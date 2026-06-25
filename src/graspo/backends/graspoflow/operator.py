"""Layer 1 — Scheduling framework primitives: Microbatch, OpBuffer, ComputeOperator.

These are the Flink-inspired building blocks.  They know nothing about models,
training objectives, or specific layer implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch


# ── Microbatch ────────────────────────────────────────────────────────────────


@dataclass
class Microbatch:
    """The unit of data that flows through the pipeline.

    A microbatch carries its identity, the raw inputs needed by the first
    (embedding) stage, the hidden states that flow between pipeline stages, and
    optional training labels used by the final stage to compute loss.
    """

    idx: int  # position in the microbatch sequence (0..N-1)

    # ── Embedding-stage inputs (populated for stage 0 only) ──
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    multimodal_inputs: dict[str, Any] | None = None

    # ── Flowing hidden states (carried between pipeline stages) ──
    hidden_states: torch.Tensor | None = None

    # ── Training metadata (used by the final stage for loss) ──
    old_log_probs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None

    # ── Internal bookkeeping ──
    _stage_input: torch.Tensor | None = field(default=None, repr=False)
    _stage_output: torch.Tensor | None = field(default=None, repr=False)

    @property
    def batch_size(self) -> int:
        if self.input_ids is not None:
            return int(self.input_ids.shape[0])
        if self.hidden_states is not None:
            return int(self.hidden_states.shape[0])
        return 0

    @property
    def seq_len(self) -> int:
        if self.input_ids is not None:
            return int(self.input_ids.shape[1])
        if self.hidden_states is not None:
            return int(self.hidden_states.shape[1])
        return 0

    def clone_for_retry(self, idx: int) -> "Microbatch":
        """Shallow-clone with a new index (used by rollout retry)."""
        return Microbatch(
            idx=idx,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            multimodal_inputs=self.multimodal_inputs,
            old_log_probs=self.old_log_probs,
            advantages=self.advantages,
            action_mask=self.action_mask,
        )


# ── OpMemoryProfile ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OpMemoryProfile:
    """Estimated memory footprint of a single microbatch inside an operator.

    All sizes in *bytes*, estimated from tensor shapes and dtype.
    """

    forward_activation_bytes: int = 0
    backward_intermediate_bytes: int = 0
    gradient_bytes: int = 0

    @property
    def total_per_microbatch(self) -> int:
        return (
            self.forward_activation_bytes
            + self.backward_intermediate_bytes
            + self.gradient_bytes
        )


# ── OpBuffer ──────────────────────────────────────────────────────────────────


class OpBuffer:
    """Bounded FIFO buffer — the backpressure primitive.

    Corresponds to Flink's ResultPartition / InputGate pair: every operator
    writes to its *output* buffer and reads from its *input* buffer.  When a
    buffer is full the upstream operator is backpressured.

    Thread-unsafe by design — pipeline execution is single-threaded per rank.
    """

    def __init__(self, max_slots: int, name: str = "") -> None:
        if max_slots < 1:
            raise ValueError("OpBuffer max_slots must be >= 1")
        self.max_slots = max_slots
        self.name = name
        self._deque: deque[Microbatch] = deque()

    # ── capacity / waterlevel ──

    @property
    def is_full(self) -> bool:
        return len(self._deque) >= self.max_slots

    @property
    def is_empty(self) -> bool:
        return len(self._deque) == 0

    @property
    def size(self) -> int:
        return len(self._deque)

    @property
    def waterlevel(self) -> float:
        """0.0 (empty) … 1.0 (full)."""
        return self.size / max(self.max_slots, 1)

    # ── push / pop ──

    def push(self, mb: Microbatch) -> bool:
        """Push a microbatch into the buffer.

        Returns:
            True if accepted, False if the buffer is full (backpressure signal).
        """
        if self.is_full:
            return False
        self._deque.append(mb)
        return True

    def pop(self) -> Microbatch | None:
        """Pop the oldest microbatch, or None if empty."""
        if self.is_empty:
            return None
        return self._deque.popleft()

    def peek(self) -> Microbatch | None:
        """Return the oldest microbatch without removing it."""
        if self.is_empty:
            return None
        return self._deque[0]

    def clear(self) -> None:
        self._deque.clear()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        tag = f"'{self.name}' " if self.name else ""
        return f"OpBuffer({tag}{self.size}/{self.max_slots})"


# ── ComputeOperator ───────────────────────────────────────────────────────────


class ComputeOperator(ABC):
    """A single pipeline stage — owns a group of transformer layers.

    An operator:
    - Reads from `self.input_buffer` (upstream)
    - Writes to `self.output_buffer` (downstream, None for the terminal op)
    - Has a fixed `tp_size` requirement (GPUs needed for tensor parallelism)

    The operator does NOT own the NCCL communication — layers are placed on
    their assigned GPU by the placement plan and P2P happens via NCCL send/recv
    inside the operator's forward/backward.

    Subclass for each model architecture (e.g. `DecoderLayerOp` wraps N Qwen
    decoder layers, `LMHeadOp` wraps final norm + lm_head).
    """

    def __init__(self, *, name: str, tp_size: int = 1) -> None:
        self.name = name
        self.tp_size = tp_size
        self.input_buffer: OpBuffer | None = None
        self.output_buffer: OpBuffer | None = None

    # ── public API ──

    def attach_buffers(
        self, input_buffer: OpBuffer | None, output_buffer: OpBuffer | None
    ) -> None:
        """Wire this operator into the pipeline DAG."""
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

    @abstractmethod
    def forward(self, mb: Microbatch) -> Microbatch:
        """Run the forward pass for one microbatch.

        The operator reads from `mb` (and potentially from `self.input_buffer`
        via NCCL recv), runs its layers, and either writes to
        `self.output_buffer` or returns the final result.
        """

    @abstractmethod
    def backward(self, mb: Microbatch) -> Microbatch:
        """Run the backward pass for one microbatch.

        Receives gradient from downstream (via NCCL recv), backpropagates
        through its layers, and sends gradient upstream (via NCCL send).
        Returns the microbatch with gradient in `hidden_states`.
        """

    @property
    @abstractmethod
    def memory_profile(self) -> OpMemoryProfile:
        """Estimated memory per microbatch for this operator."""

    @abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return parameters that require gradient updates."""

    def done(self) -> None:
        """Called when the pipeline is shutting down. Release resources."""
        if self.input_buffer is not None:
            self.input_buffer.clear()
        if self.output_buffer is not None:
            self.output_buffer.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, tp={self.tp_size})"
