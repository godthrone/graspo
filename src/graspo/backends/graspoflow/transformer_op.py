"""Layer 1 — TransformerStageOp: generic pipeline stage for all decoder-only transformers.

Extracted from ``qwen_ops.py``.  Every model family (Qwen3, Qwen3.5/3.6,
DeepSeek, …) subclasses this and only implements ``forward()`` / ``backward()``.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.distributed as dist

from graspo.backends.graspoflow.operator import (
    ComputeOperator,
    Microbatch,
    OpMemoryProfile,
)
from graspo.backends.graspoflow.parallel_state import GraspoFlowState
from graspo.backends.graspoflow.placement import NativePlacementPlan


def _now() -> float:
    import time

    return time.monotonic()


class TransformerStageOp(ComputeOperator):
    """Generic pipeline stage for all decoder-only transformer models.

    Provides:
    - P2P communication: ``_send_hidden`` / ``_recv_hidden``
    - Generic ``memory_profile`` (based on hidden_size)
    - Common attributes: ``pp_rank``, ``pp_size``, ``device``, ``placement``

    Subclasses must implement:
    - ``forward(mb)`` — stage-specific forward pass
    - ``backward(mb)`` — stage-specific backward pass
    """

    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        tp_state: GraspoFlowState,
        tp_size: int = 1,
    ) -> None:
        super().__init__(name=name, tp_size=tp_size)
        self.model = model
        self.tp_state = tp_state

    # ── Common properties ───────────────────────────────────────────────────

    @property
    def pp_rank(self) -> int:
        return self.tp_state.pp_rank

    @property
    def pp_size(self) -> int:
        return self.tp_state.pp_size

    @property
    def device(self) -> torch.device:
        return self.tp_state.device

    @property
    def placement(self) -> NativePlacementPlan | None:
        return self.model.placement

    @property
    def memory_profile(self) -> OpMemoryProfile:
        cfg = self.model.config
        hidden = int(cfg.hidden_size)
        return OpMemoryProfile(
            forward_activation_bytes=hidden * 2,  # bf16 hidden per token
            backward_intermediate_bytes=hidden * 2,  # grad per token
            gradient_bytes=0,  # LoRA grads are tiny
        )

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    # ── P2P communication ───────────────────────────────────────────────────

    def _send_hidden(self, tensor: torch.Tensor) -> None:
        """Send hidden states downstream."""
        dst = int(self.tp_state.next_pp_rank or 0)
        dist.send(tensor.contiguous(), dst=dst)

    def _recv_hidden(
        self, batch: int, seq_len: int, hidden_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Receive hidden states from upstream."""
        src = int(self.tp_state.prev_pp_rank or 0)
        tensor = torch.empty(
            (batch, seq_len, hidden_size), device=self.device, dtype=dtype
        )
        dist.recv(tensor, src=src)
        return tensor

    # ── Abstract ────────────────────────────────────────────────────────────

    @abstractmethod
    def forward(self, mb: Microbatch) -> Microbatch:
        """Run the forward pass for one microbatch."""

    @abstractmethod
    def backward(self, mb: Microbatch) -> Microbatch:
        """Run the backward pass for one microbatch."""
