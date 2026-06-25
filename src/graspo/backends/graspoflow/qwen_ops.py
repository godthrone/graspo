"""Layer 1 — Qwen-specific pipeline operators (GraspoFlow).

Each operator wraps a Qwen35HybridTextModel instance configured for one PP
stage (via NativePlacementPlan).  The operators own their NCCL P2P
communication and expose a clean forward/backward interface used by Layer 2.

TP support:
  When tp_size > 1, multiple ranks share the same pp_rank and form a TP group.
  Attention heads are sharded across the TP group (handled by
  TensorParallelQwen35DecoderLayer).  Hidden states flowing *between* stages
  are full-dimensional (not sharded), so P2P is rank-to-rank:

    send:  rank ──→ rank + tp_size  (next PP stage, same tp_rank)
    recv:  rank - tp_size ──→ rank  (prev PP stage, same tp_rank)

  This is computed by NativeTPState (parallel_state.py):

    prev_pp_rank = rank - tp_size if pp_rank > 0 else None
    next_pp_rank = rank + tp_size if pp_rank < pp_size - 1 else None

  TP all-reduce (for LoRA gradient sync within a stage) is handled separately
  by the existing _sync_nonsharded_lora_grads mechanism.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from graspo.backends.native_tp.models.qwen.modeling_hybrid import (
    Qwen35HybridTextModel,
)
from graspo.backends.native_tp.parallel_state import NativeTPState
from graspo.backends.native_tp.placement import (
    NativePlacementPlan,
)
from graspo.backends.graspoflow.operator import (
    ComputeOperator,
    Microbatch,
    OpMemoryProfile,
)


# ── Timing helpers ────────────────────────────────────────────────────────────


def _now() -> float:
    return time.monotonic()


# ── Base Stage Op ─────────────────────────────────────────────────────────────


class QwenStageOp(ComputeOperator):
    """Base for Qwen pipeline stage operators.

    Each instance owns one Qwen35HybridTextModel for a specific PP stage.
    Subclasses handle the stage-specific forward/backward logic.
    """

    def __init__(
        self,
        *,
        name: str,
        model: Qwen35HybridTextModel,
        tp_state: NativeTPState,
        tp_size: int = 1,
    ) -> None:
        super().__init__(name=name, tp_size=tp_size)
        self.model = model
        self.tp_state = tp_state

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


# ── Embedding Stage ───────────────────────────────────────────────────────────


class QwenEmbedStageOp(QwenStageOp):
    """Pipeline stage 0: embeddings + visual encoder + first group of layers.

    On forward:
      1. Embed input_ids (and visual features) → hidden_states
      2. Run assigned decoder layers → stage_output
      3. Send stage_output to next stage
      4. Return microbatch (output goes to next Op via buffer/send)

    On backward:
      1. Receive grad_output from stage 1
      2. stage_output.backward(grad_output)
      3. No upstream to send grad to (stage 0 is the root)
    """

    def forward(self, mb: Microbatch) -> Microbatch:
        assert mb.input_ids is not None, "EmbedStageOp requires input_ids"
        assert mb.attention_mask is not None, "EmbedStageOp requires attention_mask"

        # Embed + visual
        hidden = self.model.embed_inputs(
            mb.input_ids, multimodal_inputs=mb.multimodal_inputs
        )
        seq_len = int(hidden.shape[1])

        # Build position_ids
        position_ids = self.model.compute_multimodal_position_ids(
            input_ids=mb.input_ids,
            attention_mask=mb.attention_mask,
            multimodal_inputs=mb.multimodal_inputs,
            past_key_values=None,
            query_len=seq_len,
        )

        # Run decoder layers
        for layer in self.model.layers:
            hidden = layer(hidden, position_ids, mb.attention_mask)

        # Send to next stage
        if self.pp_size > 1:
            self._send_hidden(hidden.detach())

        mb.hidden_states = hidden.detach()
        mb._stage_output = hidden
        return mb

    def backward(self, mb: Microbatch) -> Microbatch:
        stage_output = mb._stage_output
        if stage_output is None:
            raise RuntimeError("EmbedStageOp.backward: no _stage_output saved")

        if self.pp_size > 1:
            dtype = next(self.model.parameters()).dtype
            batch, seq_len, hidden_size = (
                int(stage_output.shape[0]),
                int(stage_output.shape[1]),
                int(stage_output.shape[2]),
            )
            grad_output = self._recv_hidden(batch, seq_len, hidden_size, dtype)
        else:
            # Single-rank mode (no PP): grad comes from head
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("EmbedStageOp.backward: no gradient available")
            grad_output = hs

        stage_output.backward(grad_output)
        return mb


# ── Decoder Stage ─────────────────────────────────────────────────────────────


class QwenDecoderStageOp(QwenStageOp):
    """Pipeline stage 1..N-2: only decoder layers.

    On forward:
      1. Receive hidden states from upstream
      2. Run assigned decoder layers
      3. Send output to next stage

    On backward:
      1. Receive gradient from downstream
      2. stage_output.backward(received_grad)
      3. Send stage_input.grad upstream
    """

    def forward(self, mb: Microbatch) -> Microbatch:
        dtype = next(self.model.parameters()).dtype
        hidden_size = int(self.model.config.hidden_size)

        # Determine batch/seq_len from the microbatch
        if mb.hidden_states is not None:
            batch, seq_len = (
                int(mb.hidden_states.shape[0]),
                int(mb.hidden_states.shape[1]),
            )
        else:
            # Fallback: must have input_ids for batch/seq info
            assert mb.input_ids is not None
            batch, seq_len = int(mb.input_ids.shape[0]), int(mb.input_ids.shape[1])

        # Receive from upstream
        if self.pp_size > 1:
            stage_input = self._recv_hidden(batch, seq_len, hidden_size, dtype)
            stage_input.requires_grad_(True)
        else:
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("DecoderStageOp.forward: no hidden_states")
            stage_input = hs

        # Build position_ids if we have input_ids
        position_ids: torch.Tensor | None = None
        if mb.input_ids is not None and mb.attention_mask is not None:
            position_ids = self.model.compute_multimodal_position_ids(
                input_ids=mb.input_ids,
                attention_mask=mb.attention_mask,
                multimodal_inputs=mb.multimodal_inputs,
                past_key_values=None,
                query_len=seq_len,
            )

        # Run decoder layers
        hidden = stage_input
        if position_ids is not None and mb.attention_mask is not None:
            for layer in self.model.layers:
                hidden = layer(hidden, position_ids, mb.attention_mask)
        else:
            # Without position_ids/attention_mask, just run the forward pass
            for layer in self.model.layers:
                hidden = layer(
                    hidden,
                    torch.zeros(3, batch, seq_len, device=hidden.device, dtype=torch.long),
                    torch.ones(batch, seq_len, device=hidden.device, dtype=torch.bool),
                )

        # Send to next stage
        if self.pp_size > 1 and self.pp_rank < self.pp_size - 1:
            self._send_hidden(hidden.detach())

        mb.hidden_states = hidden.detach()
        mb._stage_input = stage_input
        mb._stage_output = hidden
        return mb

    def backward(self, mb: Microbatch) -> Microbatch:
        stage_output = mb._stage_output
        stage_input = mb._stage_input

        if stage_output is None:
            raise RuntimeError("DecoderStageOp.backward: no _stage_output")
        if stage_input is None:
            raise RuntimeError("DecoderStageOp.backward: no _stage_input")

        dtype = next(self.model.parameters()).dtype
        batch, seq_len, hidden_size = (
            int(stage_output.shape[0]),
            int(stage_output.shape[1]),
            int(stage_output.shape[2]),
        )

        if self.pp_size > 1:
            grad_output = self._recv_hidden(batch, seq_len, hidden_size, dtype)
        else:
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("DecoderStageOp.backward: no gradient available")
            grad_output = hs

        stage_output.backward(grad_output)

        # Send grad upstream
        if self.pp_size > 1 and self.pp_rank > 0:
            grad_input = stage_input.grad
            if grad_input is None:
                grad_input = torch.zeros_like(stage_input)
            dst = int(self.tp_state.prev_pp_rank or 0)
            dist.send(grad_input.contiguous(), dst=dst)

        return mb


# ── Head Stage (final) ────────────────────────────────────────────────────────


class QwenHeadStageOp(QwenStageOp):
    """Pipeline final stage: last decoder layers + norm + lm_head.

    On forward:
      1. Receive hidden states from upstream
      2. Run assigned decoder layers
      3. Apply norm → lm_head → logits (rollout) or compute loss (training)

    On backward:
      1. loss.backward()  (autograd propagates through norm+lm_head+layers)
      2. Send stage_input.grad upstream
    """

    def forward(self, mb: Microbatch) -> Microbatch:
        dtype = next(self.model.parameters()).dtype
        hidden_size = int(self.model.config.hidden_size)

        # Determine batch/seq_len
        if mb.input_ids is not None:
            batch, seq_len = (
                int(mb.input_ids.shape[0]),
                int(mb.input_ids.shape[1]),
            )
        elif mb.hidden_states is not None:
            batch, seq_len = (
                int(mb.hidden_states.shape[0]),
                int(mb.hidden_states.shape[1]),
            )
        else:
            raise RuntimeError("HeadStageOp.forward: no input_ids or hidden_states")

        # Receive from upstream
        if self.pp_size > 1:
            stage_input = self._recv_hidden(batch, seq_len, hidden_size, dtype)
            stage_input.requires_grad_(True)
        else:
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("HeadStageOp.forward: no hidden_states")
            stage_input = hs

        # Build position_ids
        position_ids: torch.Tensor | None = None
        if mb.input_ids is not None and mb.attention_mask is not None:
            position_ids = self.model.compute_multimodal_position_ids(
                input_ids=mb.input_ids,
                attention_mask=mb.attention_mask,
                multimodal_inputs=mb.multimodal_inputs,
                past_key_values=None,
                query_len=seq_len,
            )

        # Run decoder layers
        hidden = stage_input
        if position_ids is not None and mb.attention_mask is not None:
            for layer in self.model.layers:
                hidden = layer(hidden, position_ids, mb.attention_mask)
        else:
            for layer in self.model.layers:
                hidden = layer(
                    hidden,
                    torch.zeros(3, batch, seq_len, device=hidden.device, dtype=torch.long),
                    torch.ones(batch, seq_len, device=hidden.device, dtype=torch.bool),
                )

        # Apply norm + lm_head
        assert self.model.norm is not None, "HeadStageOp requires norm"
        assert self.model.lm_head is not None, "HeadStageOp requires lm_head"
        hidden = self.model.norm(hidden)
        logits = self.model.lm_head(hidden)

        mb.hidden_states = logits
        mb._stage_input = stage_input
        mb._stage_output = hidden  # pre-lm_head hidden (for backward)
        return mb

    def backward(self, mb: Microbatch) -> Microbatch:
        stage_input = mb._stage_input
        if stage_input is None:
            raise RuntimeError("HeadStageOp.backward: no _stage_input")

        if self.pp_size > 1 and self.pp_rank > 0:
            grad_input = stage_input.grad
            if grad_input is None:
                grad_input = torch.zeros_like(stage_input)
            dst = int(self.tp_state.prev_pp_rank or 0)
            dist.send(grad_input.contiguous(), dst=dst)

        return mb


# ── Utility: build Qwen ops from placement plan ───────────────────────────────


def build_qwen_ops(
    *,
    model: Qwen35HybridTextModel,
    tp_state: NativeTPState,
    tp_size: int,
) -> list[QwenStageOp]:
    """Build the list of Qwen pipeline operators from a model and state.

    Currently returns a single-element list (one Op = one PP stage).
    For finer granularity we could split the layers into multiple Ops.
    """
    placement = model.placement
    pp_rank = tp_state.pp_rank
    pp_size = tp_state.pp_size
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1
    has_visual = placement is not None and placement.include_embeddings and bool(
        getattr(model.config, "has_vision_config", False)
    )

    if is_first and has_visual:
        op: QwenStageOp = QwenEmbedStageOp(
            name=f"embed_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )
    elif is_first and not has_visual:
        # Text-only embedding stage
        op = QwenEmbedStageOp(
            name=f"embed_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )
    elif is_last:
        op = QwenHeadStageOp(
            name=f"head_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )
    else:
        op = QwenDecoderStageOp(
            name=f"decoder_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )

    return [op]
