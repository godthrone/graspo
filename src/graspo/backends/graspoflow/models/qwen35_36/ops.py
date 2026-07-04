"""Qwen3.5/3.6 (hybrid attention) pipeline stage operators.

Refactored from ``graspoflow/qwen_ops.py``.  Supports visual tower for multimodal.
"""

import torch

from graspo.backends.graspoflow.operator import Microbatch
from graspo.backends.graspoflow.transformer_op import TransformerStageOp


class Qwen35EmbedStageOp(TransformerStageOp):
    """Pipeline stage 0: embeddings + visual tower + first group of hybrid layers."""

    def forward(self, mb: Microbatch) -> Microbatch:
        assert mb.input_ids is not None, "EmbedStageOp requires input_ids"
        assert mb.attention_mask is not None, "EmbedStageOp requires attention_mask"

        # Embed + visual
        hidden = self.model.embed_inputs(mb.input_ids, multimodal_inputs=mb.multimodal_inputs)
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
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("EmbedStageOp.backward: no gradient available")
            grad_output = hs

        stage_output.backward(grad_output)
        return mb


class Qwen35DecoderStageOp(TransformerStageOp):
    """Pipeline stage 1..N-2: hybrid decoder layers."""

    def forward(self, mb: Microbatch) -> Microbatch:
        dtype = next(self.model.parameters()).dtype
        hidden_size = int(self.model.config.hidden_size)

        if mb.hidden_states is not None:
            batch, seq_len = (
                int(mb.hidden_states.shape[0]),
                int(mb.hidden_states.shape[1]),
            )
        else:
            assert mb.input_ids is not None
            batch, seq_len = int(mb.input_ids.shape[0]), int(mb.input_ids.shape[1])

        if self.pp_size > 1:
            stage_input = self._recv_hidden(batch, seq_len, hidden_size, dtype)
            stage_input.requires_grad_(True)
        else:
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("DecoderStageOp.forward: no hidden_states")
            stage_input = hs

        position_ids = None
        if mb.input_ids is not None and mb.attention_mask is not None:
            position_ids = self.model.compute_multimodal_position_ids(
                input_ids=mb.input_ids,
                attention_mask=mb.attention_mask,
                multimodal_inputs=mb.multimodal_inputs,
                past_key_values=None,
                query_len=seq_len,
            )

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

        if self.pp_size > 1 and self.pp_rank < self.pp_size - 1:
            self._send_hidden(hidden.detach())

        mb.hidden_states = hidden.detach()
        mb._stage_input = stage_input
        mb._stage_output = hidden
        return mb

    def backward(self, mb: Microbatch) -> Microbatch:
        stage_output = mb._stage_output
        stage_input = mb._stage_input
        if stage_output is None or stage_input is None:
            raise RuntimeError("DecoderStageOp.backward: missing _stage_output or _stage_input")

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

        if self.pp_size > 1 and self.pp_rank > 0:
            grad_input = stage_input.grad
            if grad_input is None:
                grad_input = torch.zeros_like(stage_input)
            import torch.distributed as dist

            dst = int(self.tp_state.prev_pp_rank or 0)
            dist.send(grad_input.contiguous(), dst=dst)

        return mb


class Qwen35HeadStageOp(TransformerStageOp):
    """Pipeline final stage: last hybrid layers + norm + lm_head."""

    def forward(self, mb: Microbatch) -> Microbatch:
        dtype = next(self.model.parameters()).dtype
        hidden_size = int(self.model.config.hidden_size)

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

        if self.pp_size > 1:
            stage_input = self._recv_hidden(batch, seq_len, hidden_size, dtype)
            stage_input.requires_grad_(True)
        else:
            hs = mb.hidden_states
            if hs is None:
                raise RuntimeError("HeadStageOp.forward: no hidden_states")
            stage_input = hs

        position_ids = None
        if mb.input_ids is not None and mb.attention_mask is not None:
            position_ids = self.model.compute_multimodal_position_ids(
                input_ids=mb.input_ids,
                attention_mask=mb.attention_mask,
                multimodal_inputs=mb.multimodal_inputs,
                past_key_values=None,
                query_len=seq_len,
            )

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

        assert self.model.norm is not None, "HeadStageOp requires norm"
        assert self.model.lm_head is not None, "HeadStageOp requires lm_head"
        hidden = self.model.norm(hidden)
        logits = self.model.lm_head(hidden)

        mb.hidden_states = logits
        mb._stage_input = stage_input
        mb._stage_output = hidden
        return mb

    def backward(self, mb: Microbatch) -> Microbatch:
        stage_input = mb._stage_input
        if stage_input is None:
            raise RuntimeError("HeadStageOp.backward: no _stage_input")

        if self.pp_size > 1 and self.pp_rank > 0:
            grad_input = stage_input.grad
            if grad_input is None:
                grad_input = torch.zeros_like(stage_input)
            import torch.distributed as dist

            dst = int(self.tp_state.prev_pp_rank or 0)
            dist.send(grad_input.contiguous(), dst=dst)

        return mb


def build_qwen35_ops(
    *,
    model,
    tp_state,
    tp_size: int,
) -> list[TransformerStageOp]:
    """Build the list of Qwen3.5/3.6 pipeline operators."""
    placement = model.placement
    pp_rank = tp_state.pp_rank
    pp_size = tp_state.pp_size
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1
    has_visual = (
        placement is not None
        and placement.include_embeddings
        and bool(getattr(model.config, "has_vision_config", False))
    )

    if is_first and has_visual:
        op: TransformerStageOp = Qwen35EmbedStageOp(
            name=f"embed_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )
    elif is_first:
        op = Qwen35EmbedStageOp(
            name=f"embed_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )
    elif is_last:
        op = Qwen35HeadStageOp(
            name=f"head_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )
    else:
        op = Qwen35DecoderStageOp(
            name=f"decoder_stage_pp{pp_rank}",
            model=model,
            tp_state=tp_state,
            tp_size=tp_size,
        )

    return [op]
