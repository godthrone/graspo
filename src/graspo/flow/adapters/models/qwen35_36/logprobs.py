"""Qwen3.5/3.6 adapter — sequence log-probability methods."""

import time
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.parallel.pipeline_comm import PipelineComm, wait_all
from graspo.flow.parallel.placement import (
    placement_summary,
)
from graspo.flow.parallel.tensor_utils import (
    _add_pipeline_stage_timing,
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
)


class _Qwen35LogprobsMethods:
    """Mixin: sequence log-probability methods for Qwen35Adapter."""

    # ── Sequence log probs ──────────────────────────────────────────────────

    def sequence_log_probs(
        self,
        *,
        sequences: Any,
        attention_mask: Any = None,
        metadata: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_ready()
        assert self.model is not None
        if self._is_pipeline_parallel():
            return self._pipeline_sequence_log_probs(sequences, attention_mask, metadata=metadata)
        self.model.eval()
        sequences = sequences.to(self.device)
        attention_mask = attention_mask.to(self.device).bool()
        multimodal_inputs = self._multimodal_inputs_from_metadata(
            metadata, batch_size=int(sequences.shape[0])
        )
        with torch.no_grad():
            if multimodal_inputs is not None:
                if not isinstance(self.model, Qwen35HybridTextModel):
                    raise ValueError("multimodal metadata was provided for a non-multimodal model")
                log_probs = self.model.sequence_log_probs(
                    sequences,
                    attention_mask,
                    multimodal_inputs=multimodal_inputs,
                )
            else:
                log_probs = self.model.sequence_log_probs(sequences, attention_mask)
        self._emit_rank_memory_event(
            "logprob_after",
            {
                "batch_size": int(sequences.shape[0]),
                "sequence_len": int(sequences.shape[1]),
                "multimodal_enabled": multimodal_inputs is not None,
            },
        )
        return log_probs

    def _pipeline_sequence_log_probs(
        self,
        sequences: Any,
        attention_mask: Any,
        metadata: Any | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.model, Qwen35HybridTextModel)
        self.model.eval()
        sequences = sequences.to(self.device)
        attention_mask = attention_mask.to(self.device).bool()
        stage_timing = _new_pipeline_stage_timing()
        multimodal_inputs = self._multimodal_inputs_from_metadata(
            metadata, batch_size=int(sequences.shape[0])
        )
        with torch.no_grad():
            comm = PipelineComm(
                device=self.device,
                fwd_group=self.tp_state.pp_group_fwd,
                bwd_group=self.tp_state.pp_group_bwd,
                max_inflight=int(self.config.graspoflow.pp_max_inflight_microbatches),
                chunk_count=1,
            )
            hidden, _present, _stage_input, send_work = self._pipeline_forward_hidden(
                input_ids=sequences,
                hidden_states=None,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                multimodal_inputs=multimodal_inputs,
                position_input_ids=sequences,
                apply_lm_head=False,
                timing=stage_timing,
                comm=comm,
                tag=0,
                debug_label="logprob",
            )
            if send_work is not None:
                wait_all([send_work])
            if self.pp_rank == self.pp_size - 1:
                assert hidden is not None
                assert self.model.norm is not None and self.model.lm_head is not None
                norm_started_at = time.monotonic()
                hidden = self.model.norm(hidden)
                _add_pipeline_stage_timing(stage_timing, "pipeline_norm_sec", norm_started_at)
                lm_head_started_at = time.monotonic()
                log_probs = _selected_token_log_probs_from_hidden(
                    hidden[:, :-1].float(),
                    self.model.lm_head.weight.float(),
                    sequences[:, 1:],
                )
                _add_pipeline_stage_timing(stage_timing, "pipeline_lm_head_sec", lm_head_started_at)
            else:
                log_probs = torch.empty(
                    (sequences.shape[0], max(sequences.shape[1] - 1, 0)),
                    dtype=torch.float32,
                    device=self.device,
                )
            dist.broadcast(log_probs, src=(self.pp_size - 1) * self.tp_size)
        self._emit_rank_memory_event(
            "pipeline_logprob_after",
            {
                "batch_size": int(sequences.shape[0]),
                "sequence_len": int(sequences.shape[1]),
                "multimodal_enabled": multimodal_inputs is not None,
                "placement": (
                    placement_summary(self.placement) if self.placement is not None else None
                ),
                "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            },
        )
        return log_probs
