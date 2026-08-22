"""Qwen3.5/3.6 adapter — sequence log-probability methods."""

import time
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
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
            hidden, _ = self._pipeline_forward_stage_for_logprobs(
                input_ids=sequences,
                hidden_states=None,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                multimodal_inputs=multimodal_inputs,
                timing=stage_timing,
            )
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

    def _pipeline_forward_stage_for_logprobs(
        self,
        *,
        input_ids: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: tuple[Any, ...] | None,
        use_cache: bool,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        timing: dict[str, float | int] | None = None,
    ) -> tuple[torch.Tensor | None, tuple[Any, ...] | None]:
        # NOTE(债务/RL-PP 前置债): 本条路径与训练路径（training.py/training_sft.py 的
        # _pipeline_forward_for_* 用 PipelineComm 双向通道 + 显式 tag + SP 分片）不一致：
        #   - 用阻塞 dist.recv/dist.send 且无显式 tag（可能在 1F1B 交错下死锁）
        #   - query_len = input_ids.shape[1]（完整序列长），SP 下上游发来的是分片，
        #     形状对不上
        #   - forward_stage 未传 apply_lm_head/all_gather_output，最终 stage 返回的是
        #     SP 分片，后续对分片做 norm 不正确
        # 该路径仅 RL（generation/logprobs），且多模态+PP>1 已被 preflight 拦截，当前
        # 实际跑不到。待 RL-PP (Step 4) 对齐到 PipelineComm 双向通道/tag/SP 后再修。
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        batch = int(attention_mask.shape[0])
        query_len = int(input_ids.shape[1] if input_ids is not None else hidden_states.shape[1])
        hidden_size = int(self.model.config.hidden_size)
        dtype = next(self.model.parameters()).dtype
        if self.pp_rank > 0:
            hidden_states = torch.empty(
                (batch, query_len, hidden_size), device=self.device, dtype=dtype
            )
            recv_started_at = time.monotonic()
            dist.recv(hidden_states, src=int(self.tp_state.prev_pp_rank))
            _add_pipeline_stage_timing(timing, "pipeline_recv_sec", recv_started_at)
        compute_started_at = time.monotonic()
        output = self.model.forward_stage(
            hidden_states,
            input_ids if self.pp_rank == 0 else None,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            multimodal_inputs=multimodal_inputs,
            position_input_ids=input_ids,
            apply_lm_head=False,
        )
        _add_pipeline_stage_timing(timing, "pipeline_stage_compute_sec", compute_started_at)
        present = None
        if use_cache:
            output, present = output
        assert isinstance(output, torch.Tensor)
        if self.pp_rank < self.pp_size - 1:
            send_started_at = time.monotonic()
            dist.send(output.contiguous(), dst=int(self.tp_state.next_pp_rank))
            _add_pipeline_stage_timing(timing, "pipeline_send_sec", send_started_at)
            if timing is not None:
                timing["pipeline_forward_calls"] = (
                    int(timing.get("pipeline_forward_calls") or 0) + 1
                )
            return None, present
        if timing is not None:
            timing["pipeline_forward_calls"] = int(timing.get("pipeline_forward_calls") or 0) + 1
        return output, present
