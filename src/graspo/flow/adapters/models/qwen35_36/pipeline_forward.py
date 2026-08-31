"""Qwen3.5/3.6 adapter — unified PP (pipeline-parallel) stage forward.

Boundary (BADGE §1.1 / §1.3): this module owns *one* canonical PP forward path
for the Qwen3.5/3.6 model family and the PP debug logger.  It is the only place
that calls :meth:`Qwen35HybridTextModel.forward_stage` inside a pipeline; the
training (``training.py``/``training_sft.py``), log-probability
(``logprobs.py``) and roll-out generation (``generation.py``) paths all
delegate here so the SP / tag / KV-cache / all-gather semantics can never
silently diverge between callers.
"""

import os
import sys
import time
from typing import Any

import torch

from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.logging import rotating_append, run_log_dir
from graspo.flow.parallel.pipeline_comm import PipelineComm
from graspo.flow.parallel.tensor_utils import _add_pipeline_stage_timing


def _pp_debug_log(output_dir: str, msg: str) -> None:
    """PP 调试日志：打印到 stderr + 落盘到 ``{output_dir}/logs/<run_id>/pp_debug.log``。"""
    rank = os.environ.get("RANK", "0")
    line = f"[pp-debug rank={rank}] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        rotating_append(run_log_dir(output_dir) / "pp_debug.log", line + "\n")
    # noqa: BLE001 — 调试日志落盘失败不应中断训练主流程
    except Exception:
        pass


class _Qwen35PipelineForwardMethods:
    """Unified pipeline-parallel stage forward for Qwen35Adapter."""

    # ── Unified PP stage forward ────────────────────────────────────────────
    # This is the single canonical PP forward used by training (RL/SFT),
    # sequence-log-prob and roll-out generation.  It encapsulates:
    #   * stage 0 embedding vs intermediate/final hidden-state input
    #   * SP-aware recv-buffer sizing (scatter only on stage 0, all-gather on last)
    #   * async P2P through :class:`PipelineComm` (fwd channel + explicit tag)
    #   * optional KV-cache return (``use_cache=True``) for auto-regressive decode
    #   * optional ``position_ids`` (pre-computed on stage 0 for multi-modal M-RoPE)
    # Callers must supply ``comm``/``tag`` when ``pp_size > 1``; the returned
    # ``send_work`` must be collected and ``wait_all``-ed by the caller.

    def _pipeline_forward_hidden(
        self,
        *,
        input_ids: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: tuple[Any, ...] | None = None,
        use_cache: bool = False,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        position_input_ids: torch.Tensor | None = None,
        apply_lm_head: bool = False,
        timing: dict[str, float | int] | None = None,
        comm: PipelineComm | None = None,
        tag: int = 0,
        debug_label: str = "fwd",
    ) -> tuple[torch.Tensor | None, tuple[Any, ...] | None, torch.Tensor | None, Any | None]:
        """Run one PP stage forward.

        Returns ``(output, present, stage_input, send_work)`` where
        ``output`` is the hidden (all-gathered on the last stage unless
        ``apply_lm_head``), ``present`` is the new KV cache when
        ``use_cache=True`` (else ``None``), ``stage_input`` is the recv buffer
        (for gradient propagation on backward), and ``send_work`` is the async
        send handle for this stage (``None`` on the last stage).
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        # ── Determine the sequence length and SP-aware recv-alloc size ──────
        if input_ids is not None:
            seq_len = int(input_ids.shape[1])
        elif position_input_ids is not None:
            seq_len = int(position_input_ids.shape[1])
        elif hidden_states is not None:
            seq_len = int(hidden_states.shape[1])
        else:
            raise RuntimeError("PP forward requires input_ids or hidden_states")
        if self.model._use_sp:
            sp_size = self.model.tp_size
            pad_to = ((seq_len + sp_size - 1) // sp_size) * sp_size
            recv_seq = pad_to // sp_size
        else:
            recv_seq = seq_len
        batch = int(attention_mask.shape[0])
        hidden_size = int(self.model.config.hidden_size)
        dtype = next(self.model.parameters()).dtype
        _pp_debug_log(
            str(self.config.training.output_dir),
            f"{debug_label} stage={self.pp_rank} tag={tag} pp_size={self.pp_size} "
            f"input_seq={seq_len} recv_alloc_seq={recv_seq} batch={batch} "
            f"hidden={hidden_size} use_cache={use_cache}",
        )

        stage_input: torch.Tensor | None = None
        send_work: Any | None = None
        if self.pp_rank == 0:
            compute_started_at = time.monotonic()
            output = self.model.forward_stage(
                None,
                input_ids,
                attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                multimodal_inputs=multimodal_inputs,
                position_ids=position_ids,
                position_input_ids=(
                    position_input_ids if position_input_ids is not None else input_ids
                ),
                apply_lm_head=apply_lm_head,
                all_gather_output=False,
            )
            _add_pipeline_stage_timing(timing, "pipeline_stage_compute_sec", compute_started_at)
        else:
            stage_input = torch.empty(
                (batch, recv_seq, hidden_size), device=self.device, dtype=dtype
            )
            recv_started_at = time.monotonic()
            assert comm is not None
            recv_work = comm.fwd_recv(stage_input, src=int(self.tp_state.prev_pp_rank), tag=tag)
            comm.wait(recv_work)  # 阻塞直到数据到达（上游异步 send，不会死锁）
            _add_pipeline_stage_timing(timing, "pipeline_recv_sec", recv_started_at)
            if not use_cache:
                # 仅训练路径需要梯度回流；生成（no_grad）与此无关。
                stage_input.requires_grad_(True)
            compute_started_at = time.monotonic()
            output = self.model.forward_stage(
                stage_input,
                None,
                attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                multimodal_inputs=multimodal_inputs,
                position_ids=position_ids,
                position_input_ids=position_input_ids,
                apply_lm_head=apply_lm_head,
                all_gather_output=(self.pp_rank == self.pp_size - 1),
            )
            _add_pipeline_stage_timing(timing, "pipeline_stage_compute_sec", compute_started_at)

        present: tuple[Any, ...] | None = None
        if use_cache:
            output, present = output
        assert isinstance(output, torch.Tensor)
        if self.pp_rank < self.pp_size - 1:
            send_started_at = time.monotonic()
            assert comm is not None
            _pp_debug_log(
                str(self.config.training.output_dir),
                f"send stage={self.pp_rank} tag={tag} tensor={tuple(output.shape)} "
                f"send_seq={int(output.shape[1])} dst={self.tp_state.next_pp_rank}",
            )
            send_work = comm.fwd_send(
                output.detach().contiguous(), dst=int(self.tp_state.next_pp_rank), tag=tag
            )
            _add_pipeline_stage_timing(timing, "pipeline_send_sec", send_started_at)
        if timing is not None:
            timing["pipeline_forward_calls"] = int(timing.get("pipeline_forward_calls") or 0) + 1
        return output, present, stage_input, send_work
