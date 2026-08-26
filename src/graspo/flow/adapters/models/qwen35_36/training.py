"""Qwen3.5/3.6 adapter — RL training methods (TP-only, PP 1F1B)."""

import os
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.parallel.pipeline_comm import PipelineComm, wait_all
from graspo.flow.parallel.scheduling import build_scheduler
from graspo.flow.parallel.tensor_utils import (
    _add_pipeline_stage_timing,
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
    collate_experiences,
)
from graspo.ripple.buffer import Experience
from graspo.ripple.multimodal.contract import assert_rl_training_has_multimodal


def _log_cuda_mem(label: str) -> None:
    """Log CUDA memory stats to stderr for rank 0 only."""
    if os.environ.get("RANK", "0") == "0":
        rank = os.environ.get("RANK", "0")
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"[MEM rank={rank}] {label}: allocated={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB max_allocated={max_allocated:.2f}GB",
            file=sys.stderr,
            flush=True,
        )


class _Qwen35TrainingMethods:
    """Mixin: RL training/batch optimization methods for Qwen35Adapter."""

    # ── TP-only RL training ──────────────────────────────────────────────────

    def train_batch(
        self,
        *,
        experiences: list[Experience],
        optimizer_steps: int = 1,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """RL 训练：对一批 ``Experience`` 执行 forward → GRPO loss → backward。

        支持梯度累积：每个 micro_batch 处理 ``micro_batch_size`` 个 experience，
        所有 micro-batch 的梯度累加后统一 ``optimizer.step()``。
        有效 batch size = ``micro_batch_size × gradient_accumulation_micro_batches``。
        """
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        if self._is_pipeline_parallel():
            return self._pipeline_train_batch(
                experiences,
                policy_ratio_clip_eps=policy_ratio_clip_eps,
                max_grad_norm=max_grad_norm,
            )
        self.loss_fn.policy_ratio_clip_eps = policy_ratio_clip_eps
        self.model.train()
        if bool(self.config.graspoflow.empty_cache_before_train) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        batch_size = int(self.config.graspoflow.micro_batch_size)
        train_batch_started_at = time.monotonic()
        round_secs: list[float] = []
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        micro_batch_count = 0
        # Single pass — no repeated iterations (avoids stale old_log_probs).
        optimize_round = 0
        round_started_at = time.monotonic()
        indices = self._shared_training_indices(len(experiences), optimize_round=optimize_round)
        # 梯度累积：zero_grad 只调一次，所有 micro-batch 的梯度累加后统一 step。
        num_micro_batches = max(1, len(indices) // batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        valid_micro_batches = 0
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = collate_experiences([experiences[idx] for idx in batch_indices], self.device)
            self._sync_timing()
            forward_started_at = time.monotonic()
            multimodal_inputs = self._multimodal_inputs_from_metadata(
                batch.metadata,
                batch_size=int(batch.sequences.shape[0]),
            )
            # 防呆：sequences 含图像 token 但 metadata 无 rows → 硬失败。
            # 修复前的断链正是静默走到 else 分支（图像占位符按纯文本嵌入），
            # 训练 19.5 小时视觉 LoRA 从未收到梯度（历史教训）。
            assert_rl_training_has_multimodal(
                batch.metadata,
                batch.sequences,
                image_token_id=getattr(getattr(self.model, "config", None), "image_token_id", None),
                expected_rows=int(batch.sequences.shape[0]),
            )
            if multimodal_inputs is not None:
                if not isinstance(self.model, Qwen35HybridTextModel):
                    raise ValueError("multimodal batch metadata for a non-multimodal model")
                log_probs = self.model.sequence_log_probs(
                    batch.sequences,
                    batch.attention_mask,
                    multimodal_inputs=multimodal_inputs,
                )
            else:
                log_probs = self.model.sequence_log_probs(batch.sequences, batch.attention_mask)
            self._sync_timing()
            micro_batch_forward_sec += time.monotonic() - forward_started_at
            loss = self.loss_fn(
                log_probs,
                batch.old_log_probs,
                batch.advantages,
                batch.action_mask,
            )
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue
            micro_batch_count += 1
            valid_micro_batches += 1
            loss_sum += float(loss.detach().cpu())
            # 梯度累积：loss 除以 num_micro_batches 使累加梯度等价于大 batch
            scaled_loss = loss / num_micro_batches
            self._sync_timing()
            backward_started_at = time.monotonic()
            _log_cuda_mem("before_backward")
            scaled_loss.backward()
            _log_cuda_mem("after_backward")
            self._sync_timing()
            backward_sec += time.monotonic() - backward_started_at

        # 所有 micro-batch 的 backward 完成后，统一 sync / clip / step
        # 防呆：DP/TP 梯度同步必须所有 rank 参与（集体操作），即使本 rank
        # 所有 micro-batch 都非有限（valid_micro_batches==0）也必须参加。
        # 若本 rank 无有效梯度，先 zero-fill 再参与 all_reduce。
        if valid_micro_batches == 0:
            for param in self.model.parameters():
                if param.requires_grad and param.grad is None:
                    param.grad = torch.zeros_like(param)
        from graspo.flow.lora.lora_linear import (
            _sync_dp_lora_grads,
            _sync_nonsharded_lora_grads,
        )
        from graspo.flow.parallel.tensor_utils import (
            _TENSOR_PARALLEL_GROUP,
            _TENSOR_PARALLEL_SIZE,
        )

        # DP gradient sync: AVG across DP replicas（不同数据）
        if self.tp_state is not None and self.tp_state.dp_group is not None:
            _sync_dp_lora_grads(self.model, self.tp_state.dp_group)
        # TP gradient sync: SUM across TP ranks（同数据，部分梯度）
        if _TENSOR_PARALLEL_GROUP is not None and _TENSOR_PARALLEL_SIZE > 1:
            _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)
        if valid_micro_batches > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in self.model.parameters() if param.requires_grad],
                max_grad_norm,
            )
            grad_norm_sum = float(grad_norm.detach().float().cpu())
            self._sync_timing()
            optimizer_started_at = time.monotonic()
            _log_cuda_mem("before_optimizer_step")
            self.optimizer.step()
            _log_cuda_mem("after_optimizer_step")
            self._sync_timing()
            optimizer_step_sec = time.monotonic() - optimizer_started_at
            if self.scheduler is not None:
                self.scheduler.step()
            optimizer_steps = 1
            nonzero_grad_count = self.model.nonzero_lora_grad_count()
        else:
            grad_norm_sum = 0.0
        round_secs.append(time.monotonic() - round_started_at)
        self._train_batch_call_index += 1

        lora_norm_after = self.model.lora_parameter_norm()
        metrics = {
            "optimized": optimizer_steps > 0,
            "replay_buffer_trainable_completion_count": len(experiences),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / micro_batch_count if micro_batch_count else None,
            "grad_norm_mean": grad_norm_sum,
            "nonzero_grad_count": nonzero_grad_count,
            "lora_norm_before": lora_norm_before,
            "lora_norm_after": lora_norm_after,
            "lora_norm_delta": lora_norm_after - lora_norm_before,
            "activation_checkpointing_enabled": bool(
                getattr(self.model, "gradient_checkpointing", False)
            ),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "synchronize_cuda_timing": bool(self.config.graspoflow.synchronize_cuda_timing),
            "current_lr": self._current_lr(),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("train_batch_after", {"metrics": metrics})
        return metrics

    # ── PP RL training dispatch ──────────────────────────────────────────────

    def _pipeline_train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """PP training — 1F1B 调度，梯度累积。

        Flow 层统一架构：TP + PP(1F1B)。PP=1 时走 TP-only 路径（``train_batch``），
        PP>1 时走 1F1B 流水线（本方法）。不存在其他 PP 调度策略。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        return self._pipeline_train_batch_one_f_one_b(
            experiences,
            policy_ratio_clip_eps=policy_ratio_clip_eps,
            max_grad_norm=max_grad_norm,
        )

    # ── PP forward (shared) ──────────────────────────────────────────────────

    def _pipeline_forward_for_training(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        metadata: Any | None = None,
        timing: dict[str, float | int] | None = None,
        comm: PipelineComm | None = None,
        tag: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Any | None]:
        """PP forward for RL training — delegates to the unified PP forward.

        SP / async-P2P / position_ids / tag 语义由 :meth:`_pipeline_forward_hidden`
        （pipeline_forward.py）集中维护，避免多套 PP forward 漂移。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        batch = int(sequences.shape[0])
        multimodal_inputs = self._multimodal_inputs_from_metadata(metadata, batch_size=batch)
        output, _present, stage_input, send_work = self._pipeline_forward_hidden(
            input_ids=sequences,
            hidden_states=None,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            multimodal_inputs=multimodal_inputs,
            position_input_ids=sequences,
            apply_lm_head=False,
            timing=timing,
            comm=comm,
            tag=tag,
            debug_label="fwd",
        )
        return output, stage_input, send_work

    # ── PP 1F1B schedule ─────────────────────────────────────────────────────

    def _pipeline_train_batch_one_f_one_b(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        self.loss_fn.policy_ratio_clip_eps = policy_ratio_clip_eps
        self.model.train()
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        batch_size = int(self.config.graspoflow.micro_batch_size)
        pipeline_micro_batch_size = max(1, int(self.config.graspoflow.pp_micro_batch_size))
        train_batch_started_at = time.monotonic()
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        round_secs: list[float] = []
        micro_batch_count = 0
        stage_timing = _new_pipeline_stage_timing()
        fill_sec = 0.0
        steady_sec = 0.0
        drain_sec = 0.0
        max_chunks_per_optimizer_step = 0
        configured_inflight = int(self.config.graspoflow.pp_max_inflight_microbatches)
        _pp_schedule = "one_f_one_b"  # 实际调度策略名（由 scheduler 返回更新）
        pipeline_forward_sec = 0.0
        pipeline_backward_sec = 0.0
        # Single pass — no repeated iterations.
        optimize_round = 0
        round_started_at = time.monotonic()
        indices = self._shared_training_indices(len(experiences), optimize_round=optimize_round)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[start : start + batch_size]
            chunk_batches = [
                collate_experiences(
                    [
                        experiences[idx]
                        for idx in batch_indices[
                            chunk_start : chunk_start + pipeline_micro_batch_size
                        ]
                    ],
                    self.device,
                )
                for chunk_start in range(0, len(batch_indices), pipeline_micro_batch_size)
            ]
            if not chunk_batches:
                continue
            max_chunks_per_optimizer_step = max(max_chunks_per_optimizer_step, len(chunk_batches))
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            result = self._pipeline_one_f_one_b_optimizer_step(
                chunk_batches,
                full_batch_size=len(batch_indices),
                timing=stage_timing,
                max_inflight=configured_inflight,
            )
            micro_batch_forward_sec += result["forward_sec"]
            backward_sec += result["backward_sec"]
            fill_sec += result["fill_sec"]
            steady_sec += result["steady_sec"]
            drain_sec += result["drain_sec"]
            _pp_schedule = result.get("pp_schedule", "one_f_one_b")
            pipeline_forward_sec += float(result.get("pipeline_forward_sec", 0.0))
            pipeline_backward_sec += float(result.get("pipeline_backward_sec", 0.0))
            micro_batch_count += len(chunk_batches)
            if not result["finite"]:
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            # DP/TP gradient sync: 跨 rank 同步梯度后再 clip + step
            from graspo.flow.lora.lora_linear import (
                _sync_dp_lora_grads,
                _sync_nonsharded_lora_grads,
            )
            from graspo.flow.parallel.tensor_utils import (
            _TENSOR_PARALLEL_GROUP,
            _TENSOR_PARALLEL_SIZE,
        )

            if self.tp_state is not None and self.tp_state.dp_group is not None:
                _sync_dp_lora_grads(self.model, self.tp_state.dp_group)
            if _TENSOR_PARALLEL_GROUP is not None and _TENSOR_PARALLEL_SIZE > 1:
                _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            grad_clip_started_at = time.monotonic()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                if trainable_params
                else torch.tensor(0.0)
            )
            _add_pipeline_stage_timing(stage_timing, "pipeline_grad_clip_sec", grad_clip_started_at)
            self._sync_timing()
            optimizer_started_at = time.monotonic()
            if self.optimizer is not None:
                _log_cuda_mem("before_optimizer_step_pp")
                self.optimizer.step()
                _log_cuda_mem("after_optimizer_step_pp")
            self._sync_timing()
            _add_pipeline_stage_timing(
                stage_timing, "pipeline_optimizer_step_sec", optimizer_started_at
            )
            optimizer_step_sec += time.monotonic() - optimizer_started_at
            if self.scheduler is not None and self.optimizer is not None:
                self.scheduler.step()
            optimizer_steps += 1
            loss_tensor = torch.tensor([float(result["loss_value"])], dtype=torch.float, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.MIN)
            loss_sum += float(loss_tensor.item())
            grad_norm_sum += float(grad_norm.detach().float().cpu())
            nonzero_grad_count += self.model.nonzero_lora_grad_count()
        round_secs.append(time.monotonic() - round_started_at)
        self._train_batch_call_index += 1
        lora_norm_after = self.model.lora_parameter_norm()
        effective_inflight = max_chunks_per_optimizer_step
        if configured_inflight > 0:
            effective_inflight = min(effective_inflight, configured_inflight)
        metrics = {
            "optimized": optimizer_steps > 0,
            "replay_buffer_trainable_completion_count": len(experiences),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / optimizer_steps if optimizer_steps else None,
            "grad_norm_mean": grad_norm_sum / optimizer_steps if optimizer_steps else None,
            "nonzero_grad_count": nonzero_grad_count,
            "lora_norm_before": lora_norm_before,
            "lora_norm_after": lora_norm_after,
            "lora_norm_delta": lora_norm_after - lora_norm_before,
            "activation_checkpointing_enabled": bool(
                getattr(self.model, "gradient_checkpointing", False)
            ),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "pp_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": (self.placement.strategy if self.placement is not None else None),
            "pp_schedule": _pp_schedule,
            "pipeline_pp_micro_batch_size": pipeline_micro_batch_size,
            "pipeline_chunks_per_optimizer_step": max_chunks_per_optimizer_step,
            "pp_max_inflight_microbatches": effective_inflight,
            "pipeline_inflight_bound_source": "optimizer_step_chunks",
            "pipeline_fill_sec": fill_sec,
            "pipeline_steady_sec": steady_sec,
            "pipeline_drain_sec": drain_sec,
            "pipeline_forward_sec": pipeline_forward_sec,
            "pipeline_backward_sec": pipeline_backward_sec,
            "pipeline_backpressure_wait_sec": float(stage_timing.get("pipeline_recv_sec") or 0.0)
            + float(stage_timing.get("pipeline_send_sec") or 0.0)
            + float(stage_timing.get("pipeline_grad_recv_sec") or 0.0)
            + float(stage_timing.get("pipeline_grad_send_sec") or 0.0),
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            "synchronize_cuda_timing": bool(self.config.graspoflow.synchronize_cuda_timing),
            "current_lr": self._current_lr(),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("pipeline_train_batch_after", {"metrics": metrics})
        return metrics

    def _pipeline_one_f_one_b_optimizer_step(
        self,
        chunk_batches: list[Any],
        *,
        full_batch_size: int,
        timing: dict[str, float | int],
        max_inflight: int,
    ) -> dict[str, Any]:
        # max_inflight 用于 PipelineComm 背压（有界在途 send work），此处传参以复用。
        chunk_count = len(chunk_batches)
        comm = PipelineComm(
            device=self.device,
            fwd_group=self.tp_state.pp_group_fwd,
            bwd_group=self.tp_state.pp_group_bwd,
            max_inflight=int(self.config.graspoflow.pp_max_inflight_microbatches),
            chunk_count=chunk_count,
        )
        send_works: list[Any] = []
        records: list[dict[str, Any] | None] = [None for _ in range(chunk_count)]
        finite_flags = [True for _ in range(chunk_count)]
        loss_values = [0.0 for _ in range(chunk_count)]
        forward_sec = 0.0
        backward_sec = 0.0
        fill_sec = 0.0
        steady_sec = 0.0
        drain_sec = 0.0

        def forward_chunk(chunk_idx: int) -> None:
            nonlocal forward_sec
            batch = chunk_batches[chunk_idx]
            self._sync_timing()
            forward_started_at = time.monotonic()
            stage_output, stage_input, send_work = self._pipeline_forward_for_training(
                batch.sequences,
                batch.attention_mask,
                metadata=batch.metadata,
                timing=timing,
                comm=comm,
                tag=chunk_idx,
            )
            self._sync_timing()
            forward_sec += time.monotonic() - forward_started_at
            if send_work is not None:
                send_works.append(send_work)
            loss: torch.Tensor | None = None
            finite = True
            loss_value = 0.0
            if self.pp_rank == self.pp_size - 1:
                assert stage_output is not None
                assert self.model is not None
                assert isinstance(self.model, Qwen35HybridTextModel)
                assert self.model.norm is not None and self.model.lm_head is not None
                norm_started_at = time.monotonic()
                hidden = self.model.norm(stage_output)
                _add_pipeline_stage_timing(timing, "pipeline_norm_sec", norm_started_at)
                lm_head_started_at = time.monotonic()
                log_probs = _selected_token_log_probs_from_hidden(
                    hidden[:, :-1].float(),
                    self.model.lm_head.weight.float(),
                    batch.sequences[:, 1:],
                )
                _add_pipeline_stage_timing(timing, "pipeline_lm_head_sec", lm_head_started_at)
                loss_started_at = time.monotonic()
                chunk_loss = self.loss_fn(
                    log_probs,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.action_mask,
                )
                _add_pipeline_stage_timing(timing, "pipeline_loss_sec", loss_started_at)
                finite = bool(torch.isfinite(chunk_loss).detach().cpu())
                weight = float(batch.sequences.shape[0]) / max(1, int(full_batch_size))
                loss = chunk_loss * weight if finite else None
                loss_value = float(chunk_loss.detach().cpu()) * weight if finite else 0.0
            records[chunk_idx] = {
                "stage_output": stage_output,
                "stage_input": stage_input,
                "loss": loss,
                "batch": batch,
            }
            finite_flags[chunk_idx] = finite
            loss_values[chunk_idx] = loss_value

        def backward_chunk(chunk_idx: int) -> None:
            nonlocal backward_sec
            record = records[chunk_idx]
            if record is None:
                raise RuntimeError("1F1B attempted backward before forward")
            self._sync_timing()
            backward_started_at = time.monotonic()
            assert self.tp_state is not None
            if self.pp_rank == self.pp_size - 1:
                stage_input = record["stage_input"]
                loss = record["loss"]
                if loss is not None:
                    autograd_started_at = time.monotonic()
                    _log_cuda_mem("before_backward_pp")
                    loss.backward()
                    _log_cuda_mem("after_backward_pp")
                    _add_pipeline_stage_timing(
                        timing, "pipeline_backward_autograd_sec", autograd_started_at
                    )
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    grad_send_started_at = time.monotonic()
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            else:
                stage_output = record["stage_output"]
                assert stage_output is not None
                grad_output = torch.empty_like(stage_output)
                grad_recv_started_at = time.monotonic()
                recv_work = comm.bwd_recv(
                    grad_output,
                    src=int(self.tp_state.next_pp_rank or 0),
                    tag=chunk_count + chunk_idx,
                )
                comm.wait(recv_work)  # 阻塞直到梯度到达（上游异步 send，不会死锁）
                _add_pipeline_stage_timing(timing, "pipeline_grad_recv_sec", grad_recv_started_at)
                autograd_started_at = time.monotonic()
                stage_output.backward(grad_output)
                _add_pipeline_stage_timing(
                    timing, "pipeline_backward_autograd_sec", autograd_started_at
                )
                stage_input = record["stage_input"]
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    grad_send_started_at = time.monotonic()
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            self._sync_timing()
            backward_sec += time.monotonic() - backward_started_at
            records[chunk_idx] = None

        # 可插拔调度策略：默认 1F1B（fill → steady → drain）
        scheduler = build_scheduler(
            self.config.graspoflow.pp_scheduler,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            num_chunks=chunk_count,
            forward=forward_chunk,
            backward=backward_chunk,
        )
        sched_stats = scheduler.run()
        fill_sec = float(sched_stats.get("pipeline_fill_sec", 0.0))
        steady_sec = float(sched_stats.get("pipeline_steady_sec", 0.0))
        drain_sec = float(sched_stats.get("pipeline_drain_sec", 0.0))
        # 同步所有异步发送（梯度已被上游消费），保证 buffer 生命周期安全
        wait_all(send_works)

        all_finite = all(finite_flags)
        finite_tensor = torch.tensor([all_finite], dtype=torch.int, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        all_finite = bool(finite_tensor.item())
        return {
            "finite": all_finite,
            "loss_value": sum(loss_values),
            "forward_sec": forward_sec,
            "backward_sec": backward_sec,
            "fill_sec": fill_sec,
            "steady_sec": steady_sec,
            "drain_sec": drain_sec,
            "pp_schedule": sched_stats.get("pp_schedule", "one_f_one_b"),
            "pipeline_forward_sec": float(sched_stats.get("pipeline_forward_sec", 0.0)),
            "pipeline_backward_sec": float(sched_stats.get("pipeline_backward_sec", 0.0)),
        }
