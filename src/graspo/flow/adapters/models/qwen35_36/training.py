"""Qwen3.5/3.6 adapter — RL training methods (TP-only, PP 1F1B)."""

import time
import os
import sys
from typing import Any

import torch
import torch.distributed as dist


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

from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.parallel.tensor_utils import (
    _add_pipeline_stage_timing,
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
    collate_experiences,
)
from graspo.ripple.buffer import Experience
from graspo.ripple.multimodal.contract import assert_rl_training_has_multimodal


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
        if valid_micro_batches > 0:
            from graspo.flow.lora.lora_linear import _sync_dp_lora_grads, _sync_nonsharded_lora_grads
            from graspo.flow.parallel.tensor_utils import _TENSOR_PARALLEL_GROUP

            # DP gradient sync: AVG across DP replicas（不同数据）
            if self.tp_state is not None and self.tp_state.dp_group is not None:
                _sync_dp_lora_grads(self.model, self.tp_state.dp_group)
            # TP gradient sync: SUM across TP ranks（同数据，部分梯度）
            if _TENSOR_PARALLEL_GROUP is not None:
                _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)
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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        batch = int(sequences.shape[0])
        seq_len = int(sequences.shape[1])
        hidden_size = int(self.model.config.hidden_size)
        dtype = next(self.model.parameters()).dtype
        stage_input: torch.Tensor | None = None
        multimodal_inputs = self._multimodal_inputs_from_metadata(metadata, batch_size=batch)
        if self.pp_rank == 0:
            compute_started_at = time.monotonic()
            output = self.model.forward_stage(
                None,
                sequences,
                attention_mask,
                past_key_values=None,
                use_cache=False,
                multimodal_inputs=multimodal_inputs,
                position_input_ids=sequences,
                apply_lm_head=False,
            )
            _add_pipeline_stage_timing(timing, "pipeline_stage_compute_sec", compute_started_at)
        else:
            stage_input = torch.empty(
                (batch, seq_len, hidden_size), device=self.device, dtype=dtype
            )
            recv_started_at = time.monotonic()
            dist.recv(stage_input, src=int(self.tp_state.prev_pp_rank))
            _add_pipeline_stage_timing(timing, "pipeline_recv_sec", recv_started_at)
            stage_input.requires_grad_(True)
            compute_started_at = time.monotonic()
            output = self.model.forward_stage(
                stage_input,
                None,
                attention_mask,
                past_key_values=None,
                use_cache=False,
                multimodal_inputs=multimodal_inputs,
                position_input_ids=sequences,
                apply_lm_head=False,
            )
            _add_pipeline_stage_timing(timing, "pipeline_stage_compute_sec", compute_started_at)
        assert isinstance(output, torch.Tensor)
        if self.pp_rank < self.pp_size - 1:
            send_started_at = time.monotonic()
            dist.send(output.detach().contiguous(), dst=int(self.tp_state.next_pp_rank))
            _add_pipeline_stage_timing(timing, "pipeline_send_sec", send_started_at)
        if timing is not None:
            timing["pipeline_forward_calls"] = int(timing.get("pipeline_forward_calls") or 0) + 1
        return output, stage_input

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
            micro_batch_count += len(chunk_batches)
            if not result["finite"]:
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
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
            loss_payload = [float(result["loss_value"])]
            dist.broadcast_object_list(loss_payload, src=(self.pp_size - 1) * self.tp_size)
            loss_sum += float(loss_payload[0])
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
            "pp_schedule": "one_f_one_b",
            "pipeline_pp_micro_batch_size": pipeline_micro_batch_size,
            "pipeline_chunks_per_optimizer_step": max_chunks_per_optimizer_step,
            "pp_max_inflight_microbatches": effective_inflight,
            "pipeline_inflight_bound_source": "optimizer_step_chunks",
            "pipeline_fill_sec": fill_sec,
            "pipeline_steady_sec": steady_sec,
            "pipeline_drain_sec": drain_sec,
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
        del max_inflight
        chunk_count = len(chunk_batches)
        warmup = min(self.pp_size - self.pp_rank - 1, chunk_count)
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
            stage_output, stage_input = self._pipeline_forward_for_training(
                batch.sequences,
                batch.attention_mask,
                metadata=batch.metadata,
                timing=timing,
            )
            self._sync_timing()
            forward_sec += time.monotonic() - forward_started_at
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
                    assert self.tp_state is not None
                    dist.send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                    )
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            else:
                stage_output = record["stage_output"]
                assert stage_output is not None
                grad_output = torch.empty_like(stage_output)
                grad_recv_started_at = time.monotonic()
                assert self.tp_state is not None
                dist.recv(grad_output, src=int(self.tp_state.next_pp_rank or 0))
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
                    dist.send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                    )
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            self._sync_timing()
            backward_sec += time.monotonic() - backward_started_at
            records[chunk_idx] = None

        fill_started_at = time.monotonic()
        for chunk_idx in range(warmup):
            forward_chunk(chunk_idx)
        fill_sec += time.monotonic() - fill_started_at

        remaining = chunk_count - warmup
        steady_started_at = time.monotonic()
        for offset in range(remaining):
            forward_chunk(offset + warmup)
            backward_chunk(offset)
        steady_sec += time.monotonic() - steady_started_at

        drain_started_at = time.monotonic()
        for chunk_idx in range(remaining, chunk_count):
            backward_chunk(chunk_idx)
        drain_sec += time.monotonic() - drain_started_at

        all_finite = all(finite_flags)
        finite_payload = [all_finite]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(finite_payload, src=(self.pp_size - 1) * self.tp_size)
        return {
            "finite": bool(finite_payload[0]),
            "loss_value": sum(loss_values),
            "forward_sec": forward_sec,
            "backward_sec": backward_sec,
            "fill_sec": fill_sec,
            "steady_sec": steady_sec,
            "drain_sec": drain_sec,
        }
