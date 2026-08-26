"""Qwen3.5/3.6 adapter — SFT training methods (TP+DP+PP)."""

import os
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.adapters.models.qwen35_36.helpers import collate_sft_batch
from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.flow.parallel.pipeline_comm import PipelineComm, wait_all
from graspo.flow.parallel.scheduling import build_scheduler
from graspo.flow.parallel.tensor_utils import (
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
)
from graspo.ripple.data import SFTTokenized
from graspo.ripple.multimodal.contract import assert_sft_batch_has_multimodal


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


class _Qwen35SFTTrainingMethods:
    def _compute_sft_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """从 hidden states 计算 SFT cross-entropy loss（设施层不持有算法分派）。

        复用 RL 的共享实现 ``masked_token_log_probs_from_hidden``（宪法 §1.4 单一真相源）：
        分块 logsumexp，不物化 (B,S,V) 全量 logits（v0.24.0 修复——此前 SFT 物化
        全词表 logits 导致显存不随 TP 分摊、TP=4 仍 OOM）。
        语义与 ``F.cross_entropy(ignore_index=-100, reduction="mean")`` 一致（全局均值）。
        """
        from graspo.ripple.loss import masked_token_log_probs_from_hidden

        norm = self.model.norm if hasattr(self.model, "norm") else None
        lm_head = self.model.lm_head if hasattr(self.model, "lm_head") else None
        if norm is None or lm_head is None:
            raise RuntimeError("SFT loss requires model.norm and model.lm_head")
        normalized = norm(hidden_states)
        # 因果语言模型 shift：位置 t 的 hidden 预测位置 t+1 的 token
        shift_hidden = normalized[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs = masked_token_log_probs_from_hidden(
            shift_hidden,  # 保持 bf16，函数内部分块处理，按需转换
            lm_head.weight,
            shift_labels,
            ignore_index=-100,
        )
        mask = shift_labels != -100
        return -(log_probs * mask).sum() / mask.sum().clamp_min(1)

    """Mixin: SFT training/batch optimization methods for Qwen35Adapter."""

    # ── TP-only SFT training ─────────────────────────────────────────────────

    def train_batch_sft(
        self,
        sft_batches: list[SFTTokenized],
        *,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """SFT 训练：对一批 ``SFTTokenized`` 样本执行 forward → cross-entropy loss → backward。

        支持梯度累积：当 ``micro_batch_size < len(sft_batches)`` 时，将外批拆分为多个
        micro-batch 逐个 forward/backward，梯度在 micro-batch 间累加，最后统一
        ``optimizer.step()``。有效 batch size = ``len(sft_batches)``。

        Args:
            sft_batches: ``sft_tokenize_text`` / ``sft_tokenize_multimodal`` 产出的
                ``SFTTokenized`` 列表。
            max_grad_norm: 梯度裁剪阈值
        """
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        if self._is_pipeline_parallel():
            return self._pipeline_train_batch_sft(
                sft_batches,
                max_grad_norm=max_grad_norm,
            )
        self.model.train()
        if bool(self.config.graspoflow.empty_cache_before_train) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        micro_batch_size = max(1, int(self.config.graspoflow.micro_batch_size))
        # 梯度累积：有效 batch = gradient_accumulation_micro_batches，micro-batch = micro_batch_size
        # zero_grad 只调一次，所有 micro-batch 的梯度累加后统一 step。
        num_micro_batches = max(1, (len(sft_batches) + micro_batch_size - 1) // micro_batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        train_batch_started_at = time.monotonic()
        round_secs: list[float] = []
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        micro_batch_count = 0
        valid_micro_batches = 0  # 实际贡献梯度的 micro-batch 数
        for start in range(0, len(sft_batches), micro_batch_size):
            batch_items = sft_batches[start : start + micro_batch_size]
            micro_batch = collate_sft_batch(
                batch_items,
                self.device,
                adapter=self,
                max_seq_length=int(self.config.data.max_prompt_length),
            )
            self._sync_timing()
            forward_started_at = time.monotonic()
            multimodal_inputs = micro_batch.get("multimodal_inputs")
            # 防呆：样本含媒体但 batch 无 multimodal_inputs → 硬失败。
            # 与 RL 路径的 assert_rl_training_has_multimodal 对应，
            # 防止多模态样本静默走纯文本 forward（图像丢失）。
            assert_sft_batch_has_multimodal(
                any(item.deferred_multimodal is not None for item in batch_items),
                multimodal_inputs,
                context="SFT training forward",
            )
            if multimodal_inputs is not None:
                if not isinstance(self.model, Qwen35HybridTextModel):
                    raise ValueError("multimodal SFT batch for a non-multimodal model")
                hidden = self.model._forward_hidden(
                    micro_batch["input_ids"],
                    attention_mask=micro_batch["attention_mask"],
                    multimodal_inputs=multimodal_inputs,
                )
            else:
                hidden = self.model._forward_hidden(
                    micro_batch["input_ids"],
                    attention_mask=micro_batch["attention_mask"],
                )
            assert isinstance(hidden, torch.Tensor)
            self._sync_timing()
            micro_batch_forward_sec += time.monotonic() - forward_started_at
            loss = self._compute_sft_loss(hidden, micro_batch["labels"])
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
        self._train_batch_call_index += 1

        lora_norm_after = self.model.lora_parameter_norm()
        metrics = {
            "optimized": optimizer_steps > 0,
            "sft_batch_count": len(sft_batches),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / micro_batch_count if micro_batch_count else None,
            "grad_norm_mean": grad_norm_sum,
            "nonzero_grad_count": nonzero_grad_count,
            "lora_norm_before": lora_norm_before,
            "lora_norm_after": lora_norm_after,
            "lora_norm_delta": lora_norm_after - lora_norm_before,
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "current_lr": self._current_lr(),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("sft_train_batch_after", {"metrics": metrics})
        return metrics

    # ── PP SFT training (1F1B) ─────────────────────────────────────────────────

    def _pipeline_train_batch_sft(
        self,
        sft_batches: list[SFTTokenized],
        *,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """PP SFT 训练 — 1F1B 调度，梯度累积。

        将 ``sft_batches`` 按 ``micro_batch_size`` 拆分为 micro-batch，
        通过 1F1B fill/steady/drain 三阶段流水线执行 forward/backward。
        所有 micro-batch 的梯度累加后统一 ``optimizer.step()``。
        有效 batch size = ``len(sft_batches)``。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        self.model.train()
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        micro_batch_size = max(1, int(self.config.graspoflow.micro_batch_size))
        full_batch_size = max(1, len(sft_batches))
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

        # 预 collate 所有 micro-batch（1F1B 需要提前知道所有 chunk）
        chunk_batches: list[dict[str, Any]] = []
        for start in range(0, len(sft_batches), micro_batch_size):
            batch_items = sft_batches[start : start + micro_batch_size]
            chunk_batches.append(
                collate_sft_batch(
                    batch_items,
                    self.device,
                    adapter=self,
                    max_seq_length=int(self.config.data.max_prompt_length),
                )
            )

        chunk_count = len(chunk_batches)
        if chunk_count == 0:
            self._train_batch_call_index += 1
            return self._aggregate_rank_metrics({"optimized": False, "optimizer_steps": 0})

        # 异步 P2P 通信管道（Flink 风格"调度与计算分离"）+ 可插拔调度策略
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

        # 梯度累积：zero_grad 只调一次，所有 chunk 的梯度累加后统一 step
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        def forward_chunk(chunk_idx: int) -> None:
            nonlocal micro_batch_forward_sec
            mb = chunk_batches[chunk_idx]
            self._sync_timing()
            t0 = time.monotonic()
            stage_output, stage_input, send_work = self._pipeline_forward_for_sft(
                mb["input_ids"],
                mb["attention_mask"],
                multimodal_inputs=mb.get("multimodal_inputs"),
                timing=stage_timing,
                comm=comm,
                tag=chunk_idx,
            )
            self._sync_timing()
            micro_batch_forward_sec += time.monotonic() - t0
            if send_work is not None:
                send_works.append(send_work)
            loss: torch.Tensor | None = None
            finite = True
            loss_value = 0.0
            if self.pp_rank == self.pp_size - 1:
                assert stage_output is not None
                chunk_loss = self._compute_sft_loss(stage_output, mb["labels"])
                finite = bool(torch.isfinite(chunk_loss).detach().cpu())
                # 梯度累积：loss 按 chunk_size / full_batch_size 加权
                weight = float(mb["input_ids"].shape[0]) / full_batch_size
                loss = chunk_loss * weight if finite else None
                loss_value = float(chunk_loss.detach().cpu()) * weight if finite else 0.0
            records[chunk_idx] = {
                "stage_output": stage_output,
                "stage_input": stage_input,
                "loss": loss,
            }
            finite_flags[chunk_idx] = finite
            loss_values[chunk_idx] = loss_value

        def backward_chunk(chunk_idx: int) -> None:
            nonlocal backward_sec
            record = records[chunk_idx]
            if record is None:
                raise RuntimeError("1F1B attempted backward before forward")
            self._sync_timing()
            t0 = time.monotonic()
            if self.pp_rank == self.pp_size - 1:
                stage_input = record["stage_input"]
                loss = record["loss"]
                if loss is not None:
                    _log_cuda_mem("before_backward_pp")
                    loss.backward()
                    _log_cuda_mem("after_backward_pp")
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
            else:
                stage_output = record["stage_output"]
                assert stage_output is not None
                grad_output = torch.empty_like(stage_output)
                recv_work = comm.bwd_recv(
                    grad_output,
                    src=int(self.tp_state.next_pp_rank or 0),
                    tag=chunk_count + chunk_idx,
                )
                comm.wait(recv_work)  # 阻塞直到梯度到达（上游异步 send，不会死锁）
                stage_output.backward(grad_output)
                stage_input = record["stage_input"]
                if stage_input is not None:
                    grad = (
                        stage_input.grad
                        if stage_input.grad is not None
                        else torch.zeros_like(stage_input)
                    )
                    work = comm.bwd_send(
                        grad.contiguous(),
                        dst=int(self.tp_state.prev_pp_rank or 0),
                        tag=chunk_count + chunk_idx,
                    )
                    if work is not None:
                        send_works.append(work)
            self._sync_timing()
            backward_sec += time.monotonic() - t0
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
        # 同步所有异步发送（数据已被下游消费），保证 buffer 生命周期安全
        wait_all(send_works)

        all_finite = all(finite_flags)
        finite_payload = [all_finite]
        dist.broadcast_object_list(finite_payload, src=(self.pp_size - 1) * self.tp_size)

        if bool(finite_payload[0]):
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
            # 所有 chunk 梯度已累加，统一 clip + step
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                if trainable_params
                else torch.tensor(0.0)
            )
            grad_norm_sum = float(grad_norm.detach().float().cpu())
            self._sync_timing()
            t0 = time.monotonic()
            if self.optimizer is not None:
                _log_cuda_mem("before_optimizer_step_pp")
                self.optimizer.step()
                _log_cuda_mem("after_optimizer_step_pp")
            self._sync_timing()
            optimizer_step_sec = time.monotonic() - t0
            if self.scheduler is not None and self.optimizer is not None:
                self.scheduler.step()
            optimizer_steps = 1
            nonzero_grad_count = self.model.nonzero_lora_grad_count()
            loss_sum = sum(loss_values)
            micro_batch_count = chunk_count
        else:
            skipped_nonfinite = chunk_count

        self._train_batch_call_index += 1
        lora_norm_after = self.model.lora_parameter_norm()
        metrics = {
            "optimized": optimizer_steps > 0,
            "sft_batch_count": len(sft_batches),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / micro_batch_count if micro_batch_count else None,
            "grad_norm_mean": grad_norm_sum,
            "nonzero_grad_count": nonzero_grad_count,
            "lora_norm_before": lora_norm_before,
            "lora_norm_after": lora_norm_after,
            "lora_norm_delta": lora_norm_after - lora_norm_before,
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "pp_size": self.pp_size,
            "pp_schedule": sched_stats.get("pp_schedule", "one_f_one_b"),
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            "pipeline_fill_sec": fill_sec,
            "pipeline_steady_sec": steady_sec,
            "pipeline_drain_sec": drain_sec,
            "pipeline_forward_sec": float(sched_stats.get("pipeline_forward_sec", 0.0)),
            "pipeline_backward_sec": float(sched_stats.get("pipeline_backward_sec", 0.0)),
            "current_lr": self._current_lr(),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("pipeline_sft_train_batch_after", {"metrics": metrics})
        return metrics

    def _pipeline_forward_for_sft(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        timing: dict[str, float | int] | None = None,
        comm: PipelineComm | None = None,
        tag: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, Any | None]:
        """PP forward pass for SFT — delegates to the unified PP forward.

        SP / async-P2P / position_ids / tag 语义由 :meth:`_pipeline_forward_hidden`
        （pipeline_forward.py）集中维护，避免多套 PP forward 漂移。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        output, _present, stage_input, send_work = self._pipeline_forward_hidden(
            input_ids=input_ids,
            hidden_states=None,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            multimodal_inputs=multimodal_inputs,
            position_input_ids=input_ids,
            apply_lm_head=False,
            timing=timing,
            comm=comm,
            tag=tag,
            debug_label="fwd",
        )
        return output, stage_input, send_work


# ── SFT batch collation ──────────────────────────────────────────────────────
# collate_sft_batch / collate_sft_text_batch / collate_sft_multimodal_batch /
# count_media_types_from_messages 已迁入本目录 helpers.py（纯函数，受 mypy 检查）。
