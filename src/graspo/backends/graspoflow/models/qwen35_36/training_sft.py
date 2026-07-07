"""Qwen3.5/3.6 adapter — SFT training methods (TP-only, PP simple)."""

import time
from typing import Any

import torch
import torch.distributed as dist

from graspo.backends.graspoflow.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.backends.graspoflow.tensor_utils import (
    _add_pipeline_stage_timing,
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
)


class _Qwen35SFTTrainingMethods:
    """Mixin: SFT training/batch optimization methods for Qwen35Adapter."""

    # ── TP-only SFT training ─────────────────────────────────────────────────

    def train_batch_sft(
        self,
        sft_batches: list[dict[str, Any]],
        *,
        optimize_iterations_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """SFT 训练：对一批 tokenized 样本执行 forward → cross-entropy loss → backward。

        Args:
            sft_batches: 来自 ``sft_tokenize()`` 的 dict 列表，每项包含
                ``input_ids``, ``labels``, ``attention_mask``, 可选的 ``multimodal_inputs``
            optimize_iterations_per_step: 梯度累积步数
            max_grad_norm: 梯度裁剪阈值
        """
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        if self._is_pipeline_parallel():
            return self._pipeline_train_batch_sft(
                sft_batches,
                optimize_iterations_per_step=optimize_iterations_per_step,
                max_grad_norm=max_grad_norm,
            )
        self.model.train()
        if bool(self.config.graspoflow.empty_cache_before_train) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        forward_batch_size = max(1, int(self.config.graspoflow.forward_batch_size))
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
        for _ in range(optimize_iterations_per_step):
            round_started_at = time.monotonic()
            for start in range(0, len(sft_batches), forward_batch_size):
                batch_items = sft_batches[start : start + forward_batch_size]
                micro_batch = _collate_sft_batch(batch_items, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                self._sync_timing()
                forward_started_at = time.monotonic()
                multimodal_inputs = None
                multimodal_rows = micro_batch.get("_multimodal_rows")
                if multimodal_rows:
                    multimodal_inputs = self._multimodal_inputs_from_metadata(
                        {"_multimodal_rows": multimodal_rows},
                        batch_size=int(micro_batch["input_ids"].shape[0]),
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
                loss = self.compute_loss(hidden, micro_batch)
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    continue
                self._sync_timing()
                backward_started_at = time.monotonic()
                loss.backward()
                from graspo.backends.graspoflow.lora import _sync_nonsharded_lora_grads
                from graspo.backends.graspoflow.tensor_utils import _TENSOR_PARALLEL_GROUP

                if _TENSOR_PARALLEL_GROUP is not None:
                    _sync_nonsharded_lora_grads(self.model, _TENSOR_PARALLEL_GROUP)
                self._sync_timing()
                backward_sec += time.monotonic() - backward_started_at
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [param for param in self.model.parameters() if param.requires_grad],
                    max_grad_norm,
                )
                self._sync_timing()
                optimizer_started_at = time.monotonic()
                self.optimizer.step()
                self._sync_timing()
                optimizer_step_sec += time.monotonic() - optimizer_started_at
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer_steps += 1
                micro_batch_count += 1
                loss_sum += float(loss.detach().cpu())
                grad_norm_sum += float(grad_norm.detach().float().cpu())
                nonzero_grad_count += self.model.nonzero_lora_grad_count()
            round_secs.append(time.monotonic() - round_started_at)
        self._train_batch_call_index += 1

        lora_norm_after = self.model.lora_parameter_norm()
        metrics = {
            "optimized": optimizer_steps > 0,
            "sft_batch_count": len(sft_batches),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / optimizer_steps if optimizer_steps else None,
            "grad_norm_mean": grad_norm_sum / optimizer_steps if optimizer_steps else None,
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

    # ── PP SFT training ──────────────────────────────────────────────────────

    def _pipeline_train_batch_sft(
        self,
        sft_batches: list[dict[str, Any]],
        *,
        optimize_iterations_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """PP SFT 训练 — 复用 _pipeline_forward_for_sft，替换 loss 为 cross-entropy。"""
        self.model.train()
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        forward_batch_size = max(1, int(self.config.graspoflow.forward_batch_size))
        train_batch_started_at = time.monotonic()
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        round_secs: list[float] = []
        micro_batch_count = 0
        stage_timing = _new_pipeline_stage_timing()
        for _ in range(optimize_iterations_per_step):
            round_started_at = time.monotonic()
            for start in range(0, len(sft_batches), forward_batch_size):
                batch_items = sft_batches[start : start + forward_batch_size]
                micro_batch = _collate_sft_batch(batch_items, self.device)
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                self._sync_timing()
                forward_started_at = time.monotonic()
                multimodal_inputs = None
                multimodal_rows = micro_batch.get("_multimodal_rows")
                if multimodal_rows:
                    multimodal_inputs = self._multimodal_inputs_from_metadata(
                        {"_multimodal_rows": multimodal_rows},
                        batch_size=int(micro_batch["input_ids"].shape[0]),
                    )
                stage_output, stage_input = self._pipeline_forward_for_sft(
                    micro_batch["input_ids"],
                    micro_batch["attention_mask"],
                    multimodal_inputs=multimodal_inputs,
                    timing=stage_timing,
                )
                self._sync_timing()
                micro_batch_forward_sec += time.monotonic() - forward_started_at
                loss: torch.Tensor | None = None
                loss_value = 0.0
                if self.pp_rank == self.pp_size - 1:
                    assert stage_output is not None
                    loss = self.compute_loss(stage_output, micro_batch)
                    finite = bool(torch.isfinite(loss).detach().cpu())
                    loss_value = float(loss.detach().cpu())
                else:
                    finite = True
                finite_payload = [finite]
                dist.broadcast_object_list(finite_payload, src=(self.pp_size - 1) * self.tp_size)
                if not bool(finite_payload[0]):
                    skipped_nonfinite += 1
                    continue
                self._sync_timing()
                backward_started_at = time.monotonic()
                if self.pp_rank == self.pp_size - 1:
                    assert loss is not None
                    loss.backward()
                    if stage_input is not None and stage_input.grad is not None:
                        dist.send(
                            stage_input.grad.contiguous(),
                            dst=int(self.tp_state.prev_pp_rank),
                        )
                else:
                    assert stage_output is not None
                    grad_output = torch.empty_like(stage_output)
                    dist.recv(grad_output, src=int(self.tp_state.next_pp_rank))
                    stage_output.backward(grad_output)
                    if stage_input is not None and stage_input.grad is not None:
                        dist.send(
                            stage_input.grad.contiguous(),
                            dst=int(self.tp_state.prev_pp_rank),
                        )
                self._sync_timing()
                backward_sec += time.monotonic() - backward_started_at
                trainable_params = [
                    param for param in self.model.parameters() if param.requires_grad
                ]
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    if trainable_params
                    else torch.tensor(0.0)
                )
                self._sync_timing()
                optimizer_started_at = time.monotonic()
                if self.optimizer is not None:
                    self.optimizer.step()
                self._sync_timing()
                optimizer_step_sec += time.monotonic() - optimizer_started_at
                if self.scheduler is not None and self.optimizer is not None:
                    self.scheduler.step()
                optimizer_steps += 1
                micro_batch_count += 1
                loss_payload = [loss_value]
                dist.broadcast_object_list(loss_payload, src=(self.pp_size - 1) * self.tp_size)
                loss_sum += float(loss_payload[0])
                grad_norm_sum += float(grad_norm.detach().float().cpu())
                nonzero_grad_count += self.model.nonzero_lora_grad_count()
            round_secs.append(time.monotonic() - round_started_at)
        self._train_batch_call_index += 1
        lora_norm_after = self.model.lora_parameter_norm()
        metrics = {
            "optimized": optimizer_steps > 0,
            "sft_batch_count": len(sft_batches),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / optimizer_steps if optimizer_steps else None,
            "grad_norm_mean": grad_norm_sum / optimizer_steps if optimizer_steps else None,
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
            "pp_schedule": "simple",
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """PP forward pass for SFT — 与 _pipeline_forward_for_training 相同，
        仅替换 input 参数名以匹配 SFT 的 batch 格式。
        """
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        batch = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        hidden_size = int(self.model.config.hidden_size)
        dtype = next(self.model.parameters()).dtype
        stage_input: torch.Tensor | None = None
        if self.pp_rank == 0:
            compute_started_at = time.monotonic()
            output = self.model.forward_stage(
                None,
                input_ids,
                attention_mask,
                past_key_values=None,
                use_cache=False,
                multimodal_inputs=multimodal_inputs,
                position_input_ids=input_ids,
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
                position_input_ids=input_ids,
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


# ── SFT batch collation ──────────────────────────────────────────────────────


def _collate_sft_batch(items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """将多个 SFT tokenized 样本拼接为 micro-batch。

    每个 item 来自 ``sft_tokenize()``，包含 ``input_ids``, ``labels``,
    ``attention_mask``, ``prompt_len``, ``metadata``,
    可选的 ``_multimodal_row``。
    """
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [item["input_ids"] for item in items], batch_first=True, padding_value=0
    ).to(device)
    labels = pad_sequence(
        [item["labels"] for item in items], batch_first=True, padding_value=-100
    ).to(device)
    attention_mask = (
        pad_sequence([item["attention_mask"] for item in items], batch_first=True, padding_value=0)
        .bool()
        .to(device)
    )

    micro_batch: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "metadata": [item.get("metadata", {}) for item in items],
    }

    # 多模态：收集 _multimodal_row 并按 row 组织
    multimodal_rows = [item["_multimodal_row"] for item in items if "_multimodal_row" in item]
    if multimodal_rows:
        micro_batch["_multimodal_rows"] = multimodal_rows

    return micro_batch
