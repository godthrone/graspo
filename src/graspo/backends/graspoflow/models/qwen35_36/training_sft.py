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
from graspo.core.data import SFTTokenized
from graspo.ripple.multimodal.contract import assert_sft_batch_has_multimodal


class _Qwen35SFTTrainingMethods:
    """Mixin: SFT training/batch optimization methods for Qwen35Adapter."""

    # ── TP-only SFT training ─────────────────────────────────────────────────

    def train_batch_sft(
        self,
        sft_batches: list[SFTTokenized],
        *,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """SFT 训练：对一批 ``SFTTokenized`` 样本执行 forward → cross-entropy loss → backward。

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
        for start in range(0, len(sft_batches), forward_batch_size):
            batch_items = sft_batches[start : start + forward_batch_size]
            micro_batch = _collate_sft_batch(
                batch_items,
                self.device,
                adapter=self,
                max_seq_length=int(self.config.data.max_prompt_length),
            )
            self.optimizer.zero_grad(set_to_none=True)
            self._sync_timing()
            forward_started_at = time.monotonic()
            multimodal_inputs = micro_batch.get("multimodal_inputs")
            # 防呆（§2.3）：样本含媒体但 batch 无 multimodal_inputs → 硬失败。
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
        sft_batches: list[SFTTokenized],
        *,
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
        for start in range(0, len(sft_batches), forward_batch_size):
            batch_items = sft_batches[start : start + forward_batch_size]
            micro_batch = _collate_sft_batch(
                batch_items,
                self.device,
                adapter=self,
                max_seq_length=int(self.config.data.max_prompt_length),
            )
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            self._sync_timing()
            forward_started_at = time.monotonic()
            multimodal_inputs = micro_batch.get("multimodal_inputs")
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


def _collate_sft_batch(
    items: list[SFTTokenized],
    device: torch.device,
    *,
    adapter: Any,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """将多个 ``SFTTokenized`` 样本拼接为 micro-batch（统一入口，自动分派）。

    纯文本 → ``_collate_sft_text_batch``（预 tokenized 张量 padding）
    多模态 → ``_collate_sft_multimodal_batch``（复用 RL 的 ``_encode_multimodal_rows``）
    """
    if any(item.deferred_multimodal is not None for item in items):
        if any(item.deferred_multimodal is None for item in items):
            raise ValueError(
                "Cannot mix multimodal and text-only samples in one micro-batch. "
                "Ensure all samples in a batch have the same type."
            )
        return _collate_sft_multimodal_batch(items, device, adapter=adapter, max_seq_length=max_seq_length)
    return _collate_sft_text_batch(items, device)


def _collate_sft_text_batch(
    items: list[SFTTokenized], device: torch.device
) -> dict[str, Any]:
    """纯文本 SFT batch：pad 预 tokenized 张量。"""
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [item.input_ids for item in items], batch_first=True, padding_value=0
    ).to(device)
    labels = pad_sequence(
        [item.labels for item in items], batch_first=True, padding_value=-100
    ).to(device)
    attention_mask = (
        pad_sequence(
            [item.attention_mask for item in items], batch_first=True, padding_value=0
        )
        .bool()
        .to(device)
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "metadata": [item.metadata for item in items],
    }


def _collate_sft_multimodal_batch(
    items: list[SFTTokenized],
    device: torch.device,
    *,
    adapter: Any,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """多模态 SFT batch：调用 ``_encode_multimodal_rows`` 一次性编码，复用 RL 路径。

    RL 的 ``generate_sample_groups`` → ``_encode_multimodal_rows`` 是同一次调用产出
    ``input_ids`` + ``pixel_values`` + ``image_grid_thw``。SFT 对齐：在 collate 阶段
    一次性编码，而非在 ``sft_tokenize`` 中预编码、训练时再二次编码。
    """
    tokenizer = adapter.tokenizer

    # 1. 构建 rows（与 RL 的 _multimodal_row_from_sample 格式一致）
    rows: list[dict[str, Any]] = []
    target_texts: list[str] = []
    for item in items:
        assert item.deferred_multimodal is not None
        deferred = item.deferred_multimodal
        media_types = _count_media_types_from_messages(deferred.prompt_messages)
        row: dict[str, Any] = {
            "messages": deferred.prompt_messages,
            "media": media_types,
        }
        tools = deferred.tools
        if tools is not None:
            row["tools"] = tools
        rows.append(row)
        target_texts.append(deferred.target_text)

    # 2. 单次编码 prompt（复用 RL 路径，一次性产出 input_ids + pixel_values）
    # SFT 禁用 thinking：与 sft_tokenize_text 的 setdefault("enable_thinking", False) 一致
    chat_template_kwargs = dict(adapter.config.model.chat_template_kwargs or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    encoded = adapter._encode_multimodal_rows(
        rows,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_ids = encoded["input_ids"].to(device)  # (batch, padded_prompt_len)
    prompt_mask = encoded["attention_mask"].to(device)  # (batch, padded_prompt_len)
    multimodal_inputs = adapter._multimodal_inputs_to_device(encoded)

    # 3. 编码 target text（纯文本，tokenizer 即可）
    eos_id = int(tokenizer.eos_token_id)
    response_ids_list = [
        tokenizer.encode(text, add_special_tokens=False) + [eos_id]
        for text in target_texts
    ]

    # 4. 拼接 prompt + response，构建 input_ids / labels / attention_mask
    batch_size = len(items)
    prompt_len_padded = int(prompt_ids.shape[1])
    max_response_len = max(len(ids) for ids in response_ids_list)
    total_len = prompt_len_padded + max_response_len

    input_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        actual_prompt_len = int(prompt_mask[i].sum().item())
        # 复制 prompt tokens（仅有效部分，padding 区域保持 0）
        input_ids[i, :actual_prompt_len] = prompt_ids[i, :actual_prompt_len]
        attention_mask[i, :actual_prompt_len] = True
        # 追加 response tokens
        r_ids = response_ids_list[i]
        r_len = len(r_ids)
        input_ids[i, actual_prompt_len : actual_prompt_len + r_len] = torch.tensor(
            r_ids, dtype=torch.long, device=device
        )
        labels[i, actual_prompt_len : actual_prompt_len + r_len] = torch.tensor(
            r_ids, dtype=torch.long, device=device
        )
        attention_mask[i, actual_prompt_len : actual_prompt_len + r_len] = True

    # 5. 截断
    if total_len > max_seq_length:
        input_ids = input_ids[:, :max_seq_length]
        labels = labels[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "multimodal_inputs": multimodal_inputs,
        "metadata": [item.metadata for item in items],
    }


def _count_media_types_from_messages(
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    """从 messages 中统计媒体类型数量（与 ``_media_counts`` 格式一致）。"""
    counts: dict[str, int] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            media_type = block.get("type")
            if media_type in ("image", "image_url"):
                counts["image"] = counts.get("image", 0) + 1
            elif media_type in ("video", "video_url"):
                counts["video"] = counts.get("video", 0) + 1
    return counts
