"""Qwen3 adapter — dense attention, text-only.

Lighter than Qwen35Adapter — no visual tower, no multimodal.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from graspo.backends.graspoflow.lora import native_qwen_lora_available_targets
from graspo.backends.graspoflow.lora_io import load_peft_adapter_into_native_model
from graspo.backends.graspoflow.models.qwen3.model import (
    build_native_qwen_model,
)
from graspo.backends.graspoflow.models.qwen3.ops import build_qwen3_ops
from graspo.backends.graspoflow.placement import (
    build_placement_plan,
)
from graspo.backends.graspoflow.runtime import NativeGeneration
from graspo.backends.graspoflow.tensor_utils import (
    SafetensorIndex,
    _broadcast_and_pad_finished,
    _left_pad_token_rows,
    _next_token_from_logits,
    _resolve_dtype,
    collate_experiences,
)
from graspo.backends.graspoflow.tool_parser import parse_qwen_tool_completion
from graspo.backends.graspoflow.transformer_adapter import TransformerAdapter
from graspo.core.buffer import Experience
from graspo.core.completion import ParsedCompletion
from graspo.trainer.lora import resolve_lora_target_modules


class Qwen3Adapter(TransformerAdapter):
    """Qwen3 adapter for GraspoFlow.

    Supports dense attention, text-only rollout, TP-only and PP training.
    """

    completion_parser_name = "qwen_tool_call"

    def _load_model(self, hf_config: Any, model_path: Path) -> None:
        torch_dtype = _resolve_dtype(self.config.model.torch_dtype)
        loader = SafetensorIndex(model_path)
        lora_targets = resolve_lora_target_modules(
            self.config.lora.target_modules or (self.config.lora.target_preset,),
            available=native_qwen_lora_available_targets(hf_config),
        )
        self.placement = build_placement_plan(
            strategy=self.config.graspoflow.placement_strategy,
            model_family=hf_config.family,
            num_hidden_layers=int(hf_config.num_hidden_layers),
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            layer_types=list(getattr(hf_config, "layer_types", []) or []),
            manual_ranges=[list(r) for r in self.config.graspoflow.layer_ranges]
            if self.config.graspoflow.layer_ranges is not None
            else None,
        )
        self.model = build_native_qwen_model(
            hf_config=hf_config,
            loader=loader,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            placement=self.placement,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            lora_targets=set(lora_targets.resolved),
            gradient_checkpointing=bool(self.config.model.gradient_checkpointing),
            torch_dtype=torch_dtype,
            device=self.device,
        )
        assert self.model is not None
        missing_lora_targets = sorted(
            target
            for target in set(lora_targets.resolved)
            - set(self.model.enabled_lora_target_names())
        )
        if missing_lora_targets:
            raise ValueError(
                "Resolved LoRA target(s) are not implemented by this model yet: "
                + ", ".join(missing_lora_targets)
            )
        self.model.train(False)
        if self.config.lora.adapter_path:
            load_peft_adapter_into_native_model(
                self.model,
                self.config.lora.adapter_path,
                base_model_path=str(model_path),
            )

    def _build_ops(self) -> None:
        self._ops = build_qwen3_ops(
            model=self.model,
            tp_state=self.tp_state,
            tp_size=self.tp_size,
        )

    # ── Generation (TP-only) ────────────────────────────────────────────────

    def generate_groups(
        self,
        *,
        message_batches=None,
        tool_batches=None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[NativeGeneration]:
        self._require_ready()
        assert self.model is not None
        assert self.tokenizer is not None
        if not message_batches:
            return []
        self.model.eval()
        tokenize_started_at = time.monotonic()
        prompt_texts = [
            self._format_messages(messages, chat_template_kwargs, tools=tools)
            for messages, tools in zip(
                message_batches,
                tool_batches if tool_batches else [None] * len(message_batches),
                strict=True,
            )
        ]
        encoded = self.tokenizer(
            prompt_texts,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
        )
        tokenize_sec = time.monotonic() - tokenize_started_at
        rollout_started_at = time.monotonic()
        eos_token_id = int(self.tokenizer.eos_token_id)
        pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else eos_token_id
        )
        prompt_input_ids, prompt_lens = _left_pad_token_rows(
            encoded["input_ids"],
            pad_token_id=pad_token_id,
            device=self.device,
        )
        prompt_len = int(prompt_input_ids.shape[1])
        use_kv_cache = bool(self.config.graspoflow.use_kv_cache_for_rollout) and bool(
            getattr(self.model, "supports_kv_cache", True)
        )
        requested_prompt_queue_size = len(message_batches)
        prompt_chunk_size = self._shared_rollout_prompt_chunk_size(
            prompt_len=prompt_len,
            requested_prompt_count=requested_prompt_queue_size,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        prompt_chunk_size = max(1, min(prompt_chunk_size, requested_prompt_queue_size))
        all_generations: list[NativeGeneration] = []
        all_chunk_timings: list[dict[str, float | int]] = []
        all_sequence_chunks: list[torch.Tensor] = []
        for prompt_start in range(0, requested_prompt_queue_size, prompt_chunk_size):
            prompt_stop = min(prompt_start + prompt_chunk_size, requested_prompt_queue_size)
            prompt_chunk = prompt_input_ids[prompt_start:prompt_stop]
            chunk_prompt_count = int(prompt_chunk.shape[0])
            flat_prompt_input_ids = prompt_chunk.repeat_interleave(rollout_group_size, dim=0)
            chunk_generation_micro_batch_size = self._shared_generation_micro_batch_size(
                prompt_len=prompt_len,
                rollout_group_size=chunk_prompt_count * rollout_group_size,
                max_new_tokens=max_new_tokens,
                use_kv_cache=use_kv_cache,
            )
            sequence_chunks: list[torch.Tensor] = []
            chunk_timings: list[dict[str, float | int]] = []

            with torch.no_grad():
                for start in range(
                    0, flat_prompt_input_ids.shape[0], chunk_generation_micro_batch_size
                ):
                    current_batch = flat_prompt_input_ids[
                        start : start + chunk_generation_micro_batch_size
                    ]
                    sequences = current_batch.clone()
                    finished = torch.zeros(
                        sequences.shape[0], dtype=torch.bool, device=self.device
                    )
                    if use_kv_cache:
                        sequences, chunk_timing = self._generate_group_with_kv_cache(
                            sequences=sequences,
                            finished=finished,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        sequences, chunk_timing = self._generate_group_full_forward(
                            sequences=sequences,
                            finished=finished,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    sequence_chunks.append(sequences)
                    chunk_timings.append(chunk_timing)

            flat_sequences = pad_sequence(
                [row for chunk in sequence_chunks for row in chunk],
                batch_first=True,
                padding_value=pad_token_id,
            )
            all_sequence_chunks.append(flat_sequences)
            all_chunk_timings.extend(chunk_timings)
            for local_prompt_idx in range(chunk_prompt_count):
                row_start = local_prompt_idx * rollout_group_size
                row_stop = row_start + rollout_group_size
                prompt_sequences = flat_sequences[row_start:row_stop]
                all_generations.append(
                    self._generation_from_sequences(
                        sequences=prompt_sequences,
                        prompt_len=prompt_len,
                        prompt_lens=[int(prompt_lens[prompt_start + local_prompt_idx])],
                        pad_token_id=pad_token_id,
                        rollout_group_size=rollout_group_size,
                        requested_prompt_queue_size=requested_prompt_queue_size,
                        effective_prompt_queue_size=prompt_chunk_size,
                        use_kv_cache=use_kv_cache,
                        generation_micro_batch_size=chunk_generation_micro_batch_size,
                        split_count=len(sequence_chunks),
                        tokenize_sec=tokenize_sec / max(requested_prompt_queue_size, 1),
                        chunk_timings=chunk_timings,
                        timing_divisor=chunk_prompt_count,
                        rollout_started_at=rollout_started_at,
                    )
                )

        return all_generations

    def generate_sample_groups(
        self, **kwargs: Any
    ) -> list[NativeGeneration]:
        raise NotImplementedError("Qwen3 does not support multimodal")

    # ── TP-only generation helpers ──────────────────────────────────────────

    def _generate_group_with_kv_cache(
        self,
        *,
        sequences: torch.Tensor,
        finished: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        assert self.model is not None
        attention_mask = sequences.ne(pad_token_id)
        self._sync_timing()
        prefill_started_at = time.monotonic()
        logits, past_key_values = self.model(
            sequences, attention_mask=attention_mask, use_cache=True
        )
        self._sync_timing()
        prefill_sec = time.monotonic() - prefill_started_at
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        for _ in range(max_new_tokens):
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = _next_token_from_logits(
                logits.float()[:, -1, :],
                temperature=temperature,
                top_p=top_p,
            )
            self._sync_timing()
            sampling_sec += time.monotonic() - sampling_started_at
            next_token = _broadcast_and_pad_finished(
                next_token, finished, pad_token_id
            )
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            decode_tokens += 1
            finished |= next_token.eq(eos_token_id)
            self._sync_timing()
            stop_check_started_at = time.monotonic()
            all_finished = bool(finished.all())
            self._sync_timing()
            stop_check_sec += time.monotonic() - stop_check_started_at
            if all_finished:
                break
            attention_mask = sequences.ne(pad_token_id)
            logits, past_key_values = self.model(
                next_token.unsqueeze(1),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        self._sync_timing()
        return sequences, {
            "prefill_sec": prefill_sec,
            "decode_sec": time.monotonic() - decode_started_at,
            "decode_tokens": decode_tokens,
            "sampling_sec": sampling_sec,
            "stop_check_sec": stop_check_sec,
        }

    def _generate_group_full_forward(
        self,
        *,
        sequences: torch.Tensor,
        finished: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        assert self.model is not None
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        self._sync_timing()
        for _ in range(max_new_tokens):
            attention_mask = sequences.ne(pad_token_id)
            logits = self.model(sequences, attention_mask=attention_mask).float()[:, -1, :]
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = _next_token_from_logits(
                logits, temperature=temperature, top_p=top_p
            )
            self._sync_timing()
            sampling_sec += time.monotonic() - sampling_started_at
            next_token = _broadcast_and_pad_finished(
                next_token, finished, pad_token_id
            )
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            decode_tokens += 1
            finished |= next_token.eq(eos_token_id)
            self._sync_timing()
            stop_check_started_at = time.monotonic()
            all_finished = bool(finished.all())
            self._sync_timing()
            stop_check_sec += time.monotonic() - stop_check_started_at
            if all_finished:
                break
        self._sync_timing()
        return sequences, {
            "prefill_sec": 0.0,
            "decode_sec": time.monotonic() - decode_started_at,
            "decode_tokens": decode_tokens,
            "sampling_sec": sampling_sec,
            "stop_check_sec": stop_check_sec,
        }

    # ── Training ────────────────────────────────────────────────────────────

    def train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        self.loss_fn.policy_ratio_clip_eps = policy_ratio_clip_eps
        self.model.train()
        if (
            bool(self.config.graspoflow.empty_cache_before_train)
            and self.device.type == "cuda"
        ):
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        batch_size = int(self.config.training.optimize_prompt_batch_size)
        train_batch_started_at = time.monotonic()
        round_secs: list[float] = []
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        micro_batch_count = 0
        for optimize_round in range(optimize_times_per_step):
            round_started_at = time.monotonic()
            indices = self._shared_training_indices(
                len(experiences), optimize_round=optimize_round
            )
            for start in range(0, len(indices) - batch_size + 1, batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = collate_experiences(
                    [experiences[idx] for idx in batch_indices], self.device
                )
                self.optimizer.zero_grad(set_to_none=True)
                self._sync_timing()
                forward_started_at = time.monotonic()
                log_probs = self.model.sequence_log_probs(
                    batch.sequences, batch.attention_mask
                )
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
            "synchronize_cuda_timing": bool(
                self.config.graspoflow.synchronize_cuda_timing
            ),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("train_batch_after", {"metrics": metrics})
        return metrics

    # ── Sequence log probs ──────────────────────────────────────────────────

    def sequence_log_probs(
        self,
        sequences: Any,
        attention_mask: Any,
        metadata: Any | None = None,
    ) -> torch.Tensor:
        self._require_ready()
        assert self.model is not None
        self.model.eval()
        sequences = sequences.to(self.device)
        attention_mask = attention_mask.to(self.device).bool()
        with torch.no_grad():
            log_probs = self.model.sequence_log_probs(sequences, attention_mask)
        self._emit_rank_memory_event(
            "logprob_after",
            {
                "batch_size": int(sequences.shape[0]),
                "sequence_len": int(sequences.shape[1]),
            },
        )
        return log_probs

    # ── Parse completion ────────────────────────────────────────────────────

    def parse_completion(
        self, completion: str, sample: Any | None = None
    ) -> ParsedCompletion:
        return parse_qwen_tool_completion(
            completion,
            expect_tool_calls=bool(getattr(sample, "expects_tool_calls", False)),
            tools=getattr(sample, "tools", None),
        )
