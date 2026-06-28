"""Qwen3.5/3.6 adapter — hybrid attention + visual tower + multimodal.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from graspo.backends.graspoflow.lora import native_qwen_lora_available_targets
from graspo.backends.graspoflow.lora_io import load_peft_adapter_into_native_model
from graspo.backends.graspoflow.models.qwen3.model import (
    build_native_qwen_model,
)
from graspo.backends.graspoflow.models.qwen35_36.model import Qwen35HybridTextModel
from graspo.backends.graspoflow.models.qwen35_36.ops import build_qwen35_ops
from graspo.backends.graspoflow.multimodal import (
    _compute_multimodal_offset_tables,
    _media_counts,
    _multimodal_row_from_sample,
    _normalize_tool_batches,
    _slice_multimodal_inputs_offset,
)
from graspo.backends.graspoflow.placement import (
    build_placement_plan,
    placement_summary,
)
from graspo.backends.graspoflow.runtime import NativeGeneration
from graspo.backends.graspoflow.tensor_utils import (
    SafetensorIndex,
    _add_pipeline_stage_timing,
    _broadcast_and_pad_finished,
    _left_pad_token_rows,
    _new_pipeline_stage_timing,
    _next_token_from_logits,
    _resolve_dtype,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
    collate_experiences,
)
from graspo.backends.graspoflow.tool_parser import parse_qwen_tool_completion
from graspo.backends.graspoflow.transformer_adapter import TransformerAdapter
from graspo.core.buffer import Experience
from graspo.core.completion import ParsedCompletion
from graspo.trainer.lora import resolve_lora_target_modules


class Qwen35Adapter(TransformerAdapter):
    """Qwen3.5/3.6 adapter for GraspoFlow.

    Supports hybrid attention, visual tower, multimodal rollout, and
    TP-only / PP / TP+PP training.
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
            if not (
                target.startswith("visual.") and getattr(self.model, "visual", None) is None
            )
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
        self._ops = build_qwen35_ops(
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
        tool_batches = _normalize_tool_batches(tool_batches, len(message_batches))
        if self._is_pipeline_parallel():
            return self._pipeline_generate_groups(
                message_batches=message_batches,
                tool_batches=tool_batches,
                rollout_group_size=rollout_group_size,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                chat_template_kwargs=chat_template_kwargs,
            )
        if not message_batches:
            return []
        self.model.eval()
        tokenize_started_at = time.monotonic()
        prompt_texts = [
            self._format_messages(messages, chat_template_kwargs, tools=tools)
            for messages, tools in zip(message_batches, tool_batches, strict=True)
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
        self,
        *,
        samples=None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[NativeGeneration]:
        self._require_ready()
        if any(
            any(str(item.get("type") or "") == "video" for item in sample.media)
            for sample in samples
        ):
            raise NotImplementedError(
                "Qwen3.5-family video generation is reserved for the next phase"
            )
        if any(not sample.media for sample in samples):
            raise ValueError(
                "generate_sample_groups expects every sample to contain image/video media"
            )
        if self._is_pipeline_parallel():
            return self._pipeline_generate_multimodal_groups(
                samples=samples,
                rollout_group_size=rollout_group_size,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                chat_template_kwargs=chat_template_kwargs,
            )
        return self._generate_multimodal_groups(
            samples=samples,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
        )

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

    # ── Multimodal generation (TP-only) ─────────────────────────────────────

    def _generate_multimodal_groups(
        self,
        *,
        samples: list[Any],
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> list[NativeGeneration]:
        assert self.model is not None
        assert self.tokenizer is not None
        self.model.eval()
        N = len(samples)
        G = rollout_group_size
        tokenize_started_at = time.monotonic()

        rows: list[dict[str, Any]] = []
        per_sample_image_counts: list[int] = []
        per_sample_media_counts: list[dict[str, int]] = []
        for sample in samples:
            row = _multimodal_row_from_sample(sample)
            img_count = sum(
                1 for item in sample.media if str(item.get("type") or "") == "image"
            )
            per_sample_image_counts.append(img_count)
            per_sample_media_counts.append(_media_counts(sample.media))
            for _ in range(G):
                rows.append(row)

        encoded = self._encode_multimodal_rows(
            rows,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        tokenize_sec = time.monotonic() - tokenize_started_at

        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] > max_prompt_length:
            raise ValueError(
                f"multimodal prompt length {input_ids.shape[1]} exceeds "
                f"data.max_prompt_length={max_prompt_length}"
            )
        attention_mask = encoded["attention_mask"].to(self.device).bool()
        prompt_len = int(input_ids.shape[1])
        prompt_lens = [int(mask.sum().item()) for mask in attention_mask]
        pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        eos_token_id = int(self.tokenizer.eos_token_id)
        rollout_started_at = time.monotonic()
        use_kv_cache = bool(self.config.graspoflow.use_kv_cache_for_rollout) and bool(
            getattr(self.model, "supports_kv_cache", True)
        )
        multimodal_inputs = self._multimodal_inputs_to_device(encoded)

        image_offsets, patch_offsets, video_offsets, video_patch_offsets = (
            _compute_multimodal_offset_tables(
                per_sample_image_counts=per_sample_image_counts,
                rollout_group_size=G,
                image_grid_thw=multimodal_inputs.get("image_grid_thw"),
                pixel_values=multimodal_inputs.get("pixel_values"),
            )
        )

        requested_prompt_queue_size = N
        budget_prompt_len = max(prompt_len, max_prompt_length)
        prompt_chunk_size = self._shared_rollout_prompt_chunk_size(
            prompt_len=budget_prompt_len,
            requested_prompt_count=N,
            rollout_group_size=G,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        prompt_chunk_size = max(1, min(prompt_chunk_size, N))

        all_generations: list[NativeGeneration] = []
        all_timings: list[dict[str, float | int]] = []
        with torch.no_grad():
            for prompt_start in range(0, N, prompt_chunk_size):
                prompt_stop = min(prompt_start + prompt_chunk_size, N)
                chunk_prompt_count = prompt_stop - prompt_start
                row_start = prompt_start * G
                row_stop = prompt_stop * G
                chunk_input_ids = input_ids[row_start:row_stop]
                chunk_attention_mask = attention_mask[row_start:row_stop]
                flat_B = int(chunk_input_ids.shape[0])
                chunk_prompt_lens = prompt_lens[row_start:row_stop]

                chunk_generation_micro_batch_size = self._shared_generation_micro_batch_size(
                    prompt_len=budget_prompt_len,
                    rollout_group_size=flat_B,
                    max_new_tokens=max_new_tokens,
                    use_kv_cache=use_kv_cache,
                )

                sequence_chunks: list[torch.Tensor] = []
                chunk_timings: list[dict[str, float | int]] = []
                for start in range(0, flat_B, chunk_generation_micro_batch_size):
                    stop = min(start + chunk_generation_micro_batch_size, flat_B)
                    local_start = start
                    local_stop = stop
                    global_start = row_start + local_start
                    global_stop = row_start + local_stop

                    current_input_ids = chunk_input_ids[local_start:local_stop]
                    current_attention_mask = chunk_attention_mask[local_start:local_stop]
                    current_mm_inputs = _slice_multimodal_inputs_offset(
                        multimodal_inputs,
                        global_start,
                        global_stop,
                        image_offsets=image_offsets,
                        patch_offsets=patch_offsets,
                        video_offsets=video_offsets,
                        video_patch_offsets=video_patch_offsets,
                    )
                    finished = torch.zeros(
                        local_stop - local_start, dtype=torch.bool, device=self.device
                    )
                    if use_kv_cache:
                        seq, timing = self._generate_multimodal_with_kv_cache(
                            sequences=current_input_ids,
                            attention_mask=current_attention_mask,
                            multimodal_inputs=current_mm_inputs,
                            finished=finished,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        seq, timing = self._generate_multimodal_full_forward(
                            sequences=current_input_ids,
                            multimodal_inputs=current_mm_inputs,
                            finished=finished,
                            eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    sequence_chunks.append(seq)
                    chunk_timings.append(timing)

                flat_sequences = pad_sequence(
                    [row for chunk in sequence_chunks for row in chunk],
                    batch_first=True,
                    padding_value=pad_token_id,
                )
                all_timings.extend(chunk_timings)

                for local_prompt_idx in range(chunk_prompt_count):
                    row_start_inner = local_prompt_idx * G
                    row_stop_inner = row_start_inner + G
                    prompt_sequences = flat_sequences[
                        row_start_inner:row_stop_inner
                    ].clone()
                    all_generations.append(
                        self._generation_from_sequences(
                            sequences=prompt_sequences,
                            prompt_len=prompt_len,
                            prompt_lens=chunk_prompt_lens[
                                row_start_inner : row_start_inner + 1
                            ],
                            pad_token_id=pad_token_id,
                            rollout_group_size=G,
                            requested_prompt_queue_size=requested_prompt_queue_size,
                            effective_prompt_queue_size=prompt_chunk_size,
                            use_kv_cache=use_kv_cache,
                            generation_micro_batch_size=chunk_generation_micro_batch_size,
                            split_count=len(sequence_chunks),
                            tokenize_sec=tokenize_sec / max(N, 1),
                            chunk_timings=chunk_timings,
                            timing_divisor=chunk_prompt_count,
                            rollout_started_at=rollout_started_at,
                        )
                    )

                del flat_sequences
                del sequence_chunks
                if self.device.type == "cuda" and bool(
                    self.config.graspoflow.empty_cache_after_rollout_split
                ):
                    torch.cuda.empty_cache()

        return all_generations

    def _generate_multimodal_with_kv_cache(
        self,
        *,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        multimodal_inputs: dict[str, torch.Tensor],
        finished: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        assert isinstance(self.model, Qwen35HybridTextModel)
        self._sync_timing()
        prefill_started_at = time.monotonic()
        logits, past_key_values = self.model(
            sequences,
            attention_mask=attention_mask,
            multimodal_inputs=multimodal_inputs,
            use_cache=True,
        )
        self._sync_timing()
        prefill_sec = time.monotonic() - prefill_started_at
        _actual_lens = attention_mask.sum(dim=1) - 1
        _batch_idx = torch.arange(logits.shape[0], device=logits.device)
        _first_logits = logits[_batch_idx, _actual_lens]
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        _step_logits = _first_logits
        for _ in range(max_new_tokens):
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = _next_token_from_logits(
                _step_logits.float(), temperature=temperature, top_p=top_p
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
            _step_logits = logits[:, -1, :]
        self._sync_timing()
        return sequences, {
            "prefill_sec": prefill_sec,
            "decode_sec": time.monotonic() - decode_started_at,
            "decode_tokens": decode_tokens,
            "sampling_sec": sampling_sec,
            "stop_check_sec": stop_check_sec,
        }

    def _generate_multimodal_full_forward(
        self,
        *,
        sequences: torch.Tensor,
        multimodal_inputs: dict[str, torch.Tensor],
        finished: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Non-KV-cache full forward generation (fallback)."""
        assert isinstance(self.model, Qwen35HybridTextModel)
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        _first_step = True
        for _ in range(max_new_tokens):
            attention_mask = sequences.ne(pad_token_id)
            raw_logits = self.model(
                sequences,
                attention_mask=attention_mask,
                multimodal_inputs=multimodal_inputs,
            ).float()
            if _first_step:
                _actual_lens = attention_mask.sum(dim=1) - 1
                _batch_idx = torch.arange(raw_logits.shape[0], device=raw_logits.device)
                logits = raw_logits[_batch_idx, _actual_lens]
                _first_step = False
            else:
                logits = raw_logits[:, -1, :]
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
        return sequences, {
            "prefill_sec": 0.0,
            "decode_sec": time.monotonic() - decode_started_at,
            "decode_tokens": decode_tokens,
            "sampling_sec": sampling_sec,
            "stop_check_sec": stop_check_sec,
        }

    # ── PP generation ───────────────────────────────────────────────────────

    def _pipeline_generate_groups(
        self, **kwargs
    ) -> list[NativeGeneration]:
        """PP generation delegates to the same `_pipeline_generate_sequences_with_cache`
        path used by the old adapter.  During Step 4 this will be refactored to
        use the GraspoFlow RolloutPipeline, but for now the tested path is
        preserved."""
        return self._legacy_pipeline_generate_groups(**kwargs)

    def _pipeline_generate_multimodal_groups(
        self, **kwargs
    ) -> list[NativeGeneration]:
        return self._legacy_pipeline_generate_multimodal_groups(**kwargs)

    # ── Legacy PP methods (preserved from old adapter, to be refactored in Step 4) ──

    def _legacy_pipeline_generate_groups(self, **kwargs) -> list[NativeGeneration]:
        # Delegate to the old adapter's PP methods for now.
        # This is a transitional bridge — Step 4 replaces this with
        # RolloutPipeline-based implementation.
        raise NotImplementedError(
            "PP generation will be refactored in Step 4 using RolloutPipeline"
        )

    def _legacy_pipeline_generate_multimodal_groups(self, **kwargs) -> list[NativeGeneration]:
        raise NotImplementedError(
            "PP multimodal generation will be refactored in Step 4"
        )

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
        if self._is_pipeline_parallel():
            return self._pipeline_train_batch(
                experiences,
                policy_ratio_clip_eps=policy_ratio_clip_eps,
                optimize_times_per_step=optimize_times_per_step,
                max_grad_norm=max_grad_norm,
            )
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
                multimodal_inputs = self._multimodal_inputs_from_metadata(
                    batch.metadata,
                    batch_size=int(batch.sequences.shape[0]),
                )
                if multimodal_inputs is not None:
                    if not isinstance(self.model, Qwen35HybridTextModel):
                        raise ValueError(
                            "multimodal batch metadata for a non-multimodal model"
                        )
                    log_probs = self.model.sequence_log_probs(
                        batch.sequences,
                        batch.attention_mask,
                        multimodal_inputs=multimodal_inputs,
                    )
                else:
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

    def _pipeline_train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """PP training — delegates to 1F1B or simple schedule."""
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        schedule = str(self.config.graspoflow.pp_schedule or "simple")
        if schedule == "one_f_one_b":
            return self._pipeline_train_batch_one_f_one_b(
                experiences,
                policy_ratio_clip_eps=policy_ratio_clip_eps,
                optimize_times_per_step=optimize_times_per_step,
                max_grad_norm=max_grad_norm,
            )
        return self._pipeline_train_batch_simple(
            experiences,
            policy_ratio_clip_eps=policy_ratio_clip_eps,
            optimize_times_per_step=optimize_times_per_step,
            max_grad_norm=max_grad_norm,
        )

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
        multimodal_inputs = self._multimodal_inputs_from_metadata(
            metadata, batch_size=batch
        )
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
            _add_pipeline_stage_timing(
                timing, "pipeline_stage_compute_sec", compute_started_at
            )
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
            _add_pipeline_stage_timing(
                timing, "pipeline_stage_compute_sec", compute_started_at
            )
        assert isinstance(output, torch.Tensor)
        if self.pp_rank < self.pp_size - 1:
            send_started_at = time.monotonic()
            dist.send(
                output.detach().contiguous(), dst=int(self.tp_state.next_pp_rank)
            )
            _add_pipeline_stage_timing(timing, "pipeline_send_sec", send_started_at)
        if timing is not None:
            timing["pipeline_forward_calls"] = (
                int(timing.get("pipeline_forward_calls") or 0) + 1
            )
        return output, stage_input

    def _pipeline_train_batch_simple(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        self.loss_fn.policy_ratio_clip_eps = policy_ratio_clip_eps
        self.model.train()
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        batch_size = int(self.config.training.optimize_prompt_batch_size)
        train_batch_started_at = time.monotonic()
        micro_batch_forward_sec = 0.0
        backward_sec = 0.0
        optimizer_step_sec = 0.0
        round_secs: list[float] = []
        micro_batch_count = 0
        stage_timing = _new_pipeline_stage_timing()
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
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                self._sync_timing()
                forward_started_at = time.monotonic()
                stage_output, stage_input = self._pipeline_forward_for_training(
                    batch.sequences,
                    batch.attention_mask,
                    metadata=batch.metadata,
                    timing=stage_timing,
                )
                self._sync_timing()
                micro_batch_forward_sec += time.monotonic() - forward_started_at
                loss: torch.Tensor | None = None
                if self.pp_rank == self.pp_size - 1:
                    assert stage_output is not None
                    assert self.model.norm is not None and self.model.lm_head is not None
                    norm_started_at = time.monotonic()
                    hidden = self.model.norm(stage_output)
                    _add_pipeline_stage_timing(
                        stage_timing, "pipeline_norm_sec", norm_started_at
                    )
                    lm_head_started_at = time.monotonic()
                    log_probs = _selected_token_log_probs_from_hidden(
                        hidden[:, :-1].float(),
                        self.model.lm_head.weight.float(),
                        batch.sequences[:, 1:],
                    )
                    _add_pipeline_stage_timing(
                        stage_timing, "pipeline_lm_head_sec", lm_head_started_at
                    )
                    loss_started_at = time.monotonic()
                    loss = self.loss_fn(
                        log_probs,
                        batch.old_log_probs,
                        batch.advantages,
                        batch.action_mask,
                    )
                    _add_pipeline_stage_timing(
                        stage_timing, "pipeline_loss_sec", loss_started_at
                    )
                    finite = bool(torch.isfinite(loss).detach().cpu())
                else:
                    finite = True
                finite_payload = [finite]
                dist.broadcast_object_list(
                    finite_payload, src=(self.pp_size - 1) * self.tp_size
                )
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
                    loss_value = float(loss.detach().cpu())
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
                    loss_value = 0.0
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
                optimizer_steps += 1
                micro_batch_count += 1
                loss_payload = [loss_value]
                dist.broadcast_object_list(
                    loss_payload, src=(self.pp_size - 1) * self.tp_size
                )
                loss_sum += float(loss_payload[0])
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
            "pp_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": (
                self.placement.strategy if self.placement is not None else None
            ),
            "pp_schedule": "simple",
            "pp_max_inflight_microbatches": 1,
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            "synchronize_cuda_timing": bool(
                self.config.graspoflow.synchronize_cuda_timing
            ),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("pipeline_train_batch_after", {"metrics": metrics})
        return metrics

    def _pipeline_train_batch_one_f_one_b(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
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
        batch_size = int(self.config.training.optimize_prompt_batch_size)
        pipeline_micro_batch_size = max(
            1, int(self.config.graspoflow.pp_micro_batch_size)
        )
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
        for optimize_round in range(optimize_times_per_step):
            round_started_at = time.monotonic()
            indices = self._shared_training_indices(
                len(experiences), optimize_round=optimize_round
            )
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
                    for chunk_start in range(
                        0, len(batch_indices), pipeline_micro_batch_size
                    )
                ]
                if not chunk_batches:
                    continue
                max_chunks_per_optimizer_step = max(
                    max_chunks_per_optimizer_step, len(chunk_batches)
                )
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
                trainable_params = [
                    param for param in self.model.parameters() if param.requires_grad
                ]
                grad_clip_started_at = time.monotonic()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    if trainable_params
                    else torch.tensor(0.0)
                )
                _add_pipeline_stage_timing(
                    stage_timing, "pipeline_grad_clip_sec", grad_clip_started_at
                )
                self._sync_timing()
                optimizer_started_at = time.monotonic()
                if self.optimizer is not None:
                    self.optimizer.step()
                self._sync_timing()
                _add_pipeline_stage_timing(
                    stage_timing, "pipeline_optimizer_step_sec", optimizer_started_at
                )
                optimizer_step_sec += time.monotonic() - optimizer_started_at
                optimizer_steps += 1
                loss_payload = [float(result["loss_value"])]
                dist.broadcast_object_list(
                    loss_payload, src=(self.pp_size - 1) * self.tp_size
                )
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
            "placement_strategy": (
                self.placement.strategy if self.placement is not None else None
            ),
            "pp_schedule": "one_f_one_b",
            "pipeline_pp_micro_batch_size": pipeline_micro_batch_size,
            "pipeline_chunks_per_optimizer_step": max_chunks_per_optimizer_step,
            "pp_max_inflight_microbatches": effective_inflight,
            "pipeline_inflight_bound_source": "optimizer_step_chunks",
            "pipeline_fill_sec": fill_sec,
            "pipeline_steady_sec": steady_sec,
            "pipeline_drain_sec": drain_sec,
            "pipeline_backpressure_wait_sec": float(
                stage_timing.get("pipeline_recv_sec") or 0.0
            )
            + float(stage_timing.get("pipeline_send_sec") or 0.0)
            + float(stage_timing.get("pipeline_grad_recv_sec") or 0.0)
            + float(stage_timing.get("pipeline_grad_send_sec") or 0.0),
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            "synchronize_cuda_timing": bool(
                self.config.graspoflow.synchronize_cuda_timing
            ),
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
                _add_pipeline_stage_timing(
                    timing, "pipeline_norm_sec", norm_started_at
                )
                lm_head_started_at = time.monotonic()
                log_probs = _selected_token_log_probs_from_hidden(
                    hidden[:, :-1].float(),
                    self.model.lm_head.weight.float(),
                    batch.sequences[:, 1:],
                )
                _add_pipeline_stage_timing(
                    timing, "pipeline_lm_head_sec", lm_head_started_at
                )
                loss_started_at = time.monotonic()
                chunk_loss = self.loss_fn(
                    log_probs,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.action_mask,
                )
                _add_pipeline_stage_timing(
                    timing, "pipeline_loss_sec", loss_started_at
                )
                finite = bool(torch.isfinite(chunk_loss).detach().cpu())
                weight = float(batch.sequences.shape[0]) / max(
                    1, int(full_batch_size)
                )
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
                    loss.backward()
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
                _add_pipeline_stage_timing(
                    timing, "pipeline_grad_recv_sec", grad_recv_started_at
                )
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
            dist.broadcast_object_list(
                finite_payload, src=(self.pp_size - 1) * self.tp_size
            )
        return {
            "finite": bool(finite_payload[0]),
            "loss_value": sum(loss_values),
            "forward_sec": forward_sec,
            "backward_sec": backward_sec,
            "fill_sec": fill_sec,
            "steady_sec": steady_sec,
            "drain_sec": drain_sec,
        }

    # ── Sequence log probs ──────────────────────────────────────────────────

    def sequence_log_probs(
        self,
        sequences: Any,
        attention_mask: Any,
        metadata: Any | None = None,
    ) -> torch.Tensor:
        self._require_ready()
        assert self.model is not None
        if self._is_pipeline_parallel():
            return self._pipeline_sequence_log_probs(
                sequences, attention_mask, metadata=metadata
            )
        self.model.eval()
        sequences = sequences.to(self.device)
        attention_mask = attention_mask.to(self.device).bool()
        multimodal_inputs = self._multimodal_inputs_from_metadata(
            metadata, batch_size=int(sequences.shape[0])
        )
        with torch.no_grad():
            if multimodal_inputs is not None:
                if not isinstance(self.model, Qwen35HybridTextModel):
                    raise ValueError(
                        "multimodal metadata was provided for a non-multimodal model"
                    )
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
                _add_pipeline_stage_timing(
                    stage_timing, "pipeline_norm_sec", norm_started_at
                )
                lm_head_started_at = time.monotonic()
                log_probs = _selected_token_log_probs_from_hidden(
                    hidden[:, :-1].float(),
                    self.model.lm_head.weight.float(),
                    sequences[:, 1:],
                )
                _add_pipeline_stage_timing(
                    stage_timing, "pipeline_lm_head_sec", lm_head_started_at
                )
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
                    placement_summary(self.placement)
                    if self.placement is not None
                    else None
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
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        batch = int(attention_mask.shape[0])
        query_len = int(
            input_ids.shape[1] if input_ids is not None else hidden_states.shape[1]
        )
        hidden_size = int(self.model.config.hidden_size)
        dtype = next(self.model.parameters()).dtype
        if self.pp_rank > 0:
            hidden_states = torch.empty(
                (batch, query_len, hidden_size), device=self.device, dtype=dtype
            )
            recv_started_at = time.monotonic()
            dist.recv(hidden_states, src=int(self.tp_state.prev_pp_rank))
            _add_pipeline_stage_timing(
                timing, "pipeline_recv_sec", recv_started_at
            )
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
        _add_pipeline_stage_timing(
            timing, "pipeline_stage_compute_sec", compute_started_at
        )
        present = None
        if use_cache:
            output, present = output
        assert isinstance(output, torch.Tensor)
        if self.pp_rank < self.pp_size - 1:
            send_started_at = time.monotonic()
            dist.send(output.contiguous(), dst=int(self.tp_state.next_pp_rank))
            _add_pipeline_stage_timing(
                timing, "pipeline_send_sec", send_started_at
            )
            if timing is not None:
                timing["pipeline_forward_calls"] = (
                    int(timing.get("pipeline_forward_calls") or 0) + 1
                )
            return None, present
        if timing is not None:
            timing["pipeline_forward_calls"] = (
                int(timing.get("pipeline_forward_calls") or 0) + 1
            )
        return output, present

    # ── Parse completion ────────────────────────────────────────────────────

    def parse_completion(
        self, completion: str, sample: Any | None = None
    ) -> ParsedCompletion:
        return parse_qwen_tool_completion(
            completion,
            expect_tool_calls=bool(getattr(sample, "expects_tool_calls", False)),
            tools=getattr(sample, "tools", None),
        )
