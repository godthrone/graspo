from __future__ import annotations

"""Qwen3.5/3.6 adapter — generation methods (rollout, multimodal, KV cache).
"""

import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from graspo.backends.graspoflow.lora_helpers import native_qwen_lora_available_targets
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




class _Qwen35GenerationMethods:
    """Mixin: generation/rollout methods for Qwen35Adapter."""

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
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None],
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]:
        """Pipeline-parallel generate for text-only message batches.

        Parameters are forwarded from :meth:`generate_groups` verbatim; see its
        docstring for semantics.  This stub always raises *NotImplementedError*
        until the PP generation path is refactored.
        """
        raise NotImplementedError(
            "PP generation will be refactored in Step 4 using RolloutPipeline"
        )

    def _pipeline_generate_multimodal_groups(
        self,
        *,
        samples: list[Any],
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]:
        """Pipeline-parallel generate for multimodal :class:`Sample` batches.

        Parameters are forwarded from :meth:`generate_sample_groups` verbatim;
        see its docstring for semantics.  This stub always raises
        *NotImplementedError* until the PP multimodal generation path is
        refactored.
        """
        raise NotImplementedError(
            "PP multimodal generation will be refactored in Step 4"
        )

    # ── Training ────────────────────────────────────────────────────────────
