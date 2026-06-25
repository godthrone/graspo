from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from graspo.backends.native_tp.base_adapter import BaseNativeTPAdapter
from graspo.backends.native_tp.models.qwen.checkpoint import QwenCheckpointMixin
from graspo.backends.native_tp.models.qwen.encoding import QwenEncodingMixin
from graspo.backends.native_tp.models.qwen.generator import QwenGeneratorMixin
from graspo.backends.native_tp.models.qwen.lora import (
    native_qwen_lora_available_targets,
)
from graspo.backends.native_tp.models.qwen.modeling import (
    TensorParallelQwenForCausalLM,
    load_native_qwen_config,
    build_native_qwen_model,
)
from graspo.backends.native_tp.models.qwen.modeling_hybrid import (
    Qwen35HybridTextModel,
)
from graspo.backends.native_tp.multimodal import (
    _multimodal_row_from_sample,
    _messages_from_multimodal_row,
    _processor_chat_messages,
    _tools_from_multimodal_row,
    _normalize_tool_batches,
    _tools_for_chat_template,
    _multimodal_rows_from_metadata,
    _media_counts,
    _slice_multimodal_inputs,
    _slice_multimodal_inputs_offset,
    _compute_multimodal_offset_tables,
)
from graspo.backends.native_tp.parallel_state import NativeTPState, destroy_native_tp
from graspo.backends.native_tp.placement import (
    NativePlacementPlan,
    build_placement_plan,
    placement_summary,
)
from graspo.backends.native_tp.runtime import NativeGeneration
from graspo.backends.native_tp.tensor_utils import (
    SafetensorIndex,
    collate_experiences,
    _resolve_dtype,
    _selected_token_log_probs_from_hidden,
    _mean_present,
    _rollout_timing_summary,
    _scale_rollout_timings,
    _new_pipeline_stage_timing,
    _add_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _cuda_memory_snapshot,
    _jsonable,
    _left_pad_token_rows,
    _next_token_from_logits,
    _broadcast_and_pad_finished,
)
from graspo.backends.native_tp.tool_parser import (
    parse_qwen_tool_completion,
)
from graspo.backends.native_tp.lora_io import load_peft_adapter_into_native_model
from graspo.core.buffer import Experience
from graspo.core.completion import ParsedCompletion
from graspo.core.schema import GraspoConfig
from graspo.trainer.lora import resolve_lora_target_modules
from graspo.trainer.loss import GRASPOLoss


from graspo.backends.native_tp.tensor_utils import _set_tensor_parallel_group


def _patch_transformers_float8_import_compat() -> None:
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.uint8  # type: ignore[attr-defined]


# ── Deprecation notice ──────────────────────────────────────────────────────
# The PP (pipeline parallel) methods in this adapter are superseded by
# GraspoFlow (src/graspo/backends/graspoflow/), the unified TP+PP framework.
#
#   backend: native-tp  → legacy TP/PP (this adapter, still supported)
#   backend: graspoflow → new unified Flink-style pipeline (recommended)
#
# Once GraspoFlow is validated, the PP methods in this adapter will be
# removed.  TP-only methods remain for backward compatibility.
# ────────────────────────────────────────────────────────────────────────────


class QwenNativeTPAdapter(
    BaseNativeTPAdapter,
    QwenEncodingMixin,
    QwenGeneratorMixin,
    QwenCheckpointMixin,
):
    """Qwen causal LM adapter backed by self-owned PyTorch tensor parallel."""

    completion_parser_name = "qwen_tool_call"

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.tp_size = int(config.native_tp.tp_size)
        self.tp_rank = 0
        self.pp_size = int(config.native_tp.pp_size)
        self.pp_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tp_state: NativeTPState | None = None
        self.model: TensorParallelQwenForCausalLM | None = None
        self.placement: NativePlacementPlan | None = None
        self.tokenizer: Any | None = None
        self.processor: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = GRASPOLoss(config.training.policy_ratio_clip_eps)
        self._train_batch_call_index = 0

    def setup(self) -> None:
        self._setup_distributed()
        _patch_transformers_float8_import_compat()
        from transformers import AutoProcessor, AutoTokenizer

        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model.model_path does not exist: {model_path}")

        hf_config = load_native_qwen_config(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if bool(getattr(hf_config, "has_vision_config", False)):
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = _resolve_dtype(self.config.model.torch_dtype)
        loader = SafetensorIndex(model_path)
        lora_targets = resolve_lora_target_modules(
            self.config.lora.target_modules or (self.config.lora.target_preset,),
            available=native_qwen_lora_available_targets(hf_config),
        )
        self.placement = build_placement_plan(
            strategy=self.config.native_tp.placement_strategy,
            model_family=hf_config.family,
            num_hidden_layers=int(hf_config.num_hidden_layers),
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            layer_types=list(getattr(hf_config, "layer_types", []) or []),
        )
        self.model = build_native_qwen_model(  # type: ignore[assignment]
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
            for target in set(lora_targets.resolved) - set(self.model.enabled_lora_target_names())
            if not (target.startswith("visual.") and getattr(self.model, "visual", None) is None)
        )
        if missing_lora_targets:
            raise ValueError(
                "Resolved LoRA target(s) are not implemented by this native model yet: "
                + ", ".join(missing_lora_targets)
            )
        self.model.train(False)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = (
            torch.optim.AdamW(
                trainable,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            if trainable
            else None
        )
        if self.config.lora.adapter_path:
            load_peft_adapter_into_native_model(
                self.model,
                self.config.lora.adapter_path,
                base_model_path=str(model_path),
            )
        self._emit_rank_memory_event(
            "setup_after",
            {
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
                "lora_target_modules": sorted(self.model.lora_targets),
                "lora_target_signature": self.model.lora_target_signature(),
                "rollout_kv_cache_supported": bool(getattr(self.model, "supports_kv_cache", True)),
                "placement": placement_summary(self.placement),
                "forward_batch_size": self.config.native_tp.forward_batch_size,
                "empty_cache_after_rollout_split": self.config.native_tp.empty_cache_after_rollout_split,
                "synchronize_cuda_timing": self.config.native_tp.synchronize_cuda_timing,
            },
        )
        self._print_rank0(
            {
                "event": "native_qwen_adapter_ready",
                "rank": self.rank,
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "group_batch_semantics": "rollout_prompt_queue_batch_size prompts, each with rollout_group_size completions per TP forward batch when budget permits",
                "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
                "lora_target_modules": sorted(self.model.lora_targets),
                "lora_target_signature": self.model.lora_target_signature(),
                "rollout_kv_cache_supported": bool(getattr(self.model, "supports_kv_cache", True)),
                "placement": placement_summary(self.placement),
            }
        )

    def generate_group(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> NativeGeneration:
        return self.generate_groups(
            message_batches=[messages],
            tool_batches=[tools],
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
        )[0]

    def generate_groups(  # type: ignore[override]
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
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
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id
        )
        prompt_input_ids, prompt_lens = _left_pad_token_rows(
            encoded["input_ids"],
            pad_token_id=pad_token_id,
            device=self.device,
        )
        prompt_len = int(prompt_input_ids.shape[1])
        use_kv_cache = bool(self.config.native_tp.use_kv_cache_for_rollout) and bool(
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
                    finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=self.device)
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

        rollout_summary = {
            "rollout_group_size": rollout_group_size,
            "rollout_prompt_queue_batch_size": requested_prompt_queue_size,
            "rollout_prompt_queue_effective_size": prompt_chunk_size,
            "rollout_prompt_queue_fallback": prompt_chunk_size < requested_prompt_queue_size,
            "prompt_len": prompt_len,
            "prompt_lens": prompt_lens,
            "sequence_len": max(
                (int(chunk.shape[1]) for chunk in all_sequence_chunks), default=prompt_len
            ),
            "generated_tokens_max": max(
                (int(chunk.shape[1] - prompt_len) for chunk in all_sequence_chunks), default=0
            ),
            "rollout_use_kv_cache": use_kv_cache,
            "rollout_generation_micro_batch_size": max(
                (
                    int((gen.metadata or {}).get("rollout_generation_micro_batch_size", 1))
                    for gen in all_generations
                ),
                default=1,
            ),
            "rollout_generation_split_count": sum(
                int((gen.metadata or {}).get("rollout_generation_split_count", 1))
                for gen in all_generations
            ),
            **_rollout_timing_summary(tokenize_sec, all_chunk_timings),
            "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
        }
        self._emit_rank_memory_event("rollout_after", rollout_summary)
        empty_cache_after_split = (
            self.device.type == "cuda"
            and bool(self.config.native_tp.empty_cache_after_rollout_split)
            and (
                prompt_chunk_size < requested_prompt_queue_size
                or any(
                    int((gen.metadata or {}).get("rollout_generation_split_count", 1)) > 1
                    for gen in all_generations
                )
            )
        )
        if empty_cache_after_split:
            torch.cuda.empty_cache()
            self._emit_rank_memory_event(
                "rollout_after_empty_cache",
                {**rollout_summary, "rollout_empty_cache_after_split": True},
            )
            for generation in all_generations:
                generation.metadata = {
                    **(generation.metadata or {}),
                    "rollout_empty_cache_after_split": True,
                }
        return all_generations

    def generate_sample_groups(  # type: ignore[override]
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
        self._require_ready()
        if any(
            any(str(item.get("type") or "") == "video" for item in sample.media)
            for sample in samples
        ):
            raise NotImplementedError(
                "Qwen3.5-family video generation is reserved for the next phase; "
                "image multimodal samples are supported first"
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
        """Generate rollouts for multiple multimodal samples batched together.

        Mirrors the text path generate_groups: encodes all samples×G rows in one
        processor call, then applies Level 1 (prompt chunk) and Level 2
        (micro-batch) chunking with offset-based multimodal input slicing.
        """
        assert self.model is not None
        assert self.tokenizer is not None
        self.model.eval()
        N = len(samples)
        G = rollout_group_size
        tokenize_started_at = time.monotonic()

        # Build all N*G rows and count per-sample images from media metadata
        rows: list[dict[str, Any]] = []
        per_sample_image_counts: list[int] = []
        per_sample_media_counts: list[dict[str, int]] = []
        for sample in samples:
            row = _multimodal_row_from_sample(sample)
            img_count = sum(1 for item in sample.media if str(item.get("type") or "") == "image")
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
                f"data.max_prompt_length={max_prompt_length}; "
                "increase max_prompt_length instead of truncating image placeholders"
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
        use_kv_cache = bool(self.config.native_tp.use_kv_cache_for_rollout) and bool(
            getattr(self.model, "supports_kv_cache", True)
        )
        multimodal_inputs = self._multimodal_inputs_to_device(encoded)

        # Build offset tables for heterogeneous samples
        image_offsets, patch_offsets, video_offsets, video_patch_offsets = (
            _compute_multimodal_offset_tables(
                per_sample_image_counts=per_sample_image_counts,
                rollout_group_size=G,
                image_grid_thw=multimodal_inputs.get("image_grid_thw"),
                pixel_values=multimodal_inputs.get("pixel_values"),
            )
        )

        requested_prompt_queue_size = N
        # Use max_prompt_length for budget estimation to guard against
        # heterogeneous prompt lengths across encoding chunks.  The first
        # chunk may have short prompts (→ generous budget), but later
        # chunks can be longer and would OOM.
        budget_prompt_len = max(prompt_len, max_prompt_length)
        prompt_chunk_size = self._shared_rollout_prompt_chunk_size(
            prompt_len=budget_prompt_len,
            requested_prompt_count=N,
            rollout_group_size=G,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        # Cap prompt_chunk at 3 to prevent over-ambitious batching.
        prompt_chunk_size = max(1, min(prompt_chunk_size, N))

        all_generations: list[NativeGeneration] = []
        all_timings: list[dict[str, float | int]] = []
        with torch.no_grad():
            for prompt_start in range(0, N, prompt_chunk_size):
                prompt_stop = min(prompt_start + prompt_chunk_size, N)
                chunk_prompt_count = prompt_stop - prompt_start

                # Row range in the flat batch for this prompt chunk
                row_start = prompt_start * G
                row_stop = prompt_stop * G
                chunk_input_ids = input_ids[row_start:row_stop]
                chunk_attention_mask = attention_mask[row_start:row_stop]
                flat_B = int(chunk_input_ids.shape[0])

                # Per-prompt lengths for these prompts
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
                    # local_start/stop are relative to the chunk; global positions
                    # use row_start as base for offset table indexing
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
                    # # Clone to break the view chain from flat_sequences,
                    # so the large pad_sequence tensor can be freed.
                    prompt_sequences = flat_sequences[row_start_inner:row_stop_inner].clone()
                    all_generations.append(
                        self._generation_from_sequences(
                            sequences=prompt_sequences,
                            prompt_len=prompt_len,
                            prompt_lens=chunk_prompt_lens[row_start_inner : row_start_inner + 1],
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

                # Free GPU memory between prompt chunks (respect the unified flag).
                del flat_sequences
                del sequence_chunks
                if self.device.type == "cuda" and bool(
                    self.config.native_tp.empty_cache_after_rollout_split
                ):
                    torch.cuda.empty_cache()

        # Emit memory event (once, after all chunks)
        self._emit_rank_memory_event(
            "multimodal_rollout_after",
            {
                "rollout_group_size": G,
                "prompt_len": prompt_len,
                "sequence_len": max(
                    (int(chunk.shape[1]) for chunk in [gen.sequences for gen in all_generations]),
                    default=prompt_len,
                ),
                "multimodal_media_counts": per_sample_media_counts[0]
                if len(per_sample_media_counts) == 1
                else None,
                "image_token_count": int(
                    (input_ids == int(getattr(self.model.config, "image_token_id", -1)))
                    .sum()
                    .item()
                ),
                **_rollout_timing_summary(tokenize_sec, all_timings),
            },
        )

        # Empty CUDA cache when splits occurred
        if (
            self.device.type == "cuda"
            and bool(self.config.native_tp.empty_cache_after_rollout_split)
            and (
                prompt_chunk_size < N
                or any(
                    int((gen.metadata or {}).get("rollout_generation_split_count", 1)) > 1
                    for gen in all_generations
                )
            )
        ):
            torch.cuda.empty_cache()
            self._emit_rank_memory_event(
                "multimodal_rollout_after_empty_cache",
                {"rollout_empty_cache_after_split": True},
            )
            for generation in all_generations:
                generation.metadata = {
                    **(generation.metadata or {}),
                    "rollout_empty_cache_after_split": True,
                }

        return all_generations

    def _generate_multimodal_sample_group(
        self,
        *,
        sample: Any,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> NativeGeneration:
        assert self.model is not None
        assert self.tokenizer is not None
        self.model.eval()
        tokenize_started_at = time.monotonic()
        encoded = self._encode_multimodal_rows(
            [_multimodal_row_from_sample(sample) for _ in range(rollout_group_size)],
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        tokenize_sec = time.monotonic() - tokenize_started_at
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] > max_prompt_length:
            raise ValueError(
                f"multimodal prompt length {input_ids.shape[1]} exceeds data.max_prompt_length={max_prompt_length}; "
                "increase max_prompt_length instead of truncating image placeholders"
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
        use_kv_cache = bool(self.config.native_tp.use_kv_cache_for_rollout) and bool(
            getattr(self.model, "supports_kv_cache", True)
        )
        multimodal_inputs = self._multimodal_inputs_to_device(encoded)
        B = int(input_ids.shape[0])

        # Precompute per-row image/patch counts for equal-stride slicing.
        # All rollout_group_size rows are copies of the same sample, so every
        # row contributes identical image/patch counts and even division holds.
        images_per_row = 0
        patches_per_row = 0
        if "image_grid_thw" in multimodal_inputs:
            grid_rows = int(multimodal_inputs["image_grid_thw"].shape[0])
            assert grid_rows % B == 0, (
                f"image_grid_thw rows {grid_rows} not divisible by batch size {B}; "
                "expected equal image counts per row during multimodal rollout"
            )
            images_per_row = grid_rows // B
        if "pixel_values" in multimodal_inputs:
            pv_rows = int(multimodal_inputs["pixel_values"].shape[0])
            assert pv_rows % B == 0, (
                f"pixel_values rows {pv_rows} not divisible by batch size {B}; "
                "expected equal patch counts per row during multimodal rollout"
            )
            patches_per_row = pv_rows // B

        micro_batch_size = self._shared_generation_micro_batch_size(
            prompt_len=prompt_len,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        micro_batch_size = max(1, min(micro_batch_size, B))

        sequence_chunks: list[torch.Tensor] = []
        chunk_timings: list[dict[str, float | int]] = []
        with torch.no_grad():
            for start in range(0, B, micro_batch_size):
                stop = min(start + micro_batch_size, B)
                current_input_ids = input_ids[start:stop]
                current_attention_mask = attention_mask[start:stop]
                current_mm_inputs = _slice_multimodal_inputs(
                    multimodal_inputs,
                    start,
                    stop,
                    images_per_row=images_per_row,
                    patches_per_row=patches_per_row,
                )
                finished = torch.zeros(stop - start, dtype=torch.bool, device=self.device)
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

        sequences = pad_sequence(
            [row for chunk in sequence_chunks for row in chunk],
            batch_first=True,
            padding_value=pad_token_id,
        )
        generation = self._generation_from_sequences(
            sequences=sequences,
            prompt_len=prompt_len,
            prompt_lens=prompt_lens,
            pad_token_id=pad_token_id,
            rollout_group_size=rollout_group_size,
            requested_prompt_queue_size=1,
            effective_prompt_queue_size=1,
            use_kv_cache=use_kv_cache,
            generation_micro_batch_size=micro_batch_size,
            split_count=len(sequence_chunks),
            tokenize_sec=tokenize_sec,
            chunk_timings=chunk_timings,
            timing_divisor=1,
            rollout_started_at=rollout_started_at,
        )
        generation.metadata = {
            **(generation.metadata or {}),
            "multimodal_enabled": True,
            "multimodal_media_counts": _media_counts(sample.media),
            "_multimodal_rows": [
                _multimodal_row_from_sample(sample) for _ in range(rollout_group_size)
            ],
        }
        self._emit_rank_memory_event(
            "multimodal_rollout_after",
            {
                "rollout_group_size": rollout_group_size,
                "prompt_len": prompt_len,
                "sequence_len": int(sequences.shape[1]),
                "multimodal_media_counts": _media_counts(sample.media),
                "image_token_count": int(
                    (input_ids == int(getattr(self.model.config, "image_token_id", -1)))
                    .sum()
                    .item()
                ),
                **_rollout_timing_summary(tokenize_sec, chunk_timings),
            },
        )
        return generation

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
        # Extract per-row logits at the last *real* token position (not PAD).
        # With right-padding, shorter rows have PAD tokens at the end, and
        # logits[:, -1, :] would use a PAD embedding query → corrupted output.
        # attention_mask is 1 for real tokens, 0 for PAD — sum gives per-row lengths.
        _actual_lens = attention_mask.sum(dim=1) - 1  # last real token index per row
        _batch_idx = torch.arange(logits.shape[0], device=logits.device)
        _first_logits = logits[_batch_idx, _actual_lens]  # shape (B, vocab_size)
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        _debug_decode = os.environ.get("GRASPO_DEBUG_DECODE") == "1"
        if _debug_decode and self.rank == 0:
            _row_lens = attention_mask.sum(dim=1).tolist()
            _min_len, _max_len = min(_row_lens), max(_row_lens)
            # Check rope_deltas batch size vs actual batch size
            _rd = getattr(self.model, "rope_deltas", None)
            _rd_shape = list(_rd.shape) if _rd is not None else None
            print(f"  [prefill] batch={sequences.shape[0]} pad_len={sequences.shape[1]} "
                  f"prompt_lens={_min_len}..{_max_len} rope_deltas_shape={_rd_shape}", flush=True)
            _preview_first = self.tokenizer.decode(
                [_first_logits[0].argmax().item()]
            ) if _first_logits.shape[0] > 0 else "?"
            _top5_vals, _top5_idx = _first_logits[0].float().topk(5)
            _top5_txt = [self.tokenizer.decode([int(t)]) for t in _top5_idx]
            print(f"  [prefill] seq_len={sequences.shape[1]} batch={sequences.shape[0]} "
                  f"first_tok_greedy={_preview_first!r} "
                  f"top5={list(zip(_top5_txt, _top5_vals.tolist()))}", flush=True)
        # Per-row logits for the first token; subsequent steps use logits[:, -1, :]
        _step_logits = _first_logits
        for _step_idx in range(max_new_tokens):
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = _next_token_from_logits(
                _step_logits.float(), temperature=temperature, top_p=top_p
            )
            self._sync_timing()
            sampling_sec += time.monotonic() - sampling_started_at
            next_token = _broadcast_and_pad_finished(next_token, finished, pad_token_id)
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            decode_tokens += 1
            if _debug_decode and self.rank == 0:
                _tok_ids = next_token.tolist()
                _decoded = [self.tokenizer.decode([t]) if t != pad_token_id else "<PAD>"
                            for t in _tok_ids[:3]]
                _eos_rows = [i for i, t in enumerate(_tok_ids) if t == eos_token_id]
                _pad_rows = [i for i, t in enumerate(_tok_ids) if t == pad_token_id]
                _extra = ""
                if _eos_rows:
                    _extra += f" EOS_at={_eos_rows}"
                if _pad_rows:
                    _extra += f" PAD_at={_pad_rows}"
                print(f"  [step {_step_idx}] tokens={_tok_ids[:8]} decoded={_decoded}{_extra}", flush=True)
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
            if _debug_decode and self.rank == 0 and _step_idx == 1:
                _rd = getattr(self.model, "rope_deltas", None)
                _rd_shape = list(_rd.shape) if _rd is not None else None
                _attn_lens = attention_mask.sum(dim=1).tolist()
                print(f"  [decode step 1] attn_lens[:8]={_attn_lens[:8]} "
                      f"rope_deltas_shape={_rd_shape}", flush=True)
            _step_logits = logits[:, -1, :]  # shape (B, vocab) for next decode step
        self._sync_timing()
        if _debug_decode and self.rank == 0:
            print(f"  [decode done] total_steps={decode_tokens}", flush=True)
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
                # Per-row logits at last *real* token (not PAD end position)
                _actual_lens = attention_mask.sum(dim=1) - 1
                _batch_idx = torch.arange(raw_logits.shape[0], device=raw_logits.device)
                logits = raw_logits[_batch_idx, _actual_lens]
                _first_step = False
            else:
                logits = raw_logits[:, -1, :]
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = _next_token_from_logits(logits, temperature=temperature, top_p=top_p)
            self._sync_timing()
            sampling_sec += time.monotonic() - sampling_started_at
            next_token = _broadcast_and_pad_finished(next_token, finished, pad_token_id)
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

    def _generation_from_sequences(
        self,
        *,
        sequences: torch.Tensor,
        prompt_len: int,
        prompt_lens: list[int],
        pad_token_id: int,
        rollout_group_size: int,
        requested_prompt_queue_size: int,
        effective_prompt_queue_size: int,
        use_kv_cache: bool,
        generation_micro_batch_size: int,
        split_count: int,
        tokenize_sec: float,
        chunk_timings: list[dict[str, float | int]],
        timing_divisor: int,
        rollout_started_at: float,
    ) -> NativeGeneration:
        assert self.tokenizer is not None
        attention_mask = sequences.ne(pad_token_id)
        action_mask = torch.zeros(
            (sequences.shape[0], max(sequences.shape[1] - 1, 0)),
            dtype=torch.bool,
            device=self.device,
        )
        if sequences.shape[1] > prompt_len:
            action_mask[:, prompt_len - 1 :] = True
            action_mask &= attention_mask[:, 1:]
        completions = self.tokenizer.batch_decode(
            sequences[:, prompt_len:],
            skip_special_tokens=True,
        )
        return NativeGeneration(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            completions=completions,
            prompt_len=prompt_len,
            metadata={
                "adapter": "qwen_native_tp",
                "rollout_group_size": rollout_group_size,
                "rollout_prompt_queue_batch_size": requested_prompt_queue_size,
                "rollout_prompt_queue_effective_size": effective_prompt_queue_size,
                "rollout_prompt_queue_fallback": effective_prompt_queue_size
                < requested_prompt_queue_size,
                "rollout_use_kv_cache": use_kv_cache,
                "rollout_generation_micro_batch_size": generation_micro_batch_size,
                "rollout_generation_split_count": split_count,
                "rollout_empty_cache_after_split": False,
                **_rollout_timing_summary(
                    tokenize_sec, _scale_rollout_timings(chunk_timings, timing_divisor)
                ),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
                "prefill_len": prompt_len,
                "prompt_lens": prompt_lens,
                "generated_tokens_max": max(int(sequences.shape[1] - prompt_len), 0),
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
            },
        )

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
                logits,
                temperature=temperature,
                top_p=top_p,
            )
            self._sync_timing()
            sampling_sec += time.monotonic() - sampling_started_at
            next_token = _broadcast_and_pad_finished(next_token, finished, pad_token_id)
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
            next_token = _broadcast_and_pad_finished(next_token, finished, pad_token_id)
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

    def _pipeline_generate_groups(
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> list[NativeGeneration]:
        assert self.tokenizer is not None
        assert self.model is not None
        if not message_batches:
            return []
        tool_batches = _normalize_tool_batches(tool_batches, len(message_batches))
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
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id
        )
        prompt_input_ids, prompt_lens = _left_pad_token_rows(
            encoded["input_ids"],
            pad_token_id=pad_token_id,
            device=self.device,
        )
        prompt_len = int(prompt_input_ids.shape[1])
        requested_prompt_queue_size = len(message_batches)
        prompt_chunk_size = self._shared_rollout_prompt_chunk_size(
            prompt_len=prompt_len,
            requested_prompt_count=requested_prompt_queue_size,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
        )
        prompt_chunk_size = max(1, min(prompt_chunk_size, requested_prompt_queue_size))
        all_generations: list[NativeGeneration] = []
        all_timings: list[dict[str, float | int]] = []
        all_sequence_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for prompt_start in range(0, requested_prompt_queue_size, prompt_chunk_size):
                prompt_stop = min(prompt_start + prompt_chunk_size, requested_prompt_queue_size)
                prompt_chunk = prompt_input_ids[prompt_start:prompt_stop]
                chunk_prompt_count = int(prompt_chunk.shape[0])
                flat_prompt_input_ids = prompt_chunk.repeat_interleave(rollout_group_size, dim=0)
                chunk_generation_micro_batch_size = self._shared_generation_micro_batch_size(
                    prompt_len=prompt_len,
                    rollout_group_size=chunk_prompt_count * rollout_group_size,
                    max_new_tokens=max_new_tokens,
                    use_kv_cache=True,
                )
                sequence_chunks: list[torch.Tensor] = []
                chunk_timings: list[dict[str, float | int]] = []
                for start in range(
                    0, flat_prompt_input_ids.shape[0], chunk_generation_micro_batch_size
                ):
                    current_batch = flat_prompt_input_ids[
                        start : start + chunk_generation_micro_batch_size
                    ]
                    sequences, timing = self._pipeline_generate_sequences_with_cache(
                        sequences=current_batch,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    sequence_chunks.append(sequences)
                    chunk_timings.append(timing)
                flat_sequences = pad_sequence(
                    [row for chunk in sequence_chunks for row in chunk],
                    batch_first=True,
                    padding_value=pad_token_id,
                )
                all_sequence_chunks.append(flat_sequences)
                all_timings.extend(chunk_timings)
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
                            use_kv_cache=True,
                            generation_micro_batch_size=chunk_generation_micro_batch_size,
                            split_count=len(sequence_chunks),
                            tokenize_sec=tokenize_sec / max(requested_prompt_queue_size, 1),
                            chunk_timings=chunk_timings,
                            timing_divisor=chunk_prompt_count,
                            rollout_started_at=rollout_started_at,
                        )
                    )
        self._emit_rank_memory_event(
            "pipeline_rollout_after",
            {
                "placement": placement_summary(self.placement)
                if self.placement is not None
                else None,
                "rollout_group_size": rollout_group_size,
                "rollout_prompt_queue_batch_size": requested_prompt_queue_size,
                "rollout_prompt_queue_effective_size": prompt_chunk_size,
                "rollout_prompt_queue_fallback": prompt_chunk_size < requested_prompt_queue_size,
                "prompt_len": prompt_len,
                "prompt_lens": prompt_lens,
                "sequence_len": max(
                    (int(chunk.shape[1]) for chunk in all_sequence_chunks), default=prompt_len
                ),
                "generated_tokens_max": max(
                    (int(chunk.shape[1] - prompt_len) for chunk in all_sequence_chunks), default=0
                ),
                "rollout_use_kv_cache": True,
                "rollout_generation_micro_batch_size": max(
                    (
                        int((gen.metadata or {}).get("rollout_generation_micro_batch_size", 1))
                        for gen in all_generations
                    ),
                    default=1,
                ),
                "rollout_generation_split_count": sum(
                    int((gen.metadata or {}).get("rollout_generation_split_count", 1))
                    for gen in all_generations
                ),
                **_rollout_timing_summary(tokenize_sec, all_timings),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
            },
        )
        return all_generations

    def _pipeline_generate_multimodal_groups(
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
        """Pipeline-parallel batched multimodal generation (Level 1 + Level 2)."""
        assert self.tokenizer is not None
        assert self.model is not None
        self.model.eval()
        N = len(samples)
        G = rollout_group_size
        tokenize_started_at = time.monotonic()

        rows: list[dict[str, Any]] = []
        per_sample_image_counts: list[int] = []
        for sample in samples:
            row = _multimodal_row_from_sample(sample)
            img_count = sum(1 for item in sample.media if str(item.get("type") or "") == "image")
            per_sample_image_counts.append(img_count)
            for _ in range(G):
                rows.append(row)

        encoded = self._encode_multimodal_rows(
            rows,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        tokenize_sec = time.monotonic() - tokenize_started_at

        sequences = encoded["input_ids"].to(self.device)
        if sequences.shape[1] > max_prompt_length:
            raise ValueError(
                f"multimodal prompt length {sequences.shape[1]} exceeds "
                f"data.max_prompt_length={max_prompt_length}"
            )
        attention_mask = encoded["attention_mask"].to(self.device).bool()
        prompt_len = int(sequences.shape[1])
        prompt_lens = [int(mask.sum().item()) for mask in attention_mask]
        pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        eos_token_id = int(self.tokenizer.eos_token_id)
        rollout_started_at = time.monotonic()
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
            requested_prompt_count=requested_prompt_queue_size,
            rollout_group_size=G,
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
        )
        prompt_chunk_size = max(1, min(prompt_chunk_size, requested_prompt_queue_size))

        all_generations: list[NativeGeneration] = []
        all_timings: list[dict[str, float | int]] = []
        with torch.no_grad():
            for prompt_start in range(0, N, prompt_chunk_size):
                prompt_stop = min(prompt_start + prompt_chunk_size, N)
                chunk_prompt_count = prompt_stop - prompt_start
                row_start = prompt_start * G
                row_stop = prompt_stop * G
                chunk_sequences = sequences[row_start:row_stop]
                flat_B = int(chunk_sequences.shape[0])
                chunk_prompt_lens = prompt_lens[row_start:row_stop]

                chunk_generation_micro_batch_size = self._shared_generation_micro_batch_size(
                    prompt_len=budget_prompt_len,
                    rollout_group_size=flat_B,
                    max_new_tokens=max_new_tokens,
                    use_kv_cache=True,
                )

                sequence_chunks: list[torch.Tensor] = []
                chunk_timings: list[dict[str, float | int]] = []
                for start in range(0, flat_B, chunk_generation_micro_batch_size):
                    stop = min(start + chunk_generation_micro_batch_size, flat_B)
                    local_start = start
                    local_stop = stop
                    global_start = row_start + local_start
                    global_stop = row_start + local_stop

                    current_sequences = chunk_sequences[local_start:local_stop]
                    current_mm_inputs = _slice_multimodal_inputs_offset(
                        multimodal_inputs,
                        global_start,
                        global_stop,
                        image_offsets=image_offsets,
                        patch_offsets=patch_offsets,
                        video_offsets=video_offsets,
                        video_patch_offsets=video_patch_offsets,
                    )
                    seq, timing = self._pipeline_generate_sequences_with_cache(
                        sequences=current_sequences,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        multimodal_inputs=current_mm_inputs,
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
                    # Clone to break the view chain (see TP path comment).
                    prompt_sequences = flat_sequences[row_start_inner:row_stop_inner].clone()
                    all_generations.append(
                        self._generation_from_sequences(
                            sequences=prompt_sequences,
                            prompt_len=prompt_len,
                            prompt_lens=chunk_prompt_lens[row_start_inner : row_start_inner + 1],
                            pad_token_id=pad_token_id,
                            rollout_group_size=G,
                            requested_prompt_queue_size=requested_prompt_queue_size,
                            effective_prompt_queue_size=prompt_chunk_size,
                            use_kv_cache=True,
                            generation_micro_batch_size=chunk_generation_micro_batch_size,
                            split_count=len(sequence_chunks),
                            tokenize_sec=tokenize_sec / max(N, 1),
                            chunk_timings=chunk_timings,
                            timing_divisor=chunk_prompt_count,
                            rollout_started_at=rollout_started_at,
                        )
                    )

                # Free GPU memory between prompt chunks (respect the unified flag).
                del flat_sequences
                del sequence_chunks
                if self.device.type == "cuda" and bool(
                    self.config.native_tp.empty_cache_after_rollout_split
                ):
                    torch.cuda.empty_cache()

        self._emit_rank_memory_event(
            "pipeline_multimodal_rollout_after",
            {
                "placement": placement_summary(self.placement)
                if self.placement is not None
                else None,
                "rollout_group_size": G,
                "prompt_len": prompt_len,
                "prompt_lens": prompt_lens,
                "sequence_len": max(
                    (int(chunk.shape[1]) for chunk in [gen.sequences for gen in all_generations]),
                    default=prompt_len,
                ),
                "generated_tokens_max": max(
                    (
                        int(chunk.shape[1] - prompt_len)
                        for chunk in [gen.sequences for gen in all_generations]
                    ),
                    default=0,
                ),
                "rollout_use_kv_cache": True,
                "rollout_generation_micro_batch_size": max(
                    (
                        int((gen.metadata or {}).get("rollout_generation_micro_batch_size", 1))
                        for gen in all_generations
                    ),
                    default=1,
                ),
                "rollout_generation_split_count": sum(
                    int((gen.metadata or {}).get("rollout_generation_split_count", 1))
                    for gen in all_generations
                ),
                **_rollout_timing_summary(tokenize_sec, all_timings),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
            },
        )
        return all_generations

    def _pipeline_generate_multimodal_sample_group(
        self,
        *,
        sample: Any,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> NativeGeneration:
        assert self.tokenizer is not None
        assert self.model is not None
        self.model.eval()
        tokenize_started_at = time.monotonic()
        rows = [_multimodal_row_from_sample(sample) for _ in range(rollout_group_size)]
        encoded = self._encode_multimodal_rows(
            rows,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        tokenize_sec = time.monotonic() - tokenize_started_at
        sequences = encoded["input_ids"].to(self.device)
        if sequences.shape[1] > max_prompt_length:
            raise ValueError(
                f"multimodal prompt length {sequences.shape[1]} exceeds data.max_prompt_length={max_prompt_length}"
            )
        attention_mask = encoded["attention_mask"].to(self.device).bool()
        prompt_len = int(sequences.shape[1])
        prompt_lens = [int(mask.sum().item()) for mask in attention_mask]
        pad_token_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        eos_token_id = int(self.tokenizer.eos_token_id)
        rollout_started_at = time.monotonic()
        multimodal_inputs = self._multimodal_inputs_to_device(encoded)
        B = int(sequences.shape[0])

        # Precompute per-row image/patch counts for equal-stride slicing.
        images_per_row = 0
        patches_per_row = 0
        if "image_grid_thw" in multimodal_inputs:
            grid_rows = int(multimodal_inputs["image_grid_thw"].shape[0])
            assert grid_rows % B == 0, (
                f"image_grid_thw rows {grid_rows} not divisible by batch size {B}; "
                "expected equal image counts per row during multimodal rollout"
            )
            images_per_row = grid_rows // B
        if "pixel_values" in multimodal_inputs:
            pv_rows = int(multimodal_inputs["pixel_values"].shape[0])
            assert pv_rows % B == 0, (
                f"pixel_values rows {pv_rows} not divisible by batch size {B}; "
                "expected equal patch counts per row during multimodal rollout"
            )
            patches_per_row = pv_rows // B

        micro_batch_size = self._shared_generation_micro_batch_size(
            prompt_len=prompt_len,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=True,
        )
        micro_batch_size = max(1, min(micro_batch_size, B))

        sequence_chunks: list[torch.Tensor] = []
        chunk_timings: list[dict[str, float | int]] = []
        with torch.no_grad():
            for start in range(0, B, micro_batch_size):
                stop = min(start + micro_batch_size, B)
                current_sequences = sequences[start:stop]
                current_mm_inputs = _slice_multimodal_inputs(
                    multimodal_inputs,
                    start,
                    stop,
                    images_per_row=images_per_row,
                    patches_per_row=patches_per_row,
                )
                seq, timing = self._pipeline_generate_sequences_with_cache(
                    sequences=current_sequences,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    multimodal_inputs=current_mm_inputs,
                )
                sequence_chunks.append(seq)
                chunk_timings.append(timing)

        sequences = pad_sequence(
            [row for chunk in sequence_chunks for row in chunk],
            batch_first=True,
            padding_value=pad_token_id,
        )
        generation = self._generation_from_sequences(
            sequences=sequences,
            prompt_len=prompt_len,
            prompt_lens=prompt_lens,
            pad_token_id=pad_token_id,
            rollout_group_size=rollout_group_size,
            requested_prompt_queue_size=1,
            effective_prompt_queue_size=1,
            use_kv_cache=True,
            generation_micro_batch_size=micro_batch_size,
            split_count=len(sequence_chunks),
            tokenize_sec=tokenize_sec,
            chunk_timings=chunk_timings,
            timing_divisor=1,
            rollout_started_at=rollout_started_at,
        )
        generation.metadata = {
            **(generation.metadata or {}),
            "multimodal_enabled": True,
            "multimodal_media_counts": _media_counts(sample.media),
            "_multimodal_rows": rows,
        }
        return generation

    def _pipeline_generate_sequences_with_cache(
        self,
        *,
        sequences: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=self.device)
        attention_mask = sequences.ne(pad_token_id)
        self._sync_timing()
        prefill_started_at = time.monotonic()
        stage_timing = _new_pipeline_stage_timing()
        hidden, past_key_values = self._pipeline_forward_stage(
            input_ids=sequences,
            hidden_states=None,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            multimodal_inputs=multimodal_inputs,
            timing=stage_timing,
        )
        logits = self._pipeline_logits_from_last_hidden(hidden, timing=stage_timing)
        self._sync_timing()
        prefill_sec = time.monotonic() - prefill_started_at
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        for _ in range(max_new_tokens):
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = self._pipeline_sample_next_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                timing=stage_timing,
            )
            self._sync_timing()
            sampling_sec += time.monotonic() - sampling_started_at
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_token_id), next_token
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
            hidden, past_key_values = self._pipeline_forward_stage(
                input_ids=next_token.unsqueeze(1),
                hidden_states=None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                multimodal_inputs=None,
                timing=stage_timing,
            )
            logits = self._pipeline_logits_from_last_hidden(hidden, timing=stage_timing)
        self._sync_timing()
        return sequences, {
            "prefill_sec": prefill_sec,
            "decode_sec": time.monotonic() - decode_started_at,
            "decode_tokens": decode_tokens,
            "sampling_sec": sampling_sec,
            "stop_check_sec": stop_check_sec,
            **_round_pipeline_stage_timing(stage_timing),
        }

    def _pipeline_forward_stage(
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

    def _pipeline_logits_from_last_hidden(
        self,
        hidden_states: torch.Tensor | None,
        *,
        timing: dict[str, float | int] | None = None,
    ) -> torch.Tensor | None:
        assert isinstance(self.model, Qwen35HybridTextModel)
        if self.pp_rank != self.pp_size - 1:
            return None
        assert hidden_states is not None
        assert self.model.norm is not None and self.model.lm_head is not None
        lm_head_started_at = time.monotonic()
        hidden_states = self.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        _add_pipeline_stage_timing(timing, "pipeline_lm_head_sec", lm_head_started_at)
        return logits

    def _pipeline_sample_next_token(
        self,
        logits: torch.Tensor | None,
        *,
        temperature: float,
        top_p: float,
        timing: dict[str, float | int] | None = None,
    ) -> torch.Tensor:
        assert self.tp_state is not None
        if self.pp_rank == self.pp_size - 1:
            assert logits is not None
            sample_started_at = time.monotonic()
            next_token = _next_token_from_logits(
                logits.float()[:, -1, :], temperature=temperature, top_p=top_p
            )
            _add_pipeline_stage_timing(timing, "pipeline_sample_compute_sec", sample_started_at)
        else:
            next_token = torch.empty((0,), dtype=torch.long, device=self.device)
        batch_size = torch.tensor([int(next_token.shape[0])], dtype=torch.long, device=self.device)
        broadcast_started_at = time.monotonic()
        dist.broadcast(batch_size, src=(self.pp_size - 1) * self.tp_size)
        if self.pp_rank != self.pp_size - 1:
            next_token = torch.empty(
                (int(batch_size.item()),), dtype=torch.long, device=self.device
            )
        dist.broadcast(next_token, src=(self.pp_size - 1) * self.tp_size)
        _add_pipeline_stage_timing(timing, "pipeline_token_broadcast_sec", broadcast_started_at)
        return next_token

    def sequence_log_probs(  # type: ignore[override]
        self,
        sequences: Any,
        attention_mask: Any,
        metadata: Any | None = None,
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
                    raise ValueError(
                        "multimodal metadata was provided for a non-multimodal native model"
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

    def train_batch(  # type: ignore[override]
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
        if bool(self.config.native_tp.empty_cache_before_train) and self.device.type == "cuda":
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
        self._emit_rank_memory_event(
            "train_batch_start",
            {
                "experience_count": len(experiences),
                "optimize_prompt_batch_size": batch_size,
                "optimize_times_per_step": optimize_times_per_step,
                "train_batch_call_index": self._train_batch_call_index,
            },
        )
        for optimize_round in range(optimize_times_per_step):
            round_started_at = time.monotonic()
            indices = self._shared_training_indices(len(experiences), optimize_round=optimize_round)
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
                            "multimodal batch metadata was provided for a non-multimodal model"
                        )
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
                self._sync_timing()
                backward_started_at = time.monotonic()
                loss.backward()
                # Fix: sync non-sharded LoRA gradients across TP ranks BEFORE
                # optimizer step.  In TP-sharded layers, the non-sharded LoRA
                # matrix receives different gradients per rank.  All-reducing
                # gradients before the optimizer step keeps both weights AND
                # Adam state in sync across ranks.
                from graspo.backends.native_tp.models.qwen.lora import _sync_nonsharded_lora_grads
                from graspo.backends.native_tp.tensor_utils import _TENSOR_PARALLEL_GROUP

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
            "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "synchronize_cuda_timing": bool(self.config.native_tp.synchronize_cuda_timing),
        }
        metrics = self._aggregate_rank_metrics(metrics)
        self._emit_rank_memory_event("train_batch_after", {"metrics": metrics})
        return metrics

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
            hidden, _ = self._pipeline_forward_stage(
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
                "placement": placement_summary(self.placement)
                if self.placement is not None
                else None,
                "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            },
        )
        return log_probs

    def _pipeline_train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        assert isinstance(self.model, Qwen35HybridTextModel)
        assert self.tp_state is not None
        schedule = str(self.config.native_tp.pp_schedule or "simple")
        if schedule == "one_f_one_b":
            return self._pipeline_train_batch_one_f_one_b(
                experiences,
                policy_ratio_clip_eps=policy_ratio_clip_eps,
                optimize_times_per_step=optimize_times_per_step,
                max_grad_norm=max_grad_norm,
            )
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
            indices = self._shared_training_indices(len(experiences), optimize_round=optimize_round)
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
                    _add_pipeline_stage_timing(stage_timing, "pipeline_norm_sec", norm_started_at)
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
                        log_probs, batch.old_log_probs, batch.advantages, batch.action_mask
                    )
                    _add_pipeline_stage_timing(stage_timing, "pipeline_loss_sec", loss_started_at)
                    finite = bool(torch.isfinite(loss).detach().cpu())
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
                    autograd_started_at = time.monotonic()
                    loss.backward()
                    _add_pipeline_stage_timing(
                        stage_timing, "pipeline_backward_autograd_sec", autograd_started_at
                    )
                    if stage_input is not None and stage_input.grad is not None:
                        grad_send_started_at = time.monotonic()
                        dist.send(
                            stage_input.grad.contiguous(), dst=int(self.tp_state.prev_pp_rank)
                        )
                        _add_pipeline_stage_timing(
                            stage_timing, "pipeline_grad_send_sec", grad_send_started_at
                        )
                    loss_value = float(loss.detach().cpu())
                else:
                    assert stage_output is not None
                    grad_output = torch.empty_like(stage_output)
                    grad_recv_started_at = time.monotonic()
                    dist.recv(grad_output, src=int(self.tp_state.next_pp_rank))
                    _add_pipeline_stage_timing(
                        stage_timing, "pipeline_grad_recv_sec", grad_recv_started_at
                    )
                    autograd_started_at = time.monotonic()
                    stage_output.backward(grad_output)
                    _add_pipeline_stage_timing(
                        stage_timing, "pipeline_backward_autograd_sec", autograd_started_at
                    )
                    if stage_input is not None and stage_input.grad is not None:
                        grad_send_started_at = time.monotonic()
                        dist.send(
                            stage_input.grad.contiguous(), dst=int(self.tp_state.prev_pp_rank)
                        )
                        _add_pipeline_stage_timing(
                            stage_timing, "pipeline_grad_send_sec", grad_send_started_at
                        )
                    loss_value = 0.0
                self._sync_timing()
                backward_sec += time.monotonic() - backward_started_at
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
            "replay_buffer_trainable_completion_count": len(experiences),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / optimizer_steps if optimizer_steps else None,
            "grad_norm_mean": grad_norm_sum / optimizer_steps if optimizer_steps else None,
            "nonzero_grad_count": nonzero_grad_count,
            "lora_norm_before": lora_norm_before,
            "lora_norm_after": lora_norm_after,
            "lora_norm_delta": lora_norm_after - lora_norm_before,
            "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "pp_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": self.placement.strategy if self.placement is not None else None,
            "pp_schedule": "simple",
            "pp_max_inflight_microbatches": 1,
            "pipeline_stage_timing": _round_pipeline_stage_timing(stage_timing),
            "synchronize_cuda_timing": bool(self.config.native_tp.synchronize_cuda_timing),
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
        pipeline_micro_batch_size = max(1, int(self.config.native_tp.pp_micro_batch_size))
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
        configured_inflight = int(self.config.native_tp.pp_max_inflight_microbatches)
        for optimize_round in range(optimize_times_per_step):
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
            "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
            "train_batch_total_sec": time.monotonic() - train_batch_started_at,
            "optimize_round_sec": round_secs,
            "optimize_round_sec_sum": sum(round_secs),
            "micro_batch_forward_sec": micro_batch_forward_sec,
            "backward_sec": backward_sec,
            "optimizer_step_sec": optimizer_step_sec,
            "micro_batch_count": micro_batch_count,
            "pp_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": self.placement.strategy if self.placement is not None else None,
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
            "synchronize_cuda_timing": bool(self.config.native_tp.synchronize_cuda_timing),
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
                    log_probs, batch.old_log_probs, batch.advantages, batch.action_mask
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
                    dist.send(grad.contiguous(), dst=int(self.tp_state.prev_pp_rank or 0))
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
                    dist.send(grad.contiguous(), dst=int(self.tp_state.prev_pp_rank or 0))
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

    def _shared_training_indices(self, experience_count: int, *, optimize_round: int) -> list[int]:
        indices = list(range(experience_count))
        seed = (
            int(self.config.training.seed)
            + (self._train_batch_call_index * 1_000_003)
            + int(optimize_round)
        )
        if dist.is_available() and dist.is_initialized():
            payload: list[list[int] | None] = [None]
            if self.rank == 0:
                random.Random(seed).shuffle(indices)
                payload[0] = indices
            dist.broadcast_object_list(payload, src=0)
            shared = payload[0]
            if shared is None:
                raise RuntimeError("Failed to broadcast shared GRASPO train-batch shuffle indices")
            return list(shared)
        random.Random(seed).shuffle(indices)
        return indices

    def _shared_generation_micro_batch_size(
        self,
        *,
        prompt_len: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        # User controls this directly via forward_batch_size, clamped to available rows.
        return max(1, min(int(self.config.native_tp.forward_batch_size), int(rollout_group_size)))

    def _shared_rollout_prompt_chunk_size(
        self,
        *,
        prompt_len: int,
        requested_prompt_count: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        # Encode all queued samples together (no more auto-chunking).
        return max(1, int(requested_prompt_count))

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        trainer_state: dict[str, Any] | None = None,
    ) -> None:
        self._require_ready()
        assert self.model is not None
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "adapter": "qwen_native_tp",
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "placement": placement_summary(self.placement) if self.placement is not None else None,
            "lora_target_signature": self.model.lora_target_signature(),
            "lora_tensor_metadata": self.model.lora_tensor_metadata(),
            "lora_state_dict": self.model.lora_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            "adapter_state": {
                "train_batch_call_index": self._train_batch_call_index,
            },
            "trainer_state": trainer_state,
            "config": asdict(self.config),
        }
        torch.save(
            payload, output / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}_pp_{self.pp_rank:02d}.pt"
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self._emit_rank_memory_event("checkpoint_after", {"checkpoint_dir": str(output)})
        if self.rank == 0:
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "graspo-native-tp-lora",
                        "tp_size": self.tp_size,
                        "pp_size": self.pp_size,
                        "placement": placement_summary(self.placement)
                        if self.placement is not None
                        else None,
                        "lora_target_signature": self.model.lora_target_signature(),
                        "world_size": self.world_size,
                        "checkpoint_type": "recoverable_lora_training_state",
                        "has_trainer_state": trainer_state is not None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None:
        self._require_ready()
        assert self.model is not None
        checkpoint_dir = Path(path)
        rank_path = (
            checkpoint_dir / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}_pp_{self.pp_rank:02d}.pt"
        )
        if not rank_path.exists():
            raise FileNotFoundError(
                "Missing current GRASPO checkpoint shard "
                f"for rank={self.rank} tp_rank={self.tp_rank} pp_rank={self.pp_rank}: {rank_path}"
            )
        try:
            payload = torch.load(rank_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(rank_path, map_location=self.device)
        if int(payload.get("tp_size", self.tp_size)) != self.tp_size:
            raise ValueError(
                f"Checkpoint TP size {payload.get('tp_size')} does not match runtime TP size {self.tp_size}"
            )
        if int(payload.get("pp_size", self.pp_size)) != self.pp_size:
            raise ValueError(
                f"Checkpoint PP size {payload.get('pp_size')} does not match runtime PP size {self.pp_size}"
            )
        checkpoint_signature = payload.get("lora_target_signature")
        current_signature = self.model.lora_target_signature()
        if checkpoint_signature is not None and checkpoint_signature != current_signature:
            raise ValueError(
                "Checkpoint LoRA target signature does not match runtime configuration: "
                f"checkpoint={checkpoint_signature}, runtime={current_signature}"
            )
        missing, unexpected = self.model.load_state_dict(payload["lora_state_dict"], strict=False)
        unexpected_lora = [name for name in unexpected if "lora_" in name]
        if unexpected_lora:
            raise RuntimeError(f"Unexpected LoRA tensors in checkpoint: {unexpected_lora}")
        missing_lora = [name for name in missing if "lora_" in name]
        if missing_lora:
            raise RuntimeError(f"Missing LoRA tensors while loading checkpoint: {missing_lora}")
        optimizer_state = payload.get("optimizer_state_dict")
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        elif self.optimizer is not None and optimizer_state is None:
            raise RuntimeError("Checkpoint shard is missing optimizer state for a trainable rank")
        torch.set_rng_state(payload["torch_rng_state"].detach().cpu())
        cuda_rng_state = payload.get("cuda_rng_state")
        if cuda_rng_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state.to("cpu"), self.device)
        adapter_state = payload.get("adapter_state") or {}
        self._train_batch_call_index = int(adapter_state.get("train_batch_call_index") or 0)
        self._emit_rank_memory_event(
            "checkpoint_loaded",
            {
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_rank_file": str(rank_path),
                "has_trainer_state": payload.get("trainer_state") is not None,
                "train_batch_call_index": self._train_batch_call_index,
            },
        )
        return payload.get("trainer_state")

    def close(self) -> None:
        destroy_native_tp()

    def _setup_distributed(self) -> None:
        state = NativeTPState.initialize(self.tp_size, self.pp_size)
        self.tp_state = state
        self.rank = state.rank
        self.local_rank = state.local_rank
        self.world_size = state.world_size
        self.tp_rank = state.tp_rank
        self.pp_rank = state.pp_rank
        self.device = state.device
        _set_tensor_parallel_group(state.tp_group, state.tp_size)

    def format_messages(
        self,
        messages: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        return self._format_messages(messages, chat_template_kwargs, tools=tools)

    def _format_messages(
        self,
        messages: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        assert self.tokenizer is not None
        template_kwargs = dict(chat_template_kwargs or {})
        if tools is not None:
            template_kwargs["tools"] = tools
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        tools_text = ""
        if tools is not None:
            tools_text = "\n\ntools: " + json.dumps(tools, ensure_ascii=False)
        return (
            "\n\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in messages
            )
            + tools_text
        )

    def parse_completion(self, completion: str, sample: Any | None = None) -> ParsedCompletion:
        return parse_qwen_tool_completion(
            completion,
            expect_tool_calls=bool(getattr(sample, "expects_tool_calls", False)),
            tools=getattr(sample, "tools", None),
        )

    def _encode_multimodal_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.processor is None:
            raise RuntimeError(
                "This model did not expose an AutoProcessor; image/video samples cannot be encoded"
            )
        messages = [_processor_chat_messages(_messages_from_multimodal_row(row)) for row in rows]
        tool_batches = [_tools_from_multimodal_row(row) for row in rows]
        if hasattr(self.processor, "apply_chat_template"):
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": add_generation_prompt,
                "return_dict": True,
                "return_tensors": "pt",
                **(chat_template_kwargs or {}),
            }
            tools_arg = _tools_for_chat_template(tool_batches)
            if tools_arg is not None:
                template_kwargs["tools"] = tools_arg
            try:
                encoded = self.processor.apply_chat_template(
                    messages,
                    processor_kwargs={"padding": True},
                    **template_kwargs,
                )
            except TypeError:
                encoded = self.processor.apply_chat_template(
                    messages,
                    padding=True,
                    **template_kwargs,
                )
        else:
            raise RuntimeError(
                "AutoProcessor does not implement apply_chat_template for multimodal samples"
            )
        return dict(encoded)

    def _multimodal_inputs_to_device(self, encoded: dict[str, Any]) -> dict[str, torch.Tensor]:
        keys = (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
        )
        moved: dict[str, torch.Tensor] = {}
        for key in keys:
            value = encoded.get(key)
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
        return moved

    def _multimodal_inputs_from_metadata(
        self,
        metadata: Any | None,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor] | None:
        rows = _multimodal_rows_from_metadata(metadata, expected_rows=batch_size)
        if not rows:
            return None
        encoded = self._encode_multimodal_rows(
            rows,
            add_generation_prompt=True,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        return self._multimodal_inputs_to_device(encoded)

    def _is_pipeline_parallel(self) -> bool:
        return bool(self.placement is not None and self.placement.is_pipeline)

    def _require_ready(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("QwenNativeTPAdapter is not set up")

    def _print_rank0(self, payload: dict[str, Any]) -> None:
        if self.rank == 0:
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _sync_timing(self) -> None:
        if (
            bool(self.config.native_tp.synchronize_cuda_timing)
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def is_primary(self) -> bool:
        return self.rank == 0

    def _aggregate_rank_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        local = {"rank": self.rank, "tp_rank": self.tp_rank, **metrics}
        if not (dist.is_available() and dist.is_initialized()):
            return {**metrics, "rank": self.rank, "tp_rank": self.tp_rank, "rank_metrics": [local]}
        gathered: list[dict[str, Any] | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local)
        ranks = [item for item in gathered if item is not None]
        return {
            **metrics,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "rank_metrics": ranks,
            "global_optimizer_steps_sum": sum(
                int(item.get("optimizer_steps") or 0) for item in ranks
            ),
            "global_nonzero_grad_count_sum": sum(
                int(item.get("nonzero_grad_count") or 0) for item in ranks
            ),
            "global_loss_mean": _mean_present(item.get("loss_mean") for item in ranks),
            "global_grad_norm_mean": _mean_present(item.get("grad_norm_mean") for item in ranks),
            "global_lora_norm_delta_mean": _mean_present(
                item.get("lora_norm_delta") for item in ranks
            ),
        }

    def _emit_rank_memory_event(self, phase: str, extra: dict[str, Any] | None = None) -> None:
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": "rank_memory",
            "phase": phase,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "device": str(self.device),
            "memory": _cuda_memory_snapshot(self.device),
        }
        if extra:
            payload.update(extra)
        path = output_dir / f"rank_metrics.rank_{self.rank:05d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
