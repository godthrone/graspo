from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.nn.utils.rnn import pad_sequence

from graspo.backends.native_tp.parallel_state import NativeTPState, destroy_native_tp
from graspo.backends.native_tp.placement import (
    NativePlacementPlan,
    build_placement_plan,
    placement_summary,
)
from graspo.backends.native_tp.runtime import NativeGeneration
from graspo.core.buffer import Experience
from graspo.core.schema import GraspoConfig
from graspo.backends.native_tp.lora_io import load_peft_adapter_into_native_model
from graspo.trainer.lora import resolve_lora_target_modules
from graspo.trainer.loss import GRASPOLoss


_TENSOR_PARALLEL_GROUP: dist.ProcessGroup | None = None
_TENSOR_PARALLEL_SIZE = 1


def _set_tensor_parallel_group(group: dist.ProcessGroup | None, size: int) -> None:
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_SIZE
    _TENSOR_PARALLEL_GROUP = group
    _TENSOR_PARALLEL_SIZE = int(size)


class QwenNativeTPAdapter:
    """Qwen causal LM adapter backed by self-owned PyTorch tensor parallel."""

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.tp_size = int(config.native_tp.tensor_model_parallel_size)
        self.tp_rank = 0
        self.pp_size = int(config.native_tp.pipeline_model_parallel_size)
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
                "rollout_kv_cache_max_reserved_fraction": self.config.native_tp.rollout_kv_cache_max_reserved_fraction,
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
        prompt: str,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> NativeGeneration:
        return self.generate_groups(
            prompts=[prompt],
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
        )[0]

    def generate_groups(
        self,
        *,
        prompts: list[str],
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
        if self._is_pipeline_parallel():
            return self._pipeline_generate_groups(
                prompts=prompts,
                rollout_group_size=rollout_group_size,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                chat_template_kwargs=chat_template_kwargs,
            )
        if not prompts:
            return []
        self.model.eval()
        tokenize_started_at = time.monotonic()
        prompt_texts = [self._format_prompt(prompt, chat_template_kwargs) for prompt in prompts]
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
        requested_prompt_queue_size = len(prompts)
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
                    int(gen.metadata.get("rollout_generation_micro_batch_size", 1))
                    for gen in all_generations
                ),
                default=1,
            ),
            "rollout_generation_split_count": sum(
                int(gen.metadata.get("rollout_generation_split_count", 1))
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
                    int(gen.metadata.get("rollout_generation_split_count", 1)) > 1
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

    def generate_sample_groups(
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
            return self._pipeline_generate_sample_groups(
                samples=samples,
                rollout_group_size=rollout_group_size,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                chat_template_kwargs=chat_template_kwargs,
            )
        return [
            self._generate_multimodal_sample_group(
                sample=sample,
                rollout_group_size=rollout_group_size,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                chat_template_kwargs=chat_template_kwargs,
            )
            for sample in samples
        ]

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
        sequences = input_ids
        finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            if use_kv_cache:
                sequences, timing = self._generate_multimodal_with_kv_cache(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    multimodal_inputs=multimodal_inputs,
                    finished=finished,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                sequences, timing = self._generate_multimodal_full_forward(
                    sequences=sequences,
                    multimodal_inputs=multimodal_inputs,
                    finished=finished,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
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
            generation_micro_batch_size=rollout_group_size,
            split_count=1,
            tokenize_sec=tokenize_sec,
            chunk_timings=[timing],
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
                **_rollout_timing_summary(tokenize_sec, [timing]),
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
        decode_started_at = time.monotonic()
        decode_tokens = 0
        sampling_sec = 0.0
        stop_check_sec = 0.0
        for _ in range(max_new_tokens):
            self._sync_timing()
            sampling_started_at = time.monotonic()
            next_token = _next_token_from_logits(
                logits.float()[:, -1, :], temperature=temperature, top_p=top_p
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
        prompt_len = int(sequences.shape[1])
        for _ in range(max_new_tokens):
            attention_mask = sequences.ne(pad_token_id)
            logits = self.model(
                sequences,
                attention_mask=attention_mask,
                multimodal_inputs=multimodal_inputs
                if int(sequences.shape[1]) == prompt_len
                else None,
            ).float()[:, -1, :]
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
        prompts: list[str],
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> list[NativeGeneration]:
        assert self.tokenizer is not None
        assert self.model is not None
        if not prompts:
            return []
        self.model.eval()
        tokenize_started_at = time.monotonic()
        prompt_texts = [self._format_prompt(prompt, chat_template_kwargs) for prompt in prompts]
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
        requested_prompt_queue_size = len(prompts)
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
                        int(gen.metadata.get("rollout_generation_micro_batch_size", 1))
                        for gen in all_generations
                    ),
                    default=1,
                ),
                "rollout_generation_split_count": sum(
                    int(gen.metadata.get("rollout_generation_split_count", 1))
                    for gen in all_generations
                ),
                **_rollout_timing_summary(tokenize_sec, all_timings),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
            },
        )
        return all_generations

    def _pipeline_generate_sample_groups(
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
        return [
            self._pipeline_generate_multimodal_sample_group(
                sample=sample,
                rollout_group_size=rollout_group_size,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                chat_template_kwargs=chat_template_kwargs,
            )
            for sample in samples
        ]

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
        with torch.no_grad():
            sequences, timing = self._pipeline_generate_sequences_with_cache(
                sequences=sequences,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                multimodal_inputs=multimodal_inputs,
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
            generation_micro_batch_size=rollout_group_size,
            split_count=1,
            tokenize_sec=tokenize_sec,
            chunk_timings=[timing],
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
            multimodal_inputs=multimodal_inputs if self.pp_rank == 0 else None,
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

    def sequence_log_probs(
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
        if bool(self.config.native_tp.empty_cache_before_train) and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self._emit_rank_memory_event("train_before_empty_cache")

        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        grad_norm_sum = 0.0
        nonzero_grad_count = 0
        lora_norm_before = self.model.lora_parameter_norm()
        batch_size = int(self.config.training.optimize_completion_batch_size)
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
                "optimize_completion_batch_size": batch_size,
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
        schedule = str(self.config.native_tp.pipeline_train_schedule or "simple")
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
        batch_size = int(self.config.training.optimize_completion_batch_size)
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
            "pipeline_model_parallel_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": self.placement.strategy if self.placement is not None else None,
            "pipeline_train_schedule": "simple",
            "pipeline_max_inflight_microbatches": 1,
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
        batch_size = int(self.config.training.optimize_completion_batch_size)
        pipeline_micro_batch_size = max(1, int(self.config.native_tp.train_micro_batch_size))
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
        configured_inflight = int(self.config.native_tp.pipeline_max_inflight_microbatches)
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
            "pipeline_model_parallel_size": self.pp_size,
            "pipeline_stage_rank": self.pp_rank,
            "placement_strategy": self.placement.strategy if self.placement is not None else None,
            "pipeline_train_schedule": "one_f_one_b",
            "pipeline_train_micro_batch_size": pipeline_micro_batch_size,
            "pipeline_chunks_per_optimizer_step": max_chunks_per_optimizer_step,
            "pipeline_max_inflight_microbatches": effective_inflight,
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
                    dist.send(grad.contiguous(), dst=int(self.tp_state.prev_pp_rank))
                    _add_pipeline_stage_timing(
                        timing, "pipeline_grad_send_sec", grad_send_started_at
                    )
            else:
                stage_output = record["stage_output"]
                assert stage_output is not None
                grad_output = torch.empty_like(stage_output)
                grad_recv_started_at = time.monotonic()
                assert self.tp_state is not None
                dist.recv(grad_output, src=int(self.tp_state.next_pp_rank))
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
                    dist.send(grad.contiguous(), dst=int(self.tp_state.prev_pp_rank))
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
        if self.pp_rank == 0:
            multimodal_inputs = self._multimodal_inputs_from_metadata(metadata, batch_size=batch)
            compute_started_at = time.monotonic()
            output = self.model.forward_stage(
                None,
                sequences,
                attention_mask,
                past_key_values=None,
                use_cache=False,
                multimodal_inputs=multimodal_inputs,
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
        local_size = self._resolve_generation_micro_batch_size(
            prompt_len=prompt_len,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        if not (dist.is_available() and dist.is_initialized()):
            return local_size
        payload: list[int | None] = [local_size if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        shared_size = payload[0]
        if shared_size is None:
            raise RuntimeError("Failed to broadcast shared rollout generation micro-batch size")
        return max(1, min(int(shared_size), int(rollout_group_size)))

    def _shared_rollout_prompt_chunk_size(
        self,
        *,
        prompt_len: int,
        requested_prompt_count: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        local_size = self._resolve_rollout_prompt_chunk_size(
            prompt_len=prompt_len,
            requested_prompt_count=requested_prompt_count,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        if not (dist.is_available() and dist.is_initialized()):
            return local_size
        payload: list[int | None] = [local_size if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        shared_size = payload[0]
        if shared_size is None:
            raise RuntimeError("Failed to broadcast shared rollout prompt chunk size")
        return max(1, min(int(shared_size), int(requested_prompt_count)))

    def _resolve_rollout_prompt_chunk_size(
        self,
        *,
        prompt_len: int,
        requested_prompt_count: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        requested = max(1, int(requested_prompt_count))
        if not use_kv_cache or self.device.type != "cuda" or self.model is None:
            return requested
        candidate = requested
        while candidate > 1 and not self._kv_cache_batch_fits_budget(
            batch_size=candidate * int(rollout_group_size),
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        ):
            candidate = max(1, candidate // 2)
        return max(1, min(requested, candidate))

    def _resolve_generation_micro_batch_size(
        self,
        *,
        prompt_len: int,
        rollout_group_size: int,
        max_new_tokens: int,
        use_kv_cache: bool,
    ) -> int:
        configured = max(1, int(self.config.native_tp.generation_micro_batch_size))
        if not use_kv_cache or self.device.type != "cuda" or self.model is None:
            return min(rollout_group_size, configured)
        candidate = min(rollout_group_size, max(configured, rollout_group_size))
        while candidate > 1 and not self._kv_cache_batch_fits_budget(
            batch_size=candidate,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        ):
            candidate = max(1, candidate // 2)
        return max(1, min(rollout_group_size, candidate))

    def _kv_cache_batch_fits_budget(
        self, *, batch_size: int, prompt_len: int, max_new_tokens: int
    ) -> bool:
        assert self.model is not None
        if self.device.type != "cuda":
            return True
        fraction = float(self.config.native_tp.rollout_kv_cache_max_reserved_fraction)
        fraction = min(max(fraction, 0.05), 1.0)
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        reserved = int(torch.cuda.memory_reserved(self.device))
        budget = max(0, int(total * fraction) - reserved)
        return (
            self.model.estimate_kv_cache_bytes(
                batch_size=batch_size,
                sequence_len=prompt_len + max_new_tokens,
            )
            <= budget
        )

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
            candidates = sorted(
                checkpoint_dir.glob(f"rank_*_tp_{self.tp_rank:02d}_pp_{self.pp_rank:02d}.pt")
            )
            if not candidates and self.pp_size == 1:
                candidates = sorted(checkpoint_dir.glob(f"rank_*_tp_{self.tp_rank:02d}.pt"))
            if len(candidates) == 1:
                rank_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Missing native TP checkpoint shard for rank={self.rank} tp_rank={self.tp_rank}: {rank_path}"
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

    def _format_prompt(self, prompt: str, chat_template_kwargs: dict[str, Any] | None) -> str:
        assert self.tokenizer is not None
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
        return prompt

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
        messages = [_messages_from_multimodal_row(row) for row in rows]
        if hasattr(self.processor, "apply_chat_template"):
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": add_generation_prompt,
                "return_dict": True,
                "return_tensors": "pt",
                **(chat_template_kwargs or {}),
            }
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


class NativeQwenConfig:
    def __init__(self, values: dict[str, Any], *, family: str, key_prefix: str) -> None:
        self.family = family
        self.key_prefix = key_prefix
        for key, value in values.items():
            setattr(self, key, value)


def _multimodal_row_from_sample(sample: Any) -> dict[str, Any]:
    return {
        "prompt": str(sample.prompt),
        "media": [
            {
                "type": str(item.get("type") or "unknown"),
                "path": str(item.get("path") or item.get("url") or ""),
            }
            for item in (sample.media or [])
            if isinstance(item, dict)
        ],
    }


def _messages_from_multimodal_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    for item in row.get("media") or []:
        if not isinstance(item, dict):
            continue
        media_type = str(item.get("type") or "").lower()
        path = str(item.get("path") or "")
        if not path:
            continue
        if media_type == "image":
            content.append({"type": "image", "image": path})
        elif media_type == "video":
            content.append({"type": "video", "video": path})
        else:
            raise ValueError(f"unsupported multimodal media type: {media_type!r}")
    content.append({"type": "text", "text": str(row.get("prompt") or "")})
    return [{"role": "user", "content": content}]


def _multimodal_rows_from_metadata(
    metadata: Any | None, *, expected_rows: int
) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        rows: list[dict[str, Any]] = []
        for item in metadata:
            if isinstance(item, dict):
                rows.extend(_multimodal_rows_from_metadata(item, expected_rows=1))
        if not rows:
            return []
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} multimodal metadata rows, got {len(rows)}"
            )
        return rows
    if not isinstance(metadata, dict):
        return []
    rows = metadata.get("_multimodal_rows")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise RuntimeError("metadata['_multimodal_rows'] must be a list")
    if len(rows) == 1 and expected_rows > 1:
        return [dict(rows[0]) for _ in range(expected_rows)]
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} multimodal metadata rows, got {len(rows)}")
    return [dict(row) for row in rows]


def _media_counts(media: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown") if isinstance(item, dict) else "unknown"
        counts[media_type] = counts.get(media_type, 0) + 1
    return counts


def native_qwen_lora_available_targets(hf_config: NativeQwenConfig) -> tuple[str, ...]:
    language_mlp = (
        "language.mlp.gate_proj",
        "language.mlp.up_proj",
        "language.mlp.down_proj",
    )
    if hf_config.family == "qwen3":
        return (
            "language.self_attn.q_proj",
            "language.self_attn.k_proj",
            "language.self_attn.v_proj",
            "language.self_attn.o_proj",
            *language_mlp,
        )
    if hf_config.family == "qwen3_5_text":
        targets: tuple[str, ...] = (
            "language.full_attn.q_proj",
            "language.full_attn.k_proj",
            "language.full_attn.v_proj",
            "language.full_attn.o_proj",
            "language.linear_attn.q_proj",
            "language.linear_attn.k_proj",
            "language.linear_attn.v_proj",
            "language.linear_attn.in_proj_z",
            "language.linear_attn.out_proj",
            *language_mlp,
        )
        if bool(getattr(hf_config, "has_vision_config", False)):
            depth = int((getattr(hf_config, "vision_config", {}) or {}).get("depth") or 0)
            visual_block_targets = tuple(
                target
                for idx in range(depth)
                for target in (
                    f"visual.blocks.{idx}.attn.qkv",
                    f"visual.blocks.{idx}.attn.proj",
                    f"visual.blocks.{idx}.mlp.linear_fc1",
                    f"visual.blocks.{idx}.mlp.linear_fc2",
                )
            )
            targets = (
                *targets,
                "visual.merger.linear_fc1",
                "visual.merger.linear_fc2",
                *visual_block_targets,
            )
        return targets
    return ()


def _lora_target_enabled(lora_targets: set[str], canonical_name: str) -> bool:
    return canonical_name in lora_targets or canonical_name.rsplit(".", 1)[-1] in lora_targets


class NativeTPCausalLMBase(nn.Module):
    """Shared contract for repository-native tensor-parallel causal LMs.

    Unknown models should fail closed in the registry instead of inheriting this
    class and silently attempting an unsafe best-effort sharding scheme.
    """

    supports_kv_cache = False

    def sequence_log_probs(
        self, sequences: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def estimate_kv_cache_bytes(self, *, batch_size: int, sequence_len: int) -> int:
        raise NotImplementedError

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu() for name, param in self.named_parameters() if "lora_" in name
        }

    def lora_tensor_metadata(self) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        for module_name, module in self.named_modules():
            if not isinstance(module, LoRALinear) or not module.lora_enabled:
                continue
            metadata.append(module.lora_metadata(module_name))
        return metadata

    def lora_parameter_norm(self) -> float:
        total = 0.0
        for name, param in self.named_parameters():
            if "lora_" in name:
                total += float(param.detach().float().pow(2).sum().cpu())
        return math.sqrt(total)

    def nonzero_lora_grad_count(self) -> int:
        return sum(
            int(param.grad is not None and bool(param.grad.detach().abs().sum().cpu() > 0))
            for name, param in self.named_parameters()
            if "lora_" in name
        )

    def enabled_lora_target_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for _, module in self.named_modules():
            if isinstance(module, LoRALinear) and module.lora_enabled:
                names.add(str(module.lora_target_name))
        return tuple(sorted(names))

    def lora_target_signature(self) -> dict[str, object]:
        return {
            "resolved": list(self.enabled_lora_target_names()),
            "parameter_count": sum(
                param.numel() for name, param in self.named_parameters() if "lora_" in name
            ),
        }


class QwenFamilyBase(NativeTPCausalLMBase):
    """Common Qwen native-TP helpers shared by Qwen generations."""


def load_native_qwen_config(model_path: Path) -> NativeQwenConfig:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "")
    if model_type == "qwen3":
        return NativeQwenConfig(config, family="qwen3", key_prefix="model")
    text_config = dict(config.get("text_config") or {})
    if model_type == "qwen3_5" and text_config.get("model_type") == "qwen3_5_text":
        text_config["has_vision_config"] = "vision_config" in config
        text_config["vision_config"] = dict(config.get("vision_config") or {})
        text_config["image_token_id"] = config.get("image_token_id")
        text_config["video_token_id"] = config.get("video_token_id")
        text_config["root_model_type"] = model_type
        return NativeQwenConfig(
            text_config, family="qwen3_5_text", key_prefix="model.language_model"
        )
    raise ValueError(
        f"native-tp supports text-only qwen3 and qwen3_5_text models; got model_type={model_type!r}"
    )


def build_native_qwen_model(
    *,
    hf_config: NativeQwenConfig,
    loader: "SafetensorIndex",
    tp_rank: int,
    tp_size: int,
    placement: NativePlacementPlan | None = None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: set[str],
    gradient_checkpointing: bool,
    torch_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    if hf_config.family == "qwen3":
        return Qwen3DenseModel(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            placement=placement,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device=device,
        )
    if hf_config.family == "qwen3_5_text":
        return Qwen35HybridTextModel(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            placement=placement,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported native Qwen family: {hf_config.family}")


def _build_qwen35_visual_tower(
    *,
    hf_config: NativeQwenConfig,
    loader: "SafetensorIndex",
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: set[str],
    torch_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qwen3.5-family multimodal training requires transformers Qwen3.5 vision classes"
        ) from exc

    vision_values = dict(getattr(hf_config, "vision_config", {}) or {})
    if not vision_values:
        raise RuntimeError("Qwen3.5-family config has no vision_config")
    vision_config = Qwen3_5VisionConfig(**vision_values)
    if hasattr(vision_config, "_attn_implementation"):
        vision_config._attn_implementation = "sdpa"
    visual = Qwen3_5VisionModel(vision_config).to(device=device, dtype=torch_dtype)
    state: dict[str, torch.Tensor] = {}
    prefix = "model.visual."
    for key in loader.weight_map:
        if key.startswith(prefix):
            state[key[len(prefix) :]] = loader.get(key).to(device=device, dtype=torch_dtype)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load Qwen3.5 visual tower weights: "
            f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    for param in visual.parameters():
        param.requires_grad = False
    _replace_visual_lora_modules(
        visual,
        lora_targets=lora_targets,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
        torch_dtype=torch_dtype,
    )
    return visual


def _replace_visual_lora_modules(
    visual: nn.Module,
    *,
    lora_targets: set[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    target_to_path = {
        "visual.merger.linear_fc1": "merger.linear_fc1",
        "visual.merger.linear_fc2": "merger.linear_fc2",
    }
    depth = len(getattr(visual, "blocks", []))
    for idx in range(depth):
        target_to_path.update(
            {
                f"visual.blocks.{idx}.attn.qkv": f"blocks.{idx}.attn.qkv",
                f"visual.blocks.{idx}.attn.proj": f"blocks.{idx}.attn.proj",
                f"visual.blocks.{idx}.mlp.linear_fc1": f"blocks.{idx}.mlp.linear_fc1",
                f"visual.blocks.{idx}.mlp.linear_fc2": f"blocks.{idx}.mlp.linear_fc2",
            }
        )
    for target_name, module_path in target_to_path.items():
        if not _lora_target_enabled(lora_targets, target_name):
            continue
        parent, attr = _module_parent_and_attr(visual, module_path)
        linear = getattr(parent, attr)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"visual LoRA target {target_name} is not an nn.Linear")
        replacement = LoRALinear(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            lora_enabled=True,
            target_name=target_name,
            hf_module_path=f"model.visual.{module_path}",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        setattr(parent, attr, replacement)


def _module_parent_and_attr(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent: nn.Module = module
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if part.isdigit() and isinstance(parent, nn.ModuleList)
            else getattr(parent, part)
        )
    return parent, parts[-1]


class Qwen3DenseModel(QwenFamilyBase):
    def __init__(
        self,
        *,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        placement: NativePlacementPlan | None = None,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        gradient_checkpointing: bool,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = hf_config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.placement = placement
        self.device_ref = device
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.supports_kv_cache = True
        self.lora_targets = set(lora_targets)
        self.key_prefix = str(getattr(hf_config, "key_prefix", "model"))
        self.embed_tokens = nn.Embedding(
            hf_config.vocab_size, hf_config.hidden_size, device=device, dtype=torch_dtype
        )
        self.embed_tokens.weight.data.copy_(
            loader.get(f"{self.key_prefix}.embed_tokens.weight").to(
                device=device, dtype=torch_dtype
            )
        )
        self.layers = nn.ModuleList(
            [
                TensorParallelQwenDecoderLayer(
                    layer_idx=idx,
                    key_prefix=self.key_prefix,
                    hf_config=hf_config,
                    loader=loader,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_targets=lora_targets,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                for idx in range(hf_config.num_hidden_layers)
            ]
        )
        self.norm = QwenRMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.norm.weight.data.copy_(
            loader.get(f"{self.key_prefix}.norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.lm_head = nn.Linear(
            hf_config.hidden_size,
            hf_config.vocab_size,
            bias=False,
            device=device,
            dtype=torch_dtype,
        )
        lm_head = loader.get_optional("lm_head.weight")
        if lm_head is None:
            lm_head = loader.get(f"{self.key_prefix}.embed_tokens.weight")
        self.lm_head.weight.data.copy_(lm_head.to(device=device, dtype=torch_dtype))
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            past_len = int(past_key_values[0][0].shape[2]) if past_key_values else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_len + input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        position_ids = _position_ids(attention_mask)[:, -input_ids.shape[1] :]
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present)
            elif self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    _checkpoint_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, tuple(present_key_values)
        return logits

    def sequence_log_probs(
        self, sequences: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self._forward_hidden(sequences, attention_mask=attention_mask)
        assert isinstance(hidden_states, torch.Tensor)
        return _selected_token_log_probs_from_hidden(
            hidden_states[:, :-1].float(),
            self.lm_head.weight.float(),
            sequences[:, 1:],
        )

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            past_len = int(past_key_values[0][0].shape[2]) if past_key_values else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_len + input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        position_ids = _position_ids(attention_mask)[:, -input_ids.shape[1] :]
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present)
            elif self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    _checkpoint_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states

    def estimate_kv_cache_bytes(self, *, batch_size: int, sequence_len: int) -> int:
        dtype_size = _dtype_size(self.embed_tokens.weight.dtype)
        local_kv_heads = int(self.config.num_key_value_heads) // int(self.tp_size)
        head_dim = int(
            getattr(
                self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
            )
        )
        return (
            int(batch_size)
            * int(self.config.num_hidden_layers)
            * 2
            * local_kv_heads
            * head_dim
            * int(sequence_len)
            * dtype_size
        )


class TensorParallelQwenForCausalLM(Qwen3DenseModel):
    """Compatibility alias for older tests/imports."""


class Qwen35HybridTextModel(QwenFamilyBase):
    def __init__(
        self,
        *,
        hf_config: NativeQwenConfig,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        placement: NativePlacementPlan | None = None,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        gradient_checkpointing: bool,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = hf_config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.placement = placement
        self.device_ref = device
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.supports_kv_cache = True
        self.lora_targets = set(lora_targets)
        self.key_prefix = str(getattr(hf_config, "key_prefix", "model.language_model"))
        layer_types = list(getattr(hf_config, "layer_types", []) or [])
        if len(layer_types) != int(hf_config.num_hidden_layers):
            raise ValueError("qwen3_5_text layer_types length must match num_hidden_layers")

        include_embeddings = placement.include_embeddings if placement is not None else True
        include_lm_head = placement.include_lm_head if placement is not None else True
        local_layer_indices = (
            list(placement.local_layer_indices)
            if placement is not None
            else list(range(hf_config.num_hidden_layers))
        )
        self.embed_tokens = (
            nn.Embedding(
                hf_config.vocab_size, hf_config.hidden_size, device=device, dtype=torch_dtype
            )
            if include_embeddings
            else None
        )
        if self.embed_tokens is not None:
            self.embed_tokens.weight.data.copy_(
                loader.get(f"{self.key_prefix}.embed_tokens.weight").to(
                    device=device, dtype=torch_dtype
                )
            )
        self.visual = (
            _build_qwen35_visual_tower(
                hf_config=hf_config,
                loader=loader,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_targets=lora_targets,
                torch_dtype=torch_dtype,
                device=device,
            )
            if include_embeddings and bool(getattr(hf_config, "has_vision_config", False))
            else None
        )
        self.local_layer_indices = tuple(local_layer_indices)
        self.layers = nn.ModuleList(
            [
                TensorParallelQwen35DecoderLayer(
                    layer_idx=idx,
                    layer_type=layer_types[idx],
                    key_prefix=self.key_prefix,
                    hf_config=hf_config,
                    loader=loader,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_targets=lora_targets,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                for idx in local_layer_indices
            ]
        )
        self.norm = (
            Qwen35RMSNorm(
                hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
            )
            if include_lm_head
            else None
        )
        if self.norm is not None:
            self.norm.weight.data.copy_(
                loader.get(f"{self.key_prefix}.norm.weight").to(device=device, dtype=torch_dtype)
            )
        self.lm_head = (
            nn.Linear(
                hf_config.hidden_size,
                hf_config.vocab_size,
                bias=False,
                device=device,
                dtype=torch_dtype,
            )
            if include_lm_head
            else None
        )
        if self.lm_head is not None:
            lm_head = loader.get_optional("lm_head.weight")
            if lm_head is None:
                lm_head = loader.get(f"{self.key_prefix}.embed_tokens.weight")
            self.lm_head.weight.data.copy_(lm_head.to(device=device, dtype=torch_dtype))
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[Any, ...] | None = None,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, ...]]:
        hidden_states = self._forward_hidden(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            multimodal_inputs=multimodal_inputs,
            use_cache=use_cache,
        )
        if self.lm_head is None:
            raise RuntimeError("This Qwen3.5 stage does not own lm_head")
        if use_cache:
            hidden_states, present_key_values = hidden_states
            return self.lm_head(hidden_states), present_key_values
        assert isinstance(hidden_states, torch.Tensor)
        return self.lm_head(hidden_states)

    def sequence_log_probs(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("This Qwen3.5 stage does not own lm_head")
        hidden_states = self._forward_hidden(
            sequences,
            attention_mask=attention_mask,
            multimodal_inputs=multimodal_inputs,
        )
        assert isinstance(hidden_states, torch.Tensor)
        return _selected_token_log_probs_from_hidden(
            hidden_states[:, :-1].float(),
            self.lm_head.weight.float(),
            sequences[:, 1:],
        )

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[Any, ...] | None = None,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, ...]]:
        hidden_states = self.embed_inputs(input_ids, multimodal_inputs=multimodal_inputs)
        if attention_mask is None:
            past_len = _qwen35_cache_sequence_len(past_key_values[0]) if past_key_values else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_len + input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        position_ids = _position_ids(attention_mask)[:, -input_ids.shape[1] :]
        present_key_values: list[Any] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present)
            elif self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    _checkpoint_qwen35_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        if self.norm is None:
            raise RuntimeError("This Qwen3.5 stage does not own final norm")
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states

    def embed_inputs(
        self,
        input_ids: torch.Tensor,
        *,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("This Qwen3.5 stage does not own embeddings")
        hidden_states = self.embed_tokens(input_ids)
        if multimodal_inputs is None:
            return hidden_states
        if self.visual is None:
            raise RuntimeError(
                "Qwen3.5 multimodal inputs require visual tower on the embedding stage"
            )
        image_features = self._visual_features(multimodal_inputs, kind="image")
        if image_features is not None:
            image_token_id = int(getattr(self.config, "image_token_id"))
            image_mask = input_ids.eq(image_token_id).unsqueeze(-1).expand_as(hidden_states)
            if int(image_mask.sum().item()) != int(image_features.numel()):
                raise RuntimeError(
                    "Image features and image placeholder tokens do not match: "
                    f"tokens={int(image_mask.sum().item())}, features={int(image_features.numel())}"
                )
            hidden_states = hidden_states.masked_scatter(
                image_mask, image_features.to(hidden_states.dtype)
            )
        video_features = self._visual_features(multimodal_inputs, kind="video")
        if video_features is not None:
            video_token_id = int(getattr(self.config, "video_token_id"))
            video_mask = input_ids.eq(video_token_id).unsqueeze(-1).expand_as(hidden_states)
            if int(video_mask.sum().item()) != int(video_features.numel()):
                raise RuntimeError(
                    "Video features and video placeholder tokens do not match: "
                    f"tokens={int(video_mask.sum().item())}, features={int(video_features.numel())}"
                )
            hidden_states = hidden_states.masked_scatter(
                video_mask, video_features.to(hidden_states.dtype)
            )
        return hidden_states

    def _visual_features(
        self,
        multimodal_inputs: dict[str, torch.Tensor],
        *,
        kind: str,
    ) -> torch.Tensor | None:
        if kind == "image":
            pixel_values = multimodal_inputs.get("pixel_values")
            grid_thw = multimodal_inputs.get("image_grid_thw")
        else:
            pixel_values = multimodal_inputs.get("pixel_values_videos")
            grid_thw = multimodal_inputs.get("video_grid_thw")
        if pixel_values is None:
            return None
        if grid_thw is None:
            raise RuntimeError(f"{kind} pixel values were provided without grid_thw")
        assert self.visual is not None
        dtype = next(self.visual.parameters()).dtype
        output = self.visual(pixel_values.to(dtype=dtype), grid_thw=grid_thw)
        features = output.pooler_output if hasattr(output, "pooler_output") else output[1]
        return features.to(device=pixel_values.device)

    def forward_stage(
        self,
        hidden_states: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        *,
        past_key_values: tuple[Any, ...] | None = None,
        use_cache: bool = False,
        apply_lm_head: bool = False,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, ...]]:
        if hidden_states is None:
            if input_ids is None or self.embed_tokens is None:
                raise RuntimeError("Pipeline stage requires input_ids on the embedding stage")
            hidden_states = self.embed_inputs(input_ids, multimodal_inputs=multimodal_inputs)
            query_len = int(input_ids.shape[1])
        else:
            query_len = int(hidden_states.shape[1])
        position_ids = _position_ids(attention_mask)[:, -query_len:]
        present_key_values: list[Any] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present)
            elif self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    _checkpoint_qwen35_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        if apply_lm_head:
            if self.norm is None or self.lm_head is None:
                raise RuntimeError("Pipeline final stage requires norm and lm_head")
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states

    def estimate_kv_cache_bytes(self, *, batch_size: int, sequence_len: int) -> int:
        dtype_source = (
            self.embed_tokens.weight if self.embed_tokens is not None else next(self.parameters())
        )
        dtype_size = _dtype_size(dtype_source.dtype)
        total = 0
        layer_types = list(getattr(self.config, "layer_types", []) or [])
        full_head_dim = int(
            getattr(
                self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
            )
        )
        local_layers = set(getattr(self, "local_layer_indices", tuple(range(len(layer_types)))))
        for idx, layer_type in enumerate(layer_types):
            if idx not in local_layers:
                continue
            if layer_type == "full_attention":
                local_kv_heads = max(
                    1, math.ceil(int(self.config.num_key_value_heads) / int(self.tp_size))
                )
                total += (
                    int(batch_size)
                    * 2
                    * local_kv_heads
                    * full_head_dim
                    * int(sequence_len)
                    * dtype_size
                )
            elif layer_type == "linear_attention":
                local_v_heads = int(self.config.linear_num_value_heads) // int(self.tp_size)
                local_k_heads = int(self.config.linear_num_key_heads) // int(self.tp_size)
                key_dim = local_k_heads * int(self.config.linear_key_head_dim)
                value_dim = local_v_heads * int(self.config.linear_value_head_dim)
                conv_dim = 2 * key_dim + value_dim
                recurrent = (
                    int(batch_size)
                    * local_v_heads
                    * int(self.config.linear_key_head_dim)
                    * int(self.config.linear_value_head_dim)
                )
                conv = int(batch_size) * conv_dim * int(self.config.linear_conv_kernel_dim)
                total += (recurrent + conv) * dtype_size
        return int(total)


class TensorParallelQwen35TextForCausalLM(Qwen35HybridTextModel):
    """Compatibility alias for older tests/imports."""


class TensorParallelQwen35DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        layer_type: str,
        key_prefix: str,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        prefix = f"{key_prefix}.layers.{layer_idx}"
        self.layer_type = layer_type
        self.input_layernorm = Qwen35RMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.input_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.input_layernorm.weight").to(device=device, dtype=torch_dtype)
        )
        self.post_attention_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.post_attention_layernorm.weight").to(
                device=device, dtype=torch_dtype
            )
        )
        if layer_type == "linear_attention":
            self.token_mixer = TensorParallelQwen35LinearAttention(
                prefix=f"{prefix}.linear_attn",
                hf_config=hf_config,
                loader=loader,
                tp_rank=tp_rank,
                tp_size=tp_size,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_targets=lora_targets,
                torch_dtype=torch_dtype,
                device=device,
            )
        elif layer_type == "full_attention":
            self.token_mixer = TensorParallelQwen35FullAttention(
                prefix=f"{prefix}.self_attn",
                hf_config=hf_config,
                loader=loader,
                tp_rank=tp_rank,
                tp_size=tp_size,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_targets=lora_targets,
                torch_dtype=torch_dtype,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported qwen3_5_text layer type: {layer_type}")
        self.mlp = TensorParallelQwenMLP(
            prefix=f"{prefix}.mlp",
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            torch_dtype=torch_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        past_key_value: Any | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        mixer_output = self.token_mixer(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present = None
        if use_cache:
            mixer_output, present = mixer_output
        hidden_states = hidden_states + mixer_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if use_cache:
            return hidden_states, present
        return hidden_states


def _checkpoint_qwen35_decoder_layer_forward(
    layer: "TensorParallelQwen35DecoderLayer",
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    return layer(hidden_states, position_ids, attention_mask)


class TensorParallelQwen35FullAttention(nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(
            getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        )
        if self.num_heads % tp_size != 0:
            raise ValueError("Qwen3.5 full-attention query heads must be divisible by TP size")
        self.local_heads = self.num_heads // tp_size
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = float(
            (getattr(hf_config, "rope_parameters", {}) or {}).get("rope_theta", 1000000.0)
        )
        partial = float(
            (getattr(hf_config, "rope_parameters", {}) or {}).get("partial_rotary_factor", 1.0)
        )
        self.rotary_dim = int(self.head_dim * partial)
        self.local_q_head_start = tp_rank * self.local_heads
        self.local_q_head_stop = self.local_q_head_start + self.local_heads
        self.local_kv_indices = sorted(
            {
                head // self.num_key_value_groups
                for head in range(self.local_q_head_start, self.local_q_head_stop)
            }
        )

        local_q_heads = range(self.local_q_head_start, self.local_q_head_stop)
        q_bias = loader.get_optional(f"{prefix}.q_proj.bias")
        if q_bias is not None:
            q_bias = _select_head_rows(
                q_bias, head_indices=local_q_heads, head_width=self.head_dim * 2
            )
        self.q_proj = LoRALinear(
            _select_head_rows(
                loader.get(f"{prefix}.q_proj.weight"),
                head_indices=local_q_heads,
                head_width=self.head_dim * 2,
            ),
            q_bias,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.q_proj"),
            target_name="language.full_attn.q_proj",
            hf_module_path=f"{prefix}.q_proj",
            shard_kind="rows",
            row_indices=_head_row_indices(local_q_heads, self.head_dim * 2),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        k_bias = loader.get_optional(f"{prefix}.k_proj.bias")
        if k_bias is not None:
            k_bias = _select_head_rows(
                k_bias, head_indices=self.local_kv_indices, head_width=self.head_dim
            )
        self.k_proj = LoRALinear(
            _select_head_rows(
                loader.get(f"{prefix}.k_proj.weight"),
                head_indices=self.local_kv_indices,
                head_width=self.head_dim,
            ),
            k_bias,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.k_proj"),
            target_name="language.full_attn.k_proj",
            hf_module_path=f"{prefix}.k_proj",
            shard_kind="rows",
            row_indices=_head_row_indices(self.local_kv_indices, self.head_dim),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        v_bias = loader.get_optional(f"{prefix}.v_proj.bias")
        if v_bias is not None:
            v_bias = _select_head_rows(
                v_bias, head_indices=self.local_kv_indices, head_width=self.head_dim
            )
        self.v_proj = LoRALinear(
            _select_head_rows(
                loader.get(f"{prefix}.v_proj.weight"),
                head_indices=self.local_kv_indices,
                head_width=self.head_dim,
            ),
            v_bias,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.v_proj"),
            target_name="language.full_attn.v_proj",
            hf_module_path=f"{prefix}.v_proj",
            shard_kind="rows",
            row_indices=_head_row_indices(self.local_kv_indices, self.head_dim),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.q_norm = Qwen35RMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.k_norm = Qwen35RMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.q_norm.weight.data.copy_(
            loader.get(f"{prefix}.q_norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.k_norm.weight.data.copy_(
            loader.get(f"{prefix}.k_norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.o_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.o_proj.weight"),
            bias=loader.get_optional(f"{prefix}.o_proj.bias"),
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.o_proj"),
            target_name="language.full_attn.o_proj",
            hf_module_path=f"{prefix}.o_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_len, _ = hidden_states.shape
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch, query_len, self.local_heads, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch, query_len, self.local_heads * self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, query_len, len(self.local_kv_indices), self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, query_len, len(self.local_kv_indices), self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        past_len = int(past_key_value[0].shape[2]) if past_key_value is not None else 0
        key_len = past_len + query_len
        cos, sin = _rope_cache(
            key_len, self.rotary_dim, self.rope_theta, hidden_states.device, hidden_states.dtype
        )
        query, key = _apply_rope_partial(query, key, cos, sin, position_ids)
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        present = (key, value)
        local_kv_for_heads = [
            self.local_kv_indices.index(head // self.num_key_value_groups)
            for head in range(self.local_q_head_start, self.local_q_head_stop)
        ]
        key = key[:, local_kv_for_heads]
        value = value[:, local_kv_for_heads]
        attn_mask = _causal_attention_mask(attention_mask, query_len, key_len, hidden_states.device)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch, query_len, self.local_heads * self.head_dim)
        )
        output = self.o_proj(attn * torch.sigmoid(gate))
        output = _all_reduce_tp(output)
        if use_cache:
            return output, present
        return output


class TensorParallelQwen35LinearAttention(nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_v_heads = int(hf_config.linear_num_value_heads)
        self.num_k_heads = int(hf_config.linear_num_key_heads)
        self.head_k_dim = int(hf_config.linear_key_head_dim)
        self.head_v_dim = int(hf_config.linear_value_head_dim)
        if self.num_k_heads % tp_size != 0 or self.num_v_heads % tp_size != 0:
            raise ValueError(
                "Qwen3.5 linear-attention key/value heads must be divisible by TP size"
            )
        self.local_k_heads = self.num_k_heads // tp_size
        self.local_v_heads = self.num_v_heads // tp_size
        self.local_key_dim = self.local_k_heads * self.head_k_dim
        self.local_value_dim = self.local_v_heads * self.head_v_dim
        self.conv_kernel_size = int(hf_config.linear_conv_kernel_dim)
        self.local_k_start = tp_rank * self.local_key_dim
        self.local_k_stop = self.local_k_start + self.local_key_dim
        self.local_v_start = tp_rank * self.local_value_dim
        self.local_v_stop = self.local_v_start + self.local_value_dim
        key_dim = self.num_k_heads * self.head_k_dim

        qkv_weight = loader.get(f"{prefix}.in_proj_qkv.weight")
        self.q_proj = LoRALinear(
            qkv_weight[self.local_k_start : self.local_k_stop],
            None,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.q_proj"),
            target_name="language.linear_attn.q_proj",
            hf_module_path=f"{prefix}.in_proj_qkv",
            base_weight_name=f"{prefix}.in_proj_qkv.weight",
            shard_kind="rows",
            row_start=self.local_k_start,
            row_stop=self.local_k_stop,
            peft_exportable=False,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.k_proj = LoRALinear(
            qkv_weight[key_dim + self.local_k_start : key_dim + self.local_k_stop],
            None,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.k_proj"),
            target_name="language.linear_attn.k_proj",
            hf_module_path=f"{prefix}.in_proj_qkv",
            base_weight_name=f"{prefix}.in_proj_qkv.weight",
            shard_kind="rows",
            row_start=key_dim + self.local_k_start,
            row_stop=key_dim + self.local_k_stop,
            peft_exportable=False,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.v_proj = LoRALinear(
            qkv_weight[2 * key_dim + self.local_v_start : 2 * key_dim + self.local_v_stop],
            None,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.v_proj"),
            target_name="language.linear_attn.v_proj",
            hf_module_path=f"{prefix}.in_proj_qkv",
            base_weight_name=f"{prefix}.in_proj_qkv.weight",
            shard_kind="rows",
            row_start=2 * key_dim + self.local_v_start,
            row_stop=2 * key_dim + self.local_v_stop,
            peft_exportable=False,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        conv_indices = torch.tensor(
            [
                *range(self.local_k_start, self.local_k_stop),
                *range(key_dim + self.local_k_start, key_dim + self.local_k_stop),
                *range(2 * key_dim + self.local_v_start, 2 * key_dim + self.local_v_stop),
            ],
            dtype=torch.long,
        )
        self.conv1d_weight = nn.Parameter(
            loader.get(f"{prefix}.conv1d.weight")
            .index_select(0, conv_indices)
            .to(device=device, dtype=torch_dtype),
            requires_grad=False,
        )
        self.in_proj_z = LoRALinear.from_hf(
            loader.get(f"{prefix}.in_proj_z.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.in_proj_z"),
            target_name="language.linear_attn.in_proj_z",
            hf_module_path=f"{prefix}.in_proj_z",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.in_proj_b = LoRALinear.from_hf(
            loader.get(f"{prefix}.in_proj_b.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=False,
            target_name="language.linear_attn.in_proj_b",
            hf_module_path=f"{prefix}.in_proj_b",
            r=0,
            alpha=1,
            dropout=0.0,
            device=device,
            dtype=torch_dtype,
        )
        self.in_proj_a = LoRALinear.from_hf(
            loader.get(f"{prefix}.in_proj_a.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=False,
            target_name="language.linear_attn.in_proj_a",
            hf_module_path=f"{prefix}.in_proj_a",
            r=0,
            alpha=1,
            dropout=0.0,
            device=device,
            dtype=torch_dtype,
        )
        self.dt_bias = nn.Parameter(
            _shard_tensor(
                loader.get(f"{prefix}.dt_bias"), dim=0, tp_rank=tp_rank, tp_size=tp_size
            ).to(device=device, dtype=torch_dtype),
            requires_grad=False,
        )
        self.A_log = nn.Parameter(
            _shard_tensor(
                loader.get(f"{prefix}.A_log"), dim=0, tp_rank=tp_rank, tp_size=tp_size
            ).to(device=device, dtype=torch_dtype),
            requires_grad=False,
        )
        self.norm = Qwen35RMSNormGated(
            self.head_v_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.norm.weight.data.copy_(
            loader.get(f"{prefix}.norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.out_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.out_proj.weight"),
            bias=None,
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.out_proj"),
            target_name="language.linear_attn.out_proj",
            hf_module_path=f"{prefix}.out_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        del position_ids
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        mixed_qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
        conv_state = past_key_value[0] if past_key_value is not None else None
        recurrent_state = past_key_value[1] if past_key_value is not None else None
        if conv_state is not None and seq_len == 1:
            mixed_qkv = _torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d_weight.squeeze(1),
                activation="silu",
            )
            next_conv_state = conv_state
        else:
            if conv_state is not None:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if use_cache:
                next_conv_state = _left_pad_last_dim(mixed_qkv, self.conv_kernel_size)
            else:
                next_conv_state = None
            mixed_qkv = F.silu(
                F.conv1d(
                    mixed_qkv,
                    self.conv1d_weight,
                    bias=None,
                    padding=self.conv_kernel_size - 1,
                    groups=mixed_qkv.shape[1],
                )[:, :, : mixed_qkv.shape[-1]]
            )
            if conv_state is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.local_key_dim, self.local_key_dim, self.local_value_dim],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, self.local_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.local_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.local_v_heads, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(
            batch, seq_len, self.local_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        if self.local_v_heads // self.local_k_heads > 1:
            repeat = self.local_v_heads // self.local_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        if recurrent_state is not None and seq_len == 1:
            core_attn_out, next_recurrent_state = _torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, next_recurrent_state = _torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch, seq_len, self.local_value_dim)
        output = self.out_proj(core_attn_out)
        output = _all_reduce_tp(output)
        if use_cache:
            assert next_conv_state is not None
            assert next_recurrent_state is not None
            return output, (next_conv_state, next_recurrent_state)
        return output


class TensorParallelQwenDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        key_prefix: str,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        prefix = f"{key_prefix}.layers.{layer_idx}"
        self.input_layernorm = QwenRMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.post_attention_layernorm = QwenRMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.input_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.input_layernorm.weight").to(device=device, dtype=torch_dtype)
        )
        self.post_attention_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.post_attention_layernorm.weight").to(
                device=device, dtype=torch_dtype
            )
        )
        self.self_attn = TensorParallelQwenAttention(
            prefix=f"{prefix}.self_attn",
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.mlp = TensorParallelQwenMLP(
            prefix=f"{prefix}.mlp",
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            torch_dtype=torch_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_output = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present = None
        if use_cache:
            attn_output, present = attn_output
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if use_cache:
            assert present is not None
            return hidden_states, present
        return hidden_states


def _checkpoint_decoder_layer_forward(
    layer: "TensorParallelQwenDecoderLayer",
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    return layer(hidden_states, position_ids, attention_mask)


class TensorParallelQwenAttention(nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(
            getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        )
        if self.num_heads % tp_size != 0 or self.num_kv_heads % tp_size != 0:
            raise ValueError("Qwen attention heads and kv heads must be divisible by TP size")
        self.local_heads = self.num_heads // tp_size
        self.local_kv_heads = self.num_kv_heads // tp_size
        self.hidden_size = int(hf_config.hidden_size)
        self.rope_theta = float(getattr(hf_config, "rope_theta", 1000000.0))
        self.q_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.q_proj.weight"),
            bias=loader.get_optional(f"{prefix}.q_proj.bias"),
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.q_proj"),
            target_name="language.self_attn.q_proj",
            hf_module_path=f"{prefix}.q_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.k_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.k_proj.weight"),
            bias=loader.get_optional(f"{prefix}.k_proj.bias"),
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.k_proj"),
            target_name="language.self_attn.k_proj",
            hf_module_path=f"{prefix}.k_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.v_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.v_proj.weight"),
            bias=loader.get_optional(f"{prefix}.v_proj.bias"),
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.v_proj"),
            target_name="language.self_attn.v_proj",
            hf_module_path=f"{prefix}.v_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.q_norm = QwenRMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.k_norm = QwenRMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        q_norm_weight = loader.get_optional(f"{prefix}.q_norm.weight")
        k_norm_weight = loader.get_optional(f"{prefix}.k_norm.weight")
        if q_norm_weight is not None:
            self.q_norm.weight.data.copy_(q_norm_weight.to(device=device, dtype=torch_dtype))
        if k_norm_weight is not None:
            self.k_norm.weight.data.copy_(k_norm_weight.to(device=device, dtype=torch_dtype))
        self.o_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.o_proj.weight"),
            bias=loader.get_optional(f"{prefix}.o_proj.bias"),
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.o_proj"),
            target_name="language.self_attn.o_proj",
            hf_module_path=f"{prefix}.o_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_len, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, query_len, self.local_heads, self.head_dim)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(batch, query_len, self.local_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value = (
            self.v_proj(hidden_states)
            .view(batch, query_len, self.local_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        past_len = int(past_key_value[0].shape[2]) if past_key_value is not None else 0
        key_len = past_len + query_len
        cos, sin = _rope_cache(
            key_len, self.head_dim, self.rope_theta, hidden_states.device, hidden_states.dtype
        )
        query, key = _apply_rope(query, key, cos, sin, position_ids)
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        present = (key, value)
        if self.local_kv_heads != self.local_heads:
            repeat = self.local_heads // self.local_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attn_mask = _causal_attention_mask(attention_mask, query_len, key_len, hidden_states.device)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch, query_len, self.local_heads * self.head_dim)
        )
        output = self.o_proj(attn)
        output = _all_reduce_tp(output)
        if use_cache:
            return output, present
        return output


class TensorParallelQwenMLP(nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.gate_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.gate_proj.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.mlp.gate_proj"),
            target_name="language.mlp.gate_proj",
            hf_module_path=f"{prefix}.gate_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.up_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.up_proj.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.mlp.up_proj"),
            target_name="language.mlp.up_proj",
            hf_module_path=f"{prefix}.up_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.down_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.down_proj.weight"),
            bias=None,
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.mlp.down_proj"),
            target_name="language.mlp.down_proj",
            hf_module_path=f"{prefix}.down_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return _all_reduce_tp(output)


class LoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        lora_enabled: bool,
        r: int,
        alpha: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        target_name: str | None = None,
        hf_module_path: str | None = None,
        base_weight_name: str | None = None,
        shard_kind: str = "none",
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
        row_indices: Iterable[int] | None = None,
        peft_exportable: bool = True,
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        self.bias = (
            nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
            if bias is not None
            else None
        )
        self.lora_target_name = str(target_name or "unknown")
        self.hf_module_path = str(hf_module_path) if hf_module_path else None
        self.base_weight_name = (
            str(base_weight_name)
            if base_weight_name
            else (f"{self.hf_module_path}.weight" if self.hf_module_path else None)
        )
        self.lora_shard_kind = str(shard_kind or "none")
        self.lora_row_start = row_start
        self.lora_row_stop = row_stop
        self.lora_col_start = col_start
        self.lora_col_stop = col_stop
        self.lora_row_indices = tuple(int(idx) for idx in row_indices) if row_indices else None
        self.peft_exportable = bool(peft_exportable)
        self.lora_enabled = bool(lora_enabled and r > 0)
        self.lora_r = int(r)
        self.lora_alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)
        if self.lora_enabled:
            self.lora_a = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @classmethod
    def from_hf(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        shard: str,
        tp_rank: int,
        tp_size: int,
        lora_enabled: bool,
        r: int,
        alpha: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        target_name: str | None = None,
        hf_module_path: str | None = None,
        base_weight_name: str | None = None,
        peft_exportable: bool = True,
    ) -> "LoRALinear":
        dim = 0 if shard == "out" else 1
        row_start = row_stop = col_start = col_stop = None
        if shard == "out":
            row_start, row_stop = _shard_bounds(weight.shape[0], tp_rank=tp_rank, tp_size=tp_size)
        elif shard == "in":
            col_start, col_stop = _shard_bounds(weight.shape[1], tp_rank=tp_rank, tp_size=tp_size)
        weight = _shard_tensor(weight, dim=dim, tp_rank=tp_rank, tp_size=tp_size)
        if bias is not None and shard == "out":
            bias = _shard_tensor(bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
        elif shard == "in":
            bias = None
        return cls(
            weight,
            bias,
            lora_enabled=lora_enabled,
            target_name=target_name,
            hf_module_path=hf_module_path,
            base_weight_name=base_weight_name,
            shard_kind=shard,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
            peft_exportable=peft_exportable,
            r=r,
            alpha=alpha,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

    def lora_metadata(self, module_name: str) -> dict[str, Any]:
        if self.hf_module_path is None or self.base_weight_name is None:
            raise RuntimeError(f"Enabled LoRA module {module_name} is missing HF module metadata")
        return {
            "module_name": module_name,
            "lora_a_name": f"{module_name}.lora_a",
            "lora_b_name": f"{module_name}.lora_b",
            "target_name": self.lora_target_name,
            "hf_module_path": self.hf_module_path,
            "base_weight_name": self.base_weight_name,
            "shard_kind": self.lora_shard_kind,
            "row_start": self.lora_row_start,
            "row_stop": self.lora_row_stop,
            "col_start": self.lora_col_start,
            "col_stop": self.lora_col_stop,
            "row_indices": list(self.lora_row_indices)
            if self.lora_row_indices is not None
            else None,
            "peft_exportable": self.peft_exportable,
            "r": self.lora_r,
            "alpha": self.lora_alpha,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        if self.lora_enabled:
            lora = F.linear(F.linear(self.dropout(hidden_states), self.lora_a), self.lora_b)
            output = output + lora * self.scaling
        return output


class QwenRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return hidden_states * self.weight


class Qwen35RMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        output = output * (1.0 + self.weight.float())
        return output.to(hidden_states.dtype)


class Qwen35RMSNormGated(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


class SafetensorIndex:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.weight_map: dict[str, str] = {}
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(payload["weight_map"])
        else:
            files = sorted(model_path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
            for file in files:
                for key in load_file(str(file), device="cpu").keys():
                    self.weight_map[key] = file.name
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, name: str) -> torch.Tensor:
        value = self.get_optional(name)
        if value is None:
            raise KeyError(f"Tensor not found in HF checkpoint: {name}")
        return value

    def get_optional(self, name: str) -> torch.Tensor | None:
        filename = self.weight_map.get(name)
        if filename is None:
            return None
        if filename not in self._cache:
            self._cache[filename] = load_file(str(self.model_path / filename), device="cpu")
        return self._cache[filename][name]


class CollatedExperience:
    def __init__(self, items: Iterable[Experience], device: torch.device) -> None:
        items = list(items)
        self.sequences = pad_sequence([item.sequences for item in items], batch_first=True).to(
            device
        )
        self.old_log_probs = pad_sequence(
            [item.old_log_probs for item in items], batch_first=True
        ).to(device)
        self.advantages = pad_sequence(
            [item.advantages for item in items], batch_first=True, padding_value=0.0
        ).to(device)
        self.attention_mask = (
            pad_sequence([item.attention_mask for item in items], batch_first=True)
            .bool()
            .to(device)
        )
        self.action_mask = (
            pad_sequence([item.action_mask for item in items], batch_first=True).bool().to(device)
        )
        self.metadata = [item.metadata for item in items]


def collate_experiences(items: list[Experience], device: torch.device) -> CollatedExperience:
    return CollatedExperience(items, device)


def _resolve_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _dtype_size(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    return 4


def _shard_tensor(tensor: torch.Tensor, *, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tensor.shape[dim] % tp_size != 0:
        raise ValueError(
            f"Cannot shard tensor shape {tuple(tensor.shape)} on dim={dim} by TP={tp_size}"
        )
    return tensor.chunk(tp_size, dim=dim)[tp_rank].contiguous()


def _shard_bounds(size: int, *, tp_rank: int, tp_size: int) -> tuple[int, int]:
    if int(size) % int(tp_size) != 0:
        raise ValueError(f"Cannot shard size {size} by TP={tp_size}")
    chunk = int(size) // int(tp_size)
    start = int(tp_rank) * chunk
    return start, start + chunk


def _select_head_rows(
    tensor: torch.Tensor,
    *,
    head_indices: Iterable[int],
    head_width: int,
) -> torch.Tensor:
    indices = _head_row_indices(head_indices, head_width)
    return tensor.index_select(
        0, torch.tensor(indices, dtype=torch.long, device=tensor.device)
    ).contiguous()


def _head_row_indices(head_indices: Iterable[int], head_width: int) -> list[int]:
    indices: list[int] = []
    for head in head_indices:
        start = int(head) * int(head_width)
        indices.extend(range(start, start + int(head_width)))
    return indices


class _TensorParallelAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        del ctx
        output = tensor.contiguous()
        if dist.is_available() and dist.is_initialized() and _TENSOR_PARALLEL_SIZE > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=_TENSOR_PARALLEL_GROUP)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (grad_output,)


def _all_reduce_tp(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and _TENSOR_PARALLEL_SIZE > 1:
        return _TensorParallelAllReduce.apply(tensor)
    return tensor


def _selected_token_log_probs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    output_ids: torch.Tensor,
    *,
    vocab_chunk_size: int = 32768,
) -> torch.Tensor:
    selected = lm_head_weight.index_select(0, output_ids.reshape(-1)).view(
        *output_ids.shape,
        hidden_states.shape[-1],
    )
    selected_logits = (hidden_states * selected).sum(dim=-1)
    logsumexp: torch.Tensor | None = None
    for start in range(0, lm_head_weight.shape[0], vocab_chunk_size):
        chunk = lm_head_weight[start : start + vocab_chunk_size]
        logits = F.linear(hidden_states, chunk)
        chunk_lse = torch.logsumexp(logits, dim=-1)
        logsumexp = chunk_lse if logsumexp is None else torch.logaddexp(logsumexp, chunk_lse)
    assert logsumexp is not None
    return selected_logits - logsumexp


def _mean_present(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _rollout_timing_summary(
    tokenize_sec: float, chunk_timings: list[dict[str, float | int]]
) -> dict[str, Any]:
    summary = {
        "tokenize_sec": round(float(tokenize_sec), 6),
        "prefill_sec": round(
            sum(float(item.get("prefill_sec") or 0.0) for item in chunk_timings), 6
        ),
        "decode_sec": round(sum(float(item.get("decode_sec") or 0.0) for item in chunk_timings), 6),
        "sampling_sec": round(
            sum(float(item.get("sampling_sec") or 0.0) for item in chunk_timings), 6
        ),
        "stop_check_sec": round(
            sum(float(item.get("stop_check_sec") or 0.0) for item in chunk_timings), 6
        ),
        "decode_tokens": sum(int(item.get("decode_tokens") or 0) for item in chunk_timings),
    }
    for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS:
        value = sum(float(item.get(key) or 0.0) for item in chunk_timings)
        if value:
            summary[key] = round(value, 6)
    calls = sum(int(item.get("pipeline_forward_calls") or 0) for item in chunk_timings)
    if calls:
        summary["pipeline_forward_calls"] = calls
    return summary


def _scale_rollout_timings(
    chunk_timings: list[dict[str, float | int]],
    divisor: int,
) -> list[dict[str, float | int]]:
    divisor = max(1, int(divisor))
    scaled: list[dict[str, float | int]] = []
    for item in chunk_timings:
        payload: dict[str, float | int] = {
            "prefill_sec": float(item.get("prefill_sec") or 0.0) / divisor,
            "decode_sec": float(item.get("decode_sec") or 0.0) / divisor,
            "sampling_sec": float(item.get("sampling_sec") or 0.0) / divisor,
            "stop_check_sec": float(item.get("stop_check_sec") or 0.0) / divisor,
            "decode_tokens": int(item.get("decode_tokens") or 0),
        }
        for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS:
            if key in item:
                payload[key] = float(item.get(key) or 0.0) / divisor
        if "pipeline_forward_calls" in item:
            payload["pipeline_forward_calls"] = int(item.get("pipeline_forward_calls") or 0)
        scaled.append(payload)
    return scaled


_PIPELINE_STAGE_TIMING_FLOAT_KEYS = (
    "pipeline_recv_sec",
    "pipeline_send_sec",
    "pipeline_stage_compute_sec",
    "pipeline_norm_sec",
    "pipeline_lm_head_sec",
    "pipeline_loss_sec",
    "pipeline_sample_compute_sec",
    "pipeline_token_broadcast_sec",
    "pipeline_backward_autograd_sec",
    "pipeline_grad_recv_sec",
    "pipeline_grad_send_sec",
    "pipeline_grad_clip_sec",
    "pipeline_optimizer_step_sec",
)


def _new_pipeline_stage_timing() -> dict[str, float | int]:
    payload: dict[str, float | int] = {key: 0.0 for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS}
    payload["pipeline_forward_calls"] = 0
    return payload


def _add_pipeline_stage_timing(
    timing: dict[str, float | int] | None,
    key: str,
    started_at: float,
) -> None:
    if timing is None:
        return
    timing[key] = float(timing.get(key) or 0.0) + (time.monotonic() - started_at)


def _round_pipeline_stage_timing(timing: dict[str, float | int]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS:
        value = float(timing.get(key) or 0.0)
        if value:
            payload[key] = round(value, 6)
    calls = int(timing.get("pipeline_forward_calls") or 0)
    if calls:
        payload["pipeline_forward_calls"] = calls
    return payload


def _cuda_memory_snapshot(device: torch.device) -> dict[str, float | int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "max_allocated_mib": 0.0,
            "max_reserved_mib": 0.0,
        }
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    max_allocated = int(torch.cuda.max_memory_allocated(device))
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    mib = 1024.0 * 1024.0
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": max_allocated,
        "max_reserved_bytes": max_reserved,
        "allocated_mib": allocated / mib,
        "reserved_mib": reserved / mib,
        "max_allocated_mib": max_allocated / mib,
        "max_reserved_mib": max_reserved / mib,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _left_pad_token_rows(
    rows: Iterable[Iterable[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    values = [list(row) for row in rows]
    if not values:
        return torch.empty((0, 0), dtype=torch.long, device=device), []
    lengths = [len(row) for row in values]
    width = max(max(lengths), 1)
    output = torch.full((len(values), width), int(pad_token_id), dtype=torch.long, device=device)
    for idx, row in enumerate(values):
        if not row:
            continue
        output[idx, width - len(row) :] = torch.tensor(row, dtype=torch.long, device=device)
    return output, lengths


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _apply_rope_partial(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rot = (query_rot * cos) + (_rotate_half(query_rot) * sin)
    key_rot = (key_rot * cos) + (_rotate_half(key_rot) * sin)
    return torch.cat([query_rot, query_pass], dim=-1), torch.cat([key_rot, key_pass], dim=-1)


def _causal_attention_mask(
    attention_mask: torch.Tensor | None,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(key_len - query_len, key_len, device=device).unsqueeze(1)
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    causal = key_positions <= query_positions
    if attention_mask is None:
        return causal.view(1, 1, query_len, key_len)
    key_mask = attention_mask[:, None, None, :].bool()
    return causal.view(1, 1, query_len, key_len) & key_mask


def _apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        query_mask = attention_mask[:, -hidden_states.shape[1] :]
        return (hidden_states * query_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


def _left_pad_last_dim(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.shape[-1] >= width:
        return value[:, :, -width:].contiguous()
    return F.pad(value, (width - value.shape[-1], 0))


def _qwen35_cache_sequence_len(layer_cache: Any) -> int:
    if layer_cache is None:
        return 0
    if (
        len(layer_cache) >= 2
        and hasattr(layer_cache[0], "shape")
        and len(layer_cache[0].shape) == 4
    ):
        return int(layer_cache[0].shape[2])
    return 0


def _l2norm(value: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.vector_norm(value, dim=dim, keepdim=True), min=eps)


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    output = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias=None, padding=0, groups=hidden_size
    )
    if activation == "silu":
        output = F.silu(output)
    return output[:, :, -seq_len:].to(hidden_states.dtype)


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        item.transpose(1, 2).contiguous().to(torch.float32) for item in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        item.reshape(item.shape[0], item.shape[1], -1, chunk_size, item.shape[-1])
        for item in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for idx in range(1, chunk_size):
        row = attn[..., idx, :idx].clone()
        sub = attn[..., :idx, :idx].clone()
        attn[..., idx, :idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    for idx in range(0, total_sequence_length // chunk_size):
        query_i, key_i, value_i = query[:, :, idx], key[:, :, idx], value[:, :, idx]
        attn = query_i @ key_i.transpose(-1, -2) * decay_mask[:, :, idx]
        value_prime = (k_cumdecay[:, :, idx]) @ last_recurrent_state
        value_new = value_i - value_prime
        attn_inter = (query_i * g[:, :, idx, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, idx] = attn_inter + attn @ value_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, idx, -1, None, None].exp()
            + (key_i * (g[:, :, idx, -1, None] - g[:, :, idx]).exp()[..., None]).transpose(-1, -2)
            @ value_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        item.transpose(1, 2).contiguous().to(torch.float32) for item in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    query = query * (1 / (query.shape[-1] ** 0.5))
    core_attn_out = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        value_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    for idx in range(sequence_length):
        query_t = query[:, :, idx]
        key_t = key[:, :, idx]
        value_t = value[:, :, idx]
        g_t = g[:, :, idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, idx].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_memory = (last_recurrent_state * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - kv_memory) * beta_t
        last_recurrent_state = last_recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, idx] = (last_recurrent_state * query_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


def _next_token_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature > 0:
        logits = logits / max(float(temperature), 1e-6)
        return _sample_next_token(logits, top_p=top_p)
    return logits.argmax(dim=-1)


def _broadcast_and_pad_finished(
    next_token: torch.Tensor,
    finished: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(next_token, src=0)
    return torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)


def _sample_next_token(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
