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
from graspo.backends.native_tp.runtime import NativeGeneration
from graspo.core.buffer import Experience
from graspo.core.schema import GraspoConfig
from graspo.trainer.loss import GRASPOLoss


class QwenNativeTPAdapter:
    """Qwen causal LM adapter backed by self-owned PyTorch tensor parallel."""

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.tp_size = int(config.native_tp.tensor_model_parallel_size)
        self.tp_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tp_state: NativeTPState | None = None
        self.model: TensorParallelQwenForCausalLM | None = None
        self.tokenizer: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = GRASPOLoss(config.training.policy_ratio_clip_eps)
        self._train_batch_call_index = 0

    def setup(self) -> None:
        self._setup_distributed()
        from transformers import AutoTokenizer

        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model.model_path does not exist: {model_path}")

        hf_config = load_native_qwen_config(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = _resolve_dtype(self.config.model.torch_dtype)
        loader = SafetensorIndex(model_path)
        self.model = build_native_qwen_model(
            hf_config=hf_config,
            loader=loader,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            lora_targets=set(self.config.lora.target_modules or ("q_proj", "v_proj")),
            gradient_checkpointing=bool(self.config.model.gradient_checkpointing),
            torch_dtype=torch_dtype,
            device=self.device,
        )
        self.model.train(False)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self._emit_rank_memory_event(
            "setup_after",
            {
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
                "lora_target_modules": sorted(self.model.lora_targets),
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
                "group_batch_semantics": "one prompt, rollout_group_size completions per TP forward batch",
                "activation_checkpointing_enabled": bool(self.model.gradient_checkpointing),
                "lora_target_modules": sorted(self.model.lora_targets),
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
        self._require_ready()
        assert self.model is not None
        assert self.tokenizer is not None
        self.model.eval()
        tokenize_started_at = time.monotonic()
        prompt_text = self._format_prompt(prompt, chat_template_kwargs)
        encoded = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
        )
        tokenize_sec = time.monotonic() - tokenize_started_at
        rollout_started_at = time.monotonic()
        prompt_input_ids = encoded["input_ids"].to(self.device)
        prompt_len = int(prompt_input_ids.shape[1])
        eos_token_id = int(self.tokenizer.eos_token_id)
        pad_token_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id)
        use_kv_cache = bool(self.config.native_tp.use_kv_cache_for_rollout)
        generation_micro_batch_size = self._shared_generation_micro_batch_size(
            prompt_len=prompt_len,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )
        sequence_chunks: list[torch.Tensor] = []
        chunk_timings: list[dict[str, float | int]] = []

        with torch.no_grad():
            for start in range(0, rollout_group_size, generation_micro_batch_size):
                current_batch_size = min(generation_micro_batch_size, rollout_group_size - start)
                sequences = prompt_input_ids.repeat(current_batch_size, 1)
                finished = torch.zeros(current_batch_size, dtype=torch.bool, device=self.device)
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

        sequences = pad_sequence(
            [row for chunk in sequence_chunks for row in chunk],
            batch_first=True,
            padding_value=pad_token_id,
        )

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
        self._emit_rank_memory_event(
            "rollout_after",
            {
                "rollout_group_size": rollout_group_size,
                "prompt_len": prompt_len,
                "sequence_len": int(sequences.shape[1]),
                "generated_tokens_max": max(int(sequences.shape[1] - prompt_len), 0),
                "rollout_use_kv_cache": use_kv_cache,
                "rollout_generation_micro_batch_size": generation_micro_batch_size,
                "rollout_generation_split_count": len(sequence_chunks),
                **_rollout_timing_summary(tokenize_sec, chunk_timings),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
            },
        )
        empty_cache_after_split = (
            self.device.type == "cuda"
            and len(sequence_chunks) > 1
            and bool(self.config.native_tp.empty_cache_after_rollout_split)
        )
        if empty_cache_after_split:
            torch.cuda.empty_cache()
            self._emit_rank_memory_event(
                "rollout_after_empty_cache",
                {
                    "rollout_group_size": rollout_group_size,
                    "prompt_len": prompt_len,
                    "sequence_len": int(sequences.shape[1]),
                    "generated_tokens_max": max(int(sequences.shape[1] - prompt_len), 0),
                    "rollout_use_kv_cache": use_kv_cache,
                    "rollout_generation_micro_batch_size": generation_micro_batch_size,
                    "rollout_generation_split_count": len(sequence_chunks),
                    "rollout_empty_cache_after_split": True,
                    **_rollout_timing_summary(tokenize_sec, chunk_timings),
                    "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
                },
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
                "rollout_use_kv_cache": use_kv_cache,
                "rollout_generation_micro_batch_size": generation_micro_batch_size,
                "rollout_generation_split_count": len(sequence_chunks),
                "rollout_empty_cache_after_split": empty_cache_after_split,
                **_rollout_timing_summary(tokenize_sec, chunk_timings),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
                "prefill_len": prompt_len,
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
        self._sync_timing()
        for _ in range(max_new_tokens):
            attention_mask = sequences.ne(pad_token_id)
            logits = self.model(sequences, attention_mask=attention_mask).float()[:, -1, :]
            next_token = _next_token_from_logits(logits, temperature=temperature, top_p=top_p)
            next_token = _broadcast_and_pad_finished(next_token, finished, pad_token_id)
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            decode_tokens += 1
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()):
                break
        self._sync_timing()
        return sequences, {
            "prefill_sec": 0.0,
            "decode_sec": time.monotonic() - decode_started_at,
            "decode_tokens": decode_tokens,
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
        logits, past_key_values = self.model(sequences, attention_mask=attention_mask, use_cache=True)
        self._sync_timing()
        prefill_sec = time.monotonic() - prefill_started_at
        decode_started_at = time.monotonic()
        decode_tokens = 0
        for _ in range(max_new_tokens):
            next_token = _next_token_from_logits(logits.float()[:, -1, :], temperature=temperature, top_p=top_p)
            next_token = _broadcast_and_pad_finished(next_token, finished, pad_token_id)
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            decode_tokens += 1
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()):
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
        }

    def sequence_log_probs(self, sequences: Any, attention_mask: Any) -> torch.Tensor:
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
                batch = collate_experiences([experiences[idx] for idx in batch_indices], self.device)
                self.optimizer.zero_grad(set_to_none=True)
                self._sync_timing()
                forward_started_at = time.monotonic()
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

    def _shared_training_indices(self, experience_count: int, *, optimize_round: int) -> list[int]:
        indices = list(range(experience_count))
        seed = int(self.config.training.seed) + (self._train_batch_call_index * 1_000_003) + int(optimize_round)
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

    def _kv_cache_batch_fits_budget(self, *, batch_size: int, prompt_len: int, max_new_tokens: int) -> bool:
        assert self.model is not None
        if self.device.type != "cuda":
            return True
        fraction = float(self.config.native_tp.rollout_kv_cache_max_reserved_fraction)
        fraction = min(max(fraction, 0.05), 1.0)
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        reserved = int(torch.cuda.memory_reserved(self.device))
        budget = max(0, int(total * fraction) - reserved)
        return self.model.estimate_kv_cache_bytes(
            batch_size=batch_size,
            sequence_len=prompt_len + max_new_tokens,
        ) <= budget

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        trainer_state: dict[str, Any] | None = None,
    ) -> None:
        self._require_ready()
        assert self.model is not None
        assert self.optimizer is not None
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "adapter": "qwen_native_tp",
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "lora_state_dict": self.model.lora_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "adapter_state": {
                "train_batch_call_index": self._train_batch_call_index,
            },
            "trainer_state": trainer_state,
            "config": asdict(self.config),
        }
        torch.save(payload, output / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}.pt")
        self._emit_rank_memory_event("checkpoint_after", {"checkpoint_dir": str(output)})
        if self.rank == 0:
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "graspo-native-tp-lora",
                        "tp_size": self.tp_size,
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
        assert self.optimizer is not None
        checkpoint_dir = Path(path)
        rank_path = checkpoint_dir / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}.pt"
        if not rank_path.exists():
            candidates = sorted(checkpoint_dir.glob(f"rank_*_tp_{self.tp_rank:02d}.pt"))
            if len(candidates) == 1:
                rank_path = candidates[0]
            else:
                raise FileNotFoundError(f"Missing native TP checkpoint shard for rank={self.rank} tp_rank={self.tp_rank}: {rank_path}")
        try:
            payload = torch.load(rank_path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(rank_path, map_location=self.device)
        if int(payload.get("tp_size", self.tp_size)) != self.tp_size:
            raise ValueError(
                f"Checkpoint TP size {payload.get('tp_size')} does not match runtime TP size {self.tp_size}"
            )
        missing, unexpected = self.model.load_state_dict(payload["lora_state_dict"], strict=False)
        unexpected_lora = [name for name in unexpected if "lora_" in name]
        if unexpected_lora:
            raise RuntimeError(f"Unexpected LoRA tensors in checkpoint: {unexpected_lora}")
        missing_lora = [name for name in missing if "lora_" in name]
        if missing_lora:
            raise RuntimeError(f"Missing LoRA tensors while loading checkpoint: {missing_lora}")
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
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
        state = NativeTPState.initialize(self.tp_size)
        self.tp_state = state
        self.rank = state.rank
        self.local_rank = state.local_rank
        self.world_size = state.world_size
        self.tp_rank = state.tp_rank
        self.device = state.device

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

    def _require_ready(self) -> None:
        if self.model is None or self.tokenizer is None or self.optimizer is None:
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
            "global_optimizer_steps_sum": sum(int(item.get("optimizer_steps") or 0) for item in ranks),
            "global_nonzero_grad_count_sum": sum(int(item.get("nonzero_grad_count") or 0) for item in ranks),
            "global_loss_mean": _mean_present(item.get("loss_mean") for item in ranks),
            "global_grad_norm_mean": _mean_present(item.get("grad_norm_mean") for item in ranks),
            "global_lora_norm_delta_mean": _mean_present(item.get("lora_norm_delta") for item in ranks),
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


def load_native_qwen_config(model_path: Path) -> NativeQwenConfig:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "")
    if model_type == "qwen3":
        return NativeQwenConfig(config, family="qwen3", key_prefix="model")
    text_config = dict(config.get("text_config") or {})
    if model_type == "qwen3_5" and text_config.get("model_type") == "qwen3_5_text":
        return NativeQwenConfig(text_config, family="qwen3_5_text", key_prefix="model.language_model")
    raise ValueError(
        "native-tp supports text-only qwen3 and qwen3_5_text models; "
        f"got model_type={model_type!r}"
    )


def build_native_qwen_model(
    *,
    hf_config: NativeQwenConfig,
    loader: "SafetensorIndex",
    tp_rank: int,
    tp_size: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: set[str],
    gradient_checkpointing: bool,
    torch_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    if hf_config.family == "qwen3":
        return TensorParallelQwenForCausalLM(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device=device,
        )
    if hf_config.family == "qwen3_5_text":
        return TensorParallelQwen35TextForCausalLM(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported native Qwen family: {hf_config.family}")


class TensorParallelQwenForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        hf_config: Any,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
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
        self.device_ref = device
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.lora_targets = set(lora_targets)
        self.key_prefix = str(getattr(hf_config, "key_prefix", "model"))
        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size, device=device, dtype=torch_dtype)
        self.embed_tokens.weight.data.copy_(
            loader.get(f"{self.key_prefix}.embed_tokens.weight").to(device=device, dtype=torch_dtype)
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
        self.norm = QwenRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype)
        self.norm.weight.data.copy_(loader.get(f"{self.key_prefix}.norm.weight").to(device=device, dtype=torch_dtype))
        self.lm_head = nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False, device=device, dtype=torch_dtype)
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

    def sequence_log_probs(self, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
        head_dim = int(getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads))
        return int(batch_size) * int(self.config.num_hidden_layers) * 2 * local_kv_heads * head_dim * int(sequence_len) * dtype_size

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().cpu() for name, param in self.named_parameters() if "lora_" in name}

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


class TensorParallelQwen35TextForCausalLM(nn.Module):
    """Qwen3.5 text-only model gate.

    Qwen3.5/Qwen3.6 text checkpoints mix full-attention layers with a separate
    linear-attention kernel. The model registry detects these checkpoints and
    keeps vision weights out of scope, but the linear-attention kernel must be
    implemented before training can start. Failing here is intentional: it
    prevents silently training with an incorrect approximation.
    """

    def __init__(
        self,
        *,
        hf_config: NativeQwenConfig,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        gradient_checkpointing: bool,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        del loader, tp_rank, tp_size, lora_r, lora_alpha, lora_dropout
        del lora_targets, gradient_checkpointing, torch_dtype, device
        layer_types = list(getattr(hf_config, "layer_types", []) or [])
        if any(layer_type == "linear_attention" for layer_type in layer_types):
            raise NotImplementedError(
                "native-tp detected qwen3_5_text with linear_attention layers. "
                "The checkpoint is text-only compatible, but the qwen3_5 linear-attention "
                "kernel is not implemented yet; do not fall back to an approximate layer."
            )
        raise NotImplementedError("qwen3_5_text without linear_attention is not implemented")


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
        self.input_layernorm = QwenRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype)
        self.post_attention_layernorm = QwenRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype)
        self.input_layernorm.weight.data.copy_(loader.get(f"{prefix}.input_layernorm.weight").to(device=device, dtype=torch_dtype))
        self.post_attention_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.post_attention_layernorm.weight").to(device=device, dtype=torch_dtype)
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
        self.head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
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
            lora_enabled="q_proj" in lora_targets,
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
            lora_enabled="k_proj" in lora_targets,
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
            lora_enabled="v_proj" in lora_targets,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.q_norm = QwenRMSNorm(self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype)
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
            lora_enabled="o_proj" in lora_targets,
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
        query = self.q_norm(self.q_proj(hidden_states).view(batch, query_len, self.local_heads, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, query_len, self.local_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, query_len, self.local_kv_heads, self.head_dim).transpose(1, 2)
        past_len = int(past_key_value[0].shape[2]) if past_key_value is not None else 0
        key_len = past_len + query_len
        cos, sin = _rope_cache(key_len, self.head_dim, self.rope_theta, hidden_states.device, hidden_states.dtype)
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
        attn = attn.transpose(1, 2).contiguous().view(batch, query_len, self.local_heads * self.head_dim)
        output = self.o_proj(attn)
        _all_reduce_tp(output)
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
            lora_enabled="gate_proj" in lora_targets,
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
            lora_enabled="up_proj" in lora_targets,
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
            lora_enabled="down_proj" in lora_targets,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        _all_reduce_tp(output)
        return output


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
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        self.bias = nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False) if bias is not None else None
        self.lora_enabled = bool(lora_enabled and r > 0)
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
    ) -> "LoRALinear":
        dim = 0 if shard == "out" else 1
        weight = _shard_tensor(weight, dim=dim, tp_rank=tp_rank, tp_size=tp_size)
        if bias is not None and shard == "out":
            bias = _shard_tensor(bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
        elif shard == "in":
            bias = None
        return cls(
            weight,
            bias,
            lora_enabled=lora_enabled,
            r=r,
            alpha=alpha,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        if self.lora_enabled:
            lora = F.linear(F.linear(self.dropout(hidden_states), self.lora_a), self.lora_b)
            output = output + lora * self.scaling
        return output


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return hidden_states * self.weight


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
        self.sequences = pad_sequence([item.sequences for item in items], batch_first=True).to(device)
        self.old_log_probs = pad_sequence([item.old_log_probs for item in items], batch_first=True).to(device)
        self.advantages = pad_sequence([item.advantages for item in items], batch_first=True, padding_value=0.0).to(device)
        self.attention_mask = pad_sequence([item.attention_mask for item in items], batch_first=True).bool().to(device)
        self.action_mask = pad_sequence([item.action_mask for item in items], batch_first=True).bool().to(device)


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
        raise ValueError(f"Cannot shard tensor shape {tuple(tensor.shape)} on dim={dim} by TP={tp_size}")
    return tensor.chunk(tp_size, dim=dim)[tp_rank].contiguous()


def _all_reduce_tp(tensor: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


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


def _rollout_timing_summary(tokenize_sec: float, chunk_timings: list[dict[str, float | int]]) -> dict[str, Any]:
    return {
        "tokenize_sec": round(float(tokenize_sec), 6),
        "prefill_sec": round(sum(float(item.get("prefill_sec") or 0.0) for item in chunk_timings), 6),
        "decode_sec": round(sum(float(item.get("decode_sec") or 0.0) for item in chunk_timings), 6),
        "decode_tokens": sum(int(item.get("decode_tokens") or 0) for item in chunk_timings),
    }


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
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
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


def _next_token_from_logits(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
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
