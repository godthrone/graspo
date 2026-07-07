"""Layer 1 — TransformerAdapter: common adapter logic for all decoder-only transformers.

Extracted from the original Qwen adapter.  Every model family subclasses this and
only implements the model-specific parts (``_load_model``, ``_build_ops``,
``generate_groups``, ``generate_sample_groups``, ``train_batch``,
``sequence_log_probs``, ``parse_completion``).
"""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from graspo.backends.graspoflow.base_adapter import BaseGraspoFlowAdapter
from graspo.backends.graspoflow.parallel_state import GraspoFlowState, destroy_parallel_state
from graspo.backends.graspoflow.placement import (
    NativePlacementPlan,
    placement_summary,
)
from graspo.backends.graspoflow.runtime import NativeGeneration
from graspo.backends.graspoflow.tensor_utils import (
    _cuda_memory_snapshot,
    _jsonable,
    _mean_present,
    _rollout_timing_summary,
    _scale_rollout_timings,
    _set_tensor_parallel_group,
)


class TransformerAdapter(BaseGraspoFlowAdapter):
    """Common adapter for all decoder-only transformer models.

    Provides:
    - Distributed initialization (``_setup_distributed``)
    - Tokenizer / Processor loading (``_load_tokenizer``)
    - Chat template formatting (``_format_messages``)
    - Checkpoint save/restore format (``save_checkpoint``, ``load_checkpoint``)
    - Training-loop helpers (``_shared_training_indices``, ``_aggregate_rank_metrics``)
    - Memory events (``_emit_rank_memory_event``)
    - Generation helpers (``_generation_from_sequences``, chunk-size helpers)

    Subclasses must implement:
    - ``_load_model()``
    - ``_build_ops()``
    - ``generate_groups()``
    - ``generate_sample_groups()``
    - ``train_batch()``
    - ``sequence_log_probs()``
    - ``parse_completion()``
    """

    # ── Initialization ──────────────────────────────────────────────────────

    def __init__(self, config: Any) -> None:
        self.config = config
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.tp_size = int(config.graspoflow.tp_size)
        self.tp_rank = 0
        self.pp_size = int(config.graspoflow.pp_size)
        self.pp_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tp_state: GraspoFlowState | None = None
        self.model: Any = None
        self.placement: NativePlacementPlan | None = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self._train_batch_call_index = 0

    # ── Setup (template method) ─────────────────────────────────────────────

    def setup(self) -> None:
        self._setup_distributed()
        self._patch_transformers_float8_import_compat()
        from transformers import AutoProcessor, AutoTokenizer

        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model.model_path does not exist: {model_path}")

        hf_config = self._load_native_qwen_config(model_path)
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

        self._load_model(hf_config, model_path)
        self._build_ops()
        self._build_optimizer()
        self._emit_setup_event()

    @abstractmethod
    def _load_model(self, hf_config: Any, model_path: Path) -> None:
        """Load the model.  Subclass implements."""

    @abstractmethod
    def _build_ops(self) -> None:
        """Build pipeline operators.  Subclass implements."""

    def _build_optimizer(self) -> None:
        from graspo.core.graspo_loss import GRASPOLoss

        self.loss_fn = GRASPOLoss(self.config.training.policy_ratio_clip_eps)
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
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> Any:
        """根据 ``lr_scheduler`` 配置构建学习率调度器，默认返回 None（恒定 LR）。"""
        import math

        sched_cfg = self.config.training.lr_scheduler
        if sched_cfg.type == "constant":
            return None

        base_lr = float(self.config.training.learning_rate)
        warmup_steps = max(0, int(sched_cfg.warmup_steps))
        min_lr = base_lr * float(sched_cfg.min_lr_ratio)

        max_steps = int(self.config.training.max_steps)
        if max_steps <= 0:
            raise ValueError(
                "lr_scheduler.type 非 constant 时，training.max_steps 必须 > 0，"
                "否则无法推算总 optimizer step 数"
            )
        total_steps = (
            max_steps
            * int(self.config.training.optimize_iterations_per_step)
            * int(self.config.training.rollout_group_size)
        )

        if total_steps <= warmup_steps:
            raise ValueError(
                f"lr_scheduler: 自动推算的总步数 ({total_steps})"
                f" 必须大于 warmup_steps ({warmup_steps})"
            )

        decay_steps = total_steps - warmup_steps

        if sched_cfg.type == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = min(float(step - warmup_steps) / max(1, decay_steps), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (min_lr + (base_lr - min_lr) * cosine) / base_lr

        elif sched_cfg.type == "linear":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = min(float(step - warmup_steps) / max(1, decay_steps), 1.0)
                return (min_lr + (base_lr - min_lr) * (1.0 - progress)) / base_lr

        else:
            raise ValueError(
                f"不支持的 lr_scheduler.type: {sched_cfg.type}，可选: constant, cosine, linear"
            )

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _emit_setup_event(self) -> None:
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self._emit_rank_memory_event(
            "setup_after",
            {
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "activation_checkpointing_enabled": bool(
                    getattr(self.model, "gradient_checkpointing", False)
                ),
                "lora_target_modules": sorted(self.model.lora_targets),
                "lora_target_signature": self.model.lora_target_signature(),
                "rollout_kv_cache_supported": bool(getattr(self.model, "supports_kv_cache", True)),
                "placement": placement_summary(self.placement),
                "forward_batch_size": self.config.graspoflow.forward_batch_size,
                "empty_cache_after_rollout_split": (
                    self.config.graspoflow.empty_cache_after_rollout_split
                ),
                "synchronize_cuda_timing": self.config.graspoflow.synchronize_cuda_timing,
            },
        )
        self._print_rank0(
            {
                "event": "adapter_ready",
                "rank": self.rank,
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "trainable_parameters_local": sum(param.numel() for param in trainable),
                "group_batch_semantics": (
                    "rollout_prompt_queue_batch_size prompts, each with rollout_group_size "
                    "completions per TP forward batch when budget permits"
                ),
                "activation_checkpointing_enabled": bool(
                    getattr(self.model, "gradient_checkpointing", False)
                ),
                "lora_target_modules": sorted(self.model.lora_targets),
                "lora_target_signature": self.model.lora_target_signature(),
                "rollout_kv_cache_supported": bool(getattr(self.model, "supports_kv_cache", True)),
                "placement": placement_summary(self.placement),
            }
        )

    # ── Distributed setup ───────────────────────────────────────────────────

    def _setup_distributed(self) -> None:
        state = GraspoFlowState.initialize(self.tp_size, self.pp_size)
        self.tp_state = state
        self.rank = state.rank
        self.local_rank = state.local_rank
        self.world_size = state.world_size
        self.tp_rank = state.tp_rank
        self.pp_rank = state.pp_rank
        self.device = state.device
        _set_tensor_parallel_group(state.tp_group, state.tp_size)

    def _load_native_qwen_config(self, model_path: Path) -> Any:
        from graspo.backends.graspoflow.models.qwen3.model import load_native_qwen_config

        return load_native_qwen_config(model_path)

    def _patch_transformers_float8_import_compat(self) -> None:
        # 运行时探测 PyTorch 版本以兼容 float8 类型（非接口探测，是外部库版本检测）
        if not hasattr(torch, "float8_e8m0fnu"):
            torch.float8_e8m0fnu = torch.uint8  # type: ignore[attr-defined]

    # ── Chat template ───────────────────────────────────────────────────────

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

    # ── Checkpoint ──────────────────────────────────────────────────────────

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
            "adapter": "graspoflow",
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "placement": placement_summary(self.placement) if self.placement is not None else None,
            "lora_target_signature": self.model.lora_target_signature(),
            "lora_tensor_metadata": self.model.lora_tensor_metadata(),
            "lora_state_dict": self.model.lora_state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
            "adapter_state": {
                "train_batch_call_index": self._train_batch_call_index,
            },
            "trainer_state": trainer_state,
            "config": self.config.model_dump(),
        }
        torch.save(
            payload,
            output / f"rank_{self.rank:05d}_tp_{self.tp_rank:02d}_pp_{self.pp_rank:02d}.pt",
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self._emit_rank_memory_event("checkpoint_after", {"checkpoint_dir": str(output)})
        if self.rank == 0:
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "graspoflow-lora",
                        "tp_size": self.tp_size,
                        "pp_size": self.pp_size,
                        "placement": (
                            placement_summary(self.placement)
                            if self.placement is not None
                            else None
                        ),
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
                f"Checkpoint TP size {payload.get('tp_size')} does not match runtime "
                f"TP size {self.tp_size}"
            )
        if int(payload.get("pp_size", self.pp_size)) != self.pp_size:
            raise ValueError(
                f"Checkpoint PP size {payload.get('pp_size')} does not match runtime "
                f"PP size {self.pp_size}"
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
        scheduler_state = payload.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
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

    # ── Training helpers ────────────────────────────────────────────────────

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        batch: Any,
        lm_head: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Hook: 从 hidden states 计算 loss。默认 PPO loss，SFT 子类可覆盖。

        Args:
            hidden_states: 模型最后一层的 hidden states (batch, seq_len, hidden_size)
            batch: RL 时为 CollatedExperience，SFT 时为 dict with ``labels``
            lm_head: 语言模型头（TP-only 场景为 None 时从 self.model.lm_head 获取）

        Returns:
            标量 loss
        """
        from graspo.backends.graspoflow.tensor_utils import (
            _selected_token_log_probs_from_hidden,
        )
        from graspo.core.buffer import Experience

        # 判断 batch 类型：RL (CollatedExperience) 或 SFT (dict with labels)
        if isinstance(batch, Experience) or hasattr(batch, "old_log_probs"):
            # PPO loss（默认行为）
            norm = self.model.norm if hasattr(self.model, "norm") else None
            if lm_head is None:
                lm_head = self.model.lm_head if hasattr(self.model, "lm_head") else None
            if norm is None or lm_head is None:
                raise RuntimeError("compute_loss requires model.norm and model.lm_head")
            normalized = norm(hidden_states)
            log_probs = _selected_token_log_probs_from_hidden(
                normalized[:, :-1].float(),
                lm_head.weight.float(),
                batch.sequences[:, 1:],
            )
            return self.loss_fn(log_probs, batch.old_log_probs, batch.advantages, batch.action_mask)
        # SFT path
        if isinstance(batch, dict) and "labels" in batch:
            from graspo.core.sft_loss import sft_cross_entropy_loss

            norm = self.model.norm if hasattr(self.model, "norm") else None
            if lm_head is None:
                lm_head = self.model.lm_head if hasattr(self.model, "lm_head") else None
            if norm is None or lm_head is None:
                raise RuntimeError("compute_loss requires model.norm and model.lm_head")
            normalized = norm(hidden_states)
            logits = torch.nn.functional.linear(normalized.float(), lm_head.weight.float())
            return sft_cross_entropy_loss(logits, batch["labels"].to(hidden_states.device))
        raise TypeError(
            f"compute_loss: unsupported batch type {type(batch).__name__}, "
            "expected CollatedExperience or dict with 'labels'"
        )

    def _shared_training_indices(self, experience_count: int, *, optimize_round: int) -> list[int]:
        import random

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
        return max(
            1,
            min(
                int(self.config.graspoflow.forward_batch_size),
                int(rollout_group_size),
            ),
        )

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

    def _aggregate_rank_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        local = {"rank": self.rank, "tp_rank": self.tp_rank, **metrics}
        if not (dist.is_available() and dist.is_initialized()):
            return {
                **metrics,
                "rank": self.rank,
                "tp_rank": self.tp_rank,
                "rank_metrics": [local],
            }
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

    # ── Generation helpers ──────────────────────────────────────────────────

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
                "adapter": "graspoflow",
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
                    tokenize_sec,
                    _scale_rollout_timings(chunk_timings, timing_divisor),
                ),
                "rollout_elapsed_sec": round(time.monotonic() - rollout_started_at, 6),
                "prefill_len": prompt_len,
                "prompt_lens": prompt_lens,
                "generated_tokens_max": max(int(sequences.shape[1] - prompt_len), 0),
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
            },
        )

    def _pipeline_stage_timing(self) -> dict[str, float | int]:
        from graspo.backends.graspoflow.tensor_utils import _new_pipeline_stage_timing

        return _new_pipeline_stage_timing()

    def _add_pipeline_stage_timing(
        self, timing: dict[str, float | int], key: str, started_at: float
    ) -> None:
        from graspo.backends.graspoflow.tensor_utils import _add_pipeline_stage_timing

        _add_pipeline_stage_timing(timing, key, started_at)

    def _round_pipeline_stage_timing(
        self, timing: dict[str, float | int]
    ) -> dict[str, float | int]:
        from graspo.backends.graspoflow.tensor_utils import _round_pipeline_stage_timing

        return _round_pipeline_stage_timing(timing)

    # ── Utility ─────────────────────────────────────────────────────────────

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

    def is_primary(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        destroy_parallel_state()

    def _require_ready(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__} is not set up")

    def _print_rank0(self, payload: dict[str, Any]) -> None:
        if self.rank == 0:
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _sync_timing(self) -> None:
        if (
            bool(self.config.graspoflow.synchronize_cuda_timing)
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def _is_pipeline_parallel(self) -> bool:
        return bool(self.placement is not None and self.placement.is_pipeline)

    def _current_lr(self) -> float:
        """返回当前学习率（优先从 scheduler 读取，否则从 optimizer 读取）。"""
        if self.scheduler is not None:
            return float(self.scheduler.get_last_lr()[0])
        if self.optimizer is not None:
            return float(self.optimizer.param_groups[0]["lr"])
        return float(self.config.training.learning_rate)

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
        path = output_dir / "logs" / f"rank_metrics.rank_{self.rank:05d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")

    def _encode_multimodal_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from graspo.backends.graspoflow.multimodal import (
            _messages_from_multimodal_row,
            _processor_chat_messages,
            _tools_for_chat_template,
            _tools_from_multimodal_row,
        )

        if self.processor is None:
            raise RuntimeError(
                "This model did not expose an AutoProcessor; image/video samples cannot be encoded"
            )
        messages = [_processor_chat_messages(_messages_from_multimodal_row(row)) for row in rows]
        tool_batches = [_tools_from_multimodal_row(row) for row in rows]
        # 外部 HF processor 对象：部分 processor 直接暴露 apply_chat_template，
        # 部分通过 tokenizer 间接暴露，此处检测 processor 的能力边界
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
        self, metadata: Any | None, *, batch_size: int
    ) -> dict[str, torch.Tensor] | None:
        from graspo.backends.graspoflow.multimodal import _multimodal_rows_from_metadata

        rows = _multimodal_rows_from_metadata(metadata, expected_rows=batch_size)
        if not rows:
            return None
        encoded = self._encode_multimodal_rows(
            rows,
            add_generation_prompt=True,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        return self._multimodal_inputs_to_device(encoded)
