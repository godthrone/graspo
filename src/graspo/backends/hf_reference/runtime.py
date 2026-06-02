from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from graspo.backends.native_tp.runtime import NativeGeneration
from graspo.core.buffer import Experience
from graspo.core.schema import GraspoConfig
from graspo.trainer.checkpoint import save_lora_adapter
from graspo.trainer.generation import ensure_tokenizer_ready, generate_group
from graspo.trainer.lora import build_peft_config
from graspo.trainer.loss import GRASPOLoss, sequences_log_probs


class HFReferenceRuntime:
    """Single-process Hugging Face runtime for GRASPO algorithm parity tests."""

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = GRASPOLoss(config.training.policy_ratio_clip_eps)

    def validate(self) -> None:
        if not self.config.model.model_path or self.config.model.model_path.startswith("${"):
            raise ValueError("Set model.model_path in config or MODEL_PATH environment variable")

    def setup(self) -> None:
        try:
            from peft import get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "hf-reference requires the optional PEFT dependency. "
                "Install it with `uv sync --extra reference` or `pip install -e .[reference]`. "
                "The production native-tp backend does not use PEFT at runtime."
            ) from exc
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self.validate()
        set_seed(self.config.training.seed)
        model_path = self.config.model.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        ensure_tokenizer_ready(self.tokenizer)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": _resolve_dtype(self.config.model.torch_dtype),
        }
        if self.config.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.model.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(self.device)
        if self.config.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.config.use_cache = False
        self.model = get_peft_model(self.model, build_peft_config(self.config, self.model)).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def generate_group(self, **kwargs: Any) -> NativeGeneration:
        self._require_ready()
        assert self.model is not None
        assert self.tokenizer is not None
        rollout_group_size = kwargs.pop("rollout_group_size")
        sequences, attention_mask, action_mask, completions, prompt_len = generate_group(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            synced_gpus=False,
            group_size=rollout_group_size,
            **kwargs,
        )
        return NativeGeneration(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            completions=completions,
            prompt_len=prompt_len,
            metadata={"adapter": "hf_reference", "rollout_group_size": rollout_group_size},
        )

    def sequence_log_probs(self, sequences: Any, attention_mask: Any) -> torch.Tensor:
        self._require_ready()
        assert self.model is not None
        with torch.no_grad():
            return sequences_log_probs(
                self.model,
                sequences.to(self.device),
                attention_mask.to(self.device).bool(),
            ).detach()

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
        batch_size = int(self.config.training.optimize_completion_batch_size)
        optimizer_steps = 0
        skipped_nonfinite = 0
        loss_sum = 0.0
        for _ in range(optimize_times_per_step):
            for start in range(0, len(experiences) - batch_size + 1, batch_size):
                batch = _collate(experiences[start : start + batch_size], self.device)
                self.optimizer.zero_grad(set_to_none=True)
                log_probs = sequences_log_probs(self.model, batch.sequences, batch.attention_mask)
                loss = self.loss_fn(log_probs, batch.old_log_probs, batch.advantages, batch.action_mask)
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    continue
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                optimizer_steps += 1
                loss_sum += float(loss.detach().cpu())
        return {
            "optimized": optimizer_steps > 0,
            "replay_buffer_trainable_completion_count": len(experiences),
            "optimizer_steps": optimizer_steps,
            "skipped_nonfinite": skipped_nonfinite,
            "loss_mean": loss_sum / optimizer_steps if optimizer_steps else None,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        self._require_ready()
        assert self.model is not None
        assert self.tokenizer is not None
        save_lora_adapter(self.model, self.tokenizer, Path(path))

    def close(self) -> None:
        pass

    def _require_ready(self) -> None:
        if self.model is None or self.tokenizer is None or self.optimizer is None:
            raise RuntimeError("HFReferenceRuntime is not set up")


class _Batch:
    def __init__(self, items: list[Experience], device: torch.device) -> None:
        self.sequences = pad_sequence([item.sequences for item in items], batch_first=True).to(device)
        self.old_log_probs = pad_sequence([item.old_log_probs for item in items], batch_first=True).to(device)
        self.advantages = pad_sequence([item.advantages for item in items], batch_first=True, padding_value=0.0).to(device)
        self.attention_mask = pad_sequence([item.attention_mask for item in items], batch_first=True).bool().to(device)
        self.action_mask = pad_sequence([item.action_mask for item in items], batch_first=True).bool().to(device)


def _collate(items: list[Experience], device: torch.device) -> _Batch:
    return _Batch(items, device)


def _resolve_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")
