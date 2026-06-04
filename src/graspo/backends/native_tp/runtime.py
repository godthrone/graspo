from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graspo.core.buffer import Experience
from graspo.core.schema import GraspoConfig, NativeTPConfig


DEFAULT_NATIVE_ADAPTER = "graspo.backends.native_tp.qwen_tp_adapter:QwenNativeTPAdapter"
NATIVE_TP_ADAPTER_ENV = "GRASPO_NATIVE_TP_ADAPTER"
FORBIDDEN_RUNTIME_MODULES = (
    "megatron",
    "nemo_rl",
    "vllm",
    "ray",
    "deepspeed",
    "accelerate",
    "transformer_engine",
    "apex",
)
FORBIDDEN_CONFIG_KEYS = (
    "megatron",
    "nemo",
    "vllm",
    "ray",
    "deepspeed",
    "zero",
    "fsdp",
    "ddp",
    "accelerate",
    "transformer_engine",
    "apex",
)


@dataclass(slots=True)
class NativeGeneration:
    sequences: Any
    attention_mask: Any
    action_mask: Any
    completions: list[str]
    prompt_len: int = 0
    metadata: dict[str, Any] | None = None


class NativeTPRuntimeProtocol(Protocol):
    def validate(self) -> None: ...

    def setup(self) -> None: ...

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
    ) -> NativeGeneration: ...

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
    ) -> list[NativeGeneration]: ...

    def sequence_log_probs(self, sequences: Any, attention_mask: Any) -> Any: ...

    def train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]: ...

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        trainer_state: dict[str, Any] | None = None,
    ) -> None: ...

    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None: ...

    def close(self) -> None: ...

    def is_primary(self) -> bool: ...


class NativeTPRuntime:
    """Strict self-owned tensor-parallel runtime boundary.

    The production path uses PyTorch distributed directly and intentionally does
    not import Megatron, NeMo-RL, vLLM, Ray, DeepSpeed, DDP, FSDP, Accelerate,
    TransformerEngine, or Apex.
    """

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.native_config = config.native_tp
        self._adapter: Any | None = None

    @classmethod
    def from_config(cls, config: GraspoConfig) -> "NativeTPRuntime":
        return cls(config)

    def validate(self) -> None:
        validate_native_runtime_config(self.config, self.native_config)
        assert_forbidden_runtime_modules_not_imported()

    def setup(self) -> None:
        self.validate()
        adapter_path = os.environ.get(NATIVE_TP_ADAPTER_ENV) or DEFAULT_NATIVE_ADAPTER
        module_name, sep, class_name = adapter_path.partition(":")
        if not sep:
            raise ValueError(f"{NATIVE_TP_ADAPTER_ENV} must use 'module:Class' format")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        self._adapter = adapter_cls(self.config)
        self._adapter.setup()

    def generate_group(self, **kwargs: Any) -> NativeGeneration:
        return self._require_adapter().generate_group(**kwargs)

    def generate_groups(self, **kwargs: Any) -> list[NativeGeneration]:
        adapter = self._require_adapter()
        generate_groups = getattr(adapter, "generate_groups", None)
        if callable(generate_groups):
            return generate_groups(**kwargs)
        prompts = list(kwargs.pop("prompts"))
        return [adapter.generate_group(prompt=prompt, **kwargs) for prompt in prompts]

    def sequence_log_probs(self, sequences: Any, attention_mask: Any) -> Any:
        return self._require_adapter().sequence_log_probs(sequences, attention_mask)

    def train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        return self._require_adapter().train_batch(
            experiences,
            policy_ratio_clip_eps=policy_ratio_clip_eps,
            optimize_times_per_step=optimize_times_per_step,
            max_grad_norm=max_grad_norm,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        trainer_state: dict[str, Any] | None = None,
    ) -> None:
        self._require_adapter().save_checkpoint(path, trainer_state=trainer_state)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None:
        adapter = self._require_adapter()
        if not hasattr(adapter, "load_checkpoint"):
            raise RuntimeError("Native TP adapter does not support checkpoint resume")
        return adapter.load_checkpoint(path)

    def close(self) -> None:
        if self._adapter is not None and hasattr(self._adapter, "close"):
            self._adapter.close()

    def is_primary(self) -> bool:
        adapter = self._adapter
        if adapter is None:
            return True
        if hasattr(adapter, "is_primary"):
            return bool(adapter.is_primary())
        return int(getattr(adapter, "rank", 0)) == 0

    def _require_adapter(self):
        if self._adapter is None:
            raise RuntimeError("Native TP runtime is not set up")
        return self._adapter


def validate_native_runtime_config(
    config: GraspoConfig,
    native_config: NativeTPConfig | None = None,
) -> None:
    native = native_config or config.native_tp
    if int(native.pipeline_model_parallel_size) != 1:
        raise ValueError("native-tp v1 supports pipeline_model_parallel_size=1 only")
    if bool(native.sequence_parallel):
        raise ValueError("native-tp v1 requires sequence_parallel=false")
    if int(native.tensor_model_parallel_size) < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")
    if int(native.train_micro_batch_size) != 1:
        raise ValueError("native-tp v1 requires train_micro_batch_size=1")
    if int(native.generation_micro_batch_size) != 1:
        raise ValueError("native-tp v1 requires generation_micro_batch_size=1")
    if int(config.training.rollout_prompt_queue_batch_size) < 1:
        raise ValueError("training.rollout_prompt_queue_batch_size must be >= 1")

    flattened = _flatten_keys(config.backend_config)
    forbidden = sorted(
        key for key in flattened if any(marker in key.lower() for marker in FORBIDDEN_CONFIG_KEYS)
    )
    allowed_native_prefix = "native_tp."
    forbidden = [
        key
        for key in forbidden
        if not key.startswith(allowed_native_prefix) and key != "native_tp"
    ]
    if forbidden:
        raise ValueError(
            "native-tp forbids Megatron/NeMo/vLLM/Ray/DeepSpeed/FSDP/DDP/ZeRO/Accelerate config keys: "
            + ", ".join(forbidden)
        )


def assert_forbidden_runtime_modules_not_imported() -> None:
    imported = [name for name in FORBIDDEN_RUNTIME_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(
            "native-tp runtime must not import forbidden frameworks: "
            + ", ".join(imported)
        )


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix] if prefix else []
    keys: list[str] = []
    for key, child in value.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(child_key)
        keys.extend(_flatten_keys(child, child_key))
    return keys
