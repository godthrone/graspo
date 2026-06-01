from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graspo.core.buffer import Experience
from graspo.core.schema import GraspoConfig, MegatronNativeConfig


DEFAULT_NATIVE_ADAPTER = "graspo.backends.megatron_native.qwen_tp_adapter:QwenMegatronNativeAdapter"
FORBIDDEN_RUNTIME_MODULES = (
    "nemo_rl",
    "vllm",
    "ray",
    "deepspeed",
    "accelerate",
    "transformer_engine",
    "apex",
)
FORBIDDEN_CONFIG_KEYS = (
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


class MegatronNativeRuntimeProtocol(Protocol):
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

    def sequence_log_probs(self, sequences: Any, attention_mask: Any) -> Any: ...

    def train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        optimize_times_per_step: int,
        max_grad_norm: float,
    ) -> dict[str, Any]: ...

    def save_checkpoint(self, path: str | Path) -> None: ...

    def close(self) -> None: ...

    def is_primary(self) -> bool: ...


class MegatronNativeRuntime:
    """Strict native Megatron runtime boundary.

    This class intentionally does not import NeMo-RL, vLLM, Ray, DeepSpeed,
    DDP, or FSDP. The concrete Megatron Core/L.M. model adapter lives behind
    this boundary so GRASPO control flow remains self-owned and unit-testable.
    """

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.native_config = config.megatron_native
        self._adapter: Any | None = None

    @classmethod
    def from_config(cls, config: GraspoConfig) -> "MegatronNativeRuntime":
        return cls(config)

    def validate(self) -> None:
        validate_native_runtime_config(self.config, self.native_config)
        assert_forbidden_runtime_modules_not_imported()
        if not has_native_megatron_runtime():
            raise RuntimeError(
                "megatron-native requires Megatron Core/L.M. on the training server. "
                "Install Megatron as an optional external dependency; this backend does not "
                "fall back to NeMo-RL, vLLM, Ray, DeepSpeed, DDP, FSDP, or Accelerate."
            )

    def setup(self) -> None:
        self.validate()
        adapter_path = os.environ.get("GRASPO_MEGATRON_NATIVE_ADAPTER") or DEFAULT_NATIVE_ADAPTER
        module_name, sep, class_name = adapter_path.partition(":")
        if not sep:
            raise ValueError("GRASPO_MEGATRON_NATIVE_ADAPTER must use 'module:Class' format")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        self._adapter = adapter_cls(self.config)
        self._adapter.setup()

    def generate_group(self, **kwargs: Any) -> NativeGeneration:
        return self._require_adapter().generate_group(**kwargs)

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

    def save_checkpoint(self, path: str | Path) -> None:
        self._require_adapter().save_checkpoint(path)

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
            raise RuntimeError("Megatron native runtime is not set up")
        return self._adapter


def has_native_megatron_runtime() -> bool:
    return _has_module("megatron.core") or _has_module("megatron")


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def validate_native_runtime_config(
    config: GraspoConfig,
    native_config: MegatronNativeConfig | None = None,
) -> None:
    native = native_config or config.megatron_native
    if int(native.pipeline_model_parallel_size) != 1:
        raise ValueError("megatron-native v1 supports pipeline_model_parallel_size=1 only")
    if bool(native.sequence_parallel):
        raise ValueError("megatron-native v1 requires sequence_parallel=false")
    if int(native.tensor_model_parallel_size) < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")
    if int(native.train_micro_batch_size) != 1:
        raise ValueError("megatron-native v1 requires train_micro_batch_size=1")
    if int(native.generation_micro_batch_size) != 1:
        raise ValueError("megatron-native v1 requires generation_micro_batch_size=1")

    flattened = _flatten_keys(config.backend_config)
    forbidden = sorted(
        key for key in flattened if any(marker in key.lower() for marker in FORBIDDEN_CONFIG_KEYS)
    )
    allowed_native_prefix = "megatron_native."
    forbidden = [
        key
        for key in forbidden
        if not key.startswith(allowed_native_prefix) and key != "megatron_native"
    ]
    if forbidden:
        raise ValueError(
            "megatron-native forbids NeMo/vLLM/Ray/DeepSpeed/FSDP/DDP/ZeRO/Accelerate config keys: "
            + ", ".join(forbidden)
        )


def assert_forbidden_runtime_modules_not_imported() -> None:
    imported = [name for name in FORBIDDEN_RUNTIME_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(
            "megatron-native runtime must not import forbidden frameworks: "
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
