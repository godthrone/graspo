from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graspo.core.buffer import Experience
from graspo.core.completion import ParsedCompletion, raw_parsed_completion
from graspo.core.schema import GraspoConfig, NativeTPConfig, Sample


DEFAULT_NATIVE_ADAPTER = "graspo.backends.native_tp.models.qwen.adapter:QwenNativeTPAdapter"
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
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
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
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> list[NativeGeneration]: ...

    def generate_sample_groups(
        self,
        *,
        samples: list[Sample],
        rollout_group_size: int,
        max_new_tokens: int,
        max_prompt_length: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> list[NativeGeneration]: ...

    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion: ...

    def sequence_log_probs(
        self,
        sequences: Any,
        attention_mask: Any,
        metadata: Any | None = None,
    ) -> Any: ...

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
        message_batches = list(kwargs.pop("message_batches"))
        tool_batches = kwargs.pop("tool_batches", None)
        if tool_batches is None:
            tool_batches = [None] * len(message_batches)
        return [
            adapter.generate_group(messages=messages, tools=tools, **kwargs)
            for messages, tools in zip(message_batches, tool_batches, strict=True)
        ]

    def generate_sample_groups(self, **kwargs: Any) -> list[NativeGeneration]:
        adapter = self._require_adapter()
        generate_sample_groups = getattr(adapter, "generate_sample_groups", None)
        if not callable(generate_sample_groups):
            raise RuntimeError("Native adapter does not support multimodal sample generation")
        return generate_sample_groups(**kwargs)

    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion:
        adapter = self._require_adapter()
        parse_completion = getattr(adapter, "parse_completion", None)
        if callable(parse_completion):
            return parse_completion(completion, sample)
        return raw_parsed_completion(completion)

    def sequence_log_probs(
        self,
        sequences: Any,
        attention_mask: Any,
        metadata: Any | None = None,
    ) -> Any:
        return self._require_adapter().sequence_log_probs(
            sequences, attention_mask, metadata=metadata
        )

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
    if int(native.pp_size) < 1:
        raise ValueError("pp_size must be >= 1")
    if bool(native.sequence_parallel):
        raise ValueError("native-tp v1 requires sequence_parallel=false")
    if int(native.tp_size) < 1:
        raise ValueError("tp_size must be >= 1")
    if int(native.pp_size) > 1 and int(native.tp_size) != 1:
        raise ValueError("native placement v1 supports pp_size>1 only with tp_size=1")
    if int(native.pp_micro_batch_size) < 1:
        raise ValueError("native_tp.pp_micro_batch_size must be >= 1")
    if int(native.forward_batch_size) < 1:
        raise ValueError(
            f"native_tp.forward_batch_size must be >= 1, got {native.forward_batch_size}"
        )
    schedule = str(native.pp_schedule or "simple")
    if schedule not in {"simple", "one_f_one_b"}:
        raise ValueError("native_tp.pp_schedule must be simple or one_f_one_b")
    if config.training.resume_from_checkpoint and config.lora.adapter_path:
        raise ValueError(
            "training.resume_from_checkpoint and lora.adapter_path cannot both be set; "
            "native checkpoint resume takes the full training state, while PEFT adapters are "
            "warm-start weights only"
        )
    if int(native.pp_max_inflight_microbatches) < 0:
        raise ValueError("native_tp.pp_max_inflight_microbatches must be >= 0")
    if schedule == "one_f_one_b" and int(native.pp_size) <= 1:
        raise ValueError("one_f_one_b pp_schedule requires pp_size>1")
    flattened = _flatten_keys(config.backend_config)
    forbidden = sorted(
        key for key in flattened if any(marker in key.lower() for marker in FORBIDDEN_CONFIG_KEYS)
    )
    allowed_native_prefix = "native_tp."
    forbidden = [
        key for key in forbidden if not key.startswith(allowed_native_prefix) and key != "native_tp"
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
            "native-tp runtime must not import forbidden frameworks: " + ", ".join(imported)
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
