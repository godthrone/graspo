"""Layer 2 — GraspoFlowRuntime: tensor-parallel runtime boundary.

Delegates all work to a model-specific adapter.  The adapter is loaded
dynamically via ``GRASPO_ADAPTER`` environment variable or
a sensible default.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from graspo.core.buffer import Experience
from graspo.core.completion import ParsedCompletion
from graspo.core.schema import GraspoConfig, Sample

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


@dataclass(slots=True)
class NativeGeneration:
    sequences: Any
    attention_mask: Any
    action_mask: Any
    completions: list[str]
    prompt_len: int = 0
    metadata: dict[str, Any] | None = None


class GraspoFlowRuntimeProtocol(Protocol):
    def validate(self) -> None: ...
    def setup(self) -> None: ...

    def generate_group(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> NativeGeneration: ...

    def generate_groups(
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]: ...

    def generate_sample_groups(
        self,
        *,
        samples: list[Any],
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]: ...
    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion: ...
    def sequence_log_probs(self, sequences: Any, attention_mask: Any, metadata: Any | None = None) -> Any: ...
    def train_batch(self, experiences: list[Experience], *, policy_ratio_clip_eps: float, optimize_times_per_step: int, max_grad_norm: float) -> dict[str, Any]: ...
    def save_checkpoint(self, path: str | Path, *, trainer_state: dict[str, Any] | None = None) -> None: ...
    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None: ...
    def close(self) -> None: ...
    def is_primary(self) -> bool: ...


class GraspoFlowRuntime:
    """Strict self-owned tensor-parallel runtime boundary.

    The production path uses PyTorch distributed directly and intentionally does
    not import Megatron, NeMo-RL, vLLM, Ray, DeepSpeed, DDP, FSDP, Accelerate,
    TransformerEngine, or Apex.
    """

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.graspoflow_config = config.graspoflow
        self._adapter: Any | None = None

    @classmethod
    def from_config(cls, config: GraspoConfig) -> GraspoFlowRuntime:
        return cls(config)

    def validate(self) -> None:
        validate_graspoflow_runtime_config(self.config, self.graspoflow_config)
        assert_forbidden_runtime_modules_not_imported()

    def setup(self) -> None:
        self.validate()
        adapter_path = self.graspoflow_config.adapter
        module_name, sep, class_name = adapter_path.partition(":")
        if not sep:
            raise ValueError("graspoflow.adapter 必须使用 'module:Class' 格式")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        self._adapter = adapter_cls(self.config)
        self._adapter.setup()

    def generate_group(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> NativeGeneration:
        return self._require_adapter().generate_group(
            messages=messages,
            tools=tools,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )

    def generate_groups(
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]:
        return self._require_adapter().generate_groups(
            message_batches=message_batches,
            tool_batches=tool_batches,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )

    def generate_sample_groups(
        self,
        *,
        samples: list[Any],
        rollout_group_size: int,
        max_new_tokens: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]:
        return self._require_adapter().generate_sample_groups(
            samples=samples,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )

    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion:
        return self._require_adapter().parse_completion(completion, sample)

    def sequence_log_probs(
        self, sequences: Any, attention_mask: Any, metadata: Any | None = None
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
        self, path: str | Path, *, trainer_state: dict[str, Any] | None = None
    ) -> None:
        self._require_adapter().save_checkpoint(path, trainer_state=trainer_state)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None:
        return self._require_adapter().load_checkpoint(path)

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()

    def is_primary(self) -> bool:
        adapter = self._adapter
        if adapter is None:
            return True
        return bool(adapter.is_primary())

    def _require_adapter(self):
        if self._adapter is None:
            raise RuntimeError("GraspoFlow runtime is not set up")
        return self._adapter


def validate_graspoflow_runtime_config(
    config: GraspoConfig, graspoflow_config: Any | None = None
) -> None:
    native = graspoflow_config or config.graspoflow
    if int(native.pp_size) < 1:
        raise ValueError("pp_size must be >= 1")
    if bool(native.sequence_parallel):
        raise ValueError("graspoflow requires sequence_parallel=false")
    if int(native.tp_size) < 1:
        raise ValueError("tp_size must be >= 1")
    if int(native.pp_micro_batch_size) < 1:
        raise ValueError("graspoflow.pp_micro_batch_size must be >= 1")
    if int(native.forward_batch_size) < 1:
        raise ValueError(
            f"graspoflow.forward_batch_size must be >= 1, got {native.forward_batch_size}"
        )
    schedule = str(native.pp_schedule or "simple")
    if schedule not in {"simple", "one_f_one_b"}:
        raise ValueError("graspoflow.pp_schedule must be simple or one_f_one_b")
    if config.training.resume_from_checkpoint and config.lora.adapter_path:
        raise ValueError(
            "training.resume_from_checkpoint and lora.adapter_path cannot both be set"
        )
    if int(native.pp_max_inflight_microbatches) < 0:
        raise ValueError("graspoflow.pp_max_inflight_microbatches must be >= 0")
    if schedule == "one_f_one_b" and int(native.pp_size) <= 1:
        raise ValueError("one_f_one_b pp_schedule requires pp_size>1")


def assert_forbidden_runtime_modules_not_imported() -> None:
    imported = [name for name in FORBIDDEN_RUNTIME_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(
            "graspoflow runtime must not import forbidden frameworks: "
            + ", ".join(imported)
        )
