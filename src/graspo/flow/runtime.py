"""Layer 2 — GraspoFlowRuntime: tensor-parallel runtime boundary.

Delegates all work to a model-specific adapter.  The adapter class path is
configured via ``native.adapter`` in the YAML config (default:
``graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter``).
"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict, cast

from pydantic import BaseModel, ConfigDict

from graspo.core.schema import ROLLOUT_TRAIN_METHODS, GraspoConfig, Sample
from graspo.ripple.buffer import Experience
from graspo.ripple.parsing.completion import ParsedCompletion


class TrainBatchMetrics(TypedDict, total=False):
    """train_batch / train_batch_sft 返回的指标载荷（RL 与 SFT 共有的核心键）。

    权重范数与梯度计数是**模式感知**的（``flow/progress_metrics.py``）：

    - ``tuner_type == "lora"``：``lora_norm_*`` 有值、``trainable_norm_*`` 缺省/None；
    - ``tuner_type == "full"``：``lora_norm_*`` 为 ``None``（该指标**不适用**——全参下
      不存在 lora 参数，恒 0 会被误读成"权重没变"），改用等价的 ``trainable_norm_*``。
    """

    optimized: bool
    skipped_nonfinite: int
    loss_mean: float | None
    grad_norm_mean: float | None
    nonzero_grad_count: int
    #: ``nonzero_grad_count`` 的**定义**：``nonzero_lora_grads``（lora）
    #: 或 ``grad_populated_trainable_params``（full）。
    grad_count_metric: str
    tuner_type: str
    norm_metric: str
    lora_norm_before: float | None
    lora_norm_after: float | None
    lora_norm_delta: float | None
    trainable_norm_before: float | None
    trainable_norm_after: float | None
    trainable_norm_delta: float | None
    train_batch_total_sec: float
    micro_batch_forward_sec: float
    backward_sec: float
    optimizer_step_sec: float
    micro_batch_count: int
    current_lr: float | None
    optimizer_steps: int
    sft_batch_count: int


from graspo.core.discovery import _discover  # noqa: E402

_AVAILABLE_ADAPTERS = tuple(_discover("graspo.adapters").keys())

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


class NativeGeneration(BaseModel):
    """生成结果数据容器（不可变，跨模块边界传递）。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sequences: Any
    attention_mask: Any
    action_mask: Any
    completions: list[str]
    prompt_len: int = 0
    metadata: dict[str, Any] | None = None


class GraspoFlowRuntimeBase(ABC):
    """GraspoFlow 运行时抽象基类——所有 runtime 实现的契约。

    Trainer 混合类通过此 ABC 调用 runtime，无需任何 ``getattr``/``callable()``
    探测（防呆）。子类必须实现所有抽象方法。
    """

    _adapter: Any | None  # 内部状态：setup 后加载的模型适配器

    @property
    @abstractmethod
    def rank(self) -> int:
        """当前进程的分布式 rank。"""
        ...

    @property
    @abstractmethod
    def tp_rank(self) -> int:
        """当前进程的 tensor-parallel rank。"""
        ...

    @abstractmethod
    def validate(self) -> None: ...

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def is_primary(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def generate_group(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_prompt_length: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> NativeGeneration: ...

    @abstractmethod
    def generate_groups(
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_prompt_length: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]: ...

    @abstractmethod
    def generate_sample_groups(
        self,
        *,
        samples: list[Any],
        rollout_group_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_prompt_length: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]: ...

    @abstractmethod
    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion: ...

    @abstractmethod
    def sequence_log_probs(
        self, sequences: Any, attention_mask: Any, metadata: Any | None = None
    ) -> Any: ...

    @abstractmethod
    def train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
    ) -> TrainBatchMetrics: ...

    @abstractmethod
    def train_batch_sft(
        self,
        sft_batches: list[Any],  # SFTTokenized
        *,
        max_grad_norm: float,
    ) -> TrainBatchMetrics: ...

    @abstractmethod
    def save_checkpoint(
        self, path: str | Path, *, trainer_state: dict[str, Any] | None = None
    ) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> dict[str, Any] | None: ...

    def _require_adapter(self) -> Any:
        """返回已加载的模型适配器（子类可覆盖以提供具体实现）。"""
        raise NotImplementedError


class GraspoFlowRuntime(GraspoFlowRuntimeBase):
    """Strict self-owned tensor-parallel runtime boundary.

    The production path uses PyTorch distributed directly and intentionally does
    not import Megatron, NeMo-RL, vLLM, Ray, DeepSpeed, DDP, FSDP, Accelerate,
    TransformerEngine, or Apex.
    """

    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.native_config = config.native
        self._adapter: Any | None = None

    @classmethod
    def from_config(cls, config: GraspoConfig) -> GraspoFlowRuntime:
        return cls(config)

    @property
    def rank(self) -> int:
        adapter = self._adapter
        if adapter is not None:
            return int(getattr(adapter, "rank", 0))
        return 0

    @property
    def tp_rank(self) -> int:
        adapter = self._adapter
        if adapter is not None:
            return int(getattr(adapter, "tp_rank", 0))
        return 0

    def validate(self) -> None:
        validate_native_runtime_config(self.config, self.native_config)
        assert_forbidden_runtime_modules_not_imported()

    def setup(self) -> None:
        self.validate()
        adapter_path = self.native_config.adapter
        module_name, sep, class_name = adapter_path.partition(":")
        if not sep:
            raise ValueError(
                "native.adapter 必须使用 'module:Class' 格式；"
                f"可用适配器：{', '.join(_AVAILABLE_ADAPTERS)}"
            )
        try:
            module = importlib.import_module(module_name)
            adapter_cls = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError) as exc:
            raise ValueError(
                f"无法加载适配器 {adapter_path!r}: {exc}；"
                f"可用适配器：{', '.join(_AVAILABLE_ADAPTERS)}"
            ) from exc
        self._adapter = adapter_cls(self.config)
        self._adapter.setup()

    def generate_group(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_prompt_length: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> NativeGeneration:
        return self._require_adapter().generate_group(
            messages=messages,
            tools=tools,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_prompt_length=max_prompt_length,
            chat_template_kwargs=chat_template_kwargs,
        )

    def generate_groups(
        self,
        *,
        message_batches: list[list[dict[str, Any]]],
        tool_batches: list[list[dict[str, Any]] | None] | None = None,
        rollout_group_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_prompt_length: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]:
        return self._require_adapter().generate_groups(
            message_batches=message_batches,
            tool_batches=tool_batches,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_prompt_length=max_prompt_length,
            chat_template_kwargs=chat_template_kwargs,
        )

    def generate_sample_groups(
        self,
        *,
        samples: list[Any],
        rollout_group_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_prompt_length: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[NativeGeneration]:
        return self._require_adapter().generate_sample_groups(
            samples=samples,
            rollout_group_size=rollout_group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_prompt_length=max_prompt_length,
            chat_template_kwargs=chat_template_kwargs,
        )

    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion:
        return self._require_adapter().parse_completion(completion, sample)

    def sequence_log_probs(
        self, sequences: Any, attention_mask: Any, metadata: Any | None = None
    ) -> Any:
        return self._require_adapter().sequence_log_probs(
            sequences=sequences,
            attention_mask=attention_mask,
            metadata=metadata,
        )

    def train_batch(
        self,
        experiences: list[Experience],
        *,
        policy_ratio_clip_eps: float,
        max_grad_norm: float,
    ) -> TrainBatchMetrics:
        return cast(
            TrainBatchMetrics,
            self._require_adapter().train_batch(
                experiences=experiences,
                policy_ratio_clip_eps=policy_ratio_clip_eps,
                max_grad_norm=max_grad_norm,
            ),
        )

    def train_batch_sft(
        self,
        sft_batches: list[Any],  # SFTTokenized
        *,
        max_grad_norm: float,
    ) -> TrainBatchMetrics:
        return cast(
            TrainBatchMetrics,
            self._require_adapter().train_batch_sft(
                sft_batches,
                max_grad_norm=max_grad_norm,
            ),
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

    def _require_adapter(self) -> Any:
        if self._adapter is None:
            raise RuntimeError("GraspoFlow runtime is not set up")
        return self._adapter


def validate_native_runtime_config(config: GraspoConfig, native_config: Any | None = None) -> None:
    native = native_config or config.native
    if int(native.pp_size) < 1:
        raise ValueError("pp_size must be >= 1")
    if int(native.tp_size) < 1:
        raise ValueError("tp_size must be >= 1")
    if int(native.dp_size) < 1:
        raise ValueError("dp_size must be >= 1")
    # SP: TP>1 时自动启用（激活值沿序列维度分片，零额外通信量）
    # 显式 sequence_parallel: false 可强制禁用（调试用）
    if bool(native.sequence_parallel) and int(native.tp_size) < 2:
        raise ValueError("sequence_parallel requires tp_size >= 2")
    if int(native.pp_micro_batch_size) < 1:
        raise ValueError("native.pp_micro_batch_size must be >= 1")
    if int(native.micro_batch_size) < 1:
        raise ValueError(f"native.micro_batch_size must be >= 1, got {native.micro_batch_size}")
    if config.training.resume_from_checkpoint and config.lora.adapter_path:
        raise ValueError("training.resume_from_checkpoint and lora.adapter_path cannot both be set")
    if int(native.pp_max_inflight_microbatches) < 0:
        raise ValueError("native.pp_max_inflight_microbatches must be >= 0")
    if int(native.pp_p2p_timeout_sec) < 0:
        raise ValueError(
            "native.pp_p2p_timeout_sec must be >= 0 (0 disables the bounded PP "
            "rendezvous wait, which restores the pre-P6 unbounded hang)"
        )
    if int(native.pp_rollout_no_progress_sec) <= 0:
        raise ValueError(
            "native.pp_rollout_no_progress_sec must be >= 1 (the PP rollout "
            "no-progress watchdog is what bounds stalls the NCCL watchdog cannot see)"
        )
    _validate_native_pp_rollout_gate(config, native)


def _validate_native_pp_rollout_gate(config: GraspoConfig, native: Any) -> None:
    """未验证组合 ``native + pp_size>1 + rollout`` 的**启动期**显式闸门（P6 c1）。

    为什么放在这里而不是配置加载（pydantic）：闸门要求的是"**启动期**拒绝"，
    且它只在 native 运行链路上有意义（``GraspoConfig`` 的加载期契约被既有 46 档
    与多个测试依赖，不动它 = 不改既有契约面，§2.3 收紧而非放宽）。

    为什么默认拒绝而不是"先跑着看"：该组合**从未有一档端到端通过**
    （矩阵 54 档里只有 T035/T036 命中，T036 从未跑、T035 两次实测分别被
    600s 超时掩盖与无界挂死）；默认放行等于把"能力缺口"伪装成"偶发 bug"。

    为什么不用环境变量/SKIP：那正是 §3.4 说的坏退路（让档位看起来通过）。
    这里是**显式配置预授权**：开关名、默认值、消费点各只有一处（§1.4）。
    """
    if not (native_pp_size := int(native.pp_size)) > 1:
        return
    if str(config.train_method) not in ROLLOUT_TRAIN_METHODS:
        return
    if bool(native.allow_unverified_pp_rollout):
        return
    raise RuntimeError(
        "native PP rollout is not end-to-end verified and is refused by default: "
        f"train_method={config.train_method!r} + native + pp_size={native_pp_size} "
        "runs a rollout through pipeline-parallel generation, which has never passed "
        "an end-to-end run (all 54 matrix tiers: only T035/T036 hit this combination, "
        "neither passed; the only current coverage is a CPU-mock unit test). "
        "Observed failure mode (defect P6, 2026-09-22): the run hangs after the first "
        "prefill, with no NCCL watchdog timeout (invisible to NCCL), i.e. an unbounded "
        "hang. To proceed you must opt in explicitly: set "
        "native.allow_unverified_pp_rollout=true (and expect a bounded failure thanks "
        "to native.pp_p2p_timeout_sec / native.pp_rollout_no_progress_sec), or run "
        f"pp_size=1, or use the msswift backend."
    )


def assert_forbidden_runtime_modules_not_imported() -> None:
    imported = [name for name in FORBIDDEN_RUNTIME_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(
            "native runtime must not import forbidden frameworks: " + ", ".join(imported)
        )
