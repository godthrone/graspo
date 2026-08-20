"""配置模型定义：GraspoConfig 及所有子配置段的 pydantic 模型，加载即校验。"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RewardConfig(BaseModel):
    """奖励评分配置，所有字段在加载时校验，拒绝未知字段。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["graspo"] = "graspo"
    check_think: bool = False
    check_json_markdown: bool = True
    check_list_order: bool = False
    marker_reward_weight: float = 10.0
    content_reward_weight: float = 100.0
    anti_useless_str_reward_weight: float = 1.0
    anti_useless_str_half_reward_len: int = 100
    numeric_tolerance: float = 0.2


class LoRAConfig(BaseModel):
    """LoRA 微调配置。"""

    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    adapter_path: str | None = None
    target_preset: str = "language_safe"
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseModel):
    """模型加载配置。"""

    model_config = ConfigDict(extra="forbid")

    model_path: str = ""
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict[str, Any] = {}


class LRSchedulerConfig(BaseModel):
    """学习率调度器配置。默认 ``type="constant"`` 保持 LR 不变，向后兼容。

    ``decay_steps`` 是纯调度参数：warmup 结束后的衰减跨度（optimizer-step
    粒度），调度在此步数内从 ``learning_rate`` 衰减到
    ``learning_rate × min_lr_ratio``，之后保持最低 LR 不变。训练长度只由
    ``training.max_epochs`` 控制（§7.4 参数单职责——max_steps 已于 v0.23.0
    移除：它曾同时承担提前终止与调度跨度两个角色，单位不一致互相矛盾）。
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "constant"  # "constant" | "cosine" | "linear"
    warmup_steps: int = 0  # 线性 warmup 步数（仅 cosine / linear）
    min_lr_ratio: float = 0.0  # 最终 LR = learning_rate × min_lr_ratio
    decay_steps: int = 0  # warmup 后的衰减跨度（optimizer-step）；type != constant 时必填


class TrainingConfig(BaseModel):
    """训练超参数配置。"""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = ""
    run_name: str = ""
    seed: int = 42
    # 训练长度唯一控制参数（v0.23.0 起：max_steps 已移除，见 LRSchedulerConfig）
    max_epochs: int = 100
    rollout_group_size: int = 8
    # 每次 rollout queue 的 prompt 数（默认 8）。与 optimize_prompt_batch_size
    # 解耦：队列决定采样吞吐与 replay 阈值（G × queue），optimize_prompt_batch_size
    # 决定训练 forward 每批序列数（显存峰值）。OOM 时单独调小后者即可，
    # 不需要牺牲吞吐。
    rollout_queue_batch_size: int = 8
    optimize_prompt_batch_size: int = 8
    # optimize_iterations_per_step removed — always 1 iteration per step.
    # GRPO with repeated iterations on stale old_log_probs causes catastrophic
    # forgetting of shared tokens (e.g. tool-call formatting).
    rollout_max_retries: int = 5
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    policy_ratio_clip_eps: float = 0.2
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    save_steps: int = -1
    save_checkpoint_every_epoch: bool = True
    perfect_skip_reward_threshold: float = 1.0
    reject_unparseable_groups: bool = True
    resume_from_checkpoint: str | None = None
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    # DP 学习率缩放策略：dp_size > 1 时按 linear scaling rule 调整 lr
    # "linear" = lr × dp_size；"none" = 不缩放
    lr_scaling: Literal["linear", "none"] = "linear"

    @model_validator(mode="after")
    def _validate_output_dir(self) -> TrainingConfig:
        """默认输出目录推导：outputs/<run_name>，run_name 自动生成。"""
        output_dir = str(self.output_dir or "").strip()
        run_name = str(self.run_name or "").strip()
        if not output_dir:
            if not run_name:
                run_name = _generate_run_name()
            self.output_dir = str(Path("outputs") / run_name)
            self.run_name = run_name
        elif not run_name:
            self.run_name = str(Path(output_dir).name)
        return self

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_max_steps(cls, data: Any) -> Any:
        """防呆：max_steps 已于 v0.23.0 移除，给出明确迁移提示而非模糊报错。

        训练长度只由 max_epochs 控制；学习率衰减跨度用 lr_scheduler.decay_steps。
        """
        if isinstance(data, dict) and "max_steps" in data:
            raise ValueError(
                "training.max_steps has been removed (v0.23.0): training length is "
                "controlled by training.max_epochs only; configure the LR decay span "
                "via lr_scheduler.decay_steps (in optimizer steps, warmup-exclusive)"
            )
        return data

    @model_validator(mode="after")
    def _validate_scheduler_decay_steps(self) -> TrainingConfig:
        if self.lr_scheduler.type != "constant" and self.lr_scheduler.decay_steps <= 0:
            raise ValueError(
                f"lr_scheduler.decay_steps must be > 0 when "
                f"lr_scheduler.type={self.lr_scheduler.type!r}, "
                f"got {self.lr_scheduler.decay_steps}"
            )
        return self

    @property
    def replay_buffer_optimize_threshold(self) -> int:
        return int(self.rollout_queue_batch_size) * int(self.rollout_group_size)


class DataConfig(BaseModel):
    """训练数据配置。"""

    model_config = ConfigDict(extra="forbid")

    train_path: str = ""
    max_prompt_length: int = 2048


class GraspoFlowConfig(BaseModel):
    """GraspoFlow 分布式训练配置 — TP+DP+PP+SP+Checkpoint 五位一体。

    world_size = dp_size × tp_size × pp_size，由框架自动校验。
    各维度正交：调整任何一个不影响其他维度的语义。
    """

    model_config = ConfigDict(extra="forbid")

    tp_size: int = 2
    dp_size: int = 1
    pp_size: int = 1
    # 是否在 DP rank 间复制 LoRA 权重（默认 true）。true=各 rank 独立副本，
    # 梯度 AVG all-reduce；false=权重分片，需额外通信。LoRA 参数量小，推荐 true。
    dp_replicate_lora: bool = True
    # 模型适配器路径，默认使用 qwen35_36（兼容 Qwen3.5/3.6 系列）
    adapter: str = "graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter"
    placement_strategy: str = "auto"
    # 手动指定每层的 stage 分布 [start, end) 区间，设置后覆盖 placement_strategy
    layer_ranges: list[list[int]] | None = None
    sequence_parallel: bool = False
    pp_micro_batch_size: int = 1
    forward_batch_size: int = 8
    use_kv_cache_for_rollout: bool = True
    empty_cache_after_rollout_split: bool = False
    empty_cache_before_train: bool = False
    raw_log_enabled: bool = True
    readable_log_enabled: bool = True
    synchronize_cuda_timing: bool = False
    pp_max_inflight_microbatches: int = 0

    @model_validator(mode="after")
    def _validate_dp(self) -> GraspoFlowConfig:
        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        return self


class ExportConfig(BaseModel):
    """模型导出配置。通过 ``graspo export --config <yaml>`` 驱动。"""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = ""
    export_format: str = "peft-adapter"
    export_output: str = ""
    final_formats: list[str] = []


class LaunchConfig(BaseModel):
    """分布式启动配置。

    GPU 选择由用户通过 run.sh ``--gpus`` 控制（绝不使用 CUDA_VISIBLE_DEVICES，
    与 --gpus 混用会导致 NCCL 死锁），不在配置中指定。
    ``nproc_per_node`` 默认从 ``dp_size × tp_size × pp_size`` 自动推导。
    """

    model_config = ConfigDict(extra="forbid")

    nproc_per_node: int | None = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    python: str | None = None


class GraspoConfig(BaseModel):
    """GRASPO 训练主配置，单一 YAML 入口，加载即校验。"""

    model_config = ConfigDict(extra="forbid")

    train_method: Literal["graspo", "sft"] = "graspo"
    backend: str = "graspoflow"
    graspoflow: GraspoFlowConfig = GraspoFlowConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    lora: LoRAConfig = LoRAConfig()
    export: ExportConfig = ExportConfig()
    launch: LaunchConfig = LaunchConfig()
    reward: RewardConfig = RewardConfig()
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraspoConfig:
        """从 YAML 文件加载配置，加载时完成全部校验。

        校验失败时输出人类可读的错误信息（哪个字段、期望什么、可用字段有哪些），
        而不是原始 pydantic traceback。
        """
        import yaml
        from pydantic import ValidationError

        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            return cls.from_dict(yaml.safe_load(text))
        except ValidationError as exc:
            _report_config_errors(exc, path, cls)
            raise SystemExit(1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraspoConfig:
        """从字典构建配置，pydantic ``model_validate`` 一次性校验所有字段。

        **防呆：** 不手动挑键——顶层任何未知键（拼写错误）
        由 ``extra="forbid"`` 直接拒绝，而不是静默忽略后用默认值训练。
        仅做一件显式处理：None 段防御——显式 ``section: null`` 等价于缺省
        （合法），未知键仍被拒绝。
        """
        data = dict(data or {})
        data["graspoflow"] = dict(data.get("graspoflow") or {})
        for section in ("model", "data", "lora", "export", "launch", "reward", "training"):
            if data.get(section) is None:
                data[section] = {}
        return cls.model_validate(data)


class Sample(BaseModel):
    """单条训练样本，包含 messages、targets 和可选的 tools（不可变）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = {}
    media: list[dict[str, Any]] = []

    @property
    def expects_tool_calls(self) -> bool:
        for target in self.targets:
            output = target.get("output") if isinstance(target, dict) else None
            if isinstance(output, dict) and output.get("tool_calls") is not None:
                return True
        return False

    @property
    def prompt_preview(self) -> str:
        parts: list[str] = []
        for message in self.messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            parts.append(f"{role}: {_content_preview(content)}")
        return "\n\n".join(part for part in parts if part)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


# ── 辅助函数 ────────────────────────────────────────────────────────────────


def _report_config_errors(
    exc: Any, config_path: Path, config_cls: Any
) -> None:
    """将 pydantic ValidationError 转成人类可读的错误信息。

    不输出原始 traceback，而是按字段分组，列出每个问题字段、
    期望的值、可用字段列表。
    """
    import sys

    errors = exc.errors()
    # 构建各段可用字段的映射：顶层 + 每个子段
    config_section_fields: dict[str, list[str]] = {
        section: sorted(
            [
                k
                for k in config_cls.model_fields[section].annotation.model_fields  # type: ignore[union-attr]
            ]
            if section in config_cls.model_fields
            else []
        )
        for section in (
            "graspoflow",
            "model",
            "data",
            "lora",
            "export",
            "launch",
            "reward",
            "training",
        )
    }
    config_section_fields[""] = sorted(
        k for k in config_cls.model_fields if k not in config_section_fields
    )

    lines: list[str] = []
    lines.append(f"\n配置校验失败 ({config_path}):\n")
    for error in errors:
        loc = ".".join(str(p) for p in error["loc"])
        msg = error["msg"]
        err_type = error["type"]

        if err_type == "extra_forbidden":
            # 拼写错误或已删除的字段
            lines.append(f"  {loc}: 未知字段，当前版本不支持此配置项")
            # 尝试找到所属段，列出可用字段
            section = str(error["loc"][0]) if error["loc"] else ""
            if section in config_section_fields and config_section_fields[section]:
                fields = ", ".join(config_section_fields[section])
                lines.append(f"    {section} 段可用字段: {fields}")
            elif "" in config_section_fields:
                fields = ", ".join(config_section_fields[""])
                lines.append(f"    顶层可用字段: {fields}")
        elif err_type == "missing":
            lines.append(f"  {loc}: 缺少必填字段")
        elif err_type == "string_type":
            lines.append(f"  {loc}: 期望字符串，实际收到 {error.get('input')}")
        elif err_type == "int_type":
            lines.append(f"  {loc}: 期望整数，实际收到 {error.get('input')}")
        elif err_type == "bool_type":
            lines.append(f"  {loc}: 期望布尔值，实际收到 {error.get('input')}")
        elif err_type == "literal_error":
            lines.append(f"  {loc}: 期望 {error.get('expected')}，实际收到 {error.get('input')}")
        else:
            lines.append(f"  {loc}: {msg}")

    print("\n".join(lines), file=sys.stderr)


_RUN_NAME_CACHE: dict[str, str] = {}
"""模块级缓存，确保同一秒内多次调用（如 torchrun 多 worker）得到相同的 run_name。"""


def _generate_run_name() -> str:
    """生成基于时间戳的唯一运行标识。"""
    if "current" not in _RUN_NAME_CACHE:
        _RUN_NAME_CACHE["current"] = f"graspo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return _RUN_NAME_CACHE["current"]


def _content_preview(content: Any) -> str:
    """生成消息内容的可读预览。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            item_type = str(item.get("type") or "").lower()
            if item_type == "text":
                parts.append(str(item.get("text") or ""))
            elif item_type in {"image", "image_url"}:
                parts.append("<image>")
            elif item_type in {"video", "video_url"}:
                parts.append("<video>")
            else:
                parts.append(f"<{item_type or 'content'}>")
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)
