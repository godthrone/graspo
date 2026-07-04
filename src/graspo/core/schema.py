from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from graspo.core.reward import RewardConfig


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
    """学习率调度器配置。默认 ``type="constant"`` 保持 LR 不变，向后兼容。"""

    model_config = ConfigDict(extra="forbid")

    type: str = "constant"  # "constant" | "cosine" | "linear"
    warmup_steps: int = 0  # 线性 warmup 步数（仅 cosine / linear）
    min_lr_ratio: float = 0.0  # 最终 LR = learning_rate × min_lr_ratio


class TrainingConfig(BaseModel):
    """训练超参数配置。"""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = ""
    run_name: str = ""
    seed: int = 42
    max_epochs: int = 100
    max_steps: int = -1
    rollout_group_size: int = 8
    optimize_prompt_batch_size: int = 8
    optimize_iterations_per_step: int = 3
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

    @property
    def replay_buffer_optimize_threshold(self) -> int:
        return int(self.optimize_prompt_batch_size) * int(self.rollout_group_size)


class DataConfig(BaseModel):
    """训练数据配置。"""

    model_config = ConfigDict(extra="forbid")

    train_path: str = ""
    max_prompt_length: int = 2048


class GraspoFlowConfig(BaseModel):
    """GraspoFlow 分布式训练配置。"""

    model_config = ConfigDict(extra="forbid")

    tp_size: int = 2
    pp_size: int = 1
    # 模型适配器路径，默认使用 qwen35_36（兼容 Qwen3.5/3.6 系列）
    adapter: str = "graspo.backends.graspoflow.models.qwen35_36.adapter:Qwen35Adapter"
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
    pp_schedule: str = "simple"
    pp_max_inflight_microbatches: int = 0


class ExportConfig(BaseModel):
    """模型导出配置。通过 ``graspo export --config <yaml>`` 驱动。"""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = ""
    export_format: str = "peft-adapter"
    export_output: str = ""
    final_formats: list[str] = []


class LaunchConfig(BaseModel):
    """分布式启动配置。"""

    model_config = ConfigDict(extra="forbid")

    gpus: list[int] | str | None = None
    nproc_per_node: int | None = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    python: str | None = None
    torchrun: str | None = None
    env: dict[str, str] = {}


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
    training: TrainingConfig = TrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraspoConfig:
        """从 YAML 文件加载配置，加载时完成全部校验。"""
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraspoConfig:
        """从字典构建配置，pydantic ``extra="forbid"`` 自动拒绝未知字段。"""
        data = data or {}
        flow_cfg = _resolve_graspoflow_config(data)

        # 解析 output_dir 和 run_name（宪法 §8.5：默认 Outputs + 自动 run_name）
        training_raw = dict(data.get("training", {}) or {})
        output_dir = str(training_raw.get("output_dir", "") or "").strip()
        run_name = str(training_raw.get("run_name", "") or "").strip()
        if not output_dir:
            if not run_name:
                run_name = _generate_run_name()
            output_dir = str(Path("outputs") / run_name)
            training_raw["output_dir"] = output_dir
            training_raw["run_name"] = run_name
        elif not run_name:
            training_raw["run_name"] = str(Path(output_dir).name)

        return cls(
            train_method=data.get("train_method", "graspo"),
            backend=data.get("backend", "graspoflow"),
            graspoflow=GraspoFlowConfig(**flow_cfg),
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            export=ExportConfig(**data.get("export", {})),
            launch=LaunchConfig(**data.get("launch", {})),
            reward=RewardConfig(**data.get("reward", {})),
            training=TrainingConfig(**training_raw),
        )


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

_RUN_NAME_CACHE: dict[str, str] = {}
"""模块级缓存，确保同一秒内多次调用（如 torchrun 多 worker）得到相同的 run_name。"""


def _generate_run_name() -> str:
    """生成基于时间戳的唯一运行标识（宪法 §8.5）。"""
    if "current" not in _RUN_NAME_CACHE:
        _RUN_NAME_CACHE["current"] = f"graspo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return _RUN_NAME_CACHE["current"]


def _resolve_graspoflow_config(data: dict[str, Any]) -> dict[str, Any]:
    """解析 graspoflow 配置，支持两种格式并给出迁移提示。

    规范格式：顶层 ``graspoflow:`` 键。
    旧格式：``backend_config.graspoflow:`` 嵌套键（仍然兼容，但输出 DEPRECATION 警告）。
    """
    if "graspoflow" in data and data["graspoflow"]:
        return dict(data["graspoflow"])
    if "backend_config" in data and isinstance(data["backend_config"], dict):
        bc = data["backend_config"]
        if "graspoflow" in bc and bc["graspoflow"]:
            import warnings

            msg = (
                "backend_config.graspoflow 已废弃，请将 graspoflow 配置提升到 YAML 顶层。"
                " 参考 config_example.yaml 的最新格式。"
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)
            logging.getLogger("graspo.config").warning(msg)
            return dict(bc["graspoflow"])
    return {}


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
