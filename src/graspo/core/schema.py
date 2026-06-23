from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from graspo.core.reward import RewardConfig


@dataclass(slots=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    adapter_path: str | None = None
    target_preset: str = "language_safe"
    target_modules: list[str] | None = None
    auto_target_modules: bool = True
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(slots=True)
class ModelConfig:
    model_path: str = ""
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = ""
    seed: int = 42
    training_epoch_count: int = 100
    max_steps: int = -1
    rollout_group_size: int = 8
    optimize_prompt_batch_size: int = 8
    optimize_times_per_step: int = 4
    rollout_max_retry_times: int = 5
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    policy_ratio_clip_eps: float = 0.2
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    save_steps: int = 50
    logging_steps: int = 1
    perfect_skip_reward_threshold: float = 1.0
    skip_format_broken_groups: bool = True
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None

    @property
    def replay_buffer_optimize_threshold(self) -> int:
        return int(self.optimize_prompt_batch_size) * int(self.rollout_group_size)


@dataclass(slots=True)
class DataConfig:
    train_path: str = ""
    max_prompt_length: int = 2048


@dataclass(slots=True)
class NativeTPConfig:
    tp_size: int = 2
    pp_size: int = 1
    placement_strategy: str = "auto"
    sequence_parallel: bool = False
    pp_micro_batch_size: int = 1
    forward_batch_size: int = 8
    use_kv_cache_for_rollout: bool = True
    empty_cache_after_rollout_split: bool = True
    empty_cache_before_train: bool = False
    checkpoint_format: str = "recoverable_lora_tp"
    raw_log_enabled: bool = True
    readable_log_enabled: bool = True
    synchronize_cuda_timing: bool = False
    pp_schedule: str = "simple"
    pp_max_inflight_microbatches: int = 0


@dataclass(slots=True)
class ExportConfig:
    final_formats: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LaunchConfig:
    gpus: list[int] | str | None = None
    nproc_per_node: int | None = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    python: str | None = None
    torchrun: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GraspoConfig:
    backend: str = "native-tp"
    backend_config: dict[str, Any] = field(default_factory=dict)
    native_tp: NativeTPConfig = field(default_factory=NativeTPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GraspoConfig":
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraspoConfig":
        import warnings

        data = data or {}
        backend_config = dict(data.get("backend_config", {}) or {})
        native_cfg = dict(backend_config.get("native_tp", {}) or {})
        native_cfg.update(data.get("native_tp", {}) or {})

        # Backward compat: warn about removed fields
        _removed_native = {
            "generation_micro_batch_size",
            "rollout_kv_cache_max_reserved_fraction",
            "gpu_memory_utilization",
        }
        for key in sorted(_removed_native & set(native_cfg)):
            if key == "gpu_memory_utilization":
                warnings.warn(
                    "native_tp.gpu_memory_utilization is removed. "
                    "Use native_tp.forward_batch_size instead (default 8).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"native_tp.{key} is removed.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            del native_cfg[key]

        training_cfg = _normalize_training_config(data.get("training", {}))
        return cls(
            backend=data.get("backend", "native-tp"),
            backend_config=backend_config,
            native_tp=NativeTPConfig(**native_cfg),
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**_normalize_data_config(data.get("data", {}))),
            lora=LoRAConfig(**data.get("lora", {})),
            export=ExportConfig(**data.get("export", {})),
            launch=LaunchConfig(**data.get("launch", {})),
            reward=RewardConfig(**data.get("reward", {})),
            training=TrainingConfig(**training_cfg),
        )


@dataclass(slots=True)
class Sample:
    messages: list[dict[str, Any]]
    targets: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    media: list[dict[str, Any]] = field(default_factory=list)

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
        return json.dumps(asdict(self), ensure_ascii=False)


def _content_preview(content: Any) -> str:
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


def _normalize_training_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(raw or {})
    if "replay_buffer_optimize_threshold" in config:
        raise ValueError(
            "training.replay_buffer_optimize_threshold is derived from "
            "optimize_prompt_batch_size * rollout_group_size and must not be configured"
        )
    removed = {
        "total_epochs",
        "rollout_prompt_queue_size",
        "rollout_prompt_queue_batch_size",
        "group_size",
        "train_batch_size",
        "buffer_train_rounds",
        "max_retry",
        "clip_eps",
        "perfect_reward_threshold",
        "each_step_prompts_per_device",
    }
    present = sorted(key for key in removed if key in config)
    if present:
        raise ValueError(
            "Removed training config field(s): " + ", ".join(f"training.{key}" for key in present)
        )
    return config


def _normalize_data_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(raw or {})
    removed = {"prompt_field", "messages_field", "ground_truth_field"}
    present = sorted(key for key in removed if key in config)
    if present:
        raise ValueError(
            "Removed data config field(s): "
            + ", ".join(f"data.{key}" for key in present)
            + ". GRASPO data is fixed to JSONL records with messages + optional tools + targets."
        )
    return config
