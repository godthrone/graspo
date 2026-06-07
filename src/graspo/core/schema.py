from __future__ import annotations

import json
import os
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
    model_path: str = "${MODEL_PATH}"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "${OUTPUT_DIR}"
    seed: int = 42
    training_epoch_count: int = 100
    max_steps: int = -1
    rollout_prompt_queue_batch_size: int = 1
    rollout_group_size: int = 8
    optimize_completion_batch_size: int = 4
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
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None
    legacy_config_aliases: list[str] = field(default_factory=list)

    @property
    def replay_buffer_optimize_threshold(self) -> int:
        return int(self.optimize_completion_batch_size) * int(self.rollout_group_size)


@dataclass(slots=True)
class DataConfig:
    train_path: str = "${DATA_PATH}"
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"
    messages_field: str = "messages"
    max_prompt_length: int = 2048


@dataclass(slots=True)
class NativeTPConfig:
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    placement_strategy: str = "auto"
    sequence_parallel: bool = False
    train_micro_batch_size: int = 1
    generation_micro_batch_size: int = 1
    use_kv_cache_for_rollout: bool = True
    rollout_kv_cache_max_reserved_fraction: float = 0.60
    empty_cache_after_rollout_split: bool = True
    empty_cache_before_train: bool = False
    checkpoint_format: str = "safetensors_or_native_tp"
    raw_log_enabled: bool = True
    readable_log_enabled: bool = True
    synchronize_cuda_timing: bool = False
    pipeline_train_schedule: str = "simple"
    pipeline_max_inflight_microbatches: int = 0


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
    backend: str = "auto"
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
        text = os.path.expandvars(text)
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraspoConfig":
        data = data or {}
        backend_config = dict(data.get("backend_config", {}) or {})
        native_cfg = dict(backend_config.get("native_tp", {}) or {})
        native_cfg.update(data.get("native_tp", {}) or {})
        training_cfg = _normalize_training_config(data.get("training", {}))
        return cls(
            backend=data.get("backend", "auto"),
            backend_config=backend_config,
            native_tp=NativeTPConfig(**native_cfg),
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            export=ExportConfig(**data.get("export", {})),
            launch=LaunchConfig(**data.get("launch", {})),
            reward=RewardConfig(**data.get("reward", {})),
            training=TrainingConfig(**training_cfg),
        )


@dataclass(slots=True)
class Sample:
    prompt: str
    ground_truth: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    media: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _normalize_training_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(raw or {})
    if "replay_buffer_optimize_threshold" in config:
        raise ValueError(
            "training.replay_buffer_optimize_threshold is derived from "
            "optimize_completion_batch_size * rollout_group_size and must not be configured"
        )

    aliases = {
        "total_epochs": "training_epoch_count",
        "rollout_prompt_queue_size": "rollout_prompt_queue_batch_size",
        "group_size": "rollout_group_size",
        "train_batch_size": "optimize_completion_batch_size",
        "buffer_train_rounds": "optimize_times_per_step",
        "max_retry": "rollout_max_retry_times",
        "clip_eps": "policy_ratio_clip_eps",
        "perfect_reward_threshold": "perfect_skip_reward_threshold",
    }
    used_aliases: list[str] = []
    for old_name, new_name in aliases.items():
        if old_name not in config:
            continue
        if new_name in config:
            raise ValueError(f"training.{old_name} and training.{new_name} cannot both be set")
        config[new_name] = config.pop(old_name)
        used_aliases.append(old_name)

    config.pop("each_step_prompts_per_device", None)
    if used_aliases:
        config["legacy_config_aliases"] = used_aliases
    return config
