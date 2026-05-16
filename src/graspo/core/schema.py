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
    total_epochs: int = 1
    max_steps: int = -1
    prompts_per_rank: int = 1
    group_size: int = 8
    train_batch_size: int = 4
    epochs_per_step: int = 4
    max_retry: int = 5
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    save_steps: int = 50
    logging_steps: int = 1
    perfect_reward_threshold: float = 1.0
    dataloader_num_workers: int = 0


@dataclass(slots=True)
class DataConfig:
    train_path: str = "${DATA_PATH}"
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"
    messages_field: str = "messages"
    max_prompt_length: int = 2048


@dataclass(slots=True)
class GraspoConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
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
        return cls(
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            reward=RewardConfig(**data.get("reward", {})),
            training=TrainingConfig(**data.get("training", {})),
        )


@dataclass(slots=True)
class Sample:
    prompt: str
    ground_truth: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
