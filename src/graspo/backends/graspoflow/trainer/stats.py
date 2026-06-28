"""训练统计数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graspo.core.schema import Sample


@dataclass(slots=True)
class GraspoFlowTrainStats:
    """全局训练统计，跨 epoch 累计。"""

    total_groups: int = 0
    perfect_skipped: int = 0
    retries: int = 0
    invalid: int = 0
    invalid_no_preference_gap: int = 0
    trainable: int = 0
    trainable_max_correct: int = 0
    trainable_not_correct: int = 0
    optimized_steps: int = 0


@dataclass(slots=True)
class GraspoFlowEpochStats:
    """单个 epoch 的训练统计。"""

    epoch: int = 0
    samples_seen: int = 0
    attempt_groups: int = 0
    completion_count: int = 0
    perfect_skipped: int = 0
    retries: int = 0
    invalid: int = 0
    invalid_no_preference_gap: int = 0
    trainable: int = 0
    trainable_max_correct: int = 0
    trainable_not_correct: int = 0
    reward_mean_sum: float = 0.0
    content_mean_sum: float = 0.0
    best_reward: float = 0.0


@dataclass(slots=True)
class _QueuedSample:
    """训练队列中的待处理样本。"""

    sample: Sample
    retry_count: int = 0
    attempts: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class _AttemptRecord:
    """单次 rollout 尝试的完整记录。"""

    sample: Sample
    generation: Any  # NativeGeneration
    parsed_completions: list[Any]  # ParsedCompletion
    rewards: list[float]
    content_scores: list[float]
    all_right: list[bool]
    reward_details: list[dict[str, Any]]
    decision: Any
    retry_count: int
    readable: dict[str, Any]
    timing: dict[str, Any]
