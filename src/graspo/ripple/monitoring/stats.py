"""训练统计数据结构。"""

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
    base_content_mean_sum: float = 0.0
    best_reward: float = 0.0


@dataclass(slots=True)
class QueuedSample:
    """训练队列中的待处理样本。"""

    sample: Sample
    retry_count: int = 0
    attempts: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class AttemptRecord:
    """单次 rollout 尝试的完整记录。"""

    sample: Sample
    generation: Any  # NativeGeneration
    parsed_completions: list[Any]  # ParsedCompletion
    rewards: list[float]
    content_scores: list[float]
    base_content_scores: list[float]
    all_right: list[bool]
    reward_details: list[dict[str, Any]]
    decision: Any
    retry_count: int
    readable: dict[str, Any]
    timing: dict[str, Any]


def train_stats_to_dict(stats: GraspoFlowTrainStats) -> dict[str, Any]:
    return {
        "total_groups": stats.total_groups,
        "perfect_skipped": stats.perfect_skipped,
        "retries": stats.retries,
        "invalid": stats.invalid,
        "invalid_no_preference_gap": stats.invalid_no_preference_gap,
        "trainable": stats.trainable,
        "trainable_max_correct": stats.trainable_max_correct,
        "trainable_not_correct": stats.trainable_not_correct,
        "optimized_steps": stats.optimized_steps,
    }


def epoch_stats_to_dict(stats: GraspoFlowEpochStats) -> dict[str, Any]:
    return {
        "epoch": stats.epoch,
        "samples_seen": stats.samples_seen,
        "attempt_groups": stats.attempt_groups,
        "completion_count": stats.completion_count,
        "perfect_skipped": stats.perfect_skipped,
        "retries": stats.retries,
        "invalid": stats.invalid,
        "invalid_no_preference_gap": stats.invalid_no_preference_gap,
        "trainable": stats.trainable,
        "trainable_max_correct": stats.trainable_max_correct,
        "trainable_not_correct": stats.trainable_not_correct,
        "reward_mean_sum": stats.reward_mean_sum,
        "content_mean_sum": stats.content_mean_sum,
        "base_content_mean_sum": stats.base_content_mean_sum,
        "best_reward": stats.best_reward,
    }


def train_stats_from_dict(raw: dict[str, Any]) -> GraspoFlowTrainStats:
    return GraspoFlowTrainStats(
        total_groups=int(raw.get("total_groups") or raw.get("attempt_groups") or 0),
        perfect_skipped=int(raw.get("perfect_skipped") or 0),
        retries=int(raw.get("retries") or 0),
        invalid=int(raw.get("invalid") or 0),
        invalid_no_preference_gap=int(raw.get("invalid_no_preference_gap") or 0),
        trainable=int(raw.get("trainable") or 0),
        trainable_max_correct=int(raw.get("trainable_max_correct") or 0),
        trainable_not_correct=int(raw.get("trainable_not_correct") or 0),
        optimized_steps=int(raw.get("optimized_steps") or 0),
    )


def epoch_stats_from_dict(raw: dict[str, Any]) -> GraspoFlowEpochStats:
    return GraspoFlowEpochStats(
        epoch=int(raw.get("epoch") or 0),
        samples_seen=int(raw.get("samples_seen") or 0),
        attempt_groups=int(raw.get("attempt_groups") or 0),
        completion_count=int(raw.get("completion_count") or raw.get("completions") or 0),
        perfect_skipped=int(raw.get("perfect_skipped") or 0),
        retries=int(raw.get("retries") or 0),
        invalid=int(raw.get("invalid") or 0),
        invalid_no_preference_gap=int(raw.get("invalid_no_preference_gap") or 0),
        trainable=int(raw.get("trainable") or 0),
        trainable_max_correct=int(raw.get("trainable_max_correct") or 0),
        trainable_not_correct=int(raw.get("trainable_not_correct") or 0),
        reward_mean_sum=float(raw.get("reward_mean_sum") or 0.0),
        content_mean_sum=float(raw.get("content_mean_sum") or 0.0),
        base_content_mean_sum=float(raw.get("base_content_mean_sum") or 0.0),
        best_reward=float(raw.get("best_reward") or 0.0),
    )
