from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class GroupDecision(str, Enum):
    PERFECT_SKIP = "perfect_skip"
    RETRY = "retry"
    INVALID = "invalid"
    INVALID_NO_PREFERENCE_GAP = "invalid_no_preference_gap"
    TRAINABLE_MAX_CORRECT = "trainable_max_correct"
    TRAINABLE_NOT_CORRECT = "trainable_not_correct"


@dataclass(frozen=True, slots=True)
class GroupSampleDecision:
    decision: GroupDecision
    reward_min: float
    reward_median: float
    reward_max: float
    reward_mean: float
    content_mean: float | None
    retry_count: int

    @property
    def should_retry(self) -> bool:
        return self.decision == GroupDecision.RETRY

    @property
    def should_train(self) -> bool:
        return self.decision in {
            GroupDecision.TRAINABLE_MAX_CORRECT,
            GroupDecision.TRAINABLE_NOT_CORRECT,
        }

    @property
    def reward_max_median_gap(self) -> float:
        return self.reward_max - self.reward_median


def lower_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sorted(float(value) for value in values)[(len(values) - 1) // 2]


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    if not rewards:
        return []
    values = [float(reward) for reward in rewards]
    mean = sum(values) / len(values)
    if len(values) <= 1:
        std = 0.0
    else:
        variance = sum((reward - mean) ** 2 for reward in values) / (len(values) - 1)
        std = variance**0.5
    return [(reward - mean) / (std + eps) for reward in values]


def has_reward_variance(rewards: Sequence[float], eps: float = 1e-12) -> bool:
    if len(rewards) < 2:
        return False
    values = [float(reward) for reward in rewards]
    return max(values) - min(values) > eps


def is_uniform_partial_content(content_scores: Sequence[float]) -> bool:
    if not content_scores:
        return False
    values = [float(score) for score in content_scores]
    min_value = min(values)
    max_value = max(values)
    return 0.0 < min_value == max_value < 1.0


def is_invalid_group(rewards: Sequence[float], content_scores: Sequence[float]) -> bool:
    return not has_reward_variance(rewards) or is_uniform_partial_content(content_scores)


def classify_group(
    rewards: Sequence[float],
    content_scores: Sequence[float],
    *,
    retry_count: int,
    rollout_max_retry_times: int,
    perfect_skip_reward_threshold: float = 1.0,
) -> GroupSampleDecision:
    values = [float(reward) for reward in rewards]
    if not values:
        return GroupSampleDecision(
            decision=GroupDecision.INVALID,
            reward_min=0.0,
            reward_median=0.0,
            reward_max=0.0,
            reward_mean=0.0,
            content_mean=None,
            retry_count=retry_count,
        )

    reward_min = min(values)
    reward_max = max(values)
    reward_median = lower_median(values)
    reward_mean = sum(values) / len(values)
    content_mean = (
        sum(float(score) for score in content_scores) / len(content_scores)
        if content_scores
        else None
    )

    if reward_median >= perfect_skip_reward_threshold:
        decision = GroupDecision.PERFECT_SKIP
    elif reward_max >= perfect_skip_reward_threshold:
        decision = GroupDecision.TRAINABLE_MAX_CORRECT
    elif reward_max > reward_median and reward_median >= 0.4:
        decision = GroupDecision.TRAINABLE_NOT_CORRECT
    elif reward_max < perfect_skip_reward_threshold and retry_count < rollout_max_retry_times:
        decision = GroupDecision.RETRY
    elif is_invalid_group(values, content_scores):
        decision = GroupDecision.INVALID
    elif reward_max == reward_median:
        decision = GroupDecision.INVALID_NO_PREFERENCE_GAP
    else:
        decision = GroupDecision.INVALID

    return GroupSampleDecision(
        decision=decision,
        reward_min=reward_min,
        reward_median=reward_median,
        reward_max=reward_max,
        reward_mean=reward_mean,
        content_mean=content_mean,
        retry_count=retry_count,
    )


def replay_buffer_optimize_threshold(
    optimize_prompt_batch_size: int,
    rollout_group_size: int,
) -> int:
    return int(optimize_prompt_batch_size) * int(rollout_group_size)


def replay_ready(
    replay_size: int,
    optimize_prompt_batch_size: int,
    rollout_group_size: int,
) -> bool:
    return int(replay_size) >= replay_buffer_optimize_threshold(
        optimize_prompt_batch_size,
        rollout_group_size,
    )
