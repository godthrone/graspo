"""组分类决策与 reward 工具函数（GRASPO 算法核心，纯 CPU、零 torch）。

注意：历史上曾有"公式 A 质量加权 z-score advantage"（整个 completion 给一个
标量 advantage），已被 v0.20.0 起的**字符级标注驱动的 token 级 advantage**
（`ripple.annotation.advantages.compute_group_advantages`）取代——它只能给整条
completion 一个标量推力，无法像 token 级那样在错误的 token 上给负梯度。该旧函数
已移除，本模块只保留组分类决策与 reward 工具函数。
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class GroupDecision(StrEnum):
    PERFECT_SKIP = "perfect_skip"
    RETRY = "retry"
    INVALID = "invalid"
    INVALID_NO_PREFERENCE_GAP = "invalid_no_preference_gap"
    TRAINABLE_MAX_CORRECT = "trainable_max_correct"
    TRAINABLE_NOT_CORRECT = "trainable_not_correct"


class GroupSampleDecision(BaseModel):
    """组分类决策结果（不可变数据模型）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

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
    rollout_max_retries: int,
    perfect_skip_reward_threshold: float = 1.0,
    best_completion_has_parse_error: bool = False,
    reject_unparseable_groups: bool = True,
) -> GroupSampleDecision:
    """Classify a rollout group into a training decision.

    When ``reject_unparseable_groups`` is True and the best-scoring completion
    has parse errors (invalid JSON, unclosed fences, tool-call count mismatch,
    etc.), the group is rejected — this is a **defense line** (fail-closed),
    not a fallback.  Unparseable completions are invalid data that cannot produce
    meaningful training signals; blocking them at the boundary prevents noise
    from entering the training pipeline.
    """
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
    elif reject_unparseable_groups and best_completion_has_parse_error:
        if retry_count < rollout_max_retries:
            decision = GroupDecision.RETRY
        else:
            decision = GroupDecision.INVALID
    elif reward_max > reward_median:
        decision = GroupDecision.TRAINABLE_NOT_CORRECT
    elif reward_max < perfect_skip_reward_threshold and retry_count < rollout_max_retries:
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
    rollout_queue_batch_size: int,
    rollout_group_size: int,
) -> int:
    """Replay 触发阈值 = 一次 rollout queue 的 prompt 数 × 每组 completion 数。

    语义：一个 rollout queue（rollout_queue_batch_size 个 prompt × G 条/组）恰好
    触发一次 optimize——队列节奏与训练微批大小（gradient_accumulation_micro_batches）
    解耦：队列可保持大吞吐（rollout 显存由 micro_batch_size 决定），训练
    forward 微批可单独调小（显存由 micro_batch_size 决定）。
    """
    return int(rollout_queue_batch_size) * int(rollout_group_size)


def replay_ready(
    replay_size: int,
    rollout_queue_batch_size: int,
    rollout_group_size: int,
) -> bool:
    return int(replay_size) >= replay_buffer_optimize_threshold(
        rollout_queue_batch_size,
        rollout_group_size,
    )
