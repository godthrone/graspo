from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class GroupDecision(StrEnum):
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


def group_advantages(
    rewards: Sequence[float],
    eps: float = 1e-8,
    quality_power: float = 2.0,
) -> list[float]:
    """质量加权 z-score：advantage_i = z_i × max(rewards)^quality_power。

    相对于纯 z-score 的改进（公式 A，v0.14.x）：

    1. **绝对质量进入梯度**。纯 z-score 中 ``[a, b×7]`` 组的最佳 advantage
       恒等于 ``7/√8 = 2.4749``，与 a、b 无关——考 100 分（reward=1.0）
       和考 43 分（reward=0.43）的组被授予完全相同的推力，模型学到
       "做组内最好"而非"做绝对正确"，最终崩向 not_correct 盆地。
       乘以 ``max^p`` 后，组的最佳推力随其最高 reward 单调缩放：
       max=1.0 → 满推，max=0.43 → 0.43²≈0.185 弱推。

    2. **reward 的 >1 额外奖励被保留**。reward 归一化设计下 1.0 =
       "全部正确"，>1 部分（如 anti-useless bonus，最多到 ~1.0048）
       是刻意留下的精细化激励。``max^p`` 不 clamp：全对+简洁的组
       （max=1.0048）获得 1.0096 倍推力，比全对+啰嗦的组（max=1.0）
       多约 1%——与 reward 设计语义自洽。无需 τ/clamp 的原因：
       reward 恒为 ``raw_score/max_score``，天然有界 [0, ~1.0048]。

    3. **低方差组放大被顺带压制**。rewards 挤在一起（std≈0.003）时
       纯 z-score 的 advantage 可达 ±10~100，劫持整批梯度尺度；
       这类组的 max 通常不高，``max^p`` 将其整体压回合理范围。

    性质：quality 是组内共同乘数（正数），组内相对顺序与正负号不变，
    零和性质保持（``Σz_i·g = g·Σz_i = 0``），loss 无组级偏置。
    """
    if not rewards:
        return []
    values = [float(reward) for reward in rewards]
    mean = sum(values) / len(values)
    if len(values) <= 1:
        std = 0.0
    else:
        variance = sum((reward - mean) ** 2 for reward in values) / (len(values) - 1)
        std = variance**0.5
    quality = max(values) ** quality_power
    return [((reward - mean) / (std + eps)) * quality for reward in values]


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
    etc.), the group is rejected — this is a **defense line** (Constitution 2.3),
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
        # The best completion's format is broken — the group has no valid
        # template to learn from.  Retry if we can, otherwise discard.
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
