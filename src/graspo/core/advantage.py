from __future__ import annotations

from collections.abc import Sequence


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(len(rewards) - 1, 1)
    std = variance**0.5
    return [(reward - mean) / (std + eps) for reward in rewards]


def has_reward_variance(rewards: Sequence[float], eps: float = 1e-12) -> bool:
    if len(rewards) < 2:
        return False
    return max(rewards) - min(rewards) > eps

