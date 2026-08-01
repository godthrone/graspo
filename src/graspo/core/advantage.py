from collections.abc import Sequence

from graspo.core.graspo_parity import group_advantages as _parity_group_advantages
from graspo.core.graspo_parity import has_reward_variance as _parity_has_reward_variance


def group_advantages(
    rewards: Sequence[float], eps: float = 1e-8, quality_power: float = 2.0
) -> list[float]:
    return _parity_group_advantages(rewards, eps=eps, quality_power=quality_power)


def has_reward_variance(rewards: Sequence[float], eps: float = 1e-12) -> bool:
    return _parity_has_reward_variance(rewards, eps=eps)
