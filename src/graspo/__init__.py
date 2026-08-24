"""GRASPO — Group Relative Advantage Structured Policy Optimization.

GRPO-style LoRA reinforcement learning for structured-output tasks.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("graspo")
except PackageNotFoundError:  # pragma: no cover — 未安装时回退
    __version__ = "0.0.0"

from graspo.core.schema import GraspoConfig, Sample
from graspo.ripple.reward.reward import GraspoReward, RewardConfig, RewardResult

__all__ = [
    "GraspoConfig",
    "GraspoReward",
    "RewardConfig",
    "RewardResult",
    "Sample",
]
