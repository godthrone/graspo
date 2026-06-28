"""Layer 2 — GraspoFlowTrainer（已迁移到 trainer/ 目录，类改目录模式）。

此文件保留为向后兼容的 re-export，实际实现位于 trainer/ 子目录中。
"""

from graspo.backends.graspoflow.trainer.stats import (  # noqa: F401
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
)
from graspo.backends.graspoflow.trainer.trainer import GraspoFlowTrainer  # noqa: F401
