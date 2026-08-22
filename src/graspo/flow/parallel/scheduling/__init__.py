"""flow.parallel.scheduling — PP 调度层（策略模式）。

对外透明：只导出 PipelineScheduler / OneFOneB / build_scheduler，
外部不感知内部是单文件还是目录（宪法 §8.3 类改目录）。
"""

from .base import PipelineScheduler
from .factory import build_scheduler
from .one_f_one_b import OneFOneBScheduler

__all__ = [
    "PipelineScheduler",
    "OneFOneBScheduler",
    "build_scheduler",
]
