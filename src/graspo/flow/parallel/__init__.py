"""flow.parallel — 并行计算设施：分布式状态、层放置、张量工具、PP 调度。

- state.py: TP/PP 分布式状态（RANK 等）
- placement_plan.py: 层放置规划
- tensor_utils.py: TP all-reduce / cuda snapshot / safetensors / collate
- pipeline_comm.py: 异步 P2P 通信管道（PipelineComm，isend/irecv + CUDA stream 重叠）
- scheduling/: PP 调度策略（PipelineScheduler / OneFOneB，可插拔）
"""

from .pipeline_comm import PipelineComm, wait_all
from .scheduling import OneFOneBScheduler, PipelineScheduler, build_scheduler

__all__ = [
    "PipelineComm",
    "wait_all",
    "PipelineScheduler",
    "OneFOneBScheduler",
    "build_scheduler",
]
