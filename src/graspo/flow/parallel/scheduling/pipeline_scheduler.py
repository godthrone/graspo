"""PP 调度策略基类（PipelineScheduler）—— 调度层契约。

调度层只关注"何时执行 forward/backward"，与通信层（AsyncPipe）和计算层
（adapter）解耦。新增调度（interleaved 1F1B / V-shape / ZeroBubble）只需
实现本接口并注册，不改通信或计算层——满足宪法 §1.2（对扩展开放，对修改关闭）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class PipelineScheduler(ABC):
    """PP 调度策略 — 决定 forward/backward 的执行顺序。

    :param pp_rank: 当前 pipeline stage 的 rank（0-based）
    :param pp_size: pipeline 总 stage 数
    :param num_chunks: 本批次待处理的 microbatch 数
    :param forward: ``forward(chunk_idx)`` — 执行一个 microbatch 的前向
        （内部负责接收上游 hidden、计算、发送下游 hidden；非首 stage 无输入）
    :param backward: ``backward(chunk_idx)`` — 执行一个 microbatch 的反向
        （内部负责接收下游梯度、回传梯度、计算本 stage 梯度）
    """

    def __init__(
        self,
        *,
        pp_rank: int,
        pp_size: int,
        num_chunks: int,
        forward: Callable[[int], None],
        backward: Callable[[int], None],
    ) -> None:
        self.pp_rank = int(pp_rank)
        self.pp_size = int(pp_size)
        self.num_chunks = int(num_chunks)
        self.forward = forward
        self.backward = backward

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """执行一次 pipeline 调度。

        :returns: 调度的统计信息（如 fill/steady/drain 耗时），供调用方聚合。
        """
        raise NotImplementedError
