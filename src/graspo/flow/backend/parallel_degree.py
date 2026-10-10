"""flow.backend.parallel_degree — ★ 并行度（dp/tp/pp 的规模与秩）的**单一真相源**。

★★ 一名一物（adapter 三义消歧之第 2 义）：
代码里原先写作 ``self.adapter.dp_size`` / ``self.adapter.dp_rank`` 的，**就是并行度**，
不是"算法↔后端对接契约"（``flow/adapter/``），也不是"模型 / LoRA 适配"
（``flow/backend/native/models/<m>/model_adapter.py``）。并行度是**后端设施**的量：
它描述"这次运行把模型铺在多少 rank 上、本进程是哪个"，与算法和模型族都无关。

因此：
- 类型改名 :class:`ParallelDegree`（不再叫 adapter）；
- 落位 ``flow/backend/``（跨 backend 抽象，不是某个 backend 的私有物）；
- 模型适配器只**持有一个** ``ParallelDegree``（``self.parallel``），不再自己散存
  ``dp_size`` / ``dp_rank`` / ``tp_*`` / ``pp_*`` 九個字段（宪法 §1.4）。

值来源：``config.native.{tp_size,dp_size,pp_size}``（世界大小与各维规模）与
``flow/parallel/state.py::GraspoFlowState``（setup 后的真实规模与秩）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParallelDegree:
    """一次 native 运行的并行度（不可变值对象）。

    字段语义与 ``GraspoFlowState`` 逐字对齐；``world_size == tp_size × pp_size × dp_size``。
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    @property
    def dp_groups(self) -> int:
        """DP 维度的组数（= ``dp_size``，保留具名读数口，避免调用点重推）。"""
        return self.dp_size

    @classmethod
    def from_config(cls, config: Any) -> ParallelDegree:
        """setup 之前的静态规模（秩一律为 0）。

        与旧实现 ``TransformerModelAdapter.__init__`` 逐字同值：
        ``tp_size=config.native.tp_size``、``dp_size=config.native.dp_size``、
        ``pp_size=config.native.pp_size``，其余秩为 0、``world_size=1``。
        """
        return cls(
            tp_size=int(config.native.tp_size),
            dp_size=int(config.native.dp_size),
            pp_size=int(config.native.pp_size),
        )

    @classmethod
    def from_state(cls, state: Any) -> ParallelDegree:
        """setup 之后的真实规模与秩（``GraspoFlowState`` 的只读投影）。"""
        return cls(
            rank=int(state.rank),
            local_rank=int(state.local_rank),
            world_size=int(state.world_size),
            tp_size=int(state.tp_size),
            tp_rank=int(state.tp_rank),
            pp_size=int(state.pp_size),
            pp_rank=int(state.pp_rank),
            dp_size=int(state.dp_size),
            dp_rank=int(state.dp_rank),
        )


__all__ = ["ParallelDegree"]
