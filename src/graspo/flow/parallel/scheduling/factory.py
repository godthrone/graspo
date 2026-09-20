"""调度策略工厂 — 按配置构建 PP 调度器。

集中管理调度策略的名称与实现映射。未来新增 interleaved / ZeroBubble 调度时，
在此登记（并实现对应的 ``PipelineScheduler`` 子类），调用方无需改动。
"""

from __future__ import annotations

from typing import Any

from graspo.flow.parallel.scheduling.one_f_one_b_scheduler import OneFOneBScheduler
from graspo.flow.parallel.scheduling.pipeline_scheduler import PipelineScheduler

# 注册表：调度策略名称 → 构造器。（新增策略在此登记。）
_SCHEDULERS: dict[str, type[PipelineScheduler]] = {
    "one_f_one_b": OneFOneBScheduler,
    "1f1b": OneFOneBScheduler,
}


def build_scheduler(
    name: str | None = None,
    *,
    pp_rank: int,
    pp_size: int,
    num_chunks: int,
    forward: Any,
    backward: Any,
) -> PipelineScheduler:
    """按名称构建调度器。默认（``name`` 为空或 ``default``）使用 1F1B。

    1F1B 是 PP 的唯一调度：异步 P2P 通信层（双向进程组 + 显式 tag + 背压）消除了
    早期 1F1B 在单 peer-pair FIFO 上的消息错配死锁，实测比旧 GPipe 更快
    （PP=4 冒烟 38.50s < 45.94s）。GPipe（全 forward → 全 backward）因 bubble
    最高且不重叠 forward/backward，已按宪法 §18.1 删除。

    :param name: 调度策略名（``one_f_one_b`` / ``1f1b`` / ``default``）
    :param pp_rank: pipeline stage rank
    :param pp_size: pipeline 总 stage 数
    :param num_chunks: microbatch 数
    :param forward: ``forward(chunk_idx)`` 回调
    :param backward: ``backward(chunk_idx)`` 回调
    """
    key = (name or "default").strip().lower()
    if key in ("", "default", "auto"):
        scheduler_cls = OneFOneBScheduler
    else:
        scheduler_cls = _SCHEDULERS.get(key)
    if scheduler_cls is None:
        raise ValueError(
            f"Unsupported PP scheduler: {name!r}. Available: "
            f"{sorted(set(_SCHEDULERS) | {'default'})}"
        )
    return scheduler_cls(
        pp_rank=pp_rank,
        pp_size=pp_size,
        num_chunks=num_chunks,
        forward=forward,
        backward=backward,
    )


__all__ = ["PipelineScheduler", "OneFOneBScheduler", "build_scheduler"]
