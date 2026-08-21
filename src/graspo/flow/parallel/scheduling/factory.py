"""调度策略工厂 — 按配置构建 PP 调度器。

集中管理调度策略的名称与实现映射。未来新增 interleaved / ZeroBubble 调度时，
在此登记（并实现对应的 ``PipelineScheduler`` 子类），调用方无需改动。
"""

from __future__ import annotations

from typing import Any

from .base import PipelineScheduler
from .one_f_one_b import OneFOneBScheduler

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
