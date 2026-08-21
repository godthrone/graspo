"""OneFOneBScheduler — 标准 1F1B 调度（fill → steady → drain）。

调度时序（与当前 ``training.py`` / ``training_sft.py`` 手写 1F1B 一致，但
抽为可插拔策略）：

- **fill**：前向 ``warmup`` 个 microbatch（尚未开始反向）
- **steady**：交错前向/反向
- **drain**：反向收尾

其中：:

    warmup = min(pp_size - pp_rank - 1, num_chunks)

**死锁提醒**：本调度要求调用方提供**异步 P2P 通信**（``AsyncPipe``）。若用
阻塞式 ``dist.send``/``dist.recv``，fill 阶段上游 stage 的发送会阻塞等待下游
stage 尚未到达的 receive，导致时序死锁（详见 ``docs/flow.md`` 设计决策记录）。
"""

from __future__ import annotations

import time
from typing import Any

from .base import PipelineScheduler


class OneFOneBScheduler(PipelineScheduler):
    """标准 1F1B（fill → steady → drain）。"""

    def run(self) -> dict[str, Any]:
        pp_size = self.pp_size
        pp_rank = self.pp_rank
        chunk_count = self.num_chunks

        warmup = min(pp_size - pp_rank - 1, chunk_count)
        fill_sec = 0.0
        steady_sec = 0.0
        drain_sec = 0.0

        # fill
        fill_started = time.monotonic()
        for chunk_idx in range(warmup):
            self.forward(chunk_idx)
        fill_sec = time.monotonic() - fill_started

        # steady
        remaining = chunk_count - warmup
        steady_started = time.monotonic()
        for offset in range(remaining):
            self.forward(offset + warmup)
            self.backward(offset)
        steady_sec = time.monotonic() - steady_started

        # drain
        drain_started = time.monotonic()
        for chunk_idx in range(remaining, chunk_count):
            self.backward(chunk_idx)
        drain_sec = time.monotonic() - drain_started

        return {
            "pp_schedule": "one_f_one_b",
            "pipeline_fill_sec": fill_sec,
            "pipeline_steady_sec": steady_sec,
            "pipeline_drain_sec": drain_sec,
        }
