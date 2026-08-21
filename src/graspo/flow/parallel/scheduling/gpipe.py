"""GPipeScheduler — 全 forward 后全 backward 调度（一步到位）。

调度时序（与 1F1B 的 fill/steady/drain 不同）：

- **全 forward**：每个 stage 依次处理所有 microbatch 的前向。
- **全 backward**：所有前向完成后，依次处理所有 microbatch 的反向。

**为什么用 GPipe 而非 1F1B（设计决策）**：

1. **P2P tag 对齐**：1F1B 的 fill/steady/drain 交错使上游 stage 先发出多个
   send、下游 stage 先 recv 再 send（反向），导致 NCCL P2P 的 tag 顺序跨
   stage 不一致，产生死锁（实测 "+1 collective timeout"）。GPipe 中所有 stage
   先做等量 forward（send/recv 顺序一致），tag 天然对齐，无死锁。

2. **bubble 可接受**：工作日志（2026-08-19）实测 1F1B 的 bubble 优化对当前
   工作负载无实质收益（"对 9B 是负优化；PP 只在模型太大时有用"）。GPipe 的
   bubble 略大但正确性优先（宪法 §5 先正确后可优化）。

3. **未来优化**：interleaved / ZeroBubble 仍是可插拔策略，在异步 P2P 通信层
   上作为新调度器插入，无需重写通信层。
"""

from __future__ import annotations

import time
from typing import Any

from .base import PipelineScheduler


class GPipeScheduler(PipelineScheduler):
    """GPipe 调度：全 forward 后全 backward。"""

    def run(self) -> dict[str, Any]:
        forward_sec = 0.0
        backward_sec = 0.0

        fwd_started = time.monotonic()
        for chunk_idx in range(self.num_chunks):
            self.forward(chunk_idx)
        forward_sec = time.monotonic() - fwd_started

        bwd_started = time.monotonic()
        for chunk_idx in range(self.num_chunks):
            self.backward(chunk_idx)
        backward_sec = time.monotonic() - bwd_started

        return {
            "pp_schedule": "gpipe",
            "pipeline_forward_sec": forward_sec,
            "pipeline_backward_sec": backward_sec,
        }
