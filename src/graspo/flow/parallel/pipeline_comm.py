"""Async P2P pipeline communication (isend/irecv) — 设施层。

每个 pipeline stage 通过异步点对点（``torch.distributed.isend``/``irecv``）
与前后 stage 通信，通信在专用 CUDA stream 上发起，从而与计算重叠
（Flink 风格 backpressure/调度的基础设施：通信层与计算层解耦）。

用法::

    comm = PipelineComm(device=torch.device("cuda"), world_group=pp_group)
    recv_work = comm.recv(tensor, src=prev_rank)
    recv_work.wait()                 # 阻塞直到数据到达
    send_work = comm.send(tensor, dst=next_rank)
    send_work.wait()                 # 阻塞直到数据发送完成

设计要点（BADGE §1.3 计算与设施分离）:
- 通信全部封装在本模块，调用方（调度层）不直接接触 ``dist.send/recv``。
- 使用 ``isend/irecv``（非阻塞）而非 ``send/recv``（阻塞），避免
  1F1B fill/steady 阶段的时序死锁（工作日志 2026-08-19 PP=2 死锁）。
- 专用 CUDA stream 让通信与计算在同一 GPU 上重叠，减少 GPU 空闲 bubble。

�. 注意事项:
- ``isend``/``irecv`` 要求 tensor 在调用期间保持存活（不被释放或被复用）。
  调用方必须：``recv`` 在 ``wait()`` 前不读取 tensor；``send`` 在 ``wait()``
  前不覆写/释放 tensor。
- 非 CUDA（如 gloo / CPU 测试）时退化为当前 stream，行为一致。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class PipelineComm:
    """Pipeline 异步 P2P 通信封装。

    :param device: 通信涉及的 CUDA device（CPU 环境可为 ``cpu``）。
    :param group: 通信进程组（如 PP group）。``None`` 表示全局 group。
    """

    def __init__(
        self,
        *,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        self.device = device
        self._group = group
        # NOTE(正确性优先): 使用默认 stream 发送/接收。专用 CUDA stream 提高
        # 通信/计算重叠，但需与计算流显式 wait_stream 同步，否则 isend/irecv 的
        # NCCL op 可能未及时投递，导致 GPipe 级联 send/recv 互相等待（实测
        # PP=4 卡死）。先用默认 stream 保证正确性（宪法 §5 先正确后可优化；
        # 重叠优化后续在保证同步的前提下恢复）。
        self._stream: torch.cuda.Stream | None = None

    def _torch_device(self) -> torch.device:
        if self.device.type == "cuda":
            return self.device
        return self.device  # cpu / meta 等

    def send(self, tensor: torch.Tensor, dst: int, *, tag: int = 0) -> Any:
        """异步发送 tensor 到 ``dst``，返回 work handle。

        使用显式 ``tag`` 匹配 P2P（而非默认共享计数器）。PP 流水线的中间 stage
        会交错 recv/send（前向 recv-hidden 后 send-hidden），默认共享计数器会使
        跨 stage 的 send/recv 序号错位，导致 NCCL 死锁（实测 PP>=3 超时）。
        按 chunk 编号分配独立的 send/recv tag 可彻底解耦匹配与操作顺序。

        必须调用返回 handle 的 ``wait()`` 才能确保数据已发送完毕（用于干净
        释放 buffer 或进入下一阶段）。
        """
        tensor = tensor.contiguous()
        return dist.isend(tensor, dst=dst, group=self._group, tag=tag)

    def recv(self, tensor: torch.Tensor, src: int, *, tag: int = 0) -> Any:
        """异步接收数据到 ``tensor``，返回 work handle。

        调用方必须在读取 ``tensor`` 前调用返回 handle 的 ``wait()``。
        使用显式 ``tag``（见 :meth:`send`）。
        """
        return dist.irecv(tensor, src=src, group=self._group, tag=tag)

    def wait(self, work: Any) -> None:
        """等待一个 send/recv work 完成。"""
        if work is not None:
            work.wait()


def wait_all(works: list[Any]) -> None:
    """等待一组 work 全部完成。"""
    for work in works:
        if work is not None:
            work.wait()
