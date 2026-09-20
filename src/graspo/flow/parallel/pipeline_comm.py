"""Async P2P pipeline communication (isend/irecv) — 设施层。

每个 pipeline stage 通过两个**方向独立的**异步点对点通道与前后 stage 通信：

- **forward 通道**（hidden 激活，自上游流到下游）：fwd_recv / fwd_send
- **backward 通道**（梯度，自下游流到上游）：bwd_recv / bwd_send

为什么拆成两个进程组（fwd_group / bwd_group）：
1F1B 调度把 forward 与 backward 交错到同一 peer-pair 上。若两者共用同一进程组，
NCCL 在同个 peer-pair 上是**单 FIFO 配对**的；forward-hidden（tag=chunk_idx）与
backward-grad（tag=chunk_count+chunk_idx）按不同速率、不同方向出现，时序错配即死锁。
GPipe 各方向"一整段递增"所以能对齐。拆成独立进程组 → 各得一条独立 NCCL P2P channel，
从机制上隔离（§2.1 契约即防呆）。

为什么用专用 CUDA stream + 事件：让通信与计算**重叠**——recv 在通信流上完成、计算流
显式等待其事件；send 通信流等待计算流事件后再投递。重叠是 1F1B 气泡收益的前提。
事件同步**必须显式**（record_event / wait_event），绝不依赖默认流上的隐式顺序，
否则会重演 v0.27.4 "PP=4 卡死"（§5 先正确后优化）。

背压：有界在途 send work（``max_inflight``），防止未消费的中间激活/梯度撑爆显存，
也防止 1F1B 交错下超窗死锁。``max_inflight<=0`` 表示无界（单测/调试用），>0 为窗口上限。

用法::

    comm = PipelineComm(
        device=device,
        fwd_group=tp_state.pp_group_fwd,
        bwd_group=tp_state.pp_group_bwd,
        max_inflight=config.pp_max_inflight_microbatches,
    )
    recv = comm.fwd_recv(tensor, src=prev_rank, tag=chunk_idx)
    comm.wait(recv)                 # 计算流等待 recv 完成后再读 tensor
    send = comm.fwd_send(out, dst=next_rank, tag=chunk_idx)
    wait_all(send_works)            # 管道结束时同步所有异步发送
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.distributed as dist


class PipelineComm:
    """Pipeline 异步 P2P 通信封装（双向通道 + CUDA stream 重叠 + 有界背压）。

    :param device: 通信涉及的 CUDA device（CPU 环境可为 ``cpu``）。
    :param group: 兼容单通道布局的进程组（当 fwd_group/bwd_group 缺省时用于两个方向）。
    :param fwd_group: forward-hidden 通道的进程组（通常为 ``pp_group_fwd``）。
    :param bwd_group: backward-grad 通道的进程组（通常为 ``pp_group_bwd``）。
    :param max_inflight: 每个方向的有界在途 send 数；<=0 表示无界。
    :param chunk_count: 本 optimizer step 的 microbatch 数。>0 时启用 tag 区间校验
        （forward tag∈[0,chunk_count)，backward tag∈[chunk_count,2*chunk_count)）。
    """

    def __init__(
        self,
        *,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
        fwd_group: dist.ProcessGroup | None = None,
        bwd_group: dist.ProcessGroup | None = None,
        max_inflight: int = 0,
        chunk_count: int = 0,
    ) -> None:
        self.device = device
        self._is_cuda = device.type == "cuda"
        self._fwd_group = fwd_group or group
        self._bwd_group = bwd_group or group
        # 每个方向一条独立 CUDA stream，用于通信/计算重叠。CPU/gloo 退化为当前流。
        self._fwd_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device) if self._is_cuda else None
        )
        self._bwd_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device) if self._is_cuda else None
        )
        self._max_inflight = int(max_inflight or 0)
        self._chunk_count = int(chunk_count or 0)
        # 每方向的在途 send work（用于背压：超过窗口则等待最旧 work）。
        self._inflight: dict[str, deque[Any]] = {"fwd": deque(), "bwd": deque()}

    # ── 背压 ────────────────────────────────────────────────────────────────
    def _acquire_slot(self, direction: str) -> None:
        """背压：若在途 send 已达窗口上限，等待最旧一个完成（释放 buffer）。"""
        if self._max_inflight <= 0:
            return
        queue = self._inflight[direction]
        if len(queue) > self._max_inflight:
            oldest = queue.popleft()
            if oldest is not None:
                self.wait(oldest)

    def _release_slot(self, direction: str, work: Any) -> None:
        if work is not None:
            self._inflight[direction].append(work)
            self._acquire_slot(direction)

    # ── 事件/流同步 ─────────────────────────────────────────────────────────
    def _wait_for_stream(self, comm_stream: torch.cuda.Stream | None, ev: torch.cuda.Event) -> None:
        """让通信流等待计算流记录的事件（send 前：等计算完成）。"""
        if comm_stream is not None:
            comm_stream.wait_event(ev)

    # ── 方向化 send/recv ────────────────────────────────────────────────────
    def _validate_tag(self, direction: str, tag: int) -> None:
        """边界校验（§2.3）：tag 区间须与方向一致，否则提前报错而非 NCCL 静默挂起。"""
        if self._chunk_count <= 0:
            return
        lo, hi = (
            (0, self._chunk_count)
            if direction == "fwd"
            else (
                self._chunk_count,
                2 * self._chunk_count,
            )
        )
        if not (lo <= tag < hi):
            raise RuntimeError(
                f"PipelineComm {direction} tag out of range: tag={tag} "
                f"expected [{lo}, {hi}) for chunk_count={self._chunk_count}"
            )

    def _post_send(self, tensor: torch.Tensor, dst: int, *, tag: int, direction: str) -> Any:
        self._validate_tag(direction, tag)
        tensor = tensor.contiguous()
        group = self._fwd_group if direction == "fwd" else self._bwd_group
        comm_stream = self._fwd_stream if direction == "fwd" else self._bwd_stream
        if comm_stream is not None:
            # 通信流等待计算流产生该 tensor 的事件，避免发送未算完的数据。
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            self._wait_for_stream(comm_stream, ev)
            with torch.cuda.stream(comm_stream):
                work = dist.isend(tensor, dst=dst, group=group, tag=tag)
        else:
            work = dist.isend(tensor, dst=dst, group=group, tag=tag)
        self._release_slot(direction, work)
        return work

    def _post_recv(self, tensor: torch.Tensor, src: int, *, tag: int, direction: str) -> Any:
        self._validate_tag(direction, tag)
        group = self._fwd_group if direction == "fwd" else self._bwd_group
        comm_stream = self._fwd_stream if direction == "fwd" else self._bwd_stream
        if comm_stream is not None:
            ev = torch.cuda.Event()
            with torch.cuda.stream(comm_stream):
                work = dist.irecv(tensor, src=src, group=group, tag=tag)
                ev.record(comm_stream)
            # 先返回一个"待同步"句柄；wait() 会让当前计算流等待该事件。
            return _RecvHandle(work, comm_stream, ev, self)
        work = dist.irecv(tensor, src=src, group=group, tag=tag)
        return _RecvHandle(work, None, None, self)

    def fwd_send(self, tensor: torch.Tensor, dst: int, *, tag: int = 0) -> Any:
        """异步发送 hidden 到下游 stage（forward 通道）。返回 work handle，需 wait()。"""
        return self._post_send(tensor, dst, tag=tag, direction="fwd")

    def fwd_recv(self, tensor: torch.Tensor, src: int, *, tag: int = 0) -> _RecvHandle:
        """异步接收 hidden（forward 通道）。返回 ``_RecvHandle``，调用 ``wait()`` 后读 tensor。"""
        return self._post_recv(tensor, src, tag=tag, direction="fwd")

    def bwd_send(self, tensor: torch.Tensor, dst: int, *, tag: int = 0) -> Any:
        """异步发送 grad 到上游 stage（backward 通道）。返回 work handle，需 wait()。"""
        return self._post_send(tensor, dst, tag=tag, direction="bwd")

    def bwd_recv(self, tensor: torch.Tensor, src: int, *, tag: int = 0) -> _RecvHandle:
        """异步接收 grad（backward 通道）。返回 ``_RecvHandle``，调用 ``wait()`` 后读 tensor。"""
        return self._post_recv(tensor, src, tag=tag, direction="bwd")

    def wait(self, work: Any) -> None:
        """等待一个 send/recv work 完成（send 用 CPU 同步；recv 用事件让计算流等待）。"""
        if work is None:
            return
        if isinstance(work, _RecvHandle):
            work.wait()
        else:
            work.wait()

    def wait_all(self, works: list[Any]) -> None:
        """等待一组 work 全部完成。"""
        for work in works:
            self.wait(work)


class _RecvHandle:
    """异步 recv 的句柄：封装底层 ``irecv`` work + 通信流事件同步。

    ``wait()`` 让当前（计算）流等待通信流上 recv 完成的事件，从而后续读 tensor
    是安全的；若在 CPU/gloo（无 CUDA stream），则退化为底层 ``work.wait()``。
    """

    __slots__ = ("_work", "_stream", "_ev", "_comm")

    def __init__(
        self,
        work: Any,
        stream: torch.cuda.Stream | None,
        ev: torch.cuda.Event | None,
        comm: PipelineComm,
    ) -> None:
        self._work = work
        self._stream = stream
        self._ev = ev
        self._comm = comm

    def wait(self) -> None:
        if self._stream is not None and self._ev is not None:
            # 计算流等待 recv 完成事件，再读 tensor（事件 = 数据就绪）
            torch.cuda.current_stream().wait_event(self._ev)
        else:
            self._work.wait()


def wait_all(works: list[Any]) -> None:
    """等待一组 work 全部完成（兼容旧的 ``dist`` work 列表）。"""
    for work in works:
        if work is not None:
            work.wait()
