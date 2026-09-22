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

**有界等待（a1，缺陷 P6）**：``wait()`` / ``wait_all()`` 支持 ``timedelta`` 级别的
超时（torch 2.11 原生 ``Work.wait(timeout)`` 语义）。**未配置超时（``None``/``<=0``）
时行为与修复前逐字一致**（``work.wait()`` 不带参），因此既有 pp>1 档位在健康运行下
不受影响；配置了超时后，超时会抛具名 :class:`PipelineP2PTimeoutError`，把
"永久挂死"变成"有界失败 + 明确报错"。

**★ 作用域限制（2026-09-22 指挥官裁定 3，务必遵守）**：有界等待**只准用于
PP rollout / 生成路径**（唯一消费点 ``generation.py``，键
``native.pp_rollout_p2p_timeout_sec``）。**禁止**用于 1F1B 训练热路径
（``training.py`` / ``training_sft.py`` / ``logprobs.py`` 的 ``wait_all``）：
torch 的 ``Work.wait(timeout)`` 在**设了 timeout 时会阻塞 CPU 线程**（官方文档原文
"if timeout is set, it will block the CPU thread…"），放到热路径上就是每步末尾多一次
CPU 同步、损失跨 step 重叠——用健康路径上永不触发的超时换热路径性能（§18 留债）。
训练路径的有界性由 **NCCL 自己的 watchdog** 提供（它的等待对象是"已入队 work"；
而 P6 的无界形态是 NCCL 看不见的流/事件依赖 + 尚未入队的会合，只出现在 rollout 的
紧会合序列里）。该作用域由 ``tests/flow/parallel/test_pp_bounded_wait_scope.py``
的接线守卫钉住。

**为什么单靠超时不够（必须配合会合点看门狗）**：若阻塞发生在 CUDA 流/事件依赖
（``cudaStreamWaitEvent``）或"下一次会合尚未入队"的 host 侧等待上，本层根本没有
在飞 work 可等 ⇒ 超时也无从触发。这一支由
``graspo.flow.parallel.rendezvous_watchdog`` 兜底。

**已知契约事实（§2.2 显式即防呆）**：``dist.isend/irecv`` 的 ``tag`` **在 NCCL
后端不生效**（PyTorch 文档原文 ``tag is not supported with the NCCL backend``）。
因此 fwd/bwd 的隔离**只**来自两个独立进程组；P2P 配对实际按"同 peer-pair 的入队
顺序"。:meth:`PipelineComm._validate_tag` 只保证"调用方传的 tag 与方向自洽"，
**不是**顺序配对的保护。

用法::

    comm = PipelineComm(
        device=device,
        fwd_group=tp_state.pp_group_fwd,
        bwd_group=tp_state.pp_group_bwd,
        max_inflight=config.pp_max_inflight_microbatches,
        wait_timeout_s=native.pp_rollout_p2p_timeout_sec,   # 0/None ⇒ 不设超时（旧行为）
    )
    recv = comm.fwd_recv(tensor, src=prev_rank, tag=chunk_idx)
    comm.wait(recv, label="prefill.recv")   # 计算流等待 recv 完成后再读 tensor
    send = comm.fwd_send(out, dst=next_rank, tag=chunk_idx)
    comm.wait_all(send_works)            # 管道结束时同步所有异步发送
"""

from __future__ import annotations

import time
from collections import deque
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


class PipelineP2PTimeoutError(RuntimeError):
    """PP 会合点超时（**具名失败类别**，供 runner / 判据侧识别）。

    覆盖两类会合点：

    * P2P（``fwd_send`` / ``fwd_recv`` / ``bwd_send`` / ``bwd_recv``）；
    * ``pp_group`` 内的 token 广播（:func:`bounded_broadcast`）。

    为什么必须有具名类型：缺陷 P6 的形态是"永久挂死且 NCCL watchdog 看不见"，
    失败时**没有任何异常**可判读。本类型把该形态收敛成一个可分类、可断言的失败。
    """

    def __init__(
        self,
        *,
        label: str,
        direction: str,
        peer: Any,
        shape: Any,
        timeout_s: float,
        detail: str = "",
    ) -> None:
        self.label = str(label)
        self.direction = str(direction)
        self.peer = peer
        self.shape = tuple(shape) if shape is not None else None
        self.timeout_s = float(timeout_s)
        message = (
            f"PP rendezvous timeout [{self.label}]: {self.direction} "
            f"peer={self.peer} shape={self.shape} exceeded {self.timeout_s:g}s "
            "(bounded wait; this class of stall is invisible to the NCCL "
            "watchdog — see graspo.flow.parallel.rendezvous_watchdog)"
        )
        if detail:
            message = f"{message} | {detail}"
        super().__init__(message)


def pp_rendezvous_timeout(seconds: float | None) -> timedelta | None:
    """``秒 → timedelta`` 的**唯一转换点**（§1.4 单一真相源）。

    ``None`` / ``<= 0`` ⇒ ``None`` = **不设超时**（与修复前逐字一致：``work.wait()``
    不带参）。取值来源是 ``native.pp_rollout_p2p_timeout_sec``；本模块内不出现第二个
    默认阈值字面量。
    """
    if seconds is None:
        return None
    value = float(seconds)
    if value <= 0.0:
        return None
    return timedelta(seconds=value)


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
        wait_timeout_s: float | None = None,
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
        # 有界等待（a1）：None = 不设超时（**与修复前逐字一致**）。唯一转换点
        # 是 pp_rendezvous_timeout（见模块级 docstring 的单一真相源说明）。
        self._wait_timeout: timedelta | None = pp_rendezvous_timeout(wait_timeout_s)
        self._wait_timeout_s: float | None = (
            self._wait_timeout.total_seconds() if self._wait_timeout is not None else None
        )
        # 每方向的在途 send work（用于背压：超过窗口则等待最旧 work）。
        self._inflight: dict[str, deque[Any]] = {"fwd": deque(), "bwd": deque()}
        # 在途 work 的描述信息（方向/对端/形状），用于超时报错给出可定位的上下文。
        # 键是 id(work)（work 在途期间必然被引用，不存在 id 复用）。
        self._inflight_info: dict[int, dict[str, Any]] = {}

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
        """边界校验（§2.3）：调用方传的 tag 必须与该方向自洽，否则提前报错。

        **这不是顺序配对的保护（§2.2 显式即防呆）**：``dist.isend/irecv`` 的
        ``tag`` 在 **NCCL 后端不生效**（PyTorch 文档原文
        ``tag is not supported with the NCCL backend``）。NCCL 的 P2P 配对按
        "同 peer-pair 的**入队顺序**"，所以 fwd/bwd 的隔离**只**来自
        ``pp_group_fwd`` / ``pp_group_bwd`` 两个独立进程组。本校验的作用收窄为：
        把"调用方自己把 tag 传串了"这类**代码错误**尽早暴露，避免它被静默带进
        一个看起来正常的运行。历史注释里"tag 错配会提前报错"的说法已按事实纠正。
        """
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
        self._remember(
            work,
            direction=f"{direction}_send",
            peer=f"dst={dst}",
            shape=tuple(tensor.shape),
        )
        self._release_slot(direction, work)
        return work

    def _post_recv(self, tensor: torch.Tensor, src: int, *, tag: int, direction: str) -> Any:
        self._validate_tag(direction, tag)
        group = self._fwd_group if direction == "fwd" else self._bwd_group
        comm_stream = self._fwd_stream if direction == "fwd" else self._bwd_stream
        info = {
            "direction": f"{direction}_recv",
            "peer": f"src={src}",
            "shape": tuple(tensor.shape),
        }
        if comm_stream is not None:
            ev = torch.cuda.Event()
            with torch.cuda.stream(comm_stream):
                work = dist.irecv(tensor, src=src, group=group, tag=tag)
                ev.record(comm_stream)
            # 先返回一个"待同步"句柄；wait() 会让当前计算流等待该事件。
            return _RecvHandle(work, comm_stream, ev, self, info=info)
        work = dist.irecv(tensor, src=src, group=group, tag=tag)
        return _RecvHandle(work, None, None, self, info=info)

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

    def wait(self, work: Any, *, label: str = "") -> None:
        """等待一个 send/recv work 完成。

        * 未配置 ``wait_timeout_s``（``None``/``<=0``）：**与修复前逐字一致** ——
          send 走 ``work.wait()``（让当前流挂完成事件），recv 走计算流等事件；
        * 配置了超时：走 torch 原生的有界 ``Work.wait(timedelta)``；超时抛
          :class:`PipelineP2PTimeoutError`（含 direction / peer / shape），
          绝不静默继续。

        :param label: 会合点标签（由调用方给出，如 ``"pp_rollout.prefill.send"``），
            只用于报错定位，不参与任何控制流。
        """
        if work is None:
            return
        if isinstance(work, _RecvHandle):
            work.wait(timeout=self._wait_timeout, label=label)
        else:
            self._wait_work(work, label=label)

    def wait_all(self, works: list[Any], *, label: str = "") -> None:
        """等待一组 work 全部完成（逐项走 :meth:`wait` 的有界语义）。"""
        for work in works:
            self.wait(work, label=label)

    # ── 有界等待实现 ────────────────────────────────────────────────────────
    def _remember(self, work: Any, *, direction: str, peer: str, shape: Any) -> None:
        """登记在途 work 的上下文（只在报错时使用；成功等待后即清除）。"""
        if work is None:
            return
        self._inflight_info[id(work)] = {
            "direction": direction,
            "peer": peer,
            "shape": shape,
        }

    def _describe(self, work: Any, label: str) -> dict[str, Any]:
        info = self._inflight_info.get(id(work)) or {}
        return {
            "label": label or "pipeline_comm.wait",
            "direction": info.get("direction", "unknown"),
            "peer": info.get("peer", "unknown"),
            "shape": info.get("shape"),
        }

    def _forget(self, work: Any) -> None:
        self._inflight_info.pop(id(work), None)

    def _wait_work(self, work: Any, *, label: str = "") -> None:
        """等待一个底层 work（send 路径；recv 由 :class:`_RecvHandle` 负责）。"""
        if self._wait_timeout is None:
            # 逐字旧行为：不传 timeout（让当前流挂完成事件，保留重叠）。
            work.wait()
            self._forget(work)
            return
        context = self._describe(work, label)
        started_at = time.monotonic()
        try:
            completed = work.wait(self._wait_timeout)
        except Exception as exc:  # noqa: BLE001 — 只在确认是超时时翻译成具名类型
            if _is_timeout(
                exc,
                elapsed=time.monotonic() - started_at,
                timeout_s=self._wait_timeout_s or 0.0,
            ):
                self._forget(work)
                raise PipelineP2PTimeoutError(
                    **context,
                    timeout_s=self._wait_timeout_s or 0.0,
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            raise
        self._forget(work)
        if completed is False:
            raise PipelineP2PTimeoutError(**context, timeout_s=self._wait_timeout_s or 0.0)


class _RecvHandle:
    """异步 recv 的句柄：封装底层 ``irecv`` work + 通信流事件同步。

    ``wait()`` 让当前（计算）流等待通信流上 recv 完成的事件，从而后续读 tensor
    是安全的；若在 CPU/gloo（无 CUDA stream），则退化为底层 ``work.wait()``。

    配置了有界超时时（``timeout`` 非 ``None``），改用 torch 原生的有界
    ``work.wait(timedelta)``（**它同时覆盖"数据就绪"语义**，因此不再额外挂事件），
    超时抛 :class:`PipelineP2PTimeoutError`。
    """

    __slots__ = ("_work", "_stream", "_ev", "_comm", "_info")

    def __init__(
        self,
        work: Any,
        stream: torch.cuda.Stream | None,
        ev: torch.cuda.Event | None,
        comm: PipelineComm,
        *,
        info: dict[str, Any] | None = None,
    ) -> None:
        self._work = work
        self._stream = stream
        self._ev = ev
        self._comm = comm
        self._info = dict(info or {})

    def wait(self, timeout: timedelta | None = None, *, label: str = "") -> None:
        if timeout is not None:
            context: dict[str, Any] = {
                "label": label or "pipeline_comm.recv",
                "direction": self._info.get("direction", "unknown"),
                "peer": self._info.get("peer", "unknown"),
                "shape": self._info.get("shape"),
            }
            timeout_s = timeout.total_seconds()
            started_at = time.monotonic()
            try:
                completed = self._work.wait(timeout)
            except Exception as exc:  # noqa: BLE001 — 只在确认是超时时翻译成具名类型
                if _is_timeout(exc, elapsed=time.monotonic() - started_at, timeout_s=timeout_s):
                    raise PipelineP2PTimeoutError(
                        **context, timeout_s=timeout_s, detail=f"{type(exc).__name__}: {exc}"
                    ) from exc
                raise
            self._comm._forget(self._work)  # noqa: SLF001 — 同一模块内的私有协作
            if completed is False:
                raise PipelineP2PTimeoutError(**context, timeout_s=timeout_s)
            return
        if self._stream is not None and self._ev is not None:
            # 计算流等待 recv 完成事件，再读 tensor（事件 = 数据就绪）
            torch.cuda.current_stream().wait_event(self._ev)
        else:
            self._work.wait()
        self._comm._forget(self._work)  # noqa: SLF001 — 同一模块内的私有协作


#: 超时异常消息里会出现的**特征短语**（torch/NCCL 的原文措辞，唯一真相源）。
#: 注意不能用裸 "timeout" 子串做判据：`"... not a timeout"` 这类无关错误会被误判
#: （本仓的判别力测试 `test_non_timeout_exception_is_not_masked` 钉住这一点）。
_TIMEOUT_PHRASES = (
    "timed out",
    "timing out",
    "timeout expired",
    "ran for",
    "exceeded timeout",
)


def _is_timeout(exc: BaseException, *, elapsed: float, timeout_s: float) -> bool:
    """判定一个异常是否"有界等待超时"。

    两条判据（任一命中即算）：

    1. **特征短语**：消息含 :data:`_TIMEOUT_PHRASES` 之一（torch/NCCL 的措辞）；
    2. **墙钟兜底**：异常抛出时已经等了 ``>= 0.9 * timeout_s``。

    为什么需要兜底：不同后端（gloo / NCCL）与不同 torch 版本的超时消息不统一，
    但"等了整整一个 timeout 才抛错"这一事实与措辞无关。两条都不命中时**原样
    重抛**，绝不把无关错误包装成超时（§3.4：不替换首错）。
    """
    text = str(exc).lower()
    if any(phrase in text for phrase in _TIMEOUT_PHRASES):
        return True
    return timeout_s > 0 and elapsed >= 0.9 * timeout_s


def wait_all(works: list[Any]) -> None:
    """等待一组 work 全部完成。

    兼容旧调用（模块级、无 ``PipelineComm`` 上下文 ⇒ 无超时语义）。**新代码应优先
    走** :meth:`PipelineComm.wait_all`，才能拿到 ``native.pp_rollout_p2p_timeout_sec`` 的
    有界语义（§1.4：有界性是设施层的单一契约）。
    """
    for work in works:
        if work is not None:
            work.wait()


def bounded_broadcast(
    tensor: torch.Tensor,
    *,
    src: int,
    group: dist.ProcessGroup,
    timeout_s: float | None,
    label: str,
) -> None:
    """``group`` 内广播的**有界**版本（a2，缺陷 P6 的 C1 候选会合点）。

    ``timeout_s`` 为 ``None``/``<=0`` 时退化为逐字旧行为（同步 ``dist.broadcast``）；
    否则用 ``async_op=True`` + 有界 ``wait``，超时抛 :class:`PipelineP2PTimeoutError`。

    为什么需要：``dist.broadcast`` 是同步集合，**没有 API 级超时**；PP rollout 的
    首个 decode step 正是本进程**第一次使用 ``pp_group``** 的集合，实测挂死即在
    这一带，而同步集合的 host 侧阻塞对 NCCL watchdog 不可见。
    """
    resolved = pp_rendezvous_timeout(timeout_s)
    if resolved is None:
        dist.broadcast(tensor, src=src, group=group)
        return
    handle = dist.broadcast(tensor, src=src, group=group, async_op=True)
    context: dict[str, Any] = {
        "label": label or "pp_group.broadcast",
        "direction": "pp_group_broadcast",
        "peer": f"src={src}",
        "shape": tuple(tensor.shape),
    }
    timeout_seconds = resolved.total_seconds()
    started_at = time.monotonic()
    try:
        completed = handle.wait(resolved)
    except Exception as exc:  # noqa: BLE001 — 只在确认是超时时翻译成具名类型
        if _is_timeout(exc, elapsed=time.monotonic() - started_at, timeout_s=timeout_seconds):
            raise PipelineP2PTimeoutError(
                **context,
                timeout_s=timeout_seconds,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        raise
    if completed is False:
        raise PipelineP2PTimeoutError(**context, timeout_s=timeout_seconds)
