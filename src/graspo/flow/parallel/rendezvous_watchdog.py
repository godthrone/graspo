"""PP 会合点看门狗 — 把"无界挂死"变成"有界失败 + 线程栈取证"（设施层）。

**为什么需要它（缺陷 P6，2026-09-22 在 228 实测）**

native + PP 的 rollout 在**首个 prefill 之后**无界挂死：两端 rank 都停在会合点上，
两卡 GPU util 恒 0%，而**>18 min 没有任何 NCCL watchdog 超时**。原因是卡点落在
NCCL watchdog 的**结构性盲区**里：

* NCCL watchdog 只对"**已入队、未完成**"的 work 计时；
* 当阻塞发生在 ``cudaStreamWaitEvent``（CUDA 流/事件依赖）或"下一次会合尚未入队"
  的 host 侧等待上时，**队列里根本没有可计时的 work** ⇒ 永不超时。

所以"有界失败"必须由**通信层之外的第三只眼**提供。本模块是那一只眼，只做两件事：

1. **心跳**（:func:`beat`）：主流程每跨过一个会合点就 beat 一次（由
   ``pipeline_forward`` / ``generation`` 的会合点探针调用，探针本身零行为改动）；
2. **看门狗线程**：armed 期间若连续 ``no_progress_sec`` 秒无 beat ⇒ 判定停滞，
   触发 :meth:`RendezvousWatchdog._default_on_stall`：
   ``faulthandler`` 全线程栈落盘 → ``threading.interrupt_main()``（让 Python 侧
   阻塞有机会走既有的 ``except BaseException`` 收尾路径）→ 宽限期后
   ``os._exit(EXIT_CODE_NO_PROGRESS)``（**保证有界、fail-closed**）。

另提供 :func:`install_sigusr1_handler`：人工现场取证（容器内
``kill -USR1 <pid>`` 即落一份全线程栈，含 C 栈）。

**边界（§1.1 设施层 / §1.3 计算与设施分离）**

* 本模块**不 import torch**，纯 stdlib ⇒ 可在无 GPU、无 torch 的机器上单测；
* 本模块**不改变任何控制流**：只有 heartbeat 记录、日志落盘与"停滞时才介入"的动作；
* 数值/训练语义零影响（不碰张量、不碰进程组）。

**单一真相源（§1.4）**

* 触发阈值来自配置 ``native.pp_rollout_no_progress_sec``（唯一的取值来源，见
  :func:`configure`）；本模块内**不出现**第二个默认阈值字面量；
* 退出码只有 :data:`EXIT_CODE_NO_PROGRESS` 一个定义点；
* 进程内只有一个心跳实例（:data:`PP_RENDEZVOUS`），不在别处复制。
"""

from __future__ import annotations

import _thread
import faulthandler
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: 停滞看门狗触发后的进程退出码（**唯一真相源**）。
#: 取值远离 0/1/2/126/127/128+n（shell 语义）与 137（SIGKILL/`docker stop`），
#: 便于 runner、`collect_results.py` 与人工判读一眼区分"看门狗判定停滞"与其它失败。
EXIT_CODE_NO_PROGRESS = 86

#: 判定停滞后的宽限期（秒）：先 ``interrupt_main()``，给 Python 侧异常收尾留时间；
#: 宽限期内进程自行退出就不走 ``os._exit``。
STALL_GRACE_SEC = 10.0

#: 看门狗轮询间隔上限（秒）：即使阈值很大也不至于睡着不醒。
WATCHDOG_MAX_POLL_SEC = 1.0

#: 落盘文件名（相对 ``{output_dir}/logs/<run_id>/``），唯一真相源。
DUMP_FILENAME = "pp_rendezvous_watchdog.txt"


def _dump_all_threads(path: Path | None, banner: str) -> None:
    """把 banner + 全线程栈（含 C 栈）同时写到 stderr 与 ``path``。

    任一路径失败都**不得**吞掉/替换调用方正在处理的首错——与
    ``flow/trainer/trainer.py:_abort_distributed`` 同一口径（§3.4）。
    """
    handle = None
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
        except OSError as exc:  # 落盘失败只告警，不打断取证
            print(f"[pp-watchdog] dump file open failed: {exc}", file=sys.stderr, flush=True)
    targets: list[Any] = [sys.stderr] if handle is None else [sys.stderr, handle]
    try:
        for target in targets:
            try:
                # banner 必须进**每一个** target：落盘文件要能自述（谁、何时、为什么 dump），
                # 否则事后只看到一堆无标题的调用栈。
                print(banner, file=target, flush=True)
                faulthandler.dump_traceback(file=target, all_threads=True)
            except (OSError, ValueError) as exc:
                print(f"[pp-watchdog] dump_traceback failed: {exc}", file=sys.stderr, flush=True)
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError as exc:
                print(f"[pp-watchdog] dump file close failed: {exc}", file=sys.stderr, flush=True)


class RendezvousWatchdog:
    """PP 会合点心跳 + 停滞看门狗。

    :param no_progress_sec: 连续多少秒无 beat 判为停滞（必须 > 0）。
    :param dump_path: 全线程栈落盘路径；``None`` 表示只写 stderr。
    :param on_stall: 停滞回调（收到人类可读的 reason）；**测试用注入点**，
        默认实现为 :meth:`_default_on_stall`（转储 + interrupt + 有界退出）。
    :param clock: 单调时钟注入点（测试用）。
    :param sleeper: 休眠注入点（测试用）。
    """

    def __init__(
        self,
        *,
        no_progress_sec: float,
        dump_path: Path | None = None,
        on_stall: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        grace_sec: float = STALL_GRACE_SEC,
    ) -> None:
        if not (float(no_progress_sec) > 0.0):
            raise ValueError(
                f"RendezvousWatchdog.no_progress_sec must be > 0, got {no_progress_sec!r}"
            )
        self._no_progress_sec = float(no_progress_sec)
        self._grace_sec = float(grace_sec)
        self._dump_path = dump_path
        self._clock = clock
        self._sleeper = sleeper
        self._on_stall = on_stall
        self._lock = threading.Lock()
        self._last_beat: float | None = None
        self._last_label = "<none>"
        self._armed = False
        self._fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── 心跳 ────────────────────────────────────────────────────────────────
    def beat(self, label: str) -> None:
        """记录一次会合点进展（label 用于停滞时的定位）。"""
        with self._lock:
            self._last_beat = self._clock()
            self._last_label = str(label)

    @property
    def last_label(self) -> str:
        with self._lock:
            return self._last_label

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    # ── 判据（纯逻辑，可单测）──────────────────────────────────────────────
    def stall_reason(self, now: float | None = None) -> str | None:
        """未停滞返回 ``None``；停滞返回人类可读的 reason（**不做任何副作用**）。"""
        with self._lock:
            if not self._armed:
                return None
            last = self._last_beat
            label = self._last_label
        if last is None:
            return None
        elapsed = (self._clock() if now is None else float(now)) - last
        if elapsed < self._no_progress_sec:
            return None
        return (
            f"PP rendezvous made no progress for {elapsed:.1f}s "
            f"(limit {self._no_progress_sec:.1f}s, last rendezvous={label}); "
            "no NCCL watchdog can see this class of stall (CUDA stream/event or "
            "not-yet-enqueued rendezvous)"
        )

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def arm(self, *, label: str = "pp_rollout") -> None:
        """开始监视（幂等）：重置心跳并启动看门狗线程。"""
        with self._lock:
            if self._armed:
                return
            self._armed = True
            self._last_beat = self._clock()
            self._last_label = label
            self._fired = False
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="pp-rendezvous-watchdog", daemon=True)
        self._thread = thread
        thread.start()

    def disarm(self) -> None:
        """停止监视（幂等）：看门狗线程退出，不留后台线程。"""
        with self._lock:
            self._armed = False
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, WATCHDOG_MAX_POLL_SEC * 2))
        self._thread = None

    def _run(self) -> None:
        interval = min(WATCHDOG_MAX_POLL_SEC, max(0.01, self._no_progress_sec / 4.0))
        while not self._stop.wait(interval):
            reason = self.stall_reason()
            if reason is None:
                continue
            with self._lock:
                if self._fired:
                    return
                self._fired = True
            handler = self._on_stall or self._default_on_stall
            handler(reason)
            return

    # ── 停滞动作 ────────────────────────────────────────────────────────────
    def _default_on_stall(self, reason: str) -> None:
        banner = f"[pp-watchdog] STALL DETECTED: {reason}"
        _dump_all_threads(self._dump_path, banner)
        # 让 Python 侧的阻塞有机会按既有异常路径收尾（trainer 的
        # `except BaseException` → 非阻塞 abort → 原样 raise 首错）。
        try:
            _thread.interrupt_main()
        except (RuntimeError, ValueError) as exc:
            print(f"[pp-watchdog] interrupt_main failed: {exc}", file=sys.stderr, flush=True)
        self._sleeper(self._grace_sec)
        # 宽限期内进程没退 ⇒ 强制有界退出（fail-closed；非零码，绝不静默继续）。
        print(
            f"[pp-watchdog] force-exit with code {EXIT_CODE_NO_PROGRESS} "
            "(no progress within grace period)",
            file=sys.stderr,
            flush=True,
        )
        sys.stderr.flush()
        os._exit(EXIT_CODE_NO_PROGRESS)


# ── 进程内唯一入口（§1.4）───────────────────────────────────────────────────
_WATCHDOG: RendezvousWatchdog | None = None
_CONFIGURED_SEC: float | None = None
_HANDLERS_INSTALLED = False
_GLOBAL_LOCK = threading.Lock()


def configure(
    *,
    no_progress_sec: float,
    dump_path: Path | None = None,
    on_stall: Callable[[str], None] | None = None,
) -> RendezvousWatchdog:
    """进程内唯一配置点（幂等）。

    值来自 ``native.pp_rollout_no_progress_sec``（§1.4：阈值只有这一个来源）。
    重复调用必须给同一个阈值；不一致直接报错，避免"两处阈值"的静默分叉。
    """
    global _WATCHDOG, _CONFIGURED_SEC  # noqa: PLW0603 — 进程级单例
    with _GLOBAL_LOCK:
        if _WATCHDOG is None:
            _WATCHDOG = RendezvousWatchdog(
                no_progress_sec=no_progress_sec, dump_path=dump_path, on_stall=on_stall
            )
            _CONFIGURED_SEC = float(no_progress_sec)
        elif _CONFIGURED_SEC != float(no_progress_sec):
            raise RuntimeError(
                "RendezvousWatchdog already configured with "
                f"no_progress_sec={_CONFIGURED_SEC}, cannot reconfigure to "
                f"{float(no_progress_sec)} (single source of truth, §1.4)"
            )
        else:
            if dump_path is not None:
                _WATCHDOG._dump_path = dump_path  # noqa: SLF001 — 同一模块的单例配置
        return _WATCHDOG


def watchdog() -> RendezvousWatchdog | None:
    """返回已配置的看门狗（未配置则 ``None``）。"""
    return _WATCHDOG


def beat(label: str) -> None:
    """会合点心跳（未配置时是 no-op，保证探针零依赖、零行为改动）。"""
    wd = _WATCHDOG
    if wd is not None:
        wd.beat(label)


def arm(*, label: str = "pp_rollout") -> None:
    wd = _WATCHDOG
    if wd is not None:
        wd.arm(label=label)


def disarm() -> None:
    wd = _WATCHDOG
    if wd is not None:
        wd.disarm()


def install_sigusr1_handler(dump_path: Path | None = None) -> None:
    """注册 ``SIGUSR1`` 现场取证处理器（幂等、只在主线程生效）。

    容器内取栈：``docker exec <container> kill -USR1 <pid>``。处理器只写全线程栈，
    **不改变进程状态**（不作为控制手段，只作取证手段）。
    """

    global _HANDLERS_INSTALLED  # noqa: PLW0603 — 进程级单例
    with _GLOBAL_LOCK:
        if _HANDLERS_INSTALLED:
            return
        resolved = dump_path

        def _handler(_signum: int, _frame: Any) -> None:
            _dump_all_threads(
                resolved,
                "[pp-watchdog] SIGUSR1 received — dumping all thread stacks (on-demand forensics)",
            )

        try:
            previous = signal.signal(signal.SIGUSR1, _handler)
        except (ValueError, OSError, AttributeError) as exc:
            # 非主线程 / 平台不支持：如实告警，不静默（§2.2）
            print(
                f"[pp-watchdog] SIGUSR1 handler not installed ({exc}); "
                "on-demand thread-stack forensics unavailable",
                file=sys.stderr,
                flush=True,
            )
            return
        print(
            f"[pp-watchdog] SIGUSR1 handler installed "
            f"(previous={previous!r}); `kill -USR1 <pid>` dumps all thread stacks",
            file=sys.stderr,
            flush=True,
        )
        _HANDLERS_INSTALLED = True


#: 进程内唯一的会合点心跳实例即模块级 ``_WATCHDOG``：探针只经 :func:`beat` 上报，
#: 不各自记时间戳（§1.4：会合点进展只有一个真相源）。
def configure_for_run(*, output_dir: str | Path, no_progress_sec: float) -> RendezvousWatchdog:
    """按 run 的输出目录配置看门狗并装好 SIGUSR1（PP rollout 入口唯一调用点）。

    落盘路径 = ``{output_dir}/logs/<run_id>/pp_rendezvous_watchdog.txt``。
    与 ``pipeline_forward._pp_debug_log`` 用同一个 ``run_log_dir``（§1.4），
    因此取证文件与 ``pp_debug.log`` 永远在同一个目录里。
    """
    from graspo.flow.logging import run_log_dir

    dump_path = run_log_dir(output_dir) / DUMP_FILENAME
    wd = configure(no_progress_sec=no_progress_sec, dump_path=dump_path)
    install_sigusr1_handler(dump_path)
    return wd
