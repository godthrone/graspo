"""``rendezvous_watchdog`` 单元测试 — 纯 stdlib 逻辑（不触 GPU、不起训练）。

覆盖（缺陷 P6 的 a4）：
1. 判据纯逻辑：未 arm / 未到阈值 ⇒ 不停滞；beat 会重置截止时间；
2. 看门狗线程：停滞时**只触发一次**，arm/disarm 幂等且不留后台线程；
3. 取证落盘：全线程栈写入 ``{output_dir}/logs/<run_id>/pp_rendezvous_watchdog.txt``；
4. SIGUSR1：人工现场取证（在**子进程**里验证，避免污染 pytest 的 signal 状态）；
5. 配置的单一真相源：重复 configure 同值幂等、异值直接报错。

注：``graspo`` 包的 ``__init__`` 间接 import torch，所以本文件在无 torch 的环境
整体跳过（被测模块本身不用 torch）。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

try:
    from graspo.flow.parallel import rendezvous_watchdog as wd
except ModuleNotFoundError as exc:  # pragma: no cover — 无 torch 的本机环境
    pytest.skip(
        f"rendezvous_watchdog 经 graspo 包 __init__ 间接依赖 torch（{exc}）",
        allow_module_level=True,
    )


class _Clock:
    """可控单调时钟（测试用）。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _watchdog(**kwargs):  # noqa: ANN001, ANN202
    clock = kwargs.pop("clock", None) or _Clock()
    fired: list[str] = []
    dog = wd.RendezvousWatchdog(
        no_progress_sec=kwargs.pop("no_progress_sec", 10.0),
        clock=clock,
        sleeper=kwargs.pop("sleeper", lambda _s: None),
        on_stall=kwargs.pop("on_stall", fired.append),
        **kwargs,
    )
    return dog, clock, fired


# ── 判据（纯逻辑）───────────────────────────────────────────────────────────


def test_unarmed_watchdog_never_reports_a_stall() -> None:
    dog, clock, _ = _watchdog()
    clock.advance(10_000)
    assert dog.stall_reason() is None


def test_beat_resets_the_deadline() -> None:
    dog, clock, _ = _watchdog(no_progress_sec=10.0)
    dog.arm()
    try:
        clock.advance(9.0)
        assert dog.stall_reason() is None
        dog.beat("recv_ready")
        clock.advance(9.0)
        assert dog.stall_reason() is None, "beat 之后必须重新计时"
        clock.advance(2.0)
        reason = dog.stall_reason()
        assert reason is not None
        assert "recv_ready" in reason, "停滞原因必须带最后一个会合点标签"
        assert "no progress for" in reason
    finally:
        dog.disarm()


def test_stall_reason_names_the_nccl_blind_spot() -> None:
    """报告要能让读者一眼知道"为什么 NCCL watchdog 看不见"（P6 的核心事实）。"""
    dog, clock, _ = _watchdog(no_progress_sec=5.0)
    dog.arm(label="pp_rollout.start")
    clock.advance(6.0)
    reason = dog.stall_reason() or ""
    assert "NCCL watchdog" in reason
    dog.disarm()


def test_invalid_threshold_is_rejected() -> None:
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="no_progress_sec"):
            wd.RendezvousWatchdog(no_progress_sec=bad)


# ── 看门狗线程 ──────────────────────────────────────────────────────────────


def test_thread_fires_once_on_stall_and_disarm_stops_it() -> None:
    fired: list[str] = []
    dog = wd.RendezvousWatchdog(no_progress_sec=0.2, on_stall=fired.append)
    dog.arm()
    deadline = time.monotonic() + 5.0
    while not fired and time.monotonic() < deadline:
        time.sleep(0.02)
    dog.disarm()
    assert fired, "停滞必须触发回调"
    assert len(fired) == 1, "同一次停滞只允许触发一次"
    assert dog.armed is False
    time.sleep(0.3)  # 给"本该已经退出的线程"一点时间再断言
    assert dog._thread is None  # noqa: SLF001 — 断言不留后台线程


def test_beat_keeps_a_long_running_watchdog_alive() -> None:
    fired: list[str] = []
    dog = wd.RendezvousWatchdog(no_progress_sec=0.3, on_stall=fired.append)
    dog.arm()
    try:
        for _ in range(10):
            time.sleep(0.05)
            dog.beat("steady_progress")
        assert fired == [], "持续推进时不允许误报停滞"
    finally:
        dog.disarm()


def test_arm_and_disarm_are_idempotent() -> None:
    dog, _, _ = _watchdog()
    dog.arm()
    dog.arm()
    assert dog.armed is True
    dog.disarm()
    dog.disarm()
    assert dog.armed is False


# ── 取证落盘 ────────────────────────────────────────────────────────────────


def test_dump_all_threads_writes_stacks_to_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dump.txt"
    wd._dump_all_threads(target, "[pp-watchdog] TEST BANNER")  # noqa: SLF001
    text = target.read_text(encoding="utf-8")
    assert "[pp-watchdog] TEST BANNER" in text
    # faulthandler 的头两行形态：`Current thread 0x... (most recent call first):`
    assert "Current thread" in text
    assert "most recent call first" in text


def test_configure_for_run_puts_the_dump_next_to_pp_debug(tmp_path: Path, monkeypatch) -> None:
    """取证文件与 ``pp_debug.log`` 必须在同一个 ``logs/<run_id>/`` 目录（§1.4）。"""
    monkeypatch.setattr(wd, "_WATCHDOG", None, raising=False)
    monkeypatch.setattr(wd, "_CONFIGURED_SEC", None, raising=False)
    monkeypatch.setattr(wd, "_HANDLERS_INSTALLED", True, raising=False)
    previous = signal.getsignal(signal.SIGUSR1)
    try:
        from graspo.flow.logging import run_log_dir

        dog = wd.configure_for_run(output_dir=str(tmp_path), no_progress_sec=42.0)
        assert dog._dump_path == run_log_dir(str(tmp_path)) / wd.DUMP_FILENAME  # noqa: SLF001
        assert dog._dump_path.parent.name != ""  # noqa: SLF001 — 形如 logs/<run_id>/
        assert dog._dump_path.parent.parent.name == "logs"  # noqa: SLF001
    finally:
        signal.signal(signal.SIGUSR1, previous)


def test_configure_is_idempotent_but_refuses_a_second_threshold(monkeypatch) -> None:
    monkeypatch.setattr(wd, "_WATCHDOG", None, raising=False)
    monkeypatch.setattr(wd, "_CONFIGURED_SEC", None, raising=False)
    first = wd.configure(no_progress_sec=60.0)
    assert wd.configure(no_progress_sec=60.0) is first
    with pytest.raises(RuntimeError, match="single source of truth"):
        wd.configure(no_progress_sec=30.0)


def test_beat_is_a_noop_before_configuration(monkeypatch) -> None:
    monkeypatch.setattr(wd, "_WATCHDOG", None, raising=False)
    wd.beat("x")  # 不抛、不报错：探针可以在未配置的场景（如非 PP 路径）安全调用
    wd.arm()
    wd.disarm()


# ── 退出码契约 ──────────────────────────────────────────────────────────────


def test_exit_code_is_named_and_distinct() -> None:
    code = wd.EXIT_CODE_NO_PROGRESS
    assert isinstance(code, int) and code != 0
    assert code not in (1, 2, 126, 127), "不得与 shell 语义/通用失败码相撞"
    assert code != 137, "137 = SIGKILL/`docker stop`，必须可区分"


# ── SIGUSR1：人工现场取证（子进程内验证，不污染 pytest 的 signal 状态）──────


def test_sigusr1_dumps_thread_stacks(tmp_path: Path) -> None:
    src_root = Path(__file__).resolve().parents[3] / "src"
    script = f"""
import os, signal, sys, time
from pathlib import Path
from graspo.flow.parallel import rendezvous_watchdog as wd
dump = Path({str(tmp_path / "sigusr1.txt")!r})
wd.install_sigusr1_handler(dump)
os.kill(os.getpid(), signal.SIGUSR1)
time.sleep(0.5)
print("DUMP_EXISTS", dump.exists())
print(
    "DUMP_HAS_THREADS",
    "most recent call first" in dump.read_text() if dump.exists() else False,
)
"""
    env = dict(os.environ, PYTHONPATH=str(src_root))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "DUMP_EXISTS True" in proc.stdout, proc.stdout + proc.stderr[-1000:]
    assert "DUMP_HAS_THREADS True" in proc.stdout, proc.stdout
    assert "SIGUSR1" in (tmp_path / "sigusr1.txt").read_text(encoding="utf-8")
    assert "SIGUSR1 handler installed" in proc.stderr


def test_watchdog_does_not_leave_threads_after_disarm() -> None:
    before = {t.name for t in threading.enumerate()}
    dog, _, _ = _watchdog(no_progress_sec=30.0)
    dog.arm()
    assert any(t.name == "pp-rendezvous-watchdog" for t in threading.enumerate())
    dog.disarm()
    assert {t.name for t in threading.enumerate()} == before
