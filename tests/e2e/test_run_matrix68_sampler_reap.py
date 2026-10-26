"""D1（训练结束不收尾 ⇒ 挂到外层超时）的**红 → 绿**端到端用例（不吃 GPU、不起容器）。

## 缺陷（M016 实证）

`runs/M016`：`stdout.log` 末行 `[INFO:swift] End time of running main: 2026-10-26 00:10:10`、
步数 6400/6400、8/8 epoch（训练**已跑完**），但跑次台账是 `exit_code=137`、
`elapsed=57720`（= 外层 `timeout 57600` + `kill-after 120`）。

根因在 `tests/e2e/run_matrix68.sh` 生成的 `entry.sh`：它先起一个显存采样器
（`graspo record-gpu-memory`，**无界循环**），跑完 torchrun 后用
`wait "$SAMPLER"` **无上限**地等采样器 ⇒ entry.sh 永不返回 ⇒ 容器不退出 ⇒
外层超时把整档杀掉。宿主侧的采样器子 shell 同样是"无界 wait"。

## 本模块怎么证明

不是读源码对字符串，而是**真跑生成的 `entry.sh`**：

1. 从 runner 源码里取 `cat <<ENTRY_TAIL … ENTRY_TAIL` 的**正文**（逐字），
   用**真的 heredoc 展开**生成 `entry.sh`（与生产同一条展开路径：外层展开 `$CONTAINER_PY`
   /`$CONFIG`，`\\$` 留给容器内）；
2. 把容器内的 `/out`、`/workspace/graspo` 重写到临时目录（**只做路径重写**，不动任何语句），
   并把 `torchrun` 换成一个 **5 s 后以 rc=7 退出的 stub 训练进程**、把 CONTAINER_PY 换成
   一个**与 `record-gpu-memory` 同语义的 stub 采样器**（循环采样；收到 SIGTERM 后在
   finally 里落 `gpu_memory_summary.json` 再退）；
3. 旧版 = HEAD 的 entry 段（含 `wait "$SAMPLER"`）；新版 = 工作树。两者用同一个 stub 跑，
   旧版必须**挂到超时**（演示用 `timeout 60`），新版必须**数秒内**返回且退出码 == 训
   练的 rc（⛔ 不是采样器的）。

★ 只创建临时目录、只往 PATH 放假 `torchrun`；不碰 GPU、不跑 docker、不跑任何训练档。
★ 红线：只 `kill` 本用例自己起的 stub（pid 由 stub 自报），⛔ 不 pkill、⛔ 不碰他人进程。
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "tests" / "e2e" / "run_matrix68.sh"

#: 旧版演示"挂到超时"用的短超时（秒）。验收口径：旧实现必然打满这个上限。
RED_TIMEOUT_SEC = int(os.environ.get("GRASPO_D1_RED_TIMEOUT_SEC", "60"))
#: 新版"数秒内返回"的上限（秒）：只要求远小于 RED_TIMEOUT_SEC（本用例实测 ~5 s）。
GREEN_BUDGET_SEC = 30

#: stub 训练进程（= torchrun 的位置）：5 s 后以 **7** 退出（非 0 ⇒ 能看出退出码有没有被采样器顶掉）。
_STUB_TORCHRUN = """#!/bin/bash
sleep "${STUB_TRAIN_SEC:-5}"
exit "${STUB_TRAIN_RC:-7}"
"""

#: stub 采样器（= `graspo record-gpu-memory` 的位置与语义）：无界循环采样；**SIGTERM 时
#: 在 finally 里落 `gpu_memory_summary.json` 再退**（真命令的优雅停止就是这条路，
#: 见 src/graspo/cli/gpu_monitor.py::record_gpu_memory 的 try/finally + 信号处理）。
#: `STUB_SAMPLER_IGNORE_TERM=1` ⇒ 变成"卡死不退"的采样器，用来验 SIGKILL 兜底。
_STUB_SAMPLER = '''#!/usr/bin/env python3
import json, os, signal, sys, time

summary = os.environ["STUB_SUMMARY_PATH"]
with open(os.environ["STUB_PIDFILE"], "w") as fh:
    fh.write(str(os.getpid()))

stop = {"requested": False}
if os.environ.get("STUB_SAMPLER_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, lambda *_: None)      # 故意忽略 TERM：只有 SIGKILL 能停
else:
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("requested", True))

started = time.monotonic()
try:
    while not stop["requested"]:
        time.sleep(0.2)
finally:
    with open(summary, "w") as fh:
        json.dump({"sample_count": 1, "stub": True}, fh)
        fh.write("\\n")
'''


def _entry_tail_text(text: str) -> str:
    """`cat <<ENTRY_TAIL … ENTRY_TAIL` 的**正文**（逐字、未展开）。"""
    start = text.index("cat <<ENTRY_TAIL\n") + len("cat <<ENTRY_TAIL\n")
    end = text.index("\nENTRY_TAIL\n", start)
    return text[start:end] + "\n"


def _old_tail_text() -> tuple[str, str]:
    """旧版 entry 段：优先取 **git HEAD 的 runner**（最忠实），取不到就退化为"只回退那一行"。

    退化仍然忠实：D1 的容器半边就是 `wait_for_gpu_summary "$SAMPLER"` 这一行——
    旧代码是 `wait "$SAMPLER" 2>/dev/null || true`（HEAD `73de917` 逐字）。
    返回 (正文, 来源说明)。
    """
    try:
        head = subprocess.run(
            ["git", "show", "HEAD:tests/e2e/run_matrix68.sh"],
            cwd=str(_ROOT), capture_output=True, text=True, check=False,
        ).stdout
    except OSError:                                     # pragma: no cover - 无 git 的环境
        head = ""
    if 'wait "\\$SAMPLER" 2>/dev/null || true' in head:
        return _entry_tail_text(head), "git HEAD 的 runner（真·旧脚本）"
    new = _entry_tail_text(_RUNNER.read_text(encoding="utf-8"))
    assert 'wait_for_gpu_summary "\\$SAMPLER"' in new, "runner 里找不到新的收口调用"
    return (new.replace('wait_for_gpu_summary "\\$SAMPLER"',
                        'wait "\\$SAMPLER" 2>/dev/null || true'),
            "工作树 entry 段 + 回退收口那一行（= HEAD 73de917 的逐字写法）")


def _generate_entry_sh(tmp_path: Path, tail_body: str) -> tuple[Path, dict]:
    """用**真的 heredoc** 把 entry 段展开成 entry.sh，并把容器内路径重写到临时目录。

    只做两类改写（⛔ 不动任何语句/控制流）：
      · `/out`               → `<tmp>/out`（容器内的挂载点 /out）
      · `/workspace/graspo`  → `<tmp>/ws`（容器内的只读代码挂载点）
    `ENTRY_HEAD` 里定义的 `GPU_EVIDENCE_DIR` 由本函数补一行同名同值定义（路径同样重写）。
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    torchrun = bin_dir / "torchrun"
    torchrun.write_text(_STUB_TORCHRUN, encoding="utf-8")
    torchrun.chmod(0o755)
    sampler = tmp_path / "stub_sampler.py"
    sampler.write_text(_STUB_SAMPLER, encoding="utf-8")
    sampler.chmod(0o755)
    (tmp_path / "ws").mkdir()
    (tmp_path / "out").mkdir()

    raw = tmp_path / "entry.raw.sh"
    outer = tmp_path / "generate.sh"
    outer.write_text(
        # 生产里这些变量来自 runner 的宿主侧变量（CONTAINER_PY / CONFIG / NPROC / PORT）。
        'set -o pipefail\n'
        f'CONTAINER_PY="{sampler}"\n'
        'CONFIG="M016.yaml"\n'
        'NPROC=1\n'
        'PORT=29500\n'
        f'cat <<ENTRY_TAIL > "{raw}"\n' + tail_body + 'ENTRY_TAIL\n',
        encoding="utf-8")
    done = subprocess.run(["bash", str(outer)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr

    body = raw.read_text(encoding="utf-8")
    body = body.replace("/workspace/graspo", str(tmp_path / "ws"))
    body = body.replace("/out", str(tmp_path / "out"))
    entry = tmp_path / "entry.sh"
    entry.write_text(
        # ENTRY_HEAD 的两行（同值、同语义；只重写路径），本用例只取 ENTRY_TAIL 段
        f'GPU_EVIDENCE_DIR="{tmp_path / "out" / "gpu"}"\n'
        'mkdir -p "$GPU_EVIDENCE_DIR"\n' + body, encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "STUB_SUMMARY_PATH": str(tmp_path / "out" / "gpu" / "gpu_memory_summary.json"),
        "STUB_PIDFILE": str(tmp_path / "sampler.pid"),
        "GRASPO_GPU_SAMPLER_GRACE_SEC": "2",
    }
    return entry, env


def _run_entry(entry: Path, env: dict, *, timeout_sec: int) -> tuple[subprocess.CompletedProcess, float]:
    started = time.monotonic()
    try:
        done = subprocess.run(["bash", str(entry)], env={**os.environ, **env},
                              capture_output=True, text=True, check=False,
                              timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:            # 旧版：挂到外层超时
        return subprocess.CompletedProcess(["bash", str(entry)], 124,
                                          exc.stdout or "", exc.stderr or ""), time.monotonic() - started
    return done, time.monotonic() - started


def _kill_leftover_stub(tmp_path: Path) -> None:
    """★只清理本用例自己起的 stub（pid 由 stub 自报）——⛔ 不 pkill、⛔ 不碰他人进程。"""
    pidfile = tmp_path / "sampler.pid"
    if not pidfile.is_file():
        return
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    for sig in ("TERM", "KILL"):
        try:
            subprocess.run(["kill", f"-{sig}", str(pid)], check=False,
                           capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):     # pragma: no cover
            pass


# ── ① 红：旧版 entry.sh 挂到超时 ─────────────────────────────────────────────
def test_old_entry_sh_hangs_until_the_outer_timeout(tmp_path):
    """★红（D1 的实证）：旧版 `wait "$SAMPLER"` 无界 ⇒ 训练 5 s 就退出了，entry.sh 仍不返回，
    只能被外层超时打断（rc=124/137），训练的真实退出码 7 永远传不出来。"""
    tail_body, source = _old_tail_text()
    entry, env = _generate_entry_sh(tmp_path, tail_body)
    try:
        done, elapsed = _run_entry(entry, env, timeout_sec=RED_TIMEOUT_SEC)
    finally:
        _kill_leftover_stub(tmp_path)
    assert done.returncode == 124, (
        f"旧版（来源：{source}）居然自己返回了 rc={done.returncode}？"
        f"（D1 的复现条件被破坏：采样器不再是无人收口的无界子进程）\n{done.stderr[-800:]}")
    assert elapsed >= RED_TIMEOUT_SEC * 0.9, f"旧版挂死时长 {elapsed:.1f}s 不等于超时上限"
    assert not (tmp_path / "out" / "exit_code").exists(), (
        "旧版被打断前居然写了 exit_code ⇒ 演示场景不成立")


# ── ② 绿：新版数秒返回，且退出码 == 训练的真实 rc ───────────────────────────
def test_new_entry_sh_returns_quickly_with_the_training_exit_code(tmp_path):
    """★绿：新版 SIGTERM + 有界等待 + wait 回收 ⇒ 数秒内返回，rc = 训练 stub 的 7
    （⛔ 不是采样器的），且采样器的摘要**照样落盘**（TERM 是优雅停止）。"""
    entry, env = _generate_entry_sh(
        tmp_path, _entry_tail_text(_RUNNER.read_text(encoding="utf-8")))
    done, elapsed = _run_entry(entry, env, timeout_sec=GREEN_BUDGET_SEC)
    try:
        assert done.returncode == 7, (
            f"新版退出码 {done.returncode} ≠ 训练 stub 的 7（被采样器顶掉了？）\n{done.stderr[-800:]}")
        assert elapsed < GREEN_BUDGET_SEC, f"新版耗时 {elapsed:.1f}s，不是数秒内返回"
        assert (tmp_path / "out" / "exit_code").read_text(encoding="utf-8").strip() == "7"
        assert (tmp_path / "out" / "gpu" / "gpu_memory_summary.json").is_file(), (
            "采样器摘要在 SIGTERM 后没落盘 ⇒ 有界回收把取证弄丢了")
        state = tmp_path / "out" / "gpu" / "gpu_summary.state"
        assert state.read_text(encoding="utf-8").strip() == "present"
    finally:
        _kill_leftover_stub(tmp_path)


# ── ③ SIGKILL 兜底：采样器忽略 SIGTERM 也不许拖住收口 ───────────────────────
def test_sampler_that_ignores_sigterm_is_killed_within_the_grace(tmp_path):
    entry, env = _generate_entry_sh(
        tmp_path, _entry_tail_text(_RUNNER.read_text(encoding="utf-8")))
    env = {**env, "STUB_SAMPLER_IGNORE_TERM": "1", "GRASPO_GPU_SAMPLER_GRACE_SEC": "2"}
    done, elapsed = _run_entry(entry, env, timeout_sec=GREEN_BUDGET_SEC)
    try:
        assert done.returncode == 7, done.stderr[-800:]
        assert elapsed < GREEN_BUDGET_SEC, f"忽略 TERM 的采样器把收口拖了 {elapsed:.1f}s"
        # 摘要没落盘 ⇒ 如实登记取证缺口，但**不改**退出码（train? 见上一条 rc==7）
        assert not (tmp_path / "out" / "gpu" / "gpu_memory_summary.json").exists()
        assert (tmp_path / "out" / "gpu" / "EVIDENCE_GAP_gpu_memory_summary").is_file(), (
            "摘要缺失却没登记取证缺口 ⇒ 静默把没取证当成了已取证")
        assert (tmp_path / "out" / "gpu" / "gpu_summary.state").read_text(
            encoding="utf-8").strip() == "missing"
    finally:
        _kill_leftover_stub(tmp_path)


# ── ④ 静态哨兵：几件"改回去就复发"的结构不变量 ──────────────────────────────
def test_entry_tail_serveral_invariants():
    text = _RUNNER.read_text(encoding="utf-8")
    tail = _entry_tail_text(text)
    assert 'wait "\\$SAMPLER" 2>/dev/null || true' not in tail, (
        '无界 wait "$SAMPLER" 又回来了 ⇒ D1 复发')
    assert 'wait_for_gpu_summary "\\$SAMPLER"' in tail, "收口调用不见了"
    # 收口必须在训练返回之后、写 exit_code 之前，且退出码只来自 RC
    assert tail.index("RC=\\$?") < tail.index('wait_for_gpu_summary "\\$SAMPLER"')
    assert tail.index('wait_for_gpu_summary "\\$SAMPLER"') < tail.index('echo "\\$RC" > /out/exit_code')
    assert tail.rstrip().endswith('exit "\\$RC"'), "收口行不再是 exit RC"
    # 不加引号的 heredoc 正文里，裸反引号会被**外层**当命令替换真的执行
    assert "`" not in tail.replace("\\`", ""), "entry 段出现裸反引号（外层会执行它）"
    # 宿主侧的采样器回收也必须是有界的（同一个缺陷的另一半）
    host = text[text.index("reap_sampler_pid()"):]
    host = host[:host.index("\n}\n")]
    assert "kill -TERM" in host and "kill -KILL" in host and 'wait "$pid"' in host, host


def test_generated_entry_sh_parses(tmp_path):
    """`bash -n`：外层 heredoc 展开后的产物必须是合法脚本（防注释里的 $$/反引号污染）。"""
    entry, _env = _generate_entry_sh(
        tmp_path, _entry_tail_text(_RUNNER.read_text(encoding="utf-8")))
    parsed = subprocess.run(["bash", "-n", str(entry)], capture_output=True, text=True,
                            check=False)
    assert parsed.returncode == 0, parsed.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_grace_is_configurable_and_bounded():
    """`GRASPO_GPU_SAMPLER_GRACE_SEC` 必须真被读到（默认 30，非法值回落 30）。"""
    tail = _entry_tail_text(_RUNNER.read_text(encoding="utf-8"))
    assert 'GRASPO_GPU_SAMPLER_GRACE_SEC:-30' in tail
    assert "''|*[!0-9]*" in tail, "非法 grace 没有回落 ⇒ `sleep` 会当场失败、收口失去上限"
    assert "kill -KILL" in tail, "没有 SIGKILL 兜底"
