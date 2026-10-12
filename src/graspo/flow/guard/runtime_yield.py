"""flow.guard.runtime_yield — **运行期退让**检测（设施操作，B10）。

★ 为什么运行期不能用 ``util == 0``：起跑后**我们自己的进程**就在算 ⇒ util>0 是正常的。
运行期判据改为（纯策略在 ``core.gpu_guard.evaluate_runtime_yield``）：

① 本卡出现**非我方 PID** 的计算进程 ⇒ 退让（有人挤进来算了）；
② 剩余显存**跌破安全余量** ⇒ 退让/停跑上报。

★ 本模块**只读**：不 kill 任何进程（含我方）——退让 = 上报并由调用方**具名**
``docker stop`` 自己的容器。
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from graspo.core.gpu_guard import (
    DEFAULT_GPU_YIELD_MARGIN_MIB,
    YieldDecision,
    evaluate_runtime_yield,
    resolve_gpu_yield_margin_mib,
)

#: 命令执行器注入点（测试免真跑 nvidia-smi）。
Runner = Callable[[Sequence[str]], str]

_QUERY_APPS = (
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid",
    "--format=csv,noheader,nounits",
)
_QUERY_UUID = ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits")


def _default_runner(command: Sequence[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - 固定命令、无 shell
        list(command), capture_output=True, text=True, check=True
    )
    return completed.stdout


def our_process_pids(root: Path = Path("/proc")) -> frozenset[int]:
    """本**容器 PID 命名空间**内可见的进程 PID（= 我方进程集合）。

    容器内 ``/proc`` 只列本方命名空间的进程；因此 nvidia-smi 报出的、**不在**该集合里
    的 PID 就是"外来的邻居"（宿主侧进程）。
    """
    pids: set[int] = set()
    if not root.is_dir():
        return frozenset()
    for entry in root.iterdir():
        if entry.name.isdigit():
            pids.add(int(entry.name))
    return frozenset(pids)


def _parse_csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def read_compute_pids(
    gpu_index: int | str,
    *,
    runner: Runner | None = None,
) -> tuple[int, ...]:
    """读**本卡**上的计算进程 PID（经 ``nvidia-smi --query-compute-apps``）。

    :raises RuntimeError: 查询失败（拿不到读数 ⇒ 由调用方按 fail-closed 处理）。
    """
    run = runner or _default_runner
    try:
        uuid_rows = _parse_csv_rows(run(_QUERY_UUID))
        apps = _parse_csv_rows(run(_QUERY_APPS))
    except (subprocess.CalledProcessError, OSError) as exc:  # pragma: no cover - 设施故障
        raise RuntimeError(f"nvidia-smi 计算进程查询失败：{exc}") from exc
    want = str(gpu_index)
    uuid = ""
    for row in uuid_rows:
        if len(row) >= 2 and row[0] == want:
            uuid = row[1]
            break
    pids: list[int] = []
    for row in apps:
        if len(row) < 2:
            continue
        pid_txt, app_uuid = row[0], row[1]
        if uuid and app_uuid != uuid:
            continue
        if pid_txt.isdigit():
            pids.append(int(pid_txt))
    return tuple(sorted(set(pids)))


def detect_runtime_yield(
    gpu_index: int | str,
    *,
    runner: Runner | None = None,
    free_mib: float | None = None,
    margin_mib: float | None = None,
    env: Mapping[str, str] | None = None,
    our_pids: frozenset[int] | None = None,
) -> YieldDecision:
    """运行期退让判定（读真数据 + 纯判据）。

    :param gpu_index: 目标卡号。
    :param free_mib: 当前剩余显存（None ⇒ 只用"非我方 PID"判据）。
    :param margin_mib: 安全余量；None ⇒ 走 env/默认。
    :param our_pids: 我方 PID 集合（注入点；None ⇒ 用 :func:`our_process_pids`）。
    :raises RuntimeError: 读数拿不到——调用方须按 fail-closed 上报。
    """
    mine = our_process_pids() if our_pids is None else our_pids
    reported = read_compute_pids(gpu_index, runner=runner)
    foreign = [pid for pid in reported if pid not in mine]
    margin = (
        float(margin_mib)
        if margin_mib is not None
        else resolve_gpu_yield_margin_mib(env)
    )
    return evaluate_runtime_yield(
        gpu_index=gpu_index,
        foreign_compute_pids=foreign,
        free_mib=free_mib,
        margin_mib=margin,
    )


__all__ = [
    "DEFAULT_GPU_YIELD_MARGIN_MIB",
    "Runner",
    "detect_runtime_yield",
    "our_process_pids",
    "read_compute_pids",
]
