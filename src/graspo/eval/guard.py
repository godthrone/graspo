"""GPU 锁卡与显存采样的**评测侧采样适配**（fail-closed）—— 只剩采样。

★★ 一名一物（gpu_guard 双家消除后）：
- **契约与纯策略** ⇒ ``graspo.core.gpu_guard``（``GpuPlan`` / ``GpuGuardError`` /
  ``parse_gpu_plan`` / ``resolve_gpu_plan`` / 允许集合与上限）。**唯一定义处**。
- **设施操作** ⇒ ``graspo.flow.guard``（读环境、盘点、退出、空闲断言）。
- **本模块** ⇒ 只做一件事：用 ``nvidia-smi -i`` 采**可见卡**的显存/驱动读数，
  外加评测链路承认的卡集合别名（**直接派生自 core，不另立一份规则**）。

**为什么规则不在这里**：评测链路若自己再写一份边界定义，就会出现两份会漂移的
边界——这正是宪法 §1.4 禁止的双真相源（历史教训：ELAM ``v3_eval_pipeline_v2.sh``
把 ``EXPORT_GPU=6`` / ``VLLM_GPU=7`` 硬编码进脚本）。

**采样为什么必须带 ``-i``（宪法 §2.4 操作防呆）**

历史教训：不带 ``-i`` 的 ``nvidia-smi`` 会把全部物理卡（含没被允许的卡）都报出来，
采样被非目标卡读数污染。:func:`sample_device_memory_mib` 用
``nvidia-smi --id=<可见卡>`` 并把子进程的 ``CUDA_VISIBLE_DEVICES`` 钉在可见卡上，
双保险。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graspo.core.gpu_guard import (
    ALLOWED_INDICES,
    MAX_CARDS,
    RESERVED_INDICES,
    GpuGuardError,
    GpuPlan,
)

__all__ = [
    "ALLOWED_GPU_INDICES",
    "MAX_GPU_COUNT",
    "RESERVED_GPU_INDICES",
    "DeviceMemory",
    "assert_devices_free",
    "repo_relative",
    "sample_device_memory_mib",
    "sample_driver_version",
]

#: 评测链路承认的允许集合与上限——**直接派生自 core.gpu_guard**，不另立一份。
#: ★ 允许集合 = `ALLOWED_INDICES` 本身（B10 起默认 = 本机全卡 0–7），**不再**用
#: `range(max+1)` 那种「仅上限」口径——后者会错误地把 0–3 也算成允许卡，
#: 与真相源的正向白名单判据冲突（宪法 §1.4 单一真相源）。
ALLOWED_GPU_INDICES: frozenset[int] = frozenset(ALLOWED_INDICES)
MAX_GPU_COUNT: int = MAX_CARDS

#: 仅透传（便于调用方/报告引用；规则源头仍是 core.gpu_guard）。
#: ★「生产卡 / reserved」语义已退役 ⇒ 恒为空元组（B7）。
RESERVED_GPU_INDICES: tuple[int, ...] = RESERVED_INDICES


@dataclass(slots=True)
class DeviceMemory:
    """一张卡的显存读数（MiB）。"""

    index: int
    name: str
    used_mib: int
    total_mib: int

    @property
    def used_percent(self) -> float:
        return 100.0 * self.used_mib / self.total_mib if self.total_mib > 0 else 0.0


def _nvidia_smi_command(devices: tuple[int, ...], query: str) -> list[str]:
    """构造**限定设备**的 nvidia-smi 命令（``--id`` 是硬要求，见模块 docstring）。"""
    return [
        "nvidia-smi",
        f"--id={','.join(str(index) for index in devices)}",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]


def sample_device_memory_mib(plan: GpuPlan) -> list[DeviceMemory]:
    """采样 ``plan`` 中每一张卡的显存占用——只采可见卡。

    实现上的防呆：子进程环境把 ``CUDA_VISIBLE_DEVICES`` 设为 ``plan.csv``，
    因此 nvidia-smi 眼中的设备集合恰好是这几张卡；再加 ``--id`` 限定。
    空 plan 无法采样——没有"全部卡"这个选项。

    Args:
        plan: 由 ``graspo.core.gpu_guard.resolve_gpu_plan`` 产出的锁卡计划。

    Returns:
        与 ``plan.devices`` 对应的读数列表。

    Raises:
        GpuGuardError: plan 为空、无 ``nvidia-smi``、命令失败、或输出解析不出设备。
    """
    if not plan.devices:
        raise GpuGuardError("refusing to sample: empty GPU plan (no device list to restrict to)")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = plan.csv
    try:
        completed = subprocess.run(
            _nvidia_smi_command(plan.devices, "index,name,memory.used,memory.total"),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        raise GpuGuardError(
            "nvidia-smi not found — cannot sample GPU memory; "
            "run the sampling inside the GPU container"
        ) from None
    except subprocess.CalledProcessError as exc:
        raise GpuGuardError(
            f"nvidia-smi failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from None

    readings: list[DeviceMemory] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            used_mib = int(parts[2])
            total_mib = int(parts[3])
        except ValueError:
            continue
        readings.append(
            DeviceMemory(index=index, name=parts[1], used_mib=used_mib, total_mib=total_mib)
        )
    if not readings:
        raise GpuGuardError(
            f"nvidia-smi returned no parsable device lines for {plan.devices}: "
            f"{completed.stdout.strip()!r}"
        )
    return readings


def sample_driver_version() -> str | None:
    """读 NVIDIA 驱动版本（环境指纹用）。取不到返回 ``None``，不抛异常。

    这里刻意不限定 ``--id``：驱动版本是**主机级**属性，与具体卡无关。
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return ",".join(sorted(versions)) if versions else None


def assert_devices_free(
    plan: GpuPlan,
    *,
    busy_threshold_mib: int = 20_000,
) -> list[DeviceMemory]:
    """启动前检查：``plan`` 里的卡是否已被占用。

    Args:
        plan: 锁卡计划。
        busy_threshold_mib: 显存占用超过该值即视为被占用。

    Returns:
        采样读数（供写进产物，留痕）。

    Raises:
        GpuGuardError: 任一目标卡占用超阈值。
    """
    readings = sample_device_memory_mib(plan)
    busy = [reading for reading in readings if reading.used_mib > busy_threshold_mib]
    if busy:
        detail = ", ".join(
            f"GPU {reading.index} used {reading.used_mib} MiB/{reading.total_mib} MiB"
            for reading in busy
        )
        raise GpuGuardError(
            f"target GPU(s) appear occupied (> {busy_threshold_mib} MiB): {detail}. "
            "Pick different cards — never kill the production vLLM."
        )
    return readings


def repo_relative(path: str | Path) -> str:
    """把路径归一为"尽可能不含机器信息的"形式。

    产物路径里出现宿主绝对路径属于机器相关信息（宪法 §15）。产物本身落在
    ``.local/``（不入库），但报告可能被复制出去，故这里把路径压成相对形式，
    减少外泄面。**不是**安全边界的替代品——真正的边界是"产物必须放 .local/"。
    """
    candidate = Path(path)
    try:
        return str(candidate.relative_to(Path.cwd()))
    except ValueError:
        return str(candidate)
