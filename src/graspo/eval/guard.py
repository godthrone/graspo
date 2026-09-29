"""GPU 锁卡与显存采样的评测侧适配层（fail-closed）。

**职责**：把 ``graspo.core.gpu_guard`` 的锁卡边界接到评测链路上，并提供
"只采可见卡"的显存采样。

**本文件不负责**：锁卡**规则的判定**与**卡计划契约**——两者都在 ``graspo.core.gpu_guard``
里（单一真相源，宪法 §1.4）。本模块只做两件事：① 转发 ``GpuGuardError`` /
``GpuPlan`` / ``parse_gpu_plan`` / ``resolve_gpu_plan``（**2026-09-29 D-03 边界修复**：
这些定义已下沉到 core，本模块保留同名导入路径，异常类型仍是同一个类对象）；
② 用 ``nvidia-smi -i`` 采样。

**为什么卡计划契约下沉**：修复前 ``core/schema.py`` 反向导入 ``eval.guard`` 取
``GpuGuardError`` / ``resolve_gpu_plan``，构成 ``core → eval`` 跨层环（AST 边界测试
判违规）。下沉后方向变成 ``eval → core``（合法），规则与契约各只有一份定义。

**为什么规则不在这里重写**

项目里已有 ``src/graspo/core/gpu_guard.py``（另一个工作包产出），它定义了
默认允许集合 ``{4,5,6,7}``（部署事实，可用 ``GRASPO_ALLOWED_GPU_INDICES`` 覆盖）、
上限 4 卡（``MAX_CARDS=4``）、默认保留集合 ``()``
（可用 ``GRASPO_RESERVED_GPU_INDICES`` 覆盖）以及全部拒绝文案。评测链路如果自己
再写一份，就会出现两份会漂移的边界定义——这正是宪法 §1.4 禁止的双真相源。
因此本模块**只透传 + 适配**，不重新定义规则。

**fail-closed 的体现**

``GpuPlan`` 只能由 :func:`resolve_gpu_plan` 产生，而它转手就调
``core.gpu_guard.assert_gpu_lock``：未显式给卡、越出生效的允许集合、落在生效的
保留集合里、卡数 >4 一律抛错。**没有默认值分支**——任何"顺手挑张空闲卡"的启发式
都可能踩上别人（或生产）正在用的卡。历史教训：ELAM ``v3_eval_pipeline_v2.sh`` 把
``EXPORT_GPU=6`` / ``VLLM_GPU=7`` 硬编码进脚本，那份做法严禁照抄。

**采样为什么必须带 ``-i``（宪法 §2.4 操作防呆）**

历史教训：不带 ``-i`` 的 ``nvidia-smi`` 会把全部物理卡（含 6/7）都报出来，
采样被生产卡读数污染。:func:`sample_device_memory_mib` 用
``nvidia-smi --id=<可见卡>`` 并把子进程的 ``CUDA_VISIBLE_DEVICES`` 钉在可见卡上，
双保险。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graspo.core.gpu_guard import (
    DEFAULT_ALLOWED_INDICES,
    DEFAULT_RESERVED_INDICES,
    MAX_CARDS,
    GpuGuardError,
    GpuPlan,
    # 公共同名转发（D-03）：定义已下沉到 core.gpu_guard，本模块只做再导出。
    parse_gpu_plan,  # noqa: F401
    resolve_allowed_indices,
    resolve_gpu_plan,  # noqa: F401
    resolve_reserved_indices,
)

#: 评测链路承认的**默认**允许集合与上限——**直接取自 core.gpu_guard 的默认值**，
#: 不另立一份。运行时生效值可能被配置覆盖，见 :func:`resolve_allowed_gpu_indices`。
ALLOWED_GPU_INDICES: frozenset[int] = frozenset(DEFAULT_ALLOWED_INDICES)
MAX_GPU_COUNT: int = MAX_CARDS

#: **默认**保留卡（透传，便于调用方/报告引用；规则源头仍是 core.gpu_guard）。
RESERVED_GPU_INDICES: tuple[int, ...] = DEFAULT_RESERVED_INDICES


def resolve_allowed_gpu_indices() -> frozenset[int]:
    """生效的允许集合（配置优先，回落默认值）——规则源头在 ``core.gpu_guard``。"""
    return frozenset(resolve_allowed_indices())


def resolve_reserved_gpu_indices() -> tuple[int, ...]:
    """生效的保留卡集合（配置优先，回落默认值）——规则源头在 ``core.gpu_guard``。"""
    return resolve_reserved_indices()


# ``GpuGuardError`` / ``GpuPlan`` / ``parse_gpu_plan`` / ``resolve_gpu_plan`` 的定义
# 已下沉到 ``graspo.core.gpu_guard``（2026-09-29 D-03 边界修复：core 不能反向导入
# ``eval.guard``）。此处按上面的 import 转发，既有导入路径
# （``from graspo.eval.guard import GpuGuardError, GpuPlan, resolve_gpu_plan``）
# 与异常类型身份（同一个类对象）都零变化。


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
        plan: 由 :func:`resolve_gpu_plan` 产出的锁卡计划。

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
