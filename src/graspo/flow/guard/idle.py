"""flow.guard.idle — 目标卡**起跑前可用**断言（设施操作，三判据并取）。

★ 判据（唯一真相源 = ``graspo.core.gpu_guard``）：
    **可用 ⟺ ① 已用显存 ≤ 空闲上限 ∧ 剩余显存 ≥ 本档头寸
             ∧ ② 卡上无 compute 进程 ∧ ③ 卡未被容器声明 ∧ util == 0**
★★ 2026-10-25 起由「剩余 ≥ 头寸 ∧ util == 0」两条升为**三条并取**（util 仍须为 0）：
    只看剩余/util 会在 228 上把"被人常驻占着 13–16 GiB、但 util 采样为 0"的卡
    （GPU0/1/4/5）与"卡上有宿主 compute 进程"的卡（GPU2）全部放行 ⇒ 起服/起训撞 OOM。
    ⛔ 旧的「64 MiB 绝对阈值」形态没有复活：显存面是"已用 ≤ 上限"与"剩余 ≥ 头寸"
    两条**合取**；服务档头寸按 ``0.85 × 总显存`` 解析
    （见 ``DEFAULT_MAX_USED_MIB`` / ``DEFAULT_SERVICE_POSITION_RATIO``）。

★ 起跑前必须**连续采样**（默认 3 帧、间隔 10 s）且**帧帧满足**——单帧会把
"邻居正好落在两次 kernel 之间"误判为空闲。

★ 运行期判据不同（我们自己的进程会让 util>0）⇒ 见 ``flow.guard.runtime_yield``。

本模块是**设施层**：真读 ``nvidia-smi`` / ``docker``、真等待；纯判据住在 ``core``。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

from graspo.core.gpu_guard import (
    DEFAULT_GPU_POSITION_MIB,
    DEFAULT_IDLE_SAMPLES,
    DEFAULT_IDLE_SAMPLE_INTERVAL_S,
    CardAvailability,
    CardFrame,
    CardVerdict,
    assert_card_available,
    assert_samples_all_idle,
    judge_card_frames,
    reject_unavailable,
    resolve_gpu_position_mib,
)

#: 单帧读数的形态：``(free_mib, utilization_pct)``。
Frame = tuple[float, float]


def _position(position_mib: float | None, env: Mapping[str, str] | None) -> float:
    """头寸解析（显式入参优先，其次 env，最后默认）；唯一解析入口在 core。"""
    if position_mib is not None:
        return float(position_mib)
    return resolve_gpu_position_mib(env)


def assert_gpu_idle(
    gpu_index: int | str,
    *,
    free_mib: float,
    utilization_pct: float,
    position_mib: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CardAvailability:
    """**单帧**可用断言：剩余显存 ≥ 头寸 ∧ util == 0。

    ★ 起跑入口请用 :func:`assert_gpu_idle_sampled`（连续采样）；本函数是它的单帧判据
    与手动/测试用入口。

    :param gpu_index: 目标卡号（宿主卡号或容器内本地序号；仅用于消息）。
    :param free_mib: 实测**剩余**显存（MiB）= ``memory.total - memory.used``。
    :param utilization_pct: 实测 ``utilization.gpu``（%）。
    :param position_mib: 本档头寸（MiB）；None ⇒ 走 env/默认。
    :raises GpuLockError: 该帧不满足双判据。
    """
    return assert_card_available(
        gpu_index,
        free_mib=float(free_mib),
        utilization_pct=float(utilization_pct),
        position_mib=_position(position_mib, env),
    )


def assert_gpu_idle_sampled(
    gpu_index: int | str,
    sample_reader: Callable[[int | str], Frame],
    *,
    samples: int = DEFAULT_IDLE_SAMPLES,
    interval_s: float = DEFAULT_IDLE_SAMPLE_INTERVAL_S,
    position_mib: float | None = None,
    env: Mapping[str, str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> CardAvailability:
    """**连续采样**版可用断言：逐帧真读，帧帧满足才算可用（fail-closed）。

    :param gpu_index: 目标卡号。
    :param sample_reader: 读一帧 ``(free_mib, utilization_pct)`` 的设施调用（注入点）。
    :param samples: 采样帧数（默认 3；<2 直接拒绝）。
    :param interval_s: 帧间隔秒数（默认 12 s）。
    :param sleeper: 等待函数（注入点，测试免真等）。
    :raises GpuLockError: 帧数不足，或任一帧不满足双判据。
    """
    frames: list[Frame] = [sample_reader(gpu_index)]
    for frame_no in range(max(samples, 1) - 1):
        sleeper(interval_s)
        frames.append(sample_reader(gpu_index))
    return assert_samples_all_idle(
        gpu_index, frames, position_mib=_position(position_mib, env)
    )


def judge_gpu_frames(
    gpu_index: int | str,
    frames: Sequence[CardFrame],
    *,
    position_mib: float | None = None,
    position_ratio: float | None = None,
    max_used_mib: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CardVerdict:
    """**多帧三判据**判定单卡（不抛错；抛错在 :func:`assert_gpu_frames`）。

    判据本体在 ``core.gpu_guard.judge_card_frames``——本函数只是本层的统一入口，
    让"起跑前可用"这一件事在设施层只有一个调用点。
    """
    return judge_card_frames(
        gpu_index,
        frames,
        position_mib=position_mib,
        position_ratio=position_ratio,
        max_used_mib=max_used_mib,
        env=env,
    )


def assert_gpu_frames(
    gpu_index: int | str,
    frames: Sequence[CardFrame],
    *,
    position_mib: float | None = None,
    position_ratio: float | None = None,
    max_used_mib: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CardVerdict:
    """**多帧三判据**可用断言：任一帧不满足即抛 :class:`GpuLockError`。"""
    verdict = judge_gpu_frames(
        gpu_index,
        frames,
        position_mib=position_mib,
        position_ratio=position_ratio,
        max_used_mib=max_used_mib,
        env=env,
    )
    if not verdict.available:
        raise reject_unavailable([verdict])
    return verdict


__all__ = [
    "DEFAULT_GPU_POSITION_MIB",
    "DEFAULT_IDLE_SAMPLES",
    "DEFAULT_IDLE_SAMPLE_INTERVAL_S",
    "CardFrame",
    "CardVerdict",
    "Frame",
    "assert_gpu_frames",
    "assert_gpu_idle",
    "assert_gpu_idle_sampled",
    "judge_gpu_frames",
]
