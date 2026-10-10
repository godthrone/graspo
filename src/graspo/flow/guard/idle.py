"""flow.guard.idle — 目标卡"实测空闲"断言（设施操作）。

阈值常量与判据口径的唯一真相源；由 CLI / 脚本在申请卡位前调用。
"""

from __future__ import annotations

from graspo.core.gpu_guard import (
    IDLE_MEMORY_TOLERANCE_MIB,
    IDLE_UTILIZATION_MAX_PCT,
    GpuLockError,
    _reject,
)

def assert_gpu_idle(
    gpu_index: int | str,
    memory_used_mib: float,
    utilization_gpu_pct: float,
    *,
    memory_tolerance_mib: float = IDLE_MEMORY_TOLERANCE_MIB,
    utilization_max_pct: float = IDLE_UTILIZATION_MAX_PCT,
) -> None:
    """断言目标卡**实测空闲**（F-10）：超阈值即拒绝启动——宁等不抢、不 kill 他人。

    判据与上机口径一致：``memory.used > 64 MiB`` **或** ``utilization.gpu > 5%``
    即认为目标卡上已有他人负载。这是本轮实战里救过场的一次断言（实测拦下了含被
    第三方占用的 GPU3 的 4 卡目标集）。固定进守卫后，任何走入口守卫的运行都会
    在启动前先做这道断言。

    边界语义（不放松"误放"）：

    - 阈值**严格大于**才拒绝；恰好 64 MiB / 恰好 5% 判为通过（与任务口径
      ">64 MiB 或 util>5% 即拒绝"一致，边界值属于容差内）；
    - 空载卡通常有 driver 常驻占用（数 MiB），64 MiB 容差就是为它留的；
    - 只查**这一张**卡（由设施层保证命令带 ``-i <卡号>``），绝不查全卡。

    :param gpu_index: 目标卡号（仅用于消息）。
    :param memory_used_mib: 实测 ``memory.used``（MiB）。
    :param utilization_gpu_pct: 实测 ``utilization.gpu``（%）。
    :raises GpuLockError: 目标卡不空闲。
    """
    busy: list[str] = []
    if memory_used_mib > memory_tolerance_mib:
        busy.append(f"显存占用 {memory_used_mib:.0f} MiB > 阈值 {memory_tolerance_mib:.0f} MiB")
    if utilization_gpu_pct > utilization_max_pct:
        busy.append(f"利用率 {utilization_gpu_pct:.1f}% > 阈值 {utilization_max_pct:.1f}%")
    if not busy:
        return
    raise _reject(
        f"目标卡 GPU{gpu_index} 实测非空闲：{'；'.join(busy)}",
        "该卡上已有他人负载（或本机残留进程）。**宁等不抢**：换一张实测空闲的卡，"
        "或等它空下来——绝不 kill 他人进程、绝不与生产任务混跑（混跑会让显存数字与"
        "训练结果都不可信）。",
    )
