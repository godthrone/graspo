"""flow.guard — 卡守卫的**设施实现**（跨 backend + eval 共用，与具体 backend 无关）。

★★ 一名一物（gpu_guard 双家消除）：
- **契约与纯策略** ⇒ ``graspo.core.gpu_guard``（允许/预留集合、``GpuPlan``、
  ``GpuGuardError``/``GpuLockError``、``assert_gpu_lock``、``parse_gpu_plan``/
  ``resolve_gpu_plan``、``GpuInventory``/``DeviceSource``）；
- **设施操作** ⇒ 本包（读环境、runtime 哨兵回落、探测实测可见卡、fail-closed 退出、
  采样目标选择、空闲断言）；
- **评测侧采样适配** ⇒ ``graspo.eval.guard``（只留 ``nvidia-smi`` 采样）。

为什么实现落 ``flow/guard/`` 而不是某个 backend：消费者包括 CLI（两个 backend 共用）
与评测链路 —— 落进某个 backend 会让另一个 backend 与 ``eval/`` 反向依赖它，制造新越层。
"""

from graspo.core.gpu_guard import (
    ALLOWED_INDICES,
    ALLOWED_INDICES_ENV,
    ALLOWED_MAX_INDEX,
    MAX_CARDS,
    PREFERRED_FOUR,
    RESERVED_INDICES,
    RUNTIME_MANAGED,
    DeviceSource,
    GpuGuardError,
    GpuInventory,
    GpuLockError,
    GpuPlan,
    assert_gpu_lock,
    parse_device_list,
    parse_gpu_plan,
    resolve_allowed_indices,
    resolve_device_source,
    resolve_gpu_plan,
)
from graspo.flow.guard.idle import assert_gpu_idle
from graspo.flow.guard.lock import (
    assert_gpu_lock_from_env,
    assert_gpu_lock_inventory,
    assert_inventory_visible,
    format_verdict,
    is_runtime_managed,
    require_gpu_lock_or_exit,
)
from graspo.flow.guard.sampling import (
    select_sample_targets,
    select_sample_targets_for_inventory,
)

__all__ = [
    # 契约 / 纯策略（唯一定义在 core.gpu_guard，此处只做统一入口转发）
    "ALLOWED_INDICES",
    "ALLOWED_INDICES_ENV",
    "ALLOWED_MAX_INDEX",
    "MAX_CARDS",
    "PREFERRED_FOUR",
    "RESERVED_INDICES",
    "RUNTIME_MANAGED",
    "DeviceSource",
    "GpuGuardError",
    "GpuInventory",
    "GpuLockError",
    "GpuPlan",
    "assert_gpu_lock",
    "parse_device_list",
    "parse_gpu_plan",
    "resolve_allowed_indices",
    "resolve_device_source",
    "resolve_gpu_plan",
    # 设施操作（实现住本包）
    "assert_gpu_idle",
    "assert_gpu_lock_from_env",
    "assert_gpu_lock_inventory",
    "assert_inventory_visible",
    "format_verdict",
    "is_runtime_managed",
    "require_gpu_lock_or_exit",
    "select_sample_targets",
    "select_sample_targets_for_inventory",
]
