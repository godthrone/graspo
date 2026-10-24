"""flow.guard — 卡守卫的**设施实现**（跨 backend + eval 共用，与具体 backend 无关）。

★★ 一名一物（gpu_guard 双家消除）：
- **契约与纯策略** ⇒ ``graspo.core.gpu_guard``（允许/预留集合、``GpuPlan``、
  ``GpuGuardError``/``GpuLockError``、``assert_gpu_lock``、``parse_gpu_plan``/
  ``resolve_gpu_plan``、``GpuInventory``/``DeviceSource``、B10 的头寸/util 双判据与
  运行期退让判据）；
- **设施操作** ⇒ 本包（读环境、runtime 哨兵回落、探测实测可见卡、fail-closed 退出、
  采样目标选择、起跑前可用断言、运行期退让检测）；
- **评测侧采样适配** ⇒ ``graspo.eval.guard``（只留 ``nvidia-smi`` 采样）。

为什么实现落 ``flow/guard/`` 而不是某个 backend：消费者包括 CLI（两个 backend 共用）
与评测链路 —— 落进某个 backend 会让另一个 backend 与 ``eval/`` 反向依赖它，制造新越层。
"""

from graspo.core.gpu_guard import (
    ALLOWED_INDICES,
    ALLOWED_INDICES_ENV,
    ALLOWED_MAX_INDEX,
    DEFAULT_GPU_POSITION_MIB,
    DEFAULT_GPU_YIELD_MARGIN_MIB,
    DEFAULT_MAX_USED_MIB,
    DEFAULT_SERVICE_POSITION_RATIO,
    DEFAULT_IDLE_SAMPLES,
    DEFAULT_IDLE_SAMPLE_INTERVAL_S,
    GPU_POSITION_MIB_ENV,
    GPU_YIELD_MARGIN_MIB_ENV,
    MAX_CARDS,
    PREFERRED_FOUR,
    RESERVED_INDICES,
    RUNTIME_MANAGED,
    CardAvailability,
    CardFrame,
    CardVerdict,
    ComputeProcess,
    DeviceSource,
    GpuGuardError,
    GpuInventory,
    GpuLockError,
    GpuPlan,
    YieldDecision,
    assert_card_available,
    assert_gpu_lock,
    assert_samples_all_idle,
    evaluate_runtime_yield,
    judge_card_frames,
    reject_unavailable,
    parse_device_list,
    parse_gpu_plan,
    resolve_allowed_indices,
    resolve_device_source,
    resolve_gpu_plan,
    resolve_gpu_position_mib,
    resolve_gpu_yield_margin_mib,
)
from graspo.flow.guard.containers import (
    cards_declared_by_containers,
    parse_docker_inspect_stream,
    read_container_declarations,
)
from graspo.flow.guard.idle import (
    assert_gpu_frames,
    assert_gpu_idle,
    assert_gpu_idle_sampled,
    judge_gpu_frames,
)
from graspo.flow.guard.inventory import (
    parse_visible_device_indices,
    probe_gpu_inventory,
)
from graspo.flow.guard.lock import (
    assert_gpu_lock_from_env,
    assert_gpu_lock_inventory,
    assert_inventory_visible,
    format_verdict,
    is_runtime_managed,
    require_gpu_lock_or_exit,
)
from graspo.flow.guard.runtime_yield import (
    detect_runtime_yield,
    our_process_pids,
    read_compute_pids,
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
    "DEFAULT_GPU_POSITION_MIB",
    "DEFAULT_GPU_YIELD_MARGIN_MIB",
    "DEFAULT_IDLE_SAMPLES",
    "DEFAULT_IDLE_SAMPLE_INTERVAL_S",
    "GPU_POSITION_MIB_ENV",
    "GPU_YIELD_MARGIN_MIB_ENV",
    "MAX_CARDS",
    "PREFERRED_FOUR",
    "RESERVED_INDICES",
    "RUNTIME_MANAGED",
    "CardAvailability",
    "CardFrame",
    "CardVerdict",
    "ComputeProcess",
    "DEFAULT_MAX_USED_MIB",
    "DEFAULT_SERVICE_POSITION_RATIO",
    "DeviceSource",
    "GpuGuardError",
    "GpuInventory",
    "GpuLockError",
    "GpuPlan",
    "YieldDecision",
    "assert_card_available",
    "assert_gpu_lock",
    "assert_samples_all_idle",
    "evaluate_runtime_yield",
    "judge_card_frames",
    "reject_unavailable",
    "parse_device_list",
    "parse_gpu_plan",
    "resolve_allowed_indices",
    "resolve_device_source",
    "resolve_gpu_plan",
    "resolve_gpu_position_mib",
    "resolve_gpu_yield_margin_mib",
    # 设施操作（实现住本包）
    "assert_gpu_frames",
    "assert_gpu_idle",
    "assert_gpu_idle_sampled",
    "judge_gpu_frames",
    "cards_declared_by_containers",
    "parse_docker_inspect_stream",
    "read_container_declarations",
    "detect_runtime_yield",
    "our_process_pids",
    "read_compute_pids",
    "parse_visible_device_indices",
    "probe_gpu_inventory",
    "assert_gpu_lock_from_env",
    "assert_gpu_lock_inventory",
    "assert_inventory_visible",
    "format_verdict",
    "is_runtime_managed",
    "require_gpu_lock_or_exit",
    "select_sample_targets",
    "select_sample_targets_for_inventory",
]
