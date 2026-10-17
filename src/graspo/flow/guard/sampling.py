"""flow.guard.sampling — 显存采样目标卡的选择（设施操作）。

只允许落在 ``NVIDIA_VISIBLE_DEVICES`` 之内；越界读取会把**可见集之外**的卡
（他人工作负载）算进来，污染显存读数。
"""

from __future__ import annotations

from graspo.flow.guard.lock import assert_inventory_visible

from graspo.core.gpu_guard import (
    RUNTIME_MANAGED,
    GpuInventory,
    GpuLockError,
    _ENV_KEY,
    _device_source_value,
    _reject,
    assert_gpu_lock,
    parse_device_list,
)

def select_sample_targets(
    explicit: str | None,
    visible: str | None,
) -> tuple[int, ...]:
    """确定显存采样目标卡：只允许落在 ``NVIDIA_VISIBLE_DEVICES`` 之内。

    规则（防呆）：
    - ``visible`` 必须先通过锁卡守卫（未设置 / 含**允许集合之外**的卡即拒绝；
      默认允许集合 = 本机全卡 0–7，先到先得；★ 2026-10-17 主席令「单个作业不限制卡数」
      ⇒ **不再判卡数**，旧的 >4 卡拦截已删除）；
    - ``explicit`` 为 ``None`` / 空 / ``"all"`` 时，采样全部可见卡；
    - ``explicit`` 显式列出时，必须 **⊆ visible**，否则拒绝——采样越出可见集
      正是"把集合外的卡混进显存读数"的根因（历史数据污染事故）。

    **宿主侧通道**：``visible`` 是真实宿主卡号，返回的卡号可直接喂给
    ``nvidia-smi -i``（宿主侧采样路径不变）。容器内 runtime 哨兵（``void``）
    的重编号场景走 :func:`select_sample_targets_for_inventory`。

    :param explicit: 调用方显式指定的采样卡（如 ``--gpus``）。
    :param visible: ``NVIDIA_VISIBLE_DEVICES`` 原始取值。
    :raises GpuLockError: visible 非法，或 explicit 越出 visible。
    """
    visible_devices = assert_gpu_lock(visible)
    if explicit is None or not explicit.strip() or explicit.strip().lower() == "all":
        return visible_devices

    requested = parse_device_list(explicit)
    outside = sorted(device for device in requested if device not in visible_devices)
    if outside:
        raise _reject(
            f"采样目标 {requested} 越出可见卡 {visible_devices}（越界: {outside}）",
            f"采样只允许落在 {_ENV_KEY} 之内；越界读取会混入可见集之外的卡或他人负载，"
            "使显存峰值数据不可信。",
        )
    return requested


def select_sample_targets_for_inventory(
    explicit: str | None,
    inventory: GpuInventory,
    *,
    max_cards: int | None = None,
) -> tuple[int, ...]:
    """容器内 runtime 收窄 + 重编号场景下的采样目标（F-2 修复点）。

    为什么不能沿用宿主卡号：runtime 会把宿主 GPU2 映射成容器内 index 0，
    容器内 ``nvidia-smi -i 2`` 于是报 ``exit status 6``（"没有这张卡"）。
    容器内的采样目标只能是**容器命名空间里的本地序号**，也就是
    ``GpuInventory.indices`` 给出的那一组。

    规则（防呆）：
    - ``inventory`` 先过 :func:`assert_inventory_visible`（0 卡即拒绝；
      ★ ``max_cards`` 默认 ``None`` = **不判卡数**——主席 2026-10-17「单个作业不限制卡数」）；
    - ``explicit`` 为 ``None`` / 空 / ``"all"`` 时，目标 = 全部可见卡；
    - 显式给出时，其元素必须 ⊆ 实测可见序号，否则拒绝——采样不得越出可见集
      （"只采可见卡、绝不带不带 ``-i`` 的全卡查询"这条铁律在这里继续成立）。

    :param explicit: 调用方显式指定的采样卡（容器内本地序号）。
    :param inventory: 容器内实测可见卡。
    :raises GpuLockError: 实测边界不满足，或 explicit 越出实测可见集。
    """
    observed = assert_inventory_visible(inventory, max_cards=max_cards)
    if explicit is None or not explicit.strip() or explicit.strip().lower() == "all":
        return observed

    requested = parse_device_list(explicit)
    outside = sorted(device for device in requested if device not in observed)
    if outside:
        raise _reject(
            f"采样目标 {requested} 越出实测可见卡 {observed}（越界: {outside}）",
            f"容器内采样只允许落在实测可见卡之内（来源 {inventory.source}）；"
            "越界查询会失败（重编号后按宿主卡号查必然 exit status 6）或混入他人负载。",
        )
    return requested
