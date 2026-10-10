"""flow.guard.lock — 锁卡守卫的**设施操作**（环境读取 / 盘点探测 / 进程退出）。

规则本身在 ``graspo.core.gpu_guard``（单一真相源，宪法 §1.4）；本模块只负责把
规则接到真实环境与进程上：读 ``NVIDIA_VISIBLE_DEVICES``、按需探测容器内实测
可见卡、必要时 fail-closed 退出。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence

from graspo.core.gpu_guard import (
    ALLOWED_MAX_INDEX,
    MAX_CARDS,
    RUNTIME_MANAGED,
    GpuInventory,
    GpuLockError,
    _ENV_KEY,
    _device_source_value,
    _reject,
    assert_gpu_lock,
    resolve_allowed_indices,
    resolve_device_source,
)

def assert_inventory_visible(
    inventory: GpuInventory,
    *,
    max_cards: int = MAX_CARDS,
    declared_count: int | None = None,
) -> tuple[int, ...]:
    """按**实测可见卡**做断言（F-1 容器侧通道），返回该卡集合。

    三条边界（全部 fail-closed）：

    1. 实测卡数为 0（探到空集）→ 拒绝——看不见卡说明探测或 runtime 有问题；
    2. 实测卡数 > :data:`MAX_CARDS` → 拒绝——"不得超 4 卡"在实测卡上仍然成立；
    3. ``declared_count`` 给出时，实测卡数必须与之**相等**——runtime 可以重编号
       （宿主 2 → 容器 0），但**不改变卡数**；多一张就是把没声明的卡暴露进来了。

    ⚠️ **可达性声明（2026-09-18 🟡-3 复核修正）**：第 3 条**不是生产入口的防线**，
    只在**低层 API** 生效。生产入口 :func:`require_gpu_lock_or_exit` 经
    :func:`_probe_inventory_for` 探测实测卡，而该函数**只在** ``NVIDIA_VISIBLE_DEVICES``
    是 runtime 哨兵（``void``/``none``）时才探测；**显式卡号路径恒传 ``inventory=None``**
    ⇒ 此时 `declared_count is None`、第 3 条永不执行。低层 API 调用方若显式传入
    ``inventory``（如 :func:`assert_gpu_lock_inventory` 的注入式调用），第 3 条才可达。
    为什么不把它接到显式卡号路径上：显式卡号路径是**宿主侧**语义（_ENV_KEY 在宿主上
    只是普通环境变量，runtime 不参与收窄），此时探针读到的是宿主全部卡，与"声明卡数"
    本就不该相等——接上会把合法的宿主侧单卡/双卡运行全部误拒。定位与实测证据：
    `.local/hb-workspace/20260915-152747/review-round2-fixes/review-report.md` 🟡-3。

    容器内本地序号**不做** 6/7 比对：重编号后本地 0..n-1 与宿主卡号没有对应关系，
    按本地序号判 6/7 只会得到错误的结论。6/7 的判定由宿主侧
    :func:`assert_gpu_lock` 按真实宿主卡号负责（宿主侧是唯一持有真实卡号的层级）。

    :raises GpuLockError: 任一边界不满足。
    """
    if inventory.count <= 0:
        raise _reject(
            f"实测可见卡数为 {inventory.count}（来源 {inventory.source}）",
            "看不见任何卡说明 runtime 未生效或探测失败；不确定就不启动（fail-closed）。",
        )
    if inventory.count > max_cards:
        raise _reject(
            f"实测可见 {inventory.count} 张卡，超过上限 {max_cards}"
            f"（来源 {inventory.source}，本地序号 {list(inventory.indices)}）",
            f"用户拍板每次最多 {max_cards} 卡；可见集没收住（可能配方未限定设备）时，"
            "GPU6/7 生产卡就在这堆卡里，必须拒绝。",
        )
    if declared_count is not None and inventory.count != declared_count:
        raise _reject(
            f"实测可见 {inventory.count} 张卡 != 显式声明的 {declared_count} 张"
            f"（来源 {inventory.source}）",
            "runtime 收窄不改变卡数；两者不一致说明可见集与声明不符（多出或丢卡），"
            "在确认前拒绝启动。",
        )
    return inventory.indices if inventory.indices else tuple(range(inventory.count))


def is_runtime_managed(raw: str | None) -> bool:
    """``raw`` 是否是 runtime 的"已收窄可见集"哨兵（``void``/``none``）。

    供设施层判断"该走实测可见卡通道还是宿主卡号通道"，避免两处各写一份判据
    （§1.4 单一真相源）。非法取值一律返回 ``False``——它们不是哨兵，各自的
    判定会把它们拒绝掉。
    """
    try:
        return _device_source_value(raw) is RUNTIME_MANAGED
    except GpuLockError:
        return False


def assert_gpu_lock_inventory(
    raw: str | None,
    inventory: GpuInventory | None = None,
    *,
    allowed_max_index: int = ALLOWED_MAX_INDEX,
    max_cards: int = MAX_CARDS,
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """入口守卫的统一判定：显式卡号走静态比对，runtime 哨兵走实测可见卡。

    这是训练 / 测试入口应调用的判定函数——它是 F-1 的修复点：容器内
    ``NVIDIA_VISIBLE_DEVICES=void``（runtime 已收窄）不再被误判为"未锁卡"，
    而是按实测可见卡断言；同时**不放松**任何既有限制（见 :func:`assert_inventory_visible`）。

    :param raw: ``NVIDIA_VISIBLE_DEVICES`` 原始取值。
    :param inventory: 容器内实测可见卡（设施层探测；宿主侧显式卡号时为 None）。
    :raises GpuLockError: 任一边界不满足。
    """
    source = resolve_device_source(raw, inventory)
    if source.explicit is not None:
        devices = assert_gpu_lock(
            raw,
            allowed_max_index=allowed_max_index,
            max_cards=max_cards,
            env=env,
        )
        if source.inventory is not None:
            # 有实测可见卡时，声明卡数必须与实测一致（多一张 = 可见集失守）。
            # ⚠️ 🟡-3：本分支**只在调用方显式传入 inventory 时可达**；生产入口
            # `require_gpu_lock_or_exit` 的显式卡号路径恒传 None（见
            # `assert_inventory_visible` docstring 的可达性声明）——不得再把这条
            # 当成生产入口防线对外宣称。
            assert_inventory_visible(
                source.inventory,
                max_cards=max_cards,
                declared_count=len(devices),
            )
        return devices
    assert source.inventory is not None  # resolve_device_source 保证
    return assert_inventory_visible(
        source.inventory,
        max_cards=max_cards,
    )


def assert_gpu_lock_from_env(
    env: Mapping[str, str] | None = None,
    *,
    allowed_max_index: int = ALLOWED_MAX_INDEX,
    max_cards: int = MAX_CARDS,
) -> tuple[int, ...]:
    """从环境映射读取 ``NVIDIA_VISIBLE_DEVICES`` 并按**显式宿主卡号**校验。

    接收 mapping 而非直接读 ``os.environ``，是为了让调用方与测试能注入
    确定的环境（宪法 §2.2 显式即防呆）。

    本函数**不**处理 runtime 的 ``void`` 哨兵（它要求真实卡号，见 :func:`assert_gpu_lock`）；
    需要容器侧回落语义的入口请用 :func:`require_gpu_lock_or_exit`。
    """
    source = os.environ if env is None else env
    return assert_gpu_lock(
        source.get(_ENV_KEY),
        allowed_max_index=allowed_max_index,
        max_cards=max_cards,
        env=source,
    )


def require_gpu_lock_or_exit(
    env: Mapping[str, str] | None = None,
    *,
    allowed_max_index: int = ALLOWED_MAX_INDEX,
    max_cards: int = MAX_CARDS,
    inventory_probe: Callable[[], GpuInventory] | None = None,
) -> tuple[int, ...]:
    """训练/测试入口使用的守卫包装：通过返回设备，不通过即可操作地终止。

    判定走 :func:`assert_gpu_lock_inventory`——即同时支持"宿主侧显式卡号"与
    "容器内 runtime ``void`` + 实测可见卡"两种来源（F-1）。

    :param env: 环境映射（``None`` = ``os.environ``）。
    :param inventory_probe: 探测容器内实测可见卡的可调用对象；``None`` 时用
        ``cli.gpu_monitor.probe_gpu_inventory``（延迟导入，避免本模块依赖设施层）。
        仅在 ``NVIDIA_VISIBLE_DEVICES`` 是 runtime 哨兵值时才会被调用。
    :raises SystemExit: 守卫拒绝（退出码 1，消息原样打印到 stderr）。
    """
    source = os.environ if env is None else env
    raw = source.get(_ENV_KEY)
    try:
        inventory = _probe_inventory_for(raw, inventory_probe)
        return assert_gpu_lock_inventory(
            raw,
            inventory,
            allowed_max_index=allowed_max_index,
            max_cards=max_cards,
            env=source,
        )
    except GpuLockError as exc:
        raise SystemExit(str(exc)) from None


def _probe_inventory_for(
    raw: str | None,
    probe: Callable[[], GpuInventory] | None,
) -> GpuInventory | None:
    """仅在需要时探测容器内实测可见卡；显式宿主卡号路径不做任何设施调用。

    ⚠️ **可达性声明（2026-09-18 🟡-3 复核修正）**：本函数**只在**
    ``NVIDIA_VISIBLE_DEVICES`` 是 runtime 哨兵（``void``/``none``）时返回实测卡；
    **显式卡号路径恒返回 ``None``**。其后果：`assert_inventory_visible` 的
    "声明卡数 ≠ 实测卡数 ⇒ 拒" 这一段，在**生产入口**
    :func:`require_gpu_lock_or_exit` 的显式卡号路径上**不可达**，只在低层 API
    （调用方显式传入 ``inventory``）生效。**不得**把该条当作生产入口防线对外宣称。
    为什么不接到显式路径上：显式路径是宿主侧语义，探针读到的是宿主全部卡，
    接上会误拒合法的宿主侧单卡/双卡运行（详见
    :func:`assert_inventory_visible` 的可达性声明）。
    """
    try:
        needs_inventory = _device_source_value(raw) is RUNTIME_MANAGED
    except GpuLockError:
        return None  # 判定会再次抛出同一个错误，无需探测。
    if not needs_inventory:
        return None
    if probe is None:
        from graspo.cli.gpu_monitor import probe_gpu_inventory  # noqa: PLC0415

        probe = probe_gpu_inventory
    try:
        return probe()
    except RuntimeError as exc:
        raise _reject(
            f"容器内实测可见卡探测失败：{exc}",
            "拿不到实测可见卡就无法确认边界；不确定就不启动（fail-closed）。",
        ) from None


def format_verdict(devices: Sequence[int], *, source: str = _ENV_KEY) -> str:
    """生成通过守卫后的可读确认行（供 CLI / 日志复用）。"""
    allowed = resolve_allowed_indices()
    return (
        f"[gpu-guard] OK: {source}={','.join(str(d) for d in devices)} "
        f"（{len(devices)} 卡，全部 ∈ {allowed}，≤{MAX_CARDS} 卡上限）"
    )
