"""GPU 锁卡守卫（fail-closed）—— 纯计算层，零设施依赖。

职责：把「这次运行允许用哪些 GPU」这条安全边界，编码成一条可复用、
**默认拒绝**的断言。任何训练 / 测试入口在启动前必须调用本模块。

**两种设备来源（缺一不可，§2.2 显式即防呆）：**

1. **宿主侧（``NVIDIA_VISIBLE_DEVICES`` 是卡号列表）**——守卫按**宿主卡号**
   静态比对：必须显式设置、索引 ⊆ ``{0..5}``（含生产卡 6 / 7 一律拒绝）、
   卡数 ≤ 4。宿主侧是唯一能按卡号判 6/7 的地方。
2. **容器侧（runtime 接管后）**——nvidia-container-runtime 在按设备收窄可见集
   时，会把容器内的 ``NVIDIA_VISIBLE_DEVICES`` 覆写成哨兵值 ``void``
   （**实测**：目标 GPU 服务器上加不加 ``--runtime=nvidia``、重复 ``-e`` 同变量都压不住；
   机器与环境记录见 `.local/` 与 `infra` skill）。
   ``void`` 的含义不是"没锁卡"，恰恰相反：它是 runtime **已经把可见集收窄**
   之后留下的标记。旧版守卫把 ``void`` 当成非法取值，于是目标 GPU 服务器上
   **所有训练入口必然拒绝启动**（false reject，见 F-1）——这是守卫的假设与
   runtime 语义不一致，不是调用方用错。

   容器侧的断言因此改为**以容器内实测可见卡为准**：``void`` 时必须给出
   ``GpuInventory``（由设施层 ``cli/gpu_monitor.probe_gpu_inventory`` 用
   ``nvidia-smi -L`` 探测），并按**卡数**（≤ :data:`MAX_CARDS`）+ **可见集非空**
   + **与显式声明卡数一致** 做断言。

**为什么 ``void`` 回落不放松防呆（否则就是拿安全换兼容）：**
runtime 写 ``void`` 的前提是它已按设备收窄；若收窄没收住（配方漏了设备限定），
容器会看到全部 8 张卡，此时 ``len(visible) = 8 > MAX_CARDS`` **仍然拒绝**。
未设置 / 空串 / ``all`` 仍然**一律拒绝**（``all`` 是镜像 ENV 的初值，见下）；
``nvidia-smi`` 探不到可见卡时 fail-closed；容器内本地序号（重编号后从 0 开始）
**不被解释成宿主卡号**，因此容器侧不再重复判 6/7——那由宿主侧按真实卡号负责，
容器侧改为断言"实测卡数 == 声明卡数"，多出任何一张卡都是拒绝信号。

为什么默认拒绝：本项目的运行镜像把 ``NVIDIA_VISIBLE_DEVICES`` 写死为
``all``，而 docker 默认 runtime 是 nvidia——不显式锁卡就会看到全部 8 张卡，
其中 GPU6/7 被生产 vLLM 占死。这不是"配置不当"，是"一碰就事故"，
所以守卫必须 fail-closed（宪法 §2 防呆设计：不靠调用方自觉）。

本模块不读环境、不调 ``nvidia-smi``、不碰 GPU——输入是字符串、映射与
**已探好的** ``GpuInventory``，输出是不可变设备元组，因此可在 CPU 上独立
单测（宪法 §1.3 层次边界）。环境与设施读取分别在 ``assert_gpu_lock_from_env``
（接收 mapping 以便注入）、``require_gpu_lock_or_exit``（接收探测回调以便注入）
与 ``cli/gpu_monitor.probe_gpu_inventory``。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

# ── 安全边界常量（单一真相源，§1.4）────────────────────────────────────────
#: ``NVIDIA_VISIBLE_DEVICES`` 允许出现的最大**宿主**卡号（0–5 可用，6/7 为生产卡）。
ALLOWED_MAX_INDEX = 5
#: 单次运行最多使用几张卡（用户拍板：每次最多 4 卡）。
MAX_CARDS = 4
#: 生产卡——被常驻 vLLM 占死，严禁触碰。
RESERVED_INDICES: tuple[int, ...] = (6, 7)
#: 4 卡首选集合——唯一全部位于 NUMA0。
PREFERRED_FOUR: tuple[int, ...] = (0, 1, 2, 3)

_ENV_KEY = "NVIDIA_VISIBLE_DEVICES"

#: 目标卡"实测空闲"阈值（F-10；与任务口径一致：>64 MiB 或 util>5% 即拒绝）。
IDLE_MEMORY_TOLERANCE_MIB = 64.0
IDLE_UTILIZATION_MAX_PCT = 5.0

#: runtime 的"已收窄可见集"哨兵值——**只认这两个字面量**，且只在容器侧回落
#: 路径上生效。``all`` 不在其中：它是镜像 ENV 的初值，含义是"没有锁卡"。
#: （实测确证：nvidia-container-runtime 在按设备收窄时写 ``void``；
#: ``none`` 是同一族语义的历史写法，一并接受并同样要求实测可见卡。）
_RUNTIME_MANAGED_MARKERS: frozenset[str] = frozenset({"void", "none"})

#: ``parse_device_list`` 的"显式整数卡号"结果之外的一种**受控**取值：
#: 调用方必须改用"实测可见卡"通道，不得把它当作卡号集合。
RUNTIME_MANAGED = "<runtime-managed>"

_USAGE_HINT = (
    "正确用法（显式指定 ≤4 张、且仅取自 {0,1,2,3,4,5} 的卡）：\n"
    "    NVIDIA_VISIBLE_DEVICES=0,1,2,3 <训练/测试命令>\n"
    "    run.sh <config.yaml> --gpus 0,1,2,3\n"
    "    docker run --gpus '\"device=0,1,2,3\"' ...\n"
    "  4 卡首选 {0,1,2,3}（唯一全部位于 NUMA0）；GPU6/7 为生产卡，永不使用。"
)


class GpuLockError(RuntimeError):
    """锁卡守卫拒绝启动（fail-closed）。错误消息必须可操作。"""


def _reject(reason: str, detail: str) -> GpuLockError:
    """构造统一格式的可操作拒绝消息。"""
    return GpuLockError(
        f"GPU 锁卡守卫拒绝启动（fail-closed）：\n"
        f"  原因: {reason}\n"
        f"  为什么必须拒绝: {detail}\n"
        f"  {_USAGE_HINT}"
    )


def parse_device_list(raw: str | None) -> tuple[int, ...]:
    """把设备列表字符串解析为整数元组；任何非显式、非整数的输入都拒绝。

    只接受形如 ``"0,1,2,3"`` 的整数索引列表。``None`` / 空串 / ``"all"`` /
    GPU UUID 一律拒绝——它们都意味着"没有显式锁卡"。

    **不返回** :data:`RUNTIME_MANAGED`：本函数是"宿主卡号"通道的入口，调用方
    （采样目标选择、eval 卡计划）要的是可直接比对的整数卡号。runtime 的
    ``void``/``none`` 哨兵由 :func:`resolve_device_source` 处理（它才是
    "按实测可见卡"的那条通道）。
    """
    value = _device_source_value(raw)
    if isinstance(value, tuple):
        return value
    raise _reject(
        f"{_ENV_KEY}={raw!r} 是 runtime 的托管哨兵值",
        "该取值不代表卡号；应按容器内实测可见卡判定（见 assert_gpu_lock_inventory），"
        "或用显式宿主卡号。",
    )


def _device_source_value(raw: str | None) -> tuple[int, ...] | str:
    """把 ``NVIDIA_VISIBLE_DEVICES`` 取值归一为"显式卡号元组"或 :data:`RUNTIME_MANAGED`。

    唯一允许回落为 :data:`RUNTIME_MANAGED` 的取值是 :data:`_RUNTIME_MANAGED_MARKERS`
    中的字面量。其它一律抛 :class:`GpuLockError`（fail-closed）：

    - ``None`` / 空串：本镜像 ENV 初值是 ``all``，没有取值就无从判断边界 ⇒ 拒绝；
    - ``all``：会暴露全部 8 张卡（含生产卡 GPU6/7）⇒ 拒绝；
    - UUID / 非整数 / 空元素 / 重复卡号：无法与生产卡边界做静态比对 ⇒ 拒绝。
    """
    if raw is None:
        raise _reject(
            f"环境变量 {_ENV_KEY} 未设置",
            "本镜像 ENV 默认 NVIDIA_VISIBLE_DEVICES=all，不显式锁卡会看到全部 8 张卡，"
            "其中 GPU6/7 被生产 vLLM 占死——触碰即事故。",
        )
    text = raw.strip()
    if not text:
        raise _reject(
            f"环境变量 {_ENV_KEY} 为空字符串",
            "空值等同于未锁卡（镜像 ENV 初值为 all），无法确定这次运行会用到哪些卡。",
        )
    lowered = text.lower()
    if lowered == "all":
        raise _reject(
            f"{_ENV_KEY}=all",
            "all 会暴露全部 8 张卡（含生产卡 GPU6/7）；必须显式列出要用的卡。",
        )
    if lowered in _RUNTIME_MANAGED_MARKERS:
        # runtime 已按设备收窄可见集后写入的哨兵；不是卡号，必须转"实测可见卡"通道。
        return RUNTIME_MANAGED

    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise _reject(
            f"{_ENV_KEY}={raw!r} 含空元素",
            "设备列表里出现空位（如 '0,,1'）说明取值有误，拒绝猜测调用方意图。",
        )
    devices: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise _reject(
                f"{_ENV_KEY} 含非整数索引 {part!r}",
                "本守卫只接受整数卡号；UUID / 'all' 无法与生产卡边界做静态比对。",
            )
        devices.append(int(part))
    if len(set(devices)) != len(devices):
        raise _reject(
            f"{_ENV_KEY}={raw!r} 含重复卡号",
            "重复卡号说明取值有误（同一张卡不能占两次），拒绝猜测。",
        )
    return tuple(devices)


def assert_gpu_lock(
    raw: str | None,
    *,
    allowed_max_index: int = ALLOWED_MAX_INDEX,
    max_cards: int = MAX_CARDS,
) -> tuple[int, ...]:
    """校验**宿主卡号列表**满足锁卡边界，返回设备元组；不满足即抛错。

    三条边界（全部 fail-closed）：显式设置、索引 ⊆ ``{0..allowed_max_index}``、
    卡数 ≤ ``max_cards``。

    :param raw: ``NVIDIA_VISIBLE_DEVICES`` 的原始取值（``None`` = 未设置）。
    :raises GpuLockError: 任一边界不满足（含 runtime 的 ``void`` 哨兵——本函数
        要求真实卡号；容器侧请用 :func:`assert_gpu_lock_inventory`）。
    """
    devices = parse_device_list(raw)

    reserved = sorted(device for device in devices if device > allowed_max_index)
    if reserved:
        raise _reject(
            f"设备列表 {devices} 含生产卡 {reserved}（允许上限 {allowed_max_index}）",
            f"GPU{allowed_max_index + 1} 起为生产卡（当前 GPU6/7 被常驻 vLLM 占死），"
            "在其中做测试会打断生产任务，且结果混入他人负载。",
        )
    if len(devices) > max_cards:
        raise _reject(
            f"设备列表 {devices} 使用 {len(devices)} 张卡，超过上限 {max_cards}",
            f"用户拍板每次最多 {max_cards} 卡；更多卡会挤占生产余量且超出本期验证口径。",
        )
    return devices


@dataclass(frozen=True)
class GpuInventory:
    """容器内**实测可见卡**的探针结果（见 ``cli/gpu_monitor.probe_gpu_inventory``）。

    "实测"意味着不信任环境变量：runtime 会重编号，容器内本地序号与宿主卡号
    可以完全不同（宿主 GPU2 → 容器 index 0）。这里只带**容器命名空间内**的
    信息：``count`` 是可见卡数，``indices`` 是容器内本地序号（``nvidia-smi -L``
    的顺序）。

    :param source: 探测来源标识（``"nvidia-smi"`` / ``"torch"``），仅用于消息。
    :param count: 实测可见卡数。
    :param indices: 实测可见卡的容器内本地序号；未知时为空元组。
    """

    source: str
    count: int
    indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class DeviceSource:
    """设备来源判定结果：``explicit`` 与 ``inventory`` 恰有一个非 None（§2.2）。"""

    explicit: tuple[int, ...] | None
    inventory: GpuInventory | None
    reason: str


def resolve_device_source(
    raw: str | None,
    inventory: GpuInventory | None = None,
) -> DeviceSource:
    """判定"这次运行的设备集合"从哪来：显式宿主卡号，还是 runtime 收窄后的实测卡。

    - 显式整数列表 → ``explicit``（宿主侧语义：卡号就是宿主卡号）；
    - runtime 哨兵（``void``/``none``）→ 必须给出 ``inventory``；缺失即拒绝
      （fail-closed：探不到实测卡就不启动，绝不"当作没锁卡放行"）；
    - 其它非法取值 → :class:`GpuLockError`。

    :param raw: ``NVIDIA_VISIBLE_DEVICES`` 原始取值。
    :param inventory: 容器内实测可见卡（由设施层探测后注入，便于纯逻辑测试）。
    :raises GpuLockError: 取值非法，或哨兵值下缺少实测卡。
    """
    value = _device_source_value(raw)
    if isinstance(value, tuple):
        # 显式卡号通道：判定依据是卡号本身；若同时给了实测可见卡（容器内也做了
        # 实测），则一并带上供"声明卡数 == 实测卡数"这条一致性检查使用。
        return DeviceSource(explicit=value, inventory=inventory, reason=f"{_ENV_KEY} 显式卡号")

    if inventory is None:
        raise _reject(
            f"{_ENV_KEY}={raw!r} 表示可见集已被 runtime 收窄，但未提供实测可见卡",
            "无法在容器内确认到底看得见几张卡；不确定就不启动（fail-closed）。",
        )
    return DeviceSource(
        explicit=None,
        inventory=inventory,
        reason=f"{_ENV_KEY}={raw!r}（runtime 收窄，按实测可见卡判定）",
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


def select_sample_targets(
    explicit: str | None,
    visible: str | None,
) -> tuple[int, ...]:
    """确定显存采样目标卡：只允许落在 ``NVIDIA_VISIBLE_DEVICES`` 之内。

    规则（防呆）：
    - ``visible`` 必须先通过锁卡守卫（未设置 / 含 6,7 / >4 卡即拒绝）；
    - ``explicit`` 为 ``None`` / 空 / ``"all"`` 时，采样全部可见卡；
    - ``explicit`` 显式列出时，必须 **⊆ visible**，否则拒绝——采样越出可见集
      正是"混入生产卡显存"的根因（历史数据污染事故）。

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
            f"采样只允许落在 {_ENV_KEY} 之内；越界读取会混入生产卡 GPU6/7 或他人负载，"
            "使显存峰值数据不可信。",
        )
    return requested


def select_sample_targets_for_inventory(
    explicit: str | None,
    inventory: GpuInventory,
    *,
    max_cards: int = MAX_CARDS,
) -> tuple[int, ...]:
    """容器内 runtime 收窄 + 重编号场景下的采样目标（F-2 修复点）。

    为什么不能沿用宿主卡号：runtime 会把宿主 GPU2 映射成容器内 index 0，
    容器内 ``nvidia-smi -i 2`` 于是报 ``exit status 6``（"没有这张卡"）。
    容器内的采样目标只能是**容器命名空间里的本地序号**，也就是
    ``GpuInventory.indices`` 给出的那一组。

    规则（防呆）：
    - ``inventory`` 先过 :func:`assert_inventory_visible`（0 卡 / >4 卡即拒绝）；
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


def format_verdict(devices: Sequence[int], *, source: str = _ENV_KEY) -> str:
    """生成通过守卫后的可读确认行（供 CLI / 日志复用）。"""
    return (
        f"[gpu-guard] OK: {source}={','.join(str(d) for d in devices)} "
        f"（{len(devices)} 卡，全部 ⊆ 0..{ALLOWED_MAX_INDEX}，≤{MAX_CARDS} 卡上限）"
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
