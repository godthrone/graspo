"""GPU 锁卡守卫 · **契约与纯策略**（单一真相源）—— 纯计算层，零设施依赖。

★★ 一名一物（gpu_guard 双家消除）：
- **契约 + 纯策略（本文件，`core/`）**：允许 / 预留卡号集合、卡计划数据模型
  （:class:`GpuPlan`）、异常类型（:class:`GpuLockError` / :class:`GpuGuardError`）、
  显式卡号的纯判定（:func:`assert_gpu_lock`）、卡计划解析（:func:`parse_gpu_plan` /
  :func:`resolve_gpu_plan`）、盘点数据模型（:class:`GpuInventory` /
  :class:`DeviceSource`）与来源归一（:func:`resolve_device_source`）。
  本文件**不读环境、不调 nvidia-smi、不碰 GPU、不 import flow**。
- **设施操作（`flow/guard/`）**：环境映射读取、runtime 哨兵回落、实测可见卡盘点
  探测、进程退出、采样目标选择、空闲断言 —— 见 ``graspo.flow.guard``。

为什么这样切（宪法 §1.3 计算/设施分离、§1.4 单一真相源）：
``core/`` 是跨层契约，不得反向依赖 ``flow/``（否则 ``core/schema.py`` 无法在
不拉起设施栈的前提下校验配置）；而"读环境 / 探测卡 / 退出进程"是设施操作，
必须住设施层。**同一个量只有一个解析入口**，两处不得各写一份。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

# ── 安全边界常量（单一真相源，§1.4）────────────────────────────────────────
# ★★ 卡位契约（正式口径；用户 2026-09-24 批准 A，2026-10-08 B7 收口）：
#   **允许卡 = {4,5,6,7}；其余一律拒绝。**
#   · 0–3：第三方工作负载**先到先得**，**不属**任何形式的「保留卡」；
#   · ≥8：**不存在**（8 卡机只有 0–7）。
#   ★「生产卡 / reserved」语义**已退役**（生产 vLLM 已 Exited 并释放 6/7）：
#     本文件只认**允许集合**这一个判据，不再有第二份"保留卡"名单。
#     :data:`RESERVED_INDICES` 保留常量名仅为兼容引用，恒为空元组。
#: 变体1（用户 2026-09-24 批准 A）：允许的宿主卡号**显式白名单**。
#: 取代原来的「仅上限」判据（原值 5、判据 device > max）；语义见 assert_gpu_lock。
ALLOWED_INDICES: tuple[int, ...] = (4, 5, 6, 7)
#: 允许卡号的**配置来源**（环境变量）；未设置 ⇒ 回落 :data:`ALLOWED_INDICES`。
ALLOWED_INDICES_ENV = "GRASPO_ALLOWED_GPU_INDICES"
#: 兼容既有签名的上界（= max(ALLOWED_INDICES)）；真正的判据是上面的白名单。
ALLOWED_MAX_INDEX = max(ALLOWED_INDICES)
#: 单次运行最多使用几张卡（用户拍板：每次最多 4 卡）。
MAX_CARDS = 4
#: 变体1 下不再有「生产卡」（用户批准占用 6/7；生产 vLLM 已 Exited）。
#: ★ B7 起该语义**正式退役**：唯一判据是 :data:`ALLOWED_INDICES`；本常量恒为空元组，
#: 保留名字仅为兼容既有 import（含 ``graspo.eval.guard`` 的透传别名）。
RESERVED_INDICES: tuple[int, ...] = ()
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
    "正确用法（显式指定 ≤4 张、且仅取自 {4,5,6,7} 的卡）：\n"
    "    NVIDIA_VISIBLE_DEVICES=4,5,6,7 <训练/测试命令>\n"
    "    run.sh <config.yaml> --gpus 4,5,6,7\n"
    "    docker run --gpus '\"device=4,5,6,7\"' ...\n"
    "  本期允许集合 {4,5,6,7}（用户 2026-09-24 批准 A）；GPU0–3 留给其他工作负载。"
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


def resolve_allowed_indices(env: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """生效的允许卡号集合：配置优先，未配置/空串回落 :data:`ALLOWED_INDICES`。

    这是"允许集合"这**一个量**的唯一解析入口；不设 :data:`ALLOWED_INDICES_ENV` 时
    返回值与 :data:`ALLOWED_INDICES` **逐字相等**（默认策略一个字不改）。
    ⛔ fail-closed：取值非法一律抛 :class:`GpuLockError`，⛔ 不静默回落、⛔ 不放行。
    ⛔ 只放宽"允许集合"这一个量：越界/卡数上限/显式设置/哨兵处理**均未动**。
    """
    source = os.environ if env is None else env
    raw = source.get(ALLOWED_INDICES_ENV)
    if raw is None or not raw.strip():
        return ALLOWED_INDICES
    parts = [part.strip() for part in raw.strip().split(",")]
    if any(not part.isdigit() for part in parts):
        raise _reject(
            f"{ALLOWED_INDICES_ENV}={raw!r} 取值非法",
            "该变量只接受逗号分隔的整数卡号列表（如 '0,1,2,3'）；"
            "取值非法时拒绝猜测意图（fail-closed）。",
        )
    return tuple(int(part) for part in parts)


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
    - ``all``：会暴露全部 8 张卡（含允许集合外的 GPU0–3）⇒ 拒绝；
    - UUID / 非整数 / 空元素 / 重复卡号：无法与允许集合做静态比对 ⇒ 拒绝。
    """
    if raw is None:
        raise _reject(
            f"环境变量 {_ENV_KEY} 未设置",
            "本镜像 ENV 默认 NVIDIA_VISIBLE_DEVICES=all，不显式锁卡会看到全部 8 张卡，"
            "其中 GPU0–3 属第三方工作负载（先到先得）——必须显式锁定在允许集合 {4,5,6,7} 内。",
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
            "all 会暴露全部 8 张卡（含允许集合外的 GPU0–3）；必须显式列出要用的卡。",
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
                "本守卫只接受整数卡号；UUID / 'all' 无法与允许集合做静态比对。",
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
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """校验**宿主卡号列表**满足锁卡边界，返回设备元组；不满足即抛错。

    三条边界（全部 fail-closed）：显式设置、索引 ⊆ ``{0..allowed_max_index}``、
    卡数 ≤ ``max_cards``。

    :param raw: ``NVIDIA_VISIBLE_DEVICES`` 的原始取值（``None`` = 未设置）。
    :raises GpuLockError: 任一边界不满足（含 runtime 的 ``void`` 哨兵——本函数
        要求真实卡号；容器侧请用 :func:`assert_gpu_lock_inventory`）。
    """
    devices = parse_device_list(raw)

    # ★ 变体1（用户 2026-09-24 批准 A）：**显式白名单**取代「仅上限」判据——
    #   判据由 `device > allowed_max_index` 改为 `device not in ALLOWED_INDICES`。
    #   仅当调用方沿用默认上界时启用（显式传入更小上界的调用方仍按原语义收窄，不放松、
    #   也不被本白名单误伤——例如其他工具/测试显式使用 GPU0–3 的场景）。
    # ★ 生效允许集合；未设变量时 == ALLOWED_INDICES ⇒ 判定与消息逐字节不变。
    allowed = (
        resolve_allowed_indices(env)
        if allowed_max_index == ALLOWED_MAX_INDEX
        else ALLOWED_INDICES
    )
    if allowed_max_index == ALLOWED_MAX_INDEX:
        outside = sorted(device for device in devices if device not in allowed)
        if outside:
            raise _reject(
                f"设备列表 {devices} 含白名单外的卡 {outside}（允许集合 {allowed}）",
                "本期只允许使用 GPU4–7（用户 2026-09-24 批准 A）；GPU0–3 留给其他工作负载，GPU8 起不存在。",
            )
    reserved = sorted(device for device in devices if device > allowed_max_index)
    if reserved:
        raise _reject(
            f"设备列表 {devices} 含上界外卡 {reserved}（允许集合 {allowed}，上界 {allowed_max_index}）",
            "超出允许集合的卡一律拒绝（fail-closed）。",
        )
    if len(devices) > max_cards:
        raise _reject(
            f"设备列表 {devices} 使用 {len(devices)} 张卡，超过上限 {max_cards}",
            f"用户拍板每次最多 {max_cards} 卡；更多卡会挤占生产余量且超出本期验证口径。",
        )
    return devices


@dataclass(frozen=True)
class GpuInventory:
    """容器内**实测可见卡**的探针结果（见 ``flow/guard/inventory.probe_gpu_inventory``）。

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


# ── 卡计划契约（★ 唯一定义处；原 `eval/guard.py` 的定义已迁入本文件）─────────
# 为什么在 core：`core/schema.py` 的 `EvalConfig` 必须能在**不 import flow** 的
# 前提下校验卡计划（宪法 §1.3/§1.4）；评测链路 `eval/guard.py` 只留 nvidia-smi 采样。


class GpuGuardError(RuntimeError):
    """卡守卫/采样的**契约层**异常。调用方必须让它终止流程——不要 catch 后继续跑。

    ``GpuLockError``（纯策略层）在 :func:`resolve_gpu_plan` 里被转成本异常，
    这样调用方只需捕获一种异常类型；规则文案原样保留。
    """


class GpuPlan(BaseModel):
    """一次 GPU 任务的锁卡计划。构造即校验，非法组合不可能存在。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: 显式指定的物理卡索引（保持调用方给出的顺序）。
    devices: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.devices)

    @property
    def csv(self) -> str:
        """逗号分隔形式，用于 ``NVIDIA_VISIBLE_DEVICES`` / ``docker --gpus=``。"""
        return ",".join(str(index) for index in self.devices)

    def docker_gpus_flag(self) -> str:
        """Docker ``--gpus`` 取值。用 ``device=0,1`` 形式（不是 ``all``）。"""
        return f'"device={self.csv}"'


def parse_gpu_plan(raw: str | None) -> GpuPlan:
    """把显式卡列表转成 ``GpuPlan``；规则校验交给 :func:`assert_gpu_lock`。

    Args:
        raw: 形如 ``"0,1"`` 的字符串。``None`` / 空串 = 未显式指定 → 拒绝。

    Returns:
        ``GpuPlan``。

    Raises:
        GpuGuardError: 规则层拒绝（未指定 / 含允许集合外的卡 / 卡数超限 / 取值非法）。
    """
    try:
        devices = assert_gpu_lock(raw)
    except GpuLockError as exc:
        raise GpuGuardError(str(exc)) from None
    return GpuPlan(devices=devices)


def resolve_gpu_plan(raw: str | None) -> GpuPlan:
    """从**配置**解析卡计划。

    与 :func:`parse_gpu_plan` 同义，但语义上强调"卡计划只来自配置"（宪法 §7.1
    单一配置入口）：不接受环境变量、不接受隐式默认值。分开命名是为了让调用点
    一眼看出"这里的卡来自 config"，而不是某个散落的环境变量。
    """
    return parse_gpu_plan(raw)
