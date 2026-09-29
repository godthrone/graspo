"""GPU 锁卡守卫（fail-closed）—— 规则是纯计算，探测是唯一的设施入口。

职责：把「这次运行允许用哪些 GPU」这条安全边界，编码成一条可复用、
**默认拒绝**的断言。任何训练 / 测试入口在启动前必须调用本模块。

**两种设备来源（缺一不可，§2.2 显式即防呆）：**

1. **宿主侧（``NVIDIA_VISIBLE_DEVICES`` 是卡号列表）**——守卫按**宿主卡号**
   静态比对：必须显式设置、索引 ⊆ **生效的允许集合**（默认 ``{4,5,6,7}``，
   可用 :data:`ALLOWED_INDICES_ENV` 覆盖）、索引 ∉ **生效的保留集合**
   （默认空，可用 :data:`RESERVED_INDICES_ENV` 覆盖）、卡数 ≤ 4。
   宿主侧是唯一能按真实卡号做边界判定的地方。
   ★ 允许/保留集合是**部署事实**，因此来自配置而非硬编码元组——把某台机器的
   卡号写死进共享代码会让其他机器/夹具必然误拒（见 :func:`resolve_allowed_indices`）。
2. **容器侧（runtime 接管后）**——nvidia-container-runtime 在按设备收窄可见集
   时，会把容器内的 ``NVIDIA_VISIBLE_DEVICES`` 覆写成哨兵值 ``void``
   （**实测**：目标 GPU 服务器上加不加 ``--runtime=nvidia``、重复 ``-e`` 同变量都压不住；
   机器与环境记录见 `.local/` 与 `infra` skill）。
   ``void`` 的含义不是"没锁卡"，恰恰相反：它是 runtime **已经把可见集收窄**
   之后留下的标记。旧版守卫把 ``void`` 当成非法取值，于是目标 GPU 服务器上
   **所有训练入口必然拒绝启动**（false reject，见 F-1）——这是守卫的假设与
   runtime 语义不一致，不是调用方用错。

   容器侧的断言因此改为**以容器内实测可见卡为准**：``void`` 时必须给出
   ``GpuInventory``（由 :func:`probe_gpu_inventory` 用 ``nvidia-smi -L`` 探测），
   并按**卡数**（≤ :data:`MAX_CARDS`）+ **可见集非空**
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

**规则是纯计算，探测是唯一的设施入口。** 判定函数不调 ``nvidia-smi``、不碰 GPU——
输入是字符串、映射与**已探好的** ``GpuInventory``，输出是不可变设备元组，因此规则
部分可在 CPU 上独立单测（宪法 §1.3 层次边界）。**唯一的环境读取是"允许/保留集合"**
这两个配置键（:data:`ALLOWED_INDICES_ENV` / :data:`RESERVED_INDICES_ENV`，见
:func:`resolve_allowed_indices`）：所有判定函数都接受 ``env`` mapping，未注入时才
回落 ``os.environ``；不读任何其它环境变量。设备相关的环境读取在
``assert_gpu_lock_from_env``（接收 mapping 以便注入）与 ``require_gpu_lock_or_exit``
（接收探测回调以便注入）。

**本模块内唯一的设施代码是 :func:`probe_gpu_inventory`**（``nvidia-smi -L`` 实测
容器内可见卡），以及评测链路的卡计划契约（:class:`GpuPlan` / :func:`resolve_gpu_plan`，
无设施调用）。它们留在 core，是为了让"谁来探测 / 谁来评判"不再有方向问题：下游
（``cli`` / ``eval``）都从 core 取用，**core 不反向导入任何上层模块**
（``tests/test_ast_boundary.py`` 把 ``core → cli`` / ``core → eval`` / ``core → ripple``
判为跨层违规）。**2026-09-29 D-03 修复**：修复前 core 通过延迟导入
``cli.gpu_monitor.probe_gpu_inventory`` 与 ``eval.guard`` / ``ripple.reward`` 取默认值，
构成 ``core → {cli, eval, ripple}`` 反向依赖；现在方向是 ``{cli, eval} → core``（合法）。
:func:`require_gpu_lock_or_exit` 仍接受 ``inventory_probe=`` 注入，未注入时才用本模块
的 :func:`probe_gpu_inventory`——注入点与判据都没变，只是默认值的来源方向变对了。
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

# ── 安全边界常量（单一真相源，§1.4）────────────────────────────────────────
#: 允许索引的**配置来源**（环境变量）：取值形如 ``"0,1,2,3"``。
#: 为什么可配置：允许集合是**部署事实**（哪台机器上哪几张卡可用），不是代码常量。
#: 把某台机器的元组硬编码进共享代码，会让另一台机器（或本机夹具）必然误拒——
#: 2026-09-28 的 59 项单测失败正是这样来的。部署事实由部署方通过本变量声明。
ALLOWED_INDICES_ENV = "GRASPO_ALLOWED_GPU_INDICES"
#: 保留（不可用/生产）索引的配置来源；与允许集合同样属部署事实。
RESERVED_INDICES_ENV = "GRASPO_RESERVED_GPU_INDICES"

#: 允许的宿主卡号**默认白名单**（= 变体1，用户 2026-09-24 批准 A 的部署事实）。
#: **只是默认值**：可用 :data:`ALLOWED_INDICES_ENV` 覆盖（见 :func:`resolve_allowed_indices`）。
DEFAULT_ALLOWED_INDICES: tuple[int, ...] = (4, 5, 6, 7)
#: 默认保留集合：变体1 下 6/7 由用户批准占用（生产 vLLM 已 Exited）⇒ 默认为空。
#: 可用 :data:`RESERVED_INDICES_ENV` 覆盖（例如把某台机器的 6/7 声明为不可碰）。
DEFAULT_RESERVED_INDICES: tuple[int, ...] = ()

#: 向后兼容常量：默认策略的别名，**不再作为运行时判据的唯一来源**。
#: 运行时请用 :func:`resolve_allowed_indices` / :func:`resolve_reserved_indices`
#: （它们读配置、带默认值）。保留常量名是为了不打断既有引用点。
ALLOWED_INDICES: tuple[int, ...] = DEFAULT_ALLOWED_INDICES
#: 兼容既有签名的上界（= max(默认允许集合)）；真正的判据是白名单。
ALLOWED_MAX_INDEX = max(DEFAULT_ALLOWED_INDICES)
#: 单次运行最多使用几张卡（用户拍板：每次最多 4 卡）。
MAX_CARDS = 4
RESERVED_INDICES: tuple[int, ...] = DEFAULT_RESERVED_INDICES
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

_DEFAULT_USAGE_HINT = (
    "正确用法（显式指定 ≤4 张、且仅取自 {4,5,6,7} 的卡）：\n"
    "    NVIDIA_VISIBLE_DEVICES=4,5,6,7 <训练/测试命令>\n"
    "    run.sh <config.yaml> --gpus 4,5,6,7\n"
    "    docker run --gpus '\"device=4,5,6,7\"' ...\n"
    "  默认允许集合 {4,5,6,7}（部署事实，可用 GRASPO_ALLOWED_GPU_INDICES 覆盖）。"
)


def _usage_hint(allowed: Sequence[int], reserved: Sequence[int]) -> str:
    """按**实际生效的**策略生成可操作用法提示（不再把默认元组写死进提示）。"""
    csv = ",".join(str(index) for index in allowed)
    reserved_line = (
        f"  保留卡 {tuple(reserved)} 一律拒绝（部署方通过 {RESERVED_INDICES_ENV} 声明）。\n"
        if reserved
        else ""
    )
    return (
        f"正确用法（显式指定 ≤{MAX_CARDS} 张、且仅取自 {{{csv}}} 的卡）：\n"
        f"    NVIDIA_VISIBLE_DEVICES={csv} <训练/测试命令>\n"
        f"    run.sh <config.yaml> --gpus {csv}\n"
        f"    docker run --gpus '\"device={csv}\"' ...\n"
        f"  本期允许集合 {{{csv}}}（部署事实，可用 {ALLOWED_INDICES_ENV} 覆盖）。\n"
        f"{reserved_line}"
    )


class GpuLockError(RuntimeError):
    """锁卡守卫拒绝启动（fail-closed）。错误消息必须可操作。"""


def _reject(
    reason: str,
    detail: str,
    *,
    allowed: Sequence[int] | None = None,
    reserved: Sequence[int] | None = None,
) -> GpuLockError:
    """构造统一格式的可操作拒绝消息。"""
    hint = (
        _DEFAULT_USAGE_HINT
        if allowed is None and reserved is None
        else _usage_hint(
            DEFAULT_ALLOWED_INDICES if allowed is None else allowed,
            DEFAULT_RESERVED_INDICES if reserved is None else reserved,
        )
    )
    return GpuLockError(
        f"GPU 锁卡守卫拒绝启动（fail-closed）：\n"
        f"  原因: {reason}\n"
        f"  为什么必须拒绝: {detail}\n"
        f"  {hint}"
    )


def _parse_index_set(raw: str | None, *, env_key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """解析"允许/保留索引"配置取值（``"0,1,2,3"``）。

    与 :func:`parse_device_list` 的区别：**未设置 / 空串回落默认值**（这是配置，
    有明确默认），但**取值非法一律 fail-closed 抛错**——不静默回落，
    否则一个手误的配置会悄悄放松安全边界（宪法 §2.4 操作防呆）。
    """
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise _reject(
            f"{env_key}={raw!r} 含空元素",
            "该变量是逗号分隔的整数卡号列表（如 '0,1,2,3'）；出现空位说明取值有误，"
            "拒绝猜测意图（fail-closed）。",
        )
    indices: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise _reject(
                f"{env_key} 含非整数索引 {part!r}",
                "该变量只接受整数卡号列表；非整数会让边界判据失去意义（fail-closed）。",
            )
        indices.append(int(part))
    if len(set(indices)) != len(indices):
        raise _reject(
            f"{env_key}={raw!r} 含重复卡号",
            "重复卡号说明取值有误（同一张卡不能占两次），拒绝猜测。",
        )
    return tuple(indices)


def resolve_allowed_indices(env: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """解析**生效的允许卡号集合**：配置优先，未配置回落默认值。

    配置来源 :data:`ALLOWED_INDICES_ENV`（``"0,1,2,3"`` 形式）。默认
    :data:`DEFAULT_ALLOWED_INDICES`（= 变体1 的部署事实 ``(4,5,6,7)``），
    因此不设该变量的既有调用方行为完全不变（向后兼容）。

    :param env: 环境映射（``None`` = ``os.environ``）；接收 mapping 以便测试注入。
    :raises GpuLockError: 配置取值非法（fail-closed，不静默回落）。
    """
    source = os.environ if env is None else env
    return _parse_index_set(
        source.get(ALLOWED_INDICES_ENV),
        env_key=ALLOWED_INDICES_ENV,
        default=DEFAULT_ALLOWED_INDICES,
    )


def resolve_reserved_indices(env: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """解析**生效的保留（不可碰）卡号集合**：配置优先，未配置回落默认值。

    默认 :data:`DEFAULT_RESERVED_INDICES`（当前为空）。把某台机器上的生产卡
    （如 6/7）声明为保留卡即可让守卫无条件拒绝——这是"部署事实进配置"的同一机制。
    """
    source = os.environ if env is None else env
    return _parse_index_set(
        source.get(RESERVED_INDICES_ENV),
        env_key=RESERVED_INDICES_ENV,
        default=DEFAULT_RESERVED_INDICES,
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
    allowed_indices: Sequence[int] | None = None,
    reserved_indices: Sequence[int] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """校验**宿主卡号列表**满足锁卡边界，返回设备元组；不满足即抛错。

    判据（全部 fail-closed）：显式设置、索引 ⊆ **生效的允许集合**、索引 ∉
    **生效的保留集合**、卡数 ≤ ``max_cards``。

    **允许/保留集合从哪来（部署事实进配置）**：优先用参数 ``allowed_indices`` /
    ``reserved_indices``（调用方显式注入）；未注入时读环境映射
    （``env``，默认 ``os.environ``）里的 :data:`ALLOWED_INDICES_ENV` /
    :data:`RESERVED_INDICES_ENV`；都没给就用默认值
    （:data:`DEFAULT_ALLOWED_INDICES` / :data:`DEFAULT_RESERVED_INDICES`）。
    这样"某台机器允许哪几张卡"不再硬编码进共享代码。

    向后兼容：``allowed_max_index`` 显式传入**不等于** :data:`ALLOWED_MAX_INDEX`
    时，仍沿用旧语义"只按上界收窄、不套白名单"（既有调用方不受白名单影响）。

    :param raw: ``NVIDIA_VISIBLE_DEVICES`` 的原始取值（``None`` = 未设置）。
    :param allowed_max_index: 兼容上界；仅当它等于默认上界时才启用白名单判据。
    :param allowed_indices: 显式注入的允许集合（覆盖配置与默认值）。
    :param reserved_indices: 显式注入的保留集合（覆盖配置与默认值）。
    :param env: 解析配置用的环境映射（``None`` = ``os.environ``）。
    :raises GpuLockError: 任一边界不满足（含 runtime 的 ``void`` 哨兵——本函数
        要求真实卡号；容器侧请用 :func:`assert_gpu_lock_inventory`）。
    """
    devices = parse_device_list(raw)

    # ★ 变体1（用户 2026-09-24 批准 A）：**显式白名单**取代「仅上限」判据——
    #   判据由 `device > allowed_max_index` 改为 `device not in <允许集合>`。
    #   允许集合现在来自配置（见 resolve_allowed_indices），不再是写死的机器事实。
    #   仅当调用方沿用默认上界时启用（显式传入更小上界的调用方仍按原语义收窄，不放松、
    #   也不被本白名单误伤——例如其他工具/测试显式使用 GPU0–3 的场景）。
    use_whitelist = allowed_indices is not None or allowed_max_index == ALLOWED_MAX_INDEX
    allowed = (
        tuple(allowed_indices)
        if allowed_indices is not None
        else (resolve_allowed_indices(env) if use_whitelist else ())
    )
    reserved = (
        tuple(reserved_indices) if reserved_indices is not None else resolve_reserved_indices(env)
    )
    hint_allowed = allowed if allowed else DEFAULT_ALLOWED_INDICES

    # 保留卡优先报：它是"这张卡明确不能碰"（如生产 vLLM），比"不在允许集合里"更具体。
    reserved_hits = sorted(device for device in devices if device in reserved)
    if reserved_hits:
        raise _reject(
            f"设备列表 {devices} 含保留卡 {reserved_hits}（保留集合 {reserved}）",
            "该卡被部署方声明为生产/保留卡，任何组合里出现都必须拒绝。",
            allowed=hint_allowed,
            reserved=reserved,
        )
    if use_whitelist and allowed:
        outside = sorted(device for device in devices if device not in allowed)
        if outside:
            raise _reject(
                f"设备列表 {devices} 含白名单外的卡 {outside}（允许集合 {allowed}）",
                f"本次部署只允许使用 GPU{list(allowed)}；其余卡留给其他工作负载或不存在的卡位。",
                allowed=allowed,
                reserved=reserved,
            )
    reserved_by_bound = sorted(device for device in devices if device > allowed_max_index)
    if reserved_by_bound:
        raise _reject(
            f"设备列表 {devices} 含上界外卡 {reserved_by_bound}（上界 {allowed_max_index}）",
            "超出允许集合的卡一律拒绝（fail-closed）。",
            allowed=hint_allowed,
            reserved=reserved,
        )
    if len(devices) > max_cards:
        raise _reject(
            f"设备列表 {devices} 使用 {len(devices)} 张卡，超过上限 {max_cards}",
            f"用户拍板每次最多 {max_cards} 卡；更多卡会挤占生产余量且超出本期验证口径。",
            allowed=hint_allowed,
            reserved=reserved,
        )
    return devices


@dataclass(frozen=True)
class GpuInventory:
    """容器内**实测可见卡**的探针结果（见 :func:`probe_gpu_inventory`）。

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


# ── 可见卡探测（本模块唯一的设施入口；2026-09-29 由 cli/gpu_monitor.py 下沉）──
# 为什么下沉：修复前 core/gpu_guard 通过延迟导入 ``cli.gpu_monitor.probe_gpu_inventory``
# 拿默认探测回调，构成 core → cli 反向依赖（AST 边界测试判为跨层环，D-03）。
# 探测本身是"读本机事实"的设施动作，规则部分（上面的判定函数）仍保持纯计算。
# ``cli/gpu_monitor.py`` 保留同名转发，既有 `graspo.cli.gpu_monitor.probe_gpu_inventory`
# 导入路径零破坏。


def parse_visible_device_indices(text: str) -> tuple[int, ...]:
    """从 ``nvidia-smi -L`` 输出解析**容器内本地序号**（探测可见卡的唯一入口）。

    ``-L`` 每个可见卡一行（``GPU 0: NVIDIA ... (UUID: GPU-xxx)``）。行首序号即
    容器命名空间里的本地 index——runtime 重编号后，宿主卡号在这里是看不到的
    （这正是 F-2 的根因：不能拿宿主卡号去查容器内的卡）。

    :raises RuntimeError: 行结构与 ``GPU <n>:`` 不符——宁可 fail-closed，
        也不要猜出一组卡号去查。
    """
    indices: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("GPU ") or ":" not in stripped:
            raise RuntimeError(f"无法解析 nvidia-smi -L 行：{line!r}")
        head = stripped[len("GPU ") :].split(":", 1)[0].strip()
        if not head.isdigit():
            raise RuntimeError(f"无法解析 nvidia-smi -L 卡号：{line!r}")
        indices.append(int(head))
    return tuple(indices)


def probe_gpu_inventory(
    runner: Callable[[list[str]], str] | None = None,
) -> GpuInventory:
    """实测容器内**可见卡**（F-1/F-2 的设施层入口）。

    用 ``nvidia-smi -L``（结构稳定：每个可见卡一行）数卡并取容器内本地序号。
    ``-L`` 不接受 ``-i``，但它天然只列**可见**卡——这正是我们要的口径
    （不要 ``--query-gpu`` 的全卡枚举语义）。带 ``-i`` 的查询仍由
    ``cli/gpu_monitor.build_gpu_query_command`` 唯一构造（采样路径）。

    :param runner: 命令执行器（接收 argv、返回 stdout）；默认走 ``subprocess``。
    :raises RuntimeError: ``nvidia-smi`` 不在或执行失败——调用方必须 fail-closed。
    """
    command = ["nvidia-smi", "-L"]
    if runner is None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"nvidia-smi 不可用：{exc}") from None
        if completed.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi -L 失败（rc={completed.returncode}）：{completed.stderr.strip()[:200]}"
            )
        stdout = completed.stdout
    else:
        stdout = runner(command)
    indices = parse_visible_device_indices(stdout)
    return GpuInventory(
        source="nvidia-smi -L",
        count=len(indices),
        indices=indices,
    )


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
    allowed_indices: Sequence[int] | None = None,
    reserved_indices: Sequence[int] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """入口守卫的统一判定：显式卡号走静态比对，runtime 哨兵走实测可见卡。

    这是训练 / 测试入口应调用的判定函数——它是 F-1 的修复点：容器内
    ``NVIDIA_VISIBLE_DEVICES=void``（runtime 已收窄）不再被误判为"未锁卡"，
    而是按实测可见卡断言；同时**不放松**任何既有限制（见 :func:`assert_inventory_visible`）。

    允许/保留集合的注入方式见 :func:`assert_gpu_lock`（显式参数优先，
    其次配置，最后默认值）。

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
            allowed_indices=allowed_indices,
            reserved_indices=reserved_indices,
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
    allowed_indices: Sequence[int] | None = None,
    reserved_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """从环境映射读取 ``NVIDIA_VISIBLE_DEVICES`` 并按**显式宿主卡号**校验。

    接收 mapping 而非直接读 ``os.environ``，是为了让调用方与测试能注入
    确定的环境（宪法 §2.2 显式即防呆）。允许/保留集合也从**同一个 mapping**
    解析（:data:`ALLOWED_INDICES_ENV` / :data:`RESERVED_INDICES_ENV`），
    这样"整个环境"只有一个注入点。

    本函数**不**处理 runtime 的 ``void`` 哨兵（它要求真实卡号，见 :func:`assert_gpu_lock`）；
    需要容器侧回落语义的入口请用 :func:`require_gpu_lock_or_exit`。
    """
    source = os.environ if env is None else env
    return assert_gpu_lock(
        source.get(_ENV_KEY),
        allowed_max_index=allowed_max_index,
        max_cards=max_cards,
        allowed_indices=allowed_indices,
        reserved_indices=reserved_indices,
        env=source,
    )


def require_gpu_lock_or_exit(
    env: Mapping[str, str] | None = None,
    *,
    allowed_max_index: int = ALLOWED_MAX_INDEX,
    max_cards: int = MAX_CARDS,
    inventory_probe: Callable[[], GpuInventory] | None = None,
    allowed_indices: Sequence[int] | None = None,
    reserved_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """训练/测试入口使用的守卫包装：通过返回设备，不通过即可操作地终止。

    判定走 :func:`assert_gpu_lock_inventory`——即同时支持"宿主侧显式卡号"与
    "容器内 runtime ``void`` + 实测可见卡"两种来源（F-1）。

    :param env: 环境映射（``None`` = ``os.environ``）；允许/保留集合也从这里解析。
    :param inventory_probe: 探测容器内实测可见卡的可调用对象；``None`` 时用本模块的
        :func:`probe_gpu_inventory`（``nvidia-smi -L``）。
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
            allowed_indices=allowed_indices,
            reserved_indices=reserved_indices,
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
    allowed = resolve_allowed_indices()
    return (
        f"[gpu-guard] OK: {source}={','.join(str(d) for d in devices)} "
        f"（{len(devices)} 卡，全部 ∈ {allowed}，≤{MAX_CARDS} 卡上限）"
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


# ── 评测链路的卡计划契约（2026-09-29 由 eval/guard.py 下沉）──────────────────
# 为什么下沉：判定规则必须只有一份（§1.4 单一真相源），且 core 不能反向导入
# ``eval.guard``（跨层环，D-03）。``eval/guard.py`` 保留同名转发，既有
# `graspo.eval.guard.GpuGuardError` / `GpuPlan` / `resolve_gpu_plan` 导入路径零破坏。
# 这三个名字都不做设施调用（只有格式转换 + 转发 assert_gpu_lock），是纯契约。


class GpuGuardError(RuntimeError):
    """评测链路的锁卡/采样失败。调用方必须让它终止流程——不要 catch 后继续跑。

    ``GpuLockError``（规则层）在 :func:`resolve_gpu_plan` 里被转成本异常，
    这样评测链路的调用方只需捕获一种异常类型；规则文案原样保留。
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
        GpuGuardError: 规则层拒绝（未指定 / 含生产卡 / 卡数超限 / 取值非法）。
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
