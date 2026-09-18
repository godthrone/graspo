"""GPU 锁卡守卫（fail-closed）—— 纯计算层，零设施依赖。

职责：把「这次运行允许用哪些 GPU」这条安全边界，编码成一条可复用、
**默认拒绝**的断言。任何训练 / 测试入口在启动前必须调用本模块，三种
情形一律拒绝启动（fail-closed）：

1. ``NVIDIA_VISIBLE_DEVICES`` **未显式设置**；
2. 取值越出允许集合 ``{0,1,2,3,4,5}``（含生产卡 6 / 7）；
3. 卡数超过 4。

为什么默认拒绝：本项目的运行镜像把 ``NVIDIA_VISIBLE_DEVICES`` 写死为
``all``，而 docker 默认 runtime 是 nvidia——不显式锁卡就会看到全部 8 张卡，
其中 GPU6/7 被生产 vLLM 占死。这不是"配置不当"，是"一碰就事故"，
所以守卫必须 fail-closed（宪法 §2 防呆设计：不靠调用方自觉）。

本模块不读环境、不调 ``nvidia-smi``、不碰 GPU——输入是字符串与映射，
输出是不可变设备元组，因此可在 CPU 上独立单测（宪法 §1.3 层次边界）。
环境读取集中在 ``assert_gpu_lock_from_env``，它接收一个 mapping 以便测试
注入；设施调用在 ``scripts/gpu_lock_guard.py`` 与 ``cli/gpu_monitor.py``。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

# ── 安全边界常量（单一真相源，§1.4）────────────────────────────────────────
#: ``NVIDIA_VISIBLE_DEVICES`` 允许出现的最大卡号（0–5 可用，6/7 为生产卡）。
ALLOWED_MAX_INDEX = 5
#: 单次运行最多使用几张卡（用户拍板：每次最多 4 卡）。
MAX_CARDS = 4
#: 生产卡——被常驻 vLLM 占死，严禁触碰。
RESERVED_INDICES: tuple[int, ...] = (6, 7)
#: 4 卡首选集合——唯一全部位于 NUMA0。
PREFERRED_FOUR: tuple[int, ...] = (0, 1, 2, 3)

_ENV_KEY = "NVIDIA_VISIBLE_DEVICES"

_USAGE_HINT = (
    "正确用法（显式指定 ≤4 张、且仅取自 {0,1,2,3,4,5} 的卡）：\n"
    '    NVIDIA_VISIBLE_DEVICES=0,1,2,3 <训练/测试命令>\n'
    "    run.sh <config.yaml> --gpus 0,1,2,3\n"
    '    docker run --gpus \'"device=0,1,2,3"\' ...\n'
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
    GPU UUID / ``"void"`` 一律拒绝——它们都意味着"没有显式锁卡"。
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
            "空值等同于未锁卡，无法确定这次运行会用到哪些卡。",
        )
    lowered = text.lower()
    if lowered == "all":
        raise _reject(
            f"{_ENV_KEY}=all",
            "all 会暴露全部 8 张卡（含生产卡 GPU6/7）；必须显式列出要用的卡。",
        )
    if lowered in {"void", "none"}:
        raise _reject(
            f"{_ENV_KEY}={text}",
            "该取值不指定任何卡，训练无法确定设备；必须显式列出要用的卡。",
        )

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
                "本守卫只接受整数卡号；UUID / 'all' / 'void' 无法与生产卡边界做静态比对。",
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
    """校验设备列表满足锁卡边界，返回设备元组；不满足即抛 ``GpuLockError``。

    三条边界（全部 fail-closed）：显式设置、索引 ⊆ ``{0..allowed_max_index}``、
    卡数 ≤ ``max_cards``。

    :param raw: ``NVIDIA_VISIBLE_DEVICES`` 的原始取值（``None`` = 未设置）。
    :raises GpuLockError: 任一边界不满足。
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


def assert_gpu_lock_from_env(
    env: Mapping[str, str] | None = None,
    *,
    allowed_max_index: int = ALLOWED_MAX_INDEX,
    max_cards: int = MAX_CARDS,
) -> tuple[int, ...]:
    """从环境映射读取 ``NVIDIA_VISIBLE_DEVICES`` 并校验；默认读 ``os.environ``。

    接收 mapping 而非直接读 ``os.environ``，是为了让调用方与测试能注入
    确定的环境（宪法 §2.2 显式即防呆）。
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
) -> tuple[int, ...]:
    """训练/测试入口使用的守卫包装：通过返回设备，不通过即可操作地终止。

    终止方式为 ``SystemExit``（消息原样打印到 stderr，退出码 1），避免
    在训练入口抛出原始 traceback 掩盖真正原因。
    """
    try:
        return assert_gpu_lock_from_env(
            env,
            allowed_max_index=allowed_max_index,
            max_cards=max_cards,
        )
    except GpuLockError as exc:
        raise SystemExit(str(exc)) from None


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


def format_verdict(devices: Sequence[int], *, source: str = _ENV_KEY) -> str:
    """生成通过守卫后的可读确认行（供 CLI / 日志复用）。"""
    return (
        f"[gpu-guard] OK: {source}={','.join(str(d) for d in devices)} "
        f"（{len(devices)} 卡，全部 ⊆ 0..{ALLOWED_MAX_INDEX}，≤{MAX_CARDS} 卡上限）"
    )
