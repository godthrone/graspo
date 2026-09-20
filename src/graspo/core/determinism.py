"""确定性钉定开关：**全仓唯一真相源** + **唯一渲染点**（宪法 §1.4 / §2.2 / §6）。

**为什么有本模块**

既有根因分析（`task-x1-sft-repro/report.md` §①、`task-a3-determinism/report.md`
§③-3/§④-0）已确证：本仓在 `use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` /
`cudnn.deterministic` / `NCCL_ALGO|PROTO` 上**全仓零命中** ⇒ 多卡同批首步 loss 的
1 个 bf16 ulp 级差异（`T026`：`3.2389e-3`）目前既不能判成 bug，也不能判成 §6 排除项。
宪法 §6 的口径是「这些都应该被消除」——**可控制而未被控制的差异不得先宣布为
"无法控制的差异"而排除**。本模块只做一件事：把"钉定"变成**可显式打开**的开关。

**开关放在 CLI 而不是 config（§10.1 的边界判定，理由可核）**

a3 报告 §④-0 的首推落点是 `_build_launch_env` 与 `train_worker.main()`——两者都由
`graspo launch` 的参数驱动。**不进 YAML 配置段**有两条硬理由：

1. `samples/configs/config_example.yaml` 必须覆盖 schema 的全部字段
   （``tests/core/test_schema.py::test_config_example_covers_all_schema_fields``
   与 ``tests/core/test_nested_get_falsy.py`` 的 167 字段计数守卫），而本包的边界
   明令**不得修改 ``samples/configs/``**。新增配置段会直接踩红这两条既有守卫，
   而"改守卫"是本包明令禁止的（不得削弱既有守卫）。
2. 它本来就是**实验/诊断边界**，与 ``--smoke`` 同一性质（`app.py` 里
   ``--smoke`` 的 help 原文：infrastructure param，Run 1 step then stop，
   **never mutates the config object**）。确定性与冒烟一样是"怎么跑"，不是
   "算法是什么"。

⇒ 采用与 ``--smoke`` **同构**的通道：``graspo launch --determinism ...`` ⇒
``_build_launch_env`` 注入环境变量 + worker 命令行追加 ``--determinism-spec``；
worker 侧 :func:`bind_active_switch` 把这一份声明绑定为**进程内唯一真相源**
（与 ``flow/logging.set_run_id`` 的既有范式一致），训练层只查
:func:`active_switch`，不各自解析。

**默认关**

不传任何 ``--determinism*`` ⇒ :class:`DeterminismSwitch` 全关 ⇒
:func:`determinism_env_delta` 返回**空 dict**、:func:`apply_torch_determinism`
返回**空列表**、横幅**空列表**、产物记录 ``None`` ⇒ 与打开本功能之前逐字相同
（机核见 ``tests/cli/test_determinism_launch.py``）。

**显式即防呆（§2.2）**

每个开关的**副作用**都写进表里（``DeterminismEnvKnob.side_effect``），并在生效时
打印。特别是 ``CUBLAS_WORKSPACE_CONFIG`` 会抬高 cuBLAS workspace 上限、
**可能抬高显存峰值**——这一条不允许只写在文档里，必须随开关一起出现在运行输出中。
"""

from __future__ import annotations

import dataclasses
import json
import os
import site
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class DeterminismEnvKnob:
    """一个**环境变量**型确定性开关（表项 = 定义，渲染点 = :func:`determinism_env_delta`）。

    Attributes:
        field: :class:`DeterminismSwitch` 里的字段名（该字段只做**开/关**，值由
            :attr:`default_value` 决定——单一真相源，§1.4）。
        env_var: 注入的环境变量名。
        default_value: 该变量被钉定的**唯一取值**（全仓只此一处写它）。
        side_effect: 显式记录的副作用；生效时必须打印（§2.2）。
        support_probe: 生效前是否需要做支持性探测（不支持时静默失效的变量）。
    """

    field: str
    env_var: str
    default_value: str
    side_effect: str
    support_probe: bool = False


#: ── 环境变量型开关清单（**唯一真相源**，§1.4）───────────────────────────────
#:
#: 渲染点只有 :func:`determinism_env_delta` 一处（"表 + 唯一渲染点"范式，
#: 与 ``tests/e2e/generate_matrix.py::CONTAINER_CACHE_ROOTS`` 同构）。
#: 新增开关 = 本表加一行 + :class:`DeterminismSwitch` 加一个同名字段
#: （两边由 ``tests/core/test_determinism.py`` 的同步测试守住，不靠人记）。
DETERMINISM_ENV_KNOBS: tuple[DeterminismEnvKnob, ...] = (
    DeterminismEnvKnob(
        field="cublas_workspace_config",
        env_var="CUBLAS_WORKSPACE_CONFIG",
        default_value=":4096:8",
        side_effect=(
            "把 cuBLAS workspace 上限钉到 4096 KiB（而非默认的动态分配）："
            "可能抬高显存峰值 ⇒ 27B 档有 OOM 风险，必须与峰值读数一起复核"
        ),
    ),
    DeterminismEnvKnob(
        field="nccl_algo",
        env_var="NCCL_ALGO",
        default_value="Ring",
        side_effect="固定集合通信算法：比默认的按消息大小自适应选择更慢",
        support_probe=True,
    ),
    DeterminismEnvKnob(
        field="nccl_proto",
        env_var="NCCL_PROTO",
        default_value="Simple",
        side_effect="固定集合通信协议：比默认协议更慢",
        support_probe=True,
    ),
    DeterminismEnvKnob(
        field="nccl_deterministic",
        env_var="NCCL_DETERMINISTIC",
        default_value="1",
        side_effect=(
            "NCCL 自带的确定性归约（版本相关：较新 NCCL 才有）；"
            "缺失时 NCCL **只 warn** ⇒ 静默失效，必须看支持性探测结论"
        ),
        support_probe=True,
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class DeterminismSwitch:
    """``graspo launch --determinism`` 声明的确定性开关集合（**全关 = 什么都不做**）。

    ``enabled=False`` 是总开关：细项即使为 ``True`` 也**一律不生效**——避免"细项开了
    但总开关关着"的隐性歧义（§2.2 显式）。``probe_first_step`` 是唯一的例外：它是
    **只读旁路探针**，不是"钉定"，因此独立于 ``enabled``（A/B 的臂 A 需要
    "探针开、钉定关"）。
    """

    enabled: bool = False
    #: ``torch.use_deterministic_algorithms(True, warn_only=...)``。
    torch_deterministic_algorithms: bool = True
    #: ``warn_only=True``（默认）是刻意的：严格模式下遇到没有确定性实现的算子会直接
    #: ``RuntimeError``——把"不可复现"变成"跑不起来"是明显降级（§3）。
    warn_only: bool = True
    #: ``cudnn.deterministic=True`` + ``cudnn.benchmark=False``。
    cudnn: bool = True
    #: ``CUBLAS_WORKSPACE_CONFIG``（★ 可能抬高显存峰值）。
    cublas_workspace_config: bool = True
    #: ``NCCL_ALGO`` / ``NCCL_PROTO``。
    nccl_algo: bool = True
    nccl_proto: bool = True
    #: ``NCCL_DETERMINISTIC``：**版本相关**（较新 NCCL 才有），默认 **False**——
    #: 不假定本机 NCCL 支持；打开后由静态探测显式报告"是否真的被识别"。
    nccl_deterministic: bool = False
    #: 每 rank 首步探针（只读旁路）。独立于 ``enabled``（见类 docstring）。
    probe_first_step: bool = False

    @classmethod
    def from_spec(cls, spec: str | None) -> DeterminismSwitch:
        """从 worker 命令行收到的 JSON 串还原（``""``/``None`` ⇒ 全关）。

        ``--determinism-spec`` 是**内部**通道（``graspo launch`` 生成、``train_worker``
        消费），因此这里严格校验：未知键直接报错而不是静默忽略（§2.3 边界校验即防呆）。
        """
        text = (spec or "").strip()
        if not text:
            return cls()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--determinism-spec 不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("--determinism-spec 必须是 JSON 对象")
        return switch_from_dict(payload)

    def to_spec(self) -> str:
        """渲染成只含**非默认值**的紧凑 JSON（唯一渲染点；全关 ⇒ ``""``）。

        全关返回**空串**而不是 ``"{}"``：空串让调用方"不追加参数"的默认关保证
        变成一个可断言的字符串比较（§2.2 显式），而不是"追加了一个等价的空对象"。
        """
        payload = switch_to_dict(self)
        if not payload:
            return ""
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


#: :class:`DeterminismSwitch` 的字段名集合（单一定义；测试据此守住表与类的同步）。
DETERMINISM_SWITCH_FIELDS: frozenset[str] = frozenset(
    field.name for field in dataclasses.fields(DeterminismSwitch)
)


def switch_from_dict(payload: dict[str, Any]) -> DeterminismSwitch:
    """从 dict 构造，**拒绝未知键**（§2.3 边界校验；不静默忽略拼错的字段）。"""
    unknown = sorted(set(payload) - DETERMINISM_SWITCH_FIELDS)
    if unknown:
        raise ValueError(
            f"未知的 determinism 开关：{unknown}；可用：{sorted(DETERMINISM_SWITCH_FIELDS)}"
        )
    return DeterminismSwitch(**{key: bool(value) for key, value in payload.items()})


def switch_to_dict(switch: DeterminismSwitch) -> dict[str, Any]:
    """渲染成 dict，只保留与默认值不同的字段（全关 ⇒ ``{}``）。"""
    default = DeterminismSwitch()
    return {
        field.name: getattr(switch, field.name)
        for field in dataclasses.fields(switch)
        if getattr(switch, field.name) != getattr(default, field.name)
    }


#: 进程内绑定的当前开关（由 ``cli/train_worker.main`` 从 ``--determinism-spec`` 绑定，
#: 训练层只读 :func:`active_switch`；与 ``flow/logging.set_run_id`` 同一范式，§1.4）。
_ACTIVE_SWITCH: DeterminismSwitch = DeterminismSwitch()


def bind_active_switch(switch: DeterminismSwitch) -> DeterminismSwitch:
    """把本次进程的确定性开关绑定为**唯一真相源**，返回绑定值（幂等）。"""
    global _ACTIVE_SWITCH
    _ACTIVE_SWITCH = switch
    return _ACTIVE_SWITCH


def active_switch() -> DeterminismSwitch:
    """读取本进程绑定的确定性开关（未绑定时 = 全关 ⇒ 零行为变化）。"""
    return _ACTIVE_SWITCH


def enabled_env_knobs(switch: DeterminismSwitch) -> tuple[DeterminismEnvKnob, ...]:
    """``switch`` 里**已打开**的环境变量型开关（顺序 = 表的顺序，稳定可断言）。"""
    if switch is None or not switch.enabled:
        return ()
    return tuple(knob for knob in DETERMINISM_ENV_KNOBS if getattr(switch, knob.field, False))


def determinism_env_delta(switch: DeterminismSwitch) -> dict[str, str]:
    """把开关渲染成**要注入的环境变量增量**（纯函数）。

    未启用（``enabled=False`` 或 ``None``）⇒ 返回 ``{}``：**零注入**，调用方因此
    不需要任何分支（§2.2 显式即防呆；默认关的机核就在这一行）。
    """
    return {knob.env_var: knob.default_value for knob in enabled_env_knobs(switch)}


def torch_determinism_steps(switch: DeterminismSwitch) -> list[str]:
    """返回**已启用**的进程内 torch API 开关的人类可读清单（纯函数，不导入 torch）。

    同一份清单既用于打印、也用于测试断言，避免"设了什么"与"说设了什么"两处实现。
    """
    if switch is None or not switch.enabled:
        return []
    steps: list[str] = []
    if switch.torch_deterministic_algorithms:
        steps.append(f"torch.use_deterministic_algorithms(True, warn_only={switch.warn_only})")
    if switch.cudnn:
        steps.append("torch.backends.cudnn.deterministic=True")
        steps.append("torch.backends.cudnn.benchmark=False")
    return steps


def apply_torch_determinism(switch: DeterminismSwitch) -> list[str]:
    """在**当前进程**内套用确定性开关，返回已生效的语句清单（空 = 未启用）。

    必须在 ``import torch`` 之后、训练器构造之前调用。``CUBLAS_WORKSPACE_CONFIG``
    是**环境变量**型开关（必须早于 ``import torch``），由
    ``cli/app.py::_build_launch_env`` 在启动子进程时注入——两者分工见模块 docstring。
    """
    steps = torch_determinism_steps(switch)
    if not steps:
        return []
    import torch  # 延迟导入：本模块在无 torch 的环境里仍可导入与单测（§1.3）

    if switch.torch_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=switch.warn_only)
    if switch.cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return steps


def format_determinism_banner(switch: DeterminismSwitch) -> list[str]:
    """渲染"将生效的确定性开关"多行文本（**唯一渲染点**，dry-run 所走即此）。

    未启用时**不打印任何行**（默认关的行为零变化——连 stdout 都不多一行）。
    """
    if switch is None or not switch.enabled:
        return []
    lines = ["[determinism] 确定性钉定已启用（graspo launch --determinism）"]
    delta = determinism_env_delta(switch)
    for knob in enabled_env_knobs(switch):
        lines.append(
            f"[determinism]   env {knob.env_var}={knob.default_value}"
            f" —— 副作用：{knob.side_effect}"
        )
    for statement in torch_determinism_steps(switch):
        lines.append(f"[determinism]   torch {statement}")
    lines.extend(format_nccl_support_report(delta))
    return lines


# ── NCCL 环境变量支持性探测（防"未知变量只 warn"的静默失效）──────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class NcclVariableSupport:
    """某个 NCCL 环境变量的**支持性静态探测**结论。

    Attributes:
        env_var: 环境变量名。
        accepted: ``True`` = 在 NCCL 共享库里找到该变量名字面量（被识别）；
            ``False`` = 库已定位但**不含**该字面量（该版本不认这个变量，注入后
            只会 warn）；``None`` = **探测不了**（库未定位 / 不可读）——必须显式
            报成"未确认"，不得当成"已接受"（§2.2）。
        evidence: 得到该结论的证据（可核定位）。
    """

    env_var: str
    accepted: bool | None
    evidence: str


def candidate_nccl_library_paths() -> list[Path]:
    """列出**可能**存在 libnccl 的绝对路径（只读探测，不执行任何命令）。

    来源依次为：``LD_LIBRARY_PATH``、Python 环境的 ``nvidia/nccl/lib``
    （pip 包 ``nvidia-nccl-cu12`` 的布局）、``torch`` 自带的 ``lib`` 目录、
    以及常见的系统/CUDA 路径。只返回**存在**的文件。
    """
    roots: list[Path] = []
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry))
    for base in list(site.getsitepackages()) + [site.getusersitepackages()]:
        roots.append(Path(base) / "nvidia" / "nccl" / "lib")
    try:  # torch 自带 lib（可选：无 torch 时静默跳过，不改变探测语义）
        import torch  # noqa: PLC0415

        roots.append(Path(torch.__file__).resolve().parent / "lib")
    except ImportError:
        pass
    roots.extend(
        [
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib/aarch64-linux-gnu"),
            Path("/usr/local/cuda/lib64"),
            Path("/usr/local/nccl/lib"),
        ]
    )
    nccl_home = os.environ.get("NCCL_HOME")
    if nccl_home:
        roots.append(Path(nccl_home) / "lib")

    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(
            candidate
            for candidate in sorted(root.glob("libnccl.so*"))
            if candidate.is_file() and candidate not in found
        )
    return found


def nccl_version() -> str | None:
    """返回当前 torch 链接的 NCCL 版本（``"2.27.5"`` 形式）；取不到返回 ``None``。

    ``torch.cuda.nccl.version()`` 不需要可见 GPU，因此本函数在 CPU-only 机器上
    也可用于记录版本（版本本身是探测结论的一部分）。
    """
    try:
        import torch  # noqa: PLC0415

        version = torch.cuda.nccl.version()
    except Exception:  # noqa: BLE001 - 无 torch / 无 NCCL 构建都只应降级为"未知"
        return None
    if isinstance(version, int):
        return str(version)
    if isinstance(version, (tuple, list)):
        return ".".join(str(part) for part in version)
    return str(version)


def probe_nccl_variable_support(
    env_vars: Any,
    *,
    library_paths: Any = None,
) -> list[NcclVariableSupport]:
    """逐个探测 NCCL 是否**真的认识**这些环境变量（静态证据，不需要 GPU）。

    做法：在 NCCL 共享库里搜索变量名的 ASCII 字面量。NCCL 用 ``getenv("NCCL_ALGO")``
    这类调用读取配置，变量名字面量因此留在库的只读数据段里；库里没有这个字面量
    就说明该版本不认这个变量（注入后 NCCL 只会 warn —— 静默失效）。

    这是**保守上界**：字面量存在只证明"该名字被编译进去了"，不证明在**当前拓扑**下
    一定被接受（例如 ``NCCL_ALGO=Ring`` 在拓扑不可行时仍会回退并 warn）。因此
    :func:`format_nccl_support_report` 会把结论标注为"静态探测"，并把"从 NCCL 启动
    日志核实"写成必须动作，而不是把静态结论当成运行时保证（§2.2 显式即防呆）。

    Args:
        env_vars: 要探测的变量名集合（通常 = :func:`determinism_env_delta` 的键）。
        library_paths: 显式注入的库路径（测试用）；``None`` ⇒ 自动发现。

    Returns:
        与 ``env_vars`` 同序的结论列表（``env_vars`` 为空 ⇒ 空列表）。
    """
    names = [str(name) for name in env_vars]
    if not names:
        return []
    paths = (
        candidate_nccl_library_paths()
        if library_paths is None
        else [Path(path) for path in library_paths]
    )
    readable = [path for path in paths if path.is_file()]
    if not readable:
        return [
            NcclVariableSupport(
                env_var=name,
                accepted=None,
                evidence=(
                    "未定位到可读的 libnccl.so（探测不了 ⇒ 一律按「未确认」处理，"
                    "不得当成已接受）"
                ),
            )
            for name in names
        ]
    blobs: list[tuple[Path, bytes]] = []
    for path in readable:
        try:
            blobs.append((path, path.read_bytes()))
        except OSError:
            continue
    if not blobs:
        return [
            NcclVariableSupport(
                env_var=name,
                accepted=None,
                evidence=f"定位到 {len(readable)} 个 libnccl.so 但均不可读 ⇒ 未确认",
            )
            for name in names
        ]
    results: list[NcclVariableSupport] = []
    for name in names:
        needle = name.encode("ascii")
        hits = [str(path) for path, blob in blobs if needle in blob]
        if hits:
            results.append(
                NcclVariableSupport(
                    env_var=name,
                    accepted=True,
                    evidence=(
                        f"字面量出现在 {hits[0]}（静态探测；运行时接受性仍需看 NCCL 启动日志）"
                    ),
                )
            )
        else:
            results.append(
                NcclVariableSupport(
                    env_var=name,
                    accepted=False,
                    evidence=(
                        f"字面量未出现在 {len(blobs)} 个已定位的 libnccl.so 中"
                        f"（首个：{blobs[0][0]}）⇒ 该版本不认此变量，注入只会 warn"
                    ),
                )
            )
    return results


def format_nccl_support_report(env_delta: dict[str, str]) -> list[str]:
    """把 NCCL 变量的支持性结论渲染成显式文本（未注入 NCCL 变量时返回空列表）。

    三条口径都**显式**（§2.2）：已识别 / 该版本不认识（会静默失效）/ 探测不了（未确认）。
    绝不出现"注入了就当生效"的静默假设。
    """
    probe_vars = [
        knob.env_var
        for knob in DETERMINISM_ENV_KNOBS
        if knob.support_probe and knob.env_var in env_delta
    ]
    if not probe_vars:
        return []
    lines = [f"[determinism] NCCL 变量支持性静态探测（NCCL 版本：{nccl_version() or '未知'}）："]
    for support in probe_nccl_variable_support(probe_vars):
        if support.accepted is True:
            verdict = "已识别"
        elif support.accepted is False:
            verdict = "★ 该版本不认识它 —— 注入将静默失效，须改用其它钉定手段"
        else:
            verdict = "★ 未确认（探测不了）"
        lines.append(f"[determinism]   {support.env_var}: {verdict}；证据：{support.evidence}")
    lines.append(
        "[determinism]   注意：静态探测不替代运行时核实——NCCL 对未知/不可行取值只 warn，"
        "必须在 run 日志里确认这些变量真被接受。"
    )
    return lines


def determinism_artifact(switch: DeterminismSwitch) -> dict[str, Any] | None:
    """生成**落进产物**的确定性开关记录（便于事后核对，§2.2）；未启用 ⇒ ``None``。

    与打印共用同一份渲染（:func:`determinism_env_delta` / :func:`torch_determinism_steps`），
    因此"打印了什么"和"记了什么"不可能分叉（§1.4）。
    """
    if switch is None or not (switch.enabled or switch.probe_first_step):
        return None
    delta = determinism_env_delta(switch)
    return {
        "event": "determinism",
        "kind": "diagnostic",
        "enabled": bool(switch.enabled),
        "probe_first_step": bool(switch.probe_first_step),
        "spec": switch.to_spec(),
        "env": dict(sorted(delta.items())),
        "torch": torch_determinism_steps(switch),
        "side_effects": {
            knob.env_var: knob.side_effect for knob in enabled_env_knobs(switch)
        },
        "nccl_support": [
            dataclasses.asdict(support) for support in probe_nccl_variable_support(delta.keys())
        ],
        "nccl_version": nccl_version(),
    }
