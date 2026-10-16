"""AST 依赖边界扫描：验证模块间导入方向符合架构契约。

验证目标（C5 验收）：
1. ``flow/`` 不直接 import ``ripple/``（只能通过 trainer 间接调用）
2. ``ripple/`` 不 import ``flow/``（ripple 是纯计算层，零设施依赖）
3. ``parallel/scheduling/`` 不 import ``adapters/models/`` 或 ``trainer/``
4. 导入图无环（模块间无循环依赖）

实现策略：
- 使用 ``ast`` 模块解析所有 ``src/graspo/`` 下的 .py 文件
- 构建模块导入图，报告任何违反边界的情况
- 禁止列表为白名单式：允许的导入方向显式声明，其余均为违规
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest

# ── AST 解析工具 ──────────────────────────────────────────────────────────────


class ImportEdge(NamedTuple):
    """一条导入边：source 模块 import 了 target 模块。"""

    source: str  # 如 "graspo.flow.backend.native.transformer_model_adapter"
    target: str  # 如 "graspo.ripple.loss"
    lineno: int


def _collect_imports(file_path: Path, src_root: Path) -> list[ImportEdge]:
    """解析单个 .py 文件，提取所有 graspo 内部导入。"""
    source = _file_to_module(file_path, src_root)
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    edges: list[ImportEdge] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("graspo."):
                    edges.append(ImportEdge(source, alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("graspo."):
                # 完整模块名：from graspo.ripple.loss import ...
                for alias in node.names:
                    full = f"{node.module}.{alias.name}" if alias.name != "*" else node.module
                    edges.append(ImportEdge(source, full, node.lineno))
            elif node.module is None and node.level is not None and node.level > 0:
                # 相对导入：from .xxx import ...
                resolved = _resolve_relative(source, node.level, node.module)
                for alias in node.names:
                    full = f"{resolved}.{alias.name}" if alias.name != "*" else resolved
                    edges.append(ImportEdge(source, full, node.lineno))

    return edges


def _file_to_module(file_path: Path, src_root: Path) -> str:
    """将文件路径转换为模块名。"""
    rel = file_path.relative_to(src_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)  # rel 已包含 graspo/ 前缀，无需再加


def _resolve_relative(current_module: str, level: int, target: str | None) -> str:
    """解析相对导入的目标模块名。"""
    parts = current_module.split(".")
    # level=1 表示从当前包开始，level=2 表示从父包开始
    base = parts[: len(parts) - level] if level <= len(parts) else []
    if target:
        base.append(target)
    return ".".join(base)


def _collect_all_imports() -> list[ImportEdge]:
    """收集 src/graspo/ 下所有 .py 文件的内部导入。"""
    src_root = Path("src")  # _file_to_module 需要 src/ 作为根来解析 graspo.xxx
    if not (src_root / "graspo").is_dir():
        pytest.skip(f"src/graspo/ not found (cwd={Path.cwd()})")

    all_edges: list[ImportEdge] = []
    for py_file in sorted((src_root / "graspo").rglob("*.py")):
        try:
            all_edges.extend(_collect_imports(py_file, src_root))
        except SyntaxError:
            continue  # 跳过无法解析的文件
    return all_edges


# ── ★ task-arch-b1 新增：内存源码 / 子树 导入收集（守卫与阳性对照共用）──────────
#
# 与 `_collect_imports` 同口径的别名逻辑，但：
#   1. `_imports_from_source` 可直接吃**内存源码**（阳性/阴性对照用），无需落盘；
#   2. `_runtime_imports_from_source` 跳过 `if TYPE_CHECKING:` 的**块体** ——
#      该块体永不执行 ⇒ 不构成**运行时**依赖（`else` 分支照进）。
#      这是 1a/1b 收口的口径基础：只禁运行时依赖，仅类型引用允许。

#: 仓库根 / `src/` / `src/graspo` —— 绝对路径，不依赖 cwd（测试可从任意目录跑）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_GRASPO_ROOT = _SRC_ROOT / "graspo"


def _edges_of_import_node(node: ast.AST, module: str) -> list[ImportEdge]:
    """把一个 import 节点渲染成导入边（与 `_collect_imports` 同口径）。"""
    if isinstance(node, ast.Import):
        return [
            ImportEdge(module, alias.name, node.lineno)
            for alias in node.names
            if alias.name.startswith("graspo.")
        ]
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.startswith("graspo."):
            return [
                ImportEdge(
                    module,
                    f"{node.module}.{alias.name}" if alias.name != "*" else node.module,
                    node.lineno,
                )
                for alias in node.names
            ]
        if node.module is None and node.level is not None and node.level > 0:
            resolved = _resolve_relative(module, node.level, node.module)
            return [
                ImportEdge(
                    module,
                    f"{resolved}.{alias.name}" if alias.name != "*" else resolved,
                    node.lineno,
                )
                for alias in node.names
            ]
    return []


def _imports_from_source(source: str, module: str) -> list[ImportEdge]:
    """解析**内存源码**，返回全部 graspo 内部导入边（全层级，含 TYPE_CHECKING 块内）。

    与 `_collect_all_imports` 同口径 —— 供层规则（1–5）的注入式阳性对照使用。
    """
    tree = ast.parse(source)
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        edges.extend(_edges_of_import_node(node, module))
    return edges


def _is_type_checking_guard(node: ast.If) -> bool:
    """`if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:` 守卫判断。"""
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _runtime_imports_from_source(source: str, module: str) -> list[ImportEdge]:
    """解析内存源码，返回**运行时**导入边（⛔ 跳过 `if TYPE_CHECKING:` 块体）。"""
    out: list[ImportEdge] = []

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            # 块体永不执行 ⇒ 不算运行时依赖；`else` / `elif` 分支照进。
            for stmt in node.orelse:
                _walk(stmt)
            return
        out.extend(_edges_of_import_node(node, module))
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(ast.parse(source))
    return out


def _subtree_runtime_imports(subtree: str) -> list[ImportEdge]:
    """扫 `src/graspo/<subtree>/**`（全层级，含函数体内延迟 import）的运行时导入边。"""
    root = _GRASPO_ROOT / subtree
    assert root.is_dir(), f"subtree not found: {root}"
    edges: list[ImportEdge] = []
    for py_file in sorted(root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        edges.extend(
            _runtime_imports_from_source(
                py_file.read_text(encoding="utf-8"), _file_to_module(py_file, _SRC_ROOT)
            )
        )
    return edges


# ── 边界规则 ──────────────────────────────────────────────────────────────────


def _is_under(module: str, prefix: str) -> bool:
    """模块是否在指定前缀下。"""
    return module == prefix or module.startswith(prefix + ".")


def _is_relative_import(module: str) -> bool:
    """是否为相对导入（以 . 开头）。"""
    return module.startswith(".")


# ── ★ task-arch-b1：四条层规则抽成**纯函数**（可注入内存边做阳性对照）──────────
#
# 纯函数的收益：规则本身可以被"喂一段违规边 ⇒ 必须报"直接检验。
# ⛔ 若规则实现被删空（恒返回 []），对应阳性对照用例**必然红** —— 这正是
# 「不能自证能失败的检查」的修法（原实现把规则与文件系统扫描焊死，无法注入）。


def find_ripple_importing_flow(edges: list[ImportEdge]) -> list[str]:
    """规则 1：`ripple/` 不得 import `flow/`（纯计算层，零设施依赖）。"""
    return [
        f"  {edge.source}:{edge.lineno} imports {edge.target}"
        for edge in edges
        if _is_under(edge.source, "graspo.ripple") and _is_under(edge.target, "graspo.flow")
    ]


def find_scheduling_importing_models_or_trainer(edges: list[ImportEdge]) -> list[str]:
    """规则 2：`flow/backend/native/parallel/scheduling/` 不得 import models/ 或 trainer/。"""
    violations: list[str] = []
    for edge in edges:
        if _is_under(edge.source, "graspo.flow.backend.native.parallel.scheduling"):
            if _is_under(edge.target, "graspo.flow.backend.native.models"):
                violations.append(f"  {edge.source}:{edge.lineno} imports models: {edge.target}")
            if _is_under(edge.target, "graspo.flow.backend.native.trainer"):
                violations.append(f"  {edge.source}:{edge.lineno} imports trainer: {edge.target}")
    return violations


def find_core_importing_flow_or_ripple(edges: list[ImportEdge]) -> list[str]:
    """规则 3：`core/` 不得 import `flow/` 或 `ripple/`（跨层契约不依赖上层）。"""
    violations: list[str] = []
    for edge in edges:
        if _is_under(edge.source, "graspo.core"):
            if _is_under(edge.target, "graspo.flow"):
                violations.append(f"  {edge.source}:{edge.lineno} imports flow: {edge.target}")
            if _is_under(edge.target, "graspo.ripple"):
                violations.append(f"  {edge.source}:{edge.lineno} imports ripple: {edge.target}")
    return violations


def _layer_group(module: str) -> str:
    """取架构层：graspo.flow.xxx → graspo.flow, graspo.ripple.xxx → graspo.ripple"""
    parts = module.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])  # graspo.flow, graspo.ripple, graspo.core
    return module


def find_cross_layer_cycles(edges: list[ImportEdge]) -> list[str]:
    """规则 4：不同架构层之间不得双向导入（A→B 且 B→A）。

    同层内的循环（如 flow.backend.native.models ↔ flow.lora）不在此检查范围，
    因为同层模块间可以有合理的内部依赖。
    """
    graph: dict[str, set[str]] = {}
    for edge in edges:
        src = _layer_group(edge.source)
        tgt = _layer_group(edge.target)
        if src == tgt:
            continue  # 同层内导入不检查
        graph.setdefault(src, set()).add(tgt)

    cycles: list[str] = []
    for src, tgts in graph.items():
        for tgt in tgts:
            if tgt in graph and src in graph[tgt] and src < tgt:
                cycles.append(f"  {src} ↔ {tgt}")
    return cycles


def test_flow_imports_ripple_is_documented():
    """flow → ripple 是架构文档明确的设计方向（docs/architecture.md）。

    本测试不阻止 flow→ripple 导入，而是记录当前导入数量，防止静默增长。
    若新增 flow→ripple 导入超出预期，需同步更新架构文档。
    """
    all_imports = _collect_all_imports()
    flow_ripple: list[str] = []

    for edge in all_imports:
        if _is_under(edge.source, "graspo.flow") and _is_under(edge.target, "graspo.ripple"):
            flow_ripple.append(f"  {edge.source}:{edge.lineno} → {edge.target}")

    # 当前基线：41 条 flow→ripple 导入（18 个文件），这是设计的合法依赖方向
    # 若此数字变化，说明模块边界发生变化，需检查是否合理
    assert len(flow_ripple) >= 30, (
        f"flow→ripple imports dropped below 30 ({len(flow_ripple)}); "
        "verify architecture docs still match"
    )
    # 上限宽松：允许增长但不允许爆炸（>100 则需检查）
    assert len(flow_ripple) <= 100, (
        f"flow→ripple imports grew to {len(flow_ripple)} (>100); check for unintended coupling"
    )


def test_ripple_does_not_import_flow():
    """ripple/ 不 import flow/ 的任何模块。

    ripple 是纯计算层，零设施依赖，不能引入 flow 的任何内容。
    """
    violations = find_ripple_importing_flow(_collect_all_imports())

    assert not violations, (
        f"ripple/ must not import flow/ ({len(violations)} violations):\n" + "\n".join(violations)
    )


def test_scheduling_does_not_import_models_or_trainer():
    """parallel/scheduling/ 不 import adapters/models/ 或 trainer/。

    调度层只关心何时执行 forward/backward，与模型实现和训练循环解耦。
    """
    violations = find_scheduling_importing_models_or_trainer(_collect_all_imports())

    assert not violations, (
        f"scheduling/ must not import adapters/models/ or trainer/ "
        f"({len(violations)} violations):\n" + "\n".join(violations)
    )


def test_core_does_not_import_flow_or_ripple():
    """core/ 不 import flow/ 或 ripple/。

    core 是基础设施层（schema、config），不应依赖上层模块。
    """
    violations = find_core_importing_flow_or_ripple(_collect_all_imports())

    assert not violations, (
        f"core/ must not import flow/ or ripple/ ({len(violations)} violations):\n"
        + "\n".join(violations)
    )


def test_no_circular_imports_detected():
    """AST 级别检测跨层循环依赖：不同架构层之间不应双向导入。"""
    cycles = find_cross_layer_cycles(_collect_all_imports())

    assert not cycles, f"cross-layer circular imports detected ({len(cycles)}):\n" + "\n".join(
        cycles
    )


# ── 架构合规性报告（信息性）───────────────────────────────────────────────────


def test_architecture_report():
    """打印完整的模块导入拓扑（信息性，不失败）。"""
    all_imports = _collect_all_imports()
    if not all_imports:
        pytest.skip("No graspo imports found")

    # 按源模块分组
    by_source: dict[str, list[ImportEdge]] = {}
    for edge in all_imports:
        by_source.setdefault(edge.source, []).append(edge)

    # 统计各层的导入目标
    layers = {
        "graspo.flow": {"ripple": 0, "core": 0, "flow": 0},
        "graspo.ripple": {"flow": 0, "core": 0, "ripple": 0},
        "graspo.core": {"flow": 0, "ripple": 0, "core": 0},
    }

    for edge in all_imports:
        for layer_name, counts in layers.items():
            if _is_under(edge.source, layer_name):
                for target_layer in counts:
                    if _is_under(edge.target, f"graspo.{target_layer}"):
                        counts[target_layer] += 1

    # 打印报告（用于调试）
    for layer, counts in layers.items():
        parts = [f"{k}={v}" for k, v in counts.items()]
        print(f"  {layer}: {', '.join(parts)}")

    # 至少验证 ripples 不导入 flow（已在其他测试中验证，此处仅信息性）
    assert True


# ═══════════════════════════════════════════════════════════════════════════════
# ★ P1：`ripple/` 纯算法层边界机核（U4 裁定 + 指挥部 2026-10-10 P1 裁决）
#
# 规则（每条都有阳性对照 —— 注入一个已知违规，它**必须**被抓住；⛔ 不接受恒 PASS）：
#   R1  禁 import：`ripple/` 不得 import `graspo.flow` / `graspo.eval` / `graspo.cli`
#       / `graspo.export`（★ 2026-10-15 指挥授权：新增顶层导出共享包后补上的纯强化项）
#   R2  禁词：`torch.cuda` / `torch.distributed` / `ProcessGroup` / `init_process_group`
#       / `all_reduce` / `barrier` / `broadcast`（含 `dist.*` 集合通信）
#   R3  ★ device **选择**：`device=` 实参不得是**设备字符串字面量**（`"cuda"`/`"cpu"`/…）
#       、设备序号、或新建 device（`torch.device(...)`）；
#       ✅ **允许** `device=<expr>.device` —— 读取既有张量的设备并把同一 device 传下去
#       （"张量运算的设备保持"，无设备选择语义）
#   R4  禁环境读取：`os.environ` / `os.getenv` / `os.putenv`
#   R5  禁文件 IO / 网络 / 子进程：`open(` / `subprocess` / `socket` / `requests`
#
# ✅ 明确**允许**（不是违规）：`import torch`（张量运算，算法数学底座）、`pydantic`、
#    `pathlib`（纯路径字符串计算，不读文件）、`graspo.core`（ripple→core 方向合法）。
# ═══════════════════════════════════════════════════════════════════════════════

_RIPPLE_ROOT = Path(__file__).resolve().parents[1] / "src" / "graspo" / "ripple"

_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "graspo.flow",
    "graspo.eval",
    "graspo.cli",
    "graspo.export",
)

#: R2：属性链形态的禁词（`torch.cuda.*` / `torch.distributed.*` / `dist.<集合通信>`）
_FORBIDDEN_ATTR = re.compile(
    r"torch\.cuda"
    r"|torch\.distributed"
    r"|(?:^|\.)dist\.(?:all_reduce|all_gather|barrier|broadcast|reduce|"
    r"broadcast_object_list|all_gather_object|gather|scatter)"
)
#: R2：裸标识符形态的禁词
_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "ProcessGroup",
        "init_process_group",
        "destroy_process_group",
        "new_group",
        "all_reduce",
        "barrier",
        "broadcast",
    }
)

#: R3：设备名字符串（"cuda" / "cpu" / "cuda:0" / "mps" …）
_DEVICE_LITERAL = re.compile(r"^(?:cuda|cpu|mps|xpu|npu)(?::\d+)?$")

#: R4：环境读取（`os.path.*` 等纯路径操作**不**在此列）
_ENV_ACCESS = re.compile(r"^os\.(?:environ(?:\..*)?|getenv|putenv|getenvb|putenvb)$")

#: R5：文件 IO / 网络 / 子进程
_FORBIDDEN_CALLS: frozenset[str] = frozenset({"open"})
_FORBIDDEN_CALL_PREFIXES: tuple[str, ...] = (
    "subprocess.",
    "socket.",
    "requests.",
    "urllib.",
    "http.",
)


def _facility_rule(dotted: str) -> str | None:
    """返回该点号路径命中的规则号（R4/R5），未命中返回 None。"""
    if _ENV_ACCESS.match(dotted):
        return "R4"
    if dotted in _FORBIDDEN_CALLS or dotted.startswith(_FORBIDDEN_CALL_PREFIXES):
        return "R5"
    return None


def _call_dotted(node: ast.expr) -> str:
    """把 Call.func 渲染成点号路径（`os.getenv` / `open` / `torch.device`）。"""
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_device_selection(node: ast.expr) -> bool:
    """`device=` 的实参是否是"设备选择"（R3 点名的形态）。

    ⛔ 设备字符串字面量 / 设备序号 / 新建 device 对象；
    ✅ `old.device`（Attribute）、变量名（Name）—— 是"设备保持/透传"，不是选择。
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return bool(_DEVICE_LITERAL.match(node.value)) or node.value == ""
        if isinstance(node.value, int):  # device=0 是设备序号选择
            return True
        return False
    if isinstance(node, ast.Call):  # torch.device("cuda") 等 = 新建 device
        return True
    return False


def scan_ripple_source(source: str, filename: str = "<memory>") -> list[str]:
    """扫描一段源码，返回 `ripple/` 边界违规描述（空列表 = 合规）。"""
    tree = ast.parse(source)
    out: list[str] = []

    for node in ast.walk(tree):
        # ── R1 禁 import ────────────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                    out.append(f"R1 {filename}:{node.lineno} 禁 import {alias.name}")
                if _FORBIDDEN_ATTR.search(alias.name):
                    out.append(f"R2 {filename}:{node.lineno} 禁 import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                out.append(f"R1 {filename}:{node.lineno} 禁 import {module}")
            if _FORBIDDEN_ATTR.search(module):
                out.append(f"R2 {filename}:{node.lineno} 禁 import {module}")
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    out.append(f"R2 {filename}:{node.lineno} 禁词标识符 {alias.name!r}")

        # ── R2 禁词 ─────────────────────────────────────────────────────────
        if isinstance(node, ast.Attribute) and _FORBIDDEN_ATTR.search(ast.unparse(node)):
            out.append(f"R2 {filename}:{node.lineno} 禁词 {ast.unparse(node)!r}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            out.append(f"R2 {filename}:{node.lineno} 禁词标识符 {node.id!r}")

        # ── R4/R5 环境读取与设施调用（属性形态：`os.environ[...]` 也是 Attribute）──
        if isinstance(node, ast.Attribute):
            rule = _facility_rule(ast.unparse(node))
            if rule:
                out.append(f"{rule} {filename}:{node.lineno} 禁调用 {ast.unparse(node)}")

        # ── R3 device 选择 ──────────────────────────────────────────────────
        if isinstance(node, ast.Call):
            dotted = _call_dotted(node.func)
            if dotted == "torch.device":
                out.append(f"R3 {filename}:{node.lineno} 禁新建 device：torch.device(...)")
            for kw in node.keywords:
                if kw.arg == "device" and _is_device_selection(kw.value):
                    out.append(
                        f"R3 {filename}:{node.lineno} 禁 device 选择："
                        f"device={ast.unparse(kw.value)}（允许 device=<expr>.device）"
                    )
            # ── R4 / R5 环境与设施调用 ──────────────────────────────────────
            rule = _facility_rule(dotted)
            if rule:
                out.append(f"{rule} {filename}:{node.lineno} 禁调用 {dotted}")

    return out


def _scan_ripple_tree() -> list[str]:
    """扫描真实 `ripple/` 树（只读）。"""
    violations: list[str] = []
    for path in sorted(_RIPPLE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_RIPPLE_ROOT.parent.parent.parent)
        violations += scan_ripple_source(path.read_text(encoding="utf-8"), str(rel))
    return violations


#: 阳性对照用例：每一条**必须**被 scanner 抓住。
_POSITIVE_CASES: tuple[tuple[str, str], ...] = (
    ("R1", "from graspo.flow.backend.native.runtime import GraspoFlowRuntime\n"),
    ("R1", "import graspo.eval.guard\n"),
    ("R2", "import torch\ntorch.cuda.is_available()\n"),
    ("R2", "import torch.distributed as dist\ndist.all_reduce(t)\n"),
    ("R2", "from torch.distributed import ProcessGroup\n"),
    ("R2", "init_process_group('nccl')\n"),
    ("R2", "barrier()\n"),
    ("R3", "import torch\nt = torch.zeros(1, device=\"cuda\")\n"),
    ("R3", "import torch\nt = torch.zeros(1, device=\"cpu\")\n"),
    ("R3", "import torch\nt = torch.zeros(1, device=0)\n"),
    ("R3", "import torch\nt = torch.zeros(1, device=torch.device(\"cuda\"))\n"),
    ("R4", "import os\nos.environ[\"X\"]\n"),
    ("R4", "import os\nos.getenv(\"X\")\n"),
    ("R5", "open('/tmp/x', 'w')\n"),
    ("R5", "import subprocess\nsubprocess.run(['ls'])\n"),
)

#: 阴性对照：这些**必须不**被抓住（否则机核变成"恒 FAIL"的噪声源）。
_NEGATIVE_CASES: tuple[str, ...] = (
    "import torch\nimport pydantic\nfrom pathlib import Path\n",
    "import torch\nt = torch.zeros(1, device=old_log_probs.device)\n",
    "import torch\nt = torch.zeros(1, device=ref.device)\n",
    "import torch\nt = torch.zeros(1)  # 默认 device，无选择语义\n",
    "from graspo.core.schema import Sample\n",
    "from pathlib import Path\np = Path(data_dir) / 'a.png'\n",
    # ★ 主席 2026-10-10 裁定 3：`pathlib` 纯路径运算（含 `.resolve()`）**允许**，不得误报
    "from pathlib import Path\np = (Path(base) / rel).resolve()\n",
    "import torch\nt = torch.zeros(1, device=some_device)\n",
)


@pytest.mark.parametrize(("rule", "snippet"), _POSITIVE_CASES)
def test_ripple_boundary_scanner_catches_injected_violation(rule: str, snippet: str):
    """★ 阳性对照：注入一个已知违规 ⇒ scanner **必须**抓住（⛔ 不接受恒 PASS）。"""
    hits = scan_ripple_source(snippet, "<injected>")
    assert any(h.startswith(f"{rule} ") for h in hits), f"注入的 {rule} 违规未被抓住: {hits}"


@pytest.mark.parametrize("snippet", _NEGATIVE_CASES)
def test_ripple_boundary_scanner_allows_legal_code(snippet: str):
    """★ 阴性对照：合法写法（含 `device=<expr>.device`）必须**不**被误报。"""
    assert scan_ripple_source(snippet, "<legal>") == []


def test_ripple_layer_is_within_boundary():
    """真树机核：`ripple/` 全树 0 违规（R1–R5）。"""
    violations = _scan_ripple_tree()
    assert not violations, f"ripple/ 边界违规 {len(violations)} 条：\n" + "\n".join(violations)


# ═══════════════════════════════════════════════════════════════════════════════
# ★ task-arch-b1（Phase 1a/1b/1c/1d）：结构性越界收口 + 守卫
#
#   1d  四条层规则的**注入式阳性对照**（此前只有 ripple 的 R1–R5 扫描器有对照，
#       四条层规则**无** ⇒ 不能自证"注入越界会红"）。
#   1a  `flow.adapter` 子树 ⛔ **运行时** import `flow.backend`
#       （契约层不得直取具体后端实现；收口形态 = `if TYPE_CHECKING:` 仅类型引用）。
#   1b  `cli` 子树 ⛔ **运行时**直取 `flow.backend.{native,msswift}`
#       （只许 `flow.backend.selection` facade 与 `flow.backend.base` 接口）。
#   1c  `flow → export` 是**登记允许**方向（export = 共享叶子层，反向已由
#       `tests/export/test_boundaries.py` 强制为假）；未登记的新边 ⇒ 红。
#
# ★ 口径（关键）：1a/1b 的守卫只禁**运行时**导入 —— `if TYPE_CHECKING:` 的块体
#   永不执行，不构成运行时依赖，故允许（1a 的收口形态正是它）。层规则 1–4 沿用
#   原口径（全层级扫描，含 TYPE_CHECKING 块内），⛔ 不因新守卫而放宽。
# ═══════════════════════════════════════════════════════════════════════════════


# ── 1d：四条层规则 × 注入式阳性对照 + 阴性对照 ──────────────────────────────


def test_layer_rule_ripple_to_flow_catches_injected_violation():
    """★ 1d 阳性对照（规则 1）：`ripple → flow` 注入 ⇒ 必红。

    （若 `find_ripple_importing_flow` 被删空/恒返回 []，本用例**必然** FAIL。）
    """
    edges = _imports_from_source(
        "from graspo.flow.backend.native.runtime import GraspoFlowRuntime\n",
        "graspo.ripple.annotation.advantages",
    )
    assert find_ripple_importing_flow(edges) == [
        "  graspo.ripple.annotation.advantages:1 imports "
        "graspo.flow.backend.native.runtime.GraspoFlowRuntime"
    ]


def test_layer_rule_ripple_to_flow_allows_legal_import():
    """★ 1d 阴性对照（规则 1）：`ripple → core` 合法 ⇒ 绿。"""
    edges = _imports_from_source(
        "from graspo.core.schema import Sample\n", "graspo.ripple.annotation.advantages"
    )
    assert find_ripple_importing_flow(edges) == []


def test_layer_rule_scheduling_catches_injected_models_and_trainer_violation():
    """★ 1d 阳性对照（规则 2）：`scheduling → models` 与 `→ trainer` 注入 ⇒ 必红（2 条）。"""
    edges = _imports_from_source(
        "from graspo.flow.backend.native.models.qwen3.model_adapter import Qwen3ModelAdapter\n"
        "def f():\n"
        "    from graspo.flow.backend.native.trainer.rollout import rollout_loop\n",
        "graspo.flow.backend.native.parallel.scheduling.schedule_1f1b",
    )
    assert find_scheduling_importing_models_or_trainer(edges) == [
        "  graspo.flow.backend.native.parallel.scheduling.schedule_1f1b:1 imports models: "
        "graspo.flow.backend.native.models.qwen3.model_adapter.Qwen3ModelAdapter",
        "  graspo.flow.backend.native.parallel.scheduling.schedule_1f1b:3 imports trainer: "
        "graspo.flow.backend.native.trainer.rollout.rollout_loop",
    ]


def test_layer_rule_scheduling_allows_legal_import():
    """★ 1d 阴性对照（规则 2）：`scheduling → 同层通信层` 合法 ⇒ 绿。"""
    edges = _imports_from_source(
        "from graspo.flow.backend.native.parallel.pipeline_comm import PipeComm\n",
        "graspo.flow.backend.native.parallel.scheduling.schedule_1f1b",
    )
    assert find_scheduling_importing_models_or_trainer(edges) == []


def test_layer_rule_core_catches_injected_flow_and_ripple_violation():
    """★ 1d 阳性对照（规则 3）：`core → flow` 与 `core → ripple` 注入 ⇒ 必红（2 条）。"""
    edges = _imports_from_source(
        "from graspo.flow.backend.base import RLBackend\nfrom graspo.ripple.loss import loss_fn\n",
        "graspo.core.result_judge",
    )
    assert find_core_importing_flow_or_ripple(edges) == [
        "  graspo.core.result_judge:1 imports flow: graspo.flow.backend.base.RLBackend",
        "  graspo.core.result_judge:2 imports ripple: graspo.ripple.loss.loss_fn",
    ]


def test_layer_rule_core_allows_legal_import():
    """★ 1d 阴性对照（规则 3）：`core → core` 合法 ⇒ 绿。"""
    edges = _imports_from_source(
        "from graspo.core.schema import GraspoConfig\n", "graspo.core.result_judge"
    )
    assert find_core_importing_flow_or_ripple(edges) == []


def test_layer_rule_cycles_catches_injected_cross_layer_cycle():
    """★ 1d 阳性对照（规则 4）：注入 `core → flow` ＋ `flow → core` ⇒ 必红。"""
    edges = _imports_from_source(
        "from graspo.flow.data_io import load_jsonl\n", "graspo.core.schema"
    ) + _imports_from_source("from graspo.core.schema import GraspoConfig\n", "graspo.flow.data_io")
    assert find_cross_layer_cycles(edges) == ["  graspo.core ↔ graspo.flow"]


def test_layer_rule_cycles_allows_single_direction():
    """★ 1d 阴性对照（规则 4）：单向 `flow → core` 合法 ⇒ 绿。"""
    edges = _imports_from_source(
        "from graspo.core.schema import GraspoConfig\n", "graspo.flow.data_io"
    )
    assert find_cross_layer_cycles(edges) == []


# ── 1a：契约层（flow/adapter）⛔ 运行时依赖具体后端实现 ─────────────────────

#: 1b：`cli` 允许的 `flow.backend` 入口（facade / 接口），其余子模块一律违规。
_CLI_ALLOWED_BACKEND_SUBMODULES: tuple[str, ...] = (
    "graspo.flow.backend.selection",
    "graspo.flow.backend.base",
)

#: 1c：**登记允许**的 `flow → export` 边（逐处登记；新边必须先登记再改本表）。
_REGISTERED_FLOW_TO_EXPORT: frozenset[tuple[str, str]] = frozenset(
    {
        (
            "graspo.flow.backend.native.lora.lora_io",
            "graspo.export.paths.prepare_output_dir",
        ),
        ("graspo.flow.backend.native.trainer.checkpoint", "graspo.export.export_checkpoint"),
        ("graspo.flow.backend.native.trainer.helpers", "graspo.export.paths.prepare_output_dir"),
    }
)


def find_adapter_to_backend_runtime_violations(edges: list[ImportEdge]) -> list[str]:
    """1a 规则：`flow/adapter` 子树 ⛔ 运行时 import `flow/backend`（具体后端实现）。"""
    return [
        f"  {edge.source}:{edge.lineno} imports {edge.target}"
        for edge in edges
        if _is_under(edge.source, "graspo.flow.adapter")
        and _is_under(edge.target, "graspo.flow.backend")
    ]


def find_cli_to_backend_impl_runtime_violations(edges: list[ImportEdge]) -> list[str]:
    """1b 规则：`cli` 子树 ⛔ 运行时直取 `flow.backend.{native,msswift}` 等实现子模块。"""
    return [
        f"  {edge.source}:{edge.lineno} imports {edge.target}"
        f"（cli 只许经 {' / '.join(_CLI_ALLOWED_BACKEND_SUBMODULES)}）"
        for edge in edges
        if _is_under(edge.source, "graspo.cli")
        and _is_under(edge.target, "graspo.flow.backend")
        and not any(_is_under(edge.target, allowed) for allowed in _CLI_ALLOWED_BACKEND_SUBMODULES)
    ]


def find_unregistered_flow_to_export(edges: list[ImportEdge]) -> list[str]:
    """1c 规则：`flow → export` 方向允许，但每条边必须**逐处登记**。"""
    return [
        f"  {edge.source}:{edge.lineno} → {edge.target}"
        for edge in edges
        if _is_under(edge.source, "graspo.flow")
        and _is_under(edge.target, "graspo.export")
        and (edge.source, edge.target) not in _REGISTERED_FLOW_TO_EXPORT
    ]


def test_adapter_does_not_import_backend_at_runtime():
    """★ 1a 真树守卫：`flow/adapter` 子树运行时 0 条 `flow.backend` 导入。"""
    violations = find_adapter_to_backend_runtime_violations(
        _subtree_runtime_imports("flow/adapter")
    )
    assert not violations, (
        "flow/adapter（契约层）⛔ 不得在运行时 import flow/backend（具体后端实现）：\n"
        + "\n".join(violations)
        + "\n（仅类型引用请放 `if TYPE_CHECKING:`）"
    )


def test_adapter_backend_guard_catches_planted_runtime_import():
    """★ 1a 阳性对照：把收口掉的 import **注回模块级** ⇒ 必红。"""
    planted = "from graspo.flow.backend.native.runtime import NativeGeneration\n"
    hits = find_adapter_to_backend_runtime_violations(
        _runtime_imports_from_source(planted, "graspo.flow.adapter.base_graspo_flow_adapter")
    )
    assert hits == [
        "  graspo.flow.adapter.base_graspo_flow_adapter:1 imports "
        "graspo.flow.backend.native.runtime.NativeGeneration"
    ]


def test_adapter_backend_guard_catches_deferred_import_inside_function():
    """★ 1a 阳性对照（全层级）：函数体内的延迟 import 同样命中（不是只扫模块级）。"""
    planted = "def f():\n    from graspo.flow.backend.native.runtime import NativeGeneration\n"
    hits = find_adapter_to_backend_runtime_violations(
        _runtime_imports_from_source(planted, "graspo.flow.adapter.contract")
    )
    assert len(hits) == 1 and hits[0].startswith("  graspo.flow.adapter.contract:2 imports ")


def test_adapter_backend_guard_allows_type_checking_and_legal_imports():
    """★ 1a 阴性对照：`if TYPE_CHECKING:` 仅类型引用 + 合法方向 ⇒ 绿。"""
    legal = (
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING, Any\n"
        "from graspo.ripple.parsing.completion import ParsedCompletion\n"
        "if TYPE_CHECKING:\n"
        "    from graspo.flow.backend.native.runtime import NativeGeneration\n"
    )
    assert (
        find_adapter_to_backend_runtime_violations(
            _runtime_imports_from_source(legal, "graspo.flow.adapter.base_graspo_flow_adapter")
        )
        == []
    )


def test_cli_does_not_import_backend_impl_at_runtime():
    """★ 1b 真树守卫：`cli` 子树运行时 0 条 `flow.backend.{native,msswift}` 直取。

    非空转断言：真树里必须**看得见**至少一条合法 `flow.backend.selection` 边
    （否则守卫"绿"可能只是因为它一无所见）。
    """
    edges = _subtree_runtime_imports("cli")
    violations = find_cli_to_backend_impl_runtime_violations(edges)
    assert not violations, (
        "cli（入口层）⛔ 不得在运行时直取 flow/backend 的具体实现：\n"
        + "\n".join(violations)
        + "\n（请经 flow/backend/selection.py 的 facade）"
    )
    allowed = [
        e
        for e in edges
        if _is_under(e.source, "graspo.cli")
        and any(_is_under(e.target, a) for a in _CLI_ALLOWED_BACKEND_SUBMODULES)
    ]
    assert allowed, "守卫空转：真树里没有一条 cli → flow.backend.selection/base 合法边"


def test_cli_backend_guard_catches_planted_msswift_import():
    """★ 1b 阳性对照（msswift 侧）：注回 `app.py:260` 的原直取 ⇒ 必红。"""
    planted = "from graspo.flow.backend.msswift._config_mapping import launcher_env\n"
    hits = find_cli_to_backend_impl_runtime_violations(
        _runtime_imports_from_source(planted, "graspo.cli.app")
    )
    assert len(hits) == 1 and hits[0].startswith(
        "  graspo.cli.app:1 imports graspo.flow.backend.msswift._config_mapping.launcher_env"
    )


def test_cli_backend_guard_catches_planted_native_import():
    """★ 1b 阳性对照（native 侧）：注回 `tools.py:28` 的原直取 ⇒ 必红。"""
    planted = (
        "def run_evaluate():\n"
        "    from graspo.flow.backend.native.runtime import GraspoFlowRuntime\n"
        "    return GraspoFlowRuntime.from_config(None)\n"
    )
    hits = find_cli_to_backend_impl_runtime_violations(
        _runtime_imports_from_source(planted, "graspo.cli.tools")
    )
    assert len(hits) == 1 and hits[0].startswith(
        "  graspo.cli.tools:2 imports graspo.flow.backend.native.runtime.GraspoFlowRuntime"
    )


def test_cli_backend_guard_allows_facade_and_type_checking():
    """★ 1b 阴性对照：`selection` / `base` facade ＋ `if TYPE_CHECKING:` 仅类型引用 ⇒ 绿。"""
    legal = (
        "from typing import TYPE_CHECKING\n"
        "from graspo.flow.backend.selection import create_runtime, launcher_env, select_backend\n"
        "from graspo.flow.backend.base import RLBackend\n"
        "if TYPE_CHECKING:\n"
        "    from graspo.flow.backend.native.runtime import GraspoFlowRuntime\n"
    )
    assert (
        find_cli_to_backend_impl_runtime_violations(
            _runtime_imports_from_source(legal, "graspo.cli.tools")
        )
        == []
    )


def test_flow_to_export_edges_are_all_registered():
    """★ 1c 真树守卫：`flow → export` 方向允许，但每条边必须在登记表里。"""
    edges = _subtree_runtime_imports("flow")
    violations = find_unregistered_flow_to_export(edges)
    assert not violations, (
        "flow → export 出现**未登记**新边（方向本身合法 —— export 是共享叶子层 ——"
        "但必须逐处登记到 _REGISTERED_FLOW_TO_EXPORT 并在 docs/architecture.md 登记）：\n"
        + "\n".join(violations)
    )
    registered = [
        (e.source, e.target)
        for e in edges
        if _is_under(e.source, "graspo.flow") and _is_under(e.target, "graspo.export")
    ]
    assert sorted(registered) == sorted(_REGISTERED_FLOW_TO_EXPORT), (
        "登记表与真树不一致（登记项失效或漏登记）：\n"
        f"  真树 = {sorted(registered)}\n  表 = {sorted(_REGISTERED_FLOW_TO_EXPORT)}"
    )


def test_flow_export_guard_catches_unregistered_edge():
    """★ 1c 阳性对照：未登记的 `flow → export` 新边 ⇒ 必红。"""
    planted = "from graspo.export.paths import prepare_output_dir\n"
    hits = find_unregistered_flow_to_export(
        _runtime_imports_from_source(planted, "graspo.flow.backend.native.trainer.brand_new")
    )
    assert hits == [
        "  graspo.flow.backend.native.trainer.brand_new:1 → graspo.export.paths.prepare_output_dir"
    ]


def test_flow_export_guard_allows_registered_edges():
    """★ 1c 阴性对照：三条**已登记**边的精确形态 ⇒ 绿。"""
    edges = _imports_from_source(
        "from graspo.export.paths import prepare_output_dir\n",
        "graspo.flow.backend.native.lora.lora_io",
    )
    assert find_unregistered_flow_to_export(edges) == []
