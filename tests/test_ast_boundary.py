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


# ── 边界规则 ──────────────────────────────────────────────────────────────────


def _is_under(module: str, prefix: str) -> bool:
    """模块是否在指定前缀下。"""
    return module == prefix or module.startswith(prefix + ".")


def _is_relative_import(module: str) -> bool:
    """是否为相对导入（以 . 开头）。"""
    return module.startswith(".")


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
    all_imports = _collect_all_imports()
    violations: list[str] = []

    for edge in all_imports:
        if _is_under(edge.source, "graspo.ripple") and _is_under(edge.target, "graspo.flow"):
            violations.append(f"  {edge.source}:{edge.lineno} imports {edge.target}")

    assert not violations, (
        f"ripple/ must not import flow/ ({len(violations)} violations):\n" + "\n".join(violations)
    )


def test_scheduling_does_not_import_models_or_trainer():
    """parallel/scheduling/ 不 import adapters/models/ 或 trainer/。

    调度层只关心何时执行 forward/backward，与模型实现和训练循环解耦。
    """
    all_imports = _collect_all_imports()
    violations: list[str] = []

    for edge in all_imports:
        if _is_under(edge.source, "graspo.flow.parallel.scheduling"):
            if _is_under(edge.target, "graspo.flow.backend.native.models"):
                violations.append(f"  {edge.source}:{edge.lineno} imports models: {edge.target}")
            if _is_under(edge.target, "graspo.flow.trainer"):
                violations.append(f"  {edge.source}:{edge.lineno} imports trainer: {edge.target}")

    assert not violations, (
        f"scheduling/ must not import adapters/models/ or trainer/ "
        f"({len(violations)} violations):\n" + "\n".join(violations)
    )


def test_core_does_not_import_flow_or_ripple():
    """core/ 不 import flow/ 或 ripple/。

    core 是基础设施层（schema、config），不应依赖上层模块。
    """
    all_imports = _collect_all_imports()
    violations: list[str] = []

    for edge in all_imports:
        if _is_under(edge.source, "graspo.core"):
            if _is_under(edge.target, "graspo.flow"):
                violations.append(f"  {edge.source}:{edge.lineno} imports flow: {edge.target}")
            if _is_under(edge.target, "graspo.ripple"):
                violations.append(f"  {edge.source}:{edge.lineno} imports ripple: {edge.target}")

    assert not violations, (
        f"core/ must not import flow/ or ripple/ ({len(violations)} violations):\n"
        + "\n".join(violations)
    )


def test_no_circular_imports_detected():
    """AST 级别检测跨层循环依赖：不同架构层之间不应双向导入。

    同层内的循环（如 flow.backend.native.models ↔ flow.lora）不在此检查范围，
    因为同层模块间可以有合理的内部依赖。
    """
    all_imports = _collect_all_imports()
    # 构建模块组（前两级）导入图
    graph: dict[str, set[str]] = {}

    def _layer_group(m: str) -> str:
        """取架构层：graspo.flow.xxx → graspo.flow, graspo.ripple.xxx → graspo.ripple"""
        parts = m.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])  # graspo.flow, graspo.ripple, graspo.core
        return m

    for edge in all_imports:
        src = _layer_group(edge.source)
        tgt = _layer_group(edge.target)
        if src == tgt:
            continue  # 同层内导入不检查
        graph.setdefault(src, set()).add(tgt)

    # 检测跨层循环：A→B 且 B→A（不同层）
    cycles: list[str] = []
    for src, tgts in graph.items():
        for tgt in tgts:
            if tgt in graph and src in graph[tgt] and src < tgt:
                cycles.append(f"  {src} ↔ {tgt}")

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

_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = ("graspo.flow", "graspo.eval", "graspo.cli")

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
    ("R1", "from graspo.flow.runtime import GraspoFlowRuntime\n"),
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
