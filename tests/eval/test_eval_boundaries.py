"""评测链路的模块边界扫描（AST，不导入被测模块）。

**为什么要它**：评测链路的价值在于"结论可被独立重算"。一旦纯计算模块
（口径判定、聚合、Δ 配对）偷偷依赖 torch / 网络 / flow 训练栈，这个性质就没了——
单测跑不起来、结论无法复现、CI 需要 GPU。

**契约**（宪法 §1.3 层次边界 + §1.1 一事一责）：

1. **纯逻辑模块零重设施**：``criteria`` / ``evaluate`` / ``merged_export`` /
   ``dataset`` 不得在**模块级**导入 ``torch`` / ``peft`` / ``transformers`` / ``vllm``。
   （函数体内的延迟导入是**允许且必需的**——见 ``merged_export.merge_peft_checkpoint``。）
2. **评测域不反向依赖训练栈**：``graspo.eval.*`` 不得导入 ``graspo.flow.*`` /
   ``graspo.ripple.*``。
3. **无环**：``graspo.eval`` 内部不得出现模块级循环导入。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "graspo"
_EVAL_ROOT = _SRC_ROOT / "eval"

#: 纯计算模块：模块级不得出现的重设施（它们只能出现在函数体内）。
_PURE_MODULES = ("criteria.py", "evaluate.py", "merged_export.py", "dataset.py")
_HEAVY_MODULES = ("torch", "torchvision", "peft", "transformers", "vllm")

#: 评测域不得触及的训练栈前缀。
_FORBIDDEN_PREFIXES = ("graspo.flow", "graspo.ripple")


def _eval_files() -> list[Path]:
    files = sorted(_EVAL_ROOT.glob("*.py"))
    assert files, f"no eval modules found under {_EVAL_ROOT}"
    return files


def _module_level_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """收集**模块级**（非函数体内）导入的目标名。

    函数体内的 ``import`` 不会被收录——那是延迟导入，是"模块可被无 torch 环境
    解析"的实现手段，不是违规。
    """
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 函数体内不扫描；但函数装饰器/默认值在模块级求值，保守起见跳过。
                continue
            if isinstance(child, ast.Import):
                found.extend((alias.name, child.lineno) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                found.append((child.module, child.lineno))
            visit(child)

    visit(tree)
    return found


def _all_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """收集全部层级的导入（含函数体内），用于依赖方向检查。"""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno))
    return found


@pytest.mark.parametrize("filename", _PURE_MODULES)
def test_pure_modules_have_no_module_level_heavy_dependency(filename):
    """口径/聚合/识别/数据四个模块必须能在无 torch 环境被 import。"""
    path = _EVAL_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = [
        f"{filename}:{lineno} imports {module}"
        for module, lineno in _module_level_imports(tree)
        if module.split(".")[0] in _HEAVY_MODULES
    ]
    assert not violations, (
        "pure eval modules must not import heavy facilities at module level:\n  "
        + "\n  ".join(violations)
    )


def test_heavy_imports_are_deferred_inside_functions():
    """重设施不是"不许用"，而是"必须延迟到函数体内"——否则会破坏可测性。"""
    path = _EVAL_ROOT / "merged_export.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level = {module.split(".")[0] for module, _ in _module_level_imports(tree)}
    assert not (module_level & set(_HEAVY_MODULES))

    # 合并函数里必须真的导入它们（不能静默跳过合并）。
    merge_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "merge_peft_checkpoint"
    )
    deferred = {
        alias.name.split(".")[0]
        for node in ast.walk(merge_fn)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(merge_fn)
        if isinstance(node, ast.ImportFrom)
    }
    assert "torch" in deferred
    assert "peft" in deferred
    assert "transformers" in deferred


def test_eval_package_does_not_import_training_stack():
    """评测域不反向依赖 flow / ripple：结论必须能脱离训练栈复核。"""
    violations: list[str] = []
    for path in _eval_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, lineno in _all_imports(tree):
            is_forbidden = any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in _FORBIDDEN_PREFIXES
            )
            if is_forbidden:
                violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "eval must not import the training stack:\n  " + "\n  ".join(violations)


def test_eval_package_has_no_module_level_cycles():
    """``graspo.eval`` 内部模块级导入图必须无环。"""
    graph: dict[str, set[str]] = {}
    for path in _eval_files():
        name = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets = {
            module.split(".")[2] if module.startswith("graspo.eval.") else ""
            for module, _ in _module_level_imports(tree)
            if module.startswith("graspo.eval.")
        }
        graph[name] = {target for target in targets if target and target != name}

    cycles: list[str] = []
    for source, targets in graph.items():
        for target in targets:
            if source in graph.get(target, set()) and source < target:
                cycles.append(f"  {source} ↔ {target}")
    assert not cycles, "module-level import cycles inside graspo.eval:\n" + "\n".join(cycles)


def test_test_shim_does_not_break_submodule_discovery():
    """回归：测试垫片**不得**让 ``graspo`` 下的子模块对后继测试不可见。

    锁定一次真实缺陷：垫片曾把 ``graspo.ripple`` 装成 ``__path__ = []`` 的假包，
    使**别的测试文件**（``test_schema_msswift.py``）收集期报
    ``No module named 'graspo.ripple.monitoring'`` / ``graspo.ripple.buffer``
    ——错误信息与真因完全无关。用 ``find_spec`` 断言这些子模块仍可解析。
    """
    import importlib.util
    import sys

    for submodule in ("monitoring", "buffer", "parsing"):
        dotted = f"graspo.ripple.{submodule}"
        assert importlib.util.find_spec(dotted) is not None, (
            f"{dotted} is not discoverable — a test shim has broken the package path"
        )

    suspicious = [
        name
        for name, module in sys.modules.items()
        if (name == "graspo" or name.startswith("graspo."))
        and getattr(module, "__path__", None) == []
    ]
    assert not suspicious, f"graspo packages with an empty __path__: {suspicious}"


def test_every_eval_module_declares_its_responsibility():
    """每个模块必须有文件头职责声明（宪法 §12.4）——包括 ``__init__`` 外的全部文件。"""
    missing: list[str] = []
    for path in _eval_files():
        if path.name == "__init__.py":
            continue  # §12.4 豁免：__init__ 的职责就是 re-export
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        if not docstring or len(docstring.strip()) < 20:
            missing.append(path.name)
    assert not missing, f"modules missing a file-header responsibility statement: {missing}"
