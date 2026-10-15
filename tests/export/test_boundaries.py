"""T30：``graspo.export`` **不得 import 训练栈**（``graspo.flow`` / ``graspo.ripple``）。

这是候选 (a)（顶层共享包）的核心收益：现状是"eval 不得进 flow"；本不变量更强 ——
eval 进的这一层**根本不含 flow**。全层级扫描（含函数体内延迟 import）。
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "graspo"
_EXPORT_ROOT = _SRC_ROOT / "export"
_FORBIDDEN_PREFIXES = ("graspo.flow", "graspo.ripple")


def _export_files() -> list[Path]:
    files = sorted(_EXPORT_ROOT.rglob("*.py"))
    assert files, f"no export modules found under {_EXPORT_ROOT}"
    return files


def _all_imports(source: str) -> list[tuple[str, int]]:
    tree = ast.parse(source)
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno))
    return found


def _forbidden_imports(source: str) -> list[tuple[str, int]]:
    return [
        (module, lineno)
        for module, lineno in _all_imports(source)
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in _FORBIDDEN_PREFIXES
        )
    ]


def test_export_does_not_import_training_stack():
    """AST 扫 ``src/graspo/export/**`` 的**全层级** import ⇒ flow/ripple 零命中。"""
    violations: list[str] = []
    for path in _export_files():
        for module, lineno in _forbidden_imports(path.read_text(encoding="utf-8")):
            violations.append(f"{path.relative_to(_SRC_ROOT)}:{lineno} imports {module}")
    assert not violations, (
        "graspo.export must not import the training stack:\n  " + "\n  ".join(violations)
    )


def test_scanner_catches_a_planted_violation():
    """阳性对照：扫描器**不是空转** —— 一句植入的 ``from graspo.flow...`` 必须被抓住。"""
    planted = "from graspo.flow.backend.native.runtime import X\n"
    assert _forbidden_imports(planted) == [("graspo.flow.backend.native.runtime", 1)]


def test_scanner_catches_deferred_import_inside_function():
    """阳性对照（全层级）：函数体内的延迟 import 同样命中（不是只扫模块级）。"""
    planted = "def f():\n    import graspo.ripple.reward.reward\n"
    assert _forbidden_imports(planted) == [("graspo.ripple.reward.reward", 2)]
