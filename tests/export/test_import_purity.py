"""T31：``export/`` 的**计算与设施分离**（C1/C2/C3，判决可机检）。

- **C1**：``export/**`` 模块级不得 import ``torch``/``peft``/``transformers``/``vllm``；
- **C2**：**真叶子**纯逻辑模块（``semantics.py``、自含的 ``schema.py``）**按文件路径加载
  后能真正执行**（照抄 ``tests/flow/test_full_param_support.py`` 的 ``_load_pure_module``
  手法：不执行父包 ``__init__``、无全局副作用）；★ ``classify.py`` **不在** C2 名单内
  （它必须消费 registry ⇒ 有 graspo 内部 import ⇒ 按文件路径加载必然失败，见设计 §2.3 W3）；
- **C3**：重型依赖**确实**出现在 ``_load``/``_write`` 函数体内（不是静默跳过）。

每条判据都带**阳性对照**（能构造出让该检查真失败的输入）。
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_EXPORT_ROOT = _SRC_ROOT / "graspo" / "export"
_HEAVY_MODULES = ("torch", "torchvision", "peft", "transformers", "vllm")

#: C2 的被测名单 = **真叶子**（模块级零 graspo 内部 import）。
#: ⛔ 不含 classify.py（设计 §2.3 W3 的订正）。
_C2_LEAF_MODULES = ("semantics.py", "schema.py")


def _export_files() -> list[Path]:
    return sorted(_EXPORT_ROOT.rglob("*.py"))


def _module_level_imports(source: str) -> list[tuple[str, int]]:
    """收集**模块级**（非函数体内）导入的目标名（函数体内的延迟导入不收录）。"""
    tree = ast.parse(source)
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.Import):
                found.extend((alias.name, child.lineno) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                found.append((child.module, child.lineno))
            visit(child)

    visit(tree)
    return found


def _module_level_heavy(source: str) -> list[tuple[str, int]]:
    return [
        (module, lineno)
        for module, lineno in _module_level_imports(source)
        if module.split(".")[0] in _HEAVY_MODULES
    ]


def _module_level_graspo(source: str) -> list[tuple[str, int]]:
    return [
        (module, lineno)
        for module, lineno in _module_level_imports(source)
        if module == "graspo" or module.startswith("graspo.")
    ]


def _function_imports(source: str, func_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            names: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    names.update(alias.name.split(".")[0] for alias in child.names)
                elif isinstance(child, ast.ImportFrom):
                    names.add((child.module or "").split(".")[0])
            return names
    raise AssertionError(f"function {func_name!r} not found")


def _function_imports_matching(source: str, prefix: str) -> set[str]:
    """聚合所有以 ``prefix`` 开头的函数体内的导入（``_write`` 会覆盖 ``_write_*``）。"""
    tree = ast.parse(source)
    names: set[str] = set()
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix):
            found = True
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    names.update(alias.name.split(".")[0] for alias in child.names)
                elif isinstance(child, ast.ImportFrom):
                    names.add((child.module or "").split(".")[0])
    assert found, f"no function starting with {prefix!r} found"
    return names


def _load_pure_module(dotted_name: str, relative_path: str):
    """按文件路径加载一个**叶子**模块（不执行父包 ``__init__``）。

    为了执行期间 pydantic 能按 ``__module__`` 解析前向引用（``schema.py`` 的字段注解是
    字符串），临时把模块放进 ``sys.modules``，执行完**恢复原状** ⇒ 无持续全局副作用。
    """
    path = _SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(dotted_name)
    sys.modules[dotted_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(dotted_name, None)
        else:
            sys.modules[dotted_name] = previous
    return module


# ── C1 ────────────────────────────────────────────────────────────────────────


def test_c1_no_module_level_heavy_imports():
    violations: list[str] = []
    for path in _export_files():
        for module, lineno in _module_level_heavy(path.read_text(encoding="utf-8")):
            violations.append(f"{path.relative_to(_SRC_ROOT)}:{lineno} imports {module}")
    assert not violations, (
        "export modules must not import heavy facilities at module level:\n  "
        + "\n  ".join(violations)
    )


def test_c1_positive_control():
    """阳性对照：模块级 ``import torch`` 必须被 C1 扫描器抓住。"""
    assert _module_level_heavy("import torch\n") == [("torch", 1)]
    # 反向对照：函数体内的延迟 import 不算 C1 违规。
    assert _module_level_heavy("def f():\n    import torch\n") == []


# ── C2 ────────────────────────────────────────────────────────────────────────


def test_c2_leaf_modules_are_really_required_to_be_leaf():
    """前置：C2 名单里的模块必须模块级零 graspo 内部 import（否则按路径加载必然失败）。"""
    for name in _C2_LEAF_MODULES:
        source = (_EXPORT_ROOT / name).read_text(encoding="utf-8")
        assert not _module_level_graspo(source), f"{name} is not a true leaf module"


def test_c2_semantics_module_executes_from_file_path():
    module = _load_pure_module("graspo.export.semantics", "graspo/export/semantics.py")
    assert (
        module.is_full_param_payload({"tuner_type": "lora", "full_param_state_dict": None})
        is False
    )
    assert module.is_full_param_payload({"tuner_type": "full"}) is True
    try:
        module.require_lora_semantics([{"tuner_type": "full"}], operation="merged-hf")
    except ValueError as exc:
        assert "LoRA-only" in str(exc)
    else:  # pragma: no cover - 必须抛
        raise AssertionError("full-param payload must be rejected by require_lora_semantics")


def test_c2_schema_module_executes_from_file_path():
    module = _load_pure_module("graspo.export.schema", "graspo/export/schema.py")
    assert module.CheckpointForm.NATIVE_LORA_RANK.value == "native-lora-rank"
    plan = module.ShardPlan(
        form=module.CheckpointForm.NATIVE_LORA_RANK,
        target=module.ExportTarget.PEFT_ADAPTER,
        checkpoint_name="final",
        source_dir="/tmp/final",
    )
    assert plan.topology is None and plan.replica_check is None


def test_c2_positive_control_non_leaf_module_is_detected(tmp_path):
    """阳性对照：模块级 import 父包 ⇒ 叶子判据必须命中（C2 只对真叶子成立）。

    ⚠ 不假设"按路径加载一定抛 ImportError"：本机装了 torch，父包 import 会成功
    （设计 §2.3 已实测：真正不可满足的是"不装 torch 也能 import graspo"）。C2 的
    牙齿在**叶子前置条件**这条判据上 —— 非叶子不得进入 C2 名单。
    """
    planted = "from graspo.flow.backend import native\nVALUE = 1\n"
    assert _module_level_graspo(planted) == [("graspo.flow.backend", 1)]
    bad = tmp_path / "bad_leaf.py"
    bad.write_text(planted, encoding="utf-8")
    assert _module_level_graspo(bad.read_text(encoding="utf-8"))


# ── C3 ────────────────────────────────────────────────────────────────────────


def test_c3_heavy_imports_are_deferred_but_present():
    source = (_EXPORT_ROOT / "exporters" / "native_lora.py").read_text(encoding="utf-8")
    load_imports = _function_imports(source, "_load")
    assert "torch" in load_imports, "_load must import torch inside its body"
    write_imports = _function_imports_matching(source, "_write")
    assert "safetensors" in write_imports, "_write* must import safetensors inside their bodies"


def test_c3_positive_control_module_level_heavy_is_not_deferred():
    """阳性对照：把 torch 提到模块级 ⇒ `_load` 函数体内不再有 torch ⇒ C3 失败。"""
    planted = "import torch\n\ndef _load(self, plan):\n    return plan\n"
    assert "torch" not in _function_imports(planted, "_load")
    assert _module_level_heavy(planted) == [("torch", 1)]
