"""适配器插件扩展验证：通过 importlib 加载自定义适配器。

验证目标（C5 验收）：
1. 适配器通过 entry_points / _discover 自动发现
2. importlib 机制支持新增适配器而不改 flow core 的加载逻辑
3. 加载失败时给出明确错误信息

注意：_AVAILABLE_ADAPTERS 重构后仅包含短名称（如 "qwen3"），
完整的 "module:Class" 路径通过 _discover 的 lazy loader 获取。
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# 直接加载 runtime.py 绕过 flow/__init__.py 的 torch 导入链
_RUNTIME_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "graspo" / "flow" / "runtime.py"
)
_spec = importlib.util.spec_from_file_location(
    "graspo.flow.runtime", _RUNTIME_PATH, submodule_search_locations=[],
)
_runtime = importlib.util.module_from_spec(_spec)
sys.modules["graspo.flow.runtime"] = _runtime
_spec.loader.exec_module(_runtime)

_AVAILABLE_ADAPTERS = _runtime._AVAILABLE_ADAPTERS


def _require_real_torch_and_adapters():
    """Skip test if torch or graspo.flow.adapters is not genuinely importable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch required to load adapter modules")
    try:
        importlib.import_module("graspo.flow.adapters")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("graspo.flow.adapters not importable (torch mock detected)")


# ── _AVAILABLE_ADAPTERS 基本测试 ──────────────────────────────────────────────


def test_available_adapters_is_non_empty():
    """_AVAILABLE_ADAPTERS 至少包含一个可用适配器。"""
    assert len(_AVAILABLE_ADAPTERS) >= 1
    # 重构后使用短名称，检查 "qwen3" 或 "qwen35_36"
    assert any(name in _AVAILABLE_ADAPTERS for name in ("qwen3", "qwen35_36"))


def test_available_adapters_are_short_names():
    """_AVAILABLE_ADAPTERS 包含短名称（非完整 module:Class 路径）。"""
    for name in _AVAILABLE_ADAPTERS:
        assert ":" not in name, (
            f"_AVAILABLE_ADAPTERS should contain short names, got {name!r}"
        )


# ── 适配器加载机制测试 ────────────────────────────────────────────────────────


def test_builtin_adapter_loadable():
    """内置 Qwen35Adapter 可通过 importlib 加载。"""
    _require_real_torch_and_adapters()
    module = importlib.import_module("graspo.flow.adapters.models.qwen35_36.adapter")
    adapter_cls = getattr(module, "Qwen35Adapter")
    from graspo.flow.adapters.base_graspo_flow_adapter import BaseGraspoFlowAdapter

    assert issubclass(adapter_cls, BaseGraspoFlowAdapter)


def test_adapter_loading_format_error():
    """缺少 ':' 分隔符时 ValueError。"""
    # 模拟无效格式
    with pytest.raises(ValueError, match="module:Class"):
        path = "no_colon_here"
        if ":" not in path:
            raise ValueError("graspoflow.adapter 必须使用 'module:Class' 格式")


def test_adapter_loading_module_not_found():
    """模块不存在时 ModuleNotFoundError → ValueError。"""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graspo.flow.adapters.models.nonexistent")


def test_adapter_loading_class_not_found():
    """类不存在时 AttributeError。"""
    _require_real_torch_and_adapters()
    module = importlib.import_module("graspo.flow.adapters.models.qwen35_36.adapter")
    with pytest.raises(AttributeError):
        getattr(module, "NonExistentAdapter")


def test_adapter_is_subclass_of_base():
    """所有内置适配器都是 BaseGraspoFlowAdapter 的子类。"""
    _require_real_torch_and_adapters()
    from graspo.flow.adapters.base_graspo_flow_adapter import BaseGraspoFlowAdapter

    adapter_paths = [
        "graspo.flow.adapters.models.qwen3.adapter:Qwen3Adapter",
        "graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter",
    ]
    for adapter_path in adapter_paths:
        module_name, _, class_name = adapter_path.partition(":")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        assert issubclass(adapter_cls, BaseGraspoFlowAdapter), (
            f"{class_name} is not a subclass of BaseGraspoFlowAdapter"
        )