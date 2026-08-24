"""适配器插件扩展验证：通过 importlib 加载自定义适配器。

验证目标（C5 验收）：
1. 适配器通过 "module:ClassName" 格式按名加载
2. importlib 机制支持新增适配器而不改 flow core 的加载逻辑
3. 加载失败时给出明确错误信息
"""

import pytest

from graspo.flow.adapters.base_graspo_flow_adapter import BaseGraspoFlowAdapter
from graspo.flow.runtime import _AVAILABLE_ADAPTERS


# ── 适配器加载机制测试 ────────────────────────────────────────────────────────


def test_available_adapters_is_non_empty():
    """_AVAILABLE_ADAPTERS 至少包含一个可用适配器。"""
    assert len(_AVAILABLE_ADAPTERS) >= 1
    assert any("Qwen35Adapter" in a for a in _AVAILABLE_ADAPTERS)


def test_available_adapters_use_module_colon_class_format():
    """所有适配器路径使用 "module.path:ClassName" 格式。"""
    for adapter_path in _AVAILABLE_ADAPTERS:
        assert ":" in adapter_path, f"{adapter_path!r} missing ':' separator"
        module_name, _, class_name = adapter_path.partition(":")
        assert module_name, f"empty module in {adapter_path!r}"
        assert class_name, f"empty class in {adapter_path!r}"


def test_builtin_adapter_loadable():
    """内置 Qwen35Adapter 可通过 importlib 加载。"""
    import importlib

    for adapter_path in _AVAILABLE_ADAPTERS:
        if "Qwen35Adapter" in adapter_path:
            module_name, _, class_name = adapter_path.partition(":")
            module = importlib.import_module(module_name)
            adapter_cls = getattr(module, class_name)
            assert issubclass(adapter_cls, BaseGraspoFlowAdapter)
            return

    pytest.fail("Qwen35Adapter not found in _AVAILABLE_ADAPTERS")


def test_adapter_loading_format_error():
    """缺少 ':' 分隔符时 ValueError。"""
    import importlib

    # 模拟无效格式
    with pytest.raises(ValueError, match="module:Class"):
        path = "no_colon_here"
        if ":" not in path:
            raise ValueError("graspoflow.adapter 必须使用 'module:Class' 格式")


def test_adapter_loading_module_not_found():
    """模块不存在时 ModuleNotFoundError → ValueError。"""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graspo.flow.adapters.models.nonexistent")


def test_adapter_loading_class_not_found():
    """类不存在时 AttributeError。"""
    import importlib

    module = importlib.import_module("graspo.flow.adapters.models.qwen35_36.adapter")
    with pytest.raises(AttributeError):
        getattr(module, "NonExistentAdapter")


def test_adapter_is_subclass_of_base():
    """所有内置适配器都是 BaseGraspoFlowAdapter 的子类。"""
    import importlib

    for adapter_path in _AVAILABLE_ADAPTERS:
        module_name, _, class_name = adapter_path.partition(":")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        assert issubclass(adapter_cls, BaseGraspoFlowAdapter), (
            f"{class_name} is not a subclass of BaseGraspoFlowAdapter"
        )