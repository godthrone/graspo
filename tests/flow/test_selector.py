"""后端选择器（graspo.flow.backend_selection）的单元测试。

使用 importlib.util 直接加载模块文件，绕过 flow/__init__.py 的 torch 导入链。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from graspo.core.schema import GraspoConfig

# 直接加载 backend_selection.py 模块文件，避免触发 flow/__init__.py→trainer→torch 导入链
_selector_path = (
    Path(__file__).resolve().parents[2] / "src" / "graspo" / "flow" / "backend_selection.py"
)
_spec = importlib.util.spec_from_file_location(
    "graspo.flow.backend_selection", _selector_path,
    submodule_search_locations=[],
)
_selector = importlib.util.module_from_spec(_spec)
sys.modules["graspo.flow.backend_selection"] = _selector
_spec.loader.exec_module(_selector)

select_backend = _selector.select_backend
SUPPORTED_BACKENDS = _selector.SUPPORTED_BACKENDS


def test_supported_backends_contains_graspoflow():
    """SUPPORTED_BACKENDS 包含 "graspoflow"。"""
    assert "graspoflow" in SUPPORTED_BACKENDS


def test_supported_backends_is_set():
    """SUPPORTED_BACKENDS 是 set 类型。"""
    assert isinstance(SUPPORTED_BACKENDS, set)


def test_backend_selection_defaults_to_graspoflow():
    selection = select_backend(GraspoConfig())

    assert selection.name == "graspoflow"


@pytest.mark.parametrize("backend", ["auto", "hf-reference", "megatron-vllm", "native-tp"])
def test_backend_rejects_removed_names(backend):
    config = GraspoConfig()

    with pytest.raises(ValueError, match="Unsupported backend"):
        select_backend(config, requested=backend)