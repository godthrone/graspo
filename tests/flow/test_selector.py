"""后端选择器（graspo.flow.backend.selection）的单元测试。

使用 importlib.util 直接加载模块文件，绕过 flow/__init__.py 的 torch 导入链。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from graspo.core.schema import GraspoConfig

# 直接加载 backend_selection.py 模块文件，避免触发 flow/__init__.py→trainer→torch 导入链
# ★ 选型理由（同名副本复用 ＋ 失败即摘除）：``tests/flow/test_backend_switch.py`` 与本文件
#   争同一个 canonical 键、加载同一份源码、双方原先都无 guard ⇒ 后收集者把前者顶掉，
#   先导入文件里模块级捕获的 ``select_backend`` / ``SUPPORTED_BACKENDS`` 与 sys.modules
#   现值分叉 ⇒ 对该键的任何 monkeypatch 打空、断言退化。命中即复用 ⇒ 两文件共用同一份。
_SELECTOR_MODULE_NAME = "graspo.flow.backend.selection"
_selector_path = (
    Path(__file__).resolve().parents[2] / "src" / "graspo" / "flow" / "backend" / "selection.py"
)
_selector = sys.modules.get(_SELECTOR_MODULE_NAME)
if _selector is None:
    _spec = importlib.util.spec_from_file_location(
        _SELECTOR_MODULE_NAME,
        _selector_path,
        submodule_search_locations=[],
    )
    assert _spec is not None and _spec.loader is not None, f"加载失败：{_selector_path}"
    _selector = importlib.util.module_from_spec(_spec)
    sys.modules[_SELECTOR_MODULE_NAME] = _selector
    try:
        _spec.loader.exec_module(_selector)
    except BaseException:
        sys.modules.pop(_SELECTOR_MODULE_NAME, None)
        raise
assert sys.modules[_SELECTOR_MODULE_NAME] is _selector, "同名副本未被复用 ⇒ 断言会打在第二份副本上"

select_backend = _selector.select_backend
SUPPORTED_BACKENDS = _selector.SUPPORTED_BACKENDS


def test_supported_backends_contains_native():
    """SUPPORTED_BACKENDS 包含 "native"。"""
    assert "native" in SUPPORTED_BACKENDS


def test_supported_backends_is_set():
    """SUPPORTED_BACKENDS 是 set 类型。"""
    assert isinstance(SUPPORTED_BACKENDS, set)


def test_backend_selection_defaults_to_native():
    selection = select_backend(GraspoConfig())

    assert selection.name == "native"


@pytest.mark.parametrize("backend", ["auto", "hf-reference", "megatron-vllm", "native-tp"])
def test_backend_rejects_removed_names(backend):
    config = GraspoConfig()

    with pytest.raises(ValueError, match="Unsupported backend"):
        select_backend(config, requested=backend)
