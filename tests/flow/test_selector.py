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
def _same_source_file(module, target) -> bool:
    """★ S2 加固：判定"已登记的同名模块"与本次目标是否**同一来源文件**。

    先 ``Path.samefile``（相对/绝对、符号链接等价判断）；任一侧不存在时退化为
    ``realpath`` 比较。判定不确定 ⇒ 返回 False（fail-closed），由调用方报错。
    """
    import os

    cached_file = getattr(module, "__file__", None)
    if not cached_file:
        return False
    try:
        return Path(cached_file).samefile(target)
    except OSError:
        return os.path.realpath(str(cached_file)) == os.path.realpath(str(target))


_selector = sys.modules.get(_SELECTOR_MODULE_NAME)
if _selector is not None and not _same_source_file(_selector, _selector_path):
    # ★ fail-closed（S2 加固）：复用**只按键名**不够——若同键上是"另一份来源"的同名
    #   模块，复用会让断言静默打在那一份上（方向＝可能假绿）。要求两者同一文件。
    raise RuntimeError(
        f"同键 {_SELECTOR_MODULE_NAME!r} 已被**另一份来源**登记："
        f"缓存件={getattr(_selector, '__file__', None)!r} ≠ 本次目标={str(_selector_path)!r}"
        " ⇒ 复用会让断言打在另一份同名模块上（fail-closed，⛔ 不静默）"
    )
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
