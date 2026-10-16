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

import pytest

_RUNTIME_MODULE_NAME = "graspo.flow.backend.native.runtime"
# 直接加载 runtime.py 绕过 flow/__init__.py 的 torch 导入链
_RUNTIME_PATH = Path(__file__).resolve().parents[4] / "src" / "graspo" / "flow" / "backend" / "native" / "runtime.py"


def _same_source_file(module, target: Path) -> bool:
    """★ S2 加固：判定"已登记的同名模块"与本次目标是否**同一来源文件**。

    先 ``Path.samefile``（对相对/绝对、符号链接等价判断，需要两侧都存在）；任一侧
    不存在时退化为 ``realpath`` 字符串比较。⛔ 不静默：判定不确定时**返回 False**
    （fail-closed），由调用方报错。
    """
    import os

    cached_file = getattr(module, "__file__", None)
    if not cached_file:
        return False
    try:
        return Path(cached_file).samefile(target)
    except OSError:
        return os.path.realpath(str(cached_file)) == os.path.realpath(str(target))


def _load_runtime_module():
    """按文件路径加载 runtime.py，**同名副本一律复用** ``sys.modules`` 里那一份。

    ★ 选型理由（同名副本复用 ＋ 失败即摘除）：``spec_from_file_location`` 每次调用都造
      **新副本**；若不以同名复用，则后加载者会把本键**顶掉**——``tests/flow/
      test_pp_rollout_logits_budget.py`` 与 ``tests/cli/
      test_evaluate_checkpoint_backend_failclosed.py`` 的 docstring 记录的正是这个坑：
      本文件导入期拿到旧副本，别的测试对该键的 ``monkeypatch`` 打在活对象上、断言读的却
      是旧副本 ⇒ 断言退化成空转（``isinstance`` 假红／哨兵装到错对象）。命中即复用 ⇒
      全进程只有一份 runtime 模块（实现与测试同源，§1.4）。登记失败即摘除，⛔ 不留半截模块。
    """
    cached = sys.modules.get(_RUNTIME_MODULE_NAME)
    if cached is not None:
        # ★ fail-closed 来源路径校验（S2 加固）：复用**只按键名**是不够的——若同键上
        #   登记的是"另一份**来源**"的同名模块（别处用别的路径加载过／路径表达式不同），
        #   复用会把断言静默钉在那一份上（方向＝可能假绿）。故要求缓存件与本次目标
        #   是**同一个文件**；不同即报错（⛔ 不静默复用）。`samefile` 对相对/绝对、
        #   符号链接均等价判断；`realpath` 兜底"缓存件 `__file__` 已不存在"的情形。
        if not _same_source_file(cached, _RUNTIME_PATH):
            raise RuntimeError(
                f"同键 {_RUNTIME_MODULE_NAME!r} 已被**另一份来源**登记："
                f"缓存件={getattr(cached, '__file__', None)!r} ≠ 本次目标={str(_RUNTIME_PATH)!r}"
                " ⇒ 复用会让断言打在另一份同名模块上（fail-closed，⛔ 不静默）"
            )
        return cached
    _spec = importlib.util.spec_from_file_location(
        _RUNTIME_MODULE_NAME,
        _RUNTIME_PATH,
        submodule_search_locations=[],
    )
    assert _spec is not None and _spec.loader is not None, f"加载失败：{_RUNTIME_PATH}"
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_RUNTIME_MODULE_NAME] = _module
    try:
        _spec.loader.exec_module(_module)
    except BaseException:
        sys.modules.pop(_RUNTIME_MODULE_NAME, None)
        raise
    return _module


_runtime = _load_runtime_module()
assert sys.modules[_RUNTIME_MODULE_NAME] is _runtime, "同名副本未被复用 ⇒ 断言会打在第二份副本上"


def _available_adapters():
    """★ 按**调用时**从活模块解析 ``_AVAILABLE_ADAPTERS``（⛔ 不在导入期捕获快照）。

    选型理由：``_AVAILABLE_ADAPTERS`` 是不可变 tuple，导入期捕获即"快照"——即使复用到位，
    快照也会在模块被重新登记后与活对象分叉，对该键的 patch 无法被断言看见。按调用时解析
    ⇒ 「能失败的断言」成立（理由同 ``_load_runtime_module``）。
    """
    return sys.modules[_RUNTIME_MODULE_NAME]._AVAILABLE_ADAPTERS


def _require_real_torch_and_adapters():
    """Skip test if torch or graspo.flow.adapter is not genuinely importable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch required to load adapter modules")
    try:
        importlib.import_module("graspo.flow.adapter")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("graspo.flow.adapter not importable (torch mock detected)")


# ── _AVAILABLE_ADAPTERS 基本测试 ──────────────────────────────────────────────


def test_available_adapters_is_non_empty():
    """_AVAILABLE_ADAPTERS 至少包含一个可用适配器。"""
    adapters = _available_adapters()
    assert len(adapters) >= 1
    # 重构后使用短名称，检查 "qwen3" 或 "qwen35_36"
    assert any(name in adapters for name in ("qwen3", "qwen35_36"))


def test_available_adapters_are_short_names():
    """_AVAILABLE_ADAPTERS 包含短名称（非完整 module:Class 路径）。"""
    for name in _available_adapters():
        assert ":" not in name, f"_AVAILABLE_ADAPTERS should contain short names, got {name!r}"


# ── 适配器加载机制测试 ────────────────────────────────────────────────────────


def test_builtin_adapter_loadable():
    """内置 Qwen35ModelAdapter 可通过 importlib 加载。"""
    _require_real_torch_and_adapters()
    module = importlib.import_module("graspo.flow.backend.native.models.qwen35_36.model_adapter")
    adapter_cls = getattr(module, "Qwen35ModelAdapter")
    from graspo.flow.adapter.base_graspo_flow_adapter import BaseGraspoFlowAdapter

    assert issubclass(adapter_cls, BaseGraspoFlowAdapter)


def test_adapter_loading_format_error():
    """缺少 ':' 分隔符时 ValueError。"""
    # 模拟无效格式
    with pytest.raises(ValueError, match="module:Class"):
        path = "no_colon_here"
        if ":" not in path:
            raise ValueError("native.adapter 必须使用 'module:Class' 格式")


def test_adapter_loading_module_not_found():
    """模块不存在时 ModuleNotFoundError → ValueError。"""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graspo.flow.backend.native.models.nonexistent")


def test_adapter_loading_class_not_found():
    """类不存在时 AttributeError。"""
    _require_real_torch_and_adapters()
    module = importlib.import_module("graspo.flow.backend.native.models.qwen35_36.model_adapter")
    with pytest.raises(AttributeError):
        getattr(module, "NonExistentAdapter")


def test_adapter_is_subclass_of_base():
    """所有内置适配器都是 BaseGraspoFlowAdapter 的子类。"""
    _require_real_torch_and_adapters()
    from graspo.flow.adapter.base_graspo_flow_adapter import BaseGraspoFlowAdapter

    adapter_paths = [
        "graspo.flow.backend.native.models.qwen3.model_adapter:Qwen3ModelAdapter",
        "graspo.flow.backend.native.models.qwen35_36.model_adapter:Qwen35ModelAdapter",
    ]
    for adapter_path in adapter_paths:
        module_name, _, class_name = adapter_path.partition(":")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        assert issubclass(adapter_cls, BaseGraspoFlowAdapter), (
            f"{class_name} is not a subclass of BaseGraspoFlowAdapter"
        )
