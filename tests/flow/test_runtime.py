"""Tests for ``graspo.flow.backend.native.runtime._AVAILABLE_ADAPTERS``。

历史背景（★ 本文件原为 ``sys.modules`` mock 注入式加载，见下"为什么改"）：
``spec_from_file_location`` 直接把 ``runtime.py`` 当独立模块执行会**绕过**
``graspo.flow.__init__`` 的导入链。该技巧在"torch 缺失"的年代是必需的；
现在环境里 ``torch`` 与全部 ``graspo.*`` 依赖都可导入，故**不再 stub**。

★ 为什么改（能失败的断言 ＋ 不污染同会话其它测试）：

1. 原先 ``_load_runtime_module()`` 会往 ``sys.modules`` 注入**假** ``torch``／
   ``graspo``／``graspo.core``／``graspo.flow``／``graspo.ripple``／
   ``graspo.core.schema``／``graspo.core.discovery`` 等。这些假件**键名是对的、
   内容是假的**：``graspo.flow`` 的 ``__path__ = []`` ⇒ 同会话里任何
   ``import graspo.flow.backend...`` 都变成 ``ModuleNotFoundError``；
   假 ``graspo.core.schema`` 缺 ``PP_ROLLOUT_PREFILL_LAST_ONLY`` ⇒ 别的测试
   从此 import 失败（实测：与 ``test_pp_rollout_logits_budget.py`` 同会话，
   ``test_runtime.py`` 在前 ⇒ **7 failed 的既有顺序交互**）。
2. 该注入还使本文件**自身**的收集结果依赖收集次序：若更早的测试已导入真
   ``graspo.core.schema``，假件不会安装 ⇒ 正常；否则假件安装 ⇒
   ``ImportError: cannot import name 'PP_ROLLOUT_PREFILL_LAST_ONLY'``
   ⇒ 单跑本文件**收集即失败**（路径依赖的脆弱行为，不是"防呆"）。
3. 原先的 ``_SYS_MODULES_SNAPSHOT`` 在**注入之后**才拍，``_cleanup_mocked_modules()``
   定义了却**从未被调用**（既使被调用也只会把注入件"恢复"回来）。

⇒ 处置：**只读、不注入**。本文件不再向 ``sys.modules`` 写任何东西，改从**真实包**
加载 ``graspo.flow.backend.native.runtime``（§1.4 单一真相源：测试与实现同一份模块），
并像 ``tests/flow/backend/native/test_adapter_plugin.py`` 那样**按调用时**解析活对象，
使断言在"活对象被改"时**真的会红**（而非读导入期快照而空转）。

本文件断言的是 ``_AVAILABLE_ADAPTERS`` 的行为：它是 ``tuple``、含 ``qwen3`` 与
``qwen35_36``，且 ``setup()`` 的错误信息引用它。
"""

from __future__ import annotations

import importlib
import inspect
import sys

#: 实现侧解析该模块用的 canonical 键（§1.4：测试与实现同源）。
_RUNTIME_MODULE_NAME = "graspo.flow.backend.native.runtime"


def _load_runtime_module():
    """从**真实包**加载 ``graspo.flow.backend.native.runtime``（⛔ 不注入任何 mock）。

    ★ 选型理由：本键是**实现侧**（``importlib.import_module`` /
    ``from graspo.flow.backend.native.runtime import ...``）解析的那个键，
    所以测试必须读**同一份**模块对象。若另造一份（``spec_from_file_location``
    的独立副本、或往 ``sys.modules`` 塞假依赖），则"测试看到的模块"与
    "实现 import 到的模块"分叉 ⇒ 对该键的 ``monkeypatch`` 打空、断言空转。
    """
    return importlib.import_module(_RUNTIME_MODULE_NAME)


_runtime = _load_runtime_module()


def _available_adapters():
    """★ 按**调用时**从活模块解析 ``_AVAILABLE_ADAPTERS``（⛔ 不在导入期捕获快照）。

    选型理由：``_AVAILABLE_ADAPTERS`` 是不可变 tuple，导入期捕获即"快照"——对该键的
    patch 无法被断言看见，断言会退化成空转。按调用时解析 ⇒ 「能失败的断言」成立
    （同 ``tests/flow/backend/native/test_adapter_plugin.py::_available_adapters``）。
    """
    return sys.modules[_RUNTIME_MODULE_NAME]._AVAILABLE_ADAPTERS


assert sys.modules[_RUNTIME_MODULE_NAME] is _runtime, "同名副本未被复用 ⇒ 断言会打在第二份副本上"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAvailableAdapters:
    """Tests for ``_AVAILABLE_ADAPTERS`` in ``graspo.flow.backend.native.runtime``."""

    def test_contains_qwen3(self) -> None:
        """``_AVAILABLE_ADAPTERS`` must include the 'qwen3' adapter."""
        adapters = _available_adapters()
        assert "qwen3" in adapters, f"Expected 'qwen3' in _AVAILABLE_ADAPTERS, got {adapters}"

    def test_contains_qwen35_36(self) -> None:
        """``_AVAILABLE_ADAPTERS`` must include the 'qwen35_36' adapter."""
        adapters = _available_adapters()
        assert "qwen35_36" in adapters, (
            f"Expected 'qwen35_36' in _AVAILABLE_ADAPTERS, got {adapters}"
        )

    def test_is_tuple(self) -> None:
        """``_AVAILABLE_ADAPTERS`` must be a ``tuple`` instance."""
        adapters = _available_adapters()
        assert isinstance(adapters, tuple), f"Expected tuple, got {type(adapters)}"

    def test_error_messages_reference_available_adapters(self) -> None:
        """The ``setup()`` error messages must reference ``_AVAILABLE_ADAPTERS``.

        We verify by inspecting the source code of the ``setup`` method:
        both error branches (missing ':' separator and failed import) must
        use ``_AVAILABLE_ADAPTERS`` in their f-string messages.
        """
        setup_source = inspect.getsource(_runtime.GraspoFlowRuntime.setup)

        # The error messages join _AVAILABLE_ADAPTERS with ', '.
        assert "_AVAILABLE_ADAPTERS" in setup_source, (
            "setup() must reference _AVAILABLE_ADAPTERS in its error messages"
        )

        # The joined string should contain the adapter names (read at call time).
        adapters_csv = ", ".join(_available_adapters())
        assert "qwen3" in adapters_csv
        assert "qwen35_36" in adapters_csv
