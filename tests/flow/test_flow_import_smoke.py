"""导入期冒烟：``src/graspo/flow/**`` 全部模块都能被真正 import。

**这道测试守的是什么（防呆，宪法 §2）**

2026-09-20 的缺陷 ``4ee8539``：``backend_selection.py`` 的 ``create_native_trainer`` 返回标注写成
``-> GraspoFlowTrainer``（未加引号），而该名字只在 ``if TYPE_CHECKING:`` 下导入，文件又没有
``from __future__ import annotations``。模块级函数的注解在**定义时**求值 ⇒ 导入该模块即
``NameError: name 'GraspoFlowTrainer' is not defined``。

它之所以能"静默通过"：既有测试为了避开 ``flow/__init__`` → ``trainer`` → ``torch`` 的导入链，
用 ``importlib.util.spec_from_file_location`` **按文件路径**加载该模块，从来没有按**真实点分模块名**
导入过它，因此导入期错误对它不可见；ruff 的 F821 也抓不到（TYPE_CHECKING 下的名字被 pyflakes
视为"已定义"——F821 判的是"名字有没有定义"，不是"运行期会不会被求值"）。

本测试按真实点分模块名遍历导入 ``graspo.flow.**`` 的每个子模块，**导入期错误一律判红**。
它不检查任何行为契约，只回答一个问题：**这个包还能不能被导入。**

范围与代价：只覆盖 ``graspo.flow.**``（工作包指定的范围），实测 70 个子模块、约 2 秒。
不断言可选重依赖（torch / ms-swift）是否可用——本仓 ``dev`` extra 已声明它们，
缺依赖会在导入期直接报错并暴露出来，这正是本测试要判红的情形之一。
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import graspo.flow


def _flow_submodules() -> list[str]:
    """``graspo.flow`` 下全部子模块的点分名（含命名空间包下的模块）。"""
    return sorted(
        module.name for module in pkgutil.walk_packages(graspo.flow.__path__, prefix="graspo.flow.")
    )


FLOW_SUBMODULES = _flow_submodules()


def test_flow_has_submodules() -> None:
    """防呆的前提：遍历确实拿到了模块集合（避免"空集合 ⇒ 全绿"的假通过）。"""
    assert FLOW_SUBMODULES, "未发现任何 graspo.flow 子模块——测试本身失效了"
    assert "graspo.flow.backend_selection" in FLOW_SUBMODULES


@pytest.mark.parametrize("module_name", FLOW_SUBMODULES)
def test_flow_module_imports(module_name: str) -> None:
    """每个 ``graspo.flow.**`` 子模块都必须能被导入（捕获导入期 NameError 等）。"""
    importlib.import_module(module_name)
