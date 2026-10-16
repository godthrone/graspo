"""task-arch-b1 ② 裁定落实：``evaluate-checkpoint × msswift`` 必须 **fail-closed**（且不触 GPU）。

## 为什么有这条测试（把"可达路径"变成"有覆盖"）

机检（`evidence/30`）确认全仓 ``create_runtime`` **唯一调用点**是
``cli/tools.py`` 的 ``run_evaluate``，其上游是 CLI 子命令 ``evaluate-checkpoint``，
**全链路无任何 backend 门** ⇒ 用户配置 ``backend: msswift`` 时可达：

    graspo evaluate-checkpoint --config <msswift.yaml> --data <jsonl>
      → cli/app.py::cmd_evaluate_checkpoint
      → cli/tools.py::run_evaluate
      → flow/backend/selection.py::create_runtime   ← 唯一调用点
      → _RUNTIME_PROVIDERS 只有 native ⇒ ValueError（fail-closed）

## 为什么"响亮失败"是更正确的行为（而不是功能回退）

**改前**：``cli/tools.py`` **无条件** ``GraspoFlowRuntime.from_config(config)`` ⇒ 拿
msswift 配置也会**静默跑 native 栈**（用错后端却无声）—— 那是潜在缺陷，不是功能。
**改后**：按 ``selection.name`` 查提供者表，未注册（当前＝仅 msswift）⇒ ``ValueError``。

## 能失败对照（⛔ 不接受恒 PASS）

``test_fail_closed_assertion_would_go_red_if_msswift_were_registered`` 把提供者表**临时补上**
msswift 项，证明"必须 ValueError"这条断言测的是**真行为**：补表后它必红。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from graspo.core.schema import GraspoConfig
from graspo.flow.backend import selection


def _native_runtime_class() -> Any:
    """按**实现同款**方式解析 native runtime 类（⛔ 不用模块级 `from ... import`）。

    ★ 为什么必须这样：测试套里有模块（如
    `tests/flow/backend/native/test_adapter_plugin.py`、`tests/flow/test_backend_switch.py`）
    在**导入期**用 `spec_from_file_location` 把
    `sys.modules["graspo.flow.backend.native.runtime"]` 换成**第二份副本**；而 pytest
    在**跑任何用例之前**就导入了全部测试模块 ⇒ 模块级捕获的类对象可能不是
    `create_runtime` 随后 `importlib.import_module` 解析到的那个：
    `isinstance(...)` 会**假红**，`monkeypatch` 的哨兵会装到**错误的对象**上（那样
    "setup 未被调用"就变成空断言）。按需解析 ⇒ 与实现同源，与收集顺序无关。
    """
    return importlib.import_module("graspo.flow.backend.native.runtime").GraspoFlowRuntime


def _write_msswift_config(tmp_path: Path) -> str:
    """写一份最小可加载的 msswift 配置（``backend: msswift``），返回其路径。"""
    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"p"}],"targets":[{"output":{"content":{"x":1}}}]}\n',
        encoding="utf-8",
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"backend: msswift\n"
        f"model:\n  model_path: {json.dumps(str(tmp_path / 'model'))}\n"
        f"data:\n  train_path: {json.dumps(str(data))}\n"
        f"training:\n  output_dir: {json.dumps(str(tmp_path / 'out'))}\n",
        encoding="utf-8",
    )
    return str(config)


def _fails_closed(config: GraspoConfig) -> bool:
    """``create_runtime`` 是否对该 backend fail-closed（＝抛 ``ValueError``）。"""
    try:
        selection.create_runtime(config)
    except ValueError:
        return True
    return False


def test_evaluate_checkpoint_with_msswift_config_fails_closed_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """★ 覆盖：``run_evaluate``（＝``evaluate-checkpoint`` 的真入口）拿 msswift 配置 ⇒
    ``ValueError``，且**在 ``runtime.setup()`` 之前**抛出 ⇒ 不触 GPU／装配树。

    ``setup()`` 被换成哨兵：一旦未来有人把它改成"先建 runtime 再 setup"，本用例
    不是静默跑过，而是**当场红**并阻止触碰 CUDA。
    """
    config_path = _write_msswift_config(tmp_path)
    data_path = str(tmp_path / "train.jsonl")

    setup_called: list[bool] = []

    def _sentinel_setup(self) -> None:  # pragma: no cover - 只用于证明"没被调用"
        setup_called.append(True)
        raise AssertionError("setup() 被调用 ⇒ 会触碰 GPU／装配树（必须 fail-closed）")

    monkeypatch.setattr(_native_runtime_class(), "setup", _sentinel_setup)

    from graspo.cli.tools import run_evaluate

    with pytest.raises(ValueError, match="msswift"):
        run_evaluate(config_path, data_path)

    assert setup_called == [], "fail-closed 必须发生在 runtime.setup() 之前"


def test_create_runtime_rejects_unregistered_backend(tmp_path: Path):
    """★ 边界：``create_runtime`` 对未注册 runtime 的 backend（msswift）必须 ``ValueError``。"""
    config = GraspoConfig.from_yaml(_write_msswift_config(tmp_path))
    assert selection.select_backend(config).name == "msswift"
    with pytest.raises(ValueError, match="msswift"):
        selection.create_runtime(config)


def test_create_runtime_native_still_constructs(tmp_path: Path):
    """★ 阴性对照：native 路径未被收紧误伤（注册表里 native 仍在）。"""
    config = GraspoConfig.from_yaml(_write_msswift_config(tmp_path))
    config.backend = "native"
    assert isinstance(selection.create_runtime(config), _native_runtime_class())


def test_fail_closed_assertion_would_go_red_if_msswift_were_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """★ 能失败对照：提供者表**临时补上 msswift** ⇒ 主用例的 fail-closed 断言必红。

    断言的是同一个谓词 ``_fails_closed``：补表前必须 True（＝主用例成立），
    补表后必须 False（＝主用例那条"必须 ValueError"在此**必红**）。
    """
    config = GraspoConfig.from_yaml(_write_msswift_config(tmp_path))
    assert _fails_closed(config) is True, "未补表时 msswift 必须 fail-closed"

    monkeypatch.setitem(
        selection._RUNTIME_PROVIDERS,
        "msswift",
        "graspo.flow.backend.native.runtime:GraspoFlowRuntime",
    )

    assert _fails_closed(config) is False, (
        "补表后 create_runtime 不再 fail-closed ⇒ 主用例的断言此时必红（本对照为此而设）"
    )
