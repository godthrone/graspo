"""Tests for automatic discovery mechanism — entry_points and dev-mode fallback.

覆盖场景：
1. _discover("graspo.rewards") 返回包含 "graspo" 键的 dict
2. 返回的 loader 可调用，调用后返回 GraspoReward 类
3. _discover("graspo.adapters") 返回包含 "qwen3" 和 "qwen35_36" 键的 dict
4. 获取 adapters 的 keys 不触发 torch 导入（验证 lazy-loading）
5. _discover("graspo.backends") 返回包含 "native" 键的 dict
6. _discover("graspo.unknown") 返回空 dict
7. entry_points 优先于 fallback（mock importlib.metadata.entry_points）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from graspo.core.discovery import _discover


# ── 基本发现：rewards ────────────────────────────────────────────────────────


def test_discover_rewards_has_graspo_key():
    """_discover("graspo.rewards") 返回包含 "graspo" 键的 dict。"""
    result = _discover("graspo.rewards")
    assert isinstance(result, dict)
    assert "graspo" in result


def test_discover_rewards_loader_returns_graspo_reward_class():
    """返回的 loader 可调用，调用后返回 GraspoReward 类。"""
    result = _discover("graspo.rewards")
    loader = result["graspo"]
    assert callable(loader)
    obj = loader()
    # GraspoReward 是一个类（type）
    assert isinstance(obj, type)
    assert obj.__name__ == "GraspoReward"


# ── 基本发现：adapters ───────────────────────────────────────────────────────


def test_discover_adapters_has_qwen3_and_qwen35_36_keys():
    """_discover("graspo.adapters") 返回包含 "qwen3" 和 "qwen35_36" 键的 dict。"""
    result = _discover("graspo.adapters")
    assert isinstance(result, dict)
    assert "qwen3" in result
    assert "qwen35_36" in result


def test_discover_adapters_loader_returns_adapter_class():
    """adapters 的 loader 返回对应的 Adapter 类（需 torch 和 flow 模块可用）。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch required to load adapter modules")
    # 额外检查 graspo.flow.adapters 是否可导入（可能被其他测试 mock 污染）
    try:
        importlib = __import__("importlib")
        importlib.import_module("graspo.flow.adapters.models.qwen3.adapter")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("graspo.flow.adapters not importable (torch mock detected)")
    result = _discover("graspo.adapters")
    for name in ("qwen3", "qwen35_36"):
        loader = result[name]
        assert callable(loader)
        obj = loader()
        assert isinstance(obj, type)
        assert "Adapter" in obj.__name__


# ── 基本发现：backends ───────────────────────────────────────────────────────


def test_discover_backends_has_native_key():
    """_discover("graspo.backends") 返回包含 "native" 键的 dict。"""
    result = _discover("graspo.backends")
    assert isinstance(result, dict)
    assert "native" in result


def test_discover_backends_loader_returns_callable():
    """backends 的 loader 返回可调用对象（需 torch 和 flow 模块可用）。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch required to load backend modules")
    try:
        importlib = __import__("importlib")
        importlib.import_module("graspo.flow")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("graspo.flow not importable (torch mock detected)")
    result = _discover("graspo.backends")
    loader = result["native"]
    assert callable(loader)
    obj = loader()
    assert callable(obj)  # create_native_trainer 是一个函数


# ── 未知 group ────────────────────────────────────────────────────────────────


def test_discover_unknown_group_returns_empty_dict():
    """_discover("graspo.unknown") 返回空 dict。"""
    result = _discover("graspo.unknown")
    assert isinstance(result, dict)
    assert result == {}


# ── Lazy-loading：keys() 不触发 torch 导入 ──────────────────────────────────


def test_discover_adapters_keys_does_not_import_torch():
    """获取 adapters 的 keys 不触发 torch 导入（验证 lazy-loading）。

    使用子进程确保干净环境，避免当前进程已导入 torch 的干扰。
    """
    code = """
import sys
from graspo.core.discovery import _discover

assert "torch" not in sys.modules, "torch already imported before discovery"

result = _discover("graspo.adapters")
keys = list(result.keys())

assert "torch" not in sys.modules, (
    f"torch was imported when accessing keys: {keys}"
)
print("OK: torch not imported")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0, f"Subprocess failed:\n{proc.stderr}"
    assert "OK: torch not imported" in proc.stdout


# ── entry_points 优先于 fallback ─────────────────────────────────────────────


def test_entry_points_take_priority_over_fallback():
    """mock entry_points 返回包含测试项的结果，验证优先于 fallback。

    当 entry_points 返回非空结果时，_discover 应直接使用 entry_points
    而不回退到 _DEV_FALLBACKS。
    """
    # 创建一个假的 EntryPoint，有 name 和 load 属性
    fake_obj = object()

    class FakeEntryPoint:
        def __init__(self, name: str, obj: Any) -> None:
            self._name = name
            self._obj = obj

        @property
        def name(self) -> str:
            return self._name

        def load(self) -> Any:
            return self._obj

    fake_ep = FakeEntryPoint("test_plugin", fake_obj)

    # 构造一个 mock entry_points 对象，同时支持 select() 和 get() 两种接口
    class MockEntryPoints:
        def select(self, *, group: str) -> list[FakeEntryPoint]:
            if group == "graspo.rewards":
                return [fake_ep]
            return []

        def get(self, group: str, default: Any = None) -> list[FakeEntryPoint]:
            if group == "graspo.rewards":
                return [fake_ep]
            return default if default is not None else []

    mock_eps = MockEntryPoints()

    with mock.patch("importlib.metadata.entry_points", return_value=mock_eps):
        result = _discover("graspo.rewards")
        # 应返回 entry_points 的结果，而非 fallback
        assert "test_plugin" in result
        assert result["test_plugin"]() is fake_obj
        # 不应包含 fallback 中的 "graspo" 键
        assert "graspo" not in result


def test_entry_points_empty_falls_back_to_dev():
    """entry_points 返回空时，回退到 _DEV_FALLBACKS。"""
    # 构造一个空 entry_points 对象
    class EmptyEntryPoints:
        def select(self, *, group: str) -> list[Any]:
            return []

        def get(self, group: str, default: Any = None) -> list[Any]:
            return default if default is not None else []

    mock_eps = EmptyEntryPoints()

    with mock.patch("importlib.metadata.entry_points", return_value=mock_eps):
        result = _discover("graspo.rewards")
        assert "graspo" in result
        assert callable(result["graspo"])