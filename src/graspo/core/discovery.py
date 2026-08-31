"""自动发现机制：从 setuptools entry_points 或开发模式回退注册表加载实现。

设计原则（§1.2 接口边界）：
- 新增实现只需在 pyproject.toml 声明 entry_point，不修改任何现有代码
- 开发模式下（pip install -e 未执行）回退到 _DEV_FALLBACKS 硬编码路径
- 返回值是 lazy-loading callable：调用方只遍历 .keys() 时不触发重量级导入
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any, Callable

# ── 开发模式回退注册表 ─────────────────────────────────────────────────
# 当包未通过 pip install 安装时（entry_points 不可用），从此表加载。
# 生产环境（pip install）中 entry_points 自动接管，此表不会被使用。
# 单一真相源（§1.4）：entry_points 是生产真相源，_DEV_FALLBACKS 是开发真相源。

_DEV_FALLBACKS: dict[str, dict[str, str]] = {
    "graspo.rewards": {
        "graspo": "graspo.ripple.reward.reward:GraspoReward",
    },
    "graspo.adapters": {
        "qwen3": "graspo.flow.adapters.models.qwen3.adapter:Qwen3Adapter",
        "qwen35_36": "graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter",
    },
    "graspo.backends": {
        "graspoflow": "graspo.flow.backend_selection:create_graspoflow_trainer",
    },
}


def _make_loader(module_name: str, attr_name: str) -> Callable[[], Any]:
    """创建 lazy loader：调用时才 import 模块并获取属性。"""

    def _load() -> Any:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    return _load


def _discover(group: str) -> dict[str, Callable[[], Any]]:
    """从 entry_points 自动发现注册的实现，返回 {name: lazy_loader} 映射。

    返回值是 lazy-loading callable：调用 ``loader()`` 才真正导入模块。
    这允许调用方只遍历 ``.keys()`` 而不触发重量级导入（如 torch）。

    Args:
        group: entry_point 组名，如 ``"graspo.rewards"``。

    Returns:
        {name: loader} 映射，loader 是无参 callable，调用后返回注册的实现对象。
    """
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.12+
        entries = {ep.name: ep.load for ep in eps.select(group=group)}
    else:  # Python 3.11 fallback
        entries = {ep.name: ep.load for ep in eps.get(group, [])}

    if entries:
        return entries

    # 开发模式回退：从 _DEV_FALLBACKS 加载
    if group in _DEV_FALLBACKS:
        result: dict[str, Callable[[], Any]] = {}
        for name, path in _DEV_FALLBACKS[group].items():
            module_name, _, attr_name = path.partition(":")
            result[name] = _make_loader(module_name, attr_name)
        return result

    return {}