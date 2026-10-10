"""自动发现机制：从 setuptools entry_points 或开发模式回退注册表加载实现。

设计原则（§1.2 接口边界）：
- 新增实现只需在 pyproject.toml 声明 entry_point，不修改任何现有代码
- 开发模式下（pip install -e 未执行）回退到各层（flow/ripple）自行注册的回退路径
- 返回值是 lazy-loading callable：调用方只遍历 .keys() 时不触发重量级导入
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from typing import Any

# ── 开发模式回退注册表（★ X4：core 不得硬编码 flow / ripple 的模块路径）──────
# 当包未通过 pip install 安装时（entry_points 不可用），从**各层自行注册**的
# 回退表加载。生产环境（pip install）中 entry_points 自动接管。
#
# ★ 边界（宪法 §1.3/§1.4）：具体路径属于**各层自己的知识**——
#   · flow 侧的表在 ``graspo/flow/registry.py``
#   · ripple 侧的表在 ``graspo/ripple/registry.py``
# core 只提供**注册机制**与查询入口，不认识任何 ``graspo.flow.*`` /
# ``graspo.ripple.*`` 路径（否则就是 core → flow/ripple 的字符串反向依赖）。
_REGISTERED_FALLBACKS: dict[str, dict[str, str]] = {}


def register_dev_fallbacks(group: str, mapping: dict[str, str]) -> None:
    """注册某 entry_point 组的开发模式回退表（由各层在自己的 registry 里调用）。

    同组重复注册时**合并**（后注册的覆盖同名项，不丢项），保证幂等。
    """
    _REGISTERED_FALLBACKS.setdefault(group, {}).update(dict(mapping))


#: ``train_method`` → entry_point 组名（**路由的单一真相源**，宪法 §1.4）。
#: 表驱动而不是 if/elif 链：新增训练方法 = 本表加一行 + 注册表加一个实现，
#: 既有分支 0 行改动（§1.2 对扩展开放、对修改关闭）。
_REGISTRY_BY_TRAIN_METHOD: dict[str, str] = {
    "graspo": "graspo.backends",
    "sft": "graspo.sft_backends",
    "cpt": "graspo.cpt_backends",
    "opd": "graspo.opd_backends",
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

    # 开发模式回退：从各层注册的回退表加载（core 不含任何层内路径）
    if group in _REGISTERED_FALLBACKS:
        result: dict[str, Callable[[], Any]] = {}
        for name, path in _REGISTERED_FALLBACKS[group].items():
            module_name, _, attr_name = path.partition(":")
            result[name] = _make_loader(module_name, attr_name)
        return result

    return {}


def resolve_backend_builder(backend: str, *, train_method: str) -> Callable[[], Any]:
    """按 ``(train_method, backend)`` 解析训练器工厂（**唯一的路由真相源**，§1.4 / D5）。

    路由表是 :data:`_REGISTRY_BY_TRAIN_METHOD`——四种训练方法各查自己的注册表：

    - ``graspo``（RL）→ ``graspo.backends``
    - ``sft`` → ``graspo.sft_backends``
    - ``cpt`` → ``graspo.cpt_backends``
    - ``opd`` → ``graspo.opd_backends``

    所有注册表都走 entry_points 自动发现（开发模式回退由各层 ``registry`` 注册）。
    **未知的 ``train_method`` 直接拒绝**，不再静默落到 RL 注册表——静默兜底会让
    拼错的算法名拿着另一种算法去训练（宪法 §3.4 的"坏退路"）。

    调用方拿到的是 lazy loader —— ``resolve_backend_builder(...)()`` 才真正导入
    后端模块。因此本函数本身不触发 torch / ms-swift 导入（可在无 GPU 开发机单测）。

    Args:
        backend: 后端名，如 ``"native"`` / ``"msswift"``。
        train_method: ``"graspo"``（RL）/ ``"sft"`` / ``"cpt"`` / ``"opd"``。

    Returns:
        无参 callable，调用后返回该后端该训练方法的工厂函数（形如
        ``factory(config, selection)``）。

    Raises:
        ValueError: ``train_method`` 未知，或该 ``(train_method, backend)`` 组合
            未注册任何工厂。
    """
    group = _REGISTRY_BY_TRAIN_METHOD.get(train_method)
    if group is None:
        raise ValueError(
            f"Unknown train_method {train_method!r}. "
            f"Known train methods: {', '.join(sorted(_REGISTRY_BY_TRAIN_METHOD))}"
        )
    registry = _discover(group)
    loader = registry.get(backend)
    if loader is None:
        raise ValueError(
            f"No {train_method} trainer registered for backend '{backend}'. "
            f"Registered {train_method} backends: {', '.join(sorted(registry)) or '(none)'}"
        )
    return loader()
