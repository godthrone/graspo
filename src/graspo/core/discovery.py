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
        "native": "graspo.flow.backend_selection:create_native_trainer",
        "msswift": "graspo.flow.msswift.trainer:create_msswift_trainer",
    },
    # SFT 后端注册表：与 RL（graspo.backends）分开登记，因为两者的训练器形状
    # 不同（SFT 消费「已 tokenize 的样本」，RL 消费 rollout group）。
    # 新增后端 SFT 支持 = 此表加一行 + pyproject entry_points 加一行，
    # 现有选择逻辑 0 行改动（宪法 §1.2 / 决策 D5）。
    "graspo.sft_backends": {
        "native": "graspo.flow.trainer.sft_trainer:create_native_sft_trainer",
        "msswift": "graspo.flow.msswift.sft_trainer:create_msswift_sft_trainer",
    },
    # CPT（继续预训练）注册表：形态与 SFT/RL 都不同——它走 ms-swift 的**预训练**
    # 管线（`swift.pipelines.pretrain_main`，等价于 `swift pt`），数据集是纯文本行
    # 而不是对话行。**只有 ms-swift 实现**（能力矩阵 §4：CPT · native = `⛔ 不支持`）。
    "graspo.cpt_backends": {
        "msswift": "graspo.flow.msswift.cpt_trainer:create_msswift_cpt_trainer",
    },
    # OPD（on-policy 蒸馏）注册表：走 ms-swift 的 GKD 路径
    # （`--rlhf_type gkd` + 独立冻结教师 `--teacher_model`）。**只有 ms-swift 实现**
    # （能力矩阵 §4：OPD · native = `⛔ 不支持`）。
    "graspo.opd_backends": {
        "msswift": "graspo.flow.msswift.opd_trainer:create_msswift_opd_trainer",
    },
}

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

    # 开发模式回退：从 _DEV_FALLBACKS 加载
    if group in _DEV_FALLBACKS:
        result: dict[str, Callable[[], Any]] = {}
        for name, path in _DEV_FALLBACKS[group].items():
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

    所有注册表都走 entry_points 自动发现（开发模式回退 ``_DEV_FALLBACKS``）。
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
