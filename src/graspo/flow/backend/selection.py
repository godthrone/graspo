"""后端注册表 + ABC 工厂：``select_backend`` / ``create_backend``（entry_points 自动发现）。

★ P3（backend-interface.md §2）：注册表的公开面由 :class:`RLBackend` 工厂吸收 ——
``graspo.backends`` 的每个条目都返回 :class:`RLBackend` 实现；新增 backend =
``flow/backend/<name>/`` 加一个包 + 注册表加一行，**现有代码 0 行改动**（§1.2）。

★ Phase 1b 边界收口：``cli/`` ⛔ 不得 import ``flow.backend.{native,msswift}``
（具体 backend 实现的知识只许住在 ``flow.backend`` 内）。入口层需要 backend 相关
能力时，走本模块的 facade：``select_backend`` / ``create_backend`` /
``create_runtime`` / ``launcher_env``。守卫：``tests/test_ast_boundary.py``。
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from graspo.core.discovery import _discover
from graspo.core.schema import GraspoConfig

if TYPE_CHECKING:
    from graspo.flow.backend.base import RLBackend

SUPPORTED_BACKENDS = set(_discover("graspo.backends").keys())


@dataclass(slots=True)
class BackendSelection:
    name: str
    reason: str
    requested: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def select_backend(config: GraspoConfig, requested: str | None = None) -> BackendSelection:
    requested_backend = (requested or config.backend or "native").strip()
    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{requested_backend}'. "
            f"GRASPO supports: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    reasons = {
        "native": "GRASPO native unified tensor/pipeline parallel training (TP/DP/PP/SP/GC)",
        "msswift": "ms-swift infrastructure (Megatron/DeepSpeed/vLLM) with Graspo ripple algorithm",
    }
    return BackendSelection(
        name=requested_backend,
        reason=reasons.get(requested_backend, f"Backend: {requested_backend}"),
        requested=requested_backend,
    )


def create_backend(config: GraspoConfig, selection: BackendSelection) -> "RLBackend":
    """按 ``selection.name`` 从 ``graspo.backends`` 注册表发现并构造后端（**ABC 类型**）。

    返回 :class:`graspo.flow.backend.base.RLBackend`；具体实现由各 backend 包的
    ``create_<name>_backend`` 提供（native / msswift）。新增 backend 时本函数 0 行改动。

    Raises:
        ValueError: 该 backend 名未注册任何工厂。
    """
    backends = _discover("graspo.backends")
    factory_loader = backends.get(selection.name)
    if factory_loader is None:
        raise ValueError(
            f"Unsupported backend '{selection.name}'. "
            f"GRASPO supports: {', '.join(sorted(backends))}"
        )
    factory = factory_loader()
    return factory(config, selection=selection)


def create_trainer(config: GraspoConfig, selection: BackendSelection) -> Any:
    """兼容别名：旧名 ``create_trainer`` ⇒ 转发 :func:`create_backend`（同一实现，§1.4）。"""
    return create_backend(config, selection)


# ── Phase 1b facade：生成/评测 runtime 与 launcher 环境 ────────────────────────

#: 生成/评测 runtime 的提供者表（backend 名 → ``module:Class``，**唯一真相源**）。
#: 现状登记：只有 native 提供 runtime（``evaluate-checkpoint --checkpoint`` 也是 native
#: 产物）。新增 backend 要提供 runtime ⇒ 本表加一行，入口层 0 行改动（§1.2）。
_RUNTIME_PROVIDERS: dict[str, str] = {
    "native": "graspo.flow.backend.native.runtime:GraspoFlowRuntime",
}


def create_runtime(config: GraspoConfig, selection: BackendSelection | None = None) -> Any:
    """按 ``selection.name`` 构造生成/评测 runtime（入口层唯一入口，§1.4）。

    ★ 边界：``cli/`` 不再直取 ``flow.backend.native.runtime.GraspoFlowRuntime``；
    具体 backend 的模块路径只住在本模块的 :data:`_RUNTIME_PROVIDERS` 表里。

    Raises:
        ValueError: 该 backend 未注册 runtime 提供者 —— **fail-closed**，
            ⛔ 不静默回落到 native 栈（否则 msswift 配置会跑 native runtime）。
    """
    selection = selection or select_backend(config)
    provider = _RUNTIME_PROVIDERS.get(selection.name)
    if provider is None:
        raise ValueError(
            f"backend '{selection.name}' 未注册生成/评测 runtime；"
            f"已注册：{', '.join(sorted(_RUNTIME_PROVIDERS))}"
        )
    module_name, attr_name = provider.split(":", 1)
    runtime_cls = getattr(importlib.import_module(module_name), attr_name)
    return runtime_cls.from_config(config)


def launcher_env(config: GraspoConfig, selection: BackendSelection | None = None) -> dict[str, str]:
    """后端 launcher 环境变量的唯一入口（入口层 ⛔ 不直取 backend 实现包）。

    只有 msswift 侧用 launcher 环境变量承载 S1 数据并行（``NPROC_PER_NODE`` /
    ``NNODES`` / ``NODE_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT``，真相源仍在
    ``msswift/_config_mapping.launcher_env``）；native 侧由 torchrun 参数承载 ⇒ 空 dict。
    """
    selection = selection or select_backend(config)
    if selection.name != "msswift":
        return {}
    from graspo.flow.backend.msswift._config_mapping import launcher_env as _msswift_launcher_env

    return _msswift_launcher_env(config)


__all__ = [
    "SUPPORTED_BACKENDS",
    "BackendSelection",
    "create_backend",
    "create_runtime",
    "create_trainer",
    "launcher_env",
    "select_backend",
]
