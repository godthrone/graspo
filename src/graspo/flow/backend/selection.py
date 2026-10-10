"""后端注册表 + ABC 工厂：``select_backend`` / ``create_backend``（entry_points 自动发现）。

★ P3（backend-interface.md §2）：注册表的公开面由 :class:`RLBackend` 工厂吸收 ——
``graspo.backends`` 的每个条目都返回 :class:`RLBackend` 实现；新增 backend =
``flow/backend/<name>/`` 加一个包 + 注册表加一行，**现有代码 0 行改动**（§1.2）。
"""

from __future__ import annotations

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


__all__ = [
    "SUPPORTED_BACKENDS",
    "BackendSelection",
    "create_backend",
    "create_trainer",
    "select_backend",
]
