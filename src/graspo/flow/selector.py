"""后端注册表：select_backend / create_trainer（entry_points 自动发现）。"""

import json
from dataclasses import asdict, dataclass

from graspo.core.discovery import _discover
from graspo.core.schema import GraspoConfig

SUPPORTED_BACKENDS = set(_discover("graspo.backends").keys())


@dataclass(slots=True)
class BackendSelection:
    name: str
    reason: str
    requested: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def select_backend(config: GraspoConfig, requested: str | None = None) -> BackendSelection:
    requested_backend = (requested or config.backend or "graspoflow").strip()
    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{requested_backend}'. "
            f"GRASPO supports: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    return BackendSelection(
        name="graspoflow",
        reason="GRASPO uses GraspoFlow unified tensor/pipeline parallel training",
        requested=requested_backend,
    )


def create_graspoflow_trainer(config: GraspoConfig, selection: BackendSelection):
    """graspoflow 后端的工厂函数（供 entry_points 自动发现）。"""
    from graspo.flow import GraspoFlowTrainer

    return GraspoFlowTrainer(config, selection=selection)


def create_trainer(config: GraspoConfig, selection: BackendSelection):
    backends = _discover("graspo.backends")
    loader = backends.get(selection.name)
    if loader is None:
        raise ValueError(
            f"Unsupported backend '{selection.name}'. "
            f"GRASPO supports: {', '.join(sorted(backends))}"
        )
    factory = loader()
    return factory(config, selection=selection)