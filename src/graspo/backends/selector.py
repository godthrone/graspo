from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from graspo.core.schema import GraspoConfig

SUPPORTED_BACKENDS = {"graspoflow"}


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
            "GRASPO only supports 'graspoflow'."
        )

    return BackendSelection(
        name="graspoflow",
        reason="GRASPO uses GraspoFlow unified tensor/pipeline parallel training",
        requested=requested_backend,
    )


def create_trainer(config: GraspoConfig, selection: BackendSelection):
    if selection.name == "graspoflow":
        from graspo.backends.graspoflow import GraspoFlowTrainer

        return GraspoFlowTrainer(config, selection=selection)
    raise ValueError(f"Unsupported selected backend: {selection.name}")
