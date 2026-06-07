from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from graspo.core.schema import GraspoConfig


SUPPORTED_BACKENDS = {"native-tp"}


@dataclass(slots=True)
class BackendSelection:
    name: str
    reason: str
    requested: str
    gpu_count: int
    native_tp_available: bool
    model_looks_large: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def select_backend(config: GraspoConfig, requested: str | None = None) -> BackendSelection:
    requested_backend = (requested or config.backend or "native-tp").strip()
    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{requested_backend}'. "
            "GRASPO latest-only training supports only native-tp."
        )

    return BackendSelection(
        name="native-tp",
        reason="latest-only GRASPO uses native tensor/pipeline parallel training",
        requested=requested_backend,
        gpu_count=0,
        native_tp_available=True,
    )


def create_trainer(config: GraspoConfig, selection: BackendSelection):
    if selection.name == "native-tp":
        from graspo.backends.native_tp import NativeTPGraspoTrainer

        return NativeTPGraspoTrainer(config, selection=selection)
    raise ValueError(f"Unsupported selected backend: {selection.name}")
