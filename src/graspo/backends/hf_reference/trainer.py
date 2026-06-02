from __future__ import annotations

from typing import Any

from graspo.backends.hf_reference.runtime import HFReferenceRuntime
from graspo.backends.native_tp.trainer import NativeTPGraspoTrainer
from graspo.core.schema import GraspoConfig


class HFReferenceGraspoTrainer(NativeTPGraspoTrainer):
    """Single-process Hugging Face reference backend using the native GRASPO loop."""

    def __init__(self, config: GraspoConfig, selection: Any | None = None) -> None:
        super().__init__(config, selection=selection, runtime=HFReferenceRuntime(config))
        self.backend_name = "hf-reference"
