from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass

from graspo.core.schema import GraspoConfig


SUPPORTED_BACKENDS = {"auto", "native-tp", "hf-reference"}


@dataclass(slots=True)
class BackendSelection:
    name: str
    reason: str
    requested: str
    gpu_count: int
    native_tp_available: bool
    model_looks_large: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def select_backend(config: GraspoConfig, requested: str | None = None) -> BackendSelection:
    requested_backend = (requested or config.backend or "auto").strip()
    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{requested_backend}'. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_BACKENDS))}."
        )

    gpu_count = detect_gpu_count()
    model_looks_large = looks_like_large_model(config.model.model_path)
    native_tp_available = True

    if requested_backend == "native-tp":
        return BackendSelection(
            name="native-tp",
            reason="requested explicitly",
            requested=requested_backend,
            gpu_count=gpu_count,
            native_tp_available=native_tp_available,
            model_looks_large=model_looks_large,
        )

    if requested_backend == "hf-reference":
        return BackendSelection(
            name="hf-reference",
            reason="requested explicitly",
            requested=requested_backend,
            gpu_count=gpu_count,
            native_tp_available=native_tp_available,
            model_looks_large=model_looks_large,
        )

    if gpu_count >= 2:
        return BackendSelection(
            name="native-tp",
            reason="auto: multiple GPUs detected, using self-owned native tensor parallel",
            requested=requested_backend,
            gpu_count=gpu_count,
            native_tp_available=native_tp_available,
            model_looks_large=model_looks_large,
        )

    if model_looks_large:
        return BackendSelection(
            name="native-tp",
            reason="auto: large model detected, using self-owned native tensor parallel",
            requested=requested_backend,
            gpu_count=gpu_count,
            native_tp_available=native_tp_available,
            model_looks_large=model_looks_large,
        )

    return BackendSelection(
        name="hf-reference",
        reason="auto: small/local model detected, using reference backend",
        requested=requested_backend,
        gpu_count=gpu_count,
        native_tp_available=native_tp_available,
        model_looks_large=model_looks_large,
    )


def create_trainer(config: GraspoConfig, selection: BackendSelection):
    if selection.name == "hf-reference":
        from graspo.backends.hf_reference import HFReferenceGraspoTrainer

        return HFReferenceGraspoTrainer(config, selection=selection)
    if selection.name == "native-tp":
        from graspo.backends.native_tp import NativeTPGraspoTrainer

        return NativeTPGraspoTrainer(config, selection=selection)
    raise ValueError(f"Unsupported selected backend: {selection.name}")


def detect_gpu_count() -> int:
    env_override = os.environ.get("GRASPO_GPU_COUNT") or os.environ.get("GPU_COUNT")
    if env_override and env_override.isdigit():
        return int(env_override)

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return len([line for line in proc.stdout.splitlines() if line.strip()])
    except Exception:
        pass
    return 0


def looks_like_large_model(model_path: str) -> bool:
    value = str(model_path or "").lower()
    if re.search(r"(?:^|[-_/])(?:2[0-9]|3[0-9]|4[0-9]|7[0-9]|100)b(?:$|[-_/])", value):
        return True
    return any(marker in value for marker in ("27b", "30b", "32b", "70b", "qwen3.6-27"))
