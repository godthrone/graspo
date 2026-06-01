from __future__ import annotations

from pathlib import Path
from typing import Any


def save_native_checkpoint(runtime: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime.save_checkpoint(output)
