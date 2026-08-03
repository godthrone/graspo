"""checkpoint 保存兜底：save_native_checkpoint（异常降级重试的兼容分支）。"""

from pathlib import Path
from typing import Any


def save_native_checkpoint(
    runtime: Any,
    path: str | Path,
    *,
    trainer_state: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime.save_checkpoint(output, trainer_state=trainer_state)
    except TypeError:
        runtime.save_checkpoint(output)
