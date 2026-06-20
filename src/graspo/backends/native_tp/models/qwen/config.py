from __future__ import annotations

from typing import Any

class NativeQwenConfig:
    def __init__(self, values: dict[str, Any], *, family: str, key_prefix: str) -> None:
        self.family: str = family
        self.key_prefix: str = key_prefix
        for key, value in values.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'NativeQwenConfig' object has no attribute '{name}'")


