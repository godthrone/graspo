from __future__ import annotations

from pathlib import Path


def save_lora_adapter(model, tokenizer, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = model
    if hasattr(model, "module"):
        unwrapped = model.module
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(path)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)

