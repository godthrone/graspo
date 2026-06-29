from pathlib import Path


def save_lora_adapter(model, tokenizer, output_dir: str | Path) -> None:
    """保存 LoRA adapter 和 tokenizer（兼容 DDP 包装和 HF 接口）。"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = model
    # DDP 包装检测：DistributedDataParallel 将模型放在 .module 属性中
    if hasattr(model, "module"):
        unwrapped = model.module
    # HF 标准接口：save_pretrained 是 HuggingFace 模型的约定
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(path)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)
