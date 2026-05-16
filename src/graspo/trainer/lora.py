from __future__ import annotations

from collections.abc import Iterable

COMMON_LORA_MODULE_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "wq",
    "wk",
    "wv",
    "wo",
)


def module_leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def detect_lora_target_modules(model: object, candidates: Iterable[str] | None = None) -> list[str]:
    import torch.nn as nn

    candidate_set = set(candidates or COMMON_LORA_MODULE_NAMES)
    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = module_leaf_name(name)
        if leaf in candidate_set:
            found.add(leaf)

    if not found:
        raise ValueError(
            "Could not auto-detect LoRA target modules. "
            "Set lora.target_modules explicitly, for example: "
            "['q_proj', 'k_proj', 'v_proj', 'o_proj']."
        )
    return sorted(found)


def build_peft_config(config, model):
    from peft import LoraConfig

    target_modules = config.lora.target_modules
    if target_modules is None and config.lora.auto_target_modules:
        target_modules = detect_lora_target_modules(model)
    if not target_modules:
        raise ValueError("lora.target_modules is empty and auto_target_modules is disabled")

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )

