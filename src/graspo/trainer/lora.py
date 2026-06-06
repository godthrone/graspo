from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase

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

LORA_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "language_safe": (
        "language.*.q_proj",
        "language.*.v_proj",
    ),
    "language_all_linear": (
        "language.*.q_proj",
        "language.*.k_proj",
        "language.*.v_proj",
        "language.*.o_proj",
        "language.*.in_proj_z",
        "language.*.out_proj",
        "language.*.gate_proj",
        "language.*.up_proj",
        "language.*.down_proj",
    ),
    "vision_merger": (
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
    ),
    "vision_common": (
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
        "visual.blocks.*.attn.*",
        "visual.blocks.*.mlp.linear_fc*",
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedLoRATargets:
    requested: tuple[str, ...]
    resolved: tuple[str, ...]

    @property
    def signature(self) -> dict[str, object]:
        return {
            "requested": list(self.requested),
            "resolved": list(self.resolved),
        }


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


def resolve_lora_target_modules(
    requested: Iterable[str] | None,
    *,
    available: Iterable[str],
    default_preset: str = "language_safe",
) -> ResolvedLoRATargets:
    """Resolve native LoRA targets from exact names, globs, legacy aliases, or presets."""

    available_targets = tuple(sorted(set(str(item) for item in available)))
    raw_requested = tuple(str(item) for item in (requested or (default_preset,)))
    patterns: list[str] = []
    for item in raw_requested:
        preset = LORA_TARGET_PRESETS.get(item)
        patterns.extend(preset if preset is not None else (item,))

    resolved: set[str] = set()
    unsupported: list[str] = []
    for pattern in patterns:
        matches = _match_lora_pattern(pattern, available_targets)
        if matches:
            resolved.update(matches)
        else:
            unsupported.append(pattern)

    if unsupported:
        raise ValueError(
            "Unsupported LoRA target(s): "
            + ", ".join(unsupported)
            + ". Available targets: "
            + ", ".join(available_targets)
        )
    if not resolved:
        raise ValueError("LoRA target resolution produced no trainable modules")
    return ResolvedLoRATargets(requested=raw_requested, resolved=tuple(sorted(resolved)))


def _match_lora_pattern(pattern: str, available: tuple[str, ...]) -> list[str]:
    if any(ch in pattern for ch in "*?[]"):
        return [target for target in available if fnmatchcase(target, pattern)]
    exact = [target for target in available if target == pattern]
    if exact:
        return exact
    return [target for target in available if target.rsplit(".", 1)[-1] == pattern]


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
