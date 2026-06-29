from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase

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


def resolve_lora_target_modules(
    requested: Iterable[str] | None,
    *,
    available: Iterable[str],
    default_preset: str = "language_safe",
) -> ResolvedLoRATargets:
    """Resolve native LoRA targets from exact canonical names, globs, or presets."""

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
    return [target for target in available if target == pattern]
