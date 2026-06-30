"""Pure helpers for LoRA target module resolution.

Extracted from ``lora.py`` (Type B helpers):
these functions don't depend on ``LoRALinear``'s state and are independently testable.
"""


from graspo.backends.graspoflow.models.qwen3.config import NativeQwenConfig


def native_qwen_lora_available_targets(hf_config: NativeQwenConfig) -> tuple[str, ...]:
    language_mlp = (
        "language.mlp.gate_proj",
        "language.mlp.up_proj",
        "language.mlp.down_proj",
    )
    if hf_config.family == "qwen3":
        return (
            "language.self_attn.q_proj",
            "language.self_attn.k_proj",
            "language.self_attn.v_proj",
            "language.self_attn.o_proj",
            *language_mlp,
        )
    if hf_config.family == "qwen3_5_text":
        targets: tuple[str, ...] = (
            "language.full_attn.q_proj",
            "language.full_attn.k_proj",
            "language.full_attn.v_proj",
            "language.full_attn.o_proj",
            "language.linear_attn.q_proj",
            "language.linear_attn.k_proj",
            "language.linear_attn.v_proj",
            "language.linear_attn.in_proj_z",
            "language.linear_attn.out_proj",
            *language_mlp,
        )
        if bool(getattr(hf_config, "has_vision_config", False)):
            depth = int((getattr(hf_config, "vision_config", {}) or {}).get("depth") or 0)
            visual_block_targets = tuple(
                target
                for idx in range(depth)
                for target in (
                    f"visual.blocks.{idx}.attn.qkv",
                    f"visual.blocks.{idx}.attn.proj",
                    f"visual.blocks.{idx}.mlp.linear_fc1",
                    f"visual.blocks.{idx}.mlp.linear_fc2",
                )
            )
            targets = (
                *targets,
                "visual.merger.linear_fc1",
                "visual.merger.linear_fc2",
                *visual_block_targets,
            )
        return targets
    return ()


def _lora_target_enabled(lora_targets: set[str], canonical_name: str) -> bool:
    return canonical_name in lora_targets or canonical_name.rsplit(".", 1)[-1] in lora_targets
