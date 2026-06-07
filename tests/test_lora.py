import pytest

from graspo.trainer.lora import resolve_lora_target_modules


def test_resolve_lora_targets_supports_presets_globs_and_canonical_names():
    available = [
        "language.self_attn.q_proj",
        "language.self_attn.v_proj",
        "language.self_attn.o_proj",
        "language.mlp.gate_proj",
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
    ]

    assert resolve_lora_target_modules(None, available=available).resolved == (
        "language.self_attn.q_proj",
        "language.self_attn.v_proj",
    )
    assert resolve_lora_target_modules(
        ["language.self_attn.q_proj"], available=available
    ).resolved == ("language.self_attn.q_proj",)
    assert resolve_lora_target_modules(["language.*.o_proj"], available=available).resolved == (
        "language.self_attn.o_proj",
    )
    assert resolve_lora_target_modules(["vision_merger"], available=available).resolved == (
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
    )


def test_resolve_lora_targets_rejects_leaf_aliases_and_unknown_targets():
    available = ["language.self_attn.q_proj"]

    with pytest.raises(ValueError, match="Unsupported LoRA target"):
        resolve_lora_target_modules(["q_proj"], available=available)
    with pytest.raises(ValueError, match="Unsupported LoRA target"):
        resolve_lora_target_modules(["conv1d"], available=available)
