import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from graspo.backends.native_tp.lora_io import (
    export_merged_hf_from_checkpoint,
    export_peft_adapter_from_checkpoint,
    load_peft_adapter_into_native_model,
)
from graspo.backends.native_tp.qwen_tp_adapter import LoRALinear
from graspo.backends.native_tp.runtime import validate_native_runtime_config
from graspo.core.schema import GraspoConfig


class _TinyNativeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = LoRALinear(
            torch.zeros(3, 4),
            None,
            lora_enabled=True,
            r=2,
            alpha=4,
            dropout=0.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            target_name="language.self_attn.q_proj",
            hf_module_path="model.layers.0.self_attn.q_proj",
        )

    def lora_tensor_metadata(self):
        return [self.proj.lora_metadata("proj")]


def test_load_peft_adapter_into_native_model(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/models/base",
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj"],
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    a = torch.arange(8, dtype=torch.float32).view(2, 4)
    b = torch.arange(6, dtype=torch.float32).view(3, 2)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
        },
        str(adapter / "adapter_model.safetensors"),
    )
    model = _TinyNativeModel()

    load_peft_adapter_into_native_model(model, adapter, base_model_path="/models/base")

    assert torch.equal(model.proj.lora_a, a)
    assert torch.equal(model.proj.lora_b, b)


def test_export_peft_adapter_from_native_checkpoint(tmp_path):
    checkpoint = tmp_path / "step_1"
    checkpoint.mkdir()
    a = torch.ones(2, 4)
    b = torch.arange(6, dtype=torch.float32).view(3, 2)
    _write_payload(
        checkpoint / "rank_00000_tp_00_pp_00.pt",
        metadata=[
            _record(
                module_name="layers.0.self_attn.q_proj",
                hf_module_path="model.layers.0.self_attn.q_proj",
            )
        ],
        state={
            "layers.0.self_attn.q_proj.lora_a": a,
            "layers.0.self_attn.q_proj.lora_b": b,
        },
    )

    output = tmp_path / "peft"
    export_peft_adapter_from_checkpoint(checkpoint, output, base_model_path="/models/base")

    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(output / "adapter_model.safetensors"), device="cpu")
    assert config["base_model_name_or_path"] == "/models/base"
    assert tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape == (
        2,
        4,
    )
    assert torch.equal(tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"], b)


def test_export_merged_hf_from_native_checkpoint_adds_lora_delta(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    weight = torch.zeros(3, 4)
    save_file({"model.layers.0.self_attn.q_proj.weight": weight}, str(base / "model.safetensors"))
    checkpoint = tmp_path / "step_1"
    checkpoint.mkdir()
    a = torch.ones(2, 4)
    b = torch.ones(3, 2)
    _write_payload(
        checkpoint / "rank_00000_tp_00_pp_00.pt",
        metadata=[
            _record(
                module_name="layers.0.self_attn.q_proj",
                hf_module_path="model.layers.0.self_attn.q_proj",
                alpha=4,
                r=2,
            )
        ],
        state={
            "layers.0.self_attn.q_proj.lora_a": a,
            "layers.0.self_attn.q_proj.lora_b": b,
        },
    )

    output = tmp_path / "merged"
    export_merged_hf_from_checkpoint(checkpoint, output, base_model_path=base)

    merged = load_file(str(output / "model.safetensors"), device="cpu")
    assert torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], torch.full((3, 4), 4.0))
    assert (output / "config.json").exists()


def test_export_merged_hf_infers_metadata_for_legacy_lora_state_dict(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    qkv = torch.zeros(6, 4)
    save_file(
        {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": qkv},
        str(base / "model.safetensors"),
    )
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    for tp_rank in (0, 1):
        _write_payload(
            checkpoint / f"rank_0000{tp_rank}_tp_0{tp_rank}_pp_00.pt",
            metadata=None,
            state={
                "layers.0.token_mixer.q_proj.lora_a": torch.ones(1, 4),
                "layers.0.token_mixer.q_proj.lora_b": torch.ones(1, 1),
                "layers.0.token_mixer.v_proj.lora_a": torch.ones(1, 4),
                "layers.0.token_mixer.v_proj.lora_b": torch.full((1, 1), 2.0),
            },
            tp_rank=tp_rank,
            tp_size=2,
            r=1,
            alpha=2,
        )

    output = tmp_path / "merged"
    export_merged_hf_from_checkpoint(checkpoint, output, base_model_path=base)

    merged = load_file(str(output / "model.safetensors"), device="cpu")
    actual = merged["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
    expected = torch.zeros(6, 4)
    expected[0] = 2
    expected[1] = 2
    expected[4] = 4
    expected[5] = 4
    assert torch.equal(actual, expected)


def test_export_merged_hf_maps_legacy_pp_local_layer_indices(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    save_file(
        {"model.language_model.layers.5.linear_attn.in_proj_qkv.weight": torch.zeros(6, 4)},
        str(base / "model.safetensors"),
    )
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    _write_payload(
        checkpoint / "rank_00001_tp_00_pp_01.pt",
        metadata=None,
        state={
            "layers.0.token_mixer.q_proj.lora_a": torch.ones(1, 4),
            "layers.0.token_mixer.q_proj.lora_b": torch.ones(2, 1),
        },
        tp_rank=0,
        tp_size=1,
        pp_rank=1,
        pp_size=2,
        local_layer_indices=[5],
        r=1,
        alpha=2,
    )

    output = tmp_path / "merged"
    export_merged_hf_from_checkpoint(checkpoint, output, base_model_path=base)

    merged = load_file(str(output / "model.safetensors"), device="cpu")
    actual = merged["model.language_model.layers.5.linear_attn.in_proj_qkv.weight"]
    expected = torch.zeros(6, 4)
    expected[0:2] = 2
    assert torch.equal(actual, expected)


def test_export_peft_adapter_fails_for_fused_split_target(tmp_path):
    checkpoint = tmp_path / "step_1"
    checkpoint.mkdir()
    _write_payload(
        checkpoint / "rank_00000_tp_00_pp_00.pt",
        metadata=[
            _record(
                module_name="layers.0.token_mixer.q_proj",
                hf_module_path="model.language_model.layers.0.linear_attn.in_proj_qkv",
                base_weight_name="model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                target_name="language.linear_attn.q_proj",
                peft_exportable=False,
            )
        ],
        state={
            "layers.0.token_mixer.q_proj.lora_a": torch.ones(2, 4),
            "layers.0.token_mixer.q_proj.lora_b": torch.ones(3, 2),
        },
    )

    with pytest.raises(ValueError, match="merged-hf"):
        export_peft_adapter_from_checkpoint(checkpoint, tmp_path / "peft")


def test_native_runtime_rejects_resume_checkpoint_with_peft_adapter():
    config = GraspoConfig()
    config.training.resume_from_checkpoint = "outputs/run/final"
    config.lora.adapter_path = "adapter"

    with pytest.raises(ValueError, match="cannot both be set"):
        validate_native_runtime_config(config)


def _write_payload(
    path,
    *,
    metadata,
    state,
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    local_layer_indices=None,
    r=2,
    alpha=4,
):
    payload = {
        "adapter": "qwen_native_tp",
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "lora_state_dict": state,
        "placement": {"local_layer_indices": local_layer_indices or [0]},
        "config": {
            "model": {"model_path": "/models/base"},
            "lora": {"r": r, "alpha": alpha, "dropout": 0.0, "bias": "none"},
        },
    }
    if metadata is not None:
        payload["lora_tensor_metadata"] = metadata
    torch.save(payload, path)


def _record(
    *,
    module_name,
    hf_module_path,
    base_weight_name=None,
    target_name="language.self_attn.q_proj",
    shard_kind="none",
    peft_exportable=True,
    alpha=4,
    r=2,
):
    return {
        "module_name": module_name,
        "lora_a_name": f"{module_name}.lora_a",
        "lora_b_name": f"{module_name}.lora_b",
        "target_name": target_name,
        "hf_module_path": hf_module_path,
        "base_weight_name": base_weight_name or f"{hf_module_path}.weight",
        "shard_kind": shard_kind,
        "row_start": None,
        "row_stop": None,
        "col_start": None,
        "col_stop": None,
        "row_indices": None,
        "peft_exportable": peft_exportable,
        "r": r,
        "alpha": alpha,
    }
