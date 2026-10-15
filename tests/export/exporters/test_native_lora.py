"""F3–F6：native LoRA rank 的端到端导出 + T4（禁止静默取副本）。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from helpers import make_export_config, write_hf_base, write_native_checkpoint

from graspo.export import export_checkpoint
from graspo.export.errors import ExportError

_NATIVE_LORA_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "graspo"
    / "export"
    / "exporters"
    / "native_lora.py"
)

#: 旧实现的"静默取副本"表达式形态：``[p for p in payloads if ... dp_rank ... == 0]``
_SILENT_REPLICA_FILTER = re.compile(r"dp_rank[^\n]{0,60}==\s*0\s*\]")


def test_peft_adapter_export_is_complete(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    output = tmp_path / "adapter"
    config = make_export_config(checkpoint, output, fmt="peft-adapter")
    result = export_checkpoint(config, base_model_path="models/test-base")

    adapter_config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["base_model_name_or_path"] == "models/test-base"
    assert adapter_config["r"] == 2 and adapter_config["lora_alpha"] == 4
    assert sorted(adapter_config["target_modules"]) == ["down_proj"]

    from safetensors.torch import load_file

    tensors = load_file(str(output / "adapter_model.safetensors"), device="cpu")
    assert len(tensors) == 4
    assert result.consumed_shards and result.manifest_path is not None


def test_merged_hf_applies_deltas_to_the_base(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    base = write_hf_base(tmp_path / "base")
    output = tmp_path / "merged"
    config = make_export_config(checkpoint, output, fmt="merged-hf")
    export_checkpoint(config, base_model_path=str(base))

    from safetensors.torch import load_file

    assert (output / "config.json").is_file()
    merged = load_file(str(output / "model.safetensors"), device="cpu")
    original = load_file(str(base / "model.safetensors"), device="cpu")
    assert set(merged) == set(original)
    # 阳性对照：基座是零、delta 非零 ⇒ 合并后必须出现非零值（证明真的合并了）。
    assert any(bool(value.abs().sum() > 0) for value in merged.values())


def test_merged_hf_refuses_a_mismatched_base_shape(tmp_path):
    """阳性对照（B12 保留）：基座与 delta 形状不同源 ⇒ fail-closed，⛔ 不静默错并。"""
    import torch
    from safetensors.torch import save_file

    checkpoint = write_native_checkpoint(tmp_path / "final")
    base = tmp_path / "bad-base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "test"}), encoding="utf-8")
    save_file(
        {"model.layers.0.mlp.down_proj.weight": torch.zeros(5, 4)},
        str(base / "model.safetensors"),
    )
    config = make_export_config(checkpoint, tmp_path / "merged", fmt="merged-hf")
    with pytest.raises(ExportError, match="does not match"):
        export_checkpoint(config, base_model_path=str(base))


def test_no_silent_replica_selection_in_source():
    """T4：``export/`` 内不得出现"按 rank 过滤后继续"的取副本表达式。"""
    source = _NATIVE_LORA_SOURCE.read_text(encoding="utf-8")
    assert not _SILENT_REPLICA_FILTER.search(source), (
        "found a silent 'pick dp_rank==0' filter — replica selection must be proven identical "
        "first (V7) and recorded as evidence"
    )


def test_t4_scanner_positive_control():
    planted = 'payloads = [p for p in payloads if int(p.get("dp_rank", 0)) == 0]\n'
    assert _SILENT_REPLICA_FILTER.search(planted) is not None
