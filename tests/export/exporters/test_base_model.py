"""F1：HF 基座 → merged-hf 的**显式回执**（no-op，不写盘、不假装合并）。"""

from __future__ import annotations

import json

import pytest
from helpers import make_export_config, write_hf_base

from graspo.export import export_checkpoint
from graspo.export.classify import classify_checkpoint
from graspo.export.errors import ExportError
from graspo.export.schema import CheckpointForm


def test_hf_base_is_classified_and_receipted_without_writing(tmp_path):
    base = write_hf_base(tmp_path / "base")
    before = sorted(item.name for item in base.iterdir())
    config = make_export_config(base, tmp_path / "out", fmt="merged-hf")
    result = export_checkpoint(config, base_model_path=None)

    assert result.destination == str(base)
    assert result.manifest_path is None
    assert "model.safetensors" in result.produced_files
    assert result.consumed_shards == []
    assert sorted(item.name for item in base.iterdir()) == before
    assert not (base / "export_manifest.json").exists()


def test_peft_adapter_target_is_explicitly_refused(tmp_path):
    base = write_hf_base(tmp_path / "base")
    config = make_export_config(base, tmp_path / "out", fmt="peft-adapter")
    with pytest.raises(ExportError, match="cannot be exported as 'peft-adapter'"):
        export_checkpoint(config, base_model_path=None)


def test_positive_control_directory_without_weights_is_not_a_base_model(tmp_path):
    """阳性对照：只有 ``config.json``、没有任何权重 ⇒ 不匹配 F1（零命中 fail-closed）。"""
    root = tmp_path / "not-a-model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "test"}), encoding="utf-8")
    with pytest.raises(ExportError, match="cannot classify"):
        classify_checkpoint(root)


def test_evidence_is_recorded(tmp_path):
    base = write_hf_base(tmp_path / "base")
    classification = classify_checkpoint(base)
    assert classification.form is CheckpointForm.HF_BASE_MODEL
    assert any("config.json" in item for item in classification.evidence)
