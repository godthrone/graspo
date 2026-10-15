"""形态检测（V1）：逐形态正例 + 零命中/多命中 fail-closed（注册表驱动、顺序不敏感）。"""

from __future__ import annotations

import pytest
from helpers import write_hf_base, write_native_checkpoint

from graspo.export.classify import classify_checkpoint
from graspo.export.errors import ExportError
from graspo.export.schema import CheckpointForm


def test_classifies_native_lora_rank(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    classification = classify_checkpoint(checkpoint)
    assert classification.form is CheckpointForm.NATIVE_LORA_RANK
    assert any("manifest.json" in item for item in classification.evidence)


def test_classifies_native_full_param_rank(tmp_path):
    checkpoint = write_native_checkpoint(
        tmp_path / "final", fmt="native-full-param", tuner_type="full"
    )
    classification = classify_checkpoint(checkpoint)
    assert classification.form is CheckpointForm.NATIVE_FULL_PARAM_RANK


def test_classifies_hf_base_model(tmp_path):
    base = write_hf_base(tmp_path / "base")
    classification = classify_checkpoint(base)
    assert classification.form is CheckpointForm.HF_BASE_MODEL


def test_zero_hit_fails_closed_and_lists_entries(tmp_path):
    root = tmp_path / "mystery"
    root.mkdir()
    (root / "readme.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(ExportError) as excinfo:
        classify_checkpoint(root)
    message = str(excinfo.value)
    assert "readme.txt" in message
    assert "registered forms" in message


def test_rank_shards_without_manifest_are_not_recognised(tmp_path):
    """阳性对照（V3）：只有 ``rank_*.pt``、缺 ``manifest.json`` ⇒ 零命中，⛔ 不猜拓扑。"""
    root = tmp_path / "final"
    root.mkdir()
    (root / "rank_00000_tp_00_pp_00.pt").write_bytes(b"x")
    with pytest.raises(ExportError, match="cannot classify"):
        classify_checkpoint(root)


def test_multi_hit_fails_closed_with_all_evidence(tmp_path):
    """阳性对照（V1 多命中）：同一目录既像 HF 基座又像 native rank ⇒ fail-closed 并列证据。"""
    root = write_native_checkpoint(tmp_path / "both")
    write_hf_base(root)
    with pytest.raises(ExportError) as excinfo:
        classify_checkpoint(root)
    message = str(excinfo.value)
    assert "matches multiple forms" in message
    assert "hf-base-model" in message and "native-lora-rank" in message


def test_missing_directory_fails_closed(tmp_path):
    with pytest.raises(ExportError, match="checkpoint directory not found"):
        classify_checkpoint(tmp_path / "nope")
