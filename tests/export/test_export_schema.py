"""导出域数据模型的契约：``extra="forbid"``、frozen、``None`` 语义、枚举取值。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from graspo.export.schema import (
    ArtifactManifest,
    CheckpointForm,
    ExportTarget,
    PlacementRecord,
    ReplicaCheck,
    ShardPlan,
    ShardRef,
    UnsupportedForm,
)


def _plan(**overrides):
    payload = dict(
        form=CheckpointForm.NATIVE_LORA_RANK,
        target=ExportTarget.PEFT_ADAPTER,
        checkpoint_name="final",
        source_dir="/tmp/final",
    )
    payload.update(overrides)
    return ShardPlan(**payload)


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        _plan(unexpected_field=1)


def test_frozen_model_cannot_be_mutated():
    plan = _plan()
    with pytest.raises(ValidationError):
        plan.target = ExportTarget.MERGED_HF


def test_none_semantics_are_preserved():
    plan = _plan()
    assert plan.topology is None
    assert plan.replica_check is None
    assert plan.base_model_path is None
    assert plan.consumed_shards == []


def test_manifest_carries_no_host_paths():
    manifest = ArtifactManifest(
        form=CheckpointForm.NATIVE_LORA_RANK,
        target=ExportTarget.PEFT_ADAPTER,
        checkpoint_name="final",
        consumed_shards=[
            ShardRef(name="rank_00000_tp_00_pp_00.pt", dp_rank=0, tp_rank=0, pp_rank=0, sha256="a")
        ],
        base_model_name="base",
        replica_check=ReplicaCheck(dp_size=2, identical=True, compared_keys=2, evidence=["x"]),
        produced_files=["adapter_config.json"],
        produced_keys=2,
    )
    dumped = manifest.model_dump_json()
    assert "/home/" not in dumped and "models/test" not in dumped
    assert manifest.base_model_name == "base"


def test_form_and_target_values_are_the_single_truth_vocabulary():
    assert CheckpointForm.HF_BASE_MODEL.value == "hf-base-model"
    assert CheckpointForm.NATIVE_LORA_RANK.value == "native-lora-rank"
    assert ExportTarget.MERGED_HF.value == "merged-hf"
    assert ExportTarget.PEFT_ADAPTER.value == "peft-adapter"


def test_export_target_parse_rejects_unknown_value():
    assert ExportTarget.parse("merged-hf") is ExportTarget.MERGED_HF
    with pytest.raises(ValueError, match="unsupported export target"):
        ExportTarget.parse("nonsense")


def test_unsupported_form_keeps_a_callable_detector():
    def detector(root: Path):
        return True, ("seen",)

    disposition = UnsupportedForm(
        form=CheckpointForm.NATIVE_FULL_PARAM_RANK, reason="M4", detector=detector
    )
    assert disposition.detector is not None
    assert disposition.detector(Path("/tmp")) == (True, ("seen",))


def test_placement_record_defaults_are_explicit_not_implicit_empty_strings():
    placement = PlacementRecord()
    assert placement.local_layer_indices == []
    assert placement.include_lm_head is False and placement.include_embeddings is False
