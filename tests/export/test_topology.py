"""拓扑边界（V3/V5/V6/V7/V8/V10）：分片完备性、PP 覆盖、DP 副本唯一性、LoRA 语义防线。"""

from __future__ import annotations

import json

import pytest
from helpers import make_export_config, write_native_checkpoint

from graspo.export import export_checkpoint
from graspo.export.errors import ExportError


def _run(checkpoint, output, *, fmt="peft-adapter", base="models/base"):
    config = make_export_config(checkpoint, output, fmt=fmt)
    return export_checkpoint(config, base_model_path=base)


def test_dp1_export_succeeds_and_records_consumed_shards(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    output = tmp_path / "out"
    result = _run(checkpoint, output)
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "adapter_config.json").is_file()
    assert (output / "export_manifest.json").is_file()
    assert len(result.consumed_shards) == 1
    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "graspo-export-manifest-1"
    assert manifest["produced_keys"] == 4
    assert "/home/" not in json.dumps(manifest)


def test_manifest_less_native_dir_is_fail_closed(tmp_path):
    """V3：native 形态必需 manifest.json（拓扑唯一真相源），⛔ 不从文件名反推。

    缺 manifest 时检测器**不认**该目录 ⇒ 零命中 fail-closed（而不是猜出一个拓扑）。
    """
    checkpoint = write_native_checkpoint(tmp_path / "final", write_manifest=False)
    with pytest.raises(ExportError, match="cannot classify"):
        _run(checkpoint, tmp_path / "out")


def test_missing_shard_is_fail_closed(tmp_path):
    """V5：manifest 声明 tp=2 但实际只有 1 片 ⇒ 拒绝，消息含期望数/实际数。"""
    checkpoint = write_native_checkpoint(tmp_path / "final", tp_size=2)
    sorted(checkpoint.glob("rank_*.pt"))[-1].unlink()
    with pytest.raises(ExportError) as excinfo:
        _run(checkpoint, tmp_path / "out")
    message = str(excinfo.value)
    assert "expected 2 shards" in message and "found 1" in message


def test_duplicate_shard_triple_is_fail_closed(tmp_path):
    """V5：两片声称同一 ``(dp,tp,pp)`` ⇒ 拒绝（不重不漏）。"""
    import torch

    checkpoint = write_native_checkpoint(tmp_path / "final", tp_size=2)
    shards = sorted(checkpoint.glob("rank_*.pt"))
    payload = torch.load(shards[1], map_location="cpu", weights_only=False)
    payload["tp_rank"] = 0
    torch.save(payload, shards[1])
    with pytest.raises(ExportError, match="duplicates"):
        _run(checkpoint, tmp_path / "out")


def test_dp_replicas_identical_are_merged_with_evidence(tmp_path):
    """V7/T1：dp=2 且两副本逐位相同 ⇒ 成功，ReplicaCheck.identical=True，两片都消费。"""
    checkpoint = write_native_checkpoint(tmp_path / "final", dp_size=2)
    output = tmp_path / "out"
    result = _run(checkpoint, output)
    assert len(result.consumed_shards) == 2
    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["replica_check"]["identical"] is True
    assert manifest["replica_check"]["dp_size"] == 2
    assert manifest["replica_check"]["evidence"]


def test_dp_replicas_divergent_fail_closed_without_artifacts(tmp_path):
    """V7/T2/T3：副本分歧（这里是 1 个元素差）⇒ 拒绝，且不留任何产物/成功记录。"""
    checkpoint = write_native_checkpoint(tmp_path / "final", dp_size=2, divergent=True)
    output = tmp_path / "out"
    with pytest.raises(ExportError, match="NOT identical"):
        _run(checkpoint, output)
    assert not (output / "export_manifest.json").exists()
    if output.exists():
        assert list(output.iterdir()) == []


def test_pp_coverage_gap_is_fail_closed(tmp_path):
    """V6：pp=2 但某个 stage 少了中间层 ⇒ 拒绝（缺片会静默保留基座权重）。"""
    checkpoint = write_native_checkpoint(tmp_path / "final", pp_size=2, gap_in_pp=True)
    with pytest.raises(ExportError, match="contiguous layer range"):
        _run(checkpoint, tmp_path / "out")


def test_pp_full_coverage_is_accepted(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final", pp_size=2)
    output = tmp_path / "out"
    _run(checkpoint, output)
    assert (output / "adapter_model.safetensors").is_file()


def test_full_param_payload_is_refused_by_lora_exporter(tmp_path):
    """V8：manifest 自称 LoRA 但 payload 是全参 ⇒ 逐字保留的语义防线必须拒绝。"""
    checkpoint = write_native_checkpoint(
        tmp_path / "final", fmt="native-lora", tuner_type="full"
    )
    with pytest.raises(ExportError, match="LoRA-only"):
        _run(checkpoint, tmp_path / "out")


def test_native_full_param_form_is_dispatched_away_from_the_lora_exporter(tmp_path):
    """F7 落地后：全参形态由 ``native_full_param`` 承接，⛔ 不再落到 LoRA 通道。

    这里用 LoRA 形态的合成夹具（全参 shard 的 payload 形状不同）只断言**分派**：
    全参 manifest ⇒ ``NATIVE_FULL_PARAM_RANK``，绝不会被 LoRA 导出的语义防线误判为可用。
    """
    from graspo.export.classify import classify_checkpoint
    from graspo.export.schema import CheckpointForm

    checkpoint = write_native_checkpoint(
        tmp_path / "final", fmt="native-full-param", tuner_type="full"
    )
    assert classify_checkpoint(checkpoint).form is CheckpointForm.NATIVE_FULL_PARAM_RANK


def test_missing_base_model_path_is_fail_closed(tmp_path):
    """V10：基座路径唯一来源 = 调用方传入；缺省 ⇒ 拒绝（⛔ 不回落读 checkpoint 内嵌 config）。"""
    checkpoint = write_native_checkpoint(tmp_path / "final")
    config = make_export_config(checkpoint, tmp_path / "out", fmt="merged-hf")
    with pytest.raises(ExportError, match="explicit base model path"):
        export_checkpoint(config)
