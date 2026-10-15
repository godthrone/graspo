"""落盘回读自校验（V14/V15）：产物键集合守恒、失败不留"半成品 + 成功记录"。"""

from __future__ import annotations

import json

import pytest
from helpers import make_export_config, write_native_checkpoint

from graspo.export.classify import classify_checkpoint
from graspo.export.errors import ExportError
from graspo.export.exporters.native_lora import NativeLoraRankExporter


def _exporter(checkpoint):
    classification = classify_checkpoint(checkpoint)
    return NativeLoraRankExporter(), classification


def test_produced_manifest_records_shard_hashes(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    exporter, classification = _exporter(checkpoint)
    config = make_export_config(checkpoint, tmp_path / "out")
    result = exporter.export(config, classification, base_model_path="models/base")
    manifest = json.loads((tmp_path / "out" / "export_manifest.json").read_text(encoding="utf-8"))
    assert len(result.consumed_shards) == 1
    shard = manifest["consumed_shards"][0]
    assert shard["name"] == "rank_00000_tp_00_pp_00.pt"
    assert len(shard["sha256"]) == 64
    assert manifest["produced_keys"] == 4
    assert manifest["replica_check"] is None


def test_self_check_detects_a_dropped_key_and_leaves_no_manifest(tmp_path, monkeypatch):
    """阳性对照（V14/V15）：`_write` 少写一键 ⇒ `_verify` 必抛错且无 ``export_manifest.json``。

    ``_write`` **不修改 assembled**（真实缺陷的形态：组装对了、落盘漏了），因此回读
    比对能看见"产物少了键"。
    """
    from graspo.export.exporters.native_lora import PeftAdapterAssembly

    checkpoint = write_native_checkpoint(tmp_path / "final")
    exporter, classification = _exporter(checkpoint)

    def dropping_write(self, destination, assembled, plan):
        # 模拟"组装对了、落盘漏了一键"：入参对象不变（reader 由 _write 注入），
        # 只把写盘内容换成少一个键的副本。
        reduced = PeftAdapterAssembly(
            tensors={k: v for k, v in list(assembled.tensors.items())[:-1]},
            adapter_config=assembled.adapter_config,
            graspo_metadata=assembled.graspo_metadata,
        )
        NativeLoraRankExporter._write_peft_adapter(destination, reduced)
        assembled.reader = reduced.reader
        return ("adapter_model.safetensors", "adapter_config.json")

    monkeypatch.setattr(NativeLoraRankExporter, "_write", dropping_write)

    config = make_export_config(checkpoint, tmp_path / "out")
    with pytest.raises(ExportError, match="self-check failed"):
        exporter.export(config, classification, base_model_path="models/base")

    output = tmp_path / "out"
    assert not (output / "export_manifest.json").exists()
    # 半成品可能已落盘（由调用方决定是否清理），但错误消息必须能让调用方看到它。
    assert (output / "adapter_model.safetensors").exists()


def test_self_check_detects_non_finite_values(tmp_path, monkeypatch):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    exporter, classification = _exporter(checkpoint)
    original = NativeLoraRankExporter._write

    def nan_write(self, destination, assembled, plan):
        for key in list(assembled.tensors)[:1]:
            assembled.tensors[key] = assembled.tensors[key] * float("nan")
        return original(self, destination, assembled, plan)

    monkeypatch.setattr(NativeLoraRankExporter, "_write", nan_write)
    config = make_export_config(checkpoint, tmp_path / "out")
    with pytest.raises(ExportError, match="non-finite"):
        exporter.export(config, classification, base_model_path="models/base")
    assert not (tmp_path / "out" / "export_manifest.json").exists()
