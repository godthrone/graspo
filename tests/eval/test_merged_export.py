"""``graspo.eval.merged_export`` 的单测：checkpoint 形态识别与合并前置守卫。

补齐的缺口是"ms-swift 产物（标准 PEFT 目录）→ merged-hf"。这里测的是**判定与
拒绝路径**——真正的权重合并在 GPU 容器里跑，本轮不验证（见工位 report.md）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspo.eval.merged_export import (
    CheckpointKind,
    ExportError,
    assert_base_matches,
    classify_checkpoint,
    existing_merged_output,
    prepare_output_directory,
    read_adapter_base,
    resolve_peft_adapter_dir,
)


def _write(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _base_model(root: Path) -> Path:
    _write(root / "config.json", '{"model_type": "qwen3"}')
    (root / "model.safetensors").write_bytes(b"\x00\x00")
    return root


def _peft_adapter(root: Path, base: str | None = "/models/Qwen3.5-9B") -> Path:
    payload: dict[str, object] = {"r": 128, "lora_alpha": 256}
    if base is not None:
        payload["base_model_name_or_path"] = base
    _write(root / "adapter_config.json", json.dumps(payload))
    (root / "adapter_model.safetensors").write_bytes(b"\x00\x00")
    return root


def test_classify_base_model(tmp_path):
    result = classify_checkpoint(_base_model(tmp_path / "base"))
    assert result.kind is CheckpointKind.BASE_MODEL
    assert any("config.json" in item for item in result.evidence)


def test_classify_peft_adapter(tmp_path):
    result = classify_checkpoint(_peft_adapter(tmp_path / "adapter"))
    assert result.kind is CheckpointKind.PEFT_ADAPTER


def test_classify_native_graspo_checkpoint(tmp_path):
    root = tmp_path / "native"
    _write(root / "metadata.json")
    (root / "shard_0.safetensors").write_bytes(b"\x00\x00")
    result = classify_checkpoint(root)
    assert result.kind is CheckpointKind.GRASPO_NATIVE


def test_classify_rejects_unknown_layout_with_listing(tmp_path):
    root = tmp_path / "weird"
    root.mkdir()
    (root / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ExportError, match="cannot classify checkpoint"):
        classify_checkpoint(root)


def test_classify_rejects_missing_directory(tmp_path):
    with pytest.raises(ExportError, match="not found"):
        classify_checkpoint(tmp_path / "nope")


def test_resolve_peft_adapter_finds_nested_ms_swift_output(tmp_path):
    root = tmp_path / "run" / "v3-20260901-101010" / "checkpoint-500"
    _peft_adapter(root)
    assert resolve_peft_adapter_dir(tmp_path / "run") == root.resolve()


def test_resolve_peft_adapter_returns_self_when_direct(tmp_path):
    adapter = _peft_adapter(tmp_path / "direct")
    assert resolve_peft_adapter_dir(adapter) == adapter


def test_resolve_peft_adapter_returns_none_when_absent(tmp_path):
    (tmp_path / "empty").mkdir()
    assert resolve_peft_adapter_dir(tmp_path / "empty") is None


def test_resolve_peft_adapter_refuses_ambiguous_layout(tmp_path):
    _peft_adapter(tmp_path / "out" / "a")
    _peft_adapter(tmp_path / "out" / "b")
    with pytest.raises(ExportError, match="multiple PEFT adapters found"):
        resolve_peft_adapter_dir(tmp_path / "out")


def test_read_adapter_base_and_mismatch_rejection(tmp_path):
    adapter = _peft_adapter(tmp_path / "adapter", base="/models/Qwen3.5-9B")
    assert read_adapter_base(adapter) == "/models/Qwen3.5-9B"
    assert_base_matches(adapter, "/models/Qwen3.5-9B")  # 同名即通过

    with pytest.raises(ExportError, match="adapter base mismatch"):
        assert_base_matches(adapter, "/models/Other-9B")


def test_assert_base_matches_rejects_adapter_without_recorded_base(tmp_path):
    """拿错底座会合出一个"看起来很正常"的错模型——所以无法确认时必须拒绝。"""
    adapter = _peft_adapter(tmp_path / "adapter", base=None)
    with pytest.raises(ExportError, match="does not record base_model_name_or_path"):
        assert_base_matches(adapter, "/models/Qwen3.5-9B")


def test_existing_merged_output_requires_config_and_weights(tmp_path):
    assert existing_merged_output(tmp_path) is False
    _write(tmp_path / "config.json")
    assert existing_merged_output(tmp_path) is False
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    assert existing_merged_output(tmp_path) is True


def test_prepare_output_directory_creates_when_absent(tmp_path):
    target = tmp_path / "new" / "dir"
    assert prepare_output_directory(target) == target
    assert target.is_dir()


def test_prepare_output_directory_refuses_nonempty_without_optin(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.bin").write_bytes(b"x")
    with pytest.raises(ExportError, match="refusing to overwrite"):
        prepare_output_directory(target)
    assert (target / "old.bin").exists()  # 拒绝时不动任何东西


def test_prepare_output_directory_overwrite_replaces_contents(tmp_path):
    target = tmp_path / "out"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "old.bin").write_bytes(b"x")
    (target / "old2.bin").write_bytes(b"x")
    prepare_output_directory(target, overwrite=True)
    assert list(target.iterdir()) == []


def test_merge_checks_filesystem_before_importing_heavy_deps(tmp_path):
    """合并入口的顺序契约：先做路径/底座校验，再 import torch/peft。

    顺序有实际意义——路径写错、底座不匹配这类用户级错误，应该在开始加载权重
    （几分钟）之前就被报出来。这里用一个"不是 PEFT 目录"的路径证明它先报文件系统错。
    """
    from graspo.eval.merged_export import merge_peft_checkpoint

    with pytest.raises(ExportError, match="is not a PEFT adapter"):
        merge_peft_checkpoint(tmp_path / "nope", "/models/base", tmp_path / "out")


def test_merge_refuses_base_mismatch_before_importing_heavy_deps(tmp_path):
    """底座不匹配必须在加载权重前拒绝——否则会合出一个"看起来很正常"的错模型。"""
    from graspo.eval.merged_export import merge_peft_checkpoint

    adapter = _peft_adapter(tmp_path / "adapter", base="/models/Qwen3.5-9B")
    with pytest.raises(ExportError, match="adapter base mismatch"):
        merge_peft_checkpoint(adapter, "/models/Other-9B", tmp_path / "out")
