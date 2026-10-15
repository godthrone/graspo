"""输出目录防呆（V11/V12）：非空拒绝、预授权后逐项删除、落点重叠拒绝。"""

from __future__ import annotations

import pytest
from helpers import make_export_config, write_native_checkpoint

from graspo.export import export_checkpoint
from graspo.export.errors import ExportError, OutputDirNotEmptyError
from graspo.export.paths import assert_destination_allowed, prepare_output_dir


def test_absent_directory_is_created(tmp_path):
    target = tmp_path / "new" / "nested"
    assert prepare_output_dir(target) == target
    assert target.is_dir()


def test_empty_directory_is_reused(tmp_path):
    target = tmp_path / "empty"
    target.mkdir()
    assert prepare_output_dir(target) == target


def test_nonempty_without_optin_is_refused_and_nothing_is_deleted(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.bin").write_bytes(b"x")
    with pytest.raises(OutputDirNotEmptyError) as excinfo:
        prepare_output_dir(target, overwrite=False)
    # 两个既有调用方的契约都保留：训练侧按 FileExistsError 分派，导出侧按 ExportError。
    assert isinstance(excinfo.value, FileExistsError)
    assert isinstance(excinfo.value, ExportError)
    assert "not empty" in str(excinfo.value)
    assert "overwrite_output_dir" in str(excinfo.value)
    assert (target / "keep.bin").exists()


def test_optin_deletes_listed_entries_only(tmp_path):
    """阳性对照（V11）：预授权后逐项删除，⛔ 不动目录外的任何文件。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sibling.txt").write_text("untouched", encoding="utf-8")

    target = tmp_path / "out"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "old.bin").write_bytes(b"x")
    (target / "old2.bin").write_bytes(b"x")

    assert prepare_output_dir(target, overwrite=True) == target
    assert target.is_dir()
    assert list(target.iterdir()) == []
    assert (outside / "sibling.txt").exists()


def test_destination_equal_to_checkpoint_is_refused(tmp_path):
    with pytest.raises(ExportError, match="must not be the checkpoint directory itself"):
        assert_destination_allowed(tmp_path, checkpoint_dir=tmp_path)


def test_destination_inside_base_is_refused(tmp_path):
    base = tmp_path / "base"
    with pytest.raises(ExportError, match="base model directory nor inside it"):
        assert_destination_allowed(
            base / "merged", checkpoint_dir=tmp_path / "ckpt", base_model_path=base
        )


def test_destination_containing_base_is_refused(tmp_path):
    base = tmp_path / "out" / "base"
    with pytest.raises(ExportError, match="must not contain the base model directory"):
        assert_destination_allowed(
            tmp_path / "out", checkpoint_dir=tmp_path / "ckpt", base_model_path=base
        )


def test_destination_inside_checkpoint_is_allowed(tmp_path):
    """训练结束的自动导出落点就是 ``<checkpoint_dir>/<format>`` —— 必须放行。"""
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    assert_destination_allowed(checkpoint / "merged-hf", checkpoint_dir=checkpoint)


def test_export_refuses_destructive_destination_before_writing(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    config = make_export_config(checkpoint, checkpoint, fmt="peft-adapter")
    with pytest.raises(ExportError, match="must not be the checkpoint directory itself"):
        export_checkpoint(config, base_model_path="models/base")
    assert (checkpoint / "manifest.json").exists()
    assert not (checkpoint / "export_manifest.json").exists()


def test_export_refuses_nonempty_output_without_optin(tmp_path):
    checkpoint = write_native_checkpoint(tmp_path / "final")
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.bin").write_bytes(b"x")
    config = make_export_config(checkpoint, output, fmt="peft-adapter")
    with pytest.raises(OutputDirNotEmptyError):
        export_checkpoint(config, base_model_path="models/base")
    assert (output / "existing.bin").exists()
    assert not (output / "export_manifest.json").exists()
