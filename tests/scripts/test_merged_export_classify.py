"""★ 回归：`classify_checkpoint` 对 native `.pt` 分片（manifest.json + rank_*.pt）的识别。

背景（2026-10-13 事故）：T023/T022 的 `final/` 只有 `manifest.json` + `rank_*.pt`，
而 `classify_checkpoint` 原先只认 `config.json`/`adapter_config.json`/`*.safetensors`
⇒ `.pt` 分片落到 `raise ExportError` ⇒ 评测链的 `graspo export` 步骤必失败。
"""
import pytest

from graspo.eval.merged_export import (
    CheckpointKind,
    ExportError,
    assert_native_rank_mergeable,
    classify_checkpoint,
)


# ---------- ★ native .pt 分片（manifest.json + rank_*.pt） ----------
def test_classify_native_rank_pt_shards(tmp_path):
    """★ 回归（T023/T022 事故）：manifest.json + rank_*.pt 必须可分类。"""
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "rank_00000_tp_00_pp_00.pt").write_bytes(b"x")
    c = classify_checkpoint(tmp_path)
    assert c.kind == CheckpointKind.NATIVE_RANK


def test_classify_native_rank_requires_manifest(tmp_path):
    """★ 只有 rank_*.pt、缺 manifest.json ⇒ 仍必拒（fail-closed）。"""
    (tmp_path / "rank_00000_tp_00_pp_00.pt").write_bytes(b"x")
    with pytest.raises(ExportError):
        classify_checkpoint(tmp_path)


def test_classify_safetensors_branch_unaffected(tmp_path):
    """★ 回归：*.safetensors 分支不受新分支影响。"""
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x")
    assert classify_checkpoint(tmp_path).kind == CheckpointKind.GRASPO_NATIVE


def test_assert_native_rank_mergeable_rejects_dp_gt_1():
    """★ 回归：dp_size>1 ⇒ 必拒（lora_a 分歧、无唯一正解）。"""
    problems = assert_native_rank_mergeable({"dp_size": 2, "pp_size": 1, "tp_size": 1})
    assert problems and "lora_a" in problems[0]


def test_assert_native_rank_mergeable_rejects_pp_gt_1():
    problems = assert_native_rank_mergeable({"dp_size": 1, "pp_size": 2, "tp_size": 1})
    assert problems and "PP 分片" in problems[0]


def test_assert_native_rank_mergeable_allows_dp1_pp1():
    assert assert_native_rank_mergeable({"dp_size": 1, "pp_size": 1, "tp_size": 1}) == []
