"""★ 回归：`classify_checkpoint` 对 native `.pt` 分片（manifest.json + rank_*.pt）的识别。

背景（2026-10-13 事故）：T023/T022 的 `final/` 只有 `manifest.json` + `rank_*.pt`，
而 `classify_checkpoint` 原先只认 `config.json`/`adapter_config.json`/`*.safetensors`
⇒ `.pt` 分片落到 `raise ExportError` ⇒ 评测链的 `graspo export` 步骤必失败。
"""
import pytest

# ★ 本文件的转换类用例需要 torch（本地环境通常无 torch）⇒ 跳过而非红；镜像内会自动真跑。
torch = pytest.importorskip("torch", reason="native rank 转换用例需 torch（请在镜像内跑）")

from graspo.eval.merged_export import (
    CheckpointKind,
    ExportError,
    assert_native_rank_mergeable,
    classify_checkpoint,
    convert_native_rank_to_peft,
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


def _fake_shard(tmp_path, manifest: dict, n: int = 2):
    import json
    import torch
    d = tmp_path / "final"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lora = {}
    for i in range(n):
        lora[f"layers.{i}.mlp.down_proj.lora_a"] = torch.zeros(4, 8)
        lora[f"layers.{i}.mlp.down_proj.lora_b"] = torch.zeros(8, 4)
    torch.save({"lora_state_dict": lora}, d / "rank_00000_tp_00_pp_00.pt")
    return d


def test_convert_native_rank_dp1_produces_peft_adapter(tmp_path):
    """★ dp==1 ⇒ 可转；产出 adapter_config.json + adapter_model.safetensors，键数守恒。"""
    src = _fake_shard(tmp_path, {"dp_size": 1, "pp_size": 1, "tp_size": 1})
    out = tmp_path / "adapter"
    info = convert_native_rank_to_peft(src, out, base_model_path="/models/base")
    assert info["keys"] == 4
    assert (out / "adapter_model.safetensors").is_file()
    assert (out / "adapter_config.json").is_file()


def test_convert_native_rank_rejects_dp_gt_1(tmp_path):
    """★ dp>1 ⇒ 必拒（不得用分歧权重合出错误模型）。"""
    src = _fake_shard(tmp_path, {"dp_size": 2, "pp_size": 1, "tp_size": 1})
    with pytest.raises(ExportError):
        convert_native_rank_to_peft(src, tmp_path / "adapter", base_model_path="/models/base")


def test_convert_native_rank_rejects_unmapped_key(tmp_path):
    """★ 出现规则外的键 ⇒ 必拒（不静默丢键）。"""
    src = _fake_shard(tmp_path, {"dp_size": 1, "pp_size": 1, "tp_size": 1})
    import torch
    torch.save({"lora_state_dict": {"weird_key": torch.zeros(2, 2)}},
               src / "rank_00000_tp_00_pp_00.pt")
    with pytest.raises(ExportError):
        convert_native_rank_to_peft(src, tmp_path / "adapter", base_model_path="/models/base")
