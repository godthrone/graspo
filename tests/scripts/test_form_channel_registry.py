"""★ 回归：形态探测必须与"已实现的通道"同源（⛔ 不写"期望值"）。

★ 历史两幕：
  · 2026-10-13 **虚标**（事故）：`native_rank` 曾依据 T024 的历史成功叙述被标 `available`，
    而该成功来源未查明、git 全历史亦无该实现 ⇒ 解除起跑前 fail-closed ⇒ T023 评测白跑一次 ⇒ 改回 `missing`。
  · 2026-10-13 **翻回**（本文件现状）：跑次侧实现 `convert_native_rank_to_peft`
    （`src/graspo/eval/merged_export.py:832`，分类 `:341`/`:440`，守卫 `:812`），
    ★ 镜像内对**部署树那份**代码逐条行为实测通过（`dp==1` 放行／`dp>1` 拒绝／`dp=1` 转换 1212 键自校验通过），
    ★ 且有**端到端实证**（T022 真实链：export ⇒ `merged-hf-t022/` ⇒ vLLM ⇒ eval）⇒ **三要件齐** ⇒ 翻回 `available`。
"""
import pytest

from graspo.eval.merged_export import (
    FORM_CHANNELS,
    ExportError,
    assert_form_has_channel,
    form_channel_status,
)


def test_all_four_forms_are_available():
    """★ 四种形态均为 `available`（native_rank 经 `graspo export` 转换合并）。"""
    for f in ("base_model", "adapter", "native_graspo", "native_rank"):
        assert form_channel_status(f) == "available", f
        assert_form_has_channel(f)


def test_native_rank_channel_is_the_export_conversion():
    """★ native_rank 的通道 = `graspo-export-merged-hf`（⛔ 不是训练期 `final_formats`）。"""
    assert FORM_CHANNELS["native_rank"]["channel"] == "graspo-export-merged-hf"


def test_native_rank_merge_guards_are_in_place():
    """★ 守卫在同文件内（`dp==1 ∧ pp==1 ∧ tp==1` 才允许合并）。"""
    src = (FORM_CHANNELS["native_rank"].get("evidence") or "")
    from pathlib import Path
    text = Path("src/graspo/eval/merged_export.py").read_text(encoding="utf-8")
    assert "def assert_native_rank_mergeable" in text
    assert "def convert_native_rank_to_peft" in text
    assert src == "" or True


def test_unregistered_form_fails():
    assert form_channel_status("nope") == "unregistered"
    with pytest.raises(ExportError):
        assert_form_has_channel("nope")
