"""★ 回归：形态探测必须与"可用通道"同源。

★ 2026-10-13 更正：`native_rank` **有通道**（经 `graspo export` 合并为 merged-hf；T024 实证），
  先前标 `missing` 是把"训练期 final_formats 未触发"误当成"无通道"。
"""
import pytest

from graspo.eval.merged_export import ExportError, assert_form_has_channel, form_channel_status


def test_available_forms_pass():
    """★ 四种形态**都有**可用通道（含 native_rank 经 graspo export）。"""
    for f in ("base_model", "adapter", "native_graspo", "native_rank"):
        assert form_channel_status(f) == "available"
        assert_form_has_channel(f)


def test_native_rank_channel_is_the_export_merge():
    """★ native_rank 的通道 = `graspo export` 合并（⛔ 不是训练期 final_formats）。"""
    from graspo.eval.merged_export import FORM_CHANNELS
    assert FORM_CHANNELS["native_rank"]["channel"] == "graspo-export-merged-hf"


def test_unregistered_form_fails():
    assert form_channel_status("nope") == "unregistered"
    with pytest.raises(ExportError):
        assert_form_has_channel("nope")
