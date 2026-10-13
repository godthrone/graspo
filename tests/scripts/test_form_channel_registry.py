"""★ 回归：形态探测必须与"可用通道"同源（2026-10-13 native final 缺口）。"""
import pytest

from graspo.eval.merged_export import ExportError, assert_form_has_channel, form_channel_status


def test_available_forms_pass():
    for f in ("base_model", "adapter", "native_graspo"):
        assert_form_has_channel(f)


def test_native_rank_has_no_channel_and_must_fail():
    """★ 负例（本次事故形态）：form=native_rank 当前**无渠道** ⇒ 必须 fail-closed。"""
    assert form_channel_status("native_rank") == "missing"
    with pytest.raises(ExportError):
        assert_form_has_channel("native_rank")


def test_unregistered_form_fails():
    assert form_channel_status("nope") == "unregistered"
    with pytest.raises(ExportError):
        assert_form_has_channel("nope")
