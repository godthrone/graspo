"""★ 回归：形态探测必须与"已实现的通道"同源（⛔ 不写"期望值"）。

★ 事故（2026-10-13）：`native_rank` 曾被**虚标**为 `available`（依据是 T024 的历史成功叙述，
  而该成功来源未查明、git 全历史亦无该实现）⇒ 解除了起跑前 fail-closed ⇒ T023 评测白跑一次。
"""
import pytest

from graspo.eval.merged_export import (
    FORM_CHANNELS,
    ExportError,
    assert_form_has_channel,
    form_channel_status,
)


def test_available_forms_are_only_the_implemented_ones():
    """★ 只有导出器**已支持**的形态才是 `available`。"""
    for f in ("base_model", "adapter", "native_graspo"):
        assert form_channel_status(f) == "available", f
        assert_form_has_channel(f)


def test_native_rank_is_missing_and_refused_before_launch():
    """★★ `native_rank` 当前**未实现**（导出器不认）⇒ 登记 = `missing` ⇒ 起跑前**必须拒**。"""
    assert form_channel_status("native_rank") == "missing"
    assert FORM_CHANNELS["native_rank"]["channel"] == "rank-shards-to-hf"
    with pytest.raises(ExportError):
        assert_form_has_channel("native_rank")


def test_unregistered_form_fails():
    assert form_channel_status("nope") == "unregistered"
    with pytest.raises(ExportError):
        assert_form_has_channel("nope")
