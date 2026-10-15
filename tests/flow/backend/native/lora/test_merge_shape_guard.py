"""B12：合并前的**显式切片校验** —— 防"基座与 adapter 不同源"被静默错并。

实测背景（2026-10-12，T022 = 27B LoRA native SFT）：`export --format merged-hf` 拿到
与 adapter **不同源**的基座时，旧实现对 ``out``/``rows``/``in``/``cols`` **直接切片相加、
不校验上界**：

- 基座**窄于** delta 期望（实测 MLP 中间维 12288 < 17408）⇒ 切片被截断，靠广播偶然抛
  ``RuntimeError: The size of tensor a (12288) must match the size of tensor b (17408)``，
  报错点远离真因；
- 基座**宽于** delta 期望 ⇒ **静默写到错误位置**（产物表面成功、内容错位）。

本文件钉住：两种形态都**显式 fail-closed**，且错误消息点名基座名与两侧形状。
纯 CPU、零 GPU、零 ms-swift。

★ 迁移登记：``_apply_deltas`` 随导出子集搬入 ``graspo.export.exporters.native_lora``
（设计 §7.1）；异常类型同步收敛为公开契约
:class:`graspo.export.errors.ExportError`（§2.4 单一可捕获类型）。消息文本逐字未变。
"""

from __future__ import annotations

import pytest
import torch

from graspo.export.errors import ExportError
from graspo.export.exporters.native_lora import _apply_deltas


def _rec(kind: str, **kw) -> dict:
    base = {"shard_kind": kind, "base_weight_name": "model.layers.0.mlp.down_proj.weight"}
    base.update(kw)
    return base


# ── ① 正例：形状自洽时行为不变 ──────────────────────────────────────────────


def test_full_tensor_merge_is_unchanged():
    base = torch.zeros(4, 6)
    delta = torch.ones(4, 6)
    out = _apply_deltas(base, [(_rec("none"), delta)])
    assert torch.equal(out, torch.ones(4, 6))


def test_in_kind_merges_only_its_column_slice():
    """kind="in"：只动 [col_start:col_stop] 那几列，其余列逐字不变。"""
    base = torch.zeros(3, 10)
    delta = torch.full((3, 4), 2.0)
    out = _apply_deltas(base, [(_rec("in", col_start=2, col_stop=6), delta)])
    assert torch.equal(out[:, 2:6], torch.full((3, 4), 2.0))
    assert torch.equal(out[:, :2], torch.zeros(3, 2))
    assert torch.equal(out[:, 6:], torch.zeros(3, 4))


def test_out_and_rows_kinds_merge_the_right_rows():
    base = torch.zeros(10, 3)
    out = _apply_deltas(base, [(_rec("out", row_start=1, row_stop=5), torch.full((4, 3), 3.0))])
    assert torch.equal(out[1:5], torch.full((4, 3), 3.0))
    assert torch.equal(out[0], torch.zeros(3))

    base2 = torch.zeros(10, 3)
    idx = [0, 7, 9]
    out2 = _apply_deltas(base2, [(_rec("rows", row_indices=idx), torch.full((3, 3), 5.0))])
    for i in idx:
        assert torch.equal(out2[i], torch.full((3,), 5.0))
    assert torch.equal(out2[1], torch.zeros(3))


# ── ② 负例：基座窄于 delta 期望（实测 T022 的形态）────────────────────────────


def test_base_narrower_than_the_delta_raises_actionable_error():
    """★T022 复现：基座列宽 12288 < 切片终点 17408 ⇒ **显式 ExportError**（不是广播 RuntimeError）。"""
    base = torch.zeros(2, 12288)  # 9B 的 MLP 中间维
    delta = torch.ones(2, 17408)  # 27B 的 MLP 中间维
    with pytest.raises(ExportError) as exc:
        _apply_deltas(base, [(_rec("in", col_start=0, col_stop=17408), delta)])
    msg = str(exc.value)
    assert "out of range" in msg
    assert "12288" in msg and "17408" in msg
    assert "down_proj" in msg  # 点名基座张量
    assert "does not match" in msg


# ── ③ 负例：基座**宽于** delta 期望（旧实现会静默错并）────────────────────────


def test_base_wider_than_the_delta_raises_instead_of_silently_mis_merging():
    """★最危险形态：旧实现会静默把 delta 写到错误位置；现在必须显式拒绝。"""
    base = torch.zeros(2, 20000)
    delta = torch.ones(2, 16384)  # 切片宽 17408 ≠ delta 16384
    with pytest.raises(ExportError) as exc:
        _apply_deltas(base, [(_rec("in", col_start=0, col_stop=17408), delta)])
    msg = str(exc.value)
    assert "does not match the delta" in msg
    assert "17408" in msg and "16384" in msg


def test_out_kind_slice_beyond_base_rows_is_rejected():
    base = torch.zeros(8, 3)
    with pytest.raises(ExportError, match="out of range"):
        _apply_deltas(base, [(_rec("out", row_start=0, row_stop=16), torch.ones(16, 3))])


def test_row_indices_beyond_base_rows_is_rejected():
    base = torch.zeros(4, 3)
    with pytest.raises(ExportError, match="exceeds base tensor"):
        _apply_deltas(base, [(_rec("rows", row_indices=[0, 9]), torch.ones(2, 3))])


def test_row_indices_count_mismatch_is_rejected():
    base = torch.zeros(8, 3)
    with pytest.raises(ExportError, match="row_indices count"):
        _apply_deltas(base, [(_rec("rows", row_indices=[0, 1, 2]), torch.ones(2, 3))])


def test_unsupported_kind_is_still_rejected():
    with pytest.raises(ExportError, match="Unsupported LoRA shard kind"):
        _apply_deltas(torch.zeros(2, 2), [(_rec("bogus"), torch.zeros(2, 2))])
