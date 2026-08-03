"""multimodal_tensors 设施函数测试（CPU 零 GPU 依赖）。

覆盖 attach → resolve → encode → slice 链路的关键环节（旧重构计划承诺的
端到端回归）：
- rows 契约：metadata 声明 rows 键但解析为空 → RuntimeError（B3 新增防线）
- rows 解析：单行扩展、数量校验、非法 metadata 拒绝
- 张量切片：等步长切片、offset 表切片（异构样本）
- 行转换工具：messages/tools 提取与规范化
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from graspo.flow.adapters.multimodal_tensors import (  # noqa: E402
    _compute_multimodal_offset_tables,
    _messages_from_multimodal_row,
    _multimodal_rows_from_metadata,
    _processor_chat_messages,
    _slice_multimodal_inputs,
    _slice_multimodal_inputs_offset,
    _tools_from_multimodal_row,
)
from graspo.ripple.multimodal.rows import MULTIMODAL_ROWS_KEY, attach_rows  # noqa: E402

# ── rows 解析 ────────────────────────────────────────────────────────────────


def test_rows_from_metadata_single_row_expands_to_batch():
    metadata = attach_rows({}, [{"messages": [{"role": "user", "content": "hi"}]}])
    rows = _multimodal_rows_from_metadata(metadata, expected_rows=4)
    assert len(rows) == 4
    assert all(r["messages"][0]["content"] == "hi" for r in rows)


def test_rows_from_metadata_missing_key_returns_empty():
    assert _multimodal_rows_from_metadata(None, expected_rows=2) == []
    assert _multimodal_rows_from_metadata({}, expected_rows=2) == []


def test_rows_from_metadata_wrong_count_raises():
    metadata = attach_rows({}, [{"messages": []}, {"messages": []}])
    with pytest.raises(RuntimeError, match="expected 3 multimodal metadata rows"):
        _multimodal_rows_from_metadata(metadata, expected_rows=3)


def test_rows_from_metadata_list_input_aggregates():
    m1 = attach_rows({}, [{"messages": [{"role": "user", "content": "a"}]}])
    m2 = attach_rows({}, [{"messages": [{"role": "user", "content": "b"}]}])
    rows = _multimodal_rows_from_metadata([m1, m2], expected_rows=2)
    assert [r["messages"][0]["content"] for r in rows] == ["a", "b"]


# ── 行转换工具 ──────────────────────────────────────────────────────────────


def test_messages_from_multimodal_row_requires_nonempty():
    with pytest.raises(ValueError, match="non-empty messages"):
        _messages_from_multimodal_row({"messages": []})
    with pytest.raises(ValueError, match="non-empty messages"):
        _messages_from_multimodal_row({})


def test_processor_chat_messages_normalizes_text_content():
    normalized = _processor_chat_messages([{"role": "user", "content": "hello"}])
    assert normalized == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_tools_from_multimodal_row():
    row = {"tools": [{"function": {"name": "f"}}]}
    assert _tools_from_multimodal_row(row) == [{"function": {"name": "f"}}]
    assert _tools_from_multimodal_row({"tools": None}) is None
    with pytest.raises(ValueError, match="tools must be a list"):
        _tools_from_multimodal_row({"tools": "nope"})


# ── 等步长张量切片 ───────────────────────────────────────────────────────────


def _fake_inputs(images_per_row: int = 1, patches_per_row: int = 3) -> dict[str, Any]:
    rows = 2
    return {
        "image_grid_thw": torch.zeros(rows * images_per_row, 3),
        "pixel_values": torch.zeros(rows * patches_per_row, 5),
        "mm_token_type_ids": torch.zeros(rows, 4),
    }


def test_slice_multimodal_inputs_equal_stride():
    inputs = _fake_inputs(images_per_row=2, patches_per_row=4)
    sliced = _slice_multimodal_inputs(inputs, 0, 1, images_per_row=2, patches_per_row=4)
    assert sliced["image_grid_thw"].shape[0] == 2
    assert sliced["pixel_values"].shape[0] == 4
    assert sliced["mm_token_type_ids"].shape[0] == 1


def test_slice_multimodal_inputs_zero_per_row():
    inputs = _fake_inputs()
    sliced = _slice_multimodal_inputs(inputs, 0, 1, images_per_row=0, patches_per_row=0)
    # 计数为 0 → 张量键不切片（仅 mm_token_type_ids 切片）
    assert "image_grid_thw" not in sliced
    assert "pixel_values" not in sliced
    assert sliced["mm_token_type_ids"].shape[0] == 1


# ── offset 表切片（异构样本）──────────────────────────────────────────────────


def test_compute_offset_tables_expands_across_rollout_group():
    # 样本 0：1 张图（2×2=4 patches）；样本 1：2 张图（2×2 + 1×1 = 5 patches）
    # rollout_group_size=2 → 总图数 (1+2)×2=6，总 patch 数 4×2+5×2=18
    image_grid_thw = torch.tensor(
        [[3, 2, 2], [3, 2, 2], [3, 2, 2], [3, 1, 1], [3, 2, 2], [3, 1, 1]],
        dtype=torch.int64,
    )
    pixel_values = torch.zeros(18, 5)
    offsets = _compute_multimodal_offset_tables(
        per_sample_image_counts=[1, 2],
        rollout_group_size=2,
        image_grid_thw=image_grid_thw,
        pixel_values=pixel_values,
    )
    image_offsets, patch_offsets, *_ = offsets
    # 行级累计：样本 0 两行各 1 图（行2 后=2），样本 1 两行各 2 图（行4 后=6）
    assert image_offsets.tolist() == [0, 1, 2, 4, 6]
    # patch：样本 0 每行 4（2×2），样本 1 每行 5（2×2 + 1×1）
    assert patch_offsets.tolist() == [0, 4, 8, 13, 18]


def test_slice_multimodal_inputs_offset():
    image_grid_thw = torch.tensor([[3, 2, 2], [3, 1, 1]], dtype=torch.int64)
    pixel_values = torch.zeros(5, 1)
    offsets = _compute_multimodal_offset_tables(
        per_sample_image_counts=[2],
        rollout_group_size=1,
        image_grid_thw=image_grid_thw,
        pixel_values=pixel_values,
    )
    image_offsets, patch_offsets, _, _ = offsets
    inputs = {"image_grid_thw": image_grid_thw, "pixel_values": pixel_values}
    sliced = _slice_multimodal_inputs_offset(
        inputs,
        0,
        1,
        image_offsets=image_offsets,
        patch_offsets=patch_offsets,
    )
    assert sliced["image_grid_thw"].shape[0] == 2
    assert sliced["pixel_values"].shape[0] == 5


# ── B3 契约防线：声明了 rows 键但解析为空 → 硬失败 ───────────────────────────


def test_resolve_with_stale_rows_key_raises():
    """metadata 含 MULTIMODAL_ROWS_KEY 但值为空/None → 不再静默返回 None。

    这是 v13 断链（图像 token 静默按纯文本嵌入）的回归防线（B3 新增）。
    """
    stale = {MULTIMODAL_ROWS_KEY: None}
    from graspo.flow.adapters.transformer import TransformerAdapter

    class _Stub(TransformerAdapter):
        def _encode_multimodal_rows(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("encode must not run for empty rows")

    # 测试专用：运行时解除 ABC 抽象保护（仅本测试使用，绕过抽象检查后
    # 所有抽象方法仍不可达——只调用 _multimodal_inputs_from_metadata）
    _Stub.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]
    stub = _Stub.__new__(_Stub)
    with pytest.raises(RuntimeError, match="resolved to empty rows"):
        stub._multimodal_inputs_from_metadata(stale, batch_size=1)
