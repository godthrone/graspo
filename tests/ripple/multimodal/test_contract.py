"""防呆契约单测：RL 与 SFT 两条路径的"缺图即抛异常"防线。

覆盖（宪法 §11.3：做什么、什么条件下、期望什么结果）:
- contains_image_tokens: tensor/列表/None 的图像 token 检测
- assert_rl_training_has_multimodal: RL 有图无 rows → raise 等 4 个分支
- assert_sft_batch_has_multimodal: SFT 有图无 inputs → raise 等 3 个分支
- attach_rows / rows_from_metadata: 读写契约与唯一真相源
"""

from __future__ import annotations

import pytest

from graspo.ripple.multimodal.contract import (
    assert_rl_training_has_multimodal,
    assert_sft_batch_has_multimodal,
    contains_image_tokens,
)
from graspo.ripple.multimodal.rows import (
    MULTIMODAL_ROWS_KEY,
    attach_rows,
    media_counts,
    multimodal_row_from_sample,
    rows_from_metadata,
)

IMAGE_TOKEN_ID = 151655  # Qwen3.5 系列 image_token_id


# ---------------------------------------------------------------------------
# contains_image_tokens
# ---------------------------------------------------------------------------


class _FakeTensor:
    """最小 tensor 鸭子类型：仅支持 == 与 any()，用于无 torch 的纯逻辑测试。"""

    def __init__(self, values: list[int]):
        self._values = values

    def __eq__(self, other: int) -> _FakeTensor:  # type: ignore[override]
        return _FakeTensor([1 if v == other else 0 for v in self._values])

    def any(self) -> _FakeTensor:
        return _FakeTensor([1 if any(self._values) else 0])

    def item(self) -> int:
        return self._values[0]


class TestContainsImageTokens:
    def test_finds_image_token_in_tensor(self) -> None:
        seq = _FakeTensor([100, IMAGE_TOKEN_ID, 200])
        assert contains_image_tokens(seq, IMAGE_TOKEN_ID) is True

    def test_tensor_without_image_token(self) -> None:
        seq = _FakeTensor([100, 101, 102])
        assert contains_image_tokens(seq, IMAGE_TOKEN_ID) is False

    def test_finds_image_token_in_list(self) -> None:
        seq = [100, IMAGE_TOKEN_ID, 200]
        assert contains_image_tokens(seq, IMAGE_TOKEN_ID) is True

    def test_empty_list(self) -> None:
        assert contains_image_tokens([], IMAGE_TOKEN_ID) is False

    def test_none_sequences(self) -> None:
        assert contains_image_tokens(None, IMAGE_TOKEN_ID) is False


# ---------------------------------------------------------------------------
# assert_rl_training_has_multimodal
# ---------------------------------------------------------------------------


class TestRlContract:
    def test_plain_text_model_skips_check(self) -> None:
        # image_token_id 为 None = 模型不支持多模态，不检查
        assert_rl_training_has_multimodal(None, [IMAGE_TOKEN_ID], None, expected_rows=1)

    def test_no_image_token_skips_check(self) -> None:
        # sequences 纯文本，metadata 为空也放行
        assert_rl_training_has_multimodal(None, [100, 200], IMAGE_TOKEN_ID, expected_rows=1)

    def test_image_token_with_rows_passes(self) -> None:
        metadata = attach_rows({}, [{"messages": [], "media": {"image": 1}}])
        assert_rl_training_has_multimodal(
            metadata, [IMAGE_TOKEN_ID], IMAGE_TOKEN_ID, expected_rows=1
        )

    def test_image_token_without_rows_raises(self) -> None:
        # ★ 核心防线：v13 崩溃的静默路径应在此处硬失败
        with pytest.raises(RuntimeError, match="no '_multimodal_rows' rows"):
            assert_rl_training_has_multimodal(
                None, [IMAGE_TOKEN_ID], IMAGE_TOKEN_ID, expected_rows=1
            )

    def test_image_token_with_empty_metadata_dict_raises(self) -> None:
        with pytest.raises(RuntimeError):
            assert_rl_training_has_multimodal({}, [IMAGE_TOKEN_ID], IMAGE_TOKEN_ID, expected_rows=1)


# ---------------------------------------------------------------------------
# assert_sft_batch_has_multimodal
# ---------------------------------------------------------------------------


class TestSftContract:
    def test_plain_text_batch_skips_check(self) -> None:
        assert_sft_batch_has_multimodal(False, None)

    def test_media_batch_with_inputs_passes(self) -> None:
        assert_sft_batch_has_multimodal(True, {"pixel_values": "fake"})

    def test_media_batch_without_inputs_raises(self) -> None:
        # ★ SFT 防线：样本含图但 batch 无 multimodal_inputs → 硬失败
        with pytest.raises(RuntimeError, match="no multimodal_inputs"):
            assert_sft_batch_has_multimodal(True, None)


# ---------------------------------------------------------------------------
# rows 读写契约
# ---------------------------------------------------------------------------


class TestRowsContract:
    def test_attach_then_read_roundtrip(self) -> None:
        rows = [{"messages": [], "media": {"image": 1}}]
        metadata = attach_rows({}, rows)
        assert metadata[MULTIMODAL_ROWS_KEY] == rows
        assert rows_from_metadata(metadata, expected_rows=1) == rows

    def test_attach_returns_same_metadata(self) -> None:
        metadata: dict = {}
        result = attach_rows(metadata, [{"messages": []}])
        assert result is metadata  # 就地修改

    def test_attach_copies_rows(self) -> None:
        rows = [{"messages": [], "media": {"image": 1}}]
        metadata = attach_rows({}, rows)
        rows[0]["messages"].append("mutated")
        assert metadata[MULTIMODAL_ROWS_KEY][0]["messages"] == []

    def test_read_missing_key_returns_empty(self) -> None:
        assert rows_from_metadata({}, expected_rows=1) == []
        assert rows_from_metadata(None, expected_rows=1) == []

    def test_read_single_row_expands_to_expected(self) -> None:
        metadata = attach_rows({}, [{"messages": []}])
        rows = rows_from_metadata(metadata, expected_rows=4)
        assert len(rows) == 4

    def test_read_wrong_count_raises(self) -> None:
        metadata = attach_rows({}, [{"messages": []}, {"messages": []}])
        with pytest.raises(RuntimeError, match="expected 1.*got 2"):
            rows_from_metadata(metadata, expected_rows=1)

    def test_read_list_of_metadata(self) -> None:
        # experience metadata 是 list[dict]，每项一行
        metadata = [attach_rows({}, [{"messages": []}]), attach_rows({}, [{"messages": []}])]
        rows = rows_from_metadata(metadata, expected_rows=2)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# rows 构建
# ---------------------------------------------------------------------------


class TestRowsBuild:
    def test_media_counts(self) -> None:
        media = [{"type": "image"}, {"type": "image"}, {"type": "video"}]
        assert media_counts(media) == {"image": 2, "video": 1}

    def test_media_counts_empty(self) -> None:
        assert media_counts([]) == {}

    def test_media_counts_unknown_type(self) -> None:
        assert media_counts([{"foo": "bar"}]) == {"unknown": 1}

    def test_row_from_sample_preserves_messages_and_media(self) -> None:
        sample = _Sample(
            messages=[{"role": "user", "content": [{"type": "image", "image": "a.jpg"}]}],
            media=[{"type": "image"}],
        )
        row = multimodal_row_from_sample(sample)
        assert row["media"] == {"image": 1}
        assert row["messages"][0]["role"] == "user"

    def test_row_from_sample_resolves_relative_path(self) -> None:
        sample = _Sample(
            messages=[{"role": "user", "content": [{"type": "image", "image": "a.jpg"}]}],
            media=[{"type": "image"}],
        )
        row = multimodal_row_from_sample(sample, data_dir="/data/v13_fk_scenes")
        image = row["messages"][0]["content"][0]["image"]
        assert image.startswith("/data/v13_fk_scenes/")

    def test_row_from_sample_keeps_absolute_path(self) -> None:
        sample = _Sample(
            messages=[{"role": "user", "content": [{"type": "image", "image": "/abs/a.jpg"}]}],
            media=[{"type": "image"}],
        )
        row = multimodal_row_from_sample(sample, data_dir="/data/v13_fk_scenes")
        assert row["messages"][0]["content"][0]["image"] == "/abs/a.jpg"

    def test_row_from_sample_tools(self) -> None:
        sample = _Sample(
            messages=[{"role": "user", "content": "hi"}],
            media=[],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        row = multimodal_row_from_sample(sample)
        assert row["tools"] == [{"type": "function", "function": {"name": "f"}}]

    def test_attach_rejects_non_dict_metadata(self) -> None:
        with pytest.raises(TypeError, match="metadata must be a dict"):
            attach_rows([], [])  # type: ignore[arg-type]

    def test_attach_rejects_non_list_rows(self) -> None:
        with pytest.raises(TypeError, match="rows must be a list"):
            attach_rows({}, {})  # type: ignore[arg-type]


class _Sample:
    """最小 Sample 鸭子类型：仅暴露 multimodal_row_from_sample 需要的字段。"""

    def __init__(
        self,
        *,
        messages: list[dict],
        media: list[dict],
        tools: list[dict] | None = None,
    ) -> None:
        self.messages = messages
        self.media = media
        self.tools = tools
