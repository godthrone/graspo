"""qwen35_36/helpers.py 纯函数的单元测试：SFT batch collation（CPU 可跑）。"""

import pytest
import torch

from graspo.flow.adapters.models.qwen35_36.helpers import (
    collate_sft_batch,
    collate_sft_text_batch,
    count_media_types_from_messages,
)
from graspo.ripple.data import SFTTokenized


def _text_tokenized(*, prompt_len: int, response_len: int) -> SFTTokenized:
    return SFTTokenized(
        metadata={},
        input_ids=torch.arange(prompt_len + response_len, dtype=torch.long),
        labels=torch.arange(prompt_len + response_len, dtype=torch.long),
        attention_mask=torch.ones(prompt_len + response_len, dtype=torch.bool),
        prompt_len=prompt_len,
    )


def test_collate_sft_text_batch_pads_to_longest():
    items = [
        _text_tokenized(prompt_len=2, response_len=3),
        _text_tokenized(prompt_len=2, response_len=5),
    ]
    batch = collate_sft_text_batch(items, torch.device("cpu"))

    assert batch["input_ids"].shape == (2, 7)  # 最长 2+5，短的补 0
    assert batch["labels"].shape == (2, 7)
    assert batch["attention_mask"].shape == (2, 7)
    assert batch["labels"][1, 6] == 6  # 长样本的最后一个 token 保留


def test_collate_sft_batch_dispatches_text_path():
    items = [_text_tokenized(prompt_len=1, response_len=1)]
    batch = collate_sft_batch(items, torch.device("cpu"), adapter=None)

    assert "multimodal_inputs" not in batch


def test_collate_sft_batch_rejects_mixed_multimodal_and_text():
    text_item = _text_tokenized(prompt_len=1, response_len=1)
    deferred_item = SFTTokenized(
        metadata={},
        deferred_multimodal=type(
            "D", (), {"prompt_messages": [], "tools": None, "target_text": "x"}
        )(),
    )
    import pytest

    with pytest.raises(ValueError, match="Cannot mix"):
        collate_sft_batch([text_item, deferred_item], torch.device("cpu"), adapter=None)


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([], {}),
        ([{"role": "user", "content": "text"}], {}),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "a.jpg"},
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": "b.jpg"}},
                    ],
                }
            ],
            {"image": 2},
        ),
        (
            [{"role": "user", "content": [{"type": "video", "video": "v.mp4"}]}],
            {"video": 1},
        ),
        ([{"role": "user", "content": [{"type": "audio", "audio": "a.mp3"}]}], {}),
    ],
)
def test_count_media_types_from_messages(messages, expected):
    assert count_media_types_from_messages(messages) == expected


def test_collate_sft_text_batch_metadata_preserved():
    items = [
        SFTTokenized(
            metadata={"idx": 0},
            input_ids=torch.tensor([1, 2], dtype=torch.long),
            labels=torch.tensor([1, 2], dtype=torch.long),
            attention_mask=torch.tensor([True, True]),
        )
    ]
    batch = collate_sft_text_batch(items, torch.device("cpu"))
    assert batch["metadata"] == [{"idx": 0}]
