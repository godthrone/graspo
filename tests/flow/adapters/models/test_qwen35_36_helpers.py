"""qwen35_36/helpers.py 纯函数的单元测试：SFT batch collation（CPU 可跑）。"""

import pytest
import torch

from graspo.flow.adapters.models.qwen35_36.helpers import (
    collate_sft_batch,
    collate_sft_multimodal_batch,
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


class _FakeAdapter:
    """最小 mock：只有 collate_sft_multimodal_batch 用到的接口。"""

    IMAGE_TOKEN_ID = 151655

    def __init__(self, prompt_len: int, n_placeholders: int) -> None:
        self._prompt_len = prompt_len
        self._n_placeholders = n_placeholders
        self.model = type("M", (), {"config": type("C", (), {"image_token_id": self.IMAGE_TOKEN_ID})})()
        self.tokenizer = type(
            "T",
            (),
            {"eos_token_id": 151645, "encode": lambda self, text, add_special_tokens=False: [10, 11]},
        )()
        self.config = type("CFG", (), {"model": type("MC", (), {"chat_template_kwargs": None})})()

    def _encode_multimodal_rows(self, rows, *, add_generation_prompt, chat_template_kwargs):
        # 模拟 Qwen processor 输出：input_ids 长度 = prompt_len，尾部放 n_placeholders 个占位符
        n = len(rows)
        ids = torch.zeros(n, self._prompt_len, dtype=torch.long)
        ids[:, -self._n_placeholders :] = self.IMAGE_TOKEN_ID
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(n, self._prompt_len, dtype=torch.bool),
            "pixel_values": torch.zeros(n, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, self._n_placeholders, 1]] * n, dtype=torch.long),
        }

    def _multimodal_inputs_to_device(self, encoded):
        return encoded


def _deferred_item(target_text: str = "<tool_call>ok</tool_call>") -> SFTTokenized:
    return SFTTokenized(
        metadata={},
        deferred_multimodal=type(
            "D", (), {"prompt_messages": [{"role": "user", "content": [{"type": "image", "image": "/tmp/x.jpg"}]}], "tools": None, "target_text": target_text}
        )(),
    )


def test_collate_sft_multimodal_raises_when_placeholders_truncated():
    """防呆：max_seq_length 截断视觉占位符 → 提前 ValueError，带修复建议（宪法 §2）。"""
    adapter = _FakeAdapter(prompt_len=100, n_placeholders=80)
    item = _deferred_item()
    with pytest.raises(ValueError, match="max_prompt_length 截断了视觉占位符"):
        collate_sft_multimodal_batch([item], torch.device("cpu"), adapter=adapter, max_seq_length=60)


def test_collate_sft_multimodal_ok_when_within_limit():
    """占位符完整保留时不报错。"""
    adapter = _FakeAdapter(prompt_len=100, n_placeholders=80)
    item = _deferred_item()
    batch = collate_sft_multimodal_batch([item], torch.device("cpu"), adapter=adapter, max_seq_length=120)
    assert "input_ids" in batch
    assert "multimodal_inputs" in batch
