"""停止判据的单 id / 多 id 两情形（RC-1 回归用例，纯 CPU）。

判据的**唯一来源**是 ``helpers.resolve_stop_token_ids``，判据的**唯一实现**是
``helpers.apply_stop_mask``（§1.4 单一真相源）。本文件同时覆盖三个方向：

- **不停得太晚**（= 修 RC-1 的目标）：chat 回合结束符 ``<|im_end|>`` 必须命中，
  即使它的 id 与 ``tokenizer.eos_token_id`` 不同；
- **不停得太早**（修 RC-1 的负向约束）：``pad_token``（``<|endoftext|>``）与
  普通 token 绝不能命中；词表里没有 ``<|im_end|>`` 时不得把 unk/0 当结束符；
- **单一来源**：两种形态（int / list）都从同一个函数解析。
"""

from __future__ import annotations

import pytest
import torch

from graspo.flow.adapters.models.qwen35_36.helpers import (
    CHAT_TURN_END_TOKEN,
    apply_stop_mask,
    resolve_stop_token_ids,
    rollout_chat_template_kwargs,
    stop_ids_tensor,
)


class _FakeTokenizer:
    """最小 tokenizer 替身：只暴露判据用到的三个接口。"""

    def __init__(
        self,
        *,
        eos_token_id: int | list[int] | None,
        im_end_id: int | None,
        unk_id: int = 0,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = 99  # pad 与 eos 语义分开：判据不得把它当结束符
        self._im_end_id = im_end_id
        self._unk_id = unk_id
        self._vocab = {im_end_id: CHAT_TURN_END_TOKEN} if im_end_id is not None else {}

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == CHAT_TURN_END_TOKEN:
            if self._im_end_id is None:
                return self._unk_id  # 词表里没有 ⇒ 返回 unk，不得被当作真 token
            return self._im_end_id
        raise KeyError(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._vocab.get(token_id, f"<unk_{token_id}>")


# ── resolve_stop_token_ids：单 id / 多 id / 缺失 / fail-closed ────────────────


def test_single_eos_equal_to_im_end_is_deduped():
    """Qwen3.5-9B 的真实形态：eos_token='<|im_end|>' ⇒ 两个来源同 id，去重成 1 个。"""
    tok = _FakeTokenizer(eos_token_id=248046, im_end_id=248046)
    assert resolve_stop_token_ids(tok) == [248046]


def test_eos_is_a_list_is_accepted():
    """多结束符（HF 允许 list）⇒ 全部保留，且追加 chat 结束符。"""
    tok = _FakeTokenizer(eos_token_id=[151643, 248046], im_end_id=248046)
    assert resolve_stop_token_ids(tok) == [151643, 248046]


def test_eos_distinct_from_im_end_both_accepted():
    """eos 与 chat 结束符不是同一个 id ⇒ 两个都停（这正是"停不下来"的正解）。"""
    tok = _FakeTokenizer(eos_token_id=151643, im_end_id=248046)
    assert resolve_stop_token_ids(tok) == [151643, 248046]


def test_missing_im_end_in_vocab_does_not_inject_unk():
    """词表里没有 <|im_end|> ⇒ 绝不得把 unk/0 当结束符（那会"停得太早"）。"""
    tok = _FakeTokenizer(eos_token_id=151643, im_end_id=None, unk_id=0)
    assert resolve_stop_token_ids(tok) == [151643]


def test_eos_none_falls_back_to_chat_end_token():
    tok = _FakeTokenizer(eos_token_id=None, im_end_id=248046)
    assert resolve_stop_token_ids(tok) == [248046]


def test_unresolvable_stop_ids_fail_closed():
    """解析不出任何停止 token ⇒ fail-closed（绝不静默退化成"永不停止"）。"""
    tok = _FakeTokenizer(eos_token_id=None, im_end_id=None)
    with pytest.raises(ValueError, match="无法解析停止 token"):
        resolve_stop_token_ids(tok)


def test_stop_ids_tensor_is_1d_long():
    tok = _FakeTokenizer(eos_token_id=248046, im_end_id=248046)
    tensor = stop_ids_tensor(tok, "cpu")
    assert tensor.dtype == torch.long
    assert tuple(tensor.shape) == (1,)
    assert tensor.tolist() == [248046]


# ── apply_stop_mask：正确停 / 不早停 / 不晚停 ─────────────────────────────────


def test_apply_stop_mask_single_id_stops_only_on_that_id():
    finished = torch.zeros(3, dtype=torch.bool)
    stop_ids = torch.tensor([248046])
    # 第 1 行命中；第 0 行是 pad（99，不得停）；第 2 行是普通 token（不得停）
    next_token = torch.tensor([99, 248046, 12345])
    out = apply_stop_mask(finished, next_token, stop_ids)
    assert out.tolist() == [False, True, False]


def test_apply_stop_mask_multi_id_stops_on_any():
    finished = torch.zeros(3, dtype=torch.bool)
    next_token = torch.tensor([151643, 248046, 42])
    out = apply_stop_mask(finished, next_token, [151643, 248046])
    assert out.tolist() == [True, True, False]


def test_apply_stop_mask_accepts_int_list_and_tensor_forms():
    next_token = torch.tensor([7, 8])
    base = torch.zeros(2, dtype=torch.bool)
    assert apply_stop_mask(base, next_token, 7).tolist() == [True, False]
    assert apply_stop_mask(base, next_token, [8]).tolist() == [False, True]
    assert apply_stop_mask(base, next_token, torch.tensor([7, 8])).tolist() == [True, True]


def test_apply_stop_mask_keeps_previously_finished_rows():
    """已结束的行必须保持结束（判据是单调的，不会被后续 token 抹掉）。"""
    finished = torch.tensor([True, False])
    out = apply_stop_mask(finished, torch.tensor([1, 2]), torch.tensor([248046]))
    assert out.tolist() == [True, False]


def test_apply_stop_mask_does_not_mutate_input():
    finished = torch.zeros(1, dtype=torch.bool)
    apply_stop_mask(finished, torch.tensor([248046]), torch.tensor([248046]))
    assert finished.tolist() == [False]


# ── rollout_chat_template_kwargs：rollout 与 SFT 的口径一致（RC-1 真因）────────


def test_rollout_chat_template_kwargs_disables_thinking_by_default():
    assert rollout_chat_template_kwargs(None) == {"enable_thinking": False}
    assert rollout_chat_template_kwargs({}) == {"enable_thinking": False}


def test_rollout_chat_template_kwargs_keeps_explicit_user_value():
    """透明退路 §3.2：配置显式给值时不静默覆盖。"""
    assert rollout_chat_template_kwargs({"enable_thinking": True}) == {"enable_thinking": True}


def test_rollout_chat_template_kwargs_preserves_other_keys():
    kwargs = rollout_chat_template_kwargs({"tools": [], "some_key": 1})
    assert kwargs == {"tools": [], "some_key": 1, "enable_thinking": False}
