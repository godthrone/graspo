"""GRASPORippleLoss（PPO-clip）与共享 log-prob 实现的单元测试。"""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from torch.nn import functional as F  # noqa: E402

from graspo.ripple.loss import (  # noqa: E402
    GRASPORippleLoss,
    masked_mean,
    masked_token_log_probs_from_hidden,
)


def test_ppo_clip_loss_matches_original_formula():
    log_probs = torch.tensor([[0.0, 0.2, -0.1], [0.1, -0.3, 0.4]])
    old_log_probs = torch.tensor([[-0.1, 0.1, -0.1], [0.0, -0.1, 0.2]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-0.5, -0.5, -0.5]])
    action_mask = torch.tensor([[True, True, False], [True, False, True]])

    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(0.8, 1.2) * advantages
    expected = -torch.min(surr1, surr2)
    expected = ((expected * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)).mean()

    actual = GRASPORippleLoss(policy_ratio_clip_eps=0.2)(
        log_probs, old_log_probs, advantages, action_mask
    )

    assert torch.allclose(actual, expected)


def _random_case(batch, seq, hidden_dim, vocab, seed=0):
    """构造随机 hidden/lm_head/labels，vocab 取跨 chunk 边界的值（如 70000 > 32768×2）。"""
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, hidden_dim, generator=gen)
    lm_head = torch.randn(vocab, hidden_dim, generator=gen) * 0.02
    labels = torch.randint(0, vocab, (batch, seq), generator=gen)
    labels[:, : seq // 2] = -100  # 一半位置 mask 掉，模拟 SFT 的 prompt 部分
    return hidden, lm_head, labels


def _full_log_probs_reference(hidden, lm_head, labels, ignore_index=-100):
    """完整 logits 的参考实现（对拍基准）：log_softmax 后 gather。"""
    logits = F.linear(hidden.float(), lm_head.float())
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return log_probs.masked_fill(labels == ignore_index, 0.0)


def test_masked_token_log_probs_matches_full_logits():
    """对拍：分块实现与完整 logits 参考实现数值一致（v0.24.0 共享 loss 防回归测试）。"""
    hidden, lm_head, labels = _random_case(batch=2, seq=8, hidden_dim=32, vocab=70000)
    expected = _full_log_probs_reference(hidden, lm_head, labels)
    actual = masked_token_log_probs_from_hidden(hidden, lm_head, labels, ignore_index=-100)
    assert actual.shape == expected.shape == (2, 8)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_masked_token_log_probs_without_ignore_matches_full():
    """RL 场景：token_ids 无 -100（全有效）时结果与完整 logits 一致。"""
    hidden, lm_head, labels = _random_case(batch=2, seq=8, hidden_dim=32, vocab=70000)
    valid_ids = labels.clamp(min=0)
    expected = F.log_softmax(F.linear(hidden.float(), lm_head.float()), -1).gather(
        -1, valid_ids.unsqueeze(-1)
    ).squeeze(-1)
    actual = masked_token_log_probs_from_hidden(hidden, lm_head, valid_ids, ignore_index=-100)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_sft_loss_equals_cross_entropy_reference():
    """SFT loss（共享实现 + 全局均值）与 F.cross_entropy(ignore_index) 参考语义一致。"""
    hidden, lm_head, labels = _random_case(batch=2, seq=8, hidden_dim=32, vocab=70000)
    shift_hidden = hidden[:, :-1]
    shift_labels = labels[:, 1:]

    log_probs = masked_token_log_probs_from_hidden(
        shift_hidden.float(), lm_head.float(), shift_labels, ignore_index=-100
    )
    mask = shift_labels != -100
    actual = -(log_probs * mask).sum() / mask.sum().clamp_min(1)

    logits = F.linear(shift_hidden.float(), lm_head.float())
    expected = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(actual, expected, atol=1e-4)


def test_masked_mean_global():
    """masked_mean 基础行为。"""
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, True, False], [False, True, True]])
    actual = masked_mean(tensor, mask, dim=-1)
    assert torch.allclose(actual, torch.tensor([1.5, 5.5]))
