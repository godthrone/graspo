"""Token 级梯度正确性验证：autograd.gradcheck 验证 PPO loss 的梯度计算。

验证目标（C2 验收）：
1. ``GRASPORippleLoss.forward`` 对 log_probs 的梯度通过 ``torch.autograd.gradcheck``
2. ``masked_token_log_probs_from_hidden`` 对 hidden_states 的梯度通过 gradcheck
3. 数值扰动下的梯度一致性（双精度容差）

测试策略：
- 用小规模张量（batch=2, seq=4, hidden=8, vocab=64）避免计算开销
- 使用 double 精度（float64）满足 gradcheck 默认容差
- 同时覆盖带 mask 和不带 mask 的场景
"""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from graspo.ripple.loss import (  # noqa: E402
    GRASPORippleLoss,
    masked_token_log_probs_from_hidden,
)

# ── GRASPORippleLoss.forward gradcheck ────────────────────────────────────────


def _make_loss_inputs(
    batch: int = 2,
    seq: int = 4,
    seed: int = 42,
    *,
    all_masked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 GRASPORippleLoss.forward 的输入张量（double 精度）。"""
    gen = torch.Generator().manual_seed(seed)
    log_probs = torch.randn(batch, seq, generator=gen, dtype=torch.float64)
    old_log_probs = torch.randn(batch, seq, generator=gen, dtype=torch.float64)
    advantages = torch.randn(batch, seq, generator=gen, dtype=torch.float64)
    if all_masked:
        action_mask = torch.zeros(batch, seq, dtype=torch.bool)
    else:
        action_mask = torch.randint(0, 2, (batch, seq), generator=gen).bool()
    return log_probs, old_log_probs, advantages, action_mask


def test_ppo_loss_gradcheck_log_probs():
    """gradcheck: GRASPORippleLoss.forward 对 log_probs 的梯度正确。"""
    log_probs, old_log_probs, advantages, action_mask = _make_loss_inputs()
    log_probs = log_probs.double().requires_grad_(True)

    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=0.2)

    def _forward(lp: torch.Tensor) -> torch.Tensor:
        return loss_fn(lp, old_log_probs, advantages, action_mask)

    assert torch.autograd.gradcheck(_forward, (log_probs,), eps=1e-4, atol=1e-3, rtol=1e-3)


def test_ppo_loss_gradcheck_full_mask():
    """gradcheck: 全 mask=False 时（全被忽略），loss=0，梯度仍应正确（全零）。"""
    log_probs, old_log_probs, advantages, action_mask = _make_loss_inputs(all_masked=True)
    log_probs = log_probs.double().requires_grad_(True)

    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=0.2)

    def _forward(lp: torch.Tensor) -> torch.Tensor:
        return loss_fn(lp, old_log_probs, advantages, action_mask)

    # 全 mask=False 时 masked_mean 分母 clamp_min(1)，loss=0
    loss = loss_fn(log_probs, old_log_probs, advantages, action_mask)
    assert loss.item() == 0.0

    assert torch.autograd.gradcheck(_forward, (log_probs,), eps=1e-4, atol=1e-3, rtol=1e-3)


def test_ppo_loss_gradcheck_partial_mask():
    """gradcheck: 部分 mask 时，未 mask 位置的梯度非零。"""
    log_probs = torch.tensor(
        [[0.0, 0.2, -0.1, 0.3], [0.1, -0.3, 0.4, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old_log_probs = torch.tensor(
        [[-0.1, 0.1, -0.1, 0.0], [0.0, -0.1, 0.2, 0.1]],
        dtype=torch.float64,
    )
    advantages = torch.tensor(
        [[1.0, 1.0, 1.0, 0.5], [-0.5, -0.5, -0.5, 0.0]],
        dtype=torch.float64,
    )
    action_mask = torch.tensor(
        [[True, True, False, False], [True, False, True, False]],
    )

    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=0.2)
    loss = loss_fn(log_probs, old_log_probs, advantages, action_mask)
    loss.backward()

    assert log_probs.grad is not None
    # 被 mask 的 token 梯度应为 0（mask=False 的位置）
    masked_positions = ~action_mask
    assert torch.allclose(
        log_probs.grad[masked_positions], torch.zeros_like(log_probs.grad[masked_positions])
    )
    # 未 mask 的 token 梯度非零（有 nonzero advantage）
    assert (log_probs.grad[action_mask] != 0.0).any()


# ── masked_token_log_probs_from_hidden gradcheck ──────────────────────────────


def _make_log_probs_inputs(
    batch: int = 2,
    seq: int = 4,
    hidden_dim: int = 8,
    vocab: int = 64,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, hidden_dim, generator=gen, dtype=torch.float64)
    lm_head = torch.randn(vocab, hidden_dim, generator=gen, dtype=torch.float64) * 0.02
    token_ids = torch.randint(0, vocab, (batch, seq), generator=gen)
    return hidden, lm_head, token_ids


def test_masked_token_log_probs_gradcheck_hidden():
    """gradcheck: masked_token_log_probs_from_hidden 对 hidden_states 的梯度正确。"""
    hidden, lm_head, token_ids = _make_log_probs_inputs()
    hidden = hidden.double().requires_grad_(True)

    def _forward(h: torch.Tensor) -> torch.Tensor:
        result = masked_token_log_probs_from_hidden(
            h, lm_head, token_ids, ignore_index=-100, vocab_chunk_size=32
        )
        return result.sum()

    assert torch.autograd.gradcheck(_forward, (hidden,), eps=1e-4, atol=1e-3, rtol=1e-3)


def test_masked_token_log_probs_gradcheck_with_ignore():
    """gradcheck: 带 ignore_index 时 gradient 在 ignored 位置为 0。"""
    hidden, lm_head, token_ids = _make_log_probs_inputs(vocab=128)
    # 第一半设为 ignore
    token_ids = token_ids.clone()
    token_ids[:, : token_ids.shape[1] // 2] = -100

    hidden = hidden.double().requires_grad_(True)

    result = masked_token_log_probs_from_hidden(
        hidden, lm_head, token_ids, ignore_index=-100, vocab_chunk_size=32
    )
    loss = result.sum()
    loss.backward()

    assert hidden.grad is not None
    # ignored 位置梯度应为 0
    assert (hidden.grad[:, : hidden.shape[1] // 2] == 0.0).all()
    # 有效位置梯度非零
    assert (hidden.grad[:, hidden.shape[1] // 2 :] != 0.0).any()


def test_log_probs_gradcheck_chunk_boundary():
    """gradcheck: vocab_chunk_size 跨块边界时梯度仍正确。"""
    hidden, lm_head, token_ids = _make_log_probs_inputs(vocab=128)
    hidden = hidden.double().requires_grad_(True)

    # chunk_size=33 不是 128 的整除，确保跨块边界
    def _forward(h: torch.Tensor) -> torch.Tensor:
        result = masked_token_log_probs_from_hidden(
            h, lm_head, token_ids, ignore_index=-100, vocab_chunk_size=33
        )
        return result.sum()

    assert torch.autograd.gradcheck(_forward, (hidden,), eps=1e-4, atol=1e-3, rtol=1e-3)


# ── 梯度方向验证 ──────────────────────────────────────────────────────────────


def test_positive_advantage_produces_negative_loss_gradient():
    """正 advantage → 增大 log_prob → 减小 loss → 梯度为负（loss 对 log_prob 的梯度）。"""
    log_probs = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)
    action_mask = torch.tensor([[True]])

    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=0.2)
    loss = loss_fn(log_probs, old_log_probs, advantages, action_mask)
    loss.backward()

    assert log_probs.grad is not None
    # ratio = exp(0.0) = 1.0, surr = 1.0 * 1.0 = 1.0
    # loss = -min(1.0, 1.0) = -1.0 → dloss/dlog_prob < 0（增大 log_prob 减小 loss）
    assert log_probs.grad.item() < 0


def test_negative_advantage_produces_positive_loss_gradient():
    """负 advantage → 减小 log_prob → 减小 loss → 梯度为正（loss 对 log_prob 的梯度）。"""
    log_probs = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[-1.0]], dtype=torch.float64)
    action_mask = torch.tensor([[True]])

    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=0.2)
    loss = loss_fn(log_probs, old_log_probs, advantages, action_mask)
    loss.backward()

    assert log_probs.grad is not None
    # ratio = 1.0, surr = -1.0, clamped = -1.0
    # loss = -min(-1.0, -1.0) = 1.0 → dloss/dlog_prob > 0
    assert log_probs.grad.item() > 0


def test_zero_advantage_produces_zero_loss_gradient():
    """零 advantage → loss=0 → 梯度全零。"""
    log_probs = torch.tensor([[0.0, 0.5]], dtype=torch.float64, requires_grad=True)
    old_log_probs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    advantages = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    action_mask = torch.tensor([[True, True]])

    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=0.2)
    loss = loss_fn(log_probs, old_log_probs, advantages, action_mask)
    loss.backward()

    assert log_probs.grad is not None
    assert torch.allclose(log_probs.grad, torch.zeros_like(log_probs.grad))
