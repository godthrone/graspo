"""针对性梯度验证：验证不同 CharTag 类型 token 的梯度符号与大小。

验证目标（C2 验收）：
1. S (STRUCTURE) token: advantage=+1.0 → 梯度为负（增大 log_prob 减小 loss）
2. V (VALUE) token: 梯度符号 = 负 advantage（adv>0 → 梯度<0，adv<0 → 梯度>0）
3. E (ERROR) token: advantage=−1.0 → 梯度为正（减小 log_prob 减小 loss）
4. T/W/D (THINK/WASTE/DROPPED) token: advantage=0 → loss 贡献为 0
5. action_mask=False 的 token: 梯度为 0

测试策略：
- 直接构造 log_probs/old_log_probs/advantages/action_mask 张量
- 模拟不同 CharTag 对应的 advantage 值
- 验证 loss.backward() 后各 token 位置的梯度符号
"""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from graspo.ripple.loss import GRASPORippleLoss  # noqa: E402


# ── 工具函数 ─────────────────────────────────────────────────────────────────


def _compute_grad(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """计算 GRASPORippleLoss 对 log_probs 的梯度。"""
    lp = log_probs.double().requires_grad_(True)
    loss_fn = GRASPORippleLoss(policy_ratio_clip_eps=clip_eps)
    loss = loss_fn(lp, old_log_probs.double(), advantages.double(), action_mask)
    loss.backward()
    assert lp.grad is not None
    return lp.grad


# ── 梯度符号测试 ──────────────────────────────────────────────────────────────


def test_structure_token_gradient_is_negative():
    """S token (adv=+1.0): 增大 log_prob → 减小 loss → 梯度为负。"""
    log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)  # S token
    action_mask = torch.tensor([[True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)
    assert grad.item() < 0.0, f"expected negative gradient, got {grad.item()}"


def test_error_token_gradient_is_positive():
    """E token (adv=−1.0): 减小 log_prob → 减小 loss → 梯度为正。"""
    log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[-1.0]], dtype=torch.float64)  # E token
    action_mask = torch.tensor([[True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)
    assert grad.item() > 0.0, f"expected positive gradient, got {grad.item()}"


def test_zero_advantage_token_gradient_is_zero():
    """T/W/D token (adv=0): loss 贡献为 0 → 梯度为 0。"""
    log_probs = torch.tensor([[0.0, 0.5]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    advantages = torch.tensor([[0.0, 0.0]], dtype=torch.float64)  # T/W/D tokens
    action_mask = torch.tensor([[True, True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)
    assert torch.allclose(grad, torch.zeros_like(grad))


def test_masked_token_gradient_is_zero():
    """action_mask=False: 梯度为 0（被 masked_mean 排除）。"""
    log_probs = torch.tensor([[0.0, 0.5]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    action_mask = torch.tensor([[False, False]])  # 全部 mask

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)
    assert torch.allclose(grad, torch.zeros_like(grad))


def test_gradient_sign_opposite_to_advantage():
    """梯度符号 = 负 advantage（adv>0 → grad<0, adv<0 → grad>0）。"""
    log_probs = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    # 混合：S(+1), E(-1), V(+0.5)
    advantages = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float64)
    action_mask = torch.tensor([[True, True, True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)

    # S: adv=+1.0 → grad must be negative
    assert grad[0, 0].item() < 0.0, f"S token: expected negative grad, got {grad[0, 0]}"
    # E: adv=−1.0 → grad must be positive
    assert grad[0, 1].item() > 0.0, f"E token: expected positive grad, got {grad[0, 1]}"
    # V: adv=+0.5 → grad must be negative
    assert grad[0, 2].item() < 0.0, f"V token: expected negative grad, got {grad[0, 2]}"


# ── PPO clip 边界测试 ─────────────────────────────────────────────────────────


def test_gradient_respects_ppo_clip_upper_bound():
    """ratio 超出上界 (1+eps) 时，梯度被 clip 为 0（PPO 标准行为）。

    当 ratio > 1+eps 且 advantage > 0 时：
    - surr1 = ratio * adv（大）
    - surr2 = clamped_ratio * adv（小）
    - min 选 surr2，但 surr2 中的 clamp 操作梯度为 0
    - 因此总梯度为 0（策略已移动太远，不再更新）
    """
    # ratio = exp(2.0 - 0.0) = exp(2.0) ≈ 7.389 > 1.2
    log_probs = torch.tensor([[2.0]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)
    action_mask = torch.tensor([[True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)

    # PPO clip: ratio > 1+eps, min 选 clamped side, clamp grad = 0
    assert grad.item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_respects_ppo_clip_lower_bound():
    """ratio 低于下界 (1-eps) 时，min 选 surr1（未 clamp 侧），梯度正常。"""
    # ratio = exp(-2.0 - 0.0) = exp(-2.0) ≈ 0.135 < 0.8
    log_probs = torch.tensor([[-2.0]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)
    action_mask = torch.tensor([[True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)

    # ratio = exp(-2.0) ≈ 0.135335
    # surr1 = 0.135335 * 1.0 = 0.135335
    # surr2 = 0.8 * 1.0 = 0.8 (clamped)
    # min(0.135335, 0.8) = 0.135335 (unclamped side wins)
    # loss = -0.135335, grad = d(-ratio*adv)/dlog_prob = -ratio*adv = -0.135335
    import math

    expected = -math.exp(-2.0)
    assert grad.item() == pytest.approx(expected, abs=1e-6)


def test_gradient_clip_ratio_range():
    """ratio 在 [0.1, 10.0] 范围内不会被额外 clamp。"""
    # ratio = exp(0.0) = 1.0，在 [0.8, 1.2] 内
    log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    old_log_probs = torch.tensor([[0.0]], dtype=torch.float64)
    advantages = torch.tensor([[1.0]], dtype=torch.float64)
    action_mask = torch.tensor([[True]])

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)

    # ratio = 1.0, surr1 = 1.0, surr2 = 1.0, loss = -1.0
    # grad = d(-1.0)/dlog_prob = -exp(0.0) * 1.0 = -1.0
    assert grad.item() == pytest.approx(-1.0, abs=1e-6)


# ── 批量混合场景 ──────────────────────────────────────────────────────────────


def test_mixed_advantages_batch():
    """批量混合场景：不同 token 有不同 advantage，验证梯度各自正确。"""
    log_probs = torch.zeros(2, 4, dtype=torch.float64)
    old_log_probs = torch.zeros(2, 4, dtype=torch.float64)
    # Row 0: S(+1), E(-1), V(+0.5), T(0)
    # Row 1: S(+1), V(-0.3), D(0), E(-1)
    advantages = torch.tensor(
        [[1.0, -1.0, 0.5, 0.0], [1.0, -0.3, 0.0, -1.0]],
        dtype=torch.float64,
    )
    action_mask = torch.ones(2, 4, dtype=torch.bool)

    grad = _compute_grad(log_probs, old_log_probs, advantages, action_mask)

    # Row 0
    assert grad[0, 0].item() < 0.0  # S: negative
    assert grad[0, 1].item() > 0.0  # E: positive
    assert grad[0, 2].item() < 0.0  # V(+0.5): negative
    assert grad[0, 3].item() == 0.0  # T: zero

    # Row 1
    assert grad[1, 0].item() < 0.0  # S: negative
    assert grad[1, 1].item() > 0.0  # V(-0.3): positive (opposite sign)
    assert grad[1, 2].item() == 0.0  # D: zero
    assert grad[1, 3].item() > 0.0  # E: positive

    # 梯度大小关系：|grad(S)| > |grad(V)|（因为 |adv(S)|=1.0 > |adv(V)|）
    assert abs(grad[0, 0].item()) > abs(grad[0, 2].item())  # |S| > |V(+0.5)|
    assert abs(grad[1, 0].item()) > abs(grad[1, 1].item())  # |S| > |V(-0.3)|
