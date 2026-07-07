import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from graspo.core.graspo_loss import GRASPOLoss  # noqa: E402


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

    actual = GRASPOLoss(policy_ratio_clip_eps=0.2)(
        log_probs, old_log_probs, advantages, action_mask
    )

    assert torch.allclose(actual, expected)
