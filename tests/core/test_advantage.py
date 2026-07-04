"""Tests for group advantage computation — BADGE §11.1."""

from graspo.core.advantage import group_advantages, has_reward_variance

# ── group_advantages ─────────────────────────────────────────────────────────


def test_group_advantages_standard_case():
    rewards = [0.0, 0.2, 0.4, 1.0]
    adv = group_advantages(rewards)
    assert len(adv) == 4
    # Mean = 0.4; best reward (1.0) should have positive advantage
    assert adv[3] > 0
    # Worst reward (0.0) should have negative advantage
    assert adv[0] < 0


def test_group_advantages_all_same_reward():
    rewards = [0.5, 0.5, 0.5]
    adv = group_advantages(rewards)
    # All zero (no variance, but with eps denominator)
    assert all(abs(a) < 1e-6 for a in adv)


def test_group_advantages_negative_rewards():
    rewards = [-1.0, -0.5, 0.0, 0.5]
    adv = group_advantages(rewards)
    assert len(adv) == 4
    assert adv[3] > adv[0]


def test_group_advantages_single_element_returns_zero():
    adv = group_advantages([1.0])
    assert adv == [0.0]


def test_group_advantages_empty_returns_empty():
    assert group_advantages([]) == []


def test_group_advantages_sum_is_zero():
    """GRPO advantages should sum to (near) zero."""
    rewards = [0.1, 0.3, 0.5, 0.7, 0.9]
    adv = group_advantages(rewards)
    assert abs(sum(adv)) < 1e-6


def test_group_advantages_high_variance():
    rewards = [0.0, 1.0]
    adv = group_advantages(rewards)
    # Best gets positive, worst gets negative
    assert adv[1] > 0
    assert adv[0] < 0
    # They should be opposites
    assert abs(adv[0] + adv[1]) < 1e-6


def test_group_advantages_custom_eps():
    rewards = [1.0, 1.0, 1.0]
    adv_small_eps = group_advantages(rewards, eps=1e-12)
    adv_large_eps = group_advantages(rewards, eps=1.0)
    # Larger eps → smaller advantages (more damping)
    for a_s, a_l in zip(adv_small_eps, adv_large_eps):
        assert abs(a_s) >= abs(a_l)


# ── has_reward_variance ─────────────────────────────────────────────────────


def test_has_reward_variance_true_when_values_differ():
    assert has_reward_variance([0.0, 0.5, 1.0]) is True


def test_has_reward_variance_false_when_all_same():
    assert has_reward_variance([0.5, 0.5, 0.5]) is False


def test_has_reward_variance_single_value():
    assert has_reward_variance([1.0]) is False


def test_has_reward_variance_empty():
    assert has_reward_variance([]) is False


def test_has_reward_variance_tiny_difference_below_eps():
    assert has_reward_variance([0.5, 0.5 + 1e-14]) is False


def test_has_reward_variance_tiny_difference_above_eps():
    assert has_reward_variance([0.5, 0.5 + 1e-10]) is True
