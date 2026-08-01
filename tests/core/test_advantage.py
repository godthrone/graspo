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


# ── 公式 A：质量加权（quality_power）────────────────────────────────────────


def test_quality_weighting_fairness_max_correct_vs_not_correct():
    """公平性：max_correct 组（max=1.0）满推，not_correct 组（max=0.43）弱推。"""
    max_correct = [1.0] + [0.43] * 7
    not_correct = [0.43] + [0.0] * 7
    adv_mc = group_advantages(max_correct)
    adv_nc = group_advantages(not_correct)
    # 纯 z-score 下两者最佳 advantage 相同（7/√8≈2.47）；公式 A 下应显著区分
    assert max(adv_mc) > max(adv_nc) * 3


def test_quality_weighting_preserves_rank_within_group():
    """组内相对顺序保持：quality 是组内共同正乘数。"""
    rewards = [0.0, 0.2, 0.4, 1.0]
    adv = group_advantages(rewards)
    # 单调性：reward 升序 → advantage 升序
    assert adv == sorted(adv)


def test_quality_weighting_sums_to_zero():
    """零和性质保持：组内 advantage 之和为零（无组级 loss 偏置）。"""
    rewards = [1.0, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43]
    adv = group_advantages(rewards)
    assert abs(sum(adv)) < 1e-6


def test_quality_weighting_power_one_less_aggressive():
    """p=1 比 p=2 的公平性区分弱（仍区分）。"""
    mc = [1.0] + [0.43] * 7
    nc = [0.43] + [0.0] * 7
    adv_mc_p1 = group_advantages(mc, quality_power=1.0)
    adv_nc_p1 = group_advantages(nc, quality_power=1.0)
    adv_mc_p2 = group_advantages(mc, quality_power=2.0)
    adv_nc_p2 = group_advantages(nc, quality_power=2.0)
    ratio_p1 = max(adv_mc_p1) / max(adv_nc_p1)
    ratio_p2 = max(adv_mc_p2) / max(adv_nc_p2)
    assert ratio_p2 > ratio_p1 > 1.0


def test_quality_weighting_keeps_above_one_bonus():
    """reward>1 的额外奖励（anti-useless）不被 clamp：max=1.0048 比 max=1.0 推力略高。"""
    tidy = [1.0048] + [0.43] * 7
    verbose = [1.0] + [0.43] * 7
    adv_tidy = group_advantages(tidy)
    adv_verbose = group_advantages(verbose)
    # 简洁组（>1）的最佳 advantage 应严格大于啰嗦组（=1.0）
    assert max(adv_tidy) > max(adv_verbose)
    # 且差距微小（约 1%），不破坏"满分=满推"的直觉
    assert max(adv_tidy) / max(adv_verbose) < 1.02


def test_quality_weighting_suppresses_low_variance_groups():
    """低方差组放大被压制：max 不高时 advantage 整体缩小。"""
    low_var = [0.42, 0.43, 0.43, 0.42, 0.43, 0.42, 0.43, 0.43]
    adv = group_advantages(low_var)
    # 纯 z-score 下该组 advantage 可达 ±10~100；公式 A 下应被压回合理范围
    assert all(abs(a) < 5.0 for a in adv)


def test_quality_weighting_default_power_is_two():
    """默认 quality_power=2.0（推荐值，安全窗口 1.2~3）。"""
    rewards = [0.43] + [0.0] * 7
    adv_default = group_advantages(rewards)
    adv_p2 = group_advantages(rewards, quality_power=2.0)
    assert adv_default == adv_p2
