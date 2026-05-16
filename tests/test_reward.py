import pytest

from graspo.core.reward import GraspoReward, RewardConfig


def test_reward_perfect_json_fence():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score('```json\n{"APN":"cmnet","fault_number":"138"}\n```', {"APN": "cmnet", "fault_number": "138"})

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.reward >= 1.0


def test_reward_partial_json_fence():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score('```json\n{"APN":"wrong","fault_number":"138"}\n```', {"APN": "cmnet", "fault_number": "138"})

    assert result.all_right is False
    assert 0 < result.content_score < 1.0
    assert 0 < result.reward < 1.0


def test_reward_rejects_missing_marker_when_required():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score('{"APN":"cmnet"}', {"APN": "cmnet"})

    assert result.all_right is False
    assert result.content_score == 0


def test_reward_penalizes_large_useless_text():
    reward = GraspoReward(
        RewardConfig(check_json_markdown=True, anti_useless_str_half_reward_len=10)
    )
    result = reward.score("x" * 30 + '```json\n{"APN":"cmnet"}\n```', {"APN": "cmnet"})

    assert result.all_right is False
    assert result.content_score == 0


def test_invalid_ground_truth_type():
    reward = GraspoReward()
    with pytest.raises(ValueError):
        reward.score("{}", ["not a dict"])

