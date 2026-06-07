import pytest

from graspo.core.reward import GraspoReward, RewardConfig


def test_reward_perfect_json_fence():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"APN":"cmnet","fault_number":"138"}\n```',
        {"APN": "cmnet", "fault_number": "138"},
    )

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.reward > 1.0
    assert result.max_score == 230.0
    assert result.raw_score == 231.0


def test_reward_partial_json_fence():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"APN":"wrong","fault_number":"138"}\n```',
        {"APN": "cmnet", "fault_number": "138"},
    )

    assert result.all_right is False
    assert 0 < result.content_score < 1.0
    assert 0 < result.reward < 1.0


def test_reward_rejects_missing_marker_when_required():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score('{"APN":"cmnet"}', {"APN": "cmnet"})

    assert result.all_right is False
    assert result.content_score == 0
    assert 0 <= result.reward < 1


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


def test_reward_tool_call_matches_original_scoring_shape():
    reward = GraspoReward(RewardConfig(check_json_markdown=True, check_tool_call=True))
    result = reward.score(
        '```json\n{"answer":"ok"}\n```\n<tool_call>{"name":"search","arguments":{"q":"apn"}}</tool_call>',
        {"answer": "ok"},
        {"name": "search", "arguments": {"q": "apn"}},
    )

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.max_score == 460.0
    assert result.reward > 1.0


def test_reward_list_ground_truth_uses_first_reference():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    result = reward.score('{"APN":"cmnet"}', [{"APN": "cmnet"}, {"APN": "wrong"}])

    assert result.all_right is True
    assert result.reward > 1.0
