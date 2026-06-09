import pytest

from graspo.core.completion import ParsedCompletion
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


def test_reward_parsed_tool_call_matches_canonical_ground_truth():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="<tool_call>...</tool_call>",
        tool_calls=[{"name": "robot_atomic_control", "arguments": {"action": "向下"}}],
        parser_name="qwen_xml_tool_call",
    )

    result = reward.score_parsed(
        parsed,
        {"name": "robot_atomic_control", "arguments": {"action": "向下"}},
        is_tool_call=True,
    )

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.extracted["tool_calls"] == [
        {"name": "robot_atomic_control", "arguments": {"action": "向下"}}
    ]


def test_reward_parsed_multi_tool_calls_use_ordered_canonical_list():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[
            {"name": "first", "arguments": {"x": 1}},
            {"name": "second", "arguments": {"y": 2}},
        ],
    )

    correct = reward.score_parsed(
        parsed,
        [
            {"name": "first", "arguments": {"x": 1}},
            {"name": "second", "arguments": {"y": 2}},
        ],
        is_tool_call=True,
    )
    wrong_order = reward.score_parsed(
        parsed,
        [
            {"name": "second", "arguments": {"y": 2}},
            {"name": "first", "arguments": {"x": 1}},
        ],
        is_tool_call=True,
    )

    assert correct.all_right is True
    assert wrong_order.all_right is False
    assert wrong_order.content_score < correct.content_score


def test_reward_parsed_tool_call_parse_error_is_not_all_right():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="<tool_call>bad</tool_call>",
        tool_calls=[],
        parse_errors=["bad tool call"],
    )

    result = reward.score_parsed(
        parsed,
        {"name": "robot_atomic_control", "arguments": {"action": "向下"}},
        is_tool_call=True,
    )

    assert result.all_right is False
    assert result.content_score == 0.0


def test_reward_list_ground_truth_is_rejected():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))

    with pytest.raises(ValueError, match="JSON object"):
        reward.score('{"APN":"cmnet"}', [{"APN": "cmnet"}, {"APN": "wrong"}])
