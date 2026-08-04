"""奖励计算与注册表（graspo.ripple.reward.reward）的单元测试。"""

import pytest

from graspo.ripple.parsing.completion import ParsedCompletion
from graspo.ripple.reward.reward import GraspoReward, RewardConfig


def _content_targets(*contents: dict) -> list[dict]:
    return [
        {"id": f"target-{idx}", "output": {"content": content}}
        for idx, content in enumerate(contents)
    ]


def _tool_targets(*calls: list[dict]) -> list[dict]:
    return [
        {"id": f"target-{idx}", "output": {"tool_calls": list(call_list)}}
        for idx, call_list in enumerate(calls)
    ]


def test_reward_perfect_json_fence():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"APN":"cmnet","fault_number":"138"}\n```',
        _content_targets({"APN": "cmnet", "fault_number": "138"}),
    )

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.matched_target_index == 0
    assert result.reward > 1.0
    assert result.max_score == 230.0
    assert result.raw_score == 231.0


def test_reward_partial_json_fence():
    """动作类型错误 → base_content_score < 1.0 → 使用 base_dcs 而非 dcs。"""
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"APN":"wrong","fault_number":"138"}\n```',
        _content_targets({"APN": "cmnet", "fault_number": "138"}),
    )

    assert result.all_right is False
    assert result.base_content_score < 1.0
    # 数值精度被忽略，只用 base_content_score
    assert 0 < result.content_score < 1.0
    assert 0 < result.reward < 1.0


def test_reward_numeric_json_field_uses_continuous_distance_score():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"action":"left","distance_cm":8}\n```',
        _content_targets({"action": "left", "distance_cm": 6}),
    )
    type_mismatch = reward.score(
        '```json\n{"action":"left","distance_cm":"6"}\n```',
        _content_targets({"action": "left", "distance_cm": 6}),
    )

    # all_right strips numeric leaves → both {"action": "left"} → all_right is True
    assert result.all_right is True
    # content_score includes numeric fields (gradient signal preserved).
    # leaf_compare_score uses relative error with numeric_tolerance=0.2:
    # rel error = 2/6 = 1/3 → clipped to 2/15 → leaf score 15/17;
    # check = 4 (keys + all-keys + action match) + 15/17, total = 5.
    assert result.content_score == pytest.approx((4 + 15 / 17) / 5)
    assert result.content_score > type_mismatch.content_score


def test_reward_numeric_json_field_exact_match_is_all_right():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"distance_cm":6.0}\n```',
        _content_targets({"distance_cm": 6}),
    )

    assert result.all_right is True
    assert result.content_score == 1.0


def test_reward_selects_best_content_target():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"action":"down","distance_cm":5}\n```',
        _content_targets(
            {"action": "left", "distance_cm": 6},
            {"action": "down", "distance_cm": 4},
        ),
    )

    # target-1 matches structurally ("down"), numeric distance_cm stripped → all_right is True
    assert result.all_right is True
    assert result.matched_target_index == 1
    assert result.matched_target_id == "target-1"
    assert result.target_scores is not None
    assert result.target_scores[1].content_score > result.target_scores[0].content_score


def test_reward_rejects_missing_marker_when_required():
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score('{"APN":"cmnet"}', _content_targets({"APN": "cmnet"}))

    assert result.all_right is False
    assert result.content_score == 0
    assert 0 <= result.reward < 1


def test_reward_penalizes_large_useless_text():
    reward = GraspoReward(
        RewardConfig(check_json_markdown=True, anti_useless_str_half_reward_len=10)
    )
    result = reward.score(
        "x" * 30 + '```json\n{"APN":"cmnet"}\n```',
        _content_targets({"APN": "cmnet"}),
    )

    assert result.all_right is False
    assert result.content_score == 0


def test_invalid_targets_type():
    reward = GraspoReward()
    with pytest.raises(ValueError, match="targets"):
        reward.score("{}", [{"output": {"content": {"x": 1}}}, "not a target"])


def test_reward_parsed_tool_call_matches_one_target():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="<tool_call>...</tool_call>",
        tool_calls=[{"name": "robot_atomic_control", "arguments": {"action": "down"}}],
        parser_name="qwen_xml_tool_call",
    )

    result = reward.score_parsed(
        parsed,
        _tool_targets(
            [{"name": "robot_atomic_control", "arguments": {"action": "left"}}],
            [{"name": "robot_atomic_control", "arguments": {"action": "down"}}],
        ),
        is_tool_call=True,
    )

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.matched_target_index == 1
    assert result.extracted["tool_calls"] == [
        {"name": "robot_atomic_control", "arguments": {"action": "down"}}
    ]


def test_reward_parsed_tool_call_numeric_argument_gets_partial_credit():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    partial = ParsedCompletion(
        raw_text="",
        tool_calls=[{"name": "robot_atomic_control", "arguments": {"distance_cm": 8}}],
    )
    type_mismatch = ParsedCompletion(
        raw_text="",
        tool_calls=[{"name": "robot_atomic_control", "arguments": {"distance_cm": "6"}}],
    )
    targets = _tool_targets([{"name": "robot_atomic_control", "arguments": {"distance_cm": 6}}])

    partial_result = reward.score_parsed(partial, targets, is_tool_call=True)
    mismatch_result = reward.score_parsed(type_mismatch, targets, is_tool_call=True)

    # numeric distance_cm stripped → structural match → all_right is True
    assert partial_result.all_right is True
    assert 0 < partial_result.content_score < 1
    assert partial_result.content_score > mismatch_result.content_score


def test_reward_parsed_tool_call_selects_best_numeric_target():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[{"name": "move", "arguments": {"action": "left", "distance_cm": 7}}],
    )

    result = reward.score_parsed(
        parsed,
        _tool_targets(
            [{"name": "move", "arguments": {"action": "down", "distance_cm": 4}}],
            [{"name": "move", "arguments": {"action": "left", "distance_cm": 6}}],
        ),
        is_tool_call=True,
    )

    assert result.matched_target_index == 1
    assert result.target_scores is not None
    assert result.target_scores[1].content_score > result.target_scores[0].content_score


def test_reward_parsed_multi_tool_calls_use_ordered_sequence_inside_target():
    # check_list_order is config-driven (default False = order-insensitive);
    # opt in to ordered comparison to test sequence semantics.
    reward = GraspoReward(RewardConfig(check_json_markdown=False, check_list_order=True))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[
            {"name": "first", "arguments": {"x": 1}},
            {"name": "second", "arguments": {"y": 2}},
        ],
    )

    correct = reward.score_parsed(
        parsed,
        _tool_targets(
            [
                {"name": "first", "arguments": {"x": 1}},
                {"name": "second", "arguments": {"y": 2}},
            ]
        ),
        is_tool_call=True,
    )
    wrong_order = reward.score_parsed(
        parsed,
        _tool_targets(
            [
                {"name": "second", "arguments": {"y": 2}},
                {"name": "first", "arguments": {"x": 1}},
            ]
        ),
        is_tool_call=True,
    )

    assert correct.all_right is True
    assert wrong_order.all_right is False
    assert wrong_order.content_score < correct.content_score


def test_reward_parsed_tool_call_parse_error_is_not_all_right():
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="<tool_call>bad</tool_call>",
        tool_calls=[{"name": "robot_atomic_control", "arguments": {"action": "down"}}],
        parse_errors=["bad tool call"],
    )

    result = reward.score_parsed(
        parsed,
        _tool_targets([{"name": "robot_atomic_control", "arguments": {"action": "down"}}]),
        is_tool_call=True,
    )

    assert result.all_right is False
    assert result.content_score == 1.0


def test_reward_parsed_too_many_tool_calls_gets_zero_reward():
    """模型生成多于 target 的 tool call → 按格式错误处理，reward ≈ 0。"""
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
            {"name": "move", "arguments": {"action": "backward", "distance_cm": 10}},
            {"name": "turn", "arguments": {"action": "right", "angle_deg": 45}},
        ],
    )
    targets = _tool_targets(
        [
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
        ]
    )

    result = reward.score_parsed(parsed, targets, is_tool_call=True)

    assert result.content_score == 0.0
    assert result.reward == pytest.approx(0.0, abs=0.01)
    assert result.all_right is False
    assert "too many tool calls" in result.extracted["parse_errors"]


def test_reward_parsed_fewer_tool_calls_not_penalized():
    """模型生成少于 target 的 tool call → 不惩罚，正常评分。"""
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[{"name": "move", "arguments": {"action": "forward", "distance_cm": 30}}],
    )
    targets = _tool_targets(
        [
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
        ]
    )

    result = reward.score_parsed(parsed, targets, is_tool_call=True)

    assert result.content_score > 0.0
    assert result.reward > 0.0
    assert "too many tool calls" not in result.extracted["parse_errors"]


def test_reward_parsed_exact_tool_call_count_not_penalized():
    """模型生成与 target 相同数量的 tool call → 正常评分，不触发 penalty。"""
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
        ],
    )
    targets = _tool_targets(
        [
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
        ]
    )

    result = reward.score_parsed(parsed, targets, is_tool_call=True)

    assert result.content_score == 1.0
    assert result.reward > 0.0
    assert "too many tool calls" not in result.extracted["parse_errors"]
    assert result.all_right is True


def test_reward_parsed_too_many_uses_max_target_count():
    """多个 target 有不同 tc 数量时，以最大值为准。"""
    reward = GraspoReward(RewardConfig(check_json_markdown=False))
    parsed = ParsedCompletion(
        raw_text="",
        tool_calls=[
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
            {"name": "move", "arguments": {"action": "backward", "distance_cm": 10}},
        ],
    )
    # target-0 has 2 tool calls, target-1 has 3 → max = 3, so 4 is too many
    targets = _tool_targets(
        [
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
        ],
        [
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 30}},
            {"name": "turn", "arguments": {"action": "left", "angle_deg": 90}},
            {"name": "move", "arguments": {"action": "forward", "distance_cm": 50}},
        ],
    )

    result = reward.score_parsed(parsed, targets, is_tool_call=True)

    assert result.content_score == 0.0
    assert "too many tool calls" in result.extracted["parse_errors"]


def test_reward_parsed_too_many_does_not_affect_content_score_target():
    """tc_count 惩罚不影响 content 评分路径（非 tool call 场景）。"""
    reward = GraspoReward(RewardConfig(check_json_markdown=True))
    result = reward.score(
        '```json\n{"APN":"cmnet","fault_number":"138"}\n```',
        _content_targets({"APN": "cmnet", "fault_number": "138"}),
    )

    assert result.all_right is True
    assert result.content_score == 1.0
    assert result.reward > 1.0
