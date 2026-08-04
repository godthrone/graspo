"""字典比较评分（graspo.ripple.reward.compare）的单元测试。"""

import pytest

from graspo.ripple.reward.compare import dict_compare_score


def test_dict_compare_exact():
    result = dict_compare_score({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]})

    assert result.dcs == 1.0
    assert result.total_score == result.check_score


def test_dict_compare_partial():
    result = dict_compare_score({"a": 1, "b": [1]}, {"a": 1, "b": [1, 2]})

    assert 0 < result.dcs < 1
    assert result.check_score < result.total_score


def test_dict_compare_list_order_optional():
    unordered = dict_compare_score(
        {"items": [2, 1]},
        {"items": [1, 2]},
        check_list_order=False,
    )
    ordered = dict_compare_score(
        {"items": [2, 1]},
        {"items": [1, 2]},
        check_list_order=True,
    )

    assert unordered.dcs > ordered.dcs


def test_dict_compare_numeric_leaf_uses_relative_error_score():
    """数值比较使用相对误差，target=0 时回退到绝对误差。"""
    # target=6, checked=8: relative error = 2/6 ≈ 0.333
    result = dict_compare_score(
        {"distance_cm": 8},
        {"distance_cm": 6},
    )

    rel_score = 1.0 / (1.0 + 2.0 / 6.0)  # 1/(1+0.333) = 0.75
    assert result.total_score == 3
    assert result.check_score == pytest.approx(2 + rel_score)
    assert result.dcs == pytest.approx((2 + rel_score) / 3)

    # base should exclude numeric: numeric leaves are stripped, keys
    # whose only children are numeric collapse → empty dict remains
    assert result.base_total == 1
    assert result.base_check == 1.0
    assert result.base_dcs == 1.0
    assert result.all_right is True


def test_dict_compare_numeric_relative_error_is_proportional():
    """相同绝对误差，不同相对误差 → 得分不同。"""
    # target=10, checked=0: relative = 10/10 = 1.0 → score = 1/(1+1) = 0.5
    # target=30, checked=20: relative = 10/30 ≈ 0.333 → score = 1/(1+0.333) ≈ 0.75
    result_small = dict_compare_score({"val": 0}, {"val": 10})
    result_large = dict_compare_score({"val": 20}, {"val": 30})

    assert result_small.dcs < result_large.dcs


def test_dict_compare_numeric_zero_target_falls_back_to_absolute():
    """target=0 时使用绝对误差作为分母。"""
    result = dict_compare_score({"val": 10}, {"val": 0})

    abs_score = 1.0 / (1.0 + 10.0)  # 1/11 ≈ 0.0909
    assert result.total_score == 3
    assert result.check_score == pytest.approx(2 + abs_score)


def test_dict_compare_numeric_string_type_mismatch_gets_no_leaf_score():
    result = dict_compare_score(
        {"distance_cm": "6"},
        {"distance_cm": 6},
    )

    assert result.total_score == 3
    assert result.check_score == 2
    assert result.dcs == pytest.approx(2 / 3)

    # base: "6" is string (non-numeric) → kept; 6 is numeric → key stripped
    # checked_stripped = {"distance_cm": "6"}, target_stripped = {}
    assert result.base_total == 1  # only the dict itself (numeric key stripped)
    assert result.base_check == 0.0  # key "distance_cm" not in target_stripped
    assert result.base_dcs == 0.0
    assert result.all_right is False


def test_dict_compare_bool_is_not_numeric():
    result = dict_compare_score({"enabled": True}, {"enabled": 1})

    assert result.total_score == 3
    assert result.check_score == 2
    # bool is not numeric, int 1 is numeric → target key stripped
    assert result.base_total == 1  # only dict (target key stripped)
    assert result.base_check == 0.0  # "enabled" not in target_stripped={}


def test_dict_compare_int_float_exact_match_is_all_right():
    result = dict_compare_score({"distance_cm": 6}, {"distance_cm": 6.0})

    assert result.total_score == result.check_score
    # full all-right via continuous scoring
    # base all-right should also be true (numeric stripped, structure matches)
    assert result.all_right is True


def test_dict_compare_list_dict_element_uses_nested_numeric_score():
    result = dict_compare_score(
        {"tool_calls": [{"name": "move", "arguments": {"distance_cm": 8}}]},
        {"tool_calls": [{"name": "move", "arguments": {"distance_cm": 6}}]},
        check_list_order=True,
    )

    # Full score: list dict elements are recursively expanded in denominator.
    # Numeric comparison uses relative error: target=6, checked=8 → rel=2/6≈0.333
    # leaf_compare_score = 1/(1+0.333) = 0.75
    # check_score = 2 + 0.75 (arguments dict) + ... = 8.75
    assert result.total_score == 11
    assert result.check_score == pytest.approx(8.75)
    assert result.dcs == pytest.approx(8.75 / 11)

    # Base score strips numeric → arguments dict becomes empty, stripped entirely
    # target after strip: {"tool_calls": [{"name": "move"}]}
    # checked after strip: {"tool_calls": [{"name": "move"}]}
    assert result.base_total == 7
    assert result.base_check == 7.0
    assert result.base_dcs == 1.0
    assert result.all_right is True


def test_dict_compare_numeric_tolerance_applies_to_list_element_containment():
    """List element containment check uses tolerance, not Python ==.

    When numeric_tolerance is set, leaf values that differ but are within
    tolerance should be treated as matches — including at the list-element
    containment level.  Before the fix, ``element in target_value`` used
    Python ``==`` which ignores tolerance, capping content_score at 0.9167
    for otherwise-perfect training completions.
    """
    checked = {
        "tool_calls": [
            {
                "name": "robot_atomic_control",
                "arguments": {"action_type": "逆时针旋转", "angle_deg": 90.0},
            }
        ]
    }
    target = {
        "tool_calls": [
            {
                "name": "robot_atomic_control",
                "arguments": {"action_type": "逆时针旋转", "angle_deg": 19.1},
            }
        ]
    }
    result = dict_compare_score(
        checked=checked,
        target=target,
        check_list_order=False,
        numeric_tolerance=10.0,
    )
    assert result.dcs == 1.0
    assert result.total_score == result.check_score
    assert result.all_right is True


def test_dict_compare_numeric_tolerance_partial_list_match():
    """When a checked element partially matches, containment is not granted."""
    checked = {
        "tool_calls": [
            {
                "name": "robot_atomic_control",
                "arguments": {"action_type": "伸长手臂", "distance_cm": 10.0},
            }
        ]
    }
    target = {
        "tool_calls": [
            {
                "name": "robot_atomic_control",
                "arguments": {"action_type": "逆时针旋转", "angle_deg": 19.1},
            }
        ]
    }
    result = dict_compare_score(
        checked=checked,
        target=target,
        check_list_order=False,
        numeric_tolerance=10.0,
    )
    assert result.dcs < 1.0
    assert result.base_dcs < 1.0
    assert result.all_right is False
