import pytest

from graspo.core.compare import dict_compare_score


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


def test_dict_compare_numeric_leaf_uses_absolute_error_score():
    result = dict_compare_score(
        {"distance_cm": 8},
        {"distance_cm": 6},
    )

    assert result.total_score == 3
    assert result.check_score == pytest.approx(2 + 1 / 3)
    assert result.dcs == pytest.approx((2 + 1 / 3) / 3)

    # base should exclude numeric: numeric leaves are stripped, keys
    # whose only children are numeric collapse → empty dict remains
    assert result.base_total == 1  # only the dict itself (key stripped)
    assert result.base_check == 1.0  # all-checked-keys-in-target
    assert result.base_dcs == 1.0
    assert result.all_right is True


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

    # Full score: list dict elements are recursively expanded in denominator
    assert result.total_score == 11
    assert result.check_score == pytest.approx(25 / 3)
    assert result.dcs == pytest.approx(25 / 33)

    # Base score strips numeric → arguments dict becomes empty, stripped entirely
    # target after strip: {"tool_calls": [{"name": "move"}]}
    # checked after strip: {"tool_calls": [{"name": "move"}]}
    assert result.base_total == 7
    assert result.base_check == 7.0
    assert result.base_dcs == 1.0
    assert result.all_right is True
