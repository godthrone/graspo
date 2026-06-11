import pytest

from graspo.core.compare import dict_compare_score


def test_dict_compare_exact():
    score, total, checked = dict_compare_score({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]})

    assert score == 1.0
    assert total == checked


def test_dict_compare_partial():
    score, total, checked = dict_compare_score({"a": 1, "b": [1]}, {"a": 1, "b": [1, 2]})

    assert 0 < score < 1
    assert checked < total


def test_dict_compare_list_order_optional():
    unordered, _, _ = dict_compare_score(
        {"items": [2, 1]},
        {"items": [1, 2]},
        check_list_order=False,
    )
    ordered, _, _ = dict_compare_score(
        {"items": [2, 1]},
        {"items": [1, 2]},
        check_list_order=True,
    )

    assert unordered > ordered


def test_dict_compare_numeric_leaf_uses_absolute_error_score():
    score, total, checked = dict_compare_score(
        {"distance_cm": 8},
        {"distance_cm": 6},
    )

    assert total == 3
    assert checked == pytest.approx(2 + 1 / 3)
    assert score == pytest.approx((2 + 1 / 3) / 3)


def test_dict_compare_numeric_string_type_mismatch_gets_no_leaf_score():
    score, total, checked = dict_compare_score(
        {"distance_cm": "6"},
        {"distance_cm": 6},
    )

    assert total == 3
    assert checked == 2
    assert score == pytest.approx(2 / 3)


def test_dict_compare_bool_is_not_numeric():
    _, total, checked = dict_compare_score({"enabled": True}, {"enabled": 1})

    assert total == 3
    assert checked == 2


def test_dict_compare_int_float_exact_match_is_all_right():
    _, total, checked = dict_compare_score({"distance_cm": 6}, {"distance_cm": 6.0})

    assert total == checked


def test_dict_compare_list_dict_element_uses_nested_numeric_score():
    score, total, checked = dict_compare_score(
        {"tool_calls": [{"name": "move", "arguments": {"distance_cm": 8}}]},
        {"tool_calls": [{"name": "move", "arguments": {"distance_cm": 6}}]},
        check_list_order=True,
    )

    element_score = (6 + 1 / 3) / 7
    assert total == 5
    assert checked == pytest.approx(2 + element_score)
    assert score == pytest.approx((2 + element_score) / 5)
