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

