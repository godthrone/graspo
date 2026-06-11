from __future__ import annotations

import math
from typing import Any


def count_target_score(target: dict[str, Any], check_list_order: bool, total: int = 0) -> int:
    total += 1
    for key, value in target.items():
        total += 1
        if isinstance(value, list):
            total += len(value) + 1
            if check_list_order:
                total += 1
        elif isinstance(value, dict):
            total = count_target_score(value, check_list_order, total)
        else:
            total += 1
    return total


def count_check_score(
    checked: dict[str, Any],
    target: dict[str, Any],
    check_list_order: bool,
    score: float = 0.0,
) -> float:
    for key in target:
        if key in checked:
            score += 1

    if all(key in target for key in checked):
        score += 1

    for key, target_value in target.items():
        if key not in checked:
            continue

        checked_value = checked[key]
        if isinstance(checked_value, dict) and isinstance(target_value, dict):
            score = count_check_score(checked_value, target_value, check_list_order, score)
        elif isinstance(checked_value, list) and isinstance(target_value, list):
            all_target_items_found = True
            for element in target_value:
                element_score = _list_element_match_score(checked_value, element, check_list_order)
                score += element_score
                if element_score < 1.0:
                    all_target_items_found = False

            if all(element in target_value for element in checked_value):
                score += 1
                if all_target_items_found and check_list_order and checked_value == target_value:
                    score += 1
        else:
            score += leaf_compare_score(checked_value, target_value)
    return score


def leaf_compare_score(checked: Any, target: Any) -> float:
    if isinstance(checked, bool) or isinstance(target, bool):
        return 1.0 if type(checked) is bool and type(target) is bool and checked == target else 0.0
    if _is_json_number(checked) and _is_json_number(target):
        checked_float = float(checked)
        target_float = float(target)
        if not (math.isfinite(checked_float) and math.isfinite(target_float)):
            return 1.0 if checked == target else 0.0
        return 1.0 / (1.0 + abs(checked_float - target_float))
    return 1.0 if checked == target else 0.0


def _list_element_match_score(
    checked_items: list[Any],
    target_element: Any,
    check_list_order: bool,
) -> float:
    if not isinstance(target_element, dict):
        return 1.0 if target_element in checked_items else 0.0
    scores = [
        dict_compare_score(checked_element, target_element, check_list_order)[0]
        for checked_element in checked_items
        if isinstance(checked_element, dict)
    ]
    return max(scores, default=0.0)


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def dict_compare_score(
    checked: dict[str, Any],
    target: dict[str, Any],
    check_list_order: bool = False,
) -> tuple[float, int, float]:
    total_score = count_target_score(target, check_list_order)
    check_score = count_check_score(checked, target, check_list_order)
    return check_score / total_score if total_score else 0.0, total_score, check_score
