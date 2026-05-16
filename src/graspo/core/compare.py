from __future__ import annotations

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
    score: int = 0,
) -> int:
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
                if element in checked_value:
                    score += 1
                else:
                    all_target_items_found = False

            if all(element in target_value for element in checked_value):
                score += 1
                if all_target_items_found and check_list_order and checked_value == target_value:
                    score += 1
        elif checked_value == target_value:
            score += 1
    return score


def dict_compare_score(
    checked: dict[str, Any],
    target: dict[str, Any],
    check_list_order: bool = False,
) -> tuple[float, int, int]:
    total_score = count_target_score(target, check_list_order)
    check_score = count_check_score(checked, target, check_list_order)
    return check_score / total_score if total_score else 0.0, total_score, check_score
