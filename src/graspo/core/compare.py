from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompareResult:
    """Full and non-numeric comparison result for a single dict_compare_score call.

    ``dcs`` / ``total_score`` / ``check_score`` cover the complete dict including
    numeric leaf values.  ``base_dcs`` / ``base_total`` / ``base_check`` cover
    only non-numeric content (numeric leaves are stripped from *both* sides
    before comparison), which lets ``all_right`` gate on structural correctness
    without requiring exact numeric match.
    """

    dcs: float = 0.0
    total_score: int = 0
    check_score: float = 0.0

    base_dcs: float = 0.0
    base_total: int = 0
    base_check: float = 0.0

    @property
    def all_right(self) -> bool:
        return self.base_total > 0 and self.base_total == self.base_check


# ---------------------------------------------------------------------------
# Numeric-leaf stripping
# ---------------------------------------------------------------------------


def _is_numeric_leaf(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _strip_numeric(value: Any) -> Any:
    """Return a deep copy of *value* with all numeric leaf keys removed.

    - dict: recursively strip values; drop keys whose *stripped value* is an
      empty dict (i.e. all children were numeric leaves).
    - list: recursively strip each element.
    - scalar (non-numeric): returned unchanged.
    - numeric scalar: replaced with ``None`` (caller skips missing keys).
    """
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for k, v in value.items():
            sv = _strip_numeric(v)
            if sv is not None and not (isinstance(sv, dict) and not sv):
                stripped[k] = sv
        return stripped
    if isinstance(value, list):
        return [_strip_numeric(item) for item in value]
    if _is_numeric_leaf(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Denominator / numerator
# ---------------------------------------------------------------------------


def count_target_score(target: dict[str, Any], check_list_order: bool, total: int = 0) -> int:
    total += 1
    for key, value in target.items():
        total += 1
        if isinstance(value, list):
            total += 1  # the list key itself
            for elem in value:
                if isinstance(elem, dict):
                    total = count_target_score(elem, check_list_order, total)
                else:
                    total += 1
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
                element_check, element_total = _list_element_raw_score(
                    checked_value, element, check_list_order
                )
                score += element_check
                if element_total == 0 or element_check < element_total:
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


# ---------------------------------------------------------------------------
# List-element helpers
# ---------------------------------------------------------------------------


def _list_element_match_score(
    checked_items: list[Any],
    target_element: Any,
    check_list_order: bool,
) -> float:
    """Legacy normalized 0-1 score; kept for backward compatibility."""
    if not isinstance(target_element, dict):
        return 1.0 if target_element in checked_items else 0.0
    scores = [
        dict_compare_score(checked_element, target_element, check_list_order).dcs
        for checked_element in checked_items
        if isinstance(checked_element, dict)
    ]
    return max(scores, default=0.0)


def _list_element_raw_score(
    checked_items: list[Any],
    target_element: Any,
    check_list_order: bool,
) -> tuple[float, int]:
    """Return (raw_check_score, element_total) for a single target list element."""
    if not isinstance(target_element, dict):
        return (1.0, 1) if target_element in checked_items else (0.0, 0)
    best_check = 0.0
    best_total = 1
    for checked_element in checked_items:
        if isinstance(checked_element, dict):
            result = dict_compare_score(checked_element, target_element, check_list_order)
            if result.total_score > 0 and (
                best_total == 1
                and best_check == 0.0
                or result.check_score / result.total_score
                > best_check / best_total
            ):
                best_check = result.check_score
                best_total = result.total_score
    return best_check, best_total


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def dict_compare_score(
    checked: dict[str, Any],
    target: dict[str, Any],
    check_list_order: bool = False,
) -> CompareResult:
    """Compare two dicts and return :class:`CompareResult`.

    The *checked* dict is scored against the *target* dict.  Numeric leaf values
    contribute to the full ``dcs`` score but are stripped before computing
    ``base_dcs``, so ``all_right`` only requires non-numeric fields to match
    perfectly.
    """
    total_score = count_target_score(target, check_list_order)
    check_score = count_check_score(checked, target, check_list_order)
    dcs = check_score / total_score if total_score else 0.0

    # Base (non-numeric) comparison
    checked_stripped = _strip_numeric(checked)
    target_stripped = _strip_numeric(target)

    if isinstance(checked_stripped, dict) and isinstance(target_stripped, dict):
        base_total = count_target_score(target_stripped, check_list_order)
        base_check = count_check_score(checked_stripped, target_stripped, check_list_order)
        base_dcs = base_check / base_total if base_total else 0.0
    else:
        base_total = 0
        base_check = 0.0
        base_dcs = 0.0

    return CompareResult(
        dcs=dcs,
        total_score=total_score,
        check_score=check_score,
        base_dcs=base_dcs,
        base_total=base_total,
        base_check=base_check,
    )
