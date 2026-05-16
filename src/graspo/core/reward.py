from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from graspo.core.compare import dict_compare_score


ContentField = Literal["answer", "tool_call"]
FieldItem = tuple[Literal["field"], ContentField]
CheckItem = str | FieldItem | None


@dataclass(slots=True)
class RewardConfig:
    check_think: bool = False
    check_json_markdown: bool = True
    check_tool_call: bool = False
    check_list_order: bool = False
    marker_reward_weight: float = 10.0
    content_reward_weight: float = 100.0
    anti_useless_str_reward_weight: float = 1.0
    anti_useless_str_half_reward_len: int = 100
    answer_field: str = "ground_truth"


@dataclass(slots=True)
class RewardResult:
    reward: float
    content_score: float
    all_right: bool
    extracted: dict[str, str]
    useless_text: str
    raw_score: float
    max_score: float


def is_valid_json(value: str) -> bool:
    try:
        json.loads(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_ground_truth(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("ground_truth must be a JSON object, a JSON object string, or a non-empty list")
    return parsed


class GraspoReward:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def score(self, completion: str, ground_truth: Any, tool_call: Any | None = None) -> RewardResult:
        answer_target = normalize_ground_truth(ground_truth)
        targets: dict[ContentField, dict[str, Any]] = {"answer": answer_target}
        if self.config.check_tool_call and tool_call not in (None, ""):
            targets["tool_call"] = normalize_ground_truth(tool_call)

        check_list = self._build_check_list(targets)
        max_score = self._max_reward(check_list)
        raw_score = 0.0
        content_score = 0.0
        extracted: dict[ContentField, str] = {}
        useless_text = ""
        content_type: ContentField | None = None
        mark_pos = 0
        all_right_count = 0

        for check_item in check_list:
            if isinstance(check_item, str):
                check_pos = completion.find(check_item, mark_pos)
                if check_pos < 0:
                    extracted.clear()
                    break

                raw_score += self.config.marker_reward_weight
                if content_type is None:
                    useless_text += completion[mark_pos:check_pos]
                else:
                    extracted[content_type] = completion[mark_pos:check_pos]
                mark_pos = check_pos + len(check_item)
                content_type = None
            elif check_item is None:
                content_type = None
            else:
                content_type = check_item[1]

        if content_type is None:
            useless_text += completion[mark_pos:]
        else:
            extracted[content_type] = completion[mark_pos:]

        for key, target in targets.items():
            if key not in extracted:
                continue
            text = extracted[key].strip()
            if not is_valid_json(text):
                continue

            raw_score += self.config.marker_reward_weight
            if len(useless_text) > self.config.anti_useless_str_half_reward_len:
                continue

            checked = json.loads(text)
            if not isinstance(checked, dict):
                continue
            dcs, total_score, check_score = dict_compare_score(
                checked=checked,
                target=target,
                check_list_order=self.config.check_list_order,
            )
            raw_score += dcs * self.config.content_reward_weight
            content_score += dcs
            if total_score == check_score:
                raw_score += self.config.content_reward_weight
                all_right_count += 1

        content_score /= len(targets) if targets else 1
        raw_score += self._useless_text_score(useless_text)
        all_right = all_right_count == len(targets)

        return RewardResult(
            reward=raw_score / max_score if max_score else 0.0,
            content_score=content_score,
            all_right=all_right,
            extracted={key: value for key, value in extracted.items()},
            useless_text=useless_text,
            raw_score=raw_score,
            max_score=max_score,
        )

    def _build_check_list(self, targets: dict[ContentField, dict[str, Any]]) -> list[CheckItem]:
        check_list: list[CheckItem] = []
        if self.config.check_think:
            check_list.extend(["<think>", None, "</think>"])

        if "answer" in targets:
            if check_list:
                check_list.append(None)
            if self.config.check_json_markdown:
                check_list.extend(["```json", ("field", "answer"), "```"])
            else:
                check_list.append(("field", "answer"))

        if "tool_call" in targets:
            if check_list:
                check_list.append(None)
            check_list.extend(["<tool_call>", ("field", "tool_call"), "</tool_call>"])

        return check_list

    def _max_reward(self, check_list: list[CheckItem]) -> float:
        total = 0.0
        for item in check_list:
            if isinstance(item, str):
                total += self.config.marker_reward_weight
            elif item is not None:
                total += self.config.content_reward_weight * 2 + self.config.marker_reward_weight
        return total

    def _useless_text_score(self, useless_text: str) -> float:
        return self.config.anti_useless_str_reward_weight / math.pow(
            2,
            len(useless_text) / self.config.anti_useless_str_half_reward_len,
        )
