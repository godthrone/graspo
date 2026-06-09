from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from graspo.core.compare import dict_compare_score
from graspo.core.completion import ParsedCompletion, raw_parsed_completion


ContentField = Literal["answer"]
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
    extracted: dict[str, Any]
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
    if not isinstance(value, dict):
        raise ValueError("ground_truth must be a JSON object")
    return value


def normalize_tool_call_target(value: Any) -> list[dict[str, Any]]:
    calls = value if isinstance(value, list) else [value]
    if not isinstance(calls, list) or not calls:
        raise ValueError("tool-call ground_truth must be a canonical object or non-empty list")
    normalized: list[dict[str, Any]] = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"tool-call ground_truth[{idx}] must be a JSON object")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool-call ground_truth[{idx}].name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ValueError(f"tool-call ground_truth[{idx}].arguments must be a JSON object")
        normalized.append({"name": name, "arguments": dict(arguments)})
    return normalized


class GraspoReward:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def score(self, completion: str, ground_truth: Any) -> RewardResult:
        answer_target = normalize_ground_truth(ground_truth)
        targets: dict[ContentField, dict[str, Any]] = {"answer": answer_target}

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

        # Match the original GRASPO implementation: the anti-useless bonus is
        # added after normalization's max-score denominator is computed, so a
        # clean perfect answer can be slightly above 1.0.
        normalized_reward = raw_score / max_score if max_score else 0.0

        return RewardResult(
            reward=normalized_reward,
            content_score=content_score,
            all_right=all_right,
            extracted={key: value for key, value in extracted.items()},
            useless_text=useless_text,
            raw_score=raw_score,
            max_score=max_score,
        )

    def score_parsed(
        self,
        parsed: ParsedCompletion | str,
        ground_truth: Any,
        *,
        is_tool_call: bool = False,
    ) -> RewardResult:
        if isinstance(parsed, str):
            parsed = raw_parsed_completion(parsed)
        if not is_tool_call:
            return self.score(parsed.raw_text, ground_truth)
        target_calls = normalize_tool_call_target(ground_truth)
        think_score, think_ok = self._think_marker_score(parsed.raw_text)
        max_score = self._tool_call_max_reward()
        raw_score = think_score
        content_score = 0.0
        all_right = False
        if parsed.tool_calls and think_ok:
            raw_score += self.config.marker_reward_weight
            if len(parsed.extra_text) <= self.config.anti_useless_str_half_reward_len:
                checked = {"tool_calls": parsed.tool_calls}
                target = {"tool_calls": target_calls}
                dcs, total_score, check_score = dict_compare_score(
                    checked=checked,
                    target=target,
                    check_list_order=True,
                )
                raw_score += dcs * self.config.content_reward_weight
                content_score = dcs
                all_right = total_score == check_score and not parsed.parse_errors
                if all_right:
                    raw_score += self.config.content_reward_weight
        raw_score += self._useless_text_score(parsed.extra_text)
        normalized_reward = raw_score / max_score if max_score else 0.0
        return RewardResult(
            reward=normalized_reward,
            content_score=content_score,
            all_right=all_right,
            extracted={
                "tool_calls": parsed.tool_calls,
                "think": parsed.think_text,
                "answer": parsed.answer_text,
                "parser": parsed.parser_name,
                "parse_errors": parsed.parse_errors,
                "extra_text": parsed.extra_text,
            },
            useless_text=parsed.extra_text,
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

        return check_list

    def _max_reward(self, check_list: list[CheckItem]) -> float:
        total = 0.0
        for item in check_list:
            if isinstance(item, str):
                total += self.config.marker_reward_weight
            elif item is not None:
                total += self.config.content_reward_weight * 2 + self.config.marker_reward_weight
        return total

    def _tool_call_max_reward(self) -> float:
        total = self.config.content_reward_weight * 2 + self.config.marker_reward_weight
        if self.config.check_think:
            total += self.config.marker_reward_weight * 2
        return total

    def _think_marker_score(self, text: str) -> tuple[float, bool]:
        if not self.config.check_think:
            return 0.0, True
        open_pos = text.find("<think>")
        close_pos = text.find("</think>", open_pos + len("<think>")) if open_pos >= 0 else -1
        if open_pos >= 0 and close_pos >= 0:
            return self.config.marker_reward_weight * 2, True
        return 0.0, False

    def _useless_text_score(self, useless_text: str) -> float:
        return self.config.anti_useless_str_reward_weight / math.pow(
            2,
            len(useless_text) / self.config.anti_useless_str_half_reward_len,
        )
