from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from graspo.core.compare import CompareResult, dict_compare_score
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


@dataclass(slots=True)
class RewardResult:
    reward: float
    content_score: float
    base_content_score: float
    all_right: bool
    extracted: dict[str, Any]
    useless_text: str
    raw_score: float
    max_score: float
    matched_target_index: int | None = None
    matched_target_id: str | None = None
    target_scores: list[dict[str, Any]] | None = None


def is_valid_json(value: str) -> bool:
    try:
        json.loads(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("targets must be a non-empty list")
    return [_normalize_target(item, idx) for idx, item in enumerate(value)]


def _normalize_target(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"targets[{index}] must be a JSON object")
    target_id = value.get("id")
    if target_id is not None and not isinstance(target_id, str):
        raise ValueError(f"targets[{index}].id must be a string when provided")
    output = value.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"targets[{index}].output must be a JSON object")
    normalized_output: dict[str, Any] = {}
    if "content" in output:
        content = output["content"]
        if not isinstance(content, dict):
            raise ValueError(f"targets[{index}].output.content must be a JSON object")
        normalized_output["content"] = dict(content)
    if "tool_calls" in output:
        normalized_output["tool_calls"] = normalize_tool_calls(
            output["tool_calls"], path=f"targets[{index}].output.tool_calls"
        )
    if not normalized_output:
        raise ValueError(
            f"targets[{index}].output must contain content and/or non-empty tool_calls"
        )
    return {"id": target_id, "output": normalized_output}


def normalize_tool_calls(value: Any, *, path: str = "tool_calls") -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for idx, call in enumerate(value):
        if not isinstance(call, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}[{idx}].name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ValueError(f"{path}[{idx}].arguments must be a JSON object")
        normalized.append({"name": name, "arguments": dict(arguments)})
    return normalized


class GraspoReward:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def score(self, completion: str, targets: Any) -> RewardResult:
        normalized_targets = normalize_targets(targets)
        content_targets = [target for target in normalized_targets if "content" in target["output"]]
        check_targets: dict[ContentField, dict[str, Any]] = (
            {"answer": content_targets[0]["output"]["content"]} if content_targets else {}
        )

        check_list = self._build_check_list(check_targets)
        max_score = self._max_reward(check_list)
        raw_score = 0.0
        content_score = 0.0
        base_content_score = 0.0
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

        target_scores: list[dict[str, Any]] = [
            _empty_target_score(target, idx) for idx, target in enumerate(normalized_targets)
        ]
        best: dict[str, Any] | None = None
        for key in check_targets:
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
            for score in target_scores:
                target = normalized_targets[int(score["target_index"])]
                content = target["output"].get("content")
                if not isinstance(content, dict):
                    continue
                result = dict_compare_score(
                    checked=checked,
                    target=content,
                    check_list_order=self.config.check_list_order,
                )
                score.update(
                    {
                        "content_score": result.dcs,
                        "base_content_score": result.base_dcs,
                        "all_right": result.all_right,
                    }
                )
                if best is None or result.dcs > float(best["content_score"]):
                    best = score
            if best is not None:
                content_score = float(best["content_score"])
                base_content_score = float(best.get("base_content_score", 0.0))
                raw_score += content_score * self.config.content_reward_weight
                if bool(best["all_right"]):
                    raw_score += self.config.content_reward_weight
                    all_right_count += 1

        raw_score += self._useless_text_score(useless_text)
        all_right = all_right_count > 0

        # Match the original GRASPO implementation: the anti-useless bonus is
        # added after normalization's max-score denominator is computed, so a
        # clean perfect answer can be slightly above 1.0.
        normalized_reward = raw_score / max_score if max_score else 0.0

        return RewardResult(
            reward=normalized_reward,
            content_score=content_score,
            base_content_score=base_content_score,
            all_right=all_right,
            extracted={key: value for key, value in extracted.items()},
            useless_text=useless_text,
            raw_score=raw_score,
            max_score=max_score,
            matched_target_index=(
                int(best["target_index"]) if best is not None and content_score > 0 else None
            ),
            matched_target_id=(
                str(best["target_id"])
                if best is not None and best.get("target_id") is not None and content_score > 0
                else None
            ),
            target_scores=target_scores,
        )

    def score_parsed(
        self,
        parsed: ParsedCompletion | str,
        targets: Any,
        *,
        is_tool_call: bool = False,
    ) -> RewardResult:
        if isinstance(parsed, str):
            parsed = raw_parsed_completion(parsed)
        if not is_tool_call:
            return self.score(parsed.raw_text, targets)
        normalized_targets = normalize_targets(targets)
        think_score, think_ok = self._think_marker_score(parsed.raw_text)
        max_score = self._tool_call_max_reward()
        raw_score = think_score
        content_score = 0.0
        base_content_score = 0.0
        all_right = False
        target_scores: list[dict[str, Any]] = [
            _empty_target_score(target, idx) for idx, target in enumerate(normalized_targets)
        ]
        best: dict[str, Any] | None = None
        if parsed.tool_calls and think_ok:
            raw_score += self.config.marker_reward_weight
            if len(parsed.extra_text) <= self.config.anti_useless_str_half_reward_len:
                checked = {"tool_calls": parsed.tool_calls}
                for score in target_scores:
                    target = normalized_targets[int(score["target_index"])]
                    calls = target["output"].get("tool_calls")
                    if not isinstance(calls, list):
                        continue
                    result = dict_compare_score(
                        checked=checked,
                        target={"tool_calls": calls},
                        check_list_order=True,
                    )
                    score.update(
                        {
                            "content_score": result.dcs,
                            "base_content_score": result.base_dcs,
                            "all_right": result.all_right and not parsed.parse_errors,
                        }
                    )
                    if best is None or result.dcs > float(best["content_score"]):
                        best = score
                if best is not None:
                    content_score = float(best["content_score"])
                    base_content_score = float(best.get("base_content_score", 0.0))
                raw_score += content_score * self.config.content_reward_weight
                all_right = bool(best and best["all_right"])
                if all_right:
                    raw_score += self.config.content_reward_weight
        raw_score += self._useless_text_score(parsed.extra_text)
        normalized_reward = raw_score / max_score if max_score else 0.0
        return RewardResult(
            reward=normalized_reward,
            content_score=content_score,
            base_content_score=base_content_score,
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
            matched_target_index=(
                int(best["target_index"]) if best is not None and content_score > 0 else None
            ),
            matched_target_id=(
                str(best["target_id"])
                if best is not None and best.get("target_id") is not None and content_score > 0
                else None
            ),
            target_scores=target_scores,
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


def _empty_target_score(target: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "target_index": index,
        "target_id": target.get("id"),
        "content_score": 0.0,
        "base_content_score": 0.0,
        "all_right": False,
    }
