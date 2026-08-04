"""GraspoReward：三层奖励评分（marker/content/target 加权，多 target 择优）。"""

import json
import math
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict

from graspo.core.schema import RewardConfig
from graspo.ripple.parsing.completion import ParsedCompletion, raw_parsed_completion
from graspo.ripple.reward.compare import dict_compare_score
from graspo.ripple.reward.normalize import (
    TargetScore,
    empty_target_score,
    is_valid_json,
    normalize_targets,
)

ContentField = Literal["answer"]
FieldItem = tuple[Literal["field"], ContentField]
CheckItem = str | FieldItem | None


class ExtractedFields(TypedDict, total=False):
    """从 completion 提取的内容字段（score 与 score_parsed 两种载荷）。"""

    answer: str
    think: str
    tool_calls: list[dict[str, Any]]
    parser: str
    parse_errors: list[str]
    extra_text: str


class RewardResult(BaseModel):
    """单条 completion 的奖励评分结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reward: float
    content_score: float
    base_content_score: float
    all_right: bool
    extracted: ExtractedFields
    useless_text: str
    raw_score: float
    max_score: float
    matched_target_index: int | None = None
    matched_target_id: str | None = None
    target_scores: list[TargetScore] | None = None


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

        target_scores: list[TargetScore] = [
            empty_target_score(target, idx) for idx, target in enumerate(normalized_targets)
        ]
        best: TargetScore | None = None
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
            for idx, score in enumerate(target_scores):
                target = normalized_targets[int(score.target_index)]
                content = target["output"].get("content")
                if not isinstance(content, dict):
                    continue
                result = dict_compare_score(
                    checked=checked,
                    target=content,
                    check_list_order=self.config.check_list_order,
                    numeric_tolerance=self.config.numeric_tolerance,
                )
                updated = score.model_copy(
                    update={
                        "content_score": result.dcs,
                        "base_content_score": result.base_dcs,
                        "all_right": result.all_right,
                    }
                )
                target_scores[idx] = updated
                if best is None or result.dcs > best.content_score:
                    best = updated
            if best is not None:
                content_score = best.content_score
                base_content_score = best.base_content_score
                if base_content_score >= 1.0:
                    # 动作类型全对 → 数值精度决定优劣
                    raw_score += content_score * self.config.content_reward_weight
                    if best.all_right:
                        raw_score += self.config.content_reward_weight
                        all_right_count += 1
                else:
                    # 动作类型有错 → 数值精度是噪声，只用 base_content_score
                    raw_score += base_content_score * self.config.content_reward_weight

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
            extracted={"answer": extracted["answer"]} if extracted else {},
            useless_text=useless_text,
            raw_score=raw_score,
            max_score=max_score,
            matched_target_index=(
                int(best.target_index) if best is not None and content_score > 0 else None
            ),
            matched_target_id=(
                str(best.target_id)
                if best is not None and best.target_id is not None and content_score > 0
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
        target_scores: list[TargetScore] = [
            empty_target_score(target, idx) for idx, target in enumerate(normalized_targets)
        ]
        best: TargetScore | None = None
        if parsed.tool_calls and think_ok:
            max_tc = self._max_target_tool_call_count(normalized_targets)
            if len(parsed.tool_calls) > max_tc:
                # 模型生成了比任何 target 都多的 tool call —— 这是格式错误。
                # 跳过 marker 奖励和 content 比较，raw_score 停留在 think_score 水平，
                # content_score 保持 0.0 → classify_group 将其标记为 RETRY/INVALID。
                # 不进入 replay buffer，不污染训练。
                parsed.parse_errors.append("too many tool calls")
            else:
                raw_score += self.config.marker_reward_weight
            if (
                len(parsed.extra_text) <= self.config.anti_useless_str_half_reward_len
                and len(parsed.tool_calls) <= max_tc
            ):
                checked = {"tool_calls": parsed.tool_calls}
                for idx, score in enumerate(target_scores):
                    target = normalized_targets[int(score.target_index)]
                    calls = target["output"].get("tool_calls")
                    if not isinstance(calls, list):
                        continue
                    result = dict_compare_score(
                        checked=checked,
                        target={"tool_calls": calls},
                        check_list_order=self.config.check_list_order,
                        numeric_tolerance=self.config.numeric_tolerance,
                    )
                    updated = score.model_copy(
                        update={
                            "content_score": result.dcs,
                            "base_content_score": result.base_dcs,
                            "all_right": result.all_right and not parsed.parse_errors,
                        }
                    )
                    target_scores[idx] = updated
                    if best is None or result.dcs > best.content_score:
                        best = updated
                if best is not None:
                    content_score = best.content_score
                    base_content_score = best.base_content_score
                    if base_content_score >= 1.0:
                        raw_score += content_score * self.config.content_reward_weight
                        all_right = best.all_right
                        if all_right:
                            raw_score += self.config.content_reward_weight
                    else:
                        raw_score += base_content_score * self.config.content_reward_weight
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
                int(best.target_index) if best is not None and content_score > 0 else None
            ),
            matched_target_id=(
                str(best.target_id)
                if best is not None and best.target_id is not None and content_score > 0
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

    @staticmethod
    def _max_target_tool_call_count(normalized_targets: list[dict[str, Any]]) -> int:
        """返回所有 target 中 tool call 数量的最大值。

        模型生成的 tool call 数量超过此值即视为格式错误，因为不存在需要
        那么多 tool call 的正确答案。少于或等于此值均不惩罚——少生成工具
        调用可能是空间感知不足，不应扣分。
        """
        max_count = 0
        for target in normalized_targets:
            calls = target["output"].get("tool_calls")
            if isinstance(calls, list):
                max_count = max(max_count, len(calls))
        return max_count

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


# ── 奖励实现注册表 ────────────────────────────────────────────────────


REWARD_REGISTRY: dict[str, type[GraspoReward]] = {"graspo": GraspoReward}


def create_reward(config: RewardConfig) -> GraspoReward:
    """按配置的 reward.kind 实例化奖励实现；未知实现报错并列出可用选项。"""
    reward_cls = REWARD_REGISTRY.get(config.kind)
    if reward_cls is None:
        raise ValueError(
            f"unknown reward backend {config.kind!r}; available: {sorted(REWARD_REGISTRY)}"
        )
    return reward_cls(config)
