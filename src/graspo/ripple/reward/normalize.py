"""Pure helper functions for reward target normalization and JSON validation.

Extracted from ``reward.py`` (Type B helpers):
these functions don't depend on ``GraspoReward``'s state and are independently testable.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class TargetScore(BaseModel):
    """单个 target 的评分结果（empty_target_score 与 GraspoReward 共用的数据结构）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_index: int
    target_id: str | None = None
    content_score: float = 0.0
    base_content_score: float = 0.0
    all_right: bool = False


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
    # ``reasoning``（教师推理链，ARD ``targets[i].output.reasoning``）与 content
    # 平级保留：它必须存在于 ``output`` **之内**，否则归一化会静默丢弃。
    # 保留是 D4 共享基座契约的一部分（ARD 无 logprob 是特性，reasoning 不是）。
    # 空串与缺失都归一为 None（宪法 §2.2：None 是唯一合法空值）。
    reasoning = output.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise ValueError(f"targets[{index}].output.reasoning must be a string when provided")
        normalized_output["reasoning"] = reasoning or None
    return {"id": target_id, "output": normalized_output}


def empty_target_score(target: dict[str, Any], index: int) -> TargetScore:
    """Return a zeroed score entry for a single target.

    This is used as the initial state before scoring populates real values.
    It does not depend on ``GraspoReward`` state and is independently testable.
    """
    return TargetScore(
        target_index=index,
        target_id=target.get("id"),
    )


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
