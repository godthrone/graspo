"""Pure helper functions for reward target normalization and JSON validation.

Extracted from ``reward.py`` per BADGE Constitution v1.5 §8.4 (Type B):
these functions don't depend on ``GraspoReward``'s state and are independently testable.
"""

import json
from typing import Any


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
