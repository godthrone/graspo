"""输出/任务分类判定 —— 算法层（ripple.parsing），零设施依赖。

单一真相源：logger（flow）与 monitoring（ripple）共用这些判定，
历史上有两份近似但不完全一致的实现（logger_helpers vs summary），
已归并到本模块。

- is_pure_tool_call_task: targets 是否为纯 tool-call 任务（无 content）
- summarize_json_markers / likely_truncated_json: 输出文本的 JSON 截断判定
"""

from typing import Any


def is_pure_tool_call_task(targets: Any) -> bool:
    """判断 targets 是否为纯 tool-call 任务（无 content 字段）。"""
    if not isinstance(targets, list) or not targets:
        return False
    has_content = any(
        isinstance(t, dict) and isinstance(t.get("output"), dict) and "content" in t["output"]
        for t in targets
    )
    has_tool_calls = any(
        isinstance(t, dict) and isinstance(t.get("output"), dict) and "tool_calls" in t["output"]
        for t in targets
    )
    return has_tool_calls and not has_content


def summarize_json_markers(text: str) -> dict[str, Any]:
    fence_count = text.count("```")
    has_markdown_json = "```json" in text
    return {
        "has_markdown_json": has_markdown_json,
        "fence_count": fence_count,
        "has_closing_json_fence": has_markdown_json and fence_count >= 2,
        "starts_with_object": text.lstrip().startswith("{"),
    }


def likely_truncated_json(text: str, detail: dict[str, Any] | None = None) -> bool:
    summary = summarize_json_markers(text)
    if summary["has_markdown_json"] and not summary["has_closing_json_fence"]:
        return True
    if detail and detail.get("valid_extracted_json") is False and summary["has_markdown_json"]:
        stripped = text.rstrip()
        return not (stripped.endswith("```") or stripped.endswith("}"))
    return False


def _target_tool_call_counts(targets: Any) -> set[int]:
    """计算 targets 中允许的 tool-call 数量集合（空/非列表视为 1）。"""
    counts: set[int] = set()
    if not isinstance(targets, list):
        return {1}
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        calls = output.get("tool_calls") if isinstance(output, dict) else None
        if isinstance(calls, list):
            counts.add(len(calls))
    return counts or {1}


def tool_call_count_mismatch_count(details: list[dict[str, Any]], targets: Any) -> int:
    """统计 tool-call 数量与 targets 不匹配的 completion 数量。"""
    target_counts = _target_tool_call_counts(targets)
    return sum(
        1
        for detail in details
        if detail.get("parsed_tool_calls") is not None
        and len(detail.get("parsed_tool_calls") or []) not in target_counts
    )
