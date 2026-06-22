from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NativeRolloutLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        readable_enabled: bool = True,
        raw_enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.readable_enabled = readable_enabled
        self.raw_enabled = raw_enabled
        self.readable_path = self.output_dir / "rollouts.readable.jsonl"
        self.raw_path = self.output_dir / "rollouts.raw.jsonl"
        self.train_batches_readable_path = self.output_dir / "train_batches.readable.jsonl"
        self.timing_path = self.output_dir / "timing_events.jsonl"

    def write_readable(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.readable_path, readable_payload(payload))

    def write_raw(self, payload: dict[str, Any]) -> None:
        if self.raw_enabled:
            self._append(self.raw_path, _to_jsonable(payload))

    def write_train_batch_readable(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.train_batches_readable_path, train_batch_readable_payload(payload))

    def write_timing_event(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.timing_path, timing_event_payload(payload))

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def readable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "event": payload.get("event", "graspo_group"),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "sample_index": payload.get("sample_index"),
        "attempt_index": payload.get("attempt_index"),
        "max_attempts": payload.get("max_attempts"),
        "messages": payload.get("messages"),
        "tools": payload.get("tools"),
        "prompt_preview": payload.get("prompt_preview"),
        "targets": payload.get("targets"),
        "decision": payload.get("decision"),
        "retry_count": payload.get("retry_count"),
        "reward_max_median_gap": payload.get("reward_max_median_gap"),
        "group_stats": payload.get("group_stats"),
        "group_debug": group_debug_summary(payload),
        "completions": [],
    }
    if payload.get("invalid_reason") is not None:
        compact["invalid_reason"] = payload.get("invalid_reason")
    completions = payload.get("completions", [])
    rewards = payload.get("rewards", [])
    content_scores = payload.get("content_scores", [])
    all_right = payload.get("all_right", [])
    reward_details = payload.get("reward_details", [])
    parsed_completions = payload.get("parsed_completions", [])
    generated_tokens = payload.get("generated_tokens", [])
    for idx, completion in enumerate(completions):
        detail = _get_index(reward_details, idx) or {}
        parsed = _get_index(parsed_completions, idx) or {}
        json_summary = summarize_json_markers(completion)
        compact["completions"].append(
            {
                "idx": idx,
                "completion": completion,
                "reward": _get_index(rewards, idx),
                "content_score": _get_index(content_scores, idx),
                "all_right": _get_index(all_right, idx),
                "raw_score": detail.get("raw_score"),
                "max_score": detail.get("max_score"),
                "extracted": detail.get("extracted"),
                "parsed": parsed,
                "parsed_tool_calls": detail.get("parsed_tool_calls"),
                "parser": detail.get("parser"),
                "parse_errors": detail.get("parse_errors"),
                "matched_target_index": detail.get("matched_target_index"),
                "matched_target_id": detail.get("matched_target_id"),
                "target_scores": detail.get("target_scores"),
                "useless_text_length": detail.get("useless_text_length"),
                "valid_extracted_json": detail.get("valid_extracted_json"),
                "completion_chars": len(completion),
                "generated_tokens": _get_index(generated_tokens, idx),
                "likely_truncated_json": likely_truncated_json(completion, detail),
                "has_closing_json_fence": json_summary["has_closing_json_fence"],
                "think": summarize_think(completion),
                "json": json_summary,
            }
        )
    return compact


def train_batch_readable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "graspo_train_batch",
        "backend": payload.get("backend"),
        "timestamp": payload.get("timestamp"),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "batch": payload.get("batch") or payload.get("reward_batch"),
        "optimize": payload.get("optimize"),
        "timing": payload.get("timing"),
        "health": payload.get("health"),
        "attempts": [
            train_batch_attempt_summary(attempt) for attempt in payload.get("attempts", [])
        ],
    }


def train_batch_attempt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    completions = payload.get("completions", [])
    generated_tokens = payload.get("generated_tokens", [])
    compact = {
        "event": payload.get("event", "graspo_group"),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "sample_index": payload.get("sample_index"),
        "attempt_index": payload.get("attempt_index"),
        "max_attempts": payload.get("max_attempts"),
        "decision": payload.get("decision"),
        "retry_count": payload.get("retry_count"),
        "reward_max_median_gap": payload.get("reward_max_median_gap"),
        "group_stats": payload.get("group_stats"),
        "group_debug": group_debug_summary(payload),
        "completion_count": len(completions),
        "generated_tokens_min": min(generated_tokens) if generated_tokens else None,
        "generated_tokens_max": max(generated_tokens) if generated_tokens else None,
    }
    if payload.get("invalid_reason") is not None:
        compact["invalid_reason"] = payload.get("invalid_reason")
    return compact


def timing_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "event": "timing_event",
        "timestamp": payload.get("timestamp"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "phase": payload.get("phase"),
        "duration_sec": payload.get("duration_sec"),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "sample_index": payload.get("sample_index"),
        "attempt_index": payload.get("attempt_index"),
        "retry_count": payload.get("retry_count"),
        "rank": payload.get("rank"),
        "tp_rank": payload.get("tp_rank"),
        "details": payload.get("details", {}),
    }
    return {key: value for key, value in compact.items() if value is not None}


def summarize_think(text: str) -> dict[str, Any]:
    return {
        "has_open": "<think>" in text,
        "has_close": "</think>" in text,
        "open_count": text.count("<think>"),
        "close_count": text.count("</think>"),
    }


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


def _is_pure_tool_call_task(targets: Any) -> bool:
    """Return True when targets only use tool_calls (no output.content entries)."""
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


def group_debug_summary(payload: dict[str, Any]) -> dict[str, Any]:
    completions = payload.get("completions", [])
    content_scores = payload.get("content_scores", [])
    rewards = payload.get("rewards", [])
    reward_details = payload.get("reward_details", [])
    targets = payload.get("targets")
    pure_tool_call = _is_pure_tool_call_task(targets)
    summaries = [summarize_json_markers(text) for text in completions] if not pure_tool_call else []
    target_counts = _target_tool_call_counts(targets)
    return {
        "reward_range_zero": len(set(float(value) for value in rewards)) <= 1 if rewards else True,
        "content_all_zero": bool(content_scores)
        and all(float(value) == 0.0 for value in content_scores),
        "content_all_one": bool(content_scores)
        and all(float(value) == 1.0 for value in content_scores),
        "missing_json_marker_count": (
            0 if pure_tool_call else sum(1 for item in summaries if not item["has_markdown_json"])
        ),
        "unclosed_json_fence_count": (
            0
            if pure_tool_call
            else sum(
                1
                for item in summaries
                if item["has_markdown_json"] and not item["has_closing_json_fence"]
            )
        ),
        "invalid_extracted_json_count": (
            0
            if pure_tool_call
            else sum(1 for item in reward_details if item.get("valid_extracted_json") is False)
        ),
        "likely_truncated_json_count": (
            0
            if pure_tool_call
            else sum(
                1
                for text, detail in zip(completions, reward_details, strict=False)
                if likely_truncated_json(text, detail)
            )
        ),
        "tool_call_parse_error_count": sum(
            1 for item in reward_details if item.get("parse_errors")
        ),
        "tool_call_count_mismatch_count": sum(
            1
            for item in reward_details
            if item.get("parsed_tool_calls") is not None
            and len(item.get("parsed_tool_calls") or []) not in target_counts
        ),
        "tool_call_content_all_zero": bool(content_scores)
        and any(item.get("parsed_tool_calls") is not None for item in reward_details)
        and all(float(value) == 0.0 for value in content_scores),
    }


def _get_index(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return None


def _target_tool_call_counts(targets: Any) -> set[int]:
    counts: set[int] = set()
    if not isinstance(targets, list):
        return {1}
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        calls = output.get("tool_calls") if isinstance(output, dict) else None
        if isinstance(calls, list):
            counts.add(len(calls))
    return counts or {1}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(child) for child in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
