from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graspo.core.schema import Sample


def sample_from_record(record: dict[str, Any]) -> Sample:
    removed_input_fields = {"prompt", "image", "images", "video", "videos"}
    present_removed = sorted(field for field in removed_input_fields if field in record)
    if present_removed:
        raise ValueError(
            "removed input field(s): "
            + ", ".join(present_removed)
            + "; use messages + optional tools + ground_truth JSONL"
        )
    messages = _validate_messages(record.get("messages"))

    if "ground_truth" in record:
        ground_truth = record["ground_truth"]
    else:
        raise ValueError("record must contain 'ground_truth'")
    if not isinstance(ground_truth, dict):
        raise ValueError("record 'ground_truth' must be a JSON object")

    tools = _validate_tools(record.get("tools"))
    media = _messages_media(messages)
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"messages", "ground_truth", "tools"}
    }
    return Sample(
        messages=messages,
        ground_truth=ground_truth,
        tools=tools,
        metadata=metadata,
        media=media,
    )


def _validate_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("record must contain a non-empty 'messages' list")
    messages: list[dict[str, Any]] = []
    for idx, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        role = str(message.get("role") or "").strip()
        if not role:
            raise ValueError(f"messages[{idx}].role is required")
        if "content" not in message:
            raise ValueError(f"messages[{idx}].content is required")
        messages.append(dict(message))
    if str(messages[-1].get("role") or "").lower() == "assistant":
        raise ValueError(
            "messages must be prompt/context only; final assistant messages leak the target"
        )
    return messages


def _validate_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("record 'tools' must be a list of JSON objects")
    tools: list[dict[str, Any]] = []
    for idx, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{idx}] must be an object")
        tools.append(dict(tool))
    return tools


def _messages_media(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for message in messages:
        _, content_media = _content_to_text_and_media(message.get("content", ""))
        media.extend(content_media)
    return _dedupe_media(media)


def _content_to_text_and_media(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content or ""), []
    parts: list[str] = []
    media: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type == "text":
            parts.append(str(item.get("text") or ""))
        elif item_type in {"image", "image_url"}:
            path = item.get("image") or item.get("path") or item.get("url")
            if isinstance(item.get("image_url"), dict):
                path = item["image_url"].get("url") or path
            if path:
                media.append({"type": "image", "path": str(path)})
            parts.append("<image>")
        elif item_type in {"video", "video_url"}:
            path = item.get("video") or item.get("path") or item.get("url")
            if isinstance(item.get("video_url"), dict):
                path = item["video_url"].get("url") or path
            if path:
                media.append({"type": "video", "path": str(path)})
            parts.append("<video>")
        else:
            raise ValueError(f"unsupported multimodal content type: {item_type!r}")
    return "\n".join(part for part in parts if part), media


def _dedupe_media(media: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in media:
        media_type = str(item.get("type") or "")
        path = str(item.get("path") or "")
        key = (media_type, path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def load_jsonl(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                samples.append(sample_from_record(record))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid JSONL record at {path}:{line_no}: {exc}") from exc
    return samples


def write_jsonl(samples: list[Sample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")
