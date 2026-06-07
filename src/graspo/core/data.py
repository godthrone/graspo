from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graspo.core.schema import Sample


def prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "assistant":
            continue
        content, _ = _content_to_text_and_media(message.get("content", ""))
        parts.append(content)
    return "\n\n".join(part for part in parts if part)


def sample_from_record(
    record: dict[str, Any],
    prompt_field: str = "prompt",
    ground_truth_field: str = "ground_truth",
    messages_field: str = "messages",
) -> Sample:
    if prompt_field in record:
        prompt = str(record[prompt_field])
    elif messages_field in record:
        prompt = prompt_from_messages(record[messages_field])
    else:
        raise ValueError(f"record must contain '{prompt_field}' or '{messages_field}'")

    if ground_truth_field in record:
        ground_truth = record[ground_truth_field]
    elif messages_field in record:
        assistant_messages = [
            message for message in record[messages_field] if message.get("role") == "assistant"
        ]
        if not assistant_messages:
            raise ValueError("messages record has no assistant ground truth")
        ground_truth = assistant_messages[-1].get("content", "{}")
    elif "output" in record:
        ground_truth = record["output"]
    else:
        raise ValueError(
            f"record must contain '{ground_truth_field}', assistant message, or 'output'"
        )

    media = _record_media(record, messages_field=messages_field)
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {prompt_field, ground_truth_field, "image", "images", "video", "videos"}
    }
    return Sample(prompt=prompt, ground_truth=ground_truth, metadata=metadata, media=media)


def _record_media(record: dict[str, Any], *, messages_field: str) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    if messages_field in record and isinstance(record[messages_field], list):
        for message in record[messages_field]:
            _, content_media = _content_to_text_and_media(message.get("content", ""))
            media.extend(content_media)
    for field_name, media_type in (("image", "image"), ("video", "video")):
        value = record.get(field_name)
        if value:
            media.append({"type": media_type, "path": str(value)})
    for field_name, media_type in (("images", "image"), ("videos", "video")):
        values = record.get(field_name) or []
        if isinstance(values, (str, bytes)):
            values = [values]
        for value in values:
            if value:
                media.append({"type": media_type, "path": str(value)})
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
        elif item_type == "video":
            path = item.get("video") or item.get("path") or item.get("url")
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


def load_jsonl(
    path: str | Path,
    prompt_field: str = "prompt",
    ground_truth_field: str = "ground_truth",
    messages_field: str = "messages",
) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                samples.append(
                    sample_from_record(
                        record,
                        prompt_field=prompt_field,
                        ground_truth_field=ground_truth_field,
                        messages_field=messages_field,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid JSONL record at {path}:{line_no}: {exc}") from exc
    return samples


def load_json(
    path: str | Path,
    prompt_field: str = "prompt",
    ground_truth_field: str = "ground_truth",
    messages_field: str = "messages",
) -> list[Sample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list or an object with a 'data' list")
    return [
        sample_from_record(
            record,
            prompt_field=prompt_field,
            ground_truth_field=ground_truth_field,
            messages_field=messages_field,
        )
        for record in data
    ]


def write_jsonl(samples: list[Sample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")


def convert_excel_to_samples(path: str | Path) -> list[Sample]:
    import pandas as pd

    rows: list[Sample] = []
    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str).fillna("")
        for _, row in df.iterrows():
            record = row.to_dict()
            instruction = str(record.get("instruction", "")).strip()
            user_input = str(record.get("input", "")).strip()
            output = str(record.get("output", "")).strip()
            if not user_input or not output:
                continue
            prompt = f"{instruction}\n\n{user_input}".strip()
            rows.append(
                Sample(
                    prompt=prompt,
                    ground_truth=output,
                    metadata={"sheet": sheet_name},
                )
            )
    return rows
