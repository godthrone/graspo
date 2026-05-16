from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SFTSample:
    sample_type: str
    messages: list[dict[str, str]]
    target: str
    metadata: dict[str, Any]


def _messages_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in record:
        messages = list(record["messages"])
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]
    if "prompt" in record:
        return [{"role": "user", "content": str(record["prompt"])}]
    raise ValueError("record must contain messages or prompt")


def hard_sample_from_record(record: dict[str, Any]) -> SFTSample:
    target = record.get("target", record.get("ground_truth", record.get("output")))
    if target is None:
        raise ValueError("hard sample must contain target, ground_truth, or output")
    if not isinstance(target, str):
        target = json.dumps(target, ensure_ascii=False)
    return SFTSample(
        sample_type="hard",
        messages=_messages_from_record(record),
        target=target,
        metadata={key: value for key, value in record.items() if key not in {"messages", "prompt", "target"}},
    )


def anchor_sample_from_record(record: dict[str, Any]) -> SFTSample:
    return SFTSample(
        sample_type="anchor",
        messages=_messages_from_record(record),
        target=str(record.get("teacher_answer", "")),
        metadata=dict(record.get("anchor_meta", {})),
    )


def load_sft_samples(path: str | Path, sample_type: str) -> list[SFTSample]:
    items: list[SFTSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                if sample_type == "hard":
                    items.append(hard_sample_from_record(record))
                elif sample_type == "anchor":
                    items.append(anchor_sample_from_record(record))
                else:
                    raise ValueError(f"unsupported sample_type: {sample_type}")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid {sample_type} sample at {path}:{line_no}: {exc}") from exc
    return items


def load_mixed_sft_samples(hard_path: str | Path, anchor_path: str | Path) -> list[SFTSample]:
    hard = load_sft_samples(hard_path, "hard")
    anchor = load_sft_samples(anchor_path, "anchor")
    mixed: list[SFTSample] = []
    max_len = max(len(hard), len(anchor))
    for idx in range(max_len):
        if idx < len(hard):
            mixed.append(hard[idx])
        if idx < len(anchor):
            mixed.append(anchor[idx])
    return mixed

