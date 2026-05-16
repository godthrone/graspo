from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graspo.anchor.bank import AnsweredAnchor, FilterStats


def build_anchor_manifest(
    teacher_model: str,
    generation_config: dict[str, Any],
    anchors: list[AnsweredAnchor],
    filter_stats: FilterStats | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    domains = Counter(anchor.anchor_meta.get("knowledge_top_level", "unknown") for anchor in anchors)
    tasks = Counter(anchor.anchor_meta.get("task_type", "unknown") for anchor in anchors)
    languages = Counter(anchor.anchor_meta.get("language", "unknown") for anchor in anchors)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_model": teacher_model,
        "generation_config": generation_config,
        "seed": seed,
        "counts": {
            "total": len(anchors),
            "by_domain": dict(domains),
            "by_task_type": dict(tasks),
            "by_language": dict(languages),
        },
        "filter_stats": filter_stats.to_dict() if filter_stats else None,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

