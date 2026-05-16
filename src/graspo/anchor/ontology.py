from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Ontology:
    leaves: list[dict[str, Any]]


def _extract_leaves(node: Any, path: list[str] | None = None) -> list[dict[str, Any]]:
    path = path or []
    leaves: list[dict[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            leaves.extend(_extract_leaves(value, path + [str(key)]))
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                full_path = path + [item]
                leaves.append(
                    {
                        "full_path": " -> ".join(full_path),
                        "top_level": path[0] if path else item,
                        "path": path,
                        "leaf": item,
                    }
                )
            else:
                leaves.extend(_extract_leaves(item, path))
    elif isinstance(node, str):
        full_path = path + [node]
        leaves.append(
            {
                "full_path": " -> ".join(full_path),
                "top_level": path[0] if path else node,
                "path": path,
                "leaf": node,
            }
        )
    return leaves


def load_ontology(path: str | Path, root_key: str | None = None) -> Ontology:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if root_key and isinstance(data, dict) and root_key in data:
        data = data[root_key]
    return Ontology(leaves=_extract_leaves(data))

