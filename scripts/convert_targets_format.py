#!/usr/bin/env python3
"""Convert json_output/train.jsonl targets from JSON string format to dict format.

Usage:
    python scripts/convert_targets_format.py samples/data/json_output/train.jsonl

The json_output data has targets stored as JSON-encoded strings:
    "targets": ["{\"name\": \"Alice\", ...}"]

This converts them to the dict format expected by the code:
    "targets": [{"id": "expected", "output": {"content": {"name": "Alice", ...}}}]

Already-correct targets (dict lists) are left unchanged.
The file is modified in-place; a backup is written to <file>.bak.
"""

import json
import shutil
import sys
from pathlib import Path


def is_json_string_target(target: object) -> bool:
    """Return True if target is a JSON string that should be parsed."""
    return isinstance(target, str) and target.strip().startswith("{")


def convert_targets(targets: list) -> list:
    """Convert a list of JSON string targets to dict format.

    If targets are already dicts (correct format), return unchanged.
    """
    if not targets:
        return targets

    # If first element is already a dict, assume correct format
    if isinstance(targets[0], dict):
        return targets

    converted = []
    for t in targets:
        if is_json_string_target(t):
            parsed = json.loads(t)
            converted.append(
                {
                    "id": "expected",
                    "output": {"content": parsed},
                }
            )
        elif isinstance(t, dict):
            converted.append(t)
        else:
            # Unknown format — keep as-is
            converted.append(t)
    return converted


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <jsonl_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    # Read all lines
    lines = input_path.read_text(encoding="utf-8").strip().split("\n")
    records = []
    converted_count = 0
    for line in lines:
        if not line.strip():
            records.append(line)
            continue
        record = json.loads(line)
        if "targets" in record:
            original = record["targets"]
            record["targets"] = convert_targets(original)
            if record["targets"] != original:
                converted_count += 1
        records.append(json.dumps(record, ensure_ascii=False))

    if converted_count == 0:
        print("No targets needed conversion — file is already in correct format.")
        return

    # Backup original
    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
    shutil.copy2(input_path, backup_path)

    # Write converted
    input_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    print(f"Converted {converted_count}/{len(lines)} records.")
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()
