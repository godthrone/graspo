#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-data/sample.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/fixed_eval_prompts.jsonl}"
LIMIT="${LIMIT:-16}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

python3 - "${DATA_PATH}" "${OUTPUT_PATH}" "${LIMIT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
limit = int(sys.argv[3])

written = 0
with data_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as out:
    for idx, line in enumerate(source):
        if not line.strip():
            continue
        record = json.loads(line)
        payload = {"id": str(record.get("id", idx))}
        if "messages" in record:
            payload["messages"] = record["messages"]
        else:
            payload["prompt"] = record.get("prompt", "")
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        written += 1
        if written >= limit:
            break

print(f"Wrote {written} fixed eval prompts to {output_path}")
PY
