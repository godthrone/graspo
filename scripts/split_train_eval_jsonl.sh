#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-data/sample.jsonl}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-outputs/train.jsonl}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-outputs/eval.jsonl}"
MANIFEST_OUTPUT_PATH="${MANIFEST_OUTPUT_PATH:-outputs/train_eval_split_manifest.json}"
EVAL_COUNT="${EVAL_COUNT:-48}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "${TRAIN_OUTPUT_PATH}")" "$(dirname "${EVAL_OUTPUT_PATH}")" "$(dirname "${MANIFEST_OUTPUT_PATH}")"

python3 - "${SOURCE_DATA_PATH}" "${TRAIN_OUTPUT_PATH}" "${EVAL_OUTPUT_PATH}" "${MANIFEST_OUTPUT_PATH}" "${EVAL_COUNT}" "${SEED}" <<'PY'
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
eval_path = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
eval_count = int(sys.argv[5])
seed = int(sys.argv[6])

records: list[str] = []
with source_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(line.rstrip("\n"))

if not records:
    raise SystemExit(f"No JSONL records found in {source_path}")
if not 0 < eval_count < len(records):
    raise SystemExit(f"EVAL_COUNT must be in [1, {len(records) - 1}], got {eval_count}")

indices = list(range(len(records)))
random.Random(seed).shuffle(indices)
eval_indices = set(indices[:eval_count])

with train_path.open("w", encoding="utf-8") as train_file, eval_path.open("w", encoding="utf-8") as eval_file:
    for idx, line in enumerate(records):
        target = eval_file if idx in eval_indices else train_file
        target.write(line + "\n")

manifest = {
    "source_data_path": str(source_path),
    "train_output_path": str(train_path),
    "eval_output_path": str(eval_path),
    "total_count": len(records),
    "train_count": len(records) - eval_count,
    "eval_count": eval_count,
    "seed": seed,
    "eval_indices": sorted(eval_indices),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k != "eval_indices"}, ensure_ascii=False))
PY
