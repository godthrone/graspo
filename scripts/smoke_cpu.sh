#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if python -c "import pytest" >/dev/null 2>&1; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m pytest tests
else
  echo "pytest is not installed; running lightweight stdlib smoke checks."
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -c 'from graspo.core.reward import GraspoReward, RewardConfig; r=GraspoReward(RewardConfig(check_json_markdown=False)).score("{\"a\":1}", {"a":1}); assert r.all_right'
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m graspo --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m graspo validate-reward --data data/sample.jsonl --limit 2

echo "CPU smoke checks passed."
