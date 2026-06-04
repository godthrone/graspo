#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python was not found. Set PYTHON=/path/to/python and retry." >&2
  exit 1
fi

if "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  SMOKE_TMP_DIR="${SMOKE_TMP_DIR:-${ROOT_DIR}/.pytest_smoke_tmp}"
  mkdir -p "${SMOKE_TMP_DIR}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${PYTHON_BIN}" -B -m pytest tests --basetemp "${SMOKE_TMP_DIR}/basetemp"
else
  echo "pytest is not installed; running lightweight stdlib smoke checks."
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${PYTHON_BIN}" -B -c 'from graspo.core.reward import GraspoReward, RewardConfig; r=GraspoReward(RewardConfig(check_json_markdown=False)).score("{\"a\":1}", {"a":1}); assert r.all_right'
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${PYTHON_BIN}" -B -m graspo --help >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${PYTHON_BIN}" -B -m graspo validate-reward --data data/sample.jsonl --limit 2

echo "CPU smoke checks passed."
