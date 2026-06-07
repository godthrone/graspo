#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config_example.yaml}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif [[ -x "${PROJECT_DIR}/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python was not found. Set PYTHON=/path/to/python and retry." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Config: ${CONFIG_PATH}"
exec "${PYTHON_BIN}" -m graspo launch --config "${CONFIG_PATH}"
