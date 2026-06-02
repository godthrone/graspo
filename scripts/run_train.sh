#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-native-tp}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/sample.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/run}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/graspo.yaml}"
TP_SIZE="${TP_SIZE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

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

if [[ -n "${TORCHRUN:-}" ]]; then
  TORCHRUN_BIN="${TORCHRUN}"
elif [[ -x "${PROJECT_DIR}/.venv/bin/torchrun" ]]; then
  TORCHRUN_BIN="${PROJECT_DIR}/.venv/bin/torchrun"
elif [[ -x "${PROJECT_DIR}/.venv/Scripts/torchrun.exe" ]]; then
  TORCHRUN_BIN="${PROJECT_DIR}/.venv/Scripts/torchrun.exe"
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_BIN="torchrun"
else
  TORCHRUN_BIN=""
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required. Example: MODEL_PATH=\$HOME/models/Qwen3-8B bash scripts/run_train.sh" >&2
  exit 1
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH does not exist: ${DATA_PATH}" >&2
  exit 1
fi
case "${BACKEND}" in
  native-tp|hf-reference|auto) ;;
  *) echo "ERROR: BACKEND must be native-tp, hf-reference, or auto" >&2; exit 1 ;;
esac

mkdir -p "${OUTPUT_DIR}"
export MODEL_PATH DATA_PATH OUTPUT_DIR
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${BACKEND}" == "hf-reference" ]]; then
  exec "${PYTHON_BIN}" -m graspo train --backend hf-reference --config "${CONFIG_PATH}"
fi

if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "ERROR: torchrun was not found. Run 'uv sync --extra train --python 3.11' first." >&2
  exit 1
fi

echo "Backend: ${BACKEND}"
echo "TP_SIZE: ${TP_SIZE}"
echo "Model:   ${MODEL_PATH}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "Config:  ${CONFIG_PATH}"

exec "${TORCHRUN_BIN}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${TP_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m graspo train --backend "${BACKEND}" --config "${CONFIG_PATH}"
