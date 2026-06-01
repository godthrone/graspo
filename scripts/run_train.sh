#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-megatron-native}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/run}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/graspo.yaml}"
TP_SIZE="${TP_SIZE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required. Example: MODEL_PATH=/data/models/Qwen3-8B bash scripts/run_train.sh" >&2
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
  megatron-native|hf-reference|auto) ;;
  *) echo "ERROR: BACKEND must be megatron-native, hf-reference, or auto" >&2; exit 1 ;;
esac

mkdir -p "${OUTPUT_DIR}"
export MODEL_PATH DATA_PATH OUTPUT_DIR
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${BACKEND}" == "hf-reference" ]]; then
  exec python -m graspo train --backend hf-reference --config "${CONFIG_PATH}"
fi

echo "Backend: ${BACKEND}"
echo "TP_SIZE: ${TP_SIZE}"
echo "Model:   ${MODEL_PATH}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "Config:  ${CONFIG_PATH}"

exec torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${TP_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m graspo train --backend "${BACKEND}" --config "${CONFIG_PATH}"
