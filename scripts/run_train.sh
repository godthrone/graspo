#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-graspo:cuda12.4}"
GPU_COUNT="${GPU_COUNT:-8}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/run}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/fsdp_lora_graspo.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/configs/accelerate_fsdp_8gpu.yaml}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required. Example: MODEL_PATH=/data/models/llama bash scripts/run_train.sh" >&2
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

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: ACCELERATE_CONFIG does not exist: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${GPU_COUNT}" == "all" ]]; then
  GPU_FLAG="all"
else
  GPU_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
  GPU_FLAG="device=${GPU_DEVICES}"
fi

echo "Image:       ${IMAGE_NAME}"
echo "Project:     ${PROJECT_DIR}"
echo "Model:       ${MODEL_PATH}"
echo "Data:        ${DATA_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPUs:        ${GPU_COUNT}"
echo "Config:      ${CONFIG_PATH}"
echo "Accelerate:  ${ACCELERATE_CONFIG}"

docker run --rm -it \
  --gpus "${GPU_FLAG}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace/graspo" \
  -v "${MODEL_PATH}:/workspace/model:ro" \
  -v "${DATA_PATH}:/workspace/data/train.jsonl:ro" \
  -v "${OUTPUT_DIR}:/workspace/outputs" \
  -v "${CONFIG_PATH}:/workspace/config/train.yaml:ro" \
  -v "${ACCELERATE_CONFIG}:/workspace/config/accelerate.yaml:ro" \
  -e MODEL_PATH=/workspace/model \
  -e DATA_PATH=/workspace/data/train.jsonl \
  -e OUTPUT_DIR=/workspace/outputs \
  -w /workspace/graspo \
  "${IMAGE_NAME}" \
  bash -lc "python3 -m pip install -e . && accelerate launch --config_file /workspace/config/accelerate.yaml \$(command -v graspo) train --config /workspace/config/train.yaml"
