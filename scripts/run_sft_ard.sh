#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-graspo:cuda12.4}"
GPU_COUNT="${GPU_COUNT:-8}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
HARD_DATA_PATH="${HARD_DATA_PATH:-${PROJECT_DIR}/data/hard_samples.jsonl}"
ANCHOR_DATA_PATH="${ANCHOR_DATA_PATH:-${PROJECT_DIR}/anchor_bank/base_model/anchor_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/ard_sft}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/ard_sft_lora.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/configs/accelerate_fsdp_8gpu.yaml}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required. Example: MODEL_PATH=/data/models/base bash scripts/run_sft_ard.sh" >&2
  exit 1
fi

if [[ ! -f "${HARD_DATA_PATH}" ]]; then
  echo "ERROR: HARD_DATA_PATH does not exist: ${HARD_DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ANCHOR_DATA_PATH}" ]]; then
  echo "ERROR: ANCHOR_DATA_PATH does not exist: ${ANCHOR_DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${GPU_COUNT}" == "all" ]]; then
  GPU_FLAG="all"
else
  GPU_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
  GPU_FLAG="device=${GPU_DEVICES}"
fi

docker run --rm -it \
  --gpus "${GPU_FLAG}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace/graspo" \
  -v "${MODEL_PATH}:/workspace/model:ro" \
  -v "${HARD_DATA_PATH}:/workspace/data/hard_samples.jsonl:ro" \
  -v "${ANCHOR_DATA_PATH}:/workspace/data/anchor_train.jsonl:ro" \
  -v "${OUTPUT_DIR}:/workspace/outputs" \
  -v "${CONFIG_PATH}:/workspace/config/ard_sft.yaml:ro" \
  -v "${ACCELERATE_CONFIG}:/workspace/config/accelerate.yaml:ro" \
  -e MODEL_PATH=/workspace/model \
  -e HARD_DATA_PATH=/workspace/data/hard_samples.jsonl \
  -e ANCHOR_DATA_PATH=/workspace/data/anchor_train.jsonl \
  -e OUTPUT_DIR=/workspace/outputs \
  -w /workspace/graspo \
  "${IMAGE_NAME}" \
  bash -lc "python3 -m pip install -e . && accelerate launch --config_file /workspace/config/accelerate.yaml \$(command -v graspo) train-sft-ard --config /workspace/config/ard_sft.yaml"
