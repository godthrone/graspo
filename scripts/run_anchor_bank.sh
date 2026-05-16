#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-graspo:cuda12.4}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
BASE_MODEL_ID="${BASE_MODEL_ID:-base_model}"
ANCHOR_DIR="${ANCHOR_DIR:-${PROJECT_DIR}/anchor_bank/${BASE_MODEL_ID}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/anchor_generation.yaml}"
GPU_COUNT="${GPU_COUNT:-1}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required. Example: MODEL_PATH=/data/models/base bash scripts/run_anchor_bank.sh" >&2
  exit 1
fi

mkdir -p "${ANCHOR_DIR}"

if [[ "${GPU_COUNT}" == "all" ]]; then
  GPU_FLAG="all"
else
  GPU_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
  GPU_FLAG="device=${GPU_DEVICES}"
fi

docker run --rm -it \
  --gpus "${GPU_FLAG}" \
  --ipc=host \
  -v "${PROJECT_DIR}:/workspace/graspo" \
  -v "${MODEL_PATH}:/workspace/model:ro" \
  -v "${ANCHOR_DIR}:/workspace/anchor_bank" \
  -v "${CONFIG_PATH}:/workspace/config/anchor_generation.yaml:ro" \
  -e MODEL_PATH=/workspace/model \
  -e BASE_MODEL_ID="${BASE_MODEL_ID}" \
  -w /workspace/graspo \
  "${IMAGE_NAME}" \
  bash -lc "python3 -m pip install -e . && \
    graspo anchor-generate --config /workspace/config/anchor_generation.yaml --output /workspace/anchor_bank/anchor_prompts.jsonl && \
    graspo anchor-answer --model-path /workspace/model --input /workspace/anchor_bank/anchor_prompts.jsonl --output /workspace/anchor_bank/anchor_answered.jsonl && \
    graspo anchor-filter --input /workspace/anchor_bank/anchor_answered.jsonl --output /workspace/anchor_bank/anchor_filtered.jsonl --manifest-output /workspace/anchor_bank/manifest.json --teacher-model /workspace/model && \
    graspo anchor-split --input /workspace/anchor_bank/anchor_filtered.jsonl --train-output /workspace/anchor_bank/anchor_train.jsonl --eval-output /workspace/anchor_bank/anchor_eval.jsonl"
