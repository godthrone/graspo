#!/usr/bin/env bash
set -euo pipefail

# Edit these three lines on the A800 server, then run:
#   bash scripts/run_server_example.sh
export MODEL_PATH="${MODEL_PATH:-/data/models/base-model}"
export DATA_PATH="${DATA_PATH:-$(pwd)/data/train.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/graspo-run}"
export GPU_COUNT="${GPU_COUNT:-8}"
export IMAGE_NAME="${IMAGE_NAME:-graspo:cuda12.4}"

bash scripts/run_train.sh

