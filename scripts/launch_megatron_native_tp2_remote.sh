#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(pwd)}
VENV=${VENV:-"$CODE_DIR/.venv"}
MODEL_PATH=${MODEL_PATH:-}
DATA_PATH=${DATA_PATH:-"$CODE_DIR/data/sample.jsonl"}
PROFILE=${PROFILE:-"$CODE_DIR/configs/profiles/qwen3_8b_megatron_native_tp2_overnight.yaml"}
GPUS=${GPUS:-0,1}
PORT=${PORT:-29623}
TAG=${TAG:-longrun}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-20}
LATEST_PATH=${LATEST_PATH:-"$CODE_DIR/latest_graspo_longrun.out"}
MEMORY_INTERVAL_SEC=${MEMORY_INTERVAL_SEC:-1}
TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

usage() {
  cat <<'USAGE'
Usage: launch_megatron_native_tp2_remote.sh [options]

Options:
  --code-dir PATH       Deployed GRASPO code directory. Default: current directory.
  --venv PATH           Python venv containing torchrun. Default: CODE_DIR/.venv.
  --model-path PATH     Qwen3-8B HF weight directory.
  --data-path PATH      Training JSONL path.
  --profile PATH        Base YAML profile to copy into the run output directory.
  --gpus IDS            CUDA_VISIBLE_DEVICES value. Default: 0,1.
  --port PORT           torchrun master port. Default: 29623.
  --tag TAG             Output tag. Default: longrun.
  --max-steps N         Override training.max_steps. Default: -1.
  --save-steps N        Override training.save_steps. Default: 20.
  --latest-path PATH    File updated with output directory.
  --memory-interval N   nvidia-smi recorder interval seconds. Default: 1.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --code-dir) CODE_DIR=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --model-path) MODEL_PATH=$2; shift 2 ;;
    --data-path) DATA_PATH=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --save-steps) SAVE_STEPS=$2; shift 2 ;;
    --latest-path) LATEST_PATH=$2; shift 2 ;;
    --memory-interval) MEMORY_INTERVAL_SEC=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: --model-path or MODEL_PATH is required." >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/python3" ]]; then
  echo "ERROR: venv python not found: $VENV/bin/python3" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: data file not found: $DATA_PATH" >&2
  exit 1
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT="$CODE_DIR/outputs/tp2_${TAG}_2048_$RUN_TS"
CONFIG="$OUT/$(basename "$PROFILE")"

mkdir -p "$OUT/gpu_memory"
cp "$PROFILE" "$CONFIG"

python3 - "$CONFIG" "$MAX_STEPS" "$SAVE_STEPS" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
max_steps = sys.argv[2]
save_steps = sys.argv[3]
text = path.read_text(encoding="utf-8")
text = re.sub(r"(?m)^(\s*)max_steps:\s*[-0-9]+", rf"\1max_steps: {max_steps}", text)
text = re.sub(r"(?m)^(\s*)save_steps:\s*[-0-9]+", rf"\1save_steps: {save_steps}", text)
path.write_text(text, encoding="utf-8")
PY

{
  echo "code_dir=$CODE_DIR"
  echo "output_dir=$OUT"
  echo "config=$CONFIG"
  echo "model_path=$MODEL_PATH"
  echo "data_path=$DATA_PATH"
  echo "venv=$VENV"
  echo "gpus=$GPUS"
  echo "port=$PORT"
  echo "max_steps=$MAX_STEPS"
  echo "save_steps=$SAVE_STEPS"
  echo "torchinductor_compile_threads=$TORCHINDUCTOR_COMPILE_THREADS"
  echo "started_at=$(date -Is)"
  cd "$CODE_DIR"
  git status --short || true
  "$VENV/bin/python3" --version
  "$VENV/bin/python3" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY
  nvidia-smi -i "$GPUS"
} > "$OUT/run_env.log" 2>&1

echo "$OUT" > "$LATEST_PATH"

nohup "$VENV/bin/python3" "$CODE_DIR/scripts/record_gpu_memory.py" \
  --gpus "$GPUS" \
  --interval-sec "$MEMORY_INTERVAL_SEC" \
  --output-dir "$OUT/gpu_memory" \
  --tag "$TAG" \
  --pid-filter torchrun,python \
  > "$OUT/gpu_memory_recorder.nohup.out" 2>&1 &
echo $! > "$OUT/gpu_memory_recorder.pid"

cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"
export MODEL_PATH
export DATA_PATH
export OUTPUT_DIR="$OUT"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_COMPILE_THREADS

nohup "$VENV/bin/torchrun" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port="$PORT" \
  -m graspo train \
  --backend megatron-native \
  --config "$CONFIG" \
  > "$OUT/nohup.out" 2>&1 &
echo $! > "$OUT/torchrun.pid"

echo "started output_dir=$OUT"
echo "torchrun_pid=$(cat "$OUT/torchrun.pid")"
echo "gpu_memory_recorder_pid=$(cat "$OUT/gpu_memory_recorder.pid")"
