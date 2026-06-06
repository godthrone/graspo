#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:-$(pwd)}
VENV=${VENV:-"$CODE_DIR/.venv"}
MODEL_PATH=${MODEL_PATH:-}
DATA_PATH=${DATA_PATH:-"$CODE_DIR/data/sample.jsonl"}
PROFILE=${PROFILE:-"$CODE_DIR/configs/profiles/qwen3_8b_native_tp2_overnight.yaml"}
GPUS=${GPUS:-0,1}
PORT=${PORT:-29623}
TAG=${TAG:-placement_run}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-20}
LATEST_PATH=${LATEST_PATH:-"$CODE_DIR/latest_graspo_longrun.out"}
MEMORY_INTERVAL_SEC=${MEMORY_INTERVAL_SEC:-1}
TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-}
ROLLOUT_QUEUE_BATCH_SIZE=${ROLLOUT_QUEUE_BATCH_SIZE:-}
KV_FRACTION=${KV_FRACTION:-}
PLACEMENT_STRATEGY=${PLACEMENT_STRATEGY:-}
PIPELINE_SCHEDULE=${PIPELINE_SCHEDULE:-}
PIPELINE_MAX_INFLIGHT=${PIPELINE_MAX_INFLIGHT:-}
SYNCHRONIZE_CUDA_TIMING=${SYNCHRONIZE_CUDA_TIMING:-}
OPTIMIZE_COMPLETION_BATCH_SIZE=${OPTIMIZE_COMPLETION_BATCH_SIZE:-}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-}

usage() {
  cat <<'USAGE'
Usage: launch_native_placement_remote.sh [options]

Generic detached launcher for native placement runs: Qwen3 TP, Qwen3.6 PP, and
future TP/PP profiles. The profile controls tensor/pipeline placement; this
script controls the environment, output directory, torchrun, and GPU recorder.

Options:
  --code-dir PATH       Deployed GRASPO code directory. Default: current directory.
  --venv PATH           Python venv containing torchrun. Default: CODE_DIR/.venv.
  --model-path PATH     Local Hugging Face weight directory.
  --data-path PATH      Training JSONL path.
  --profile PATH        Base YAML profile to copy into the run output directory.
  --gpus IDS            CUDA_VISIBLE_DEVICES value. Default: 0,1.
  --port PORT           torchrun master port. Default: 29623.
  --tag TAG             Output tag. Default: placement_run.
  --max-steps N         Override training.max_steps. Default: -1.
  --save-steps N        Override training.save_steps. Default: 20.
  --rollout-queue N     Override training.rollout_prompt_queue_batch_size.
  --kv-fraction FLOAT   Override native_tp.rollout_kv_cache_max_reserved_fraction.
  --placement STRATEGY  Override native_tp.placement_strategy.
  --pipeline-schedule S Override native_tp.pipeline_train_schedule.
  --max-inflight N      Override native_tp.pipeline_max_inflight_microbatches.
  --optimize-batch N    Override training.optimize_completion_batch_size.
  --train-micro-batch N Override native_tp.train_micro_batch_size.
  --sync-cuda-timing B  Override native_tp.synchronize_cuda_timing.
  --latest-path PATH    File updated with output directory.
  --memory-interval N   nvidia-smi recorder interval seconds. Default: 1.
  --nproc N             torchrun processes. Default: number of comma-separated GPUs.
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
    --rollout-queue) ROLLOUT_QUEUE_BATCH_SIZE=$2; shift 2 ;;
    --kv-fraction) KV_FRACTION=$2; shift 2 ;;
    --placement) PLACEMENT_STRATEGY=$2; shift 2 ;;
    --pipeline-schedule) PIPELINE_SCHEDULE=$2; shift 2 ;;
    --max-inflight) PIPELINE_MAX_INFLIGHT=$2; shift 2 ;;
    --optimize-batch) OPTIMIZE_COMPLETION_BATCH_SIZE=$2; shift 2 ;;
    --train-micro-batch) TRAIN_MICRO_BATCH_SIZE=$2; shift 2 ;;
    --sync-cuda-timing) SYNCHRONIZE_CUDA_TIMING=$2; shift 2 ;;
    --latest-path) LATEST_PATH=$2; shift 2 ;;
    --memory-interval) MEMORY_INTERVAL_SEC=$2; shift 2 ;;
    --nproc) NPROC_PER_NODE=$2; shift 2 ;;
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
if [[ ! -x "$VENV/bin/torchrun" ]]; then
  echo "ERROR: torchrun not found: $VENV/bin/torchrun" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: data file not found: $DATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: profile not found: $PROFILE" >&2
  exit 1
fi

if [[ -z "$NPROC_PER_NODE" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
  NPROC_PER_NODE=${#GPU_ARRAY[@]}
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT="$CODE_DIR/outputs/${TAG}_$RUN_TS"
CONFIG="$OUT/$(basename "$PROFILE")"

mkdir -p "$OUT/gpu_memory"
cp "$PROFILE" "$CONFIG"

python3 - "$CONFIG" "$MAX_STEPS" "$SAVE_STEPS" "$ROLLOUT_QUEUE_BATCH_SIZE" "$KV_FRACTION" "$PLACEMENT_STRATEGY" "$PIPELINE_SCHEDULE" "$PIPELINE_MAX_INFLIGHT" "$SYNCHRONIZE_CUDA_TIMING" "$OPTIMIZE_COMPLETION_BATCH_SIZE" "$TRAIN_MICRO_BATCH_SIZE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
max_steps = sys.argv[2]
save_steps = sys.argv[3]
rollout_queue = sys.argv[4]
kv_fraction = sys.argv[5]
placement_strategy = sys.argv[6]
pipeline_schedule = sys.argv[7]
pipeline_max_inflight = sys.argv[8]
synchronize_cuda_timing = sys.argv[9]
optimize_completion_batch_size = sys.argv[10]
train_micro_batch_size = sys.argv[11]
text = path.read_text(encoding="utf-8")
text = re.sub(r"(?m)^(\s*)max_steps:\s*[-0-9]+", rf"\1max_steps: {max_steps}", text)
text = re.sub(r"(?m)^(\s*)save_steps:\s*[-0-9]+", rf"\1save_steps: {save_steps}", text)
if rollout_queue:
    text = re.sub(
        r"(?m)^(\s*)rollout_prompt_queue_batch_size:\s*[-0-9]+",
        rf"\1rollout_prompt_queue_batch_size: {rollout_queue}",
        text,
    )
if kv_fraction:
    text = re.sub(
        r"(?m)^(\s*)rollout_kv_cache_max_reserved_fraction:\s*[0-9.]+",
        rf"\1rollout_kv_cache_max_reserved_fraction: {kv_fraction}",
        text,
    )
if placement_strategy:
    text = re.sub(
        r"(?m)^(\s*)placement_strategy:\s*[-_a-zA-Z0-9]+",
        rf"\1placement_strategy: {placement_strategy}",
        text,
    )
if pipeline_schedule:
    if re.search(r"(?m)^\s*pipeline_train_schedule:", text):
        text = re.sub(
            r"(?m)^(\s*)pipeline_train_schedule:\s*[-_a-zA-Z0-9]+",
            rf"\1pipeline_train_schedule: {pipeline_schedule}",
            text,
        )
    else:
        text = re.sub(
            r"(?m)^(\s*)readable_log_enabled:\s*(true|false)",
            rf"\1readable_log_enabled: \2\n\1pipeline_train_schedule: {pipeline_schedule}",
            text,
        )
if pipeline_max_inflight:
    if re.search(r"(?m)^\s*pipeline_max_inflight_microbatches:", text):
        text = re.sub(
            r"(?m)^(\s*)pipeline_max_inflight_microbatches:\s*[-0-9]+",
            rf"\1pipeline_max_inflight_microbatches: {pipeline_max_inflight}",
            text,
        )
    else:
        text = re.sub(
            r"(?m)^(\s*)readable_log_enabled:\s*(true|false)",
            rf"\1readable_log_enabled: \2\n\1pipeline_max_inflight_microbatches: {pipeline_max_inflight}",
            text,
        )
if synchronize_cuda_timing:
    if re.search(r"(?m)^\s*synchronize_cuda_timing:", text):
        text = re.sub(
            r"(?m)^(\s*)synchronize_cuda_timing:\s*(true|false)",
            rf"\1synchronize_cuda_timing: {synchronize_cuda_timing}",
            text,
        )
    else:
        text = re.sub(
            r"(?m)^(\s*)readable_log_enabled:\s*(true|false)",
            rf"\1readable_log_enabled: \2\n\1synchronize_cuda_timing: {synchronize_cuda_timing}",
            text,
        )
if optimize_completion_batch_size:
    text = re.sub(
        r"(?m)^(\s*)optimize_completion_batch_size:\s*[-0-9]+",
        rf"\1optimize_completion_batch_size: {optimize_completion_batch_size}",
        text,
    )
if train_micro_batch_size:
    text = re.sub(
        r"(?m)^(\s*)train_micro_batch_size:\s*[-0-9]+",
        rf"\1train_micro_batch_size: {train_micro_batch_size}",
        text,
    )
path.write_text(text, encoding="utf-8")
PY

export CODE_DIR
export CUDA_VISIBLE_DEVICES="$GPUS"
export MODEL_PATH
export DATA_PATH
export OUTPUT_DIR="$OUT"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_COMPILE_THREADS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$VENV/bin/python3" - <<'PY'
import os
from pathlib import Path
import sys

import graspo

code_dir = Path(os.environ["CODE_DIR"]).resolve()
actual = Path(graspo.__file__).resolve()
expected_src = code_dir / "src" / "graspo"
if expected_src not in actual.parents:
    print(
        "ERROR: imported graspo is not from this CODE_DIR/src. "
        f"actual={actual} expected_under={expected_src}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"graspo_import_path={actual}")
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
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "max_steps=$MAX_STEPS"
  echo "save_steps=$SAVE_STEPS"
  echo "rollout_queue_batch_size=${ROLLOUT_QUEUE_BATCH_SIZE:-profile_default}"
  echo "kv_fraction=${KV_FRACTION:-profile_default}"
  echo "placement_strategy=${PLACEMENT_STRATEGY:-profile_default}"
  echo "pipeline_train_schedule=${PIPELINE_SCHEDULE:-profile_default}"
  echo "pipeline_max_inflight_microbatches=${PIPELINE_MAX_INFLIGHT:-profile_default}"
  echo "optimize_completion_batch_size=${OPTIMIZE_COMPLETION_BATCH_SIZE:-profile_default}"
  echo "train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE:-profile_default}"
  echo "synchronize_cuda_timing=${SYNCHRONIZE_CUDA_TIMING:-profile_default}"
  echo "pythonpath=$PYTHONPATH"
  echo "torchinductor_compile_threads=$TORCHINDUCTOR_COMPILE_THREADS"
  echo "started_at=$(date -Is)"
  cd "$CODE_DIR"
  git status --short || true
  "$VENV/bin/python3" --version
  "$VENV/bin/python3" - <<'PY'
import graspo
import torch
print("graspo", graspo.__file__)
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
nohup "$VENV/bin/torchrun" \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr=127.0.0.1 \
  --master_port="$PORT" \
  -m graspo train \
  --backend native-tp \
  --config "$CONFIG" \
  > "$OUT/nohup.out" 2>&1 &
echo $! > "$OUT/torchrun.pid"

echo "started output_dir=$OUT"
echo "torchrun_pid=$(cat "$OUT/torchrun.pid")"
echo "gpu_memory_recorder_pid=$(cat "$OUT/gpu_memory_recorder.pid")"
