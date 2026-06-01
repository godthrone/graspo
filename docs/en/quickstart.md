# Quickstart

## 1. Prepare Data

Create a JSONL file:

```jsonl
{"prompt": "Extract JSON from this ticket: ...", "ground_truth": {"field": "value"}}
```

Convert existing JSONL, JSON, or Excel files when needed:

```bash
python -m graspo prepare-data --input data/raw.xlsx --output data/train.jsonl
```

## 2. Native Megatron Server Smoke

Run the first validation on Qwen3-8B with TP=2:

```bash
TARGET_SERVER=user@gpu-host \
TARGET_PROJECT_DIR=/data/projects/graspo \
bash scripts/sync_to_server.sh

ssh user@gpu-host
cd /data/projects/graspo

TP_SIZE=2 \
MODEL_PATH=/data/models/Qwen3-8B \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/native-tp2-smoke \
CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml \
bash scripts/run_train.sh
```

This path requires PyTorch plus open-source Megatron-LM/Core on the server. It
does not use NeMo, NeMo-RL, vLLM, Ray, DeepSpeed, FSDP, DDP, or Accelerate.

## 3. Local Reference Training

```bash
MODEL_PATH=/data/models/small-causal-lm \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/reference-run \
BACKEND=hf-reference \
bash scripts/run_train.sh
```

## 4. Local CPU Smoke

```bash
bash scripts/smoke_cpu.sh
```
