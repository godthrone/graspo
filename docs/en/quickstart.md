# Quickstart

## 1. Prepare data

Create a JSONL file:

```jsonl
{"prompt": "Extract JSON from this ticket: ...", "ground_truth": {"field": "value"}}
```

You can also convert an existing JSONL, JSON, or Excel file:

```bash
python -m graspo prepare-data --input data/raw.xlsx --output data/train.jsonl
```

## 2. Build Docker image

```bash
bash scripts/build_docker.sh
```

## 3. Run training

```bash
MODEL_PATH=/data/models/your-base-model \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/run-001 \
GPU_COUNT=8 \
bash scripts/run_train.sh
```

## 4. Local CPU smoke check

```bash
bash scripts/smoke_cpu.sh
```

