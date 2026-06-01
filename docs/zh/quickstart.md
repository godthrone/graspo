# 快速开始

## 1. 准备数据

创建 JSONL 文件：

```jsonl
{"prompt": "从工单中提取 JSON: ...", "ground_truth": {"field": "value"}}
```

也可以从 JSONL、JSON 或 Excel 转换：

```bash
python -m graspo prepare-data --input data/raw.xlsx --output data/train.jsonl
```

## 2. Native Megatron 服务器冒烟

第一阶段固定用 Qwen3-8B、TP=2 验证：

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

服务器需要 PyTorch 和开源 Megatron-LM/Core。该路线不使用 NeMo、NeMo-RL、
vLLM、Ray、DeepSpeed、FSDP、DDP 或 Accelerate。

## 3. 本地 reference 训练

```bash
MODEL_PATH=/data/models/small-causal-lm \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/reference-run \
BACKEND=hf-reference \
bash scripts/run_train.sh
```

## 4. 本地 CPU 冒烟

```bash
bash scripts/smoke_cpu.sh
```
