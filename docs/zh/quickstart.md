# 快速开始

## 1. 准备数据

创建 JSONL 文件：

```jsonl
{"prompt": "从工单中提取 JSON：...", "ground_truth": {"field": "value"}}
```

也可以从已有 JSONL、JSON 或 Excel 转换：

```bash
python -m graspo prepare-data --input data/raw.xlsx --output data/train.jsonl
```

## 2. 构建 Docker 镜像

```bash
bash scripts/build_docker.sh
```

## 3. 启动训练

```bash
MODEL_PATH=/data/models/your-base-model \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/run-001 \
GPU_COUNT=8 \
bash scripts/run_train.sh
```

## 4. 本机 CPU 冒烟检查

```bash
bash scripts/smoke_cpu.sh
```

