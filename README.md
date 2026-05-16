# GRASPO

[English Docs](docs/en/README.md) | [中文文档](docs/zh/README.md)

GRASPO is a model-agnostic implementation of **Group Relative Adaptive Structured
Policy Optimization** for structured-output language-model training.

This repository is intentionally independent of NeMo, Megatron, Ray, and private
NVIDIA training images. The first training backend targets Hugging Face
`AutoModelForCausalLM` models with PEFT LoRA and single-node multi-GPU FSDP.

## What v0.1 supports

- Hugging Face causal language models (`AutoModelForCausalLM` + `AutoTokenizer`)
- LoRA fine-tuning with automatic target-module detection and manual override
- Structured rewards for JSON, markdown JSON fences, optional think tags, and tool calls
- GRASPO rollout behavior: group sampling, adaptive retry, perfect-first skip, invalid group filtering, group-relative advantages, replay buffer, and clipped policy loss
- Anchor Replay Distillation (ARD) workflow for hard-sample SFT with anti-forgetting anchor replay
- Docker image based on CUDA 12.4 for A800 x 8 servers

## Quick start

Prepare a JSONL file:

```jsonl
{"prompt": "Extract JSON from this ticket: ...", "ground_truth": {"field": "value"}}
```

Build the image:

```bash
bash scripts/build_docker.sh
```

Run training on the target GPU server:

```bash
MODEL_PATH=/models/base-model \
DATA_PATH=/workspace/graspo/data/train.jsonl \
OUTPUT_DIR=/workspace/outputs/graspo-run \
GPU_COUNT=8 \
bash scripts/run_train.sh
```

For local CPU-only checks:

```bash
bash scripts/smoke_cpu.sh
```

Generate and reuse an offline anchor bank for ARD:

```bash
MODEL_PATH=/models/base-model \
BASE_MODEL_ID=base-model \
bash scripts/run_anchor_bank.sh
```

## Documentation

- English: [docs/en/README.md](docs/en/README.md)
- 中文: [docs/zh/README.md](docs/zh/README.md)
- Documentation index: [docs/README.md](docs/README.md)
