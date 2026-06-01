# GRASPO

[English Docs](docs/en/README.md) | [Chinese Docs](docs/zh/README.md)

GRASPO is a self-owned implementation of **Group Relative Adaptive Structured
Policy Optimization** for structured-output language-model training.

The production route is now:

- `megatron-native`: large-model backend using open-source Megatron-LM/Core
  tensor-parallel process groups. GRASPO owns rollout, retry/filter/replay,
  reward, advantage, loss, readable JSONL logs, raw replay JSONL logs, and
  recoverable LoRA TP checkpoints.
- `hf-reference`: single-process Hugging Face reference backend for small
  models, local tests, and algorithm parity checks.

The repository no longer provides a NeMo/NeMo-RL/vLLM/Ray/DeepSpeed/FSDP/DDP
production path.

## What v0.1 supports

- `backend: auto | megatron-native | hf-reference`
- Built-in Qwen native tensor-parallel adapter for first-stage Qwen3-8B TP=2
  smoke validation.
- LoRA-only training on frozen base weights.
- Original GRASPO queue semantics: one prompt is sampled at a time, its
  `group_size` completions are generated as one TP batch, trainable samples go
  into ReplayBuffer, and optimization starts when the replay threshold is met.
- Structured rewards for JSON, markdown JSON fences, optional think tags, and
  tool calls.
- Readable/raw rollout JSONL split and recoverable per-rank LoRA TP checkpoint
  state.

## Quick Start

Prepare a JSONL file:

```jsonl
{"prompt": "Extract JSON from this ticket: ...", "ground_truth": {"field": "value"}}
```

For the native Megatron path, run inside a server environment with PyTorch and
open-source Megatron-LM/Core installed:

```bash
TP_SIZE=2 \
MODEL_PATH=/data/models/Qwen3-8B \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/tp2-smoke \
CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml \
bash scripts/run_train.sh
```

For a local single-process reference run:

```bash
BACKEND=hf-reference \
MODEL_PATH=/models/small-causal-lm \
DATA_PATH=data/sample.jsonl \
bash scripts/run_train.sh
```

For local CPU-only checks:

```bash
bash scripts/smoke_cpu.sh
```

## Documentation

- English: [docs/en/README.md](docs/en/README.md)
- Chinese: [docs/zh/README.md](docs/zh/README.md)
- Documentation index: [docs/README.md](docs/README.md)
