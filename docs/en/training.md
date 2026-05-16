# Training

## Server command

```bash
MODEL_PATH=/data/models/your-base-model \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/run-001 \
GPU_COUNT=8 \
bash scripts/run_train.sh
```

## Multi-GPU backend

v0.1 uses:

- Hugging Face Accelerate
- PyTorch FSDP `FULL_SHARD`
- PEFT LoRA

The base model is mounted at runtime and is not baked into the Docker image.

## Checkpoints

GRASPO saves LoRA adapters under the configured output directory. v0.1 does not
force automatic adapter merging into the base model.

## GRASPO + ARD Iteration

Recommended workflow:

```text
anchor bank -> GRASPO -> hard sample mining -> ARD-SFT -> GRASPO
```

ARD-SFT command:

```bash
MODEL_PATH=/data/models/your-base-model \
HARD_DATA_PATH=/data/graspo/hard_samples.jsonl \
ANCHOR_DATA_PATH=/data/graspo/anchor_bank/base-model/anchor_train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/ard-sft-001 \
GPU_COUNT=8 \
bash scripts/run_sft_ard.sh
```

See [Anchor Replay Distillation](ard.md).

## First server validation

Start with a short run:

```yaml
training:
  max_steps: 1
  save_steps: 1
```

Confirm that rollout, logprob computation, backward, and adapter checkpoint
saving all work before running a longer job.
