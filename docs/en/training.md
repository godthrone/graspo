# Training

## Backends

GRASPO now exposes two training backends:

- `megatron-native`: production backend. GRASPO owns rollout, retry/filter,
  ReplayBuffer, reward, advantage, loss, JSONL logging, and checkpoint control.
  Open-source Megatron-LM/Core provides tensor-parallel process groups only.
- `hf-reference`: single-process Hugging Face backend for small-model parity and
  local debugging.

The removed legacy stack is not a supported runtime: NeMo, NeMo-RL, vLLM, Ray,
DeepSpeed, FSDP, DDP, and Accelerate are not production training paths.

## Native Megatron Command

```bash
TP_SIZE=2 \
MODEL_PATH=/data/models/Qwen3-8B \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/native-tp2-smoke \
CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml \
bash scripts/run_train.sh
```

The first accepted target is Qwen3-8B, single node, TP=2, PP=1, LoRA-only,
with `max_new_tokens=2048`. Use `MAX_STEPS=1-3` only to cap the number of
optimization steps during validation; do not lower generation length for real
training. TP=8 and Qwen3.6-27B are second-stage targets.

## Data Queue Semantics

The first implementation follows the original GRASPO queue:

- Consume one prompt at a time.
- Generate that prompt's `rollout_group_size` completions as one tensor-parallel batch.
- Retry or skip according to original GRASPO group decisions.
- Append trainable completion-level experiences to ReplayBuffer.
- Optimize when `replay_buffer_optimize_threshold` completions are available.
  This threshold is derived from
  `optimize_completion_batch_size * rollout_group_size`, and
  `optimize_times_per_step` controls how many passes to train over that same
  replay batch.

## Checkpoints

`megatron-native` writes recoverable per-rank LoRA TP checkpoints containing the
local LoRA tensors, optimizer state, RNG state, and config snapshot. v0.1 does
not require HF PEFT export or merged full-model export.

## First Validation

For TP=2 smoke, confirm:

- no forbidden framework import appears in logs,
- `native_qwen_adapter_ready` is printed,
- readable and raw rollout JSONL files exist,
- at least one finite loss is logged,
- LoRA grad count is non-zero,
- `step_1/` and `final/` checkpoints contain per-rank checkpoint files.

Long-run monitoring must also inspect reward health: reward trend, group reward
range, content_score distribution, invalid/retry/perfect/trainable decision
mix, JSON truncation diagnostics in `rollouts.readable.jsonl`, and LoRA tensor
changes. Stop and diagnose sustained NaN/inf metrics, zero LoRA delta, reward
collapse, no group variance, or abnormal JSON truncation.
