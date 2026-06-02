# GRASPO

[中文说明](README.zh-CN.md)

GRASPO is an improved reinforcement-learning algorithm based on GRPO, designed
for language-model tasks whose outputs can be checked structurally, such as JSON
generation, information extraction, classification, form parsing, and tool-call
argument generation.

Compared with generic GRPO training, GRASPO adds rollout retry, perfect-answer
skipping, invalid group filtering, no-preference-gap filtering, completion-level
ReplayBuffer training, and readable reward/debug logs. These changes make the
algorithm better suited to formatted outputs where a response can be valid,
malformed, partially correct, or fully correct.

## Why GRASPO

GRPO works well when a group of sampled completions contains useful reward
differences. Structured-output tasks add a few extra problems:

- many completions are invalid because they miss JSON fences, tool-call markers,
  required fields, or parseable structure;
- some prompts are already solved and should not consume optimizer budget;
- some no-right groups have identical or near-identical rewards, so they do not
  provide a useful preference signal;
- long formatted answers can be truncated, making reward debugging impossible
  without storing the actual model outputs.

GRASPO keeps the useful GRPO idea of comparing completions inside a rollout
group, then adds task-specific filtering and replay behavior so training focuses
on groups that can teach the model something.

This repository focuses on a production path that is easy to inspect and adapt:

- the GRASPO algorithm, replay queue, reward, logging, and LoRA training are
  kept in this codebase;
- the multi-GPU backend is self-owned `native-tp`, implemented with PyTorch
  distributed tensor parallelism;
- the production path does not require Megatron, NeMo, vLLM, Ray, DeepSpeed,
  FSDP, DDP, Accelerate, TransformerEngine, Apex, or ZeRO fallbacks.

## How It Works

For each dataset sample, GRASPO runs this loop:

1. render one prompt;
2. generate `rollout_group_size` completions;
3. score every completion against `ground_truth`;
4. classify the group as perfect, trainable, retry, or invalid;
5. push trainable completion-level experiences into ReplayBuffer;
6. optimize LoRA parameters when the replay queue reaches the threshold;
7. keep readable/raw JSONL logs so low rewards can be debugged from actual model
   outputs.

The main algorithmic changes from plain GRPO are:

- `retry`: retry low-quality groups before giving up, up to
  `rollout_max_retry_times`;
- `perfect_skip`: skip groups whose lower median reward is already perfect;
- `invalid`: drop hard-invalid groups such as no reward variance or uniform
  partial content;
- `invalid_no_preference_gap`: drop no-right groups whose max reward does not
  beat the median;
- ReplayBuffer optimization: store completion-level experiences and optimize
  them for `optimize_times_per_step` passes;
- readable/raw logging: save model outputs, reward details, masks, logprobs, and
  metadata separately for monitoring and debugging.

Group decisions are evaluated in order:

1. `perfect_skip`: lower median reward reaches the perfect threshold;
2. `trainable_max_correct`: at least one completion is fully correct;
3. `trainable_not_correct`: no full solution, but `reward_max > reward_median`;
4. `retry`: retry budget remains;
5. `invalid`: original hard invalid filters, such as no reward variance or
   uniform partial content;
6. `invalid_no_preference_gap`: no useful preference gap after retries.

`invalid_no_preference_gap` is an information-extraction guard: a no-right group
that is not already hard-invalid and has `reward_max == reward_median` does not
enter ReplayBuffer, because it has no useful preference signal.

## Current Status

Supported today:

- Qwen3 dense causal LM, such as Qwen3-8B, through the native TP adapter;
- LoRA-only training with frozen base weights;
- TP sharding for large dense attention and MLP matrices;
- exact selected-token logprob for the training loss path;
- rollout KV cache with generation micro-batch splitting;
- readable rollout logs, raw replay logs, per-rank metrics, and recoverable LoRA
  native TP checkpoints;
- `hf-reference`, a single-process Hugging Face backend for parity and smoke
  tests.

Planned next:

- exact native text-only hybrid `linear_attention` kernel for Qwen3.5/Qwen3.6;
- PEFT adapter import/export helpers for offline compatibility;
- optional vocab-parallel embedding/lm_head if replicated vocabulary weights
  become the bottleneck.

Qwen3.5/Qwen3.6 text configs are detected, and vision weights are kept out of
scope, but checkpoints with hybrid `linear_attention` fail closed until the
exact kernel exists. GRASPO will not silently train them with an approximate
Qwen3 full-attention layer.

## Install

Python 3.11+ is recommended.

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev
```

Without `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Keep model weights, real datasets, logs, and checkpoints outside the repository.

## Quick Start

Run the local CPU smoke test:

```bash
bash scripts/smoke_cpu.sh
```

Validate the bundled sample data and reward function:

```bash
python -m graspo validate-reward --data data/sample.jsonl --limit 2
```

Run a single-process reference job with a small local causal LM:

```bash
BACKEND=hf-reference \
MODEL_PATH=$HOME/models/small-causal-lm \
DATA_PATH=data/sample.jsonl \
OUTPUT_DIR=outputs/hf-reference-demo \
CONFIG_PATH=configs/graspo.yaml \
bash scripts/run_train.sh
```

Run native TP TP=2 training on Qwen3-8B:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
BACKEND=native-tp \
MODEL_PATH=$HOME/models/Qwen3-8B \
DATA_PATH=$HOME/datasets/graspo/train.jsonl \
OUTPUT_DIR=outputs/qwen3-8b-tp2 \
CONFIG_PATH=configs/profiles/qwen3_8b_native_tp2_overnight.yaml \
bash scripts/run_train.sh
```

Run a detached long training job on a server:

```bash
bash scripts/launch_native_tp2_remote.sh \
  --model-path $HOME/models/Qwen3-8B \
  --data-path $HOME/datasets/graspo/train.jsonl \
  --gpus 0,1 \
  --tag longrun
```

The launcher writes the output directory to `latest_graspo_longrun.out`, starts
`nohup` training, and starts `scripts/record_gpu_memory.py` beside the run.

## Data Format

Training data is JSONL. Each line is one prompt:

```jsonl
{"prompt":"Extract JSON with the APN and fault number.\nTicket: user 13800138000 cannot use apn cmnet.","ground_truth":{"APN":"cmnet","fault_number":"13800138000"}}
```

Supported fields:

- `prompt`: plain text prompt;
- `ground_truth`: expected structured output, usually a JSON object;
- `messages`: optional chat messages for tokenizer chat templates;
- extra fields are treated as metadata.

Prepare data:

```bash
python -m graspo prepare-data --input raw_data.jsonl --output outputs/train.jsonl
```

Split train/eval JSONL:

```bash
SOURCE_DATA_PATH=outputs/train.jsonl \
TRAIN_OUTPUT_PATH=outputs/train_split.jsonl \
EVAL_OUTPUT_PATH=outputs/eval_split.jsonl \
bash scripts/split_train_eval_jsonl.sh
```

## Core Configuration

The production profile uses canonical GRASPO names:

- `training.training_epoch_count=100`: full dataset training epochs;
- `training.rollout_group_size=8`: completions sampled per prompt attempt;
- `training.optimize_completion_batch_size=4`: completion micro-batch size for
  one optimizer step;
- `training.replay_buffer_optimize_threshold=32`: derived from
  `optimize_completion_batch_size * rollout_group_size`;
- `training.optimize_times_per_step=4`: repeated optimization passes over the
  same replay completions;
- `training.rollout_max_retry_times=5`: extra rollout attempts after the first
  attempt;
- `training.max_new_tokens=2048`: real training generation length. Use
  `max_steps` for short checks; do not lower generation length for production
  profiles;
- `training.policy_ratio_clip_eps=0.2`: clipped policy-ratio objective epsilon.

## Outputs And Monitoring

Each run writes to `training.output_dir`:

- `nohup.out` or stdout: compact progress JSON;
- `train.log`: compact rank-0 training events;
- `rollouts.readable.jsonl`: prompt, completions, rewards, and debug details;
- `rollouts.raw.jsonl`: replay tensors, masks, old logprobs, advantages, and
  reward metadata;
- `train_batches.readable.jsonl`: one line per optimize-trigger batch;
- `rank_metrics.rank_*.jsonl`: per-rank memory, timing, LoRA, and optimizer
  diagnostics;
- `checkpoints/step_*`: recoverable LoRA native TP checkpoints and optimizer
  state.

Useful monitoring command:

```bash
tail -f outputs/qwen3-8b-tp2/nohup.out
```

A healthy GRASPO run is not just a process that stays alive. Watch reward,
content score, group reward range, decision distribution, finite loss/grad,
nonzero LoRA deltas, checkpoint writes, and GPU/NCCL health.

## Development

Run local checks:

```bash
python -m pytest -q
ruff check src tests
python -m graspo --help
```

Open-source hygiene checks:

```bash
git ls-files
rg -n "10\\.1\\.|192\\.168|ssh -p" .
```

Only code, configs, tests, scripts, README files, sample data, license files,
and the dependency lock file should be tracked.

## Troubleshooting

- `MODEL_PATH is required`: set `MODEL_PATH` to a local Hugging Face model
  directory.
- `DATA_PATH does not exist`: pass a JSONL file with `prompt` and
  `ground_truth`.
- Tokenizer has no pad token: GRASPO tries to use `eos_token` as `pad_token`.
- Native backend import fails: install PyTorch in the active environment.
- OOM during rollout: keep `max_new_tokens=2048`; reduce concurrency or use KV
  cache splitting rather than lowering the generation budget.

## License

GRASPO is released under the MIT License. See [LICENSE](LICENSE).
