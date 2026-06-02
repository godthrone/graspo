# GRASPO

[中文说明](README.zh-CN.md)

GRASPO is a standalone implementation of **Group Relative Adaptive Structured
Policy Optimization** for structured-output language-model training. It is built
for tasks where outputs can be checked field by field, such as JSON extraction,
classification, form parsing, and tool-call argument generation.

The project is intentionally README-first: install it, run the CPU smoke test,
prepare a JSONL file, then launch either the local reference backend or the
native Megatron tensor-parallel backend.

## What This Repository Provides

- `megatron-native`: the production path. GRASPO owns rollout, retry/filtering,
  ReplayBuffer, reward, advantage, policy-ratio clipped loss, readable/raw logs,
  LoRA checkpoints, and run monitoring while using open-source Megatron-LM/Core
  tensor-parallel process groups.
- `hf-reference`: a single-process Hugging Face backend for algorithm parity,
  small-model debugging, and local smoke tests.
- Qwen-style native tensor-parallel adapter with LoRA-only training on frozen
  base weights.
- Human-readable rollout logs, raw replay JSONL logs, per-rank diagnostics, and
  recoverable LoRA TP checkpoints.

The production training route does **not** depend on NeMo, NeMo-RL, vLLM, Ray,
DeepSpeed, FSDP, DDP, Accelerate, TransformerEngine, or Apex.

## Install

Python 3.11-3.13 is supported by the pinned training dependency stack. The
quick-start path uses Python 3.11 because large PyTorch wheels often lag the
newest Python releases on Linux servers.

```bash
# Install uv if it is not already available. uv will create/use a
# Python 3.11 environment even when the system Python is older.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11

git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
```

If you do not use `uv`, create a virtual environment and install the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the native Megatron backend, install a compatible open-source Megatron-LM or
Megatron Core package in the same environment. Keep model weights outside the
repository.

GRASPO pins PyTorch to the `2.5.x` line in `pyproject.toml` because some
long-lived Linux GPU servers still expose a `manylinux_2_17` baseline, while
newer PyTorch wheels may require a newer platform baseline.

## Quick Start

Run the local smoke test first:

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

Run native Megatron TP=2 training:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
BACKEND=megatron-native \
MODEL_PATH=$HOME/models/Qwen3-8B \
DATA_PATH=$HOME/datasets/graspo/train.jsonl \
OUTPUT_DIR=outputs/qwen3-8b-tp2 \
CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_overnight.yaml \
bash scripts/run_train.sh
```

For a detached long run on a server, use the generic launcher:

```bash
bash scripts/launch_megatron_native_tp2_remote.sh \
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

- `prompt`: plain text prompt.
- `ground_truth`: expected structured output, usually a JSON object.
- `messages`: optional chat messages. If present, the tokenizer chat template
  can render the prompt.
- extra fields are treated as metadata.

Convert local JSON, JSONL, or spreadsheet data into the standard JSONL shape:

```bash
python -m graspo prepare-data --input raw_data.jsonl --output outputs/train.jsonl
```

Split a dataset into train/eval JSONL files:

```bash
SOURCE_DATA_PATH=outputs/train.jsonl \
TRAIN_OUTPUT_PATH=outputs/train_split.jsonl \
EVAL_OUTPUT_PATH=outputs/eval_split.jsonl \
bash scripts/split_train_eval_jsonl.sh
```

## Core Algorithm Settings

The default profile follows the GRASPO queue semantics used by this project:

- `training.training_epoch_count=100`: full dataset training epochs.
- `training.rollout_group_size=8`: completions sampled per prompt attempt.
- `training.optimize_completion_batch_size=4`: completion micro-batch size for
  each optimizer step.
- `training.replay_buffer_optimize_threshold=32`: derived from
  `optimize_completion_batch_size * rollout_group_size`.
- `training.optimize_times_per_step=4`: repeat optimization passes over the same
  replay completions.
- `training.rollout_max_retry_times=5`: extra rollout attempts after the first
  attempt.
- `training.max_new_tokens=2048`: real training generation length. Use
  `max_steps` for short checks; do not lower generation length for production
  profiles.
- `training.policy_ratio_clip_eps=0.2`: clipped policy-ratio objective epsilon.

Group decisions are evaluated in order:

1. `perfect_skip`: lower median reward reaches the perfect threshold.
2. `trainable_max_correct`: at least one completion is fully correct.
3. `trainable_not_correct`: no full solution, but `reward_max > reward_median`.
4. `retry`: retry budget remains.
5. `invalid`: hard invalid group.
6. `invalid_no_preference_gap`: no useful preference gap after retries.

`invalid_no_preference_gap` is an information-extraction guard: when a no-right
group has `reward_max == reward_median`, it does not enter ReplayBuffer because
there is no useful preference signal.

## Outputs

Each run writes to `training.output_dir`:

- `nohup.out` or stdout: compact progress JSON.
- `train.log`: compact rank-0 training events.
- `rollouts.readable.jsonl`: prompt, completions, rewards, and debug details for
  humans.
- `rollouts.raw.jsonl`: replay tensors, masks, old logprobs, advantages, and
  reward metadata.
- `train_batches.readable.jsonl`: one line per optimize-trigger batch.
- `rank_metrics.rank_*.jsonl`: per-rank memory, timing, LoRA, and optimizer
  diagnostics.
- `checkpoints/step_*`: recoverable LoRA TP checkpoints and optimizer state.

Useful monitoring command:

```bash
tail -f outputs/qwen3-8b-tp2/nohup.out
```

## Docker

Build the project image:

```bash
bash scripts/build_docker.sh
```

Mount model and data directories when running containers. Model weights,
datasets, logs, and checkpoints are intentionally ignored by Git.

## Development

Run the full local check:

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
  Define one in the tokenizer if both are missing.
- Native backend import fails: install PyTorch and open-source Megatron-LM/Core
  in the active environment.
- OOM during rollout: keep `max_new_tokens=2048`; reduce concurrency or use KV
  cache splitting rather than lowering the generation budget.

## License

GRASPO is released under the MIT License. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
