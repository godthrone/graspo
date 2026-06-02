# GRASPO

[中文 README](README.zh-CN.md)

GRASPO is a README-first implementation of **Group Relative Adaptive Structured
Policy Optimization** for structured-output language-model training. It is built
for tasks where outputs can be checked field by field, such as JSON extraction,
classification, form parsing, and tool-call argument generation.

The production route is **self-owned `native-tp`**: GRASPO keeps rollout,
retry/filtering, ReplayBuffer, reward, advantage, policy-ratio clipped loss,
LoRA checkpoints, and monitoring in this repository, and uses only PyTorch
distributed process groups for tensor parallelism.

## What This Repository Provides

- `native-tp`: the production tensor-parallel backend. No Megatron, NeMo, vLLM,
  Ray, DeepSpeed, FSDP, DDP, Accelerate, TransformerEngine, or Apex is required.
- `hf-reference`: a single-process Hugging Face backend for algorithm parity,
  small-model debugging, and CPU/GPU smoke tests.
- Frozen-base LoRA training with repository-native LoRA modules.
- Human-readable rollout logs, raw replay JSONL logs, per-rank diagnostics, and
  recoverable native TP LoRA checkpoints.

Current model status:

- Qwen3 dense causal LM, for example Qwen3-8B: supported by the current native TP
  adapter.
- Qwen3.5/Qwen3.6 text-only checkpoints: config registry and text weight prefix
  detection are implemented. Hybrid `linear_attention` layers fail closed until
  the exact native linear-attention kernel is implemented, so the project will
  not silently train them with an incorrect approximation.

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

Keep model weights and real datasets outside the repository.

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

For a detached long run on a server, use the generic launcher:

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

- `prompt`: plain text prompt.
- `ground_truth`: expected structured output, usually a JSON object.
- `messages`: optional chat messages. If present, the tokenizer chat template can
  render the prompt.
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

## Core Algorithm Settings

The production profile uses these canonical names:

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
5. `invalid_no_preference_gap`: no useful preference gap after retries.
6. `invalid`: fallback invalid group.

`invalid_no_preference_gap` is an information-extraction guard: when a no-right
group has `reward_max == reward_median`, it does not enter ReplayBuffer because
there is no useful preference signal. Groups with max reward at or above the
perfect threshold must become `trainable_max_correct` or `perfect_skip`, never an
invalid bucket.

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
- `checkpoints/step_*`: recoverable LoRA native TP checkpoints and optimizer
  state.

Useful monitoring command:

```bash
tail -f outputs/qwen3-8b-tp2/nohup.out
```

## Native TP Notes

The first production implementation shards the large Qwen dense matrices:
attention `q/k/v`, attention output, MLP `gate/up/down`, and LoRA targets.
Embedding and LM head are currently replicated. Training logprob uses exact
selected-token logprob to avoid keeping full vocab logits alive for the loss
path.

Qwen3.5/Qwen3.6 support requires an exact native implementation of their hybrid
linear-attention text layers. The registry already detects those checkpoints and
ignores vision weights, but training intentionally fails closed until that kernel
is present.

## Development

Run the local checks:

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

GRASPO is released under the MIT License. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).