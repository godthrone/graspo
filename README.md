# GRASPO

[中文说明](README.zh-CN.md)

GRASPO is a GRPO-style reinforcement-learning trainer for language-model tasks
whose answers can be checked structurally, such as JSON generation, information
extraction, classification, form parsing, and tool-call argument generation.
It is designed for low-cost LoRA-based structured-output training: keep the base
model frozen, train only compact LoRA adapters, and use reward rules that can be
audited from the generated text.
The built-in reward checks required markers, parses fenced JSON or tool-call
payloads, compares structured fields against `ground_truth`, and turns the
result into group-level preference signals for GRASPO.

GRASPO keeps the useful GRPO idea of comparing multiple completions for the same
prompt, then adds production-oriented behavior for structured outputs:

- practical RL fine-tuning for 9B-class models on a single 80 GB GPU when using
  the LoRA/native memory-aware path;
- rollout retry when a group is too weak to train on;
- perfect-answer skip so solved prompts do not consume optimizer budget;
- invalid and no-preference-gap filtering for groups with no useful reward
  signal;
- completion-level ReplayBuffer optimization;
- readable reward/debug logs that preserve real model outputs for inspection;
- self-owned `native-tp` training with tensor parallel and pipeline placement;
- LoRA-only training with frozen base weights.

The production training path uses native TP/PP LoRA modules. PEFT is treated as
an external compatibility format for warm-start import and offline export. Full
parameter training is not supported in v1.

## Install

Python 3.11 is recommended.

```bash
git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
```

## Quick Start

Edit the root sample config:

```bash
cp config_example.yaml my_graspo.yaml
```

Set at least these fields in `my_graspo.yaml`:

- `model.model_path`: local Hugging Face model directory or model id;
- `data.train_path`: JSONL training data;
- `training.output_dir`: run output directory;
- `launch.gpus`: local GPU ids for this node;
- `backend_config.native_tp.tensor_model_parallel_size` and
  `backend_config.native_tp.pipeline_model_parallel_size`: native placement
  world size.

Launch training with one YAML argument:

```bash
uv run graspo launch --config config_example.yaml
```

For a short smoke, keep `training.max_new_tokens=2048` and reduce
`training.max_steps`. Real GRASPO training should keep
`training.training_epoch_count=100` unless you intentionally run a bounded test.

Validate sample data and reward behavior:

```bash
uv run graspo validate-reward --data data/sample.jsonl --limit 2
```

## Data Format

Training data is JSONL. Each line is one prompt/context represented as chat
messages plus a reward target:

```jsonl
{"messages":[{"role":"system","content":"You extract structured telecom ticket fields as fenced JSON."},{"role":"user","content":"Ticket: user 13800138000 cannot use apn cmnet."},{"role":"assistant","content":"I will identify the phone number and APN from the ticket."},{"role":"user","content":"Extract JSON with the APN and fault number."}],"ground_truth":{"APN":"cmnet","fault_number":"13800138000"}}
```

Multimodal records use the same `messages` field and preserve message roles and
content order:

```jsonl
{"messages":[{"role":"system","content":"Extract fields from ticket screenshots."},{"role":"user","content":"Use exact snake_case values."},{"role":"assistant","content":"Understood."},{"role":"user","content":[{"type":"image","image":"images/panel_0001.png"},{"type":"text","text":"Extract the ticket fields as strict JSON."}]}],"ground_truth":{"ticket_id":"T-0001","status":"critical"}}
```

Supported fields:

- `ground_truth`: expected structured output as a JSON object;
- `messages`: required prompt/context messages for tokenizer or processor chat templates;
- image/video items inside `messages[].content`: parsed by the data layer for
  multimodal routing; image training is supported, while video should be
  smoke-tested before production use;
- extra fields are kept as metadata.

The final message must not have role `assistant`; `ground_truth` is the reward
target and must not be leaked into the input messages. GRASPO only accepts JSONL
records with `messages` and JSON-object `ground_truth`; plain `prompt`, JSON,
Excel, and top-level media fields are not supported.

## Reward Scoring

GRASPO currently ships one built-in structured-output reward. It is rule-based,
auditable, and designed for tasks where the target answer is a JSON object. A
completion is scored in four steps:

1. Check output markers. Depending on `reward` config, the scorer can require
   `<think>...</think>`, fenced JSON Markdown blocks, and
   `<tool_call>...</tool_call>` sections.
2. Extract answer text. The answer is read from the configured marker region;
   if `check_json_markdown=false`, the full answer region is treated as JSON.
3. Parse and compare JSON. Parsed completion JSON is compared with
   `ground_truth` using recursive dictionary/list matching. Fields present in
   the target get partial credit; exact values get additional credit; unexpected
   extra keys reduce the structural match.
4. Normalize reward. Marker score, structured content score, perfect-match
   bonus, and the extra-text penalty/bonus are combined into `reward`,
   `content_score`, and `all_right`.

The important outputs are:

- `reward`: scalar used for GRASPO group decisions, advantage calculation, and
  ReplayBuffer training;
- `content_score`: normalized structured-content match before group filtering;
- `all_right`: true only when every checked target is fully correct.

GRASPO uses the reward distribution inside each rollout group, not just one
absolute score. Groups with useful differences become trainable; already-perfect
groups can be skipped; groups with no reward variance or no preference gap are
discarded or retried. The readable rollout log stores the completion, extracted
fields, reward details, and invalid reason so reward behavior can be inspected
without rerunning generation.

## Configuration

All normal training configuration lives in YAML. `config_example.yaml` is the
complete public example.

### `backend`

- `native-tp`: the only supported training backend. It uses native TP/PP LoRA
  placement and frozen base weights.

### `model`

- `model_path`: base Hugging Face model path or id.
- `trust_remote_code`: passed to Hugging Face loaders.
- `torch_dtype`: model dtype, usually `bfloat16`.
- `attn_implementation`: optional Hugging Face attention implementation.
- `gradient_checkpointing`: enable model gradient checkpointing where supported.
- `chat_template_kwargs`: extra tokenizer chat-template options.

### `data`

- `train_path`: JSONL training file.
- `max_prompt_length`: prompt truncation/tokenization limit.

### `lora`

- `r`: LoRA rank.
- `alpha`: LoRA alpha.
- `dropout`: LoRA dropout.
- `adapter_path`: optional PEFT adapter directory used only for warm-start.
- `target_preset`: safe named target set, such as `language_safe`.
- `target_modules`: explicit LoRA targets. If set, `lora.target_modules` takes
  precedence over `target_preset`.
- `auto_target_modules`: allow automatic target detection when explicit targets
  are absent.
- `bias`: PEFT-compatible bias setting, usually `none`.
- `task_type`: PEFT-compatible task type, usually `CAUSAL_LM`.

GRASPO currently supports LoRA training only. It does not support full parameter
training.

### `reward`

- `check_think`: require `<think>...</think>` markers before the answer.
- `check_json_markdown`: require fenced JSON output.
- `check_tool_call`: score a tool-call target in addition to the answer.
- `check_list_order`: make list order matter in structured comparison.
- `marker_reward_weight`: reward for required output markers.
- `content_reward_weight`: reward for structured content match.
- `anti_useless_str_reward_weight`: bonus/penalty weight for extra text.
- `anti_useless_str_half_reward_len`: length scale for extra-text penalty.
- `answer_field`: ground-truth answer field.

### `training`

- `output_dir`: run output directory.
- `seed`: random seed.
- `training_epoch_count`: full dataset training epochs. Production default is
  `100`.
- `max_steps`: optional step cap for smoke/debug runs. `-1` means no cap.
- `rollout_prompt_queue_batch_size`: prompt groups scheduled together for
  rollout.
- `rollout_group_size`: completions sampled per prompt attempt.
- `optimize_completion_batch_size`: completion micro-batch size for one
  optimizer step.
- `optimize_times_per_step`: repeated optimization passes over the same replay
  completions.
- `rollout_max_retry_times`: retry budget after the initial rollout attempt.
- `learning_rate`, `weight_decay`, `max_grad_norm`: optimizer settings.
- `policy_ratio_clip_eps`: clipped policy-ratio objective epsilon.
- `max_new_tokens`: real training generation length. Keep
  `training.max_new_tokens=2048`.
- `temperature`, `top_p`: rollout sampling settings.
- `save_steps`: native checkpoint interval.
- `logging_steps`: compact training log interval.
- `perfect_skip_reward_threshold`: threshold for skipping already-solved groups.
- `dataloader_num_workers`: data loading worker count.
- `resume_from_checkpoint`: recoverable GRASPO native checkpoint directory.

`training.replay_buffer_optimize_threshold` is derived as
`optimize_completion_batch_size * rollout_group_size` and must not be configured.
`training.resume_from_checkpoint` and `lora.adapter_path` are mutually
exclusive: resume restores native checkpoint state, while PEFT adapter loading
is only a LoRA warm-start.

### `backend_config.native_tp`

- `tensor_model_parallel_size`: TP size.
- `pipeline_model_parallel_size`: PP size.
- `placement_strategy`: placement policy such as `qwen3_tp` or
  `qwen36_pp8_static`.
- `sequence_parallel`: must stay `false` in v1.
- `train_micro_batch_size`: native train micro-batch size.
- `generation_micro_batch_size`: native generation micro-batch split.
- `use_kv_cache_for_rollout`: use KV cache only for rollout generation.
- `rollout_kv_cache_max_reserved_fraction`: rollout KV memory reservation.
- `empty_cache_after_rollout_split`, `empty_cache_before_train`: CUDA cache
  controls.
- `checkpoint_format`: native recoverable checkpoint format label.
- `raw_log_enabled`, `readable_log_enabled`: rollout/replay log toggles.
- `synchronize_cuda_timing`: synchronize CUDA events for timing diagnostics.
- `pipeline_train_schedule`: pipeline train schedule, default `simple`.
- `pipeline_max_inflight_microbatches`: 1F1B inflight cap for experiments.

### `export`

- `final_formats`: optional list of formats to export after the clean `final/`
  checkpoint, for example `["peft-adapter"]`. Step checkpoints are not
  auto-exported.

### `launch`

- `gpus`: local GPU ids for this node.
- `nproc_per_node`: worker count per node. If omitted for `native-tp`, it is
  derived from TP * PP / nodes.
- `nnodes`, `node_rank`, `master_addr`, `master_port`: distributed launch
  settings.
- `python`: optional Python executable override.
- `torchrun`: optional torchrun executable override.
- `env`: extra environment variables for the launched training process.

## LoRA Targets

Preset values:

- `language_safe`: language-side `q_proj` and `v_proj`;
- `language_all_linear`: supported language attention, linear-attention, and
  MLP matrices;
- `vision_merger`: visual merger linear layers only;
- `vision_common`: visual merger plus supported visual attention/MLP linear
  layers.

Explicit `lora.target_modules` may use canonical names such as
`language.self_attn.q_proj` or glob patterns such as `visual.blocks.*.attn.*`.
Leaf aliases such as `q_proj` are not accepted. Resolution is fail-closed:
unknown targets, unsupported conv/norm parameters, and empty matches stop before
training. Native checkpoints store the resolved LoRA target signature and reject
resume with a different target configuration.

## Export

GRASPO native checkpoints are recoverable training checkpoints. Portable model
artifacts are produced with `graspo export`.

Export a PEFT LoRA adapter:

```bash
uv run graspo export --config config_example.yaml --checkpoint outputs/example-run/final --format peft-adapter --output outputs/export/adapter
```

Export a merged Hugging Face full model:

```bash
uv run graspo export --config config_example.yaml --checkpoint outputs/example-run/final --format merged-hf --output outputs/export/merged
```

`peft-adapter` reconstructs PEFT `adapter_config.json` and
`adapter_model.safetensors` from GRASPO native rank shards. For fused/split
native targets, GRASPO writes an additional `graspo_adapter_metadata.json` so
GRASPO can warm-start those adapters losslessly. Standard PEFT tools can read
the adapter tensors, but the GRASPO metadata is required to map fused/split
targets back into native training modules without ambiguity.

`merged-hf` streams the base HF safetensors on CPU, applies LoRA deltas, copies
tokenizer/config sidecar files, and writes a HF-compatible merged model
directory.

Exported PEFT adapters and merged full models are deployment/compatibility
artifacts. They do not contain optimizer, RNG, replay buffer, or trainer state,
and cannot replace `step_*` or `final` for full training resume.

## Outputs And Monitoring

Each run writes to `training.output_dir`:

- `train.log`: compact rank-0 training events;
- `rollouts.readable.jsonl`: human-readable messages, completion, reward, and
  debug details;
- `rollouts.raw.jsonl`: replay tensors, masks, old logprobs, advantages, and
  reward metadata;
- `train_batches.readable.jsonl`: one row per optimize-trigger batch;
- `rank_metrics.rank_*.jsonl`: per-rank memory, timing, LoRA, and optimizer
  diagnostics;
- `step_*`: periodic recoverable GRASPO native training checkpoints;
- `final`: final recoverable checkpoint after a clean exit.

A healthy GRASPO run is not just a process that stays alive. Watch reward trend,
reward range inside each group, content-score validity, decision distribution,
finite loss/grad, nonzero LoRA gradients, LoRA tensor changes, replay-buffer
progress, checkpoint writes, and GPU/NCCL health.

## Development

```bash
uv run --extra dev ruff check src tests scripts
uv run --extra dev ruff format --check src tests scripts
uv run --extra dev pytest -q
uv run --extra dev python -m graspo --help
```

## FAQ

- `model.model_path must be set`: edit `config_example.yaml` and point it at a
  real base model.
- `data.train_path does not exist`: point `data.train_path` at a JSONL file.
- Native launch world size mismatch: make `launch.nproc_per_node * launch.nnodes`
  equal `tensor_model_parallel_size * pipeline_model_parallel_size`.
- Rollout OOM: keep `training.max_new_tokens=2048`; reduce rollout concurrency
  or KV cache reservation instead of lowering production generation length.
- Need PEFT compatibility: load PEFT adapters through `lora.adapter_path`, and
  export portable artifacts with `graspo export`.

## License

GRASPO is released under the MIT License. See [LICENSE](LICENSE).
