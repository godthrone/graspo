# GRASPO (Group Relative Adaptive Structured Policy Optimization)

[中文说明](README.zh-CN.md)

GRASPO is a GRPO-style reinforcement-learning trainer for language-model tasks
whose answers can be checked structurally, such as JSON generation, information
extraction, classification, form parsing, and tool-call argument generation.
It is designed for low-cost LoRA-based structured-output training: keep the base
model frozen, train only compact LoRA adapters, and use reward rules that can be
audited from the generated text.
The built-in reward checks required markers, parses fenced JSON or tool-call
payloads, compares structured fields against `targets`, and turns the result
into group-level preference signals for GRASPO.

GRASPO keeps the useful GRPO idea of comparing multiple completions for the same
prompt, then adds production-oriented behavior for structured outputs:

- practical RL fine-tuning for 9B-class models on a single 80 GB GPU when using
  the LoRA/native memory-aware path;
- rollout retry when a group is too weak to train on;
- perfect-answer skip so solved prompts do not consume optimizer budget;
- format-broken group filtering: when the best completion has parse errors or
  tool-call count mismatch, the group is retried or discarded instead of
  training on broken output;
- invalid and no-preference-gap filtering for groups with no useful reward
  signal;
- completion-level ReplayBuffer optimization;
- readable reward/debug logs that preserve real model outputs for inspection;
- self-owned GraspoFlow training with tensor parallel and pipeline placement;
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

### RL Training (GRASPO)

Edit the root sample config:

```bash
cp config_example.yaml my_graspo.yaml
```

Set at least these fields in `my_graspo.yaml`:

- `train_method`: `graspo` (RL) or `sft` (supervised fine-tuning);
- `model.model_path`: local Hugging Face model directory or model id;
- `data.train_path`: JSONL training data;
- `training.output_dir`: run output directory;
- `launch.gpus`: local GPU ids for this node;
- `graspoflow.tp_size` and
  `graspoflow.pp_size`: native placement
  world size.

Launch training with one YAML argument:

```bash
uv run graspo launch --config config_example.yaml
```

For a short smoke, keep `training.max_new_tokens=2048` and reduce
`training.max_steps`. Real GRASPO training should keep
`training.max_epochs=100` unless you intentionally run a bounded test.

### SFT Training

SFT mode reuses the same GraspoFlow infrastructure (TP/PP, LoRA, checkpoint)
and the same JSONL data format. Copy the dedicated SFT example config:

```bash
cp configs/sft_example.yaml my_sft.yaml
```

Key differences from RL:

- `train_method: sft` — dispatches to supervised fine-tuning instead of RL;
- `forward_batch_size` acts as micro-batch size;
- `optimize_iterations_per_step` acts as gradient accumulation steps;
- `max_prompt_length` is the full sequence length (prompt + response);
- `learning_rate` is typically higher than RL (e.g. `5e-5` vs `5e-6`);
- `reward` section is ignored by SFT.

Launch the same way:

```bash
uv run graspo launch --config my_sft.yaml
```

After SFT, continue with RL by changing `train_method` to `graspo` and
pointing `lora.adapter_path` to the SFT checkpoint.

### Validate Data

Validate sample data and reward behavior:

```bash
uv run graspo validate-reward --data data/sample.jsonl --limit 2
```

For reward validation testing, use the scripts in `scripts/` directory.

## Data Format

Training data is JSONL. Each line is one prompt/context represented as chat
messages, optional tool declarations, and one or more acceptable targets:

```jsonl
{"messages":[{"role":"system","content":"You extract structured telecom ticket fields as fenced JSON."},{"role":"user","content":"Ticket: user 13800138000 cannot use apn cmnet."},{"role":"assistant","content":"I will identify the phone number and APN from the ticket."},{"role":"user","content":"Extract JSON with the APN and fault number."}],"targets":[{"id":"expected","output":{"content":{"APN":"cmnet","fault_number":"13800138000"}}}]}
```

Multimodal records use the same `messages` field and preserve message roles and
content order:

```jsonl
{"messages":[{"role":"system","content":"Extract fields from ticket screenshots."},{"role":"user","content":"Use exact snake_case values."},{"role":"assistant","content":"Understood."},{"role":"user","content":[{"type":"image","image":"images/panel_0001.png"},{"type":"text","text":"Extract the ticket fields as strict JSON."}]}],"targets":[{"id":"expected","output":{"content":{"ticket_id":"T-0001","status":"critical"}}}]}
```

Tool-call records can provide model-native tool declarations in the optional
`tools` field. GRASPO passes `messages + tools` to the model tokenizer or
processor chat template at runtime; users should not pre-render model template
strings in the dataset:

```jsonl
{"messages":[{"role":"system","content":"Use tools when needed. Output only the tool call."},{"role":"user","content":"Query device OLT-17 status at 2026-06-08 10:30."}],"tools":[{"type":"function","function":{"name":"query_device_status","description":"Query network device panel status.","parameters":{"type":"object","properties":{"device_id":{"type":"string"},"panel_time":{"type":"string"}},"required":["device_id","panel_time"]}}}],"targets":[{"id":"expected","output":{"tool_calls":[{"name":"query_device_status","arguments":{"device_id":"OLT-17","panel_time":"2026-06-08T10:30:00+08:00"}}]}}]}
```

See `data/sample_tool_call.jsonl` for a runnable tool-call dataset row.

Alternative targets are expressed as multiple `targets` entries. Ordered
multi-step tool execution is expressed only inside `output.tool_calls`:

```jsonl
{"messages":[{"role":"user","content":"Move toward the object."}],"tools":[{"type":"function","function":{"name":"robot_atomic_control","parameters":{"type":"object","properties":{"action":{"type":"string"},"distance_cm":{"type":"integer"}},"required":["action","distance_cm"]}}}],"targets":[{"id":"left-first","output":{"tool_calls":[{"name":"robot_atomic_control","arguments":{"action":"向左","distance_cm":6}}]}},{"id":"down-first","output":{"tool_calls":[{"name":"robot_atomic_control","arguments":{"action":"向下","distance_cm":4}}]}}]}
{"messages":[{"role":"user","content":"Move, then inspect."}],"tools":[{"type":"function","function":{"name":"move","parameters":{"type":"object"}}},{"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}}],"targets":[{"id":"move-inspect","output":{"tool_calls":[{"name":"move","arguments":{"action":"left"}},{"name":"inspect","arguments":{"object":"target"}}]}}]}
```

Supported fields:

- `messages`: required prompt/context messages for tokenizer or processor chat templates;
- `tools`: optional list of tool declarations in OpenAI function-calling format,
  passed to model chat templates. Each entry is
  `{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}`;
- `targets`: required non-empty list of acceptable outputs. Each target has an
  optional `id` and an `output` object. `output.content` is a JSON object for
  normal answer tasks; `output.tool_calls` is an ordered list of canonical tool
  calls for tool-call tasks;
- image/video items inside `messages[].content`: parsed by the data layer for
  multimodal routing; image training is supported, while video should be
  smoke-tested before production use;
- extra fields are kept as metadata.

For tool-call records, `targets[].output.tool_calls` is canonical tool-call
JSON: each item is `{"name":"...","arguments":{...}}`, and list order is the
execution order. Alternative valid answers are separate entries in `targets`.
Model-specific output formats, such as Qwen XML tool calls, are parsed by the
model adapter before reward scoring.

The final message must not have role `assistant`; `targets` are raw reward
targets and must not be leaked into the input messages or converted to a
model chat template. GRASPO only accepts JSONL records with `messages`, optional
`tools`, and `targets`; plain `prompt`, JSON, Excel, legacy `ground_truth`, and
top-level media fields are not supported.

### Assistant messages with tool calls

In multi-turn conversations, assistant messages that contain tool calls MUST
use the structured `tool_calls` field. GRASPO validates this at startup and
rejects any record that embeds raw tool-call text in `content`:

```json
{
  "role": "assistant",
  "content": "I will rotate the arm toward the target.",
  "tool_calls": [
    {"name": "robot_atomic_control", "arguments": {"action_type": "顺时针旋转", "angle_deg": 38.3}}
  ]
}
```

Raw Qwen XML (`<function=...><parameter=...>`), raw JSON strings, and any
other model-specific tool-call formats MUST NOT be placed in `content`.
Use `tool_calls` with canonical JSON `{"name":"...","arguments":{...}}`.
The model's chat template renders `tool_calls` into the correct native format
automatically.

## Reward Scoring

GRASPO currently ships one built-in structured-output reward. It is rule-based,
auditable, and designed for tasks where one or more acceptable targets contain
a JSON object or canonical tool-call sequence. A completion is scored in four
steps:

1. Parse model-specific completion format. The model adapter converts raw
   output, including Qwen XML tool calls, into canonical parsed fields while
   preserving raw text and `<think>...</think>`. For Qwen XML tool calls,
   `integer`, `number`, and `boolean` parameters are typed from the declared
   tool schema before reward scoring.
2. Check output markers. Depending on `reward` config, the scorer can require
   `<think>...</think>` and fenced JSON Markdown blocks for normal answer tasks.
3. Compare structured content. Normal answer tasks compare parsed JSON with
   each `targets[].output.content`; tool-call tasks compare canonical tool
   calls with each `targets[].output.tool_calls`, preserving sequence order
   inside each target. GRASPO uses the best-scoring target. JSON number fields
   are scored with `1 / (1 + abs(predicted - target))`; non-numeric fields and
   mismatched types still use strict equality.

   Dict elements inside lists are recursively expanded in the denominator:
   `count_target_score` counts each dict element's full structure, and
   `count_check_score` uses the raw check score rather than a compressed 0-1
   normalized value.  This gives deep dict lists (e.g. tool-call `arguments`)
   proper reward differentiation while leaving scalar lists and large flat JSON
   (e.g. field-extraction tasks) unchanged.
4. Normalize reward. Marker score, structured content score, perfect-match
   bonus, and the extra-text penalty/bonus are combined into `reward`,
   `content_score`, and `all_right`.

   `dict_compare_score` returns a ``CompareResult`` that carries two parallel
   scores: the full ``dcs`` (numeric leaf values included, for gradient signal)
   and ``base_dcs`` (numeric leaves stripped from both sides, for ``all_right``
   gating).  This means numeric fields like ``distance_cm`` or ``angle_deg``
   still flow through ``content_score`` for training, but ``all_right`` only
   requires non-numeric structure to match — an action-type-correct completion
   with a slightly-off distance is still considered "all right", so
   `perfect_skip` and `max_correct` group decisions are no longer blocked by
   continuous numeric scores.

The important outputs are:

- `reward`: scalar used for GRASPO group decisions, advantage calculation, and
  ReplayBuffer training;
- `content_score`: normalized structured-content match before group filtering
  (includes numeric continuous scoring);
- `base_content_score`: structural match with numeric fields stripped, used to
  diagnose numeric field contribution;
- `all_right`: true when at least one target's non-numeric structure is fully
  correct — numeric fields like distances and angles do not need to match exactly.

Identifiers or categorical codes that should not receive continuous numeric
credit should be represented as JSON strings in the dataset.

GRASPO uses the reward distribution inside each rollout group, not just one
absolute score. Groups with useful differences become trainable; already-perfect
groups can be skipped; groups with no reward variance or no preference gap are
discarded or retried. The readable rollout log stores the completion, parsed
tool calls, extracted fields, reward details, parser errors, and invalid reason
so reward behavior can be inspected without rerunning generation.

## Configuration

All normal training configuration lives in YAML. `config_example.yaml` is the
complete public example for RL training. `configs/sft_example.yaml` is the
dedicated SFT template.

### `train_method`

- `graspo`: RL training with GRASPO algorithm (default).
- `sft`: supervised fine-tuning with cross-entropy loss. Reuses the same
  config fields — no new fields needed. `reward` section is ignored.

### `backend`

- `graspoflow`: **The only backend.** Unified TP+PP Flink-style streaming pipeline.
  Supports all parallel modes: single-GPU (`tp=1,pp=1`), pure TP (`tp=N,pp=1`),
  pure PP (`tp=1,pp=N`), and TP+PP mixed (`tp=M,pp=N`).
  See `configs/graspoflow_example.yaml`.

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
- `adapter_path`: optional PEFT or GRASPO-PEFT adapter directory used only for warm-start.
- `target_preset`: safe named target set, such as `language_safe`.
- `target_modules`: explicit LoRA targets. If set, takes precedence over
  `target_preset`.
- `bias`: PEFT-compatible bias setting, usually `none`.
- `task_type`: PEFT-compatible task type, usually `CAUSAL_LM`.

GRASPO currently supports LoRA training only. It does not support full parameter
training.

### `reward`

- `check_think`: require `<think>...</think>` markers before the answer.
- `check_json_markdown`: require fenced JSON output.
- `check_tool_call`: legacy switch retained in config; current training infers
  tool-call scoring from `targets[].output.tool_calls`.
- `check_list_order`: make list order matter in structured comparison.
- `marker_reward_weight`: reward for required output markers.
- `content_reward_weight`: reward for structured content match.
- `anti_useless_str_reward_weight`: bonus/penalty weight for extra text.
- `anti_useless_str_half_reward_len`: length scale for extra-text penalty.

### `training`

- `output_dir`: run output directory.
- `seed`: random seed.
- `max_epochs`: full dataset training epochs. Production default is
  `100`.
- `max_steps`: optional step cap for smoke/debug runs. `-1` means no cap.
- `rollout_group_size`: completions sampled per prompt.
- `optimize_prompt_batch_size`: prompts scheduled together for one optimize
  step; replay buffer threshold is `optimize_prompt_batch_size × rollout_group_size`.
- `optimize_iterations_per_step`: repeated optimization passes over the same replay
  completions.
- `rollout_max_retries`: retry budget after the initial rollout attempt.
- `learning_rate`, `weight_decay`, `max_grad_norm`: optimizer settings.
- `policy_ratio_clip_eps`: clipped policy-ratio objective epsilon.
- `max_new_tokens`: real training generation length. Keep
  `training.max_new_tokens=2048`.
- `temperature`, `top_p`: rollout sampling settings.
- `save_steps`: native checkpoint interval. `-1` (default) disables per-step
  checkpoints, leaving only epoch checkpoints.
- `save_checkpoint_every_epoch`: save a recoverable checkpoint at the end of each
  epoch (default `true`). Recommended for production training.
- `perfect_skip_reward_threshold`: threshold for skipping already-solved groups.
- `reject_unparseable_groups`: when true (default), groups whose best completion
  has parse errors or tool-call count mismatch are retried or discarded instead
  of being used for training.
- `resume_from_checkpoint`: recoverable GRASPO native checkpoint directory.

`training.replay_buffer_optimize_threshold` is derived as
`optimize_prompt_batch_size * rollout_group_size` and must not be configured.
`training.resume_from_checkpoint` and `lora.adapter_path` are mutually
exclusive: resume restores native checkpoint state, while PEFT adapter loading
is only a LoRA warm-start.

### `graspoflow`

- `tp_size`: TP size (default 2).
- `pp_size`: PP size (default 1).
- `placement_strategy`: placement policy such as `qwen3_tp` or
  `qwen36_pp8_static` (default `auto`).
- `layer_ranges`: manual per-stage layer distribution. Example for pp=4, 32 layers:
  `[[0,9], [9,17], [17,25], [25,32]]`. Overrides `placement_strategy` when set.
- `sequence_parallel`: must stay `false` in v1.
- `pp_micro_batch_size`: PP micro-batch size (default 1).
- `forward_batch_size`: rollout forward batch size (default 8). Replaces
  the old `gpu_memory_utilization`.
- `use_kv_cache_for_rollout`: use KV cache only for rollout generation.
- `empty_cache_after_rollout_split`, `empty_cache_before_train`: CUDA cache
  controls.
- `raw_log_enabled`, `readable_log_enabled`: rollout/replay log toggles.
- `synchronize_cuda_timing`: synchronize CUDA events for timing diagnostics.
- `pp_schedule`: pipeline schedule, `simple` (default) or `one_f_one_b`.
- `pp_max_inflight_microbatches`: 1F1B inflight cap for experiments.

### `export`

- `final_formats`: optional list of formats to export after the clean `final/`
  checkpoint, for example `["peft-adapter"]`. Step checkpoints are not
  auto-exported.

### `launch`

- `gpus`: local GPU ids for this node.
- `nproc_per_node`: worker count per node. If omitted, it is
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

## Native Model Implementation Boundary

Native model math belongs in the native model classes. RoPE/M-RoPE, position
IDs, KV-cache continuation, visual feature injection, TP shard-local layer
math, and LoRA target metadata should live on classes such as
`Qwen3DenseModel`, `Qwen35HybridTextModel`, and their attention/layer modules.

`TransformerAdapter` and its model-family subclasses (e.g. `Qwen3Adapter`,
`Qwen35Adapter`) are responsible for processor/tokenizer calls, batching,
rollout splitting, sampling, pipeline send/recv orchestration, checkpoint
delegation, and logging. Runtime and placement modules own backend lifecycle,
config validation, and TP/PP layout only; they should not implement
model-family math.

Qwen3.6 uses the Qwen3.5-family hybrid text/vision native class in GRASPO
because its architecture is compatible with that family. If a future model uses
a different `model_type`, such as `qwen3_vl` or `qwen3_omni`, add a dedicated
native model class instead of introducing adapter-level special cases.

## Export

GRASPO native checkpoints are recoverable training checkpoints. Portable model
artifacts are produced with `graspo export`. Set `export.checkpoint_path`,
`export.export_format`, and `export.export_output` in your YAML config, then run:

```bash
uv run graspo export --config config_example.yaml
```

Example minimal export config:
```yaml
backend: graspoflow
model:
  model_path: models/Qwen3-8B
export:
  checkpoint_path: outputs/example-run/final
  export_format: peft-adapter   # or "merged-hf"
  export_output: outputs/export/adapter
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

- `logs/training.log`: compact rank-0 training events;
- `logs/rollouts.readable.jsonl`: human-readable messages, completion, reward, and
  debug details;
- `logs/rollouts.raw.jsonl`: replay tensors, masks, old logprobs, advantages, and
  reward metadata;
- `logs/train_batches.readable.jsonl`: one row per optimize-trigger batch;
- `logs/rank_metrics.rank_*.jsonl`: per-rank memory, timing, LoRA, and optimizer
  diagnostics;
- `logs/error.log`: aggregated ERROR-level events (invalid groups, reward variance
  failures, format-broken groups);
- `logs/timing_events.jsonl`: timing diagnostics for each phase;
- `epoch_*`: epoch-end recoverable checkpoints (when `save_checkpoint_every_epoch` is true);
- `step_*`: periodic recoverable checkpoints (when `save_steps > 0`);
- `final`: final recoverable checkpoint after a clean exit;
- `config.yaml`: configuration backup for full reproducibility.

All log files live under the `logs/` subdirectory.

SFT runs produce a subset of these outputs: `training.log`, `rank_metrics.*.jsonl`,
`error.log`, checkpoints, `final`, and `config.yaml`. Rollout and replay logs are
RL-only and not written during SFT training.

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
  equal `tp_size * pp_size`.
- Rollout OOM: keep `training.max_new_tokens=2048`; reduce rollout concurrency
  or KV cache reservation instead of lowering production generation length.
- Need PEFT compatibility: load PEFT/GRASPO-PEFT adapters through `lora.adapter_path`, and
  export portable artifacts with `graspo export --config <yaml>`.
- **SFT to RL**: after SFT training, set `train_method: graspo`, point
  `lora.adapter_path` to the SFT checkpoint's adapter, and adjust
  `learning_rate` down (e.g. `1e-6`). The SFT LoRA adapter is directly
  compatible with GRASPO RL training.
- **SFT OOM**: reduce `forward_batch_size` (micro-batch) or `max_prompt_length`;
  increase `optimize_iterations_per_step` (gradient accumulation) to keep the
  effective batch size.

## License

GRASPO is released under the MIT License. See [LICENSE](LICENSE).
