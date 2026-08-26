# GRASPO (Group Relative Advantage Structured Policy Optimization)

[中文说明](README.zh-CN.md)

GRASPO is a GRPO-style reinforcement-learning trainer purpose-built for
structured-output LLM tasks: JSON generation, information extraction, tool
calling, and any task whose answers can be checked structurally.  LoRA-only:
train 9B-class models on a single 80 GB GPU.

**Three layers of structured-output RL, from token signal to group decision:**

- **Token-level reward via character annotation (the core innovation).** Completion
  text is annotated character-by-character against a structural template, then
  mapped to tokens through offset_mapping — naturally tokenizer-agnostic. The
  annotation module gives each character/token a credit role (S/V/T/W/E/D), so it
  has strong format recognition. Only the first mismatched character is penalized;
  everything after it is excluded from training.
- **Structured-output reward, decoupled and extensible.** Reward scores a value (a
  JSON field / tool-call argument) or the whole JSON sequence treated as one value,
  via the `GraspoReward` class + `REWARD_REGISTRY`. Recursive dict comparison with
  dual scoring (numeric accuracy for gradient signal, structural correctness for
  gating), multi-target best-match, and numeric tolerance. To add more scoring
  abilities you extend the reward module only — the annotation module and
  token-level gradient training stay unchanged.
- **Group decision with defense-in-depth.** Six-way classification
  (perfect_skip / trainable / invalid / retry / no_preference_gap) filters
  noisy groups before they enter training. Token-level advantage drives only the
  trainable tokens, so the model is not pushed to be "best in a bad group."

**Infrastructure designed for production:**

- **User-specified parallelism, always a viable configuration.** You pick
  `tp_size` / `dp_size` / `pp_size` (and optional `sequence_parallel` when
  TP≥2). GRASPO never tells you "this won't run on your hardware" — as long
  as the basic resource constraint is met, there is a performant
  TP+DP+PP+SP+Checkpoint combination that runs your task. You don't need to
  understand NCCL topology or PCIe vs NVLink, and each GPU runs exactly one
  process, memory-balanced by design.
- SFT → RL unified pipeline: same data format, same model loading, same
  checkpoint format.
- GraspoFlow backend: the design goal is five-in-one parallelism
  (TP+DP+PP+SP+Checkpoint) usable across **SFT and RL**, at **9B and 27B**,
  from a single GPU to many, behind one configuration switch.
  ``world_size = dp_size × tp_size × pp_size``. The implementation plan
  closes the remaining gaps (RL PP>1, SP native transport, full-coverage
  testing) toward that goal.
- Pluggable model adapters with ABC contracts: new model families require
  zero changes to existing code.
- Multimodal training with three-layer contract-based defense against silent
  image dropout.
- ReplayBuffer, readable rollout logs, and built-in `analyze-profile` tooling.

## Quick Start

> **Sample files** live in `samples/`:
> - `samples/configs/sft_example.yaml` — 最保守单卡 SFT 配置（开箱即用）；
> - `samples/configs/rl_example.yaml` — 最保守单卡 RL 配置；
> - `samples/configs/a800x8_qwen35_9b_tp1_dp8_pp1.yaml` — 8×A800 多卡验证配置；
> - `samples/data/` — small JSONL datasets for validation and smoke tests.

### Docker (production path)

Docker is the primary training method. It locks the runtime environment and
avoids host dependency conflicts.

**Launch via `run.sh` (the single, defensive entry point):**

```bash
# 1. Build the image (reads version from git tag automatically)
bash docker/build.sh

# 2. Train — run.sh auto-picks free GPUs, mounts directories from YAML config,
#    and sets NCCL-safe flags
bash run.sh my_config.yaml

# 3. Smoke test first (optional but recommended): run 1 step to verify
#    the environment (model load, multimodal pipeline, training forward)
bash run.sh my_config.yaml --smoke

# 4. Pin specific GPUs (e.g. 4 and 5)
bash run.sh my_config.yaml --gpus 4,5

# 5. Override image tag (default comes from git describe)
bash run.sh my_config.yaml --image graspo:v0.22.0
```

`run.sh` is defensive by design:

- **Auto-selects free GPUs** via `nvidia-smi` (no manual GPU counting);
- **Only passes `--gpus device=<ids>`** — never injects
  `CUDA_VISIBLE_DEVICES`, which combined with Docker device binding
  deadlocks NCCL initialization;
- **Always sets `--ipc=host --shm-size=16g`** (required for NCCL shared memory);
- **Mount directories are derived from your YAML config** — `model.model_path`,
  `data.train_path`, `training.output_dir`, and the config file's own directory
  are all mounted automatically. No manual `--model-dir` or `-v` flags needed;
- **Resolves the image tag from `git describe`** — never hardcodes a version
  (overridable with `--image`);
- **`--smoke` goes through the CLI** (`graspo launch --smoke`) as a run-boundary
  flag: training stops after the first step — semantically equivalent to a
  `max_steps=1` config, and your config file is never modified.

> **`run.sh` is the only supported launch path for training.** Hand-written
> `docker run` invocations are for diagnostics only — they have repeatedly
> tripped on `--gpus` JSON syntax (Docker 29) and path resolution. When the
> training data's image references are relative (`../images/...`, e.g. datasets
> with `data/` and `images/` side by side), run.sh additionally
> mounts the dataset root on demand — mount footprint stays minimal
> (exact data/model/output dirs only).

Manual invocation (for reference, e.g. inside your own orchestration):

```bash
docker run --gpus "device=0,1" --ipc=host --shm-size=16g \
  -v /data:/data \
  graspo:<version> \
  launch --config /data/outputs/my_config.yaml
```

> **Never combine `--gpus device=` with `CUDA_VISIBLE_DEVICES`** — the
> mismatch between Docker's device mapping and the env var deadlocks NCCL.
> GPU selection is handled by `run.sh`; see `launch` in your config for
> distributed settings (nnodes, master_addr, port).

> **Need a model?** The default config points to `models/Qwen3.5-9B`. Download it with:
> ```bash
> # On the host, before launching the container
> huggingface-cli download Qwen/Qwen3.5-9B --local-dir /path/to/models/Qwen3.5-9B
> ```

For a smoke test, keep `training.max_new_tokens=2048` and reduce
`training.max_steps`. Real GRASPO training keeps
`training.max_epochs=100` unless you intentionally run a bounded test.

**Custom image name:**
```bash
IMAGE_NAME=graspo:test bash docker/build.sh
```

### Local Install (development)

Python 3.11 or 3.12 is required (`>=3.11,<3.13`).

```bash
git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
```

Now copy a sample config and point it at your model and data, then launch:

```bash
cp samples/configs/sft_example.yaml my_config.yaml
uv run graspo launch --config my_config.yaml
```

### RL Training (GRASPO)

Copy and edit the RL sample config:

```bash
cp samples/configs/rl_example.yaml my_graspo.yaml
```

Set at least these fields in `my_graspo.yaml`:

- `model.model_path`: local Hugging Face model directory or model id;
- `data.train_path`: JSONL training data;
- `training.output_dir`: run output directory;
- GPU selection is **not** in the config — use `run.sh` (auto-picks free
  GPUs) or `bash run.sh config.yaml --gpus 4,5` (see Docker section);
- `graspoflow.tp_size`, `graspoflow.dp_size`, and
  `graspoflow.pp_size`: world size = tp × dp × pp.

### SFT Training

SFT mode reuses the same GraspoFlow infrastructure (TP/PP, LoRA, checkpoint)
and the same JSONL data format. Copy the dedicated SFT example config:

```bash
cp samples/configs/sft_example.yaml my_sft.yaml
```

Key differences from RL:

- `train_method: sft` — dispatches to supervised fine-tuning instead of RL;
- `micro_batch_size` and `gradient_accumulation_micro_batches` control batch size;
  effective batch/GPU = micro_batch_size × gradient_accumulation_micro_batches;
- `max_prompt_length` is the full sequence length (prompt + response);
- `learning_rate` is typically higher than RL (e.g. `5e-5` vs `5e-6`);
- `reward` section is ignored by SFT.

Launch the same way:

```bash
uv run graspo launch --config my_sft.yaml
```

After SFT, continue with RL by changing `train_method` to `graspo` and
pointing `lora.adapter_path` to the SFT checkpoint.

### Tool Commands

Tool commands accept input-locating flags only and either print results or
write to config-decided locations — they never override config values.

**Validate reward scoring** (prints per-sample scores, writes nothing):

```bash
uv run graspo validate-reward --data samples/data/sample.jsonl --limit 2
```

**Evaluate a checkpoint** (generates rollout groups, scores rewards, writes
`summary.json` + `completions.jsonl` to `<config output_dir>/evaluate/`):

```bash
uv run graspo evaluate-checkpoint --config my_config.yaml \
    --data samples/data/sample.jsonl --checkpoint outputs/my_run/step_100
```

**Summarize profiling outputs**:

```bash
uv run graspo analyze-profile outputs/my_run
```

Besides the run summary, `analyze-profile` (v0.22+) writes six analysis
files per run dir (the CLI prints only their paths — no tables on stdout):
1. `logs/analysis_profile.json` — perf/timing/latest-step window summary
2. `logs/analysis_steps.jsonl` — training progress per step: sample range
   (samples_start/end within epoch), terminal decision counts
   (perfect/invalid/no_gap/mc/nc), mc_ratio, reward/content/loss means,
   retry_rate, alarm category counts, wall-clock seconds
3. `logs/analysis_epochs.json` — same metric tuple at epoch granularity
   (loss_mean and alarms aggregated from train_step events)
4. `logs/analysis_errors.jsonl` — completion-level mutually-exclusive error
   causes (L1 format: no_tool_call / malformed_xml / missing_param /
   multi_call / other_parse; L2 match: tool_mismatch / param_name_mismatch /
   param_value_mismatch; ok / other), at both step and epoch granularity,
   with deduplicated sample references for traceback
5. `logs/analysis_attribution.json` — group-level attribution: not_correct
   cause classification (tool_mismatch / content_all_wrong /
   format_shortfall), tool-name/param-match trend per step, decision × match
   cross-table
6. `logs/analysis_perf.jsonl` — performance from train_step timing blocks
   (zero training intrusion): rollout / queue-wait% / prefill / decode /
   throughput tok/s / optimize / retry-rate, at step and epoch granularity

All analysis is **independent of training-data semantics** (no field-name or
numeric assumptions — L3 semantic/numeric diagnosis is left to AI/humans
working from the rollouts detail logs), per user ruling.

## CLI Reference

All commands are config-driven: input-locating flags only, outputs
are either printed or written to config-decided locations.

- `graspo launch --config <yaml> [--smoke]` — training entry. `run.sh` is the
  only supported launcher in production (auto GPU selection, `--ipc=host`,
  mount inference); `--smoke` runs one step to verify the environment.
- `graspo export --config <yaml>` — export a LoRA checkpoint
  (`export.checkpoint_path` → `export.export_output` in `export_format`).
- `graspo validate-reward --data <jsonl> [--limit N] [--completions <jsonl>]`
  — score samples/reward-link check; prints per-sample scores, writes nothing.
- `graspo evaluate-checkpoint --config <yaml> --data <jsonl>
  [--checkpoint <dir>] [--limit N]` — generate rollout groups and score them;
  writes `summary.json` + `completions.jsonl` to `<output_dir>/evaluate/`.
- `graspo analyze-profile <run_dir>... [--skip-warmup-steps N]` — write six
  analysis files (`analysis_profile.json` / `analysis_steps.jsonl` /
  `analysis_epochs.json` / `analysis_errors.jsonl` /
  `analysis_attribution.json` / `analysis_perf.jsonl`) into `<run_dir>/logs/`
  and print only their paths.

Run `graspo --help` for the full flag list.

## Data Format

Training data is JSONL and is **shared by SFT and RL** (one unified format).
Each line is one prompt/context represented as OpenAI-compatible chat
`messages` (plus optional `tools` in OpenAI function-calling format), and
one or more acceptable reward `targets` — the GRASPO-specific field that
carries the expected answer for structured-output eligibility:

```jsonl
{"messages":[{"role":"system","content":"You extract structured support ticket fields as fenced JSON."},{"role":"user","content":"Ticket: user 99999000000 cannot use apn apn01."},{"role":"assistant","content":"I will identify the phone number and APN from the ticket."},{"role":"user","content":"Extract JSON with the APN and fault number."}],"targets":[{"id":"expected","output":{"content":{"APN":"apn01","fault_number":"99999000000"}}}]}
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
{"messages":[{"role":"system","content":"Use tools when needed. Output only the tool call."},{"role":"user","content":"Query device DEV-01 status at 2026-06-08 10:30."}],"tools":[{"type":"function","function":{"name":"query_device_status","description":"Query network device panel status.","parameters":{"type":"object","properties":{"device_id":{"type":"string"},"panel_time":{"type":"string"}},"required":["device_id","panel_time"]}}}],"targets":[{"id":"expected","output":{"tool_calls":[{"name":"query_device_status","arguments":{"device_id":"DEV-01","panel_time":"2026-06-08T10:30:00+08:00"}}]}}]}
```

See `samples/data/sample_tool_call.jsonl` for a runnable tool-call dataset row.

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

GRASPO ships the reward machinery as an extensible **`GraspoReward` class**
registered in `REWARD_REGISTRY` (currently one built-in `graspo` reward;
new reward functions register as classes without touching the caller). The
built-in reward is rule-based and auditable, designed for tasks where one or
more acceptable targets contain a JSON object or canonical tool-call sequence.
A completion is scored in four steps:

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

## Token-Level Annotation (v0.20.0)

After rollout, every completion is annotated **character by character** with a
structural role (`CharTag`), decoupled from scoring. The annotation module
(`src/graspo/ripple/annotation/`) is the single input for token-level
advantages:

| Tag | Meaning | Downstream |
|-----|---------|------------|
| `S` | structure char, matches expected template | +1.0 |
| `V` | value char (parameter value / JSON value) | similarity 0~1 |
| `T` | think content | 0 (not trained) |
| `W` | lead/trail/gap text | 0 (not trained) |
| `E` | **first** mismatched char vs expected template | -1.0 |
| `D` | all chars after `E` | 0 (not trained) |

Design principles (motivated by four consecutive training collapses in
v0.16-v0.19):

- **Character-level alignment, tokenizer-agnostic**: the expected mark
  sequence (e.g. `<tool_call> <function=...> <parameter=...> ...`) is compared
  char-by-char against the model output; the first mismatched char becomes
  `E`. Token labels are derived via `offset_mapping`, so any tokenizer variant
  never mislabels a correct prefix token (e.g. `</parametr>` shares the `</`
  and `param` prefix tokens with `</parameter>`).
- **Strict-alignment truncation**: everything after `E` is `D` (not trained) —
  tokens under a wrong prefix have no training value.
- **Truncation/incomplete structure does not set `E`**: existing chars are all
  correct; the incompleteness signal is left to the reward layer
  (`content_score = 0` → group RETRY/INVALID).
- **Value errors do not set `E`**: content similarity is scored downstream;
  only structural/type errors set `E`.
- **Tool-call parameters are matched as an unordered set** (JSON object
  semantics): order swap is not an error; a missing GT parameter or an extra
  parameter is (the expected missing `<parameter=NAME>` is compared against
  the actual output to locate `E`).
- **Consistent with the reward layer**: only semantic errors set `E` — missing
  fields, extra fields, typos, type mismatches; semantically-correct output
  (e.g. parameter order) does not.

A 65-case test dataset (`tests/data/annotation_testset_v3.jsonl`, 44 tool_call +
21 JSON) covers perfect outputs, value errors, lead text, typos, duplicate
tags, extra/missing fields, order swaps, truncation, gibberish, think mode,
nested structures, and truncation combinations, with per-case expected
annotations verified by independent agents.
`tests/data/generate_annotation_viewer.py` renders an HTML color-coded view
for manual inspection.

## Configuration

All normal training configuration lives in YAML.

- `samples/configs/sft_example.yaml` — 最保守单卡 SFT 模板；
- `samples/configs/rl_example.yaml` — 最保守单卡 RL 模板；
- `samples/configs/a800x8_qwen35_9b_tp1_dp8_pp1.yaml` — 8×A800 多卡验证配置。

### `train_method`

- `graspo`: RL training with GRASPO algorithm (default).
- `sft`: supervised fine-tuning with cross-entropy loss. Reuses the same
  config fields — no new fields needed. `reward` section is ignored.

### `backend`

- `graspoflow`: **The only backend.** Unified TP+DP+PP+SP+Checkpoint
  five-dimensional parallelism. world_size = tp_size × dp_size × pp_size.
  Supports all parallel modes: single-GPU (`tp=1,dp=1,pp=1`), pure TP,
  pure DP, and mixed modes.

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
- `check_list_order`: make list order matter in structured comparison.
- `marker_reward_weight`: reward for required output markers.
- `content_reward_weight`: reward for structured content match.
- `anti_useless_str_reward_weight`: bonus/penalty weight for extra text.
- `anti_useless_str_half_reward_len`: length scale for extra-text penalty.
- `numeric_tolerance`: relative numeric error tolerance for full marks
  (default 0.2).

### `training`

- `output_dir`: run output directory; when empty, derived as
  `outputs/<run_name>` from an auto-generated timestamp-based `run_name`.
- `run_name`: optional run name (default auto-generated).
- `seed`: random seed.
- `max_epochs`: full dataset training epochs. Production default is
  `100`. Training length is controlled solely by `max_epochs` — the old
  `max_steps` was removed in v0.23.0. For bounded/smoke runs use the
  `--smoke` flag (runs one step) instead.
- `rollout_group_size`: completions sampled per prompt.
- `rollout_queue_batch_size`: prompts fetched from the rollout queue per step
  (default 8); drives the replay buffer threshold together with
  `rollout_group_size`.
- `gradient_accumulation_micro_batches`: prompts scheduled together for one optimize
  step; replay buffer threshold is `rollout_queue_batch_size × rollout_group_size`.
- `rollout_max_retries`: retry budget after the initial rollout attempt.
- `learning_rate`, `weight_decay`, `max_grad_norm`: optimizer settings.
- `policy_ratio_clip_eps`: clipped policy-ratio objective epsilon.
- `max_new_tokens`: real training generation length. Keep
  `training.max_new_tokens=2048`.
- `lr_scheduler`: `type` (`constant`/`cosine`/`linear`), `warmup_steps`,
  `min_lr_ratio`, `decay_steps`. When `type` is not `constant`, `decay_steps`
  must be set to a positive value.
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
`rollout_queue_batch_size * rollout_group_size` (default 8 × 8 = 64 completions) and must not be configured.
`training.resume_from_checkpoint` and `lora.adapter_path` are mutually
exclusive: resume restores native checkpoint state, while PEFT adapter loading
is only a LoRA warm-start.

### `graspoflow`

- `adapter`: model adapter path (default
  `graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter`).
- `tp_size`: TP size (default 2).
- `dp_size`: DP size (default 1). `world_size = dp_size × tp_size × pp_size`.
- `pp_size`: PP size (default 1).
- `dp_replicate_lora`: replicate LoRA across DP ranks (default `true`);
  gradients are AVG all-reduced across `dp_group`.
- `lr_scaling`: DP learning-rate scaling (`"linear"` = lr × dp_size, default;
  `"none"` = no scaling).
- `placement_strategy`: placement policy such as `qwen3_tp` or
  `qwen36_pp8_static` (default `auto`).
- `layer_ranges`: manual per-stage layer distribution. Example for pp=4, 32 layers:
  `[[0,9], [9,17], [17,25], [25,32]]`. Overrides `placement_strategy` when set.
- `sequence_parallel`: optional; requires `tp_size >= 2`. Shards the sequence
  dimension within a TP group (reduce_scatter + all_gather) to cut activation
  memory.
- `pp_micro_batch_size`: PP micro-batch size (default 1).
- `micro_batch_size`: rollout forward batch size (default 8). Replaces
  the old `gpu_memory_utilization`.
- `pp_scheduler`: PP schedule strategy (`one_f_one_b`/`1f1b`, default).
  `one_f_one_b` interleaves forward/backward for a smaller pipeline bubble; it relies on
  the two-way process groups (`pp_group_fwd`/`pp_group_bwd`) so forward-hidden and
  backward-grad do not share a single peer-pair FIFO. 1F1B is the only PP schedule (the
  old bubble-heavy `gpipe` was removed).
- `pp_max_inflight_microbatches`: bounded in-flight microbatch cap for PP backpressure.
  Default `0` = auto (framework derives a bound); `>0` = explicit cap.
- `use_kv_cache_for_rollout`: use KV cache only for rollout generation.
- `empty_cache_after_rollout_split`, `empty_cache_before_train`: CUDA cache
  controls.
- `raw_log_enabled`, `readable_log_enabled`: rollout/replay log toggles.
- `synchronize_cuda_timing`: synchronize CUDA events for timing diagnostics.

### `export`

- `final_formats`: optional list of formats to export after the clean `final/`
  checkpoint, for example `["peft-adapter"]`. Step checkpoints are not
  auto-exported.

### `launch`

- GPU selection is **not** configured here — it is handled by `run.sh`
  (`bash run.sh config.yaml --gpus 4,5`) or by Docker `--gpus` directly.
- `nproc_per_node`: worker count per node. If omitted, it is derived from
  `tp_size × dp_size × pp_size / nnodes`.
- `nnodes`, `node_rank`, `master_addr`, `master_port`: distributed launch
  settings.
- `python`: optional Python executable override.

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
math, and LoRA target metadata should live on the supported hybrid text/vision
family's native class (e.g. `Qwen35HybridTextModel`) and its attention/layer
modules.

`TransformerAdapter` and its model-family subclasses (e.g. `Qwen35Adapter`)
are responsible for processor/tokenizer calls, batching,
rollout splitting, sampling, pipeline send/recv orchestration, checkpoint
delegation, and logging. Runtime and placement modules own backend lifecycle,
config validation, and TP/PP layout only; they should not implement
model-family math.

The supported model family is the Qwen3.5/3.6 hybrid text/vision class;
Qwen3.6 reuses the Qwen3.5-family native class in GRASPO because its
architecture is compatible with that family. If a future model uses a
different `model_type`, add a dedicated native model class instead of
introducing adapter-level special cases.

## Export

GRASPO native checkpoints are recoverable training checkpoints. Portable model
artifacts are produced with `graspo export`. Set `export.checkpoint_path`,
`export.export_format`, and `export.export_output` in your YAML config, then run:

```bash
uv run graspo export --config samples/configs/sft_example.yaml
```

Example minimal export config:
```yaml
backend: graspoflow
model:
  model_path: models/Qwen3.5-9B
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

- `logs/training.log`: human-readable rank-0 text log (INFO/DEBUG lines only —
  structured JSON events go to `events.jsonl`);
- `logs/events.jsonl`: structured event stream (`train_step`, `epoch_summary`,
  `checkpoint_saved`, `group_decision`, ...) with `timestamp` + `run_id` keys;
- `logs/train.log`: alternative training log path (legacy fallback for
  `analyze-profile`);
- `logs/rollouts.readable.jsonl`: human-readable messages, completion, reward, and
  debug details (every attempt including retries);
- `logs/rollouts.raw.jsonl`: replay tensors, masks, old logprobs, advantages, and
  reward metadata (terminal attempts only);
- `logs/train_batches.readable.jsonl`: one row per optimize-trigger batch;
- `logs/rank_metrics.rank_*.jsonl`: per-rank memory, timing, LoRA, and optimizer
  diagnostics;
- `logs/error.log`: aggregated ERROR-level text events (invalid groups, reward
  variance failures, format-broken groups);
- `logs/timing_events.jsonl`: timing diagnostics for each phase;
- `epoch_*`: epoch-end recoverable checkpoints (when `save_checkpoint_every_epoch` is true);
- `step_*`: periodic recoverable checkpoints (when `save_steps > 0`);
- `final`: final recoverable checkpoint after a clean exit;
- `config.yaml`: configuration backup for full reproducibility.

All log files live under the `logs/` subdirectory.

SFT runs produce a subset of these outputs: `training.log`, `rank_metrics.*.jsonl`,
`error.log`, checkpoints, `final`, and `config.yaml`. Rollout and replay logs are
RL-only and not written during SFT training.

### Key Monitoring Indicators

**Group Classification** (per step):
- `perfect_skip`: already stable, no training needed
- `trainable` (max_correct / not_correct): has training value
- `invalid`: format-broken, discarded
- `invalid_no_preference_gap`: all completions identical, no signal
- `retry`: rollout failed, retrying

**Reward Trends**:
- `reward_mean` / `reward_median` rising → training is effective
- `reward_max_median_gap_mean` > 0 → group still has preference signal
- `nonzero_range_rate` near 0 → model may be overfitting

**Content Score**:
- `content_all_zero_rate` ≥ 0.8 → model cannot produce correct content
- `content_all_one_rate` rising → model approaching perfection

**Training Health** (auto-detected):
- `nonfinite_loss_or_grad`: loss/grad has inf/nan
- `zero_lora_delta`: LoRA weights unchanged
- `batch_reward_all_zero`: all rewards = 0 in this batch
- `batch_high_retry_rate`: excessive retries
- `reward_all_zero_window`: recent 10+ steps all zero
- `content_score_all_zero_window`: recent steps ≥80% zero content

### SFT → RL Two-Phase Training

For complex structured-output tasks, the recommended pipeline is:

1. **SFT phase** (10 epochs): teach the model output format via `train_method: sft`
2. **Export**: `graspo export` merges LoRA into the base model
3. **RL phase** (100 epochs): optimize output quality via `train_method: graspo`

SFT and RL share the same JSONL data format. SFT target text is generated by
`build_sft_target_text` — it produces raw XML matching the model's inference
output character-for-character, without going through `tokenizer.apply_chat_template`.
The principle: SFT should teach **what to say**, not **how to say it** — format
is a pre-trained capability that should not be overwritten by LoRA.

## Development

### Local

```bash
uv run --extra dev ruff check src tests scripts
uv run --extra dev ruff format --check src tests scripts
uv run --extra dev pytest -q
uv run --extra dev python -m graspo --help
```

### Docker

```bash
# Check CLI works
docker run --rm graspo:<version>
# → shows graspo --help output

# Run quick smoke test (requires a mounted model):
#   graspo launch --smoke runs 1 training step, verifies model load,
#   multimodal pipeline, and training forward, then stops.
bash run.sh samples/configs/sft_example.yaml --smoke
```

## FAQ

- `model.model_path must be set`: edit `samples/configs/sft_example.yaml` and point it at a
  real base model.
- `data.train_path does not exist`: point `data.train_path` at a JSONL file.
- **Docker: model not found in container**: make sure `model.model_path` in your
  YAML is an absolute host path — `run.sh` auto-mounts its parent directory.
  If the path is outside the auto-detected mounts, check with `bash run.sh --help`.
- **Docker: `torchrun` not found**: the Docker image installs GRASPO as a CLI
  entry point. Run `graspo launch --config ...` directly; the container's PATH
  includes the venv with torch and torchrun.
- Native launch world size mismatch: make `launch.nproc_per_node * launch.nnodes`
  equal `tp_size × dp_size × pp_size`.
- Rollout OOM: keep `training.max_new_tokens=2048`; reduce rollout concurrency
  or KV cache reservation instead of lowering production generation length.
- Need PEFT compatibility: load PEFT/GRASPO-PEFT adapters through `lora.adapter_path`, and
  export portable artifacts with `graspo export --config <yaml>`.
- **SFT to RL**: after SFT training, set `train_method: graspo`, point
  `lora.adapter_path` to the SFT checkpoint's adapter, and adjust
  `learning_rate` down (e.g. `1e-6`). The SFT LoRA adapter is directly
  compatible with GRASPO RL training.
- **SFT OOM**: reduce `micro_batch_size` (micro-batch) or `max_prompt_length`;
  increase `gradient_accumulation_micro_batches` to keep the
  effective batch size.
- **A800 PCIe DP hang (NCCL all-reduce timeout)**: on A800 PCIe-topology
  machines where GPUs are arranged in NVLink pairs connected by PCIe bridges
  (PXB), NCCL's P2P/CUMEM path can hang on small-tensor all-reduce during
  DP training.  TP training is unaffected because it uses large tensors and
  bypasses the problematic P2P path.  **Fix**: `NCCL_P2P_DISABLE=1`.  The
  framework now **detects this automatically and sets it for you — no manual
  `-e` needed**:
  - `run.sh` and the in-container Python startup check
    (`GraspoFlowState.initialize`) both inspect `nvidia-smi topo -m`: if any
    selected GPU pair is linked via PXB/PHB/SYS they set `NCCL_P2P_DISABLE=1`;
    on a full-NVLink mesh (all `NV#`) they keep P2P enabled (faster and avoids
    the asymmetric-VRAM device-0 heap).  If `nvidia-smi` is unavailable or the
    topology can't be parsed they safely fall back to disabling P2P.
  - On a PCIe topology the framework **forces** `NCCL_P2P_DISABLE=1` even if you
    pass `=0` (passing `=0` there would hang); `=1` is always safe.
  - On a full-NVLink mesh, setting `=1` manually is only a VRAM/performance
    tradeoff, not required for correctness.
  The performance impact of disabling P2P is negligible (<0.1% of total
  training time).

## License

GRASPO is released under the MIT License. See [LICENSE](LICENSE).

Dependency licenses: all runtime dependencies are permissively licensed
(MIT/Apache-2.0/BSD). The PyTorch CUDA wheels bundle NVIDIA runtime libraries
under NVIDIA's proprietary EULA (redistributable, non-copyleft); see the
NVIDIA Software License for details.
