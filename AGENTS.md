# GRASPO Trainer Constitution

This file is the standing instruction set for GRASPO work in this repository.
Treat these rules as higher priority than runbooks, profiles, or ad hoc smoke
commands unless the user explicitly overrides them in the current conversation.

## Non-Negotiable Token Budget Rules

- Do not set `max_tokens`, `max_output_tokens`, or provider-equivalent output
  caps on large-model API calls. Let the model/API use its default output
  budget unless the user explicitly overrides this rule.
- Do not use low generation caps for real GRASPO training.
- Real GRASPO training must use `training.max_new_tokens=2048`.
- Do not lower `training.max_new_tokens` in production profiles, runbooks, or
  launch scripts. If a quick check is needed, reduce `training.max_steps`, not
  generation length.
- Unit tests may use small synthetic token counts only inside fake runtimes or
  fixtures. Those values must not leak into production configs.

## GRASPO Training Doctrine

- GRASPO is a long-training algorithm with monitored early stopping.
- Default training length is `training.training_epoch_count=100`.
- A run is not healthy merely because it does not crash. Monitoring must check:
  reward trend, reward range inside each group, content-score validity,
  decision distribution, finite loss/grad, nonzero LoRA gradients, LoRA tensor
  changes, replay-buffer progress, checkpoint writes, and GPU/NCCL health.
- Stop and diagnose when any sustained window shows NaN/inf loss or grad,
  LoRA delta stuck at zero, reward all zero or all one unexpectedly, reward
  range stuck at zero with no trainable groups, JSON truncation rate abnormal,
  content_score stuck at all zero, OOM, NCCL hang, or rank failure.
- Always inspect model outputs and reward details when reward looks wrong.
  `rollouts.readable.jsonl` is the human debug artifact; `rollouts.raw.jsonl`
  is the tensor/logprob replay artifact.

## GRASPO Core Vocabulary

- A prompt is one dataset sample.
- A rollout attempt is one generation pass for one prompt.
- A completion is one sampled model output from a rollout attempt; it is not a
  dataset sample.
- A rollout group is the set of completions sampled for the same prompt in the
  same attempt. `training.rollout_group_size=8` means one rollout group contains
  eight completions.
- ReplayBuffer stores completion-level experiences: sequence, old logprob,
  advantage, masks, and reward.
- `training.optimize_completion_batch_size=4` is the completion micro-batch
  size used by one optimizer step. It is not four prompts or four rollout
  groups.
- `training.replay_buffer_optimize_threshold` is derived as
  `optimize_completion_batch_size * rollout_group_size`; with defaults this is
  32 completions, usually four trainable rollout groups.
- `training.optimize_times_per_step=4` means the same ReplayBuffer completions
  are optimized for four passes. This replaces the ambiguous original
  `epochs_per_step` name.
- Compact `train_step` logs should separate generated work from training work:
  `decisions.rollout_attempts` counts all rollout attempts including retry,
  `decisions.terminal` counts final prompt outcomes, and
  `decisions.trainable` counts groups that enter ReplayBuffer. With defaults,
  `decisions.trainable.total * rollout_group_size` is the optimize input size.
- `batch` in logs is the optimize-trigger batch since the previous optimize
  step. It is not a dataset mini-batch and should not expose a generic
  `progress`; progress belongs to `run` or `epoch`.

## Native TP Boundary

- Production backend is `native-tp`.
- Allowed dependencies: PyTorch, Transformers tokenizer/config utilities,
  safetensors, PyYAML, PyTorch distributed, and repository-native LoRA/training
  code.
- Forbidden production training dependencies: NeMo, NeMo-RL, NGC NeMo
  containers, vLLM, Ray, DeepSpeed, FSDP, DDP, Accelerate,
  TransformerEngine/Apex as required dependencies, and ZeRO-style fallbacks.
- `hf-reference` may exist only as a single-process parity/debug backend. It is
  not the production multi-card route.
- KV cache may be used only for `generate_group()` rollout acceleration.
  `sequence_log_probs()`, ReplayBuffer contents, advantage computation,
  policy-ratio clipped loss, and `train_batch()` must remain full-sequence
  training semantics. Do not reuse rollout KV cache after a LoRA update.

## GPU And Remote Run Safety

Public repository files must not contain private hostnames, IP addresses,
usernames, relay endpoints, API keys, or concrete internal filesystem paths.
Keep site-specific deployment notes in ignored local files such as
`AGENTS.local.md`.

Host-key safety matters. If SSH reports a changed host key, stop and get an
explicit fingerprint confirmation before removing or bypassing known-hosts
entries.

## Remote Run Discipline

- Use isolated timestamped directories for server experiments unless the user
  explicitly asks to reuse a fixed directory.
- Record environment, command, config, git/worktree state, GPU state, and output
  directory in the run output.
- Choose target GPUs explicitly for every remote run; do not bake private GPU
  allocation policy into public scripts or docs.
- Long runs must be watched. Monitoring is about reward and group quality, not
  just process liveness.
- Start an independent GPU memory recorder before remote long runs when
  diagnosing OOM or allocator behavior. Prefer `scripts/record_gpu_memory.py`
  with the target GPUs and write its output under the run directory.

## Known Follow-Ups

- Native Qwen3.5/Qwen3.6 support still needs an exact text-only hybrid
  linear-attention kernel. Do not approximate those layers or silently fall back
  to Qwen3 full attention.
- Native Qwen TP adapter should eventually add vocab-parallel embedding/lm_head
  when replicated vocabulary weights become the bottleneck.
- Rank logs should be aggregated so global training events are printed once,
  while per-rank metrics remain available for diagnostics.
- Checkpoint resume needs an explicit smoke test: resume from `step_N`, continue
  for 1-2 steps, and verify global step, optimizer state, rng state, and LoRA
  tensor continuity.
