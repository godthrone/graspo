# Configuration

The main config is `configs/graspo.yaml`. The first native server profile is
`configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml`.

## Backend

```yaml
backend: auto
```

Supported values:

- `megatron-native`: production backend for single-node tensor parallelism.
- `hf-reference`: single-process reference backend.
- `auto`: chooses `megatron-native` when multiple GPUs and Megatron-LM/Core are
  detected, otherwise chooses `hf-reference`; large-model paths fail early if
  Megatron is unavailable.

Native Megatron v1 supports single-node TP only:

```yaml
backend_config:
  megatron_native:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    sequence_parallel: false
    train_micro_batch_size: 1
    generation_micro_batch_size: 1
    raw_log_enabled: true
    readable_log_enabled: true
```

## LoRA

The first production target is LoRA-only:

```yaml
lora:
  auto_target_modules: false
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Training

Important knobs:

- `rollout_group_size`: completions sampled for one prompt in one rollout attempt.
- `rollout_max_retry_times`: extra rollout attempts after the initial group.
- `optimize_completion_batch_size`: completion micro-batch size for one optimizer step.
- `replay_buffer_optimize_threshold`: derived as
  `optimize_completion_batch_size * rollout_group_size`; this is the number of
  completion-level experiences required before one optimize step is triggered.
- `optimize_times_per_step`: how many passes to train over the same ReplayBuffer batch.
- `max_new_tokens`: generation budget. Real GRASPO training uses `2048`;
  reduce `max_steps` for quick checks instead of lowering generation length.
- `training_epoch_count`: default long-training budget is `100`; GRASPO relies on
  monitored early stopping, not short fixed epochs.
