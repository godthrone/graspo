# 配置说明

主配置是 `configs/graspo.yaml`。当前生产路线只保留 `megatron-native`
和本地 parity 用的 `hf-reference`。

## Backend

```yaml
backend: auto
```

- `megatron-native`：单机 tensor parallel 生产后端。
- `hf-reference`：单进程参考后端，只用于小模型 parity 和本地调试。
- `auto`：多 GPU 且检测到 Megatron-LM/Core 时选择 `megatron-native`。

Native Megatron v1 只支持单机 TP、PP=1：

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

第一阶段生产目标是 LoRA-only：

```yaml
lora:
  auto_target_modules: false
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## 训练默认

- 真实 GRASPO 训练固定使用 `max_new_tokens: 2048`。
- 快速检查只能减少 `max_steps`，不能降低生成长度。
- 默认 `training_epoch_count: 100`。
- GRASPO 依赖长训练监控和异常早停，不是短 epoch 试跑。

关键参数：

- `rollout_group_size`：每个 prompt 的 completion 数。
- `rollout_max_retry_times`：自适应重试次数。
- `optimize_completion_batch_size`：优化阶段 completion micro-batch，不是 prompt group 数。
- `replay_buffer_optimize_threshold`：派生值，等于
  `optimize_completion_batch_size * rollout_group_size`，默认 `32`，不允许在 YAML 中手配。
- `optimize_times_per_step`：ReplayBuffer 同一批经验复用轮数。
- `max_new_tokens`：生成长度，真实训练必须为 `2048`。
