# 工程实现说明

更新时间：2026-05-29。

本文档记录当前工程实现如何对齐 ELAM/原版 GRASPO 设计，以及哪些地方是有意偏离或后续待补。

## 核对依据

仓库中可直接核对的原始实现位于 `原版graspo算法/graspo/`：

- `graspo_training_arguments.py`：原版训练参数默认值。
- `group_relative_adaptive_supervised_policy_optimization.py`：epoch 主循环、ReplayBuffer 触发条件。
- `graspo_group_sample.py`：rollout、perfect skip、retry、invalid、trainable 判定。
- `graspo_replay_buffer.py`：completion-level ReplayBuffer。
- `graspo_group_optimize.py`：ReplayBuffer DataLoader、重复优化轮数、梯度裁剪。
- `graspo_rollout.py`：group rollout、old logprob、sample-std advantage。
- `graspo_loss.py`：policy ratio clipped loss。

当前工程实现对应位置：

- `src/graspo/core/graspo_parity.py`：原版 group decision 与 replay threshold 的纯逻辑。
- `src/graspo/backends/megatron_native/trainer.py`：生产训练控制流。
- `src/graspo/backends/megatron_native/qwen_tp_adapter.py`：Qwen3 Megatron TP runtime。
- `src/graspo/core/schema.py`：canonical 配置字段与旧字段 alias。

## 核心术语

- `prompt`：数据集中的一条训练样本输入。
- `completion`：某个 prompt 的一条模型输出，不是数据集样本。
- `rollout_group`：同一个 prompt 一次生成出的 `rollout_group_size` 条 completion。
- `rollout_attempt`：一次实际生成动作。初始 rollout 和 retry rollout 都是 attempt。
- `retry_attempt`：因为当前组 `reward_max` 未达到阈值而追加生成的中间 attempt。
- `terminal_group`：一个 prompt 最终结束后的组，去向只能是 perfect skip、trainable 或 invalid。
- `ReplayBuffer`：只保存 trainable group 拆出来的 completion-level experience。
- `optimize step`：ReplayBuffer 达到阈值后执行的一次训练触发。
- `batch`：日志中的 batch 指“从上一次 optimize 到本次 optimize 之间，为攒够 ReplayBuffer 阈值经历的所有 rollout attempts”。它不是数据集 mini-batch，也不应该有独立 progress。

## 参数对齐

| 原版字段 | 当前 canonical 字段 | 默认 | 含义 |
|---|---|---:|---|
| `total_epoch` | `training_epoch_count` | `100` | 整个数据集训练轮数 |
| `group_size` | `rollout_group_size` | `8` | 每个 prompt 一次 rollout 生成的 completion 数 |
| `train_batch_size` | `optimize_completion_batch_size` | `4` | 优化阶段每个 optimizer step 使用的 completion micro-batch |
| `train_batch_size * group_size` | `replay_buffer_optimize_threshold` | `32` | ReplayBuffer 攒够多少条 completion 后触发 optimize，派生值，不允许手配 |
| `epochs_per_step` | `optimize_times_per_step` | `4` | 同一批 ReplayBuffer completion 重复优化几轮 |
| `max_make_new_group_retry_times` | `rollout_max_retry_times` | `5` | 初始 rollout 后最多额外 retry 次数 |
| `clip_eps` | `policy_ratio_clip_eps` | `0.2` | 新旧策略 logprob ratio 的裁剪 epsilon |
| `max_norm` | `max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `lr` | `learning_rate` | `5e-6` | AdamW 学习率 |
| 首轮 median 阈值 | `perfect_skip_reward_threshold` | `1.0` | 初始 rollout lower-median 达到阈值则 perfect skip |
| `max_new_tokens` | `max_new_tokens` | `2048` | 当前真实训练固定为 2048，避免结构化 JSON 被截断 |

旧字段只作为 YAML 读取 alias 兼容，不应出现在生产 profile 和新文档中。

## 原版训练语义

原版主循环每个 epoch 遍历训练集，每条 prompt 执行以下流程：

1. 对一个 prompt 生成 `group_size` 条 completion。
2. 计算每条 completion 的 reward 和 content score。
3. 若首次 rollout 的 median reward `>= 1`，该 prompt 计入 perfect skip，不训练。
4. 若当前组 `reward_max < 1`，且 retry 次数未超过 `max_make_new_group_retry_times`，继续 retry rollout。
5. 若 retry 耗尽后仍未拿到满分样本，记录困难样本；当前组可能成为 `trainable_not_correct` 或 invalid。
6. 若 `0 < content_score.min == content_score.max < 1`，或 reward 全相同，判为 invalid，不进入 ReplayBuffer。
7. 其余可训练组进入 ReplayBuffer：
   - `reward_max >= 1` 为 `trainable_max_correct`。
   - `reward_max < 1` 为 `trainable_not_correct`。
8. ReplayBuffer 长度达到 `train_batch_size * group_size`，或 epoch 末尾仍有剩余 experience，则触发 optimize。
9. optimize 中用 `DataLoader(batch_size=train_batch_size, shuffle=True, drop_last=True)` 取 completion micro-batch，重复 `epochs_per_step` 轮。
10. loss 为无 critic 的 policy-ratio clipped objective，advantage 为组内 sample-std 标准化 reward。

## 当前实现状态

当前 `megatron-native` 生产路径保持上述核心语义：

- 每次消费一个 prompt。
- 每个 prompt 的 `rollout_group_size` 条 completion 在同一个 TP runtime 中批量生成。
- `perfect_skip_reward_threshold=1.0`，首轮 lower-median 达标则跳过。
- retry 上限为 `rollout_max_retry_times`，因此最多 attempts 为 `rollout_max_retry_times + 1`。
- invalid 过滤保留 reward 无方差和 uniform partial content 规则。
- ReplayBuffer 保存 completion-level experience。
- optimize 触发阈值为 `optimize_completion_batch_size * rollout_group_size`。
- `sequence_log_probs()`、advantage、policy-ratio clipped loss、`train_batch()` 语义不因 KV cache 改变。
- `use_kv_cache_for_rollout=true` 只加速 rollout 生成，不复用到训练 forward。
- KV cache 只在单次 rollout attempt 内有效；LoRA 更新后不会复用旧 cache，避免 stale cache 影响下一轮生成。

当前有意偏离：

- 原版 `max_new_tokens=1024`；当前真实训练固定 `2048`，防止复杂 JSON 被截断。
- 原版可选困难样本 SFT/LoRA merge 流程当前不在生产训练路径中。
- 当前周末长跑以 Qwen3-8B、单机 TP=2、LoRA-only 为目标，不包含 TP=8 或更大模型。

需要继续核对或补齐：

- 原版每个 epoch 会 `random.shuffle(prepared_data_list)`；当前 native trainer 是否需要显式 epoch shuffle 仍需单独确认。
- checkpoint resume 需要继续做 `step_N` 恢复后再跑 1-2 step 的 smoke。
- Megatron Core 并行 primitive 化、KV cache 显存峰值优化、rank 日志聚合仍是后续工程债。

## 监控日志口径

stdout 的 `train_step` 只保留监控必要信息，详细 completion 放在 `rollouts.readable.jsonl`。

`train_step` 分三层看：

- `run`：整个训练进程累计状态。
- `epoch`：当前 epoch 内数据集样本进度，允许有 `progress`。
- `batch`：本次 optimize batch 的组成和健康度，不放 `progress`。

batch 级决策使用以下结构，避免把 retry attempt 和最终样本去向混在一起：

```json
"decisions": {
  "rollout_attempts": {
    "total": 59,
    "retry": 21,
    "terminal": 38
  },
  "terminal": {
    "perfect_skip": 31,
    "trainable": 4,
    "invalid": 3,
    "total": 38
  },
  "trainable": {
    "max_correct": 4,
    "not_correct": 0,
    "total": 4
  }
}
```

对账方式：

- 实际生成 completion 数：`rollout_attempts.total * rollout_group_size`。
- 最终样本去向：`perfect_skip + trainable + invalid = terminal.total`。
- 本次入训 completion 数：`trainable.total * rollout_group_size`。

例如 `59 * 8 = 472` 条实际生成 completion，其中 `4 * 8 = 32` 条进入 ReplayBuffer 并触发 optimize。

注意：2026-05-29 周末长跑启动后，本地代码已将日志结构改为上述 `rollout_attempts` / `terminal` / `trainable` 三层形式；228 上已在跑的周末长训没有自动同步和重启，因此它的现有日志仍可能保留旧的扁平 `decisions` 字段。后续重启或部署新代码后，以本节结构为准。

## 文件分工

- `nohup.out` / `train.log`：人类和 heartbeat 监控读取，短、稳定、有时间戳。
- `train_batches.readable.jsonl`：每行一个 optimize batch summary，不嵌完整 completion。
- `rollouts.readable.jsonl`：逐 attempt 的 prompt、completion、reward 和 debug 细节。
- `rollouts.raw.jsonl`：tensor、logprob、advantage 等 replay/debug 原始数据。
- `rank_metrics.rank_*.jsonl`：per-rank 显存、KV rollout timing、checkpoint 后状态。

## KV Cache 边界

KV cache 是 rollout 加速，不是训练语义改动：

- `generate_group()` 可使用 `use_kv_cache_for_rollout=true` 做 prefill + incremental decoding。
- attention 支持 `query_len != key_len`，prefill 后每步只输入最后一个 token。
- `rollout_kv_cache_max_reserved_fraction=0.60` 限制单次 rollout cache 的显存预算。
- `empty_cache_after_rollout_split=true` 时，被 KV 预算拆分的长 rollout 会在 attempt 后主动释放 allocator cache。
- 若估算超过预算，只拆分 generation micro-batch，例如 `8 -> 4 -> 2 -> 1`；不降低 `max_new_tokens=2048`。
- cache 不写入 ReplayBuffer，不进入 raw JSONL，不参与 checkpoint。
- `sequence_log_probs()` 继续 full-sequence forward，用于计算 old/new policy logprob。
- `train_batch()` 继续 full-sequence forward，loss、advantage、mask、policy-ratio clip 语义保持不变。
- 每个 rollout attempt 内的 cache 在 attempt 结束后丢弃；optimizer 更新 LoRA 后，下一次 rollout 重新 prefill。

验收 KV cache 时要同时看速度和语义健康：`rollout_use_kv_cache=true`、`rollout_generation_micro_batch_size`、rollout timing、reward/JSON 统计不漂移、loss/grad finite、LoRA delta 非零。

## 训练显存治理

显存治理不能改变 GRASPO 训练语义。当前优先级如下：

1. activation checkpointing：native Qwen decoder layer 在训练 forward 中 recompute activation，降低 `loss.backward()` 峰值；rollout 和 no-grad logprob 不启用。
2. 原版 LoRA target：原版复现 profile 默认只训练 `q_proj` / `v_proj`，保留实验 profile 扩大 target 的能力。
3. KV cache budget：限制 rollout cache 并发，不限制输出长度。
4. Megatron-style TP：后续继续做 vocab-parallel lm_head / selected-token logprob，以及 `ColumnParallelLinear` / `RowParallelLinear` primitive 化。

远端 OOM 调试需要先启动独立显存记录脚本：

```bash
nohup python scripts/record_gpu_memory.py \
  --gpus 6,7 \
  --interval-sec 1 \
  --output-dir "$OUTPUT_DIR/gpu_memory" \
  --tag tp2-longrun \
  --pid-filter torchrun,python \
  > "$OUTPUT_DIR/gpu_memory/nohup.out" 2>&1 &
```

该脚本输出 `gpu_memory.jsonl`、`gpu_processes.jsonl`、`gpu_memory_summary.json`，用于和 `rank_metrics.rank_*.jsonl` 对齐分析 `allocated`、`reserved`、`nvidia-smi` 和进程显存差异。

## 周末长跑验收重点

- `run_start.config` 使用 canonical 字段，且 `max_new_tokens=2048`。
- `replay_buffer_optimize_threshold=32`，约 4 个 trainable group 触发一次 optimize。
- `loss_mean`、`grad_norm_mean` finite，`lora_delta_mean` 非零。
- reward 不坍塌为全 0 或全 1，组内 reward range 不长期为 0。
- JSON truncation、invalid JSON、missing marker 不异常。
- checkpoint 每 `save_steps` 写出，且后续需要恢复验证。
- GPU 6/7 显存峰值和 reserved/allocated 差异持续记录。
