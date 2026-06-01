# 训练

## 后端

当前保留两条后端：

- `megatron-native`：生产路线。GRASPO 自己负责 rollout、retry/filter、
  ReplayBuffer、reward、advantage、loss、JSONL 日志和 checkpoint；开源
  Megatron-LM/Core 只提供 tensor-parallel 进程组与并行基础能力。
- `hf-reference`：单进程 Hugging Face 参考后端，用于小模型 parity 和本地调试。

NeMo、NeMo-RL、vLLM、Ray、DeepSpeed、FSDP、DDP、Accelerate 都不是生产训练路径。

## Native Megatron 命令

```bash
TP_SIZE=2 \
MODEL_PATH=/data/models/Qwen3-8B \
DATA_PATH=/data/graspo/train.jsonl \
OUTPUT_DIR=/data/graspo/outputs/native-tp2 \
CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_smoke.yaml \
bash scripts/run_train.sh
```

第一阶段目标是 Qwen3-8B、单机 TP=2、PP=1、LoRA-only、`max_new_tokens=2048`。
快速验收用 `MAX_STEPS=1-3` 或配置里的 `max_steps` 限制优化步数，不允许为了省时间
把真实训练的生成长度降到 128/256。

OOM 调试或长跑前建议先单独启动显存记录：

```bash
nohup python scripts/record_gpu_memory.py \
  --gpus 6,7 \
  --interval-sec 1 \
  --output-dir "$OUTPUT_DIR/gpu_memory" \
  --tag tp2-longrun \
  --pid-filter torchrun,python \
  > "$OUTPUT_DIR/gpu_memory/nohup.out" 2>&1 &
```

## 数据队列语义

首版严格跟随原版 GRASPO 队列：

- 每次消费一个 prompt。
- 该 prompt 的 `rollout_group_size` 条 completion 作为一个 tensor-parallel batch 生成。
- 按原版 group decision 执行 retry、perfect skip 和 invalid filtering。
- 可训练样本进入 ReplayBuffer。
- ReplayBuffer 达到 `optimize_completion_batch_size * rollout_group_size` 后开始优化，并执行
  `optimize_times_per_step` 轮。

其中 `optimize_completion_batch_size` 是优化阶段的 completion micro-batch，
不是 prompt group 数。默认 `4 * 8 = 32` 表示 ReplayBuffer 里累计 32 条
trainable completion 后触发一次 optimize，通常对应 4 个可训练 rollout group。
retry attempt 和 perfect-skip group 会产生 completion，但不会进入 ReplayBuffer。

`train_step.batch.decisions` 建议按三层理解：

- `rollout_attempts`：实际发生了多少次 rollout，包含 retry。
- `terminal`：prompt 最终去向，包含 perfect skip、trainable、invalid。
- `trainable`：最终进入 ReplayBuffer 的可训练组细分。

示例：

```json
"decisions": {
  "rollout_attempts": {"total": 59, "retry": 21, "terminal": 38},
  "terminal": {"perfect_skip": 31, "trainable": 4, "invalid": 3, "total": 38},
  "trainable": {"max_correct": 4, "not_correct": 0, "total": 4}
}
```

这样可以同时对齐“实际生成 completion 数”和“本次训练 completion 数”：

- `59 * 8 = 472` 条实际生成 completion。
- `31 + 4 + 3 = 38` 个最终结束的 prompt group。
- `4 * 8 = 32` 条 completion 进入 ReplayBuffer 并触发本次 optimize。

`batch` 是本次 optimize 的形成过程，不是数据集 mini-batch，因此不放独立
`progress`；进度属于 `epoch.progress` 或全局 run 统计。
更完整的实现口径见 [工程实现说明](engineering-implementation.md)。

## 长训练监控

GRASPO 的核心是长训练监控早停。监控不只是看进程是否报错，还要持续检查：

- reward 趋势、reward max/mean、组内 reward range。
- content_score 分布，以及是否长期全 0 或全 1。
- decision 分布：retry、invalid、perfect_skip、trainable。
- readable JSONL 里的 JSON 截断、缺少 ```json、JSON 无法解析等低分原因。
- finite loss/grad、非零 LoRA grad、LoRA norm delta、checkpoint 写入。
- GPU 显存、NCCL、rank 存活状态。
- `gpu_memory/gpu_memory.jsonl` 与 `rank_metrics.rank_*.jsonl` 的峰值和 reserved/allocated 差异。

若出现 NaN/inf、LoRA delta 长期为 0、reward 全 0/全 1、组内长期无差异、
content_score 全 0 或异常 JSON 截断窗口，应提前停止并修复。

## Checkpoint

`megatron-native` 保存可恢复的 per-rank LoRA TP checkpoint，包含本 rank LoRA
tensor、optimizer state、RNG state 和配置快照。v0.1 不要求导出 HF PEFT adapter
或合并完整模型。

## 首次验收

TP=2 验收需要确认：

- 日志里没有 forbidden framework import。
- 出现 `native_qwen_adapter_ready`。
- readable/raw rollout JSONL 存在。
- readable JSONL 能解释低分原因，并包含 ground truth、reward 细节和截断诊断。
- 至少一次 finite loss。
- LoRA grad count 非零，LoRA tensor 有变化。
- `step_N/` 和 `final/` checkpoint 含 per-rank 文件。
