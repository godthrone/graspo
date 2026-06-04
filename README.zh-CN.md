# GRASPO

[English README](README.md)

GRASPO 是一个基于 GRPO 改进的强化学习算法，适用于可以做结构化校验的语言模型任务，例如 JSON 生成、信息抽取、分类、表单解析和工具调用参数生成。

相比通用 GRPO，GRASPO 增加了 rollout retry、perfect-answer skip、invalid group filtering、no-preference-gap filtering、completion-level ReplayBuffer 训练，以及 readable reward/debug 日志。这些机制让它更适合格式化输出场景，因为模型回复可能是格式错误、部分正确、完全正确，或者只是缺少少数字段。

## 为什么需要 GRASPO

GRPO 适合一组 sampled completions 内存在有效 reward 差异的场景。结构化输出任务会额外遇到几类问题：

- 很多 completion 会因为缺少 JSON fence、tool-call marker、必填字段或可解析结构而无效；
- 有些 prompt 已经稳定答对，不应该继续消耗 optimizer budget；
- 有些没有全对答案的组 reward 完全相同或几乎相同，没有有效偏好信号；
- 长格式化答案可能被截断，如果不保存真实模型输出，很难 debug reward 为什么低。

GRASPO 保留 GRPO 在 rollout group 内比较 completion 的核心思想，同时加入面向结构化输出的过滤、重试和 replay 机制，让训练集中在真正能教会模型的样本组上。

这个仓库关注一条容易审查、容易改造的生产路线：

- GRASPO 算法、replay queue、reward、日志和 LoRA 训练都保留在本仓库；
- 多 GPU 后端是自研 `native-tp`，基于 PyTorch distributed tensor parallel；
- 生产训练不需要 Megatron、NeMo、vLLM、Ray、DeepSpeed、FSDP、DDP、Accelerate、TransformerEngine、Apex 或 ZeRO fallback。

## 算法流程

对每条数据，GRASPO 执行：

1. 渲染一个 prompt；
2. 生成 `rollout_group_size` 条 completion；
3. 将每条 completion 和 `ground_truth` 对比打分；
4. 将这个组判定为 perfect、trainable、retry 或 invalid；
5. 将可训练 completion-level experience 放进 ReplayBuffer；
6. ReplayBuffer 达到阈值后优化 LoRA 参数；
7. 保存 readable/raw JSONL 日志，方便从真实模型输出 debug 低分原因。

相对 plain GRPO，主要改进点是：

- `retry`：低质量组先重试，最多到 `rollout_max_retry_times`；
- `perfect_skip`：lower median reward 已经 perfect 的组直接跳过；
- `invalid`：丢弃无 reward 方差、uniform partial content 等硬无效组；
- `invalid_no_preference_gap`：丢弃没有全对且 max reward 不高于 median 的无偏好差异组；
- ReplayBuffer 优化：保存 completion-level experience，并重复优化 `optimize_times_per_step` 轮；
- readable/raw 日志：分开保存模型输出、reward 细节、mask、logprob 和 metadata，便于监控和 debug。

组决策顺序：

1. `perfect_skip`：lower median reward 达到 perfect 阈值；
2. `trainable_max_correct`：至少一条 completion 全对；
3. `trainable_not_correct`：没有全对，但 `reward_max > reward_median`；
4. `retry`：还有 retry 预算；
5. `invalid`：原始硬过滤，例如无 reward 方差或 uniform partial content；
6. `invalid_no_preference_gap`：retry 后仍没有有效偏好差异。

`invalid_no_preference_gap` 是信息抽取增强过滤：没有全对、没有触发原始 invalid、且 `reward_max == reward_median` 的组不进入 ReplayBuffer，因为它没有有效偏好信号。

## 当前状态

当前已支持：

- Qwen3 dense causal LM，例如 Qwen3-8B；
- 冻结 base weights，只训练 LoRA；
- attention 和 MLP 大矩阵 native TP 分片；
- 训练 loss 路径使用精确 selected-token logprob；
- rollout KV cache 和 generation micro-batch split；
- readable rollout 日志、raw replay 日志、per-rank metrics 和可恢复 LoRA native TP checkpoint；
- `hf-reference` 单进程 Hugging Face 后端，用于 parity 和 smoke test。

下一步计划：

- 为 Qwen3.5/Qwen3.6 实现精确 text-only hybrid `linear_attention` kernel；
- 增加 PEFT adapter 离线导入/导出；
- 如果 replicated vocab weights 成为瓶颈，再做 vocab-parallel embedding/lm_head。

Qwen3.5/Qwen3.6 的 text config 已能识别，vision weights 不进入训练范围；但包含 hybrid `linear_attention` 的 checkpoint 会在 kernel 实现前明确失败，不会用近似的 Qwen3 full-attention 层悄悄训练。仓库里对应 profile 已命名为 `experimental_*`，只用于 config/load smoke，不是生产长训入口。

## 安装

推荐并固定使用 Python 3.11。

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
```

不用 `uv` 也可以：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

模型权重、真实数据、日志和 checkpoint 请放在仓库外。
只有需要 Excel 转换时才安装可选数据依赖：

```bash
uv sync --extra dev --extra data --python 3.11
```

## 快速开始

先跑 CPU smoke：

```bash
bash scripts/smoke_cpu.sh
```

验证样例数据和 reward：

```bash
python -m graspo validate-reward --data data/sample.jsonl --limit 2
```

用小模型跑单进程 reference：

```bash
BACKEND=hf-reference \
MODEL_PATH=$HOME/models/small-causal-lm \
DATA_PATH=data/sample.jsonl \
OUTPUT_DIR=outputs/hf-reference-demo \
CONFIG_PATH=configs/graspo.yaml \
bash scripts/run_train.sh
```

用 Qwen3-8B 跑 native TP TP=2：

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

服务器 nohup 长训：

```bash
bash scripts/launch_native_tp2_remote.sh \
  --model-path $HOME/models/Qwen3-8B \
  --data-path $HOME/datasets/graspo/train.jsonl \
  --gpus 0,1 \
  --tag longrun
```

launcher 会把输出目录写入 `latest_graspo_longrun.out`，启动 `nohup` 训练，并在同一个 run 目录下启动 `scripts/record_gpu_memory.py`。

## 数据格式

训练数据是 JSONL，每行一个 prompt：

```jsonl
{"prompt":"Extract JSON with the APN and fault number.\nTicket: user 13800138000 cannot use apn cmnet.","ground_truth":{"APN":"cmnet","fault_number":"13800138000"}}
```

支持字段：

- `prompt`：纯文本 prompt；
- `ground_truth`：期望结构化输出，通常是 JSON object；
- `messages`：可选 chat messages，用于 tokenizer chat template；
- 其它字段会作为 metadata。

准备数据：

```bash
python -m graspo prepare-data --input raw_data.jsonl --output outputs/train.jsonl
```

拆分 train/eval：

```bash
SOURCE_DATA_PATH=outputs/train.jsonl \
TRAIN_OUTPUT_PATH=outputs/train_split.jsonl \
EVAL_OUTPUT_PATH=outputs/eval_split.jsonl \
bash scripts/split_train_eval_jsonl.sh
```

## 核心配置

生产 profile 使用这些 GRASPO canonical 名字：

- `training.training_epoch_count=100`：完整数据集训练轮数；
- `training.rollout_prompt_queue_batch_size=1`：每次从 prompt 队列中取多少个 prompt 一起 rollout。每个 prompt 仍然独立拥有自己的 `rollout_group_size` 条 completion，以及独立 retry/decision 路径；
- `training.rollout_group_size=8`：每个 prompt attempt 生成多少条 completion；
- `training.optimize_completion_batch_size=4`：每个 optimizer step 的 completion micro-batch；
- `training.replay_buffer_optimize_threshold=32`：由 `optimize_completion_batch_size * rollout_group_size` 派生；
- `training.optimize_times_per_step=4`：同一批 replay completion 重复优化几轮；
- `training.rollout_max_retry_times=5`：初始 attempt 后最多额外 retry 次数；
- `training.max_new_tokens=2048`：真实训练生成长度。短测请改 `max_steps`，不要降低生产生成长度；
- `training.policy_ratio_clip_eps=0.2`：policy-ratio clipped objective 的 epsilon。

## Rollout 性能

Rollout 通常是最大的耗时来源。Native TP 支持一次批量调度多个 prompt group 做 rollout，同时保持 GRASPO 语义不变：retry、perfect skip、invalid filtering、ReplayBuffer、advantage 和 loss 都仍然按 prompt group 独立计算。

主要调参项：

```yaml
training:
  rollout_prompt_queue_batch_size: 2

backend_config:
  native_tp:
    rollout_kv_cache_max_reserved_fraction: 0.90
```

在两张 80 GB GPU 上训练 Qwen3-8B 时，推荐加速配置是 `rollout_prompt_queue_batch_size=2`、KV fraction `0.90`。它能显著提升 rollout GPU 利用率，同时长训峰值显存远离 80 GB 边界。`rollout_prompt_queue_batch_size=3` 可能通过短 profiling，但真实长训会遇到 allocator 和长序列峰值，因此只建议作为 profiling 边界，不建议作为稳定默认。

## 输出和监控

每个 run 写入 `training.output_dir`：

- `nohup.out` 或 stdout：紧凑进度 JSON；
- `train.log`：rank-0 训练事件；
- `rollouts.readable.jsonl`：prompt、completion、reward 和 debug 细节；
- `rollouts.raw.jsonl`：replay tensor、mask、old logprob、advantage 和 reward metadata；
- `train_batches.readable.jsonl`：每个 optimize-trigger batch 一行；
- `rank_metrics.rank_*.jsonl`：每 rank 显存、耗时、LoRA 和 optimizer 诊断；
- `checkpoints/step_*`：可恢复 LoRA native TP checkpoint 和 optimizer state。

常用监控命令：

```bash
tail -f outputs/qwen3-8b-tp2/nohup.out
```

健康的 GRASPO 训练不只是“进程没挂”。需要观察 reward、content score、组内 reward range、decision 分布、finite loss/grad、非零 LoRA delta、checkpoint 写入和 GPU/NCCL 健康。

## 开发检查

```bash
python -m pytest -q
ruff check src tests
python -m graspo --help
```

Git 只应该跟踪代码、配置、测试、脚本、README、样例数据、license 文件和依赖 lock 文件。

## 常见问题

- `MODEL_PATH is required`：设置 `MODEL_PATH` 到本地 Hugging Face 模型目录；
- `DATA_PATH does not exist`：传入包含 `prompt` 和 `ground_truth` 的 JSONL；
- tokenizer 没有 pad token：GRASPO 会尝试使用 `eos_token` 作为 `pad_token`；
- native backend import 失败：确认当前环境安装了 PyTorch；
- rollout OOM：保持 `max_new_tokens=2048`；降低 `rollout_prompt_queue_batch_size` 或使用 KV cache split，不要降低生产生成长度。

## License

GRASPO 使用 MIT License。见 [LICENSE](LICENSE)。
