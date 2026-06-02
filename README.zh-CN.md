# GRASPO

[English README](README.md)

GRASPO 是一个 README-first 的 **Group Relative Adaptive Structured Policy Optimization**
实现，用于结构化输出语言模型训练。它适合 JSON 信息抽取、分类、表单解析、tool-call
参数生成等可以逐字段打分的任务。

生产路线是自研 `native-tp`：GRASPO 在仓库内自己控制 rollout、retry/filter、ReplayBuffer、
reward、advantage、policy-ratio clipped loss、LoRA checkpoint 和监控，只使用 PyTorch
distributed process group 做 tensor parallel。

## 项目提供什么

- `native-tp`：生产级 tensor-parallel 后端。不依赖 Megatron、NeMo、vLLM、Ray、DeepSpeed、
  FSDP、DDP、Accelerate、TransformerEngine 或 Apex。
- `hf-reference`：单进程 Hugging Face 后端，用于算法 parity、小模型调试和 CPU/GPU smoke。
- 冻结 base 权重，只训练仓库自研 LoRA 模块。
- readable rollout 日志、raw replay JSONL、per-rank 诊断和可恢复 native TP LoRA checkpoint。

当前模型状态：

- Qwen3 dense causal LM，例如 Qwen3-8B：当前 native TP adapter 已支持。
- Qwen3.5/Qwen3.6 text-only checkpoint：已支持 config registry 和文本权重前缀识别；包含
  `linear_attention` 的 hybrid 文本层会 fail closed，直到实现精确 native linear-attention kernel。
  项目不会用错误近似层悄悄训练。

## 安装

推荐 Python 3.11+。

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev
```

不用 `uv` 也可以：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

模型权重和真实数据请放在仓库外，不要提交到 Git。

## 快速开始

先跑本地 smoke：

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

服务器 nohup 长训可以用通用 launcher：

```bash
bash scripts/launch_native_tp2_remote.sh \
  --model-path $HOME/models/Qwen3-8B \
  --data-path $HOME/datasets/graspo/train.jsonl \
  --gpus 0,1 \
  --tag longrun
```

launcher 会把输出目录写到 `latest_graspo_longrun.out`，启动 `nohup` 训练，并在旁边启动
`scripts/record_gpu_memory.py`。

## 数据格式

训练数据是 JSONL，每行一个 prompt：

```jsonl
{"prompt":"Extract JSON with the APN and fault number.\nTicket: user 13800138000 cannot use apn cmnet.","ground_truth":{"APN":"cmnet","fault_number":"13800138000"}}
```

支持字段：

- `prompt`：纯文本 prompt。
- `ground_truth`：期望结构化输出，通常是 JSON object。
- `messages`：可选 chat messages；存在时 tokenizer chat template 可以渲染 prompt。
- 其它字段会当作 metadata。

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

## 核心算法参数

生产 profile 使用这些 canonical 名字：

- `training.training_epoch_count=100`：完整数据集训练轮数。
- `training.rollout_group_size=8`：每个 prompt attempt 生成多少条 completion。
- `training.optimize_completion_batch_size=4`：每个 optimizer step 的 completion micro-batch。
- `training.replay_buffer_optimize_threshold=32`：由 `optimize_completion_batch_size * rollout_group_size` 派生。
- `training.optimize_times_per_step=4`：同一批 replay completion 重复优化几轮。
- `training.rollout_max_retry_times=5`：初始 attempt 之后最多额外 retry 次数。
- `training.max_new_tokens=2048`：真实训练生成长度。短测请改 `max_steps`，不要降低生产生成长度。
- `training.policy_ratio_clip_eps=0.2`：policy-ratio clipped objective 的 epsilon。

组决策顺序：

1. `perfect_skip`：lower median reward 达到 perfect 阈值。
2. `trainable_max_correct`：至少一条 completion 全对。
3. `trainable_not_correct`：没有全对，但 `reward_max > reward_median`。
4. `retry`：还有 retry 预算。
5. `invalid_no_preference_gap`：retry 后仍没有有效偏好差异。
6. `invalid`：fallback invalid group。

`invalid_no_preference_gap` 是信息抽取增强过滤：没有全对且 `reward_max == reward_median` 的组不进入
ReplayBuffer，因为没有有效偏好信号。最大 reward 达到 perfect 阈值的组必须是 `trainable_max_correct` 或
`perfect_skip`，不能落入 invalid。

## 输出文件

每个 run 写入 `training.output_dir`：

- `nohup.out` 或 stdout：紧凑进度 JSON。
- `train.log`：rank-0 训练事件。
- `rollouts.readable.jsonl`：prompt、completion、reward 和 debug 细节。
- `rollouts.raw.jsonl`：replay tensor、mask、old logprob、advantage 和 reward metadata。
- `train_batches.readable.jsonl`：每个 optimize-trigger batch 一行。
- `rank_metrics.rank_*.jsonl`：每 rank 显存、耗时、LoRA 和 optimizer 诊断。
- `checkpoints/step_*`：可恢复 LoRA native TP checkpoint 和 optimizer state。

常用监控命令：

```bash
tail -f outputs/qwen3-8b-tp2/nohup.out
```

## Native TP 说明

第一版生产实现会分片 Qwen dense 大矩阵：attention `q/k/v`、attention output、MLP
`gate/up/down` 和 LoRA target。embedding 和 LM head 当前复制保存。训练 logprob 使用精确
selected-token logprob，避免 loss 路径长期保留完整 vocab logits。

Qwen3.5/Qwen3.6 支持需要精确实现 hybrid linear-attention 文本层。registry 已经能识别这些
checkpoint 并忽略 vision weights，但训练会在 kernel 缺失时明确失败。

## 开发检查

```bash
python -m pytest -q
ruff check src tests
python -m graspo --help
```

开源卫生检查：

```bash
git ls-files
rg -n "10\\.1\\.|192\\.168|ssh -p" .
```

Git 只应跟踪代码、配置、测试、脚本、README、样例数据、license 文件和依赖 lock 文件。

## 常见问题

- `MODEL_PATH is required`：设置 `MODEL_PATH` 到本地 Hugging Face 模型目录。
- `DATA_PATH does not exist`：传入包含 `prompt` 和 `ground_truth` 的 JSONL。
- tokenizer 没有 pad token：GRASPO 会尝试使用 `eos_token` 作为 `pad_token`。
- native backend import 失败：确认当前环境安装了 PyTorch。
- rollout OOM：保持 `max_new_tokens=2048`；降低并发或使用 KV cache split，不要降低生产生成长度。

## License

GRASPO 使用 MIT License。见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。