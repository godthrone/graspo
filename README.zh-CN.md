# GRASPO

[English README](README.md)

GRASPO 是一个独立实现的 **Group Relative Adaptive Structured Policy
Optimization** 训练项目，面向字段可验证的结构化输出任务，例如 JSON 抽取、分类、表单解析和
tool-call 参数生成。

这个仓库按 README-first 方式组织：安装依赖，跑 CPU smoke，准备 JSONL 数据，然后启动本地
reference 后端或 native Megatron 张量并行后端。

## 项目提供什么

- `megatron-native`：生产训练路线。GRASPO 自己控制 rollout、retry/filter、
  ReplayBuffer、reward、advantage、policy-ratio clipped loss、readable/raw 日志、
  LoRA checkpoint 和监控，同时使用开源 Megatron-LM/Core tensor parallel 进程组。
- `hf-reference`：单进程 Hugging Face 后端，用于算法 parity、小模型调试和本地 smoke。
- Qwen 风格 native tensor-parallel adapter，冻结 base weights，只训练 LoRA。
- 人类可读 rollout 日志、raw replay JSONL、per-rank 诊断和可恢复 LoRA TP checkpoint。

生产训练路线不依赖 NeMo、NeMo-RL、vLLM、Ray、DeepSpeed、FSDP、DDP、Accelerate、
TransformerEngine 或 Apex。

## 安装

推荐 Python 3.11+。

```bash
# 如果机器上还没有 uv，先安装 uv。即使系统 Python 较旧，uv 也会创建/使用
# Python >=3.11 的环境。
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev
```

如果不用 `uv`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

如果要使用 native Megatron 后端，请在同一个环境中安装兼容的开源 Megatron-LM 或 Megatron
Core。模型权重不要放进仓库。

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

启动 native Megatron TP=2 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
BACKEND=megatron-native \
MODEL_PATH=$HOME/models/Qwen3-8B \
DATA_PATH=$HOME/datasets/graspo/train.jsonl \
OUTPUT_DIR=outputs/qwen3-8b-tp2 \
CONFIG_PATH=configs/profiles/qwen3_8b_megatron_native_tp2_overnight.yaml \
bash scripts/run_train.sh
```

服务器上启动 nohup 长训可以用通用 launcher：

```bash
bash scripts/launch_megatron_native_tp2_remote.sh \
  --model-path $HOME/models/Qwen3-8B \
  --data-path $HOME/datasets/graspo/train.jsonl \
  --gpus 0,1 \
  --tag longrun
```

launcher 会把输出目录写到 `latest_graspo_longrun.out`，启动 `nohup` 训练，并同步启动
`scripts/record_gpu_memory.py`。

## 数据格式

训练数据是 JSONL，每行一条 prompt：

```jsonl
{"prompt":"Extract JSON with the APN and fault number.\nTicket: user 13800138000 cannot use apn cmnet.","ground_truth":{"APN":"cmnet","fault_number":"13800138000"}}
```

字段说明：

- `prompt`：文本 prompt。
- `ground_truth`：期望结构化输出，通常是 JSON object。
- `messages`：可选 chat messages；如果存在，可以用 tokenizer chat template 渲染。
- 其它字段作为 metadata 保留。

把本地 JSON、JSONL 或 spreadsheet 转成标准 JSONL：

```bash
python -m graspo prepare-data --input raw_data.jsonl --output outputs/train.jsonl
```

切分 train/eval：

```bash
SOURCE_DATA_PATH=outputs/train.jsonl \
TRAIN_OUTPUT_PATH=outputs/train_split.jsonl \
EVAL_OUTPUT_PATH=outputs/eval_split.jsonl \
bash scripts/split_train_eval_jsonl.sh
```

## 核心算法参数

默认 profile 使用本项目的 GRASPO 队列语义：

- `training.training_epoch_count=100`：完整数据集训练轮数。
- `training.rollout_group_size=8`：每个 prompt attempt 采样多少条 completion。
- `training.optimize_completion_batch_size=4`：每个 optimizer step 的 completion micro-batch。
- `training.replay_buffer_optimize_threshold=32`：由
  `optimize_completion_batch_size * rollout_group_size` 派生。
- `training.optimize_times_per_step=4`：同一批 replay completion 重复优化几轮。
- `training.rollout_max_retry_times=5`：首次 attempt 后最多额外 retry 次数。
- `training.max_new_tokens=2048`：真实训练生成长度。短测用 `max_steps` 截断，不降低生成长度。
- `training.policy_ratio_clip_eps=0.2`：policy-ratio clipped objective 的 epsilon。

组决策顺序：

1. `perfect_skip`：lower median reward 达到 perfect 阈值。
2. `trainable_max_correct`：至少一条 completion 完全正确。
3. `trainable_not_correct`：没有完全正确，但 `reward_max > reward_median`。
4. `retry`：还有 retry 预算。
5. `invalid`：硬 invalid 组。
6. `invalid_no_preference_gap`：retry 后仍没有有效偏好差异。

`invalid_no_preference_gap` 是信息抽取增强过滤：无满分组如果 `reward_max == reward_median`，说明没有可用偏好信号，不进入 ReplayBuffer。

## 输出文件

每次运行写入 `training.output_dir`：

- `nohup.out` 或 stdout：紧凑进度 JSON。
- `train.log`：rank-0 compact train events。
- `rollouts.readable.jsonl`：给人看的 prompt、completion、reward 和 debug 细节。
- `rollouts.raw.jsonl`：replay tensor、mask、old logprob、advantage 和 reward metadata。
- `train_batches.readable.jsonl`：每个 optimize-trigger batch 一行。
- `rank_metrics.rank_*.jsonl`：per-rank 显存、耗时、LoRA、optimizer 诊断。
- `checkpoints/step_*`：可恢复 LoRA TP checkpoint 和 optimizer state。

常用监控命令：

```bash
tail -f outputs/qwen3-8b-tp2/nohup.out
```

## Docker

构建镜像：

```bash
bash scripts/build_docker.sh
```

运行容器时挂载模型和数据目录。模型、数据、日志和 checkpoint 默认都被 Git 忽略。

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

仓库应只提交代码、配置、测试、脚本、README、样例数据、license 文件和依赖锁文件。

## 常见问题

- `MODEL_PATH is required`：把 `MODEL_PATH` 指向本地 Hugging Face 模型目录。
- `DATA_PATH does not exist`：传入包含 `prompt` 和 `ground_truth` 的 JSONL。
- tokenizer 没有 pad token：GRASPO 会尝试用 `eos_token` 作为 `pad_token`；如果两者都没有，需要先在 tokenizer 中定义。
- native backend import 失败：确认 PyTorch 和开源 Megatron-LM/Core 已安装在当前环境。
- rollout OOM：保持 `max_new_tokens=2048`，通过降低并发或 KV cache split 处理，不用降低生成长度换稳定性。

## License

GRASPO 使用 MIT License。见 [LICENSE](LICENSE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
