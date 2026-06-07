# GRASPO

[English README](README.md)

GRASPO 是一个基于 GRPO 思路改进的强化学习训练器，适用于可以做结构化校验的语言模型任务，例如 JSON 生成、信息抽取、分类、表单解析和工具调用参数生成。
它面向基于 LoRA 的低成本结构化输出训练：冻结 base model，只训练紧凑的 LoRA adapter，并用可审计的规则 reward 从真实生成文本中判断好坏。
内置 reward 会检查必要标记、解析 fenced JSON 或 tool-call 内容，把结构化字段和 `ground_truth` 对齐比较，再转成 GRASPO 需要的组内偏好信号。

GRASPO 保留“同一个 prompt 采样多条 completion，并在组内比较 reward”的核心思想，同时加入更适合结构化输出训练的机制：

- 使用 LoRA/native memory-aware 路径时，可在单张 80 GB GPU 上对 9B 级模型做强化学习训练；
- rollout retry：低质量组先重试；
- perfect-answer skip：已经稳定答对的 prompt 不再消耗 optimizer budget；
- invalid 和 no-preference-gap filtering：丢弃没有有效偏好信号的组；
- completion-level ReplayBuffer 优化；
- readable reward/debug 日志，保留真实模型输出方便排查；
- 自研 `native-tp` 训练路径，支持 tensor parallel 和 pipeline placement；
- 冻结 base weights，只训练 LoRA。

生产训练内部使用 native TP/PP LoRA modules。PEFT 只作为外部兼容格式，用于 warm-start 导入和离线导出。v1 暂不支持全参数训练。

## 安装

推荐使用 Python 3.11。

```bash
git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
```

只有需要 Excel 转换时才安装可选数据依赖：

```bash
uv sync --extra dev --extra data --python 3.11
```

## 快速开始

复制并编辑根目录样例配置：

```bash
cp config_example.yaml my_graspo.yaml
```

至少需要设置这些字段：

- `model.model_path`：本地 Hugging Face 模型目录或模型 id；
- `data.train_path`：JSONL 训练数据；
- `training.output_dir`：输出目录；
- `launch.gpus`：当前节点使用的 GPU id；
- `backend_config.native_tp.tensor_model_parallel_size` 和
  `backend_config.native_tp.pipeline_model_parallel_size`：native placement 的 world size。

只传一个 YAML 参数启动训练：

```bash
uv run graspo launch --config config_example.yaml
```

短测时保持 `training.max_new_tokens=2048`，只降低 `training.max_steps`。真实 GRASPO 训练默认保持 `training.training_epoch_count=100`。

验证样例数据和 reward：

```bash
uv run graspo validate-reward --data data/sample.jsonl --limit 2
```

## 数据格式

训练数据是 JSONL，每行一个 prompt：

```jsonl
{"prompt":"Extract JSON with the APN and fault number.\nTicket: user 13800138000 cannot use apn cmnet.","ground_truth":{"APN":"cmnet","fault_number":"13800138000"}}
```

也支持 chat-style 和多模态记录：

```jsonl
{"messages":[{"role":"user","content":[{"type":"image","image":"images/panel_0001.png"},{"type":"text","text":"Extract the ticket fields as strict JSON."}]}],"ground_truth":{"ticket_id":"T-0001","status":"critical"}}
```

支持字段：

- `prompt`：纯文本 prompt；
- `ground_truth`：期望结构化输出，通常是 JSON object；
- `messages`：可选 chat messages，用于 tokenizer chat template；
- `image` / `images`：单张图片路径或图片路径列表；
- `video` / `videos`：数据层可以解析，但生产训练前应单独 smoke；
- 其它字段会作为 metadata。

## Reward 计分方式

GRASPO 当前提供一个内置结构化输出 reward。它是规则型、可审计的，适合目标答案为 JSON object 的任务。每条 completion 的计分流程分四步：

1. 检查输出标记。根据 `reward` 配置，scorer 可以要求 `<think>...</think>`、fenced JSON Markdown block，以及 `<tool_call>...</tool_call>`。
2. 抽取答案文本。answer 会从配置要求的标记区域里取出；如果 `check_json_markdown=false`，则直接把 answer 区域当作 JSON。
3. 解析并比较 JSON。completion JSON 会和 `ground_truth` 做递归 dict/list 匹配。目标字段存在会得部分分，值完全相等会继续加分，额外字段会降低结构化匹配度。
4. 归一化 reward。标记分、结构化内容分、完全正确 bonus，以及多余文本惩罚/奖励会合成 `reward`、`content_score` 和 `all_right`。

几个关键输出：

- `reward`：用于 GRASPO group decision、advantage 计算和 ReplayBuffer 训练的标量；
- `content_score`：组过滤前的结构化内容匹配分；
- `all_right`：只有所有检查目标都完全正确时才为 true。

GRASPO 使用的是同一 rollout group 内的 reward 分布，而不是单条 completion 的绝对分数。有有效差异的组会进入训练；已经 perfect 的组可以跳过；没有 reward 方差或没有偏好差异的组会被丢弃或重试。`rollouts.readable.jsonl` 会记录 completion、抽取字段、reward 细节和 invalid reason，方便不用重新生成就检查 reward 行为。

## 配置说明

所有常规训练配置都在 YAML 内完成。`config_example.yaml` 是完整公开样例。

### `backend`

- `native-tp`：生产多 GPU 路线。
- `hf-reference`：单进程 Hugging Face 参考后端，用于 parity 和小 smoke。
- `auto`：根据本机 GPU 和模型线索自动选择。

### `model`

- `model_path`：base Hugging Face 模型路径或 id。
- `trust_remote_code`：传给 Hugging Face loader。
- `torch_dtype`：模型 dtype，通常为 `bfloat16`。
- `attn_implementation`：可选 Hugging Face attention implementation。
- `gradient_checkpointing`：在支持时开启 gradient checkpointing。
- `chat_template_kwargs`：tokenizer chat template 的额外参数。

### `data`

- `train_path`：JSONL 训练文件。
- `prompt_field`：纯文本 prompt 字段名。
- `ground_truth_field`：期望答案字段名。
- `messages_field`：chat messages 字段名。
- `max_prompt_length`：prompt token 长度限制。

### `lora`

- `r`：LoRA rank。
- `alpha`：LoRA alpha。
- `dropout`：LoRA dropout。
- `adapter_path`：可选 PEFT adapter 目录，只用于 warm-start。
- `target_preset`：安全 target preset，例如 `language_safe`。
- `target_modules`：显式 LoRA targets。设置后，`lora.target_modules` 优先于 `target_preset`。
- `auto_target_modules`：未显式设置 targets 时是否允许自动检测。
- `bias`：PEFT-compatible bias 设置，通常为 `none`。
- `task_type`：PEFT-compatible task type，通常为 `CAUSAL_LM`。

GRASPO 当前只支持 LoRA 训练，不支持全参数训练。

### `reward`

- `check_think`：要求 `<think>...</think>` 标记。
- `check_json_markdown`：要求 fenced JSON 输出。
- `check_tool_call`：额外检查 tool-call target。
- `check_list_order`：结构化比较时 list 顺序是否敏感。
- `marker_reward_weight`：输出标记 reward 权重。
- `content_reward_weight`：结构化内容匹配 reward 权重。
- `anti_useless_str_reward_weight`：多余文本惩罚/奖励权重。
- `anti_useless_str_half_reward_len`：多余文本惩罚长度尺度。
- `answer_field`：ground-truth answer 字段。

### `training`

- `output_dir`：run 输出目录。
- `seed`：随机种子。
- `training_epoch_count`：完整数据集训练轮数，生产默认是 `100`。
- `max_steps`：短测/debug step 上限，`-1` 表示不限制。
- `rollout_prompt_queue_batch_size`：一次调度多少个 prompt group 做 rollout。
- `rollout_group_size`：每个 prompt attempt 采样多少条 completion。
- `optimize_completion_batch_size`：每个 optimizer step 的 completion micro-batch。
- `optimize_times_per_step`：同一批 replay completion 重复优化几轮。
- `rollout_max_retry_times`：初始 rollout 后的 retry 预算。
- `learning_rate`、`weight_decay`、`max_grad_norm`：optimizer 设置。
- `policy_ratio_clip_eps`：policy-ratio clipped objective epsilon。
- `max_new_tokens`：真实训练生成长度。保持 `training.max_new_tokens=2048`。
- `temperature`、`top_p`：rollout sampling 设置。
- `save_steps`：native checkpoint 间隔。
- `logging_steps`：紧凑训练日志间隔。
- `perfect_skip_reward_threshold`：跳过已解 prompt 的阈值。
- `dataloader_num_workers`：数据加载 worker 数。
- `resume_from_checkpoint`：可恢复 GRASPO native checkpoint 目录。

`training.replay_buffer_optimize_threshold` 由 `optimize_completion_batch_size * rollout_group_size` 派生，不能手动配置。`training.resume_from_checkpoint` 和 `lora.adapter_path` 互斥：前者恢复 native checkpoint 状态，后者只是 PEFT LoRA warm-start。

### `backend_config.native_tp`

- `tensor_model_parallel_size`：TP size。
- `pipeline_model_parallel_size`：PP size。
- `placement_strategy`：placement 策略，例如 `qwen3_tp` 或 `qwen36_pp8_static`。
- `sequence_parallel`：v1 必须保持 `false`。
- `train_micro_batch_size`：native train micro-batch size。
- `generation_micro_batch_size`：native generation micro-batch split。
- `use_kv_cache_for_rollout`：KV cache 只用于 rollout generation。
- `rollout_kv_cache_max_reserved_fraction`：rollout KV 显存预留比例。
- `empty_cache_after_rollout_split`、`empty_cache_before_train`：CUDA cache 控制。
- `checkpoint_format`：native recoverable checkpoint 格式标签。
- `raw_log_enabled`、`readable_log_enabled`：rollout/replay 日志开关。
- `synchronize_cuda_timing`：是否同步 CUDA timing。
- `pipeline_train_schedule`：pipeline train schedule，默认 `simple`。
- `pipeline_max_inflight_microbatches`：实验性 1F1B inflight 上限。

### `export`

- `final_formats`：可选 final checkpoint 后自动导出的格式列表，例如 `["peft-adapter"]`。step checkpoint 不会自动导出。

### `launch`

- `gpus`：当前节点使用的 GPU id。
- `nproc_per_node`：当前节点 worker 数。native-tp 下为空时会从 TP * PP / nodes 派生。
- `nnodes`、`node_rank`、`master_addr`、`master_port`：distributed launch 设置。
- `python`：可选 Python executable override。
- `torchrun`：可选 torchrun executable override。
- `env`：传给训练进程的额外环境变量。

## LoRA Targets

Preset 取值：

- `language_safe`：语言侧 `q_proj` 和 `v_proj`；
- `language_all_linear`：语言侧 attention、linear-attention 和 MLP 中已支持的线性矩阵；
- `vision_merger`：只训练 visual merger 线性层；
- `vision_common`：visual merger 加上已支持的 visual attention/MLP 线性层。

显式 `lora.target_modules` 可以使用 `language.self_attn.q_proj` 这样的 canonical name、`q_proj` 这样的 legacy leaf name，或 `visual.blocks.*.attn.*` 这样的 glob pattern。解析是 fail-closed：未知 target、不支持的 conv/norm 参数和空匹配都会在训练前报错。Native checkpoint 会保存 resolved LoRA target signature，并拒绝用不同 target 配置 resume。

## 导出

GRASPO native checkpoint 是可恢复训练 checkpoint。便携模型产物通过 `graspo export` 生成。

导出 PEFT LoRA adapter：

```bash
uv run graspo export --config config_example.yaml --checkpoint outputs/example-run/final --format peft-adapter --output outputs/export/adapter
```

导出 Hugging Face merged full model：

```bash
uv run graspo export --config config_example.yaml --checkpoint outputs/example-run/final --format merged-hf --output outputs/export/merged
```

`peft-adapter` 会从 GRASPO native rank shards 重建 PEFT `adapter_config.json` 和 `adapter_model.safetensors`，只支持 PEFT 能一对一表达的 targets。

`merged-hf` 会在 CPU 上流式读取 base HF safetensors，注入 LoRA delta，复制 tokenizer/config 等 sidecar 文件，并写出 HF-compatible merged model 目录。

Qwen3.5/Qwen3.6 中无法严格表示为 PEFT adapter 的 fused/split native targets，会在 `peft-adapter` 导出时严格失败。此类 checkpoint 请使用 `merged-hf`。

导出的 PEFT adapter 和 merged full model 是部署/兼容产物，不包含 optimizer、RNG、replay buffer 或 trainer state，不能替代 `step_*` 或 `final` 做完整训练恢复。

## 输出和监控

每个 run 写入 `training.output_dir`：

- `train.log`：rank-0 紧凑训练事件；
- `rollouts.readable.jsonl`：人类可读的 prompt、completion、reward 和 debug 细节；
- `rollouts.raw.jsonl`：replay tensors、masks、old logprobs、advantages 和 reward metadata；
- `train_batches.readable.jsonl`：每个 optimize-trigger batch 一行；
- `rank_metrics.rank_*.jsonl`：每 rank 显存、耗时、LoRA 和 optimizer 诊断；
- `step_*`：周期性可恢复 GRASPO native training checkpoint；
- `final`：干净退出后的最终可恢复 checkpoint。

健康的 GRASPO 训练不只是“进程没挂”。需要观察 reward trend、组内 reward range、content-score validity、decision distribution、finite loss/grad、非零 LoRA gradients、LoRA tensor changes、replay-buffer progress、checkpoint writes 和 GPU/NCCL health。

## 开发检查

```bash
uv run --extra dev ruff check src tests scripts
uv run --extra dev ruff format --check src tests scripts
uv run --extra dev pytest -q
uv run --extra dev python -m graspo --help
```

## 常见问题

- `model.model_path must be set`：编辑 `config_example.yaml`，指向真实 base model。
- `data.train_path does not exist`：将 `data.train_path` 指向 JSONL 文件。
- Native launch world size mismatch：让 `launch.nproc_per_node * launch.nnodes` 等于 `tensor_model_parallel_size * pipeline_model_parallel_size`。
- rollout OOM：保持 `training.max_new_tokens=2048`；降低 rollout 并发或 KV cache 预留，而不是降低生产生成长度。
- 需要 PEFT 兼容：通过 `lora.adapter_path` 加载 PEFT adapter，通过 `graspo export` 导出便携产物。

## License

GRASPO 使用 MIT License。见 [LICENSE](LICENSE)。
