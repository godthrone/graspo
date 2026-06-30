# GRASPO

[English README](README.md)

GRASPO 是一个面向结构化输出任务的 GRPO-style 强化学习训练器，适合 JSON 生成、信息抽取、分类、表单解析和工具调用参数生成等可以自动校验答案的场景。它面向基于 LoRA 的低成本训练：冻结 base model，只训练紧凑 LoRA adapter，用可审计的 reward 规则从真实生成文本里判断好坏。

内置 reward 会检查必要输出标记，解析 fenced JSON 或 tool-call 内容，将结构化字段和 `targets` 对齐比较，再转成 GRASPO 需要的组内偏好信号。使用 LoRA/native memory-aware 路径时，GRASPO 的目标是支持单张 80 GB GPU 对 9B 级别模型做强化学习训练。

GRASPO 的训练主线是：

- 同一条 prompt/context 采样多条 completion，并在组内比较 reward；
- 低质量 rollout group 自动 retry；
- 已经稳定答对的 prompt 可以 perfect-skip，避免浪费 optimizer budget；
- 格式损坏的 group 会被过滤：当最好 completion 有 parse error 或 tool-call count mismatch 时，group 会被 retry 或丢弃，不参与训练；
- 没有 reward 方差或没有偏好差异的 group 会被过滤；
- ReplayBuffer 保存 completion-level experience；
- readable 日志保留真实 messages、completion、reward 和 debug 细节；
- 生产训练只使用自研 GraspoFlow TP/PP LoRA 路径；
- 只支持 LoRA 训练，不支持全参数训练。

训练内部使用 native TP/PP LoRA modules。PEFT 只作为外部兼容格式，用于 warm-start 导入和离线导出。

## 安装

推荐 Python 3.11。

```bash
git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
```

## 快速开始

复制并编辑根目录完整样例配置：

```bash
cp config_example.yaml my_graspo.yaml
```

至少需要设置：

- `model.model_path`：本地 Hugging Face 模型目录或模型 id；
- `data.train_path`：JSONL 训练数据；
- `training.output_dir`：run 输出目录；
- `launch.gpus`：当前节点使用的 GPU id；
- `graspoflow.tp_size` 和 `graspoflow.pp_size`：native TP/PP world size。

训练只需要一个 YAML 参数：

```bash
uv run graspo launch --config config_example.yaml
```

短测时保持 `training.max_new_tokens=2048`，只降低 `training.max_steps`。真实训练默认保持 `training.training_epoch_count=100`，除非你刻意做有限步数测试。

验证样例数据和 reward：

```bash
uv run graspo validate-reward --data data/sample.jsonl --limit 2
```

## 数据格式

训练数据只支持 JSONL。每行是一条由 chat messages 表示的 prompt/context、可选工具声明，以及一个或多个可接受的 reward 目标：

```jsonl
{"messages":[{"role":"system","content":"You extract structured telecom ticket fields as fenced JSON."},{"role":"user","content":"Ticket: user 13800138000 cannot use apn cmnet."},{"role":"assistant","content":"I will identify the phone number and APN from the ticket."},{"role":"user","content":"Extract JSON with the APN and fault number."}],"targets":[{"id":"expected","output":{"content":{"APN":"cmnet","fault_number":"13800138000"}}}]}
```

多模态数据也使用同一个 `messages` 字段，role 和 content 顺序会保真进入 tokenizer/processor：

```jsonl
{"messages":[{"role":"system","content":"Extract fields from ticket screenshots."},{"role":"user","content":"Use exact snake_case values."},{"role":"assistant","content":"Understood."},{"role":"user","content":[{"type":"image","image":"images/panel_0001.png"},{"type":"text","text":"Extract the ticket fields as strict JSON."}]}],"targets":[{"id":"expected","output":{"content":{"ticket_id":"T-0001","status":"critical"}}}]}
```

工具调用数据可以在可选 `tools` 字段中提供模型原生工具声明。GRASPO 会在运行时把 `messages + tools` 交给 tokenizer 或 processor 的 chat template；用户不需要、也不应该在数据集中提前渲染模型模板字符串：

```jsonl
{"messages":[{"role":"system","content":"Use tools when needed. Output only the tool call."},{"role":"user","content":"Query device OLT-17 status at 2026-06-08 10:30."}],"tools":[{"type":"function","function":{"name":"query_device_status","description":"Query network device panel status.","parameters":{"type":"object","properties":{"device_id":{"type":"string"},"panel_time":{"type":"string"}},"required":["device_id","panel_time"]}}}],"targets":[{"id":"expected","output":{"tool_calls":[{"name":"query_device_status","arguments":{"device_id":"OLT-17","panel_time":"2026-06-08T10:30:00+08:00"}}]}}]}
```

可运行的工具调用数据样例见 `data/sample_tool_call.jsonl`。

多个合理答案写成多个 `targets`；顺序执行的多步工具调用只写在单个 target 的 `output.tool_calls` 里：

```jsonl
{"messages":[{"role":"user","content":"Move toward the object."}],"tools":[{"type":"function","function":{"name":"robot_atomic_control","parameters":{"type":"object","properties":{"action":{"type":"string"},"distance_cm":{"type":"integer"}},"required":["action","distance_cm"]}}}],"targets":[{"id":"left-first","output":{"tool_calls":[{"name":"robot_atomic_control","arguments":{"action":"向左","distance_cm":6}}]}},{"id":"down-first","output":{"tool_calls":[{"name":"robot_atomic_control","arguments":{"action":"向下","distance_cm":4}}]}}]}
{"messages":[{"role":"user","content":"Move, then inspect."}],"tools":[{"type":"function","function":{"name":"move","parameters":{"type":"object"}}},{"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}}],"targets":[{"id":"move-inspect","output":{"tool_calls":[{"name":"move","arguments":{"action":"left"}},{"name":"inspect","arguments":{"object":"target"}}]}}]}
```

支持字段：

- `messages`：必填 prompt/context messages，用于 tokenizer 或 processor chat template；
- `tools`：可选工具声明列表，使用 OpenAI function-calling 格式，传给模型 chat
  template。每项为 `{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}`；
- `targets`：必填非空 list，表示多个可接受答案；每个 target 可以有 `id`，并必须有 `output`。普通答案写 `output.content` JSON object，工具调用写 `output.tool_calls` ordered canonical tool-call list；
- `messages[].content` 内的 image/video 条目：用于多模态路由；图片训练已支持，视频训练前应单独 smoke；
- 其它字段会作为 metadata。

工具调用样本的 `targets[].output.tool_calls` 使用 canonical tool-call JSON：每一项都是 `{"name":"...","arguments":{...}}`，列表顺序就是执行顺序。多个可接受答案必须拆成多个 `targets`。Qwen XML 等模型私有输出格式由对应模型 adapter 在 reward 前解析成 canonical 结构，不能写入数据集。

最后一条 message 不能是 `assistant`；`targets` 是原始 reward 目标，不能泄漏进输入，也不能转换成模型 chat template。GRASPO 只接受 `messages + 可选 tools + targets` 的 JSONL 记录，不支持纯文本 prompt 字段、JSON 文件、Excel 文件、旧 `ground_truth` 字段或 top-level `image/images/video/videos` 字段。

### 含工具调用的 assistant 消息

多轮对话中，含工具调用的 assistant 消息**必须**使用结构化 `tool_calls` 字段。
GRASPO 在启动时会校验数据，任何在 `content` 中嵌入裸工具调用文本的记录都会被拒绝：

```json
{
  "role": "assistant",
  "content": "我来旋转机械臂靠近目标。",
  "tool_calls": [
    {"name": "robot_atomic_control", "arguments": {"action_type": "顺时针旋转", "angle_deg": 38.3}}
  ]
}
```

裸 Qwen XML（`<function=...><parameter=...>`）、裸 JSON 字符串以及其他模型私有的
工具调用格式**不能**放在 `content` 中。请使用 `tool_calls` 字段，格式为 canonical JSON
`{"name":"...","arguments":{...}}`。模型的 chat template 会自动将 `tool_calls` 渲染为
正确的原生格式。

## Reward 计分方式

GRASPO 当前提供一个内置结构化输出 reward，适合目标答案为 JSON object 或 canonical tool-call sequence 的任务，并支持多个可接受 target。每条 completion 的计分流程：

1. 解析模型私有输出格式：模型 adapter 把 raw completion 中的 Qwen XML tool call 等格式转成 canonical 结构，同时保留 raw text 和 `<think>...</think>`。Qwen XML tool-call 参数会根据工具 schema 中的 `integer`、`number`、`boolean` 类型先转成对应 JSON 类型，再进入 reward。
2. 检查输出标记：根据 reward 配置，可要求 `<think>...</think>`；普通 answer 任务还可以要求 fenced JSON Markdown block。
3. 比较结构化内容：普通 answer 任务逐个比较 parsed JSON 和 `targets[].output.content`；工具调用任务逐个比较 canonical tool-call sequence 和 `targets[].output.tool_calls`，并保持同一 target 内的多步调用顺序。最终使用得分最高的 target。JSON number 字段使用 `1 / (1 + abs(predicted - target))` 连续计分；非数字字段和类型不匹配仍按严格相等比较。

   列表（list）内的 dict 元素会递归展开计分：`count_target_score` 将 dict 元素按完整结构计入分母，`count_check_score` 使用 raw check score 而非压缩后的 normalized 0-1 分。这确保复杂 dict 列表（如 tool-call `arguments`）的字段差异能正确反映在 reward 信号中，同时不影响 scalar 列表和扁平大 JSON（如工单字段抽取）的计分行为。
4. 归一化 reward：标记分、结构化内容分、完全正确 bonus 和多余文本惩罚/奖励合成 `reward`、`content_score` 和 `all_right`。

   `dict_compare_score` 返回 `CompareResult`，同时携带两个并行分数：完整 `dcs`（含数值叶节点，保留训练梯度）和 `base_dcs`（两边同步去除数值字段后比较，用于 `all_right` 判定）。这样 `distance_cm`、`angle_deg` 等数值字段仍通过 `content_score` 参与训练，但 `all_right` 只要求非数值结构匹配——动作正确但距离略有偏差的 completion 也算 "all right"，`perfect_skip` 和 `max_correct` 组决策不再被连续数值打分阻塞。

关键输出：

- `reward`：用于 GRASPO group decision、advantage 计算和 ReplayBuffer 训练；
- `content_score`：组过滤前的结构化内容匹配分（含数值连续打分）；
- `base_content_score`：去数值后的结构匹配分，用于诊断数值字段贡献；
- `all_right`：非数值字段完全匹配即为 true，不再要求数值误差为 0。

不应该获得连续数字分的 ID 或类别编码，应该在数据集中写成 JSON string。

GRASPO 使用同一 rollout group 内的 reward 分布，而不是单条 completion 的绝对分数。有有效差异的 group 会进入训练；已经 perfect 的 group 可以跳过；没有 reward 方差或没有偏好差异的 group 会被丢弃或重试。`rollouts.readable.jsonl` 会记录 messages、completion、parsed tool calls、抽取字段、parser errors、reward 细节和 invalid reason，方便检查 reward 行为。

## 配置说明

所有常规训练配置都在 YAML 内完成。`config_example.yaml` 是完整公开样例。

### `backend`

- `graspoflow`：**唯一后端。** 统一 TP+PP Flink 风格流式流水线框架。
  支持所有并行模式：单卡（`tp=1,pp=1`）、纯 TP（`tp=N,pp=1`）、
  纯 PP（`tp=1,pp=N`）、TP+PP 混合（`tp=M,pp=N`）。
  参见 `configs/graspoflow_example.yaml`。

### `model`

- `model_path`：base Hugging Face 模型路径或 id。
- `trust_remote_code`：传给 Hugging Face loader。
- `torch_dtype`：模型 dtype，通常为 `bfloat16`。
- `attn_implementation`：可选 Hugging Face attention implementation。
- `gradient_checkpointing`：在支持时开启 gradient checkpointing。
- `chat_template_kwargs`：tokenizer chat template 的额外参数。

### `data`

- `train_path`：JSONL 训练文件。
- `max_prompt_length`：prompt token 长度限制。

### `lora`

- `r`：LoRA rank。
- `alpha`：LoRA alpha。
- `dropout`：LoRA dropout。
- `adapter_path`：可选 PEFT 或 GRASPO-PEFT adapter 目录，只用于 warm-start。
- `target_preset`：target preset，例如 `language_safe`。
- `target_modules`：显式 LoRA targets；设置后优先于 `target_preset`。
- `auto_target_modules`：保留为配置字段；native 训练推荐使用 preset 或显式 canonical targets。
- `bias`：PEFT-compatible bias 设置，通常为 `none`。
- `task_type`：PEFT-compatible task type，通常为 `CAUSAL_LM`。

### `reward`

- `check_think`：要求 `<think>...</think>` 标记。
- `check_json_markdown`：要求 fenced JSON 输出。
- `check_tool_call`：旧版 tool-call 开关；当前训练会从 `targets[].output.tool_calls` 推断 tool-call 评分目标。
- `check_list_order`：结构化比较时 list 顺序是否敏感。
- `marker_reward_weight`：输出标记 reward 权重。
- `content_reward_weight`：结构化内容匹配 reward 权重。
- `anti_useless_str_reward_weight`：多余文本惩罚/奖励权重。
- `anti_useless_str_half_reward_len`：多余文本惩罚长度尺度。

### `training`

- `output_dir`：run 输出目录。
- `seed`：随机种子。
- `training_epoch_count`：完整数据集训练轮数；生产默认 `100`。
- `max_steps`：短测/debug step 上限；`-1` 表示不限制。
- `rollout_group_size`：每个 prompt attempt 采样多少条 completion。
- `optimize_prompt_batch_size`：每个 optimizer step 的 prompt 数量；
  replay buffer threshold = `optimize_prompt_batch_size × rollout_group_size`。
- `optimize_times_per_step`：同一批 replay completion 重复优化几轮。
- `rollout_max_retry_times`：初始 rollout 后的 retry 预算。
- `learning_rate`、`weight_decay`、`max_grad_norm`：optimizer 设置。
- `policy_ratio_clip_eps`：policy-ratio clipped objective epsilon。
- `max_new_tokens`：真实训练生成长度；保持 `training.max_new_tokens=2048`。
- `temperature`、`top_p`：rollout sampling 设置。
- `save_steps`：native checkpoint 间隔。`-1`（默认）禁用 step 级别 checkpoint，仅保留 epoch checkpoint。
- `save_epoch_checkpoint`：每个 epoch 结束时保存可恢复 checkpoint（默认 `true`）。生产训练推荐保持开启。
- `logging_steps`：紧凑训练日志间隔。
- `perfect_skip_reward_threshold`：跳过已解 prompt 的阈值。
- `reject_unparseable_groups`：默认 true，当最好 completion 有 parse error 或 tool-call count mismatch 时，group 会被 retry 或丢弃，不参与训练。
- `dataloader_num_workers`：数据加载 worker 数。
- `resume_from_checkpoint`：可恢复 GRASPO native checkpoint 目录。

`training.replay_buffer_optimize_threshold` 由 `optimize_prompt_batch_size * rollout_group_size` 派生，不能手动配置。`training.resume_from_checkpoint` 和 `lora.adapter_path` 互斥：前者恢复 native checkpoint 状态，后者只是 PEFT/GRASPO-PEFT LoRA warm-start。

### `graspoflow`

- `tp_size`：TP size（默认 2）。
- `pp_size`：PP size（默认 1）。
- `placement_strategy`：placement 策略，例如 `qwen3_tp` 或 `qwen36_pp8_static`（默认 `auto`）。
- `layer_ranges`：手动逐 stage 层数分配。例如 pp=4, 32 层：
  `[[0,9], [9,17], [17,25], [25,32]]`。设置后覆盖 `placement_strategy`。
- `sequence_parallel`：v1 必须保持 `false`。
- `pp_micro_batch_size`：PP micro-batch size（默认 1）。
- `forward_batch_size`：rollout forward batch size（默认 8），替代旧的 `gpu_memory_utilization`。
- `use_kv_cache_for_rollout`：KV cache 只用于 rollout generation。
- `empty_cache_after_rollout_split`、`empty_cache_before_train`：CUDA cache 控制。
- `checkpoint_format`：native recoverable checkpoint 格式标签。
- `raw_log_enabled`、`readable_log_enabled`：rollout/replay 日志开关。
- `synchronize_cuda_timing`：是否同步 CUDA timing。
- `pp_schedule`：pipeline schedule，`simple`（默认）或 `one_f_one_b`。
- `pp_max_inflight_microbatches`：1F1B inflight 上限。

### `export`

- `final_formats`：可选 final checkpoint 后自动导出的格式列表，例如 `["peft-adapter"]`。step checkpoint 不会自动导出。

### `launch`

- `gpus`：当前节点使用的 GPU id。
- `nproc_per_node`：当前节点 worker 数；为空时从 TP * PP / nodes 派生。
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

显式 `lora.target_modules` 只能使用 canonical name，例如 `language.self_attn.q_proj`，或 glob pattern，例如 `visual.blocks.*.attn.*`。不接受 `q_proj` 这样的 leaf alias。解析是 fail-closed：未知 target、不支持的 conv/norm 参数和空匹配都会在训练前报错。Native checkpoint 会保存 resolved LoRA target signature，并拒绝使用不同 target 配置 resume。

## Native 模型实现边界

Native 模型数学必须落在 native model class 内。RoPE/M-RoPE、position
IDs、KV-cache continuation、visual feature injection、TP shard-local layer
math 和 LoRA target metadata，应由 `Qwen3DenseModel`、
`Qwen35HybridTextModel` 及其 attention/layer modules 负责。

`TransformerAdapter` 及其模型家族子类（如 `Qwen3Adapter`、`Qwen35Adapter`）只负责
processor/tokenizer 调用、batch/split、sampling、pipeline send/recv 编排、
checkpoint delegation 和 logging。Runtime/placement 只负责 backend lifecycle、
config validation 和 TP/PP layout，不实现模型 family 的数学逻辑。

GRASPO 中 Qwen3.6 复用 Qwen3.5-family hybrid text/vision native class，因为
它的结构与该 family 兼容。未来如果出现 `qwen3_vl`、`qwen3_omni` 等不同
`model_type`，需要新增对应 native model class，不能在 adapter 层塞特判。

## 导出

GRASPO native checkpoint 是可恢复训练 checkpoint。便携模型产物通过 `graspo export` 生成。在 YAML 配置中设置 `export.checkpoint_path`、`export.export_format` 和 `export.export_output`，然后运行：

```bash
uv run graspo export --config config_example.yaml
```

最小导出配置示例：
```yaml
backend: graspoflow
model:
  model_path: models/Qwen3-8B
export:
  checkpoint_path: outputs/example-run/final
  export_format: peft-adapter   # 或 "merged-hf"
  export_output: outputs/export/adapter
```

`peft-adapter` 会从 GRASPO native rank shards 重建 PEFT `adapter_config.json` 和 `adapter_model.safetensors`。对于 fused/split native targets，GRASPO 会额外写出 `graspo_adapter_metadata.json`，使 GRASPO 可以无损 warm-start 这些 adapter。普通 PEFT 工具可以读取 adapter tensors，但要无歧义地映射回 native fused/split 训练模块，需要 GRASPO metadata。

`merged-hf` 会在 CPU 上流式读取 base HF safetensors，注入 LoRA delta，复制 tokenizer/config 等 sidecar 文件，并写出 HF-compatible merged model 目录。

导出的 PEFT adapter 和 merged full model 是部署/兼容产物，不包含 optimizer、RNG、replay buffer 或 trainer state，不能替代 `step_*` 或 `final` 做完整训练恢复。

## 输出和监控

每个 run 写入 `training.output_dir`：

- `logs/training.log`：rank-0 紧凑训练事件；
- `logs/rollouts.readable.jsonl`：人类可读的 messages、completion、reward 和 debug 细节；
- `logs/rollouts.raw.jsonl`：replay tensors、masks、old logprobs、advantages 和 reward metadata；
- `logs/train_batches.readable.jsonl`：每个 optimize-trigger batch 一行；
- `logs/rank_metrics.rank_*.jsonl`：每 rank 显存、耗时、LoRA 和 optimizer 诊断；
- `logs/error.log`：ERROR 级别事件汇聚（无效 group、reward 方差失败、格式损坏 group）；
- `logs/timing_events.jsonl`：各阶段 timing 诊断；
- `epoch_*`：每个 epoch 结束时的可恢复 checkpoint（当 `save_epoch_checkpoint` 为 true 时）；
- `step_*`：周期性可恢复 checkpoint（当 `save_steps > 0` 时）；
- `final`：干净退出后的最终可恢复 checkpoint；
- `config.yaml`：本次运行的配置备份，确保可完整复现。

所有日志文件位于 `logs/` 子目录下。

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
- Native launch world size mismatch：让 `launch.nproc_per_node * launch.nnodes` 等于 `tp_size * pp_size`。
- Rollout OOM：保持 `training.max_new_tokens=2048`；降低 rollout 并发或 KV cache 预留，而不是降低生产生成长度。
- 需要 PEFT 兼容：通过 `lora.adapter_path` 加载 PEFT/GRASPO-PEFT adapter，通过 `graspo export --config <yaml>` 导出便携产物。

## License

GRASPO 使用 MIT License。见 [LICENSE](LICENSE)。
