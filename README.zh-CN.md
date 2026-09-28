# GRASPO (Group Relative Advantage Structured Policy Optimization)

[English README](README.md)

## 简介

GRASPO (Group Relative Advantage Structured Policy Optimization) — GRPO 风格的 RL 与 SFT，面向结构化 LLM 输出。基于字符标注的 token 级 reward。单张 80GB GPU 即可训练 9B 模型。GraspoFlow 统一 TP+DP+PP+SP+Checkpoint 五维并行。组过滤、perfect-skip、多模态、工具调用 reward。

**结构化输出 RL 的三层防御纵深，从 token 信号到组决策：**

- **字符级标注驱动 token 级 reward（核心创新）。** 对 completion 文本逐字符比对结构模板，
  通过 offset_mapping 映射到 token——天然 tokenizer 无关。标注模块给每个字符/token 赋予
  credit 角色（S/V/T/W/E/D），对格式本身有极强的结构化识别能力。仅首个不匹配字符受罚，
  其后所有字符排除在训练之外。
- **结构化输出 reward，分离且可扩展。** reward 只对**某个 value 打分**（单个 JSON value /
  tool-call 参数），或**把整段 JSON 序列视作一个 value 赋分**，通过 `GraspoReward` 类 +
  `REWARD_REGISTRY` 注册实现。递归 dict 比较，双分数（数值精度用于梯度，结构正确用于门控），
  多目标最优匹配，数值容差。要扩展更多打分能力，**只需改 reward 模块**——底层的标注模块与
  token 级梯度训练都不用动。
- **组决策体系与防御纵深。** 六路分类（perfect_skip / trainable / invalid / retry /
  no_preference_gap）在训练边界拦截噪声。token 级 advantage 只驱动可训练 token，避免模型
  收敛到"差组里最好"。

**面向生产的基础设施：**

- **用户指定并行，框架保证可运行。** 你显式指定 `tp_size` / `dp_size` /
  `pp_size`（TP≥2 时可选用 `sequence_parallel`）。GRASPO 从不告诉你"这套设备
  跑不了这个任务"——只要基本资源约束满足，就一定存在一组性能最优的
  TP+DP+PP+SP+Checkpoint 组合能运行该任务。你不需要理解 NCCL 拓扑、PCIe vs
  NVLink 差异；每张 GPU 只跑一个进程，且按设计保持显存对称（负载均衡）。
- SFT → RL 统一管道：同一数据格式、同一模型加载、同一 checkpoint 格式。
- GraspoFlow 后端：设计目标是五位一体（TP+DP+PP+SP+Checkpoint）
  在 **SFT 与 RL**、**9B 与 27B** 上全部可用，单卡到多卡同配置切换。
  ``world_size = dp_size × tp_size × pp_size``。实现计划补齐剩余缺口
  （RL PP>1、SP 原生通信、全组合实测覆盖）以逼近该目标。
- 插件化模型适配器：ABC 契约，新模型族零侵入现有代码。
- 多模态训练：三层契约防线防止静默丢图。
- ReplayBuffer、可读 rollout 日志、内置 `analyze-profile` 分析工具。

## 快速开始

> **双语文档**：`README.md` 与 `README.zh-CN.md` 内容镜像（§17.1）——每个章节、
> 配置键、命令块与 FAQ 条目两侧都有。可用 `python3 tools/readme_mirror_check.py`
> 核验：该脚本对比两侧章节结构与内联代码 token，并打印"仅见 EN / 仅见 ZH"清单。
> 已登记的纯记号差异（不算内容缺失）写在该脚本里，附理由。

> **示例文件** 在 `samples/` 目录下：
> - `samples/configs/sft_example.yaml` — 单卡 SFT 配置（开箱即用）；
> - `samples/configs/rl_example.yaml` — 单卡 RL 配置；
> - `samples/configs/a800x8_qwen35_9b_tp1_dp8_pp1.yaml` — 8×A800 多卡配置；
> - `samples/configs/config_example_msswift.yaml` — ms-swift 后端配置（需 `pip install graspo[msswift]`）；
> - `samples/data/json_output/train.jsonl` — 小型 JSONL 数据集，用于冒烟测试。

### 三步冒烟测试（Docker，推荐）

```bash
# 第一步：克隆仓库
git clone https://github.com/godthrone/graspo.git
cd graspo

# 第二步：构建 Docker 镜像
bash docker/build.sh

# 第三步：运行冒烟测试（跑 1 步训练，验证模型加载 + 前向传播）
bash run.sh samples/configs/sft_example.yaml --smoke
```

就这么简单。冒烟测试通过后，即可开始正式训练。

> **没有 git tag？** 如果是 shallow clone（不带 tag），请显式指定版本号：
> ```bash
> VERSION=0.0.0 bash docker/build.sh
> bash run.sh samples/configs/sft_example.yaml --smoke --image graspo:0.0.0
> ```

> **需要模型？** 默认配置指向 `models/Qwen3.5-9B`。下载方式：
> ```bash
> huggingface-cli download Qwen/Qwen3.5-9B --local-dir /path/to/models/Qwen3.5-9B
> ```
> 然后将 config 中的 `model.model_path` 改为宿主机的绝对路径。

### 正式训练

冒烟测试通过后，复制并编辑一份配置用于你的任务：

```bash
# SFT 训练
cp samples/configs/sft_example.yaml my_config.yaml
# 编辑：model.model_path、data.train_path、training.output_dir
bash run.sh my_config.yaml

# RL 训练（GRASPO）
cp samples/configs/rl_example.yaml my_config.yaml
# 编辑：model.model_path、data.train_path、training.output_dir
bash run.sh my_config.yaml
```

`run.sh` 是防呆设计：
- **自动选择空闲 GPU**（通过 `nvidia-smi` 检测），无需手动数卡；
- **只传 `--gpus device=<ids>`**，绝不注入 `CUDA_VISIBLE_DEVICES`；
- **固定 `--ipc=host --shm-size=16g`**（NCCL 必需）；
- **挂载目录从 YAML 配置自动推导**；
- **镜像 tag 默认取 `git describe`**（可通过 `--image` 覆盖）；
- **`--smoke` 跑 1 步即停止**，不修改你的 config 文件。

> **`run.sh` 是训练启动的唯一受支持入口。** 手写 `docker run` 仅限诊断用途——
> 曾多次踩 `--gpus` JSON 语法（Docker 29）与路径解析的坑。如果必须在 Docker 29+
> 上手动调用 `docker run`，请用单引号包裹 GPU 列表：
> ```bash
> docker run --gpus '"device=0,1"' --ipc=host --shm-size=16g \
>   -v /data:/data \
>   graspo:<version> \
>   launch --config /data/outputs/my_config.yaml
> ```

### 本地安装（开发用）

```bash
git clone https://github.com/godthrone/graspo.git
cd graspo
uv sync --extra dev --python 3.11
cp samples/configs/sft_example.yaml my_config.yaml
uv run graspo launch --config my_config.yaml
```

### 自定义镜像名

```bash
IMAGE_NAME=graspo:test bash docker/build.sh
```

> **需要 HTTP 代理？** 传入标准代理环境变量：
> ```bash
> HTTP_PROXY=http://your-proxy:port HTTPS_PROXY=http://your-proxy:port bash docker/build.sh
> ```

### RL 训练（GRASPO）

复制并编辑根目录完整样例配置：

```bash
cp samples/configs/rl_example.yaml my_config.yaml
```

至少需要在 `my_config.yaml` 中设置：

- `model.model_path`：本地 Hugging Face 模型目录或模型 id；
- `data.train_path`：JSONL 训练数据；
- `training.output_dir`：run 输出目录；
- GPU 选择**不在配置中**——用 `run.sh`（自动选空闲 GPU）或
  `bash run.sh config.yaml --gpus 4,5`（见 Docker 章节）；
- `native.tp_size`、`native.dp_size` 和
  `native.pp_size`：world_size = tp_size × dp_size × pp_size。

### SFT 训练

SFT 模式复用同一套 GraspoFlow 基础设施（TP/PP、LoRA、checkpoint），
使用相同的 JSONL 数据格式。复制专用 SFT 配置模板：

```bash
cp samples/configs/sft_example.yaml my_sft.yaml
```

与 RL 的主要区别：

- `train_method: sft` — 切换到监督微调，而非 RL；
- `micro_batch_size` 和 `gradient_accumulation_micro_batches` 控制 batch size；
  effective batch/GPU = micro_batch_size × gradient_accumulation_micro_batches；
- `max_prompt_length` 是完整序列长度（prompt + response）；
- `learning_rate` 通常比 RL 高（如 `5e-5` vs `5e-6`）；
- `reward` 配置段在 SFT 中被忽略。

启动方式相同：

```bash
uv run graspo launch --config my_sft.yaml
```

SFT 完成后可以无缝切换到 RL：将 `train_method: graspo`，
将 `lora.adapter_path` 指向 SFT checkpoint 的 adapter 目录即可。

### 工具命令

工具命令只接受输入定位参数，输出要么打印、要么写入 config 决定的位置——绝不覆盖配置值。

**校验 reward 评分**（逐样本打印，不落盘）：

```bash
uv run graspo validate-reward --data samples/data/json_output/train.jsonl --limit 2
```

**评测 checkpoint**（生成 rollout groups 并评分，输出
`summary.json` + `completions.jsonl` 到 `<config output_dir>/evaluate/`）：

```bash
uv run graspo evaluate-checkpoint --config my_config.yaml \
    --data samples/data/json_output/train.jsonl --checkpoint outputs/my_run/step_100
```

**汇总性能分析输出**：

```bash
uv run graspo analyze-profile outputs/my_run
```

除运行汇总外，`analyze-profile`（v0.22+）对每个 run 目录产出
**六份分析文件**（CLI 只打印文件路径，不打印数据表格）：
1. `logs/analysis_profile.json` — 性能/timing/最新步窗口汇总
2. `logs/analysis_steps.jsonl` — step 粒度训练进度：样本区间
   （samples_start/end，同 epoch 内差分）、终结决策计数
   （perfect/invalid/no_gap/mc/nc）、mc_ratio、reward/content/loss 均值、
   retry_rate、告警分类计数、墙钟耗时
3. `logs/analysis_epochs.json` — 同一度量元组的 epoch 粒度（loss_mean 与
   alarms 由分析端从 train_step 事件聚合补齐）
4. `logs/analysis_errors.jsonl` — completion 级互斥错误原因（L1 格式层：
   no_tool_call / malformed_xml / missing_param / multi_call / other_parse；
   L2 匹配层：tool_mismatch / param_name_mismatch / param_value_mismatch；
   ok / other），step 与 epoch 双粒度，带去重样本引用可回溯
5. `logs/analysis_attribution.json` — 组级归因：not_correct 组原因分类
   （tool_mismatch / content_all_wrong / format_shortfall）、工具名/参数
   匹配趋势（按 step）、决策 × 匹配交叉表
6. `logs/analysis_perf.jsonl` — 性能统计（只聚合 train_step timing 块，
   零训练侵入）：rollout / 队列等待% / prefill / decode / 吞吐 tok/s /
   optimize / retry 占比，step 与 epoch 双粒度

全部为**与训练数据内容无关**的结构级分析（不假设字段名/数值语义——
L3 语义/数值层诊断由 AI/人工基于 rollouts 详表离线统计，用户裁定），
供脚本消费。
## CLI 参考

所有命令均为配置驱动：只接受输入定位参数，输出要么打印、
要么写入 config 决定的位置。

- `graspo launch --config <yaml> [--smoke]` — 训练入口。生产环境唯一受支持
  的启动方式是 `run.sh`（自动选 GPU、`--ipc=host`、挂载推导）；
  `--smoke` 跑 1 步验证环境。
- `graspo export --config <yaml>` — 导出 LoRA checkpoint
  （`export.checkpoint_path` → `export.export_output`，格式 `export_format`）。
- `graspo validate-reward --data <jsonl> [--limit N] [--completions <jsonl>]`
  — 校验 reward 评分链路；逐样本打印分数，不落盘。
- `graspo evaluate-checkpoint --config <yaml> --data <jsonl>
  [--checkpoint <dir>] [--limit N]` — 生成 rollout groups 并评分；
  写入 `<output_dir>/evaluate/` 下的 `summary.json` + `completions.jsonl`。
- `graspo analyze-profile <run_dir>... [--skip-warmup-steps N]` — 向
  `<run_dir>/logs/` 写入六份分析文件（`analysis_profile.json` /
  `analysis_steps.jsonl` / `analysis_epochs.json` / `analysis_errors.jsonl` /
  `analysis_attribution.json` / `analysis_perf.jsonl`），只打印文件路径。

完整参数列表见 `graspo --help`。

## 数据格式

训练数据只支持 JSONL，且 **SFT 与 RL 共用同一套格式**。每行是一条由
OpenAI 兼容的 chat `messages` 表示的 prompt/context、可选工具声明
（`tools`，OpenAI function-calling 格式），以及一个或多个可接受的 reward
目标（`targets`）——`targets` 是 GRASPO 特有的、承载期望答案（用于结构化输出判分）的字段：

```jsonl
{"messages":[{"role":"system","content":"You extract structured support ticket fields as fenced JSON."},{"role":"user","content":"Ticket: user 99999000000 cannot use apn apn01."},{"role":"assistant","content":"I will identify the phone number and APN from the ticket."},{"role":"user","content":"Extract JSON with the APN and fault number."}],"targets":[{"id":"expected","output":{"content":{"APN":"apn01","fault_number":"99999000000"}}}]}
```

多模态数据也使用同一个 `messages` 字段，role 和 content 顺序会保真进入 tokenizer/processor：

```jsonl
{"messages":[{"role":"system","content":"Extract fields from ticket screenshots."},{"role":"user","content":"Use exact snake_case values."},{"role":"assistant","content":"Understood."},{"role":"user","content":[{"type":"image","image":"images/panel_0001.png"},{"type":"text","text":"Extract the ticket fields as strict JSON."}]}],"targets":[{"id":"expected","output":{"content":{"ticket_id":"T-0001","status":"critical"}}}]}
```

工具调用数据可以在可选 `tools` 字段中提供模型原生工具声明。GRASPO 会在运行时把 `messages + tools` 交给 tokenizer 或 processor 的 chat template；用户不需要、也不应该在数据集中提前渲染模型模板字符串：

```jsonl
{"messages":[{"role":"system","content":"Use tools when needed. Output only the tool call."},{"role":"user","content":"Query device DEV-01 status at 2026-06-08 10:30."}],"tools":[{"type":"function","function":{"name":"query_device_status","description":"Query network device panel status.","parameters":{"type":"object","properties":{"device_id":{"type":"string"},"panel_time":{"type":"string"}},"required":["device_id","panel_time"]}}}],"targets":[{"id":"expected","output":{"tool_calls":[{"name":"query_device_status","arguments":{"device_id":"DEV-01","panel_time":"2026-06-08T10:30:00+08:00"}}]}}]}
```

可运行的工具调用数据样例见 `samples/data/tool_call_mm/train.jsonl`。

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

最后一条 message 不能是 `assistant`；`targets` 是原始 reward 目标，不能泄漏进输入，也不能转换成模型 chat template。GRASPO 只接受 `messages + 可选 tools + targets` 的 JSONL 记录，不支持纯文本 `prompt` 字段、JSON 文件、Excel 文件、旧 `ground_truth` 字段或 top-level `image/images/video/videos` 字段。

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

GRASPO 通过可扩展的 **`GraspoReward` 类**提供 reward 机制，并注册于
`REWARD_REGISTRY`（当前内置 `graspo` 一种；未来新增 reward 类注册即可，无需改动调用端）。
内置 reward 适合目标答案为 JSON object 或 canonical tool-call sequence 的任务，并支持多个可接受 target。每条 completion 的计分流程：

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

GRASPO 使用同一 rollout group 内的 reward 分布，而不是单条 completion 的绝对分数。有有效差异的 group 会进入训练；已经 perfect 的 group 可以跳过；没有 reward 方差或没有偏好差异的 group 会被丢弃或重试。`logs/rollouts.readable.jsonl` 会记录 messages、completion、parsed tool calls、抽取字段、parser errors、reward 细节和 invalid reason，方便检查 reward 行为。

## Token 级标注（v0.20.0）

rollout 完成后，每条 completion 会被**逐字符标注**结构角色（`CharTag`），与打分解耦。标注模块（`src/graspo/ripple/annotation/`）是 token 级 advantage 的唯一输入：

| 枚举 | 含义 | 下游 |
|------|------|------|
| `S` | 结构字符，与期望模板字符相等 | +1.0 |
| `V` | 值字符（参数值 / JSON value） | 相似度 0~1 |
| `T` | think 内容 | 0（不训练） |
| `W` | 前导/尾随/夹缝文本 | 0（不训练） |
| `E` | **首个**与期望模板不匹配的字符 | -1.0 |
| `D` | E 之后所有字符 | 0（不训练） |

设计原则（源于 v0.16-v0.19 四次连续训练崩溃）：

- **字符级比对，tokenizer 无关**：期望 mark 序列（如 `<tool_call> <function=...> <parameter=...> ...`）与模型输出逐字符比对，首个不匹配字符标 E。token 标注经 `offset_mapping` 派生——任何 tokenizer 变体都不会误标正确前缀 token（如 `</parametr>` 与 `</parameter>` 共享 `</`、`param` 前缀 token）
- **严格对齐截断**：E 之后全部 D（不训练）——错误前缀下的 token 无训练价值
- **截断/结构不完整不标 E**：已有字符全部正确，硬标 E 只误标正确 token；不完整性信号交给 reward 层（`content_score = 0` → 组级 RETRY/INVALID）
- **值错误不触发 E**：内容相似度由下游打分，仅结构/类型错误触发 E
- **tool call 参数无序集合匹配**：参数顺序颠倒不是错误（JSON 对象语义）；GT 参数缺失或多余参数是错误（期望缺失 `<parameter=NAME>` 与实际输出比对定位 E）
- **与 reward 层语义一致**：只有语义错误标 E——缺字段/多余字段/拼错/类型错；语义正确不标（如参数顺序）

65 条测试数据集（`tests/data/annotation_testset_v3.jsonl`，tool call 44 + JSON 21）覆盖完美输出/值错误/前导文本/拼错/双开标记/多余字段/缺字段/顺序颠倒/截断/乱码/think/嵌套结构等场景，每条含期望标注，经独立 agent 核验。`tests/data/generate_annotation_viewer.py` 生成 HTML 逐字符着色视图供人工检查。

## 配置说明

所有常规训练配置都在 YAML 内完成。

- `samples/configs/sft_example.yaml` — 最保守单卡 SFT 模板；
- `samples/configs/rl_example.yaml` — 最保守单卡 RL 模板；
- `samples/configs/a800x8_qwen35_9b_tp1_dp8_pp1.yaml` — 8×A800 多卡验证配置。

### `train_method`

- `graspo`：使用 GRASPO 算法进行 RL 训练（默认）。
- `sft`：使用交叉熵损失进行监督微调。复用相同配置字段，无需新增字段。
  `reward` 配置段在 SFT 中被忽略。

### `backend`

GRASPO 支持两种后端，通过 `config.backend` 切换：

| 后端       | 配置值          | 说明 | 安装方式 |
|-----------|----------------|------|---------|
| **Native** | `native`（默认） | 自研 TP+DP+PP+SP+Checkpoint 五位一体并行。适合精细并行控制和单卡到多卡弹性伸缩。 | 内置 |
| **MsSwift** | `msswift`      | 委托给 ms-swift 生态：vLLM rollout、DeepSpeed/Megatron、600+ 模型、150+ 数据集、量化、部署。 | `pip install graspo[msswift]` |

- `native`：统一 TP+DP+PP+SP+Checkpoint 五位一体
  （TP 分片参数、DP 分片数据、PP 分片层、SP 分片序列、Checkpoint 省显存）。
  `world_size = dp_size × tp_size × pp_size`。支持从单卡（`tp=1,dp=1,pp=1`）
  到全并行（`tp=T,dp=D,pp=P`），同一个配置切换。`native` 配置段控制并行参数。
  参见 `samples/configs/a800x8_qwen35_9b_tp1_dp8_pp1.yaml`。
- `msswift`：利用 ms-swift 的工业级训练基础设施（vLLM rollout、DeepSpeed ZeRO、
  FSDP），同时注入 Graspo 的 ripple 算法（字符级标注、token 级 advantage、
  GRASPORippleLoss）。需要 `pip install graspo[msswift]`。
  参见 `samples/configs/config_example_msswift.yaml`。

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
- `bias`：PEFT-compatible bias 设置，通常为 `none`。
- `task_type`：PEFT-compatible task type，通常为 `CAUSAL_LM`。

GRASPO 目前仅支持 LoRA 训练，不支持全参数训练。

### `reward`

- `check_think`：要求 `<think>...</think>` 标记。
- `check_json_markdown`：要求 fenced JSON 输出。
- `check_list_order`：结构化比较时 list 顺序是否敏感。
- `marker_reward_weight`：输出标记 reward 权重。
- `content_reward_weight`：结构化内容匹配 reward 权重。
- `anti_useless_str_reward_weight`：多余文本惩罚/奖励权重。
- `anti_useless_str_half_reward_len`：多余文本惩罚长度尺度。
- `numeric_tolerance`：数值相对误差容差（满分阈值，默认 0.2）。

### `training`

- `output_dir`：run 输出目录；为空时由自动生成的时间戳 `run_name`
  推导为 `outputs/<run_name>`。
- `run_name`：可选 run 名（默认自动生成）。
- `seed`：随机种子。
- `max_epochs`：完整数据集训练轮数；生产默认 `100`。训练长度仅由 `max_epochs`
  控制——旧的 `max_steps` 已在 v0.23.0 移除；短测用 `--smoke`（跑 1 步）。
- `rollout_group_size`：每个 prompt attempt 采样多少条 completion。
- `rollout_queue_batch_size`：每个 step 从 rollout queue 取多少 prompt（默认 8）；
  与 `rollout_group_size` 共同决定 replay buffer threshold。
- `gradient_accumulation_micro_batches`：每个 optimizer step 的 prompt 数量；
  replay buffer threshold = `rollout_queue_batch_size × rollout_group_size`。
- `rollout_max_retries`：初始 rollout 后的 retry 预算。
- `learning_rate`、`weight_decay`、`max_grad_norm`：optimizer 设置。
- `policy_ratio_clip_eps`：policy-ratio clipped objective epsilon。
- `max_new_tokens`：真实训练生成长度；保持 `training.max_new_tokens=2048`。
- `lr_scheduler`：`type`（`constant`/`cosine`/`linear`）、`warmup_steps`、
  `min_lr_ratio`、`decay_steps`。`type` 非 `constant` 时 `decay_steps` 必须为正数。
- `temperature`、`top_p`：rollout sampling 设置。
- `save_steps`：native checkpoint 间隔。`-1`（默认）禁用 step 级别 checkpoint，仅保留 epoch checkpoint。
- `save_checkpoint_every_epoch`：每个 epoch 结束时保存可恢复 checkpoint（默认 `true`）。生产训练推荐保持开启。
- `perfect_skip_reward_threshold`：跳过已解 prompt 的阈值。
- `reject_unparseable_groups`：默认 true，当最好 completion 有 parse error 或 tool-call count mismatch 时，group 会被 retry 或丢弃，不参与训练。
- `resume_from_checkpoint`：可恢复 GRASPO native checkpoint 目录。

`training.replay_buffer_optimize_threshold` 由 `rollout_queue_batch_size * rollout_group_size` 派生（默认 8 × 8 = 64 条 completion），不能手动配置。`training.resume_from_checkpoint` 和 `lora.adapter_path` 互斥：前者恢复 native checkpoint 状态，后者只是 PEFT/GRASPO-PEFT LoRA warm-start。

### `native`

- `adapter`：模型适配器路径（默认
  `graspo.flow.adapters.models.qwen35_36.adapter:Qwen35Adapter`）。
- `tp_size`：TP size（默认 2）。
- `dp_size`：DP size（默认 1）。`world_size = dp_size × tp_size × pp_size`。
- `pp_size`：PP size（默认 1）。
- `dp_replicate_lora`：DP rank 间复制 LoRA（默认 true）；梯度跨 `dp_group` 做 AVG all-reduce。
- `lr_scaling`：DP 学习率缩放策略（`"linear"`=lr×dp_size，默认；`"none"`=不缩放）。
- `placement_strategy`：placement 策略，例如 `qwen3_tp` 或 `qwen36_pp8_static`（默认 `auto`）。
- `layer_ranges`：手动逐 stage 层数分配。例如 pp=4, 32 层：
  `[[0,9], [9,17], [17,25], [25,32]]`。设置后覆盖 `placement_strategy`。
- `sequence_parallel`：可选；需 `tp_size >= 2`。在 TP 组内沿序列维度分片激活值（reduce_scatter + all_gather）以省显存。
- `pp_micro_batch_size`：PP micro-batch size（默认 1）。
- `micro_batch_size`：rollout forward batch size（默认 8），替代旧的 `gpu_memory_utilization`。
- `pp_scheduler`：PP 调度策略（默认 `one_f_one_b`/`1f1b`）。`one_f_one_b` 交错 forward/backward，气泡更小，需要双向进程组（`pp_group_fwd`/`pp_group_bwd`），让 forward-hidden 与 backward-grad 不共用同一 peer-pair 单 FIFO。1F1B 是 PP 的唯一调度策略（旧的气泡最大的 `gpipe` 已删除）。
- `pp_max_inflight_microbatches`：PP 背压的有界在途 microbatch 上限。默认 `0`=auto（框架推导），`>0`=显式上限。
- `use_kv_cache_for_rollout`：KV cache 只用于 rollout generation。
- `empty_cache_after_rollout_split`、`empty_cache_before_train`：CUDA cache 控制。
- `raw_log_enabled`、`readable_log_enabled`：rollout/replay 日志开关。
- `synchronize_cuda_timing`：是否同步 CUDA timing。

### `export`

- `final_formats`：可选 final checkpoint 后自动导出的格式列表，例如 `["peft-adapter"]`。step checkpoint 不会自动导出。

### `launch`

- GPU 选择**不在此配置**——由 `run.sh`（`bash run.sh config.yaml --gpus 4,5`）
  或 Docker `--gpus` 直接处理。
- `nproc_per_node`：当前节点 worker 数；为空时从 `tp_size × dp_size × pp_size / nnodes` 派生。
- `nnodes`、`node_rank`、`master_addr`、`master_port`：distributed launch 设置。
- `python`：可选 Python executable override。

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
math 和 LoRA target metadata，应由受支持的 hybrid text/vision 家族的
native class（如 `Qwen35HybridTextModel`）及其 attention/layer modules 负责。

`TransformerAdapter` 及其模型家族子类（如 `Qwen35Adapter`）只负责
processor/tokenizer 调用、batch/split、sampling、pipeline send/recv 编排、
checkpoint delegation 和 logging。Runtime/placement 只负责 backend lifecycle、
config validation 和 TP/PP layout，不实现模型 family 的数学逻辑。

GRASPO 受支持的模型家族是 **Qwen3.5 / Qwen3.6 hybrid text/vision** 这一类；
Qwen3.6 复用 Qwen3.5-family hybrid text/vision native class，因为它的结构与
该 family 兼容。未来如果出现不同 `model_type`，需要新增对应 native model
class，不能在 adapter 层塞特判。

## 导出

GRASPO native checkpoint 是可恢复训练 checkpoint。便携模型产物通过 `graspo export` 生成。在 YAML 配置中设置 `export.checkpoint_path`、`export.export_format` 和 `export.export_output`，然后运行：

```bash
uv run graspo export --config samples/configs/sft_example.yaml
```

最小导出配置示例：
```yaml
backend: native
model:
  model_path: models/Qwen3.5-9B
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

- `logs/training.log`：人类可读的 rank-0 文本日志（仅 INFO/DEBUG 文本行——结构化 JSON 事件在 `events.jsonl`）；
- `logs/events.jsonl`：结构化事件流（`train_step`、`epoch_summary`、`checkpoint_saved`、`group_decision` 等），带 `timestamp` + `run_id` 关联键；
- `logs/train.log`：备用训练日志路径（`analyze-profile` 的旧版 fallback）；
- `logs/rollouts.readable.jsonl`：人类可读的 messages、completion、reward 和 debug 细节（全部 attempt，含 retry）；
- `logs/rollouts.raw.jsonl`：replay tensors、masks、old logprobs、advantages 和 reward metadata（仅终结 attempt）；
- `logs/train_batches.readable.jsonl`：每个 optimize-trigger batch 一行；
- `logs/rank_metrics.rank_*.jsonl`：每 rank 显存、耗时、LoRA 和 optimizer 诊断；
- `logs/error.log`：ERROR 级别文本事件汇聚（无效 group、reward 方差失败、格式损坏 group）；
- `logs/timing_events.jsonl`：各阶段 timing 诊断；
- `epoch_*`：每个 epoch 结束时的可恢复 checkpoint（当 `save_checkpoint_every_epoch` 为 true 时）；
- `step_*`：周期性可恢复 checkpoint（当 `save_steps > 0` 时）；
- `final`：干净退出后的最终可恢复 checkpoint；
- `config.yaml`：本次运行的配置备份，确保可完整复现。

所有日志文件位于 `logs/` 子目录下。

SFT 运行只产生其中一部分：`training.log`、`rank_metrics.*.jsonl`、`error.log`、
checkpoint、`final` 和 `config.yaml`。rollout 和 replay 日志仅 RL 模式产生，
SFT 训练不写入。

### 关键监控指标

**组决策分类**（每 step）：
- `perfect_skip`：已稳定答对，无需训练
- `trainable`（max_correct / not_correct）：有训练价值
- `invalid`：格式错误被丢弃
- `invalid_no_preference_gap`：所有 completion 相同，无偏好信号
- `retry`：rollout 失败后重试

**Reward 趋势**：
- `reward_mean` / `reward_median` 上升 → 训练有效
- `reward_max_median_gap_mean` > 0 → 组内仍有偏好信号
- `nonzero_range_rate` 接近 0 → 模型可能过拟合

**Content Score**：
- `content_all_zero_rate` ≥ 0.8 → 模型无法产生正确内容
- `content_all_one_rate` 上升 → 模型趋于完美

**Training Health**（自动检测）：
- `nonfinite_loss_or_grad`：loss/grad 出现 inf/nan
- `zero_lora_delta`：LoRA 权重无变化
- `batch_reward_all_zero`：本 batch 所有 reward = 0
- `batch_high_retry_rate`：retry 率过高
- `reward_all_zero_window`：近期 10+ 步 reward 全部为 0
- `content_score_all_zero_window`：近期 ≥80% 为内容零分

### SFT → RL 两阶段训练

对于复杂结构化输出任务，推荐的训练流程：

1. **SFT 阶段**（10 epochs）：用 `train_method: sft` 教模型输出格式
2. **导出**：`graspo export` 将 LoRA 合并到 base 模型
3. **RL 阶段**（100 epochs）：用 `train_method: graspo` 优化输出质量

SFT 和 RL 共享同一套 JSONL 数据格式。SFT target text 由 `build_sft_target_text`
直接生成原始 XML，与模型推理输出字符级一致，不经过 `tokenizer.apply_chat_template`。
原则：SFT 应该教模型**说什么**（what to say），不改变**怎么说**（how to say）——
格式是预训练已经学会的能力，不应该被 LoRA 覆盖。

## 开发检查

### 本地

```bash
uv run --extra dev ruff check src tests scripts
uv run --extra dev ruff format --check src tests scripts
uv run --extra dev pytest -q
uv run --extra dev python -m graspo --help
```

### Docker

```bash
# 检查 CLI 是否正常
docker run --rm graspo:<version>
# → 显示 graspo --help 输出

# 快速冒烟测试（需要挂载模型）：
#   graspo launch --smoke 跑 1 步训练，验证模型加载、多模态链路、
#   训练前向后停止。
bash run.sh samples/configs/sft_example.yaml --smoke
```

## 常见问题

- `model.model_path must be set`：编辑 `samples/configs/sft_example.yaml`，指向真实 base model。
- `data.train_path does not exist`：将 `data.train_path` 指向 JSONL 文件。
- **Docker 容器内找不到模型**：确认 `model.model_path` 在 YAML 中写的是宿主机上的绝对路径，`run.sh` 会自动挂载其父目录。如果路径不在常见位置，用 `bash run.sh --help` 检查挂载逻辑。
- **Docker 提示 `torchrun` 找不到**：镜像已将 GRASPO 安装为 CLI 入口，直接运行 `graspo launch --config ...` 即可，PATH 已包含 torch 和 torchrun。
- Native launch world size mismatch：让 `launch.nproc_per_node * launch.nnodes` 等于 `tp_size × dp_size × pp_size`。
- Rollout OOM：保持 `training.max_new_tokens=2048`；降低 rollout 并发或 KV cache 预留，而不是降低生产生成长度。
- 需要 PEFT 兼容：通过 `lora.adapter_path` 加载 PEFT/GRASPO-PEFT adapter，通过 `graspo export --config <yaml>` 导出便携产物。
- **SFT 转 RL**：SFT 训练完成后，将 `train_method: graspo`，
  将 `lora.adapter_path` 指向 SFT checkpoint 的 adapter，
  降低 `learning_rate`（如 `1e-6`）。SFT LoRA adapter 可直接用于 GRASPO RL 训练。
- **SFT OOM**：减小 `micro_batch_size`（micro-batch）或 `max_prompt_length`；
  增大 `gradient_accumulation_micro_batches` 以保持有效 batch size 不变。
- **A800 PCIe 拓扑 DP 训练 hang（NCCL all-reduce 超时）**：在 A800 PCIe 拓扑的机器上（GPU 以 NVLink pair 成对、pair 之间通过 PCIe bridge(PXB) 连接），NCCL 的 P2P/CUMEM 路径在小张量 all-reduce 时可能 hang。TP 训练不受影响（大张量走 Simple 协议，不经过 P2P 路径）。**修复**：`NCCL_P2P_DISABLE=1`。框架现在会**自动检测并设置，无需手动 `-e`**：
  - `run.sh` 与容器内 Python 启动检查（`GraspoFlowState.initialize`）都会读取 `nvidia-smi topo -m`：选中 GPU 间存在 PXB/PHB/SYS 则设 `NCCL_P2P_DISABLE=1`；全 NVLink mesh（全 `NV#`）保留 P2P（更快，且避免默认组缓冲集中到 device0 造成的非对称显存/GPU0 先 OOM）。`nvidia-smi` 不可用或拓扑解析失败时安全回退为禁用 P2P（避免 hang）。
  - 在 PCIe 拓扑上框架会**强制** `NCCL_P2P_DISABLE=1`，即使你传 `=0`（该值在此拓扑必然 hang）；`=1` 始终安全。
  - 在全 NVLink mesh 上手动设 `=1` 仅是显存/性能取舍，不影响正确性。
  - 禁用 P2P 的性能影响极小（<0.1% 训练总耗时）。

## License

GRASPO 使用 MIT License。见 [LICENSE](LICENSE)。

依赖许可：所有运行时依赖均为宽松许可（MIT/Apache-2.0/BSD）。PyTorch CUDA wheel
捆绑的 NVIDIA 运行时库采用 NVIDIA 专有 EULA（可再分发、无传染性），详见
NVIDIA Software License。
