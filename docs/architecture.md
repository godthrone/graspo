# GRASPO 架构设计

## 核心理念

GRASPO 是一个 GRPO 风格的 LoRA 强化学习训练器，面向结构化输出任务（JSON 生成、工具调用、信息抽取等）。设计遵循边界思维和防呆原则：模块边界清晰、接口稳定、防线前置。

详细的分话题文档：
- **[GraspoFlow 后端](graspoflow.md)** — Flink 风格 TP+PP 分布式调度框架
- **[训练方法论](training_methodology.md)** — GRPO 改进版算法、奖励系统、效果评估

## 三层架构

```mermaid
flowchart TB
    subgraph L0 ["入口层 cli/"]
        CLI["cli/app.py<br/>launch / export / validate-reward / evaluate-checkpoint / analyze-profile"]
    end
    subgraph L1 ["算法层 ripple/ · 纯计算"]
        R["reward/ · parity · loss · buffer<br/>data · parsing/ · monitoring/ · multimodal/<br/>annotation/"]
    end
    subgraph L2 ["通用件 core/"]
        C["schema.py · chat_template.py · lora.py"]
    end
    subgraph L3 ["设施层 flow/"]
        T["trainer/"]
        A["adapters/<br/>models/common + qwen3 + qwen35_36"]
        S["scheduling/ · parallel/ · lora/<br/>runtime · memory · selector"]
    end
    CLI --> C
    CLI --> T
    T --> R
    T --> A
    A --> S
    T --> S
```

依赖方向：`cli → trainer → adapters → scheduling/parallel`；`flow → ripple/core` 单向（设施消费算法）。

## 三层边界（单向依赖）

| 层 | 职责 | 允许依赖 | 禁止 |
|----|------|---------|------|
| **ripple/**（算法层） | 训练方法论：奖励评分、advantage、loss、解析、监控、多模态行构建 | 标准库、pydantic、torch 纯张量计算（CPU 可单测） | GPU 设备调用、分布式、网络、文件 IO |
| **core/**（通用件） | 跨层契约：配置模型、chat template | 标准库、pydantic、yaml（仅配置读取） | 其他设施 |
| **flow/**（设施层） | 执行载体：TP/PP 分布式、模型加载、checkpoint、训练循环 | 一切设施 + 单向依赖 ripple/core | 无（反向被 FORBIDDEN 守卫拦截） |

**边界判断**：改这个文件会让训练结果变吗？会 → ripple；不会但和配置有关 → core；其余 → flow。

## 训练模式

GRASPO 支持两种训练模式，通过配置 `train_method` 选择：

| 模式 | train_method | 训练器 | 用途 |
|------|-------------|--------|------|
| **RL** | `graspo` | `GraspoFlowTrainer` | GRPO 风格强化学习，group-based 相对优势 |
| **SFT** | `sft` | `SftTrainer` | 监督微调，直接教模型输出 target text |

### SFT → RL 两阶段训练

典型 pipeline：先用 SFT 教模型输出格式，再用 RL 优化输出质量。

```mermaid
flowchart LR
    SFT["SFT 配置<br/>train_method: sft"] --> LOAD["SftTrainer 加载 JSONL"]
    LOAD --> TXT["纯文本: sft_tokenize_text"]
    LOAD --> MM["多模态: sft_tokenize_multimodal<br/>→ MultimodalDeferred 延迟编码"]
    TXT --> EPOCH["训练 10 epochs<br/>保存 LoRA checkpoint"]
    MM --> EPOCH
    EPOCH --> EXPORT["graspo export<br/>合并 LoRA 到 HF 模型"]
    EXPORT --> RL["RL 配置<br/>train_method: graspo"]
    RL --> RLN["GraspoFlowTrainer<br/>加载合并模型 + 新 LoRA"]
    RLN --> RLL["100 epochs RL 长训"]
```

### SFT 数据格式对齐（关键不变式）

SFT 和 RL 共享同一套 JSONL 数据格式，但 target text 的生成方式不同：

- **RL**：推理时模型生成 completion → parser 解析 → 与 target 比较计算 reward
- **SFT**：`build_sft_target_text` 直接生成 XML，**不经过** `tokenizer.apply_chat_template`

**为什么不能走 chat template？** Qwen 的 chat template 在 assistant 消息前插入
`\n response\n` 前缀，但 RL 推理时 `enable_thinking=false` 的 assistant prefix 是
`<|im_start|>assistant\n`。两者不一致会导致 SFT 教的格式和模型实际输出格式错位，
模型会输出 `\n\nfunction\n response\n\n response\n` 等垃圾。

**正确做法**：`ripple/parsing/xml.py` 的 `build_sft_target_text`/`tool_calls_to_xml` 直接生成纯
`<tool_call>...</tool_call>` XML，与模型 RL 推理时的实际输出字符级一致。

### XML 格式对齐：与 Base Model 原生输出一致

仅仅"不走 chat template"还不够。Qwen3.5 在预训练中学会的 XML 工具调用格式是参数值
位于独立行：

```xml
<parameter=action_type>
逆时针旋转
</parameter>
```

而早期实现使用内联紧凑格式 `<parameter=action_type>逆时针旋转</parameter>`。
这种格式差异导致 **灾难性干扰（catastrophic interference）**：LoRA 仅 30M 参数
（模型 0.3%），被迫同时改写 XML 格式风格和内容知识。在 405 条小样本上，模型在
新旧格式间摇摆，输出崩溃的 XML（如 `<<tool_call>`、`<parameter=distance</parameter>`）。

**验证方法**：对 base model 推理一条纯文本 prompt，观察其原生 XML 输出格式，
然后确保 `tool_calls_to_xml`（ripple/parsing/xml.py）产出的格式与之完全一致。

**修复效果**：格式对齐后，step 1 loss 从 0.64 降至 0.28，模型不再需要为格式风格
消耗 LoRA 容量，所有参数专注于学习内容（动作名、数值）。

### 灾难性干扰与 LoRA 容量

灾难性干扰不是过拟合——过拟合是 loss 很低但泛化差。这里是 loss 也降不下去（卡在
0.15），模型处于"半吊子"状态，新旧知识混在一起。根本原因是 LoRA 容量不足以同时：
1. 改写预训练中嵌入的格式习惯
2. 学习新的领域知识

**原则**：SFT 应该只教模型**内容**（what to say），不改变**格式**（how to say）。
格式是预训练已经学会的能力，不应该被 LoRA 覆盖。如果确实需要改变格式，要么增大
LoRA 容量（r >= 64），要么考虑全量微调。

### 多模态延迟编码

SFT 多模态路径与 RL 完全对齐编码流程：

```mermaid
flowchart LR
    subgraph SFT_PATH ["SFT 路径"]
        S1["sft_tokenize_multimodal"] --> S2["MultimodalDeferred"]
        S2 --> S3["_collate_sft_multimodal_batch"]
        S3 --> ENC["_encode_multimodal_rows<br/>单次 processor 调用"]
    end
    subgraph RL_PATH ["RL 路径"]
        R1["ripple.multimodal.rows<br/>multimodal_row_from_sample"]
        R1 --> ENC
    end
```

两者都通过 `_encode_multimodal_rows` 一次性完成 tokenize + 视觉编码，确保
`input_ids` 和 `pixel_values` 来自同一次 processor 调用。SFT 与 RL 均通过 `ripple/multimodal/rows.py` 的 `resolve_messages_media_paths`
（`multimodal_row_from_sample(data_dir=...)` 内部调用）将相对图像路径解析为绝对路径。

## 数据流

```mermaid
flowchart TD
    CFG["配置文件 YAML"] --> SCHEMA["GraspoConfig<br/>pydantic 校验"] --> CLI["CLI"]
    CLI --> TRAINER["SftTrainer / GraspoFlowTrainer"]
    DATA["JSONL 数据"] --> LOAD["load_jsonl"] --> SAMPLE["Sample"]
    SAMPLE --> RLO["rollout / tokenize"] --> REW["reward 评分"]
    RLO --> BUF["ReplayBuffer / collate batch"] --> OPT["优化步骤"]
    OPT --> CKPT["checkpoint 保存"]
```

## 标注模块（annotation/，v0.20.0）

rollout 完成后对每条 completion 做**字符级结构标注**，作为 token 级 advantage 的唯一输入。

### 设计哲学

- **字符级标注** `List[CharTag]`（S/V/T/W/E/D），长度恒等于 `len(completion)`，只标角色**不打分**（打分由 advantage 层消费枚举后完成）
- **逐字符比对**：期望 mark 序列与模型输出逐字符比对，首个不匹配字符标 E——不按 XML 元素整体标（Qwen tokenizer 中 `</parametr>` 与 `</parameter>` 共享 `</`/`param` 前缀 token，元素级标注会误标正确 token）。标注定义在字符级，token 由 offset_mapping 派生，天然 tokenizer 无关
- **严格对齐截断**：E 之后全部 D（不训练）；值错误不触发 E（下游相似度打分），仅结构/类型错误触发 E
- **与 reward 层语义一致**：只有语义错误标 E（缺字段/多余字段/拼错/类型错），语义正确不标（参数顺序颠倒）

### CharTag 枚举

| 枚举 | 含义 | 下游 |
|------|------|------|
| S | 结构字符，与期望模板字符相等 | +1.0 |
| V | 值字符（参数值 / JSON value） | 相似度 0~1 |
| T | think 内容 | 0（不训练） |
| W | 前导/尾随/夹缝文本 | 0（不训练） |
| E | 首个与期望不匹配字符 | -1.0 |
| D | E 之后所有字符 | 0（不训练） |

### 模块结构

- `char_tag.py` — CharTag 枚举（StrEnum，含 trainable 属性）
- `tool_call_labeler.py` — Qwen XML tool call 标注：期望 mark 序列逐字符比对、参数无序集合匹配 + 缺失检测、值类型校验
- `json_labeler.py` — JSON 标注：围栏提取（可有可无）、状态机扫描、字段名拼错/多余检测、类型校验
- `tokenize.py` — 字符标注 → token 标注（offset_mapping，E 优先合并语义）
- `labeler.py` — `annotate()` 主入口，format_type 由 `Sample.expects_tool_calls` 传入（不猜测）

### 测试数据集

`tests/data/annotation_testset_v3.jsonl`（65 条：tool call 44 + JSON 21）覆盖完美输出/值错误/前导文本/拼错/双开标记/多余字段/缺字段/顺序颠倒/截断/乱码/think/嵌套结构/闭合后多余内容（v3.1 一刀切 E+D）等场景，每条含期望标注与 error_pos，作为标注模块回归基准。`tests/data/generate_annotation_viewer.py` 生成 HTML 逐字符着色视图。

## 为什么用 ABC 模板方法（模板方法）

**运行时 ABC 契约**：`GraspoFlowRuntimeBase(ABC)` 定义所有 runtime 必须实现的抽象方法（`generate_group`、`sequence_log_probs`、`train_batch`、`save_checkpoint` 等）。`GraspoFlowRuntime` 是生产实现，`GraspoFlowTrainer`/`SFTTrainer` 的 mixin 通过 ABC 调用 runtime，无需任何 `getattr`/`callable()` 探测（防呆）。

**适配器 ABC 继承**：每个模型族（Qwen3、Qwen3.5/3.6）有大量共享逻辑（tokenizer、chat template、batch 管理），但模型结构不同（dense vs hybrid text+vision、full-attn vs linear-attn）。ABC 基类 `TransformerAdapter` 定义流程骨架，子类只覆盖差异部分。新增模型只需定义新类并注册，零侵入现有代码。

**核心数据模型**：`Experience`、`NativeGeneration`、`ParsedCompletion`、`GroupSampleDecision` 均使用 pydantic `BaseModel` + `extra="forbid"` + `frozen=True`，在模块边界上完成校验，非法数据在边界被拦截（防呆）。

## 为什么用类改目录（类改目录）

`GraspoFlowTrainer` 包含训练循环、rollout、优化、checkpoint 四个关注点。按功能域拆分为多个 mixin 文件后，每个文件聚焦一个概念。外部使用者通过 `__init__.py` 只 import 类名，完全不感知内部拆分。

## 为什么只有 GraspoFlow 一个后端（不留负债）

历史上有过 `native_tp` 后端。v0.9 完成 GraspoFlow 迁移后立即删除旧代码——不保留"兼容模式"，不保留 `legacy/` 目录。代码库中只存在一套当前架构。这是宪法"不留技术负债"原则的直接体现。

## 确定性边界（复现性）

训练的可复现性遵循"同一 config + 同一 seed 下，排除不可控随机性后，落盘输出一致"的
定义（见宪法复现定义）。GRASPO 的确定性保证范围：

- **受控随机性**：`training.seed`（配置显式设定）覆盖 random/numpy/torch/cuda RNG；
  epoch shuffle 用独立 `random.Random(seed + epoch)` 实例；多 rank 的 batch shuffle
  由 rank 0 广播保持一致；torch/cuda RNG 状态随 checkpoint 保存与恢复。
- **不可控随机性**：GPU 内核选择（cudnn autotune 等）、机器负载导致的耗时差异
  不在复现保证范围内——同 config 两次运行的落盘结果文件一致，日志中的耗时字段
  允许有差异。
- **边界**：跨不同 GPU 型号/驱动版本训练不承诺比特级一致；需要比特级复现的场景
  应在同一台机器同一驱动下运行。
