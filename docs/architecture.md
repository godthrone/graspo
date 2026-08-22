# GRASPO 架构设计

GRASPO 是一个 GRPO 风格的 LoRA 强化学习训练器，面向结构化输出任务（JSON 生成、工具调用、信息抽取等）。设计遵循**边界思维**和**防呆原则**：模块边界清晰、接口稳定、防线前置。

## 核心特性

| 特性 | 说明 |
|------|------|
| **字符级标注驱动 token 级 reward** | 逐字符比对结构模板，通过 offset_mapping 映射到 token——天然 tokenizer 无关。仅首个不匹配字符受罚，其后排除在训练之外 |
| **结构化输出专用 reward** | 递归 dict 比较，双分数（数值精度用于梯度，结构正确用于门控），多目标最优匹配 |
| **组决策体系与防御纵深** | 六路分类（perfect_skip/trainable/invalid/retry/no_preference_gap）在训练边界拦截噪声，质量加权 advantage 防止收敛到"差组里最好" |
| **RL+SFT 同构** | 同一套 JSONL 数据格式、同一套模型加载、同一套 checkpoint 格式。SFT 教格式，RL 优质量 |
| **ripple/flow 分层** | 算法层（ripple，涟漪）纯计算，命名来自 token 间信用分配的涟漪效应；设施层（flow，水流）负责分布式执行，命名来自数据在流水线中的持续流动 |
| **TP+DP+PP+SP+Checkpoint 五位一体** | 统一 GraspoFlow 后端，五维并行正交：TP 分片参数、DP 分片数据、PP 分片层、SP 分片序列、Checkpoint 节省显存。用户只需配置 `tp_size`、`dp_size`、`pp_size`，框架自动推导 world_size。多投入资源 = 跑更大模型 + 跑更快 |
| **插件化模型适配** | ABC 模板方法 + 注册表，新增模型族只需定义子类并注册，零侵入现有代码 |
| **多模态训练** | 图像+文本联合训练，三层防线防止静默丢图，SFT/RL 双路径编码对齐 |

## 五维并行模型

### 设计哲学：自适应并行

GRASPO 的设计目标是**让分布式训练对用户透明**。用户只需要给出**资源**（几张 GPU）和**训练任务**（模型 + 数据 + 训练方法），除非资源低于任务的最低要求，否则 GRASPO 就能通过 TP+PP+SP+DP+Checkpoint 五位一体的参数组合，自动让训练任务以最高效率运行。

```mermaid
flowchart LR
    USER["用户<br/>给出资源 + 任务"] --> GRASPO["GRASPO<br/>自动选择最优并行策略"]
    GRASPO --> TP["TP 分片参数"]
    GRASPO --> DP["DP 分片数据"]
    GRASPO --> PP["PP 分片层"]
    GRASPO --> SP["SP 分片序列"]
    GRASPO --> CKPT["Checkpoint 省显存"]
    TP & DP & PP & SP & CKPT --> TRAIN["高效训练"]
```

**核心价值**：在生产领域模型训练中，调试分布式训练框架通常占据大量开发时间。GRASPO 将这部分工作消除——用户不需要理解 NCCL 拓扑、不需要手动调 TP/DP/PP 配比、不需要处理 PCIe vs NVLink 的差异。这是 GRASPO 作为开源框架的核心竞争力。

**当前状态**（2026-08-21 实测，4×A800，Qwen3.5-9B，多模态 SFT/RL）：

| 并行维度 | 可用组合 | 不可用 |
|---------|:--:|------|
| TP+DP+SP+GC | 5/9 组合验证通过 | — |
| 含 PP>1 | — | 4/9 组合（PP 多模态未实现） |

详见 [并行测试矩阵](../.local/parallelism-test-matrix-20260821.md)。

```mermaid
flowchart TB
    subgraph PARALLEL["五位一体并行"]
        TP["TP（Tensor Parallel）<br/>分片模型参数 · 同数据"]
        DP["DP（Data Parallel）<br/>分片训练数据 · 不同数据"]
        PP["PP（Pipeline Parallel）<br/>分片模型层 · 流水线"]
        SP["SP（Sequence Parallel）<br/>分片序列维度 · 省显存"]
        CKPT["Gradient Checkpoint<br/>重计算换显存"]
    end
    TP --> DP --> PP --> SP --> CKPT
```

| 维度 | 配置 | 默认值 | 作用 | 通信 |
|------|------|:----:|------|------|
| **TP** | `tp_size` | 2 | 分片 attention heads / MLP 维度 | all_reduce(SUM) |
| **DP** | `dp_size` | 1 | 不同数据分片独立训练 | all_reduce(AVG) |
| **PP** | `pp_size` | 1 | 分片模型层到不同 stage | 异步 P2P（isend/irecv）+ 背压 |
| **SP** | `sequence_parallel` | false | TP 组内沿序列维度分片激活值 | reduce_scatter + all_gather |
| **Checkpoint** | `gradient_checkpointing` | true | 前向不存中间激活，反向重计算 | 无 |

**五维正交**：各维度独立配置、独立生效。`world_size = dp_size × tp_size × pp_size`。

**PP 流水线架构**：PP 是 Flow 设施层中最复杂的形态，采用 **Flink 风格"调度与计算分离"**——调度策略（`parallel/scheduling/`，默认 1F1B（唯一调度））决定 forward/backward 时序，异步 P2P 通信层（`parallel/pipeline_comm.py`，双向进程组 + CUDA stream 重叠 + 有界背压）决定数据流动，二者与模型计算层解耦。未来 interleaved / ZeroBubble 作为新调度策略插上即可，无需重写通信或计算层。详见 [Flow 设施层](flow.md)。

## 三层架构

```mermaid
flowchart TB
    subgraph L0["入口层 cli/"]
        CLI["launch / export / validate-reward / evaluate-checkpoint / analyze-profile"]
    end
    subgraph L1["算法层 ripple/ · 纯计算"]
        R["reward/ · parity/ · loss · buffer<br/>data · parsing/ · monitoring/ · multimodal/<br/>annotation/"]
    end
    subgraph L2["通用件 core/"]
        C["schema.py · chat_template.py · lora.py"]
    end
    subgraph L3["设施层 flow/"]
        T["trainer/"]
        A["adapters/ · models/qwen3 + qwen35_36"]
        P["parallel/ · pipeline_comm.py · scheduling/ · lora/ · runtime · memory"]
    end
    CLI --> C
    CLI --> T
    T --> R
    T --> A
    A --> P
    T --> P
```

依赖方向：`cli → trainer → adapters → parallel`；`flow → ripple/core` 单向。

### 三层边界

| 层 | 职责 | 单测条件 |
|----|------|---------|
| **ripple/** | 算法逻辑：奖励、advantage、loss、解析、标注、监控 | 纯 CPU，无 GPU/网络/文件 IO |
| **core/** | 跨层契约：配置模型、chat template、LoRA 工具 | 同上 |
| **flow/** | 执行载体：TP/DP/PP/SP 分布式、模型加载、checkpoint、训练循环 | 需 GPU（纯设施逻辑可在单 GPU 上测试） |

**边界判断**：改这个文件会让训练结果变吗？会 → ripple。不会但和配置有关 → core。其余 → flow。

**DP 的边界**：DP 完全在 flow/ 中实现。ripple 永远不知道 `dp_rank`、`dp_size`、`dp_group` 的存在——它只看到"一份数据"，flow 负责把不同数据分片喂给不同的 DP rank。

详细设计见：
- **[Ripple 算法层](ripple.md)** — 奖励、标注、advantage、group 决策、loss
- **[Flow 设施层](flow.md)** — TP+DP+PP+SP 调度、模型适配、训练循环、LoRA 管理

## 数据流

```mermaid
flowchart TD
    CFG["配置文件 YAML"] --> SCHEMA["GraspoConfig pydantic 校验"] --> CLI["CLI"]
    CLI --> TRAINER["SftTrainer / GraspoFlowTrainer"]
    DATA["JSONL 数据"] --> LOAD["load_jsonl"] --> SAMPLE["Sample"]
    SAMPLE --> RLO["rollout / tokenize"] --> REW["reward 评分"]
    RLO --> BUF["ReplayBuffer / collate"] --> OPT["优化步骤"]
    OPT --> CKPT["checkpoint 保存"]
```

## RL+SFT 同构

GRASPO 的 RL 和 SFT 共享同一套 JSONL 数据格式、同一套模型加载、同一套 checkpoint 格式。用户只需在配置中切换 `train_method`。

```mermaid
flowchart LR
    SFT["SFT<br/>train_method: sft"] --> LOAD["SftTrainer 加载 JSONL"]
    LOAD --> TXT["纯文本路径"]
    LOAD --> MM["多模态路径<br/>MultimodalDeferred"]
    TXT --> EPOCH["训练 N epochs<br/>保存 LoRA"]
    MM --> EPOCH
    EPOCH --> EXPORT["graspo export<br/>合并 LoRA"]
    EXPORT --> RL["RL<br/>train_method: graspo"]
    RL --> RLN["GraspoFlowTrainer<br/>加载合并模型 + 新 LoRA"]
    RLN --> RLL["M epochs RL 长训"]
```

### SFT 格式对齐（关键不变式）

SFT 和 RL 共享同一套 JSONL 数据，但 target text 生成方式不同：

- **RL**：模型生成 completion → parser 解析 → 与 target 比较计算 reward
- **SFT**：`build_sft_target_text` 直接生成 XML，**不经过** `tokenizer.apply_chat_template`

**为什么不能走 chat template？** Qwen 的 chat template 在 assistant 消息前插入 `\n response\n` 前缀，但 RL 推理时的 assistant prefix 是 `<|im_start|>assistant\n`。两者不一致会导致格式错位。

**正确做法**：直接生成纯 `<tool_call>...</tool_call>` XML，与模型 RL 推理时的实际输出字符级一致。

### XML 格式对齐：与 Base Model 原生输出一致

Qwen3.5 在预训练中学会的 XML 工具调用格式是参数值位于独立行。如果 SFT 使用内联紧凑格式，LoRA（30M 参数，模型 0.3%）被迫同时改写格式和内容，在 405 条小样本上导致灾难性干扰——模型在新旧格式间摇摆，输出崩溃 XML。

**原则**：SFT 应该只教模型**内容**（what to say），不改变**格式**（how to say）。格式是预训练已经学会的能力，不应该被 LoRA 覆盖。

## 关键设计决策

### ABC 模板方法

运行时和适配器均使用 ABC 定义契约。`GraspoFlowRuntimeBase(ABC)` 定义所有 runtime 必须实现的抽象方法，`TransformerAdapter` 定义模型族适配流程骨架。新增模型只需定义新类并注册，零侵入现有代码。

### 类改目录

`GraspoFlowTrainer` 包含训练循环、rollout、优化、checkpoint 四个关注点，按功能域拆分为多个文件后，每个文件聚焦一个概念。外部通过 `__init__.py` 只 import 类名，完全不感知内部拆分。

### 单后端原则

历史上存在过 `native_tp` 后端。v0.9 完成 GraspoFlow 迁移后立即删除旧代码——不保留"兼容模式"，不保留 `legacy/` 目录。代码库中只存在一套当前架构。

### 五维并行正交

TP、DP、PP、SP、Checkpoint 五个维度彼此独立——用户调整任何一个不影响其他维度的语义。DP 的 `dp_size` 是独立配置项，不自动推导（显式优于隐式，宪法 §2.2）。`world_size = dp_size × tp_size × pp_size` 由框架自动校验。

## 确定性保证

训练可复现性遵循"同一 config + 同一 seed 下，排除不可控随机性后，落盘输出一致"的定义。

- **受控随机性**：`training.seed` 覆盖 random/numpy/torch/cuda RNG；epoch shuffle 用独立 `random.Random(seed + epoch)` 实例
- **不可控随机性**：GPU 内核选择、机器负载导致的耗时差异不在复现保证范围内
- **边界**：跨不同 GPU 型号/驱动版本不承诺比特级一致；需要比特级复现时应在同一台机器同一驱动下运行