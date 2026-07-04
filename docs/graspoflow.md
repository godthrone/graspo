# GraspoFlow — 统一 TP+PP 分布式训练后端

## 设计目标

GraspoFlow 是 GRASPO 的唯一训练后端。它将 tensor parallel (TP)、pipeline parallel (PP) 和单 GPU 模式统一在一个框架下：`pp=1,tp=1`（单卡）、`pp=1,tp=N`（纯 TP）、`pp=N,tp=1`（纯 PP）、`pp=M,tp=N`（TP+PP 混合）。用户只需在配置文件中设 `tp_size` 和 `pp_size`，不需要理解后端差异。

核心设计思想来自 Flink 等大数据分布式系统：**调度与计算分离**。调度器决定"什么时候执行哪个 microbatch"，算子负责"怎么计算和通信"。两者各自独立，可以分别测试和优化。

## 四层架构

```
Layer 3: 模型族         models/qwen3/          models/qwen35_36/
                        架构特定实现              hybrid text+vision

Layer 2: 训练编排       GraspoFlowTrainer       GraspoFlowRuntime
                        GRASPO 训练循环          分布式运行时边界

Layer 1: 通用适配       TransformerAdapter      TransformerStageOp
                        模型族共享逻辑            PP 阶段封装

Layer 0: 调度框架       operator schedule graph memory
                        完全模型无关的 Flink 原语
```

### Layer 0：调度框架（模型无关）

Layer 0 是整个系统的基石，完全不知道模型、训练目标或层的内部实现。它只操作三个抽象概念：

**Microbatch** — 流经流水线的数据单元。携带 input token IDs、hidden states、training labels，以及位置标记 `idx`（在 micro-batch 序列中的位置）。在流水线各阶段之间传递的 hidden states 是唯一流动数据。

**OpBuffer** — 算子间的 FIFO 缓冲。维护 `waterlevel` 水位线（正=有空位，负=欠数据）。支持 `push`/`pop`/`peek`/`clear`，实现流水线背压（backpressure）：下游消费慢时自动阻塞上游生产。

**ComputeOperator** — 流水线中的一个计算阶段。绑定一个输入 buffer 和一个输出 buffer，封装 forward 和 backward 方法。算子不知道自己处理的是哪个模型层——它只从 buffer 取数据、计算、写回 buffer。

**PipelineScheduler** — 抽象调度策略，接收 micro-batch 列表和流水线拓扑，产出有序执行计划。两种实现：
- `GPipeScheduler`：所有 forward 先于 backward，简单但显存峰值高
- `OneFOneBScheduler`：预热后交替执行 forward/backward，每个 micro-batch 被一个 forward 和一个 backward 覆盖，显存更均衡

**PipelineGraph** — 流水线物理拓扑：依次连接的 `ComputeOperator` 节点 + 转发 `OpBuffer`。通过 `max_inflight_microbatches` 限制同时飞行的 batch 数，控制显存峰值。

**Memory budget** — `MemoryBudget` 类根据激活值估算和 KV cache 估算计算单阶段可安全容纳的最大 micro-batch 数。`_kv_cache_batch_fits_budget` 比较 `max(kv_bytes, act_bytes × 1.5)` 与 GPU 空闲显存。

### Layer 1：通用 Transformer 适配

**TransformerStageOp** — 继承 `ComputeOperator`，是流水线中的一个 transformer 阶段。每个 op 持有模型中的若干 decoder layer，封装 layer forward、gradient checkpointing、TP all-reduce。

**TransformerAdapter** — 所有模型族的共享基类。提供：
- Tokenizer 加载、chat template 应用、batch 整理
- KV cache 管理和 full-forward fallback 选择
- Rollout 分块生成（`forward_batch_size` 控制 micro-batch 大小）
- Rank 间 metric 聚合（SUM reduction）
- `shared_rollout_prompt_chunk_size` / `shared_generation_micro_batch_size` 等共享策略
- 随机数种子管理（`config.training.seed` + epoch offset，确保可复现）

### Layer 2：训练编排

**GraspoFlowRuntime** — TP/PP 运行时的入口。负责：
- 初始化并行状态（`GraspoFlowState`）
- 加载模型权重（通过 `SafetensorIndex` per-rank 缓存）
- 构建流水线图（`PipelineGraph`）
- 管理模型 sharding 和 placement plan
- 提供 `generate_group` / `train_batch` 等高层接口

**GraspoFlowTrainer** — 训练循环主类。通过 mixin 组合：`RolloutMixin`（生成+打分）、`OptimizeMixin`（优化步骤）、`CheckpointMixin`（保存恢复）、`stats.py`（统计追踪）。训练主循环：
1. 从 ReplayBuffer 或 DataLoader 获取 prompt batch
2. Rollout：对每个 prompt 采样 `rollout_group_size` 个 completion
3. 打分：`GraspoReward.score()` 计算每个 completion 的 reward
4. 决策：根据 reward 分布和规则将 group 分类为 `perfect_skip`/`trainable`/`retry`/`invalid`
5. 训练：对 trainable group 计算 advantage、构建 Experience 并优化 LoRA

### Layer 3：模型族

每个模型族在 `models/` 下有独立目录，包含：
- `adapter.py` — TransformerAdapter 子类，提供模型特定的初始化和配置
- `model.py` — 模型定义（causal LM wrapper）
- `layers.py` — re-export shim（从 `common/layers*.py` 导入）
- `config.py` — 模型族特定的配置解析
- `ops.py` — 模型特定的算子
- `generation.py` / `logprobs.py` / `training.py` — 模型特定的方法

公共层实现在 `models/common/layers.py`（Qwen3.5/3.6 族）和 `layers_qwen3.py`（Qwen3 族），按模型族拆分以避免单文件过大（宪法 §8.4）。

## TP LoRA 梯度同步

TP 模式下，所有 rank 处理相同数据并各自计算部分梯度。完整梯度是各 rank **部分梯度之和（SUM）**，而非 DDP 中的平均值。

- `lora_a`（input projection）：权重非分片，TP rank 间共享 → 梯度需在 TP rank 间做 **SUM all-reduce**
- `lora_b`（output projection）：按 output dimension 分片 → 每个 rank 只负责自己的分片维度，**不需要同步**

这个区别体现在 `lora.py` 的 `LoRALinear` 实现中：`lora_a` 的同步和 `lora_b` 的独立分片是两条不同路径。

## KV Cache 策略

Rollout 生成阶段支持两种路径：
- **KV cache 路径**（默认）：逐 token decode 时复用已计算的 KV 对，大幅减少计算量。适用于支持 KV cache 的模型。
- **Full forward 路径**（fallback）：每次 decode 都做完整 forward。适用于不支持 KV cache 的模型（如早期 Qwen3-8B）。

`TransformerAdapter._generate_group_with_kv_cache` 和 `_generate_group_full_forward` 分别在两端实现，通过 `model.supports_kv_cache` 属性在运行时选择。

## Placement 策略

`NativePlacementPlan` 决定每层放到哪个 pipeline stage。默认策略：
- `qwen3_tp`：对 Qwen3 系列优化的均匀分布策略
- `auto`：基于层数的均匀划分
- 手动：通过 `layer_ranges: [[start, end), ...]` 精确控制
