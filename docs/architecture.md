# GRASPO 架构设计

## 核心理念

GRASPO 是一个 GRPO 风格的 LoRA 强化学习训练器，面向结构化输出任务（JSON 生成、工具调用、信息抽取等）。设计遵循 BADGE 开发宪法的边界思维和防呆原则。

详细的分话题文档：
- **[GraspoFlow 后端](graspoflow.md)** — Flink 风格 TP+PP 分布式调度框架
- **[训练方法论](training_methodology.md)** — GRPO 改进版算法、奖励系统、效果评估

## 三层架构

```
┌─────────────────────────────────────────────────┐
│  Layer 3: 模型族                                  │
│  models/qwen3/       models/qwen35_36/          │
│  纯模型实现，不依赖训练框架                        │
├─────────────────────────────────────────────────┤
│  Layer 2: 训练编排                                │
│  trainer/            runtime.py                 │
│  GRASPO 训练循环 + 分布式运行时边界                │
├─────────────────────────────────────────────────┤
│  Layer 1: 通用 Transformer 适配                   │
│  transformer_adapter.py  transformer_op.py      │
│  跨模型族的通用逻辑                                │
├─────────────────────────────────────────────────┤
│  Layer 0: 调度框架                                │
│  operator.py  schedule.py  graph.py  memory.py  │
│  Flink 风格的计算-通信分离流水线                    │
└─────────────────────────────────────────────────┘
```

依赖方向：`trainer → runtime → adapter → transformer_adapter → transformer_op → Layer 0`

## 计算与设施分离（宪法 §1.3）

- **计算层（core/）**：纯逻辑，零 GPU/网络/IO 依赖。奖励计算、advantage 计算、结构化比较、缓冲区管理。可在单线程本地运行，可独立测试。
- **设施层（backends/graspoflow/）**：GPU 通信、TP/PP 分布式、模型加载、checkpoint 读写。唯一训练后端。

## 数据流

```
配置文件(YAML) → GraspoConfig(pydantic校验) → CLI → GraspoFlowTrainer
                                                      ↓
                    JSONL数据 → load_jsonl → Sample → rollout → reward评分
                                                      ↓
                                              ReplayBuffer → 优化步骤
                                                      ↓
                                              checkpoint 保存
```

## 为什么用 ABC 模板方法（宪法 §9.2）

每个模型族（Qwen3、Qwen3.5/3.6）有大量共享逻辑（tokenizer、chat template、batch 管理），但模型结构不同（dense vs hybrid text+vision、full-attn vs linear-attn）。ABC 基类 `TransformerAdapter` 定义流程骨架，子类只覆盖差异部分。新增模型只需定义新类并注册，零侵入现有代码。

## 为什么用类改目录（宪法 §8.3）

`GraspoFlowTrainer` 包含训练循环、rollout、优化、checkpoint 四个关注点。按功能域拆分为多个 mixin 文件后，每个文件聚焦一个概念。外部使用者通过 `__init__.py` 只 import 类名，完全不感知内部拆分。

## 为什么只有 GraspoFlow 一个后端（宪法 §16.1）

历史上有过 `native_tp` 后端。v0.9 完成 GraspoFlow 迁移后立即删除旧代码——不保留"兼容模式"，不保留 `legacy/` 目录。代码库中只存在一套当前架构。这是宪法"不留技术负债"原则的直接体现。
