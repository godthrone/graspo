# GRASPO 能力矩阵：训练框架能做什么 + 实测效果如何

> **里程碑对照**：本文件的 **2026-09-22 基线快照**已冻结为 `docs/capability-matrix-20260922.md`（逐字节快照，不再修改）。此后本文件继续演进；要与该里程碑对照，直接 diff 两个文件即可。

## 1. 本文档是什么

本文档是 **GRASPO 训练框架的能力矩阵手册**，只回答四问：**支持什么模型**、**有哪些功能**（哪些后端支持哪些能力）、**什么参数组合可以用**、**实测效果如何**。

**本期目标：让每个参数组合（模型 × 算法 × 模式 × 卡数）至少有一个后端能跑通。** GRASPO 有两套训练后端——**ms-swift**（复用上游、省开发）与 **native**（自研；ms-swift 覆盖不了的参数由 native 自己实现）。因此判定一个参数“能用”，看的是**该参数组合下是否至少存在一个可用后端**，而不是某个特定后端是否可用。推进原则：**能简单修复的先修；复杂的（更换并行框架、native 补实现）按覆盖缺口排期**。

全文只保留：能力表、参数组合规则、测试条件、实测台账、读懂它们所需的术语与图例。启动与使用方式见仓库根 `README.md`。四问分别落在第 3 节（支持什么模型）、第 4 节（有哪些功能）、第 5 节（什么参数组合能用）、第 7 节（实测效果如何）。

## 2. 怎么读

**① 可用性三态**（这能力现在能不能用）：`✅ 可用` = 配置可表达 + 入口完整 + 有跑通记录；`⚠️ 未验证` = **尚未可用或尚未端到端验证**（含未实现、仅官方声明支持、有入口但未测）——**这些是工程团队要开发或验证的目标**；`⛔ 不支持` = 架构不支持或用户已明确排除。

**② 实测状态词**（实验跑出什么结果）：`— 未测` = 尚未按本期模型与配置跑过端到端；`✅ 通过` = A1–A6 判据全过；`❌ 失败` = 跑过但判据未过；`⚠ 口径不可测` = 功能层跑通，但判据无法作出有意义判定（档族未标定 / 仅 n=0 兜底容差 / 结构性步数门槛不足）；`⛔ 无配方` = 当前无配方、将来可能做得了（blocked，依据 `samples/configs/matrix54/T016.blocked.md`、`T034.blocked.md`）；`⛔ 不适用` = 逻辑上不适用（not_applicable，依据 `samples/configs/matrix54/T052.not_applicable.md`）。**`⛔ 无配方` 与 `⛔ 不适用` 不是一回事**：前者 = 暂时做不了、可解锁；后者 = 该配置逻辑上不存在。**原则：「官方声明支持」≠「已验证」**，未验证项不得按已验证使用。

| 术语 | 一句话 |
|---|---|
| DP / TP / PP | 数据并行 / 张量并行 / 流水线并行。 |
| SP / CP | 都沿序列维切分：序列并行 / 上下文并行。 |
| GC / GA | 激活重算省显存 / 多步累积再更新。 |
| l2k | 只对少量 token 计算输出层，省 logits 及其梯度。 |
| packing / padding_free | 多条短样本拼进同一序列 / 免 padding 的变长批处理。 |
| FA2 / liger | 省显存的注意力实现 / 融合算子。 |
| ZeRO-1/2/3 | 优化器态 / 梯度 / 权重分片（DeepSpeed 预设）。 |
| FSDP2 | 全分片数据并行（v2），补充路径。 |
| 按层切分 | 按层把模型切到多卡。 |
| offload | 把优化器态与权重换出到 CPU 内存。 |

## 3. 支持的模型

| 大类 | 模型名称 | 是否多模态 |
|---|---|---|
| Qwen 系列（当前仅此系列） | `Qwen3.5-9B` | 是 |
| Qwen 系列（当前仅此系列） | `Qwen3.8-27B` | 是 |

## 4. 有哪些功能（后端 × 能力）

**算法**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| CPT（预训练） | 继续预训练 | ⛔ 不支持 | ⚠️ 未验证 |
| SFT（监督微调） | 用标注数据做监督微调 | ✅ 可用 | ⚠️ 未验证 |
| GRASPO（RL） | 用奖励做强化学习训练 | ⚠️ 未验证 | ⚠️ 未验证 |
| OPD（on-policy 蒸馏） | 在线策略蒸馏 | ⛔ 不支持 | ⚠️ 未验证 |

**模式**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| LoRA | 只训低秩适配器，省显存 | ✅ 可用 | ✅ 可用 |
| 全量（全参）· 9B | 训练全部权重（仅 9B） | ⚠️ 未验证 | ⚠️ 未验证 |
| 全量（全参）· 27B | 训练全部权重（27B × 全量） | ⛔ 不支持 | ⛔ 不支持 |

**层 1 · 并行维度**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| DP | 数据并行 | ✅ 可用 | ✅ 可用 |
| TP | 张量并行 | ✅ 可用 | ⚠️ 未验证 |
| PP | 流水线并行 | ⚠️ 未验证 | ⚠️ 未验证 |
| SP | 序列并行 | ✅ 可用 | ✅ 可用 |
| CP | 上下文并行 | ⛔ 不支持 | ⚠️ 未验证 |
| 按层切分 | 按层把模型切到多卡 | ⛔ 不支持 | ✅ 可用 |
| AutoTP | 自动张量并行 | ⛔ 不支持 | ⚠️ 未验证 |

**层 2 · 分片策略**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| 优化器态与梯度分片（ZeRO-1/2） | 把优化器态与梯度摊到各卡 | ⛔ 不支持 | ✅ 可用 |
| 权重分片与 CPU offload | 把权重也摊开；常驻状态换出到 CPU | ⛔ 不支持 | ✅ 可用 |
| FSDP2 | 全分片数据并行（v2） | ⛔ 不支持 | ✅ 可用 |

**层 3 · 激活与 logits 削峰**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| GC（激活重算） | 用重算换显存 | ✅ 可用 | ⚠️ 未验证 |
| l2k | 少算输出层，省 logits 及其梯度 | ⛔ 不支持 | ✅ 可用 |
| FA2（flash attention） | 更省显存的注意力实现 | ⚠️ 未验证 | ✅ 可用 |
| liger kernel | 融合算子降中间激活 | ⛔ 不支持 | ✅ 可用 |

**层 4 · 批次与序列组织**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| micro-batch size | 单次前向的样本数 | ✅ 可用 | ✅ 可用 |
| 梯度累积 | 多步累积再更新 | ✅ 可用 | ✅ 可用 |
| packing | 短样本拼进同一序列 | ⛔ 不支持 | ✅ 可用 |
| padding_free | 免 padding 的变长批处理 | ⛔ 不支持 | ✅ 可用 |
| 序列长度上限与截断 | 一条样本最多算多少 token、超长如何截断 | ⚠️ 未验证 | ✅ 可用 |

**层 5 · 精度与拓扑**

| 能力 | 一句话作用 | native | ms-swift |
|---|---|---|---|
| bf16 混合精度 | 训练数值精度 | ✅ 可用 | ✅ 可用 |
| 量化 | 低比特权重与计算 | ⛔ 不支持 | ⛔ 不支持 |
| 多机 | 跨节点扩展 | ⛔ 不支持 | ⛔ 不支持 |

> **第 4 节的可用性 ≠ 第 7 节的实测通过**：第 4 节答"能力入口是否存在"，其中 `✅ 可用` 里的"有跑通记录"指**既有历史 / 冒烟级记录**（含未达台账门槛者，且可能来自旧模型），**不等于第 7 节的台账判据**；第 7 节答"本期模型 + 本期配置下是否跑通"——**本期台账 54 档状态与计数直通机器基线 `runs/batch/results.tsv`（单一真相源，本表不自定口径）**（因**实测数据集与本次不同、方法有变、代码有变**，本节曾于上一轮重置为**全新未测**，现已按机器基线回填完毕）；**`✅ 通过 5` / `❌ 失败 22` / `⚠ 口径不可测 22` / `⛔ 无配方 2` / `⛔ 不适用 3` / `— 未测 0`**（**逐档状态与计数以第 7 节为准**）。

## 5. 参数组合规则（什么能一起开）

- **层间关系**：层 1 决定"卡怎么分" → 层 2 在同一世界大小内再摊常驻状态 → 层 3 不改分配、只削峰 → 层 4 决定"一次算多少" → 层 5 管精度与跨机。
- **native**：`TP × PP × DP = 卡数`，不匹配即报错；SP 与 TP 同进程组、**不额外占卡**、须 `TP ≥ 2`；不做 CP。
- **ms-swift**：SP 是独立轴，`DP 度 = 卡数 ÷ SP 度`（与 native 不是同一条恒等式）；可用手段 DP / SP / ZeRO / offload / FSDP2 / 按层切分 / packing / padding_free / l2k / FA2 / liger；TP / PP / CP 需依托其内部 Megatron 引擎，当前 `⚠️ 未验证`。
- **可整除性**：native 的 TP 须**整除注意力头数**且 `TP ≤ KV 头数`；ms-swift 的 SP 度须**整除 KV 头数**。
- **三套 SP 机制不可混用**：native 布尔开关 / ms-swift 标准路径整数度 / Megatron 段布尔开关。
- **互斥与前置**：DeepSpeed / FSDP2 / 按层切分三者互斥；`packing ⇒ padding_free ⇒ 强制 FA2`；`SP > 1` 时 l2k 不可用（默认即关）；多模态下 l2k 强制关闭。
- **经验不可直接外推**：既有的 **CP / TP / PP 实战经验来自另一个旧模型**，**不能直接外推到本期目标模型**。

## 6. 测试条件

| 项 | 取值 |
|---|---|
| 卡数与机器 | 单机 8×A800-80G，**只用 1 / 2 / 4 卡；8 卡不做** |
| 上下文长度口径 | **递增加长法实测得出，不预设档位**；**长上下文必须靠序列维分片（SP / CP）**，DP / TP / PP 不切序列维 |
| 档位构成 | `2 模型 × 2 模式 × 2 后端 × 3 卡数 × 2 算法（SFT/GRASPO）= 48`；`+ 2 模型 × 2 模式 × 3 卡数 × 2 算法（CPT/OPD，仅 ms-swift）= 24`；`− 27B 全量 = −18` ⇒ **54 档** |
| 正式记录门槛 | 每档 ≥ 1 完整 epoch 且 ≥ 5 optimizer step；SFT ≥ 100 条、RL ≥ 20 条。★ **其中「≥ 5 optimizer step」以「该档可产出步数上限 ≥ 5」为前提**——上限 < 5 的档（见 §7 后 mini 披露：`GRASPO × native` 在 `subset_size=20`、`max_epochs=1` 下为 1 卡 3 / 2 卡 2 / 4 卡 1 步）该条门槛**结构上不适用**，台账标 `⚠ 口径不可测`（**不是** `❌ 失败`、**不是** `✅ 通过`）。**门槛值本身不放宽**（仍是 5）——判据侧实现见 `src/graspo/core/result_judge.py` 的 `resolve_step_gate_applicability` |
| 评测集 | `ELAM V5` 的 test 集，与训练同源、不重叠 |
| 范围约束 | **所有训练都必须支持多模态（视觉）通路**；MoE 不在本期；**不在生产任务占用的卡上做测试** |

## 7. 实测效果（54 档台账）

| 条件档编号 | 模型 | 算法 | 模式 | 后端 | 卡数 | 最大可行上下文（实测） | 实测每卡峰值显存(GiB) | 实测状态 | 实测日期 | 业界情况调研（含出处） |
|---|---|---|---|---|:--:|---|---|---|---|---|
| T001 | 9B | CPT | LoRA | ms-swift | 1 | — 未测 | 25.48 GiB | ⚠ 口径不可测 | 2026-09-22 | ms-swift 官方 8B(非9B) LoRA 单卡 22GB 可跑【S1】;9B 同量级,单卡 80G 富余。 |
| T002 | 9B | CPT | LoRA | ms-swift | 2 | — 未测 | 26.67 GiB | ⚠ 口径不可测 | 2026-09-22 | 同 T001 档:单卡已够,2 卡 DDP 仅提速、不降每卡显存,业界无 2 卡专档【S1】。 |
| T003 | 9B | CPT | LoRA | ms-swift | 4 | — 未测 | 26.68 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;LLaMA-Factory 折 9B LoRA≈18GB 总量,与卡数无关【S7】。 |
| T004 | 9B | CPT | 全量 | ms-swift | 1 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 单卡不可行:9B 全量 bf16≈162GB 总量(18x 折算)【S7】;需 ZeRO-3+offload【S5/S9】。 |
| T005 | 9B | CPT | 全量 | ms-swift | 2 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ≈162/2≈81GB/卡>80G(折算)【S7】;业界须 ZeRO-3 分片+offload【S5/S9】。 |
| T006 | 9B | CPT | 全量 | ms-swift | 4 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ≈162/4≈40GB/卡+激活(折算)【S7】;ZeRO-2 分片即可,业界常规【S10】。 |
| T007 | 27B | CPT | LoRA | ms-swift | 1 | — 未测 | 60.62 GiB | ⚠ 口径不可测 | 2026-09-22 | 官方最近档为 30B-A3B(MoE,非稠密 27B)【S1】;折 27B LoRA≈54GB 总量【S7】,单卡临界。 |
| T008 | 27B | CPT | LoRA | ms-swift | 2 | — 未测 | 60.95 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;30B-A3B 全量 ZeRO2 用 16×60GiB、ZeRO3 在 16×80GiB OOM;无 27B 稠密档【S1】。 |
| T009 | 27B | CPT | LoRA | ms-swift | 4 | — 未测 | 60.95 GiB | ⚠ 口径不可测 | 2026-09-22 | 同 2 卡;官方 30B-A3B(MoE) LoRA 档为 4×60GiB【S1】,稠密 27B 量级相近。 |
| T010 | 9B | SFT | LoRA | native | 1 | — 未测 | 20.36 GiB | ✅ 通过 | 2026-09-22 | TRL+PEFT 单卡 LoRA 8B 级≈16-18GB(2x 折算)【S7/S11】;Unsloth 再省约 70% 显存【S13】。 |
| T011 | 9B | SFT | LoRA | native | 2 | — 未测 | 20.36 GiB | ❌ 失败 | 2026-09-22 | 业界单卡即够;2 卡走 DDP/FSDP2 只提速,每卡显存不降,无专档【S14】。 |
| T012 | 9B | SFT | LoRA | native | 4 | — 未测 | 20.36 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;FSDP2 全分片可再降每卡常驻态,9B LoRA 不必用【S14/S15】。 |
| T013 | 9B | SFT | LoRA | ms-swift | 1 | — 未测 | 31.02 GiB | ✅ 通过 | 2026-09-22 | ms-swift 官方 8B(非9B) LoRA 22GB/单卡 A10【S1】;我方 16105 长序列实测 43.87GiB 更高。 |
| T014 | 9B | SFT | LoRA | ms-swift | 2 | — 未测 | 28.53 GiB | ✅ 通过 | 2026-09-22 | 同上;长序列业界靠 SP/FA2 降每卡显存【S5】,无 2 卡专档。 |
| T015 | 9B | SFT | LoRA | ms-swift | 4 | — 未测 | 28.54 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;折 9B LoRA≈18GB 总量,与卡数无关【S7】。 |
| T016 | 9B | SFT | 全量 | native | 1 | — 未测 | — 未测 | ⛔ 无配方 | 2026-09-22 | 原为全量档(单卡须 ZeRO-Offload【S9】);本轮改用 fp16+LoRA ⇒ 业界 8B(非9B) LoRA 单卡 22GB【S1】、折 9B LoRA≈18GB 总量【S7】,单卡 80G 可行。 |
| T017 | 9B | SFT | 全量 | native | 2 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ≈81GB/卡>80G(折算)【S7】;业界须 ZeRO-3 分片或 offload【S9】。 |
| T018 | 9B | SFT | 全量 | native | 4 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ≈40GB/卡+激活(折算)【S7】;ZeRO-2 分片为业界常规,4 卡可行【S10】。 |
| T019 | 9B | SFT | 全量 | ms-swift | 1 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ms-swift 支持 ZeRO-2/3+offload【S5】;单卡须 ZeRO-3+offload(官方 30B-A3B 全量 ZeRO3 仍 OOM)【S1】。 |
| T020 | 9B | SFT | 全量 | ms-swift | 2 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ≈81GB/卡>80G(折算)【S7】;ZeRO-3 分片后可行【S5】。 |
| T021 | 9B | SFT | 全量 | ms-swift | 4 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ≈40GB/卡+激活(折算)【S7】;ZeRO-2 可,官方对比中 ZeRO-2 档为 16×60GiB【S1】。 |
| T022 | 27B | SFT | LoRA | native | 1 | — 未测 | 55.44 GiB | ✅ 通过 | 2026-09-22 | 折 27B LoRA≈54GB 总量【S7】,单卡临界(需 GC/FA2);官方无稠密 27B 档,最近 30B-A3B MoE【S1】。 |
| T023 | 27B | SFT | LoRA | native | 2 | — 未测 | 55.44 GiB | ❌ 失败 | 2026-09-22 | 同 1 卡;2 卡 DDP 不降每卡显存,FSDP2/ZeRO 分片可降【S14/S15】。 |
| T024 | 27B | SFT | LoRA | native | 4 | — 未测 | 55.44 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;官方 30B-A3B(MoE) LoRA 参考 4×60GiB【S1】,非稠密 27B。 |
| T025 | 27B | SFT | LoRA | ms-swift | 1 | — 未测 | 66.17 GiB | ✅ 通过 | 2026-09-22 | 官方最近为 30B-A3B(MoE,非稠密 27B) LoRA 4×60GiB【S1】;27B LoRA 折≈54GB 总量【S7】。 |
| T026 | 27B | SFT | LoRA | ms-swift | 2 | — 未测 | 64.65 GiB | ❌ 失败 | 2026-09-22 | 同 1 卡;业界无 27B 稠密 2 卡专档,ZeRO-2 可分片【S5】。 |
| T027 | 27B | SFT | LoRA | ms-swift | 4 | — 未测 | 64.65 GiB | ⚠ 口径不可测 | 2026-09-22 | 4×60GiB 为 MoE 30B-A3B 档,稠密 27B 每卡更高【S1】;LoRA+ZeRO-2 可行。 |
| T028 | 9B | GRASPO | LoRA | native | 1 | — 未测 | 27.53 GiB | ⚠ 口径不可测 | 2026-09-22 | TRL GRPOTrainer 比 PPO 省显存【S14/S12】;RL 另有 rollout 显存,须释放训练显存【S4】。 |
| T029 | 9B | GRASPO | LoRA | native | 2 | — 未测 | 27.37 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;业界无 2 卡专档;vLLM 用 TP 摊 rollout 显存【S4】。 |
| T030 | 9B | GRASPO | LoRA | native | 4 | — 未测 | 27.37 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;4 卡可将训练与 rollout 分置以降每卡压力【S4】。 |
| T031 | 9B | GRASPO | LoRA | ms-swift | 1 | — 未测 | 24.45 GiB | ⚠ 口径不可测 | 2026-09-22 | ms-swift GRPO colocate 下须按官方释放 vLLM/训练显存【S4】;LoRA 单卡业界可行【S1】。 |
| T032 | 9B | GRASPO | LoRA | ms-swift | 2 | — 未测 | 23.05 GiB | ❌ 失败 | 2026-09-22 | 同上;无 2 卡专档;vLLM TP 摊 rollout 显存【S4】。 |
| T033 | 9B | GRASPO | LoRA | ms-swift | 4 | — 未测 | 23.06 GiB | ❌ 失败 | 2026-09-22 | 同上;4 卡可分置训练与 rollout【S4】。 |
| T034 | 9B | GRASPO | 全量 | native | 1 | — 未测 | — 未测 | ⛔ 无配方 | 2026-09-22 | 原为最重组合(全量+rollout,须 ZeRO-3+offload、常需多机【S9】);本轮改用 fp16+LoRA ⇒ 9B LoRA≈18GB 总量【S7】+rollout,业界 GRPO 的 LoRA 单卡可行(Unsloth 8B GRPO 总占用 54.33GB【S13】),仍须按官方释放 vLLM/训练显存【S4】。 |
| T035 | 9B | GRASPO | 全量 | native | 2 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 同上;2 卡≈81GB/卡>80G(折算)【S7】,须 ZeRO-3+offload【S9】。 |
| T036 | 9B | GRASPO | 全量 | native | 4 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 同上;4 卡 ZeRO-2 可容训练段,rollout 另计【S4/S10】。 |
| T037 | 9B | GRASPO | 全量 | ms-swift | 1 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | ms-swift 全量 RL 需 ZeRO-3/offload,官方 GRPO 有释放训练显存开关【S4】;9B 全量≈162GB【S7】。 |
| T038 | 9B | GRASPO | 全量 | ms-swift | 2 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 同上;2 卡≈81GB/卡>80G(折算)【S7】,须 ZeRO-3+offload【S4】。 |
| T039 | 9B | GRASPO | 全量 | ms-swift | 4 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 同上;4 卡 ZeRO-2 可容训练段【S4/S10】。 |
| T040 | 27B | GRASPO | LoRA | native | 1 | — 未测 | 61.77 GiB | ⚠ 口径不可测 | 2026-09-22 | 折 27B LoRA≈54GB 总量【S7】,叠加 rollout 后单卡业界需 offload/l2k;无稠密 27B 档【S1】。 |
| T041 | 27B | GRASPO | LoRA | native | 2 | — 未测 | 61.73 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;2 卡分片后可行【S9】。 |
| T042 | 27B | GRASPO | LoRA | native | 4 | — 未测 | 61.72 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;4 卡余量充足【S10】。 |
| T043 | 27B | GRASPO | LoRA | ms-swift | 1 | — 未测 | 60.75 GiB | ⚠ 口径不可测 | 2026-09-22 | ms-swift 无稠密 27B GRPO 档,最近为 30B-A3B MoE【S1】;需 ZeRO-2+l2k 并释放训练显存【S4】。 |
| T044 | 27B | GRASPO | LoRA | ms-swift | 2 | — 未测 | 56.15 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;2 卡 ZeRO-2 可分片【S4】。 |
| T045 | 27B | GRASPO | LoRA | ms-swift | 4 | — 未测 | 55.53 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;4 卡余量最足【S4】。 |
| T046 | 9B | OPD | LoRA | ms-swift | 1 | — 未测 | 75.22 GiB | ⚠ 口径不可测 | 2026-09-22 | OPD=ms-swift OPD-RL【S3】;官方称默认全词表 KL 易 OOM,可用 --gkd_logits_topk 降显存【S3】。 |
| T047 | 9B | OPD | LoRA | ms-swift | 2 | — 未测 | 75.09 GiB | ❌ 失败 | 2026-09-22 | 同上;2 卡可分片,仍建议 top-K 蒸馏省显存【S3】。 |
| T048 | 9B | OPD | LoRA | ms-swift | 4 | — 未测 | 75.09 GiB | ⚠ 口径不可测 | 2026-09-22 | 同上;4 卡余量足,top-K 仍推荐【S3】。 |
| T049 | 9B | OPD | 全量 | ms-swift | 1 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 全量 OPD:官方 OPD-RL 全词表 KL 易 OOM【S3】;须 ZeRO-3/offload+top-K,9B 全量≈162GB【S3/S7】。 |
| T050 | 9B | OPD | 全量 | ms-swift | 2 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 同上;2 卡≈81GB/卡>80G(折算)【S7】,须 ZeRO-3+offload【S3】。 |
| T051 | 9B | OPD | 全量 | ms-swift | 4 | — 未测 | — 未测 | ❌ 失败 | 2026-09-22 | 同上;4 卡 ZeRO-2 可容训练段【S3/S10】。 |
| T052 | 27B | OPD | LoRA | ms-swift | 1 | — 未测 | — 未测 | ⛔ 不适用 | 2026-09-22 | 我方不适用;业界同能力为 OPD-RL,官方无稠密 27B 档,最近 30B-A3B MoE【S1/S3】。 |
| T053 | 27B | OPD | LoRA | ms-swift | 2 | — 未测 | — 未测 | ⛔ 不适用 | 2026-09-22 | 同 T052;业界须 ZeRO-2/offload 分片,单卡 27B 折≈54GB 总量【S7/S3】。 |
| T054 | 27B | OPD | LoRA | ms-swift | 4 | — 未测 | — 未测 | ⛔ 不适用 | 2026-09-22 | 同 T052;4 卡余量较足,仍受全词表 KL 显存约束,建议 top-K【S3】。 |

> **出处清单（编号 → 完整 URL）**：`【S1】` https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3-Best-Practice.html ｜ `【S2】` https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3-VL-Best-Practice.html ｜ `【S3】` https://swift.readthedocs.io/zh-cn/latest/Instruction/Distillation.html ｜ `【S4】` https://swift.readthedocs.io/zh-cn/latest/BestPractices/GRPO.html ｜ `【S5】` https://swift.readthedocs.io/zh-cn/latest/Instruction/Pre-training-and-Fine-tuning.html ｜ `【S6】` https://github.com/modelscope/ms-swift ｜ `【S7】` https://github.com/hiyouga/LLaMA-Factory#hardware-requirement（README `### Hardware Requirement`，原文标注 `*estimated*`）｜ `【S8】` https://qwen.readthedocs.io/zh-cn/latest/training/ms_swift.html ｜ `【S9】` https://www.deepspeed.ai/tutorials/zero-offload/ ｜ `【S10】` arXiv:1910.02054（ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, SC'20）｜ `【S11】` arXiv:2106.09685（LoRA: Low-Rank Adaptation of Large Language Models, ICLR'22）｜ `【S12】` arXiv:2402.03300（DeepSeekMath）｜ `【S13】` https://github.com/unslothai/unsloth ＋ https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide ｜ `【S14】` https://github.com/huggingface/trl ｜ `【S15】` https://github.com/huggingface/peft

**结构性计数**：目标档位总数 **54**（CPT 9 ｜ SFT 18 ｜ GRASPO 18 ｜ OPD 9）；`✅ 通过 5` ｜ `❌ 失败 22` ｜ `⚠ 口径不可测 22` ｜ `⛔ 无配方 2` ｜ `⛔ 不适用 3` ｜ `— 未测 0`——等式 `✅5 + ❌22 + ⚠22 + ⛔无配方2 + ⛔不适用3 + —未测0 = 54` 成立。
> §7 逐档状态与计数**直通机器基线** `runs/batch/results.tsv`（单一真相源），本表不自定口径。
> **`❌ 失败 22` 构成**：`environment_cuda_mismatch`（镜像 CUDA 13.2 工具链 vs torch `+cu130`，**环境缺陷、可修**）**8** ｜ `A4_only`（3 首步零容差 + 3 终态容差兜底，**功能层全过、仅判据未过**）**6** ｜ `多模态 LoRA 视觉预检`**4** ｜ `形状不匹配`**2** ｜ `TP 组数据不一致`**2**。
> **`⚠ 口径不可测 22` 构成**：`structural_limited`（可达步数 3/2/1 < §6 门槛 5，**训练本身成功**，补标定也救不了）**6**（T028/T029/T030/T040/T041/T042） ｜ `indeterminate`（仅 A4 档族未标定 fail-closed，**补标定后可转明确 ✅/❌**）**16**。
> 「最大可行上下文」列本轮**未做 ramp，一律 `— 未测`**。

## 8. 达标线与发布条件

| 算法 | 判据 | 口径 |
|---|---|---|
| SFT | 测试集**综合准确率 ≥ 50%** | **阈值由用户拍板**；本期各档须在同一评测集上自测 |
| GRASPO（RL） | **Δ ≥ 20 个百分点** | 同一 vLLM 评测流程、同一评测集，**训练前后各评一次**；`Δ = 训练后正确率 − 训练前正确率`（**绝对百分点口径**） |
| CPT / OPD | 本期**只判跑通**，效果判据待定 | **不阻塞发布** |

**发布条件**：全部目标档位跑通，且 SFT / GRASPO 效果达标 ⇒ 本期目标达成，可发布 `v1.1.0`。**CPT / OPD 不阻塞发布**——这两类算法的效果判据本期未给定，只按「跑通」处置。
