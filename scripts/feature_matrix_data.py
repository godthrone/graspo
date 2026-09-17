#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能矩阵的需求层数据源：目标档位常量 + 结构性计数，供 `gen_feature_matrix.py` 渲染。

职责范围：只保存**标识性内容**（目标范围、状态口径、分层矩阵、负面清单、实现缺口）与
**结构性计数**；**不承担**渲染职责（在 `gen_feature_matrix.py`），**不承担**实测数据职责——
实测列一律为空，不出现任何估算 / 推算 / 外推数值。本模块不做 I/O。

已拍板口径（用户 + 指挥官）：
  · native 目标 = 并行 DP / TP / PP / **SP**（SP 与 TP 同组，须 TP≥2，不额外占卡）
  · **需求以「分片/降显存方法」为单位，不以"支持哪个框架"为单位**；不要求同时实现三套框架
  · **Megatron 引擎优先**（TP/PP/CP/SP/分布式优化器最全，且唯一覆盖 CP）；
    DeepSpeed / FSDP2 / device_map 作为补充（Megatron 覆盖不到或成本过高时启用）。
    **前提（声明层 ≠ 事实层）**：对目标两模型只有注册层证据（官方支持表 ✔、注册表在 `mcore_bridge`），
    **运行时未实测**（既有实测均为 `Qwen3-8B`）⇒ 须先过 §2.8 G-12 的 1-iter 冒烟
  · **对外只有 native / ms-swift 两档后端**；引擎/路径选择是 ms-swift 的内部实现策略
  · 27B × 全量 = 不做｜CPT / OPD = 仅 ms-swift，**LoRA 与全量都要支持**｜8 卡 = 不做｜CP 不加 native
  · 多模态 = 所有训练的目标能力
"""

MODELS = ("9B", "27B")
ALGOS = ("CPT", "SFT", "GRASPO", "OPD")
MODES = ("LoRA", "全量")
BACKENDS = ("native", "ms-swift")
CARDS = (1, 2, 4)

IMPLEMENTED = "✅ 已实现可跑"
PARTIAL = "⚠️ 有入口未验证"
TODO = "⛔ 未实现"
FALLBACK = "⛔ 本期不作为需求（按需补充）"
NO = "⛔ 不做（用户已定）"
FIELD = "⚠️ 引擎字段存在，通路未接通"

# ------------------------------------------------------------------ §2.1 算法轴
ALGO_ROWS = [
    ("**CPT（预训练）**", "本期目标（**仅 ms-swift**）", "**不做**（用户已定）", NO, TODO,
     "native 目标算法只有 SFT 与 GRASPO；CPT 配置层当前无法表达（`train_method` 只有 `graspo`/`sft`，"
     "`src/graspo/core/schema.py:410`），ms-swift 侧需新通道 + 新数据契约。"
     "**模式 = LoRA 与全量均支持**（用户已定，见 §2.2）"),
    ("**SFT（监督微调）**", "本期目标（两后端）", "**目标**", IMPLEMENTED, PARTIAL,
     "native 侧有完整 SFT 路径；ms-swift 侧入口为进程内调 `sft_main`，且 SFT 后端注册表未登记进打包元数据"
     "（生产环境走开发回退表）"),
    ("**GRASPO（RL）**", "本期目标（两后端）", "**目标**", PARTIAL, PARTIAL,
     "两后端均有入口；native 侧多卡实测曾失败（RL + PP 出现过进程级崩溃），故只能算有入口未验证"),
    ("**OPD（on-policy distillation）**", "本期目标（**仅 ms-swift**）", "**不做**（用户已定）", NO, TODO,
     "当前仅契约预留（ARD 契约提及 OPD 阶段由教师现场给出），零消费端。"
     "**模式 = LoRA 与全量均支持**（用户已定，见 §2.2）"),
]

# ------------------------------------------------------------------ §2.2 模式轴
MODE_ROWS = [
    ("**LoRA**", "本期目标：两后端 × 两模型 × **四算法**（含 CPT / OPD）", IMPLEMENTED, IMPLEMENTED,
     "两后端唯一当前可用的模式"),
    ("**全量（全参）**", "本期目标：**仅 9B**；两后端 × **四算法**（含 CPT / OPD）", TODO, TODO,
     "**27B × 全量 = 不做**（用户已定：卡数不够），见 §2.7 N-5，**对四算法一律适用**；"
     "native：基座权重硬冻结（按名白名单只放行 `lora_` 参数）且配置层无全参开关；"
     "ms-swift：`--tuner_type` 写死 `lora`（`src/graspo/flow/msswift/_config_mapping.py:382`）"),
]

# ------------------------------------------------------------------ §2.3 后端轴
BACKEND_ROWS = [
    ("**native**", "**并行 = DP / TP / PP / SP；算法 = SFT / GRASPO；须支持多模态**", IMPLEMENTED,
     "用户已定：native 不做 CPT / OPD，不做 CP，不做 l2k / ZeRO / FSDP2 / device_map / offload 等手段。"
     "**SP 保留实现**（与 TP 同进程组，须 `TP ≥ 2`，不额外占卡，是 native 侧唯一的序列维分片手段）"),
    ("**ms-swift**", "本期目标（主力路径，须支持多模态）", IMPLEMENTED,
     "**对外只有这一档**；引擎 / 路径选择是 ms-swift 后端的**内部实现策略**，不暴露给用户做选择。"
     "当前可用：DP / SP（`sequence_parallel_size`）/ ZeRO-0~3+offload / FSDP2 / device_map / packing / "
     "padding_free / l2k / FA2 / liger；TP / PP / CP 需依托其 Megatron 引擎接通（见 §2.4 与 §2.8）"),
]

# ------------------------------------------------- §2.4 五层结构（方法视角；ms-swift 合并为一列）
# 每层 = (层标题, 层说明, 层内硬约束, [行…])
# 每行 = (项, 作用+备注, native 目标, native 当前, ms-swift 目标, ms-swift 当前, 对应配置项)
LAYERS = [
    ("**层 1 · 并行维度**（世界大小如何切分）",
     "决定「卡怎么分」：把数据、权重、层、序列切到多卡。**这一层决定世界大小的分解方式**，是其余各层的前提。",
     "**native**：`TP × PP × DP = 卡数`（源码校验：`expected_world_size = dp_size * tp_size * pp_size`，"
     "不匹配即抛错；SP 不占卡，与 TP 同进程组，须 `TP ≥ 2`）。源码位置见附录 A。"
     "**ms-swift**：SP 是独立轴，`sp_world_size = gcd(num_kv_heads, world_size)` 且须整除 `num_kv_heads`，"
     "`DP 度 = world_size / sp_world_size` —— 与 native 不是同一条恒等式。",
     [
         ("**DP**（数据并行）", "把数据切到多卡；复制模型。native 为自研 all_reduce 实现（非 DDP）",
          "**目标**", IMPLEMENTED, "**目标**", IMPLEMENTED, "`dp_size`"),
         ("**TP**（张量并行）", "把权重的 hidden / 头 / FFN 中间维切到多卡。"
          "native 须整除注意力头数且 `TP ≤ num_kv_heads`",
          "**目标**", IMPLEMENTED, "**目标**（依托 Megatron 引擎）", FIELD,
          "`tp_size`｜Megatron `tensor_model_parallel_size`"),
         ("**PP**（流水线并行）", "把层切到多卡，多段流水。native 当前仅 `1F1B` 一种调度且多卡实测曾失败",
          "**目标**", PARTIAL, "**目标**（依托 Megatron 引擎）", FIELD,
          "`pp_size`｜`pp_scheduler`｜`pp_micro_batch_size`｜"
          "`pp_max_inflight_microbatches`｜Megatron `pipeline_model_parallel_size`"),
         ("**SP**（序列并行）", "**沿序列维**切激活，长上下文的关键手段之一。"
          "**注意：本仓库有【三套】SP 机制，源码注释原文警告「是两套机制，勿混用」**"
          "——native 布尔开关、ms-swift 标准路径整数度、Megatron 段布尔开关",
          "**目标**", IMPLEMENTED, "**目标**",
          "② `✅ 已实现可跑`（标准路径）｜③ `⚠️ 引擎字段存在，通路未接通`（Megatron，见 G-5）",
          "① native `sequence_parallel`（布尔；与 TP 同进程组）｜"
          "② ms-swift 标准路径 `sequence_parallel_size`（整数度；`sp_world_size = "
          "gcd(num_kv_heads, world_size)`）｜"
          "③ **Megatron 引擎 `sequence_parallel`（布尔，仅当 TP>1 生效；源码注释原文："
          "「与标准路径的 `msswift.sequence_parallel_size` 是两套机制，勿混用」）**"),
         ("**CP**（上下文并行）", "**沿序列维**切分，长上下文的关键手段之一；"
          "**CP 只切序列、不切权重**；**唯一能覆盖 CP 的是 Megatron 引擎**（用户已定：native 不做 CP）",
          "**不做**（用户已定，见 §2.7 N-7）", "⛔ 硬禁用",
          "**目标**（须依托 Megatron 引擎接通）", FIELD, "Megatron `context_parallel_size`"),
         ("**device_map**（层切分）", "按层把模型切到多卡。**它决定卡上放什么，属并行维度**（原列在层 2，"
          "与层 2「不改变卡分配」的定义冲突，已移入层 1）；**同时是 Z3 的互斥约束参与方**",
          "**不做**（用户已定）", "⛔ 未实现", FALLBACK, IMPLEMENTED, "`device_map`"),
         ("**AutoTP**（自动张量并行）", "自动张量并行（上游仅全参）。**它是对 TP 的自动化，属并行维度**"
          "（原列在层 2，已移入层 1）", "**不做**（用户已定）", "⛔ 未实现",
          "**待定**（全量入口打通后评估）", FIELD, "`deepspeed_autotp_size`"),
     ]),
    ("**层 2 · 分片策略**（权重 / 梯度 / 优化器状态如何摊）",
     "在**同一世界大小内**把常驻状态再摊开：降低权重、梯度、优化器态的每卡常驻占用。**这不改变卡的分配方式**。"
     "**路线**：优先用 Megatron 引擎的分布式优化器与参数 / 梯度分片；"
     "DeepSpeed（ZeRO 预设 + CPU offload）与 FSDP2 / device_map 作为补充，仅在 Megatron 覆盖不到或成本过高时启用。"
     "**⚠️「Megatron 优先」的成立前提（声明层 ≠ 事实层）**：该优先级对**目标两模型**目前只有**注册层证据**"
     "（官方支持表标 ✔、注册表在外部包 `mcore_bridge`），**运行时从未实测**——既有 Megatron 实测全部跑在 "
     "`Qwen3-8B`（非混合注意力、非多模态）上。因此本层各项在 Megatron 引擎下的可达性，"
     "**须以 §2.8 G-12 的 1-iter 冒烟结果为准**；未过冒烟前不得按「已验证」使用。",
     "`deepspeed` ✗ `fsdp`、`deepspeed` ✗ `device_map`、`fsdp` ✗ `device_map`（三者互斥）；"
     "ZeRO++ 须同时给 `deepspeed`；`deepspeed_autotp_size` 仅全参可用。"
     "**Megatron 路径三条硬约束**（上游源码）：① **LoRA 必须 `bridge_backend=mcore-bridge`**"
     "（megatron-bridge 不支持 LoRA，非 full 时直接报错）；② `language_model_only=True` 时**禁用 `lora_llm`**"
     "（只能 `lora`）；③ `freeze_vit` 默认 True，且其冻结逻辑**仅 `tuner_type='full'` 生效**。",
     [
         ("**优化器态 / 梯度分片**（ZeRO-1/2 等价物）", "把优化器态与梯度摊到各卡。"
          "**Megatron 侧的分布式优化器是其原生能力**；DeepSpeed ZeRO-1/2 为补充",
          "**不做**（用户已定）", "⛔ 架构硬禁用",
          "**目标**（优先 Megatron 分布式优化器；**前提见本层「Megatron 优先」说明——未过 §2.8 G-12 冒烟前为未验证态**）",
          IMPLEMENTED,
          "Megatron `use_distributed_optimizer`｜`deepspeed`"),
         ("**权重分片**（ZeRO-3 等价物）", "把权重也摊到各卡。"
          "**Megatron-FSDP 的分片阶段由 `data_parallel_sharding_strategy` 选择**"
          "（`no_shard` / `optim` / `optim_grads` / `optim_grads_params`）",
          "**不做**（用户已定）", "⛔ 架构硬禁用", "**目标**（Megatron-FSDP / ZeRO-3）", IMPLEMENTED,
          "Megatron `use_megatron_fsdp`｜"
          "**`data_parallel_sharding_strategy`（Megatron-FSDP 的 ZeRO 阶段选择器）**｜"
          "`strict_fsdp_dtensor_load`｜`deepspeed`"),
         ("**CPU offload**", "把优化器态 / 权重换出到 CPU 内存",
          "**不做**（用户已定）", "⛔ 架构硬禁用", "**目标**（ZeRO-3 offload 预设）", IMPLEMENTED,
          "`deepspeed` 预设｜`zero_hpz_partition_size`（ZeRO++）"),
         ("**FSDP2**", "全分片数据并行（v2）。**补充路径**，Megatron 覆盖不到时启用",
          "**不做**（用户已定）", "⛔ 未实现", FALLBACK, IMPLEMENTED, "`fsdp`"),
     ]),
    ("**层 3 · 激活与 logits 削峰**（不改卡数分配，只降峰值）",
     "**不改世界的切分方式**，只把训练过程中的峰值显存压低（激活重算、少算 logits、换更省显存的注意力实现）。",
     "**l2k 与 `SP>1` 的关系（按源码重写，不是\u201c上游硬互斥\u201d）**：① **默认推导**——`get_use_logits_to_keep` "
     "使 `SP>1` 时 l2k **默认**为 False（显式设 True 时该函数不 raise）；② **运行时硬拦**——真启用 l2k 且 `SP>1` 时，"
     "`prepare_logits_to_keep()` 抛 `NotImplementedError`；"
     "③ **多模态下 l2k 被强制关闭**（非 transformers_5）。"
     "另：`packing ⇒ padding_free ⇒ 强制 FA2`（见层 4）。源码位置见附录 A。",
     [
         ("**GC / 激活重算**", "用激活重算换显存。native 默认开启。"
          "**Megatron 引擎侧用 `recompute_granularity` / `recompute_method` / `recompute_num_layers`，"
          "而本仓库配置段【查无任何 `recompute_*` 字段】⇒ 字段缺失，待映射**（已计入 §2.8 G-7）",
          "**目标**", IMPLEMENTED, "**目标（字段缺失，待映射）**", "⚠️ 字段缺失，待映射",
          "native `gradient_checkpointing`｜"
          "**Megatron `recompute_granularity` / `recompute_method` / `recompute_num_layers`"
          "（本仓库 `schema.py` 无对应字段 ⇒ 待新增映射）**"),
         ("**l2k**（`use_logits_to_keep`）", "只对少量 token 计算 `lm_head`，省 logits 及其梯度。"
          "**与 `SP>1` 的关系（两处引用，本表已按源码重写）**："
          "① **默认推导**——`get_use_logits_to_keep("
          "self.template.sequence_parallel_size == 1)` 使 `SP>1` 时**默认**取 False（显式设为 True 时该函数不 raise）；"
          "② **运行时硬拦**——一旦真启用 l2k 且 `SP>1`，`prepare_logits_to_keep()` 抛 `NotImplementedError`。"
          "**另：多模态模型在非 transformers_5 下 l2k 被强制置 False**",
          "**不做**（用户已定）", "⛔ 未实现", "**目标**", IMPLEMENTED,
          "`use_logits_to_keep`"),
         ("**FA2**（flash attention）", "更省显存的注意力实现；SP / packing 的硬前置。"
          "native 侧为可选实现，未在 native 上验证",
          "**不做**（用户已定）", PARTIAL, "**目标**", IMPLEMENTED,
          "`attn_implementation`（native）｜`attn_impl`（ms-swift）"),
         ("**liger kernel**", "融合算子，降低中间激活", "**不做**（用户已定）", "⛔ 未实现", "**目标**",
          IMPLEMENTED, "`use_liger_kernel`"),
     ]),
    ("**层 4 · 批次与序列组织**（一次算多少、样本怎么拼）",
     "决定「一次算多少」：micro-batch、梯度累积决定一次优化步的样本量；packing 决定一条序列里塞多少样本。"
     "**这一层直接影响吞吐与有效序列长度**。",
     "`packing ⇒ padding_free ⇒ 强制 FA2`；`padding_free` 单独开时也要求 flash attention 系实现。",
     [
         ("**micro-batch size**", "单次前向的样本数", "**目标**", IMPLEMENTED, "**目标**", IMPLEMENTED,
          "`micro_batch_size`（native）｜`per_device_train_batch_size`（ms-swift）"),
         ("**梯度累积（GA）**", "多步累积再更新，放大有效 batch；与 micro-batch 共同决定有效 batch",
          "**目标**", IMPLEMENTED, "**目标**", IMPLEMENTED,
          "`gradient_accumulation_micro_batches`"),
         ("**packing**", "把多条短样本拼进同一序列，减少 padding 浪费；开启后自动置 `padding_free`，"
          "进而强制 flash attention", "**不做**（用户已定）", "⛔ 未实现", "**目标**", IMPLEMENTED,
          "`packing`"),
         ("**padding_free**", "免 padding 的变长批处理", "**不做**（用户已定）", "⛔ 未实现", "**目标**",
          IMPLEMENTED, "`padding_free`"),
         ("**序列长度上限与截断边界**", "**控制上下文的入口**——决定一条样本最多算多少 token、"
          "超长如何截断；**它直接决定 §3 递增测试的可用长度范围**。"
          "native 侧在配置段**无显式长度上限字段**（由数据侧决定）；ms-swift 侧 `max_model_len` 即该入口",
          "**目标（字段待补）**", "⚠️ 字段缺失（配置段无显式上限字段）", "**目标**", IMPLEMENTED,
          "ms-swift `max_model_len`｜"
          "**native：配置段无对应字段（长度由数据侧控制）⇒ 若需在配置层显式控制，须新增字段**"),
         ("**attention mask / 序列窗口**", "控制注意力可见范围（因果掩码、窗口、padding 掩码）；"
          "**本仓库两后端的配置段均未暴露该入口**（由框架按模型结构自动决定）",
          "**无对应配置项**", "—", "**无对应配置项**", "—", "—（两后端均无配置项，记为无映射）"),
     ]),
    ("**层 5 · 精度与拓扑**",
     "数值精度与跨机形态。**本期多数项已被用户排除**，列在此处是为了让配置设计者看到全貌。",
     "量化需 Hopper 及以上硬件与 `transformer_engine`；本仓库两后端均未暴露量化配置。",
     [
         ("**混合精度**（bf16）", "训练数值精度；本期口径为 bf16", "**目标**", IMPLEMENTED, "**目标**",
          IMPLEMENTED, "`torch_dtype`"),
         ("**量化**（QLoRA / 4bit / FP8 / FP4）", "低比特权重与计算", "**不做**（§2.7 N-1）", "—",
          "**不做**（§2.7 N-1）", "—", "—"),
         ("**多机**（`nnodes > 1`）", "跨节点扩展", "**不做**（§2.7 N-3）", "⛔ 未实现",
          "**不做**（§2.7 N-3）", "—", "`nnodes`"),
     ]),
]

# ------------------------------------------- 附录 A：§2.4 字段 / 标识 → 源码位置
# 每条 = (字段 / 标识, 文件相对路径:行号, §2.4 出现位置)
# 出处：从 §2.4 正文逐处迁出（正文只保留字段名；行号属实现细节，见宪法 §1.6）。
FIELD_SOURCE_INDEX = [
    ("`TP × PP × DP = 卡数`（world size 恒等式校验）", "src/graspo/flow/parallel/state.py:99-103",
     "层 1 · 层内硬约束"),
    ("SP 与 TP 同进程组（须 `TP ≥ 2`）", "flow/runtime.py:384-385", "层 1 · 层内硬约束"),
    ("`dp_size`", "schema.py:214", "层 1 · DP 行"),
    ("`tp_size`", "schema.py:213", "层 1 · TP 行"),
    ("Megatron `tensor_model_parallel_size`", "schema.py:277", "层 1 · TP 行"),
    ("`pp_size`", "schema.py:215", "层 1 · PP 行"),
    ("`pp_scheduler`", "schema.py:241", "层 1 · PP 行"),
    ("`pp_micro_batch_size`", "schema.py:225", "层 1 · PP 行"),
    ("`pp_max_inflight_microbatches`", "schema.py:236", "层 1 · PP 行"),
    ("Megatron `pipeline_model_parallel_size`", "schema.py:280", "层 1 · PP 行"),
    ("三套 SP 机制注释告警（作用列）", "schema.py:287-288", "层 1 · SP 行"),
    ("native `sequence_parallel`（布尔）", "schema.py:224", "层 1 · SP 行"),
    ("ms-swift 标准路径 `sequence_parallel_size`（整数度）", "schema.py:344", "层 1 · SP 行"),
    ("Megatron 引擎 `sequence_parallel`（布尔）", "schema.py:288", "层 1 · SP 行"),
    ("三套 SP 机制源码注释原文", "schema.py:287", "层 1 · SP 行"),
    ("Megatron `context_parallel_size`", "schema.py:290", "层 1 · CP 行"),
    ("`device_map`", "schema.py:333", "层 1 · device_map 行"),
    ("`deepspeed_autotp_size`", "schema.py:340", "层 1 · AutoTP 行"),
    ("Megatron `use_distributed_optimizer`", "schema.py:269", "层 2 · 优化器态 / 梯度分片行"),
    ("`deepspeed`", "schema.py:336", "层 2 · 优化器态 / 梯度分片行"),
    ("Megatron `use_megatron_fsdp`", "schema.py:271", "层 2 · 权重分片行"),
    ("`data_parallel_sharding_strategy`（Megatron-FSDP 的 ZeRO 阶段选择器）", "schema.py:272-274",
     "层 2 · 权重分片行"),
    ("`strict_fsdp_dtensor_load`", "schema.py:275", "层 2 · 权重分片行"),
    ("`deepspeed`", "schema.py:336", "层 2 · 权重分片行"),
    ("`deepspeed`（预设）", "schema.py:336", "层 2 · CPU offload 行"),
    ("`zero_hpz_partition_size`（ZeRO++）", "schema.py:338", "层 2 · CPU offload 行"),
    ("`fsdp`", "schema.py:342", "层 2 · FSDP2 行"),
    ("l2k 默认推导 `get_use_logits_to_keep`", "trainers/mixin.py:211-223", "层 3 · 层内硬约束"),
    ("l2k 运行时硬拦 `prepare_logits_to_keep`", "trainers/mixin.py:1261-1263", "层 3 · 层内硬约束"),
    ("多模态下 l2k 被强制关闭（非 transformers_5）", "trainers/mixin.py:215-216", "层 3 · 层内硬约束"),
    ("native `gradient_checkpointing`", "schema.py:65", "层 3 · GC / 激活重算行"),
    ("l2k 默认推导 `get_use_logits_to_keep`", "trainers/mixin.py:211-223", "层 3 · l2k 行"),
    ("l2k 运行时硬拦 `prepare_logits_to_keep`", "trainers/mixin.py:1261-1263", "层 3 · l2k 行"),
    ("多模态下 l2k 被强制置 False", "trainers/mixin.py:215-216", "层 3 · l2k 行"),
    ("`use_logits_to_keep`", "schema.py:356", "层 3 · l2k 行"),
    ("`attn_implementation`（native）", "schema.py:64", "层 3 · FA2 行"),
    ("`attn_impl`（ms-swift）", "schema.py:350", "层 3 · FA2 行"),
    ("`use_liger_kernel`", "schema.py:351", "层 3 · liger kernel 行"),
    ("`micro_batch_size`（native）", "schema.py:229", "层 4 · micro-batch size 行"),
    ("`per_device_train_batch_size`（ms-swift）", "schema.py:369", "层 4 · micro-batch size 行"),
    ("`gradient_accumulation_micro_batches`", "schema.py:109", "层 4 · 梯度累积（GA）行"),
    ("`packing`", "schema.py:348", "层 4 · packing 行"),
    ("`padding_free`", "schema.py:349", "层 4 · padding_free 行"),
    ("ms-swift `max_model_len`", "schema.py:347", "层 4 · 序列长度上限与截断边界行"),
    ("`torch_dtype`", "schema.py:63", "层 5 · 混合精度（bf16）行"),
    ("`nnodes`", "schema.py:328", "层 5 · 多机（`nnodes > 1`）行"),
]

# ------------------------------------------------------- §2.6 模型与硬件
MODEL_ROWS = [
    ("模型", "`Qwen3.5-9B`、`Qwen3.8-27B`",
     "两者均为 `model_type=qwen3_5`、`architectures=['Qwen3_5ForConditionalGeneration']` 的"
     "**原生多模态 VLM（vision + video）**；权重含**整座视觉塔**（333 个 `model.visual.*` 参数，两模型同数）。"
     "它们**非 MoE（属 dense）**，但**不是普通 dense Transformer**——文本侧为**混合线性注意力（GDN）+ MTP**（见下行）"),
    ("**架构**（混合线性注意力 GDN + MTP）",
     "`text_config.layer_types` 在 `linear_attention` / `full_attention` 间**交替**"
     "（模式形如 `[L,L,L,F,…]`），`full_attention_interval=4`；含 `mamba_ssm_dtype`、"
     "`linear_conv_kernel_dim`、`linear_key/value_head_dim`、`attn_output_gate=True` 等线性注意力字段；"
     "`mtp_num_hidden_layers=1`（MTP / NextN 预测层）",
     "**该架构即官方所称 GatedDeltaNet（GDN）混合线性注意力**。"
     "**含义**：它**不能按普通全注意力 Transformer 的显存 / 并行假设外推**——"
     "线性注意力层与全注意力层交替，且另有 MTP 层与视觉塔"),
    ("**关键规格**",
     "**9B** = 32 层 / hidden 4096 / 16 注意力头 / 4 KV 头 / `head_dim` 256 / 词表 248320 / FFN 12888；"
     "**27B** = 64 层 / hidden 5120 / 24 注意力头 / 4 KV 头 / `head_dim` 256 / 词表 248320 / FFN 17408",
     "以上字段均在 `text_config` 内；两模型 `tie_word_embeddings=False`、`dtype=bfloat16`、"
     "`max_position_embeddings=262144`"),
    ("**多模态（视觉）**", "**所有训练都要支持多模态**（要训练多模态数据集）", "**这是目标能力**，不是可选项；"
     "当前实现状态：现有运行均为「视觉 deferred」的纯文本路径 ⇒ 视觉通路要补（§2.8 G-8）。"
     "**Megatron 引擎侧的多模态支持注册层已核实存在（运行时未验证）**（§2.8 G-11）；接通时须落实三条硬约束（§2.4 层 2 层内硬约束）："
     "LoRA 须用 `mcore-bridge`、`language_model_only` 禁 `lora_llm`、`freeze_vit` 仅对全参生效"),
    ("实物状态", "两模型**已就位**（在测试节点上）",
     "**分片数与总大小已核实**：9B = 4 分片 / `metadata.total_size` 19 306 216 416 B；"
     "27B = 18 分片 / `metadata.total_size` 55 562 855 904 B"),
    ("⚠️ **结论性提醒**",
     "**既有的 CP / TP / PP 实战经验来自 `Qwen3-8B`**（`model_type=qwen3`，非混合注意力、非多模态）",
     "⇒ **不能直接外推**到这两个模型：目标模型是「混合线性注意力（GDN）+ 视觉塔 + MTP」三合一，"
     "任一环节未打通都会翻转并行可行性结论（见 §2.8 G-11 / G-12）"),
    ("卡数", "**1 / 2 / 4**（8 卡 = 不做，见 §2.7 N-6）", "单节点 8 × A800-80G 机器上只使用 1/2/4 卡三档"),
    ("上下文长度", "**由实测得到，不作预设档位**", "从基准长度起递增加长直到失败（§3）"),
    ("测试数据量", "**SFT ≥ 100 条；RL ≥ 20 条**；每档至少 1 个完整 epoch 且 ≥ 5 个 optimizer step", "见 §3 测试方法"),
    ("评测集", "**ELAM V5 的 test 集**（与训练同源、不重叠）", "SFT / GRASPO 的效果达标在同一评测集上对比初始模型与训练后模型（§3.4）"),
]

# ------------------------------------------------------- §2.7 负面清单
NEG_ROWS = [
    ("N-1", "**量化**（QLoRA / 4bit / 8bit / GPTQ / AWQ / FP8 / FP4 参数分片）", "用户明确排除",
     "用户显式改变该拍板"),
    ("N-2", "**MoE 全部**（EP / 专家并行 / `expert_*`）", "不考虑 MoE 模型；本期两模型均为 dense",
     "引入 MoE 模型时重新纳入"),
    ("N-3", "**多机训练**（`nnodes > 1`）", "本期单机多卡", "用户拍板纳入，且单机多卡全绿"),
    ("N-4", "**生产占用卡上的任何测试**", "生产任务常驻", "生产负载迁移"),
    ("N-5", "**27B × 全量**（任何卡数、任何后端、**任何算法**）", "用户已定：卡数不够（全量只做 9B）",
     "用户显式纳入，且卡数上限提升到足以容纳 27B 全参训练"),
    ("N-6", "**8 卡档**（全部组合）", "用户已定：本期只用 1 / 2 / 4 卡", "用户显式提升卡数上限"),
    ("N-7", "**CP @ native**（native 侧上下文并行）", "用户已定：native 不加入 CP",
     "用户显式要求 native 支持 CP（架构级解禁 + 新实现）"),
    ("N-8", "**CPT @ native**", "用户已定：CPT 只经 ms-swift", "用户显式要求 native 支持 CPT"),
    ("N-9", "**OPD @ native**", "用户已定：OPD 只经 ms-swift", "用户显式要求 native 支持 OPD"),
]

# ------------------------------------------------------- §2.8 实现缺口
GAP_ROWS = [
    ("G-1", "**全量训练入口（native，9B）**——基座权重硬冻结 + 配置层无全参开关",
     "9B × 全量 × native 的档位", "⛔ 未实现"),
    ("G-2", "**全量训练入口（ms-swift，9B）**——`--tuner_type` 写死 `lora`",
     "9B × 全量 × ms-swift 的档位", "⛔ 未实现"),
    ("G-3", "**CPT 通道（ms-swift）**——配置层无法表达预训练；需新数据契约（纯文本 vs 结构化目标）",
     "全部 CPT 档（9 档）", "⛔ 未实现"),
    ("G-4", "**OPD 实现（ms-swift）**——仅契约预留，零消费端", "全部 OPD 档（9 档）", "⛔ 未实现"),
    ("G-5", "**承载结构调整：接通 ms-swift 的 Megatron 引擎通路**——它是 TP / PP / **CP** / SP / "
     "分布式优化器的承载载体；当前引擎字段齐全但不进启动参数，且缺 Megatron 专属超参映射表。"
     "**接通时须同时落实三条硬约束**：① LoRA 必须 `bridge_backend=mcore-bridge`；"
     "② `language_model_only=True` 禁用 `lora_llm`；③ `freeze_vit` 冻结逻辑仅对 `tuner_type='full'` 生效",
     "所有需要 TP / PP / CP 的档位（含全部长文目标档）", "⚠️ 引擎字段存在，通路未接通"),
    ("G-6", "**CP（上下文并行，ms-swift）**——`context_parallel_size` 已有字段，须依托 Megatron 引擎接通；"
     "**CP 是长上下文的关键手段**且只切序列、不切权重；已核实**多模态下 CP 由 `input_embeds` 承载**"
     "（`megatron_lm_utils.py:869-870`）", "所有需要上下文分片的长文档位", "⚠️ 引擎字段存在，通路未接通"),
    ("G-7", "**Megatron 激活重算字段缺失**——Megatron 用 `recompute_granularity` / `recompute_method` / "
     "`recompute_num_layers` 控制激活重算，而本仓库 `schema.py` **查无任何 `recompute_*` 字段** ⇒ "
     "需新增映射；否则层 3 的 GC 在 Megatron 引擎下无配置入口",
     "所有经 Megatron 引擎的长文档位（GC 是本期的目标手段）", "⚠️ 字段缺失，待映射"),
    ("G-8", "**多模态（视觉）训练通路**——现有运行均为视觉 deferred 的纯文本路径；"
     "Megatron 引擎侧的多模态支持**已核实存在**（见 §2.8 G-11 的证据），但本仓库的两模型通路仍待打通",
     "全部目标档", "⚠️ 有入口未验证（视觉路径零证据）"),
    ("G-9", "**SFT 后端注册表未登记进打包元数据**——生产环境走开发回退表", "ms-swift 的 SFT 档",
     "⚠️ 有入口未验证"),
    ("G-10", "**RL 多卡稳定性**（native 侧 PP 多卡曾崩溃）", "native 的 GRASPO 多卡档", "⚠️ 有入口未验证"),
    ("G-11", "**Megatron 引擎支持多模态 / VLM（注册层已核实、运行时未验证）**——"
     "**注册层证据**：官方受支持模型表 `Support Megatron` 列对 `Qwen3.5-9B` / `Qwen3.8-27B` **均为 ✔**；"
     "注册表实现在**外部包 `mcore_bridge`**（ms-swift 经 `get_model_meta` 查询，返回 `None` 即报 not supported）；"
     "官方示例 `examples/megatron/multimodal/` 下有 VLM 脚本；`megatron_args.py` 的 `_init_multimodal_full()`；"
     "`megatron_lm_utils.py` 的「Multimodal models will handle CP in input_embeds」。"
     "**运行时事实**：**从未在这两个模型上跑过 Megatron**（既有 Megatron 实测只跑过 `Qwen3-8B`）"
     "⇒ 标为 **`⚠️ 官方声明支持，运行时未验证`**。"
     "**硬约束**：多模态**强制 `mcore-bridge`**（`megatron-bridge` 不支持多模态，会直接报错），"
     "且需额外依赖（`transformers>=5.0.0.dev`、`qwen_vl_utils`、`decord`）。"
     "**口径提醒**：官方对 **dense 模型推荐 transformers 后端**（Megatron 的推荐对象是 MoE）；"
     "且官方**无**「`qwen3_5` dense 9B/27B + Megatron」现成脚本——Megatron 侧 `qwen3_5` 现成脚本只有 "
     "2B/4B dense 与 35B-A3B **MoE**，9B 在官方脚本里只作 teacher 出现。"
     "**待确认**：线性注意力（GDN）的 CP 官方要求 `megatron-core` **main 分支**，而测试节点装的是 **release 版**。"
     "**备选回退路径（若 Megatron 走不通）**：ms-swift **标准路径 + SP（Ulysses / Ring，同样沿序列维分片）+ ZeRO-3**；"
     "native 侧用 **SP + TP / PP**。",
     "全部多模态目标档（与 G-5 叠加）", "⚠️ 官方声明支持，运行时未验证"),
    ("G-12", "**`qwen3_5` 两模型在 Megatron 下的运行时可行性未实测**——"
     "注册层已核实（§2.8 G-11），但缺一次真机验证 ⇒ **须先做 1-iter 冒烟**："
     "`train_iters=1` / `micro_batch_size=1`，让 `get_model_meta('qwen3_5')` 给出「非 None / 抛 not supported」"
     "的二值答案，并验证权重键映射（尤其 `model.visual.*` 与 `mtp.*`）。"
     "**冒烟通过前，Megatron 优先策略对目标两模型均为未验证态**；在它通过之前不宜并行投入 27B 的大规模实现",
     "全部需要 TP / PP / CP / Megatron 的目标档", "⚠️ 未实测（须先做 1-iter 冒烟）"),
    ("G-13", "**`lora.r` 无边界校验，LoRA 静默失效（fail-open 静默降级）**——"
     "`LoRAConfig`（`src/graspo/core/schema.py:41-53`）**没有 `r > 0` 校验**。"
     "**已证事实（源码）**：`src/graspo/flow/lora/lora_linear.py:123` 为 "
     "`lora_enabled = bool(lora_enabled and r > 0)` ⇒ `r <= 0` 时该层**不建立 LoRA**；"
     "同文件 `:132-134` 随即把 `lora_a` / `lora_b` **注册为 `None`**；"
     "而 `src/graspo/flow/adapters/models/qwen35_36/model.py:156`（Qwen3.5/3.8 侧）与 "
     "`src/graspo/flow/adapters/models/qwen3/model.py:100`（Qwen3-8B 侧）均为**无条件**的 "
     "`param.requires_grad = \"lora_\" in name` ⇒ **没有任何参数名含 `lora_`** ⇒ "
     "**native 侧零可训练参数**（训练空转或报错）；配置边界**不报错、静默失效**。"
     "**未验证（不作结论）**：ms-swift 侧 `--lora_rank 0` 的行为尚未验证。"
     "**本期要补**：在配置边界加校验——**`r <= 0` 直接报错（fail-closed）**，不得静默降级",
     "native 侧全部 LoRA 档（12 档 = 2 模型 × SFT / GRASPO × 1 / 2 / 4 卡，已证零可训练参数）；"
     "ms-swift 侧 `--lora_rank 0` 的行为未验证，暂不计入影响范围", "⛔ 未实现"),
]

# ------------------------------------------------------- §6 开放项（应为 0）
QUESTIONS = []


def ledger_rows():
    """有效目标组合（与已拍板范围一致）：

    ① CPT / OPD 只出 ms-swift；② 全量只出 9B（对四算法一律适用）；
    ③ native 的算法仅 SFT / GRASPO。
    计数式 = 2 模型 × 2 模式 × 2 后端 × 3 卡数 × 2 算法(SFT/GRASPO)
           + 2 模型 × 2 模式 × 3 卡数 × 2 算法(CPT/OPD，仅 ms-swift)
           − 27B × 全量（SFT/GRASPO 的 2 后端×3 卡 + CPT/OPD 的 1 后端×3 卡）
    """
    rows = []
    n = 0
    for algo in ALGOS:
        backends = BACKENDS if algo in ("SFT", "GRASPO") else ("ms-swift",)
        for model in MODELS:
            for mode in MODES:
                if mode == "全量" and model == "27B":
                    continue
                for backend in backends:
                    for cards in CARDS:
                        n += 1
                        rows.append(dict(id="T{:03d}".format(n), algo=algo, model=model,
                                         mode=mode, backend=backend, cards=cards))
    return rows


def ledger_table():
    head = ("| 条件档编号 | 模型 | 算法 | 模式 | 后端 | 卡数 | 最大可行上下文（实测） | "
            "实测每卡峰值显存(GiB) | 实测配方 | 实测状态 | 实测日期 | 产物 | 备注 |")
    sep = "|---|---|---|---|---|:--:|---|---|---|---|---|---|---|"
    out = []
    rows = ledger_rows()
    for algo in ALGOS:
        sub = [r for r in rows if r["algo"] == algo]
        out.append("")
        out.append("#### {}（{} 档）".format(algo, len(sub)))
        out.append("")
        out.append(head)
        out.append(sep)
        for r in sub:
            note = ""
            if r["mode"] == "全量":
                note = "**待补齐**：全量入口（§2.8 G-{}）".format(1 if r["backend"] == "native" else 2)
            out.append("| {} | {} | {} | {} | {} | {} | — 未测 | — 未测 | — 未测 | — 未测 | — | — | {} |".format(
                r["id"], r["model"], r["algo"], r["mode"], r["backend"], r["cards"], note))
    return "\n".join(out)


def report():
    rows = ledger_rows()
    tot_sg = 2 * 2 * 2 * 3 * 2
    tot_co = 2 * 2 * 3 * 2
    excl = 2 * 2 * 3 + 2 * 3
    out = []
    out.append("=" * 72)
    out.append("功能矩阵结构性计数（不含任何估算）")
    out.append("=" * 72)
    out.append("目标档位总数 = {}".format(len(rows)))
    out.append("  计数式 = {}（SFT/GRASPO）+ {}（CPT/OPD，仅 ms-swift）− {}（27B × 全量）"
               " = {}（自数一致：{}）".format(tot_sg, tot_co, excl, tot_sg + tot_co - excl,
                                              tot_sg + tot_co - excl == len(rows)))
    out.append("  按算法：{}".format(" ｜ ".join(
        "{} {}".format(a, sum(1 for r in rows if r["algo"] == a)) for a in ALGOS)))
    out.append("  按模型：{}".format(" ｜ ".join(
        "{} {}".format(m, sum(1 for r in rows if r["model"] == m)) for m in MODELS)))
    out.append("  按后端：{}".format(" ｜ ".join(
        "{} {}".format(b, sum(1 for r in rows if r["backend"] == b)) for b in BACKENDS)))
    out.append("  按模式：{}".format(" ｜ ".join(
        "{} {}".format(m, sum(1 for r in rows if r["mode"] == m)) for m in MODES)))
    out.append("")
    out.append("实测列非空行数 = 0 ｜ 实测状态取值集合 = ['— 未测']")
    out.append("")
    out.append("§2.4 分层：{} 层，共 {} 行".format(len(LAYERS), sum(len(l[3]) for l in LAYERS)))
    for i, l in enumerate(LAYERS, 1):
        out.append("    层 {}：{} 行".format(i, len(l[3])))
    out.append("")
    out.append("负面清单条数 = {}".format(len(NEG_ROWS)))
    out.append("实现缺口条数 = {}".format(len(GAP_ROWS)))
    out.append("开放项条数 = {}".format(len(QUESTIONS)))
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
