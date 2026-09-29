"""A1–A6 验收判据与失败分类 —— 纯计算层，零设施依赖。

职责：把一次训练运行抽出的**证据**（退出码、日志、步数、权重变化、产物清单、
loss / grad_norm 序列、双跑对照）判成六条验收判据 A1–A6 的通过与否，并给出
失败分类与「是否计入最大可行上下文」的结论。本模块不读文件、不调子进程，
证据抽取在 ``scripts/collect_results.py``。

**硬要求（用户拍板）**：只有**真 OOM** 才计入"最大可行上下文"；其余任何失败
一律记为 `❌ 失败` + 失败类型，修复后重测。否则框架 bug 会被固化成"能力上限"。

判据：
- **A1** 进程成功（exit=0）；
- **A2** 训练步真推进且权重真变化（LoRA / 全参两套判据）；
- **A3** checkpoint 可被重新加载；
- **A4** 同 config 同 seed 双跑一致；
- **A5** 四件套产物落盘；
- **A6** 数值健康（loss 与 grad_norm 全程 finite，NaN/Inf 即不通过；loss 首末走向
  只作**记录项** ``loss_trend``，不作阻断——见 :data:`A6_LOSS_TREND_BLOCKS`）。

证据缺失一律判**不通过**（fail-closed）——不能因为"读不到"就默认通过。
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ── 常量 ────────────────────────────────────────────────────────────────────

#: ── rank_metrics 旁路的 phase 名注册表（**全仓唯一真相源**，§1.4） ──────────
#:
#: 背景（2026-09-20 实测伪否）：采集层曾把"哪条 phase 承载逐步权威指标"写死成
#: **一个**字面量 ``pipeline_sft_train_batch_after``。训练侧却有**两条** SFT 路径，
#: 各自 emit 不同的 phase：pipeline（pp_size>1）emit 带 pipeline 前缀的那个，
#: 普通/单卡路径 emit 不带的那个。于是 T010（9B·SFT·LoRA·native·**1 卡**，
#: 100/100 步、exit=0、loss/grad_norm 全程 finite）的 100 条逐步指标被**整批静默丢弃**，
#: 判成"A2 缺少 optimizer step 证据 / A6 缺少 loss 序列证据"——采集侧伪否。
#:
#: 为什么注册表放在**纯计算层**（本模块，零设施依赖）：采集脚本
#: ``scripts/collect_results.py`` 已经按文件路径加载本模块（它不能走
#: ``import graspo``，那会拉入 torch/pydantic）。把 phase 名单放在这里，训练侧与
#: 采集侧才能共享同一份定义，而**不必**在采集侧再抄一遍字面量。
#:
#: 每个名字的**出处行号**（本仓库当前 HEAD；改动 emit 处时必须同步本表）：
#:
#: .. code-block:: text
#:
#:     ── 训练指标（payload 带 ``metrics``，是本注册表存在的理由）──────────────
#:     src/graspo/flow/adapters/models/qwen35_36/training_sft.py:753   SFT  · PP 路径（pp_size>1）
#:     src/graspo/flow/adapters/models/qwen35_36/training_sft.py:478   SFT  · 普通路径（pp_size==1，
#:                                                                    含全部单卡/DP/TP 档）
#:     src/graspo/flow/adapters/models/qwen35_36/training.py:440       RL/GRASPO · PP 路径
#:     src/graspo/flow/adapters/models/qwen35_36/training.py:204       RL/GRASPO · 普通路径
#:     src/graspo/flow/adapters/models/qwen3/adapter.py:486            qwen3 后端 · 普通路径
#:
#:     ── 诊断事件（**不**带 metrics，不进注册表，登记以免被误加）──────────────
#:     src/graspo/flow/adapters/transformer_adapter.py:271             setup_after
#:     src/graspo/flow/adapters/transformer_adapter.py:441             checkpoint_after
#:     src/graspo/flow/adapters/transformer_adapter.py:591             checkpoint_loaded
#:     src/graspo/flow/adapters/models/qwen35_36/training_sft.py:353   train_before_empty_cache
#:     src/graspo/flow/adapters/models/qwen35_36/training.py:58        train_before_empty_cache
#:     src/graspo/flow/adapters/models/qwen3/adapter.py:394            train_before_empty_cache
#:     src/graspo/flow/adapters/models/qwen35_36/logprobs.py:57        logprob_after
#:     src/graspo/flow/adapters/models/qwen35_36/logprobs.py:125       pipeline_logprob_after
#:     src/graspo/flow/adapters/models/qwen3/adapter.py:508            logprob_after
#:
#: ★ 加新路径时改的**只有本表**；采集侧不得再出现任何 phase 字面量（§2.2 显式即防呆）。

#: 承载**逐步权威指标**的 phase 名（payload 含 ``metrics``：全局聚合 + 逐 rank 明细）。
#: 采集侧用它决定"哪些 rank_metrics 行是 loss/grad_norm/optimizer_steps 的证据源"。
#: 判断依据是**语义**（"这一步训练的权威读数"），不是后端标签，也不是卡数——
#: 判据语义因此与后端/并行度无关（与 :func:`find_output_dirs` 同一原则）。
STEP_METRICS_PHASES: frozenset[str] = frozenset(
    {
        "pipeline_sft_train_batch_after",
        "sft_train_batch_after",
        "pipeline_train_batch_after",
        "train_batch_after",
    }
)

#: **PP 专属**训练指标 phase 名。用于只读 PP 结构性字段（``pipeline_stage_timing`` /
#: ``placement_strategy`` 等）的展示层：这些键在非 PP 档里结构性不存在，所以这里的
#: 过滤是**语义正确**的，不是"漏认一个名字"。
#: 使用方：``src/graspo/cli/tools.py:_read_rank_summary``。
PIPELINE_TRAIN_METRICS_PHASE: str = "pipeline_train_batch_after"

#: 已知的**诊断类** phase 名（``_emit_rank_memory_event`` 的其余取值）：payload 只带
#: 显存快照或结构信息，**不带** ``metrics``。登记在此供自检/文档用——若某天诊断事件
#: 开始携带 metrics，采集侧会把它计入"含 metrics 的未登记 phase"并显式暴露（不静默）。
#: **例外（显式登记）**：``fail_closed`` 有意携带 ``metrics``——失败步必须留下一行可被
#: A7/取证链读到的读数（``skipped_nonfinite`` 等），否则"触发失败的那一步"永远没有读数
#: （T018 实测缺口：raise 早于 metrics 构造 ⇒ 崩溃步零行）。该行**不进** ``STEP_METRICS_PHASES``
#: （它不是成功步的逐步指标），因此采集侧会按既有规则把它计入 ``ignored_phase_records``
#: 并写明 phase 名——这正是"显式暴露而非静默"的期望行为。
DIAGNOSTIC_PHASES: frozenset[str] = frozenset(
    {
        "setup_after",
        "checkpoint_after",
        "checkpoint_loaded",
        "train_before_empty_cache",
        "logprob_after",
        "pipeline_logprob_after",
        # 每 rank 首步探针（只读旁路：首步 local loss + 首批 input_ids sha256）。
        # 唯一真相源是 ``graspo/flow/msswift/first_step_probe.py::PROBE_PHASE``；
        # 这里只能写字面量——采集侧按文件路径加载本模块（不能 import graspo），
        # 因此两个字面量由 ``tests/core/test_determinism.py`` 的同步测试守住。
        # 本行 payload **不带** ``metrics`` ⇒ 采集侧不会误当逐步指标（§2.2 显式）。
        "first_step_probe",
        # native PP 首步逐 rank 数值探针（R3，2026-09-23）：只读梯度/loss 读数，
        # **不带** ``metrics``（见上条同理）。唯一真相源是
        # ``graspo/flow/adapters/models/common/grad_probe.py`` 的调用点常量；
        # 同步由 ``tests/flow/trainer/test_grad_fail_closed.py`` 守住。
        "pp_numeric_probe",
        # 失败步诊断行（R1，2026-09-23）：**先落盘、再 raise**，见上方例外说明。
        "fail_closed",
    }
)

#: ── 「实测每卡峰值显存(GiB)」列的**取数口径**（**全仓唯一真相源**，§1.4） ────────
#:
#: 权威定义不在本仓库的代码里，而在 `.local/本期工程跟踪.md` **§9.1**「★ §7 各列的
#: 读取口径」表中 `实测每卡峰值显存(GiB)` 那一行的**原文**——**可核定位**：
#: `.local/本期工程跟踪.md:420`（表头 `★ §7 各列的读取口径` 在 `:415`）：
#:
#:     **二分口径（按 A §7 的「后端」列即可推导，不必逐格加标签）**：
#:     `native` 档 = 容器内 PyTorch allocator 的 rank0 `max_allocated`；
#:     `ms-swift` 档 = 该进程的 `max_memory_reserved`（同进程全部可见卡取 max）。
#:     ★ 同一后端内部口径完全一致；跨后端差异显式登记在台账
#:     `peak_memory_caliber` 字段且可由「后端」列单值推导（§1.4 / §2.2）。
#:     ★ `ms-swift` 档是**保守上界**（`reserved >= allocated`）。
#:     **其它口径一律不得混入本列**（OOM 报文的进程占用、宿主 `nvidia-smi` 采样）；**不可得时留空并显式标注，绝不静默回落**。
#:
#: 本常量 = 上述声明的 native 支路在**可执行代码里的唯一落点**；后端→口径的**唯一映射**
#: 见 :data:`PEAK_MEMORY_CALIBER_BY_BACKEND`（改口径 = 改那一处 + 同步
#: `docs/capability-matrix.html` 的峰值列口径（§3 字段说明；实测值见 §6 表）；
#: 两处之外不得再出现第三份口径定义）。
#:
#: 训练侧落点（allocator 值怎么来的）：
#:   `src/graspo/flow/parallel/tensor_utils.py:_cuda_memory_snapshot`
#:     → `max_allocated_mib = torch.cuda.max_memory_allocated(device) / 1024**2`
#:   `src/graspo/flow/adapters/transformer_adapter.py:_emit_rank_memory_event`
#:     → 追加一行 `{"event": "rank_memory", "memory": {...}}` 到 rank0 的
#:       `metrics/rank_metrics.rank_00000.jsonl`。
#: 采集侧落点（值怎么被取走）：`scripts/collect_results.py:_read_peak_memory`。
PEAK_MEMORY_CALIBER: str = "rank0_max_allocated"

#: allocator 口径**不可得**时，台账峰值列必须带上的**显式标注**（§2.2 显式即防呆）。
#: 为什么必须有：宿主采样摘要 `gpu/gpu_memory_summary.json` 在不产 `rank_metrics` 的
#: 后端（如 ms-swift）上**仍然存在**；若 `None` 被下游读成"没跑过"，就会诱发
#: "拿宿主采样把这一格补上"的静默混口径——那正是本字段要防住的缺陷（§9.1 明文禁止）。
PEAK_MEMORY_UNAVAILABLE: str = "未取得（allocator 口径不可得）"

#: 宿主 `nvidia-smi` 采样峰值在台账里的**独立字段名**：**另存，不进 §7 峰值列**
#: （§9.1 口径分离的机器可核形式）。它回答的是"这张卡实际用了多少" + 取证链锚点。
#: ★ 为什么它是**必需的诊断项而不是可选装饰**：228 是**共享机**，`task-j1-batch-msswift`
#: 实测 `T033/run2` 的 GPU2 = 27451 MiB 而 `max_peak_memory_gap_mib = 2388`（采样缺口）
#: ⇒ 真值应 ≈ 25063 MiB——**他人作业的显存被算到了本档头上**。宿主口径因此不仅
#: "口径不同"，在共享机上会**系统性高估**本档占用；用户拿 §7 那列判断"我的硬件够不够"
#: 会被带偏。allocator 口径按进程统计，天然不被同租户污染——这是口径分离的**正确性**
#: 理由，不只是形式理由。
HOST_SAMPLE_PEAK_FIELD: str = "host_sample_peak_gib"

#: 同一次宿主采样的**采样缺口**（MiB）：`gpu_memory_summary.json.max_peak_memory_gap_mib`。
#: 它把"这张卡的峰值可能被他人作业污染"这件事**显式**留在台账里（§2.2）——
#: 缺口 > 0 即说明该次采样期间卡上的占用有非本档成分，副读数须打折看。
HOST_SAMPLE_GAP_FIELD: str = "host_sample_peak_gap_mib"

#: **ms-swift 后端自报的显存峰值口径标识** —— `max_memory_reserved`，**不是** allocator 口径。
#:
#: 源码依据（`.local/refs/ms-swift-4.5.3/git-v4.5.3`，`HEAD=faed594`，`tag v4.5.3`）：
#:   `swift/trainers/patcher.py:27`   `state.max_memory = max(..., get_max_reserved_memory())`
#:   `swift/trainers/patcher.py:29`   `logs['memory(GiB)'] = round(state.max_memory, 2)`
#:   `swift/utils/torch_utils.py:413-419`
#:       `mems = [get_torch_device().max_memory_reserved(device=device) for device in devices]`
#:       `return max(mems) / 1024**3`
#:   `swift/megatron/callbacks/print.py:64`
#:       `memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)`
#:       `logs['memory(GiB)'] = round(memory, 2)`
#:
#: 三处**没有一处**调用 `max_memory_allocated()` ⇒ ms-swift 的 `memory(GiB)` 是
#: **reserved（缓存分配器已向 CUDA 申请并持有的高水位）**，与 §9.1 声明的
#: `rank0 max_allocated`（`.local/本期工程跟踪.md:420`）是**两个量**：
#: reserved ≥ allocated 恒成立（reserved 含 caching allocator 的空闲缓存块）。
#: 语言侧对照：ms-swift 自己在 `swift/rlhf_trainers/utils.py:423/424` 把
#: `memory_allocated()` / `memory_reserved()` 分别命名 —— 两个词在上游就是两个量。
#:
#: ⇒ **它只在 ms-swift 档上是 §7 峰值列的合法口径**（2026-09-21 指挥官裁定「方案 A：
#: 二分口径」）：ms-swift 后端结构性不产 `rank_metrics` 旁路 ⇒ `max_allocated` 不可得；
#: 若坚持单一口径，54 档里 36 个 ms-swift 档**永久填不出值**，§7 便无法回答用户的
#: "我的硬件够不够"。§9.1 已改为**二分口径**：同一**后端**内部口径一致，跨后端差异
#: **显式登记**（台账 `peak_memory_caliber`）且**可由「后端」列单值推导**（§1.4 / §2.2）。
#: ★ 它是**保守上界**：`reserved >= allocated` 恒成立 ⇒ 拿它判断"够不够"不会被低估。
#: 唯一映射（后端 → 口径）见 :data:`PEAK_MEMORY_CALIBER_BY_BACKEND`。
MSSWIFT_RESERVED_CALIBER: str = "msswift_rank0_max_reserved"

#: ms-swift 自报 reserved 峰值在台账里的**独立字段名**：保留为该后端自报值的**原始读数**
#: （与 §7 峰值列同值但自带来源标签）。它与 :data:`HOST_SAMPLE_PEAK_FIELD` 是同一种设计
#: ——把"另一个来源的实测值"显式留在台账里，让"这一格是什么口径"可回答（§1.4 / §2.2）。
MSSWIFT_RESERVED_PEAK_FIELD: str = "msswift_reserved_peak_gib"

#: 与上面那个值**成对落库**的口径标签字段名 —— 让下游"看到数字就同时看到口径"，
#: 不必回查本模块或源码（§2.2 显式即防呆）。
MSSWIFT_RESERVED_CALIBER_FIELD: str = "msswift_reserved_peak_caliber"

#: 与上面那个值**成对落库**的说明字段名（有值 = 口径理由；无值 = 未取得原因）。
MSSWIFT_RESERVED_NOTE_FIELD: str = "msswift_reserved_peak_note"

#: ms-swift 自报峰值**未取得**时的显式标注（**不得**用宿主采样或 native 值兜底）。
MSSWIFT_RESERVED_UNAVAILABLE: str = "未取得（ms-swift 未打印 memory(GiB)）"

#: ms-swift 自报值**为什么是保守上界、以及它在二分口径下的地位**的一句话理由（随字段
#: 一起落台账，让下游不必读源码就知道该数与本档 §7 峰值列同源）。
MSSWIFT_RESERVED_NOTE: str = (
    "口径=max_memory_reserved（swift/trainers/patcher.py:27 → swift/utils/torch_utils.py:416）；"
    "§9.1 二分口径下它即 ms-swift 档的 §7 峰值口径；reserved>=allocated ⇒ 保守上界（不会被低估）"
)

#: ── ★ 后端 → §7 峰值列口径 的**唯一映射**（§1.4 单一真相源） ──────────────────
#:
#: 为什么必须只有一处：口径是**后端的函数**——`native` 经
#: `transformer_adapter._emit_rank_memory_event` 落 rank0 旁路（allocator 可得），
#: `ms-swift` 结构性不产该旁路、只有自报的 `memory(GiB)`（reserved）。
#: 若采集层与落盘层各写一份"哪个后端用哪个口径"，两者漂移时**没有任何测试能发现**
#: ——那正是"这一格是什么口径"变得不可回答的通道（§2.2 显式即防呆）。
#: 取值来源与依据见 `.local/本期工程跟踪.md:420`（§9.1 的该列口径行）。
PEAK_MEMORY_CALIBER_BY_BACKEND: dict[str, str] = {
    "native": PEAK_MEMORY_CALIBER,
    "ms-swift": MSSWIFT_RESERVED_CALIBER,
}

#: **后端未登记**时的显式"无口径"标签——**不猜、不默认取某一支**（§2.2）。
#: 为什么不用 `None`：台账 `jsonl` 里 `null` 与"字段缺失"不可分辨，且下游 `md` 渲染
#: 需要一个可显示的字面量；给一个**自证为"未登记"**的字符串比留空更难被误读。
PEAK_MEMORY_CALIBER_UNKNOWN: str = "unknown_backend_no_caliber"

#: 口径 → **不可得**时的显式标注（§2.2）。沿用既有 `PEAK_MEMORY_UNAVAILABLE` 风格：
#: "没拿到值"必须与"没跑过"在下游可区分，且**绝不**回落到另一种口径或宿主值。
PEAK_MEMORY_UNAVAILABLE_BY_CALIBER: dict[str, str] = {
    PEAK_MEMORY_CALIBER: PEAK_MEMORY_UNAVAILABLE,
    MSSWIFT_RESERVED_CALIBER: MSSWIFT_RESERVED_UNAVAILABLE,
}

#: 后端未登记 ⇒ 无口径可谈 ⇒ 单独的不可得标注（原因与两种已登记口径都不同）。
PEAK_MEMORY_UNAVAILABLE_UNKNOWN: str = "未取得（后端未登记峰值口径）"


def peak_memory_caliber_for_backend(backend: str) -> str:
    """按后端给出该档 §7 峰值列的口径标签（读取 :data:`PEAK_MEMORY_CALIBER_BY_BACKEND` 的唯一入口）。

    **唯一真相源（§1.4）**：本函数是"这个后端用哪个口径"的**唯一**判定点——采集层的
    取数分支与落盘层的 `peak_memory_caliber` 字段**都**从这里取值，因此两者不可能漂移。
    未登记的后端返回 :data:`PEAK_MEMORY_CALIBER_UNKNOWN`（显式，绝不默认取某一支）。
    """
    return PEAK_MEMORY_CALIBER_BY_BACKEND.get(backend, PEAK_MEMORY_CALIBER_UNKNOWN)


def peak_memory_unavailable_note(caliber: str) -> str:
    """给定口径标签，返回其**不可得**时该落的显式标注（§2.2，绝不静默回落）。"""
    return PEAK_MEMORY_UNAVAILABLE_BY_CALIBER.get(caliber, PEAK_MEMORY_UNAVAILABLE_UNKNOWN)

#: A5 四件套产物：配置备份 / 训练日志 / 可恢复 checkpoint / 运行指标。
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "config_backup",
    "training_log",
    "checkpoint",
    "metrics",
)

#: 正式记录门槛（capability-matrix.html §6）：≥1 epoch 且 ≥5 optimizer step。
#:
#: ★ 这是**缺省值**（清单未给门槛时用它），**不是**唯一真相源。权威值在
#: 清单 ``tiers[*].acceptance.formal_gate.min_optimizer_steps``——由采集层读出后
#: 经 :attr:`RunEvidence.min_optimizer_steps` 传入（§1.4 单一真相源）。
#: 曾经本模块把这个常量**硬编码**在 :func:`judge_a2` 里 ⇒ 已批准并登记在清单里的
#: 档位偏离（合成档 ``min_optimizer_steps: 1``）对判定器不可见 ⇒ **假否定**
#: （2026-09-19 task-r3-bulk §⑦-6 实测：合成 4k/8k/16k 的 A2 被判否）。
#: 改法只把"门槛从哪来"变成可配置，**缺省值不动**（= 不放松任何现有档位）。
MIN_OPTIMIZER_STEPS = 5

#: ★ **已核实的口径债（AF1 §⑧-5 / 本包 §⑧，2026-09-21 登记，未改任何行为）**：
#: 清单 ``acceptance.formal_gate`` 里另有 ``min_epochs`` 与 ``min_train_samples`` 两个键，
#: **全仓没有任何消费点**——它们只被写（``tests/e2e/generate_matrix.py`` 与
#: ``tests/e2e/rig_synth/emit_manifests.py``），没有任何判据或采集逻辑读它们
#: （机核见 ``tests/e2e/test_generate_matrix.py::test_min_train_samples_and_min_epochs_are_declared_orphans``，
#: 用 ``ast`` 证明本模块不读这两个键）。
#: ⇒ **它们是纯声明字段（orphan）**，读者不应把它们当成"已生效的门槛"。
#: 本包**不**给它们加消费点：加 ``min_epochs`` 断言会改变既有 A2 通过集
#: （``RunEvidence.epochs_completed`` 在既有读数上不一定可靠），属**收紧判据**，
#: 不在本包"判据口径"范围内，须由指挥官另立工作包拍板。
ORPHAN_FORMAL_GATE_FIELDS: tuple[str, ...] = ("min_epochs", "min_train_samples")

#: 门槛的**缺省值哨兵**：``RunEvidence.min_optimizer_steps is None`` =
#: "清单没给门槛" ⇒ 回落到 :data:`MIN_OPTIMIZER_STEPS`。
#: 为什么用 ``None`` 而不是 0/-1（§2.2）：0 与负数在"最少步数"这个语义里**没有合法
#: 解释**，拿它们当"未提供"会把非法输入静默变成合法缺省 ⇒ 判定器必须把它们
#: **判否**（fail-closed），而不是回落到 5。两者语义必须能分开。

#: ── A4（同 config 同 seed 双跑一致）的两个容差口径：**由实测推导，不拍脑袋** ──
#:
#: **实测依据（唯一真相源，禁止凭感觉改数）**：2026-09-19 T013
#: （9B · SFT · LoRA · ms-swift · 1 卡 · 100 步 · seq 8192，镜像 graspo-msswift:4.5.3）
#: 同 config 同 seed 双跑：
#:
#:   - **首步（step 1）loss 逐位相同**：``1.3092858791351318`` == ``1.3092858791351318``；
#:   - 分歧自 step 5 起（``0.49435022`` vs ``0.50010920``），
#:     **终态 logged loss 差 = 1.0302e-3**（``0.06767760515213013`` vs ``0.06870781779289245``）。
#:
#: 该量级的差异来自 **bf16 归约顺序 / GPU 内核选择**——宪法 §6 明文把"GPU 内核选择等
#: **无法控制**的差异"排除在"复现破坏"之外，故 A4 的终态 loss 容差按此量级推导。
#:
#: **为什么首步是零容差、终态才给容差**：首步 loss 只取决于 种子 / 初始化 /
#: 数据顺序 / 首批样本——全是**可控**因素，任何差异都是真缺陷的指纹；而终态 loss
#: 已经累积了 100 步的不可控浮点漂移，用零容差会变成必然失败的空断言。
#: 两条并列，**任何一条不满足都判 A4 不通过**。
#: ⚠️ **本常量只保留为"T013 单档实测记录"**（历史出处）。它**不再**是全局终态容差的
#: 推导基准——2026-09-21 裁定（依据 ``task-ad1-a4-validity/report.md`` §⑥）：该值由
#: **单档** T013 推导、**不跨档泛化**（T010 1 卡 native SFT 同配置池最坏对
#: 0.034423828125 = 33.4× 它）。终态容差改由 :data:`A4_TIER_CALIBRATIONS` **按档族**
#: 标定；"实测常量 × 固定倍数"的重推方式**已废止**（见 :func:`validate_tier_tolerance`）。
A4_MEASURED_BF16_FINAL_LOSS_DRIFT = 1.0302e-3

#: ── A4 终态容差：**分档标定表**（2026-09-21 裁定，换标定方法，不是放宽）─────────
#:
#: 依据：``task-ad1-a4-validity/report.md`` §⑥（含五要素方案与三道防放行锁）。
#: 标定方法（每条取值都必须能追到实测读数，**不得凭空取数**）：
#:
#:   ``tol_tier = min(1.5 × worst_pair, A4_TOLERANCE_HARD_UPPER_BOUND)``
#:
#: 其中 ``worst_pair = max |L_i − L_j|`` 取自**同档族 n≥5 次重复跑**的终态读数池。
#: - **n≥5**：AD1 §⑦ 指出该分布是**重尾/双峰**（同一配置的终态 loss 落在相距 ~0.03 的
#:   两个簇里）⇒ n=2 双跑会**严重低估**（T010 sealed n=2 = 0.0044 vs 全池 0.0344，差 7.8×）。
#:   因此 n<5 的档族**不得**用低估值定档：只能取"实测下界 + 证据不足"标注，且值**只下调**。
#: - **同一卡集合（mandatory）**：一个档族的容差读数池（以及 A4 比较的两跑）**必须同卡集**——
#:   容差标的是"同一 workload 在同一组卡上的内核漂移"。AO1 实测：`T032` 在卡集 `{4,5}` 上
#:   n=5 的最坏对是 `4.0139e-3`（⇒ tol 6.0209e-3）；把 W2 在 `{0,1}` 上的两次并入后最坏对抬到
#:   `1.1014e-2`、`1.5×` = `1.6522e-2`，隔离带从 15.53× 掉到 5.66×——**跨卡集把方差抬高 ≈2.74×**。
#:   因此 W2 `{0,1}` 的两次**不并入**本表该档族读数池（它同时是"为什么必须同卡集合"的实测依据）。
#: - **硬上界锁一**：任何档族的容差 **≤ 本档族最小真 bug 签名 / 2**。AD1 点名的最小真
#:   bug 签名是 T032 的 **9.35e-2**（符号翻转 −0.0217 → +0.0718）⇒ 全库硬上界 **4.675e-2**
#:   （= 9.35e-2 / 2，精确取半；AD1 口头值"4.7e-2"是同一数的上取整）。
#:   这条锁是**跨档族**的（不只是 T010 档族）：没有任何档族可以把容差抬到 4.675e-2 以上。
#:
#: ⚠️ **本表不是"把 1e-2 放大"**：它是把"单档 1e-2"换成"逐档族实测最坏对 ×1.5、再被
#: 真 bug 半量封顶"。T010 档族的取值 4.675e-2 是**被封顶**得到的
#: （1.5×0.0344=5.16e-2 > 4.675e-2），不是 1e-2 的任何倍数。
#:
#: - **档族键 = backend × cards × algorithm × model × mode**（2026-09-22 指挥官 J2/P1 裁定）：
#:   三维键粗于 workload——本批实测同族 ms-swift/1/CPT 的 9B 档差 `8.7e-3`、27B 档差 `4.14e-2`
#:   （**4.7×**），ms-swift/2/CPT 9B `2.5e-3` vs 27B `4.35e-2`（**17.5×**）⇒ 单一容差被迫取
#:   大者、贴近硬上界、把 9B 的隔离带吃光（§18 债）。`model`/`mode` 为 ``None`` 的行是
#:   **通配行**（见 :attr:`A4TierCalibration.model`），2026-09-22 之前的历史行属**已登记债**。
#: - **算法名归一**：清单写 ``GRASPO``、本表写 ``GRPO``（同一机制），由
#:   :func:`normalize_a4_algorithm` 做**显式精确**映射（J5 裁定；不改生成器，避免清单指纹失效）。
#:
#: ★ **n=5 不是统计意义上的容差上界**：分布无关的单侧 95% 覆盖 / 90% 置信区间需
#: n = ln(0.10)/ln(0.95) ≈ 45 个样本；`A4_WORST_PAIR_SAFETY_FACTOR = 1.5` 是**工程安全余量**
#: （人工判断常数），不是从分布推出的分位数。口径说明见 `docs/a4-tolerance-calibration.md`。

#: 全库硬上界（锁一）：AD1 点名的最小真 bug 签名 / 2。任何档族容差都不得越过它。
A4_TOLERANCE_HARD_UPPER_BOUND = 4.675e-2
#: 真 bug 签名（AD1 §⑥ 量化隔离表 + §③ 逐档回查表）。
#: 用途：① 硬上界 4.675e-2 = 9.35e-2 / 2；② 隔离带断言（容差 ×2 必须仍 < 最小签名）。
A4_MIN_TRUE_BUG_SIGNATURE = 9.35e-2  # T032：-0.0217 → +0.0718（符号翻转）

#: **已废止**的全局终态容差 = `A4_MEASURED_BF16_FINAL_LOSS_DRIFT × ≈10`（T013 单档）。
#: 仅作为兼容别名保留给"历史台账复算"（AD1 §④ 第 10 条：同一对数值在 1e-3 与 1e-2 下
#: 判出相反结论）。**判据已不再使用它**——新判据用 :data:`A4_TIER_CALIBRATIONS`。
A4_LEGACY_GLOBAL_TOLERANCE_ABS = 1e-2
#: ⚠️ 兼容别名（历史名字，值为被废止的全局容差）。**新代码不得引用**；保留只为让旧台账
#: 复算脚本与旧测试文件可以逐字对照"改前 vs 改后"。
A4_FINAL_LOSS_TOLERANCE_ABS = A4_LEGACY_GLOBAL_TOLERANCE_ABS

#: n≥5 的标定样本量门槛（锁二的一部分：禁止用 n=2 的低估定档）。
A4_CALIBRATION_MIN_N = 5
#: 最坏对 → 容差的余量倍数。**这是一个"由统计量推导"的倍数，不是"常量 × 固定倍数"**：
#: 它作用在**逐档族实测的最坏对**上，而不是作用在一个单档常量上——两者不可混同。
A4_WORST_PAIR_SAFETY_FACTOR = 1.5

#: 双通道一致（锁三）：loss 通道与"权重/首步"通道**冲突 ⇒ 判不可判定**。
#: 「冲突」的可判定定义（避免把正常的跨跑内核漂移误判成冲突）：
#: 终态 loss 漂移 ≤ 该档族容差（loss 通道说"一致"）
#: **且** 两跑末步 checkpoint 的 sha256 **完全相同**（权重通道说"一致"）
#: **且** 两跑 loss **逐位相同**（差恰为 0）⇒ 末步 loss 是**无鉴别力的死通道**
#: （AD1 §④ 退化类：GRPO 无信号步恒 0 / 单步扫描）⇒ 判「不可判定」。
#: 反向（loss 差恰为 0 而权重 sha 不同）**不判冲突**：GPU 内核残余下权重层分叉是
#: AD1 §6.2 已证的**预期现象**（两跑 ckpt sha 不同），把它当冲突会把所有正常双跑
#: 都判死。判据边界见 :func:`judge_a4` 的 docstring。
A4_DUAL_CHANNEL_CONFLICT_LEDGER = "⚠ 不可判定（A4 双通道冲突：loss 通道无鉴别力）"

#: 符号翻转检验（五要素第 4 条）：真 bug `T032` 表现为**终态 loss 值符号翻转**
#: （−0.0217 → +0.0718）。判据见 :func:`_is_sign_flip`：异号，**且**【平均幅度 >
#: ``A4_SIGN_FLIP_RELATIVE_FLOOR`` × 该档容差】或【两端幅度 ≥ 容差 且 相对差 > 2.0】。
#: 地板的作用是避免把"零附近符号抖动的内核噪声"（如 ±1e-12）误报成 bug——
#: 同时保住 T032 这种"一端幅度不大、均值被容差盖住"的真翻转。
A4_SIGN_FLIP_RELATIVE_FLOOR = 0.5

#: 中位数检验（五要素第 4 条）：无 bug 的漂移应关于 0 **对称**；真 bug 常表现为**单向偏移**。
#: 判据用 ``median(delta)``（两跑集合的全部两两差值）与容差的比值。
A4_MEDIAN_SHIFT_WARN_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class A4TierCalibration:
    """一个档族的终态容差标定记录（**单一真相源**：每个档族一行，含出处）。

    ``tol`` **一律**满足 ``A4_MEASURED_BF16_FINAL_LOSS_DRIFT <= tol <=
    A4_TOLERANCE_HARD_UPPER_BOUND``（见 :func:`validate_tier_tolerance`），
    因此不存在"标定表里塞一个无出处的大数"的空间。
    """

    #: 档族标识：``backend × cards × algorithm``（五要素第 1 条）。
    backend: str
    cards: int
    algorithm: str
    #: 该档族的终态容差（绝对值，loss 量纲）。判定器的**唯一**取数入口。
    tol: float
    #: 本档族标定所用**最坏对**（max |Li − Lj|，n≥5 读数池）。
    worst_pair: float
    #: 参与标定的读数个数。
    n: int
    #: 结局：``calibrated``（n≥5）/ ``insufficient_n``（n<5，证据不足）/ ``degenerate``
    #: （末步无信号或单步 ⇒ 终态子检查零鉴别力）。
    outcome: str
    #: **出处**：逐字写出读数池与数值来源（AD1 报告行号 / 工位文件）。不得为空。
    provenance: str
    #: ── 档族键的第四、第五维：**model × mode**（2026-09-22 指挥官 J2/P1 裁定）──
    #:
    #: 为什么必须扩维（§1.4 / §18）：`backend × cards × algorithm` 粗于 workload——
    #: 本批实测同族 ms-swift/1/CPT 的 9B 档差 8.7e-3、27B 档差 4.14e-2（**4.7×**），
    #: ms-swift/2/CPT 9B 2.5e-3 vs 27B 4.35e-2（**17.5×**）。单一容差被迫取大者、
    #: 贴近硬上界 ⇒ 9B 档的隔离带被 27B 吃光。
    #:
    #: ``None`` = **通配**（该行覆盖全部 model/mode）。通配只用于两类行：
    #: ① 2026-09-22 之前的历史行（当时键只有三维，属**已登记债**，`note` 必须写明）；
    #: ② 经"上包络"标定、确有证据覆盖全部 model/mode 的行。
    #: 通配行**不构成放宽通道**：其 ``tol`` 照样要过 :func:`validate_tier_tolerance`。
    model: str | None = None
    #: 模式（``LoRA`` / ``全量``）。语义与 :attr:`model` 相同（``None`` = 通配）。
    mode: str | None = None
    #: ── 标定元数据（§1.4 可追溯：每个容差必须能指回数据来源与样本量）──
    #:
    #: 代表档号（该读数池实际跑的 tier，如 ``"T032"``）。``calibrated`` 行**不得为空**。
    representative_tier: str = ""
    #: 读数池的卡集合（AO1 立下的「同一卡集合 mandatory」纪律）。空元组 = 历史行未登记。
    card_set: tuple[int, ...] = ()
    #: 本容差**实际覆盖**的 model/mode 范围（人读；``model``/``mode`` 为 ``None`` 时必须写明
    #: 实测的是哪一个）。``calibrated`` 行**不得为空**。
    model_scope: str = ""
    #: 采样日期/时段（人读）。``calibrated`` 行**不得为空**。
    sampled_at: str = ""
    #: 小样本稳定性核对结论（留一/半分；``calibrated`` 行**不得为空**，历史行写"未做留一"）。
    split_check: str = ""
    #: 退化/证据不足的人读说明（``calibrated`` 时为 ``""``）。
    note: str = ""


A4_OUTCOME_CALIBRATED = "calibrated"
A4_OUTCOME_INSUFFICIENT_N = "insufficient_n"
A4_OUTCOME_DEGENERATE = "degenerate"

#: ── 算法名的**显式等价**映射（2026-09-22 指挥官 J5 裁定）────────────────────────
#:
#: 清单侧把 RL 档写作 ``GRASPO``（唯一真相源是生成器
#: ``tests/e2e/generate_matrix.py`` 的 ``_block("GRASPO", …)``），而 A4 标定表把
#: 同一条 RL 机制的实测读数记作 ``GRPO``（GRASPO = GRPO trainer + graspo reward，
#: 见 ``scripts/collect_results.py`` 由清单逐字透传 algorithm、**不做归一**）。
#:
#: 为什么用**显式映射**而不是改生成器：改生成器会连带 rmtree+重写
#: ``samples/configs/matrix54/`` 与 ``run_matrix54.sh`` ⇒ 228 上已跑批次的清单指纹
#: 全部失效（``task-matrix-driver/run-plan.md`` §⑨ 记录）。改判定器只动取数入口，
#: **不放宽任何判据**——映射只声明一条已文档化的等价。
#:
#: ★ 边界（§2.2 显式即防呆）：只做**精确串**映射，不做大小写归一、不做前缀匹配、
#: 不做 guess。未登记的算法串一律原样比较 ⇒ 未命中即 fail-closed。
_A4_ALGORITHM_EQUIVALENTS: Mapping[str, str] = {"GRASPO": "GRPO"}


def normalize_a4_algorithm(algorithm: str) -> str:
    """把算法名归一到标定表使用的规范串（唯一入口，见 :data:`_A4_ALGORITHM_EQUIVALENTS`）。"""
    return _A4_ALGORITHM_EQUIVALENTS.get(algorithm, algorithm)

#: 档族标定表（唯一真相源）。**取值出处逐条可追**（§1.4）。
#: 为什么 T010 的 note 写"封顶"：1.5 × 0.034423828125 = 5.1636e-2 **越过**硬上界
#: 4.675e-2（= 最小真 bug 9.35e-2 / 2，精确取半）⇒ 按锁一**封顶到 4.675e-2**，
#: 隔离带仅剩 2.0×（9.35e-2 / 4.675e-2）。
A4_TIER_CALIBRATIONS: tuple[A4TierCalibration, ...] = (
    A4TierCalibration(
        backend="native",
        cards=1,
        algorithm="SFT",
        tol=A4_TOLERANCE_HARD_UPPER_BOUND,
        worst_pair=0.034423828125,
        n=5,
        outcome=A4_OUTCOME_CALIBRATED,
        provenance=(
            "T010 · 9B · SFT · LoRA · native · 1 卡 同配置终态读数池 n=5"
            "（AA1 sealed×2：0.038818359375 / 0.0517578125；AA1 determinism×2："
            "0.043212890625 / 0.0732421875；w1：0.064453125）。"
            "最坏对 0.034423828125 = AA1-A2 vs AA1-B1。"
            "出处：task-ad1-a4-validity/report.md §6.1（:190-198）、§7.2（:261）"
        ),
        # ★ P1 迁移遗留（2026-09-22）：本行是**宽键行**（model/mode 未分维）。
        #   实测只在 T010（9B/LoRA）；同族 27B 档 T022 是**借用**本容差——T022 实测差
        #   4.30e-2 ≤ 4.675e-2 ⇒ 通过，故保留宽键以避免 fail-closed 回归。
        #   重新标定时应拆成精确行（§18 已登记债）。
        model=None,
        mode=None,
        representative_tier="T010",
        card_set=(),
        model_scope="9B/LoRA（实测 T010）；宽键行覆盖全部 model/mode",
        sampled_at="2026-09-19/20（AA1 + w1）",
        split_check="半分：sealed n=3 最坏对 2.56e-2 < 全池 3.44e-2（AD1 §7.2；未做留一）",
        note=(
            "1.5 × 0.034423828125 = 5.1636e-2 越过硬上界 4.675e-2 ⇒ **按锁一封顶**；"
            "隔离带 = 9.35e-2 / 4.675e-2 = 2.00×。"
            "★ 宽键行（P1 遗留）：实测仅 9B/LoRA，27B/LoRA 属借用，重新标定时拆精确行"
        ),
    ),
    A4TierCalibration(
        backend="ms-swift",
        cards=1,
        algorithm="SFT",
        tol=6.72e-3,
        worst_pair=4.48e-3,
        n=5,
        outcome=A4_OUTCOME_CALIBRATED,
        provenance=(
            "T013 · ms-swift · SFT · LoRA · 1 卡 **sealed 臂** n=5 两两差值（6 对）："
            "4.48e-3 / 3.16e-3 / 5.03e-4 / 0.00127 / 0.00659 / 0.00198（示例值见 AD1 表）"
            "⇒ 最坏对 4.48e-3。出处：task-ad1-a4-validity/report.md §3（r3-bulk/r3-lenramp/"
            "r3-retest2 T013 行，:98-105）×5 对；AD1 §6.3（:218-221）"
        ),
        note="1.5 × 4.48e-3 = 6.72e-3；隔离带 = 9.35e-2 / 6.72e-3 = 13.9×",
        # ★ P1 迁移遗留：宽键行；实测仅 T013（9B/LoRA），27B 档 T025 属借用
        #   （T025 实测差 4.85e-3 ≤ 6.72e-3 ⇒ 通过）。
        model=None,
        mode=None,
        representative_tier="T013",
        card_set=(),
        model_scope="9B/LoRA（实测 T013）；宽键行覆盖全部 model/mode",
        sampled_at="2026-09-20/21（r3-bulk / r3-lenramp / r3-retest2）",
        split_check="未做留一（历史行）；AD1 §3 的 6 对逐对读数见 provenance",
    ),
    A4TierCalibration(
        backend="ms-swift",
        cards=1,
        algorithm="GRPO",
        tol=A4_MEASURED_BF16_FINAL_LOSS_DRIFT,
        worst_pair=0.0,
        n=6,
        outcome=A4_OUTCOME_DEGENERATE,
        provenance=(
            "★ 2026-09-22 **清洗后重 derive**（指挥官第 1 批工作包 J3 裁定）："
            "**剔除全部 4 卡 T033 读数**——原出处写 `T031/T033`，把 4 卡档的读数并进 1 卡族，"
            "违反 AO1 立下的「同一卡集合（mandatory）」纪律（跨卡集把方差抬高 ≈2.74×）。"
            "清洗后池 = **仅 1 卡 T031 的 6 个终态读数**（j1 run1/run2、p1 run1/run2、"
            "p2 run1/run2，AD1 §3 :88-94 逐批标注「0 vs 0（差 0）」）⇒ **6/6 全部 = 0.0**"
            "（GRPO 无信号步：末步 loss=0 / grad_norm=0 / group skipped）"
            "⇒ 最坏对 = 0.0 ⇒ 终态通道**零鉴别力**，按 AD1 §④ 标 `degenerate`；"
            "容差取基线下界（只下调不上调）。"
            "★ 旧值留证（已作废）：tol=1.38e-2、worst_pair=9.2e-3、n=4、"
            "outcome=insufficient_n——该值由 4 卡 T033 读数主导，对 1 卡族是跨卡集污染。"
            "注：6 个读数来自 j1/p1/p2 三批，各批卡集未在台账登记（card_set 空）；"
            "因池内恒为 0，卡集不影响 worst_pair（恒 0）。"
        ),
        representative_tier="T031",
        card_set=(),
        model_scope="9B/LoRA（实测 T031）；宽键行覆盖全部 model/mode",
        sampled_at="2026-09-20（j1 / p1 / p2 三批）",
        split_check="未做留一；退化池（6/6 = 0）估计量恒为 0，无需稳定性核对",
        note=(
            "**退化（degenerate）**：池内 6/6 终态 loss = 0 ⇒ 终态子检查零鉴别力；"
            "容差取基线下界 1.0302e-3（只下调不上调）。"
            "★ 本行**不得**再引用任何 4 卡读数（2026-09-22 清洗）"
        ),
    ),
    A4TierCalibration(
        backend="native",
        cards=2,
        algorithm="SFT",
        tol=A4_MEASURED_BF16_FINAL_LOSS_DRIFT,
        worst_pair=0.0,
        n=0,
        outcome=A4_OUTCOME_INSUFFICIENT_N,
        provenance=(
            "AD1 §6.3 五要素第 1 条要求覆盖 native-2 卡-SFT，但**既有台账无该档族双跑读数**"
            "（AD1 §⑦：需补 n=5，约 2.5–4 GPU·小时）⇒ 本档族 **未标定**"
        ),
        note=(
            "**证据不足**：无实测读数 ⇒ 容差取基线下界 1.0302e-3（**只下调不上调**）；"
            "需按 AD1 §⑦ 补 n≥5 后重新定档"
        ),
    ),
    A4TierCalibration(
        backend="ms-swift",
        cards=2,
        algorithm="SFT",
        tol=A4_MEASURED_BF16_FINAL_LOSS_DRIFT,
        worst_pair=0.0,
        n=0,
        outcome=A4_OUTCOME_INSUFFICIENT_N,
        provenance=(
            "AD1 §6.3 五要素第 1 条要求覆盖 ms-swift-多卡，但**既有台账无该档族双跑读数**"
            "（AD1 §⑦：需补 n=5）⇒ 本档族 **未标定**"
        ),
        note="**证据不足**：无实测读数 ⇒ 容差取基线下界 1.0302e-3（**只下调不上调**）",
    ),
    A4TierCalibration(
        backend="ms-swift",
        cards=2,
        algorithm="GRPO",
        tol=6.020861864089967e-3,
        worst_pair=0.004013907909393311,
        n=5,
        outcome=A4_OUTCOME_CALIBRATED,
        provenance=(
            "T032 · ms-swift · 2 卡 · GRPO · LoRA · 80 步 · 卡集 {4,5} 同配置 **独立跑** 读数池 n=5"
            "（5 次各自完整 runner 进程 + 各自 RUN_ROOT；同一清单 ao1.json sha256 "
            "8d155330fbbde4d772eb3b5d19744420340339e83045d15ea7b56c10313b29a1；5/5 exit_code=0）。"
            "终态 loss（末个训练步）：-0.019665956497192383 / -0.016562902927398683 / "
            "-0.01565204858779907 / -0.01878647804260254 / -0.016518282890319824。"
            "最坏对 = max|Li−Lj| = 0.004013907909393311（第 2 高 0.003147673607，"
            "中位数 0.002245885133743285）；1.5 × 最坏对 = 0.006020861864089967；"
            "未触硬上界 ⇒ tol = min(1.5×最坏对, 4.675e-2) = 0.006020861864089967；"
            "隔离带 = 9.35e-2 / 6.020861864089967e-3 = 15.53×（≥ 2.0×，AE1 §③ 锁一）。"
            "采样日期：228 时钟 2026-09-21 01:25–02:25。"
            "出处：.local/hb-workspace/20260915-152747/task-ao1-2card-calibration/report.md "
            "§②（逐跑读数表）/§③（两两差值全表 + 公式推导）/§④（隔离带核算）；"
            "原始读数 evidence/ao1-readings-local.txt、复算 evidence/ao1-calib-derive.txt；"
            "入表裁定：指挥官 task-aq1-calibration-entry（2026-09-21）"
        ),
        note=(
            "隔离带 = 9.35e-2 / 6.020861864089967e-3 = 15.53×。"
            "★ **不得**被同 backend×cards 的 **OPD** 档复用：算法不同、机制不同"
            "（AO1 §⑩-2）；OPD 档仍为 `None` ⇒ fail-closed，需独立补 n≥5 后另行定档。"
            "★ P1 遗留：本行是宽键行（model/mode 未分维），实测仅 9B/LoRA（T032）；"
            "27B 档 T044 属借用（T044 实测差 8.11e-5 ⇒ 通过），重新标定时拆精确行"
        ),
        # ★ P1 迁移遗留：宽键行；读数池只有 T032（9B/LoRA）。
        model=None,
        mode=None,
        representative_tier="T032",
        card_set=(4, 5),
        model_scope="9B/LoRA（实测 T032）；宽键行覆盖全部 model/mode",
        sampled_at="228 时钟 2026-09-21 01:25–02:25",
        split_check=(
            "n=5 全池；第 2 高对 0.003147673607、中位数 0.002245885133743285"
            "（AO1 §③；未做留一）"
        ),
    ),
)

#: 无算法字段时的算法回落序（最具体 → 最泛化）。**只降维、不抬容差**。
_A4_ALGORITHM_FALLBACK_ORDER: tuple[str, ...] = ("SFT", "GRPO")
#: 无卡数/无后端字段时的回落序。同样只降维。
_A4_CARDS_FALLBACK_ORDER: tuple[int, ...] = (1, 2)
_A4_BACKEND_FALLBACK_ORDER: tuple[str, ...] = ("native", "ms-swift")


def validate_tier_tolerance(tol: float, *, min_true_bug_signature: float) -> None:
    """锁一 + 锁二：**拒绝**任何越过硬上界、或落到实测基线下方的档族容差。

    为什么这是一个**独立可测的函数**（§2.1 契约即防呆 / §2.3 边界校验）：
    危险值必须在"定档那一刻"被拦下，而不是靠人眼看表。它同时服务两处——
    ① 模块导入时对 :data:`A4_TIER_CALIBRATIONS` 全表校验（表被改坏 ⇒ 启动即炸）；
    ② 负向对照测试直接喂一个"放大到能放过 T032（0.0935）"的变体（0.34），
       证明既有测试会拦住它。

    两条界：
    - **上界**：``tol <= min_true_bug_signature / 2``（锁一，真 bug 半量封顶）；
    - **下界**：``tol >= A4_MEASURED_BF16_FINAL_LOSS_DRIFT``（实测基线下界；
      低于它等于用一个没被测到过的量当容差，会把正常的 bf16 漂移判成失败）。
    """
    hard_upper = min_true_bug_signature / 2.0
    if tol > hard_upper:
        raise ValueError(
            f"A4 档族容差 {tol!r} 越过硬上界 {hard_upper!r}"
            f"（= 最小真 bug 签名 {min_true_bug_signature!r} / 2，锁一）。"
            "任何容差都必须能放过真 bug：真 bug 量级是最小签名，容差超过它一半即失去隔离带。"
        )
    if tol < A4_MEASURED_BF16_FINAL_LOSS_DRIFT:
        raise ValueError(
            f"A4 档族容差 {tol!r} 低于实测基线下界 {A4_MEASURED_BF16_FINAL_LOSS_DRIFT!r}"
            "（会把正常的 §6 内核漂移判成失败）。"
        )


#: 入表元数据的**必填项**（仅对 ``calibrated`` 行强制；见
#: :func:`validate_tier_calibration_entry`）。
#: 为什么这几项：§1.4 要求"每个容差能指回数据来源与样本量"——单靠 `provenance` 一段散文
#: 无法机核，把可机核的部分拆成字段才能在下游自动核对。
A4_CALIBRATION_REQUIRED_METADATA: tuple[str, ...] = (
    "representative_tier",
    "model_scope",
    "sampled_at",
    "split_check",
)


def validate_tier_calibration_entry(calibration: A4TierCalibration) -> None:
    """**入表**校验（§2.1 契约即防呆 / §2.3 边界校验）：一条标定记录必须自证。

    两类界：
    - **所有行**：``provenance`` 非空（题目：取值必须能追到实测读数）；
    - **``calibrated`` 行**：``n >= A4_CALIBRATION_MIN_N``、``worst_pair > 0``、
      :data:`A4_CALIBRATION_REQUIRED_METADATA` 全非空。

    为什么 ``worst_pair > 0`` 必须卡：``worst_pair == 0`` 表示池内读数逐位相同
    （退化池）⇒ 终态通道**零鉴别力**，此时 `1.5 × 0 = 0` 会低于基线下界、把正常的
    §6 内核漂移判成失败。这种池**不得**标 ``calibrated``，应标 ``degenerate`` +
    基线下界（见 :func:`calibrated_tolerance`）。

    ``insufficient_n`` / ``degenerate`` 行**不要求**元数据（它们本就没有可用读数池）；
    但仍要求 ``provenance`` 说明"为什么没有"。
    """
    if not calibration.provenance.strip():
        raise ValueError(
            f"A4 标定行 {calibration.backend}/{calibration.cards}/{calibration.algorithm} "
            f"(model={calibration.model!r}, mode={calibration.mode!r}) 的 provenance 为空："
            "§1.4 要求每个容差能指回读数池与数值来源。"
        )
    if calibration.outcome != A4_OUTCOME_CALIBRATED:
        return
    if calibration.n < A4_CALIBRATION_MIN_N:
        raise ValueError(
            f"A4 标定行 ... 标 `calibrated` 但 n={calibration.n} < {A4_CALIBRATION_MIN_N}："
            "n<5 的最坏对严重低估（AD1 §7.2 实测 7.8×）⇒ 只能标 insufficient_n。"
        )
    if not calibration.worst_pair > 0.0:
        raise ValueError(
            "A4 标定行标 `calibrated` 但 worst_pair="
            f"{calibration.worst_pair!r}（非正）：池内读数逐位相同 ⇒ 终态通道零鉴别力，"
            "应标 degenerate + 基线下界，不得标 calibrated。"
        )
    missing = [
        name
        for name in A4_CALIBRATION_REQUIRED_METADATA
        if not str(getattr(calibration, name)).strip()
    ]
    if missing:
        raise ValueError(
            f"A4 标定行 ... 标 `calibrated` 但元数据为空：{missing}。"
            "§1.4：每个容差必须能指回代表档 / 覆盖面 / 采样时间 / 小样本核对。"
        )


#: 导入时**立即**校验全表（§2.3 边界校验在边界上）：标定表被改坏 ⇒ 启动即失败，
#: 而不是等到某一档跑完才发现判据失去隔离带。
for _calibration in A4_TIER_CALIBRATIONS:
    validate_tier_tolerance(_calibration.tol, min_true_bug_signature=A4_MIN_TRUE_BUG_SIGNATURE)
    validate_tier_calibration_entry(_calibration)
del _calibration

#: 硬上界必须正好是"最小真 bug 签名的一半"（锁一的定义本身，逐字锁死）。
assert A4_TOLERANCE_HARD_UPPER_BOUND == A4_MIN_TRUE_BUG_SIGNATURE / 2.0, (
    "A4_TOLERANCE_HARD_UPPER_BOUND 必须 == A4_MIN_TRUE_BUG_SIGNATURE / 2（锁一）"
)
#: 隔离带断言：每个已定档的容差 ×2 必须仍**严格小于**最小真 bug 签名
#:（即容差本身 < 签名/2，留出 ≥2× 的判别余量——AD1 §6.3 方案验收判据 ①）。
for _calibration in A4_TIER_CALIBRATIONS:
    if _calibration.outcome == A4_OUTCOME_CALIBRATED:
        # 浮点算术不带容差 ⇒ 用相对 epsilon 表达"严格小于"（0.04675×2 在 IEEE754 下
        # 恰好 = 0.0935 还是 = 0.0934999… 取决于具体值，不能靠裸 `<`）。
        assert (
            _calibration.tol * 2
            < A4_MIN_TRUE_BUG_SIGNATURE + 1e-12 * max(1.0, A4_MIN_TRUE_BUG_SIGNATURE)
        ), (
            f"档族 {_calibration.backend}/{_calibration.cards}/{_calibration.algorithm}"
            f"（model={_calibration.model!r}, mode={_calibration.mode!r}）"
            f"容差 {_calibration.tol!r} ×2 不小于最小真 bug 签名 "
            f"{A4_MIN_TRUE_BUG_SIGNATURE!r}（隔离带消失）"
        )
del _calibration


def worst_pair(readings: Sequence[float]) -> float:
    """读数池的**最坏对** ``max |L_i − L_j|``（五要素第 2 条的统计量）。

    ``n < 2`` ⇒ 没有"对"可言，返回 ``0.0``（调用方必须结合 ``len(readings)`` 判证据是否足够，
    见 :func:`calibrated_tolerance`——不得把 0.0 当成"实测漂移为零"）。
    """
    values = [float(value) for value in readings]
    if len(values) < 2:
        return 0.0
    return max(abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :])


def calibrated_tolerance(
    readings: Sequence[float],
    *,
    min_true_bug_signature: float = A4_MIN_TRUE_BUG_SIGNATURE,
    min_n: int = A4_CALIBRATION_MIN_N,
) -> tuple[float, str]:
    """由读数池推导容差，返回 ``(tol, outcome)``。

    **统计量 = n≥5 最坏对 × 1.5，再被硬上界封顶、再被实测基线下界兜底**（五要素第 2 条 + 锁一）。

    三个分支：

    - ``n < min_n`` ⇒ ``(A4_MEASURED_BF16_FINAL_LOSS_DRIFT, A4_OUTCOME_INSUFFICIENT_N)``
      ——**绝不允许**用 n=2 的低估去定档（AD1 §7.2：n=2 = 0.0044 vs 全池 0.0344，低估 7.8×）。
    - ``n >= min_n`` 但 **``worst_pair == 0``（池内逐位相同）** ⇒
      ``(A4_MEASURED_BF16_FINAL_LOSS_DRIFT, A4_OUTCOME_DEGENERATE)``。
      为什么不能返回 0：``1.5 × 0 = 0`` 低于基线下界，会把正常的内核漂移判成失败；
      退化池的终态通道**零鉴别力**（AD1 §④），只能取下界并显式标退化。
    - 否则 ⇒ ``max(min(1.5 × worst_pair, 硬上界), 基线下界)``，``calibrated``。
      ★ 为什么要 `max(..., 基线下界)` 而不是抛错：若某族实测漂移**小于**基线下界
      （如 ms-swift/2/SFT 9B 实测差 ~1.9e-4 ⇒ 1.5× = 2.8e-4 < 1.0302e-3），
      说明"这一族比已知最小内核漂移还稳"——此时正确的容差是**下界**（下界锁的
      本意就是"不得用没被测到过的更小量当容差"），而不是让模块导入炸掉。
      `validate_tier_tolerance` 只负责**拒绝**表里手写越界的值，不负责给合法读数池判死。

    ★ 注意：本函数**不做** `常量 × 固定倍数` 的推算——倍数只作用在"逐档族实测最坏对"上。
    """
    if len(readings) < min_n:
        return A4_MEASURED_BF16_FINAL_LOSS_DRIFT, A4_OUTCOME_INSUFFICIENT_N
    measured = worst_pair(readings)
    if measured <= 0.0:
        return A4_MEASURED_BF16_FINAL_LOSS_DRIFT, A4_OUTCOME_DEGENERATE
    tol = A4_WORST_PAIR_SAFETY_FACTOR * measured
    capped = min(tol, min_true_bug_signature / 2.0)
    floored = max(capped, A4_MEASURED_BF16_FINAL_LOSS_DRIFT)
    validate_tier_tolerance(floored, min_true_bug_signature=min_true_bug_signature)
    return floored, A4_OUTCOME_CALIBRATED


def _is_sign_flip(first_loss: float, second_loss: float, tolerance: float) -> bool:
    """符号翻转检验（五要素第 4 条）：真 bug ``T032`` 的指纹。

    ``T032`` 的终态是 ``−0.0217349 → +0.0718016``（差 0.0935）——真 bug 里最常见的一种：
    训练方向被破坏到 loss 值**越过 0**。无 bug 的内核漂移不会让 loss 越零（它只在
    原值附近抖动），所以这条检验的误报率极低。

    两个**或**起来的判据（异号是前置条件）：

    ① 平均幅度 ``(|L1|+|L2|)/2`` 超过 ``A4_SIGN_FLIP_RELATIVE_FLOOR × 容差``；
    ② 两端幅度**各自**都 ≥ 该档族容差，**且**相对差
       ``|L1−L2| / max(|L1|,|L2|)`` 超过 ``A4_SIGN_FLIP_RELATIVE_FLOOR × 4``（= 2.0）。

    为什么需要两条（缺一条就会漏掉真 bug 的一种形状）：

    - T032 的形状是"一端幅度小、一端幅度大"：``−0.0217`` 只有容差的 0.36 倍（在
      native 档族容差 4.675e-2 下），补一个稍宽的容差就会把判据①的均值压到地板之下——
      此时只有判据②（两端都 ≥ 容差 + 相对差 4.3 倍）能开火；
    - "两端都成量级、均值很大"的形状由判据①覆盖。

    判据②的相对差门槛是防误报的：零附近的内核抖动 ``−1e-12 → +1e-12`` 虽然相对差
    高达 2.0，但两端幅度远小于容差 ⇒ 被"各自 ≥ 容差"挡掉，不误报。

    ★ 在"容差很紧"的档族上这条闸是**冗余**的（任何异号的差值都会先越容量级 ⇒ 子检查②
    已判否）；它的价值在容差被定得较松时**兜住**量级检查漏掉的方向破坏。
    """
    if (first_loss > 0.0) == (second_loss > 0.0):
        return False
    mean_magnitude = (abs(first_loss) + abs(second_loss)) / 2.0
    if mean_magnitude > A4_SIGN_FLIP_RELATIVE_FLOOR * tolerance:
        return True
    if abs(first_loss) < tolerance or abs(second_loss) < tolerance:
        return False
    larger = max(abs(first_loss), abs(second_loss))
    relative_gap = abs(first_loss - second_loss) / larger
    return relative_gap > A4_SIGN_FLIP_RELATIVE_FLOOR * 4.0


def _median_shift_ratio(deltas: Sequence[float], tolerance: float) -> float | None:
    """中位数检验（五要素第 4 条）：``median(|Δ|) / tol``。

    无 bug 的漂移关于 0 对称 ⇒ 多跑的两两差值中位数应远小于容差；真 bug 的**单向偏移**
    会把中位数顶起来。返回 ``None`` 表示没有可比的对（n<2）。
    """
    if len(deltas) < 2 or tolerance <= 0.0:
        return None
    ordered = sorted(abs(delta) for delta in deltas)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return median / tolerance


def degenerate_final_check_reason(losses: Sequence[float], grad_norms: Sequence[float]) -> str | None:
    """AD1 §④ 的 7 条「退化通过」：末步无信号步或 ``losses`` 长度=1 ⇒ 终态子检查零鉴别力。

    两种退化形态（都来自 AD1 台账实测）：

    1. **``len(losses) == 1``**：终态 == 首步。终态子检查只是把首步值又比一遍
       （synth T013 4k/8k/16k）；
    2. **末步是无信号步**：``loss == 0``（GRPO ``loss 0 / grad_norm 0 / group skipped``）。
       此时任何终态破坏都被平凡满足（T031 ×3、T043）。

    返回**人读原因**（``None`` = 不退化）。★ 本函数**只做标注**，不改判定方向：
    这类档是"**证据不足**"，**不是**"不通过"（不得直接改判失败）。
    """
    if len(losses) == 1:
        return "losses 长度=1（单步）⇒ 终态 == 首步，终态子检查零鉴别力"
    if losses:
        last = losses[-1]
        if isinstance(last, float) and last == 0.0:
            tail = grad_norms[-1] if grad_norms else None
            hint = "、grad_norm=0" if (isinstance(tail, float) and tail == 0.0) else ""
            return f"末步 loss=0{hint}（无训练信号步）⇒ 终态子检查零鉴别力"
    return None


def _model_mode_specificity_order(
    model: str | None, mode: str | None
) -> tuple[tuple[str | None, str | None], ...]:
    """给出 ``(model, mode)`` 的匹配候选，**从最具体到最泛化**（§2.2 显式即防呆）。

    ``None`` = 该维**未知**（调用方没给）或该行**通配**（见 :attr:`A4TierCalibration.model`）。
    规则：

    - 某一维**未知**时，**不得**匹配该维的精确行（例如 mode 未知就不得用 ``LoRA`` 行）；
      只允许该维通配的行 —— 这是 fail-closed，不是放宽。
    - 顺序 = 五元组精确 → 单维通配 → 全通配；先命中先返回。
    """
    order: list[tuple[str | None, str | None]] = [(model, mode)]
    if model is not None:
        order.append((model, None))
    if mode is not None:
        order.append((None, mode))
    order.append((None, None))
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[tuple[str | None, str | None]] = []
    for item in order:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def find_tier_calibration(
    backend: str,
    cards: int,
    algorithm: str,
    model: str | None = None,
    mode: str | None = None,
) -> A4TierCalibration | None:
    """按 ``backend × cards × algorithm × model × mode`` 取标定记录（五要素第 1 条 + J2/P1）。

    取数规则（**只降维、不抬容差**）：

    1. **五元组精确命中**优先；
    2. 未命中 ⇒ 回落 ``model``/``mode`` 维的**通配行**（``A4TierCalibration.model is None``），
       顺序见 :func:`_model_mode_specificity_order`；
    3. 仍未命中 ⇒ ``None``（调用方 fail-closed）。
       ★ **不返回任何全局默认容差**——"废止把单档 1.0302e-3 全局化"正是本包的核心要求。

    算法名先经 :func:`normalize_a4_algorithm` 归一（``GRASPO`` → ``GRPO``，J5 裁定）。
    通配行不构成放宽通道：其 ``tol`` 照样过 :func:`validate_tier_tolerance` 与
    :func:`validate_tier_calibration_entry`（模块导入时全表校验）。
    """
    normalized = normalize_a4_algorithm(algorithm)
    for candidate_model, candidate_mode in _model_mode_specificity_order(model, mode):
        for calibration in A4_TIER_CALIBRATIONS:
            if (
                calibration.backend == backend
                and calibration.cards == cards
                and calibration.algorithm == normalized
                and calibration.model == candidate_model
                and calibration.mode == candidate_mode
            ):
                return calibration
    return None


def _resolve_tier_calibration(
    backend: str | None,
    cards: int | None,
    algorithm: str | None,
    model: str | None = None,
    mode: str | None = None,
) -> A4TierCalibration | None:
    """把"缺失的档族字段"用**只降维**的回落序补全，直接返回标定记录（§2.2 显式即防呆）。

    判定器拿不到后端/卡数/算法时（如纯逻辑测试只造了 ``RunEvidence``），必须回落到
    **更泛化**的档族——回落序是**显式注册表**（:data:`_A4_BACKEND_FALLBACK_ORDER` 等），
    不是 ``try/except`` 里猜。回落**只降维、不抬容差**：目标档族的容差照样要过
    :func:`validate_tier_tolerance` / :func:`validate_tier_calibration_entry`。

    ``model``/``mode`` 为 ``None`` 时**只**允许命中通配行（见 :func:`find_tier_calibration`）。
    返回 ``None`` = 连最泛化的档族都没有 ⇒ 调用方 fail-closed。
    """
    backends = (backend,) if backend else _A4_BACKEND_FALLBACK_ORDER
    cards_seq = (cards,) if isinstance(cards, int) else _A4_CARDS_FALLBACK_ORDER
    algorithms = (algorithm,) if algorithm else _A4_ALGORITHM_FALLBACK_ORDER
    for candidate_backend in backends:
        for candidate_cards in cards_seq:
            for candidate_algorithm in algorithms:
                hit = find_tier_calibration(
                    candidate_backend, candidate_cards, candidate_algorithm, model, mode
                )
                if hit is not None:
                    return hit
    return None


def build_tier_calibration(
    readings: Sequence[float],
    *,
    backend: str,
    cards: int,
    algorithm: str,
    provenance: str,
    model: str | None = None,
    mode: str | None = None,
    representative_tier: str = "",
    card_set: tuple[int, ...] = (),
    model_scope: str = "",
    sampled_at: str = "",
    split_check: str = "",
) -> A4TierCalibration:
    """由实测读数池**造一条**档族标定记录（定档入口，含 n<5 / 退化两条 fail-closed 分支）。

    ★ 这是唯一允许"产生容差"的入口：任何新档族必须先有读数池与出处，才能定档。
    没有读数（``readings`` 为空）⇒ 走 ``insufficient_n`` 分支，容差落到基线下界。
    池内读数逐位相同（``worst_pair == 0``）⇒ 走 ``degenerate`` 分支，容差同样落下界。
    """
    tol, outcome = calibrated_tolerance(readings)
    measured = worst_pair(readings)
    note = ""
    if outcome == A4_OUTCOME_INSUFFICIENT_N:
        note = (
            f"**证据不足**：n={len(readings)} < {A4_CALIBRATION_MIN_N}"
            f"（AD1 §7.2：n=2 低估 7.8×）⇒ 不下档，容差取基线下界 "
            f"{A4_MEASURED_BF16_FINAL_LOSS_DRIFT:g}（只下调不上调）"
        )
    elif outcome == A4_OUTCOME_DEGENERATE:
        note = (
            f"**退化（degenerate）**：n={len(readings)} 但池内读数逐位相同"
            f"（worst_pair=0）⇒ 终态子检查零鉴别力，不下档，容差取基线下界 "
            f"{A4_MEASURED_BF16_FINAL_LOSS_DRIFT:g}（只下调不上调）"
        )
    elif A4_WORST_PAIR_SAFETY_FACTOR * measured < A4_MEASURED_BF16_FINAL_LOSS_DRIFT:
        note = (
            f"**下界锁生效**：1.5 × 最坏对 {measured:.6g} = "
            f"{A4_WORST_PAIR_SAFETY_FACTOR * measured:.6g} 低于实测基线下界 "
            f"{A4_MEASURED_BF16_FINAL_LOSS_DRIFT:g} ⇒ 容差取下界（只上调到已知最小内核漂移）"
        )
    calibration = A4TierCalibration(
        backend=backend,
        cards=cards,
        algorithm=normalize_a4_algorithm(algorithm),
        tol=tol,
        worst_pair=worst_pair(readings),
        n=len(readings),
        outcome=outcome,
        provenance=provenance,
        model=model,
        mode=mode,
        representative_tier=representative_tier,
        card_set=card_set,
        model_scope=model_scope,
        sampled_at=sampled_at,
        split_check=split_check,
        note=note,
    )
    validate_tier_calibration_entry(calibration)
    return calibration


#: 「该字段没有可解释的数值」的 **NaN 哨兵**（``RunEvidence.losses`` 里用它表达
#: ``loss: null`` / 字段缺失）。为什么用 NaN 而不是 None：判定链是
#: ``tuple[float, ...]``，NaN 天然让 finite 检查 fail-closed，不需要把类型契约
#: 改成 ``float | None``（§5 先正确后可优化）。
#:
#: ⚠️ **语义边界**：NaN 只能表示"读不到值"，**不得**用来表示"读到 NaN"——
#: 后者是**数值异常**（一个关于训练数值的事实断言）。两者必须能分开：
#: 见 :data:`MISSING_SENTINEL` 与 :func:`judge_a6` 的检查顺序。
NAN_SENTINEL = float("nan")

#: 「loss 字段无值」的哨兵。刻意**不是** float（类型是 str）：数值分析层会
#: ``TypeError`` 大声失败，而不是把一个"没有读数"静默当成数值参与计算。
#: 与 :data:`NAN_SENTINEL` 的分工：本哨兵只用于"明确知道某步没有 loss 读数"。
MISSING_SENTINEL: Any = "MISSING"

#: 训练器在首个非有限梯度处硬失败时打的标记（唯一真相源：
#: ``flow/adapters/models/qwen35_36/training_sft.py``）。
NONFINITE_GRAD_MARKER = "非有限梯度"

#: A6 的「最终 loss 不高于初始 loss」子检查的**记录字段名**：appears verbatim in
#: :func:`judge_a6` 的明细文本，供人眼一眼看到、并由采集层
#: （``scripts/collect_results.py`` 的 ledger 组装处与 :func:`ledger_row`）原样带进台账。
#: 唯一真相源——不得在任何地方另拼这个字面量（§1.4/§2.2）。
A6_LOSS_TREND_FIELD = "loss_trend"

#: ── ★ 裁定：A6 的「最终 loss ≤ 初始 loss」子检查**降为记录项**，不作阻断 ──────────
#:
#: **裁定人/时间**：指挥官裁定（2026-09-20），依据真机实测。
#:
#: **为什么降级**（统计意义 + 误判实例）：
#: 首末两点比较在整个 loss 序列的波动区间里**没有统计意义**。T010
#: （9B·SFT·LoRA·native·1 卡）跑了 100/100 步、``exit_code=0``、ckpt 可重载、
#: 每一步 loss/grad_norm 全程 finite、``skipped_nonfinite=0``，但 ``losses[0]=0.0250``
#: → ``losses[-1]=0.0603``，而**同一 run 的逐步波动区间是 0.0042–0.4043**。
#: 于是这条子检查把一次**完全健康**的训练判成「数值异常」（T010 实测伪否）。
#:
#: **代码内旁证**：``scripts/collect_results.py`` 的注释把"末点高于起点"
#: 记作**窄噪声区间**（``min 0.00418 / max 0.404``）——同一语义在代码里当噪声、
#: 在判据里却当阻断，是双口径（违反 §1.4）。
#:
#: **语义依据**：A6 的 docstring 自称「**数值健康**」判据；"loss 是否真下降"是
#: **效果**信号，应由**准确率**类指标回答（能力矩阵 §8：SFT ≥50% / GRASPO Δ≥20pp），
#: 不该由单点 loss 比较承担。
#:
#: **为什么保留为记录项而不是删除**（§2.2 显式即防呆）：降级 ≠ 静默丢弃。
#: "这档末点高于起点"仍要**显式可见**（进入 A6 明细文本与台账 ``loss_trend``），
#: 只是不再使 A6 判否。
#:
#: **不回退的部分**（红线，见 ``tests/core/test_result_judge.py`` 的守卫用例）：
#: ① 真读到非有限 ``loss``/``grad_norm`` ⇒ A6 仍然不通过；
#: ② 读不到读数（``NAN_SENTINEL``/``MISSING_SENTINEL``）⇒ 仍然 fail-closed。
#: 本条只放开"首末两点的走向"，**没有**放松任何数值健康事实断言。
#:
#: **如何回退**：把本常量改成 ``True`` 即恢复旧行为（阻断 + 分类数值异常）；
#: 若整条子检查都要下线，删掉 :func:`judge_a6` 里读它的那一个 if 块即可，
#: 其余（记录字段、台账字段、测试）都可保留——它们对两种取值都成立。
A6_LOSS_TREND_BLOCKS = False


class FailureClass(StrEnum):
    """失败分类。前 7 个是规定的分类；``UNCLASSIFIED`` 与 ``EVIDENCE_GAP`` 是安全兜底。

    兜底存在的理由：把"分类不出来"硬塞进 7 类中的某一类，可能把框架 bug
    误标成配置非法或数值异常，从而在修复前被当成结论。两个兜底都
    永远不计入最大可行上下文，必须人工判定。

    ``EVIDENCE_GAP`` 与 ``UNCLASSIFIED`` 的分工（本轮新增，见 :func:`judge_tier`）：

    - ``EVIDENCE_GAP``：**判据一条都没被证据否定**，只是证据读不到 ⇒ 台账状态是
      "⚠ 不可判定（取证缺口）"。这不是训练的失败，而是**判定器/采集侧的缺口**；
      把它记成「❌ 失败」等于把 collector 的缺陷记成训练失败（F-D 实测踩过）。
    - ``UNCLASSIFIED``：确实有证据说明某条判据不成立，但归不进前 7 类。
    """

    REAL_OOM = "真 OOM"
    FRAMEWORK_UNIMPLEMENTED = "框架未实现"
    CONFIG_INVALID = "配置非法"
    DATA_PROBLEM = "数据问题"
    COMM_HARDWARE = "通信硬件"
    NUMERIC_ANOMALY = "数值异常"
    TIMEOUT = "超时"
    EVIDENCE_GAP = "取证不足（不可判定）"
    #: ★ **第四类"非失败"**（AF1/指挥官裁定，2026-09-21）：该档的 step 门槛**结构上
    #: 不可测**——取数口径所限，用本档 config 无论跑多久都达不到门槛。
    #: 为什么单列一类而不并进 ``EVIDENCE_GAP``：取证缺口是"这次没取到证据"，
    #: 口径不可测是"这档根本测不了这件事"——前者修采集可解，后者要改档位定义
    #: （属目标/预算层）。两者在台账上的动作完全不同，混在一起会把用户引向错误的修法。
    #: 与 ``EVIDENCE_GAP`` 一样**不计入**最大可行上下文（见 :data:`MAX_CONTEXT_FAILURE_CLASSES`）。
    GATE_NOT_APPLICABLE = "口径不可测（步数上限 < 门槛）"
    #: ★ 新增（2026-09-22 裁定 3，T035 实测）：**集合通信超时**导致的进程组崩溃。
    #: 为什么单列而不并进 ``COMM_HARDWARE``：``COMM_HARDWARE`` 的既有签名
    #: （``watchdog timeout`` / ``NCCL error|WARN|timeout``）**匹配不到**真实形态
    #: ``Watchdog caught collective operation timeout`` + ``DistBackendError``
    #: ⇒ T035 被迫落 ``UNCLASSIFIED``。单列还能自证"各 rank 步调不一致"这一**修法方向**，
    #: 而``通信硬件``会把人引向换卡/查线。
    #: 取值与跑批装置（``rig/matrix_batch_driver.py``）**逐字一致**（同一口径两个消费者）。
    NCCL_COLLECTIVE_TIMEOUT = "nccl_collective_timeout（集合通信超时 ⇒ PP/DP 各 rank 步调不一致）"
    #: ★ 新增（2026-09-22 指挥官裁定）：**PP 流水线 P2P 通信超时**，退出码 **86**。
    #: 类名逐字取自裁定（跨工具口径一致）。判据**优先按退出码**——因为 torchrun/弹性启动
    #: 会把异常文本前缀化（如 `[rank1]:` / `ChildFailedError` 包裹），文本不可靠。
    PIPELINE_P2P_TIMEOUT = "pipeline_p2p_timeout"
    #: ★ 新增（2026-09-22 指挥官裁定）：**PP rollout 墙钟上限到点**，退出码 **21**。
    #: 与 `TIMEOUT`（124，OS/runner 层整体超时）分开：这是 **rollout 自身的预算闸**，
    #: 指向"这一档的 rollout 太慢/无界"，修法在 PP rollout 而不是调度层。
    PP_ROLLOUT_WALL_CLOCK_CAP = "pp_rollout_wall_clock_cap"
    UNCLASSIFIED = "未分类（需人工判定）"


#: **退出码 → 失败类别**（唯一真相源；2026-09-22 指挥官裁定"**优先按退出码分类**"）。
#:
#: 为什么优先于文本匹配：这些退出码是**进程交还的事实**，而异常文本可能被 torchrun /
#: 弹性启动前缀化（`[rank1]:`、`ChildFailedError` 包裹）⇒ 文本匹配会漏判或错判。
#: 只登记**有明确语义**的码；未登记的码继续走原有文本/兜底路径（**不放宽**）。
EXIT_CODE_FAILURE_CLASSES: dict[int, FailureClass] = {
    86: FailureClass.PIPELINE_P2P_TIMEOUT,
    21: FailureClass.PP_ROLLOUT_WALL_CLOCK_CAP,
}

#: 只有这一类失败允许被解释为"上下文太长"。
MAX_CONTEXT_FAILURE_CLASSES: frozenset[str] = frozenset({FailureClass.REAL_OOM})

#: 日志文本 → 失败类型的匹配表。**顺序即优先级**（先命中先返回）。
#:
#: ★ **数值异常必须排在表头**（🟡-4，2026-09-18 复核修正）：`非有限梯度` 是
#: 训练器在首个非有限梯度处**主动打出的硬失败标记**（唯一真相源：
#: ``flow/adapters/models/qwen35_36/training_sft.py``），是关于训练数值的**事实
#: 断言**；而 `FileNotFoundError` / `NotImplementedError` 是通用文本。两者同现在
#: 同一份日志时，旧顺序（数值异常排第 5）会先命中数据/框架文本，把排查引向数据
#: 而不是数值链路。数值异常与数据问题**都不计入最大可行上下文**，故本次修正只
#: 改变"归类给谁看"，不改变能力边界（也不改变 `passed` 语义）。
_LOG_PATTERNS: tuple[tuple[FailureClass, tuple[re.Pattern[str], ...]], ...] = (
    (
        FailureClass.NUMERIC_ANOMALY,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                # F-4 P0 修法①：训练器在首个非有限梯度处硬失败时打的显式标记
                # （``flow/adapters/models/qwen35_36/training_sft.py`` 的唯一真相源）。
                r"非有限梯度",
                r"non-?finite gradient",
                r"nonfinite_loss_or_grad",
            )
        ),
    ),
    (
        FailureClass.FRAMEWORK_UNIMPLEMENTED,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"NotImplementedError",
                r"is not implemented",
                r"not supported (?:yet|in|by)",
                r"unsupported (?:argument|feature|op)",
                r"not implemented for",
            )
        ),
    ),
    (
        FailureClass.CONFIG_INVALID,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"配置校验失败",
                r"extra_forbidden",
                r"unknown (?:argument|field|parameter)",
                r"unrecognized arguments",
                r"pydantic.*ValidationError",
                r"ValidationError",
                r"must be (?:>=|<=|>|<|one of)",
                # detached loss（T046 OPD(GKD) 冒烟实测，2026-09-19）：第 0 步
                # `accelerator.backward(loss)` 抛这一句，真因是**接线/参数透传**
                # （OPD 少传 `--lmbda` ⇒ 上游默认 0.5 ⇒ GKD 分流到"按数据集既有回答
                # 算散度"的 off-policy 分支 ⇒ 纯提示词行零监督 token ⇒
                # `gkd_loss` 返回 detached 零张量）。它是 torch 对该情形的**唯一、
                # 确定性**文本，只可能来自反向传播前的配置/接线错误，不会与真 OOM /
                # NaN / 数据坏样本混淆；归到「配置非法」才能把读者引向配置，而不是
                # 数据或上下文长度（旧行为下 0 命中 ⇒ 归 UNCLASSIFIED，是**假的
                # "未知"**）。能力边界不变：`MAX_CONTEXT_FAILURE_CLASSES={REAL_OOM}`。
                r"does not require grad and does not have a grad_fn",
            )
        ),
    ),
    (
        FailureClass.DATA_PROBLEM,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"No such file or directory",
                r"FileNotFoundError",
                r"json\.decoder\.JSONDecodeError",
                r"dataset .* not found",
                r"empty (?:dataset|sample)",
                r"KeyError.*(?:targets|messages|media)",
            )
        ),
    ),
    (
        FailureClass.PIPELINE_P2P_TIMEOUT,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                # 退出码 86 的**文本兜底**：若退出码被外层归一化（如 torchrun 交还 1），
                # 仍能从异常名认出该缺陷。放在 NCCL 之前（更具体者先匹配）。
                r"PipelineP2PTimeoutError",
                r"pipeline p2p timeout",
            )
        ),
    ),
    (
        FailureClass.PP_ROLLOUT_WALL_CLOCK_CAP,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"rollout wall.?clock cap",
                r"rollout 墙钟上限",
            )
        ),
    ),
    (
        FailureClass.NCCL_COLLECTIVE_TIMEOUT,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                # 真实形态（T035 实测原文）：进程组 watchdog 报集合算子超时，
                # 随后 `terminate called after throwing an instance of 'c10::DistBackendError'`。
                r"Watchdog caught collective operation timeout",
                r"Process group watchdog thread terminated",
                r"DistBackendError",
            )
        ),
    ),
    (
        FailureClass.COMM_HARDWARE,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"NCCL (?:error|WARN|timeout)",
                r"unhandled cuda error",
                r"CUDA error",
                r"device-side assert",
                r"Xid",
                r"ECC",
                r"watchdog timeout",
                r"invalid device ordinal",
            )
        ),
    ),
    (
        FailureClass.REAL_OOM,
        tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"torch\.OutOfMemoryError",
                r"OutOfMemoryError",
                r"CUDA out of memory",
                r"HIP out of memory",
                r"out of memory\. Tried to allocate",
            )
        ),
    ),
)


# ── 判据结果 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """单条判据的结果。

    ``evidence_missing`` 区分两种"不通过"（F-D 实测缺陷的收口）：

    - ``False``（默认）：**证据证明判据不成立**（如 exit≠0、真读到 NaN、权重未变化）
      ⇒ 这是关于这次训练的**事实断言**，台账记「❌ 失败」；
    - ``True``：**证据读不到**（collector 定位不到产物 / 缺 checkpoint / 缺序列）
      ⇒ 我们**没有**否定这次训练，台账记「⚠ 不可判定（取证缺口）」。

    两者都 ``passed=False``（fail-closed 方向不变：读不到绝不当成通过）。

    ``not_applicable`` 是**第三种**"不通过"（AF1/指挥官裁定，2026-09-21）：

    - ``True``：**该判据对这一档结构上不可测**——不是"训练失败了"，也不是"读不到证据"，
      而是"用这个 config 无论跑多久，这条判据都不可能被满足/证伪"
      （唯一已知实例：native GRASPO 在 mini 取数下的 optimizer 步数上限 < 清单门槛，
      AF1 诊断的 9 档）。台账必须记「⚠ 口径不可测」，**不得**记成「❌ 失败」，
      也**不得**记成「✅ 通过」。

    ★ 字段方向不变：``passed`` 仍为 ``False``（本判据**没有被证明成立**，绝不因"不适用"
    而被当成通过）；变的只是"这条不通过该记成什么"。判定侧靠 ``detail`` 里的
    :data:`STEP_GATE_NOT_APPLICABLE_MARKER` 识别第三态，:func:`judge_tier` 据此把它
    从"阻断项"里**移出**单列（见 :class:`TierJudgement`）。
    """

    criterion: str
    passed: bool
    detail: str
    evidence_missing: bool = False
    #: ``True`` = 该判据对该档**结构上不可测**（第三态）。与 ``evidence_missing`` 互斥：
    #: 前者是"这档根本测不了这件事"，后者是"这次没取到证据"。
    not_applicable: bool = False


@dataclass(frozen=True, slots=True)
class RunEvidence:
    """一次运行抽取出的全部证据（缺失用 ``None`` 表达，§2.2 None 语义）。"""

    tier_id: str
    exit_code: int | None
    timed_out: bool
    log_text: str
    tuner_type: str
    optimizer_steps: int | None
    epochs_completed: float | None
    weight_changed: bool | None
    checkpoint_reloadable: bool | None
    artifacts_present: Mapping[str, bool]
    losses: tuple[float, ...]
    grad_norms: tuple[float, ...]
    # ── 「训练步真推进」的逐步证据（F-4 P0 修法③）─────────────────────────────
    #: 每个训练步实际执行的 optimizer step 数（全局口径，来自 rank_metrics 旁路
    #: 的 ``global_optimizer_steps_sum``）。PP>1 下 rank0 局部恒为 1、全局和各 rank
    #: 的 ``trainable_norm_after`` 不受此字段影响，因此这是唯一能抓住
    #: 「梯度被跳过、权重冻结」的逐步读数。缺失用 ``None``（§2.2）。
    optimizer_steps_per_step: tuple[int, ...] | None = None
    #: 被判定为非有限而**跳过优化器步**的累积次数（全局口径，来自
    #: ``skipped_nonfinite``）。> 0 即证明本次运行发生过"权重冻结"的步。
    #: **三态**（A7 用）：``int`` = 有读数；``None`` = **该后端不上报且推断不出**
    #: ⇒ A7 记「口径不可测」，**绝不自动通过**（2026-09-22 指挥官裁定）。
    nonfinite_skips: int | None = None
    #: 该计数的**来源自证**（§2.2）：native 精确计数 / ms-swift 从 NaN grad_norm 推断 /
    #: 空 = 不可得。
    nonfinite_skips_source: str = ""
    #: ★ **A7 三分类之 (A)**（2026-09-22 指挥官裁定）：该 run 走的路径**遇到非有限会中止**
    #: ⇒ "**跑完且 rc=0**"本身就是"没发生非有限跳过"的**正面证据**。
    #: 值 = **该档实测 stdout 证明其确实走这条路径** + `文件:行` 锚点（**逐档核，不按后端整批推定**）。
    nonfinite_fail_closed_guarantee: str = ""
    # ── 「读到了什么」与「什么都没读到」的分界（F-4 P0 修法④）───────────────
    #: loss 序列里**真读到**非有限值（NaN/Inf）⇒ 数值异常（关于训练的**事实**）。
    losses_nonfinite: bool | None = None
    #: loss 字段**无值**（``loss: null`` / 缺失）⇒ 证据缺口，判"未分类"而非
    #: "数值异常"。断言后者是假陈述（我们根本没读到数），会把排查引向数值问题。
    losses_unavailable: bool | None = None
    #: 本次运行**计划**跑完的 optimizer step 数（ms-swift ``logging.jsonl`` 的
    #: ``global_step/max_steps`` 分母；native 侧无此读数 ⇒ ``None``）。
    #: A2 用它做"训练步真推进"的后端无关断言：实际步数必须达到计划步数。
    steps_declared_total: int | None = None
    #: collector 是否**定位到了产物根**。用于 A5：产物根找不到 ⇒ 判据是取证缺口；
    #: 产物根找到了但四件套不齐 ⇒ 那是"产物没落盘"这个事实，判失败。
    output_located: bool = True
    #: ``losses[0]`` 对应的**训练步号**（来自 trainer_state/logging 的 ``log_history[*].step``；
    #: rank_metrics 旁路的第一行即 step 1）。A4 的**零容差子检查**只在能确认
    #: "两端都确实是首步"时生效（``== 1``）——拿不到就如实声明"未适用"，不假装做过。
    first_logged_step: int | None = None
    #: A2 的**正式门槛**（最少 optimizer step），来自清单
    #: ``tiers[*].acceptance.formal_gate.min_optimizer_steps``（§1.4 单一真相源）。
    #: ``None`` = 清单未给 ⇒ 回落到 :data:`MIN_OPTIMIZER_STEPS`（**不放宽缺省**）。
    #: ``<= 0`` 是**非法门槛**（"最少步数"没有 0/负数的合法解释，§2.2）⇒ A2 判否
    #: （fail-closed），**绝不**静默回落到 5——静默回落会把"清单被改坏"伪装成正常。
    min_optimizer_steps: int | None = None
    # ── "该档最多能跑出几步"（AF1 §⑥ 方案 S1/S2，2026-09-21）────────────────
    #: 该档**每 epoch** 可产出的 optimizer 步数上限，来自清单
    #: ``tiers[*].expected_optimizer_steps_per_epoch``。**只用于台账自证**（让读者看到
    #: "上限为什么是这个数"），不参与任何断言。
    expected_optimizer_steps_per_epoch: int | None = None
    #: 该档**整个 run** 可产出的 optimizer 步数上限（= 每 epoch 上限 × ``max_epochs``），
    #: 来自清单 ``tiers[*].expected_optimizer_steps_reachable``。
    #:
    #: ★ **它是 A2 的"结构可达性"前置判定的唯一输入**（§1.4）：当
    #: ``reachable < min_optimizer_steps`` 时，"步数少于门槛"这件事**结构上必然发生**，
    #: 把它记成「❌ 训练失败」就是把**口径债**误报成**能力失败**（AF1 诊断的
    #: ``T030``/``T036``/``T042``：4 卡时 ``20÷4=5`` 条/rank × ``G=8`` = 40 条 experience
    #: < 阈值 ``queue×group=64`` ⇒ 阈值永不触发、``max_epochs=1`` ⇒ 全 run 最多 1 步）。
    #:
    #: ``None`` = 清单没给（老清单 / 非矩阵档）⇒ **判定退回原样**：门槛照常硬判
    #: （fail-closed：拿不到"结构上不可达"的证据，就不给它开第三态）。
    expected_optimizer_steps_reachable: int | None = None
    # ── 证据来源自证（§1.4 单一真相源 / §2.2 显式即防呆）─────────────────────
    #: A2「权重真变化」这条证据**来自哪里、是哪种等级**（如
    #: ``run_metrics:lora_norm_delta`` / ``checkpoint:torch_load(lora_b)``）。
    #: 为什么必须有它：同一份 run 产物在**有 torch 的宿主**与**无 torch 的宿主**上
    #: 证据落点不同（228 实测无 torch）。没有这一列，"采集机缺 torch 造成的取证缺口"
    #: 与"训练真失败"在台账上长得一模一样 —— T010 真机就是这样被记成 ❌ 失败的。
    #: ``None`` = 没有取到任何权重证据（与 ``weight_changed is None`` 同源）。
    weight_evidence_source: str | None = None
    #: A2 权重证据的**人读明细**（读到了什么 / 为什么没读到）。
    weight_evidence_detail: str = ""
    #: A3「checkpoint 可重载」这条证据的**等级与来源**（如
    #: ``container_torch_probe`` / ``host_torch_load`` / ``structural_only:zip_crc``）。
    #: 弱证据等级（``structural_only``）**不改变** ``checkpoint_reloadable`` 的值
    #: （仍是缺口 ⇒ A3 判否），只让"为什么明明是缺口"在台账里可解释、不被误读成训练失败。
    reload_evidence_source: str | None = None
    #: ★ **A3 证据来源的机器可核取值域**（2026-09-23 裁定）：
    #: `native_probe` / `hf_shard_probe`（**内容级**，可支撑 A3=True）/
    #: `safetensors_header` / `skipped` / `missing`（**弱/缺，一律不得支撑 A3=True**）。
    a3_source: str = ""
    #: A3 重载证据的人读明细。
    reload_evidence_detail: str = ""
    # ── A4 分档标定所需要的档族坐标（五要素第 1 条 + J2/P1）──
    # 键 = backend × cards × algorithm × model × mode。
    #: 训练后端（``native`` / ``ms-swift``）。``None`` = 这条证据不知道后端。
    #: ★ 为什么档族坐标随**证据**走、而不是当参数传（§1.4 单一真相源）：容差属于
    #: "这一档是什么硬件/栈/算法"这个事实，事实的权威来源是清单；把它从清单一路
    #: 透传到 RunEvidence，A4 的取数就只有一个入口，不可能出现"调用方另喂一个容差"。
    backend: str | None = None
    #: 卡数（清单 ``cards``）。``None`` = 未知。
    cards: int | None = None
    #: 算法（清单 ``algorithm``，如 ``SFT`` / ``GRPO``）。``None`` = 未知。
    #: ★ 判定器**不假设**调用方已归一：取数时会经
    #: :func:`normalize_a4_algorithm` 把 ``GRASPO`` 映射到 ``GRPO``（J5 裁定）。
    algorithm: str | None = None
    #: 档族键的第四、第五维（J2/P1 裁定，2026-09-22）：``model``（如 ``9B``/``27B``）
    #: 与 ``mode``（``LoRA``/``全量``）。``None`` = 未知 ⇒ 只允许命中**通配行**
    #: （fail-closed；不得用某个精确 model 的行去充数）。
    #: 为什么必须随证据走：同族 9B 与 27B 的实测漂移差 4.7×–17.5×，用一条容差覆盖两者
    #: 会把 9B 的隔离带吃光（见 :attr:`A4TierCalibration.model`）。
    model: str | None = None
    mode: str | None = None
    # ── A4 独立通道（五要素第 5 条 + 锁三）：权重层指纹 ──────────────────────
    #: **末步** checkpoint 的权重文件内容 sha256（十六进制小写）。末步 = 与
    #: ``optimizer_steps`` 同名的那一个（``checkpoint-<step>`` / ``final``），
    #: 因此不会把中间 ckpt 当末态（§1.4：末步的权威定义只有一个）。
    #: ``None`` = 取不到（无 ckpt / 无权重文件 / 读失败）⇒ 双通道退化为单通道，
    #: 台账必须如实标注"另一通道无证据"（**不得**当成两通道都通过）。
    final_ckpt_sha256: str | None = None
    #: 该指纹的人读明细（取自哪个文件、为什么没取到）。
    final_ckpt_sha256_detail: str = ""
    #: 该指纹的**来源标识**（如 ``file_sha256:<relpath>``）。
    final_ckpt_sha256_source: str | None = None
    # ── A4 形态①的"独立性"输入（2026-09-22 裁定 2）─────────────────────────
    #: 该 run 的**运行根**（绝对路径）。两跑不同 ⇒ 一条独立证据。
    run_root: str | None = None
    #: 该 run **产物根**的文件系统身份 ``"st_dev:st_ino"``。两跑不同 ⇒ 一条独立证据，
    #: 且能直接识破"两个路径指向同一实体目录"（符号链接/重复挂载）。
    output_identity: str | None = None
    #: 该 run **首条带 timestamp 的逐步记录**（ISO8601）。它是**内容级**证据：
    #: 把同一份产物拷到另一个路径再比一次，时间戳不会变 ⇒ 这条**不会**给独立性加分，
    #: 从而堵住"拷贝造伪独立"的洞。取不到 ⇒ ``None``（不猜）。
    first_metric_timestamp: str | None = None


#: 台账状态（唯一真相源）。三态而非两态的理由见 :class:`FailureClass` 的
#: ``EVIDENCE_GAP`` 条目与 :func:`judge_tier`：取证缺口既不是通过，也不是训练失败。
#: ★ **主判据**（决定 ✅/❌，2026-09-22 用户口径 = "能不能拿来训练"）：
#: `A1`=进程正常退出、`A2`=跑满预定步数/epoch 且权重确实被更新、
#: `A3`=checkpoint 可重载、`A5`=训练产物齐全、`A6`=数值健康（无 NaN/Inf）、
#: **`A7`=训练真推进（没有任何一步因非有限梯度被跳过）**。
PRIMARY_CRITERIA: tuple[str, ...] = ("A1", "A2", "A3", "A5", "A6", "A7")

#: ★ **诊断判据**（**保留计算与记录，但不参与 ✅/❌**，2026-09-22 用户纠正）：
#: `A4`=同配置双跑一致性（含首步零容差 / 终态容差 / 档族标定 / 运行独立性四个子检查）。
#: 为什么只降级 A4：用户原话"你的判定标准应该是**能顺利跑完一个 epoch**……我们现在看的
#: **不是效果，是训练可用**"——双跑一致性是**可复现性**信号，不是"能不能训练"。
#: 它仍是有价值的诊断数据（例如"这个族在硬件上的正常漂移有多大"）。
DIAGNOSTIC_CRITERIA: tuple[str, ...] = ("A4",)


#: ★ **2026-09-22 用户纠正口径后的状态词**（唯一的两个结论态）：
#: 用户原话："你的判定标准应该是**能顺利跑完一个 epoch**……我们现在看的**不是效果，是训练可用**。"
#: ⇒ `✅ 可用` = 能拿来训练（跑完预定步数/epoch + 正常退出 + 产物齐全 + 数值健康）；
#:    `❌ 不可用` = 跑不完（崩 / 挂死 / 环境缺陷 / 数值崩坏）。
LEDGER_PASS = "✅ 可用"
LEDGER_FAIL = "❌ 不可用"
#: `⚠ 口径不可测` **只在真正"跑不完或跑不了"时使用**（结构性步数上限 < 门槛、缺配方、
#: 取证缺口导致主判据无法判定）——**不再用于"A4 没标定"**（A4 已降级为诊断字段）。
LEDGER_INDETERMINATE = "⚠ 口径不可测"
#: ★ **第四态**（AF1/指挥官裁定，2026-09-21）：该档的 step 门槛**结构上不可测**——
#: 不是通过、不是训练失败、也不是取证缺口。
#: 为什么必须与 :data:`LEDGER_FAIL` 分开：AF1 诊断的 9 档 native GRASPO 里，
#: ``T030``/``T036``/``T042`` 用这个 config **无论跑多久都最多 1 步**（阈值
#: ``queue×group`` 在 4 卡下永不可触发）；把它们的"步数 < 门槛"记成「❌ 失败」，
#: 读者会读成"这档能力不行"，而事实是"**这档压根没被真正测过**"。
#: 方向不变：它**不是** ✅，也**不放宽**门槛（``min_optimizer_steps`` 一个字没动）。
LEDGER_GATE_NOT_APPLICABLE = "⚠ 口径不可测"   # 与 LEDGER_INDETERMINATE 同字：都是"测不了"，理由写在 note
#: ★ **当前无配方、将来可能做得了**（blocked）——**不得**与"逻辑上不适用"或"还没跑"混同
#: （2026-09-22 指挥官裁定：用户专门问过这几个词的区别，塌成一个词是信息量倒退）。
#: 判据来源必须可查：`samples/configs/matrix54/<T>.blocked.md`。
LEDGER_NO_RECIPE = "⛔ 无配方"
#: ★ **逻辑上不适用**（not_applicable）。判据来源：`samples/configs/matrix54/<T>.not_applicable.md`。
LEDGER_NOT_APPLICABLE = "⛔ 不适用"
#: **尚未纳入跑批**。
LEDGER_UNTESTED = "— 未测"

#: ★ **状态词全集（6 个，唯一真相源）**：任何产出都不得出现第 7 种白名单外的词。
LEDGER_STATUS_PHRASES: tuple[str, ...] = (
    LEDGER_PASS,
    LEDGER_FAIL,
    LEDGER_INDETERMINATE,
    LEDGER_NO_RECIPE,
    LEDGER_NOT_APPLICABLE,
    LEDGER_UNTESTED,
)


@dataclass(frozen=True, slots=True)
class TierJudgement:
    """一档的最终判定，可直接落成台账记录。"""

    tier_id: str
    criteria: tuple[CriterionResult, ...]
    passed: bool
    failure_class: FailureClass | None
    counts_toward_max_context: bool
    note: str
    #: 本次 A2 实际使用的**正式门槛**（清单值或缺省值；门槛非法时为 ``None``）。
    #: 落台账用（§1.4）：让"这一档按几判的"可审计——合成档与正式档的门槛不同，
    #: 台账不写清就会被下游误读成"同一把尺子"。
    a2_step_threshold: int | None = None
    #: A2/A3 的**证据来源自证**，从 :class:`RunEvidence` 原样透传到台账（§1.4）。
    #: 只加字段、不改任何判据语义：``None`` = 本次没取到该条证据的来源标识。
    weight_evidence_source: str | None = None
    reload_evidence_source: str | None = None
    #: ★ A3 证据来源的机器可核取值域（随台账落盘；§1.4：让"为什么 A3 不可判定"可机核）。
    a3_source: str = ""
    #: ★ **A2 的 step 门槛是否结构上不可测**（AF1/指挥官裁定，2026-09-21）。
    #: ``True`` ⇒ 该档的"步数 < 门槛"**不是**训练失败，而是取数口径所限 ⇒
    #: :attr:`ledger_status` 落 :data:`LEDGER_GATE_NOT_APPLICABLE`（**不是** ❌、**不是** ✅）。
    #: 只在"由清单 ``expected_optimizer_steps_reachable`` 可证"时为 ``True``
    #: （见 :func:`resolve_step_gate_applicability`）——真失败（该跑 100 步只跑 3 步）
    #: 仍然判 ❌。默认 ``False`` = 门槛照常判（**不放宽任何现有档位**）。
    step_gate_not_applicable: bool = False
    #: 该档可产出步数上限（清单值原样透传，§1.4）⇒ 台账能回答"这一档最多几步"。
    expected_optimizer_steps_reachable: int | None = None
    #: 本次 A4 是否启用**双通道强制口径**（见 :func:`judge_tier`）。落台账用（§1.4）：
    #: 同一档在两种口径下可能得到不同结论，台账不写清就会被下游误读成"同一把尺子"。
    a4_require_dual_channel: bool = False
    #: ★ **主判据**（决定 ✅/❌，2026-09-22 用户口径 = 训练可用性）；见 :data:`PRIMARY_CRITERIA`。
    primary_criteria: tuple[str, ...] = PRIMARY_CRITERIA
    #: ★ **诊断判据**（仍计算/仍记录，**不参与判定**）；见 :data:`DIAGNOSTIC_CRITERIA`。
    diagnostic_criteria: tuple[str, ...] = DIAGNOSTIC_CRITERIA
    #: **主判据**是否只因"读不到证据"而未满足（决定是否落「⚠ 口径不可测（取证缺口）」）。
    #: 只看主判据：A4 的取证缺口**不再**制造第三态。
    primary_evidence_gap: bool = False
    #: ★ **除 A2 以外的主判据是否全部通过**——第三态（步数口径不可测）的**唯一**放行条件。
    #: 为什么必须有它（2026-09-22 实测回归）：简化 `ledger_status` 时若只看
    #: ``step_gate_not_applicable``，T035/T036（A1 exit=1 **真失败** + A2 口径不可测）
    #: 会被误标成「⚠ 口径不可测」——**第三态掩盖真失败**。真失败优先。
    primary_others_ok: bool = False

    @property
    def indeterminate(self) -> bool:
        """未通过是否**完全**由取证缺口造成（没有任何判据被证据否定）。

        ★ 第三态（结构上不可测）**不算**取证缺口：它是"这档测不了这件事"，
        不是"这次没取到证据"。两者在台账上必须分开（AF1/指挥官裁定）。
        """
        # ★ 2026-09-22 口径变更：只看**主判据**的取证缺口。
        #   旧实现遍历全部六条判据 ⇒ A4 未标定/无指纹会把档拖成"取证缺口"，
        #   而 A4 已降级为诊断，**不得**再参与状态判定。
        return self.primary_evidence_gap


    @property
    def ledger_status(self) -> str:
        if self.passed:
            return LEDGER_PASS
        if self.indeterminate:
            return LEDGER_INDETERMINATE
        # ★ A2 门槛结构上不可测：**仅在"除它以外全部通过"时**落第四态。
        #   为什么要求"其余全过"：第三态只说"步数这条判据测不了"，**不替其他判据说话**
        #   —— 若 A1/A3/A4/A6 有真失败，仍是「❌ 失败」（真失败绝不被第三态掩掉）。
        # ★ 第三态只在"**除 A2 以外的主判据全部通过**"时成立（守卫由 judge_tier 计算）。
        #   少了这个守卫，T035/T036 那种"真失败 + A2 口径不可测"会被误判为口径不可测。
        if self.step_gate_not_applicable and self.primary_others_ok:
            return LEDGER_GATE_NOT_APPLICABLE
        return LEDGER_FAIL

    def criterion(self, name: str) -> CriterionResult:
        for result in self.criteria:
            if result.criterion == name:
                return result
        raise KeyError(name)


# ── A1–A6 ───────────────────────────────────────────────────────────────────


def judge_a1(evidence: RunEvidence) -> CriterionResult:
    """A1 进程成功：exit=0 且未超时。"""
    if evidence.timed_out:
        return CriterionResult("A1", False, "运行超时（timeout），非正常退出")
    if evidence.exit_code is None:
        return CriterionResult(
            "A1", False, "缺少 exit_code 证据（fail-closed）", evidence_missing=True
        )
    if evidence.exit_code != 0:
        return CriterionResult("A1", False, f"exit={evidence.exit_code} != 0")
    return CriterionResult("A1", True, "exit=0")


def resolve_min_optimizer_steps(raw: int | None) -> tuple[int | None, str | None]:
    """由清单值解析 A2 的正式门槛：``(门槛, 拒绝原因)``。

    纯函数、零设施依赖——判定链的每一层都能复用它，口径只有这一处（§1.4）。

    三种输入，三种**互不混淆**的产出（§2.2 空值语义 + §2.3 边界校验即防呆）：

    ===========================  ======================  ==========================
    清单 ``min_optimizer_steps``  产出                    语义
    ===========================  ======================  ==========================
    ``None``（未提供）            ``(5, None)``           回落到缺省值（**不放宽**）
    ``>= 1`` 的整数               ``(n, None)``           按清单判（唯一真相源）
    ``<= 0`` / 非整数             ``(None, 原因)``        **非法门槛** ⇒ A2 判否
    ===========================  ======================  ==========================

    为什么 ``<= 0`` 必须判否而不是回落到 5：把 0/负数静默当"未提供"，等于用一个
    **非法配置**去换一个**宽松结论**的前置条件；"最少步数 ≥ 0"在逻辑上也永远为真
    ⇒ 那才是真正的放松判据。真实数字与缺省值必须可区分。
    """
    if raw is None:
        return MIN_OPTIMIZER_STEPS, None
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, (
            f"清单里的门槛 min_optimizer_steps={raw!r} 不是整数（fail-closed：不猜执法意图）"
        )
    if raw <= 0:
        return None, (
            f"清单里的门槛 min_optimizer_steps={raw} 非法（必须 ≥ 1；"
            "「最少步数」没有 0/负数的合法解释）⇒ fail-closed 判否，"
            "**不**静默回落到缺省值"
        )
    return raw, None


#: ★ A2「门槛结构上不可测」的**识别标记**（唯一真相源，§1.4）：写在
#: :attr:`CriterionResult.detail` 的**开头**。:func:`judge_tier` 靠它把第三态从
#: "阻断项"里移出单列；采集层/台账读者也靠它在文本里一眼分辨
#: "⚠ 口径不可测" 与 "❌ 训练失败"。
#: 与生成器侧 ``generate_matrix.STEP_GATE_NOT_APPLICABLE_STRUCTURAL`` 是**同一件事的
#: 两侧**：清单标"该档门槛不适用"，判据据此产出第三态（防漂移测试见
#: ``tests/core/test_result_judge_threshold.py``）。
STEP_GATE_NOT_APPLICABLE_MARKER = "⚠ 口径不可测"


def _reachable_suffix(evidence: RunEvidence) -> str:
    """台账文本里补一句"该档可产出上限是多少"（§2.2 显式即防呆）。

    AF1 的核心担忧不是"步数少"，而是"**步数少这件事被读成能力不足**"。因此在
    A2 的任何步数结论里都把上限写在明面上：读者不需要去翻别的字段就能看到
    "上限就是 N，实测就是 N"。
    """
    if evidence.expected_optimizer_steps_reachable is None:
        return ""
    return (
        f"；该档可产出步数上限={evidence.expected_optimizer_steps_reachable}"
        f"（每 epoch 上限={evidence.expected_optimizer_steps_per_epoch}）"
    )


def resolve_step_gate_applicability(
    evidence: RunEvidence, threshold: int
) -> tuple[bool, str]:
    """A2 的 step 门槛是否**结构上不可测** ⇒ ``(是否不适用, 理由)``。

    纯函数、零设施依赖，口径只有这一处（§1.4）。

    判定（**"不可达"必须由清单证明，不靠推测**）::

        reachable 为 None            ⇒ 不适用=False ⇒ **门槛照常硬判**（fail-closed）
        reachable >= 门槛            ⇒ 不适用=False ⇒ 门槛照常判（现有档位一字不改）
        reachable <  门槛            ⇒ 不适用=True  ⇒ 第三态「⚠ 口径不可测」

    ★ **为什么不会放过真失败**（本函数只做一件事：判断"这道题可不可能考"）:

    1. 它**排在所有实质断言之后**（``judge_a2`` 里）：非有限跳过、逐步
       ``optimizer_steps_per_step <= 0``、``steps_declared_total`` 未达标——
       这些是**关于这次训练的事实**，先于"门槛可不可能达到"被判；
    2. 它**只**在清单显式给出 ``reachable < 门槛`` 时触发。老清单 / 非矩阵档
       （``reachable is None``）⇒ **退回原样硬判**，不为任何档开第三态；
    3. ``reachable`` 本身是**上限**（AF1 推导：``floor(P/Q)+1``，``P = subset÷cards``），
       不是实际步数 ⇒ 拿"实际上限"去比门槛，天然保守；
    4. 第三态**不是 ✅**：``passed`` 仍为 ``False``，"这档的步数门槛没有被满足"
       这个事实照样写在台账里（只是分类从"训练失败"改成"口径不可测"）。
       ``TierJudgement.ledger_status`` 只在"除 A2 以外全部通过"时才给第四态；
       其它判据（A1/A3/A4/A6）有真失败 ⇒ 仍然「❌ 失败」。
    """
    reachable = evidence.expected_optimizer_steps_reachable
    if reachable is None:
        return False, (
            "清单未给 expected_optimizer_steps_reachable ⇒ 无结构性证据，门槛照常硬判（fail-closed）"
        )
    if isinstance(reachable, bool) or not isinstance(reachable, int) or reachable < 1:
        return False, (
            f"清单里的 expected_optimizer_steps_reachable={reachable!r} 非法"
            "（必须为 ≥1 的整数）⇒ 不据此开第三态，门槛照常硬判（fail-closed）"
        )
    if reachable >= threshold:
        return False, (
            f"可产出步数上限={reachable} ≥ 门槛 {threshold} ⇒ 门槛适用，照常判"
        )
    return True, (
        f"该档可产出步数上限={reachable} < 门槛 {threshold}"
        f"（每 epoch 上限={evidence.expected_optimizer_steps_per_epoch}）"
        "⇒ 用本档 config 无论跑多久都达不到门槛，属**取数口径所限**"
        "（子集取数 × 训练器步进语义），**不是**训练失败；"
        "门槛值本身未放宽，改判需先调档位定义（属目标/预算层）"
    )


def judge_a2(evidence: RunEvidence) -> CriterionResult:
    """A2 训练步真推进 且 权重真变化（LoRA / 全参两套判据）。

    - **非有限跳过**：``nonfinite_skips > 0`` 直接不通过。这是 F-4（2026-09-18）
      实测缺陷的收口：一次 run 建了 final checkpoint、exit=0，但从第 3 步起
      ``optimizer_steps=0``、权重逐位冻结，逐 rank ``grad_norm`` 却全是 finite
      （26.25 / 1.03e7 / 0.0 / 0.0）。没有这条断言，判据层会把"坏 run"记成
      "权重已变化"。
    - **逐步断言**：``optimizer_steps_per_step`` 中任一步 ``<= 0`` 即不通过——
      "每个训练步都必须真的推进了优化器"。
    - **计划步数断言**（ms-swift 侧的等价证据）：``steps_declared_total`` 可用时，
      实际 ``optimizer_steps`` 必须**达到**计划步数。ms-swift 不产 ``rank_metrics``
      逐步旁路（实测 ``series_source: none``），但它自己的 ``logging.jsonl`` 逐步记
      ``global_step/max_steps``；实际步数 < 计划步数即证明有步没推进。**判据语义与
      native 的逐步断言一致，不是为后端放松**（native 无此读数时该断言自然跳过）。
    - 步数：``optimizer_steps >= 门槛``，门槛来自清单
      ``acceptance.formal_gate.min_optimizer_steps``（§1.4 单一真相源；清单未给 ⇒
      回落到 :data:`MIN_OPTIMIZER_STEPS`；清单给了非法值（``<=0``/非整数）⇒ fail-closed
      判否，见 :func:`resolve_min_optimizer_steps`）；
    - LoRA：至少一个 ``lora_b`` 权重非零（LoRA B 初始为 0，非零即证明被更新）；
    - 全参：终态权重与基座权重不同（由抽取层给出 ``weight_changed``）。

    检查顺序（为什么"跳过"排在门槛之前）：门槛报"步数不够"会把一次**数值崩坏**
    误读成"冒烟太短"，从而诱导重复跑而不是修缺陷。跳过是更硬的事实。
    """
    # ★ 2026-09-22：`nonfinite_skips > 0` 这条断言**已提升为独立主判据 A7**
    #   （见 :func:`judge_a7`）——同一结论不再写两处（§1.4 单一真相源）。
    if evidence.optimizer_steps_per_step is not None:
        stalled = [
            index for index, count in enumerate(evidence.optimizer_steps_per_step) if count <= 0
        ]
        if stalled:
            return CriterionResult(
                "A2",
                False,
                f"训练未真推进：第 {', '.join(str(i + 1) for i in stalled[:5])} 步"
                f"optimizer_steps=0（共 {len(evidence.optimizer_steps_per_step)} 步；"
                "梯度被跳过 ⇒ 该步权重冻结）",
            )
    if evidence.optimizer_steps is None:
        return CriterionResult(
            "A2", False, "缺少 optimizer step 证据（fail-closed）", evidence_missing=True
        )
    if (
        evidence.steps_declared_total is not None
        and evidence.optimizer_steps < evidence.steps_declared_total
    ):
        return CriterionResult(
            "A2",
            False,
            f"训练未真推进：计划 {evidence.steps_declared_total} 个 optimizer step，"
            f"实际只推进了 {evidence.optimizer_steps} 个（训练器自报的 "
            "global_step/max_steps 未跑到分母）",
        )
    # ★ 门槛来自**清单**（``acceptance.formal_gate.min_optimizer_steps``，§1.4）。
    #   检查顺序：计划步数断言（更硬的事实：训练器自报的 global_step/max_steps
    #   没跑到分母）先于门槛，与"跳过先于门槛"同一理由。
    threshold, rejected = resolve_min_optimizer_steps(evidence.min_optimizer_steps)
    if rejected is not None:
        return CriterionResult("A2", False, rejected)
    assert threshold is not None  # rejected is None ⇒ threshold 必非 None
    # ★ **"压根没训练"永远判否**（AF1/指挥官裁定的第一条硬要求）——**排在第三态之前**。
    #   为什么：结构可达性只说"这档**最多**能跑 N 步"；N=1 时"最多 1 步"**不等于**
    #   "0 步也没关系"。0 步是"根本没推进优化器"，是关于这次训练的事实，
    #   优先级高于"这道题可不可能考"。没有这一条，第三态会把最硬的失败吞掉
    #   （负向对照见 tests/core/test_result_judge_threshold.py）。
    #   注：`optimizer_steps is None`（读不到读数）已在上面单独判否，走不到这里。
    if evidence.optimizer_steps == 0:
        source = "清单" if evidence.min_optimizer_steps is not None else "缺省（清单未给门槛）"
        return CriterionResult(
            "A2",
            False,
            "训练未真推进：optimizer_steps = 0（一步都没推进优化器）"
            f"< 门槛 {threshold}（{source}）" + _reachable_suffix(evidence),
        )
    # ★ **结构可达性前置判定**（AF1 §⑥ 方案 S2，2026-09-21）——**在"步数 < 门槛"之前**。
    #   为什么必须前置：这一步不是放宽门槛，而是先回答"这道题本身可不可能考"。
    #   当该档可产出步数上限 < 门槛时，"步数少于门槛"**结构上必然发生**，
    #   把它记成训练失败是**误判能力**（AF1 诊断的 T030/T036/T042）。
    #   ★ 顺序上它排在**所有实质断言之后**（非有限跳过 / 逐步推进 / 计划步数达标 /
    #     0 步）——那些断言是"关于这次训练的事实"，优先级高于"这道题可不可能考"。
    not_applicable, gate_reason = resolve_step_gate_applicability(evidence, threshold)
    if not_applicable:
        # ★ **口径变更（2026-09-22 指挥官裁定）**：`min_optimizer_steps`（默认 5）是**我们
        #   自己外挂的质量门槛**，属"效果/充分性"判断；用户的判据是"**能顺利跑完一个
        #   epoch**"且明确"**不看效果**" ⇒ 把"跑完了的档"记成「口径不可测」正是过度严格。
        #   ⇒ 门槛**退出闸门角色、只保留标注**；改以"**该档 config 的计划步数**"为准。
        #   （`A2` 其它断言——权重确实更新、逐步推进、非有限跳过——**照旧不放宽**。）
        planned = evidence.expected_optimizer_steps_per_epoch
        if planned is None:
            planned = evidence.expected_optimizer_steps_reachable
        steps = evidence.optimizer_steps
        annotation = (
            f"（标注：本档每 epoch 计划步数仅 **{planned}**，属**本期最小配置**；"
            f"原挂门槛 {threshold} 与可达上限 "
            f"{evidence.expected_optimizer_steps_reachable} 两个数字原样保留作对照 —— "
            "该门槛已**退出闸门角色**，不再据此判否）"
        )
        if isinstance(planned, int) and planned > 0 and steps is not None and steps >= planned:
            return CriterionResult(
                "A2", True,
                f"跑满**该档计划步数**：optimizer step={steps} ≥ 计划 {planned}{annotation}",
            )
        return CriterionResult(
            "A2", False,
            f"**未跑满该档计划步数**：optimizer step={steps} < 计划 {planned}{annotation}",
        )
    if evidence.optimizer_steps < threshold:
        source = "清单" if evidence.min_optimizer_steps is not None else "缺省（清单未给门槛）"
        return CriterionResult(
            "A2",
            False,
            f"optimizer step={evidence.optimizer_steps} < 门槛 {threshold}（{source}）"
            + _reachable_suffix(evidence),
        )
    if evidence.weight_changed is None:
        suffix = f"：{evidence.weight_evidence_detail}" if evidence.weight_evidence_detail else ""
        return CriterionResult(
            "A2",
            False,
            f"缺少权重变化证据（tuner_type={evidence.tuner_type}，fail-closed）{suffix}",
            evidence_missing=True,
        )
    if not evidence.weight_changed:
        # ★ 文案必须与**实际证据来源**一致（§2.2 显式即防呆；判定方向不变，仍是"未变化"）。
        # 2026-09-22：全参新增"就地 `trainable_norm_delta`"证据源后，旧文案一律写
        # "全参（与基座一致）"——可那条路径**根本没和基座比过**。把来源写错会把排查
        # 引向"基座比对"，而真因是"run 指标全零"。
        if evidence.tuner_type == "lora":
            mode = "LoRA（lora_b 全零）"
        elif str(evidence.weight_evidence_source or "").startswith("run_metrics:"):
            mode = "全参（run 自产的可训练参数 L2 变化指标全零）"
        else:
            mode = "全参（与基座权重逐字节一致）"
        return CriterionResult(
            "A2", False, f"权重未变化：{mode}（来源：{evidence.weight_evidence_source}）"
        )
    plan = (
        f"，计划 {evidence.steps_declared_total}"
        if evidence.steps_declared_total is not None
        else ""
    )
    return CriterionResult(
        "A2",
        True,
        f"optimizer step={evidence.optimizer_steps}{plan} ≥ 门槛 {threshold}，"
        f"权重已变化（tuner_type={evidence.tuner_type}，"
        f"来源：{evidence.weight_evidence_source}；{evidence.weight_evidence_detail}）",
    )


def judge_a7(evidence: RunEvidence) -> CriterionResult:
    """**A7 训练真推进**：没有任何一步因**非有限梯度**被跳过优化器步。

    为什么单列成一条主判据（2026-09-22 指挥官批准的两级收紧之①）：
    这条断言原本**埋在 A2 的明细里**（`judge_a2` 的第一条分支），虽然确实阻断了，
    但**读者容易忽略**——真实教训：T035（**native** 全参 GRASPO —— 勘误：曾被误记为 ms-swift；见 `T035-verify.yaml` 的 `backend: native`）机制层面的无界挂死/OOM
    被 P6 修好后，"跑完了"很容易被读成"可用"，而它**跳过了 33 次 nonfinite 步**
    （修复后仍有 1 次）+ `grad_norm 7.8e13`。用户要的是"**顺利**跑完一个 epoch"，
    跳过 33 次非有限步不叫顺利。

    三态（与其它判据一致，**不放宽**）：

    - 读到 ``nonfinite_skips > 0`` ⇒ **不通过**（这是关于本次运行的**事实**）；
    - 读到 ``nonfinite_skips == 0`` ⇒ 通过；
    - **读数缺失（``None``）** ⇒ 不通过且标 ``evidence_missing``（fail-closed；
      由 :func:`judge_tier` 归成「⚠ 口径不可测（取证缺口）」而不是训练失败）。
    """
    if evidence.nonfinite_fail_closed_guarantee:
        return CriterionResult(
            "A7", True,
            "由 **fail-closed 机制保证**（跑完且 rc=0 ⇒ 未发生非有限跳过）："
            f"{evidence.nonfinite_fail_closed_guarantee}",
        )
    skips = evidence.nonfinite_skips
    if skips is None:
        return CriterionResult(
            "A7",
            False,
            "**A7 口径不可测**：该后端不上报跳过计数，且无法从逐步读数推断"
            "（fail-closed —— **绝不放行**；同一判据对不同后端必须等价，"
            "见 collector 的 `nonfinite_skips_source`）",
            evidence_missing=True,
        )
    if skips:
        return CriterionResult(
            "A7",
            False,
            f"训练未真推进：累计 **{skips} 次**因非有限梯度跳过优化器步"
            "（权重冻结；即使进程正常退出，也不满足「**顺利**跑完一个 epoch」）",
        )
    source = evidence.nonfinite_skips_source or "（未标注来源）"
    return CriterionResult("A7", True, f"nonfinite 跳过次数 = 0（训练真推进；来源：{source}）")


def judge_a3(evidence: RunEvidence) -> CriterionResult:
    """A3 checkpoint 可被重新加载。

    **证据等级显式化**（§2.2）：明细里必须能看出这条结论来自哪种证据
    （``container_torch_probe`` / ``host_torch_load`` / ``structural_only:zip_crc``）。
    为什么要这样：同一个"缺证据"在台账上可能是**采集机环境**造成的（宿主无 torch），
    也可能是**训练真的没落盘**。不标来源，两者在台账里长得一样，228 上的 T010 就因此
    被记成 ❌ 失败。注意方向不变：**弱证据不升格为通过**，``checkpoint_reloadable is
    None`` 仍然是 fail-closed 判否。
    """
    if evidence.checkpoint_reloadable is None:
        suffix = f"：{evidence.reload_evidence_detail}" if evidence.reload_evidence_detail else ""
        return CriterionResult(
            "A3", False, f"缺少 checkpoint 重载证据（fail-closed）{suffix}", evidence_missing=True
        )
    # ★ **硬守卫（假绿最后一环，2026-09-23 裁定）**：**弱/缺证据一律不得支撑 A3=True**，
    #   即使上游把它抽成了 True 也要在此拦下 ⇒ 落「不可判定」（`evidence_missing`），
    #   **不是 True（假绿）、也不是 False（会被误读成训练失败）**。
    #   注意**方向与范围**：只拦"**弱证据却说通过**"（`checkpoint_reloadable is True`）
    #   —— **`False`（头部不可解析/归档损坏 = 真损坏）必须原样保留为实质失败**，
    #   否则会把"文件真坏了"误记成"测不出来"，那是另一种失真（既有用例守卫）。
    if evidence.checkpoint_reloadable and evidence.a3_source in (
        "safetensors_header", "skipped", "missing",
    ):
        return CriterionResult(
            "A3", False,
            f"A3 证据等级不足（a3_source=`{evidence.a3_source or 'missing'}`）："
            "**仅头部级/SKIP/缺失证据，不构成内容级重载证据** ⇒ 落「不可判定」，绝不放行；"
            "需 `native_probe`（container_torch_probe）或 `hf_shard_probe`（HF 分片内容级审计）",
            evidence_missing=True,
        )
    source = ""
    if evidence.reload_evidence_source:
        source = f"（来源：{evidence.reload_evidence_source}）"
    if not evidence.checkpoint_reloadable:
        detail = f"：{evidence.reload_evidence_detail}" if evidence.reload_evidence_detail else ""
        return CriterionResult("A3", False, f"checkpoint 无法重新加载{source}{detail}")
    return CriterionResult(
        "A3", True, f"checkpoint 重载成功{source}：{evidence.reload_evidence_detail}"
    )


def _a4_first_step_zero_tolerance_check(
    first: RunEvidence, second: RunEvidence
) -> CriterionResult | None:
    """★ **子检查①（零容差）：首步 loss 逐位相同** —— **一个字都不能改**。

    本函数是从 :func:`judge_a4` 里**原样抽出**的（2026-09-21 重构，为了给它单独立测），
    抽出的方式保证：**条件、比较运算、返回值与消息文本与抽出前逐字相同**。

    ★ **为什么它是本轮唯一不能碰的判据**（AD1 §⑤ 结论）：27 条证据里，被它拦下的
    真 bug 差值全部 **≤ 3.3e-3**（T015/T026/T033/T046/T047/T048 差 8.31e-4~3.24e-3，
    GRPO 播种缺陷 3.3e-3），**全都小于任何合理的终态容差** ⇒ 靠终态容差一个都抓不到。
    它是区分"真 bug"与"GPU 内核残余"的**唯一尺子**（宪法 §6 排除项 vs §10.1 破坏）。

    返回 ``None`` = 本条子检查通过（或未适用）；返回 ``CriterionResult`` = 直接判否。
    """
    if first.first_logged_step == 1 and second.first_logged_step == 1:
        first_initial = first.losses[0]
        second_initial = second.losses[0]
        if not isinstance(first_initial, float) or not isinstance(second_initial, float):
            return CriterionResult(
                "A4",
                False,
                "首步 loss 无读数（loss 字段缺失），零容差子检查无法进行（fail-closed）",
            )
        if first_initial != second_initial:
            return CriterionResult(
                "A4",
                False,
                f"首步 loss 不一致（零容差子检查）：{first_initial!r} vs {second_initial!r} "
                "—— 首步只取决于 种子/初始化/数据顺序/首批样本，全是**可控**因素 ⇒ "
                "这是可复现性被破坏的指纹（§10.1），不是 §6 排除的 GPU 内核差异",
            )
    return None


def _a4_first_step_was_checked(first: RunEvidence, second: RunEvidence) -> bool:
    """子检查①是否**实际执行**过（两端都确认是首步）。

    ★ 这个判据必须与 :func:`_a4_first_step_zero_tolerance_check` 的**进入条件逐字一致**
    （``first_logged_step == 1`` 两端）——否则"首步 loss 逐位相同"这句话会在没比过的时候
    被写进台账（§2.2 显式即防呆：不能让台账说一句没验证过的话）。
    """
    return first.first_logged_step == 1 and second.first_logged_step == 1


#: A4 形态①判定"确是两次独立运行"所需的**最少独立证据数**（2026-09-22 裁定 2）。
#: 为什么至少 2：单凭"路径不同"可能只是同一实体目录的两个入口；
#: 必须再有第二个**互不依赖**的身份维度交叉确认。
A4_MIN_INDEPENDENCE_SIGNALS = 2


def _a4_independence_signals(first: RunEvidence, second: RunEvidence) -> tuple[int, str]:
    """统计"两跑是两次**独立**运行"的独立证据条数（裁定 2 的实现）。

    三个互不依赖的维度（任一条单独都不足以定论）：

    1. **运行根不同**（``run_root`` 两端非空且不等）；
    2. **产物根的文件系统身份不同**（``output_identity`` 两端非空且不等，``st_dev:st_ino``）
       —— 这条能直接识破"两个路径指向同一实体目录"；
    3. **首条逐步记录的时间戳不同** —— **内容级**证据；拷贝产物不会改变它。

    **额外硬约束（堵拷贝造伪独立）**：若两端都有时间戳却**相同** ⇒ 直接判"不独立"
    （返回 ``0``），哪怕前两条都满足 —— 因为那正是"同一份产物被复制了一份"的指纹。

    取不到证据就**少记一条**（不猜、不默认独立）：这正是"缺元数据 ⇒ 继续 fail-closed"。
    返回 ``(条数, 人读明细)``。
    """
    signals: list[str] = []
    if first.run_root and second.run_root and first.run_root != second.run_root:
        signals.append(f"运行根不同（{first.run_root} ≠ {second.run_root}）")
    if (
        first.output_identity
        and second.output_identity
        and first.output_identity != second.output_identity
    ):
        signals.append(
            f"产物根文件系统身份不同（{first.output_identity} ≠ {second.output_identity}）"
        )
    stamps_present = bool(first.first_metric_timestamp and second.first_metric_timestamp)
    if stamps_present and first.first_metric_timestamp == second.first_metric_timestamp:
        return 0, (
            f"时间戳相同（{first.first_metric_timestamp}）⇒ 判为**同一份产物的两次比对**"
            "（拷贝造伪独立）⇒ 不给独立性加分"
        )
    if stamps_present:
        signals.append(
            f"首条逐步记录时间戳不同（{first.first_metric_timestamp} ≠ {second.first_metric_timestamp}）"
        )
    detail = (
        "；".join(signals)
        if signals
        else "未能取得任何独立身份证据（元数据缺失，或两跑指向同一实体目录）"
    )
    return len(signals), detail


def judge_a4(
    first: RunEvidence,
    second: RunEvidence | None,
    *,
    require_dual_channel: bool = False,
) -> CriterionResult:
    """A4 同 config 同 seed 双跑一致（**四条并列子检查**，见下）。

    一致性口径（§10.1 复现定义 + §6 排除项）：

    - 两跑都成功、``optimizer_steps`` 相同、``weight_changed`` 结论相同；
    - **子检查①（零容差）**：**首步 loss 必须逐位相同**（``losses[0]`` 在两端
      ``first_logged_step == 1`` 且 ``==`` 严格相等）。
      首步 loss 只取决于 种子 / 初始化 / 数据顺序 / 首批样本——全是**可控**因素，
      因此零容忍：这是"去掉种子""打乱数据顺序""换初始化"这类破坏的**即时指纹**。
      **实现见 :func:`_a4_first_step_zero_tolerance_check`（本轮一字未改）。**
    - **子检查②（分档实测容差）**：终态 loss 之差 ≤ **该档族**的标定容差
      ``backend × cards × algorithm × model × mode`` —— 见 :data:`A4_TIER_CALIBRATIONS`
      （J2/P1 扩维，2026-09-22：同族 9B 与 27B 实测漂移差 4.7×–17.5×，三维键会把
      9B 的隔离带吃光）。算法名先经 :func:`normalize_a4_algorithm` 归一
      （``GRASPO`` → ``GRPO``，J5 裁定）；``model``/``mode`` 未知时只允许命中通配行。
      依据：宪法 §6 明文把"GPU 内核选择等**无法控制**的差异"排除在复现破坏之外；
      容差**按档族**由 n≥5 实测最坏对 ×1.5 推出、并被"最小真 bug 签名 / 2"封顶
      （2026-09-21 裁定，依据 ``task-ad1-a4-validity``；**废止** T013 单档常量的
      10× 全局放大——那会批准 0.34 这种能放过真 bug 0.0935 的值）。
      判据取不到档族标定 ⇒ **fail-closed**（不下发默认容差）。
      ★ 本子检查的容差按**同卡集合**标定，因此被比较的两跑也**必须同卡集合**——AO1 实测：
      跨卡集把最坏对从 `4.0139e-3` 抬到 `1.1014e-2`（方差 ≈2.74×），容差随即失去意义；
      实测依据与出处见 :data:`A4_TIER_CALIBRATIONS` 上方的「同一卡集合」条。
    - **子检查③（符号翻转）**：两跑终态 loss **异号**且差值超过一个容差量级 ⇒ 判否。
      依据：真 bug ``T032`` 的指纹就是终态从 ``−0.0217`` 翻到 ``+0.0718``。
    - **子检查④（独立通道）**：末步 checkpoint 的 ``final_ckpt_sha256`` 与 loss 通道
      **冲突** ⇒ 判「不可判定」，**不判通过也不判失败**。两种冲突形态：
      ① 两跑权重指纹**逐位相同**、而 loss 也逐位相同 ⇒ 末步 loss 是死通道；
      ② ``require_dual_channel=True`` 且两跑 loss 逐位相同、而独立通道**没有证据**
         （两跑都取不到指纹）⇒ 双通道从未一致过（**矩阵采集必须开这个开关**：
         否则"指纹取不到"会静默退化成"单通道通过"，退化通过就漏过去了）。
      注意"两跑权重指纹**不同**"**不是**冲突——那是 GPU 内核残余下的预期现象
      （AD1 §6.2 实测两跑 ckpt sha 不同）。

    **退化通过**（AD1 §④ 的 7 条）：末步是无信号步（``loss 0 / grad_norm 0 /
    group skipped``）或 ``losses`` 长度=1 ⇒ 终态子检查**零鉴别力**。此时 detail 里
    **显式标注**"终态子检查退化，未提供鉴别力"，且**不放行、不改判失败**——它是
    「证据不足」，不是「不通过」（见 :func:`degenerate_final_check_reason`）。

    **诚实性**：拿不到步号（``first_logged_step is None``）或多步累积后无法确认"首步"时，
    子检查①**如实声明"未适用"**，不得假装做过（detail 里写明），此时①不参与判定，
    但②照常生效。缺第二跑/loss 证据 → 不通过（fail-closed）。
    """
    if second is None:
        return CriterionResult("A4", False, "缺少第二跑证据（fail-closed）", evidence_missing=True)
    if first.exit_code != 0 or second.exit_code != 0:
        return CriterionResult(
            "A4", False, f"双跑未都成功（exit={first.exit_code}/{second.exit_code}）"
        )
    if first.optimizer_steps != second.optimizer_steps:
        return CriterionResult(
            "A4",
            False,
            f"optimizer step 不一致：{first.optimizer_steps} vs {second.optimizer_steps}",
        )
    if first.weight_changed != second.weight_changed:
        return CriterionResult("A4", False, "权重变化结论不一致（一次变化一次未变化）")
    if not first.losses or not second.losses:
        return CriterionResult(
            "A4", False, "缺少 loss 序列证据（fail-closed）", evidence_missing=True
        )

    # ── 子检查①（零容差）：首步 loss 逐位相同 ────────────────────────────────
    # ★ 抽出为独立函数（见其 docstring）：本轮**一字未改**。所有返回值原样透传。
    first_step_verdict = _a4_first_step_zero_tolerance_check(first, second)
    if first_step_verdict is not None:
        return first_step_verdict
    first_step_checked = _a4_first_step_was_checked(first, second)
    step_note = (
        "首步 loss 逐位相同；"
        if first_step_checked
        else "首步零容差子检查未适用（拿不到步号或首步无读数）；"
    )

    # ── 子检查②（分档实测容差）：先取档族，再比终态 loss ─────────────────────
    calibration = _resolve_tier_calibration(
        first.backend, first.cards, first.algorithm, first.model, first.mode
    )
    if calibration is None:
        # 取不到档族标定 ⇒ fail-closed（**绝不下发"全局默认容差"**，那正是本包要废止的东西）。
        return CriterionResult(
            "A4",
            False,
            "缺少该档族的终态容差标定（fail-closed）："
            f"backend={first.backend!r} cards={first.cards!r} algorithm={first.algorithm!r} "
            f"model={first.model!r} mode={first.mode!r} "
            "在 A4_TIER_CALIBRATIONS 里没有对应档族 ⇒ 不下发默认容差、不判通过",
            evidence_missing=True,
        )
    tolerance = calibration.tol
    evidence_note = ""
    if calibration.outcome != A4_OUTCOME_CALIBRATED:
        evidence_note = f"；{calibration.note}"

    first_loss = first.losses[-1]
    second_loss = second.losses[-1]
    if not isinstance(first_loss, float) or not isinstance(second_loss, float):
        # 末步 loss 无读数（MISSING 哨兵）⇒ 无法比较，fail-closed。
        # （不让哨兵流进下面的算术：那是 TypeError，会把"判不通过"变成"崩溃"。）
        return CriterionResult(
            "A4", False, "末步 loss 无读数（loss 字段缺失），无法做双跑一致性比较（fail-closed）"
        )
    delta = abs(first_loss - second_loss)
    if delta > tolerance:
        return CriterionResult(
            "A4",
            False,
            f"最终 loss 不一致：{first_loss:.6g} vs {second_loss:.6g}"
            f"（差 {delta:.3g} > 档族容差 {tolerance:.3g}；"
            f"档族 {calibration.backend}/{calibration.cards} 卡/{calibration.algorithm}"
            f"·{calibration.model or '*'}/{calibration.mode or '*'}，"
            f"标定 n={calibration.n}，最坏对 {calibration.worst_pair:.3g}）{evidence_note}",
        )

    # ── 子检查③（符号翻转）：真 bug T032 的指纹 ─────────────────────────────
    if _is_sign_flip(first_loss, second_loss, tolerance):
        return CriterionResult(
            "A4",
            False,
            f"终态 loss 符号翻转（真 bug 指纹）：{first_loss:.6g} vs {second_loss:.6g}"
            f"（差 {delta:.3g}）—— 无 bug 的内核漂移只在原值附近抖动，不会越过 0；"
            "AD1 点名的 T032 真 bug 正是这个形态（−0.0217 → +0.0718）",
        )

    # ── 子检查④（独立通道）：final_ckpt_sha256 与 loss 通道的冲突检查 ────────
    # 「冲突」的定义刻意收得很窄（见 A4_DUAL_CHANNEL_CONFLICT_LEDGER 的注释）：
    # 只有"权重逐位相同、loss 也逐位相同"才说明 loss 通道是无鉴别力的死通道。
    # 权重不同是 GPU 内核残余下的**预期现象**（AD1 §6.2），不构成冲突。
    sha_first = first.final_ckpt_sha256
    sha_second = second.final_ckpt_sha256
    hashes_present = sha_first is not None and sha_second is not None
    hashes_identical = hashes_present and sha_first == sha_second
    # 形态①：指纹有证据且逐位相同 + loss 逐位相同 ⇒ loss 是死通道。
    if hashes_identical and delta == 0.0:
        # ★ 2026-09-22 裁定 2：两通道逐位相同有两种**性质相反**的成因，必须先分开——
        #   ① 同一份产物比了两次（自我比较陷阱）⇒ 真·无鉴别力 ⇒ 继续 fail-closed；
        #   ② 两次**真实独立**运行且逐位相同 ⇒ 这是**最强的复现证据**（确定性复现），
        #      旧实现会永远报"不可判定"，等于把完美可复现判成不可判定（方向性错误）。
        #   分开的判据 = 运行级独立身份证据（≥2 项，见 _a4_independence_signals）。
        signals, independence_detail = _a4_independence_signals(first, second)
        if signals >= A4_MIN_INDEPENDENCE_SIGNALS:
            return CriterionResult(
                "A4",
                True,
                "**完美可复现**（判通过）：两跑末步 checkpoint 逐位相同"
                f"（sha256={sha_first[:12]}…）且末步 loss 逐位相同（{first_loss:.6g}，差 0）——"
                "该形态本是「死通道」的样子，但已用**运行级独立身份证据**确认它们是"
                f"**两次真实独立运行**（{signals} 项 ≥ {A4_MIN_INDEPENDENCE_SIGNALS}："
                f"{independence_detail}）⇒ 逐位相同只能解释为**确定性复现**。"
                "（备注：本判定未改任何容差/门槛；两通道均逐位相同属最强复现证据。）",
            )
        return CriterionResult(
            "A4",
            False,
            "双通道冲突（判「不可判定」，既不判通过也不判失败）："
            f"两跑末步 checkpoint 逐位相同（sha256={sha_first[:12]}…）"
            f"且末步 loss 逐位相同（{first_loss:.6g}，差 0）⇒ 末步 loss 是无鉴别力的"
            "死通道（无信号步/单步）；且**无法证明是两次独立运行**"
            f"（独立身份证据仅 {signals} 项 < {A4_MIN_INDEPENDENCE_SIGNALS}："
            f"{independence_detail}）⇒ 保持 fail-closed。"
            f"{A4_DUAL_CHANNEL_CONFLICT_LEDGER}",
            evidence_missing=True,
        )
    # 形态②（矩阵采集口径）：两跑 loss 逐位相同 ⇒ loss 通道**已证明**无鉴别力；
    # 此时必须有独立通道证据说"权重也不同"才能算两通道一致。指纹两端都取不到
    # ⇒ 双通道从未一致过，如实判「不可判定」（不得静默当单通道通过）。
    if require_dual_channel and delta == 0.0 and not hashes_present:
        return CriterionResult(
            "A4",
            False,
            "双通道未一致（判「不可判定」）：两跑末步 loss 逐位相同"
            f"（{first_loss:.6g}，差 0）⇒ loss 通道**已证明**无鉴别力"
            "（无信号步/单步），而独立通道（final_ckpt_sha256）两跑都取不到证据"
            "⇒ 没有任何一条通道能证明两跑真的可复现。"
            "要恢复 A4 可判性：让末步是有效训练步（loss≠0、步数≥2），"
            "并让采集器取到末步权重指纹。",
            evidence_missing=True,
        )

    # ── 通过：区分"有鉴别力的通过"与"退化通过"（AD1 §④ 标注，不改判定方向）──
    degenerate = degenerate_final_check_reason(
        first.losses, first.grad_norms
    ) or degenerate_final_check_reason(second.losses, second.grad_norms)
    if degenerate is not None:
        return CriterionResult(
            "A4",
            True,
            f"双跑一致（**终态子检查退化，未提供鉴别力**）：{step_note}"
            f"final loss {first_loss:.6g} vs {second_loss:.6g}"
            f"（差 {delta:.3g} ≤ 档族容差 {tolerance:.3g}）"
            f"；退化原因：{degenerate}"
            "—— 本条 A4 ✅ **证据不足**，只等于首步零容差通过，不得引用为终态复现证据",
        )
    return CriterionResult(
        "A4",
        True,
        f"双跑一致：{step_note}final loss {first_loss:.6g} vs {second_loss:.6g}"
        f"（差 {delta:.3g} ≤ 档族容差 {tolerance:.3g}；"
        f"档族 {calibration.backend}/{calibration.cards} 卡/{calibration.algorithm}"
        f"·{calibration.model or '*'}/{calibration.mode or '*'}，"
        f"标定 n={calibration.n}，最坏对 {calibration.worst_pair:.3g}）{evidence_note}",
    )


def judge_a5(evidence: RunEvidence) -> CriterionResult:
    """A5 四件套产物落盘。

    取证缺口的边界：**产物根都没定位到** ⇒ 判据是取证缺口（我们没找到该去哪里看，
    这是 collector 的缺口）；产物根找到了但四件套不齐 ⇒ 那是"产物没落盘"这个事实，
    判**失败**。两者都 ``passed=False``（fail-closed 不变）。
    """
    missing = [name for name in REQUIRED_ARTIFACTS if not evidence.artifacts_present.get(name)]
    if missing:
        detail = f"缺少产物：{', '.join(missing)}"
        if not evidence.output_located:
            detail += "（且未定位到任何产物根 ⇒ 取证缺口，不得记为训练失败）"
            return CriterionResult("A5", False, detail, evidence_missing=True)
        return CriterionResult("A5", False, detail)
    return CriterionResult("A5", True, f"四件套齐全：{', '.join(REQUIRED_ARTIFACTS)}")


def judge_a6(evidence: RunEvidence) -> CriterionResult:
    """A6 数值健康：loss / grad_norm 全程 finite（NaN/Inf 即不通过）。

    **三种情形必须分清（F-4 §④.1-4 的核心）**：

    1. **真读到 NaN/Inf** ⇒ 判**不通过**，明细为「出现 NaN/Inf」⇒ 分类 **数值异常**。
       这是关于训练数值的**事实断言**（F-4 那个 run 的 step2 就是真 NaN）。
    2. **loss 字段无值**（``loss: null`` / 缺失）⇒ 判**不通过**（fail-closed），
       明细为 :data:`NUMERIC_INDETERMINATE_DETAIL` ⇒ 分类 **未分类（需人工判定）**。
       我们**没有读数**，所以不得断言"数值异常"——那会把排查引向数值问题，而真因可能
       是进程被杀、超时或环境问题。这**不是**"偶发地依赖 null 被解析成 NaN"。
    3. 两者同时出现（F-4 实测正是如此）⇒ **真 NaN 优先**，分类数值异常。
       已确证的数值崩坏不得被"某几步没有读数"降格。

    情形 1 与 2 一律**不通过、不计入最大可行上下文**（fail-closed 不变）。

    **「最终 loss 不高于初始 loss」自 2026-09-20 起降为记录项**（见
    :data:`A6_LOSS_TREND_BLOCKS` 的完整裁定依据）：首末两点的走向**不再**使 A6 判否，
    但仍以 ``loss_trend=decreased|increased`` 写进明细文本（§2.2 显式即可见），
    由采集层带进台账。判据自称「数值健康」，而"loss 是否真下降"属**效果**信号，
    应由准确率类指标（能力矩阵 §8）回答。
    """
    if not evidence.losses:
        return CriterionResult(
            "A6", False, "缺少 loss 序列证据（fail-closed）", evidence_missing=True
        )
    if not evidence.grad_norms:
        return CriterionResult(
            "A6", False, "缺少 grad_norm 序列证据（fail-closed）", evidence_missing=True
        )

    # ① 真读到非有限值。类型上必须是 float：MISSING_SENTINEL 是 str，会被这个
    #    isinstance 挡住，不会污染"数值异常"这个事实断言。
    real_nonfinite_losses = [
        value for value in evidence.losses if isinstance(value, float) and not math.isfinite(value)
    ]
    real_nonfinite_grads = [
        value
        for value in evidence.grad_norms
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if evidence.losses_nonfinite or real_nonfinite_losses or real_nonfinite_grads:
        return CriterionResult(
            "A6",
            False,
            f"出现 NaN/Inf：loss {len(real_nonfinite_losses) or 1} 个、"
            f"grad_norm {len(real_nonfinite_grads)} 个",
        )
    # ② 只有"没有读数"（loss: null / 字段缺失 / 哨兵）⇒ 证据缺口，不是数值异常。
    if evidence.losses_unavailable or any(
        value is NAN_SENTINEL or value is MISSING_SENTINEL for value in evidence.losses
    ):
        return CriterionResult("A6", False, NUMERIC_INDETERMINATE_DETAIL, evidence_missing=True)

    # ③ 数值健康（全程 finite）⇒ 首末走向只**记录**，不阻断（见 A6_LOSS_TREND_BLOCKS）。
    #    顺序刻意排在 ①② 之后：真实的数值崩坏/证据缺口不得被"走向正常"盖过。
    initial = evidence.losses[0]
    final = evidence.losses[-1]
    assert isinstance(initial, float) and isinstance(final, float)  # 哨兵已在 ② 排除
    trend = "increased" if final > initial else "decreased"
    trend_note = (
        f"{A6_LOSS_TREND_FIELD}={trend}（初始 {initial:.6g} → 最终 {final:.6g}"
        f"，{len(evidence.losses)} 个样本全程 finite"
    )
    if trend == "increased":
        trend_note += "；★ 首末走向为记录项，不阻断 A6（2026-09-20 裁定，见 A6_LOSS_TREND_BLOCKS）"
    trend_note += "）"
    if trend == "increased" and A6_LOSS_TREND_BLOCKS:
        return CriterionResult("A6", False, f"最终 loss 高于初始 loss：{trend_note}")
    # ── ★ grad_norm **量级指纹**（诊断项，**不单独判否**；2026-09-22 收紧之②）──────
    # 为什么只做诊断：单看量级会**误伤**——T017（native 全参）grad_norm≈8e8、loss 却从
    # 2.6 降到 0.009 且全程 finite；而 T035（**native** 全参，勘误同上）7.8e13 **且**跳过了 nonfinite 步
    # 才是真问题。区分这两者靠的是"**是否伴随**跳过/非有限"，而不是量级本身。
    grad_note = ""
    finite_grads = [
        value
        for value in evidence.grad_norms
        if isinstance(value, float) and math.isfinite(value) and value > 0.0
    ]
    if finite_grads:
        grad_max = max(finite_grads)
        grad_median = statistics.median(finite_grads)
        if grad_median > 0.0:
            orders = math.log10(grad_max / grad_median)
            grad_note = (
                f"；grad_norm 量级指纹：max={grad_max:.3g} / 中位数={grad_median:.3g}"
                f"（相差 {orders:.1f} 个数量级）"
            )
            if orders >= 10:
                grad_note += " ⇒ **⚠ 量级尖峰（诊断项，不单独判否；需结合是否有 nonfinite 跳过判定）**"
    return CriterionResult("A6", True, f"数值健康：{trend_note}{grad_note}")


# ── 失败分类 ────────────────────────────────────────────────────────────────


#: 数值不可判定（loss 字段无值可读）的明细前缀 —— 单一真相源。
#: ``judge_a6`` 写它、``classify_failure`` 认它；两者不得各自拼字符串。
NUMERIC_INDETERMINATE_DETAIL = (
    "数值健康不可判定：loss 序列存在无法解释的取值（loss: null / 字段缺失）"
    "——按 fail-closed 判不通过，不得默认通过"
)

#: A6 明细里出现这些字样 ⇒ 数值异常（唯一真相源，见 ``judge_a6``）。
#: 注意：「不可判定」不在此列——它是 fail-closed 的**证据缺口**，不是数值异常；
#: 真正的非有限梯度由训练器的显式标记走 ``_LOG_PATTERNS`` 分类。
#: **2026-09-20 起不再包含「高于初始」**：首末 loss 走向已降为记录项
#: （:data:`A6_LOSS_TREND_BLOCKS`），``judge_a6`` 不会再产出那个字样，
#: 留着它就是一条永不可达的分支（违 §2.2/§18.1）。真数值崩坏由 "NaN/Inf" 覆盖。
_NUMERIC_ANOMALY_DETAILS: tuple[str, ...] = ("NaN/Inf",)


def _is_nan_sentinel(value: float) -> bool:
    """该值是"字段没有可解释的数值"的 **NaN 哨兵**（而非 loss 本身为 NaN）。

    两者都是 NaN，但语义不同：前者是证据缺口，后者是数值异常。判定链上的优先级
    也不同（见 ``classify_failure``），因此必须用同一个对象在两处识别，避免
    `judge_a6` 与 `classify_failure` 对"同一条 NaN"作出不同解释。

    **用 ``is`` 判对象身份**（``float("nan") is NAN_SENTINEL`` 为 False）：
    数值 NaN 即便是"另一个 NaN 实例"也不会被误判成证据缺口——这正是修复本轮
    回归的关键（提取层只允许用 :data:`NAN_SENTINEL` 表达"读不到值"，日志里真实
    读取到的 NaN **不得**用同一个对象表达）。
    """
    return value is NAN_SENTINEL


def numeric_anomaly(a6: CriterionResult) -> bool:
    """A6 是否因**数值异常**未通过（真读到 NaN/Inf）。

    单独抽成函数是为了给 ``classify_failure`` 一个显式、可测的**优先级判据**：
    数值健康是比"日志里出现 OOM 字样"更硬的事实（见该函数的排序说明）。

    **2026-09-20 变更**：不再把"最终 loss 高于初始"算作数值异常——该子检查已降为
    记录项（:data:`A6_LOSS_TREND_BLOCKS`）。判据的取值面因此变窄，但"真读到非有限值
    ⇒ 数值异常"这条**不变**。
    """
    return not a6.passed and any(mark in a6.detail for mark in _NUMERIC_ANOMALY_DETAILS)


def numeric_indeterminate(a6: CriterionResult) -> bool:
    """A6 是否因**证据缺口**（loss 字段无值）未通过 —— 与"数值异常"区分。

    **可达性**：可达。提取层对 ``loss: null`` / 字段缺失写入
    :data:`MISSING_SENTINEL`（类型是 str ⇒ 不参与"真读到 NaN"的判定），
    ``judge_a6`` 因此进入"只有没有读数"分支，明细为
    :data:`NUMERIC_INDETERMINATE_DETAIL`，本函数返回 ``True``
    ⇒ 分类为 ``UNCLASSIFIED``（需人工判定）。

    为什么必须可达：断言"数值异常"是关于训练数值的**事实陈述**；当我们**只是
    没有读数**时，那个陈述是假的，还会把排查引向数值问题。
    """
    return not a6.passed and NUMERIC_INDETERMINATE_DETAIL in a6.detail


def a6_loss_trend(a6: CriterionResult) -> str | None:
    """从 A6 明细文本里取出 ``loss_trend`` 的记录值（``"decreased"`` / ``"increased"``）。

    存在的理由（§1.4 单一真相源）：A6 明细文本是 ``loss_trend`` 的**唯一权威来源**，
    台账字段只是它的一个视图。与其让采集层另起一套"首末比较"（那就是第二个真相源，
    且迟早与判定器漂移），不如从已经写好、已经判过的文本里原样取出。

    取不到 ⇒ 返回 ``None``（§2.2）。可达情形且都有意义：

    - A6 未通过且是 NaN/Inf 或证据缺口 ⇒ 明细里没有走向记录（当时无从比较）；
    - 用到本函数的是采集层，**不是**判定路径——它不参与任何 passed 计算。
    """
    marker = f"{A6_LOSS_TREND_FIELD}="
    start = a6.detail.find(marker)
    if start < 0:
        return None
    rest = a6.detail[start + len(marker) :]
    for known in ("increased", "decreased"):
        if rest.startswith(known):
            return known
    return None


def _mean_finite(values: Sequence[float | None]) -> float | None:
    """逐 rank 序列求均值，**保留**非有限值（通信层/日志层共用的一致口径）。

    为什么不用 ``sum(...)/len(...)`` 之外的任何"过滤"：过滤 NaN 会把
    "一个 rank 崩了"洗成"健康"——F-4 实测的 step2 正是 r0/r1 有限、r2/r3 NaN，
    一旦过滤，全局读数就变成有限值。``None``（该 rank 本步无读数）被跳过；
    全为 ``None`` 时返回 ``None``（而非 0.0，§2.2）。
    """
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def classify_failure(evidence: RunEvidence, a6: CriterionResult) -> FailureClass | None:
    """把一次未通过的运行归类。

    返回 ``None`` 表示"没有失败"（exit=0、无 NaN、非超时）。匹配不到则返回
    ``UNCLASSIFIED``（不计入最大可行上下文）。

    **判定优先级序列（唯一真相源，改动此处必须同步更新本段说明与测试）**：

    0. **退出码映射**（:data:`EXIT_CODE_FAILURE_CLASSES`，2026-09-22 新增）：
       ``86`` ⇒ ``pipeline_p2p_timeout``、``21`` ⇒ ``pp_rollout_wall_clock_cap``。
       排在文本匹配**之前**——异常文本常被 torchrun 前缀化，退出码才是可信事实。
       （它与第 1 步的 ``124`` 不冲突：不同的码表达不同的事。）
    1. **超时**（``timed_out`` / ``exit_code == 124``）⇒ ``TIMEOUT``；
    2. **数值异常**（A6 明细含 "NaN/Inf"）⇒ ``NUMERIC_ANOMALY``；
    3. **数值不可判定**（A6 明细含 :data:`NUMERIC_INDETERMINATE_DETAIL`，即
       ``loss: null`` / 字段缺失 / 全程无读数）⇒ ``UNCLASSIFIED``（需人工判定）；
    4. **显式非有限梯度标记**（训练器硬失败的 ``非有限梯度`` /
       ``non-finite gradient`` / ``nonfinite_loss_or_grad``）⇒ ``NUMERIC_ANOMALY``；
       这一步由 ``_LOG_PATTERNS`` **表头第一条**实现（2026-09-18 🟡-4 修正前的
       旧序把它排在第 5 位，与本节描述不符，已修正）；
    5. **其余日志模式**（框架未实现 / 配置非法 / 数据问题 / **集合通信超时** / 通信硬件 / 真 OOM）——
       即 ``_LOG_PATTERNS`` 中**数值异常之后的**条目，按表内顺序匹配
       （``NCCL_COLLECTIVE_TIMEOUT`` 排在 ``COMM_HARDWARE`` **之前**：前者签名更具体）；
    6. 非零退出码且未匹配 ⇒ ``UNCLASSIFIED``。

    **每一级的可达性与触发条件**（不允许存在"写着却永不触发"的分级）：

    | # | 触发条件（可判定的事实） | 归类 | 可达 |
    |---|---|---|:--:|
    | 0 | ``exit_code`` ∈ ``{86, 21}`` | ``pipeline_p2p_timeout`` / ``pp_rollout_wall_clock_cap`` | ✔ |
    | 1 | ``timed_out`` 或 ``exit_code == 124`` | ``TIMEOUT`` | ✔ |
    | 2 | loss/grad_norm 序列里**真读到** NaN/Inf | ``NUMERIC_ANOMALY`` | ✔ |
    | 3 | loss 字段无值（``MISSING_SENTINEL``）且**无任何**真 NaN | ``UNCLASSIFIED`` | ✔ |
    | 4 | 日志出现 ``非有限梯度`` 等硬失败标记 | ``NUMERIC_ANOMALY`` | ✔ |
    | 5 | 日志匹配到 OOM / 未实现 / 配置 / 数据 / **集合通信超时** / 通信模式（且不含第 4 步标记） | 对应类型 | ✔ |
    | 6 | 非零退出、以上都不匹配 | ``UNCLASSIFIED`` | ✔ |

    **为什么第 4 步必须排在其余日志模式之前**（🟡-4 修正的判据）：``非有限梯度``
    是**本项目的训练器主动打出的**硬失败标记，出现即证明"训练在数值处硬停"——
    这是关于训练数值的事实断言；而 ``FileNotFoundError`` / ``NotImplementedError``
    是通用文本，可能来自同一份日志里与死因无关的旁路。旧序下两者同现会先命中
    数据/框架文本，实测把 ``FileNotFoundError`` + ``非有限梯度`` 归成「数据问题」，
    把排查引向数据而不是数值链路。两者的 ``counts_toward_max_context`` 都是
    ``False``，故本修正不改变能力边界，只修正"归类给谁看"。

    **为什么"数值异常"必须排在"日志里出现 OOM 字样"之前**：A6 的 NaN/Inf /
    loss 上升是从证据序列直接算出的事实，而日志里的 `CUDA out of memory` 只是
    文本证据——一次真 OOM 之后常伴随 NaN/数值崩溃，若按文本归类就会把该档上下文
    记进"最大可行上下文"（放宽"只有真 OOM 才计入"的硬要求并污染能力边界）。
    端到端用例见 ``tests/e2e/test_collect_results.py::
    test_collector_numeric_anomaly_wins_over_oom_text``。

    **为什么第 3 步与第 2 步分家**：``loss: null`` 是"字段没有值"的证据缺口，
    既不是"run 数值崩坏"、也不是"上下文太长"。若与第 2 步合并，就会凭**没有读数**
    断言"数值异常"（假陈述，且把排查引向数值问题）；若落到第 5 步，日志里恰好有
    OOM 字样时又会被误记成真 OOM（污染能力边界）。两条路都错，故单列
    ``UNCLASSIFIED``，且与第 2 步一样**不计入最大可行上下文**。
    """
    if evidence.timed_out or evidence.exit_code == 124:
        return FailureClass.TIMEOUT
    # ★ 退出码优先（2026-09-22 裁定）：86 = PP P2P 超时、21 = PP rollout 墙钟上限。
    #   排在文本匹配**之前**，因为异常文本常被 torchrun 前缀化而匹配不到。
    by_exit = EXIT_CODE_FAILURE_CLASSES.get(evidence.exit_code) if evidence.exit_code else None
    if by_exit is not None:
        return by_exit
    if numeric_anomaly(a6):
        return FailureClass.NUMERIC_ANOMALY
    if numeric_indeterminate(a6):
        return FailureClass.UNCLASSIFIED
    for failure_class, patterns in _LOG_PATTERNS:
        if any(pattern.search(evidence.log_text) for pattern in patterns):
            return failure_class
    if evidence.exit_code not in (None, 0):
        return FailureClass.UNCLASSIFIED
    return None


def counts_toward_max_context(failure_class: FailureClass | None) -> bool:
    """只有真 OOM 计入最大可行上下文（硬要求）。"""
    return failure_class in MAX_CONTEXT_FAILURE_CLASSES


# ── 总判定 ──────────────────────────────────────────────────────────────────


def usability_step_criterion(evidence: RunEvidence) -> tuple[bool, bool, str]:
    """**主判据之一**：这次运行是否"跑满了预定步数/epoch"。

    返回 ``(跑满, 结构性不可测, 人读说明)``。判据来源与 :func:`judge_a2` **逐字共用**
    同一对纯函数（:func:`resolve_min_optimizer_steps` / :func:`resolve_step_gate_applicability`），
    不另立门槛口径（§1.4 单一真相源）。

    与 :func:`judge_a2` 的**唯一差别**：本函数**不看**"权重是否变化"这条附属断言
    （那属效果/取证，不是"能不能跑完"）；`optimizer_steps is None`（读不到步数）⇒
    不是"跑满"，由调用方按**取证缺口**处理。
    """
    threshold, _reject = resolve_min_optimizer_steps(evidence.min_optimizer_steps)
    if threshold is None:
        return False, False, f"缺省门槛不可解析（min_optimizer_steps={evidence.min_optimizer_steps}）"
    # ★ 顺序与 :func:`judge_a2` **逐条一致**（§1.4 单一真相源）：
    #   **先判"读不到步数"这个取证缺口**，再谈"门槛结构上不可测"。
    #   反过来的话，`optimizer_steps=None` + `reachable < 门槛` 会被错判成
    #   「口径不可测（结构性）」，而事实是**这次没取到读数**——两者修法完全不同
    #   （前者修采集，后者要改档位定义）。这条区分有既有用例守卫。
    steps = evidence.optimizer_steps
    if steps is None:
        return False, False, "读不到 optimizer step 数（取证缺口，fail-closed）"
    not_applicable, why = resolve_step_gate_applicability(evidence, threshold)
    if not_applicable:
        planned = evidence.expected_optimizer_steps_per_epoch
        if planned is None:
            planned = evidence.expected_optimizer_steps_reachable
        if isinstance(planned, int) and planned > 0 and steps >= planned:
            return (
                True,
                False,
                f"跑满该档计划步数 {steps} ≥ {planned}"
                f"（外挂门槛 {threshold} 已退出闸门角色、仅作标注）",
            )
        return False, True, f"{STEP_GATE_NOT_APPLICABLE_MARKER}：{why}"
    if steps >= threshold:
        return True, False, f"optimizer step={steps} ≥ 门槛 {threshold}"
    return False, False, f"optimizer step={steps} < 门槛 {threshold}（未跑满预定步数/epoch）"


def judge_tier(
    first: RunEvidence,
    second: RunEvidence | None = None,
    *,
    context_length: int | None = None,
    require_dual_channel: bool = False,
) -> TierJudgement:
    """对一档做 A1–A6 总判定，产出可直接落台账的记录。

    ``require_dual_channel``：矩阵采集口径 —— **两跑 loss 逐位相同而独立通道无证据**
    时判「不可判定」（见 :func:`judge_a4` 子检查④形态②）。纯逻辑调用方可以不开，
    但**能力矩阵的采集必须开**，否则 AD1 §④ 的退化通过会静默通过。此开关会被记进
    :class:`TierJudgement` 并由 :func:`ledger_row` 落盘（"这一档按哪套口径判的"可审计）。
    """
    a1 = judge_a1(first)
    a2 = judge_a2(first)
    a3 = judge_a3(first)
    a4 = judge_a4(first, second, require_dual_channel=require_dual_channel)
    a5 = judge_a5(first)
    a6 = judge_a6(first)
    a7 = judge_a7(first)
    criteria = (a1, a2, a3, a4, a5, a6, a7)

    # ══ 新口径（2026-09-22 用户纠正；本函数的判定轴）══════════════════════════
    #   用户原话："你的判定标准应该是**能顺利跑完一个 epoch**……我们现在看的
    #   **不是效果，是训练可用**。"
    #   ⇒ 主判据 = **训练可用性**：A1（正常退出）+ steps（跑满预定步数/epoch）
    #     + A5（产物齐全）+ A6（数值健康）。
    #   ⇒ A3 / A4 **整体降级为诊断**：仍计算、仍记录（证据不删），但**不参与 ✅/❌**。
    _steps_passed, step_gate_not_applicable, steps_detail = usability_step_criterion(first)
    primary_status: dict[str, bool] = {
        "A1": a1.passed,
        "A2": a2.passed,
        "A3": a3.passed,
        "A5": a5.passed,
        "A6": a6.passed,
        "A7": a7.passed,
    }
    passed = all(primary_status.values())
    diagnostics_note = (
        "【诊断字段·不参与判定】"
        f"A4 双跑一致性 = {a4.passed}（{a4.detail[:240]}）"
    )

    failure_class: FailureClass | None = None
    note = ""
    primary_evidence_gap = False
    #: 「除 A2 以外的主判据是否全过」——**通过分支里当然全过**（含 A2）⇒ 默认 True。
    others_ok = True
    if passed:
        note = (
            "✅ 可用（主判据 = 训练可用性）："
            f"{steps_detail}；进程正常退出；权重已更新；checkpoint 可重载；产物齐全；"
            "数值健康；训练真推进（无 nonfinite 跳过）。"
            f"｜{diagnostics_note}"
        )
    else:
        failing_primary = [name for name, ok in primary_status.items() if not ok]
        others_ok = all(ok for name, ok in primary_status.items() if name != "A2")
        # 每条失败的主判据**是否只是"读不到证据"**
        primary_gap_flags = {
            "A1": bool(a1.evidence_missing),
            "A2": bool(a2.evidence_missing),
            "A3": bool(a3.evidence_missing),
            "A5": bool(a5.evidence_missing),
            "A6": bool(a6.evidence_missing),
            "A7": bool(a7.evidence_missing),
        }
        primary_evidence_gap = all(primary_gap_flags[name] for name in failing_primary)
        substantive = classify_failure(first, a6)
        # ★ **一致性修复 v2**（2026-09-23；v1 方向错了，见下）：
        #   症状：T031/T038/T043/T044/T045（+T032）**同一条记录三个说法**——
        #     `status=⚠ 口径不可测` 但 note 前缀 `❌ 不可用`、failure_class 记"通信硬件"。
        #   **v1 的错**：我把状态词改成跟着 `substantive` 走 ⇒ 这 5 档变成 `❌ 不可用`，
        #     而那个"通信硬件"**恰恰是不可靠标签**（指挥官复核：6 档里 5 档实为 `⚠-A7`）。
        #   真根因：`classify_failure` 从**日志文本**里抓信号，而这些档 **`A1` 通过
        #     （`exit_code=0`、步数精确跑满）** ⇒ 日志里的"通信/NCCL"文本很可能是
        #     **无关输出**（例如被引用/被回显的字符串），**不能**当作这次运行的失败证据。
        #   修法：**只有"这次运行本身没正常退出"（A1 未过）时，日志文本里的硬失败信号才可信**；
        #     A1 已过 ⇒ 该信号降级为**诊断记录**，不参与状态与 failure_class。
        log_signal = substantive
        # ★ **只对"已证实不可靠"的那一类**这么做（指挥官复核：`通信硬件` 标签 6 档里
        #   5 档实为 `⚠-A7`）——**不**推广到 OOM 等：真正的 OOM 有独立证据路径
        #   （A6 数值异常 / 独立日志断言），且 OOM 与"A1 正常退出"通常互斥，
        #   幸存的 OOM 用例必须继续照判（有既有用例守卫）。
        if substantive is FailureClass.COMM_HARDWARE and a1.passed:
            substantive = None
            note_log_signal = (
                f"（诊断：日志文本里出现过 `{log_signal}` 类字样，但本次 **A1 正常退出**，"
                "该字样**不作为失败证据**——它更可能是无关输出；"
                "这正是 `failure_class=通信硬件` 不可靠的来源）"
            )
        else:
            note_log_signal = ""
        # 真失败（本次运行自身的硬信号）优先于"采集侧读不到"
        primary_evidence_gap = primary_evidence_gap and substantive is None
        if step_gate_not_applicable and others_ok:
            # 结构性不可测（口径层，不是能力层）：仅当其余主判据都过时落此态。
            failure_class = FailureClass.GATE_NOT_APPLICABLE
            note = (
                f"⚠ 口径不可测：{steps_detail}——该档用本 config 无论跑多久都达不到门槛"
                " ⇒ **不是训练失败**：能力**未被本次取数证伪**，门槛值一个字未放宽。"
                f"｜{diagnostics_note}"
            )
        elif primary_evidence_gap and substantive is None:
            failure_class = FailureClass.EVIDENCE_GAP
            note = (
                f"⚠ 口径不可测（取证缺口）：主判据读不到证据（{', '.join(failing_primary)}）"
                " ⇒ 既不算可用、也不算不可用（fail-closed）。"
                "（注：A3/A4 已降级为诊断，**不**再制造此态。）"
                f"{note_log_signal}"
                f"｜{diagnostics_note}"
            )
        else:
            failure_class = substantive if substantive is not None else FailureClass.UNCLASSIFIED
            note = (
                f"❌ 不可用：未满足主判据 {', '.join(failing_primary)}；"
                f"失败类型：{failure_class}；{steps_detail}"
            )
            if not counts_toward_max_context(failure_class):
                note += "（不计入最大可行上下文，修复后重测）"
            elif context_length is not None:
                note += (
                    f"（真 OOM：当前上下文 {context_length} 可作为该档最大可行上下文的候选，"
                    "需按递增加长法确认边界）"
                )
            else:
                note += "（真 OOM）"
            if a6.passed is False and "NaN/Inf" in a6.detail:
                # 数值崩坏 ⇒ ❌ 不可用（用户保留"数值健康"为主判据的理由见 §2.2 说明）
                note += "；★ 数值崩坏（NaN/Inf）⇒ 产出不可用，属硬性不可用"
            note += f"｜{diagnostics_note}"

    return TierJudgement(
        tier_id=first.tier_id,
        criteria=criteria,
        passed=passed,
        failure_class=failure_class,
        counts_toward_max_context=counts_toward_max_context(failure_class),
        note=note,
        a2_step_threshold=resolve_min_optimizer_steps(first.min_optimizer_steps)[0],
        weight_evidence_source=first.weight_evidence_source,
        reload_evidence_source=first.reload_evidence_source,
        a3_source=first.a3_source,
        a4_require_dual_channel=require_dual_channel,
        # 第三态自证（§1.4）：台账必须能回答"A2 的步数门槛为什么没判失败"。
        step_gate_not_applicable=step_gate_not_applicable,
        primary_evidence_gap=primary_evidence_gap,
        primary_others_ok=others_ok,
        expected_optimizer_steps_reachable=first.expected_optimizer_steps_reachable,
    )


#: 台账 `max_context` 的**取值口径标记**——回答"这一列的数字从哪来"（§1.4）。
#: 为什么必须有它：真 OOM 档写进该列的是**边界候选**（该 run 整体未通过），
#: 通过档写进去的是**实测可行值**；两者都在同一列，不标口径就会被下游读成
#: "这个长度跑得通"，从而把不可行的长度当成能力上限（§7.1 第 4 条）。
MAX_CONTEXT_KIND_FEASIBLE = "实测通过"
MAX_CONTEXT_KIND_OOM_BOUNDARY = "真 OOM 边界候选"

#: A4 独立通道在台账里的**键名**（唯一真相源，§1.4）：把"键名"集中在这里，
#: 采集层与下游读同一份定义，不可能出现"写的是这个名、读的是那个名"。
A4_CKPT_SHA256_FIRST_FIELD = "final_ckpt_sha256_first"
A4_CKPT_SHA256_SECOND_FIELD = "final_ckpt_sha256_second"
A4_CKPT_SHA256_AGREE_FIELD = "final_ckpt_sha256_agree"


def a4_ckpt_sha_fields(first_sha: str | None, second_sha: str | None) -> dict[str, Any]:
    """构造台账里 A4 独立通道的四个键值（两跑指纹 + 结论 + 缺失原因）。

    ``agree`` 三态（**None 不是 False**，§2.2）：
    - ``True``：两端都取到且逐位相同；
    - ``False``：两端都取到但不同；
    - ``None``：至少一端取不到 ⇒ **没有结论**（不能把"读不到"说成"不一致"，也不能说成"一致"）。

    为什么必须把"取不到"和"不同"分开：AD1 §6.2 已证无 bug 的两跑权重 sha **不同**是
    GPU 内核残余下的**预期现象**；把"取不到"混进 ``False`` 会让台账读起来像"两跑权重
    不一致 = 有 bug"，从而制造假怀疑（§2.2 显式即防呆）。
    """
    agree: bool | None
    if first_sha is not None and second_sha is not None:
        agree = first_sha == second_sha
    else:
        agree = None
    return {
        A4_CKPT_SHA256_FIRST_FIELD: first_sha,
        A4_CKPT_SHA256_SECOND_FIELD: second_sha,
        A4_CKPT_SHA256_AGREE_FIELD: agree,
    }


def ledger_row(
    judgement: TierJudgement,
    *,
    model: str,
    algorithm: str,
    mode: str,
    backend: str,
    cards: int,
    max_context: int | None,
    peak_memory_gib: float | None,
    date: str,
    host_sample_peak_gib: float | None = None,
    host_sample_peak_gap_mib: float | None = None,
    msswift_reserved_peak_gib: float | None = None,
    peak_memory_caliber: str | None = None,
) -> dict[str, object]:
    """把判定落成 capability-matrix.html §6 台账的一行（可直接填表）。

    **``peak_memory_gib`` 的口径（§9.1 二分口径，2026-09-21 指挥官裁定「方案 A」）**：
    **按本档 ``backend`` 决定**，映射的唯一真相源是
    :data:`PEAK_MEMORY_CALIBER_BY_BACKEND`（读取入口
    :func:`peak_memory_caliber_for_backend`）：

    - ``native`` ⇒ :data:`PEAK_MEMORY_CALIBER`（容器内 PyTorch allocator 的 rank0
      ``max_allocated``）；
    - ``ms-swift`` ⇒ :data:`MSSWIFT_RESERVED_CALIBER`（该进程的 ``max_memory_reserved``，
      **保守上界**：``reserved >= allocated``）；
    - 未登记的后端 ⇒ :data:`PEAK_MEMORY_CALIBER_UNKNOWN`（显式，绝不默认取某一支）。

    宿主 `nvidia-smi` 采样峰值**不得**经本参数进入该列——它只能走
    ``host_sample_peak_gib`` 另存（:data:`HOST_SAMPLE_PEAK_FIELD`）。
    ``peak_memory_gib is None`` ⇒ 台账按**该档口径**落对应的显式标注
    （:func:`peak_memory_unavailable_note`），让"该口径不可得"与"没跑过"在下游可区分，
    且**绝不静默回落**到另一种口径或宿主值（§2.2 显式即防呆）。

    **``peak_memory_caliber``（可选）**：调用方（采集层取数函数）核算出的口径标签。
    默认 ``None`` ⇒ 由 ``backend`` 就地推导。给值时**必须**与推导值相同——否则说明
    取数与落盘对"哪个后端用哪个口径"理解不一致，本函数**当场拒绝**而不是把错标签落盘。

    **``msswift_reserved_peak_gib``**：ms-swift 后端自报的
    `memory(GiB)`（= `max_memory_reserved`，见 :data:`MSSWIFT_RESERVED_CALIBER`）的
    **原始读数**，经 :data:`MSSWIFT_RESERVED_PEAK_FIELD` 另存并配对落口径标签与理由。
    ★ 在二分口径下它与 ms-swift 档的 ``peak_memory_gib`` **同值同源**（不是第二个真相源：
    一个是"本档 §7 峰值列的值"，一个是"上游自报字段的原值 + 来源标签"）。
    非 ms-swift 档上它恒为 ``None``（不适用）。

    **``max_context`` 的资格口径（🔴-1 修正，2026-09-18）**：该列的资格由
    :func:`counts_toward_max_context` **或**整档通过共同决定，**不再**只由
    ``judgement.passed`` 决定。

    旧实现 ``max_context if judgement.passed else None`` 在**结构上不可能**满足硬
    要求「只有真 OOM 才能写入最大可行上下文」：真 OOM 的 run 必然 ``exit != 0``
    ⇒ A1 不过 ⇒ ``passed=False`` ⇒ 该列恒为 ``None``；而能写进该列的那类恰好
    永远不是真 OOM。分类层 :func:`counts_toward_max_context` 一直返回 ``True``，
    是落盘层把值丢掉了。

    修正后该列有两种来源，用 ``max_context_kind`` 显式区分：

    - :data:`MAX_CONTEXT_KIND_FEASIBLE`：整档通过，该长度**已被实测证明可行**；
    - :data:`MAX_CONTEXT_KIND_OOM_BOUNDARY`：整档未通过但失败类型是真 OOM，
      该长度是**边界候选**（run 在这里 OOM 了，说明上限低于它；仍须按 §7.1
      第 4 条的"上一个通过长度 + 二分细化"确认，不得直接当成可行值）。

    **不放松任何判定**：``passed`` 的语义与 A1–A6 都不动；非真 OOM 的失败
    （数值异常 / 数据问题 / 框架未实现 / 配置非法 / 通信硬件 / 超时 / 未分类）
    仍然一律 ``None``。
    """
    eligible = judgement.passed or judgement.counts_toward_max_context
    recorded_context = max_context if eligible else None
    if recorded_context is None:
        context_kind: str | None = None
    elif judgement.passed:
        context_kind = MAX_CONTEXT_KIND_FEASIBLE
    else:
        context_kind = MAX_CONTEXT_KIND_OOM_BOUNDARY
    # 本档峰值列的**口径由 backend 唯一决定**（§1.4）：采集层核算的标签若与本处推导
    # 不一致，说明两边对"哪个后端用哪个口径"理解不同 —— 当场拒绝，不把错标签落盘。
    derived_caliber = peak_memory_caliber_for_backend(backend)
    if peak_memory_caliber is not None and peak_memory_caliber != derived_caliber:
        raise ValueError(
            "峰值口径不一致：采集层给 "
            f"{peak_memory_caliber!r}，而后端 {backend!r} 的唯一映射是 {derived_caliber!r}"
            "（§1.4 单一真相源：PEAK_MEMORY_CALIBER_BY_BACKEND）"
        )
    caliber = derived_caliber
    return {
        "tier_id": judgement.tier_id,
        "model": model,
        "algorithm": algorithm,
        "mode": mode,
        "backend": backend,
        "cards": cards,
        "max_context": recorded_context,
        "max_context_kind": context_kind,
        "peak_memory_gib": peak_memory_gib,
        # 峰值列的**口径自证 + 空值原因 + 宿主副读数**（§1.4 单一真相源 / §2.2 显式即防呆）：
        # 台账必须能回答"这一格是什么口径、从哪个字段来""不可得时为什么空""卡实际用了多少"。
        # **只加字段**：`peak_memory_gib` 的值域与任何判定都不动；变的只是"按后端二分"。
        "peak_memory_caliber": caliber,
        "peak_memory_note": "" if peak_memory_gib is not None else peak_memory_unavailable_note(caliber),
        HOST_SAMPLE_PEAK_FIELD: host_sample_peak_gib,
        # 同一次宿主采样的缺口：> 0 = 该卡峰值含**非本档**成分（共享机他人作业），
        # 副读数必须打折看。**只加字段**，不进 §7 峰值列（二分口径下仍然禁止）。
        HOST_SAMPLE_GAP_FIELD: host_sample_peak_gap_mib,
        # ms-swift 后端自报的 **reserved** 峰值的**原始读数**：另存 + 带口径标签（§1.4 / §2.2）。
        # 二分口径下（2026-09-21 裁定）它同时就是 ms-swift 档 §7 峰值列的取数来源；
        # 本字段保留为"上游自报字段的原值 + 来源标签"，非 ms-swift 档恒为 None。
        # `None` ⇒ 显式标注原因，绝不静默、绝不用宿主采样补 —— 与 `PEAK_MEMORY_UNAVAILABLE` 同一防呆模式。
        MSSWIFT_RESERVED_PEAK_FIELD: msswift_reserved_peak_gib,
        MSSWIFT_RESERVED_CALIBER_FIELD: MSSWIFT_RESERVED_CALIBER,
        MSSWIFT_RESERVED_NOTE_FIELD: (
            MSSWIFT_RESERVED_NOTE if msswift_reserved_peak_gib is not None else MSSWIFT_RESERVED_UNAVAILABLE
        ),
        "status": judgement.ledger_status,
        "failure_class": judgement.failure_class,
        "note": judgement.note,
        # 读数口径自证（§1.4）：台账必须能回答"A2 用的门槛是多少"。只加字段，
        # 不改任何判定与既有字段的语义（`None` = 本次未解析出合法门槛）。
        "min_optimizer_steps": judgement.a2_step_threshold,
        # ★ step 门槛的**结构可达性自证**（AF1 §⑥ 方案 S2，§1.4 单一真相源 / §2.2 显式即防呆）：
        # 台账必须能回答"这一档最多能跑几步""为什么步数不够却不判失败"。只加字段，
        # `status` 的取值域只多了第四态（口径不可测），**不放宽**任何现有档位。
        "expected_optimizer_steps_reachable": judgement.expected_optimizer_steps_reachable,
        "step_gate_not_applicable": judgement.step_gate_not_applicable,
        # 证据来源自证（§1.4 单一真相源 / §2.2 显式即防呆）：台账必须能回答"A2 的权重
        # 变化证据是哪种等级、从哪来""A3 的重载证据是哪种等级"。**只加字段**，不改任何
        # 判定与既有字段的语义。为什么必须有：宿主无 torch 造成的取证缺口与训练真失败
        # 在旧台账上无法区分（T010 真机被记成 ❌ 失败，实为采集机环境性伪否）。
        "weight_evidence_source": judgement.weight_evidence_source,
        "reload_evidence_source": judgement.reload_evidence_source,
        "a3_source": judgement.a3_source,
        # A4 的双通道强制口径自证（§1.4）：台账必须能回答"这一档的 A4 是按哪套口径判的"。
        # 同一对 loss 读数在两种口径下可能一个 ✅ 一个 ⚠ 不可判定，不写清就是第二个真相源。
        "a4_require_dual_channel": judgement.a4_require_dual_channel,
        # A6 首末 loss 走向的**记录项**（2026-09-20 裁定降级，见 A6_LOSS_TREND_BLOCKS）。
        # 只加字段、不改 passed 语义：`None` = 本次 A6 没产出走向记录（NaN/Inf 或取证缺口）。
        # 与 A6 明细文本同源（a6_loss_trend 从明细取），不是第二个真相源。
        A6_LOSS_TREND_FIELD: a6_loss_trend(judgement.criterion("A6")),
        "date": date,
    }


def loss_series_summary(values: Sequence[float]) -> str:
    """诊断用：把 loss/grad_norm 序列压成一行摘要。"""
    if not values:
        return "empty"
    return f"n={len(values)} first={values[0]:.6g} last={values[-1]:.6g}"


#: 允许调用方遍历的失败类型清单（含兜底），供 CLI 打印。
ALL_FAILURE_CLASSES: tuple[FailureClass, ...] = tuple(FailureClass)
