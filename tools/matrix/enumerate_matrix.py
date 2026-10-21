#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""档位矩阵**唯一生成入口**：全量枚举 → 剔除不合理 → 逐档选后端 → 写 `docs/capability-matrix.data.json`。

★★ 本脚本是档位表的唯一生成入口，⛔ 不许手改 `docs/capability-matrix.data.json` 的 `rows[]`。
   行数据 = 本脚本的输出；要改档位/判据/常数/证据 ⇒ 改本脚本的数据结构并重跑（宪法 §1.4）。
   `rows[]` 里**没有任何一个是手写的**：`--dry-run` 与正式运行走同一条构造路径。

背景（主笔原话，逐字保留在这里作为需求锚点）：
  「矩阵为什么有 50 多档？其实他最开始用了一套全量组合的方法，把张量并行、数据并行、
   序列并行、流水线并行等等各种显存分配方法都全量组合了一遍，把不合理的都删掉……
   保证最多用 4 张卡……既然单张卡放不下 27B 的模型，说明排列组合里有些档位不合理。」
  ⇒ 本脚本做的就是：**全量枚举 → 按判据剔除 → 留下的逐档选后端**，并把判据/常数/证据
   全部变成可审计的数据结构（⛔ 不靠文档里手填）。

用法：
    python tools/matrix/enumerate_matrix.py --dry-run            # 只打印汇总结论，不写任何文件
    python tools/matrix/enumerate_matrix.py                      # 写 data.json + 剔除清单（.local）
    python tools/matrix/enumerate_matrix.py --card-limit 4 --budget-gib 78.0 \
        --out docs/capability-matrix.data.json --generated-at 2026-10-21T23:59:00+08:00

参数：
    --card-limit     档位设计约束「总卡数 ≤ N」（**矩阵设计参数，不是平台上限**：
                      `docs/card-pool-contract.md:47-51` 明文无「每次最多 4 卡」这类平台级上限）
    --budget-gib     单卡显存预算（判定线），默认 78.0（`tests/e2e/generate_matrix.py:1355`）
    --out            输出数据件，默认 `docs/capability-matrix.data.json`
    --report-json    剔除清单输出路径，默认写到 `.local/` 下本工位（⛔ 不污染仓）
    --dry-run        不写任何文件
    --generated-at   固定 `meta.enumeration.generated_at`（**确定性的唯一非确定字段**）

确定性：同输入（含 `--generated-at`）⇒ 逐字节同输出。排序键见 `SORT_KEY_DOC`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ════════════════════════════════════════════════════════════════════════════
# §0 · 路径与常量
# ════════════════════════════════════════════════════════════════════════════

GIB = float(2 ** 30)
REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR_REL = ".local/hb-workspace/20261017-matrix-realign/task-enum-matrix"
DEFAULT_REPORT_JSON = f"{WORK_DIR_REL}/excluded-report.json"
#: ★ W8-②（2026-10-22）：原先默认固定写到 `archive-54/`（而且写死了**他人工位**）⇒ 复核者用默认参数
#:   复跑会**覆盖**上一次的 `meta.before.json` / `rows.before.json`（退路自毁）。现改为
#:   **默认带时间戳**（`archive-<YYYYmmdd-HHMMSS>/`），复跑只会新增目录，⛔ 不再覆盖；
#:   需要指定固定位置时用 `--archive-dir` 显式给出。
DEFAULT_ARCHIVE_DIR = ""      # 空 = 运行时按时间戳现算（见 main）
DEFAULT_AUDIT_JSON = f"{WORK_DIR_REL}/enumerate_audit.json"
CRITERIA_VERSION = "enum-v1-2026-10-21"

#: 单卡总容量（**实测**）：显存不足 原文逐字 "GPU 0 has a total capacity of 79.25 GiB"
#:   —— `.local/hb-workspace/20260921-150941/fix-native-fullparam-numerics/pp1-control-report.md:45`
#:   交叉印证：宿主 `81153 MiB = 79.25 GiB`（`.local/opd-teacher-evidence-20261014.md` §⑥）。
CARD_CAPACITY_GIB = 79.25

# ════════════════════════════════════════════════════════════════════════════
# §1 · 输入域（枚举维度）
# ════════════════════════════════════════════════════════════════════════════

#: 算法取值 —— 单一真相源是 `src/graspo/core/schema.py:28` 的 `TrainMethod`。
ALGOS = ("SFT", "GRASPO", "CPT", "OPD")
ALGO_TO_TRAIN_METHOD = {"SFT": "sft", "GRASPO": "graspo", "CPT": "cpt", "OPD": "opd"}

#: 模型取值。`params` 与 `act_gib` 的证据等级逐项见 `MODEL_SPEC` 的注释。
MODELS = ("9B", "27B")

#: 模式取值（显示名 → `tuner_type` 字面量）。
MODES = ("LoRA", "全参")
MODE_TO_TUNER_TYPE = {"LoRA": "lora", "全参": "full"}

#: 并行形状取值域：每维 ∈ {1,2,4}，且 `tp*pp*dp ≤ CARD_LIMIT`（CARD_LIMIT 是 CLI 参数）。
PARALLEL_VALUES = (1, 2, 4)

#: 后端**不是**枚举维度，它是每档的**输出**（选型结果）。这里只定义候选全序。
BACKENDS = ("native", "ms-swift")

#: 排序键（档号 = 按此键升序编号 M001…）。⛔ 改排序键 = 改档号，必须重跑本脚本。
SORT_KEY_DOC = (
    "model(9B<27B) → algo(SFT<GRASPO<CPT<OPD) → mode(LoRA<全参) "
    "→ tp(升序) → pp(升序) → dp(升序) → backend(仅作尾键，rows 里已唯一) "
    "→ 档号 M001.. 按此顺序连续编号"
)

#: 算法 → **支持它的后端**。逐字来源 `src/graspo/core/schema.py:29-34`：
#:   `TRAIN_METHOD_BACKENDS = {"graspo": ("native","msswift"), "sft": ("native","msswift"),
#:                             "cpt": ("msswift",), "opd": ("msswift",)}`
#: ★ 主笔裁定「缺实现 ≠ 不合理」⇒ 这里**只**用于候选后端集合与备注卡点，
#:   ⛔ 绝不用它把 CPT/OPD 的 native 档"剔除"——后者根本不是一个可枚举出来的档位，
#:   它只是"该档在 native 上还缺实现"，必须作为待测试档的卡点登记（见 `ALGO_BACKEND_NOTE`）。
ALGO_BACKEND_SUPPORT: dict[str, tuple[str, ...]] = {
    "SFT": ("native", "ms-swift"),
    "GRASPO": ("native", "ms-swift"),
    "CPT": ("ms-swift",),
    "OPD": ("ms-swift",),
}

#: 「某算法在某后端缺实现」的**卡点描述**（缺实现 ≠ 不合理；只写清卡在哪一层）。
ALGO_BACKEND_NOTE: dict[tuple[str, str], str] = {
    ("CPT", "native"): "native 侧尚无 CPT 训练通道（算法层未提供该训练方法在 native 后端的实现）",
    ("OPD", "native"): "native 侧尚无 OPD 训练通道（算法层未提供该训练方法在 native 后端的实现）",
}

# ════════════════════════════════════════════════════════════════════════════
# §2 · 显存模型（常数逐项写在这里，**每一项都标证据等级**）
# ════════════════════════════════════════════════════════════════════════════
#
# 计算骨架（单位 GiB）：
#
#   全参: 每卡 = k × P / share + 激活(模型常数) [+ 教师常驻(OPD)]
#   LoRA: 每卡 = 2 × P / share_base + 12 × P_lora / share_lora + 激活(模型常数) [+ 教师]
#
# 证据等级语汇沿用仓内既有口径：
#   【实测】真机读数 ｜【配置可读】仓内配置/代码直读 ｜【推断】外推或假设（不得当实测）
#
# ▸ 每参数字节 k（`tests/e2e/generate_matrix.py:1347-1352` 逐行核实）：
#     - native 全参       k = 8   = bf16 权重 2 + bf16 梯度 2 + bf16 Adam(m+v) 4  【推断·来源假设】
#       `generate_matrix.py:1348-1351` 自述 "两侧的 k 都是假设、不是实测"。
#     - ms-swift 全参     k = 12  = bf16 权重 2 + bf16 梯度 2 + fp32 Adam(m+v) 8   【推断·保守】
#     - native 全参+offload k = 4 = 权重 2 + 梯度 2（优化器态在宿主内存，不占卡）    【推断】
#       `native.offload_optimizer_state` 语义：`src/graspo/core/schema.py:756-779`；
#       实现：`src/graspo/models/qwen35_36/cpu_offloaded_adam_w.py:1-58`。
#     - LoRA 基座         k = 2   = bf16 冻结权重（两后端一致）                      【配置可读】
#     - LoRA 可训练态     k = 12  = 2 权重 + 2 梯度 + fp32 Adam(m+v) 8              【推断】
#       ★ 现行生成器**漏了这一项**（`generate_matrix.py:1477-1479` 明文"适配器参数量忽略"），
#         9B 实测 384,757,760 个可训练参数 ⇒ 4.30 GiB（recon-enum-domain §3.6 第 2 条）。
#
# ▸ 分片语义（**每项都可判真假，别算错**）：
#     - native 全参：**只支持 PP 分片**（`src/graspo/core/schema.py:171-188` 拒绝 full+tp>1 或 dp>1，
#       理由逐字"native 的梯度同步实现只覆盖 LoRA 参数"⇒ 全参 DP 副本基座梯度不会 all-reduce）
#       ⇒ 除以 `pp`。
#     - native LoRA：TP/PP 都可摊基座（TP SUM 同步 `src/graspo/flow/backend/native/lora/lora_linear.py:223-259`；
#       DP AVG 同步 `:262-282`）⇒ 基座与适配器按 `tp*pp` 摊；**DP 不摊基座**（冻结权重每 rank 一份）。
#     - ms-swift 全参：走 FSDP2 ⇒ 可按 `tp*pp*dp` 摊（`msswift.fsdp` 真透传：
#       `src/graspo/flow/backend/msswift/_config_mapping.py:206-213,353-362`；配方
#       `samples/configs/fsdp/fsdp2_full_state_dict.json`）。★ FSDP2 **零 退出码 0 实测**（见证据表 EV20）。
#     - ms-swift LoRA：标准通道只有 DDP+SP（`_config_mapping.py:409-436`）；Megatron 段未接线
#       （`_config_mapping.py:682` 写死 `include_megatron=False`）⇒ 基座与适配器按 `tp*pp` 摊、DP 不摊。
#       ★ `tp>1/pp>1` 在 ms-swift 上的"分摊"实际只能靠 FSDP2 全分片承载，属**待实测卡点**。
#
# ▸ 激活项（每模型常数，**实测反推**，不是结构算出来的）：
#     - 9B  = 12.18 GiB：79.24（9B 全参 1 卡实测峰值）− 67.06（8 B/param × 9e9）
#       ⇒ 与 `ACTIVATION_ALLOWANCE_GIB = 12.0` 的常数差 0.18 GiB（recon-enum-domain §3.6 ①）。
#     - 27B = 28.54 GiB（总残余）− 12.85（27B 适配器态估算）= 15.69 GiB：
#       28.54 = 实测 78.64 − 实测 bf16 权重 50.10（`.local/repo-moved-out/rig/run-plan.md:488` U-7）。
#       ★ 27B 的"激活 vs 适配器态"拆分**不可直接测**：本脚本按"先估适配器参数、再从实测总残余里扣"
#         的顺序落地（结果：27B LoRA tp1pp1dp1 = 50.10+12.85+15.69 = 78.64，与实测逐位吻合）。
#         不确定性：适配器参数估算 ±30% ⇒ 激活项同幅反向漂移，**总占用不变**（总占用锚在实测值上）。
#     - 激活项**不按 tp/pp 再摊**（与 `generate_matrix.py:1465-1555` 现行算式同口径，保守）：
#       依据 T035（9B 全参 native PP2）算式 45.53 却实测 显存不足 的冲突（recon §3.6 ⑥）——
#       native PP 是否真摊掉优化器态/激活，仓内无证据 ⇒ 宁可保守。
#
# ▸ LoRA 可训练参数量 P_lora：
#     - 9B  = 384,757,760  【实测】`docs/capability-matrix.data.json` meta
#       `lora_target_backends.native.signature_evidence.parameter_count`（2026-10-12 探针 + 解析式，
#       两后端逐项一致）。
#     - 27B = 9B × (26.896e9 / 9.0e9) = 1,149,827,190 【推断·按参数量等比例外推】
#       ★ 仓内**没有** 27B 的 LoRA 可训练参数量实测（模型 config 不在仓内：层数/隐藏维/头数零证据）。
#       外推假设：LoRA 目标正则与 rank 相同、可训练参数占比随模型参数量同比放大。
#       该假设的不确定性由 §2 的"激活项反推"吸收（两者之和恒等于实测 78.64）。
#
# ▸ OPD 教师（**单独计，不计入卡数**，主笔原文"不包括 OPD 的教师"）：
#     - GPU 常驻情形：每 rank 一份、非分片 ⇒ +50.10 GiB/卡 【实测】
#       （`tests/e2e/matrix54_manifest.json:128` `teacher_weights_gib_bf16_measured: 50.1`；
#       机理逐字"teacher 每 rank 各持一份（非分片）⇒ 加卡只增总显存、不降单卡占用" `:152`）
#     - offload 情形：`distill.offload_teacher_model=true` 真透传
#       （`src/graspo/core/schema.py:666-668`；`_config_mapping.py:663`）⇒ GPU 侧不计教师常驻。
#       ★ 该路径**零 退出码 0 实测**（证据表 EV16）⇒ 选它必须写进备注卡点。
#     - 每档按**两种情形分别估算**，取更优者作为该档的判定；恶劣情形写入备注与剔除清单。

#: 模型参数量与激活常数（逐项标证据等级）。
MODEL_SPEC: dict[str, dict[str, Any]] = {
    "9B": {
        "label": "Qwen3.5-9B",
        "params": 9.0e9,
        "params_level": "【推断】估算常量（`tests/e2e/generate_matrix.py:1332`）；docstring 另提 9.41B 但无出处 ⇒ 待实测",
        "weights_gib_level": "【推断】由 9.0e9 × 2 B 推出（与 9B 实测峰值口径自洽，见激活项反推）",
        "act_gib": 12.18,
        "act_level": "【实测反推】79.24 − 67.06（显存不足 原文 `pp1-control-report.md:44-46,63`）",
        "lora_trainable": 384_757_760,
        "lora_trainable_level": "【实测】meta.lora_target_backends.native.signature_evidence（2026-10-12 探针）",
    },
    "27B": {
        "label": "Qwen3.8-27B",
        "params": 26.896e9,
        "params_level": "【实测】上机 probe（`tests/e2e/matrix54_manifest.json:129`）",
        "weights_gib_level": "【实测】50.10 GiB（`matrix54_manifest.json:128`；26.896e9×2/2³⁰ 逐位吻合）",
        "act_gib": None,  # 由 §2 注释里的推导在模块加载时算出，见 `_derive_act_27b()`
        "act_level": "【实测反推】78.64 − 50.10 − 12.85（适配器态估算）= 15.69",
        "lora_trainable": None,  # 同上，等比例外推
        "lora_trainable_level": "【推断】9B 实测值 × 参数量比（26.896/9.0）；27B 无实测 ⇒ 不确定性 ±30%",
    },
}

#: 27B 族实测峰值（族级、零余量）：78.64 GiB / 卡容量 79.25 GiB ⇒ 余量 0.61 GiB
#:   来源：`.local/repo-moved-out/rig/run-plan.md:488`（U-7，标"实测"）。
#:   ⚠ 口径：ms-swift 侧报 `max_memory_reserved`（保守上界，`scripts/collect_results.py:2032-2071`）；
#:   native 侧报 `max_allocated`。两者不可直接互比 ⇒ 本脚本对 27B 的标定一律按 reserved 上界取，
#:   **当上界用**（偏保守），不据它下调任何档位。
PEAK_27B_FAMILY_GIB = 78.64

#: 教师权重（bf16，每 rank 一份）：50.10 GiB 【实测】`matrix54_manifest.json:128`
TEACHER_WEIGHT_GIB = 50.10

#: LoRA 可训练态每参数字节：2 权重 + 2 梯度 + fp32 Adam(m+v) 8 = 12 【推断】
LORA_STATE_BYTES_PER_PARAM = 12.0

#: 全参每参数字节（k），按后端 × 配方分列。
FULL_K_BYTES: dict[str, float] = {
    "native": 8.0,            # bf16 权重2+梯度2+bf16 Adam4   【推断】
    "native_offload": 4.0,    # 权重2+梯度2（优化器态在 CPU） 【推断】
    "ms-swift": 12.0,         # 权重2+梯度2+fp32 Adam8        【推断·保守】
}

#: 三档判定的阈值：`0.90 × BUDGET`（BUDGET 由 CLI 给，默认 78.0）。
TIGHT_BAND_RATIO = 0.90


def _derive_act_27b() -> float:
    """27B 激活项 = 实测总残余(28.54) − 27B 适配器态估算(12.85)。

    28.54 = 实测族级峰值 78.64 − 实测 bf16 权重 50.10 ⇒ **总残余是实测的**；
    适配器态是本脚本的估算项 ⇒ 两者相减得到的激活项与适配器估算强耦合（不确定性约 ±30%），
    但**总占用恒锚在实测值上**：参数估算变大 ⇒ 激活项同步变小，反之亦然。
    """
    weights = 2 * MODEL_SPEC["27B"]["params"] / GIB
    lora_state = _lora_trainable("27B") * LORA_STATE_BYTES_PER_PARAM / GIB
    return PEAK_27B_FAMILY_GIB - weights - lora_state


def _lora_trainable(model: str) -> int:
    """LoRA 可训练参数量。9B = 实测值；27B = 按参数量等比例外推（【推断】）。"""
    spec = MODEL_SPEC[model]
    if spec["lora_trainable"] is not None:
        return int(spec["lora_trainable"])
    ratio = MODEL_SPEC["27B"]["params"] / MODEL_SPEC["9B"]["params"]
    return int(round(MODEL_SPEC["9B"]["lora_trainable"] * ratio))


def _act_gib(model: str) -> float:
    spec = MODEL_SPEC[model]
    return float(spec["act_gib"]) if spec["act_gib"] is not None else _derive_act_27b()


def _weights_gib(model: str) -> float:
    return 2 * MODEL_SPEC[model]["params"] / GIB


# ════════════════════════════════════════════════════════════════════════════
# §3 · 实测证据表（逐条带出处；证据优先级：实测 > 算式）
# ════════════════════════════════════════════════════════════════════════════
#
# 每条记录的字段：
#   id       唯一编号
#   kind     oom（逐字 显存不足 原文）/ runtime_failure（非 显存不足 的运行期失败）/ success（判据① 通过）
#            / peak_calibration（族级实测峰值，用于标定常数）/ contested（互斥记载，采信不足）/ warn（其它实测事实）
#   scope    (algo|"*", model|"*", mode|"*", backend|"*", shapes|"*" 或 [(tp,pp,dp), ...])
#   reading  读数原文（逐字或近逐字）
#   source   出处（`file:line`；`.local` 件标"归档"）
#   effect   对判定的作用：
#              backend_infeasible  该后端在该档 **实测不可行** ⇒ 从候选里剔除（证据 > 算式）
#              recipe_forbid       该后端在该档的**该配方**被实测否掉 ⇒ 换配方（如 9B 全参 1 卡必须 offload）
#              calibrate           用于标定常数（27B 族峰值）
#              success             判据①（训练跑通/产物完整）实测通过 ⇒ 选型排序加分
#              warn_only           只写进备注，不改判定（定性冲突/未定性异常/交付层缺口）
#
# ★ 断言口径（裁定 2026-10-21 后补，由自检 V13 机核）：
#   「训练可跑 / 训练能跑 / 训练跑通」**只允许**出现在证据表里有具名 `success` 类跑次的档上；
#   没有跑次的形状只能写「从未实跑 ⇒ 训练可行性未建立（产物侧另需补导出层）」。⛔ 不许用形状推断代替跑次。
#   反方向同理：note 里的「不可行 / 必然失败」必须有实测证据兜底，纯算式的只能写「估算超预算」。
#
# ★ 口径纪律（`scripts/collect_results.py:2032-2071`）：native = `max_allocated`，
#   ms-swift = `max_memory_reserved`（保守上界）；两者不可直接互比。

Shape = tuple[int, int, int]

MEASURED_EVIDENCE: tuple[dict[str, Any], ...] = (
    # ── A. 9B 全参 × native ────────────────────────────────────────────────
    {
        "id": "EV01", "label": "9B 全参 1 卡无 offload 实测 显存不足 79.24 GiB", "kind": "oom",
        "scope": ("SFT", "9B", "全参", "native", ((1, 1, 1),)),
        "reading": "峰值 79.24 GiB ⇒ 显存不足（卡容量 79.25 GiB，`optimizer.step()` 处爆）；估算 67.06+12.18=79.24 逐位吻合",
        "source": "归档 .local/hb-workspace/20260921-150941/fix-native-fullparam-numerics/pp1-control-report.md:44-46,63",
        "effect": "recipe_forbid",
        "recipe": "no_offload",
    },
    {
        "id": "EV02", "label": "9B 全参 1 卡 offload 实测 35.19 GB 跑通", "kind": "success",
        "scope": ("SFT", "9B", "全参", "native", ((1, 1, 1),)),
        "reading": "`offload_optimizer_state=true` 后 `max_allocated = 35.19 GB`，训练跑完 steps=100、skipped_nonfinite=0",
        "source": "归档 .local/repo-moved-out/rig/run-plan.md:887,893",
        "effect": "success",
        "recipe": "offload",
    },
    {
        "id": "EV03", "label": "9B 全参 PP2/PP4 训练跑通但导出未实现", "kind": "success",
        "scope": ("SFT", "9B", "全参", "native", ((1, 2, 1), (1, 4, 1))),
        "reading": "PP2/PP4 训练 ec=0（T018: 200 步 / 536.9 s）；★ 但产物不可合并 ⇒ 导出/评测 fail-closed",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:52,119；缺口代码 src/graspo/export/exporters/native_full_param.py:462-471",
        "effect": "warn_only",   # 训练成功但交付层缺实现 ⇒ 记卡点，不剔除
        "delivery_gap": "导出只认 dp=1∧tp=1∧pp=1 的单份 rank_*.pt（native_full_param.py:462-471）",
    },
    # ── B. 27B LoRA（族级零余量 + 逐档失败） ───────────────────────────────
    {
        "id": "EV04", "label": "27B LoRA 族实测峰值 78.64 GiB（余 0.61 GiB）", "kind": "peak_calibration",
        "scope": ("*", "27B", "LoRA", "*", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "27B 族实测峰值 78.64 GiB / 卡容量 79.25 GiB ⇒ 余量 0.61 GiB（零余量；不是「放不下」）",
        "source": "归档 .local/repo-moved-out/rig/run-plan.md:488（U-7）",
        "effect": "calibrate",
    },
    {
        "id": "EV05", "label": "27B GRASPO LoRA native 1 卡实测 显存不足 78.77 GiB", "kind": "oom",
        "scope": ("GRASPO", "27B", "LoRA", "native", ((1, 1, 1),)),
        "reading": "「Tried to allocate 572.00 MiB … 485.88 MiB is free … 78.77 GiB in use」exit_code=1",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:72；tests/e2e/matrix54_manifest.json `graspo_lora_27b_cards.by_backend.native`",
        "effect": "backend_infeasible",
    },
    {
        "id": "EV06", "label": "27B GRASPO LoRA ms-swift dp2 实测 显存不足 78.01 GiB", "kind": "oom",
        "scope": ("GRASPO", "27B", "LoRA", "ms-swift", ((1, 1, 2),)),
        "reading": "「Tried to allocate 1.89 GiB … 1.23 GiB free … 78.01 GiB in use」exitcode=1",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:76",
        "effect": "backend_infeasible",
    },
    {
        "id": "EV07", "label": "27B GRASPO LoRA native dp2 实测异常退出且异常类型未落盘（未定性）", "kind": "runtime_failure",
        "scope": ("GRASPO", "27B", "LoRA", "native", ((1, 1, 2),)),
        "reading": ("T041：僵持 51+ min；逐字 `exception-path teardown: aborted=True … WITHOUT a rendezvous "
                    "(no WORLD barrier on the failure path)`；卡侧 `73.65 GiB in use` ⇒ **异常类型未落盘、原文明确禁止据此判 显存不足**"),
        "source": "归档 .local/T041-evidence-20261014.md:16,22",
        "effect": "warn_only",   # 未定性 ⇒ 不据此剔除（主笔：宁可留待测试，不误删）
    },
    {
        "id": "EV08", "label": "27B GRASPO LoRA native dp4 NCCL 报错", "kind": "runtime_failure",
        "scope": ("GRASPO", "27B", "LoRA", "native", ((1, 1, 4),)),
        "reading": "`NCCL error in ProcessGroup` → `torch.distributed.DistBackendError` → `ChildFailedError`；训练早期即挂",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:27,76,143",
        "effect": "warn_only",
    },
    {
        "id": "EV09", "label": "27B GRASPO LoRA ms-swift dp1/dp4 判据① 通过", "kind": "success",
        "scope": ("GRASPO", "27B", "LoRA", "ms-swift", ((1, 1, 1), (1, 1, 4))),
        "reading": "判据① 通过：T043 1280/1280 步、T045 320/320 步（判据② 未过：Δ 为负）",
        "source": "docs/capability-matrix.data.json rows[T043/T045].runs（归档 54 档台账）",
        "effect": "success",
    },
    {
        "id": "EV10", "label": "27B SFT LoRA ms-swift dp1/2/4 判据① 通过", "kind": "success",
        "scope": ("SFT", "27B", "LoRA", "ms-swift", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "T025/T026 判据① 通过且判据② 达标（Δ=+7.83/+7.12 pp）；T027（dp4）判据① 通过",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:59-61",
        "effect": "success",
    },
    {
        "id": "EV11", "label": "27B CPT LoRA ms-swift dp1/2/4 判据① 通过", "kind": "success",
        "scope": ("CPT", "27B", "LoRA", "ms-swift", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "T007/T008/T009 判据① 通过（200/100/50 步全跑满）",
        "source": "docs/capability-matrix.data.json rows[T007-T009].runs（归档 54 档台账）",
        "effect": "success",
    },
    {
        "id": "EV12", "label": "27B SFT LoRA native dp1 曾因显存被移卡", "kind": "warn",   # 记录性质：单卡 27B LoRA native 需要 67.36 GiB 空闲
        "scope": ("SFT", "27B", "LoRA", "native", ((1, 1, 1),)),
        "reading": "移卡原因逐字 `Free memory on device cuda:0 (30.82/79.25 GiB) < 0.85 (67.36 GiB)` ⇒ 该档要求 67.36 GiB 空闲",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:56",
        "effect": "warn_only",
    },
    {
        "id": "EV13", "label": "27B SFT LoRA native dp2/dp4 判据① 通过且 DP 副本逐元素相等（可合并）", "kind": "success",
        "scope": ("SFT", "27B", "LoRA", "native", ((1, 1, 2), (1, 1, 4))),
        "reading": ("T023/T024（epoch=4 新跑次）：判据① 通过；DP 副本唯一性已逐键证明（`lora_a` 606/606、"
                    "`lora_b` 606/606 逐元素相等、不等 0 ⇒ 产物唯一、可合并）；T024 明写「旧依据 = epoch=2 旧 dp4 "
                    "跑次的副本分歧，已被本跑次推翻」⇒ **旧归档件 `:57-58` 的「副本不一致、导出 fail-closed」是 epoch=2 旧读数，已作废**"),
        "source": "`docs/capability-matrix.data.json`（54 档归档件）rows[T023/T024].runs；实现 src/graspo/export/exporters/native_lora.py:455-476（`_dedupe_dp_replicas`：以 `Tensor.equal` 证明副本一致后才化简）",
        "effect": "success",
        "delivery_gap": "native LoRA 的 DP 合并**已实现**，但依赖「DP 副本位等价」这一窄口前置（分歧即 fail-closed）",
    },
    # ── C. 9B LoRA / 全参 成功与失败 ───────────────────────────────────────
    {
        "id": "EV14", "label": "9B SFT LoRA native dp1/2/4 判据① 通过", "kind": "success",
        "scope": ("SFT", "9B", "LoRA", "native", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "T010/T011/T012 判据①② 全过（Δ=+13.39 / +11.25 / +11.40 pp；T011/T012 另证 DP 副本逐元素相等 ⇒ 可合并）",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:44-46",
        "effect": "success",
    },
    {
        "id": "EV15", "label": "9B SFT LoRA ms-swift dp1/2/4 判据① 通过", "kind": "success",
        "scope": ("SFT", "9B", "LoRA", "ms-swift", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "T013/T014/T015 判据① 通过（T013 判据② 达标 Δ=+8.69 pp）",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:47-49",
        "effect": "success",
    },
    {
        "id": "EV16", "label": "9B GRASPO LoRA native dp2/dp4 NCCL 看门狗超时", "kind": "runtime_failure",
        "scope": ("GRASPO", "9B", "LoRA", "native", ((1, 1, 2), (1, 1, 4))),
        "reading": "T029/T030：NCCL 看门狗超时 600 s ⇒ Signal 6 (SIGABRT)；原文「★ 非 显存不足」",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:63-64",
        "effect": "warn_only",
    },
    {
        "id": "EV17", "label": "9B GRASPO LoRA ms-swift dp1/2/4 判据① 通过", "kind": "success",
        "scope": ("GRASPO", "9B", "LoRA", "ms-swift", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "T031/T032/T033 判据① 通过（三档判据② 未过）",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:65-67",
        "effect": "success",
    },
    {
        "id": "EV18", "label": "9B GRASPO 全参 native 1/2 卡 显存不足 记载互斥", "kind": "contested",
        "scope": ("GRASPO", "9B", "全参", "native", ((1, 1, 1), (1, 2, 1))),
        "reading": "一处记 `torch.OutOfMemoryError … Tried to allocate 96.00 MiB`（T034/T035）；另一处记这两档「从未上机」⇒ **互斥记载**",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:68-69 vs .local/本期工程跟踪.md:418-419",
        "effect": "warn_only",   # 定性证据采信不足 ⇒ 只登记冲突，不据此剔除
    },
    {
        "id": "EV19", "label": "9B GRASPO 全参 native pp4 pp-watchdog 强退", "kind": "runtime_failure",
        "scope": ("GRASPO", "9B", "全参", "native", ((1, 4, 1),)),
        "reading": "T036：`pp-watchdog` 判 PP rendezvous 停滞 300 s ⇒ force-exit 86；**非 显存不足**",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:70",
        "effect": "warn_only",
    },
    # ── D. ms-swift 全参载体（DeepSpeed 禁用后证据作废）与 FSDP2 ─────────────
    {
        "id": "EV20", "label": "ms-swift 全参的正面证据只带 DeepSpeed；FSDP2 零实测", "kind": "warn",
        "scope": ("*", "*", "全参", "ms-swift", "*"),
        "reading": "ms-swift 全参的历史正面证据全部带 DeepSpeed（T004 `zero2_offload` Δ=+8.55 pp、T006 `zero2` Δ=+7.55 pp）⇒ **DS 禁用后该证据作废**；替代载体 FSDP2 接点齐（字段+配方+12 档派生清单）但 **0 档 退出码 0 实测**",
        "source": "归档 .local/capability-baseline-milestone-20261014-v2.md:38,40；tests/e2e/matrix54_manifest_fsdp.json；src/graspo/flow/backend/msswift/_config_mapping.py:206-213",
        "effect": "warn_only",
        "delivery_gap": "FSDP2 零实测；且 `use_vllm=false` 的纯 generate rollout 在 FSDP 下的支持（上游只对 vLLM 分支做 FSDP 适配）未验证",
    },
    # ── E. OPD 教师 ───────────────────────────────────────────────────────
    {
        "id": "EV21", "label": "OPD 教师不换出 1/2/4 卡实测全 显存不足", "kind": "oom",
        "scope": ("OPD", "9B", "*", "ms-swift", ((1, 1, 1), (1, 1, 2), (1, 1, 4))),
        "reading": "教师不换出：1 卡 显存不足@1/40 步（峰值 78.06 GiB）；2 卡 79.06/79.08 GiB；4 卡四 rank 各 79.08 GiB（余 15.88 MiB）⇒ 加卡不降单卡占用",
        "source": "tests/e2e/matrix54_manifest.json:152",
        "effect": "recipe_forbid",
        "recipe": "teacher_resident",
    },
    {
        "id": "EV22", "label": "教师 offload 真透传但零 退出码 0 实测", "kind": "warn",
        "scope": ("OPD", "*", "*", "*", "*"),
        "reading": "`offload_teacher_model=true` 是真透传，但**无 退出码 0 实测**；27B×OPD 历史上「主席 2026-10-14 裁定不做」",
        "source": "src/graspo/flow/backend/msswift/_config_mapping.py:663；src/graspo/core/schema.py:666-668；归档 .local/capability-baseline-milestone-20261014-v2.md:86-88",
        "effect": "warn_only",
        "delivery_gap": "教师 offload 路径未实测（student 侧可行 ≠ 端到端可跑）",
    },
    # ── F. 并行维度真生效性（选型的"能分片"判据） ──────────────────────────
    {
        "id": "EV23", "label": "Megatron 未接线 ⇒ ms-swift 无 TP/PP 通道", "kind": "warn",
        "scope": ("*", "*", "*", "ms-swift", "*"),
        "reading": "Megatron 在本仓生产路径**未接线**（`include_megatron=False` 写死）⇒ ms-swift 的 TP/PP 无可用通道；标准通道只有 DDP + SP + FSDP2",
        "source": "src/graspo/flow/backend/msswift/_config_mapping.py:682,383-407；schema.py:858-911",
        "effect": "warn_only",
        "delivery_gap": "ms-swift 的 TP/PP 语义需一条 `swift.megatron.*` 启动通道（缺实现）",
    },
    {
        "id": "EV24", "label": "native PP+rollout 默认被闸门拒绝", "kind": "warn",
        "scope": ("GRASPO", "*", "全参", "native", "*"),
        "reading": "native 的 PP+rollout 默认被启动期闸门拒绝（`allow_unverified_pp_rollout=false`），须显式预授权才跑",
        "source": "src/graspo/flow/backend/native/runtime.py:507-522；src/graspo/core/schema.py:836-844",
        "effect": "warn_only",
    },
)


_EVIDENCE_BY_ID: dict[str, dict[str, Any]] = {}


def _labels(evs: list[Any]) -> str:
    """证据（dict 或 id）列表 → 人读标签串（备注里 ⛔ 不许出现裸 id：读者要能直接读懂）。"""
    out = []
    for ev in evs:
        if isinstance(ev, str):
            ev = _EVIDENCE_BY_ID.get(ev, {"label": ev})
        out.append(ev["label"])
    return "、".join(out)


def _shape_match(scope_shapes: Any, shape: Shape) -> bool:
    return scope_shapes == "*" or shape in scope_shapes


_EVIDENCE_BY_ID.update({ev["id"]: ev for ev in MEASURED_EVIDENCE})


def evidence_for(
    algo: str, model: str, mode: str, backend: str, shape: Shape,
    *, effects: tuple[str, ...] | None = None, kinds: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """按 (算法, 模型, 模式, 后端, 形状) 过滤证据表（`"*"` = 通配）。"""
    out = []
    for ev in MEASURED_EVIDENCE:
        a, m, md, be, sh = ev["scope"]
        if a not in ("*", algo) or m not in ("*", model) or md not in ("*", mode) or be not in ("*", backend):
            continue
        if not _shape_match(sh, shape):
            continue
        if effects is not None and ev["effect"] not in effects:
            continue
        if kinds is not None and ev["kind"] not in kinds:
            continue
        out.append(ev)
    return out


# ════════════════════════════════════════════════════════════════════════════
# §4 · 显存估算
# ════════════════════════════════════════════════════════════════════════════


def _lora_share(backend: str, tp: int, pp: int) -> int:
    """LoRA 的基座/适配器分摊倍数（DP **不**摊基座：冻结权重每 rank 一份）。"""
    return tp * pp


def _full_share(backend: str, tp: int, pp: int, dp: int) -> int:
    """全参的分摊倍数：native 只 PP（`core/schema.py:171-188`）；ms-swift 走 FSDP2 摊 tp*pp*dp。"""
    return pp if backend == "native" else tp * pp * dp


def estimate_components(
    *, model: str, mode: str, backend: str, tp: int, pp: int, dp: int,
    algo: str, recipe: str, teacher_variant: str,
) -> dict[str, float]:
    """返回每卡占用分项（GiB），可逐项审计。"""
    weights = _weights_gib(model)
    act = _act_gib(model)
    comp: dict[str, float] = {"weights": 0.0, "grad": 0.0, "optimizer": 0.0,
                              "lora_state": 0.0, "activation": act, "teacher": 0.0}
    if mode == "LoRA":
        share = _lora_share(backend, tp, pp)
        comp["weights"] = weights / share
        comp["lora_state"] = _lora_trainable(model) * LORA_STATE_BYTES_PER_PARAM / GIB / share
    else:
        k = FULL_K_BYTES["native_offload" if recipe == "offload" else backend]
        share = _full_share(backend, tp, pp, dp)
        comp["weights"] = 2.0 * MODEL_SPEC[model]["params"] / GIB / share
        comp["grad"] = 2.0 * MODEL_SPEC[model]["params"] / GIB / share
        comp["optimizer"] = (k - 4.0) * MODEL_SPEC[model]["params"] / GIB / share
    if algo == "OPD" and teacher_variant == "gpu":
        comp["teacher"] = TEACHER_WEIGHT_GIB
    return comp


def peak_gib(comp: dict[str, float]) -> float:
    return round(sum(comp.values()), 4)


def band_of(peak: float, budget: float, *, measured_anchor: bool = False) -> str:
    """三档判定：≤0.90×budget ⇒ 可行；≤budget ⇒ 余量紧；>budget ⇒ 超预算。

    ★ 第四种情形（**证据优先**）：估算值锚在**实测**族级峰值上（`measured_anchor=True`），
      且该值越过 78.0 的设计预算但**未越过卡容量 79.25** ⇒ 判「余量紧」而不是「超预算」。
      依据：主笔裁定"证据优先级：实测高于算式"；27B LoRA 族实测峰值 78.64 GiB（余 0.61 GiB）
      并且**实测跑通过**（T043 1280/1280 步）⇒ 拿设计预算把一个实测跑得起来的形状删掉，
      等于用算式否定测量。该情形必须在备注里写清"越过设计预算 X GiB、余量 = 卡容量 − 实测值"。
    """
    if peak <= TIGHT_BAND_RATIO * budget:
        return "可行"
    if peak <= budget:
        return "余量紧"
    if measured_anchor and peak <= CARD_CAPACITY_GIB:
        return "余量紧"
    return "超预算"


# ════════════════════════════════════════════════════════════════════════════
# §5 · 后端候选与选型
# ════════════════════════════════════════════════════════════════════════════

#: 每后端 × 模式的**并行能力事实**（选型理由的原料；⛔ 不许写"更方便"这类空话）。
BACKEND_PARALLEL_FACT: dict[tuple[str, str], dict[str, Any]] = {
    ("native", "全参"): {
        "supports_tp": False, "supports_pp": True, "supports_dp": False,
        "fact": "native 全参只支持 PP 分片：full + tp>1/dp>1 被加载期校验拒绝",
        "source": "src/graspo/core/schema.py:171-188",
        "tier": 0,
    },
    ("native", "LoRA"): {
        "supports_tp": True, "supports_pp": True, "supports_dp": True,
        "fact": "native LoRA 的 TP/PP/DP 三维都接线（TP SUM 同步、DP AVG 同步、PP 1F1B）",
        "short_fact": "native LoRA 的 TP/PP/DP 三维原生接线",
        "source": "src/graspo/flow/backend/native/lora/lora_linear.py:223-282；native/parallel/pipeline_comm.py",
        "tier": 0,
    },
    ("ms-swift", "全参"): {
        "supports_tp": False, "supports_pp": False, "supports_dp": True,
        "fact": "ms-swift 全参只能靠 FSDP2 摊（Megatron 段未接线 ⇒ TP/PP 无通道；DeepSpeed 已禁用）",
        "source": "src/graspo/flow/backend/msswift/_config_mapping.py:206-213,682",
        "tier": 1,
    },
    ("ms-swift", "LoRA"): {
        "supports_tp": False, "supports_pp": False, "supports_dp": True,
        "fact": "ms-swift 标准通道只有 DDP+SP（无 tp/pp 字段；Megatron 未接线）",
        "short_fact": "ms-swift 标准通道只有 DDP+SP",
        "source": "src/graspo/flow/backend/msswift/_config_mapping.py:409-436,682",
        "tier": 1,
    },
}


def backend_legal(backend: str, mode: str, tp: int, pp: int, dp: int) -> tuple[bool, str]:
    """该后端在该模式下**语义上**能不能承载这个并行形状（不改契约的前提下）。"""
    if backend == "native":
        if mode == "全参" and (tp > 1 or dp > 1):
            return False, (
                "native 全参不支持 tp>1/dp>1：加载期校验直接拒绝"
                "（原因：native 的梯度同步实现只覆盖 LoRA 参数）"
            )
        return True, ""
    return True, ""


def eval_backend(
    *, algo: str, model: str, mode: str, backend: str, shape: Shape,
    budget: float,
) -> dict[str, Any]:
    """评估一个后端候选：语义合法性 → 配方 → 实测证据 → 估算峰值。"""
    tp, pp, dp = shape
    legal, illegal_why = backend_legal(backend, mode, tp, pp, dp)
    if not legal:
        return {"backend": backend, "eligible": False, "why": illegal_why,
                "why_short": "语义被拒（native 全参只支持 PP 分片）"}

    # ① 实测证据：该后端在该档"实测不可行" ⇒ 直接出局（证据 > 算式）
    infeasible_evs = evidence_for(algo, model, mode, backend, shape, effects=("backend_infeasible",))
    if infeasible_evs:
        return {
            "backend": backend, "eligible": False,
            "why": "实测不可行证据优先于算式：" + infeasible_evs[0]["reading"],
            "why_short": f"实测不可行（{','.join(e['id'] for e in infeasible_evs)}）",
            "evidence": [e["id"] for e in infeasible_evs],
        }

    # ② 配方：全参 native 的两种配方（标准 k=8 / offload k=4）；被实测否掉的配方不进候选
    recipes = [("standard", "no_offload")]
    if mode == "全参" and backend == "native" and pp == 1:
        recipes.append(("offload", "offload"))
    if mode == "全参" and backend == "ms-swift":
        recipes = [("fsdp2", "no_offload")]

    teacher_variants = ["gpu", "offload"] if algo == "OPD" else ["none"]
    #: 该 (模型, 模式, 形状) 是否有**实测族级峰值标定**（`peak_calibration`）⇒ 估算值按实测锚定
    calibrated = bool(evidence_for(algo, model, mode, backend, shape, effects=("calibrate",)))
    forbidden_recipes = {
        ev.get("recipe") for ev in evidence_for(algo, model, mode, backend, shape, effects=("recipe_forbid",))
    }

    best: dict[str, Any] | None = None
    all_variants: list[dict[str, Any]] = []
    for recipe_name, recipe_key in recipes:
        if recipe_key in forbidden_recipes:
            all_variants.append({"recipe": recipe_name, "recipe_key": recipe_key,
                                 "forbidden_by_evidence": True, "peak_gib": None})
            continue
        for tv in teacher_variants:
            if tv == "gpu" and "teacher_resident" in forbidden_recipes:
                all_variants.append({"recipe": recipe_name, "teacher": tv,
                                     "forbidden_by_evidence": True, "peak_gib": None})
                continue
            comp = estimate_components(model=model, mode=mode, backend=backend, tp=tp, pp=pp, dp=dp,
                                       algo=algo, recipe=recipe_key, teacher_variant=tv)
            peak = peak_gib(comp)
            item = {
                "recipe": recipe_name, "recipe_key": recipe_key, "teacher": tv,
                "components_gib": {k: round(v, 4) for k, v in comp.items()},
                "peak_gib": peak, "band": band_of(peak, budget, measured_anchor=calibrated),
                "measured_anchor": calibrated,
            }
            all_variants.append(item)
            if item["band"] == "超预算":
                continue                      # ⛔ 超预算的配方不进候选
            if best is None or peak < best["peak_gib"]:
                best = item

    if best is None:
        overs = [v for v in all_variants if v.get("peak_gib") is not None]
        if overs:
            best_over = min(overs, key=lambda v: v["peak_gib"])
            return {
                "backend": backend, "eligible": False,
                "why": (f"最佳情形仍超预算：{best_over['peak_gib']:.2f} GiB > {budget:.1f} GiB"
                        f"（配方 {best_over['recipe']}"
                        + (f" / 教师 {best_over['teacher']}" if algo == 'OPD' else "") + "）"),
                "why_short": f"最佳情形仍超预算 {best_over['peak_gib']:.2f} > {budget:.1f} GiB",
                "peak_gib": best_over["peak_gib"],
                "variants": all_variants,
            }
        return {"backend": backend, "eligible": False,
                "why": "全部配方都被实测证据否掉", "variants": all_variants}

    # ── 产物可推理（裁定 2026-10-21 ②）：native 的导出层只认 dp=tp=pp=1 ──────
    #   `export/exporters/native_full_param.py:462-471` 逐字：DP replica reconciliation /
    #   TP row reassembly / PP layer assembly are NOT implemented, MUST stay fail-closed；
    #   `export/exporters/native_lora.py:455-476` 要求同 (tp,pp) 副本 Tensor.equal 位等价，
    #   T023/T024 实测 `lora_a` 606/606 全不同 ⇒ 导出 fail-closed。
    #   ⇒ `native + LoRA + tp=pp=1 + dp>1`：**训练能跑、产物交不出** ⇒ 同等条件下让 ms-swift 上位。
    #      （tp>1/pp>1 的 LoRA 档不参与此判据：那些形状 ms-swift 根本没通道，仍选 native，只写卡点。）
    # ★ 判据④ 的**准确**语义（裁定后复核更正）：
    #   · native LoRA + **tp>1**：仓内**没有 TP 合并实现**（`native_lora.py` 只做 DP 副本化简 `_dedupe_dp_replicas`
    #     与 PP 覆盖校验 `_assert_pp_coverage`；`native_full_param.py:462-471` 亦逐字标注 TP row reassembly 未实现）
    #     ⇒ 产物真的交不出。
    #   · native LoRA + **dp>1（tp=pp=1）**：DP 副本化简**已实现**（以 `Tensor.equal` 证明副本一致后才化简），
    #     且 T011/T012/T023/T024 实测副本逐元素相等 ⇒ 该形状产物**可合并**；但该实现把"副本位等价"当硬前置，
    #     历史上同形状 epoch=2 旧跑次曾分歧 ⇒ 按裁定② 的口径（同等条件下产标准 HF checkpoint 的一方更稳）不首选。
    #   ⛔ 因此 note 里**不许**把 dp>1 写成"交不出产物"。
    deliverable, delivery_gap, delivery_gap_short = True, None, None
    if backend == "native" and mode == "LoRA" and tp > 1:
        deliverable = False
        delivery_gap_short = ("native 的多卡产物导出只覆盖数据并行的副本合并，张量并行的结果拼接尚未实现")
        delivery_gap = delivery_gap_short
    elif backend == "native" and mode == "LoRA" and tp == 1 and pp == 1 and dp > 1:
        deliverable = False
        delivery_gap_short = ("native 的 DP 产物合并要求各卡副本逐元素相等，否则拒绝导出；"
                              "既有 9B/27B 的 LoRA dp2/dp4 跑次实测副本相等、可合并，"
                              "epoch=2 旧 dp4 跑次曾出现分歧")
        delivery_gap = delivery_gap_short

    cap_fact = BACKEND_PARALLEL_FACT[(backend, mode)]
    fits_shape = (
        (tp == 1 or cap_fact["supports_tp"])
        and (pp == 1 or cap_fact["supports_pp"])
        and (dp == 1 or cap_fact["supports_dp"])
    )
    success_evs = evidence_for(algo, model, mode, backend, shape, effects=("success",))
    warn_evs = evidence_for(algo, model, mode, backend, shape, effects=("warn_only",))
    return {
        "backend": backend,
        "eligible": True,
        "peak_gib": best["peak_gib"],
        "band": best["band"],
        "measured_anchor": bool(best.get("measured_anchor")),
        "recipe": best["recipe"],
        "recipe_key": best["recipe_key"],
        "teacher_variant": best["teacher"],
        "components_gib": best["components_gib"],
        "variants": all_variants,
        "evidence": [e["id"] for e in success_evs] + [e["id"] for e in warn_evs],
        "success_evidence": [e["id"] for e in success_evs],
        "warn_evidence": [e["id"] for e in warn_evs],
        "fits_shape": fits_shape,
        "deliverable": deliverable,
        "delivery_gap": delivery_gap,
        "delivery_gap_short": delivery_gap_short,
        "capability_tier": 0 if fits_shape else 2,   # 0 = 该形状真被后端消费；2 = 形状要靠别的载体兜
        "fact": cap_fact["fact"],
        "source": cap_fact["source"],
    }


def select_backend(
    *, algo: str, model: str, mode: str, shape: Shape, budget: float,
) -> dict[str, Any]:
    """逐档选后端。排序键（升序取第一个）：

        ① 实测成功证据（有 > 无）
        ② 该形状是否被后端真消费（是 > 否：TP/PP 在 native 真接线；ms-swift 无通道）
        ③ 分片载体是否已实现且已实测（FSDP2 接点齐但零实测 ⇒ 排后）
        ④ 产物可推理（是 > 否：native 导出层只认 dp=tp=pp=1 ⇒ LoRA 的 dp>1 档交不出东西）
        ⑤ 估算峰值（更小者优先，等价于"能用更少卡/更省显存"）
        ⑥ 后端名（native < ms-swift，确定性尾键）

    理由必须可判真假 ⇒ 每个候选都带 `fact` + `source`，不写"更方便"。
    """
    cands = [eval_backend(algo=algo, model=model, mode=mode, backend=be, shape=shape, budget=budget)
             for be in ALGO_BACKEND_SUPPORT[algo]]
    # CPT/OPD 的 native 缺失是**算法层缺实现**：登记卡点，不进候选（不是剔除档位）
    missing = [ALGO_BACKEND_NOTE[(algo, be)] for be in BACKENDS
               if be not in ALGO_BACKEND_SUPPORT[algo] and (algo, be) in ALGO_BACKEND_NOTE]

    eligible = [c for c in cands if c["eligible"]]
    for c in cands:
        c["rank_key"] = None
    ranked = sorted(
        eligible,
        key=lambda c: (
            0 if c["success_evidence"] else 1,          # ①
            c["capability_tier"],                        # ② + ③
            0 if c.get("deliverable", True) else 1,      # ④ 产物可推理
            round(c["peak_gib"], 3),                     # ⑤
            BACKENDS.index(c["backend"]),                # ⑥
        ),
    )
    for i, c in enumerate(ranked):
        c["rank_key"] = i
    chosen = ranked[0] if ranked else None
    return {"candidates": cands, "chosen": chosen, "missing_backends": missing}


# ════════════════════════════════════════════════════════════════════════════
# §6 · 枚举 → 剔除 → 行构造
# ════════════════════════════════════════════════════════════════════════════

#: 原因码（剔除清单里的枚举；`ALGO_BACKEND` **不是剔除**，见主笔裁定「缺实现 ≠ 不合理」）。
REASON_CODES: dict[str, dict[str, Any]] = {
    "EXCEED_CARDS": {
        "desc": "tp*pp*dp > card_limit ⇒ 枚举阶段就不生成档位",
        "excludes": True,
    },
    "MEMORY_UNFIT": {
        "desc": "按显存模型估算，**所有**可选后端的最佳情形仍超预算 ⇒ 不合理",
        "excludes": True,
    },
    "KNOWN_MEMORY_FAIL": {
        "desc": "该档所有可选后端都有**实测** 显存不足/不可用证据 ⇒ 不合理（证据优先于算式）",
        "excludes": True,
    },
    "ALGO_BACKEND": {
        "desc": "某算法在某后端缺实现 —— ★ 定稿判据：**不算剔除**，保留为待测试并写清卡点",
        "excludes": False,
    },
    "MISSING_CHANNEL": {
        "desc": ("**唯一能装下的后端缺该算法通道**（物理组合成立、拦住它的是我们自己的欠账）"
                 "⇒ ★ 定稿判据 2026-10-21：**不算剔除**，保留为待测试，按补齐通道后的配方估算"),
        "excludes": False,
    },
    "UNSUPPORTED_MODE": {
        "desc": "某算法在**任何**后端都不支持该模式 ⇒ 该组合不生成",
        "excludes": True,
    },
}

#: 学习率策略：lr = f(算法, 模式)，**不按卡数缩放**。
#:   单一真相源 `tools/matrix/extract_spec.py:45-54`（`LR_POLICY`），落盘
#:   `samples/configs/matrix54-generated/T010.yaml:18`（2.0e-05）/ `T001.yaml:18`（5.0e-06）。
LR_POLICY: dict[tuple[str, str], str] = {
    ("SFT", "LoRA"): "2.0e-05",
    ("SFT", "全参"): "5.0e-06",
    ("CPT", "LoRA"): "5.0e-06",
    ("CPT", "全参"): "5.0e-06",
    ("GRASPO", "LoRA"): "5.0e-06",
    ("GRASPO", "全参"): "5.0e-06",
    ("OPD", "LoRA"): "2.0e-05",
    ("OPD", "全参"): "5.0e-06",
}

#: 训练轮次：epochs = f(算法)，与后端/卡数无关（`docs/capability-matrix.data.json` rows[].max_epochs）。
EPOCH_POLICY: dict[str, int] = {"CPT": 2, "SFT": 4, "GRASPO": 8, "OPD": 2}

#: 每档训练子集条数（口径沿用 `tests/e2e/generate_matrix.py:504` `SUBSET_SIZE_BY_ALGORITHM`）。
SUBSET_SIZE_BY_ALGO: dict[str, int] = {"CPT": 100, "SFT": 100, "GRASPO": 20, "OPD": 20}

#: 备注/开关里用到的固定事实。
TEACHER_MODEL_PATH = "/models/Qwen3.8-27B"


def enumerate_shapes(card_limit: int) -> list[Shape]:
    return [(tp, pp, dp)
            for tp in PARALLEL_VALUES
            for pp in PARALLEL_VALUES
            for dp in PARALLEL_VALUES
            if tp * pp * dp <= card_limit]


def sort_key(rec: dict[str, Any]) -> tuple:
    tp, pp, dp = rec["shape"]
    return (MODELS.index(rec["model"]), ALGOS.index(rec["algo"]), MODES.index(rec["mode"]),
            tp, pp, dp)


def build_candidates(card_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回 (枚举出来的候选组合, 被 EXCEED_CARDS 挡在枚举外的组合)。

    ★ 语义：`tp*pp*dp > card_limit` 的组合**不进候选**（枚举阶段就不生成），
      但仍计数登记（否则"全量排列组合 432"这个数字无从核对）。
    """
    kept: list[dict[str, Any]] = []
    exceeded: list[dict[str, Any]] = []
    for model in MODELS:
        for algo in ALGOS:
            for mode in MODES:
                for tp in PARALLEL_VALUES:
                    for pp in PARALLEL_VALUES:
                        for dp in PARALLEL_VALUES:
                            rec = {"model": model, "algo": algo, "mode": mode, "shape": (tp, pp, dp)}
                            if tp * pp * dp > card_limit:
                                exceeded.append(rec)
                            else:
                                kept.append(rec)
    kept.sort(key=sort_key)
    exceeded.sort(key=sort_key)
    return kept, exceeded


def row_of(tier: str, rec: dict[str, Any], sel: dict[str, Any], budget: float) -> dict[str, Any]:
    """把 (组合, 选型结果) 落成 data.json 的一行。"""
    model, algo, mode = rec["model"], rec["algo"], rec["mode"]
    tp, pp, dp = rec["shape"]
    cards = tp * pp * dp
    chosen = sel["chosen"]
    backend = chosen["backend"]
    peak = chosen["peak_gib"]
    band = chosen["band"]
    train_method = ALGO_TO_TRAIN_METHOD[algo]
    tuner_type = MODE_TO_TUNER_TYPE[mode]

    # ── 卡点（缺实现 ≠ 不合理）：逐条写清卡在哪一层 ────────────────────────
    gaps: list[str] = []
    if sel.get("missing_channel"):
        gaps.append(f"native 侧尚无 {algo} 训练通道（算法层未提供该训练方法在 native 后端的实现）；"
                    f"补齐后按 native 配方估算 {peak:.2f} GiB 可行")
    if backend == "ms-swift":
        if mode == "全参":
            gaps.append("FSDP2 分片载体零 退出码 0 实测")
        if tp > 1 or pp > 1:
            gaps.append("ms-swift 的 TP/PP 无通道（并行实现尚未接入，标准通道只覆盖 DDP+SP）"
                        "⇒ 该形状的分摊只能靠 FSDP2 全分片，语义待实测")
    if backend == "native" and mode == "全参":
        if pp > 1:
            gaps.append("native 全参的流水线并行产物导出尚未实现（目前只支持单卡单机导出）")
        if algo == "GRASPO" and pp > 1:
            gaps.append("native 的流水线并行 + rollout 组合须显式解除未验证保护"
                        "（`native.allow_unverified_pp_rollout=true`）")
    if backend == "native" and mode == "LoRA" and (tp > 1 or pp > 1):
        # ★ 必修（裁定后补）：历史 54 档里 LoRA 的 tp=pp=1 恒成立（TP/PP 只出现在全参档）
        #   ⇒ 这些形状**从未实跑**，⛔ 不许写"训练可跑/能跑"。
        if tp > 1:
            gaps.append("该形状从未实跑（既有档位里 LoRA 恒为 tp=pp=1，TP/PP 只出现在全参档）"
                        "⇒ 训练可行性未建立；产物侧另需补导出层：多卡导出仅覆盖数据并行的副本合并，"
                        "张量并行的结果拼接尚未实现")
        else:
            gaps.append("该形状从未实跑（既有档位里 LoRA 恒为 tp=pp=1，TP/PP 只出现在全参档）"
                        "⇒ 训练可行性未建立；产物侧流水线并行分片的覆盖校验已实现，但无实跑验证")
    if chosen["recipe"] == "offload":
        gaps.append("需 `native.offload_optimizer_state=true`（实测 35.19 GB）")
        if algo == "GRASPO":
            gaps.append("offload 目前只在 SFT 通道生效，RL 通道尚未接入")
    if algo == "OPD" and chosen["teacher_variant"] == "offload":
        gaps.append("需 `offload_teacher_model=true`（该路径零 退出码 0 实测）")
    gaps = gaps[:4]

    # ── 边缘/冲突提示（实测 vs 算式） ─────────────────────────────────────
    notes_extra: list[str] = []
    if band == "余量紧":
        if peak > budget:
            if chosen["success_evidence"]:
                basis = f"该形状实测跑通过（{_labels(chosen['success_evidence'])}）⇒ 证据优先"
            elif chosen.get("measured_anchor"):
                basis = f"该值锚在实测族级峰值（27B LoRA 78.64 GiB、余 0.61 GiB）"
            else:
                basis = "算式估算"
            notes_extra.append(
                f"余量紧：估算 {peak:.2f} GiB 越过 {budget:.1f} 设计预算 {peak - budget:.2f} GiB、"
                f"卡容量内余量仅 {CARD_CAPACITY_GIB - peak:.2f} GiB（{basis}），需实测确认")
        else:
            notes_extra.append(f"余量紧：估算 {peak:.2f}/{budget:.1f} GiB，需实测确认")
    if rec["algo"] == "GRASPO" and model == "27B" and mode == "LoRA" and (tp, pp) == (1, 1) and backend == "native":
        notes_extra.append("该形状 native 曾异常退出（未定性为 显存不足）、ms-swift 实测 显存不足")
    contested = [e for e in MEASURED_EVIDENCE
                 if e["kind"] == "contested"
                 and e["scope"][0] in ("*", algo) and e["scope"][1] in ("*", model)
                 and e["scope"][2] in ("*", mode) and _shape_match(e["scope"][4], (tp, pp, dp))]
    for ev in contested:
        notes_extra.append(f"该形状历史记载互斥（{ev['label']}）⇒ 须实测确认")
    if algo == "OPD":
        gpu_variant = next((v for v in chosen.get("variants", [])
                            if v.get("teacher") == "gpu" and v.get("peak_gib") is not None), None)
        if chosen["teacher_variant"] == "offload" and gpu_variant is not None:
            notes_extra.append(
                f"教师必须 offload：GPU 常驻情形估算 {gpu_variant['peak_gib']:.2f} GiB（超预算，"
                "实测 1/2/4 卡全 显存不足）；offload 路径零 退出码 0 实测")
        elif chosen["teacher_variant"] == "gpu":
            notes_extra.append("教师 GPU 常驻：实测 1/2/4 卡全 显存不足")

    # ── 选型理由（一句话，可判真假） ─────────────────────────────────────
    reason_line = _reason_line(chosen, sel, backend, mode, tp, pp, dp, band, budget, peak, algo)

    note_parts = [reason_line]
    if notes_extra:
        note_parts.extend(notes_extra)
    if gaps:
        note_parts.append("卡点：" + "；".join(gaps))
    note = "；".join(p for p in note_parts if p)

    # ── 专用：native LoRA 的 dp>1 档（双方都无成功跑次）—— 必修（裁定后补）
    #    要素：① 两方均无成功跑次（各带具名跑次）② 选 native 的唯一理由 = 对方有实测 显存不足、
    #    本方是通信/teardown 异常（可诊断）③ 余量紧 ⇒ 先解决余量与通信稳定性 ④ 产物另需补导出层。
    #    ⛔ 一个字都不许写"训练可跑/能跑"。
    if backend == "native" and mode == "LoRA" and tp == 1 and pp == 1 and dp > 1 \
            and not chosen["success_evidence"]:
        ms = next((c for c in sel["candidates"] if c["backend"] == "ms-swift"), None)
        native_fail = (f"native —— {_labels(chosen['warn_evidence'])}" if chosen.get("warn_evidence")
                       else "native —— 无成功跑次记录")
        ms_fail = (f"ms-swift —— {_labels(ms['evidence'])}" if (ms and ms.get("evidence"))
                   else "ms-swift —— 无成功跑次记录")
        note = (f"两者在该形状均无成功跑次：{native_fail}；{ms_fail}。"
                f"选 native 的唯一理由 = 对方有实测 显存不足、本方是通信/teardown 异常（可诊断、可重试），"
                f"⛔ 不表示该形状可行。估算 {peak:.2f}/{budget:.1f} GiB（{band}）⇒ "
                "测试前须先解决显存余量与通信稳定性。"
                f"卡点：产物侧另需补导出层 —— {chosen.get('delivery_gap_short') or 'DP 合并依赖副本位等价'}")

    # ── 步数推导（沿用旧口径：子集条数 ÷ 每 epoch 卡数 × epochs；本轮不跑批次） ──
    subset = SUBSET_SIZE_BY_ALGO[algo]
    # GRASPO 的采样步数按 rollout 组计（`tests/e2e/generate_matrix.py:581` ROLLOUT_GROUP_SIZE=8），
    # 本轮不重生成配置 ⇒ 只登记朴素推导值，并在 `steps_basis` 说明它待生成侧口径对齐。
    per_epoch = max(1, math.ceil(subset / max(1, cards if not (mode == "全参" and backend == "native") else pp)))
    epochs = EPOCH_POLICY[algo]
    derived = per_epoch * epochs

    switches = _switches_of(algo, mode, backend, chosen, pp)
    allow_pp_rollout = "true" if (backend == "native" and algo == "GRASPO" and pp > 1) else "—"

    row: dict[str, Any] = {
        "tier": tier,
        "id": tier,
        "model": model,
        "algo": algo,
        "mode": mode,
        "backend": backend,
        "parallel": (f"nproc_per_node={cards}（{cards} 卡）" if backend == "ms-swift"
                     else f"tp={tp} pp={pp} dp={dp}（{cards} 卡）"),
        # ★ gen_configs 的同源断言读 `native_tp/native_pp/native_dp`（`gen_configs.py:79-110`）：
        #   两个后端都填，让「卡数 == tp×pp×dp」这条防呆**真的能跑**（⛔ 不靠字段缺失跳过校验）。
        "native_tp": str(tp), "native_pp": str(pp), "native_dp": str(dp),
        "parallel_scheme": (
            "native: tp_size/dp_size/pp_size（三维真接线）" if backend == "native"
            else "ms-swift: nproc_per_node(+nnodes)；tp/pp 语义由 FSDP2 分片承载（Megatron 未接线）"
        ),
        "parallel_dims": f"TP={tp} PP={pp} DP={dp}；SP={'auto（TP>1 自动开）' if backend == 'native' and tp > 1 else 'off'}",
        "tp": str(tp), "pp": str(pp), "dp": str(dp),
        "tp_size": str(tp), "pp_size": str(pp), "dp_size": str(dp),
        # 卡数三个键各有唯一消费者（⛔ 不是三份真相：值同源、由本行唯一处生成）：
        #   `cards`（gen_configs 读）｜`cards_total`（render_matrix 的「总卡数」列读）｜
        #   `total_cards`（主笔交付规格里点名的字段名）。
        "cards": str(cards), "cards_total": str(cards), "total_cards": str(cards),
        "nproc_per_node": str(cards) if backend == "ms-swift" else "—",
        "nnodes": "1" if backend == "ms-swift" else "—",
        "gpus": "—",                      # 尚未排产：本轮只出设计矩阵，不跑批次
        "train_method": train_method,
        "tuner_type": tuner_type,
        "model_path": MODEL_SPEC[model]["label"],
        "lr": LR_POLICY[(algo, mode)],
        "max_epochs": epochs,
        "save_steps": derived + 1,
        "steps": f"{epochs} epoch × {per_epoch} 步/epoch = 推导 {derived} 步",
        "steps_basis": "沿用旧生成规则的换算（子集条数 ÷ 卡数 × epochs）；GRASPO 的 rollout 组规模待生成侧统一",
        "switches": switches,
        "deepspeed": "—",                 # DeepSpeed 已禁用：任何档都不再以 DS 为配方
        "allow_unverified_pp_rollout": allow_pp_rollout,
        "cp_size": "1",
        "sp_size": "auto" if backend == "native" and tp > 1 else "off",
        "subset_size": str(subset),
        "prompts_per_rank": str(max(1, math.ceil(subset / cards))),
        "per_epoch_steps": str(per_epoch),
        "derived_steps": str(derived),
        "planned_steps": str(per_epoch),
        "train_path": f"subsets-v2/{tier}.jsonl",
        "data_source": "ELAM V5 mini 冒烟集（按算法子集条数）",
        "data_subset": "待生成",
        "lora_target": "真实 HF 名正则（两后端共用同一条；取值以核心 LoRA 配置的 target 名单为准）" if mode == "LoRA" else "—",
        "applicable": True,
        # `runnable`：**今天**这个后端能不能起跑该算法（⛔ 与 `status` 正交：status 是"测没测过"，
        #   runnable 是"通道在不在"）。MISSING_CHANNEL 档 = 通道缺 ⇒ false，配置装载期会被
        #   `core/schema.py:29-34` 拒绝；下游生成配置时应按此跳过（见 meta.enumeration.known_gaps）。
        "runnable": not bool(sel.get("missing_channel")),
        "blocked_on": (f"{backend} 侧尚无 {algo} 训练通道"
                       f"（算法层未提供该训练方法在 {backend} 后端的实现）"
                       if sel.get("missing_channel") else "—"),
        "status": "待测试",               # ★ 主笔要求：每一个档位都是待测试
        # ── 渲染需要的实验数据字段：全部置 "—"（§17.7：未验证不得携带读数） ──
        "runs": "—",
        "peak_mem": "—",
        "steps_actual": "—",
        "elapsed": "—",
        "throughput": "—",
        "effect": "—",
        "not_failed_reason": "—",
        "note": note,
        "backend_reason": {
            "selected": backend,
            "sort_key": ("①实测成功证据 ②形状是否被后端真消费 ③分片载体已实现且已实测 "
                         "④产物可推理（native 导出只认 dp=tp=pp=1） ⑤估算峰值 ⑥后端名"),
            "rule": _reason_rule(sel, backend),
            "peak_gib": round(peak, 4),
            "budget_gib": budget,
            "band": band,
            "measured_anchor": bool(chosen.get("measured_anchor")),
            "over_budget_gib": round(max(0.0, peak - budget), 4),
            "components_gib": chosen["components_gib"],
            "recipe": chosen["recipe"],
            "teacher_variant": chosen["teacher_variant"],
            "backend_implemented": not bool(sel.get("missing_channel")),
            "deliverable": bool(chosen.get("deliverable", True)),
            "delivery_gap": chosen.get("delivery_gap"),
            "evidence": chosen["evidence"],
            "candidates": [
                {
                    "backend": c["backend"],
                    "eligible": c["eligible"],
                    "why": c.get("why") or c.get("fact"),
                    "source": c.get("source"),
                    "peak_gib": c.get("peak_gib"),
                    "band": c.get("band"),
                    "recipe": c.get("recipe"),
                    "success_evidence": c.get("success_evidence", []),
                    "rank_key": c.get("rank_key"),
                }
                for c in sorted(sel["candidates"],
                                key=lambda c: (c.get("rank_key") is None, c.get("rank_key") or 99,
                                               BACKENDS.index(c["backend"])))
            ],
            "gaps": gaps,
        },
        "spec_json": "—",
    }
    if algo == "OPD":
        # ★ gen_configs 的 fail-closed 断言（`gen_configs.py:65-76`）：`meta.distill_teacher`
        #   标了 measured 但 `min_cards` 未定 ⇒ OPD 档必须带 `cards_pending` 标记，
        #   否则「教师卡数未定」会被静默当已定。本矩阵的教师不计入卡数（主笔原文）⇒ 恰属未定态。
        row["cards_pending"] = True
    # ── 发布件 / 审计件分离（阻断修复）：内部审计字段收进 `_audit`，发布件只留渲染+消费字段 ──
    #   ★ 发布件行键必须 ⊆ `build_official.HUMAN_VIEW_ROW_KEYS`（机核见 V16）；
    #     审计字段（候选排序/落选原文/证据号/源码路径/生成依据全文）只进 `.local` 审计件。
    blocked_on = row.get("blocked_on") or "—"
    row["blocked_reason"] = blocked_on          # 人读卡点（允许清单内的既有字段）
    audit: dict[str, Any] = {}
    for k in ("id", "native_tp", "native_pp", "native_dp", "total_cards",
              "runnable", "blocked_on", "steps_basis", "backend_reason"):
        if k in row:
            audit[k] = row.pop(k)
    row["_audit"] = audit
    return row


def _reason_line(chosen, sel, backend, mode, tp, pp, dp, band, budget, peak, algo: str) -> str:
    """一句话选型理由（可判真假：每个分句都能用候选的实测/代码事实反驳或印证）。"""
    others = [c for c in sel["candidates"] if c["backend"] != backend]
    other = others[0] if others else None
    ev = _labels(chosen["success_evidence"]) or "无"
    if sel.get("missing_channel"):
        # ★ 裁定 1：唯一能装下的是缺通道的那个后端 ⇒ 保留待测试，理由写清"是欠账不是不合理"
        over = [c for c in sel["candidates"] if c["backend"] != backend and c.get("peak_gib") is not None]
        over_txt = "、".join(f"{c['backend']} 最佳情形 {c['peak_gib']:.2f} GiB 超预算" for c in over) \
            or "其余后端均不可行"
        return (f"选 {backend}：{over_txt}；只有 {backend} 补齐 {algo} 通道后能装下"
                f"（按补齐后的配方估算 {peak:.2f}/{budget:.1f} GiB，{band}）"
                "⇒ 缺实现≠档位不合理，保留待测试")
    if (other is not None and other["eligible"] and chosen.get("deliverable", True)
            and not other.get("deliverable", True)):
        # ⚠ 只在"双方都有实测成功跑次"时才敢这么写；否则走下面的通用分支
        if chosen["success_evidence"] and other["success_evidence"]:
            core = (f"选 {backend}：两者都有实测成功跑次（{backend} {_labels(chosen['success_evidence'])}；"
                    f"{other['backend']} {_labels(other['success_evidence'])}），但 {other['backend']} 的产物交付"
                    f"带一道未闭合前置 —— {other.get('delivery_gap_short') or other.get('delivery_gap')} "
                    f"⇒ 按「产物可推理」这一判据取 {backend}")
        else:
            core = (f"选 {backend}：{other['backend']} 的产物交付带一道未闭合前置 —— "
                    f"{other.get('delivery_gap_short') or other.get('delivery_gap')}；"
                    f"且 {other['backend']} 在该形状无成功跑次 ⇒ 按「产物可推理」这一判据取 {backend}")
    elif other is None:
        if sel["missing_backends"]:
            core = (f"选 {backend}：{algo} 只有 {backend} 通道 —— {sel['missing_backends'][0]}；"
                    "缺实现≠档位不合理 ⇒ 保留待测试")
        else:
            core = f"选 {backend}：该档只有这一个候选后端"
    elif other["eligible"]:
        if chosen["success_evidence"] and other["success_evidence"]:
            core = (f"选 {backend}：两个后端都有实测成功证据（{backend} {ev}；"
                    f"{other['backend']} {_labels(other['success_evidence'])}）⇒ 按「更简单路径」取 {backend}"
                    f"（{chosen.get('short_fact') or chosen['fact']}）")
        elif chosen["success_evidence"]:
            why_other = ("实测记录只有：" + _labels([e for e in MEASURED_EVIDENCE
                                                    if e["id"] in other["warn_evidence"]])
                         if other.get("warn_evidence") else "无实测跑次")
            core = (f"选 {backend}：{backend} 有实测成功证据（{ev}），"
                    f"{other['backend']} 无实测成功证据（{why_other}）")
        else:
            core = (f"选 {backend}：两者都无实测成功证据 ⇒ 按「形状是否被后端真消费」取 {backend}"
                    f"（{chosen['fact']}；{other['backend']} 的形状要靠别的载体兜）")
    else:
        short = other.get("why_short") or other.get("why") or ""
        if other.get("evidence"):
            short = f"实测不可行（{_labels(other['evidence'])}）"
        core = f"选 {backend}：{other['backend']} 出局 —— {short}"
    tail = f"估算 {peak:.2f}/{budget:.1f} GiB（{band}）"
    return f"{core}；{tail}"


def _reason_rule(sel, backend) -> str:
    """结构化理由里的"为什么是它" —— 与 `backend_rank_sort` 一一对应。"""
    ranked = sorted([c for c in sel["candidates"] if c["eligible"]],
                    key=lambda c: c.get("rank_key") if c.get("rank_key") is not None else 99)
    if not ranked:
        return "无可行候选"
    top = ranked[0]
    why = []
    why.append(f"排序键第 1 位 = 实测成功证据：{backend} {'有' if top['success_evidence'] else '无'}"
               f"（{','.join(top['success_evidence']) if top['success_evidence'] else '—'}）")
    why.append(f"第 2 位 = 形状是否被后端真消费：{'是' if top['fits_shape'] else '否'}")
    why.append(f"第 4 位 = 产物可推理：{'是' if top.get('deliverable', True) else '否（' + str(top.get('delivery_gap')) + '）'}")
    for other in ranked[1:]:
        why.append(f"次优 {other['backend']}：峰值 {other['peak_gib']:.2f} GiB、"
                   f"实测成功 {','.join(other['success_evidence']) if other['success_evidence'] else '无'}、"
                   f"形状真消费 {'是' if other['fits_shape'] else '否'}、"
                   f"产物可推理 {'是' if other.get('deliverable', True) else '否'}")
    for c in sel["candidates"]:
        if not c["eligible"]:
            why.append(f"落选 {c['backend']}：{c['why']}")
    return "；".join(why)


def _switches_of(algo: str, mode: str, backend: str, chosen: dict[str, Any], pp: int) -> str:
    """关键开关：算法级 token + 后端级 token（⛔ 不含 deepspeed）。"""
    toks: list[str] = []
    if algo == "CPT":
        toks += ["loss_scale=all", "chat_template=false"]
    if algo == "OPD":
        toks += [f"teacher={Path(TEACHER_MODEL_PATH).name}", "offload_teacher=true"]
    if backend == "ms-swift" and mode == "全参":
        toks.append("fsdp2=full_state_dict")
    if chosen.get("recipe") == "offload":
        toks.append("offload_optimizer_state=true")
    if backend == "native" and algo == "GRASPO" and pp > 1:
        toks.append("allow_unverified_pp_rollout=true")
    return ";".join(toks) if toks else "—"


# ════════════════════════════════════════════════════════════════════════════
# §7 · 自检（把报告里的回验表变成可执行断言 —— 断言不过 ⇒ rc=2，不写文件）
# ════════════════════════════════════════════════════════════════════════════


def _sel_by(rows, algo, model, mode, shape):
    tp, pp, dp = shape
    for r in rows:
        if (r["algo"], r["model"], r["mode"], int(r["tp"]), int(r["pp"]), int(r["dp"])) == (algo, model, mode, tp, pp, dp):
            return r
    return None


def _est(algo, model, mode, backend, shape, budget, recipe=None, teacher_variant=None):
    """直接算一个 (backend, 配方, 教师情形) 的估算峰值 —— 供自检用。"""
    tp, pp, dp = shape
    if recipe is None:
        recipe = "offload" if (mode == "全参" and backend == "native" and pp == 1) else "no_offload"
    if teacher_variant is None:
        teacher_variant = "gpu" if algo == "OPD" else "none"
    comp = estimate_components(model=model, mode=mode, backend=backend, tp=tp, pp=pp, dp=dp,
                               algo=algo, recipe=recipe, teacher_variant=teacher_variant)
    return peak_gib(comp)


def _v11_lora_dp_deliverable(rows, budget: float) -> bool:
    """裁定 2026-10-21 ②：LoRA ∧ dp>1 ∧ tp=pp=1 一律 ms-swift，除非 ms-swift 在该形状实测不可行。"""
    for r in rows:
        if r["mode"] != "LoRA" or int(r["tp"]) != 1 or int(r["pp"]) != 1 or int(r["dp"]) <= 1:
            continue
        if r["backend"] == "ms-swift":
            continue
        ms = next((c for c in r["backend_reason"]["candidates"] if c["backend"] == "ms-swift"), None)
        if ms is None or ms["eligible"] or "实测不可行" not in (ms.get("why") or ""):
            return False
    return True


def _v12_missing_channel(rows, budget: float) -> bool:
    """裁定 2026-10-21 ①：MISSING_CHANNEL 档必须保留为待测试，且 note 写清卡点与复活估算。"""
    mc = [r for r in rows if r["backend_reason"].get("backend_implemented") is False]
    if not mc:
        return False
    for r in mc:
        note = r["note"]
        if r["status"] != "待测试" or "卡点" not in note:
            return False
        if f"native 侧尚无 {r['algo']} 训练通道" not in note or "补齐后按 native 配方估算" not in note:
            return False
    return True


#: 旧 54 档档号形态（V14 机核）：发布件（`docs/capability-matrix.data.json`）里必须为 0。
#:   ★ 溯源（档号 + 路径 + 行号）只留在本脚本证据表的 `reading`/`source` 与 `.local` 工位报告里；
#:     ⛔ 发布件里的人读文案一律用描述性指代（"既有 9B/27B LoRA dp2/dp4 跑次"）。
TIER_ID_RE = re.compile(r"T\d{3}")


#: 发布件用词中性化（**顺序敏感**：先短语、后兜底）。
#:   ★ 只换过程用语，⛔ 不改结论、⛔ 不动数值、⛔ 不动 file:line 溯源（机读台账的审计线索要留）。
PUBLISH_WORD_REWRITES: tuple[tuple[str, str], ...] = (
    ("评测集口径", "评测集定义"),
    ("fit 口径", "fit 判定标准"),
    ("lr 口径", "lr 规则"),
    ("参数口径名", "参数判据名"),
    ("效果口径名", "效果判据名"),
    ("口径变更", "判据变更"),
    ("口径理由", "判据理由"),
    ("口径订正", "判据订正"),
    ("现行口径", "现行判据"),
    ("旧口径", "旧判据"),
    ("完整正确口径", "完整正确判据"),
    ("主笔裁定", "定稿判据"),
    ("★ 裁定", "★ 规则"),
    ("按裁定", "按判据"),
    ("裁定", "规则"),
    ("口径", "判据"),
)

#: ★ 发布件**行键契约**（V16 机核）：必须 ⊆ 本表；对齐 `build_official.HUMAN_VIEW_ROW_KEYS`。
#:   ⛔ 审计字段（backend_reason / steps_basis / id / native_tp,dp / total_cards / runnable /
#:   blocked_on）不在本表 ⇒ 只进 `.local` 审计件。
PUBLISH_ROW_KEYS: frozenset[str] = frozenset({
    "tier", "status", "model", "algo", "mode", "backend", "parallel", "parallel_dims",
    "parallel_scheme", "lora_target", "lr", "data_subset", "steps", "switches", "spec_json",
    "model_path", "train_method", "tuner_type", "cards", "tp_size", "pp_size", "dp_size",
    "nproc_per_node", "nnodes", "cp_size", "sp_size", "gpus", "target_modules",
    "target_modules_mode", "deepspeed", "allow_unverified_pp_rollout", "subset_generation",
    "subset_r3_historical", "expected_optimizer_steps_historical", "steps_formula_v2",
    "min_optimizer_steps", "data_source", "subset_size", "prompts_per_rank", "train_path",
    "planned_steps", "save_steps", "expected_optimizer_steps_per_epoch",
    "expected_optimizer_steps_reachable", "applicable", "max_epochs", "derived_steps",
    "per_epoch_steps", "save_steps_basis", "data_scale", "tp", "pp", "dp", "cards_total",
    "peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note", "runs",
    "inference", "algo_checks", "verdict", "excluded_reason", "availability",
    "not_failed_reason", "executability", "blocked_reason",
    "final_formats", "family", "family_min_cards", "cards_pending", "live_evidence",
    "caliber_change",
})

#: ★ 发布件 **meta 键契约**（V16 机核）：必须 ⊆ 本表；对齐 `build_official.HUMAN_VIEW_META_KEYS`。
#:   ⛔ `enumeration`（计数/排序键/证据索引/已知缺口/原因码说明）整体进 `.local` 审计件。
PUBLISH_META_KEYS: frozenset[str] = frozenset({
    "title", "version", "doc_date", "scope", "venue", "code_version", "authority",
    "status_vocabulary", "boundary_declaration", "criteria_caliber", "criteria_version",
    "results_policy", "distill_teacher", "lora_target_backends", "enumeration",
})

#: ★ 发布件 `meta.enumeration` 的**精确键集合**（V16：多一个键就拒写）。
#:   只许「结论级计数」；排序键/原因码说明/证据索引/已知缺口/后端选择规则 ⛔ 一律在审计件。
ENUMERATION_KEYS: frozenset[str] = frozenset({
    "criteria_version", "card_limit", "budget_gib", "counts", "script", "script_sha256", "audit_ref",
})
ENUMERATION_COUNT_KEYS: frozenset[str] = frozenset({
    "raw_cube", "exceed_cards", "enumerated", "kept", "excluded", "kept_split", "tiers",
})
ENUMERATION_SUBKEYS: dict[str, frozenset[str]] = {
    "excluded": frozenset({"EXCEED_CARDS", "MEMORY_UNFIT", "KNOWN_MEMORY_FAIL", "UNSUPPORTED_MODE", "ALGO_BACKEND"}),
    "kept_split": frozenset({"direct", "missing_channel"}),
    "tiers": frozenset({"feasible", "tight", "over_budget"}),
}

#: ★ 发布件里**不得出现**的内部审计线索形态（V16 机核）。
AUDIT_ONLY_PATTERNS = (
    (re.compile(r"src/graspo/"), "内部源码路径"),
    (re.compile(r"\bEV\d{2}\b"), "内部证据号"),
    (re.compile(r"\.[A-Za-z]{2,4}:\d+"), "file:line 溯源"),
    (re.compile(r"backend_reason|steps_basis|native_tp|total_cards|\brunnable\b"), "内部审计键名"),
)


#: 进页面的字段（档一：⛔ 不许出现 file:line 与内部符号名）。
RENDER_FIELDS: tuple[str, ...] = ("note", "blocked_reason")
#: 档一字段里禁止的形态：源码行号 + 内部模块/函数名 + 内部目录名。
RENDER_FIELD_BANNED = re.compile(
    r"\.py:\d+|_dedupe_dp_replicas|native_lora|native_full_param|_config_mapping"
    r"|lora_linear|pipeline_comm|gen_configs|core/schema|Ripple|flow 层"
)


#: 人读字段名（发布前必须过 `build_official.human_view_notation()`）—— 按键名判定，⛔ 不按位置写死。
HUMAN_TEXT_KEYS = frozenset({"note", "blocked_reason"})


def _human_view(text: str) -> str:
    """人读字段的**唯一**清理器 = `build_official.human_view_notation()`（⛔ 不另写一份剥离逻辑）。

    病史（第四轮复核）：本脚本落盘发布件时**没有**调用该清理器，只走脱敏扫描器的禁词表；
    而禁词表当时不含 `★` / `⛔` ⇒ M093 的备注正文里漏出一个 `⛔`（页面可见）。
    现改为：发布前对 `note` / `blocked_reason` 逐字段复用同一函数。
    """
    import build_official as _bo      # 同目录；纯函数，无副作用
    return _bo.human_view_notation(text)


def _publish_words(text: str) -> str:
    for old, new in PUBLISH_WORD_REWRITES:
        text = text.replace(old, new)
    return text


def _polish_doc(node: Any) -> Any:
    """深拷贝并把**所有字符串值**过一遍发布用词中性化（键名⛔ 不动）。"""
    if isinstance(node, dict):
        return {k: _polish_doc(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_polish_doc(v) for v in node]
    if isinstance(node, str):
        return _publish_words(node)
    return node


def serialize_doc(doc: dict[str, Any]) -> str:
    """发布件落盘字节的唯一产生点（V14 与写盘共用，⛔ 两处各 dumps 一次会漂移）。"""
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def leak_hits_in_text(text: str) -> list[str]:
    """**写盘前 fail-closed**：用 `scan_matrix_leaks.py` 的同一套禁词与规则扫发布件全文。

    ★ §1.4 单一真相源：⛔ 不在这里另抄一份禁词表 —— 直接 import 同目录扫描器的
      `PATTERNS`（其中已同源引用 `build_official.DEV_CHURN_TERMS`／`EVIDENCE_PATH_PATTERN`）
      与可选外部清单 `load_denylist()`。生成端与校验端因此**不会漂移**。
    """
    try:
        import scan_matrix_leaks as _leaks  # 同目录，纯常量与纯函数，⛔ 无副作用
    except Exception as exc:  # 扫描器不可用 ⇒ 失败即拒写（⛔ 不许"扫不了就放行"）
        return [f"扫描器不可用（{exc}）⇒ 无法完成脱敏机核"]
    rules = list(_leaks.PATTERNS) + list(_leaks.load_denylist())
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat, why in rules:
            m = re.search(pat, line)
            if m:
                hits.append(f"第 {i} 行：{why} ⇒ {m.group(0)!r}")
                break
    return hits


def _v16_publish_vs_audit(data_text: str | None) -> tuple[bool, str]:
    r"""V16（阻断修复）：**发布件 / 审计件分离**四条机核 ——
    ① 行键 ⊆ `PUBLISH_ROW_KEYS`；② meta 键 ⊆ `PUBLISH_META_KEYS`；
    ③ 全文不含 `src/graspo/`、`EV\d{2}`、file:line 溯源、审计键名；
    ④ 审计字段确实进了审计件（由调用方另行核对行数）。"""
    if data_text is None:
        return False, "数据件文本未提供"
    try:
        doc = json.loads(data_text)
    except Exception as exc:
        return False, f"数据件不可解析（{exc}）"
    rows = doc.get("rows") or []
    meta = doc.get("meta") or {}
    bad_row = sorted({k for r in rows for k in r} - PUBLISH_ROW_KEYS)
    if bad_row:
        return False, f"① 发布件行键越界：{bad_row}"
    bad_meta = sorted(set(meta) - PUBLISH_META_KEYS)
    if bad_meta:
        return False, f"② 发布件 meta 键越界：{bad_meta}"
    # ④ `meta.enumeration` 只许「结论级计数」：键集合必须**恰好**等于契约（多一个就拒写）
    enum = meta.get("enumeration")
    if enum is not None:
        if not isinstance(enum, dict):
            return False, "④ meta.enumeration 不是对象"
        if sorted(set(enum)) != sorted(ENUMERATION_KEYS):
            return False, (f"④ meta.enumeration 键集合不符（多：{sorted(set(enum) - ENUMERATION_KEYS)}；"
                           f"缺：{sorted(ENUMERATION_KEYS - set(enum))}）")
        cnt = enum.get("counts") or {}
        if set(cnt) != ENUMERATION_COUNT_KEYS:
            return False, f"④ meta.enumeration.counts 键集合不符：{sorted(set(cnt) ^ ENUMERATION_COUNT_KEYS)}"
        for sub, keys in ENUMERATION_SUBKEYS.items():
            got = set(cnt.get(sub) or {})
            if got != keys:
                return False, f"④ meta.enumeration.counts.{sub} 键集合不符：{sorted(got ^ keys)}"

    leaks: list[str] = []
    for pat, why in AUDIT_ONLY_PATTERNS:
        m = pat.search(data_text)
        if m:
            leaks.append(f"{why} ⇒ {m.group(0)!r}")
    if leaks:
        return False, f"③ 发布件里出现审计线索 {len(leaks)} 类：" + "；".join(leaks[:4])
    return True, "①②③ 全部合规（行键/meta 键均 ⊆ 允许清单，无源码路径·证据号·file:line）"


def _v15_publish_clean(data_text: str | None) -> tuple[bool, str]:
    """V15（三条硬断言）：① 脱敏扫描器词表命中 = 0；② 进页面字段里 file:line/内部符号 = 0；
    ③ 旧档号 `T` + 三位数字 = 0。命中 ⇒ rc=2 且 ⛔ 不写盘。"""
    if data_text is None:
        return False, "数据件文本未提供"
    hits = leak_hits_in_text(data_text)
    if hits:
        return False, f"① 脱敏词表命中 {len(hits)} 处：" + "；".join(hits[:5])
    try:
        doc = json.loads(data_text)
        rows = doc.get("rows") or []
    except Exception as exc:
        return False, f"数据件不可解析（{exc}）"
    bad: list[str] = []
    for r in rows:
        for field in RENDER_FIELDS:
            val = r.get(field)
            if isinstance(val, str):
                m = RENDER_FIELD_BANNED.search(val)
                if m:
                    bad.append(f"{r.get('tier')}.{field} ⇒ {m.group(0)!r}")
    if bad:
        return False, f"② 进页面字段含开发过程细节 {len(bad)} 处：" + "；".join(bad[:5])
    tier_ids = TIER_ID_RE.findall(data_text)
    if tier_ids:
        return False, f"③ 旧档号命中 {len(tier_ids)} 处（{', '.join(sorted(set(tier_ids)))}）"
    return True, "①②③ 全部 0 处"


def _v14_no_tier_ids(data_text: str | None) -> tuple[bool, str]:
    """V14：发布件里旧档号（`T` + 三位数字）命中数必须为 0。"""
    if data_text is None:
        return False, "数据件文本未提供"
    hits = TIER_ID_RE.findall(data_text)
    if hits:
        uniq = sorted(set(hits))
        return False, f"命中 {len(hits)} 处（{', '.join(uniq)}）"
    return True, "0 处"


#: note 里**只有**拿到具名 success 跑次才允许出现的断言词（V13 机核）。
CLAIM_WORDS: tuple[str, ...] = ("训练可跑", "训练能跑", "训练跑通")
#: note 里**只有**拿到实测证据才允许出现的断言词（V13 反方向机核）；纯算式的只能写「估算超预算」。
NEGATIVE_CLAIM_WORDS: tuple[str, ...] = ("不可行", "必然失败", "交不出")


def _any_backend_evidence(algo, model, mode, shape, effects) -> bool:
    """不看后端，只看 (算法,模型,模式,形状) 是否有该类证据（供"某后端出局"类断言兜底）。"""
    for ev in MEASURED_EVIDENCE:
        a, m, md, _be, sh = ev["scope"]
        if (a in ("*", algo) and m in ("*", model) and md in ("*", mode)
                and _shape_match(sh, shape) and ev["effect"] in effects):
            return True
    return False


def _v13_claims_backed_by_runs(rows, budget: float) -> bool:
    """裁定后补（必修 2）：note 的成败断言必须有实测跑次兜底 —— 正反两个方向都机核。

    正向：「训练可跑/能跑/跑通」⇒ 该档（算法,模型,模式,后端,形状）必须有 `success` 类证据。
    反向：「不可行/必然失败/交不出」⇒ 该档（不限后端）必须有 `backend_infeasible`/`recipe_forbid` 证据，
          纯算式结论只能写「估算超预算」。
    """
    for r in rows:
        note = r["note"]
        shape = (int(r["tp"]), int(r["pp"]), int(r["dp"]))
        if any(w in note for w in CLAIM_WORDS):
            if not evidence_for(r["algo"], r["model"], r["mode"], r["backend"], shape,
                                effects=("success",)):
                return False
        if any(w in note for w in NEGATIVE_CLAIM_WORDS):
            if not _any_backend_evidence(r["algo"], r["model"], r["mode"], shape,
                                         ("backend_infeasible", "recipe_forbid")):
                return False
    return True


VERIFICATION_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "V1",
        "claim": "27B LoRA 的 dp1/dp2/dp4（tp=pp=1）必须判「不可行」或标「余量紧」",
        "check": lambda rows, budget: all(
            (_sel_by(rows, algo, "27B", "LoRA", (1, 1, dp)) is None)
            or (_sel_by(rows, algo, "27B", "LoRA", (1, 1, dp))["backend_reason"]["band"] in ("余量紧", "超预算"))
            for algo in ALGOS for dp in (1, 2, 4)),
    },
    {
        "id": "V2",
        "claim": "9B LoRA dp2 判「可行」（估算 ≈33.2 GiB ≤ 0.90×78.0）",
        "check": lambda rows, budget: all(
            (r := _sel_by(rows, algo, "9B", "LoRA", (1, 1, 2))) is not None
            and r["backend_reason"]["band"] == "可行"
            for algo in ALGOS),
    },
    {
        "id": "V3",
        "min_card_limit": 4,
        "claim": "27B 全参 native PP4 按生成器算式（k=8/pp）应「可行」且被选为 native",
        "check": lambda rows, budget: all(
            (r := _sel_by(rows, algo, "27B", "全参", (1, 4, 1))) is not None
            and r["backend"] == "native" and r["backend_reason"]["band"] == "可行"
            for algo in ("SFT", "GRASPO")),
    },
    {
        "id": "V4",
        "claim": "9B 全参 1 卡必须靠 offload：无 offload 估算 79.24 > 78.0，offload 后 45.71 ≤ 78.0",
        "check": lambda rows, budget: (
            _est("SFT", "9B", "全参", "native", (1, 1, 1), budget, recipe="no_offload") > budget
            and _est("SFT", "9B", "全参", "native", (1, 1, 1), budget, recipe="offload") <= budget
            and (r := _sel_by(rows, "SFT", "9B", "全参", (1, 1, 1))) is not None
            and r["backend_reason"]["recipe"] == "offload"),
    },
    {
        "id": "V5",
        "claim": "OPD 1 卡教师常驻必须判不可行（估算 83.34 > 78.0，实测 78.06 显存不足）；仅 teacher offload 情形可留待测试",
        "check": lambda rows, budget: (
            _est("OPD", "9B", "LoRA", "ms-swift", (1, 1, 1), budget, teacher_variant="gpu") > budget
            and _est("OPD", "9B", "LoRA", "ms-swift", (1, 1, 1), budget, teacher_variant="offload") <= budget
            and (r := _sel_by(rows, "OPD", "9B", "LoRA", (1, 1, 1))) is not None
            and r["backend_reason"]["teacher_variant"] == "offload"),
    },
    {
        "id": "V6",
        "claim": ("CPT 无 native 实现 ⇒ 必须保留为待测试（不许剔除）：能装下的档后端 = ms-swift；"
                  "只有 native 能装下的档以 MISSING_CHANNEL 保留（backend=native 但 backend_implemented=false）"),
        "check": lambda rows, budget: (
            sum(1 for r in rows if r["algo"] == "CPT") > 0
            and all("尚无 CPT 训练通道" in r["note"] for r in rows if r["algo"] == "CPT")
            and all((r["backend"] == "ms-swift")
                    or (r["backend_reason"].get("backend_implemented") is False)
                    for r in rows if r["algo"] == "CPT")),
    },
    {
        "id": "V7",
        "claim": "全部保留档位 status = 待测试，且结果列全为「—」",
        "check": lambda rows, budget: all(
            r["status"] == "待测试"
            and all(r[k] == "—" for k in ("peak_mem", "steps_actual", "elapsed", "throughput", "effect"))
            for r in rows),
    },
    {
        "id": "V8",
        "claim": "任何档位都不以 DeepSpeed 为选型/配方（DS 已禁用）；Megatron 只以「未接线卡点」出现",
        "check": lambda rows, budget: all(
            r["deepspeed"] == "—" and "deepspeed" not in r["switches"].lower() for r in rows)
        and all("Megatron 段未接线" in json.dumps(r["backend_reason"], ensure_ascii=False)
                or "Megatron 未接线" in json.dumps(r["backend_reason"], ensure_ascii=False)
                or r["backend"] == "native"
                for r in rows),
    },
    {
        "id": "V9",
        "claim": "保留档位的判定只允许「可行 / 余量紧」（超预算的一律在剔除清单里）",
        "check": lambda rows, budget: all(r["backend_reason"]["band"] in ("可行", "余量紧") for r in rows),
    },
    {
        "id": "V10",
        "claim": "计数守恒：raw_cube = exceed_cards + enumerated；enumerated = kept + Σ剔除；kept = direct + missing_channel",
        "check": lambda rows, budget: True,   # 由 `run_self_check` 外层的守恒断言单独核（见下）
    },
    {
        "id": "V11",
        "claim": ("LoRA ∧ dp>1 ∧ tp=pp=1 ⇒ backend 取 ms-swift（native 的 DP 产物交付带一道未闭合前置："
                  "副本位等价窄口，`native_lora.py:455-476`；ms-swift 走 FSDP2 full_state_dict 产标准 HF checkpoint）；"
                  "唯一例外 = ms-swift 在该形状有**实测不可行**证据（此时选 native 并在 note 写明该实测）"),
        "check": _v11_lora_dp_deliverable,
    },
    {
        "id": "V12",
        "claim": "MISSING_CHANNEL 档必须 status=待测试、note 含「卡点」且写明缺哪条通道与补齐后的估算",
        "check": _v12_missing_channel,
    },
    {
        "id": "V13",
        "claim": ("note 的成败断言必须有实测跑次兜底：「训练可跑/能跑/跑通」需具名 success 证据"
                  "（历史 54 档里 LoRA 恒为 tp=pp=1 ⇒ tp>1/pp>1 的 LoRA 档只能写「从未实跑」）；"
                  "反方向「不可行/必然失败/交不出」需 backend_infeasible/recipe_forbid 证据，纯算式只能写「估算超预算」"),
        "check": _v13_claims_backed_by_runs,
    },
    {
        "id": "V14",
        "claim": ("发布件（docs/capability-matrix.data.json）里旧档号形态 `T` + 三位数字的命中数必须为 0；"
                  "人读文案一律用描述性指代，完整溯源只留在工位件"),
        "check": None,   # 由 `run_self_check` 单独核（需要发布件序列化文本，见下）
    },
    {
        "id": "V16",
        "claim": ("发布件/审计件分离：行键 ⊆ 允许清单、meta 键 ⊆ 允许清单、`meta.enumeration` 键集合"
                  "恰好 = 结论级计数契约（多一键即拒写），且发布件里 `src/graspo/`、`EV` + 两位数字、"
                  "file:line 溯源、审计键名命中数 = 0"),
        "check": None,
    },
    {
        "id": "V15",
        "claim": ("发布件三条硬断言：①脱敏扫描器词表命中 = 0（含「裁定/口径」等过程用语）；"
                  "②进页面字段（note/blocked_reason）里源码行号与内部符号名命中 = 0；"
                  "③旧档号 T+三位数字 = 0"),
        "check": None,
    },
)


def run_self_check(rows, budget: float, counts: dict[str, Any] | None = None,
                   card_limit: int | None = None,
                   data_text: str | None = None) -> list[dict[str, Any]]:
    out = []
    for case in VERIFICATION_CASES:
        need = case.get("min_card_limit")
        if need is not None and card_limit is not None and card_limit < need:
            out.append({"id": case["id"], "claim": case["claim"], "ok": True, "skipped": True,
                        "reason": f"min_card_limit={need} 而 card_limit={card_limit} ⇒ 该形状不在枚举域，本回验不适用"})
            continue
        if case["id"] == "V16":
            ok, detail = _v16_publish_vs_audit(data_text)
            out.append({"id": "V16", "claim": case["claim"], "ok": ok, "reading": detail})
            continue
        if case["id"] == "V15":
            ok, detail = _v15_publish_clean(data_text)
            out.append({"id": "V15", "claim": case["claim"], "ok": ok, "reading": detail})
            continue
        if case["id"] == "V14":
            ok, detail = _v14_no_tier_ids(data_text)
            out.append({"id": "V14", "claim": case["claim"], "ok": ok, "reading": detail})
            continue
        if case["id"] == "V10":
            if counts is None:
                out.append({"id": "V10", "claim": case["claim"], "ok": False,
                            "error": "counts 未提供"})
                continue
            # EXCEED_CARDS 发生在枚举之前 ⇒ 不进"枚举候选的剔除"合计（否则重复计数）
            excl_total = sum(v for k, v in counts["excluded"].items()
                             if k != "EXCEED_CARDS" and REASON_CODES.get(k, {}).get("excludes"))
            kb = counts.get("kept_breakdown", {})
            mc_n = sum(1 for r in rows if r["backend_reason"].get("backend_implemented") is False)
            ok = (counts["raw_cube"] == counts["exceed_cards"] + counts["enumerated"]
                  and counts["excluded"].get("EXCEED_CARDS") == counts["exceed_cards"]
                  and counts["enumerated"] == counts["kept"] + excl_total
                  and counts["kept"] == len(rows)
                  and kb.get("direct", 0) + kb.get("missing_channel", 0) == counts["kept"]
                  and counts["excluded"].get("MISSING_CHANNEL", 0) == mc_n == kb.get("missing_channel", 0))
            out.append({"id": "V10", "claim": case["claim"], "ok": bool(ok),
                        "detail": {"raw": counts["raw_cube"], "exceed": counts["exceed_cards"],
                                   "enumerated": counts["enumerated"], "kept": counts["kept"],
                                   "kept_breakdown": counts.get("kept_breakdown"),
                                   "missing_channel": mc_n, "excluded_total": excl_total}})
            continue
        try:
            ok = bool(case["check"](rows, budget))
        except Exception as exc:                      # noqa: BLE001 — 自检自身出错必须显式失败
            out.append({"id": case["id"], "claim": case["claim"], "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append({"id": case["id"], "claim": case["claim"], "ok": ok})
    return out


# ════════════════════════════════════════════════════════════════════════════
# §8 · meta 同步（三态词表 + enumeration 块 + 旧 54 档遗留措辞清理）
# ════════════════════════════════════════════════════════════════════════════

#: 旧 54 档遗留的**状态词**（⛔ 不得留在发布件 meta 里）。
LEGACY_STATUS_WORDS = ("尚未验证", "未通过", "已验证可用", "已验证不可用", "未纳入验收", "四态")

#: 旧 54 档档号（`T001`…`T054`）：⛔ 不得留在发布件 meta 里（历史留痕另存 `.local`）。
LEGACY_TIER_RE = re.compile(r"T\d{3}")

#: meta 字段级**显式覆写**（路径 → 新值）。⛔ 逐条写，不靠正则猜：
#:   - 状态词表从四态收敛为三态（主笔要求：待测试 / 通过 / 不通过）；
#:   - 承诺面措辞必须与「全部档位 = 待测试」一致（否则发布件自相矛盾）；
#:   - 旧 54 档的档号清单与逐档读数史**一律移除**（归档见 `archive-54/`）。
META_OVERRIDES: tuple[dict[str, Any], ...] = (
    {"path": "doc_date", "value": "2026-10-22",
     "why": "档位表整体重生成，发布日期随生成日"},
    # ★ 2026-10-22（复核 W7）：`code_version.commit` 落后 HEAD 61 个提交 ⇒ 由**生成时的
    #   `git rev-parse --short HEAD`**（可用 `--code-commit` 显式覆盖）写入，⛔ 不留手写常量。
    {"path": "code_version.commit", "value": "COMMIT",
     "why": "本件是就本 commit 的代码生成的 ⇒ 记录生成基线，避免读成旧代码的产物"},
    {"path": "status_vocabulary", "value": ["待测试", "通过", "不通过"],
     "why": "主笔要求：三态词表（旧四态作废）"},
    {"path": "boundary_declaration.text",
     "value": ("能力与可用档位矩阵说明各档位的可用性与性能表现：本版是**待测试的设计矩阵** —— "
               "全部档位状态 = 「待测试」，结果列一律「—」；跑通后由跑次记录回填实测读数与「通过 / 不通过」判定。"
               "训练效果取决于使用者自身的任务、数据集与参数，不在本矩阵的承诺范围。"),
     "why": "旧文本宣称「每个列出的档位都已完整跑通」，与全部档待测试相矛盾"},
    {"path": "boundary_declaration.points[0]",
     "value": "① 各档可用性：状态为「通过」的档位才计入承诺面；「待测试」档只表示设计上成立、尚未取得实测结论。",
     "why": "三态词表同步"},
    {"path": "boundary_declaration.points[1]",
     "value": ("①′ 待测试档不计入承诺面：状态为「待测试」的档位既不充当可用证据、也不充当不可用结论 —— "
               "它们尚无实测读数，不改变任何档位的结论与状态计数。"),
     "why": "旧文本用四态与「本环境不可执行」措辞，与本版词表不一致"},
    {"path": "results_policy",
     "value": "结果列由跑次记录回填；尚无实测结论的档位状态为「待测试」，结果列一律「—」；不得手工填写。",
     "why": "三态词表同步"},
    {"path": "criteria_version",
     "value": "v4-2026-10-17（判据② 定死；四条判据 + 算法正确性检查全过 ⇒ 才可标「通过」）",
     "why": "三态词表同步"},
    {"path": "criteria_caliber.conclusion_rule.rule",
     "value": ("结论必须与同一行的参数一一对应。若一行的参数已被更正，而结论来自旧参数下的跑次 ⇒ 该结论作废、"
               "该档回退为「待测试」；仅在参数冻定后重跑，才可给出「通过 / 不通过」判定。"),
     "why": "三态词表同步（旧文本写「尚未验证 / 已验证可用 / 已验证不可用」）"},
    {"path": "criteria_caliber.conclusion_rule.pass_rule",
     "value": ("四条判据（① 能完整跑完、② 效果变好、③ 算法正确性检查、④ 能推理）全部通过 ⇒ 才能标「通过」；"
               "任一条不过 ⇒ 标「不通过」；尚无实测结论 ⇒ 标「待测试」。"),
     "why": "三态词表同步"},
    {"path": "criteria_caliber.conclusion_rule.status_wording",
     "value": ("本文档的状态词表固定为三态：「待测试 / 通过 / 不通过」；「待测试」等同「未测试」，"
               "不得读作能力否定（也不得读作可用证据）。"),
     "why": "三态词表同步"},
    {"path": "criteria_caliber.dataset.id",
     "value": ("平台方自有评测集（名称平台方自有数据集（不公开））＋ 100 行档（判据 = 该档 subset_size == 100）"
               "的评测集 = 该档自身训练用的同一份文件（fit 口径：训练集自评）"),
     "why": "移除旧 54 档的 45 档档号清单"},
    {"path": "criteria_caliber.dataset.train_eval_relation",
     "value": ("评测集口径：100 行档 = fit 口径（训练集自评）—— 评测集 = 该档自身训练用的同一份文件"
               "（100 行、dataset_split: train）⇒ 读数回答的是「该档训练是否生效」；不得读作训练分布之外的表现"
               "（本矩阵不评测该问题）。档位清单一律从行数据现算（判据 = subset_size == 100）。"
               "判别力（100 行档）：该 100 行的 (工具, 动作) 类别最大类占 19%（19/100）⇒ 常数预测器上界 19%；"
               "而通过线 = after − base ≥ +10.0 pp（点估计）⇒ 跨过它必须真的读图，任何常数输出都拿不到。"
               "旧 54 档矩阵的 702 行口径读数与「重叠剔除」次级读数已随旧矩阵归档（不得引用）：mini 档实测 "
               "mini100 × 702 图像交集 = 0 ⇒ 702 行口径对 mini 档自始不成立；100 行档下评测集 ≡ 训练集 "
               "⇒ 重叠剔除类次级读数按定义退化、不构成次级读数。"),
     "why": "移除旧档号清单与四态措辞；保留口径与判别力事实"},
    {"path": "criteria_caliber.eval_caliber.rule",
     "value": ("100 行档（判据 = 该档 subset_size == 100）：评测集 = 该档自身训练用的同一份文件"
               "（100 行、dataset_split: train）⇒ fit 口径（训练集自评）；其余档：平台方自有 702 行评测集 "
               "+ dataset_split: test。"),
     "why": "移除旧 54 档的 45 档档号清单"},
    {"path": "criteria_caliber.eval_caliber.not_applicable",
     "value": ("本矩阵所有档位状态一律为「待测试」，不存在「不适用」档。旧 54 档矩阵的 20 行档 / 不可执行档"
               "清单已作为历史留痕随旧矩阵归档，不随本发布件发布（口径理由保留：n=20 时每样本 5.00 pp，"
               "而 GRASPO 通过线正好 +5.0 pp ⇒ 判据会退化为「1 个样本决定成败」；放大到 100 行后每样本粒度"
               "回到 1.00 pp、判别力恢复）。"),
     "why": "移除旧档号清单与四态「未纳入验收」措辞"},
    {"path": "criteria_caliber.eval_caliber.untouched",
     "value": ("旧矩阵的 9 个 formal 档（702 行评测集口径：训练源 = 全量训练集前 N，与 702 评测集同分布）"
               "已随旧矩阵归档；本矩阵不逐一声明该口径的适用性。"),
     "why": "移除旧档号清单"},
    {"path": "criteria_caliber.eval_caliber.consequence",
     "value": ("后果判据：旧读数取自 702 行评测集、与本行现行评测集口径不再一一对应 ⇒ 结论作废、该档回退为"
               "「待测试」（旧读数保留在历史归档件，不抹）。与 lr 口径的叠加：两条原因并列写清。"),
     "why": "三态词表同步 + 移除旧档号"},
    {"path": "criteria_caliber.lr_policy.consequence",
     "value": ("口径变更后读数与配置的一一对应性须逐档复核：凡读数来自旧口径跑次（旧 lr 或旧 epoch）而本行"
               "参数已更正者 ⇒ 结论作废、该档回退为「待测试」（旧读数保留在历史归档件，不抹）。"),
     "why": "三态词表同步"},
    {"path": "criteria_caliber.caliber_change.changed",
     "value": ("改动一 = 判据②（效果）：由「训练类与 GRASPO 统一 δ = +5.0 pp」改为逐算法分列；SFT epoch 由 2 "
               "改为 4；OPD 教师改为 SFT 之后的 27B 产物；明确「不做假设检验」。改动二（2026-10-16）= 评测集口径："
               "100 行档的评测集由平台方自有 702 行评测集改为该档自身训练用的同一份文件（100 行、"
               "dataset_split: train）⇒ 读数的含义是 fit 口径（训练集自评），不得读作训练分布之外的表现；"
               "该改动不改通过线数值，旧读数按「结论与口径一一对应」纪律作废（历史留痕已归档，"
               "不随发布件发布）。改动三（2026-10-16）= 20 行档扩面到 100 行（口径变更）：旧矩阵里 "
               "GRASPO×ms-swift 与 OPD 的 20 行档放大到 100 行、适用同一 fit 口径；本矩阵的档位与子集"
               "一律从行数据现算。"),
     "why": "移除旧 54 档的档号清单与逐档读数史引用"},
    {"path": "meta.lora_target_backends.native.signature_evidence.runs",
     "value": "旧矩阵的 1 卡与 2 卡 LoRA 档签名完全相同",
     "why": "移除旧档号"},
    {"path": "meta.lora_target_backends.native.signature_evidence.note",
     "value": ("该值即字段 parameter_count 的现行值（08a5654 融合 QKV 后按形状表："
               "409,923,584 − 24 × 8192 × 128 = 384,757,760）；旧矩阵 1 卡/2 卡档的两份融合前 checkpoint 的 "
               "manifest 原值为 409923584（与旧形状表逐位相等，可作交叉验证）—— 旧 adapter 与现行签名不再同构，"
               "不得混用。语义按字段名判（parameter_count 是 LoRA 可训练参数量，本脚本用它算可训练态显存）。"),
     "why": "移除旧档号（数值与结论逐字保留）"},
    {"path": "meta.lora_target_backends.native.note",
     "value": ("native 侧支持 preset 名（glob），由本仓 lora 模块解析。实测证明 language_all_linear 确实命中"
               "语言层 10 项、不是空转（与旧矩阵 1 卡 SFT 档实测 Δ=+13.39 pp 互证：语言层空转不可能有该增益）。"),
     "why": "移除旧档号（数值与结论逐字保留）"},
    {"path": "criteria_caliber.epoch_policy.known_issue",
     "value": ("登记待修（未修）：推导步数公式对 native GRASPO 不成立，根因已读清：两个后端的 step 语义本就不同。"
               "① ms-swift：步数 = epochs × (subset × num_generations) / (micro_batch × grad_accum)；"
               "② native：步数 = replay 缓冲区触发次数（optimize.py::_maybe_optimize）；阈值 = "
               "rollout_queue_batch_size × rollout_group_size（core/schema.py:458 / ripple/group_decision.py:152），"
               "每次仅消耗整组（whole_group_count，组不拆），epoch 尾可有 force 刷尾；与 micro_batch/grad_accum 解耦。"
               "③ 可训练补全数是运行期结果（取决于 retry/perfect_skip）⇒ 无法从配置先验算出；公式仍不改，"
               "维持「native 以实跑 run_cumulative.step 为准」。"),
     "why": "移除旧档号锚（机理与未修状态逐字保留）"},
    {"path": "meta.lora_target_backends.msswift.evidence[7]",
     "value": "旧矩阵 1 卡 SFT 档实测 Δ=+13.39 pp（与 native 签名互证）",
     "why": "移除旧档号"},
)

#: 直接**移除**的 meta 键（旧 54 档逐档读数史 ⇒ 归档，不随发布件发布）。
META_DROPS: tuple[dict[str, str], ...] = (
    {"path": "criteria_caliber.caliber_change.reading_change_registry",
     "why": "逐档「四态翻盘」读数史：全部以旧档号锚定、旧读数已作废 ⇒ 归档"},
    {"path": "criteria_caliber.caliber_change.reading_change_registry_note",
     "why": "同上（该键只解释 registry 怎么读）"},
    {"path": "criteria_caliber.step_reconciliation",
     "why": "旧 54 档逐档步数对账史（全部以旧档号为锚，含已作废读数）⇒ 归档，不随发布件发布"},
)


def _get_path(doc: dict[str, Any], path: str) -> Any:
    """按 `a.b[1].c` 取值。"""
    cur: Any = doc
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            cur = cur[int(part[1:-1])]
        else:
            cur = cur[part]
    return cur


def _set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur: Any = doc
    for part in parts[:-1]:
        cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
    last = parts[-1]
    if last.startswith("["):
        cur[int(last[1:-1])] = value
    else:
        cur[last] = value


def _del_path(doc: dict[str, Any], path: str) -> bool:
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur: Any = doc
    for part in parts[:-1]:
        if part.startswith("["):
            cur = cur[int(part[1:-1])]
        else:
            if part not in cur:
                return False
            cur = cur[part]
    last = parts[-1]
    if last.startswith("["):
        idx = int(last[1:-1])
        if idx >= len(cur):
            return False
        cur.pop(idx)
        return True
    if last not in cur:
        return False
    cur.pop(last)
    return True


def _git_short_head() -> str:
    """当前 HEAD 的短 sha（⭐ W7：`meta.code_version.commit` 由它现算，⛔ 不再手写常量）。

    拿不到（无 git / 非仓）时 fail-closed：记录一个错误的基线 commit 比留空更有害。
    """
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"FATAL: 取不到 HEAD commit（{exc}）⇒ 请用 --code-commit 显式指定")
    sha = (p.stdout or "").strip()
    if p.returncode != 0 or not sha:
        raise SystemExit(f"FATAL: `git rev-parse --short HEAD` 失败（rc={p.returncode}）"
                         f"⇒ 请用 --code-commit 显式指定")
    return sha


def apply_meta_sync(meta: dict[str, Any], *, card_limit: int, budget: float,
                    counts: dict[str, Any], script_sha: str, generated_at: str,
                    code_commit: str = ""
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """同步 meta：三态词表 + 旧 54 档遗留措辞清理。

    ★ 返回 `(发布 meta, 枚举审计块)`：**发布 meta 只含 `HUMAN_VIEW_META_KEYS` 允许的键**
      （⛔ 不许把内部审计明细塞进发布件）；枚举/证据/缺口等明细全部进审计件（`.local`）。
    """
    new = json.loads(json.dumps(meta))  # 深拷贝（⛔ 不改输入）

    # ★ 发布件只放**结论级计数**（第 4 章章首「保留 124 档、剔除 36 档」由数据渲染，⛔ 不手填数字）；
    #   排序键 / 原因码说明 / 证据索引 / 已知缺口 / 后端选择规则等明细⛔ 不进发布件 ⇒ 在审计件。
    _ex = counts.get("excluded", {})
    _kb = counts.get("kept_breakdown", {})
    _bd = counts.get("bands", {})
    new["enumeration"] = {
        "criteria_version": CRITERIA_VERSION,
        "card_limit": card_limit,
        "budget_gib": budget,
        "counts": {
            "raw_cube": counts["raw_cube"],
            "exceed_cards": counts["exceed_cards"],
            "enumerated": counts["enumerated"],
            "kept": counts["kept"],
            "excluded": {k: _ex.get(k, 0) for k in
                         ("EXCEED_CARDS", "MEMORY_UNFIT", "KNOWN_MEMORY_FAIL", "UNSUPPORTED_MODE", "ALGO_BACKEND")},
            "kept_split": {"direct": _kb.get("direct", 0), "missing_channel": _kb.get("missing_channel", 0)},
            "tiers": {"feasible": _bd.get("可行", 0), "tight": _bd.get("余量紧", 0),
                      "over_budget": _bd.get("超预算", 0)},
        },
        "script": "tools/matrix/enumerate_matrix.py",
        "script_sha256": script_sha,
        "audit_ref": "逐档审计明细见枚举脚本 --audit-out 输出（随工作区，不入库）",
    }

    # ★★ 2026-10-22（第四轮复核 W7）：`code_version.commit` 记录**生成基线**（提交时 HEAD）。
    #   自引用限制：数据件记录不了"自己所在的那个提交"（内容含该值 ⇒ sha 变化 ⇒ 鸡生蛋）。
    #   ⇒ 语义写进 `code_version.note`：commit = 生成基线；**脚本 sha256 以
    #   `meta.enumeration.script_sha256` 为准**（它在同一提交里与脚本逐字节对应）。
    _cv = new.get("code_version")
    if isinstance(_cv, dict):
        _cv["note"] = ("commit = 本件的生成基线（生成时的 HEAD）；脚本以 "
                       "meta.enumeration.script_sha256 为准 —— 该 sha256 与同提交内的 "
                       "tools/matrix/enumerate_matrix.py 逐字节一致。")

    for drop in META_DROPS:
        _del_path(new, drop["path"])   # 不存在也不算错（幂等）

    for ov in META_OVERRIDES:
        path = ov["path"][5:] if ov["path"].startswith("meta.") else ov["path"]
        _val = ov["value"]
        if _val == "\x01COMMIT\x01":          # ★ 生成基线 commit：调用方现算后传进来
            _val = code_commit
        try:
            _get_path(new, path)
        except (KeyError, IndexError, TypeError):
            raise SystemExit(f"FATAL: meta 覆写路径不存在：{path}（meta 结构变了 ⇒ 必须同步本脚本）")
        _set_path(new, path, _val)

    # 兜底扫描：⛔ 旧档号 / 旧状态词不得留在发布件 meta（fail-closed）
    leftovers: list[str] = []

    def _scan(node: Any, path: str) -> None:
        # ★ 兜底扫描的**适用面**：只扫"从旧 meta 继承下来的字段"。
        #   本脚本自己写的 `enumeration` 块必须能引用具名跑次（T041/T044…）与证据表标签，
        #   否则无法把"实测 > 算式"写清；这些内容是本轮新生成、已逐条可核，不是旧档遗留。
        if path.startswith("meta.enumeration"):
            return
        if isinstance(node, dict):
            for k, v in node.items():
                _scan(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _scan(v, f"{path}[{i}]")
        elif isinstance(node, str):
            if LEGACY_TIER_RE.search(node):
                leftovers.append(f"{path}: 含旧档号")
            for w in LEGACY_STATUS_WORDS:
                if w in node:
                    leftovers.append(f"{path}: 含旧状态词「{w}」")

    _scan(new, "meta")
    if leftovers:
        joined = "\n  ".join(leftovers[:40])
        raise SystemExit("FATAL: meta 仍含旧 54 档遗留措辞/档号 ⇒ 补 META_OVERRIDES 后重跑：\n  " + joined)

    enumeration = {
        "script": "tools/matrix/enumerate_matrix.py",
        "script_sha256": script_sha,
        "criteria_version": CRITERIA_VERSION,
        "card_limit": card_limit,
        "budget_gib": budget,
        "card_capacity_gib": CARD_CAPACITY_GIB,
        "tight_band": f"(0.90×budget, budget] = ({TIGHT_BAND_RATIO * budget:.2f}, {budget:.1f}] GiB ⇒ 保留并标「余量紧，需实测确认」",
        "domain": {
            "algo": list(ALGOS), "model": list(MODELS), "mode": list(MODES),
            "parallel_shape": f"(tp, pp, dp) ∈ {{1,2,4}}³ 且 tp*pp*dp ≤ card_limit（每维 3 值，全量 27 形状）",
            "backend": "**不是**枚举维度：后端是每档的选型输出（native / ms-swift 二选一）",
            "opd_teacher": "教师不计入卡数（主笔原文「不包括 OPD 的教师」）；教师常驻另计 50.10 GiB/卡",
        },
        "sort_key": SORT_KEY_DOC,
        "backend_selection": {
            "sort_keys": [
                "①实测成功证据（该算法×模型×模式×后端×形状有判据① 通过的跑次）",
                "②该形状是否被后端真消费（native 的 TP/PP 真接线；ms-swift 的 TP/PP 只在未接线的 Megatron 段）",
                "③分片载体是否已实现且已实测（FSDP2 接点齐但零 退出码 0 ⇒ 排后）",
                "④产物可推理（native LoRA：tp>1 无合并实现；dp>1 的 DP 合并依赖「副本位等价」窄口前置）",
                "⑤估算峰值　⑥后端名（确定性尾键）",
            ],
            "exception_rule": (
                "排序键只允许被**唯一一种**例外推翻：被选方（本应排第一的一方）在该形状有**实测不可行证据**时，"
                "可让「有实测成功证据 / 失败更可诊断」的一方上位；⛔ 不许出现第二种未登记的例外。"
                "当前唯一实例：M093（27B GRASPO LoRA dp2）—— native 实测异常退出且异常类型未落盘（未定性）、"
                "ms-swift 实测 显存不足（78.01 GiB）⇒ 保留 native（对方有实测 显存不足 死刑、本方是通信异常可诊断）。"
            ),
            "note_claim_policy": (
                "「训练可跑 / 训练能跑 / 训练跑通」只允许出现在证据表里有具名 `success` 跑次的档上；"
                "无跑次的形状只能写「从未实跑 ⇒ 训练可行性未建立（产物侧另需补导出层）」。"
                "反方向同理：「不可行 / 必然失败 / 交不出」必须有实测证据，纯算式的只能写「估算超预算」。自检 V13 机核。"
            ),
        },
        "counts": counts,
        "reason_codes": {k: {"desc": v["desc"], "excludes": v["excludes"]} for k, v in REASON_CODES.items()},
        "evidence_policy": ("证据优先级 = 实测（跑次记录里的 显存不足 / 峰值）> 算式；算式判「可行」而有实测不可行证据的档，"
                           "以实测为准并在备注写明冲突。实测证据表逐条登记在脚本 `MEASURED_EVIDENCE`（含出处），"
                           "不随发布件发布。"),
        "evidence_index": [
            {"id": e["id"], "label": e["label"], "kind": e["kind"]} for e in MEASURED_EVIDENCE
        ],
        "recipe_not_modeled": [
            "fsdp2_full_state_dict_offload（`samples/configs/fsdp/fsdp2_full_state_dict_offload.json` 存在、"
            "cpu_offload=true 会把参数/梯度/优化器态全放 CPU，但**零 退出码 0 实测**）⇒ ⛔ 未计入显存估算；"
            "受此影响的被剔档在剔除清单里带 `if_recipe_measured` 标注（若该配方实测可行则可能复活）",
        ],
        "known_gaps": [
            "配置生成侧待补：`tools/matrix/gen_configs.py` 全场不出现 `fsdp` ⇒ 本表里 ms-swift 全参档生成的 yaml "
            "只会下发 `deepspeed: —`，**FSDP2 配方缺**（`switches` 里的 fsdp2 只是提示串）。（生成器侧待补，不在本表改造范围内）",
            "MISSING_CHANNEL 档与配置生成：`gen_configs.py:690` 只按 `applicable` 过滤 ⇒ 本表 4 档 "
            "`runnable=false`（backend=native 但缺 CPT/OPD 通道）仍会被它生成 yaml，而那套 `train_method`+`backend` "
            "组合在装载期会被 `core/schema.py:29-34` 拒绝。配置生成侧应改为按 `runnable` 跳过。（生成器侧待补，不在本表改造范围内）",
            "交付侧待补：native 的导出层只认 dp=tp=pp=1（`export/exporters/native_full_param.py:462-471` 逐字 "
            "DP/TP/PP 合并均未实现、MUST stay fail-closed）⇒ native 的 LoRA/全参 TP·PP·DP 档「训练能跑、产物交不出」，"
            "须补 flow 层 export 子系统。",
        ],
        "status": "全部档位 status = 「待测试」，结果列全为「—」",
        "generated_at": generated_at,
        "rule": "★ 本脚本是档位表的唯一生成入口，⛔ 不许手改 data.json 的 rows[]",
    }
    # ★ 渲染型 meta 子树（lora_target_backends / criteria_caliber）里剥掉审计线索：
    #   它们是**人读视图**的一部分 ⇒ ⛔ 不携带 `src/graspo/**`、`*.py:NNN`、`EV##`。
    for sub in ("lora_target_backends", "criteria_caliber"):
        if sub in new:
            new[sub] = _strip_audit_refs(new[sub])

    return new, enumeration


#: 渲染型 meta 子树里的审计线索 → 面向使用者的落点（顺序敏感）。
_AUDIT_REF_REWRITES: tuple[tuple[str, str], ...] = (
    (r"core/schema\.py:\d+(?:-\d+)?\s*/\s*ripple/group_decision\.py:\d+",
     "rollout 队列表与 rollout 组规模两个开关"),
    (r"src/graspo/core/lora\.py", "核心 LoRA 配置"),
    (r"swift/arguments/tuner_args\.py:\d+(?:-\d+)?", "ms-swift 侧"),
    (r"swift/tuners_utils\.py:\d+", "ms-swift 侧"),
    (r"[A-Za-z0-9_./-]+\.py:\d+(?:-\d+)?", ""),
    (r"src/graspo/[A-Za-z0-9_./-]+", ""),
    (r"\bEV\d{2}\b", "实测台账"),
    # 收尾：替换后残留的空格与指代不顺
    ("核心 LoRA 配置 的", "核心 LoRA 配置里的"),
    ("ms-swift 侧 的", "ms-swift 侧的"),
    ("（ms-swift 官方字段 ms-swift 侧；:201-202 覆盖 target_modules）",
     "（ms-swift 官方字段，覆盖其 target_modules）"),
)


def _strip_audit_refs(node: Any) -> Any:
    """深拷贝后把字符串值里的源码路径 / file:line / 证据号换成面向使用者的落点。"""
    if isinstance(node, dict):
        return {k: _strip_audit_refs(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_audit_refs(v) for v in node]
    if isinstance(node, str):
        text = node
        for pat, repl in _AUDIT_REF_REWRITES:
            text = re.sub(pat, repl, text)
        return text
    return node


# ════════════════════════════════════════════════════════════════════════════
# §9 · 主流程
# ════════════════════════════════════════════════════════════════════════════


def build(
    *, card_limit: int, budget: float, generated_at: str, code_commit: str = "", script_sha: str, base_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """核心：枚举 → 剔除 → 选型 → **发布件** + **审计件** + 剔除清单。纯函数（无 IO、无时钟）。

    ★ 发布件（`docs/capability-matrix.data.json`）：只含渲染字段 + 生成器消费字段 + 人读 `note`，
      行键 ⊆ `PUBLISH_ROW_KEYS`、meta 键 ⊆ `PUBLISH_META_KEYS`（V16 机核）。
    ★ 审计件（`.local/…/enumerate_audit.json`，⛔ 不入库）：`backend_reason`（候选排序/落选原文/
      证据号/源码路径）、`steps_basis`、枚举块、证据表、已知缺口。
    """
    enumerated, exceeded = build_candidates(card_limit)

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "raw_cube": len(enumerated) + len(exceeded), "exceed_cards": len(exceeded),
        "enumerated": len(enumerated), "kept": 0,
        "kept_breakdown": {"direct": 0, "missing_channel": 0},
        "excluded": {"EXCEED_CARDS": len(exceeded)},
    }

    for rec in enumerated:
        algo, model, mode, shape = rec["algo"], rec["model"], rec["mode"], rec["shape"]
        tp, pp, dp = shape
        sel = select_backend(algo=algo, model=model, mode=mode, shape=shape, budget=budget)
        chosen = sel["chosen"]
        cands = sel["candidates"]

        if chosen is None:
            # ★ 主笔裁定「缺实现 ≠ 不合理」的落地：该算法在某后端**缺实现**因而没进候选，
            #   但若补齐实现本档就能装下 ⇒ 该档**不是不合理**，而是我们自己的欠账：
            #   保留为待测试（新原因码 MISSING_CHANNEL），按补齐通道后的配方登记估算。
            hyp_all = [eval_backend(algo=algo, model=model, mode=mode, backend=be,
                                    shape=shape, budget=budget)
                       for be in BACKENDS if be not in ALGO_BACKEND_SUPPORT[algo]]
            fittable = [h for h in hyp_all if h["eligible"]]
            if fittable:
                pick = min(fittable, key=lambda h: (round(h["peak_gib"], 3), BACKENDS.index(h["backend"])))
                sel = {
                    "candidates": cands + fittable,
                    "chosen": pick,
                    "missing_backends": [ALGO_BACKEND_NOTE[(algo, be)] for be in BACKENDS
                                         if be not in ALGO_BACKEND_SUPPORT[algo] and (algo, be) in ALGO_BACKEND_NOTE],
                    "missing_channel": True,
                }
                counts["kept_breakdown"]["missing_channel"] += 1
                rows.append(row_of(f"M{len(rows) + 1:03d}", rec, sel, budget))
                continue

            # 真正装不下 ⇒ 分原因码：有实测不可行证据 = KNOWN_MEMORY_FAIL；否则 = MEMORY_UNFIT
            known = [c for c in cands if "evidence" in c]
            code = "KNOWN_MEMORY_FAIL" if known else "MEMORY_UNFIT"
            counts["excluded"][code] = counts["excluded"].get(code, 0) + 1
            revive = [{"backend": h["backend"], "eligible": h["eligible"],
                       "peak_gib": h.get("peak_gib"), "band": h.get("band"),
                       "gap": ALGO_BACKEND_NOTE.get((algo, h["backend"]))} for h in hyp_all]
            detail = "；".join(f"{c['backend']}: {c['why']}" for c in cands)
            for rv in revive:
                if rv["eligible"]:
                    detail += (f"；★ 复活路径：{rv['backend']} 补齐 {algo} 通道后估算 "
                               f"{rv['peak_gib']:.2f} GiB（{rv['band']}）⇒ 本档届时可复活")
            if_recipe = None
            if mode == "全参" and any(c["backend"] == "ms-swift" and not c["eligible"] for c in cands):
                if_recipe = ("若 `samples/configs/fsdp/fsdp2_full_state_dict_offload.json` 的 cpu_offload 配方"
                             "实测可行则本档可能复活（该配方把参数/梯度/优化器态都放 CPU ⇒ 每卡常住接近激活项；"
                             "⛔ 零 退出码 0 实测 ⇒ 未计入显存估算）")
            excluded.append({
                "key": {"algo": algo, "model": model, "mode": mode,
                        "shape": {"tp": tp, "pp": pp, "dp": dp}, "total_cards": tp * pp * dp},
                "reason_code": code,
                "if_recipe_measured": if_recipe,
                "detail": detail[:1200],
                "revive_if_implemented": revive,
                "candidates": [
                    {"backend": c["backend"], "eligible": c["eligible"], "why": c.get("why"),
                     "peak_gib": c.get("peak_gib"), "evidence": c.get("evidence")}
                    for c in cands
                ],
            })
            continue

        counts["kept_breakdown"]["direct"] += 1
        rows.append(row_of(f"M{len(rows) + 1:03d}", rec, sel, budget))

    counts["kept"] = len(rows)
    for code in REASON_CODES:
        if REASON_CODES[code]["excludes"]:
            counts["excluded"].setdefault(code, 0)
    # 非剔除原因码：计的是"保留档位数"，⛔ 不与剔除混算
    counts["excluded"].setdefault("ALGO_BACKEND", 0)
    counts["excluded"].setdefault("MISSING_CHANNEL", counts["kept_breakdown"]["missing_channel"])
    counts["exceed_cards"] = len(exceeded)
    _bands: dict[str, int] = {}
    for _r in rows:
        _b = ((_r.get("_audit") or {}).get("backend_reason") or {}).get("band")
        if _b:
            _bands[_b] = _bands.get(_b, 0) + 1
    counts["bands"] = _bands

    pub_meta, enumeration = apply_meta_sync(
        base_doc.get("meta", {}), card_limit=card_limit, budget=budget,
        counts=counts, script_sha=script_sha, generated_at=generated_at,
        code_commit=code_commit)
    # ★ meta 里的人读字段同样过唯一清理器（如 `lora_target_backends.*.note`）
    def _clean_meta_human(node):
        if isinstance(node, dict):
            return {k: (_human_view(v) if k in HUMAN_TEXT_KEYS and isinstance(v, str)
                        else _clean_meta_human(v))
                    for k, v in node.items()}
        if isinstance(node, list):
            return [_clean_meta_human(v) for v in node]
        return node

    pub_meta = _clean_meta_human(pub_meta)

    # ★ 发布/审计分离：`_audit` 摘出行外；自检与 report 用**合并后的完整行**（含 backend_reason 等）
    checks_rows = [dict(r, **(r.get("_audit") or {})) for r in rows]
    audit_rows = [(r["tier"], r.pop("_audit")) for r in rows]

    # ★★ 2026-10-22（第四轮复核）：**人读字段**过唯一清理器（★/⛔ 等内部记号一律不发布）。
    #   覆盖 `rows[*].note` / `rows[*].blocked_reason` 与 `meta` 里同名的说明字段
    #   （如 `meta.lora_target_backends.msswift.note`）——递归按**键名**判定，⛔ 不按位置写死。
    def _clean_human(node):
        if isinstance(node, dict):
            return {k: (_human_view(v) if k in HUMAN_TEXT_KEYS and isinstance(v, str)
                        else _clean_human(v))
                    for k, v in node.items()}
        if isinstance(node, list):
            return [_clean_human(v) for v in node]
        return node

    rows = _clean_human(rows)

    doc = {
        "schema": base_doc.get("schema", "capability-matrix/v3-official"),
        "meta": pub_meta,
        "rows": rows,
        "data_keys": base_doc.get("data_keys", ["peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note"]),
    }

    # ★ 发布件专用：过程用语中性化（⛔ 只动字符串值；审计件与 report 保留原始溯源）
    doc = _polish_doc(doc)

    audit = {
        "schema": "capability-matrix/audit-v1",
        "note": ("内部审计件（⛔ 不随发布件提供）：发布件里被摘出的逐档审计线索 —— 后端候选与落选原文、"
                 "估算分项、证据号与源码路径、生成依据；另有枚举块（计数/排序键/原因码/证据索引/已知缺口）。"),
        "generated_at": generated_at,
        "script_sha256": script_sha,
        "criteria_version": CRITERIA_VERSION,
        "card_limit": card_limit,
        "budget_gib": budget,
        "counts": counts,
        "enumeration": enumeration,
        "evidence_table": [
            {"id": e["id"], "label": e["label"], "kind": e["kind"], "effect": e["effect"],
             "scope": {"algo": e["scope"][0], "model": e["scope"][1], "mode": e["scope"][2],
                       "backend": e["scope"][3],
                       "shapes": e["scope"][4] if e["scope"][4] == "*" else [list(x) for x in e["scope"][4]]},
             "reading": e["reading"], "source": e["source"],
             "delivery_gap": e.get("delivery_gap")}
            for e in MEASURED_EVIDENCE
        ],
        "rows": [{"tier": tier, **a} for tier, a in audit_rows],
        "excluded": excluded,
    }

    report = {
        "criteria_version": CRITERIA_VERSION,
        "card_limit": card_limit,
        "budget_gib": budget,
        "card_capacity_gib": CARD_CAPACITY_GIB,
        "counts": counts,
        "excluded": excluded,
        "verification_cases": run_self_check(
            checks_rows, budget, counts, card_limit,
            # ★ V14 要的是**发布件序列化文本**（= 落盘那一刻的字节）⇒ 与写盘共用 `serialize_doc`
            data_text=serialize_doc(doc),
        ),
        "rows_diag": [
            {
                "tier": r["tier"], "model": r["model"], "algo": r["algo"], "mode": r["mode"],
                "shape": {"tp": int(r["tp"]), "pp": int(r["pp"]), "dp": int(r["dp"])},
                "total_cards": int(r["cards_total"]), "backend": r["backend"],
                "peak_gib": r["backend_reason"]["peak_gib"],
                "band": r["backend_reason"]["band"],
                "recipe": r["backend_reason"]["recipe"],
                "teacher_variant": r["backend_reason"]["teacher_variant"],
                "components_gib": r["backend_reason"]["components_gib"],
                "evidence": r["backend_reason"]["evidence"],
                "gaps": r["backend_reason"]["gaps"],
            }
            for r in checks_rows
        ],
        "evidence_table": audit["evidence_table"],
        "constants": {
            "CARD_CAPACITY_GIB": {"value": CARD_CAPACITY_GIB, "level": "【实测】显存不足 原文 79.25 GiB"},
            "budget_gib": {"value": budget, "level": "设计选择（tests/e2e/generate_matrix.py:1355 默认 78.0）"},
            "FULL_K_BYTES": {"value": FULL_K_BYTES, "level": "【推断】generate_matrix.py:1347-1352（自述「都是假设」）"},
            "LORA_STATE_BYTES_PER_PARAM": {"value": LORA_STATE_BYTES_PER_PARAM, "level": "【推断】2+2+8"},
            "activation_gib": {
                "9B": {"value": _act_gib("9B"), "level": MODEL_SPEC["9B"]["act_level"]},
                "27B": {"value": round(_act_gib("27B"), 4), "level": MODEL_SPEC["27B"]["act_level"]},
            },
            "lora_trainable": {
                "9B": {"value": _lora_trainable("9B"), "level": MODEL_SPEC["9B"]["lora_trainable_level"]},
                "27B": {"value": _lora_trainable("27B"), "level": MODEL_SPEC["27B"]["lora_trainable_level"]},
            },
            "teacher_weight_gib": {"value": TEACHER_WEIGHT_GIB, "level": "【实测】matrix54_manifest.json:128"},
        },
    }
    return doc, report, audit


def summarize(doc: dict[str, Any], report: dict[str, Any]) -> str:
    """人读摘要（`--dry-run` / 正式运行都打印；限流：≤ 60 行）。"""
    counts = report["counts"]
    rows = doc["rows"]
    lines: list[str] = []
    lines.append("═" * 78)
    lines.append(f"档位枚举汇总（criteria={CRITERIA_VERSION}，card_limit={report['card_limit']}，"
                 f"budget={report['budget_gib']} GiB，卡容量={CARD_CAPACITY_GIB} GiB）")
    lines.append("─" * 78)
    lines.append(f"全量立方（含越界）         : {counts['raw_cube']} 档"
                 f"（4 算法 × 2 模型 × 2 模式 × 27 形状）")
    lines.append(f"  − EXCEED_CARDS           : {counts['exceed_cards']} 档（tp*pp*dp > {report['card_limit']}，枚举阶段不生成）")
    lines.append(f"  = 枚举候选               : {counts['enumerated']} 档")
    for code in REASON_CODES:
        if code == "EXCEED_CARDS":
            continue
        n = counts["excluded"].get(code, 0)
        tag = "剔除" if REASON_CODES[code]["excludes"] else "不剔除（保留待测试并记卡点）"
        lines.append(f"  − {code:<22} : {n} 档（{tag}）")
    kb = counts.get("kept_breakdown", {})
    lines.append(f"  = 最终保留               : {counts['kept']} 档（全部 status=待测试；"
                 f"其中常规 {kb.get('direct', 0)} + MISSING_CHANNEL {kb.get('missing_channel', 0)}）")
    lines.append("─" * 78)
    dist: dict[tuple[str, str, str, str], int] = {}
    for r in rows:
        k = (r["model"], r["algo"], r["mode"], r["backend"])
        dist[k] = dist.get(k, 0) + 1
    lines.append("分布（模型×算法×模式×后端）:")
    for model in MODELS:
        for algo in ALGOS:
            for mode in MODES:
                cells = [f"{be}={dist.get((model, algo, mode, be), 0)}" for be in BACKENDS
                         if (model, algo, mode, be) in dist]
                if cells:
                    lines.append(f"  {model:<3} {algo:<6} {mode:<3} : " + "，".join(cells))
    bands: dict[str, int] = dict(counts.get("bands") or {})
    lines.append(f"三档判定：可行 {bands.get('可行', 0)}，余量紧 {bands.get('余量紧', 0)}，超预算 {bands.get('超预算', 0)}")
    lines.append("─" * 78)
    lines.append("自检（报告里的回验表 = 可执行断言）:")
    for case in report["verification_cases"]:
        lines.append(f"  [{'PASS' if case['ok'] else 'FAIL'}] {case['id']} {case['claim']}")
    lines.append("═" * 78)
    return "\n".join(lines)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="档位矩阵唯一生成入口（枚举 → 剔除 → 选后端 → 写 data.json）")
    ap.add_argument("--card-limit", type=int, default=4, help="档位设计约束「总卡数 ≤ N」（默认 4；非平台上限）")
    ap.add_argument("--budget-gib", type=float, default=78.0, help="单卡显存预算（判定线），默认 78.0")
    ap.add_argument("--out", default="docs/capability-matrix.data.json", help="输出数据件路径")
    ap.add_argument("--report-json", default=DEFAULT_REPORT_JSON,
                    help="剔除清单输出路径（默认写到 .local 工位，⛔ 不污染仓）")
    ap.add_argument("--dry-run", action="store_true", help="不写任何文件，只打印汇总")
    ap.add_argument("--generated-at", default=None,
                    help="固定 meta.enumeration.generated_at（确定性：同输入同输出）")
    ap.add_argument("--overwrite-archive", action="store_true",
                    help="允许写入已存在且非空的归档目录（默认拒绝：归档是退路，⛔ 不覆盖）")
    ap.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR,
                    help="旧矩阵 meta/rows 归档目录（默认 .local 工位）")
    ap.add_argument("--code-commit", default=None,
                    help="写入 meta.code_version.commit 的生成基线（默认取 `git rev-parse --short HEAD`）")
    ap.add_argument("--audit-out", default=DEFAULT_AUDIT_JSON,
                    help="审计件输出路径（内部审计线索；默认 .local 工位，⛔ 不入库）")
    a = ap.parse_args(argv)

    out_path = (REPO_ROOT / a.out).resolve()
    report_path = (REPO_ROOT / a.report_json).resolve()
    audit_path = (REPO_ROOT / a.audit_out).resolve()
    _archive_rel = a.archive_dir or (
        f"{WORK_DIR_REL}/archive-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    archive_dir = (REPO_ROOT / _archive_rel).resolve()
    # ⛔ 防呆：归档目录若已存在且非空 ⇒ 拒绝覆盖（退路必须是**只增不减**的）
    if archive_dir.exists() and any(archive_dir.iterdir()) and not a.overwrite_archive:
        raise SystemExit(
            f"FATAL: 归档目录已存在且非空：{archive_dir}\n"
            "  ⛔ 归档是退路，⛔ 不得覆盖（默认已带时间戳；如确要覆盖请显式加 --overwrite-archive）"
        )
    script_sha = _sha256_of(Path(__file__).resolve())
    generated_at = a.generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    # ★ W7：生成基线 commit（默认取当前 HEAD 短 sha；可用 `--code-commit` 显式覆盖）
    code_commit = a.code_commit or _git_short_head()

    # ★ W8-①（2026-10-22）：父目录缺失时**创建**（⛔ 不再一上来 FATAL）；数据件本身缺失才拒绝，
    #   且文案写清"怎么才能有它"。
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        raise SystemExit(
            f"FATAL: 输出数据件不存在：{out_path}\n"
            "  本脚本是**增量更新**：它以在盘数据件的 meta 为基线重算 rows[]，不凭空造基线。\n"
            f"  要首次生成：先建基线（`tools/matrix/build_data.py` / `backfill_matrix.py`），"
            f"或用 `--out` 指向已有数据件。已确保父目录存在：{out_path.parent}"
        )
    base_doc = json.loads(out_path.read_text(encoding="utf-8"))

    doc, report, audit = build(card_limit=a.card_limit, budget=a.budget_gib,
                        generated_at=generated_at, script_sha=script_sha, base_doc=base_doc,
                        code_commit=code_commit)

    # ⓪ 写盘前 fail-closed：用 `scan_matrix_leaks` 的同一套规则扫发布件全文
    data_text = serialize_doc(doc)
    leaks = leak_hits_in_text(data_text)
    if leaks:
        print(f"\nFATAL: 发布件脱敏机核命中 {len(leaks)} 处 ⇒ ⛔ 拒绝写盘（⛔ 不放宽禁词、不加白名单）：",
              file=sys.stderr)
        for h in leaks[:20]:
            print(f"  ✗ {h}", file=sys.stderr)
        return 2

    print(summarize(doc, report))
    failed = [c for c in report["verification_cases"] if not c["ok"]]
    if failed:
        print(f"\nFATAL: 自检未通过 {len(failed)} 条 ⇒ ⛔ 不写任何文件（{', '.join(c['id'] for c in failed)}）",
              file=sys.stderr)
        return 2

    if a.dry_run:
        print("\n（--dry-run：未写任何文件）")
        return 0

    # ① 归档旧 meta/rows（历史留痕另存 .local，⛔ 不留在发布件里）
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "meta.before.json").write_text(
        json.dumps(base_doc.get("meta", {}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (archive_dir / "rows.before.json").write_text(
        json.dumps(base_doc.get("rows", []), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (archive_dir / "README.md").write_text(
        "# 旧 54 档矩阵归档（历史留痕）\n\n"
        "本目录由 `tools/matrix/enumerate_matrix.py` 在重生成档位表前自动写入，"
        "用于保留旧 54 档（T001–T054）的 `meta` 与 `rows[]` 原文。\n\n"
        "- 发布件 `docs/capability-matrix.data.json` **不再**携带旧档号与旧状态词（四态作废，改为三态）。\n"
        "- ⛔ 本目录在 `.local/` 下，不入 git。\n",
        encoding="utf-8")

    # ② 写数据件与剔除清单（确定性：固定键序、无时钟；generated_at 由 CLI 固定）
    out_path.write_text(data_text, encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # ★ 审计件（内部线索：backend_reason / steps_basis / 枚举块 / 证据表 / 已知缺口）⇒ .local，⛔ 不入库
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    audit_path.write_text(audit_text, encoding="utf-8")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"\n已写：{_rel(out_path)}（{len(doc['rows'])} 行）")
    print(f"已写：{_rel(report_path)}（剔除清单）")
    print(f"已写：{_rel(audit_path)}（审计件：backend_reason/steps_basis/枚举块/证据表）")
    print(f"已归档：{_rel(archive_dir)}/（旧 meta + 旧 rows）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
