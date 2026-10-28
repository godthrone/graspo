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

# ★ 2026-10-23（基准件唯一 + 按轮副本）：路径的**唯一真相源** = 同目录 `paths.py`（宪法 §1.4）。
#   ⛔ 本文件不再自写「基准件文件名 / 副本命名模板」字面量（病史：主席改名去空格后，生成端仍注入
#   带空格的旧名 ⇒ 重跑本脚本会把数据件的 `human_view_path` 悄悄改回一个不存在的路径）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _PATHS  # noqa: E402

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
    ("CPT", "native"): "native 后端无 CPT 训练通道（能力边界：该后端不提供 CPT 训练方法）",
    ("OPD", "native"): "native 后端无 OPD 训练通道（能力边界：该后端不提供 OPD 训练方法）",
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
# ★ 蓝图口径（2026-10-22 用户拍板）：第 4 章备注**只写前提 / 条件 / 边界**，
#   ⛔ 不写「我们跑过 / 没跑过」这类现状叙述（也不拿历史跑次当选型依据）。
#   由自检 V13 机核的等价防线：万一备注里出现成败断言，必须有具名证据兜底
#   ——「训练可跑 / 能跑 / 跑通」需 `success` 证据；「不可行 / 必然失败」需
#   `backend_infeasible` / `recipe_forbid` 证据；纯算式只能写「估算超预算」。
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
        "fact": "native LoRA 的 TP/PP/DP 三维都支持（TP SUM 同步、DP AVG 同步、PP 按层切分流水）",
        "short_fact": "native LoRA 的 TP/PP/DP 三维原生支持",
        "source": "src/graspo/flow/backend/native/lora/lora_linear.py:223-282；native/parallel/pipeline_comm.py",
        "tier": 0,
    },
    ("ms-swift", "全参"): {
        "supports_tp": False, "supports_pp": False, "supports_dp": True,
        "fact": "ms-swift 全参只能靠 FSDP2 摊（TP/PP 要走 Megatron 段；DeepSpeed 已禁用）",
        "source": "src/graspo/flow/backend/msswift/_config_mapping.py:206-213,682",
        "tier": 1,
    },
    ("ms-swift", "LoRA"): {
        "supports_tp": False, "supports_pp": False, "supports_dp": True,
        "fact": "ms-swift 标准通道只做纯数据并行（无 tp/pp 字段：tp 恒为 1 ⇒ SP 恒关闭；TP/PP 要走 Megatron 段）",
        "short_fact": "ms-swift 标准通道只做纯数据并行",
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
                "why_short": "native 全参只支持 PP 分片"
                           "（full 档 tp>1／dp>1 在加载期即不允许）"}

    # ① ★ 2026-10-22（用户拍板：选型**只按能力驱动**）：原「该后端在该档有历史跑次 ⇒ 直接出局」
    #   分支已删除 —— 选型不再拿任何历史跑次当选型依据，只看「后端能否承载该形状 + 估算是否装得下」。
    #   证据仍留在 `backend_reason.candidates[].gaps` 与审计件里（⛔ 不抹事实），但**不参与选型**。
    #   受影响档：27B GRASPO LoRA dp2（两个后端都能吃该形状 ⇒ 按优先级取 ms-swift）。

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

    # ── 产物可推理（★ 2026-10-22 更正：侦察已推翻「native LoRA 张量并行拼接未实现」） ──
    #   ❌ 旧陈述（已删，⛔ 不许再传播）："native 的多卡导出只覆盖 DP 副本合并，TP 结果拼接尚未实现"。
    #   ✅ 事实：**算法侧导出已实现**（LoRA 的 TP/PP/DP 三维都有合并路径），
    #      只是**真机尚未验证**（既有档位里 LoRA 恒为 tp=pp=1 ⇒ TP/PP 形状零实跑）。
    #   ⛔ 因此这里不再给任何 LoRA 档打 `deliverable=False`：卡点只写「已实现、真机未验证」。
    #   native 全参的 PP 产物导出/合并仍未实现（那是真正的欠账，写在 row_of 的卡点里）。
    deliverable, delivery_gap, delivery_gap_short = True, None, None

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


#: 后端选型优先级（⭐ 2026-10-22 用户拍板：**能力驱动**，⛔ 不再有一堆形状特例）。
#:   LoRA：两个都能做 ⇒ 取 ms-swift；全参 dp>1 ⇒ 取 ms-swift；全参 dp==1 ⇒ 取 native。
BACKEND_PRIORITY: dict[tuple[str, str], tuple[str, ...]] = {
    ("LoRA", "any"): ("ms-swift", "native"),
    ("全参", "dp>1"): ("ms-swift", "native"),
    ("全参", "dp==1"): ("native", "ms-swift"),
}


def backend_can_consume(backend: str, mode: str, tp: int, pp: int, dp: int) -> bool:
    """该后端是否能**消费**这个并行形状（能力表 `BACKEND_PARALLEL_FACT` 是唯一真相源）。

    ★ 2026-10-22（用户拍板）：native 能吃 LoRA 任意 tp/pp/dp；native 全参只吃 `tp==1 ∧ dp==1`
      （pp 任意）；ms-swift 只吃 `tp==1 ∧ pp==1`（dp 任意）。
    """
    f = BACKEND_PARALLEL_FACT[(backend, mode)]
    return ((tp == 1 or f["supports_tp"])
            and (pp == 1 or f["supports_pp"])
            and (dp == 1 or f["supports_dp"]))


def priority_order(mode: str, dp: int) -> tuple[str, ...]:
    return BACKEND_PRIORITY[(mode, "any" if mode == "LoRA" else ("dp>1" if dp > 1 else "dp==1"))]


def deletion_reason_code(algo: str, mode: str, tp: int, pp: int, dp: int) -> str:
    """**一个后端都做不了**时的原因码（两条显式判据，⛔ 不靠"排除法"猜）。

    · `NATIVE_NO_CHANNEL`：形状本就是 native 全参能吃的形状（`tp==1 ∧ dp==1`，pp 任意），
      但 native 侧没有该算法的训练通道（CPT / OPD）。
    · `MSSWIFT_NO_PARALLEL`：其余情形都是"要靠 ms-swift 的 TP/PP，而 Megatron 通道未接线"
      （LoRA 的 TP/PP 形状、全参的 `tp>1`/`dp>1` 形状）。
    """
    if mode == "全参" and "native" not in ALGO_BACKEND_SUPPORT[algo] and tp == 1 and dp == 1:
        return "NATIVE_NO_CHANNEL"
    return "MSSWIFT_NO_PARALLEL"


def select_backend(
    *, algo: str, model: str, mode: str, shape: Shape, budget: float,
) -> dict[str, Any]:
    """逐档选后端：**先算每档有哪些后端真能做，再按固定优先级取**（⭐ 2026-10-22 用户拍板）。

        · native 能做的形状：LoRA 任意 `tp/pp/dp`；全参 `tp==1 ∧ dp==1`（`pp` 任意）。
        · ms-swift 能做的形状：`tp==1 ∧ pp==1`（`dp` 任意）。
        · 两个都能做时的优先级（`BACKEND_PRIORITY`，单一真相源）：
              LoRA ⇒ ms-swift；全参 `dp>1` ⇒ ms-swift；全参 `dp==1` ⇒ native。
        · 一个都不能做 ⇒ 该档不入表（原因码见 `deletion_reason_code`）。

    ⛔ 不再有"形状特例"式排序（旧规则里的「LoRA ∧ dp>1 ∧ tp=pp=1 ⇒ ms-swift」等一律收掉）：
    `candidates` 仍逐后端评估（审计件要留落选原文），但**只按优先级在"能做且装得下"的候选里取**。
    """
    tp, pp, dp = shape
    cands = [eval_backend(algo=algo, model=model, mode=mode, backend=be, shape=shape, budget=budget)
             for be in ALGO_BACKEND_SUPPORT[algo]]
    # CPT/OPD 的 native 缺失是**算法层缺实现**（现在按 `NATIVE_NO_CHANNEL` 剔除，不再"保留待测试"）
    missing = [ALGO_BACKEND_NOTE[(algo, be)] for be in BACKENDS
               if be not in ALGO_BACKEND_SUPPORT[algo] and (algo, be) in ALGO_BACKEND_NOTE]

    capable = [c["backend"] for c in cands if backend_can_consume(c["backend"], mode, tp, pp, dp)]
    for c in cands:
        c["rank_key"] = None
        c["shape_capable"] = c["backend"] in capable
    order = priority_order(mode, dp)
    ranked = sorted([c for c in cands if c["eligible"] and c["shape_capable"]],
                    key=lambda c: order.index(c["backend"]))
    for i, c in enumerate(ranked):
        c["rank_key"] = i
    chosen = ranked[0] if ranked else None
    return {"candidates": cands, "chosen": chosen, "missing_backends": missing,
            "shape_capable": capable, "priority": list(order)}


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
    "NATIVE_NO_CHANNEL": {
        "desc": ("**该形状本该由 native 承担，而 native 后端没有这个训练方法的通道**"
                 "（算法层未提供 CPT / OPD 在 native 后端的实现）⇒ 剔除"),
        "excludes": True,
    },
    "MSSWIFT_NO_PARALLEL": {
        "desc": ("**ms-swift 的张量 / 流水并行要走 Megatron 通道，该通道当前未接线**"
                 "⇒ ms-swift 只能纯数据并行，需要 TP/PP 的形状一律剔除（此前按 tp*pp 打折估算属估算缺陷）"),
        "excludes": True,
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

    # ── 前提 / 边界：逐条写清"这档要跑成需要什么条件、有什么边界"（⛔ 不写"我们做过/没做过什么"）
    gaps: list[str] = []
    if sel.get("missing_channel"):
        gaps.append(f"native 后端无 {algo} 训练通道（能力边界）；"
                    f"按 native 配方估算 {peak:.2f} GiB 可行")
    if backend == "ms-swift":
        if mode == "全参":
            gaps.append("优化器态须走 FSDP2 全分片载体（全参档在 ms-swift 上的分片方式）")
        if tp > 1 or pp > 1:
            gaps.append("ms-swift 的 TP/PP 无通道（标准通道只覆盖 DDP+SP）"
                        "⇒ 该形状的分摊只能靠 FSDP2 全分片来承载")
    if backend == "native" and mode == "全参":
        if pp > 1:
            # ★ 2026-10-22（用户拍板 C 留）：这 6 档**保留、状态仍待测试**，理由写清是"待实现"。
            # ★ 边界句的**唯一措辞** = 本模块常量 `NATIVE_FULL_PP_GAP`（⛔ 别在这里另写一份）。
            #   2026-10-22 主席令清空备注后，该句只进 `.local` 审计件的 `backend_reason.gaps`
            #   （页面备注为空）⇒ 原 V17「必须出现在备注里」随之退役（见 `VERIFICATION_CASES`）。
            gaps.append(NATIVE_FULL_PP_GAP)
        if algo == "GRASPO" and pp > 1:
            gaps.append("native 的流水线并行 + rollout 组合须显式解除未验证保护"
                        "（`native.allow_unverified_pp_rollout=true`）")
    if backend == "native" and mode == "LoRA" and (tp > 1 or pp > 1):
        # ★ 2026-10-22（蓝图口径）：只写**前提**——该形状的训练可行性由本档测试建立；
        #   ⛔ 不写"从未实跑 / 真机未验证"这类叙述我们做过什么的话。
        gaps.append("LoRA 的 TP/PP 组合是本矩阵新增形状 ⇒ 训练可行性由本档测试建立；"
                    "产物侧合并路径覆盖 TP/PP/DP 三维")
    if chosen["recipe"] == "offload" and mode == "全参" and tp * pp * dp == 1:
        # ★ 2026-10-22（用户拍板）：保留档里"必须开优化器态 offload 才装得下"的只有 1 卡全参档，
        #   在审计件的 `backend_reason.gaps` 里写清（⛔ 不新增列；`switches` 已按同一配方写
        #   `offload_optimizer_state=true`）。页面备注按主席令清空，⛔ 不再承载该句。
        gaps.insert(0, "1 卡全参必须开优化器态 offload 才能装下（否则估算 79.24 GiB 超预算，"
                       "开 offload 后 45.71 GiB）")
    if chosen["recipe"] == "offload":
        gaps.append("需 `native.offload_optimizer_state=true`")
        # ★ 2026-10-22 更正（侦察已推翻旧陈述"只在 SFT 通道生效、GRASPO 是否生效未确认"）：
        #   该开关是**后端级单一钩子**，SFT / RL 两条 trainer 共用同一 override。
        gaps.append("offload 是后端级单一钩子，SFT / RL 两条 trainer 共用")
    if algo == "OPD" and chosen["teacher_variant"] == "offload":
        gaps.append("需 `offload_teacher_model=true`")
    gaps = gaps[:4]

    # ── 备注列：**按主席 2026-10-22 指令清空**（备注在测试完成后回填，枚举阶段⛔ 不写任何内容）──
    #   主席原话：「备注本来就是测试完成之后写的，现在什么都没测，当然不应该有内容。」
    #   ⇒ `rows[].note` 保留字段、值为空串（页面渲染出空 `<td>`；字段契约不变，供回填脚本使用）。
    #   ★ 选型理由 / 前提边界仍在 `backend_reason`（含 `gaps`）里，落 `.local` 审计件；
    #     ⛔ 不再合成进发布件的 `rows[].note`（旧 `notes_extra` / `_reason_line` 只喂这一处 ⇒ 同批删除，
    #     ⛔ 不留只写不读的死代码）。
    note = ""

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
        # ★ 2026-10-22（复核条目 6）：⚠ 本字段的唯一含义 = **每 epoch 步数**；
        #   `derived_steps` = 总轮数 × 每 epoch 步数（配置侧推导量）。
        #   ★ 2026-10-23：判据① 已按主席原文改为「跑完总训练轮数（Epoch）」⇒ 步数
        #   **不再是判据① 的通过门槛**；本字段仅作配置/回填的推导量，⛔ 两个词不混用。
        "planned_steps": str(per_epoch),
        "train_path": f"subsets-v2/{tier}.jsonl",
        "data_source": "ELAM V5 均衡子集（12 格分层 · seed 20261022 · 100 条）",
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
            # ★ 2026-10-22（用户拍板：选型只按能力驱动）：排序键重写 —— ⛔ 不再以历史跑次排序。
            "sort_key": ("①后端能否承载该形状（能力表） ②估算峰值是否装得下（≤ 设计预算） "
                         "③固定优先级（LoRA ⇒ ms-swift；全参 dp>1 ⇒ ms-swift；全参 dp==1 ⇒ native）"),
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


def _reason_rule(sel, backend) -> str:
    """结构化理由里的"为什么是它" —— **只按能力面排序**（⛔ 不以历史跑次排序）。

    ★ 2026-10-22（用户拍板：选型只按能力驱动）：排序键改为
      ① 后端能否承载该形状（能力表 `BACKEND_PARALLEL_FACT`）② 估算峰值是否装得下
      ③ 固定优先级（`BACKEND_PRIORITY`）。历史跑次只作为审计线索留在 `candidates[].why` 里，
      ⛔ 不参与排序、⛔ 不进人读件。
    """
    ranked = sorted([c for c in sel["candidates"] if c["eligible"]],
                    key=lambda c: c.get("rank_key") if c.get("rank_key") is not None else 99)
    if not ranked:
        return "无可行候选"
    top = ranked[0]
    why = []
    why.append(f"选型 = 能承载该形状的后端里按固定优先级取：{backend}"
               f"（优先级 {sel.get('priority')}）")
    why.append(f"形状是否被后端真消费：{'是' if top['fits_shape'] else '否'}")
    why.append(f"估算峰值 {top['peak_gib']:.2f} GiB（判定 {top['band']}）")
    for other in ranked[1:]:
        why.append(f"次优 {other['backend']}：峰值 {other['peak_gib']:.2f} GiB、"
                   f"形状真消费 {'是' if other['fits_shape'] else '否'}")
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


def _v11_capability_selection(rows, budget: float) -> bool:
    """能力驱动选型不变量（⭐ 2026-10-22 用户拍板）：每档的后端必须是**它能吃这个形状**的那个，
    且两个都能吃时按固定优先级（LoRA ⇒ ms-swift；全参 dp>1 ⇒ ms-swift；全参 dp==1 ⇒ native）。

    旧特例（「LoRA ∧ dp>1 ∧ tp=pp=1 ⇒ 一律 ms-swift」）已随选型重写收掉 ⇒ 本机核改为查不变量本身。
    """
    for r in rows:
        mode, backend = r["mode"], r["backend"]
        tp, pp, dp = int(r["tp"]), int(r["pp"]), int(r["dp"])
        if not backend_can_consume(backend, mode, tp, pp, dp):
            return False
        cands = r["backend_reason"]["candidates"]
        capable_feasible = [c["backend"] for c in cands
                            if c["eligible"] and backend_can_consume(c["backend"], mode, tp, pp, dp)]
        order = priority_order(mode, dp)
        want = min(capable_feasible, key=order.index) if capable_feasible else None
        if want != backend:
            return False
    return True


def _v12_deleted_reason_codes(excluded: list, budget: float) -> bool:
    """两条新剔除判据必须真的落地（⛔ 不许"删了档却说不出为什么"）。

    · `NATIVE_NO_CHANNEL` 只许出现在 CPT / OPD 的**全参**档（形状本是 native 全参能吃的 `tp==1 ∧ dp==1`）；
    · `MSSWIFT_NO_PARALLEL` 只许出现在"要靠 ms-swift 的 TP/PP"的档（`tp>1` 或 `pp>1`）；
    · 两条码的删除档数都必须 > 0（判据失效即红，⛔ 不是恒真断言）。
    """
    nc = [e for e in excluded if e["reason_code"] == "NATIVE_NO_CHANNEL"]
    npar = [e for e in excluded if e["reason_code"] == "MSSWIFT_NO_PARALLEL"]
    if not nc or not npar:
        return False
    for e in nc:
        k = e["key"]
        if k["mode"] != "全参" or k["algo"] in ("SFT", "GRASPO"):
            return False
        if (k["shape"]["tp"], k["shape"]["dp"]) != (1, 1):
            return False
    for e in npar:
        k = e["key"]
        if not (k["shape"]["tp"] > 1 or k["shape"]["pp"] > 1):
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
    "excluded": frozenset({"EXCEED_CARDS", "MEMORY_UNFIT", "KNOWN_MEMORY_FAIL", "UNSUPPORTED_MODE",
                              "ALGO_BACKEND", "NATIVE_NO_CHANNEL", "MSSWIFT_NO_PARALLEL"}),
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


#: ★ 2026-10-22（主席令：清空备注内容）—— `NATIVE_FULL_PP_GAP` 的**页面载体已不存在**
#:   （备注列按令清空，备注在测试完成后才回填）⇒ 该句只留在 `.local` 审计件的
#:   `backend_reason.gaps` 里；原 V17「每档必须逐字出现在备注里」随之**正式退役**
#:   （退役登记见 `VERIFICATION_CASES` 的 `retired` 字段 + `run_self_check` 的 `[RETIRED]` 打印）。
#: native 全参 PP>1 的**产物交付边界句**（逐字，⛔ 不许改写成"别的载体"这类悬空说法）。
#: ★ 2026-10-22（蓝图口径）：删掉尾句「本档保留并按待测试推进」——它与第 4 章状态列重复；
#:   边界事实（需跨层分片合并才能导出、导出能力只到单卡单机）一字不动。
NATIVE_FULL_PP_GAP = ("native 全参的 PP>1 档要交付产物，须跨层分片合并导出"
                      "（当前导出能力只到单卡单机）")


def _v13_notes_cleared(rows, budget: float) -> bool:
    """V13（2026-10-22 主席令清空备注）：全部档位的 `rows[].note` 必须是**空串**。

    主席原话：「备注本来就是测试完成之后写的，现在什么都没测，当然不应该有内容。」
    ⇒ 枚举阶段⛔ 不写备注；字段**保留**、值为空串（供测试完成后的回填脚本写入）。
    ★ 该断言仍**能失败**（⛔ 不是恒真）：任何档位被写进内容（含 `—` 占位）即红；
      它只判"是否为空"，⛔ 不判内容语义（内容语义属将来的回填侧）。
    """
    return all(r.get("note") == "" for r in rows)


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
        "claim": ("CPT / OPD 无 native 通道 ⇒ 保留档**一律** ms-swift（2026-10-22 用户拍板："
                  "native 侧做不了的档按 `NATIVE_NO_CHANNEL` 剔除，⛔ 不再以 MISSING_CHANNEL 保留）"),
        "check": lambda rows, budget: (
            sum(1 for r in rows if r["algo"] == "CPT") > 0
            and sum(1 for r in rows if r["algo"] == "OPD") > 0
            and all(r["backend"] == "ms-swift" for r in rows if r["algo"] in ("CPT", "OPD"))),
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
        "claim": ("任何档位都不以 DeepSpeed 为选型/配方（DS 已禁用）；Megatron 只以**能力边界**出现"
                  "—— 页面备注已按主席 2026-10-22 指令清空（见 V13），故页面侧不发布任何工程叙述；"
                  "能力边界原文仍留在 `.local` 审计件（`backend_reason` 里的「TP/PP 要走 Megatron 段／通道」）"),
        "check": lambda rows, budget: all(
            r["deepspeed"] == "—" and "deepspeed" not in r["switches"].lower() for r in rows)
        # 审计件里仍必须留下这条能力边界的原文（⛔ 不是"删了就当没有"）
        and any("Megatron" in json.dumps(r["backend_reason"], ensure_ascii=False) for r in rows),
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
        "claim": ("能力驱动选型不变量：每档后端必须**能吃该形状**（native 全参只吃 tp==1∧dp==1；"
                  "ms-swift 只吃 tp==1∧pp==1），且两后端都能吃时按固定优先级"
                  "（LoRA ⇒ ms-swift / 全参 dp>1 ⇒ ms-swift / 全参 dp==1 ⇒ native）"
                  "—— 旧「LoRA∧dp>1 特例」已收掉，本机核查的是不变量本身"),
        "check": _v11_capability_selection,
    },
    {
        "id": "V12",
        "claim": ("两条新剔除判据必须真的落地：`NATIVE_NO_CHANNEL` 只出现在 CPT/OPD 的全参档"
                  "（形状是 native 全参能吃的 tp==1∧dp==1）；`MSSWIFT_NO_PARALLEL` 只出现在 tp>1∨pp>1 的档；"
                  "两码的档数都必须 > 0"),
        "check": lambda rows, budget: True,   # 由 `run_self_check` 传入剔除清单单独核（见 V12 分支）
    },
    {
        "id": "V13",
        "claim": ("备注内容一律为空（主席 2026-10-22 指令：备注在测试完成后回填，枚举阶段⛔ 不写任何内容）"
                  "—— `rows[].note` 字段保留、值为空串；页面备注单元格因此也为空"),
        "check": _v13_notes_cleared,
    },
    {
        "id": "V17",
        "claim": ("`全参 ∧ pp>1 ∧ native` 的**每一档**都必须逐字披露产物交付边界句"
                  "（「" + NATIVE_FULL_PP_GAP + "」）—— 该组档现在交不出产物，少写一档即隐瞒"),
        # ★ 正式退役（2026-10-22 主席令「备注内容全清空」）：该断言的**载体** = 第 4 章备注列，
        #   备注按令为空 ⇒ 页面不再发布该句（边界原文仍在 `.local` 审计件 `backend_reason.gaps`）。
        #   ⛔ 不是改期望值变绿、⛔ 不是静默跳过：命中后由 `run_self_check` 逐条打 `[RETIRED]` + 理由。
        #   若将来备注回填重新承载该句，删掉本 `retired` 键即可恢复机核（判据函数仍在 git 历史里）。
        "retired": ("主席 2026-10-22 指令清空备注内容 ⇒ 该断言的载体（备注列里的产物交付边界句）"
                    "不再发布；边界原文保留在审计件 backend_reason.gaps"),
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
                   data_text: str | None = None,
                   excluded: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    out = []
    for case in VERIFICATION_CASES:
        # ★ 正式退役面（2026-10-22）：断言的**对象**已随主席令/换代不存在 ⇒ 不吞掉、不静默跳过：
        #   仍逐条登记 `retired` + 理由，由 `summarize` 打 `[RETIRED]`；`ok=True` 故不阻断写盘。
        #   ⛔ 未登记的判据照旧 fail-closed（`failed` 过滤只认 `ok`）。
        if case.get("retired"):
            out.append({"id": case["id"], "claim": case["claim"], "ok": True,
                        "retired": True, "reason": case["retired"]})
            continue
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
        if case["id"] == "V12":
            ok = excluded is not None and _v12_deleted_reason_codes(excluded, budget)
            nc = [e for e in (excluded or []) if e["reason_code"] == "NATIVE_NO_CHANNEL"]
            npar = [e for e in (excluded or []) if e["reason_code"] == "MSSWIFT_NO_PARALLEL"]
            out.append({"id": "V12", "claim": case["claim"], "ok": ok,
                        "reading": f"NATIVE_NO_CHANNEL={len(nc)}；MSSWIFT_NO_PARALLEL={len(npar)}"})
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
    # ★★ 2026-10-22（主席第 2/3 条指令）：人读件改名 ⇒ **文档名与 `authority` 里的路径**由生成器注入。
    #   病史：`title` 与 `authority.*` 原先只从在盘数据件**继承**（本脚本从不写这两处），
    #   改名后旧路径会原样留在数据件里 ⇒ 页面 meta 与数据件不同源、`human_view_path` 指向不存在的文件。
    #   ⇒ 逐条显式覆写（⛔ 不靠正则替换，⛔ 不手改数据件）。
    {"path": "title", "value": "GRASPO 训练框架功能测试报告",
     "why": "人读件改名：页面 <title> 由本字段渲染，须与 <h1> 同名"},
    {"path": "authority.editable_source",
     "value": ("docs/GRASPO 训练框架功能测试报告.html"
               "（唯一真相源 / 权威；结论以人读件为准）"),
     "why": "人读件改名：authority 里的旧路径会把人指到不存在的文件"},
    {"path": "authority.human_view",
     "value": ("docs/GRASPO 训练框架功能测试报告.html"
               "（本页即测试结论本身；人读件为唯一真相源，可逐字节重建）"),
     "why": "同上"},
    {"path": "authority.human_view_path",
     # ★ 2026-10-23（基准件唯一 + 按轮副本）：本字段是**路径**、且就是机核用来比对页面路径的那一个
     #   ⇒ 取值一律取 `paths.py` 的基准件常量（宪法 §1.4 唯一真相源；⛔ 本文件不再抄一份路径字面量）。
     #   ★ 其余三条 `authority.*` 是**人读散文**（含「（唯一真相源 / 权威…）」等说明文字），
     #     其旧名残留属本包**保留项**（见交付报告 ⑦）：散文改字要连带重渲染基准件，
     #     而基准件已 444 冻结（sha 见 `paths.py`）⇒ 属「基准件内容改写」授权面，不在本包。
     "value": _PATHS.REL_BASELINE_HTML,
     "why": "同上（本字段就是机核用来比对页面路径的那一个）"},
    {"path": "authority.direction",
     "value": ("结果 JSON → capability-matrix.data.json（仅本页的渲染输入，非权威）→ "
               "人读视图 docs/GRASPO 训练框架功能测试报告.html（唯一真相源；单方向，禁止互推）"),
     "why": "同上"},
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
    # ★ 2026-10-22 / 10-23（判据 ①④ 改写）：页面第 2 章四条 `<li>` ↔ 数据件 `pass_criteria` 必须**逐字一致**
    #   （机核 = verify_matrix ④X-b）。此处按同一份文本覆盖 ⇒ ⛔ 不允许页面改了、数据件留在旧句上。
    {"path": "criteria_caliber.pass_criteria[0].text",
     "value": ("训练正常结束、无中断、无报错、退出码 0、跑完总训练轮数（Epoch）、"
               "最终检查点完整可加载、没有 NaN/Inf。"),
     "why": ("★ 2026-10-23 主席原文重写：判据① 的通过条件改为「跑完总训练轮数（Epoch）」；"
             "删除「实跑步数达到该档总训练步数」及其步数公式（⛔ 步数不再是判据① 的通过门槛）；"
             "「最终检查点完整可加载」「没有 NaN/Inf」两项保留")},
    # ★ 2026-10-23（主席原文重写）：判据② 只留短名「效果变好」+ 四个算法小项；
    #   平台方评测条件长括号与 SFT 的「与参考答案逐字段完全一致」括注按主席原文删除
    #   （评测口径改由数据件 `threshold` / `eval_caliber` 承载，判据文本不再重复）。
    {"path": "criteria_caliber.pass_criteria[1].text",
     "value": ("数据集完全正确率 = 与参考答案逐字段一致的样本数 ÷ 100；"
               "base = 同一基座未训练时的同口径读数，after = 训练后的同口径读数，"
               "每档训练前后各评一次。"),
     "why": ("★ 2026-10-24 主席原文：判据② 正文补「完全正确率与 base 定义」一句"
             "（判定口径，⛔ 不提温度——温度由判据④ 与「公共超参」承载）；"
             "四个算法小项与其数值/通过线一字未动")},
    # ★ 2026-10-22（判据③ 改写）：页面第 2 章判据③ 的那一句话 ↔ 数据件同一条文本必须逐字一致。
    #   GRASPO 一侧改指**人读口径件**《GRASPO 算法说明》（docs/graspo-algorithm.html，页面相对链接）；
    #   机核 = verify_matrix ④ALG（句子恰 1 次 + 链接在盘 + 新文档 7 锚点齐备）。
    # ★★ 2026-10-22（复核条目 3 / 14）：判据③ 补 SFT / CPT / OPD 的可执行对照项；
    #   **并把链接标记写进数据件**（人读件里的 `<a href="graspo-algorithm.html">` 必须与
    #   数据件同句同源，否则重生成 / 回填链路会把链接丢掉）。
    #   机核：`verify_matrix` ④X-b（按**原文含标签**逐字比对）+ ④X-d（链接标记恰 1 次、
    #   去标签后与 `ALGO_CRITERION_TEXT` 逐字一致）；阳性对照：`run_self_proof` ㊱。
    {"path": "criteria_caliber.pass_criteria[2].text",
     "value": ("SFT / CPT / OPD 按各自公开的标准实现逻辑判断"
               "（以 HuggingFace Transformers 与 ms-swift 官方实现为准）；"
               "GRASPO 按《<a href=\"graspo-algorithm.html\">GRASPO 算法说明</a>》逐条判断"
               "（算法逻辑严禁改动）。SFT / CPT / OPD 另按公开标准核对损失函数与标签掩码"
               "的处理规则（OPD 另含教师信号的处理规则）。"),
     "why": ("判据③ 由「各按自己的清单」改为可判定的一句：GRASPO 以独立人读件逐条对照"
             "（算法逻辑不改）；并补 SFT / CPT / OPD 的可执行对照项")},
    {"path": "criteria_caliber.pass_criteria[3].text",
     "value": ("LoRA 档合并后、全参档直接，用 vLLM 启动（温度=0），在训练数据集上完整推理一遍，"
               "能完整产出可解析的推理结果。"),
     "why": ("★ 2026-10-23 主席原文重写：④ 加「（温度=0）」；删除"
             "「（效果是否变好由判据② 判定，本条只判能否推理）」括注；"
             "④ 仍同时覆盖两种模式（LoRA 档合并后 / 全参档直接）")},
    # ★ 2026-10-22（复核条目 8）：OPD 教师线由「高于」改为「≥」表达（数值 +10.0 pp 不动）。
    #   学生线 after − base ≥ +5.0 pp 一字不动。
    {"path": "criteria_caliber.threshold.by_method.opd.teacher",
     "value": "教师 − 学生未训练模型 ≥ +10.0 个百分点（教师在训练数据集上的完全正确率）",
     "why": "OPD 教师线改用 ≥ 表达（允许相等）；数值 10.0 不变；学生线不变"},
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
     "value": ("平台方自有、不公开数据集；评测集 = 该档自身训练用的同一份文件（fit 口径：训练集自评）"),
     "why": ("移除旧 54 档的 45 档档号清单；★ 2026-10-22 复核第 2 轮：删掉残句"
             "「＋ 100 行档（判据 = 该档 subset_size == 100）的评测集」⇒ 评测集不再分档"
             "（⛔ 不改任何数值）")},
    {"path": "criteria_caliber.dataset.train_eval_relation",
     "value": ("评测集口径：fit 口径（训练集自评）—— 评测集 = 该档自身训练用的同一份文件"
               "（100 行、dataset_split: train）⇒ 读数回答的是「该档训练是否生效」；不得读作训练分布之外的表现"
               "（本矩阵不评测该问题）。档位清单一律从行数据现算（不写死档号）。"
               "判别力：该 100 行的 (工具, 动作) 类别最大类占 17%（17/100）⇒ 常数预测器上界 17%；"
               "而通过线 = after − base ≥ +10.0 pp（点估计）⇒ 跨过它必须真的读图，任何常数输出都拿不到。"
               "旧 54 档矩阵的 702 行口径读数与「重叠剔除」次级读数已随旧矩阵归档（不得引用）：mini 档实测 "
               "mini100 × 702 图像交集 = 0 ⇒ 702 行口径对 mini 档自始不成立；评测集 ≡ 训练集 "
               "⇒ 重叠剔除类次级读数按定义退化、不构成次级读数。"),
     "why": ("移除旧档号清单与四态措辞；保留口径与判别力事实；★ 2026-10-22 复核第 2 轮：删掉分档前缀"
             "「100 行档 =」「判别力（100 行档）」「100 行档下」与「（判据 = subset_size == 100）」"
             "⇒ 与「评测集不分档」统一（⛔ 不改任何数值；702 的历史留痕句逐字保留）")},
    {"path": "criteria_caliber.eval_caliber.rule",
     "value": ("评测集 = 该档自身训练用的同一份文件（训练集自评）；不另设平台方评测集分支。"),
     "why": ("移除旧 54 档的 45 档档号清单；★ 2026-10-22 复核阻断-2：删掉与页面相反的"
             "「其余档 = 平台方自有 702 行评测集 + dataset_split: test」分支 ⇒ 与页面判据②"
             "「训练数据集即评测集（同一份 100 条）」一致（⛔ 不改任何数值）；"
             "★★ 复核第 2 轮：前半句的分档前缀「100 行档（判据 = 该档 subset_size == 100）…；」"
             "与后半句「不分档」自相矛盾 ⇒ 只留不分档那一句（⛔ 不新增、不改任何数值）")},
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
    # ════════════════════════════════════════════════════════════════════════
    # ★★ 2026-10-22（复核收口 · 阻断-2 / 警告 1-2）：数据件的**当前判据字段**必须与页面一致。
    #   页面判据② 已写「训练数据集即评测集（同一份 100 条）」；数据件里凡仍带
    #   「其余档 = 平台方 702 行评测集」的**当前判据**字段一律删掉该分支（⛔ 不改任何数值）。
    #   历史留痕字段（caliber_change.* / eval_caliber.untouched / eval_caliber.consequence /
    #   dataset.train_eval_relation 的「旧矩阵 702 行读数已归档」句 / caliber_evidence 的
    #   噪声底证据）**保留** 702 —— 那是历史事实，⛔ 不抹。
    # ════════════════════════════════════════════════════════════════════════
    {"path": "criteria_caliber.threshold.by_method.sft.metric",
     "value": "数据集完全正确率（after − base）；评测集 = 该档自身训练用的同一份文件（fit 判定标准，100 行）",
     "why": "当前判据字段：删掉与页面相反的「其余档 = 平台方自有 702 行评测集」分支（⛔ 数值不动）"},
    {"path": "criteria_caliber.threshold.by_method.graspo.metric",
     "value": "数据集完全正确率（after − base）；评测集 = 该档自身训练用的同一份文件（fit 判定标准，100 行）",
     "why": "同上（GRASPO 与 SFT 同式；通过线数值不在此字段）"},
    {"path": "criteria_caliber.threshold.by_method.cpt.metric",
     "value": "数据集完全正确率（after − base）；评测集 = 该档自身训练用的同一份文件（fit 判定标准，100 行）",
     "why": "同上（CPT 走不劣化线，数值在 op/value 上，本字段只描述度量）"},
    {"path": "criteria_caliber.threshold.by_method.opd.metric",
     "value": "数据集完全正确率（学生 after − base）；评测集 = 该档自身训练用的同一份文件（fit 判定标准，100 行）",
     "why": "同上（OPD 度量的是学生 Δ；⛔ 不改 +5.0 / +10.0 两个数值）"},
    {"path": "criteria_caliber.dataset.sample_count",
     "value": "评测集 = 该档自身训练文件（fit 判定标准）",
     "why": ("当前判据字段：删掉「其余档 = 702」分支（⛔ 100 一字不动）；"
             "★ 复核第 2 轮：再删残句「100 行档 = 100」⇒ 不分档（⛔ 不改任何数值）")},
    {"path": "criteria_caliber.dataset.version_anchor_sha256",
     "value": "评测集 = 该档自身训练文件（100 行，sha256 5ad2d7cc…）",
     "why": "当前判据字段：删掉「其余档 = 平台方自有评测集（702 行，c202b1e1…）」分支（⛔ 100 / sha256 不动）"},
    {"path": "criteria_caliber.dataset.availability",
     "value": "平台方自有 / 不公开（评测集用该档自身训练文件，同为非公开数据）",
     "why": ("★ 2026-10-22 复核第 2 轮：删掉分档残句「100 行档用该档自身训练文件」⇒ 不分档"
             "（⛔ 不改任何数值）")},
    {"path": "criteria_caliber.dataset.train_eval_relation_role",
     "value": ("本条是评测集口径（不是能力承诺）：矩阵只声明「已在本口径下完整跑通并取数」，"
               "使用者自身任务/数据集/参数下的效果不在承诺范围（见边界声明）。"
               "评测集为 fit 口径（训练集自评）：其读数只证明该档在其自身训练数据上的拟合与训练有效性，"
               "不得据此外推到其他数据上的表现。"),
     "why": ("★ 2026-10-22 复核第 2 轮：分档前缀「100 行档为 fit 口径」⇒「评测集为 fit 口径」"
             "（⛔ 不改任何数值与承诺边界语义）")},
    {"path": "criteria_caliber.eval_caliber.discriminative_power",
     "value": ("训练集 (工具, 动作) 类别最大类 17/100 = 17% ⇒ 常数预测器上界 17% ⇒ "
               "跨 +10.0 pp 必须真读图；每样本粒度 1.00 pp（达线需净跳 ≥10 条）。"),
     "why": ("★ 2026-10-22 复核第 2 轮：删掉分档前缀「100 行档：」⇒ 不分档"
             "（⛔ 19/100、19%、10.0、1.00、10 条等数值一个不动）")},
    {"path": "criteria_caliber.baseline.caliber",
     "value": ("未训练基座在同一评测集上的数据集完全正确率（accuracy），来自 graspo eval 的 role=base 通道；"
               "评测集 = 该档自身训练用的同一份文件（fit 判定标准）。"
               "本项是判据② 的被减数（after − base 中的 base）；与 CPT 的「训练 loss 正常下降」判据不是一回事。"),
     "why": "当前判据字段：删掉「其余档 = 702 行评测集」分支（⛔ 不改任何数值与减法关系）"},
    # ★★ 2026-10-22（警告-1）：当前判据字段不得把页面现在采用的说法称作「已作废」。
    #   ⛔ 两个数值（+10.0 / +5.0）与 epoch=2 一个都不动。
    {"path": "criteria_caliber.threshold.by_method.opd.epoch_note",
     "value": ("epoch=2 不变；教师 = Qwen3.8-27B 在 SFT / LoRA / dp=1 这一档训练完成后的产物"
               "（不是基座权重本身）；教师线：教师 − 学生未训练模型 ≥ +10.0 pp。"),
     "why": ("改写「教师说明已作废」的旧措辞：页面现行说法（教师 = 27B 的 SFT / LoRA / dp=1 档产物）"
             "就是当前判据；教师线按 ≥ 表达。⛔ 数值 +10.0 pp / epoch=2 不动")},
    # ★★ 2026-10-22（警告-2）：status 只追加当日两条，⛔ 不改写旧记录（逐字留痕）。
    {"path": "criteria_caliber.threshold.status",
     "value": ("已冻结（现行判据，2026-10-15；OPD 通过线已于 2026-10-19 变更，见 "
               "caliber_change.caliber_change_20261019；2026-10-22：教师身份已指定为"
               "「27B / SFT / LoRA / dp=1」档产物，教师线已明确为 ≥）：判据② 逐算法分列；不放宽判据；"
               "判定一律用点估计 Δ —— 不做假设检验、不引入置信区间 / 显著性。"),
     "why": ("补 2026-10-22 两条（教师身份指定到档位 / 教师线定 ≥）；旧记录保留、只追加；"
             "「已冻结」前缀保留（回填侧按「含冻结且不含未冻结」判冻结，⛔ 不解除 fail-closed）")},
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

    # ★ 发布件只放**结论级计数**（第 4 章章首「保留 68 档、剔除 56 档」由数据渲染，⛔ 不手填数字）；
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
                         ("EXCEED_CARDS", "MEMORY_UNFIT", "KNOWN_MEMORY_FAIL", "UNSUPPORTED_MODE",
                          "ALGO_BACKEND", "NATIVE_NO_CHANNEL", "MSSWIFT_NO_PARALLEL")},
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
    _enum = new.get("enumeration")
    if isinstance(_enum, dict):
        # ★ 2026-10-28（审查警告 6）：原口径宣称「script_sha256 与同提交内的脚本逐字节一致」，
        #   但实测 9926431a… 对应的是 `ef0a9f3`，而 code_version.commit=53b0c24 内是 944e2647…
        #   ⇒ 两句不能同时为真。改为**如实说明两个提交不同**：枚举脚本的版本单独记 `script_commit`
        #   （生成时 HEAD；缺省 = 本次生成所在提交），⛔ 不再宣称与 code_version.commit 同一提交。
        _enum["script_commit"] = _git_short_head()
        _enum["note"] = ("script_sha256 = 实际跑枚举的 tools/matrix/enumerate_matrix.py 的 sha256；"
                         "它与 script_commit 内的同名文件逐字节一致。"
                         "该提交与 meta.code_version.commit（本件生成基线）不是同一个提交——"
                         "枚举早于本件基线，两个提交不同属正常，不得据此推断脚本被改过。")
    _cv = new.get("code_version")
    if isinstance(_cv, dict):
        _cv["note"] = ("commit = 本件的生成基线（生成时的 HEAD），与枚举脚本版本无关；"
                       "枚举脚本身份以 meta.enumeration.script_sha256 + "
                       "meta.enumeration.script_commit 为准。")

    for drop in META_DROPS:
        _del_path(new, drop["path"])   # 不存在也不算错（幂等）

    # ★★ 2026-10-22（复核条目 15）：OPD 教师身份与教师线的**判据变更记录**。
    #   病史：数据件里 `threshold.by_method.opd.epoch_note` 与
    #   `caliber_change.caliber_change_20261019.changed` 仍写「原教师说明『SFT 之后的 27B 产物』
    #   已作废」，与人读件的新句（教师 = 27B 在 SFT / LoRA / dp=1 这一档训练完成后的产物）相反。
    #   ⇒ ⛔ 不改写历史条目（逐字留痕），而是**追加一条当日变更记录**说明：
    #     ① 教师已指定到可复现的具体档位；② 教师线已定为 ≥（数值 +10.0 pp 不变）。
    #   ★ 走生成器注入（本函数是发布 meta 的唯一出口）⇒ 数据件仍可由本脚本逐字节重建。
    new.setdefault("criteria_caliber", {}).setdefault("caliber_change", {})[
        "caliber_change_20261022"] = {
        "effective": "2026-10-22",
        "source": "2026-10-22 判据变更记录（OPD 教师身份指定到档位；教师线改用 ≥ 表达）",
        "changed": (
            "改动一 = OPD 教师身份指定到可复现的具体档位：教师 = Qwen3.8-27B 在 "
            "SFT / LoRA / dp=1 这一档训练完成后的产物（不是基座权重本身）"
            "—— 目的是让各 OPD 档的教师取值可复现。"
            "改动二 = OPD 教师线改用「≥」表达：教师 − 学生未训练模型 ≥ +10.0 pp"
            "（数值 +10.0 pp 不动，仅把「高于」改为允许相等的「≥」）。"),
        "unchanged": (
            "OPD 学生线 after − base ≥ +5.0 pp 一字未动；+10.0 pp 这一数值未动；"
            "OPD epoch = 2 未动；SFT / GRASPO / CPT 三条通过线与判定语义未动；"
            "判据条数（四条）与四个短名未动。"),
        "superseded": (
            "旧教师说明「SFT 之后的 27B 产物」现精确到档位（SFT / LoRA / dp=1）；"
            "旧教师线表述「高于学生未训练模型 10 个百分点」改由「≥」表达，数值不变。"),
    }

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
            # ★★ 2026-10-22（用户拍板 A 删 / B 删 / C 留）：**一个后端都做不了**的档直接剔除，
            #   原因码由 `deletion_reason_code()` 给（两条显式判据，⛔ 不靠排除法猜）：
            #     · NATIVE_NO_CHANNEL  = 形状本是 native 全参能吃的（tp==1 ∧ dp==1），但缺该算法通道；
            #     · MSSWIFT_NO_PARALLEL = 要靠 ms-swift 的 TP/PP，而 Megatron 通道未接线。
            #   `shape_capable` 非空 ⇒ 说明形状有人能做、只是装不下 ⇒ 走原显存原因码（⛔ 别错记成能力缺口）。
            #   判定顺序（⭐ 与 §一 的 4 / 52 分类一一对应，⛔ 不混算）：
            #     ① 有后端**装得下**、却吃不了这个形状 ⇒ MSSWIFT_NO_PARALLEL（靠 Megatron 的 TP/PP，未接线）；
            #     ② 谁都装不下、但"补齐 native 通道后能装下" ⇒ NATIVE_NO_CHANNEL（形状本是 native 全参的菜）；
            #     ③ 谁都装不下也没有复活路径 ⇒ 原显存原因码（MEMORY_UNFIT / KNOWN_MEMORY_FAIL）。
            _eligible_any = [c for c in cands if c["eligible"]]
            if _eligible_any:
                code = "MSSWIFT_NO_PARALLEL"
            else:
                hyp = [eval_backend(algo=algo, model=model, mode=mode, backend=be,
                                    shape=shape, budget=budget)
                       for be in BACKENDS if be not in ALGO_BACKEND_SUPPORT[algo]]
                _hyp_fit = [h for h in hyp if h["eligible"]]
                if _hyp_fit:
                    code = "NATIVE_NO_CHANNEL"
                else:
                    known = [c for c in cands if "evidence" in c]
                    code = "KNOWN_MEMORY_FAIL" if known else "MEMORY_UNFIT"
            counts["excluded"][code] = counts["excluded"].get(code, 0) + 1
            detail = "；".join(f"{c['backend']}: {c.get('why') or ('装得下（%.2f GiB）' % c.get('peak_gib', 0.0))}"
                               for c in cands)
            if code in ("NATIVE_NO_CHANNEL", "MSSWIFT_NO_PARALLEL"):
                detail += (f"；★ 剔除判据：{REASON_CODES[code]['desc']}")
            excluded.append({
                "key": {"algo": algo, "model": model, "mode": mode,
                        "shape": {"tp": tp, "pp": pp, "dp": dp}, "total_cards": tp * pp * dp},
                "reason_code": code,
                "if_recipe_measured": None,
                "detail": detail[:1200],
                "revive_if_implemented": [
                    {"backend": c["backend"], "eligible": c["eligible"],
                     "peak_gib": c.get("peak_gib"), "band": c.get("band"),
                     "shape_capable": c.get("shape_capable"),
                     "gap": ALGO_BACKEND_NOTE.get((algo, c["backend"]))}
                    for c in cands],
                "candidates": [
                    {"backend": c["backend"], "eligible": c["eligible"], "why": c.get("why"),
                     "peak_gib": c.get("peak_gib"), "evidence": c.get("evidence"),
                     "shape_capable": c.get("shape_capable")}
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
            excluded=excluded,
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
        if case.get("retired"):
            lines.append(f"  [RETIRED] {case['id']} {case['claim']}（{case.get('reason', '')}）")
        else:
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
