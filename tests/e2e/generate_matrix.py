#!/usr/bin/env python3
"""GRASPO 本期 54 档测试矩阵生成器（重写版，替代旧的全排列生成器）。

职责：把 `docs/capability-matrix.md` §7 的 54 档实测台账（T001–T054）逐档
翻译成「配置骨架 + 运行清单」，供后续 GPU 实验按档执行。生成物：

  - `samples/configs/matrix54/T###.yaml` —— 每档一个配置骨架
  - `samples/configs/matrix54/T###.blocked.md` —— 当前不可表达档位的阻塞说明
  - `tests/e2e/matrix54_manifest.json` —— 运行清单（44+ 字段/档，含 GPU 集合、
    验收门槛、可表达性状态），是结果收集器 `scripts/collect_results.py` 的输入
  - `tests/e2e/run_matrix54.sh` —— 批量执行脚本骨架（自带锁卡守卫与可信采样）

**口径（用户拍板，权威）**：只用 1/2/4 卡，每次最多 4 卡且仅取 GPU0-5；
4 卡首选 {0,1,2,3}；上下文长度由递增加长法实测得出、不预设档位。

**为什么是重写而不是改造旧生成器**：旧 `generate_matrix.py` 的口径已废——
数学全排列出 56 个 yaml、镜像名 `graspo:0.29.0`（不存在）、`WORLD_SIZES=[1,2,4]`
全排列、无锁卡守卫、与 54 档台账不对应。旧口径的任何残留都会把"能力上限"
测错，因此按宪法 §18.1 重写。

用法：
    python3 tests/e2e/generate_matrix.py --dry-run --assert-count 54   # 只自检不落盘
    python3 tests/e2e/generate_matrix.py                              # 生成全部产物
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "samples" / "configs" / "matrix54"
MANIFEST_PATH = PROJECT_ROOT / "tests" / "e2e" / "matrix54_manifest.json"
RUNNER_PATH = PROJECT_ROOT / "tests" / "e2e" / "run_matrix54.sh"

# ── 台账口径常量（单一真相源，§1.4）────────────────────────────────────────
CARDS: tuple[int, ...] = (1, 2, 4)
EXPECTED_TOTAL = 54
EXPECTED_BY_ALGORITHM = {"CPT": 9, "SFT": 18, "GRASPO": 18, "OPD": 9}
EXPECTED_BY_CARDS = {1: 18, 2: 18, 4: 18}
EXPECTED_BY_BACKEND = {"ms-swift": 36, "native": 18}
EXPECTED_BY_MODE = {"LoRA": 36, "全量": 18}

#: 4 卡首选 {0,1,2,3}（唯一全部位于 NUMA0）；GPU6/7 为生产卡，永不出现。
GPU_SETS: dict[int, tuple[int, ...]] = {1: (0,), 2: (0, 1), 4: (0, 1, 2, 3)}

# ── 模型权重：运行链路的一环，必须显式挂载（不能只写容器内路径就了事）──────
# **宿主机上的模型根目录属于环境信息**（§15.1/§16），不写进 tracked 文件：
# 运行时由 runner 从环境变量 ``GRASPO_MODELS_HOST_ROOT`` 取（见 .local/ 下的本地配置），
# 只读挂载到容器内 ``/models``。生成器只记录**容器内**路径；"模型从哪来"这一环
# 由 runner 的 fail-closed 前置断言守住（缺失即拒绝启动，不会掉到"数据问题"分类）。
MODELS_HOST_ROOT_ENV = "GRASPO_MODELS_HOST_ROOT"
MODELS_CONTAINER_ROOT = "/models"

#: 容器内目录名（= 宿主侧相对模型根的目录名）；显示名是用户给定的专名，不含内网信息。
_MODEL_DIR_NAMES: dict[str, str] = {"9B": "Qwen3.5-9B", "27B": "Qwen3.8-27B"}


def models_host_dir_env(model: str) -> str:
    """该档模型目录的宿主侧**可选**覆盖环境变量名。

    默认取 ``$GRASPO_MODELS_HOST_ROOT/<容器内目录名>``；仅当某档模型不在同一
    模型根下时，才用 ``GRASPO_9B_HOST_DIR`` / ``GRASPO_27B_HOST_DIR`` 单点覆盖。
    """
    return f"GRASPO_{model.upper()}_HOST_DIR"


MODELS: dict[str, dict[str, str]] = {
    size: {"name": dirname, "path": f"{MODELS_CONTAINER_ROOT}/{dirname}"}
    for size, dirname in _MODEL_DIR_NAMES.items()
}

# ── 训练数据：ELAM V5（只读确认过结构，非占位符）────────────────────────────
# 【实测确证】结构（数据根目录下）：
#   data/train.jsonl    （6378 行）
#   data/test.jsonl     （ 702 行）
#   images/*.jpg        （15954 个）
# 图像在 jsonl 里写作 `"image": "../images/xxx_left_eye.jpg"`（**相对 data 文件所在目录**），
# 由 grasp 的 resolve_messages_media_paths 以 (data_dir / path).resolve() 展开为绝对路径。
# 因此容器内约定：整棵数据目录挂到 /work/elam-v5，
#   data_dir = /work/elam-v5/data，`../images` → /work/elam-v5/images。✓
#
# **宿主机上的数据根目录属于环境信息**（§16），不写进 tracked 文件：
# 运行时由 runner 从环境变量 `GRASPO_ELAM_HOST_ROOT` 取（见 .local/ 下的本地配置）。
ELAM_HOST_ROOT_ENV = "GRASPO_ELAM_HOST_ROOT"
ELAM_CONTAINER_ROOT = "/work/elam-v5"
ELAM_TRAIN_JSONL = f"{ELAM_CONTAINER_ROOT}/data/train.jsonl"
ELAM_TEST_JSONL = f"{ELAM_CONTAINER_ROOT}/data/test.jsonl"
ELAM_SUBSET_DIR = f"{ELAM_CONTAINER_ROOT}/subsets"
ELAM_TRAIN_COUNT = 6378
ELAM_TEST_COUNT = 702
ELAM_IMAGE_COUNT = 15954

#: 每档训练子集大小：按 §6 门槛取下限即可（用户已定"尽量省资源"，不整集跑）。
#: SFT ≥100 条、RL ≥20 条（GRASPO 属 RL）。
SUBSET_SIZE_BY_ALGORITHM: dict[str, int] = {"CPT": 100, "SFT": 100, "GRASPO": 20, "OPD": 20}

#: ⚠️ 已知数据问题，**必须显式标注、不得静默忽略**（评测口径由评测链路负责，
#: 本工程只负责不掩盖）：train/test 样本 id 不重叠，但图像去重存在交集。
DATA_INTEGRITY_CAVEAT = (
    "ELAM V5 train/test 样本 id 不重叠；但**图像去重交集 752 张**（占测试图像 38.5%），"
    "且 **169/702 个测试样本的图像集是某训练样本的子集**。训练集与评测集「同源不重叠」"
    "的口径在图像层面不成立，评测结论须在对齐该口径后再下判断。"
)

INITIAL_MAX_PROMPT_LENGTH = 8192
TIMEOUT_SEC = 7200

#: 已作废口径——生成物中出现任何一条即自检失败（防"把框架 bug 固化成能力上限"）。
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "2卡96K",
    "FSDP2 权重分片",
    "95格",
    "graspo:0.29.0",
    "WORLD_SIZES",
    "DOCKER_IMAGE",
)

#: 全参档位的可表达性说明：配置字段已存在（`tuner_type: full`），但能力本身
#: 是本期的 ⚠️ 开发目标，必须实测后才能标 ✅。
FULL_MODE_NOTE = (
    "全参路径由 `tuner_type: full` 表达（native 与 ms-swift 共用同一开关）。"
    "该能力当前 ⚠️ 未验证（capability-matrix §4「全量 · native / ms-swift」），"
    "且 native 全参只支持 PP 分片（tp_size>1 或 dp_size>1 会被配置校验 fail-closed 拒绝），"
    "因此本档 native 布局固定为 pp_size=卡数、dp=tp=1。"
)

ALGORITHM_TO_TRAIN_METHOD = {"SFT": "sft", "GRASPO": "graspo"}


# ── 档位台账构造 ────────────────────────────────────────────────────────────


def _block(algorithm: str, backend: str, model: str, mode: str) -> list[dict[str, Any]]:
    """按固定顺序生成一个 (算法, 后端, 模型, 模式) 块：卡数 1 → 2 → 4。"""
    return [
        {"algorithm": algorithm, "backend": backend, "model": model, "mode": mode, "cards": c}
        for c in CARDS
    ]


def build_ledger() -> list[dict[str, Any]]:
    """按 §7 台账的逐行顺序生成 54 档（T001 → T054）。"""
    tiers: list[dict[str, Any]] = []

    # CPT 9 = 9B(LoRA,全量)×3卡 + 27B(LoRA)×3卡
    tiers += _block("CPT", "ms-swift", "9B", "LoRA")
    tiers += _block("CPT", "ms-swift", "9B", "全量")
    tiers += _block("CPT", "ms-swift", "27B", "LoRA")
    # SFT 18 = 9B(LoRA native, LoRA ms-swift, 全量 native, 全量 ms-swift) + 27B(LoRA native, LoRA ms-swift)
    tiers += _block("SFT", "native", "9B", "LoRA")
    tiers += _block("SFT", "ms-swift", "9B", "LoRA")
    tiers += _block("SFT", "native", "9B", "全量")
    tiers += _block("SFT", "ms-swift", "9B", "全量")
    tiers += _block("SFT", "native", "27B", "LoRA")
    tiers += _block("SFT", "ms-swift", "27B", "LoRA")
    # GRASPO 18 = 与 SFT 同构
    tiers += _block("GRASPO", "native", "9B", "LoRA")
    tiers += _block("GRASPO", "ms-swift", "9B", "LoRA")
    tiers += _block("GRASPO", "native", "9B", "全量")
    tiers += _block("GRASPO", "ms-swift", "9B", "全量")
    tiers += _block("GRASPO", "native", "27B", "LoRA")
    tiers += _block("GRASPO", "ms-swift", "27B", "LoRA")
    # OPD 9 = 与 CPT 同构
    tiers += _block("OPD", "ms-swift", "9B", "LoRA")
    tiers += _block("OPD", "ms-swift", "9B", "全量")
    tiers += _block("OPD", "ms-swift", "27B", "LoRA")

    for index, tier in enumerate(tiers, start=1):
        tier["tier_id"] = f"T{index:03d}"
        tier["gpus"] = list(GPU_SETS[tier["cards"]])
    return tiers


def classify_expressibility(tier: dict[str, Any]) -> tuple[str, str | None]:
    """判定该档当前能否被配置模型表达，返回 (status, reason)。

    - ``ready``：配置可表达，对应能力格当前为 ✅ / 常规路径（本期仍需实测）；
    - ``unverified``：配置可表达，但对应能力格当前 ⚠️ 未验证；
    - ``blocked``：配置模型/映射当前**无法表达**；或虽可表达，但按当前口径
      估算给不出可靠配方（如 native 全参 1 卡，见 ``BLOCKED_REASONS`` 的
      诚实表述——那是**保守判定，不是已证的必然 OOM**）。
    """
    algorithm = str(tier["algorithm"])
    if algorithm in {"CPT", "OPD"}:
        return "blocked", BLOCKED_REASONS[algorithm.lower()]
    # native 全参只有 PP 分片这一种手段：1 卡 ⇒ 无分片（无 offload）⇒ 保守判 blocked。
    # 理由不写「必然 OOM」——见 BLOCKED_REASONS["native_full_1card"] 的诚实表述。
    if tier["mode"] == "全量" and tier["backend"] == "native" and int(tier["cards"]) == 1:
        return "blocked", BLOCKED_REASONS["native_full_1card"]
    if tier["mode"] == "全量":
        return "unverified", FULL_MODE_NOTE
    return "ready", None


# ── 配置骨架 ────────────────────────────────────────────────────────────────


def _learning_rate(algorithm: str) -> float:
    return 5.0e-5 if algorithm == "SFT" else 5.0e-6


def subset_path(tier_id: str) -> str:
    """该档训练子集在容器内的固定路径（runner 负责按 manifest 生成）。"""
    return f"{ELAM_SUBSET_DIR}/{tier_id}.jsonl"


def subset_size(algorithm: str) -> int:
    """该档训练子集大小（§6 门槛下限；省资源不整集跑）。"""
    return SUBSET_SIZE_BY_ALGORITHM[algorithm]


# ── 显存可行性模型（生成期断言，§2.3 边界校验即防呆）────────────────────────
# **为什么要有这个**：配置能生成 ≠ 能跑。上一轮的坑就是"安静地产出一个跑起来才爆的
# 配置"，把错误推迟到花掉 GPU 时间之后才暴露。这里在生成期用保守算式估算每卡需求，
# 超过单卡可用显存就直接拒绝生成并报原因。
#
# 算式（保守，单位 bytes/param；9B=9.0e9，27B=27.0e9）：
#   - bf16 权重          2  （两个后端一致）
#   - bf16 梯度          2  （两个后端一致）
#   - 优化器态(m+v)      fp32=8 / bf16=4 —— **按后端分别假设，见下**
#   - LoRA：基座冻结 ⇒ 只算权重 2 B/param（适配器参数量忽略，远小于基座）
#   - ZeRO-2：梯度 + 优化器按卡数摊 ⇒ 2 + (2+优化器)/N
#   - ZeRO-2 + CPU offload：优化器态在宿主内存 ⇒ 2 + 2（梯度全量）
#   - AutoTP=T：权重再按 T 摊 ⇒ 权重 2/T
#   - native 全参：只支持 PP 分片 ⇒ 权重/梯度/优化器都按 PP=卡数 摊 ⇒ k/N
# 激活值用一个名义常量（上下文长度由递增加长法实测得出、不预设档位）。
PARAMS_BILLION = {"9B": 9.0, "27B": 27.0}
_BF16 = 2.0
_GRAD = 2.0

#: ── 常驻口径「单一真相源」（§1.4）：**按后端分别假设，不混用一个 k** ──────────
#: 此前 native 与 ms-swift 共用 k=12（fp32 优化器态），于是 native 1 卡被算成
#: 112.6 GiB ⇒ 写成「必然 OOM」；而 native 侧的实际实现口径是 k=8（bf16 Adam，
#: 见 task-i2-fullparam §3.1：`torch.optim.AdamW` 用 `zeros_like(param)` 建 m/v，
#: dtype 跟随 bf16 参数 ⇒ 常数 70.1 GiB、余量 ≈10 GiB）。两处口径不一致，且
#: **两侧的 k 都是假设、不是实测**。本生成器现在的统一口径是：
#:   - native 全参  : k = 8  B/param（bf16 Adam 假设，来源 task-i2-fullparam §3.1）
#:   - ms-swift 全参: k = 12 B/param（DeepSpeed fp32 优化器态假设，保守）
#:   - LoRA         : 2 B/param（基座冻结）
#: 每个假设都写进 manifest 的 `feasibility_model.assumptions`，便于事后核账。
_ADAM_FP32 = 8.0
_ADAM_BF16 = 4.0
#: native 侧实际实现口径（bf16 矩估计）。**这是假设**：若工程上改为 fp32 矩估计
#: （k=12），native 1 卡的估算会升到 105 GiB+（算式级必爆）。
NATIVE_OPTIMIZER_BYTES_PER_PARAM = _BF16 + _GRAD + _ADAM_BF16
#: ms-swift / DeepSpeed 侧保守口径（fp32 优化器态）。
MSSWIFT_OPTIMIZER_BYTES_PER_PARAM = _BF16 + _GRAD + _ADAM_FP32
ACTIVATION_ALLOWANCE_GIB = 12.0
#: 单卡 80 GiB，留出余量后允许的上限。
CARD_BUDGET_GIB = 78.0

NATIVE_ACCOUNTING_ASSUMPTION = (
    "native 全参 k=8 B/param —— 假设 `torch.optim.AdamW` 的 m/v 矩估计跟随 bf16 参数"
    "（`zeros_like(param)`，无 fp32 master）。来源：task-i2-fullparam §3.1。"
    "**该假设未经上机实测**：若改为 fp32 矩估计（k=12）则 1 卡估算升至 105 GiB+。"
)
MSSWIFT_ACCOUNTING_ASSUMPTION = (
    "ms-swift/DeepSpeed 全参 k=12 B/param —— 保守假设 fp32 优化器态（m+v = 8）。"
    "**该假设未经上机实测，方向偏保守。**"
)


def _native_static_gib(params: float) -> float:
    """native 全参 1 卡（PP=1，无分片）的常驻显存估算（GiB）。"""
    return params * NATIVE_OPTIMIZER_BYTES_PER_PARAM / (1024.0**3)


#: 当前不可表达的原因（每条都指向能力矩阵里的 ⚠️ 格）。
#: native 1 卡那条**不写「必然 OOM」**——算式依赖未经实测的 bf16 Adam 假设，
#: 且余量只有约 10 GiB，只能如实说「判定不可靠、本轮不承诺」。
BLOCKED_REASONS: dict[str, str] = {
    "cpt": (
        "配置模型不支持 CPT：`train_method` 的 Literal 只有 {'graspo','sft'}，"
        "ms-swift 映射也没有 pretrain 通道（capability-matrix §4「CPT · ms-swift」⚠️）。"
    ),
    "opd": (
        "配置模型不支持 OPD：`train_method` 的 Literal 只有 {'graspo','sft'}，"
        "没有 on-policy 蒸馏通道（capability-matrix §4「OPD · ms-swift」⚠️）。"
    ),
    "native_full_1card": (
        "native 全参当前只支持 PP 分片、且无 offload ⇒ 1 卡无任何分片手段。"
        f"按 native 实际实现口径（k={NATIVE_OPTIMIZER_BYTES_PER_PARAM:g} B/param，"
        "bf16 Adam 假设）9B 常驻 ≈"
        f"{_native_static_gib(PARAMS_BILLION['9B'] * 1.0e9):.1f} GiB，"
        f"加激活/其他预算 {ACTIVATION_ALLOWANCE_GIB:g} GiB ⇒ ≈"
        f"{_native_static_gib(PARAMS_BILLION['9B'] * 1.0e9) + ACTIVATION_ALLOWANCE_GIB:.1f} GiB"
        f" > 预算 {CARD_BUDGET_GIB:g} GiB，余量仅约 10 GiB。"
        "**该结论依赖 bf16 Adam 假设、未经上机实测，判定不可靠 ⇒ 本轮不承诺**；"
        "若工程上改为 fp32 矩估计则算式级必爆。"
        "需 native 侧实现全参 offload、改用 ≥2 卡走 PP、或先用 1 卡边界取证档实测后再定。"
    ),
}


def ms_swift_full_recipe(cards: int) -> dict[str, Any]:
    """ms-swift 全参的可行配方（≤4 卡；依据全量入口工作包的方案）。

    - 1 / 2 卡 → ``zero2_offload``（优化器态换出到 CPU，否则单卡常驻超预算）
    - 4 卡 → ``zero2`` + ``deepspeed_autotp_size: 4``（ZeRO-2 + AutoTP 权重分片）
    """
    if cards <= 2:
        return {"deepspeed": "zero2_offload"}
    return {"deepspeed": "zero2", "deepspeed_autotp_size": 4}


def estimate_config_per_card_gib(
    tier: dict[str, Any], config: dict[str, Any]
) -> tuple[float, str]:
    """按**实际生成的配置**估算每卡显存需求，返回 (GiB, 算式依据)。

    校验的是"我们真正要产出的配置"而不是"我们以为的配方"——否则生成器漏配
    deepspeed 时断言照样会通过（这正是上一轮的坑）。
    """
    params = PARAMS_BILLION[str(tier["model"])] * 1.0e9
    cards = int(tier["cards"])
    backend = str(tier["backend"])
    mode = str(tier["mode"])

    if mode == "LoRA":
        bytes_per_param = _BF16
        basis = f"LoRA：基座冻结 ⇒ 权重 {_BF16:g} B/param"
    elif backend == "native":
        pp_size = int(config.get("native", {}).get("pp_size", 1) or 1)
        bytes_per_param = NATIVE_OPTIMIZER_BYTES_PER_PARAM / pp_size
        basis = (
            f"native 全参 PP 分片（k={NATIVE_OPTIMIZER_BYTES_PER_PARAM:g} B/param，bf16 Adam 假设）："
            f"({_BF16:g}+{_GRAD:g}+{_ADAM_BF16:g})/{pp_size} B/param"
            + ("" if pp_size > 1 else "（无分片）")
        )
    else:
        msswift = config.get("msswift", {})
        deepspeed = msswift.get("deepspeed")
        autotp = float(msswift.get("deepspeed_autotp_size") or 1)
        if deepspeed == "zero2_offload":
            weight = _BF16 / autotp
            sharded = _GRAD  # 优化器态 offload 到 CPU，梯度仍全量
            basis = (
                f"ms-swift 全参 zero2_offload：权重 {_BF16:g}/{autotp:g}"
                f" + 梯度 {_GRAD:g} B/param（优化器态在 CPU）"
            )
        elif deepspeed in {"zero2", "zero3", "zero2_offload", "zero3_offload"}:
            weight = _BF16 / autotp
            sharded = (_GRAD + _ADAM_FP32) / cards
            basis = (
                f"ms-swift 全参 {deepspeed}+AutoTP{autotp:g}：权重 {weight:g}"
                f" + (梯度+优化器) {_GRAD + _ADAM_FP32:g}/{cards} B/param"
            )
        else:
            # 没配任何分片/offload：全参 ⇒ 权重+梯度+优化器全量压在一张卡上。
            weight = _BF16
            sharded = _GRAD + _ADAM_FP32
            basis = (
                f"ms-swift 全参**未配 deepspeed/offload**：权重 {_BF16:g} + 梯度 {_GRAD:g}"
                f" + 优化器 {_ADAM_FP32:g} = {MSSWIFT_OPTIMIZER_BYTES_PER_PARAM:g} B/param 全量压单卡"
            )
        bytes_per_param = weight + sharded

    gib = params * bytes_per_param / (1024.0**3) + ACTIVATION_ALLOWANCE_GIB
    return gib, f"{basis}；激活/其他预算 {ACTIVATION_ALLOWANCE_GIB:g} GiB"


def estimate_per_card_gib(tier: dict[str, Any]) -> tuple[float, str]:
    """按该档的规范配置估算每卡显存需求。"""
    return estimate_config_per_card_gib(tier, build_config(tier))


def feasibility(tier: dict[str, Any]) -> tuple[bool, str]:
    """该档是否显存可行（保守估算 ≤ 单卡预算）。

    ``ok=False`` **不等于"已证的必然 OOM"**：算式含未实测假设（各后端优化器态精度），
    故措辞为"估算超预算 ⇒ 判定不可靠"，而不是"必然 OOM"。
    """
    gib, basis = estimate_per_card_gib(tier)
    ok = gib <= CARD_BUDGET_GIB
    verdict = "可行" if ok else "估算超预算 ⇒ 判定不可靠（假设未实测）"
    return ok, f"估算每卡 {gib:.1f} GiB（预算 {CARD_BUDGET_GIB:g} GiB）⇒ {verdict}：{basis}"


def feasibility_for(tier: dict[str, Any]) -> tuple[bool | None, str]:
    """给 blocked 档返回 ``None``（不估配方），其余走正常估算。"""
    status, _ = classify_expressibility(tier)
    if status == "blocked":
        return None, "blocked：当前给不出可行配方（原因见 status_reason）"
    return feasibility(tier)


def build_config(tier: dict[str, Any]) -> dict[str, Any]:
    """构造一档的配置骨架（只使用 schema 中确实存在的字段）。"""
    model = MODELS[str(tier["model"])]
    tier_id = str(tier["tier_id"])
    cards = int(tier["cards"])
    backend = str(tier["backend"])
    algorithm = str(tier["algorithm"])

    config: dict[str, Any] = {
        "train_method": ALGORITHM_TO_TRAIN_METHOD[algorithm],
        "backend": "msswift" if backend == "ms-swift" else "native",
        "tuner_type": "full" if tier["mode"] == "全量" else "lora",
        "model": {
            "model_path": model["path"],
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "data": {
            # 本档训练子集：由 run_matrix54.sh 从 ELAM V5 data/train.jsonl 取前 N 条生成。
            # 路径固定 ⇒ config 仍是产物的唯一描述（§10.1）；N 记录在 manifest 里。
            # 图像相对路径 `../images/...` 由该路径的父目录解析到 /work/elam-v5/images。
            "train_path": subset_path(tier_id),
            "max_prompt_length": INITIAL_MAX_PROMPT_LENGTH,
        },
        "training": {
            "output_dir": f"/out/{tier_id}",
            "run_name": tier_id,
            "overwrite_output_dir": True,
            "seed": 42,
            "max_epochs": 1,
            "learning_rate": _learning_rate(algorithm),
            "gradient_accumulation_micro_batches": 1,
            "save_checkpoint_every_epoch": False,
            "save_steps": 1,
        },
    }

    if tier["mode"] == "LoRA":
        # LoRA 参数档位：r=8/alpha=16 与既有 ms-swift 冒烟口径一致（可再调）。
        config["lora"] = {
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_preset": "language_safe",
        }
    # 全参档位不写 lora 段：`tuner_type: full` 与 lora.adapter_path 互斥，
    # 配置校验会在加载时拒绝两者并存（schema.validate_tuner_type_combination）。

    if backend == "native":
        if tier["mode"] == "全量" and cards > 1:
            # native 全参只支持 PP 分片；DP/TP 会被校验 fail-closed 拒绝。
            config["native"] = {
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": cards,
                "micro_batch_size": 1,
            }
        else:
            config["native"] = {
                "tp_size": 1,
                "dp_size": cards,
                "pp_size": 1,
                "micro_batch_size": 1,
            }
    else:
        config["msswift"] = {
            "nproc_per_node": cards,
            "per_device_train_batch_size": 1,
            "sequence_parallel_size": 1,
            "use_vllm": False,
        }
        if tier["mode"] == "全量":
            # 全参必须给可行配方，否则 1 卡估算超预算（生成期断言会兜底）。
            config["msswift"].update(ms_swift_full_recipe(cards))

    if algorithm == "GRASPO":
        config["reward"] = {"kind": "graspo"}
    return config


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def render_config_yaml(tier: dict[str, Any], status: str, reason: str | None) -> str:
    """把配置 dict 渲染成带口径注释的 YAML 文本。"""
    tier_id = str(tier["tier_id"])
    header = [
        f"# {tier_id} | {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡",
        "# 生成器: tests/e2e/generate_matrix.py（手改无效，请改生成器）",
        "# 口径: 只用 1/2/4 卡；GPU 集合取自 {0,1,2,3}；上下文长度由递增加长法实测得出，不预设档位。",
        f"# 数据: ELAM V5 训练子集前 {subset_size(str(tier['algorithm']))} 条（宿主数据根目录经 "
        f"{ELAM_HOST_ROOT_ENV} 注入，见 .local/）；子集由 run_matrix54.sh 生成，图像经 ../images 解析。",
        f"# ⚠️ 数据口径告警: {DATA_INTEGRITY_CAVEAT}",
        f"# 可表达性: {status}",
        f"# 显存可行性: {feasibility(tier)[1]}",
    ]
    if reason:
        header.append(f"# 注意: {reason}")
    lines: list[str] = list(header)

    config = build_config(tier)
    for section, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{section}:")
            for key, item in value.items():
                if key == "max_prompt_length":
                    lines.append(f"  {key}: {_yaml_scalar(item)}")
                    continue
                lines.append(f"  {key}: {_yaml_scalar(item)}")
        else:
            lines.append(f"{section}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def render_blocked_stub(tier: dict[str, Any], reason: str) -> str:
    """不可表达档位的阻塞说明（不产出一个"看似能跑"的错误配置）。"""
    return (
        f"# {tier['tier_id']} 当前不可执行（blocked）\n\n"
        f"档位: {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡\n\n"
        f"阻塞原因: {reason}\n\n"
        "处置: 该档所需能力是本期 14 个 ⚠️ 能力格的开发目标；能力落地并通过验证后，\n"
        "     由 tests/e2e/generate_matrix.py 重新生成本档配置再执行。\n"
    )


# ── 运行清单 ────────────────────────────────────────────────────────────────


def build_manifest(tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """构造运行清单（结果收集器的输入）。"""
    entries = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        entry: dict[str, Any] = {
            "tier_id": tier["tier_id"],
            "model": tier["model"],
            "model_name": MODELS[str(tier["model"])]["name"],
            "model_path": MODELS[str(tier["model"])]["path"],
            "algorithm": tier["algorithm"],
            "mode": tier["mode"],
            "backend": tier["backend"],
            "cards": tier["cards"],
            "gpus": tier["gpus"],
            "status": status,
            #: 宿主侧模型目录的两级来源（环境信息，值只在 .local/）：
            #: 默认 ``$GRASPO_MODELS_HOST_ROOT/<容器内目录名>``，逐档可用
            #: ``GRASPO_9B_HOST_DIR`` / ``GRASPO_27B_HOST_DIR`` 单点覆盖。
            "model_host_root_env_var": MODELS_HOST_ROOT_ENV,
            "model_host_dir_env_var": models_host_dir_env(str(tier["model"])),
            "model_host_dir_default": (
                f"<{MODELS_HOST_ROOT_ENV}>/{_MODEL_DIR_NAMES[str(tier['model'])]}"
            ),
        }
        if reason is not None:
            entry["status_reason"] = reason
        if status == "blocked":
            entry["config"] = None
            entry["blocked_doc"] = f"samples/configs/matrix54/{tier['tier_id']}.blocked.md"
        else:
            entry["config"] = f"samples/configs/matrix54/{tier['tier_id']}.yaml"
        entry["data"] = {
            "train_path": subset_path(str(tier["tier_id"])),
            "subset_size": subset_size(str(tier["algorithm"])),
            "full_train_jsonl": ELAM_TRAIN_JSONL,
            "caveat": DATA_INTEGRITY_CAVEAT,
        }
        feasible, estimate = feasibility_for(tier)
        entry["feasibility"] = {
            "feasible": feasible,
            "estimate": estimate,
            "recipe": (
                {"native": build_config(tier).get("native")}
                if tier["backend"] == "native"
                else {"msswift": build_config(tier).get("msswift")}
            )
            if status != "blocked"
            else None,
        }
        entry["acceptance"] = {
            "formal_gate": {
                "min_epochs": 1,
                "min_optimizer_steps": 5,
                "min_train_samples": 20 if tier["algorithm"] in {"GRASPO", "OPD"} else 100,
            },
            "criteria": ["A1", "A2", "A3", "A4", "A5", "A6"],
        }
        entries.append(entry)

    return {
        "schema_version": 1,
        "source": "docs/capability-matrix.md §6/§7",
        "counts": {
            "total": len(entries),
            "by_algorithm": _count_by(entries, "algorithm"),
            "by_cards": {str(k): v for k, v in sorted(_count_by_cards(entries).items())},
            "by_backend": _count_by(entries, "backend"),
            "by_mode": _count_by(entries, "mode"),
            "by_status": _count_by(entries, "status"),
        },
        "runtime": {
            "image": "graspo-msswift:4.5.3",
            "timeout_sec": TIMEOUT_SEC,
            "initial_max_prompt_length": INITIAL_MAX_PROMPT_LENGTH,
        },
        "models": {
            "container_root": MODELS_CONTAINER_ROOT,
            "host_root_env_var": MODELS_HOST_ROOT_ENV,
            "host_root_note": (
                "宿主模型根目录属环境信息（§15.1/§16），由运行时环境变量注入，"
                "不入 tracked 文件；runner 只读挂载到容器内 " + MODELS_CONTAINER_ROOT + "。"
                "缺失时 runner fail-closed 拒绝启动（运行链路错误，不是数据问题）。"
            ),
            "overrides": {
                size: {
                    "container_path": MODELS[size]["path"],
                    "host_dir_env_var": models_host_dir_env(size),
                    "host_dir_default": f"<{MODELS_HOST_ROOT_ENV}>/{_MODEL_DIR_NAMES[size]}",
                }
                for size in MODELS
            },
            "mount_in_runner": (
                f'-v "$MODELS_ROOT:{MODELS_CONTAINER_ROOT}:ro"（container_root 只读挂载）'
            ),
        },
        "feasibility_model": {
            "card_budget_gib": CARD_BUDGET_GIB,
            "activation_allowance_gib": ACTIVATION_ALLOWANCE_GIB,
            "bf16_param_bytes": _BF16,
            "grad_bytes": _GRAD,
            "adam_bytes": {
                "native": _ADAM_BF16,
                "ms-swift": _ADAM_FP32,
                "note": (
                    "按后端分别假设（统一口径，§1.4）：native k=8 B/param（bf16 矩估计）；"
                    "ms-swift k=12 B/param（DeepSpeed fp32 优化器态，保守）。"
                ),
            },
            "bytes_per_param_by_backend": {
                "native_full": NATIVE_OPTIMIZER_BYTES_PER_PARAM,
                "msswift_full": MSSWIFT_OPTIMIZER_BYTES_PER_PARAM,
                "lora": _BF16,
            },
            "assumptions": {
                "native_full": NATIVE_ACCOUNTING_ASSUMPTION,
                "msswift_full": MSSWIFT_ACCOUNTING_ASSUMPTION,
                "measured": False,
                "note": (
                    "两侧的优化器态精度都是**假设**、不是实测值；因此「估算超预算」只能表述为"
                    "『判定不可靠』，**不得写成『必然 OOM』**。"
                ),
            },
            "rules": [
                "LoRA：基座冻结 ⇒ 权重 2 B/param",
                "ms-swift 全参 zero2_offload（1/2 卡）：权重 2 + 梯度 2（优化器态在 CPU）",
                "ms-swift 全参 zero2+AutoTP4（4 卡）：权重 2/4 + (梯度+优化器) 10/4",
                "native 全参：仅 PP 分片 ⇒ k/卡数（k=8，bf16 Adam 假设）；"
                "1 卡无分片 ⇒ 估算超预算、判定不可靠（标 blocked，理由为诚实表述而非『必然 OOM』）",
            ],
            "generation_gate": (
                "assert_memory_feasible：可执行档估算超预算即拒绝生成"
                "（不允许安静产出不可靠配置）"
            ),
        },
        "data": {
            "source": "ELAM V5 balanced（与开发机副本 md5 一致）",
            "host_root_env_var": ELAM_HOST_ROOT_ENV,
            "host_root_note": "宿主数据根目录属环境信息（§16），由运行时环境变量注入，不入 tracked 文件",
            "container_root": ELAM_CONTAINER_ROOT,
            "train_jsonl": ELAM_TRAIN_JSONL,
            "test_jsonl": ELAM_TEST_JSONL,
            "subset_dir": ELAM_SUBSET_DIR,
            "counts": {
                "train": ELAM_TRAIN_COUNT,
                "test": ELAM_TEST_COUNT,
                "images": ELAM_IMAGE_COUNT,
            },
            "image_reference_style": '"image": "../images/<name>.jpg"（相对 data 文件父目录）',
            "subset_size_by_algorithm": dict(SUBSET_SIZE_BY_ALGORITHM),
            "subset_rationale": "按 §6 门槛取下限（SFT ≥100 / RL ≥20），不整集跑（省资源）",
            "integrity_caveat": DATA_INTEGRITY_CAVEAT,
        },
        "tiers": entries,
    }


def _count_by(entries: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry[field])] = counts.get(str(entry[field]), 0) + 1
    return counts


def _count_by_cards(entries: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for entry in entries:
        cards = int(entry["cards"])
        counts[cards] = counts.get(cards, 0) + 1
    return counts


# ── 自检 ────────────────────────────────────────────────────────────────────


def assert_ledger(tiers: list[dict[str, Any]], expected_total: int) -> None:
    """结构自检：逐档对应台账，计数与 §6/§7 完全一致。"""
    assert len(tiers) == expected_total, f"expected {expected_total} tiers, got {len(tiers)}"
    ids = [tier["tier_id"] for tier in tiers]
    expected_ids = [f"T{index:03d}" for index in range(1, expected_total + 1)]
    assert ids == expected_ids, "tier ids must be contiguous T001..T{expected_total}"

    by_algorithm = _count_by(tiers, "algorithm")
    assert by_algorithm == EXPECTED_BY_ALGORITHM, f"algorithm counts: {by_algorithm}"
    assert _count_by_cards(tiers) == EXPECTED_BY_CARDS, "card counts mismatch"
    assert _count_by(tiers, "backend") == EXPECTED_BY_BACKEND, "backend counts mismatch"
    assert _count_by(tiers, "mode") == EXPECTED_BY_MODE, "mode counts mismatch"

    # 逐档指纹：抽 5 个关键档位，锁死 §7 的行序（防止"数量对、顺序错"）。
    fingerprints = {
        "T001": ("9B", "CPT", "LoRA", "ms-swift", 1),
        "T010": ("9B", "SFT", "LoRA", "native", 1),
        "T021": ("9B", "SFT", "全量", "ms-swift", 4),
        "T028": ("9B", "GRASPO", "LoRA", "native", 1),
        "T046": ("9B", "OPD", "LoRA", "ms-swift", 1),
        "T054": ("27B", "OPD", "LoRA", "ms-swift", 4),
    }
    by_id = {tier["tier_id"]: tier for tier in tiers}
    for tier_id, expected in fingerprints.items():
        tier = by_id[tier_id]
        actual = (
            tier["model"],
            tier["algorithm"],
            tier["mode"],
            tier["backend"],
            tier["cards"],
        )
        assert actual == expected, f"{tier_id}: expected {expected}, got {actual}"

    # GPU 集合永远 ⊆ {0,1,2,3} ⊆ 允许集合，且 ≤ 4 卡。
    for tier in tiers:
        gpus = tier["gpus"]
        assert len(gpus) == tier["cards"], f"{tier['tier_id']}: gpu/card mismatch"
        assert max(gpus) <= 3 and len(gpus) <= 4, f"{tier['tier_id']}: unsafe gpus {gpus}"


def assert_no_legacy_terms(rendered: list[tuple[str, str]]) -> None:
    """生成物中不得出现已作废口径；也不得有顶层 `r:` 键。"""
    for name, text in rendered:
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in text, f"{name}: forbidden legacy term {pattern!r}"
        for line in text.splitlines():
            assert not line.startswith("r:"), f"{name}: illegal top-level `r:` key"


def assert_expressibility(tiers: list[dict[str, Any]]) -> dict[str, int]:
    """统计可表达性，并断言各状态之和等于总数。"""
    counts: dict[str, int] = {"ready": 0, "unverified": 0, "blocked": 0}
    for tier in tiers:
        status, _ = classify_expressibility(tier)
        counts[status] += 1
    assert sum(counts.values()) == EXPECTED_TOTAL, counts
    return counts


# ── 生成期显存可行性断言（§2.3 边界校验即防呆）──────────────────────────────


def assert_memory_feasible(tiers: list[dict[str, Any]]) -> dict[str, tuple[bool, str]]:
    """对每个**可执行**档做保守显存估算；任一档估算超预算即**拒绝生成**。

    这是本轮补齐的关键防线：配置能生成 ≠ 能跑。宁可在生成期失败并报出算式，
    也不要在花掉 GPU 时间之后才发现不可靠。blocked 档不参与断言（已知给不出配方，
    由 ``classify_expressibility`` 显式说明原因）。
    """
    verdicts: dict[str, tuple[bool, str]] = {}
    violations: list[str] = []
    for tier in tiers:
        status, _ = classify_expressibility(tier)
        if status == "blocked":
            continue
        # 校验**实际生成的配置**（不是"我们以为的配方"）。
        gib, detail = estimate_config_per_card_gib(tier, build_config(tier))
        ok = gib <= CARD_BUDGET_GIB
        verdicts[str(tier["tier_id"])] = (ok, detail)
        if not ok:
            violations.append(f"{tier['tier_id']} ({tier['algorithm']}/{tier['mode']}/"
                              f"{tier['backend']}/{tier['cards']}卡): {detail}")
    if violations:
        raise AssertionError(
            "生成期显存可行性断言失败——拒绝生成估算超预算（判定不可靠）的配置：\n  - "
            + "\n  - ".join(violations)
        )
    return verdicts


def feasibility_table(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个档位的配方 + 可行性自检结果（写进 manifest / 报告）。"""
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        ok, detail = feasibility_for(tier)
        config = build_config(tier) if status != "blocked" else {}
        recipe: dict[str, Any] = {}
        if status != "blocked":
            if tier["backend"] == "native":
                recipe = {"native": config.get("native")}
            else:
                recipe = {"msswift": config.get("msswift")}
        rows.append(
            {
                "tier_id": tier["tier_id"],
                "algorithm": tier["algorithm"],
                "mode": tier["mode"],
                "backend": tier["backend"],
                "cards": tier["cards"],
                "status": status,
                "recipe": recipe,
                "feasible": ok,
                "estimate": detail,
                "status_reason": reason,
            }
        )
    return rows


# ── 运行脚本骨架 ────────────────────────────────────────────────────────────


def render_runner() -> str:
    """生成批量执行脚本骨架（锁卡守卫 + 可信采样 + 数据子集 + 逐档记录）。"""
    return f"""#!/bin/bash
# GRASPO 54 档批量执行骨架 —— 由 tests/e2e/generate_matrix.py 生成（手改无效）。
#
# 用法: bash tests/e2e/run_matrix54.sh <T###> [--dry-run]
#   - 每档：宿主侧锁卡守卫 → **模型挂载前置断言** → 生成训练子集 → docker run
#           （单一路径锁卡，只认 NVIDIA_VISIBLE_DEVICES）→ 容器内可信采样 + torchrun 训练
#   - 产物落 <RUN_ROOT>/<T###>/：exit_code, stdout.log, gpu/, subsets/, <T###>/（训练输出）
#   - **必须先导出宿主数据根目录**：export {ELAM_HOST_ROOT_ENV}=<宿主 ELAM V5 数据根目录>
#   - **必须先导出宿主模型根目录**：export {MODELS_HOST_ROOT_ENV}=<宿主模型根目录>
#     该目录只读挂载到容器内 {MODELS_CONTAINER_ROOT}（模型**不在镜像里，必须挂**）。
#     个别档模型不在同一根下时，可用 `GRASPO_9B_HOST_DIR` / `GRASPO_27B_HOST_DIR` 单点覆盖。
#     （上面两者都是环境信息，见 .local/ 本地配置；不写进 tracked 文件，宪法 §16）
#
# 数据: ELAM V5 数据根目录挂到 {ELAM_CONTAINER_ROOT}；
#       每档按 §6 门槛取下限生成训练子集 {ELAM_SUBSET_DIR}/<T###>.jsonl（省资源，不整集跑）。
# 模型: 每档配置里的 `model.model_path` 都指向容器内 {MODELS_CONTAINER_ROOT}/<模型目录名>；
#       runner 从 manifest 读出该路径并把它解析回宿主路径做存在性断言。
# ⚠️ 数据口径告警: {DATA_INTEGRITY_CAVEAT}
#
# 红线：只用 GPU0-5、每次最多 4 卡、GPU6/7 永不触碰。本脚本不自动执行矩阵。
set -uo pipefail

TIER="${{1:?usage: run_matrix54.sh <T###> [--dry-run]}}"
MODE="${{2:-run}}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${{GRASPO_IMAGE:-graspo-msswift:4.5.3}}"
RUN_ROOT="${{RUN_ROOT:-$ROOT_DIR/.local/matrix54-runs}}"
ELAM_HOST="${{{ELAM_HOST_ROOT_ENV}:-}}"
MODELS_ROOT="${{{MODELS_HOST_ROOT_ENV}:-}}"
MANIFEST="$ROOT_DIR/tests/e2e/matrix54_manifest.json"
PYBIN="${{PYTHON:-python3}}"
CONTAINER_PY="${{CONTAINER_PYTHON:-python}}"

# 0) 从运行清单取出该档的 GPU / 配置 / 卡数 / 子集大小 / 容器内模型路径。
# 交接格式是**单行 JSON**：旧的"read 多个变量 < <(python …)"在 stdout 是管道时
# 会受 Python 块缓冲影响，字段可能错位（潜在隐患，本轮改为显式 JSON 交接）。
TIER_JSON=$("$PYBIN" - "$MANIFEST" "$TIER" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
tier = next(t for t in manifest["tiers"] if t["tier_id"] == sys.argv[2])
data = tier.get("data") or {{}}
print(json.dumps({{
    "gpus": ",".join(str(g) for g in tier["gpus"]),
    "config": tier.get("config"),
    "status": tier["status"],
    "cards": tier["cards"],
    "subset_size": data.get("subset_size") or 0,
    "train_path": data.get("train_path") or "-",
    "model_path": tier.get("model_path") or "-",
    "model_env": tier.get("model_host_dir_env_var") or "-",
}}, ensure_ascii=False))
PY
) || exit 5

read -r GPUS CONFIG_STATUS NPROC SUBSET TRAINPATH MODEL_PATH MODEL_ENV < <("$PYBIN" - "$TIER_JSON" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
print(
    data["gpus"],
    (data["config"] or "-") + "|" + data["status"],
    data["cards"],
    data["subset_size"],
    data["train_path"],
    data["model_path"],
    data["model_env"],
)
PY
)
CONFIG="${{CONFIG_STATUS%%|*}}"
TIER_STATUS="${{CONFIG_STATUS#*|}}"

# 0a) **不可表达（blocked）档必须先于一切放行路径被拒**，包括 --dry-run。
# 顺序即语义：dry-run 是"预检"入口，预检把 blocked 档报成 rc=0 会让批处理
# 把不可跑的档当成可跑（历史缺陷 F-6）。因此这条检查排在 dry-run 分支**之前**。
if [ "$CONFIG" = "-" ]; then
    echo "FATAL: $TIER 当前不可表达（blocked），见 samples/configs/matrix54/$TIER.blocked.md" >&2
    exit 2
fi

# 0b) 环境前置：数据根与模型根都必须显式给出（两者都是环境信息，见 .local/）。
if [ -z "$ELAM_HOST" ]; then
    echo "FATAL(runtime-link): 数据未挂载 —— 请先 export {ELAM_HOST_ROOT_ENV}=<宿主 ELAM V5 数据根目录>" >&2
    exit 3
fi
if [ -z "$MODELS_ROOT" ]; then
    echo "FATAL(runtime-link): 模型未挂载 —— 请先 export {MODELS_HOST_ROOT_ENV}=<宿主模型根目录>" >&2
    echo "  容器内 {MODELS_CONTAINER_ROOT} 的宿主来源；模型不在镜像里，必须挂载。" >&2
    exit 4
fi

# 2) 宿主侧锁卡守卫（fail-closed；与容器内 train_worker 是同一实现）
"$PYBIN" "$ROOT_DIR/scripts/gpu_lock_guard.py" --visible "$GPUS" || exit 1

# 2a) 宿主侧**目标卡实测空闲断言**（F-10，fail-closed）：逐卡实测
#     memory.used ≤64 MiB 且 utilization.gpu ≤5%；任一卡被占即拒绝启动。
#     本轮实战里这道断言拦下过含被第三方占用的 GPU3 的 4 卡目标集。
#     宁等不抢：不 kill 他人进程，改选实测空闲的卡或等它空下来。
#     GRASPO_SKIP_IDLE_ASSERT=1 仅供无 GPU 的脚手架自检（如假 docker 的 dry-run 回归），
#     真实上机**不得**设置——跳过即失去这道防线，属显式降级。
if [ "${{GRASPO_SKIP_IDLE_ASSERT:-0}}" != "1" ]; then
    "$PYBIN" "$ROOT_DIR/scripts/gpu_idle_assert.py" --visible "$GPUS" || exit 1
else
    # 透明退路（宪法 §3.2）：跳过防线必须留痕，不能只在日志里"没有输出"。
    # 两处痕迹：① stderr 的显式 WARNING；② **环境指纹/产物** `skipped_guards.log`，
    # 使"这次运行的 F-10 防线被关过"可事后审计。
    echo "WARNING(透明退路 §3.2): GRASPO_SKIP_IDLE_ASSERT=1 ⇒ 本次运行**已跳过** F-10 目标卡空闲断言（GPU ${{GPUS}} 未做实测空闲校验）；真实上机不得设置本变量。" >&2
    mkdir -p "$RUN_ROOT"
    printf '%s skip_idle_assert=1 tier=%s gpus=%s image=%s\\n' \\
        "$(date -Is 2>/dev/null || date)" "$TIER" "$GPUS" "$IMAGE" \\
        >> "$RUN_ROOT/skipped_guards.log"
fi

# 2b) **模型挂载 fail-closed 前置断言**（与数据侧同一模式）。
# 为什么必须在 docker run **之前**：模型不在获批镜像里，配置却指向容器内
# {MODELS_CONTAINER_ROOT}/<模型目录名>；若缺失，容器会以"找不到模型目录"退出，
# 而 collect_results.py 的日志分类器会把 FileNotFoundError/No such file or directory
# 归成「数据问题」且 counts_toward_max_context=False ⇒ 真因（运行链路缺模型）
# 会被记成一次普通失败。这里先拒绝启动并把原因标成**运行链路错误**。
MODEL_DIR_NAME="${{MODEL_PATH#{MODELS_CONTAINER_ROOT}/}}"
if [ "$MODEL_PATH" != "{MODELS_CONTAINER_ROOT}/$MODEL_DIR_NAME" ] || [ -z "$MODEL_DIR_NAME" ]; then
    echo "FATAL(runtime-link): 模型未挂载 —— 配置的 model_path 不在 {MODELS_CONTAINER_ROOT}/ 下：$MODEL_PATH" >&2
    exit 4
fi
MODEL_HOST_DIR="${{!MODEL_ENV:-$MODELS_ROOT/$MODEL_DIR_NAME}}"
if [ ! -d "$MODEL_HOST_DIR" ]; then
    echo "FATAL(runtime-link): 模型未挂载 —— 宿主模型目录不存在：$MODEL_HOST_DIR" >&2
    echo "  期望目录名：$MODEL_DIR_NAME（容器内路径 $MODEL_PATH；模型不在镜像里，必须挂载）" >&2
    echo "  模型根 {MODELS_HOST_ROOT_ENV}=$MODELS_ROOT；可用 $MODEL_ENV=<该档模型目录> 单点覆盖。" >&2
    echo "  ⚠ 这是**运行链路错误**，不是数据问题；修好挂载后再跑，不要记为数据问题。" >&2
    exit 4
fi

if [ "$MODE" = "--dry-run" ]; then
    echo "[dry-run] $TIER gpus=$GPUS nproc=$NPROC status=$TIER_STATUS image=$IMAGE"
    echo "[dry-run] config=$CONFIG train_subset=$SUBSET 条 -> $TRAINPATH"
    echo "[dry-run] models=$MODELS_ROOT:$MODEL_DIR_NAME -> {MODELS_CONTAINER_ROOT}（只读）"
    exit 0
fi

RUN_DIR="$RUN_ROOT/$TIER"
mkdir -p "$RUN_DIR/subsets"

# 3) 生成训练子集：只取门槛下限（SFT ≥100 / RL ≥20），不整集跑。
if [ ! -f "$ELAM_HOST/data/train.jsonl" ]; then
    echo "FATAL: ELAM V5 train.jsonl not found: $ELAM_HOST/data/train.jsonl" >&2
    exit 3
fi
head -n "$SUBSET" "$ELAM_HOST/data/train.jsonl" > "$RUN_DIR/subsets/$TIER.jsonl"
LINES=$(wc -l < "$RUN_DIR/subsets/$TIER.jsonl")
if [ "$LINES" -lt "$SUBSET" ]; then
    echo "FATAL: subset too small: $LINES < $SUBSET" >&2
    exit 3
fi

# 4) 容器内入口：先做空闲断言（F-10）+ 起可信采样（只采实测可见卡），再 torchrun 训练。
PORT=$(( 29500 + RANDOM % 400 ))
cat > "$RUN_DIR/entry.sh" <<ENTRY
set -o pipefail
"$CONTAINER_PY" -m graspo record-gpu-memory --idle-only || exit 1
"$CONTAINER_PY" -m graspo record-gpu-memory --output-dir /out/gpu --tag "$TIER" --interval-sec 2 &
SAMPLER=\\$!
torchrun --standalone --nproc_per_node="$NPROC" --master_port="$PORT" \\
  -m graspo.cli.train_worker --config "/workspace/graspo/$CONFIG" > /out/stdout.log 2>&1
RC=\\$?
kill "\\$SAMPLER" 2>/dev/null || true
wait "\\$SAMPLER" 2>/dev/null || true
exit "\\$RC"
ENTRY

# 锁卡采用**单一路径**：宿主侧先由 gpu_lock_guard 断言，容器只认
# NVIDIA_VISIBLE_DEVICES（与容器内 train_worker 的守卫同源）。
# 不同时用 `--gpus`——两者可能互相覆盖，语义有歧义。
docker run --rm --runtime=nvidia \\
    -e NVIDIA_VISIBLE_DEVICES="$GPUS" \\
    -e PYTHONPATH=/workspace/graspo/src \\
    -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \\
    -v "$ROOT_DIR:/workspace/graspo:ro" \\
    -v "$MODELS_ROOT:{MODELS_CONTAINER_ROOT}:ro" \\
    -v "$ELAM_HOST:{ELAM_CONTAINER_ROOT}:ro" \\
    -v "$RUN_DIR/subsets:{ELAM_SUBSET_DIR}:ro" \\
    -v "$RUN_DIR:/out" \\
    -v "$RUN_DIR/entry.sh:/entry.sh:ro" \\
    -w /workspace/graspo \\
    "$IMAGE" bash /entry.sh
echo "$?" > "$RUN_DIR/exit_code"
exit "$(cat "$RUN_DIR/exit_code")"
"""


# ── 主流程 ──────────────────────────────────────────────────────────────────


def generate(expected_total: int) -> dict[str, Any]:
    tiers = build_ledger()
    assert_ledger(tiers, expected_total)
    expressibility = assert_expressibility(tiers)
    # 生成期显存可行性断言：任一可执行档估算超预算 ⇒ 拒绝生成并报算式。
    assert_memory_feasible(tiers)

    rendered: list[tuple[str, str]] = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        if status == "blocked":
            rendered.append(
                (f"{tier['tier_id']}.blocked.md", render_blocked_stub(tier, reason or ""))
            )
        else:
            rendered.append(
                (f"{tier['tier_id']}.yaml", render_config_yaml(tier, status, reason))
            )
    assert_no_legacy_terms(rendered)

    manifest = build_manifest(tiers)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    runner_text = render_runner()
    assert_no_legacy_terms([("manifest", manifest_text), ("runner", runner_text)])

    if CONFIG_DIR.exists():
        shutil.rmtree(CONFIG_DIR)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in rendered:
        (CONFIG_DIR / name).write_text(text, encoding="utf-8")
    MANIFEST_PATH.write_text(manifest_text, encoding="utf-8")
    RUNNER_PATH.write_text(runner_text, encoding="utf-8")
    RUNNER_PATH.chmod(0o755)
    write_feasibility_table(tiers)

    return {"tiers": tiers, "manifest": manifest, "expressibility": expressibility}


def write_feasibility_table(tiers: list[dict[str, Any]]) -> Path:
    """把"每档配方 + 可行性自检结果"落成一份可读表格（工位证据）。"""
    rows = feasibility_table(tiers)
    lines = [
        "| 档位 | 算法 | 模式 | 后端 | 卡数 | 状态 | 配方 | 可行性自检 |",
        "|---|---|---|---|:--:|---|---|---|",
    ]
    for row in rows:
        recipe = json.dumps(row["recipe"], ensure_ascii=False) if row["recipe"] else "—"
        lines.append(
            f"| {row['tier_id']} | {row['algorithm']} | {row['mode']} | {row['backend']} "
            f"| {row['cards']} | {row['status']} | {recipe} | {row['estimate']} |"
        )
    path = PROJECT_ROOT / ".local" / "matrix54_feasibility.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_summary(result: dict[str, Any]) -> None:
    manifest = result["manifest"]
    counts = manifest["counts"]
    print(f"档位总数: {counts['total']}")
    print(f"  算法: {counts['by_algorithm']}")
    print(f"  卡数: {counts['by_cards']}")
    print(f"  后端: {counts['by_backend']}")
    print(f"  模式: {counts['by_mode']}")
    print(f"  可表达性: {result['expressibility']}")
    print(f"配置目录: {CONFIG_DIR}")
    print(f"运行清单: {MANIFEST_PATH}")
    print(f"执行骨架: {RUNNER_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRASPO 54-tier matrix generator.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all self-checks without writing any file.",
    )
    parser.add_argument(
        "--assert-count",
        type=int,
        default=EXPECTED_TOTAL,
        help="Expected tier count asserted by --dry-run (default: 54).",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        tiers = build_ledger()
        assert_ledger(tiers, args.assert_count)
        expressibility = assert_expressibility(tiers)
        # 生成期显存可行性断言也必须在 dry-run 里跑（自检不含它就是假自检）。
        assert_memory_feasible(tiers)
        rendered = []
        for tier in tiers:
            status, reason = classify_expressibility(tier)
            if status == "blocked":
                rendered.append(
                    (f"{tier['tier_id']}.blocked.md", render_blocked_stub(tier, reason or ""))
                )
            else:
                rendered.append(
                    (f"{tier['tier_id']}.yaml", render_config_yaml(tier, status, reason))
                )
        assert_no_legacy_terms(rendered)
        print(
            f"dry-run OK: {len(tiers)} tiers, expressibility={expressibility}, "
            "内存可行性断言通过（可执行档均有可行配方）"
        )
        return 0

    result = generate(args.assert_count)
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
