#!/usr/bin/env python3
"""GRASPO 本期 54 档测试矩阵生成器（重写版，替代旧的全排列生成器）。

职责：把 `docs/capability-matrix.md` §7 的 54 档实测台账（T001–T054）逐档
翻译成「配置骨架 + 运行清单」，供后续 GPU 实验按档执行。生成物：

  - `samples/configs/matrix54/T###.yaml` —— 每档一个配置骨架
  - `samples/configs/matrix54/T###.blocked.md` —— 当前给不出配方档位的阻塞说明
  - `tests/e2e/matrix54_manifest.json` —— 运行清单（44+ 字段/档，含 GPU 集合、
    验收门槛、可表达性状态、**显存可行性三态**），是结果收集器 `scripts/collect_results.py` 的输入
  - `tests/e2e/run_matrix54.sh` —— 批量执行脚本骨架（自带锁卡守卫与可信采样）

**两条正交的口径（不要混读）**：

- **可表达性**（`ready` / `unverified` / `blocked`）答"**配置能不能被接受、能不能生成**"。
  CPT / OPD 已于 2026-09-19（`WP-X3`）解除 `blocked`：`train_method` 枚举含 `cpt`/`opd`，
  各有专用注册表与 ms-swift 映射（`swift pt` / GKD）。
- **显存可行性**（`feasible` / `infeasible` / `unmeasured` / `blocked`）答"**能不能声称装得下**"。
  `unmeasured` = 确定性算式通过、但存在**未测算**的额外显存消费者（目前仅 OPD 的教师，
  P-25）⇒ **不得写成「可行」**，等上机冒烟回填。

**★ 「可表达」≠「能跑通」≠「显存可行」**：三者都不互相蕴含。

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

# ── 显存可行性**三态**判定（进 manifest 的 `feasibility.verdict`，显式优于布尔二值）──
# 为什么不是布尔：CPT/OPD 之前，"估算过预算"与"没得估"两件事被压进同一个 None；
# OPD 引入了第三种情况——**确定性算式通过，但存在未测算的额外显存消费者（教师）**。
# 混进 True 就是"为凑数放行"，混进 False 就是"把没测过的说成跑不了"。
VERDICT_FEASIBLE = "feasible"      # 确定性算式 ≤ 预算，且无未测算的额外消费者
VERDICT_INFEASIBLE = "infeasible"  # 确定性算式已超预算 ⇒ 拒绝生成
VERDICT_UNMEASURED = "unmeasured"  # 算式通过，但有未测算的额外消费者 ⇒ 不得声称"可行"
VERDICT_BLOCKED = "blocked"        # 当前给不出配方（能力未落地 / 无分片手段）
#: ★ 合法终态（用户授权）：该档"确实不适用"——**不存在用户想要的那种配置**，不是"还没做"。
#: 与 `blocked` 的区别：`blocked` = 现在给不出配方、能力落地后可能给得出；
#: `not_applicable` = **给不出也不该给**（用户已拍板的口径下不存在该档的合法形态）。
VERDICT_NOT_APPLICABLE = "not_applicable"

ALGORITHM_TO_TRAIN_METHOD = {"CPT": "cpt", "SFT": "sft", "GRASPO": "graspo", "OPD": "opd"}

#: ★ CPT / OPD 的**通道**已落地（`WP-X3`，2026-09-19）：`train_method` 枚举含 `cpt`/`opd`，
#: 配置层对这两个算法**接受**、对 `cpt|opd + native` **fail-closed 拒绝**；路由表
#: （`core/discovery._REGISTRY_BY_TRAIN_METHOD`）把它们分别落到 `graspo.cpt_backends` /
#: `graspo.opd_backends`。⇒ 本生成器**不再**把这两类档判为 `blocked`。
#: 依据：`docs/capability-matrix.md` §4（CPT · ms-swift = ⚠️ 未验证；OPD · ms-swift = ⚠️ 未验证；
#: 两者 native 侧 = ⛔ 不支持）。
CPT_OPD_UNBLOCKED_NOTE = (
    "CPT/OPD 的**配置通道**已由 WP-X3 落地（train_method 枚举 + 专用注册表 + ms-swift "
    "pretrain/GKD 映射）。因此本档**可表达**。注意两种口径不要混："
    "**「可表达」= 配置能被接受、能生成、能启动**；**「跑通」= 真机跑出 A1–A6 证据**。"
    "本档当前只到前者。"
)

#: CPT（继续预训练）走了 ms-swift 的 `swift pt` 通道（= `swift sft --use_chat_template false
#: --loss_scale all`）。数据集形态是**纯文本续训行**，由
#: `flow/msswift/dataset.py::build_cpt_rows` 从 ELAM V5 形态的样本摊平得到。
CPT_NOTE = (
    CPT_OPD_UNBLOCKED_NOTE + " 数据形态：ms-swift 官方预训练行（单条 assistant 纯文本 + 媒体块）。"
    "**★ 语料口径（不得误读）**：本档用的是 **ELAM V5 形态**的指令数据摊平后的纯文本，"
    "目的是**跑通通道**，**不是「CPT 语料已就绪」**——本期没有预训练语料，"
    "语料问题按用户 2026-09-18 拍板登记为「回头再说」，本生成器不造语料。"
)

#: OPD（on-policy 蒸馏）走了 ms-swift 的 **GKD** 通道（`--rlhf_type gkd`），
#: 教师是同一训练进程内的**独立冻结模型**（`--teacher_model`），学生现场采样、
#: 教师现场打分（逐 token 稠密信号）。教师/学生对由用户 2026-09-18 拍板：
#: 教师 = `Qwen3.8-27B`、学生 = `Qwen3.5-9B`。
OPD_NOTE = (
    CPT_OPD_UNBLOCKED_NOTE + " 算法路径：GKD（`--rlhf_type gkd`）+ 独立冻结教师。"
    "**★ 教师侧显存未测算（P-25）**：教师模型在训练进程内**额外占一份显存**，"
    "本生成器没有教师侧的实测数字 ⇒ 所有 OPD 档的可行性判定为"
    f"**「待上机测量」（{VERDICT_UNMEASURED}）**，**不得**预先写成「可行」。"
)

#: OPD 教师模型在台账里的键（教师 = 27B，学生 = 台账的 `model`）。
OPD_TEACHER_MODEL = "27B"

#: ★ OPD 的**学生固定为 9B**（用户拍板，2026-09-19）：
#: 用户的原话是"OPD 要的是**教师 27B → 学生 9B**（省显存，价值高）"⇒ OPD 这一条通道的
#: **学生恒为 9B**，教师恒为 27B 常量。台账 §7 的 `model` 列在 graspo 里处处是
#: "**被训练的学生**"（`model.model_path`）⇒ 台账里那三行 **27B OPD 档**（`T052`–`T054`）
#: 若按现行口径读作"学生 = 27B"，就需要 **27B 学生 + 27B 教师**：教师权重**实测 50.10 GiB**
#: （上机 probe，bf16），再加 27B 学生权重（≈52 GiB）
#: ⇒ **权重常驻就 ≈100 GiB**，与用户"省显存"的意图**正好相反**。
#: ⇒ **不存在用户想要的那种配置** ⇒ 这三档判 **`not_applicable`（确实不适用）**，
#: 理由见 :data:`OPD_NOT_APPLICABLE_NOTE`。这是用户授权的**合法终态**
#: （目标验收口径明文："「失败」与「确实不适用」是合法产出"），**不是 blocked（还没做）**。
OPD_STUDENT_MODEL = "9B"

OPD_NOT_APPLICABLE_NOTE = (
    "★ 确实不适用（not_applicable）：用户拍板 **OPD 教师 = 27B 常量、学生 = 9B**（省显存，价值高）"
    "⇒ OPD 通道的学生**固定 9B**。台账该档的 `model` 列在 graspo 里是**被训练的学生**，"
    "按现行口径读作 27B 学生 ⇒ 需要 **27B 学生 + 27B 教师**：教师 bf16 权重**实测 50.10 GiB**"
    "（上机 probe），再加 27B 学生权重（≈52 GiB）⇒ **权重常驻 ≈100 GiB**，"
    "与用户「省显存」的意图正好相反。**因此用户想要的那种配置不存在** ⇒ 本档不适用。"
    "★ 这是**合法终态**（验收口径明文：「失败」与「确实不适用」都是合法产出），"
    "**不是「blocked（当前给不出配方、将来可能给得出）」**，也不是「跑失败了」。"
)


def opd_not_applicable(tier: dict[str, Any]) -> tuple[bool, str | None]:
    """该档是否属于"**确实不适用**"（用户已拍板的口径下不存在合法配置）。

    单一真相源（§1.4）：配置头部、manifest 的 ``applicability``、可行性与可表达性判定
    都从这里取值，不各判一次。
    """
    if str(tier["algorithm"]) != "OPD" or str(tier["model"]) != "27B":
        return False, None
    return True, OPD_NOT_APPLICABLE_NOTE


def opd_teacher_relation(tier: dict[str, Any]) -> str | None:
    """OPD 档的教师/学生关系：``independent`` / ``self-distillation`` / ``not-applicable``。

    - 非 OPD 档 ⇒ ``None``（"关系"这个概念对它们不适用，不编造一个值）；
    - OPD 但不适用档 ⇒ ``not-applicable``（拒绝按"自蒸馏"渲染）；
    - 其余 OPD ⇒ 比较学生与教师路径：同名 ``self-distillation``，否则 ``independent``。

    单一真相源：配置头部注释、manifest、报告都引用本函数，不各判一次（§1.4）。
    """
    if str(tier["algorithm"]) != "OPD":
        return None
    not_applicable, _ = opd_not_applicable(tier)
    if not_applicable:
        return "not-applicable"
    student = MODELS[str(tier["model"])]["path"]
    teacher = MODELS[OPD_TEACHER_MODEL]["path"]
    return "self-distillation" if student == teacher else "independent"


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

    - ``ready``：配置可表达，常规路径（LoRA）；
    - ``unverified``：配置可表达，但对应能力格当前 ⚠️ 未验证（全量）；
    - ``blocked``：配置模型/映射当前**无法表达**；或虽可表达，但按当前口径
      估算给不出可靠配方（如 native 全参 1 卡，见 ``BLOCKED_REASONS`` 的
      诚实表述——那是**保守判定，不是已证的必然 OOM**）。
    - ``not_applicable``：**确实不适用**——用户已拍板的口径下**不存在**该档的合法配置
      （现有唯一来源：台账的 27B OPD 档 `T052`–`T054`，见 :func:`opd_not_applicable`）。
      这是**合法终态**，不是 ``blocked``（"还没做"），也不产出可跑配置。

    **CPT / OPD 已于 2026-09-19（`WP-X3`）解除 ``blocked``**：`train_method` 枚举
    含 `cpt`/`opd`、各有专用注册表与 ms-swift 映射（pretrain / GKD）。
    它们按**模式**落 ``ready``（LoRA）/ ``unverified``（全量），与 SFT/GRASPO 同口径。

    **注意 ``可表达`` ≠ ``跑通``，也 ≠ ``显存可行``**：显存那一维由
    :func:`feasibility_verdict` 的三态判定给（CPT/OPD 的通道可行性见 ``CPT_NOTE`` /
    ``OPD_NOTE``；OPD 因教师侧未测算恒为 ``unmeasured``，但 ``not_applicable`` 档
    连"待测量"都不成立——它们不适用）。
    """
    # ★ 先判"确实不适用"：一个**不存在合法形态**的档，不该走后面任何一条"能跑/待测"的路径。
    not_applicable, na_reason = opd_not_applicable(tier)
    if not_applicable:
        return "not_applicable", na_reason
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


# ── ckpt 保留策略（2026-09-19，用户已授权；单一真相源）────────────────────────


def expected_optimizer_steps(tier: dict[str, Any]) -> int:
    """该档的期望全局优化步数（= 子集条数 ÷ 卡数，向下取整）。

    依据（**实测口径，不是猜测**）：矩阵档每卡 micro batch = 1、
    ``gradient_accumulation_micro_batches`` = 1 ⇒ 每卡每步 1 个样本、全局步数 = 每卡步数。
    上机实测对得上：``T013``（1 卡/100 条）= 100 步、``T014``（2 卡/100 条）= 50 步、
    ``T015``（4 卡/100 条）= 25 步（``task-r3-retest2`` 的 ``Train: N%|k/k`` 与
    ``last_model_checkpoint: checkpoint-<k>``）。

    **取向下取整**（不是向上）：它保证 ``save_steps ≤ 实际步数`` ⇒ **至少落一份 ckpt**。
    若取向上而实际步数不足（数据末端被 drop 的情形），就会一份都不落 ⇒ A3 无证据可判。
    """
    cards = max(1, int(tier["cards"]))
    return max(1, subset_size(str(tier["algorithm"])) // cards)


def checkpoint_save_steps(tier: dict[str, Any]) -> int:
    """该档的 ``training.save_steps``：**总步数**（⇒ 每档只落一份 ckpt）。

    ★ 为什么是"总步数"而不是 0/负数：``save_steps <= 0`` 在映射层会落到
    ``save_strategy: epoch`` 分支（``msswift/_config_mapping.py``），那是**另一种落盘形态**，
    不是"不落盘"。要"每档只落一份"，正确做法是让保存间隔等于总步数。

    ★ **A3 判据不受影响**：A3 = "checkpoint 能否被重新加载"，判据实现
    （``scripts/collect_results.py`` 的 ``find_checkpoint_dirs()`` /
    ``extract_checkpoint_reloadable()``）只要求**存在至少一份** ``checkpoint-<step>/``
    （ms-swift/LoRA 结构完整 + ``adapter_model.safetensors`` 可反序列化）
    ——**没有**"必须有多少份"的要求。所以"留一份"完全满足 A3；
    而"每步一份"浪费的是磁盘，不是判据。
    """
    return expected_optimizer_steps(tier)


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
#:
#: ★ ``cpt`` / ``opd`` 两条已于 2026-09-19 删除（`WP-X3` 落地了配置通道）：
#: 旧文案「`train_method` 的 Literal 只有 {'graspo','sft'}」**已是过期事实**，
#: 保留它就会把已解决的问题永远钉在「不可表达」上。留痕见 ``CPT_NOTE`` / ``OPD_NOTE``。
BLOCKED_REASONS: dict[str, str] = {
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


#: 教师权重常驻量（GiB，bf16 权重 2 B/param）——**只用来说明"不换出为什么不行"**，
#: 不是教师侧显存需求（教师前向的工作集未测算，见 ``OPD_TEACHER_UNMEASURED_NOTE``）。
def _teacher_weights_gib() -> float:
    params = PARAMS_BILLION[OPD_TEACHER_MODEL] * 1.0e9
    return params * _BF16 / (1024.0**3)


OPD_TEACHER_UNMEASURED_NOTE = (
    "★ **待上机测量（P-25）**：OPD 的教师模型（Qwen3.8-27B）在训练进程内**额外占一份显存**"
    "（每个 rank 一份），本生成器**没有**教师侧的实测数字 ⇒ **不得**判定为「可行」。"
    "配方已显式设 `distill.offload_teacher_model: true`（教师权重常驻 CPU、前向时换入），"
    "但教师**前向时的 GPU 工作集**仍是未测算量 ⇒ 本档标 "
    f"`{VERDICT_UNMEASURED}`，由**上机冒烟**（建议先跑 `T046`）回填实测后再改判定。"
    "参考算式（教师**不**换出时）：27B bf16 权重 ≈ "
    f"{_teacher_weights_gib():.1f} GiB/卡，与学生侧常驻相加在任何卡数下都超预算 ⇒ "
    "「不换出」不构成可用配方（这是权重常驻的算术事实，不是实测结论）。"
)


def feasibility_verdict(tier: dict[str, Any]) -> tuple[str, str]:
    """该档的显存可行性**三态判定**（单一真相源，§1.4），返回 (verdict, detail)。

    判定顺序即语义（先确定性、后未测算）：

    1. ``not_applicable``：该档**确实不适用**（用户拍板口径下不存在合法配置）——
       不参与任何"可行性"判断（对不存在的东西问"装不装得下"没有意义）。
    2. ``blocked``：该档当前给不出配方（能力未落地 / 无分片手段）——由
       :func:`classify_expressibility` 决定，这里不做二次判断。
    3. ``infeasible``：**确定性算式**（学生侧常驻 + 配方里显式给出的分片/offload）
       已超单卡预算 ⇒ 生成期断言**拒绝生成**并报出算式。
    4. ``unmeasured``：确定性算式通过，但存在**未测算的额外显存消费者**
       （目前唯一来源：OPD 的教师）⇒ **不得写「可行」**，标「待上机测量」（P-25）。
    5. ``feasible``：确定性算式通过，且无未测算的额外消费者。

    **为什么第 3 步不能并进第 4 步**：把"没量过的教师"当成"不存在"，就是拿未实测的
    假设去支撑一个「可行」结论（跟踪文档 `D-23`：推定值不得入账）。
    **为什么不能并进第 2 步**：算式通过就说它"必然 OOM"，同样是把假设伪装成结论。
    """
    # ★ "确实不适用"最先判：它既不是 infeasible（没说它放不下），也不是 unmeasured
    #   （它连"待上机测量"都不成立——没有该测量的配置）。
    not_applicable, na_reason = opd_not_applicable(tier)
    if not_applicable:
        return VERDICT_NOT_APPLICABLE, na_reason or ""

    status, _ = classify_expressibility(tier)
    if status == "blocked":
        return VERDICT_BLOCKED, "blocked：当前给不出可行配方（原因见 status_reason）"

    gib, basis = estimate_per_card_gib(tier)
    within_budget = gib <= CARD_BUDGET_GIB
    verdict_text = "可行" if within_budget else "估算超预算 ⇒ 判定不可靠（假设未实测）"
    detail = (
        f"估算每卡 {gib:.1f} GiB（预算 {CARD_BUDGET_GIB:g} GiB）⇒ {verdict_text}：{basis}"
    )
    if not within_budget:
        return VERDICT_INFEASIBLE, detail
    if str(tier["algorithm"]) == "OPD":
        # 先堵一个后门：`unmeasured` **不得**成为"任何 OPD 配置都放行"的借口。
        # 教师**不换出**时，其 bf16 权重是每卡常驻的——这是权重常驻的**算术事实**
        # （不是实测结论），与学生侧相加必然超预算 ⇒ 确定性判定为 infeasible。
        distill = build_config(tier).get("distill") or {}
        if distill.get("offload_teacher_model") is not True:
            teacher_gib = _teacher_weights_gib()
            return VERDICT_INFEASIBLE, (
                "OPD 教师**未换出**（`distill.offload_teacher_model` 不是 true）："
                f"教师 27B bf16 权重 ≈ {teacher_gib:.1f} GiB/卡是**常驻**的，"
                f"与学生侧估算 {gib:.1f} GiB 相加 ≈ {gib + teacher_gib:.1f} GiB > "
                f"预算 {CARD_BUDGET_GIB:g} GiB ⇒ 拒绝生成。"
                "（依据是权重常驻的算术事实，不是上机实测；教师**换出后**的 GPU 工作集"
                "仍是未测算量，见 OPD_TEACHER_UNMEASURED_NOTE。）"
            )
        # 学生侧确定性算式通过、教师已换出，但教师前向的 GPU 工作集未测算
        # ⇒ 只能标"待上机测量"。
        return VERDICT_UNMEASURED, f"{OPD_TEACHER_UNMEASURED_NOTE}；学生侧算式：{detail}"
    return VERDICT_FEASIBLE, detail


def feasibility(tier: dict[str, Any]) -> tuple[bool, str]:
    """兼容视图：只有 ``feasible`` 才为 ``True``。

    ★ **``False`` 不等于「必然 OOM」，也不等于「不可生成」**——它只表示
    **「不得声称可行」**。"超预算"与"未测算"必须用 :func:`feasibility_verdict` 区分。
    """
    verdict, detail = feasibility_verdict(tier)
    return verdict == VERDICT_FEASIBLE, detail


def feasibility_for(tier: dict[str, Any]) -> tuple[bool | None, str]:
    """给 blocked / unmeasured / not_applicable 档返回 ``None``（**均不得声称可行**）。

    三者用 ``feasibility_verdict`` 的字符串区分（``blocked`` / ``unmeasured`` /
    ``not_applicable``），manifest 里三者都带显式 ``verdict`` 字段——不靠 ``None`` 猜。
    """
    verdict, detail = feasibility_verdict(tier)
    if verdict in (VERDICT_BLOCKED, VERDICT_UNMEASURED, VERDICT_NOT_APPLICABLE):
        return None, detail
    return verdict == VERDICT_FEASIBLE, detail


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
            # epoch 末 checkpoint 关掉：全参档**不受** `save_steps` 控制的落盘（native trainer
            # 的 final、ms-swift 的 epoch 末）也要挡一道；容器内另有兜底保留步骤。
            "save_checkpoint_every_epoch": False,
            # ★ ckpt 保留策略（2026-09-19，用户已授权）：`save_steps` = 该档**总优化步数**
            #   ⇒ 每档只落一份 ckpt。原值 `1` ⇒ 每 step 一份 ⇒ 单档 100 份、≈25 GB/档
            #   （228 实测，磁盘 96%）。口径与 A3 论证见 `checkpoint_save_steps()`。
            #   **不能**设 0/负：那会落到 `save_strategy: epoch`（另一种落盘形态）。
            "save_steps": checkpoint_save_steps(tier),
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
    if algorithm == "CPT":
        # CPT 的算法级配置是**后端中立**段（`core/schema.py::PretrainConfig`），
        # 由 `_config_mapping` 的 `stage="cpt"` 分支映射成 `swift pt` 的两条等价参数。
        # 这里**显式写出**（而不是靠 schema 默认值），因为"配置仍是产物的唯一描述"
        # （§10.1）：读档位配置即可回答"这档是不是按 CPT 语义在跑"。
        config["pretrain"] = {"loss_scale": "all", "use_chat_template": False}
    if algorithm == "OPD":
        # OPD 的教师/学生对是**后端中立**段（`core/schema.py::DistillConfig`）；
        # 学生 = 上面的 `model.model_path`（不另设字段），教师显式给出（配置层强制非空）。
        # `offload_teacher_model: true` 是**起始配方**：教师 27B 权重常驻 CPU，
        # 否则其 bf16 权重（≈50.3 GiB/卡）与学生侧常驻相加在任何卡数下都超预算。
        # ★ 这只是"起跑姿势"，不是"可行性结论"——教师前向的 GPU 工作集未测算，
        #   全部 OPD 档的 verdict 恒为 UNMEASURED（见 OPD_TEACHER_UNMEASURED_NOTE）。
        config["distill"] = {
            "teacher_model_path": MODELS[OPD_TEACHER_MODEL]["path"],
            "offload_teacher_model": True,
        }
    return config


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def opd_relation_header(tier: dict[str, Any]) -> str:
    """OPD 档**一眼可辨**的教师/学生关系声明（**每档都有**，不只在自蒸馏档）。

    ★ 为什么每档都要写：教师是**用户拍板的常量**（27B），学生随台账变；两者同名时
    配置看起来像"教师路径写错了"。把关系写进**每一条** OPD 配置的头部，
    读配置的人（或 AI）不必去翻 manifest 就知道**这不是笔误**。
    """
    relation = opd_teacher_relation(tier)
    student = MODELS[str(tier["model"])]["name"]
    teacher = MODELS[OPD_TEACHER_MODEL]["name"]
    pointer = (
        "逐档关系见 manifest 的 `feasibility_model.opd_teacher.relation_by_tier`"
        "（逐档字段名 `opd_teacher_relation`）。"
    )
    if relation == "self-distillation":
        return (
            f"★ 教师/学生关系: **self-distillation** —— 本档**教师 = 学生**（均为 {teacher}），"
            "**这不是笔误**：台账该档的 `model` 列 = **学生**（graspo 处处如此），"
            f"而教师是用户拍板的**常量** {teacher}，两者同名即自蒸馏"
            "（ms-swift 明确支持的合法跑法：LoRA 用 `disable_adapter()` 取教师 logits，"
            "不额外加载模型）。" + pointer
        )
    if relation == "not-applicable":
        # 防御性分支：`not_applicable` 档**不产出 YAML**（由 classify_expressibility 拦在
        # 渲染之前）。留这个分支是为了"即使被误渲染，也绝不写成自蒸馏/独立教师"——
        # 那种输出会让人以为这档有配方。
        return (
            f"★ 教师/学生关系: **not-applicable** —— 本档（{student} 学生 × {teacher} 教师常量）"
            "**确实不适用**（见 OPD_NOT_APPLICABLE_NOTE）；本档不应产出可跑配置。" + pointer
        )
    return (
        f"★ 教师/学生关系: **independent** —— 学生 {student} ≠ 教师 {teacher}"
        "（用户拍板常量）。" + pointer
    )


def _algorithm_note(tier: dict[str, Any]) -> str | None:
    """该档位的算法口径说明（写进档位配置头部；``None`` = 无额外口径需要声明）。

    单一真相源：CPT / OPD 的说明各只有一份（``CPT_NOTE`` / ``OPD_NOTE``），
    配置头部、报告都引用它，不各写一遍（§1.4）。OPD 额外带上**逐档**的教师/学生关系
    （``opd_relation_header``）——自蒸馏与独立教师是**两种不同的跑法**，必须显式。
    """
    algorithm = str(tier["algorithm"])
    note = {"CPT": CPT_NOTE, "OPD": OPD_NOTE}.get(algorithm)
    if algorithm == "OPD" and note:
        note = f"{note} {opd_relation_header(tier)}"
    return note


def render_config_yaml(tier: dict[str, Any], status: str, reason: str | None) -> str:
    """把配置 dict 渲染成带口径注释的 YAML 文本。"""
    tier_id = str(tier["tier_id"])
    verdict, verdict_detail = feasibility_verdict(tier)
    header = [
        f"# {tier_id} | {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡",
        "# 生成器: tests/e2e/generate_matrix.py（手改无效，请改生成器）",
        "# 口径: 只用 1/2/4 卡；GPU 集合取自 {0,1,2,3}；上下文长度由递增加长法实测得出，不预设档位。",
        f"# 数据: ELAM V5 训练子集前 {subset_size(str(tier['algorithm']))} 条（宿主数据根目录经 "
        f"{ELAM_HOST_ROOT_ENV} 注入，见 .local/）；子集由 run_matrix54.sh 生成，图像经 ../images 解析。",
        f"# ⚠️ 数据口径告警: {DATA_INTEGRITY_CAVEAT}",
        f"# 可表达性: {status}",
        # 显存可行性必须**带 verdict 标签**：`unmeasured` 与 `infeasible` 的文案完全不同，
        # 只看一句人话会被读混（"不得声称可行" vs "估算超预算"）。
        f"# 显存可行性: [{verdict}] {verdict_detail}",
    ]
    if algorithm_note := _algorithm_note(tier):
        header.append(f"# 算法口径: {algorithm_note}")
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


def render_not_applicable_stub(tier: dict[str, Any], reason: str) -> str:
    """**确实不适用**档位的说明（不产出一个"看似能跑"的错误配置）。

    ★ 与 :func:`render_blocked_stub` 的区别**必须写在纸面上**：`blocked` 是
    "现在给不出配方、能力落地后可再生成"；`not_applicable` 是
    "**用户已拍板的口径下不存在该档的合法形态**"——它**不是**待办，不会被
    "能力落地"解除。把两者写成同一段文字，就会有人把它当成待办去排期。
    """
    return (
        f"# {tier['tier_id']} 确实不适用（not_applicable）—— 合法终态，不产出可跑配置\n\n"
        f"档位: {tier['model']} | {tier['algorithm']} | {tier['mode']} | "
        f"{tier['backend']} | {tier['cards']}卡\n\n"
        f"不适用的理由: {reason}\n\n"
        "定性: **确实不适用**（用户已拍板的口径下不存在该档的合法配置）——\n"
        "      **不是** `blocked`（尚未给出配方、将来可能做得了），**不是** `❌ 失败`（跑过没通过）。\n"
        "      「失败」与「确实不适用」都是验收口径明文允许的**合法产出**。\n"
        "处置: 无需排期、无需上机；台账侧按 `⛔ 不适用` 登记，理由引用本节。\n"
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
        # ★ 适用性（2026-09-19，指挥官裁定 A）：与"可表达性"正交的一维。
        #   `applicable: false` = **确实不适用**（用户拍板口径下不存在合法配置），
        #   是**合法终态**；只有 `true` 的档才谈"能不能跑 / 装不装得下"。
        applicable = status != "not_applicable"
        entry["applicable"] = applicable
        if not applicable:
            entry["applicability_reason"] = reason
        if str(tier["algorithm"]) == "OPD":
            # 教师/学生关系必须进 manifest：独立教师与自蒸馏是两种跑法，
            # 事后只看 "teacher_model_path" 无法区分（同名的档）。
            # ★ `T052`–`T054` 现为 `not-applicable`（不再按自蒸馏渲染）。
            entry["opd_teacher_relation"] = opd_teacher_relation(tier)
        if status in ("blocked", "not_applicable"):
            entry["config"] = None
            # 不适用档不产出"可跑配置"，也不产出 blocked 的"能力落地后再生成"说明——
            # 后者会被读成"这档将来能跑"。理由进 `applicability_reason`。
            entry["blocked_doc"] = (
                f"samples/configs/matrix54/{tier['tier_id']}.blocked.md"
                if status == "blocked"
                else None
            )
        else:
            entry["config"] = f"samples/configs/matrix54/{tier['tier_id']}.yaml"
        entry["data"] = {
            "train_path": subset_path(str(tier["tier_id"])),
            "subset_size": subset_size(str(tier["algorithm"])),
            "full_train_jsonl": ELAM_TRAIN_JSONL,
            "caveat": DATA_INTEGRITY_CAVEAT,
        }
        # 期望总优化步数 = `training.save_steps`（ckpt 保留策略，见 runtime.checkpoint_retention）
        entry["expected_optimizer_steps"] = expected_optimizer_steps(tier)
        entry["checkpoint_save_steps"] = checkpoint_save_steps(tier)
        feasible, estimate = feasibility_for(tier)
        verdict, _ = feasibility_verdict(tier)
        entry["feasibility"] = {
            # ★ `verdict` 是显式四态（+infeasible 共五值），`feasible` 只保留向后兼容语义：
            #   feasible ⇒ True；infeasible ⇒ False；unmeasured / blocked / not_applicable
            #   ⇒ None（**不得声称可行**）。不要用 `feasible is None` 区分原因 —— 读 `verdict`。
            "verdict": verdict,
            "feasible": feasible,
            "requires_measurement": verdict == VERDICT_UNMEASURED,
            "estimate": estimate,
            "recipe": (
                {"native": build_config(tier).get("native")}
                if tier["backend"] == "native"
                else {"msswift": build_config(tier).get("msswift")}
            )
            if verdict not in (VERDICT_BLOCKED, VERDICT_NOT_APPLICABLE)
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
            "by_applicability": _count_by(
                [
                    {"applicable": "applicable" if e["applicable"] else "not_applicable"}
                    for e in entries
                ],
                "applicable",
            ),
            "by_feasibility_verdict": _count_by(
                [entry["feasibility"] for entry in entries], "verdict"
            ),
        },
        "runtime": {
            "image": "graspo-msswift:4.5.3",
            "timeout_sec": TIMEOUT_SEC,
            "initial_max_prompt_length": INITIAL_MAX_PROMPT_LENGTH,
            # ── ckpt 保留策略（2026-09-19，用户已授权；主机侧 rig 照此执行，不另立口径）──
            # 原本 `save_steps: 1` ⇒ 每 step 一份 ⇒ ≈25 GB/档（228 实测，磁盘 96%）。
            # 现策略：每档只留**一份** ckpt（A3 判据只需一份可重载的 ckpt）。
            "checkpoint_retention": {
                "policy": "keep-latest-single-checkpoint",
                "keep_per_tier": 1,
                "how": (
                    "① 配置侧：`training.save_steps` = 该档总优化步数（见每档 "
                    "`checkpoint_save_steps`），=「每档只落一份」；"
                    "② runner 侧：容器内落盘后立即逐项目具名删除旧 ckpt（`[ckpt-retention]` "
                    "日志行），并断言至少还剩一份（否则报运行链路错误，不静默通过）。"
                ),
                "keep": [
                    "最新一份 `checkpoint-<step>/`（A2 权重变化证据 + A3 重载证据）",
                    "轻量证据一律保留：`stdout.log`、`logging.jsonl`、`args.json`、"
                    "`trainer_state.json`、`gpu/` 读数、`subsets/`、`exit_code`、"
                    "`ckpt_retention.state`",
                ],
                "delete": [
                    "同一档内除最新一份以外的 `checkpoint-*/`（逐项具名，不用通配符展开做操作数）",
                ],
                "a3_unaffected": (
                    "A3 = checkpoint 能否重载；`scripts/collect_results.py` 的 "
                    "`find_checkpoint_dirs()` 只要求**存在 ≥1 份**可解析的 `checkpoint-<step>/`，"
                    "没有份数要求 ⇒ 留一份即满足 A3。"
                ),
                "blocked_by_env": (
                    "checkpoint 由容器内 root 落盘，宿主普通用户删不掉（228 实测：`sudo` 需密码）"
                    "⇒ 删除只能发生在容器内，故实现落在 runner 的 entry.sh。"
                ),
            },
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
                "CPT：与学生侧同一套学生侧算式（算法不改变常驻口径）；全量档沿用 ms-swift 全参配方",
                "OPD：学生侧算式同上；**教师侧不参与算式**——教师（27B）显存是本生成器"
                "**未测算**的量 ⇒ 判定恒为 unmeasured（不得声称可行，P-25）",
            ],
            "verdicts": {
                "values": [
                    VERDICT_FEASIBLE,
                    VERDICT_INFEASIBLE,
                    VERDICT_UNMEASURED,
                    VERDICT_BLOCKED,
                ],
                "meaning": {
                    VERDICT_FEASIBLE: "确定性算式 ≤ 预算且无未测算的额外消费者",
                    VERDICT_INFEASIBLE: "确定性算式已超预算 ⇒ 拒绝生成",
                    VERDICT_UNMEASURED: "算式通过但有未测算的额外消费者 ⇒ 不得声称可行，待上机测量",
                    VERDICT_BLOCKED: "当前给不出配方（能力未落地 / 无分片手段）",
                    VERDICT_NOT_APPLICABLE: (
                        "确实不适用——用户拍板口径下不存在该档的合法配置 ⇒ 合法终态，"
                        "不产出可跑配置、不参与可行性判断"
                    ),
                },
                "note": (
                    "`feasibility.feasible` 是向后兼容的布尔视图：True/False/None。"
                    "`None` 同时覆盖 unmeasured / blocked / not_applicable —— "
                    "**必须读 `verdict` 区分**，不得用 `feasible is None` 推断原因。"
                ),
            },
            "opd_teacher": {
                "teacher_model": MODELS[OPD_TEACHER_MODEL]["name"],
                "teacher_model_path": MODELS[OPD_TEACHER_MODEL]["path"],
                "student_model": (
                    f"OPD 学生**固定 {OPD_STUDENT_MODEL}**（用户拍板：省显存，价值高）；"
                    "教师 = 27B 常量。台账 `model` 列在 graspo 里 = 被训练的学生。"
                ),
                "teacher_weights_gib_bf16": round(_teacher_weights_gib(), 1),
                "teacher_weights_gib_bf16_measured": 50.1,
                "teacher_weights_measurement": (
                    "上机 probe 实测（PyTorch allocator `max_memory_allocated`，bf16，"
                    "26.896 B 参数）⇒ 50.10 GiB；出处 task-cpt-opd / task-r3-retest2 报告。"
                ),
                "measured": False,
                "unmeasured_note": OPD_TEACHER_UNMEASURED_NOTE,
                "not_applicable_note": OPD_NOT_APPLICABLE_NOTE,
                "not_applicable_tiers": [
                    str(tier["tier_id"]) for tier in tiers if opd_not_applicable(tier)[0]
                ],
                "relation_by_tier": {
                    str(tier["tier_id"]): opd_teacher_relation(tier)
                    for tier in tiers
                    if str(tier["algorithm"]) == "OPD"
                },
                "decision": (
                    "★ 已由指挥官裁定并落地（2026-09-19）：`T052`–`T054`（台账的 27B OPD 档）"
                    "由原口径 A（27B→27B 自蒸馏）改判为 **`not_applicable`（确实不适用）**。"
                    "理由：用户拍板 **OPD 教师 = 27B 常量、学生 = 9B**（省显存）⇒ OPD 学生固定 9B；"
                    "27B 行若作为学生则需 27B 学生 + 27B 教师，教师权重实测 50.10 GiB、"
                    "加 27B 学生权重 ≈100 GiB 权重常驻，与用户意图相反 ⇒ **确实不适用**。"
                    "`T046`–`T051`（9B OPD）保持 **independent（独立教师）**关系不变。"
                    "这是**合法终态**（验收口径明文：「失败」与「确实不适用」都是合法产出）。"
                ),
            },
            "generation_gate": (
                "assert_memory_feasible：确定性算式超预算（infeasible）即拒绝生成"
                "（不允许安静产出不可靠配置）；unmeasured 档**允许生成但不声称可行**"
                "（未测算的量只能靠上机量出来），并在 stdout 打出待测清单；"
                "`not_applicable` 档**不产出可跑配置**（确实不适用、无待测量）"
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
    """统计可表达性，并断言各状态之和等于总数。

    ★ 四态（2026-09-19 起）：`not_applicable` 是**确实不适用**的合法终态
    （现有唯一来源 `T052`–`T054`）——它与 `blocked` 分开计数，否则
    "还没做"与"不该做"会被压成同一个数（§1.4 单一真相源要求可回答"为什么"）。
    """
    counts: dict[str, int] = {
        "ready": 0,
        "unverified": 0,
        "blocked": 0,
        "not_applicable": 0,
    }
    for tier in tiers:
        status, _ = classify_expressibility(tier)
        counts[status] += 1
    assert sum(counts.values()) == EXPECTED_TOTAL, counts
    return counts


# ── 生成期显存可行性断言（§2.3 边界校验即防呆）──────────────────────────────


def assert_memory_feasible(tiers: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """对每个**可执行**档做保守显存估算；任一档确定性算式超预算即**拒绝生成**。

    这是本轮补齐的关键防线：配置能生成 ≠ 能跑。宁可在生成期失败并报出算式，
    也不要在花掉 GPU 时间之后才发现不可靠。

    **四态口径（单一真相源 = :func:`feasibility_verdict`）**：

    - ``infeasible`` ⇒ **拒绝生成**（报出算式）；
    - ``unmeasured`` ⇒ **允许生成**，但**不声称可行**：把"待上机测量"打到 stdout，
      由 manifest 的 ``feasibility.requires_measurement`` 承接（★ 允许生成是**刻意的**：
      未测算的量只能靠上机量出来，"因为没量过所以不生成"会让它永远量不了）；
    - ``feasible`` ⇒ 通过；
    - ``blocked`` ⇒ 不参与断言（已知给不出配方，由 ``classify_expressibility`` 说明原因）；
    - ``not_applicable`` ⇒ **不参与断言**（确实不适用：既不是"放不下"也不是"待测量"，
      由 ``classify_expressibility`` / manifest 的 ``applicability`` 说明原因）。

    Returns:
        ``{tier_id: (verdict, detail)}``——第一项是**字符串四态**，不是布尔
        （布尔会把 "未测算" / "超预算" / "不适用" 压成同一个 False）。
    """
    verdicts: dict[str, tuple[str, str]] = {}
    violations: list[str] = []
    pending_measurement: list[str] = []
    for tier in tiers:
        verdict, detail = feasibility_verdict(tier)
        if verdict in (VERDICT_BLOCKED, VERDICT_NOT_APPLICABLE):
            continue
        verdicts[str(tier["tier_id"])] = (verdict, detail)
        if verdict == VERDICT_INFEASIBLE:
            violations.append(
                f"{tier['tier_id']} ({tier['algorithm']}/{tier['mode']}/"
                f"{tier['backend']}/{tier['cards']}卡): {detail}"
            )
        elif verdict == VERDICT_UNMEASURED:
            pending_measurement.append(f"{tier['tier_id']} ({tier['algorithm']})")
    if violations:
        raise AssertionError(
            "生成期显存可行性断言失败——拒绝生成确定性算式已超预算的配置：\n  - "
            + "\n  - ".join(violations)
        )
    if pending_measurement:
        print(
            "⚠️  待上机测量（未测算的额外显存消费者：OPD 教师，见 P-25）"
            f"——共 {len(pending_measurement)} 档，**不得**在此之前写成「可行」：\n  - "
            + "\n  - ".join(pending_measurement)
        )
    return verdicts


def feasibility_table(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个档位的配方 + 可行性自检结果（写进 manifest / 报告）。"""
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        verdict, detail = feasibility_verdict(tier)
        ok, _ = feasibility_for(tier)
        config = build_config(tier) if status != "blocked" else {}
        recipe: dict[str, Any] = {}
        if status != "blocked":
            if tier["backend"] == "native":
                recipe = {"native": config.get("native")}
            else:
                recipe = {"msswift": config.get("msswift")}
            for section in ("pretrain", "distill"):
                if section in config:
                    recipe[section] = config[section]
        rows.append(
            {
                "tier_id": tier["tier_id"],
                "algorithm": tier["algorithm"],
                "mode": tier["mode"],
                "backend": tier["backend"],
                "cards": tier["cards"],
                "status": status,
                "verdict": verdict,
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
# 数据（**挂载目的地由配置反推，不硬编码**）:
#   每档按 §6 门槛取下限生成训练子集 -> 宿主 <RUN_ROOT>/<T###>/subsets/<T###>.jsonl，
#   只读挂到**该档配置 data.train_path 的所在目录**；图像根只读挂到**由 train_path
#   反推出来的同级 images 目录**。两者都从 manifest 的 data.train_path 用 dirname 推出来，
#   而不是写死的常量（单一真相源，见脚本 1) 段）。
#
#   ⚠ 为什么不再用"整棵数据根 + 其下嵌套子集"（实测确证的运行链路缺陷，rc=125）：
#     旧命令 `-v "$ELAM_HOST:{ELAM_CONTAINER_ROOT}:ro"` +
#     `-v "$RUN_DIR/subsets:{ELAM_SUBSET_DIR}:ro"` 把子挂载点放在一个**只读**绑定挂载之下，
#     dockerd 必须在只读文件系统上创建嵌套 mountpoint：
#     `mkdirat .../subsets: read-only file system` ⇒ `docker run` 返回 **125，一档都起不来**。
#     现方案让所有挂载目标**彼此互不嵌套**（子集目录与图像目录是兄弟，父目录留在容器
#     可写层里），从结构上消除"在只读挂载下建 mountpoint"这一步。
#     该不变量在启动前由 `assert_mount_targets_not_under_readonly` 断言（fail-closed，
#     给出可操作报错，而不是让 docker 报一个语义模糊的 125）。
#
# 模型: 每档配置里的 `model.model_path` 都指向容器内 {MODELS_CONTAINER_ROOT}/<模型目录名>；
#       runner 从 manifest 读出该路径并把它解析回宿主路径做存在性断言。
# ⚠️ 数据口径告警: {DATA_INTEGRITY_CAVEAT}
#
# 红线：只用 GPU0-5、每次最多 4 卡、GPU6/7 永不触碰。本脚本不自动执行矩阵。
set -uo pipefail

# ── 挂载结构不变量（fail-closed，纯 argv 分析，不需要 docker）──────────────────
# 唯一真相源：**任何挂载目标都不得位于另一个只读（ro）挂载目标之下**。
# 原因（实测确证，rc=125）：父挂载只读时 dockerd 无法在其内部创建嵌套 mountpoint，
# `mkdirat ...: read-only file system` ⇒ docker 返回 125，容器根本没起来。
# 本函数把这条"运行链路缺陷"提前变成可操作报错；参数是 `宿主源|容器目标|模式` 三元组。
#   - 目标规范化为无尾斜杠；
#   - 只读父目标下的任何挂载都判违规（含相等的情况——同一目标挂两次也是错误）。
# 允许测试单独 source 进来调用（见 GRASPO_RUNNER_LIB_ONLY）。
assert_mount_targets_not_under_readonly() {{
    local -a specs=("$@")
    local i j src dst mode src2 dst2 mode2
    for ((i = 0; i < ${{#specs[@]}}; i++)); do
        IFS='|' read -r src dst mode <<< "${{specs[$i]}}"
        for ((j = 0; j < ${{#specs[@]}}; j++)); do
            [ "$i" -eq "$j" ] && continue
            IFS='|' read -r src2 dst2 mode2 <<< "${{specs[$j]}}"
            [ "$mode2" = "ro" ] || continue
            case "${{dst}}/" in
                "${{dst2}}/"*)
                    echo "FATAL(runtime-link): 挂载目标 '$dst' 位于只读挂载 '$dst2' 之下 ——" >&2
                    echo "  dockerd 无法在只读文件系统上创建嵌套 mountpoint，会以 rc=125 失败。" >&2
                    echo "  修法：让挂载目标彼此互不嵌套（例如把子目录与它的父目录分开挂到同一层的兄弟位置）。" >&2
                    return 6
                    ;;
            esac
        done
    done
    return 0
}}

# 测试用的库模式：只定义上面的断言函数，不执行主流程（不读 manifest、不起容器）。
if [ "${{GRASPO_RUNNER_LIB_ONLY:-0}}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

TIER="${{1:?usage: run_matrix54.sh <T###> [--dry-run]}}"
MODE="${{2:-run}}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${{GRASPO_IMAGE:-graspo-msswift:4.5.3}}"
RUN_ROOT="${{RUN_ROOT:-$ROOT_DIR/.local/matrix54-runs}}"
ELAM_HOST="${{{ELAM_HOST_ROOT_ENV}:-}}"
MODELS_ROOT="${{{MODELS_HOST_ROOT_ENV}:-}}"
MANIFEST="${{GRASPO_RUNNER_MANIFEST:-$ROOT_DIR/tests/e2e/matrix54_manifest.json}}"
# ↑ 清单路径可被 `GRASPO_RUNNER_MANIFEST` 覆盖：给"逐长度递增批"这类需要**单档清单**
#   的场景用（rig 只换清单、不换 runner 逻辑）；默认路径与行为逐字不变。
#   本行与 `run_matrix54.sh` 的同一行**逐字对应**（由 rig/verify_runner_parity.py 校验）。
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
    "verdict": (tier.get("feasibility") or {{}}).get("verdict") or "-",
}}, ensure_ascii=False))
PY
) || exit 5

read -r GPUS CONFIG_STATUS NPROC SUBSET TRAINPATH MODEL_PATH MODEL_ENV VERDICT \\
  < <("$PYBIN" - "$TIER_JSON" <<'PY'
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
    data["verdict"],
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

# 0a-bis) **待上机测量**档（OPD：教师侧显存未测算，P-25）：允许跑（跑就是为了量出来），
# 但必须显式提示——并把"教师侧峰值显存"记进产物，否则这一档的可行性永远无据。
# 判据：manifest 的 `feasibility.verdict == unmeasured`（不是"没配方"，两者语义不同）。
if [ "$VERDICT" = "unmeasured" ]; then
    echo "WARN(待上机测量): $TIER 的额外显存消费者（OPD 教师）**未测算** ——" >&2
    echo "     本次跑的产物里必须单独记录**教师侧**与**学生侧**峰值显存，回填后才能写「可行」。" >&2
    echo "     参考：教师 27B bf16 权重 ≈ 50.3 GiB/卡（不换出时）；本档配方已 offload_teacher_model=true。" >&2
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

# 1) **由配置自身反推挂载目的地**（单一真相源，不硬编码目的地）：
#    相对媒体路径的锚点口径（native 与 ms-swift **共用同一条**）是"训练 JSONL 的父目录"
#    （native: flow/trainer/sft_trainer.py 的 Path(train_path).parent；
#      ms-swift: flow/msswift/dataset.py. 两处说的是同一件事，本脚本只认这一条）。
#    ELAM V5 的样本把图像写作 `../images/<file>.jpg` ⇒ 容器内必须让
#    `dirname(train_path)/../images` 可达。因此：
#      TRAIN_PATH_DIR_CONTAINER = dirname(train_path)          ← 子集的宿主目录挂到这里
#      IMAGE_DIR_CONTAINER      = dirname(dirname(train_path))/images  ← 图像根挂到这里
#    布局若变（如 train_path 移到 <根>/data/v2/xxx.jsonl），这两处**自动跟随**，
#    不需要改本脚本——这就是"不再断裂"的原因；另有容器内媒体预检兜底（见 4) 段）。
TRAIN_PATH="$TRAINPATH"
TRAIN_PATH_DIR_CONTAINER="$(dirname "$TRAIN_PATH")"
IMAGE_DIR_CONTAINER="$(dirname "$TRAIN_PATH_DIR_CONTAINER")/images"
if [ "$TRAIN_PATH" != "$TRAIN_PATH_DIR_CONTAINER/$TIER.jsonl" ]; then
    echo "FATAL(runtime-link): 数据子集路径与配置不一致 ——" >&2
    echo "  配置 data.train_path=$TRAIN_PATH，但本档生成的子集文件名是 $TIER.jsonl" >&2
    echo "  （runner 只生成 <RUN_ROOT>/$TIER/subsets/$TIER.jsonl）。" >&2
    echo "  ⚠ 这是**运行链路错误**：容器里打开的数据文件与实际产物不是同一个。" >&2
    exit 3
fi

# 1b) **媒体锚点可达性（宿主侧，fail-closed）**：图像根缺失时立刻拒绝启动。
#     若不拦，训练会以 FileNotFoundError / PIL.UnidentifiedImageError 退出，
#     被 collect_results.py 的日志分类器记成「数据问题」，真因（运行链路缺图像根）被掩盖。
#     ⚠ 不用符号链接兜底：符号链接在容器内指向宿主路径，必然断开（r3 实测踩过）。
if [ ! -d "$ELAM_HOST/images" ]; then
    echo "FATAL(runtime-link): 媒体锚点不可达 —— 宿主图像根不存在：$ELAM_HOST/images" >&2
    echo "  锚点口径：相对媒体路径相对训练 JSONL 的父目录解析 ⇒ 容器内需要" >&2
    echo "  <dirname(train_path)>/../images = $IMAGE_DIR_CONTAINER" >&2
    echo "  宿主来源：$ELAM_HOST/images（环境变量 {ELAM_HOST_ROOT_ENV}）。" >&2
    echo "  ⚠ 这是**运行链路错误**，不是数据问题；补齐图像根后再跑。" >&2
    exit 3
fi

if [ "$MODE" = "--dry-run" ]; then
    echo "[dry-run] $TIER gpus=$GPUS nproc=$NPROC status=$TIER_STATUS image=$IMAGE"
    echo "[dry-run] config=$CONFIG train_subset=$SUBSET 条 -> $TRAINPATH"
    echo "[dry-run] models=$MODELS_ROOT:$MODEL_DIR_NAME -> {MODELS_CONTAINER_ROOT}（只读）"
    echo "[dry-run] subsets=$RUN_ROOT/$TIER/subsets -> $TRAIN_PATH_DIR_CONTAINER（只读）"
    echo "[dry-run] images=$ELAM_HOST/images -> $IMAGE_DIR_CONTAINER（只读；媒体锚点）"
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

# 4) 容器内媒体预检脚本（写在运行目录里，经 /out 只读进容器；训练前先跑）。
#    它是**断言**不是第二套解析器：只验证"训练器将要打开的绝对路径确实存在"。
#    锚点口径与两个后端共用同一条：训练 JSONL 的父目录。
cat > "$RUN_DIR/preflight_media.py" <<'PY'
#!/usr/bin/env python3
# 媒体锚点可达性前置断言（容器内，训练之前；只读、不改写任何路径）。
#
# 锚点口径（全项目单一真相源，native 与 ms-swift 共用同一条）：
#     训练 JSONL 里的相对媒体路径，相对于该 JSONL 文件所在目录解析。
# 本脚本验证"训练器将要打开的绝对路径在容器内确实存在"，因此布局漂移会在训练
# 启动前以可操作报错 fail-closed；否则 ms-swift / native 会在数据准备阶段抛
# FileNotFoundError / PIL.UnidentifiedImageError，被日志分类器记成「数据问题」，
# 把运行链路缺口的真因掩盖掉。

import json
import os
import sys

MEDIA_KEYS = ("image", "path", "url", "video")
MEDIA_TYPES = ("image", "image_url", "video", "video_url")
MAX_EXAMPLES = 5


def is_relative(value):
    return not value.startswith(("http://", "https://", "/", "data:"))


def iter_media_paths(raw):
    # 产出该样本里的媒体路径（读的是训练侧同一个字段：messages[*].content[*]）。
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "").lower() not in MEDIA_TYPES:
                continue
            for key in MEDIA_KEYS:
                value = block.get(key)
                if isinstance(value, str):
                    yield value


def main():
    train_path = sys.argv[1]
    if not os.path.isabs(train_path):
        train_path = os.path.join(os.getcwd(), train_path)
    anchor = os.path.dirname(train_path)
    checked = 0
    blocks_found = 0
    missing = []
    shape_mismatch = 0
    with open(train_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                print("[preflight] FATAL(runtime-link): %s:%d 不是合法 JSON: %s" % (train_path, line_no, exc))
                return 3
            if not isinstance(raw, dict):
                continue
            found_here = 0
            for value in iter_media_paths(raw):
                found_here += 1
                if not is_relative(value):
                    # 绝对路径 / http(s) / data: 不属本预检范围（训练侧同样跳过）。
                    continue
                checked += 1
                resolved = os.path.join(anchor, value)
                if not os.path.exists(resolved):
                    if len(missing) < MAX_EXAMPLES:
                        missing.append("%s:%d -> %s" % (train_path, line_no, resolved))
            blocks_found += found_here
            if found_here == 0 and '"image"' in line:
                # 文本里出现 image 字段却按契约解析不到任何媒体块 ⇒ 形状变了。
                # 不做静默放行：预检若"什么都没查"就等于没有防线。
                shape_mismatch += 1
    if missing:
        print("[preflight] FATAL(runtime-link): 媒体锚点不可达 —— 相对媒体路径解析后不存在。")
        print("[preflight]   锚点口径：相对路径相对训练 JSONL 的父目录解析；anchor=%s" % anchor)
        print("[preflight]   共检查 %d 条相对媒体路径，缺失示例：" % checked)
        for item in missing:
            print("[preflight]     %s" % item)
        print("[preflight]   这是运行链路错误（挂载/布局），不是数据问题。")
        print("[preflight]   修法：确认图像根被挂到 dirname(dirname(train_path))/images。")
        return 3
    if shape_mismatch:
        print("[preflight] FATAL(runtime-link): 预检解析器与数据形状不一致 ——")
        print("[preflight]   %d 行含 'image' 字段，但按 messages[*].content[*] 契约解析不到任何媒体块。" % shape_mismatch)
        print("[preflight]   不做静默放行（否则预检形同虚设）：请同步本脚本与数据契约。")
        return 3
    print("[preflight] media anchor OK: %d 条相对媒体路径在 %s 下全部可达" % (checked, anchor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

PORT=$(( 29500 + RANDOM % 400 ))
cat > "$RUN_DIR/entry.sh" <<ENTRY
set -o pipefail
"$CONTAINER_PY" /out/preflight_media.py "$TRAIN_PATH" || exit 3
"$CONTAINER_PY" -m graspo record-gpu-memory --idle-only || exit 1
"$CONTAINER_PY" -m graspo record-gpu-memory --output-dir /out/gpu --tag "\$TIER" --interval-sec 2 &
SAMPLER=\\$!
torchrun --standalone --nproc_per_node="$NPROC" --master_port="$PORT" \\
  -m graspo.cli.train_worker --config "/workspace/graspo/$CONFIG" > /out/stdout.log 2>&1
RC=\\$?
# 产物放开读权限（**必须**）：容器内训练以 root 运行 ⇒ ms-swift/HF 默认落 root:0600，
# 宿主侧的 collect_results.py（普通用户）**打不开** checkpoint 权重 ⇒ A2/A3 读不到证据
# （228 实测踩过：A3 一度被记成"checkpoint 无法重新加载"，属方向性错误）。
# 产物必须可被宿主侧的收集/复现链路读取（§6）；chmod 只改元数据、不碰内容。
chmod -R a+rX /out 2>/dev/null || echo "WARN: chmod -R a+rX /out failed — 宿主侧可能读不到产物" >&2
# ── ckpt 保留策略（落盘后立即执行，§2.4 具名清理；**通配符只用于 find 的 -name 匹配，不作为删除操作数**）──
# 为什么在容器内做：checkpoint 由**容器内 root** 落盘，宿主普通用户删不掉
# （228 实测：sudo 需要密码，非交互不可用）。容器内是 root ⇒ 只有这里删得掉。
# ★ 写法约束（防呆）：本块**只使用内层 shell 自己的变量**（\\$1/\\$@/函数内局部量），
#   不读写外层 runner 的任何变量，也不内联任何函数调用。这样"外层展开"与"内层展开"
#   在语义上重合，不会因为外层 set -u 误判内层变量（228/本机都踩过这个坑）。
#   ★ 唯一的例外是档号 \\$TIER：它由**外层 `docker run -e "TIER=$TIER"` 显式注入**容器，
#     不是"恰好未定义⇒空串"的巧合（那种隐性依赖是脆弱设计，§2 显式即防呆）。
retain_single_checkpoint() {{
    local output_root="\\${{1:-}}"
    local -a all=() ckpts=()
    local path step newest_path="" newest_step="" deleted=0 keep_count=0

    if [ ! -d "\\$output_root" ]; then
        return 0
    fi
    mapfile -t all < <(find "\\$output_root" -type d -name 'checkpoint-*' -print 2>/dev/null)
    if [ "\\${{#all[@]}}" -eq 0 ]; then
        echo "[ckpt-retention] FATAL: 未找到任何 checkpoint-* 目录（output_root=\\$output_root）——" >&2
        echo "[ckpt-retention]   A3（checkpoint 重载）将无证据可判。这是**运行链路错误**，" >&2
        echo "[ckpt-retention]   不是训练失败：请检查 save_steps 是否大于实际优化步数。" >&2
        return 4
    fi
    for path in "\\${{all[@]}}"; do
        if [ -f "\\$path/trainer_state.json" ]; then
            ckpts+=("\\$path")
        else
            echo "[ckpt-retention] WARN: 跳过（缺 trainer_state.json）\\$path" >&2
        fi
    done
    if [ "\\${{#ckpts[@]}}" -eq 0 ]; then
        ckpts=("\\${{all[@]}}")
    fi
    # 只留**最新一份**：按 checkpoint-<step> 的步数取最大
    #（同一 run 内步数唯一且单调递增；末步的目录名即最大步数）
    for path in "\\${{ckpts[@]}}"; do
        step="\\${{path##*-}}"
        case "\\$step" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ -z "\\$newest_path" ] || [ "\\$step" -gt "\\$newest_step" ]; then
            newest_path="\\$path"; newest_step="\\$step"
        fi
    done
    if [ -z "\\$newest_path" ]; then
        echo "[ckpt-retention] WARN: ckpt 目录名不含步数，跳过保留动作（不猜、不删）" >&2
        return 0
    fi
    for path in "\\${{ckpts[@]}}"; do
        if [ "\\$path" = "\\$newest_path" ]; then
            keep_count=\\$((keep_count + 1))
            echo "[ckpt-retention] KEEP   \\$path"
            continue
        fi
        echo "[ckpt-retention] DELETE \\$path（中间段 ckpt；A3 只需最新一份）"
        if ! rm -rf -- "\\$path"; then
            echo "[ckpt-retention] WARN: rm 失败 \\$path" >&2
        fi
        deleted=\\$((deleted + 1))
    done
    echo "[ckpt-retention] 保留 ckpt 数=\\$keep_count 删除数=\\$deleted"
    return 0
}}
OUTPUT_ROOT="/out/\\$TIER"
CKPT_STATE="none"
if [ -d "\\$OUTPUT_ROOT" ]; then
    if retain_single_checkpoint "\\$OUTPUT_ROOT"; then
        CKPT_STATE="ok"
    else
        CKPT_STATE="missing"
        echo "[ckpt-retention] FATAL: 保留动作未完成（见上方 FATAL/WARN）——" >&2
        echo "[ckpt-retention]   训练产物目录存在但**无一份可留的 checkpoint**；A3 无证据可判。" >&2
        echo "[ckpt-retention]   这是**运行链路错误**，不是训练失败：请检查 save_steps 是否大于实际优化步数。" >&2
        printf '%s\\n' "\\$CKPT_STATE" > /out/ckpt_retention.state
        # 训练失败（RC≠0）时保留原 rc，绝不掩盖训练失败；训练成功却无 ckpt 才是链路错误。
        if [ "\\$RC" = "0" ]; then exit 4; fi
    fi
else
    # ★ 目录不存在**不得静默跳过**（保留策略悄悄失效 ⇒ 磁盘迟早爆）。
    #   分两种情形：
    #     · 训练本身失败（RC≠0）：目录本就不会存在 ⇒ 记 state=none 并**保留原 rc**，
    #       不掩盖训练失败（这是唯一合法情形）。
    #     · 训练成功（RC=0）却无产物目录 ⇒ **产物通道/档号定位坏了** ⇒ FATAL + exit 4。
    CKPT_STATE="none"
    echo "[ckpt-retention] FATAL: 训练产物目录不存在（output_root=\\$OUTPUT_ROOT，RC=\\$RC）——" >&2
    echo "[ckpt-retention]   保留策略无法执行。若 RC≠0 则是训练失败（按原 rc 返回，不掩盖）；" >&2
    echo "[ckpt-retention]   若 RC=0 则说明**产物目录没落出来或档号定位错误**（TIER=\\${{TIER:-<未设置>}}），" >&2
    echo "[ckpt-retention]   这是运行链路错误，不是训练失败。" >&2
    printf '%s\\n' "\\$CKPT_STATE" > /out/ckpt_retention.state
    if [ "\\$RC" = "0" ]; then exit 4; fi
fi
printf '%s\\n' "\\$CKPT_STATE" > /out/ckpt_retention.state
# ★ 轻量证据（stdout.log / logging.jsonl / args.json / trainer_state.json / gpu 读数）
#   **一律不删**——"删权重不删证据"（宪法 §16 + 验收锚点必须是产物证据）。

ENTRY

# 5) **挂载表（唯一真相源）**：三个数据/代码/模型来源 + 运行目录。
#    目的地全部互不嵌套：子集目录与图像目录是兄弟，它们的父目录留在容器可写层里，
#    因此不存在"在只读绑定挂载之下创建 mountpoint"这一步（旧方案 rc=125 的根因）。
#    图像根的目的地由 train_path 反推（1) 段）——与锚点口径同源，布局变化自动跟随。
MOUNT_SPECS=(
    "$ROOT_DIR|/workspace/graspo|ro"
    "$MODELS_ROOT|{MODELS_CONTAINER_ROOT}|ro"
    "$ELAM_HOST/images|$IMAGE_DIR_CONTAINER|ro"
    "$RUN_DIR/subsets|$TRAIN_PATH_DIR_CONTAINER|ro"
    "$RUN_DIR|/out|rw"
    "$RUN_DIR/entry.sh|/entry.sh|ro"
)
assert_mount_targets_not_under_readonly "${{MOUNT_SPECS[@]}}" || exit 6

# 锁卡采用**单一路径**：宿主侧先由 gpu_lock_guard 断言，容器只认
# NVIDIA_VISIBLE_DEVICES（与容器内 train_worker 的守卫同源）。
# 不同时用 `--gpus`——两者可能互相覆盖，语义有歧义。
DOCKER_ARGS=(
    --rm --runtime=nvidia
    # 共享内存（**≥2 卡必需**）：torchrun 多 rank 按 fd 共享 CPU 张量，docker 默认
    # /dev/shm 仅 64 MiB ⇒ 多卡启动即 `unable to allocate shared memory(shm)` /
    # `Resource temporarily unavailable (11)`（228 实测：4 个 rank 同抛，Train: 0%）。
    # 与 run.sh / tests/e2e/run_matrix.sh 的既有约定逐字一致（§1.4 单一真相源）。
    --ipc=host --shm-size=16g
    -e "NVIDIA_VISIBLE_DEVICES=$GPUS"
    # 档号**显式**进容器：容器内入口脚本（entry.sh）的 ckpt 保留块要按档号定位
    # `<RUN_ROOT>/<T###>`（= 容器内 `/out/<T###>`）。不靠"内层恰好未定义⇒空串"这种
    # 隐性巧合——那是脆弱设计（§2 显式即防呆）。与上面 NVIDIA_VISIBLE_DEVICES 同范式。
    -e "TIER=$TIER"
    -e PYTHONPATH=/workspace/graspo/src
    -e HF_HUB_OFFLINE=1
    -e TOKENIZERS_PARALLELISM=false
)
for SPEC in "${{MOUNT_SPECS[@]}}"; do
    IFS='|' read -r SRC DST MODE <<< "$SPEC"
    if [ "$MODE" = "ro" ]; then
        DOCKER_ARGS+=(-v "$SRC:$DST:ro")
    else
        DOCKER_ARGS+=(-v "$SRC:$DST")
    fi
done
docker run "${{DOCKER_ARGS[@]}}" -w /workspace/graspo "$IMAGE" bash /entry.sh
echo "$?" > "$RUN_DIR/exit_code"
exit "$(cat "$RUN_DIR/exit_code")"
"""


# ── 主流程 ──────────────────────────────────────────────────────────────────


def render_all_tiers(tiers: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """按可表达性把每档渲染成"要落盘的文件"（文件名, 内容）。

    单一真相源（§1.4）：`--dry-run` 自检与正式生成**走同一条渲染路径**——否则
    "自检通过、生成出来的东西不一样"，自检就是假自检（这正是本轮把两处重复循环
    抽成一个函数的原因）。

    - ``blocked`` ⇒ ``T###.blocked.md``（能力落地后可再生成，配 `.blocked` 说明）；
    - ``not_applicable`` ⇒ ``T###.not_applicable.md``（**确实不适用**的合法终态；
      **不产出任何可跑配置**，理由必须可读）；
    - 其余 ⇒ ``T###.yaml``。
    """
    rendered: list[tuple[str, str]] = []
    for tier in tiers:
        status, reason = classify_expressibility(tier)
        tier_id = str(tier["tier_id"])
        if status == "blocked":
            rendered.append((f"{tier_id}.blocked.md", render_blocked_stub(tier, reason or "")))
        elif status == "not_applicable":
            rendered.append(
                (f"{tier_id}.not_applicable.md", render_not_applicable_stub(tier, reason or ""))
            )
        else:
            rendered.append((f"{tier_id}.yaml", render_config_yaml(tier, status, reason)))
    return rendered


def generate(expected_total: int) -> dict[str, Any]:
    tiers = build_ledger()
    assert_ledger(tiers, expected_total)
    expressibility = assert_expressibility(tiers)
    # 生成期显存可行性断言：任一可执行档估算超预算 ⇒ 拒绝生成并报算式。
    assert_memory_feasible(tiers)

    rendered = render_all_tiers(tiers)
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
        "| 档位 | 算法 | 模式 | 后端 | 卡数 | 状态 | 可行性判定 | 配方 | 可行性自检 |",
        "|---|---|---|---|:--:|---|---|---|---|",
    ]
    for row in rows:
        recipe = json.dumps(row["recipe"], ensure_ascii=False) if row["recipe"] else "—"
        lines.append(
            f"| {row['tier_id']} | {row['algorithm']} | {row['mode']} | {row['backend']} "
            f"| {row['cards']} | {row['status']} | {row['verdict']} "
            f"| {recipe} | {row['estimate']} |"
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
    print(f"  显存可行性: {counts['by_feasibility_verdict']}")
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
        rendered = render_all_tiers(tiers)
        assert_no_legacy_terms(rendered)
        verdict_counts: dict[str, int] = {}
        for tier in tiers:
            verdict, _ = feasibility_verdict(tier)
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        print(
            f"dry-run OK: {len(tiers)} tiers, expressibility={expressibility}, "
            f"feasibility_verdict={verdict_counts}"
        )
        print(
            "说明：`infeasible` 档已被拒绝生成；`unmeasured` 档**允许生成但不声称可行**"
            "（未测算的量只能靠上机量出来，见上方待测量清单）；`blocked` 档不产出配置；"
            "`not_applicable` 档**确实不适用**（合法终态）——既不产出配置，也不参与可行性判断。"
        )
        return 0

    result = generate(args.assert_count)
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
