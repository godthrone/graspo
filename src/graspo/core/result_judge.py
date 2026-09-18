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
- **A6** 数值健康（loss 与 grad_norm 全程 finite、最终 loss 不高于初始；NaN/Inf 即不通过）。

证据缺失一律判**不通过**（fail-closed）——不能因为"读不到"就默认通过。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from dataclasses import dataclass
from enum import StrEnum

# ── 常量 ────────────────────────────────────────────────────────────────────

#: A5 四件套产物：配置备份 / 训练日志 / 可恢复 checkpoint / 运行指标。
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "config_backup",
    "training_log",
    "checkpoint",
    "metrics",
)

#: 正式记录门槛（capability-matrix §6）：≥1 epoch 且 ≥5 optimizer step。
MIN_OPTIMIZER_STEPS = 5

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
A4_MEASURED_BF16_FINAL_LOSS_DRIFT = 1.0302e-3
#: 终态 loss 的绝对容差 = 实测量级 ×≈10 的余量。
#: ⚠️ **余量必须随实测更新**：若今后实测漂移量级上升（更长序列 / 更大模型 / 更多步），
#: 必须先重测、更新 :data:`A4_MEASURED_BF16_FINAL_LOSS_DRIFT` 并重新推导本常量，
#: **不得**为了方便让测试变绿而放宽（``tests/core/test_result_judge.py`` 两侧都锁住）。
A4_FINAL_LOSS_TOLERANCE_ABS = 1e-2

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
    UNCLASSIFIED = "未分类（需人工判定）"


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
    """

    criterion: str
    passed: bool
    detail: str
    evidence_missing: bool = False


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
    nonfinite_skips: int | None = None
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


#: 台账状态（唯一真相源）。三态而非两态的理由见 :class:`FailureClass` 的
#: ``EVIDENCE_GAP`` 条目与 :func:`judge_tier`：取证缺口既不是通过，也不是训练失败。
LEDGER_PASS = "✅ 通过"
LEDGER_FAIL = "❌ 失败"
LEDGER_INDETERMINATE = "⚠ 不可判定（取证缺口）"


@dataclass(frozen=True, slots=True)
class TierJudgement:
    """一档的最终判定，可直接落成台账记录。"""

    tier_id: str
    criteria: tuple[CriterionResult, ...]
    passed: bool
    failure_class: FailureClass | None
    counts_toward_max_context: bool
    note: str

    @property
    def indeterminate(self) -> bool:
        """未通过是否**完全**由取证缺口造成（没有任何判据被证据否定）。"""
        failing = [item for item in self.criteria if not item.passed]
        return bool(failing) and all(item.evidence_missing for item in failing)

    @property
    def ledger_status(self) -> str:
        if self.passed:
            return LEDGER_PASS
        if self.indeterminate:
            return LEDGER_INDETERMINATE
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
    - 步数：``optimizer_steps >= MIN_OPTIMIZER_STEPS``（§6 正式门槛）；
    - LoRA：至少一个 ``lora_b`` 权重非零（LoRA B 初始为 0，非零即证明被更新）；
    - 全参：终态权重与基座权重不同（由抽取层给出 ``weight_changed``）。

    检查顺序（为什么"跳过"排在门槛之前）：门槛报"步数不够"会把一次**数值崩坏**
    误读成"冒烟太短"，从而诱导重复跑而不是修缺陷。跳过是更硬的事实。
    """
    if evidence.nonfinite_skips:
        return CriterionResult(
            "A2",
            False,
            f"训练未真推进：累计 {evidence.nonfinite_skips} 次因非有限梯度跳过"
            "优化器步（权重冻结，§3.4 防线：不得记为成功）",
        )
    if evidence.optimizer_steps_per_step is not None:
        stalled = [
            index
            for index, count in enumerate(evidence.optimizer_steps_per_step)
            if count <= 0
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
    if evidence.optimizer_steps < MIN_OPTIMIZER_STEPS:
        return CriterionResult(
            "A2",
            False,
            f"optimizer step={evidence.optimizer_steps} < 门槛 {MIN_OPTIMIZER_STEPS}",
        )
    if evidence.weight_changed is None:
        return CriterionResult(
            "A2",
            False,
            f"缺少权重变化证据（tuner_type={evidence.tuner_type}，fail-closed）",
            evidence_missing=True,
        )
    if not evidence.weight_changed:
        mode = "LoRA（lora_b 全零）" if evidence.tuner_type == "lora" else "全参（与基座一致）"
        return CriterionResult("A2", False, f"权重未变化：{mode}")
    plan = (
        f"，计划 {evidence.steps_declared_total}"
        if evidence.steps_declared_total is not None
        else ""
    )
    return CriterionResult(
        "A2",
        True,
        f"optimizer step={evidence.optimizer_steps}{plan}，权重已变化"
        f"（tuner_type={evidence.tuner_type}）",
    )


def judge_a3(evidence: RunEvidence) -> CriterionResult:
    """A3 checkpoint 可被重新加载。"""
    if evidence.checkpoint_reloadable is None:
        return CriterionResult(
            "A3", False, "缺少 checkpoint 重载证据（fail-closed）", evidence_missing=True
        )
    if not evidence.checkpoint_reloadable:
        return CriterionResult("A3", False, "checkpoint 无法重新加载")
    return CriterionResult("A3", True, "checkpoint 重载成功")


def judge_a4(first: RunEvidence, second: RunEvidence | None) -> CriterionResult:
    """A4 同 config 同 seed 双跑一致（**两条并列子检查**，见下）。

    一致性口径（§10.1 复现定义 + §6 排除项）：

    - 两跑都成功、``optimizer_steps`` 相同、``weight_changed`` 结论相同；
    - **子检查①（零容差）**：**首步 loss 必须逐位相同**（``losses[0]`` 在两端
      ``first_logged_step == 1`` 且 ``==`` 严格相等）。
      首步 loss 只取决于 种子 / 初始化 / 数据顺序 / 首批样本——全是**可控**因素，
      因此零容忍：这是"去掉种子""打乱数据顺序""换初始化"这类破坏的**即时指纹**。
    - **子检查②（实测容差）**：终态 loss 之差 ≤ :data:`A4_FINAL_LOSS_TOLERANCE_ABS`
      （= :data:`A4_MEASURED_BF16_FINAL_LOSS_DRIFT` 的约 10 倍）。
      依据：宪法 §6 明文把"GPU 内核选择等**无法控制**的差异"排除在复现破坏之外；
      bf16 归约顺序正属此类（2026-09-19 T013 实测：首步逐位相同、终态累计 1.0302e-3）。
      **不得**把这个容差当成"放宽判据"——它由实测推导，且子检查①仍在零容忍地把守
      可控差异；两者缺一不可（见 ``tests/core/test_result_judge.py`` 的四条守卫用例）。

    **诚实性**：拿不到步号（``first_logged_step is None``）或多步累积后无法确认"首步"时，
    子检查①**如实声明"未适用"**，不得假装做过（detail 里写明），此时①不参与判定，
    但②照常生效。缺第二跑/loss 证据 → 不通过（fail-closed）。
    """
    if second is None:
        return CriterionResult(
            "A4", False, "缺少第二跑证据（fail-closed）", evidence_missing=True
        )
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
        return CriterionResult(
            "A4", False, "权重变化结论不一致（一次变化一次未变化）"
        )
    if not first.losses or not second.losses:
        return CriterionResult(
            "A4", False, "缺少 loss 序列证据（fail-closed）", evidence_missing=True
        )

    # ── 子检查①（零容差）：首步 loss 逐位相同 ────────────────────────────────
    first_step_checked = False
    if first.first_logged_step == 1 and second.first_logged_step == 1:
        first_initial = first.losses[0]
        second_initial = second.losses[0]
        if not isinstance(first_initial, float) or not isinstance(second_initial, float):
            return CriterionResult(
                "A4",
                False,
                "首步 loss 无读数（loss 字段缺失），零容差子检查无法进行（fail-closed）",
            )
        first_step_checked = True
        if first_initial != second_initial:
            return CriterionResult(
                "A4",
                False,
                f"首步 loss 不一致（零容差子检查）：{first_initial!r} vs {second_initial!r} "
                "—— 首步只取决于 种子/初始化/数据顺序/首批样本，全是**可控**因素 ⇒ "
                "这是可复现性被破坏的指纹（§10.1），不是 §6 排除的 GPU 内核差异",
            )
    step_note = (
        "首步 loss 逐位相同；"
        if first_step_checked
        else "首步零容差子检查未适用（拿不到步号或首步无读数）；"
    )

    # ── 子检查②（实测容差）：终态 loss ──────────────────────────────────────
    first_loss = first.losses[-1]
    second_loss = second.losses[-1]
    if not isinstance(first_loss, float) or not isinstance(second_loss, float):
        # 末步 loss 无读数（MISSING 哨兵）⇒ 无法比较，fail-closed。
        # （不让哨兵流进下面的算术：那是 TypeError，会把"判不通过"变成"崩溃"。）
        return CriterionResult(
            "A4", False, "末步 loss 无读数（loss 字段缺失），无法做双跑一致性比较（fail-closed）"
        )
    delta = abs(first_loss - second_loss)
    if delta > A4_FINAL_LOSS_TOLERANCE_ABS:
        return CriterionResult(
            "A4",
            False,
            f"最终 loss 不一致：{first_loss:.6g} vs {second_loss:.6g}"
            f"（差 {delta:.3g} > 容差 {A4_FINAL_LOSS_TOLERANCE_ABS:.3g}；"
            f"该容差按实测 bf16 漂移 {A4_MEASURED_BF16_FINAL_LOSS_DRIFT:.3g} 推导）",
        )
    return CriterionResult(
        "A4",
        True,
        f"双跑一致：{step_note}final loss {first_loss:.6g} vs {second_loss:.6g}"
        f"（差 {delta:.3g} ≤ 容差 {A4_FINAL_LOSS_TOLERANCE_ABS:.3g}）",
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
    """A6 数值健康：loss / grad_norm 全程 finite，最终 loss ≤ 初始 loss。

    **三种情形必须分清（F-4 §④.1-4 的核心）**：

    1. **真读到 NaN/Inf** ⇒ 判**不通过**，明细为「出现 NaN/Inf」⇒ 分类 **数值异常**。
       这是关于训练数值的**事实断言**（F-4 那个 run 的 step2 就是真 NaN）。
    2. **loss 字段无值**（``loss: null`` / 缺失）⇒ 判**不通过**（fail-closed），
       明细为 :data:`NUMERIC_INDETERMINATE_DETAIL` ⇒ 分类 **未分类（需人工判定）**。
       我们**没有读数**，所以不得断言"数值异常"——那会把排查引向数值问题，而真因可能
       是进程被杀、超时或环境问题。这**不是**"偶发地依赖 null 被解析成 NaN"。
    3. 两者同时出现（F-4 实测正是如此）⇒ **真 NaN 优先**，分类数值异常。
       已确证的数值崩坏不得被"某几步没有读数"降格。

    三种情形一律**不通过、不计入最大可行上下文**（fail-closed 不变）。
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
        value
        for value in evidence.losses
        if isinstance(value, float) and not math.isfinite(value)
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
    initial = evidence.losses[0]
    final = evidence.losses[-1]
    if final > initial:
        return CriterionResult(
            "A6", False, f"最终 loss {final:.6g} 高于初始 loss {initial:.6g}"
        )
    return CriterionResult(
        "A6",
        True,
        f"数值健康：loss {initial:.6g} → {final:.6g}，{len(evidence.losses)} 个样本全程 finite",
    )


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
_NUMERIC_ANOMALY_DETAILS: tuple[str, ...] = ("NaN/Inf", "高于初始")


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
    """A6 是否因**数值异常**未通过（NaN/Inf，或最终 loss 高于初始）。

    单独抽成函数是为了给 ``classify_failure`` 一个显式、可测的**优先级判据**：
    数值健康是比"日志里出现 OOM 字样"更硬的事实（见该函数的排序说明）。
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

    1. **超时**（``timed_out`` / ``exit_code == 124``）⇒ ``TIMEOUT``；
    2. **数值异常**（A6 明细含 "NaN/Inf" 或 "高于初始"）⇒ ``NUMERIC_ANOMALY``；
    3. **数值不可判定**（A6 明细含 :data:`NUMERIC_INDETERMINATE_DETAIL`，即
       ``loss: null`` / 字段缺失 / 全程无读数）⇒ ``UNCLASSIFIED``（需人工判定）；
    4. **显式非有限梯度标记**（训练器硬失败的 ``非有限梯度`` /
       ``non-finite gradient`` / ``nonfinite_loss_or_grad``）⇒ ``NUMERIC_ANOMALY``；
       这一步由 ``_LOG_PATTERNS`` **表头第一条**实现（2026-09-18 🟡-4 修正前的
       旧序把它排在第 5 位，与本节描述不符，已修正）；
    5. **其余日志模式**（框架未实现 / 配置非法 / 数据问题 / 通信硬件 / 真 OOM）——
       即 ``_LOG_PATTERNS`` 中**数值异常之后的**条目，按表内顺序匹配；
    6. 非零退出码且未匹配 ⇒ ``UNCLASSIFIED``。

    **每一级的可达性与触发条件**（不允许存在"写着却永不触发"的分级）：

    | # | 触发条件（可判定的事实） | 归类 | 可达 |
    |---|---|---|:--:|
    | 1 | ``timed_out`` 或 ``exit_code == 124`` | ``TIMEOUT`` | ✔ |
    | 2 | loss/grad_norm 序列里**真读到** NaN/Inf，或最终 loss > 初始 | ``NUMERIC_ANOMALY`` | ✔ |
    | 3 | loss 字段无值（``MISSING_SENTINEL``）且**无任何**真 NaN | ``UNCLASSIFIED`` | ✔ |
    | 4 | 日志出现 ``非有限梯度`` 等硬失败标记 | ``NUMERIC_ANOMALY`` | ✔ |
    | 5 | 日志匹配到 OOM / 未实现 / 配置 / 数据 / 通信模式（且不含第 4 步标记） | 对应类型 | ✔ |
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


def judge_tier(
    first: RunEvidence,
    second: RunEvidence | None = None,
    *,
    context_length: int | None = None,
) -> TierJudgement:
    """对一档做 A1–A6 总判定，产出可直接落台账的记录。"""
    a1 = judge_a1(first)
    a2 = judge_a2(first)
    a3 = judge_a3(first)
    a4 = judge_a4(first, second)
    a5 = judge_a5(first)
    a6 = judge_a6(first)
    criteria = (a1, a2, a3, a4, a5, a6)
    passed = all(result.passed for result in criteria)

    failure_class: FailureClass | None = None
    note = ""
    if not passed:
        failing = [result for result in criteria if not result.passed]
        failed_names = ", ".join(result.criterion for result in failing)
        substantive = classify_failure(first, a6)
        # 取证缺口 vs 真失败：全部不通过的判据都只因"读不到证据"、且日志里**没有任何**
        # 实质性失败信号 ⇒ 这是采集侧的缺口，**不得记成训练失败**
        # （F-D 实测：一次 exit=0、100 步、loss 正常下降的训练被记成「❌ 失败」，
        #  只因为 collector 读不到 ms-swift 的产物布局）。
        # 注意方向不变：`passed` 仍然 False（fail-closed，读不到绝不当成通过）。
        # 也注意优先级不变：日志里的硬失败信号（OOM / 非有限梯度 / 数据问题…）
        # 是**来自这次运行本身**的证据，优先级高于"采集侧读不到"。
        if failing and all(result.evidence_missing for result in failing) and substantive is None:
            failure_class = FailureClass.EVIDENCE_GAP
            note = (
                f"未过判据：{failed_names}；**取证不足/不可判定**（判定器读不到证据，"
                "不是训练失败）；修复采集后复评"
            )
        else:
            failure_class = substantive if substantive is not None else FailureClass.UNCLASSIFIED
            note = f"未过判据：{failed_names}；失败类型：{failure_class}"
            if not counts_toward_max_context(failure_class):
                note += "（不计入最大可行上下文，修复后重测）"
            else:
                note += (
                    f"（真 OOM：当前上下文 {context_length} 可作为该档最大可行上下文的候选，"
                    "需按递增加长法确认边界）"
                    if context_length is not None
                    else "（真 OOM）"
                )

    return TierJudgement(
        tier_id=first.tier_id,
        criteria=criteria,
        passed=passed,
        failure_class=failure_class,
        counts_toward_max_context=counts_toward_max_context(failure_class),
        note=note,
    )


#: 台账 `max_context` 的**取值口径标记**——回答"这一列的数字从哪来"（§1.4）。
#: 为什么必须有它：真 OOM 档写进该列的是**边界候选**（该 run 整体未通过），
#: 通过档写进去的是**实测可行值**；两者都在同一列，不标口径就会被下游读成
#: "这个长度跑得通"，从而把不可行的长度当成能力上限（§7.1 第 4 条）。
MAX_CONTEXT_KIND_FEASIBLE = "实测通过"
MAX_CONTEXT_KIND_OOM_BOUNDARY = "真 OOM 边界候选"


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
) -> dict[str, object]:
    """把判定落成 capability-matrix §7 台账的一行（可直接填表）。

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
        "status": judgement.ledger_status,
        "failure_class": judgement.failure_class,
        "note": judgement.note,
        "date": date,
    }


def loss_series_summary(values: Sequence[float]) -> str:
    """诊断用：把 loss/grad_norm 序列压成一行摘要。"""
    if not values:
        return "empty"
    return f"n={len(values)} first={values[0]:.6g} last={values[-1]:.6g}"


#: 允许调用方遍历的失败类型清单（含兜底），供 CLI 打印。
ALL_FAILURE_CLASSES: tuple[FailureClass, ...] = tuple(FailureClass)
