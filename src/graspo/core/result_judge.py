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


class FailureClass(StrEnum):
    """失败分类。前 7 个是规定的分类；``UNCLASSIFIED`` 是安全兜底。

    兜底存在的理由：把"分类不出来"硬塞进 7 类中的某一类，可能把框架 bug
    误标成配置非法或数值异常，从而在修复前被当成结论。``UNCLASSIFIED``
    永远不计入最大可行上下文，必须人工判定。
    """

    REAL_OOM = "真 OOM"
    FRAMEWORK_UNIMPLEMENTED = "框架未实现"
    CONFIG_INVALID = "配置非法"
    DATA_PROBLEM = "数据问题"
    COMM_HARDWARE = "通信硬件"
    NUMERIC_ANOMALY = "数值异常"
    TIMEOUT = "超时"
    UNCLASSIFIED = "未分类（需人工判定）"


#: 只有这一类失败允许被解释为"上下文太长"。
MAX_CONTEXT_FAILURE_CLASSES: frozenset[str] = frozenset({FailureClass.REAL_OOM})

_LOG_PATTERNS: tuple[tuple[FailureClass, tuple[re.Pattern[str], ...]], ...] = (
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
    """单条判据的结果。"""

    criterion: str
    passed: bool
    detail: str


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
    def ledger_status(self) -> str:
        return "✅ 通过" if self.passed else "❌ 失败"

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
        return CriterionResult("A1", False, "缺少 exit_code 证据（fail-closed）")
    if evidence.exit_code != 0:
        return CriterionResult("A1", False, f"exit={evidence.exit_code} != 0")
    return CriterionResult("A1", True, "exit=0")


def judge_a2(evidence: RunEvidence) -> CriterionResult:
    """A2 训练步真推进 且 权重真变化（LoRA / 全参两套判据）。

    - 步数：``optimizer_steps >= MIN_OPTIMIZER_STEPS``（§6 正式门槛）；
    - LoRA：至少一个 ``lora_b`` 权重非零（LoRA B 初始为 0，非零即证明被更新）；
    - 全参：终态权重与基座权重不同（由抽取层给出 ``weight_changed``）。
    """
    if evidence.optimizer_steps is None:
        return CriterionResult("A2", False, "缺少 optimizer step 证据（fail-closed）")
    if evidence.optimizer_steps < MIN_OPTIMIZER_STEPS:
        return CriterionResult(
            "A2",
            False,
            f"optimizer step={evidence.optimizer_steps} < 门槛 {MIN_OPTIMIZER_STEPS}",
        )
    if evidence.weight_changed is None:
        return CriterionResult(
            "A2", False, f"缺少权重变化证据（tuner_type={evidence.tuner_type}，fail-closed）"
        )
    if not evidence.weight_changed:
        mode = "LoRA（lora_b 全零）" if evidence.tuner_type == "lora" else "全参（与基座一致）"
        return CriterionResult("A2", False, f"权重未变化：{mode}")
    return CriterionResult(
        "A2",
        True,
        f"optimizer step={evidence.optimizer_steps}，权重已变化（tuner_type={evidence.tuner_type}）",
    )


def judge_a3(evidence: RunEvidence) -> CriterionResult:
    """A3 checkpoint 可被重新加载。"""
    if evidence.checkpoint_reloadable is None:
        return CriterionResult("A3", False, "缺少 checkpoint 重载证据（fail-closed）")
    if not evidence.checkpoint_reloadable:
        return CriterionResult("A3", False, "checkpoint 无法重新加载")
    return CriterionResult("A3", True, "checkpoint 重载成功")


def judge_a4(first: RunEvidence, second: RunEvidence | None) -> CriterionResult:
    """A4 同 config 同 seed 双跑一致。

    一致性口径（§10.1 复现定义）：两跑都成功、optimizer step 相同、最终 loss
    在相对容差内相同、权重变化结论相同。缺第二跑证据 → 不通过（fail-closed）。
    """
    if second is None:
        return CriterionResult("A4", False, "缺少第二跑证据（fail-closed）")
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
        return CriterionResult("A4", False, "缺少 loss 序列证据（fail-closed）")
    first_loss = first.losses[-1]
    second_loss = second.losses[-1]
    tolerance = 1e-3 * max(1.0, abs(first_loss))
    if abs(first_loss - second_loss) > tolerance:
        return CriterionResult(
            "A4",
            False,
            f"最终 loss 不一致：{first_loss:.6g} vs {second_loss:.6g}（容差 {tolerance:.3g}）",
        )
    return CriterionResult("A4", True, f"双跑一致（final loss {first_loss:.6g}）")


def judge_a5(evidence: RunEvidence) -> CriterionResult:
    """A5 四件套产物落盘。"""
    missing = [name for name in REQUIRED_ARTIFACTS if not evidence.artifacts_present.get(name)]
    if missing:
        return CriterionResult("A5", False, f"缺少产物：{', '.join(missing)}")
    return CriterionResult("A5", True, f"四件套齐全：{', '.join(REQUIRED_ARTIFACTS)}")


def judge_a6(evidence: RunEvidence) -> CriterionResult:
    """A6 数值健康：loss / grad_norm 全程 finite，最终 loss ≤ 初始 loss。"""
    if not evidence.losses:
        return CriterionResult("A6", False, "缺少 loss 序列证据（fail-closed）")
    if not evidence.grad_norms:
        return CriterionResult("A6", False, "缺少 grad_norm 序列证据（fail-closed）")

    bad_losses = [value for value in evidence.losses if not math.isfinite(value)]
    bad_grads = [value for value in evidence.grad_norms if not math.isfinite(value)]
    if bad_losses or bad_grads:
        return CriterionResult(
            "A6",
            False,
            f"出现 NaN/Inf：loss {len(bad_losses)} 个、grad_norm {len(bad_grads)} 个",
        )
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


#: A6 明细里出现这些字样 ⇒ 数值异常（唯一真相源，见 ``judge_a6``）。
_NUMERIC_ANOMALY_DETAILS: tuple[str, ...] = ("NaN/Inf", "高于初始")


def numeric_anomaly(a6: CriterionResult) -> bool:
    """A6 是否因**数值异常**未通过（NaN/Inf，或最终 loss 高于初始）。

    单独抽成函数是为了给 ``classify_failure`` 一个显式、可测的**优先级判据**：
    数值健康是比"日志里出现 OOM 字样"更硬的事实（见该函数的排序说明）。
    """
    return not a6.passed and any(mark in a6.detail for mark in _NUMERIC_ANOMALY_DETAILS)


def classify_failure(evidence: RunEvidence, a6: CriterionResult) -> FailureClass | None:
    """把一次未通过的运行归类。

    返回 ``None`` 表示"没有失败"（exit=0、无 NaN、非超时）。匹配不到则返回
    ``UNCLASSIFIED``（不计入最大可行上下文）。

    **优先级（为什么数值异常排在日志模式之前）**：A6 的 NaN/Inf / loss 上升是
    从证据序列直接算出的事实，而"日志里出现 `CUDA out of memory`"只是文本证据——
    一次真 OOM 之后常伴随 NaN/数值崩溃，此时真正的结论是"数值异常"，
    不能因此把该档上下文长度记进"最大可行上下文"（那会放宽"只有真 OOM 才计入"
    这条硬要求并污染能力边界）。因此：超时 → **数值异常** → 其余日志模式。
    """
    if evidence.timed_out or evidence.exit_code == 124:
        return FailureClass.TIMEOUT
    if numeric_anomaly(a6):
        return FailureClass.NUMERIC_ANOMALY
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
        failure_class = classify_failure(first, a6)
        if failure_class is None:
            failure_class = FailureClass.UNCLASSIFIED
        failed_names = ", ".join(result.criterion for result in criteria if not result.passed)
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
    """把判定落成 capability-matrix §7 台账的一行（可直接填表）。"""
    return {
        "tier_id": judgement.tier_id,
        "model": model,
        "algorithm": algorithm,
        "mode": mode,
        "backend": backend,
        "cards": cards,
        "max_context": max_context if judgement.passed else None,
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
