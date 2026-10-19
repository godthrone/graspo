"""聚合与 Δ 判定：把逐样本明细变成结论，并做基线/训练后配对。

**职责**：
1. 把 ``SampleRecord`` 明细聚合成 ``EvalSummary``（准确率 + 分工具/分动作诊断
   + 图像重叠剔除口径）。
2. 读回已落盘的产物（``eval_report.json``）。
3. Δ 配对：给定 base 与训练后两份产物，算绝对百分点差值并与阈值比较。

**本文件不负责**：判定单样本对错（`criteria.py`）、发请求（`vllm_client.py`）、
锁卡/起服务（`guard.py` / 调用方）。聚合是纯函数，可在 CPU 上单测。

**Δ 口径（冻结口径，2026-10-15 起；以真相源为准）**

``Δ = acc_after − acc_before``，**绝对百分点**；``acc_before`` 的基线锚点是
**base 模型**（不是 SFT 产物）。判据② **逐算法分列**（一律**点估计**，
⛔ 不做假设检验、不引置信区间 / 显著性）：

- SFT    ⇒ ``Δ ≥ +10.0 pp``（epoch=4）
- GRASPO ⇒ ``Δ ≥ +5.0 pp``（epoch=8）
- CPT    ⇒ ``after ≥ base``（不劣化，non-inferiority；epoch=2）
- OPD    ⇒ ``Δ ≥ +5.0 pp``（criterion = ``delta_pp``；epoch=2）——★ 2026-10-19 口径变更（原为不劣化）

★ 通过线**不在本模块**、也不在任何配置里：唯一真相源 =
``docs/capability-matrix.data.json`` 的 ``meta.criteria_caliber.threshold.by_method``，
由 :mod:`graspo.eval.thresholds` 现读（⛔ 不得新增第二份数值副本）。
"CPT 的『训练 loss 正常下降』另有强制方"不属本模块（本模块只判 ``after ≥ base``）。

**量纲纪律**：SFT 与 GRASPO 比的都是 ``Δ``（百分点）——⛔ 不得拿 ``after`` 的
**绝对准确率**去比通过线（那会把"底子本来就好"误判成"训练有效"，结论必错）。

**为什么 Δ 必须配对同一个数据集**

两个准确率只有在**同一数据、同一口径、同一温度**下相减才有意义。本模块在配对时
强制校验：数据集 sha256 一致、口径版本一致、温度都为 0、有效样本数一致。
任何一条不满足即抛出——宁可拒绝出结论，也不给出一个不可比的 Δ。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from graspo.eval.criteria import ACCURACY_CRITERIA_VERSION, accuracy
from graspo.eval.schema import EvalReport, EvalSummary
from graspo.eval.thresholds import KNOWN_METHODS, frozen_method_threshold


class DeltaError(RuntimeError):
    """Δ 配对失败（两份产物不可比 / 算法名不合法）。"""


@dataclass(slots=True)
class DeltaResult:
    """Δ 配对结论（判据② **逐算法分列**，一律点估计）。

    通过线**不是**本对象的输入：``threshold_pp`` / ``criterion`` / ``pass_line_text``
    都是 :func:`compute_delta` 从唯一真相源现读后**照抄**进来的留证字段（§2.2 显式即可见）。
    """

    acc_before_percent: float
    acc_after_percent: float
    delta_percentage_points: float
    #: 判据② 用的算法（``sft`` / ``graspo`` / ``cpt`` / ``opd``）——决定用哪条通过线。
    method: str
    #: 冻结口径登记的判据类型：``delta_pp``（SFT / GRASPO / OPD）或 ``non_inferiority``（CPT）。
    criterion: str
    #: 该通过线的数值（百分点），逐字来自真相源（SFT 10.0 / GRASPO 5.0 / OPD 5.0 / CPT 0.0）。
    threshold_pp: float
    #: 真相源里那条通过线的原文（如 ``after − base ≥ +5.0 pp``），便于人读核对。
    pass_line_text: str
    #: 按 ``criterion`` 对 ``Δ`` 的判定（点估计；⛔ 不做假设检验）。
    passed: bool
    before_run_id: str
    after_run_id: str
    comparability_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "acc_before_percent": round(self.acc_before_percent, 4),
            "acc_after_percent": round(self.acc_after_percent, 4),
            "delta_percentage_points": round(self.delta_percentage_points, 4),
            "method": self.method,
            "criterion": self.criterion,
            "threshold_pp": self.threshold_pp,
            "pass_line_text": self.pass_line_text,
            "passed": self.passed,
            "before_run_id": self.before_run_id,
            "after_run_id": self.after_run_id,
            "comparability_notes": self.comparability_notes,
        }


def summarize(
    records: list[dict[str, Any]],
    *,
    overlap_indices: set[int] | None = None,
) -> EvalSummary:
    """把逐样本明细聚合成 ``EvalSummary``。

    分母口径与 v3 ``stat_matrix.py`` 一致：``error`` 非空的样本既不入分子也不入
    分母（``n_err`` 剔除）。

    Args:
        records: ``SampleRecord.to_dict()`` 形式的明细列表。
        overlap_indices: 图像集被训练集完全覆盖的样本下标。``None`` = **未做分析**
            （数值字段留 ``None`` 且 ``overlap_analysis_performed=False``）；
            空集 ``set()`` = **做了分析、重叠集为空**（数值字段为 0 / 0.0 且
            ``overlap_analysis_performed=True``）。两者在产物里**必须可区分**。

    Returns:
        ``EvalSummary``。
    """
    total = len(records)
    correct = 0
    valid = 0
    errors = 0
    by_tool: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_action: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    overlap_correct = 0
    overlap_valid = 0

    for record in records:
        if record.get("error"):
            errors += 1
            continue
        valid += 1
        hit = bool(record.get("all_right"))
        if hit:
            correct += 1
        by_tool[str(record.get("gt_name") or "unknown")][0] += int(hit)
        by_tool[str(record.get("gt_name") or "unknown")][1] += 1
        by_action[str(record.get("gt_action") or "unknown")][0] += int(hit)
        by_action[str(record.get("gt_action") or "unknown")][1] += 1

        if overlap_indices is not None and int(record["sample_index"]) not in overlap_indices:
            overlap_valid += 1
            overlap_correct += int(hit)

    accuracy_value = accuracy(correct, valid)
    excluding: float | None = None
    excluded_count: int | None = None
    overlap_performed = overlap_indices is not None
    if overlap_performed:
        # 分析执行过：重叠集为空时这里是 0.0 / 0，与主口径同值——这是**合法**的
        # "做了且结果为零"，靠 overlap_analysis_performed 与"没做"区分。
        excluding = round(100.0 * accuracy(overlap_correct, overlap_valid), 4)
        excluded_count = total - overlap_valid
    return EvalSummary(
        overlap_analysis_performed=overlap_performed,
        sample_count_total=total,
        sample_count_valid=valid,
        sample_count_error=errors,
        correct=correct,
        incorrect=valid - correct,
        accuracy=round(accuracy_value, 6),
        accuracy_percent=round(100.0 * accuracy_value, 4),
        by_tool={key: value for key, value in sorted(by_tool.items())},
        by_action={key: value for key, value in sorted(by_action.items())},
        accuracy_percent_excluding_overlap=excluding,
        overlap_excluded_count=excluded_count,
    )


def load_report(path: str | Path) -> EvalReport:
    """读回一份已落盘的评测产物。

    Raises:
        DeltaError: 文件不存在或结构不符合契约。
    """
    report_path = Path(path)
    if not report_path.is_file():
        raise DeltaError(f"eval report not found: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DeltaError(f"{report_path} is not valid JSON: {exc}") from None
    try:
        return EvalReport.model_validate(payload)
    except ValidationError as exc:  # 契约不符 → 转成可读信息（§13.1 不吞异常）
        raise DeltaError(
            f"{report_path} does not match the eval artifact contract: {exc}"
        ) from None


def compute_delta(
    before: EvalReport,
    after: EvalReport,
    *,
    method: str | None = None,
    expect_role_before: str = "base",
    role_after: str | None = None,
) -> DeltaResult:
    """算出 Δ 并按**冻结口径**给出结论（判据② 逐算法分列，一律点估计）。

    Args:
        before: 基线产物。**必须是 base 模型**（用户拍板），默认强制校验。
        after: 训练后产物。
        method: 判据② 适用的算法（``sft`` / ``graspo`` / ``cpt`` / ``opd``）。
            ``None`` = 由 after 角色推断：``role == "sft"`` ⇒ ``sft``，否则 ``graspo``。
            ★ CPT / OPD **必须显式给出**（``role`` 字面量里没有它们；不显式给就会
            被当成 GRASPO 判）。
        expect_role_before: 期望的基线角色；``""`` 表示不校验。
        role_after: 明确指定 after 的角色；``None`` 表示沿用产物自带角色。

    Returns:
        ``DeltaResult``：``method`` / ``criterion`` / ``threshold_pp`` / ``pass_line_text``
        全部照抄自唯一真相源，``passed`` = 按该算法的判定式对 ``Δ`` 的点估计判定。

    Raises:
        DeltaError: 两份产物不可比（数据集/口径/温度/有效样本数不一致）、
            基线角色不符合要求，或 ``method`` 不是冻结口径认得的那四个。
        FrozenCaliberError: 冻结口径真相源读不到 / 形状不对（fail-closed）。
    """
    notes = _comparability_notes(before, after)
    if notes:
        raise DeltaError(
            "eval reports are not comparable — refusing to compute delta:\n  " + "\n  ".join(notes)
        )
    if expect_role_before and before.model.role != expect_role_before:
        raise DeltaError(
            f"baseline report role is {before.model.role!r} but {expect_role_before!r} is required "
            "(the GRASPO delta anchor is the base model, by user ruling)"
        )

    role = role_after or after.model.role
    resolved_method = method or ("sft" if role == "sft" else "graspo")
    if resolved_method not in KNOWN_METHODS:
        # 算法名不认识属"调用参数错"；真相源读不到是另一回事（让 FrozenCaliberError 冒上去）。
        raise DeltaError(
            f"未知算法 {resolved_method!r}：冻结口径只认 {KNOWN_METHODS}（判据② 逐算法分列）"
        )
    threshold = frozen_method_threshold(resolved_method)

    # 量纲纪律：SFT / GRASPO 比的都是 Δ（百分点），⛔ 不是 after 的绝对准确率。
    delta = after.summary.accuracy_percent - before.summary.accuracy_percent
    return DeltaResult(
        acc_before_percent=before.summary.accuracy_percent,
        acc_after_percent=after.summary.accuracy_percent,
        delta_percentage_points=delta,
        method=resolved_method,
        criterion=threshold.criterion,
        threshold_pp=threshold.value_pp,
        pass_line_text=threshold.pass_line_text,
        passed=threshold.judge(delta),
        before_run_id=before.run_id,
        after_run_id=after.run_id,
        comparability_notes=[
            f"dataset sha256 match: {before.dataset.sha256[:12]}…",
            f"criteria version: {before.criteria.version}",
            f"temperature: {before.decoding.temperature} / {after.decoding.temperature}",
            f"valid samples: {before.summary.sample_count_valid}",
        ],
    )


def _comparability_notes(before: EvalReport, after: EvalReport) -> list[str]:
    """列出不满足可比性的项；空列表 = 可比。"""
    problems: list[str] = []
    if before.dataset.sha256 != after.dataset.sha256:
        problems.append(
            f"dataset sha256 differs: {before.dataset.sha256} vs {after.dataset.sha256} "
            "(delta across different data is meaningless)"
        )
    if before.criteria.version != after.criteria.version:
        problems.append(
            f"criteria version differs: {before.criteria.version} vs {after.criteria.version}"
        )
    if before.criteria.version != ACCURACY_CRITERIA_VERSION:
        problems.append(
            f"baseline report used criteria {before.criteria.version}, "
            f"current implementation is {ACCURACY_CRITERIA_VERSION}"
        )
    for report, label in ((before, "before"), (after, "after")):
        if report.decoding.temperature != 0.0:
            problems.append(f"{label} report temperature is {report.decoding.temperature}, not 0")
    if before.decoding.max_tokens != after.decoding.max_tokens:
        problems.append(
            f"max_tokens differs: {before.decoding.max_tokens} vs {after.decoding.max_tokens}"
        )
    if before.summary.sample_count_valid != after.summary.sample_count_valid:
        problems.append(
            "valid sample count differs: "
            f"{before.summary.sample_count_valid} vs {after.summary.sample_count_valid} "
            "(error samples shift the denominator)"
        )
    return problems
