"""A1–A6 判定与失败分类的单元测试（纯逻辑，不触 GPU、不读文件）。

重点覆盖硬要求：只有"真 OOM"计入最大可行上下文，其余失败一律 ❌ 失败
且不计入。
"""

import pytest

from graspo.core.result_judge import (
    A4_CALIBRATION_MIN_N,
    A4_CALIBRATION_REQUIRED_METADATA,
    A4_FINAL_LOSS_TOLERANCE_ABS,
    A4_LEGACY_GLOBAL_TOLERANCE_ABS,
    A4_MEASURED_BF16_FINAL_LOSS_DRIFT,
    A4_MIN_TRUE_BUG_SIGNATURE,
    A4_OUTCOME_CALIBRATED,
    A4_OUTCOME_DEGENERATE,
    A4_OUTCOME_INSUFFICIENT_N,
    A4_TIER_CALIBRATIONS,
    A4_TOLERANCE_HARD_UPPER_BOUND,
    A4_WORST_PAIR_SAFETY_FACTOR,
    A6_LOSS_TREND_BLOCKS,
    A6_LOSS_TREND_FIELD,
    MAX_CONTEXT_KIND_FEASIBLE,
    MAX_CONTEXT_KIND_OOM_BOUNDARY,
    MISSING_SENTINEL,
    A4TierCalibration,
    FailureClass,
    RunEvidence,
    _a4_first_step_was_checked,
    _a4_first_step_zero_tolerance_check,
    _is_sign_flip,
    _model_mode_specificity_order,
    a6_loss_trend,
    build_tier_calibration,
    calibrated_tolerance,
    classify_failure,
    counts_toward_max_context,
    degenerate_final_check_reason,
    find_tier_calibration,
    judge_a1,
    judge_a2,
    judge_a3,
    judge_a4,
    judge_a5,
    judge_a6,
    judge_tier,
    ledger_row,
    normalize_a4_algorithm,
    validate_tier_calibration_entry,
    validate_tier_tolerance,
    worst_pair,
)


def make_evidence(**overrides) -> RunEvidence:
    base = dict(
        tier_id="T010",
        exit_code=0,
        timed_out=False,
        log_text="",
        tuner_type="lora",
        optimizer_steps=6,
        epochs_completed=1.0,
        weight_changed=True,
        checkpoint_reloadable=True,
        artifacts_present={
            "config_backup": True,
            "training_log": True,
            "checkpoint": True,
            "metrics": True,
        },
        losses=(1.0, 0.8, 0.6),
        grad_norms=(1.0, 0.9, 0.8),
    )
    base.update(overrides)
    return RunEvidence(**base)


# ── A1 ──────────────────────────────────────────────────────────────────────


def test_a1_passes_on_zero_exit():
    assert judge_a1(make_evidence()).passed


def test_a1_fails_on_nonzero_exit_and_timeout():
    assert not judge_a1(make_evidence(exit_code=1)).passed
    assert not judge_a1(make_evidence(timed_out=True, exit_code=124)).passed
    assert not judge_a1(make_evidence(exit_code=None)).passed


# ── A2（LoRA / 全参两套判据）────────────────────────────────────────────────


def test_a2_passes_for_lora_with_nonzero_weights():
    result = judge_a2(make_evidence(tuner_type="lora", weight_changed=True))

    assert result.passed


def test_a2_passes_for_full_with_changed_weights():
    assert judge_a2(make_evidence(tuner_type="full", weight_changed=True)).passed


def test_a2_fails_when_weights_unchanged_or_steps_too_few():
    assert not judge_a2(make_evidence(weight_changed=False)).passed
    assert not judge_a2(make_evidence(optimizer_steps=4)).passed


def test_a2_fails_closed_when_evidence_missing():
    assert not judge_a2(make_evidence(optimizer_steps=None)).passed
    assert not judge_a2(make_evidence(weight_changed=None)).passed


# ── A3 / A5 ─────────────────────────────────────────────────────────────────


def test_a3_fails_closed_on_missing_evidence():
    assert judge_a3(make_evidence()).passed
    assert not judge_a3(make_evidence(checkpoint_reloadable=False)).passed
    assert not judge_a3(make_evidence(checkpoint_reloadable=None)).passed


def test_a5_requires_all_four_artifacts():
    assert judge_a5(make_evidence()).passed
    broken = dict(make_evidence().artifacts_present)
    broken["metrics"] = False
    assert not judge_a5(make_evidence(artifacts_present=broken)).passed


# ── A4 ──────────────────────────────────────────────────────────────────────


def test_a4_requires_second_run():
    assert not judge_a4(make_evidence(), None).passed


def test_a4_passes_on_consistent_double_run():
    assert judge_a4(make_evidence(), make_evidence()).passed


def test_a4_fails_on_inconsistent_double_run():
    assert not judge_a4(make_evidence(), make_evidence(optimizer_steps=7)).passed
    assert not judge_a4(make_evidence(), make_evidence(losses=(1.0, 0.8, 5.0))).passed


# ── A6 ──────────────────────────────────────────────────────────────────────
#
# 2026-09-20 裁定（指挥官，依据 T010 真机实测）：A6 的「最终 loss 不高于初始 loss」
# 子检查**降为记录项**（A6_LOSS_TREND_BLOCKS=False）。下面三条一组锁死这个改动的
# 边界：**只**放开首末走向，NaN/Inf 与取证缺口两条 fail-closed 红线一字不动。

_TREND_MARKER = f"{A6_LOSS_TREND_FIELD}="


def _trend_row(losses):
    """把一组 loss 走完整条判定链，返回台账行（验证走向真的进了 ledger）。"""
    evidence = make_evidence(losses=losses)
    judgement = judge_tier(evidence, make_evidence(losses=losses))
    return ledger_row(
        judgement,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="native",
        cards=1,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-20",
    )


def test_a6_passes_on_healthy_numerics():
    assert judge_a6(make_evidence()).passed


def test_a6_records_increased_loss_trend_instead_of_blocking():
    """★裁定守卫：末点高于起点、数值全 finite ⇒ A6 **通过**，且走向被**显式记录**。

    依据 T010（9B·SFT·LoRA·native·1 卡）：100/100 步、exit=0、全程 finite、
    skipped_nonfinite=0，但 losses 0.0250 → 0.0603，而逐步波动区间是 0.0042–0.4043
    ⇒ 首末两点比较在统计上没有意义，旧判据把完全健康的训练判成「数值异常」。
    降级 ≠ 静默丢弃：走向必须仍然可见（§2.2 显式即防呆）。
    """
    result = judge_a6(make_evidence(losses=(0.5, 0.42, 0.7)))

    assert result.passed, "首末走向是记录项，不得再阻断 A6"
    assert a6_loss_trend(result) == "increased"
    assert _TREND_MARKER in result.detail, "走向必须进入明细文本，不得静默丢弃"
    assert "0.7" in result.detail and "0.5" in result.detail, "首末值必须随记录可见"
    # 台账视图与明细同源（§1.4）：同一个走向，不得出现第二个口径。
    assert _trend_row((0.5, 0.42, 0.7))[A6_LOSS_TREND_FIELD] == "increased"


def test_a6_records_decreased_loss_trend_and_still_passes():
    result = judge_a6(make_evidence(losses=(0.7, 0.42, 0.5)))

    assert result.passed
    assert a6_loss_trend(result) == "decreased"
    assert _TREND_MARKER in result.detail


def test_a6_fails_on_nan_or_inf_even_after_the_trend_demotion():
    """★红线守卫：降级**没有**放松数值健康的事实断言。

    "出现 NaN/Inf" 与 "缺读数" 两侧都必须仍然判否——否则这次改动就不是降级一条
    子检查，而是把整个 A6 变成了空断言。
    """
    for evidence in (
        make_evidence(losses=(1.0, float("nan"))),
        make_evidence(losses=(1.0, float("inf"))),
        make_evidence(losses=(1.0, float("-inf"))),
    ):
        result = judge_a6(evidence)

        assert not result.passed, "真读到非有限 loss ⇒ A6 必须仍然不通过"
        assert "NaN/Inf" in result.detail
        assert classify_failure(evidence, result) is FailureClass.NUMERIC_ANOMALY
    assert not judge_a6(make_evidence(grad_norms=(1.0, float("inf")))).passed
    assert not judge_a6(make_evidence(grad_norms=(1.0, float("nan")))).passed


def test_a6_fail_closed_on_missing_reading_is_unaffected_by_the_demotion():
    """★红线守卫：取证缺口仍然 fail-closed（既不是通过，也不是数值异常）。"""
    for evidence in (
        make_evidence(losses=(1.0, MISSING_SENTINEL, 0.5)),
        make_evidence(losses_unavailable=True, losses=(1.0, 0.5)),
    ):
        result = judge_a6(evidence)

        assert not result.passed, "读不到读数 ⇒ 必须仍然 fail-closed"
        assert result.evidence_missing, "缺读数必须记成取证缺口，不是数值异常"
        assert classify_failure(evidence, result) is FailureClass.UNCLASSIFIED
    # 空序列同样仍然 fail-closed。
    assert not judge_a6(make_evidence(losses=(), grad_norms=(1.0,))).passed


def test_a6_trend_switch_single_source_of_truth():
    """★防呆：降级由**一处**开关表达；把它翻回 True 必须恢复阻断（可回退性验证）。

    这条用例同时证明非默认路径仍然可达——若哪天有人在别处硬编码"不阻断"，
    这里的断言会失败（单点开关是唯一的裁决处，§1.4）。回退后走向**仍记录**，
    因为记录与阻断是两件事——回退只恢复"阻断"，不取消"可见"。
    """
    import graspo.core.result_judge as judge_mod

    assert A6_LOSS_TREND_BLOCKS is False, "2026-09-20 裁定后的默认口径"
    try:
        judge_mod.A6_LOSS_TREND_BLOCKS = True
        reverted = judge_a6(make_evidence(losses=(0.5, 0.7)))

        assert not reverted.passed, "开关翻回 True 必须恢复旧行为（阻断）"
        assert a6_loss_trend(reverted) == "increased", "回退只恢复阻断，走向记录仍在"
    finally:
        judge_mod.A6_LOSS_TREND_BLOCKS = False


def test_a6_fails_closed_on_empty_series():
    assert not judge_a6(make_evidence(losses=(), grad_norms=(1.0,))).passed
    assert not judge_a6(make_evidence(grad_norms=())).passed


# ── 失败分类与"只有真 OOM 计入上下文" ───────────────────────────────────────


def test_real_oom_is_the_only_class_counted_toward_max_context():
    oom = make_evidence(exit_code=1, log_text="torch.OutOfMemoryError: CUDA out of memory")
    a6 = judge_a6(oom)

    failure = classify_failure(oom, a6)

    assert failure is FailureClass.REAL_OOM
    assert counts_toward_max_context(failure) is True
    for other in FailureClass:
        if other is FailureClass.REAL_OOM:
            continue
        assert counts_toward_max_context(other) is False


def test_framework_unimplemented_is_not_counted():
    evidence = make_evidence(exit_code=1, log_text="NotImplementedError: l2k is not implemented")

    failure = classify_failure(evidence, judge_a6(evidence))

    assert failure is FailureClass.FRAMEWORK_UNIMPLEMENTED
    assert not counts_toward_max_context(failure)


def test_timeout_and_unclassified_are_not_counted():
    timed = make_evidence(exit_code=124, timed_out=True, log_text="timeout: sending signal TERM")
    assert classify_failure(timed, judge_a6(timed)) is FailureClass.TIMEOUT

    unknown = make_evidence(exit_code=3, log_text="something odd happened")
    failure = classify_failure(unknown, judge_a6(unknown))
    assert failure is FailureClass.UNCLASSIFIED
    assert not counts_toward_max_context(failure)


# ── 总判定 ──────────────────────────────────────────────────────────────────


def test_judge_tier_passes_when_all_criteria_pass():
    judgement = judge_tier(make_evidence(), make_evidence())

    assert judgement.passed
    assert judgement.ledger_status == "✅ 通过"
    assert judgement.failure_class is None
    assert not judgement.counts_toward_max_context


def test_judge_tier_marks_oom_as_context_candidate():
    oom = make_evidence(exit_code=1, log_text="CUDA out of memory. Tried to allocate 2 GiB")
    judgement = judge_tier(oom, context_length=65536)

    assert not judgement.passed
    assert judgement.ledger_status == "❌ 失败"
    assert judgement.failure_class is FailureClass.REAL_OOM
    assert judgement.counts_toward_max_context
    assert "65536" in judgement.note


def test_judge_tier_non_oom_failure_is_not_context_candidate():
    broken = make_evidence(exit_code=1, log_text="NotImplementedError: unsupported feature")
    judgement = judge_tier(broken, context_length=65536)

    assert not judgement.passed
    assert not judgement.counts_toward_max_context
    assert "修复后重测" in judgement.note


# ── 🟡-4：显式非有限梯度标记必须前置（不得被数据/框架文本覆盖）──────────────


def test_nonfinite_grad_marker_wins_over_data_text():
    """`FileNotFoundError` + `非有限梯度` ⇒ 数值异常（不是数据问题）。

    旧序下 `数据问题` 排在 `数值异常` 之前，实测把这一组合归成「数据问题」，把
    排查引向数据而不是数值链路。两者都不计入最大可行上下文，故这是**归类方向**
    问题，不是能力边界问题。
    """
    evidence = make_evidence(
        exit_code=1,
        log_text=(
            "FileNotFoundError: dataset.jsonl\n"
            "RuntimeError: 非有限梯度：SFT 训练硬失败（fail-closed，宪法 §3.4）。"
        ),
    )

    failure = classify_failure(evidence, judge_a6(evidence))

    assert failure is FailureClass.NUMERIC_ANOMALY
    assert not counts_toward_max_context(failure)


def test_nonfinite_grad_marker_wins_over_framework_text():
    """`NotImplementedError` + `非有限梯度` ⇒ 同样是数值异常（标记优先）。"""
    evidence = make_evidence(
        exit_code=1,
        log_text="NotImplementedError\nRuntimeError: 非有限梯度：SFT 训练硬失败。",
    )

    assert classify_failure(evidence, judge_a6(evidence)) is FailureClass.NUMERIC_ANOMALY


def test_data_problem_still_reachable_without_the_marker():
    """对照：同一份数据文本**没有**标记时仍是数据问题（修正没有把该级顶掉）。"""
    evidence = make_evidence(exit_code=1, log_text="FileNotFoundError: dataset.jsonl")

    assert classify_failure(evidence, judge_a6(evidence)) is FailureClass.DATA_PROBLEM


# ── OPD(GKD) detached loss：接线/配置类，不得记成"未知" ──────────────────────


def test_detached_loss_is_classified_as_config_error():
    """T046 实测栈：第 0 步 backward 抛 detached loss ⇒ **配置非法**。

    真因是接线/参数透传（OPD 少传 `--lmbda` ⇒ 上游默认 0.5 ⇒ GKD 分流到按"数据集
    既有回答"算散度的 off-policy 分支 ⇒ 纯提示词行零监督 token），不是数据坏样本、
    也不是上下文过长。修正前 ``_LOG_PATTERNS`` 对该文本 **0 命中** ⇒ 落
    ``UNCLASSIFIED``——那是一个**假的"未知"**：我们有确定性文本却没有用它。
    """
    evidence = make_evidence(
        exit_code=1,
        log_text=(
            "[rank0]: RuntimeError: element 0 of tensors does not require grad "
            "and does not have a grad_fn\n"
        ),
    )

    failure = classify_failure(evidence, judge_a6(evidence))

    assert failure is FailureClass.CONFIG_INVALID
    assert not counts_toward_max_context(failure)


def test_detached_loss_tier_is_not_a_max_context_candidate():
    """同一条失败在台账层：❌ 失败 + 不计入最大可行上下文（能力边界不变）。"""
    evidence = make_evidence(
        exit_code=1,
        log_text=(
            "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn\n"
        ),
    )

    judgement = judge_tier(evidence, context_length=65536)

    assert not judgement.passed
    assert not judgement.indeterminate  # 有实质证据，不是取证缺口
    assert judgement.failure_class is FailureClass.CONFIG_INVALID
    assert not judgement.counts_toward_max_context
    assert "修复后重测" in judgement.note


# ── 🔴-1：ledger_row 的 max_context 资格口径 ─────────────────────────────────


def _row(judgement, max_context: int | None) -> dict:
    return ledger_row(
        judgement,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="native",
        cards=1,
        max_context=max_context,
        peak_memory_gib=None,
        date="2026-09-18",
    )


def test_ledger_row_passed_writes_feasible_context():
    judgement = judge_tier(make_evidence(), make_evidence(), context_length=8192)

    row = _row(judgement, 8192)

    assert judgement.passed
    assert row["max_context"] == 8192
    assert row["max_context_kind"] == MAX_CONTEXT_KIND_FEASIBLE


def test_ledger_row_real_oom_writes_boundary_candidate():
    """真 OOM（exit≠0 ⇒ passed=False）必须**仍然**能写进台账（阻断点）。"""
    oom = make_evidence(exit_code=1, log_text="CUDA out of memory. Tried to allocate 2 GiB")
    judgement = judge_tier(oom, context_length=65536)

    row = _row(judgement, 65536)

    assert not judgement.passed  # 判定语义不变：坏 run 仍判失败
    assert row["status"] == "❌ 失败"
    assert row["max_context"] == 65536
    assert row["max_context_kind"] == MAX_CONTEXT_KIND_OOM_BOUNDARY


def test_ledger_row_non_oom_failure_writes_nothing():
    """数值异常 / 数据问题 / 框架问题 / 超时 / 未分类 ⇒ 一律不写上下文。"""
    for log_text in (
        "FileNotFoundError: dataset.jsonl",
        "NotImplementedError: l2k",
        "NCCL error: unhandled cuda error",
        "something odd happened",
    ):
        judgement = judge_tier(make_evidence(exit_code=1, log_text=log_text), context_length=65536)
        row = _row(judgement, 65536)

        assert row["max_context"] is None, log_text
        assert row["max_context_kind"] is None, log_text


# ── A4（2026-09-19 裁定：放宽容差 + 三条硬要求）──────────────────────────────
#
# 裁定要点：宪法 §6 明文把"GPU 内核选择等**无法控制的差异**"排除在复现破坏之外；
# bf16 归约顺序正属此类（实测：同 config 同 seed 双跑，**首步 loss 逐位相同**，
# 分歧自 step 5 起、step-100 终态累计 1.03e-3）。
# 但"放宽阈值"本身会让 A4 变成空断言 ⇒ 必须同时具备：
#   ① **零容差子检查**：首步（step 1）loss 必须逐位相同 —— 种子/初始化/数据顺序坏了它立刻抓住；
#   ② 终态 loss 容差**由实测量级推导**（留明确余量），并在判据里写明依据；
#   ③ 下面这些用例就是"守卫"：没有它们，不允许把 A4 判成通过。


def test_a4_guards_step_one_loss_bit_exactly():
    """★守卫①：首步 loss 逐位不同 ⇒ A4 必须不通过。

    首步 loss 只取决于 种子 / 初始化 / 数据顺序 / 首批样本 —— 全是**可控**因素。
    两端终态 loss 完全相同（0.6 == 0.6），只有首步差 1e-6：这正是"去掉种子"或
    "打乱数据顺序"这类破坏的指纹。旧判据只比终态 loss，会把它放过去。
    """
    first = make_evidence(losses=(1.0, 0.8, 0.6), first_logged_step=1)
    second = make_evidence(losses=(1.000001, 0.8, 0.6), first_logged_step=1)

    result = judge_a4(first, second)

    assert not result.passed
    assert "首步" in result.detail


def test_a4_accepts_measured_bf16_drift_but_rejects_gross_divergence():
    """★守卫②：容差按实测推导（约 10× 余量），但仍必须挡住量级更大的分歧。"""
    base = make_evidence(losses=(1.3, 0.5, 0.2), first_logged_step=1)
    measured = make_evidence(
        losses=(1.3, 0.5, 0.2 + A4_MEASURED_BF16_FINAL_LOSS_DRIFT), first_logged_step=1
    )
    # ★ 2026-09-21：终态容差改成**分档**标定后，本用例的"gross"不能再写死 0.05
    #（native/1 卡/SFT 档族容差 4.675e-2，0.05 只略越界，不足以表达"量级更大的分歧"）。
    # 改为"明显超过该档族容差"（2×），让本用例继续只测它要测的东西：容差不是空断言。
    tier_tol = find_tier_calibration("native", 1, "SFT").tol
    gross = make_evidence(losses=(1.3, 0.5, 0.2 + 2 * tier_tol), first_logged_step=1)

    assert judge_a4(base, measured).passed, "实测 bf16 漂移量级必须被接受（§6 排除项）"
    assert not judge_a4(base, gross).passed, "量级更大的分歧仍必须被拒绝（容差不是空断言）"


def test_a4_tolerance_is_derived_per_tier_from_recorded_measurements():
    """★防呆③（**锁二**，2026-09-21 逐字改写）：容差不得拍脑袋，也不得"常量×固定倍数"。

    **改前**（旧断言，已被 AD1 §6.3 锁二判为失效）::

        assert A4_MEASURED_BF16_FINAL_LOSS_DRIFT > 0
        assert A4_FINAL_LOSS_TOLERANCE_ABS >= 5 * A4_MEASURED_BF16_FINAL_LOSS_DRIFT
        assert A4_FINAL_LOSS_TOLERANCE_ABS <= 100 * A4_MEASURED_BF16_FINAL_LOSS_DRIFT

    它失效的原因：这是一个"全局常量 × 固定倍数"的区间断言，**默认"更大的容差 = 合法的"**，
    因此把常量乘到 ~330×（= 0.34）它照样放行——而 0.34 能放过真 bug 0.0935。
    改后的断言不再检查"倍率区间"，而是检查**每一条档族标定的出处与隔离带**：

    ① 每条已定档的容差都 ≤ 最小真 bug 签名的一半（锁一，真 bug 半量封顶）；
    ② 每条都 ≥ 实测基线下界（不能低于实测漂移）；
    ③ 每条都必须带**出处**（provenance 非空）——凭空取数过不了；
    ④ 容差必须是"最坏对 × 安全倍数"推出的量级（不是某个常量的固定倍数）。
    """
    assert A4_MEASURED_BF16_FINAL_LOSS_DRIFT > 0
    assert A4_TOLERANCE_HARD_UPPER_BOUND == A4_MIN_TRUE_BUG_SIGNATURE / 2.0
    assert A4_TIER_CALIBRATIONS, "标定表不得为空（否则 A4 全部 fail-closed）"
    for calibration in A4_TIER_CALIBRATIONS:
        assert calibration.provenance.strip(), f"{calibration} 缺出处 ⇒ 凭空取数"
        assert calibration.tol >= A4_MEASURED_BF16_FINAL_LOSS_DRIFT
        assert calibration.tol <= A4_MIN_TRUE_BUG_SIGNATURE / 2.0
        if calibration.outcome == A4_OUTCOME_CALIBRATED:
            assert calibration.n >= A4_CALIBRATION_MIN_N
            # 容差 == min(1.5 × 最坏对, 硬上界)：要么正好是"最坏对 ×1.5"，
            # 要么是被硬上界封顶（此时 1.5×最坏对 > 上界，仍是"实测推出来的"）。
            scaled = A4_WORST_PAIR_SAFETY_FACTOR * calibration.worst_pair
            assert calibration.tol == pytest.approx(
                min(scaled, A4_TOLERANCE_HARD_UPPER_BOUND), rel=1e-9, abs=1e-12
            ), (
                f"{calibration.backend}/{calibration.cards}/{calibration.algorithm} "
                f"容差 {calibration.tol} 既不等于 1.5×最坏对 {scaled} 也不等于硬上界"
                " ⇒ 这个数没有从实测推出来"
            )
    # 被废止的历史全局容差只作为兼容别名存在，**不得**再被当作判据取数入口。
    assert A4_FINAL_LOSS_TOLERANCE_ABS == A4_LEGACY_GLOBAL_TOLERANCE_ABS
    assert find_tier_calibration("native", 1, "SFT").tol != A4_LEGACY_GLOBAL_TOLERANCE_ABS


def test_a4_first_step_zero_tolerance_subcheck_is_untouched():
    """★★ **首步零容差子检查一个字都不能改**（AD1 §⑤：区分真 bug 与内核残余的唯一尺子）。

    本用例锁住它的**行为契约**（不是实现细节）：27 条证据里被它拦下的真 bug 差值
    全部 ≤ 3.3e-3（T015 8.31e-4 / T033 1.54e-3 / T047 1.58e-3 / T048 2.05e-3 /
    T046 2.31e-3 / T026 3.24e-3 / GRPO 播种缺陷 3.3e-3）——**全部小于任何终态容差**，
    所以只要这条子检查被削弱，这些真 bug 会集体漏网。
    """
    # ① 首位差 1e-6（**远小于**任何档族容差）⇒ 必须判否，且消息里点名"首步"。
    first = make_evidence(losses=(1.0, 0.8, 0.6), first_logged_step=1)
    second = make_evidence(losses=(1.000001, 0.8, 0.6), first_logged_step=1)
    result = judge_a4(first, second)
    assert not result.passed
    assert "首步 loss 不一致（零容差子检查）" in result.detail
    # ② 逐位相同 ⇒ 通过，且台账明写"首步 loss 逐位相同"。
    ok = judge_a4(first, make_evidence(losses=(1.0, 0.8, 0.6), first_logged_step=1))
    assert ok.passed
    assert "首步 loss 逐位相同" in ok.detail
    # ③ 拿不到步号 ⇒ 如实声明"未适用"，**不得**假装比过。
    no_step = judge_a4(make_evidence(), make_evidence())
    assert no_step.passed
    assert "未适用" in no_step.detail
    # ④ 首步无读数（MISSING 哨兵）⇒ fail-closed。
    missing_first = make_evidence(losses=(MISSING_SENTINEL, 0.8, 0.6), first_logged_step=1)
    assert not judge_a4(missing_first, first).passed
    assert "零容差子检查无法进行" in judge_a4(missing_first, first).detail


def test_a4_declares_when_the_zero_tolerance_check_cannot_apply():
    """诚实性：拿不到"首步"的步号时，零容差子检查必须**声明未适用**，不得假装做过。"""
    result = judge_a4(make_evidence(), make_evidence())

    assert result.passed
    assert "未适用" in result.detail


# ── A4 分档标定 + 三道防放行锁（2026-09-21 裁定，依据 task-ad1-a4-validity）──────
#
# 本组用例锁住"换标定方法、不是放宽"这件事本身：
#   · 每条档族容差都有出处、都在隔离带内（锁一）；
#   · "常量 × 固定倍数"的重推路径被移除，且一个 0.34 的危险变体会被拦住（锁二）；
#   · 双通道冲突判「不可判定」（锁三）；
#   · 自比（同一目录同传）在 collector 边界上被拒绝（独立用例在 e2e）。
# 同时覆盖 AD1 §④ 的"退化通过"标注。


def test_a4_calibration_table_covers_the_required_tiers_with_provenance():
    """五要素第 1 条：**按 backend × cards × algorithm（× model × mode）分档**，每条带出处。"""
    native1 = find_tier_calibration("native", 1, "SFT")
    swift1 = find_tier_calibration("ms-swift", 1, "SFT")
    grpo1 = find_tier_calibration("ms-swift", 1, "GRPO")
    assert native1 is not None and swift1 is not None and grpo1 is not None
    # 取值出处（AD1 §6.1/§7.2 与 §3）：T013 单档常量的 10× 全局化**已废止**。
    assert native1.tol == A4_TOLERANCE_HARD_UPPER_BOUND  # 1.5×0.0344 被锁一封顶
    assert native1.worst_pair == 0.034423828125
    assert native1.n == 5
    assert swift1.tol == 6.72e-3
    # 2026-09-22 J3 清洗后：1 卡 GRPO 行只剩 T031（6 个全 0 读数）⇒ 退化，不再是 n=4。
    assert grpo1.outcome == A4_OUTCOME_DEGENERATE
    assert grpo1.n == 6
    assert grpo1.worst_pair == 0.0
    assert grpo1.tol == A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    for calibration in A4_TIER_CALIBRATIONS:
        assert calibration.provenance.strip()
        assert calibration.worst_pair >= 0.0
        # 元数据完整性（§1.4）：calibrated 行必须能指回代表档/覆盖面/时间/小样本核对。
        if calibration.outcome == A4_OUTCOME_CALIBRATED:
            for field in A4_CALIBRATION_REQUIRED_METADATA:
                assert str(getattr(calibration, field)).strip(), (calibration.algorithm, field)


def test_a4_msswift_1card_grpo_row_is_cleaned_of_four_card_readings():
    """J3（2026-09-22）：1 卡 GRPO 行**不得**再引用 4 卡 T033 的读数。

    旧行出处写 ``T031/T033``、worst_pair 9.2e-3（由 4 卡读数主导）⇒ 跨卡集污染，
    违反 AO1 的「同一卡集合（mandatory）」纪律。清洗后：只剩 1 卡 T031 的 6 个全 0
    读数 ⇒ outcome=degenerate、tol=基线下界；旧值必须留证在 note/provenance 里。
    """
    row = find_tier_calibration("ms-swift", 1, "GRPO")
    assert row is not None
    assert row.outcome == A4_OUTCOME_DEGENERATE
    assert row.n == 6
    assert row.worst_pair == 0.0
    assert row.tol == A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    # 清洗证据：不得再出现"把 T033 当 1 卡读数"的表述。
    assert "T031/T033 ·" not in row.provenance
    assert "剔除全部 4 卡 T033" in row.provenance
    # 旧值留证（可审计性，§1.4）。
    assert "1.38e-2" in row.provenance
    assert "9.2e-3" in row.provenance


def test_a4_algorithm_alias_graspo_resolves_to_the_grpo_rows():
    """J5（2026-09-22）：清单写 ``GRASPO``、标定表写 ``GRPO`` ⇒ 显式别名必须命中。

    ★ 4 卡 GRASPO 仍必须 fail-closed：别名只让 1/2 卡的两条 GRPO 行可达，
    4 卡族在表里**没有**对应行。
    """
    assert normalize_a4_algorithm("GRASPO") == "GRPO"
    assert normalize_a4_algorithm("GRPO") == "GRPO"
    assert normalize_a4_algorithm("SFT") == "SFT"
    # 未知算法原样返回（精确匹配 ⇒ 不命中 ⇒ fail-closed；不做 fuzzy）。
    assert normalize_a4_algorithm("graspo") == "graspo"
    assert find_tier_calibration("ms-swift", 1, "GRASPO") is find_tier_calibration(
        "ms-swift", 1, "GRPO"
    )
    assert find_tier_calibration("ms-swift", 2, "GRASPO") is find_tier_calibration(
        "ms-swift", 2, "GRPO"
    )
    # ★ 4 卡 GRASPO 必须仍为 None（别名救不了它，需要独立标定）。
    assert find_tier_calibration("ms-swift", 4, "GRASPO") is None
    assert find_tier_calibration("native", 4, "GRASPO") is None


def test_a4_five_tuple_key_exact_wins_over_wildcard_unknown_model_fails_closed():
    """J2/P1：五元组键**精确优先**、通配只降维、维度未知时 fail-closed。"""
    # 现有历史行是通配行（model/mode 均 None）⇒ 带任意 model/mode 都命中它。
    wildcard = find_tier_calibration("native", 1, "SFT", "9B", "LoRA")
    assert wildcard is not None and wildcard.model is None and wildcard.mode is None
    assert find_tier_calibration("native", 1, "SFT", "27B", "inexistent-mode") is wildcard
    # 候选顺序：最具体 → 最泛化，且**已知维度不得匹配未知维度的精确行**。
    assert _model_mode_specificity_order("9B", "LoRA") == (
        ("9B", "LoRA"),
        ("9B", None),
        (None, "LoRA"),
        (None, None),
    )
    assert _model_mode_specificity_order(None, None) == ((None, None),)
    assert _model_mode_specificity_order("9B", None) == (("9B", None), (None, None))
    assert _model_mode_specificity_order(None, "LoRA") == ((None, "LoRA"), (None, None))
    # 精确行必须赢过通配行（用一条内存里的临时表验证匹配逻辑）。
    exact = A4TierCalibration(
        backend="ms-swift",
        cards=1,
        algorithm="CPT",
        tol=A4_MEASURED_BF16_FINAL_LOSS_DRIFT,
        worst_pair=0.002,
        n=5,
        outcome=A4_OUTCOME_CALIBRATED,
        provenance="test",
        model="9B",
        mode="LoRA",
        representative_tier="T001",
        model_scope="9B/LoRA",
        sampled_at="test",
        split_check="test",
    )
    validate_tier_calibration_entry(exact)  # 元数据齐全 ⇒ 通过
    # 未命中任何行 ⇒ None（调用方据此 fail-closed）。
    assert find_tier_calibration("native", 1, "CPT") is None
    assert find_tier_calibration("ms-swift", 4, "SFT") is None


def test_a4_entry_validation_rejects_incomplete_or_impossible_rows():
    """§2.3：入表校验必须在边界上拒绝"证据不足却标 calibrated"的行。"""
    base = dict(
        backend="native",
        cards=1,
        algorithm="CPT",
        tol=2e-3,
        worst_pair=1e-3,
        n=A4_CALIBRATION_MIN_N,
        outcome=A4_OUTCOME_CALIBRATED,
        provenance="test",
        model="9B",
        mode="LoRA",
        representative_tier="T001",
        model_scope="9B/LoRA",
        sampled_at="test",
        split_check="test",
    )
    validate_tier_calibration_entry(A4TierCalibration(**base))  # 正例
    # provenance 为空 ⇒ 拒（所有行）。
    for outcome in (A4_OUTCOME_CALIBRATED, A4_OUTCOME_INSUFFICIENT_N, A4_OUTCOME_DEGENERATE):
        with pytest.raises(ValueError):
            validate_tier_calibration_entry(
                A4TierCalibration(**{**base, "provenance": "  ", "outcome": outcome})
            )
    # n < min_n 却标 calibrated ⇒ 拒。
    with pytest.raises(ValueError):
        validate_tier_calibration_entry(
            A4TierCalibration(**{**base, "n": A4_CALIBRATION_MIN_N - 1})
        )
    # worst_pair == 0 却标 calibrated ⇒ 拒（退化池必须标 degenerate）。
    with pytest.raises(ValueError):
        validate_tier_calibration_entry(A4TierCalibration(**{**base, "worst_pair": 0.0}))
    # 元数据缺失 ⇒ 拒（逐项）。
    for field in A4_CALIBRATION_REQUIRED_METADATA:
        with pytest.raises(ValueError):
            validate_tier_calibration_entry(A4TierCalibration(**{**base, field: ""}))
    # 非 calibrated 行不要求元数据（它们本就没有可用读数池）。
    validate_tier_calibration_entry(
        A4TierCalibration(
            **{**base, "outcome": A4_OUTCOME_INSUFFICIENT_N, "n": 0, "worst_pair": 0.0,
               "representative_tier": "", "model_scope": "", "sampled_at": "", "split_check": ""}
        )
    )


def test_a4_calibrated_tolerance_floors_at_base_and_handles_degenerate_pool():
    """§2.3 边界：退化池（全同）与"比下界还稳"的池都必须返回可用容差，不得抛错或给 0。"""
    # 退化池：n≥5 但读数逐位相同 ⇒ degenerate + 基线下界（绝不给 0）。
    tol, outcome = calibrated_tolerance([0.0] * 6)
    assert outcome == A4_OUTCOME_DEGENERATE
    assert tol == A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    # 比基线下界还稳的池 ⇒ 容差**上调**到下界（下界锁的本意），仍标 calibrated。
    readings = [0.0, 0.0001, 0.0002, 0.0001, 0.0002]
    assert A4_WORST_PAIR_SAFETY_FACTOR * worst_pair(readings) < A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    tol_small, outcome_small = calibrated_tolerance(readings)
    assert outcome_small == A4_OUTCOME_CALIBRATED
    assert tol_small == A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    calibration = build_tier_calibration(
        readings, backend="ms-swift", cards=2, algorithm="SFT", provenance="test",
        model="9B", mode="LoRA", representative_tier="T014", model_scope="9B/LoRA",
        sampled_at="test", split_check="test",
    )
    assert "下界锁生效" in calibration.note


def test_a4_first_step_zero_tolerance_guard_is_unchanged_and_runs_first():
    """★★ **红线回归守卫**：首步零容差子检查一字未改，且**先于**终态容差生效。

    锁住五件事（2026-09-22 指挥官硬约束 ②）：
    ─ ① 首步逐位相同 ⇒ 不判否（返回 None）；
    ─ ② 首步任一不同 ⇒ 判否，并给"可复现性被破坏"的原文指纹；
    ─ ③ 首步无读数（MISSING 哨兵）⇒ fail-closed；
    ─ ④ 拿不到步号 ⇒ **不适用**（不得假装比过）；
    ─ ⑤ 与终态容差**无关**：即使某档族已有可用容差、且终态差极小，
      首步不一致仍必须判 ❌（T033 形状：首步差 0.0316、终态差 5.1e-4）。
    """
    equal_first = make_evidence(losses=(1.0, 0.5), grad_norms=(1.0, 1.0), first_logged_step=1)
    other_equal_first = make_evidence(
        losses=(1.0, 0.9), grad_norms=(1.0, 1.0), first_logged_step=1
    )
    assert _a4_first_step_zero_tolerance_check(equal_first, other_equal_first) is None
    assert _a4_first_step_was_checked(equal_first, other_equal_first) is True

    different_first = make_evidence(
        losses=(1.0316, 0.2), grad_norms=(1.0, 1.0), first_logged_step=1
    )
    verdict = _a4_first_step_zero_tolerance_check(equal_first, different_first)
    assert verdict is not None and verdict.passed is False
    assert "首步 loss 不一致（零容差子检查）" in verdict.detail
    assert "可控" in verdict.detail

    missing_first = make_evidence(
        losses=(MISSING_SENTINEL, 0.5), grad_norms=(1.0, 1.0), first_logged_step=1
    )
    missing_verdict = _a4_first_step_zero_tolerance_check(equal_first, missing_first)
    assert missing_verdict is not None and missing_verdict.passed is False
    # 注意：该分支**既有实现**不置 evidence_missing（本轮红线：一字不改）——
    # 守卫只锁"判否 + 原文"，不替它改语义。
    assert "首步 loss 无读数" in missing_verdict.detail

    no_step = make_evidence(losses=(1.0, 0.5), grad_norms=(1.0, 1.0), first_logged_step=None)
    assert _a4_first_step_zero_tolerance_check(no_step, other_equal_first) is None
    assert _a4_first_step_was_checked(no_step, other_equal_first) is False

    # ★ ⑤：T033 形状 —— 终态差极小、首步不一致 ⇒ 仍 ❌（终态容差不参与）。
    t033_like_a = make_evidence(
        losses=(0.20522165298461914, 0.17088532), grad_norms=(1.0, 1.0),
        first_logged_step=1, backend="ms-swift", cards=4, algorithm="GRASPO",
        model="9B", mode="LoRA",
    )
    t033_like_b = make_evidence(
        losses=(0.1736421287059784, 0.17037117), grad_norms=(1.0, 1.0),
        first_logged_step=1, backend="ms-swift", cards=4, algorithm="GRASPO",
        model="9B", mode="LoRA",
    )
    result = judge_a4(t033_like_a, t033_like_b)
    assert result.passed is False
    assert "首步 loss 不一致（零容差子检查）" in result.detail


def test_a4_msswift_2card_grpo_is_calibrated_and_opd_must_stay_fail_closed():
    """2026-09-21 入表：`ms-swift/2卡/GRPO` 由 AO1 的 n=5 独立跑标定。

    锁住三件事：
    ─ ① 取值 = `min(1.5 × 最坏对 0.004013907909393311, 硬上界)`（**未被封顶**）；
    ─ ② 出处非空 + 隔离带 ≥ AE1 要求的 2.0×（实际 15.53×）；
    ─ ③ **`ms-swift/2卡/OPD` 不得复用 GRPO 的读数**（算法不同、机制不同）
      ⇒ 必须仍为 `None`（judge_a4 据此 fail-closed）。
    """
    grpo = find_tier_calibration("ms-swift", 2, "GRPO")
    assert grpo is not None
    assert grpo.tol == 6.020861864089967e-3
    assert grpo.worst_pair == 0.004013907909393311
    assert grpo.n == A4_CALIBRATION_MIN_N
    assert grpo.outcome == A4_OUTCOME_CALIBRATED
    assert grpo.provenance.strip()
    # 本档未被硬上界封顶：tol 恰好 == 1.5 × 最坏对。
    assert grpo.tol == A4_WORST_PAIR_SAFETY_FACTOR * grpo.worst_pair
    assert grpo.tol < A4_TOLERANCE_HARD_UPPER_BOUND
    # 隔离带（≥ 2.0× 是 AE1 的硬要求；本档实际 15.53×）。
    assert A4_MIN_TRUE_BUG_SIGNATURE / grpo.tol >= 2.0
    assert A4_MIN_TRUE_BUG_SIGNATURE / grpo.tol == pytest.approx(15.53, rel=1e-3)
    # ★ OPD 必须仍 fail-closed：不得让算法不同的档族复用同一个读数池。
    assert find_tier_calibration("ms-swift", 2, "OPD") is None


def test_a4_no_tier_tolerance_exceeds_half_the_smallest_true_bug_signature():
    """**锁一（硬上界）**：任何档族容差 ×2 必须仍 < 最小真 bug 签名 9.35e-2（T032）。"""
    assert A4_MIN_TRUE_BUG_SIGNATURE == 9.35e-2
    assert A4_TOLERANCE_HARD_UPPER_BOUND <= A4_MIN_TRUE_BUG_SIGNATURE / 2.0
    for calibration in A4_TIER_CALIBRATIONS:
        assert calibration.tol * 2 <= A4_MIN_TRUE_BUG_SIGNATURE + 1e-12
        assert calibration.tol <= A4_TOLERANCE_HARD_UPPER_BOUND


def test_a4_tier_resolution_is_exact_then_narrowing_fallback_never_unknown_default():
    """档族取数：精确命中优先；缺字段只**降维回落**；连最泛化档族都没有 ⇒ None。"""
    assert find_tier_calibration("native", 1, "SFT") is not None
    # 精确未命中、且没有任何字段可回落 ⇒ None（判定器据此 fail-closed）。
    assert find_tier_calibration("native", 1, "CPT") is None
    assert find_tier_calibration("totally-unknown", 1, "SFT") is None


def test_a4_inflated_tolerance_variant_is_rejected_by_the_boundary_check():
    """★★ **锁二负向对照**：把容差放大到能放过 T032（0.0935）的变体必须被拦住。

    构造一个 0.34 的档族容差 —— 它 > 0.0935，意味着真 bug T032 会被判"通过"。
    这正是 AD1 §6.3 锁二点名的危险值（旧单测的 [5×,100×] 区间会**自动批准**它）。
    本用例证明：**现有测试**（与模块导入时的全表校验走同一个函数）会拦住它。
    """
    for dangerous in (0.34, A4_MIN_TRUE_BUG_SIGNATURE, 1.0):
        with pytest.raises(ValueError):
            validate_tier_tolerance(dangerous, min_true_bug_signature=A4_MIN_TRUE_BUG_SIGNATURE)
    # 危险值一旦进了标定表，`calibrated_tolerance` 也不会放行（它内部走同一把尺子）。
    with pytest.raises(ValueError):
        validate_tier_tolerance(
            A4_TOLERANCE_HARD_UPPER_BOUND * 1.0000001,
            min_true_bug_signature=A4_MIN_TRUE_BUG_SIGNATURE,
        )
    # 正向对照：恰好等于硬上界是合法的。
    validate_tier_tolerance(
        A4_TOLERANCE_HARD_UPPER_BOUND, min_true_bug_signature=A4_MIN_TRUE_BUG_SIGNATURE
    )


def test_a4_calibration_n_below_five_never_uses_the_low_estimate():
    """五要素第 2 条：**n<5 不得用低估值定档** ⇒ 落到基线下界 + 显式「证据不足」。"""
    readings = [0.0, 0.0044]  # n=2（AD1 §7.2：这一对低估 7.8×）
    tol, outcome = calibrated_tolerance(readings)
    assert outcome == A4_OUTCOME_INSUFFICIENT_N
    assert tol == A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    calibration = build_tier_calibration(
        readings, backend="native", cards=1, algorithm="SFT", provenance="test"
    )
    assert "证据不足" in calibration.note
    assert calibration.n == 2
    # n≥5 才允许按最坏对 ×1.5 定档。
    tol5, outcome5 = calibrated_tolerance([0.0, 0.001, 0.002, 0.003, 0.02])
    assert outcome5 == A4_OUTCOME_CALIBRATED
    assert tol5 == 1.5 * 0.02


def test_a4_sign_flip_check_catches_the_t032_shape():
    """五要素第 4 条：真 bug `T032`（−0.0217 → +0.0718）是**符号翻转**。"""
    # T032 实测值：差值 0.0935 同时越过任何档族容差 ⇒ 先被子检查②拦下。
    t032_first = make_evidence(losses=(-0.021734893321990967,), grad_norms=(1.0,))
    t032_second = make_evidence(losses=(0.07180161476135254,), grad_norms=(1.0,))
    assert not judge_a4(t032_first, t032_second).passed
    # 但"符号翻转"这条守卫**独立**成立——把容差放宽到 0.06（落在"量级检查已放过（>0.04675）、而两端幅度却≥容差"的窗口里，
    # 即"量级检查已经放过它"的尺度）时，它必须单独把 T032 拦下：
    assert _is_sign_flip(-0.021734893321990967, 0.07180161476135254, tolerance=0.06)
    # 边界诚实性：容差放大到 0.5（比 T032 两端幅度都大一个量级）时，本闸**不**开火——
    # 这是刻意的取舍：那个尺度下"−0.02 → +0.07"与内核噪声在纯数值上不可分，
    # 判据只能记为"不可判定/证据不足"，不能凭空断言是 bug（§2.2）。
    assert not _is_sign_flip(-0.021734893321990967, 0.07180161476135254, tolerance=0.5)
    assert not _is_sign_flip(0.000289464, 0.000526679, tolerance=0.5)  # 同号
    assert not _is_sign_flip(-1e-12, 1e-12, tolerance=0.5)  # 零附近抖动，不误报
    # 反向：**同号但幅度大**不报（这是内核漂移的正常形状，不是翻转）。
    assert not _is_sign_flip(0.2, 0.21, tolerance=0.5)
    # 均值远大于容差 ⇒ 报（第二种翻转形状）。
    assert _is_sign_flip(0.001, -0.002, tolerance=1e-6)


def test_a4_dual_channel_conflict_is_indeterminate_never_pass():
    """**锁三（双通道一致）**：权重指纹与 loss 通道冲突 ⇒ 判「不可判定」。"""
    same_hash = "a" * 64
    first = make_evidence(losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
                          final_ckpt_sha256=same_hash)
    second = make_evidence(losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
                           final_ckpt_sha256=same_hash)
    result = judge_a4(first, second)
    assert not result.passed
    assert result.evidence_missing is True, "冲突是「不可判定」，不是「训练失败」"
    assert "双通道冲突" in result.detail
    assert "不可判定" in result.detail
    # 权重不同（真实双跑的预期现象，AD1 §6.2）**不**构成冲突 ⇒ 照常判通过。
    second_diff = make_evidence(losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
                                final_ckpt_sha256="b" * 64)
    assert judge_a4(first, second_diff).passed
    # 任一端没有指纹 ⇒ 单通道，不制造假的"冲突"，也不假装两通道都验过。
    assert judge_a4(first, make_evidence(losses=(1.0, 0.8, 0.6))).passed


def test_a4_bit_identical_two_independent_runs_is_perfect_reproducibility():
    """★裁定 2（正向）：两通道逐位相同 **且**已确认为两次独立运行 ⇒ 判**通过**。

    旧实现对这种形态一律判「双通道冲突/不可判定」——那会**误伤真·完美可复现**：
    若某配置真的逐位确定，A4 会永远报"不可判定"，是方向性错误。
    独立性证据 3 项：运行根不同 / 产物根 ``st_dev:st_ino`` 不同 / 首条记录时间戳不同。
    """
    same_hash = "c" * 64
    first = make_evidence(
        losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
        final_ckpt_sha256=same_hash,
        run_root="/exp/r1/T010", output_identity="8:1001",
        first_metric_timestamp="2026-09-22T05:39:48+00:00",
    )
    second = make_evidence(
        losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
        final_ckpt_sha256=same_hash,
        run_root="/exp/r2/T010", output_identity="8:2002",
        first_metric_timestamp="2026-09-22T05:57:54+00:00",
    )
    result = judge_a4(first, second)
    assert result.passed, result.detail
    assert "完美可复现" in result.detail
    assert "两次真实独立运行" in result.detail
    # 备注必须自证"没动容差/门槛"（§2.2 显式即防呆）。
    assert "未改任何容差/门槛" in result.detail


def test_a4_bit_identical_copied_artifacts_stay_indeterminate():
    """★裁定 2（负向·堵"拷贝造伪独立"）：位置证据拿满但**时间戳相同** ⇒ 仍判不可判定。

    把同一份产物拷到另一个路径：``run_root`` 与 inode 都会变（位置级证据 2 项），
    但产物里**记录的时间戳不会变** ⇒ 内容级证据一票否决独立性 ⇒ fail-closed 保持。
    这条是"不许把自我比较陷阱放行"的守卫。
    """
    same_hash = "d" * 64
    first = make_evidence(
        losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
        final_ckpt_sha256=same_hash,
        run_root="/exp/r1/T010", output_identity="8:1001",
        first_metric_timestamp="2026-09-22T05:39:48+00:00",
    )
    second = make_evidence(
        losses=(1.0, 0.8, 0.6), grad_norms=(1.0, 0.9, 0.8),
        final_ckpt_sha256=same_hash,
        run_root="/exp/copy-of-r1/T010", output_identity="8:9999",
        first_metric_timestamp="2026-09-22T05:39:48+00:00",   # 同一时间戳 ⇒ 拷贝指纹
    )
    result = judge_a4(first, second)
    assert not result.passed
    assert result.evidence_missing is True, "冲突/不可证明是「不可判定」，不是「训练失败」"
    assert "无法证明是两次独立运行" in result.detail
    assert "同一份产物的两次比对" in result.detail


def test_classify_failure_recognises_nccl_collective_timeout():
    """★裁定 3：集合通信超时必须被归类为 ``nccl_collective_timeout``（不再"未分类"）。

    真实形态（T035 实测 stdout 原文）：``Watchdog caught collective operation timeout``
    + ``terminate called after throwing an instance of 'c10::DistBackendError'``。
    既有 ``COMM_HARDWARE`` 的签名（``watchdog timeout`` / ``NCCL error|WARN|timeout``）
    **匹配不到**它 ⇒ 修前落 ``UNCLASSIFIED``。
    取值必须与跑批装置 ``rig/matrix_batch_driver.py`` 的口径**逐字一致**（同一口径两消费者）。
    """
    evidence = make_evidence(
        exit_code=1,
        losses=(1.0, 0.8),
        grad_norms=(1.0, 0.9),
        log_text=(
            "[rank1]:[E922 06:10:39.470785264 ProcessGroupNCCL.cpp:689] [Rank 1] Watchdog "
            "caught collective operation timeout: WorkNCCL(SeqNum=3, OpType=ALLREDUCE, "
            "NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600043 milliseconds before "
            "timing out.\n"
            "terminate called after throwing an instance of 'c10::DistBackendError'\n"
        ),
    )
    a6 = judge_a6(evidence)
    assert a6.passed, a6.detail          # 数值本身健康 ⇒ 分类只由日志模式决定
    assert classify_failure(evidence, a6) is FailureClass.NCCL_COLLECTIVE_TIMEOUT
    assert FailureClass.NCCL_COLLECTIVE_TIMEOUT == (
        "nccl_collective_timeout（集合通信超时 ⇒ PP/DP 各 rank 步调不一致）"
    )
    # 反向守卫：既有的 `通信硬件` 签名仍须可用（未被新条目顶掉）。
    generic = make_evidence(
        exit_code=1, losses=(1.0,), grad_norms=(1.0,),
        log_text="NCCL WARN some socket error\n",
    )
    assert classify_failure(generic, judge_a6(generic)) is FailureClass.COMM_HARDWARE


def test_classify_failure_prefers_exit_code_for_pipeline_p2p_timeout():
    """★裁定：退出码 **86** ⇒ ``pipeline_p2p_timeout``，且**优先于文本**。

    为什么必须优先：torchrun/弹性启动会把异常文本前缀化并包成
    ``ChildFailedError``，文本里可能根本找不到 ``PipelineP2PTimeoutError``
    ⇒ 只按文本会漏判。退出码是进程交还的**事实**。
    """
    evidence = make_evidence(
        exit_code=86, losses=(1.0,), grad_norms=(1.0,),
        log_text="[rank1]: torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n",
    )
    a6 = judge_a6(evidence)
    assert a6.passed
    assert classify_failure(evidence, a6) is FailureClass.PIPELINE_P2P_TIMEOUT
    assert FailureClass.PIPELINE_P2P_TIMEOUT == "pipeline_p2p_timeout"


def test_classify_failure_prefers_exit_code_for_pp_rollout_wall_clock_cap():
    """★裁定：退出码 **21** ⇒ ``pp_rollout_wall_clock_cap``（与 124 的整体超时区分）。"""
    evidence = make_evidence(
        exit_code=21, losses=(1.0,), grad_norms=(1.0,), log_text="(无异常文本)\n",
    )
    a6 = judge_a6(evidence)
    assert classify_failure(evidence, a6) is FailureClass.PP_ROLLOUT_WALL_CLOCK_CAP
    assert FailureClass.PP_ROLLOUT_WALL_CLOCK_CAP == "pp_rollout_wall_clock_cap"
    # 与既有的 124 不冲突：124 仍是 TIMEOUT。
    assert classify_failure(
        make_evidence(exit_code=124, losses=(1.0,), grad_norms=(1.0,), log_text=""), a6
    ) is FailureClass.TIMEOUT


def test_classify_failure_text_fallback_when_exit_code_is_normalised():
    """退出码被外层归一成 1 时，仍能从**异常名**认出（文本兜底，不退化成"未分类"）。"""
    p2p = make_evidence(
        exit_code=1, losses=(1.0,), grad_norms=(1.0,),
        log_text="PipelineP2PTimeoutError: p2p wait exceeded budget\n",
    )
    assert classify_failure(p2p, judge_a6(p2p)) is FailureClass.PIPELINE_P2P_TIMEOUT


def test_new_exit_code_classes_never_count_toward_max_context():
    """两个新类**都不计入最大可行上下文**（只有真 OOM 计入）。"""
    for failure_class in (
        FailureClass.PIPELINE_P2P_TIMEOUT,
        FailureClass.PP_ROLLOUT_WALL_CLOCK_CAP,
    ):
        assert not counts_toward_max_context(failure_class)
    assert counts_toward_max_context(FailureClass.REAL_OOM)


def test_a4_unknown_tier_fails_closed_without_a_default_tolerance():
    """五要素第 1 条：取不到档族标定 ⇒ **不下发默认容差**，fail-closed。"""
    unknown_first = make_evidence(losses=(1.0, 0.8, 0.6), backend="native", cards=1,
                                  algorithm="CPT")
    unknown_second = make_evidence(losses=(1.0, 0.8, 0.6), backend="native", cards=1,
                                   algorithm="CPT")
    result = judge_a4(unknown_first, unknown_second)
    assert not result.passed
    assert result.evidence_missing is True
    assert "缺少该档族的终态容差标定" in result.detail


def test_a4_degenerate_final_check_is_annotated_not_silently_passed():
    """AD1 §④ 的 7 条「退化通过」：必须**显式标注**"零鉴别力"，且不直接改判失败。"""
    assert degenerate_final_check_reason([0.5], [1.0]) is not None      # 单步
    assert degenerate_final_check_reason([1.0, 0.0], [1.0, 0.0]) is not None  # 无信号末步
    assert degenerate_final_check_reason([1.0, 0.5], [1.0, 0.9]) is None
    # T031 形状：两跑末步都是 GRPO 无信号 0 ⇒ ✅ 但台账必须写"未提供鉴别力"。
    first = make_evidence(losses=(1.0, 0.0), grad_norms=(1.0, 0.0), first_logged_step=1)
    second = make_evidence(losses=(1.0, 0.0), grad_norms=(1.0, 0.0), first_logged_step=1)
    result = judge_a4(first, second)
    assert result.passed, "退化档是「证据不足」，不是「不通过」——不得改判失败"
    assert "退化" in result.detail
    assert "未提供鉴别力" in result.detail
    assert "证据不足" in result.detail
