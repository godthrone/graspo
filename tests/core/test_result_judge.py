"""A1–A6 判定与失败分类的单元测试（纯逻辑，不触 GPU、不读文件）。

重点覆盖硬要求：只有"真 OOM"计入最大可行上下文，其余失败一律 ❌ 失败
且不计入。
"""

from graspo.core.result_judge import (
    A4_FINAL_LOSS_TOLERANCE_ABS,
    A4_MEASURED_BF16_FINAL_LOSS_DRIFT,
    A6_LOSS_TREND_BLOCKS,
    A6_LOSS_TREND_FIELD,
    MAX_CONTEXT_KIND_FEASIBLE,
    MAX_CONTEXT_KIND_OOM_BOUNDARY,
    MISSING_SENTINEL,
    FailureClass,
    RunEvidence,
    a6_loss_trend,
    classify_failure,
    counts_toward_max_context,
    judge_a1,
    judge_a2,
    judge_a3,
    judge_a4,
    judge_a5,
    judge_a6,
    judge_tier,
    ledger_row,
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
    gross = make_evidence(losses=(1.3, 0.5, 0.25), first_logged_step=1)

    assert judge_a4(base, measured).passed, "实测 bf16 漂移量级必须被接受（§6 排除项）"
    assert not judge_a4(base, gross).passed, "量级更大的分歧仍必须被拒绝（容差不是空断言）"


def test_a4_tolerance_constant_is_derived_from_a_recorded_measurement():
    """★防呆③：容差不得拍脑袋 —— 必须相对"记录在案的实测量级"留明确余量（两侧都锁）。

    调小到实测漂移之下 ⇒ 失败（说明它是拍脑袋的）；
    调大到 >100× ⇒ 也失败（说明它松到没有判别力）。
    """
    assert A4_MEASURED_BF16_FINAL_LOSS_DRIFT > 0
    assert A4_FINAL_LOSS_TOLERANCE_ABS >= 5 * A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    assert A4_FINAL_LOSS_TOLERANCE_ABS <= 100 * A4_MEASURED_BF16_FINAL_LOSS_DRIFT


def test_a4_declares_when_the_zero_tolerance_check_cannot_apply():
    """诚实性：拿不到"首步"的步号时，零容差子检查必须**声明未适用**，不得假装做过。"""
    result = judge_a4(make_evidence(), make_evidence())

    assert result.passed
    assert "未适用" in result.detail
