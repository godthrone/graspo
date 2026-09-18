"""A1–A6 判定与失败分类的单元测试（纯逻辑，不触 GPU、不读文件）。

重点覆盖硬要求：只有"真 OOM"计入最大可行上下文，其余失败一律 ❌ 失败
且不计入。
"""

from graspo.core.result_judge import (
    FailureClass,
    RunEvidence,
    classify_failure,
    counts_toward_max_context,
    judge_a1,
    judge_a2,
    judge_a3,
    judge_a4,
    judge_a5,
    judge_a6,
    judge_tier,
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


def test_a6_passes_on_healthy_numerics():
    assert judge_a6(make_evidence()).passed


def test_a6_fails_on_nan_or_inf():
    assert not judge_a6(make_evidence(losses=(1.0, float("nan")))).passed
    assert not judge_a6(make_evidence(grad_norms=(1.0, float("inf")))).passed


def test_a6_fails_when_final_loss_above_initial():
    assert not judge_a6(make_evidence(losses=(0.5, 0.7))).passed


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
