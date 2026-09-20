"""P0 验收隐患（F-4 2026-09-18）的负向测试 —— 纯逻辑，零设施依赖。

覆盖工作包 `task-p0-acceptance` 的任务 A：

- **A①** stdout 出现训练器硬失败标记「非有限梯度」⇒ 该 run 判不通过；
- **A②** SFT 侧健康检查（``training_health``）对 ``nonfinite_loss_or_grad`` 告警；
- **A③** 每步 ``optimizer_steps > 0`` 断言：抓住「梯度被跳过、权重冻结」；
- **A④** ``loss: null`` 被**显式**处理：即使把 null 改写成 ``0.0``，
  该 NaN run **仍判不通过**（不得因为"修了解析"反而变通过）；
- **判定优先级**：数值异常必须优先于日志里的 OOM 文本（既有判据，不得回归）。

每个用例都配一个"注入缺陷就真失败"的对照，见 ``test_injected_defect_*``。
"""

import math

from graspo.core.result_judge import (
    FailureClass,
    RunEvidence,
    classify_failure,
    judge_a2,
    judge_a6,
    judge_tier,
)


def make_evidence(**overrides) -> RunEvidence:
    base = dict(
        tier_id="T018",
        exit_code=0,
        timed_out=False,
        log_text="",
        tuner_type="full",
        optimizer_steps=4,
        epochs_completed=1.0,
        weight_changed=True,
        checkpoint_reloadable=True,
        artifacts_present={
            "config_backup": True,
            "training_log": True,
            "checkpoint": True,
            "metrics": True,
        },
        losses=(0.0209, 13.125, float("nan"), float("nan")),
        grad_norms=(10.0, 1.035e7, 0.0, 0.0),
        optimizer_steps_per_step=(4, 4, 0, 0),
        nonfinite_skips=2,
    )
    base.update(overrides)
    return RunEvidence(**base)


# ── A③：每步 optimizer_steps > 0 ────────────────────────────────────────────


def test_a2_fails_when_a_step_skipped_the_optimizer():
    """F-4 实测形态：4 步里后 2 步 optimizer_steps=0 ⇒ 权重冻结。

    两条独立防线都要各自可测：
      - 有 ``skipped_nonfinite`` 证据时走"累计跳过"分支；
      - 只有逐 step 证据时走"某步 optimizer_steps=0"分支。
    """
    with_skips = judge_a2(make_evidence())
    assert not with_skips.passed
    assert "跳过优化器步" in with_skips.detail

    steps_only = judge_a2(
        make_evidence(
            optimizer_steps_per_step=(4, 4, 0, 0),
            nonfinite_skips=None,
            losses=(0.0209, 13.125, 0.0, 0.0),
        )
    )
    assert not steps_only.passed
    assert "optimizer_steps=0" in steps_only.detail
    assert "第 3, 4 步" in steps_only.detail


def test_a2_fails_on_nonfinite_skips_even_with_enough_steps():
    """即使步数已达门槛、权重"看起来变了"，有过跳过也必须不通过。"""
    result = judge_a2(
        make_evidence(
            optimizer_steps=9,
            optimizer_steps_per_step=(4, 4, 4, 4, 4, 4, 4, 4, 4),
            nonfinite_skips=1,
        )
    )
    assert not result.passed
    assert "非有限" in result.detail


def test_a2_passes_on_healthy_run():
    """对照：每步都推进、无跳过 ⇒ 通过（修复没有把正常路径关掉）。"""
    result = judge_a2(
        make_evidence(
            optimizer_steps_per_step=(1, 1, 1, 1, 1, 1),
            nonfinite_skips=0,
            optimizer_steps=6,
        )
    )
    assert result.passed


def test_injected_defect_step_assertion_removed_would_pass():
    """注入缺陷对照：删掉"每步 optimizer_steps>0"断言后，坏 run 会通过。

    这条不修改源码，而是**模拟**缺陷后的判定输入（把逐 step 证据抹掉，
    只剩下旧口径的步数与 ``weight_changed=True``）——证明旧口径确实会放过
    这个坏 run，从而证明新断言是必需的而非装饰。
    """
    # 步数必须取到门槛之上：旧口径只数"总步数"，4 步会被 *另一条* 判据
    # （MIN_OPTIMIZER_STEPS）拦下，那构不成对本隐患的证明。
    old_style = make_evidence(
        optimizer_steps=9, optimizer_steps_per_step=None, nonfinite_skips=None
    )
    assert judge_a2(old_style).passed  # 旧口径：9 步 ≥ 门槛、权重"变了" ⇒ 通过


# ── A④：loss: null 显式处理（NaN 哨兵）─────────────────────────────────────


def test_a6_nan_sentinel_is_never_healthy():
    """``loss: null`` 被提取层显式表示成 NaN 哨兵 ⇒ A6 必须不通过。"""
    result = judge_a6(make_evidence(losses=(0.0209, 13.125, float("nan"), float("nan"))))
    assert not result.passed


def test_null_run_still_fails_after_null_is_rewritten_to_zero():
    """★ 核心负向证据（F-4 ④）：把 null 改写成 0.0 后，该 run **仍判不通过**。

    旧路径下 ``grad_norms`` 全 finite（26.25 / 1.03e7 / 0.0 / 0.0），一旦
    ``loss: null`` 变成 ``0.0``，A6 会判"数值健康"。本用例断言：现在不会——
    原因不再是"null 恰好被解析成 NaN"，而是 ``optimizer_steps_per_step`` 抓到
    第 3/4 步 ``optimizer_steps=0``、``nonfinite_skips=2`` 抓到跳过。
    """
    zero_filled = make_evidence(
        losses=(0.0209, 13.125, 0.0, 0.0),
        grad_norms=(26.25, 1.035e7, 0.0, 0.0),
        optimizer_steps_per_step=(4, 4, 0, 0),
        nonfinite_skips=2,
    )
    assert not judge_a2(zero_filled).passed
    # A6 本身现在判"健康"（loss 全 finite 且不上升）——这正是旧隐患的入口。
    assert judge_a6(zero_filled).passed
    # 但整条判定链仍拦住它（A2 新断言是唯一拦住它的那条）。
    judgement = judge_tier(zero_filled, zero_filled, context_length=8192)
    assert judgement.ledger_status == "❌ 失败"


def test_injected_defect_zero_filled_null_would_pass_old_judgement():
    """注入缺陷对照：只保留旧口径的证据（无逐步断言、无跳过计数）时，
    把 null 改写成 0.0 就会让整个 run **通过** —— 这就是当初的隐患本体。

    该断言是"旧口径真的会漏"的可执行证明：一旦有人删掉 A③ 的新字段，
    本对照会失败。
    """
    old_style = make_evidence(
        optimizer_steps=9,
        losses=(0.0209, 13.125, 0.0, 0.0),
        grad_norms=(26.25, 1.035e7, 0.0, 0.0),
        optimizer_steps_per_step=None,
        nonfinite_skips=None,
    )
    # 旧口径下 A2/A6 **双双通过**（权重"变了"、数值"健康"）⇒ 坏 run 会被记成功。
    assert judge_a2(old_style).passed
    assert judge_a6(old_style).passed


# ── A①：非有限梯度硬失败标记 ────────────────────────────────────────────────


def test_hard_fail_marker_classified_as_numeric_anomaly():
    evidence = make_evidence(
        exit_code=1,
        losses=(0.5, float("nan")),
        grad_norms=(1.0, float("nan")),
        optimizer_steps_per_step=(1, 0),
        nonfinite_skips=1,
        log_text="RuntimeError: 非有限梯度：SFT 训练硬失败（fail-closed，宪法 §3.4）。",
    )
    # A6 只看损失/梯度序列；这里给它一个健康的序列，验证分类仍由日志里的
    # 硬失败标记兜底（A6 未抓到 NaN 时不得落到 OOM 文本分支）。
    a6 = judge_a6(
        RunEvidence(
            tier_id="T018",
            exit_code=1,
            timed_out=False,
            log_text=evidence.log_text,
            tuner_type="full",
            optimizer_steps=None,
            epochs_completed=None,
            weight_changed=None,
            checkpoint_reloadable=None,
            artifacts_present={},
            losses=(0.5,),
            grad_norms=(1.0,),
        )
    )
    assert a6.passed
    assert classify_failure(evidence, a6) == FailureClass.NUMERIC_ANOMALY


# ── 判定优先级（不得回归既有判据）──────────────────────────────────────────


def test_numeric_anomaly_wins_over_oom_text():
    """数值异常（A6 NaN）必须优先于日志里的 `CUDA out of memory` 文本。"""
    a6 = judge_a6(make_evidence(losses=(1.0, float("nan")), grad_norms=(1.0, 1.0)))
    assert not a6.passed
    evidence = make_evidence(log_text="torch.OutOfMemoryError: CUDA out of memory.")
    assert classify_failure(evidence, a6) == FailureClass.NUMERIC_ANOMALY


def test_real_oom_without_numeric_anomaly_still_counts():
    """对照：数值健康 + 日志含 OOM ⇒ 仍是真 OOM（该通道不得被关掉）。"""
    a6 = judge_a6(make_evidence(losses=(1.0, 0.9), grad_norms=(1.0, 0.9)))
    assert a6.passed
    evidence = make_evidence(log_text="torch.OutOfMemoryError: CUDA out of memory.")
    assert classify_failure(evidence, a6) == FailureClass.REAL_OOM


def test_numeric_indeterminate_never_becomes_real_oom():
    """``loss: null``（证据缺口）不得被 OOM 文本"劫持"成真 OOM。

    若它落进日志文本分支，该档上下文长度就会被记进"最大可行上下文"——
    正是必须堵死的那条污染路径。
    """
    from graspo.core.result_judge import NUMERIC_INDETERMINATE_DETAIL, CriterionResult

    indeterminate = CriterionResult("A6", False, NUMERIC_INDETERMINATE_DETAIL)
    evidence = make_evidence(log_text="torch.OutOfMemoryError: CUDA out of memory.")
    assert classify_failure(evidence, indeterminate) == FailureClass.UNCLASSIFIED
    assert not judge_tier(evidence, None).counts_toward_max_context


# ── 指挥官改判：三种情形必须各自可达、各自归类 ──────────────────────────────


def test_three_way_distinction_is_reachable():
    """三种情形（真 NaN / 仅 null / 真 OOM 无 NaN）各自可达且归类不同。

    这是"不允许保留写着却永不触发的分级"的可执行证明：把三种输入分别喂给
    A6 + classify_failure，断言：

    | 输入 | A6 | 归类 | fail-closed |
    |---|---|---|---|
    | 真读到 NaN | 不通过 "出现 NaN/Inf" | ``数值异常`` | ✔ |
    | 只有 null 无读数 | 不通过 "不可判定" | ``未分类（需人工判定）`` | ✔ |
    | 数值健康 + OOM 文本 | 通过 | ``真 OOM`` | 计入最大可行上下文 |
    """
    from graspo.core.result_judge import MISSING_SENTINEL

    oom_text = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"

    real_nan = make_evidence(
        losses=(0.5, float("nan")), grad_norms=(1.0, float("nan")), log_text=oom_text
    )
    a6 = judge_a6(real_nan)
    assert not a6.passed and "NaN/Inf" in a6.detail
    assert classify_failure(real_nan, a6) == FailureClass.NUMERIC_ANOMALY

    only_null = make_evidence(
        losses=(0.5, MISSING_SENTINEL),
        grad_norms=(1.0, 1.0),
        losses_nonfinite=False,
        losses_unavailable=True,
        log_text=oom_text,
    )
    a6_null = judge_a6(only_null)
    assert not a6_null.passed
    assert "不可判定" in a6_null.detail
    assert classify_failure(only_null, a6_null) == FailureClass.UNCLASSIFIED
    assert not judge_tier(only_null, None).counts_toward_max_context

    healthy_oom = make_evidence(losses=(1.0, 0.9), grad_norms=(1.0, 0.9), log_text=oom_text)
    a6_ok = judge_a6(healthy_oom)
    assert a6_ok.passed
    assert classify_failure(healthy_oom, a6_ok) == FailureClass.REAL_OOM

    # 前两种都 fail-closed（不计入最大可行上下文）
    assert not judge_tier(real_nan, None).counts_toward_max_context
    assert not judge_tier(only_null, None).counts_toward_max_context


def test_timeout_still_wins_over_everything():
    """超时排在最前（既有优先级不得被新分支顶掉）。"""
    a6 = judge_a6(make_evidence(losses=(1.0, float("nan")), grad_norms=(1.0, 1.0)))
    evidence = make_evidence(timed_out=True, exit_code=124, log_text="CUDA out of memory")
    assert classify_failure(evidence, a6) == FailureClass.TIMEOUT


def test_nan_sentinel_identity_is_strict():
    """哨兵必须是**同一个对象**：另一个 NaN 实例是"数值异常"，不是证据缺口。"""
    from graspo.core.result_judge import NAN_SENTINEL

    assert (float("nan") is NAN_SENTINEL) is False
    assert math.isnan(NAN_SENTINEL)
