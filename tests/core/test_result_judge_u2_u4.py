"""U2（loss 正常下降 / reward 正常上升）与 U4（产物可推理）的判据单测。

★ 本文件的四条纪律（对应工作包 `task-judge-b1` 的 ②③，含 2026-10-17 上级 **γ 裁定**）：

1. **窗口算法与唯一现存实现同源**：U2 的窗口 = `min(max(3, round(n*0.10)), n)`，逐字复刻
   `.local/.../task-run-T028/evidence/u2_window.py:27`（该脚本是**基线对照工具**：
   给 CUR/BASE 各打一组窗口，自身不做比较判定）；
2. **每条判据都必须**能被反例判红**（"恒 PASS"禁令）：末段更高的 loss 序列必须红、
   前向抛异常必须红、输出空/不可解析必须红、缺序列/无 GPU 必须 fail-closed
   且**显式** `evidence_missing`（"未测" ≠ "失败"）；
3. ★ **γ：U2/U4 都只在诊断轴** —— 判据层该红就红，但**不得**翻转整档 ✅/❌；
   且「为什么没接入」必须**逐字登记**在代码常量与档位备注里（本文件有专门用例锁死）；
4. ⛔ **不碰** `A6_LOSS_TREND_BLOCKS`。
"""

import pytest

from graspo.core.result_judge import (
    DIAGNOSTIC_CRITERIA,
    MISSING_SENTINEL,
    PRIMARY_CRITERIA,
    U2_CRITERION,
    U2_EVIDENCE_MISSING_DETAIL,
    U2_NONFINITE_DETAIL,
    U2_WIRING_PENDING_NOTE,
    U4_CRITERION,
    U4_WIRING_PENDING_NOTE,
    RunEvidence,
    a6_loss_trend,
    judge_a6,
    judge_tier,
    judge_u2,
    judge_u4,
    u2_window_means,
    u2_window_size,
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
        losses=(1.0, 0.8, 0.6, 0.5, 0.4, 0.3),
        grad_norms=(1.0, 0.9, 0.8),
        # A7 要"训练真推进"的正面读数；不给 ⇒ A7 落取证缺口、整档就判不了通过，
        # 那会把 U2/U4 的对照淹没在 A7 的缺证据里。
        nonfinite_skips=0,
        nonfinite_skips_source="unit_test",
    )
    base.update(overrides)
    return RunEvidence(**base)


# ── U2 · 窗口口径 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("point_count", "expected"),
    [
        (0, 0),
        (-1, 0),
        (1, 1),  # min(k, n)：窗口不得长于序列
        (3, 3),  # 下界 3 ⇒ max(3, 0)=3、min(3, 3)=3
        (20, 3),  # round(2.0)=2 ⇒ max(3, 2)=3
        (30, 3),
        (41, 4),  # round(4.1)=4
        (50, 5),  # round(5.0)=5
        (100, 10),  # ★ 与 228 实测 CPT 档 k=10 一致
        (200, 20),  # ★ 与 228 实测 CPT 档 k=20 一致
    ],
)
def test_u2_window_size_matches_the_in_force_implementation(point_count, expected):
    assert u2_window_size(point_count) == expected


def test_u2_window_means_is_first_and_last_k():
    start, end, k = u2_window_means((1.0, 2.0, 3.0, 4.0, 5.0))

    assert k == 3
    assert start == pytest.approx(2.0)  # mean(1,2,3)
    assert end == pytest.approx(4.0)  # mean(3,4,5)


def test_u2_window_means_on_empty_series_is_none_never_zero():
    """§2.2：读不到就返回 None，**不得**猜 0.0（0.0 会伪装成一个真实读数）。"""
    assert u2_window_means(()) is None


def test_u2_series_too_short_for_two_windows_is_an_evidence_gap_not_a_failure():
    """★退化边界：``k == len(values)`` 时首窗 == 终窗 ⇒ **无从比较** ⇒ 取证缺口。

    n ≤ 3 ⇒ k = min(max(3, 10%n), n) = n ⇒ 两窗是同一个集合。此时判红会把
    「这条序列承载不了该口径」误报成「训练没有下降」——「未测 ≠ 失败」要拦的正是这个。
    """
    for losses in ((0.5,), (0.5, 0.3), (0.5, 0.3, 0.1)):
        result = judge_u2(make_evidence(losses=losses))

        assert not result.passed, "无从比较 ⇒ fail-closed，绝不放行"
        assert result.evidence_missing is True, "这是取证缺口，不是训练失败"
        assert U2_EVIDENCE_MISSING_DETAIL in result.detail
        assert "无从比较" in result.detail


# ── U2 · 能失败对照（红色路径）────────────────────────────────────────────


def test_u2_fails_when_the_terminal_window_is_higher():
    """★能失败对照①：正常下降后**末段反弹更高** ⇒ U2 必须红。

    这不是"均值巧合"——反例被构造成末段窗严格高于起点窗；
    并且它**不是**取证缺口（序列完整读到了），必须记成**事实判否**。
    """
    result = judge_u2(make_evidence(losses=(1.0, 0.8, 0.5, 0.2, 0.1, 2.0, 3.0, 4.0)))

    assert not result.passed
    assert result.evidence_missing is False, "读全了序列 ⇒ 这是事实判否，不是取证缺口"
    assert result.criterion == U2_CRITERION
    assert "起点窗" in result.detail and "终点窗" in result.detail
    assert "未正常下降" in result.detail


def test_u2_red_control_is_red_at_judge_level_but_not_on_the_blocking_axis():
    """★ γ 裁定锁（2026-10-17 上级）：U2 判红 **不得**翻转整档结论，但必须**逐次记录**。

    背景：U2 的实现口径（终窗 对 自身起点窗｜自比）与文档化主判据（终窗 对
    基线同口径终点窗）**不是等价读法**，本层入参又拿不到基线证据 ⇒ U2 暂降回
    **诊断轴**。本用例把该状态锁死：
      * 判据层：这条序列必须**红**（截断第 129 行那段红色对照不变）；
      * 整档层：**不许**因为 U2 红而判 ❌（`passed` 仍为 True、状态仍 ✅）；
      * 记录层：U2 必须出现在 `criteria` 与档位备注里（不是静默丢弃）。
    """
    judgement = judge_tier(make_evidence(losses=(1.0, 0.8, 0.5, 0.2, 0.1, 2.0, 3.0, 4.0)))

    assert judgement.criterion(U2_CRITERION).passed is False, "判据层该红还得红"
    assert judgement.passed, "U2 暂不接入阻断轴 ⇒ 不得因它翻转整档结论"
    assert judgement.ledger_status == "✅ 可用"
    assert "U2 loss 正常下降 = False" in judgement.note, "U2 的结论必须写进档位备注"
    assert U2_WIRING_PENDING_NOTE in judgement.note, "偏离与不接入状态必须显式可见"


def test_u2_nonfinite_series_is_a_fact_failure_not_an_evidence_gap():
    """真读到 NaN/Inf ⇒ fail-loud 判否，**不**降格为取证缺口。"""
    for losses in (
        (1.0, float("nan"), 0.5),
        (1.0, float("inf"), 0.5),
        (1.0, 0.5, float("-inf")),
    ):
        result = judge_u2(make_evidence(losses=losses))

        assert not result.passed
        assert result.evidence_missing is False
        assert U2_NONFINITE_DETAIL in result.detail


# ── U2 · 绿色路径 ───────────────────────────────────────────────────────────


def test_u2_passes_on_normal_decline():
    result = judge_u2(make_evidence(losses=(1.0, 0.9, 0.8, 0.7, 0.6)))

    assert result.passed, result.detail
    assert result.evidence_missing is False
    assert "正常下降成立" in result.detail


def test_u2_passes_for_rising_reward():
    """GRASPO 支路：reward 越高越好（``higher_is_better=True``）。"""
    rising = judge_u2(make_evidence(losses=(0.1, 0.2, 0.3, 0.4, 0.5)), higher_is_better=True)
    falling = judge_u2(make_evidence(losses=(0.5, 0.4, 0.3, 0.2, 0.1)), higher_is_better=True)

    assert rising.passed
    assert not falling.passed, "reward 下降必须红（方向不能被忽略）"


# ── U2 · fail-closed（未测 ≠ 失败）────────────────────────────────────────


def test_u2_fail_closed_and_explicitly_marked_evidence_missing():
    """★能失败对照③：缺序列 ⇒ fail-closed，且**显式**标 `evidence_missing`。"""
    for evidence in (
        make_evidence(losses=()),
        make_evidence(losses_unavailable=True, losses=(1.0, 0.5)),
        make_evidence(losses=(1.0, MISSING_SENTINEL, 0.5)),
    ):
        result = judge_u2(evidence)

        assert not result.passed, "读不到序列绝不放行"
        assert result.evidence_missing is True, "未测必须记成取证缺口，不是训练失败"
        assert U2_EVIDENCE_MISSING_DETAIL in result.detail
        assert "fail-closed" in result.detail


def test_u2_missing_series_keeps_the_tier_indeterminate_not_failed():
    """整档视角：缺 loss 序列 ⇒ 「⚠ 口径不可测（取证缺口）」，⛔ 不是 ❌。"""
    judgement = judge_tier(make_evidence(losses=()))

    assert not judgement.passed
    assert judgement.ledger_status == "⚠ 口径不可测"
    assert judgement.failure_class is not None
    assert judgement.criterion(U2_CRITERION).evidence_missing is True


# ── U2 · 与 A6「首末两点」不同源（单一真相源守卫）─────────────────────────


def test_u2_window_and_a6_endpoints_are_different_sources_both_ways():
    """★证明 U2（窗口均值）与 A6 记录项（首末两点）**不同源**——两个方向各一例。

    两例都被构造成"首末两点说一套、窗口均值说另一套"：
    ⇒ 若哪天有人把 U2 实现成首末两点（或把 A6 那条记录项拿去顶 U2），
    本用例必红。
    """
    # ① 首末两点 = increased（A6 记记录项），窗口均值 = 下降（U2 绿）
    case_a = make_evidence(losses=(0.05, 5.0, 5.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.30))
    assert a6_loss_trend(judge_a6(case_a)) == "increased"
    assert judge_u2(case_a).passed, "窗口均值下降 ⇒ U2 必须绿（首末两点不是 U2 的口径）"

    # ② 首末两点 = decreased（A6 记记录项），窗口均值 = 上升（U2 红）
    case_b = make_evidence(losses=(0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0, 0.09))
    assert a6_loss_trend(judge_a6(case_b)) == "decreased"
    assert not judge_u2(case_b).passed, "窗口均值上升 ⇒ U2 必须红（不得用首末两点顶替）"


def test_u2_does_not_disturb_the_a6_trend_record():
    """⛔ 不碰 A6：接入 U2 **没有**改变 A6 的「首末走向只记录不阻断」裁定。"""
    evidence = make_evidence(losses=(0.5, 0.42, 0.7))

    a6 = judge_a6(evidence)

    assert a6.passed, "A6 仍只看数值健康（首末走向是记录项）"
    assert a6_loss_trend(a6) == "increased"


# ── U4 · 能失败对照 ─────────────────────────────────────────────────────────


def test_u4_fails_when_the_forward_pass_raises():
    """★能失败对照①：前向抛异常 ⇒ U4 必须红（事实判否，不是取证缺口）。"""
    result = judge_u4(
        make_evidence(
            inference_forward_ok=False,
            inference_output_parseable=False,
            inference_error="RuntimeError: CUDA error: no kernel image is available",
            inference_evidence_source="vllm_forward_once:bogus.log",
        )
    )

    assert not result.passed
    assert result.evidence_missing is False
    assert "抛异常" in result.detail
    assert "CUDA error" in result.detail


def test_u4_fails_when_the_output_is_empty_or_unparseable():
    """★能失败对照②：前向成功但输出空 / 不可解析 ⇒ U4 必须红。"""
    result = judge_u4(
        make_evidence(
            inference_forward_ok=True,
            inference_output_parseable=False,
            inference_error="空输出：未解析出任何 tool_call / JSON",
        )
    )

    assert not result.passed
    assert result.evidence_missing is False
    assert "为空 / 不可解析" in result.detail


def test_u4_fail_closed_when_there_is_no_gpu_or_no_forward_evidence():
    """★能失败对照③：无 GPU / 未做前向 ⇒ fail-closed 且**显式** `evidence_missing`。"""
    for evidence in (
        make_evidence(),  # 采集面根本没落这两个字段
        make_evidence(inference_forward_ok=None, inference_output_parseable=None),
        make_evidence(
            inference_forward_ok=None,
            inference_output_parseable=None,
            inference_error="no CUDA device available（本机无 GPU）",
        ),
    ):
        result = judge_u4(evidence)

        assert not result.passed, "无 GPU 绝不放行、也绝不默认通过"
        assert result.evidence_missing is True, "「未测」必须记成取证缺口"
        assert "fail-closed" in result.detail
        assert "未测" in result.detail or "无 GPU" in result.detail


def test_u4_passes_only_when_forward_succeeded_and_output_parsed():
    result = judge_u4(
        make_evidence(
            inference_forward_ok=True,
            inference_output_parseable=True,
            inference_evidence_source="vllm_forward_once:serve.log",
        )
    )

    assert result.passed, result.detail
    assert result.evidence_missing is False
    assert "可解析" in result.detail


# ── U4 · 接入状态自证（显式登记，不静默）──────────────────────────────────


def test_u4_is_computed_and_recorded_by_judge_tier():
    judgement = judge_tier(make_evidence())

    assert judgement.criterion(U4_CRITERION).passed is False
    assert judgement.criterion(U4_CRITERION).evidence_missing is True
    assert U4_CRITERION in DIAGNOSTIC_CRITERIA
    assert U4_CRITERION not in PRIMARY_CRITERIA


def test_u4_not_being_wired_is_explicit_in_the_tier_note_not_silent():
    """★防呆：U4 尚未接入阻断轴这件事必须**写在台账备注里**，不得静默（§2.2）。

    反例保护：若有人悄悄把 U4 接进阻断轴（或悄悄把这段说明删掉），本用例必红。
    """
    judgement = judge_tier(make_evidence())

    assert judgement.passed, "U4 未接入阻断轴 ⇒ 本档仍可判通过"
    assert U4_WIRING_PENDING_NOTE in judgement.note
    assert "U4 产物可推理 = False" in judgement.note


def test_u2_and_u4_are_in_the_declared_criteria_tuples():
    """台账自述（§1.4）：判据清单元组必须与 judge_tier 的实际判定轴一致。

    ★ γ 裁定（2026-10-17 上级）：`U2` 与 `U4` **都在诊断轴、都不在阻断轴**。
    """
    assert U2_CRITERION in DIAGNOSTIC_CRITERIA
    assert U2_CRITERION not in PRIMARY_CRITERIA, "U2 必须已撤出阻断轴（γ）"
    assert U4_CRITERION in DIAGNOSTIC_CRITERIA
    assert U4_CRITERION not in PRIMARY_CRITERIA


# ── U2 · 接入状态自证（γ：撤出阻断轴 ＋ 显式登记 ＋ 测试锁死）───────────────


def test_u2_is_computed_and_recorded_by_judge_tier():
    """U2 撤出阻断轴 ≠ 删除：仍必须逐次计算并记录在 `criteria` 里。"""
    judgement = judge_tier(make_evidence())

    assert judgement.criterion(U2_CRITERION).passed is True  # 100 点下降那条夹具
    assert U2_CRITERION in {c.criterion for c in judgement.criteria}


def test_u2_not_being_wired_is_explicit_in_the_tier_note_not_silent():
    """★γ 锁：U2 暂不接入阻断轴 ＋「实现≠文档口径」这条偏离，必须**写在台账备注里**。

    反例保护：若有人悄悄把 U2 接回 `primary_status`（或悄悄删掉 `U2_WIRING_PENDING_NOTE`
    这段登记），本用例必红——这正是 v14 判 🔴 的「同一文件两个口径且无一句登记偏离」。
    """
    judgement = judge_tier(make_evidence())

    assert judgement.passed, "U2 未接入阻断轴 ⇒ 本档仍可判通过"
    assert U2_WIRING_PENDING_NOTE in judgement.note
    assert "U2 loss 正常下降 = True" in judgement.note


def test_u2_wiring_pending_note_records_the_divergence_verbatim():
    """★γ 锁：登记文本必须**点名**两套读法与两条待裁路径（不能只剩一句"暂不接入"）。"""
    note = U2_WIRING_PENDING_NOTE

    assert "自比" in note, "必须写明实现口径 = 对自身起点窗"
    assert "基线" in note, "必须写明文档化主判据 = 对基线同口径终点窗"
    assert "scripts/collect_results.py" in note, "必须写明 α 路径的证据落点"
    assert "docs/**" in note, "必须写明 β 路径需同批改写原文"
