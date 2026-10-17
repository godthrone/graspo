"""CPT_LOSS（判据② 的 CPT 支）与 U4（产物可推理）的判据单测。

★ **本文件的历史与退役登记**（2026-10-17，工作包 `task-judge-retire`，P3 判据体系面）：

  旧 ``U2``（"loss 正常下降 / reward 正常上升"、窗口 + 基线同口径终窗 + 方向开关）
  整套机制**已随判据体系收口为四条而整体退役**：它既不在冻结口径的四条判据里，
  又被逐字点名为作废（U1–U4 判据体系 / U2 窗口 / 终点窗口 / 起点窗口 /
  "基线同口径终窗" / 单卡曲线基线）。⇒ ``judge_u2`` / ``u2_window_*`` /
  ``U2_*`` 常量 / ``RunEvidence.baseline_end_window`` **删净**，其单测**逐条具名删除**。

  **保留下来并升级为真实强制点的是 CPT 的 loss 判据**：
  ``judge_cpt_loss``（= :data:`CPT_LOSS_CRITERION`）现列在 :data:`PRIMARY_CRITERIA`
  里 —— 主席裁定「CPT 的『训练 loss 正常下降』必须保留」，而旧 U2 落在诊断轴时
  这句话落不到任何 ✅/❌ 上。它**只对** ``algorithm == "CPT"`` 生效。

★ 本文件的纪律（对应工作包 ②③⑤⑥）：

1. **窗口算法沿用现行口径**：``min(max(3, round(n*0.10)), n)``；
2. ★★ **CPT_LOSS 的判定式 = 「终窗均值 < 起窗均值」**（严格、loss 越低越好）；
   ⛔ **退出基线依赖**（不再与"未训练底座 loss 终窗"比较，单卡曲线基线已作废）；
3. **每条判据都必须能被反例判红**（"恒 PASS"禁令）：loss 未下降必须红、缺序列
   fail-closed 且**显式** ``evidence_missing``（"未测" ≠ "失败"）、NaN/Inf fail-loud、
   ⛔ **只对 CPT 生效**（SFT/GRASPO/OPD 判 loss ⇒ 必红）；
4. ⛔ **不碰** ``A6_LOSS_TREND_BLOCKS``；CPT_LOSS（首尾窗口均值）与 A6（首末两点）
   **不同源**，两个方向各有一例钉死。

★ **逐条具名删除的旧 U2 用例**（每条的理由一律 =「对应判据已不存在」；⛔ 未用
  skip/xfail，⛔ 未删断言凑绿）：

  1. ``test_u2_window_size_matches_the_in_force_implementation``
  2. ``test_u2_window_means_is_first_and_last_k``
  3. ``test_u2_window_means_on_empty_series_is_none_never_zero``
  4. ``test_u2_short_series_is_judged_against_the_baseline_like_the_caliber_says``
  5. ``test_u2_fails_when_the_terminal_window_is_higher``
  6. ``test_u2_red_control_is_red_at_judge_level_but_not_on_the_blocking_axis``
  7. ``test_u2_caliber_is_locked_against_self_comparison``
  8. ``test_u2_nonfinite_series_is_a_fact_failure_not_an_evidence_gap``
  9. ``test_u2_nonfinite_or_unusable_baseline_is_fail_loud_not_an_evidence_gap``
  10. ``test_u2_passes_on_normal_decline``
  11. ``test_u2_attaining_the_baseline_exactly_passes``
  12. ``test_u2_fails_when_the_end_window_is_worse_than_the_baseline``
  13. ``test_u2_missing_baseline_is_an_evidence_gap_not_a_pass``
  14. ``test_u2_keeps_the_start_window_as_attribution_diagnostic_only``
  15. ``test_u2_passes_for_rising_reward``
  16. ``test_u2_fail_closed_and_explicitly_marked_evidence_missing``
  17. ``test_u2_missing_series_keeps_the_tier_indeterminate_not_failed``
  18. ``test_u2_window_and_a6_endpoints_are_different_sources_both_ways``
  19. ``test_u2_does_not_disturb_the_a6_trend_record``
  20. ``test_u2_direction_mapping_is_explicit_exact_and_fail_closed``
  21. ``test_u2_reward_evidence_is_judged_by_reward_direction_both_ways``
  22. ``test_u2_loss_evidence_keeps_the_mirror_direction``
  23. ``test_u2_direction_unknown_is_fail_closed_and_never_defaults_to_loss``
  24. ``test_u2_direction_unknown_keeps_the_tier_indeterminate_in_the_note_not_silent``
  25. ``test_u2_judge_tier_call_site_derives_the_direction_from_the_algorithm``
  26. ``test_u2_explicit_direction_argument_is_an_explicit_override_never_a_silent_default``
  27. ``test_u2_is_computed_and_recorded_by_judge_tier``
  28. ``test_u2_not_being_wired_is_explicit_in_the_tier_note_not_silent``
  29. ``test_u2_wiring_pending_note_records_the_new_state_and_both_readings``

  ``test_u2_and_u4_are_in_the_declared_criteria_tuples`` **未删**，按其新语义改写为
  ``test_cpt_loss_is_primary_and_u4_is_diagnostic``。
"""

import pytest

from graspo.core.result_judge import (
    CPT_LOSS_ALGORITHM,
    CPT_LOSS_CRITERION,
    CPT_LOSS_EVIDENCE_MISSING_DETAIL,
    CPT_LOSS_NONFINITE_DETAIL,
    CPT_LOSS_NOT_APPLICABLE_DETAIL,
    DIAGNOSTIC_CRITERIA,
    LEDGER_FAIL,
    LEDGER_INDETERMINATE,
    LEDGER_PASS,
    LEDGER_STATUS_PHRASES,
    MISSING_SENTINEL,
    PRIMARY_CRITERIA,
    U4_CRITERION,
    U4_WIRING_PENDING_NOTE,
    RunEvidence,
    a6_loss_trend,
    cpt_loss_window_means,
    cpt_loss_window_size,
    judge_a6,
    judge_cpt_loss,
    judge_tier,
    judge_u4,
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
        # ★ 缺省算法 = `SFT`：CPT_LOSS **不适用**于它（⛔ 不得让 SFT 去判 loss）。
        #   CPT 支的用例一律**显式**给 ``algorithm="CPT"``。
        algorithm="SFT",
        # A7 要"训练真推进"的正面读数；不给 ⇒ A7 落取证缺口、整档就判不了通过，
        # 那会把 CPT_LOSS/U4 的对照淹没在 A7 的缺证据里。
        nonfinite_skips=0,
        nonfinite_skips_source="unit_test",
    )
    base.update(overrides)
    return RunEvidence(**base)


#: 一条"明显下降"的 CPT loss 序列（n=20 ⇒ k=3）。
CPT_DECLINING = tuple(1.0 - 0.04 * index for index in range(20))

#: 一条"未下降（反而上升）"的 CPT loss 序列（n=20 ⇒ k=3）。
CPT_NOT_DECLINING = tuple(0.5 + 0.03 * index for index in range(20))


# ── CPT_LOSS · 窗口口径 ─────────────────────────────────────────────────────


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
def test_cpt_loss_window_size_matches_the_caliber(point_count, expected):
    assert cpt_loss_window_size(point_count) == expected


def test_cpt_loss_window_means_is_first_and_last_k():
    start, end, k = cpt_loss_window_means((1.0, 2.0, 3.0, 4.0, 5.0))

    assert k == 3
    assert start == pytest.approx(2.0)  # mean(1,2,3)
    assert end == pytest.approx(4.0)  # mean(3,4,5)


def test_cpt_loss_window_means_on_empty_series_is_none_never_zero():
    """§2.2：读不到就返回 None，**不得**猜 0.0（0.0 会伪装成一个真实读数）。"""
    assert cpt_loss_window_means(()) is None


# ── CPT_LOSS · 能失败对照（红色路径）────────────────────────────────────────


def test_cpt_loss_fails_when_the_terminal_window_is_not_lower():
    """★能失败对照①：loss **未下降**（上升）⇒ CPT_LOSS 必须红。

    这不是"均值巧合"——反例被构造成终窗均值严格高于起窗均值；
    并且它**不是**取证缺口（序列读到了），必须记成**事实判否**。
    """
    rising = judge_cpt_loss(make_evidence(algorithm="CPT", losses=CPT_NOT_DECLINING))

    assert not rising.passed, rising.detail
    assert rising.evidence_missing is False, "读全了序列 ⇒ 这是事实判否，不是取证缺口"
    assert rising.criterion == CPT_LOSS_CRITERION
    assert "起窗" in rising.detail and "终窗" in rising.detail
    assert "未下降" in rising.detail


def test_cpt_loss_is_strict_equality_does_not_pass():
    """操作定义 = 「终窗均值 **<** 起窗均值」⇒ 持平（逐位相等）**不通过**（严格小于）。"""
    flat = judge_cpt_loss(make_evidence(algorithm="CPT", losses=(0.5,) * 10))

    assert not flat.passed, "起窗 == 终窗 ⇒ 不构成下降，必须红"
    assert flat.evidence_missing is False


def test_cpt_loss_nonfinite_series_is_a_fact_failure_not_an_evidence_gap():
    """真读到 NaN/Inf ⇒ fail-loud 判否，**不**降格为取证缺口。"""
    for losses in (
        (1.0, float("nan"), 0.5, 0.2),
        (1.0, float("inf"), 0.5, 0.2),
        (1.0, 0.5, float("-inf"), 0.2),
    ):
        result = judge_cpt_loss(make_evidence(algorithm="CPT", losses=losses))

        assert not result.passed
        assert result.evidence_missing is False
        assert CPT_LOSS_NONFINITE_DETAIL in result.detail


# ── CPT_LOSS · 绿色路径 ─────────────────────────────────────────────────────


def test_cpt_loss_passes_on_normal_decline():
    result = judge_cpt_loss(make_evidence(algorithm="CPT", losses=CPT_DECLINING))

    assert result.passed, result.detail
    assert result.evidence_missing is False
    assert "下降成立" in result.detail


def test_cpt_loss_has_no_baseline_dependency_at_all():
    """★★ **退出基线依赖的机核证明**：判定式只比本跑自身两窗。

    ① ``RunEvidence`` **不再有** ``baseline_end_window`` 字段（旧基线通道已删）；
    ② 两条除"起窗 → 终窗走向"外**完全相同**的序列给出相反结论 —— 若实现偷偷
       依赖任何外部基线，两者会得到同一结论。
    """
    assert not hasattr(RunEvidence, "baseline_end_window"), (
        "旧 U2 的基线字段必须已删净（单卡曲线基线已作废）"
    )

    declining = make_evidence(
        algorithm="CPT", losses=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
    )
    not_declining = make_evidence(
        algorithm="CPT", losses=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    )

    assert judge_cpt_loss(declining).passed
    assert not judge_cpt_loss(not_declining).passed


@pytest.mark.parametrize("algorithm", ["SFT", "GRASPO", "OPD", None, "", "cpt", "GRPO"])
def test_cpt_loss_only_applies_to_cpt(algorithm):
    """⛔ **只对 CPT 生效**：其余任何算法串 ⇒ 「不适用」（通过且**显式**登记）。

    关键反例保护：若有人把本判据做成"对所有算法判 loss"，则上表里的 SFT/GRASPO
    档会因 loss 未下降被判红 —— 那正是本用例要拦下的越权。
    """
    result = judge_cpt_loss(make_evidence(algorithm=algorithm, losses=CPT_NOT_DECLINING))

    assert result.passed, f"algorithm={algorithm!r} 不是 CPT ⇒ 必须判「不适用」而不是红"
    assert result.not_applicable is True
    assert result.evidence_missing is False
    assert "不适用" in result.detail
    assert CPT_LOSS_NOT_APPLICABLE_DETAIL in result.detail
    assert CPT_LOSS_ALGORITHM == "CPT", "适用算法串必须逐字 = CPT（精确匹配，⛔ 不猜）"


# ── CPT_LOSS · fail-closed（未测 ≠ 失败）────────────────────────────────────


def test_cpt_loss_fail_closed_and_explicitly_marked_evidence_missing():
    """★能失败对照③：缺序列 ⇒ fail-closed（**绝不放行**），且**显式**标 `evidence_missing`。"""
    for evidence in (
        make_evidence(algorithm="CPT", losses=()),
        make_evidence(algorithm="CPT", losses_unavailable=True, losses=(1.0, 0.5)),
        make_evidence(algorithm="CPT", losses=(1.0, MISSING_SENTINEL, 0.5)),
    ):
        result = judge_cpt_loss(evidence)

        assert not result.passed, "读不到序列绝不放行（缺数据 ⇒ fail-closed）"
        assert result.evidence_missing is True, "未测必须记成取证缺口，不是训练失败"
        assert CPT_LOSS_EVIDENCE_MISSING_DETAIL in result.detail
        assert "fail-closed" in result.detail


# ── CPT_LOSS · 与 A6「首末两点」不同源（单一真相源守卫）─────────────────────


def test_cpt_loss_window_and_a6_endpoints_are_different_sources_both_ways():
    """★证明 CPT_LOSS（窗口均值）与 A6 记录项（首末两点）**不同源**——两个方向各一例。

    两例都被构造成"首末两点说一套、窗口均值说另一套"：
    ⇒ 若哪天有人把 CPT_LOSS 实现成首末两点（或把 A6 那条记录项拿去顶它），本用例必红。
    """
    # ① 首末两点 = increased（A6 记记录项），窗口均值下降 ⇒ CPT_LOSS 绿
    case_a = make_evidence(
        algorithm="CPT", losses=(0.05, 5.0, 5.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.30)
    )
    assert a6_loss_trend(judge_a6(case_a)) == "increased"
    assert judge_cpt_loss(case_a).passed, "窗口均值下降 ⇒ CPT_LOSS 必须绿（首末两点不是它的口径）"

    # ② 首末两点 = decreased（A6 记记录项），窗口均值上升 ⇒ CPT_LOSS 红
    case_b = make_evidence(
        algorithm="CPT", losses=(0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0, 0.09)
    )
    assert a6_loss_trend(judge_a6(case_b)) == "decreased"
    assert not judge_cpt_loss(case_b).passed, "窗口均值上升 ⇒ CPT_LOSS 必须红（不得用首末两点顶替）"


def test_cpt_loss_does_not_disturb_the_a6_trend_record():
    """⛔ 不碰 A6：接入 CPT_LOSS **没有**改变 A6 的「首末走向只记录不阻断」裁定。"""
    evidence = make_evidence(algorithm="CPT", losses=(0.5, 0.42, 0.7))

    a6 = judge_a6(evidence)

    assert a6.passed, "A6 仍只看数值健康（首末走向是记录项）"
    assert a6_loss_trend(a6) == "increased"


# ── CPT_LOSS · 真实强制点（判据② 的 CPT 支）与整档判定 ──────────────────────


def test_cpt_loss_is_primary_and_u4_is_diagnostic():
    """★台账自述（§1.4）：判据清单元组必须与 `judge_tier` 的实际判定轴一致。

    ★ 2026-10-17 状态：``CPT_LOSS`` 是**主判据**（判据② 的 CPT 支，**真能判 ✅/❌**）；
    ``A4`` 与 ``U4`` 是**诊断项**（⛔ 不参与 ✅/❌）；旧 ``U2`` 名**不再存在**。
    """
    assert CPT_LOSS_CRITERION in PRIMARY_CRITERIA, "CPT 的 loss 判据必须是真实强制点"
    assert U4_CRITERION in DIAGNOSTIC_CRITERIA
    assert U4_CRITERION not in PRIMARY_CRITERIA
    assert "A4" in DIAGNOSTIC_CRITERIA
    assert "A4" not in PRIMARY_CRITERIA, "A4 容差体系已退役「判据」身份、降级为内部工具"
    assert "U2" not in DIAGNOSTIC_CRITERIA and "U2" not in PRIMARY_CRITERIA


def test_retired_u2_symbols_are_gone():
    """★★ 退役自证：旧 U2 机制的符号**一个都不在**模块命名空间里。"""
    import graspo.core.result_judge as module

    for name in (
        "judge_u2",
        "u2_higher_is_better_from_algorithm",
        "u2_direction_label",
        "u2_window_size",
        "u2_window_means",
        "U2_CRITERION",
        "U2_REWARD_ALGORITHMS",
        "U2_LOSS_ALGORITHMS",
        "U2_WIRING_PENDING_NOTE",
        "U2_BASELINE_MISSING_DETAIL",
        "U2_EVIDENCE_MISSING_DETAIL",
        "U2_NONFINITE_DETAIL",
        "U2_DIRECTION_UNKNOWN_DETAIL",
        "U2_WINDOW_MIN",
        "U2_WINDOW_FRACTION",
    ):
        assert not hasattr(module, name), f"已退役的 U2 符号 {name} 不得残留"
    assert not hasattr(module.RunEvidence, "baseline_end_window")


def test_cpt_loss_is_a_real_blocking_point_on_the_tier_axis():
    """★★**本包最重要的断言**：CPT 的 loss 未下降 ⇒ **整档必须判不通过**。

    构造：除 CPT_LOSS 外全部主判据都过（exit=0 / 步数达标 / 权重变化 / ckpt 可重载 /
    产物齐全 / 数值 finite / 无 nonfinite 跳过），唯独 loss 上升。
    ⇒ 旧口径下这是「✅ 可用」；接上真实强制点后必须翻成「已验证不可用」。
    """
    judgement = judge_tier(make_evidence(algorithm="CPT", losses=CPT_NOT_DECLINING))

    assert judgement.criterion(CPT_LOSS_CRITERION).passed is False
    assert judgement.passed is False, "CPT 的 loss 判据必须是真强制点，不得只记录"
    assert judgement.ledger_status == LEDGER_FAIL


def test_cpt_loss_passing_keeps_the_tier_passing():
    """镜像（正向）：同一档只把 loss 换成明显下降 ⇒ 整档恢复「已验证可用」。"""
    judgement = judge_tier(make_evidence(algorithm="CPT", losses=CPT_DECLINING))

    assert judgement.criterion(CPT_LOSS_CRITERION).passed is True
    assert judgement.passed is True
    assert judgement.ledger_status == LEDGER_PASS


def test_cpt_loss_missing_series_keeps_the_tier_unverified_not_failed():
    """★整档视角：CPT 缺 loss 序列 ⇒ 「尚未验证」（取证缺口），⛔ **不是**「已验证不可用」。"""
    judgement = judge_tier(make_evidence(algorithm="CPT", losses=()))

    assert not judgement.passed
    assert judgement.ledger_status == LEDGER_INDETERMINATE
    assert judgement.failure_class is not None
    assert judgement.criterion(CPT_LOSS_CRITERION).evidence_missing is True


def test_cpt_loss_does_not_apply_to_non_cpt_tiers_on_the_tier_axis():
    """⛔ **不得让 SFT / GRASPO / OPD 也去判 loss**：同一份"未下降"的序列，
    在非 CPT 档上**不得**翻转整档结论。"""
    for algorithm in ("SFT", "GRASPO", "OPD"):
        judgement = judge_tier(make_evidence(algorithm=algorithm, losses=CPT_NOT_DECLINING))

        assert judgement.criterion(CPT_LOSS_CRITERION).passed is True, algorithm
        assert judgement.criterion(CPT_LOSS_CRITERION).not_applicable is True, algorithm
        assert judgement.passed is True, f"{algorithm} 档不判 loss ⇒ 结论不得被翻转"


# ── 台账状态词 = 冻结四态（六态/三态标签已作废）────────────────────────────


def test_ledger_status_vocabulary_is_the_frozen_four_states():
    """★台账状态词 = 冻结四态；⛔ 旧的六态/三态标签不得再出现。"""
    assert LEDGER_STATUS_PHRASES == (
        "已验证可用",
        "已验证不可用",
        "尚未验证",
        "未纳入验收",
    )
    assert len(set(LEDGER_STATUS_PHRASES)) == 4
    assert LEDGER_PASS == "已验证可用"
    assert LEDGER_FAIL == "已验证不可用"
    assert LEDGER_INDETERMINATE == "尚未验证"


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
