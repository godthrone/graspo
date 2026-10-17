"""U2（loss 正常下降 / reward 正常上升）与 U4（产物可推理）的判据单测。

★ 本文件的四条纪律（对应工作包 `task-u2-align` 的 ②③⑤⑥；2026-10-17 起 U2 口径按
**文档原文**对齐）：

1. **窗口算法与唯一现存实现同源**：U2 的窗口 = `min(max(3, round(n*0.10)), n)`，逐字复刻
   `.local/.../task-run-T028/evidence/u2_window.py:27`（该脚本是**基线对照工具**：
   给 CUR/BASE 各打一组窗口，自身不做比较判定）；
2. ★★ **U2 主判据 = 文档化口径「终点窗口达到/超过基线同口径终点窗口」**，起点窗口
   **仅作归因诊断**；⛔ **不得**改回「终窗 对 起窗」的**自比**读法
   （`test_u2_caliber_is_locked_against_self_comparison` **双向**钉死：两个方向的用例
   在自比实现下都会红）；
3. **每条判据都必须能被反例判红**（"恒 PASS"禁令）：终窗劣于基线必须红、前向抛异常
   必须红、输出空/不可解析必须红、缺序列/缺基线/无 GPU 必须 fail-closed 且**显式**
   `evidence_missing`（"未测" ≠ "失败"）；U2/U4 都只在**诊断轴**（判据层该红就红，
   ⛔ 不得翻转整档 ✅/❌），且「为什么没接线」必须**逐字登记**在代码常量与档位备注里；
4. ⛔ **不碰** `A6_LOSS_TREND_BLOCKS`。
"""

import pytest

from graspo.core.result_judge import (
    DIAGNOSTIC_CRITERIA,
    MISSING_SENTINEL,
    PRIMARY_CRITERIA,
    U2_BASELINE_MISSING_DETAIL,
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
        # ★ U2 的比较基准（口径 = `meta.criteria_caliber.baseline`）。
        #   缺省给一个"本跑终窗（0.4）达到/超过它"的基线 ⇒ 缺省夹具在 U2 上判绿，
        #   免得 U2/U4 的对照被"基线证据未到"淹没。缺基线那一态有专门用例。
        baseline_end_window=0.5,
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


def test_u2_short_series_is_judged_against_the_baseline_like_the_caliber_says():
    """★ 退化边界（**口径对齐后的归因更正**，2026-10-17）：``k == len(values)`` 的短序列
    **不再**因为"首窗 == 终窗、无从自比"落取证缺口 —— 那个理由属于**自比**读法。

    口径原文（`docs/capability-matrix-milestone-20261014.html:280`）给的是
    「**末段** max(3, 10% 步数) 的均值」与「终点窗口 达到/超过**基线**同口径终点窗口」：
    末段窗对任何 ``n ≥ 1`` 都可算，文档**没有**把"序列长度"设为判定前置 ⇒ 本层
    ⛔ 不自造附加条件。但**放行方向不变**：短序列照样要能判红（见本用例的两条断言）。

    ★ 变更登记（⛔ 不是顺手改）：本用例替换旧用例
    `test_u2_series_too_short_for_two_windows_is_an_evidence_gap_not_a_failure`
    （旧的"取证缺口"结论在自比口径下成立、在文档口径下不成立）。同时**加强**：
    ① 缺基线 ⇒ 仍是 fail-closed 取证缺口；② 短序列劣于基线 ⇒ **必红**（绝不放行）。
    """
    # ① 短序列 + 基线在 ⇒ 按「终点窗 对 基线窗」判：末段窗 = 全序列均值 0.333… ≤ 0.4 ⇒ 绿
    short_ok = judge_u2(make_evidence(losses=(0.5, 0.3, 0.2), baseline_end_window=0.4))
    assert short_ok.passed, short_ok.detail
    assert short_ok.evidence_missing is False

    # ② 短序列劣于基线 ⇒ **必红**（事实判否）：均值 0.566… > 0.3
    short_bad = judge_u2(make_evidence(losses=(0.5, 0.3, 0.9), baseline_end_window=0.3))
    assert not short_bad.passed
    assert short_bad.evidence_missing is False
    assert "未正常下降" in short_bad.detail

    # ③ 短序列 + **缺基线** ⇒ 仍 fail-closed（取证缺口，⛔ 不得降格为通过）
    for losses in ((0.5,), (0.5, 0.3), (0.5, 0.3, 0.1)):
        result = judge_u2(make_evidence(losses=losses, baseline_end_window=None))

        assert not result.passed, "缺基线 ⇒ fail-closed，绝不放行"
        assert result.evidence_missing is True, "这是取证缺口，不是训练失败"
        assert U2_BASELINE_MISSING_DETAIL in result.detail


# ── U2 · 能失败对照（红色路径）────────────────────────────────────────────


def test_u2_fails_when_the_terminal_window_is_higher():
    """★能失败对照①：正常下降后**末段反弹更高** ⇒ U2 必须红。

    这不是"均值巧合"——反例被构造成末段窗（3.0）严格劣于基线窗（0.5）；
    并且它**不是**取证缺口（序列与基线都读到了），必须记成**事实判否**。
    """
    result = judge_u2(make_evidence(losses=(1.0, 0.8, 0.5, 0.2, 0.1, 2.0, 3.0, 4.0)))

    assert not result.passed
    assert result.evidence_missing is False, "读全了序列与基线 ⇒ 这是事实判否，不是取证缺口"
    assert result.criterion == U2_CRITERION
    assert "起点窗" in result.detail and "终点窗" in result.detail
    assert "未正常下降" in result.detail


def test_u2_red_control_is_red_at_judge_level_but_not_on_the_blocking_axis():
    """★ 诊断轴锁：U2 判红 **不得**翻转整档结论，但必须**逐次记录**。

    背景：U2 的口径已按文档对齐（终窗 对 基线同口径终点窗），但**接线待裁**、
    且基线证据的采集落点在本层边界外 ⇒ U2 落在**诊断轴**。本用例把该状态锁死：
      * 判据层：这条序列必须**红**（上一条红色对照不变）；
      * 整档层：**不许**因为 U2 红而判 ❌（`passed` 仍为 True、状态仍 ✅）；
      * 记录层：U2 必须出现在 `criteria` 与档位备注里（不是静默丢弃）。
    """
    judgement = judge_tier(make_evidence(losses=(1.0, 0.8, 0.5, 0.2, 0.1, 2.0, 3.0, 4.0)))

    assert judgement.criterion(U2_CRITERION).passed is False, "判据层该红还得红"
    assert judgement.passed, "U2 未接入阻断轴 ⇒ 不得因它翻转整档结论"
    assert judgement.ledger_status == "✅ 可用"
    assert "U2 loss 正常下降 = False" in judgement.note, "U2 的结论必须写进档位备注"
    assert U2_WIRING_PENDING_NOTE in judgement.note, "接线状态必须显式可见"


def test_u2_caliber_is_locked_against_self_comparison():
    """★★ **口径锁（本包最关键的新增用例）**：把 U2 改回「终窗 对 起窗」（自比）⇒ **必红**。

    两个方向各构造一例，使「自比读法」与「文档口径（对基线）**结论相反**」：
      ① **终窗优于起点窗**（自比 ⇒ 绿）但 **终窗劣于基线窗**（文档口径 ⇒ 红）
         ⇒ 必须 **红**；
      ② **终窗劣于起点窗**（自比 ⇒ 红）但 **终窗达到/超过基线窗**（文档口径 ⇒ 绿）
         ⇒ 必须 **绿**。
    任一条只要有人把主判据换回起点窗（或拿起点窗顶替基线窗），本用例必红。
    """
    # ① k=3：起窗 (1.0,0.9,0.8)=0.9 → 终窗 (0.2,0.15,0.1)=0.15（自比=降）；基线窗 0.05
    case_self_green_baseline_red = make_evidence(
        losses=(1.0, 0.9, 0.8, 0.7, 0.4, 0.2, 0.15, 0.1), baseline_end_window=0.05
    )
    result_a = judge_u2(case_self_green_baseline_red)
    assert not result_a.passed, "终窗劣于基线 ⇒ 必红（自比读法会误判为绿）"
    assert result_a.evidence_missing is False

    # ② k=3：起窗 (0.1,0.1,0.1)=0.1 → 终窗 (0.6,0.6,0.6)=0.6（自比=升）；基线窗 0.7
    case_self_red_baseline_green = make_evidence(
        losses=(0.1, 0.1, 0.1, 0.2, 0.3, 0.6, 0.6, 0.6), baseline_end_window=0.7
    )
    result_b = judge_u2(case_self_red_baseline_green)
    assert result_b.passed, "终窗达到/超过基线 ⇒ 必绿（自比读法会误判为红）"
    assert "正常下降成立" in result_b.detail


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


def test_u2_nonfinite_or_unusable_baseline_is_fail_loud_not_an_evidence_gap():
    """基线窗读数本身不可用（NaN/Inf/非数值）⇒ fail-loud 判否，**不**降格为取证缺口。

    为什么与"缺基线"（取证缺口）分开：这是**读到了、但读到的是坏数**——关于证据的
    事实，与「没读到」不同（§2.2 三态语义）。
    """
    for baseline in (float("nan"), float("inf"), float("-inf"), "n/a"):
        result = judge_u2(make_evidence(baseline_end_window=baseline))

        assert not result.passed
        assert result.evidence_missing is False, "坏数 ≠ 没读到"
        assert U2_NONFINITE_DETAIL in result.detail


# ── U2 · 绿色路径 ───────────────────────────────────────────────────────────


def test_u2_passes_on_normal_decline():
    result = judge_u2(make_evidence(losses=(1.0, 0.9, 0.8, 0.7, 0.6), baseline_end_window=0.75))

    assert result.passed, result.detail
    assert result.evidence_missing is False
    assert "正常下降成立" in result.detail


def test_u2_attaining_the_baseline_exactly_passes():
    """口径原文「终点窗口 **达到** / 超过基线同口径的终点窗口」⇒ 相等 = **通过**（非严格）。

    ★ 用**二进制可精确表示**的值（0.5）构造"逐位相等"，⛔ 不用 0.4 这类十进制近似值
    （那会在非严格比较的另一侧因浮点尾巴翻出假红，与口径无关）。
    """
    # k=3：终窗 = mean(0.5, 0.5, 0.5) = 0.5，与基线窗**逐位相等**
    result = judge_u2(
        make_evidence(losses=(1.0, 0.75, 0.5, 0.5, 0.5, 0.5), baseline_end_window=0.5)
    )

    assert result.passed, result.detail
    assert result.evidence_missing is False


def test_u2_fails_when_the_end_window_is_worse_than_the_baseline():
    """★能失败对照①（口径对齐后）：终窗**劣于**基线同口径终点窗 ⇒ 必红（事实判否）。"""
    # k=3：起窗 0.9 → 终窗 0.15（相对自身起点在降），但基线窗 0.05 ⇒ 未达到 ⇒ 红
    result = judge_u2(
        make_evidence(losses=(1.0, 0.9, 0.8, 0.7, 0.4, 0.2, 0.15, 0.1), baseline_end_window=0.05)
    )

    assert not result.passed
    assert result.evidence_missing is False, "序列与基线都读到了 ⇒ 事实判否"
    assert "未正常下降" in result.detail
    assert "基线同口径终点窗" in result.detail


def test_u2_missing_baseline_is_an_evidence_gap_not_a_pass():
    """★能失败对照②（口径对齐后）：缺基线窗 ⇒ `evidence_missing=True` 且 fail-closed。

    ★ 关键：本条用的序列**相对自身起点是在下降的**（自比读法会给绿）——它必须仍然
    **不放行**，这正是「基线证据未到 ⇒ 不可判」而非「默认通过」。
    """
    result = judge_u2(
        make_evidence(losses=(1.0, 0.9, 0.8, 0.7, 0.6), baseline_end_window=None)
    )

    assert not result.passed, "缺基线绝不放行"
    assert result.evidence_missing is True, "未测必须记成取证缺口，不是训练失败"
    assert U2_BASELINE_MISSING_DETAIL in result.detail
    assert "fail-closed" in result.detail
    # ★ 读数不丢：起点窗/终点窗仍写进明细（归因诊断）
    assert "终点窗" in result.detail and "起点窗" in result.detail


def test_u2_keeps_the_start_window_as_attribution_diagnostic_only():
    """口径原文「起点窗口仅作**归因诊断**」⇒ 它必须**被记录**、且**不被当判据**。

    反例保护：若有人把起点窗接回判定式（自比复辟），`test_u2_caliber_is_locked_...`
    与本用例必红。
    """
    # 起窗 0.1 → 终窗 0.6（相对自身起点在**升**），但基线窗 0.7 ⇒ 达标 ⇒ 绿
    rising_but_ok = judge_u2(
        make_evidence(losses=(0.1, 0.1, 0.1, 0.2, 0.3, 0.6, 0.6, 0.6), baseline_end_window=0.7)
    )

    assert rising_but_ok.passed, "主判据只对基线 ⇒ 起点窗上升不构成判否"
    assert "归因诊断" in rising_but_ok.detail
    assert "起点窗" in rising_but_ok.detail


def test_u2_passes_for_rising_reward():
    """GRASPO 支路：reward 越高越好（``higher_is_better=True``），同样对**基线**比。"""
    rising = judge_u2(
        make_evidence(losses=(0.1, 0.2, 0.3, 0.4, 0.5), baseline_end_window=0.39),
        higher_is_better=True,
    )
    falling = judge_u2(
        make_evidence(losses=(0.5, 0.4, 0.3, 0.2, 0.1), baseline_end_window=0.39),
        higher_is_better=True,
    )

    assert rising.passed, "终窗 reward 0.4 ≥ 基线窗 0.39"
    assert not falling.passed, "reward 终窗 0.2 < 基线窗 0.39 ⇒ 必须红（方向不能被忽略）"


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

    两例都被构造成"首末两点说一套、窗口均值说另一套"（基线窗由夹具给，两例同值）：
    ⇒ 若哪天有人把 U2 实现成首末两点（或把 A6 那条记录项拿去顶 U2），
    本用例必红。
    """
    # ① 首末两点 = increased（A6 记记录项），窗口均值下降 ⇒ 终窗 0.15 ≤ 基线窗 0.5（U2 绿）
    case_a = make_evidence(losses=(0.05, 5.0, 5.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.30))
    assert a6_loss_trend(judge_a6(case_a)) == "increased"
    assert judge_u2(case_a).passed, "窗口均值下降 ⇒ U2 必须绿（首末两点不是 U2 的口径）"

    # ② 首末两点 = decreased（A6 记记录项），窗口均值上升 ⇒ 终窗 1.713 > 基线窗 0.5（U2 红）
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

    ★ 2026-10-17 状态：`U2` 口径已按文档对齐、但**接线待裁** ⇒ `U2` 与 `U4`
    **都在诊断轴、都不在阻断轴**（`U2_WIRING_PENDING_NOTE` 逐字登记）。
    """
    assert U2_CRITERION in DIAGNOSTIC_CRITERIA
    assert U2_CRITERION not in PRIMARY_CRITERIA, "U2 未接线 ⇒ 不得出现在阻断轴"
    assert U4_CRITERION in DIAGNOSTIC_CRITERIA
    assert U4_CRITERION not in PRIMARY_CRITERIA


# ── U2 · 接线状态自证（口径已对齐 ＋ 未接线显式登记 ＋ 测试锁死）────────────


def test_u2_is_computed_and_recorded_by_judge_tier():
    """U2 不在阻断轴 ≠ 删除：仍必须逐次计算并记录在 `criteria` 里。"""
    judgement = judge_tier(make_evidence())

    assert judgement.criterion(U2_CRITERION).passed is True  # 0.4 ≤ 基线窗 0.5
    assert U2_CRITERION in {c.criterion for c in judgement.criteria}


def test_u2_not_being_wired_is_explicit_in_the_tier_note_not_silent():
    """★锁：U2 **未接线** ＋「口径已对齐文档」这条新状态，必须**写在台账备注里**。

    反例保护：若有人悄悄把 U2 接回 `primary_status`（或悄悄删掉 `U2_WIRING_PENDING_NOTE`
    这段登记），本用例必红。
    """
    judgement = judge_tier(make_evidence())

    assert judgement.passed, "U2 未接入阻断轴 ⇒ 本档仍可判通过"
    assert U2_WIRING_PENDING_NOTE in judgement.note
    assert "U2 loss 正常下降 = True" in judgement.note


def test_u2_wiring_pending_note_records_the_new_state_and_both_readings():
    """★锁：登记文本必须写明**新状态**（口径已对齐 / 接线待裁 / 采集落点），
    并**两造并列**保留被替换的旧读法与 v14 的 🔴 判词原文（⛔ 不得只剩一句"暂不接入"）。
    """
    note = U2_WIRING_PENDING_NOTE

    assert "口径已对齐" in note, "必须写明新状态：口径已按文档对齐"
    assert "基线" in note, "必须写明主判据 = 对基线同口径终点窗"
    assert "待上级裁定" in note and "本包未接" in note, "必须写明接线待裁、本包未接"
    assert "scripts/collect_results.py" in note, "必须写明基线证据的采集落点"
    assert "自比" in note, "必须两造并列保留被替换的旧读法（自比）"
    assert "🔴" in note and "判词①" in note, "必须保留 v14 的 🔴 判词原文引用"
    assert "docs/capability-matrix-milestone-20261014.html:280" in note, "必须点名文档口径原文落点"
    assert "docs/capability-matrix.data.json:189-191" in note, "必须点名基线定义的机读落点"
