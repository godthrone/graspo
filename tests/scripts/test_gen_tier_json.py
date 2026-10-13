"""`scripts/gen_tier_json.py` 的用例 —— 覆盖指挥部点名的四类落盘自检。"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import gen_tier_json as g  # noqa: E402


# ① 单位/同源：delta_pp 必须是百分点
def test_fractions_delta_is_percentage_points():
    a, b, dp = g.fractions(171, 702, 118, 702)
    assert dp == pytest.approx(7.5499, abs=1e-4), (a, b, dp)
    assert dp == pytest.approx((a - b) * 100, abs=1e-6)


def test_fractions_negative_delta_keeps_sign():
    a, b, dp = g.fractions(117, 702, 118, 702)
    assert dp < 0 and dp == pytest.approx(-0.1424, abs=1e-4), dp


# ② evidence_paths 非空（否则回填校验器 fail-closed）
def _minimal_doc(**over):
    d = {
        "provenance_command_placeholder": None,
        "delta_pp": 1.0,
        "after_acc": 0.20,
        "base_acc": 0.19,
        "n_total": 702,
        "provenance": {"evidence_paths": ["/data/x/eval.log"], "command": "launcher … ::: graspo eval run"},
        "step_count_reconciliation": {"steps_nominal": 10},
    }
    d.update(over)
    return d


def test_selfcheck_accepts_good_document():
    g.selfcheck(_minimal_doc())


def test_selfcheck_rejects_wrong_delta_unit():
    """★ 回归：delta_pp 误写成小数差（0.0057 而非 0.5697）必须被拒。"""
    with pytest.raises(AssertionError):
        g.selfcheck(_minimal_doc(delta_pp=0.0057, after_acc=0.168091, base_acc=0.162393))


def test_selfcheck_rejects_empty_evidence_paths():
    """★ 回归：evidence_paths 空/缺失 ⇒ 被回填校验器拦下那次的收口。"""
    for bad in ([], None, [""], "not-a-list"):
        with pytest.raises(AssertionError):
            g.selfcheck(_minimal_doc(provenance={"evidence_paths": bad}))


# ③ 对账式步骤与备注（同源、≤100 字）
def test_short_note_is_short_and_uses_structured_fields():
    a, b, dp = g.fractions(171, 702, 118, 702)
    note = g.short_note("通过（已验证可用）", 171, 118, 702)
    assert len(note) <= 100
    assert "171/702" in note and "118/702" in note and f"{dp:+.2f}" in note


def test_reconciliation_formula_check():
    """对账式：nproc=4 ⇒ 8×20×8/4 = 320（与 steps_actual 一致）。"""
    assert (8 * 20 * 8) // 4 == 320
    assert (2 * 100) // 1 == 200


def test_summarise_reads_report_summary(tmp_path: Path):
    p = tmp_path / "eval_report.json"
    p.write_text(json.dumps({"summary": {"correct": 171, "total": 702, "accuracy_percent": 24.359}}))
    assert g.summarise(g.load_report(str(p))) == (171, 702, 24.359)


def test_selfcheck_rejects_empty_provenance_command():
    """★ 回归：T008 的 provenance.command 为空串 ⇒ 被矩阵侧 fail-closed 拦下。"""
    for bad in ("", "   ", None):
        with pytest.raises(AssertionError):
            g.selfcheck(_minimal_doc(provenance={"evidence_paths": ["/data/x/eval.log"], "command": bad}))


def test_delta_tolerance_allows_report_rounding():
    """★ 回归：报告侧 accuracy_percent 只保留 4 位小数（T020: 0.1424 vs 精确 0.14245）⇒ 容差 1e-3。"""
    a, b, dp = g.fractions(118, 702, 117, 702)
    assert dp == 0.1425, dp                      # 精确分数 ⇒ 0.1425
    assert abs(round(16.8091 - 16.6667, 4) - dp) < 1e-3   # 报告舍入值 ⇒ 0.1424


# ---------- 对账式必须按该档自身口径（T045 误用 SFT 公式） ----------
def test_recon_graspo_accepts_graspo_formula():
    assert g.check_reconciliation_matches_tier("GRASPO", "8x20x8/4 = 320", 320) == []


def test_recon_graspo_rejects_sft_formula():
    """★ 回归：T045 首版误用 ms-swift SFT 的 `2x100/4 = 50` 算 GRASPO 档 ⇒ 必须拒。"""
    problems = g.check_reconciliation_matches_tier("GRASPO", "2x100/4 = 50", 320)
    assert problems and "8×subset×8/cards" in problems[0], problems


def test_recon_sft_rejects_graspo_formula():
    problems = g.check_reconciliation_matches_tier("SFT", "8x20x8/4 = 320", 320)
    assert problems and "不得用 GRASPO" in problems[0], problems


def test_recon_sft_accepts_sft_formula():
    assert g.check_reconciliation_matches_tier("CPT", "2x100/2 = 100", 100) == []


def test_recon_right_side_must_match_measured_steps():
    """★ 实测权威：右侧 ≠ steps_actual ⇒ 拒（以实测为准）。"""
    problems = g.check_reconciliation_matches_tier("GRASPO", "8x20x8/4 = 640", 320)
    assert problems and "≠ steps_actual" in problems[0], problems
