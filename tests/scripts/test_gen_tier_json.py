"""`scripts/gen_tier_json.py` 的用例 —— 覆盖指挥部点名的四类落盘自检。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


# ---------- ★★ 判据② 通过线：逐算法现读唯一真相源（修正轮 1，审 B1） ----------
# 病史：本脚本曾对所有算法一律写死 `status = PASS if delta_pp >= 5.0 else FAIL`
# ⇒ SFT 档 Δ=+7.3 pp 被判成「通过（已验证可用）」（冻结口径 SFT 线 = +10.0 pp ⇒ 结论错），
# 且 `effect` 文本自述「冻结阈值 +5.0 pp」⇒ 回填器 `backfill_matrix.run_claimed_pp`
# 按算法交叉核对时 SFT/CPT/OPD 三档会被直接拒绝。下列断言即该事故的回归网。

def _write_reports(eval_dir: Path, base_correct: int, after_correct: int, total: int) -> None:
    for side, correct in (("base", base_correct), ("after", after_correct)):
        d = eval_dir / ".local/reports" / side
        d.mkdir(parents=True, exist_ok=True)
        (d / "eval_report.json").write_text(json.dumps({"summary": {
            "correct": correct, "total": total,
            "accuracy_percent": round(correct / total * 100, 4)}}), encoding="utf-8")


def _gen_args(label: str, eval_dir: Path) -> SimpleNamespace:
    """与 `main()` 的 argparse 缺省值同形的最小 Namespace（够 `build()` 走评测分支）。"""
    return SimpleNamespace(
        tier="T999", eval_dir=str(eval_dir), no_eval=False, not_evaluable_reason="",
        nproc=1, steps=1, recon="1", model="9B", label=label, mode="full", backend="native",
        parallel="TP=1 PP=1 DP=1", hparams="", lora_target="", runs="", inference="",
        algo_checks="", throughput="", peak_mem="", run_path="", produced_by="x", command="x",
        tree_fp="", yaml="", yaml_sha="", img_run="", img_base="", host="example-host",
        gpu="0", epochs="2", exit_code="0", criteria="c", temp="0", serving_note="", p5p6="",
        evidence=["e.txt"], anomaly=[], non_graspo=True, identity_from_config=None)


def _build_with_delta(tmp_path: Path, label: str, delta_pp: float) -> dict:
    """造一份 base=500/1000 的报告，after 使其 Δ = `delta_pp` pp（1000 档 ⇒ 整数样本）。"""
    after = 500 + int(round(delta_pp * 10))          # Δ pp = (after−base)/1000×100 = 10×Δ
    d = tmp_path / f"ev-{label}-{delta_pp}"
    _write_reports(d, 500, after, 1000)
    return g.build(_gen_args(label, d))


# ★ 能失败对照（落证见工位 evidence/B1-before-fix.txt）：
#   本断言在旧代码（`>= 5.0`）上**必红** —— 旧代码把 SFT +7.3 判成 PASS。
def test_sft_seven_point_three_must_not_pass(tmp_path: Path):
    """★ 回归（B1 核心）：SFT Δ=+7.3 pp < 冻结线 +10.0 pp ⇒ **不得**判「通过」。"""
    doc = _build_with_delta(tmp_path, "SFT", 7.3)
    assert doc["delta_pp"] == pytest.approx(7.3)
    assert doc["status"] != g.PASS, f"★ SFT +7.3 pp 被判成了通过 ⇒ 第二份阈值副本又回来了：{doc['status']}"
    assert doc["status"] == g.FAIL


def test_sft_at_or_above_ten_passes(tmp_path: Path):
    doc = _build_with_delta(tmp_path, "SFT", 10.5)
    assert doc["status"] == g.PASS
    # 边界：恰好 10.0 ⇒ 通过（算子 `>=`，逐字取自真相源）
    assert _build_with_delta(tmp_path, "SFT", 10.0)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "SFT", 9.9)["status"] == g.FAIL


def test_graspo_line_is_five(tmp_path: Path):
    assert _build_with_delta(tmp_path, "GRASPO", 7.3)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "GRASPO", 5.0)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "GRASPO", 4.9)["status"] == g.FAIL


def test_cpt_uses_non_inferiority_zero(tmp_path: Path):
    """CPT 走「不劣化线」（真相源 value = 0.0 pp）⇒ `Δ ≥ 0` 通过、`Δ < 0` 不过。"""
    assert _build_with_delta(tmp_path, "CPT", 0.0)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "CPT", 7.3)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "CPT", -0.1)["status"] == g.FAIL


def test_opd_uses_delta_plus_five(tmp_path: Path):
    """OPD 走 Δ 通过线（真相源 value = 5.0 pp）⇒ `Δ ≥ +5.0` 通过、`Δ < +5.0` 不过。

    ★ 2026-10-19 口径变更（原为不劣化线）：原先那条把 CPT/OPD 参数化的用例**按新语义拆成两条**
    （⛔ 不删用例、⛔ 不放宽断言：OPD 这边比原来**更严** —— 旧口径下 `Δ=+0.0` 算通过，现在必须判否）。
    """
    assert _build_with_delta(tmp_path, "OPD", 5.0)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "OPD", 7.3)["status"] == g.PASS
    assert _build_with_delta(tmp_path, "OPD", 4.9)["status"] == g.FAIL
    assert _build_with_delta(tmp_path, "OPD", 0.0)["status"] == g.FAIL
    assert _build_with_delta(tmp_path, "OPD", -0.1)["status"] == g.FAIL


def test_effect_renders_current_pass_line_and_no_stale_wording(tmp_path: Path):
    """★ `effect` 必须按**该算法**渲染现行通过线，且**不得**再带已作废的旧口径措辞。"""
    sft = _build_with_delta(tmp_path, "SFT", 7.3)["effect"]
    assert "after − base ≥ +10.0 pp" in sft, sft
    graspo = _build_with_delta(tmp_path, "GRASPO", 7.3)["effect"]
    assert "after − base ≥ +5.0 pp" in graspo, graspo
    cpt = _build_with_delta(tmp_path, "CPT", 0.0)["effect"]
    assert "after ≥ base（不劣化）" in cpt, cpt
    for effect in (sft, graspo, cpt):
        for stale in ("目标 +10.0 pp", "仅展示", "冻结阈值 +5.0 pp"):
            assert stale not in effect, f"残留旧口径措辞 {stale!r}：{effect}"


def test_self_claimed_threshold_matches_matrix_frozen_caliber(tmp_path: Path):
    """★ 回填器交叉核对的前置：`effect` 自述阈值 == 矩阵冻结口径（SFT/GRASPO）。"""
    matrix = json.loads((_ROOT / "docs/capability-matrix.data.json").read_text(encoding="utf-8"))
    by_method = matrix["meta"]["criteria_caliber"]["threshold"]["by_method"]
    for label, method in (("SFT", "sft"), ("GRASPO", "graspo")):
        effect = _build_with_delta(tmp_path, label, 7.3)["effect"]
        assert by_method[method]["pass_line_text"] in effect


def test_unknown_label_fails_closed():
    """★ 不认识的算法名 ⇒ FAIL-CLOSED（⛔ 不猜、⛔ 不回落某条线套所有算法）。"""
    for bad in ("GRPO", "", "RLHF"):
        with pytest.raises(SystemExit):
            g.method_from_label(bad)


def test_method_from_label_maps_the_four_frozen_methods():
    assert {g.method_from_label(k) for k in ("SFT", "GRASPO", "CPT", "OPD")} == \
        {"sft", "graspo", "cpt", "opd"}


# ── D9 附带：`MODE_ALLOWED` 与 `MODE_BY_TUNER` 的取值域必须自洽（2026-10-27 实测驱动）──
def test_mode_allowed_covers_every_mode_by_tuner_value():
    """★ 同文件里「必须等于 全参」与「只许是 {LoRA, full, CPT, SFT}」此前互相打架：

    实测驱动 = 远端机在役树 `runs/M011/tier.json` 的 `tuner_type: "full"` + 矩阵数据件里
    16 档 `mode="全参"`。`MODE_BY_TUNER` 能产出的取值**必须**都在 `MODE_ALLOWED` 里，
    否则全参档按 `--identity-from-config` 一律 FAIL-CLOSED（台账一格写不进去）。
    """
    assert set(g.MODE_BY_TUNER.values()) <= g.MODE_ALLOWED, (
        f"MODE_BY_TUNER 产出 {sorted(set(g.MODE_BY_TUNER.values()) - g.MODE_ALLOWED)} "
        f"但 MODE_ALLOWED={sorted(g.MODE_ALLOWED)} 不收 ⇒ 自相矛盾")


def test_full_param_tier_identity_passes_with_real_config(tmp_path: Path):
    """★ 全参档（tuner_type=full）走身份同源断言 + 结构化断言**必须放行**（修复前必红）。"""
    cfg = tmp_path / "M011.yaml"
    cfg.write_text("tp_size: 1\npp_size: 1\ndp_size: 1\ntuner_type: full\n"
                   "backend: native\nmodel_name: Qwen3.5-9B\n", encoding="utf-8")
    args = SimpleNamespace(parallel="TP=1 PP=1 DP=1", mode="全参", backend="native",
                           model="9B", nproc=1, identity_from_config=str(cfg))
    g.assert_structured_params(args)          # 修复前：mode=全参 不在 MODE_ALLOWED ⇒ SystemExit
    g.assert_identity_consistent(args)        # 修复前：断言 mode 应为 全参、却又被上一条拒收
