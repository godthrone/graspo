"""判据② 通过线与**冻结口径**对齐的负向守卫（评测侧 P1）。

**为什么有这份测试**（宪法 §1.4 单一真相源 ＋ §2 防呆）

评测侧 `eval/evaluate.compute_delta` 曾另存一份旧口径的**数值副本**
（`graspo_threshold_pp=20.0` / `sft_threshold_percent=50.0`），且 SFT 那支比的是
**绝对准确率**（`after ≥ 50%`），而冻结口径要求比 **Δ**（`after − base ≥ +10.0 pp`）
⇒ **量纲错配，结论必错**。

冻结口径的唯一真相源 = `docs/capability-matrix.data.json` 的
`meta.criteria_caliber.threshold.by_method`：

- SFT ⇒ Δ ≥ +10.0 pp（epoch=4）
- GRASPO ⇒ Δ ≥ +5.0 pp（epoch=8）
- CPT ⇒ non-inferiority：`after ≥ base`（Δ ≥ 0.0 pp；epoch=2）
- OPD ⇒ Δ ≥ +5.0 pp（criterion = `gain_pp`；epoch=2）—— ★ 2026-10-19 口径变更（原为不劣化）

一律**点估计**，⛔ 不做假设检验、不引置信区间 / 显著性。

本文件里的每条断言都是**能真失败**的负向用例：把修法退回去（旧数值副本或旧量纲），
至少有一条必须变红。逐条负向对照的取证见工位
`.local/hb-workspace/20261014-sft-kickoff/task-threshold-align/evidence/`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graspo.eval import thresholds as th
from graspo.eval.criteria import ACCURACY_CRITERIA_VERSION
from graspo.eval.evaluate import DeltaError, compute_delta, summarize
from graspo.eval.schema import (
    EvalCriteria,
    EvalDataset,
    EvalDecoding,
    EvalEnvironmentPointer,
    EvalModel,
    EvalReport,
    utc_now_iso,
)

_REPO = Path(__file__).resolve().parents[2]


def _summary(accuracy_percent: float, valid: int = 1000):
    hits = round(valid * accuracy_percent / 100)
    records = [
        {
            "sample_index": i,
            "sample_key": f"t#{i}",
            "gt_name": "rotate_arm",
            "gt_action": "left",
            "pred_name": "rotate_arm",
            "pred_action": "left",
            "all_right": i < hits,
            "completion": "",
            "parse_error": None,
            "error": None,
            "elapsed_sec": 0.1,
        }
        for i in range(valid)
    ]
    return summarize(records)


def _report(role: str, accuracy_percent: float) -> EvalReport:
    return EvalReport(
        run_id=f"{role}-20261017-000000",
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        elapsed_sec=1.0,
        model=EvalModel(served_name="s", role=role, path="/tmp/model"),  # type: ignore[arg-type]
        dataset=EvalDataset(
            path="data/test.jsonl",
            split="test",
            sha256="a" * 64,
            sample_count_total=1000,
            sample_count_requested=1000,
        ),
        decoding=EvalDecoding(temperature=0.0, top_p=0.9, max_tokens=128, enable_thinking=False),
        criteria=EvalCriteria(version=ACCURACY_CRITERIA_VERSION, source="src"),
        environment=EvalEnvironmentPointer(
            visible_gpus=[0], gpu_count=1, fingerprint_path="/tmp/env.json"
        ),
        summary=_summary(accuracy_percent),
        samples=[],
    )


# ── ① 唯一真相源：值、量纲、判据类型都逐字来自 data.json ────────────────────────


def test_pass_lines_come_from_the_frozen_single_source():
    """四算法的通过线 = 真相源 `by_method.*`（⛔ 不是代码里的第二份副本）。"""
    by_method = th.load_frozen_method_thresholds()
    assert set(by_method) == {"sft", "graspo", "cpt", "opd"}

    assert (by_method["sft"].criterion, by_method["sft"].value_pp) == ("gain_pp", 10.0)
    assert (by_method["graspo"].criterion, by_method["graspo"].value_pp) == ("gain_pp", 5.0)
    assert (by_method["cpt"].criterion, by_method["cpt"].value_pp) == ("non_inferiority", 0.0)
    # ★ 2026-10-19 口径变更（OPD：不劣化 ⇒ Δ ≥ +5.0 pp）⇒ 本断言按**新口径**改期望值；
    #   ⛔ 不是放宽、⛔ 不是删断言：负向用例（下列分别验 CPT 不劣化 / OPD Δ 线）强度不变。
    assert (by_method["opd"].criterion, by_method["opd"].value_pp) == ("gain_pp", 5.0)


def test_no_second_threshold_copy_survives_in_src():
    """★ 防呆：`src/` 里不得再出现旧阈值字段的**定义/赋值**（第二份数值副本）。

    只认"行首就是字段名"的形态（`name:` / `name=`）——本模块自己的说明性文档
    引用旧字段名不算副本。"""
    import re

    pattern = re.compile(r"^\s*(graspo_threshold_pp|sft_threshold_percent)\s*[:=]", re.M)
    offenders: list[str] = []
    for path in (_REPO / "src").rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(_REPO)))
    assert offenders == [], f"旧阈值字段定义仍在 src/：{offenders}"


def test_frozen_source_path_is_the_repo_data_file():
    assert th.caliber_source_path() == _REPO / "docs" / "capability-matrix.data.json"
    assert th.caliber_source_path().is_file()


def test_missing_source_is_fail_closed(monkeypatch, tmp_path):
    """真相源读不到 ⇒ 抛 `FrozenCaliberError`，⛔ 不回落任何旧值。"""
    monkeypatch.setattr(th, "caliber_source_path", lambda: tmp_path / "missing.json")
    with pytest.raises(th.FrozenCaliberError, match="真相源不存在"):
        th.load_frozen_method_thresholds()


# ── ② SFT 量纲专项：比 Δ，⛔ 不比绝对准确率 ────────────────────────────────────


def test_sft_compares_delta_not_absolute_accuracy():
    """★ 量纲专项对照（口径要求的两个反例/正例）。

    `after=52, base=50` ⇒ Δ=+2.0 < +10.0 ⇒ **不通过**（旧代码按 52 ≥ 50 会误判通过）；
    `after=65, base=50` ⇒ Δ=+15.0 ≥ +10.0 ⇒ 通过。
    """
    base50 = _report("base", 50.0)

    flat = compute_delta(base50, _report("sft", 52.0), method="sft")
    assert flat.delta_percentage_points == pytest.approx(2.0)
    assert flat.criterion == "gain_pp"
    assert flat.threshold_pp == pytest.approx(10.0)
    assert flat.passed is False, "Δ=+2.0 竟按旧的『绝对准确率 ≥50%』判通过 = 量纲错配"

    jump = compute_delta(base50, _report("sft", 65.0), method="sft")
    assert jump.delta_percentage_points == pytest.approx(15.0)
    assert jump.passed is True


def test_sft_boundary_is_inclusive_at_plus_ten():
    """通过线是「≥」：Δ = +10.0 ⇒ 通过；Δ = +9.9 ⇒ 不通过（点估计，不四舍五入放水）。"""
    assert compute_delta(_report("base", 40.0), _report("sft", 50.0), method="sft").passed is True
    assert compute_delta(_report("base", 40.0), _report("sft", 49.9), method="sft").passed is False


def test_sft_role_suffix_is_inferred_without_the_flag():
    """不传 method 时由 after 角色推断：role=sft ⇒ SFT 线（而不是 GRASPO 线）。"""
    result = compute_delta(_report("base", 40.0), _report("sft", 52.0))
    assert (result.method, result.threshold_pp) == ("sft", 10.0)
    assert result.passed is True


# ── ③ GRASPO：+5.0 pp（⛔ 不是旧副本的 20.0 pp）──────────────────────────────


def test_graspo_line_is_five_pp_not_twenty(tmp_path):
    """base=50, after=55 ⇒ Δ=+5.0 ⇒ 通过；旧副本 20.0 会把它判否。"""
    result = compute_delta(_report("base", 50.0), _report("after", 55.0))
    assert (result.method, result.criterion, result.threshold_pp) == ("graspo", "gain_pp", 5.0)
    assert result.passed is True

    assert compute_delta(_report("base", 50.0), _report("after", 54.9)).passed is False


# ── ④ CPT：non-inferiority（after ≥ base）／OPD：Δ ≥ +5.0 pp，都必须显式给 method ──
#   ★ 2026-10-19 口径变更：OPD 由「不劣化」改为 **Δ ≥ +5.0 pp**（criterion = `gain_pp`、value = 5.0）
#     ⇒ 原先那条把 CPT/OPD 合并成"不劣化线"的用例**按新语义拆成两条**（⛔ 不删用例、⛔ 不放宽断言：
#     CPT 仍验 Δ ≥ 0（`Δ=0` 过 / `Δ<0` 不过），OPD 改验 Δ ≥ +5.0（`+5.0` 过 / `+4.9`、`<0` 不过）。


def test_cpt_stays_non_inferiority():
    """CPT：`after == base` ⇒ 通过（不劣化）；`after < base` ⇒ 不通过。"""
    ok = compute_delta(_report("base", 50.0), _report("after", 50.0), method="cpt")
    assert ok.criterion == "non_inferiority"
    assert ok.threshold_pp == pytest.approx(0.0)
    assert ok.passed is True

    assert compute_delta(_report("base", 50.0), _report("after", 49.9), method="cpt").passed is (
        False
    )


def test_opd_applies_delta_plus_five():
    """OPD：criterion = `gain_pp`、线 = +5.0 pp ⇒ `Δ=+5.0` 过、`Δ=+4.9` 不过、`Δ<0` 不过。"""
    at_line = compute_delta(_report("base", 50.0), _report("after", 55.0), method="opd")
    assert at_line.criterion == "gain_pp"
    assert at_line.threshold_pp == pytest.approx(5.0)
    assert at_line.passed is True
    # 边界：恰好 +5.0 ⇒ 通过（算子 `>=`，逐字取自真相源）
    assert compute_delta(_report("base", 50.0), _report("after", 55.0),
                         method="opd").delta_percentage_points == pytest.approx(5.0)

    assert compute_delta(_report("base", 50.0), _report("after", 54.9),
                         method="opd").passed is False          # Δ = +4.9 < +5.0
    assert compute_delta(_report("base", 50.0), _report("after", 45.0),
                         method="opd").passed is False          # Δ = −5.0 < +5.0


def test_cpt_opd_cannot_be_inferred_from_role():
    """★ 不得假装能推断：role=after + method=cpt 才是 CPT 线（role 无法表达 CPT/OPD）。"""
    cpt = compute_delta(_report("base", 50.0), _report("after", 50.0), method="cpt")
    opd = compute_delta(_report("base", 50.0), _report("after", 50.0), method="opd")
    inferred = compute_delta(_report("base", 50.0), _report("after", 50.0))
    # ★ 新口径：CPT 走不劣化、OPD 走 Δ 线、不传 method 走 GRASPO 的 Δ 线 ⇒ 三者互不可推断。
    assert cpt.criterion == "non_inferiority"
    assert opd.criterion == "gain_pp" and opd.threshold_pp == pytest.approx(5.0)
    assert inferred.criterion == "gain_pp"


# ── ⑤ 参数防呆 ────────────────────────────────────────────────────────────────


def test_unknown_method_is_rejected():
    with pytest.raises(DeltaError, match="未知算法"):
        compute_delta(_report("base", 50.0), _report("after", 60.0), method="dpo")
