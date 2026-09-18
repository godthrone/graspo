"""``graspo.eval.evaluate`` 的单测：聚合口径与 Δ 配对可比性守卫。

Δ 配对的**拒绝路径**比通过路径更重要：一个不可比的 Δ 会直接导致错误的发布决策。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspo.eval.criteria import ACCURACY_CRITERIA_VERSION
from graspo.eval.evaluate import DeltaError, compute_delta, load_report, summarize
from graspo.eval.schema import (
    EvalCriteria,
    EvalDataset,
    EvalDecoding,
    EvalEnvironmentPointer,
    EvalModel,
    EvalReport,
    utc_now_iso,
)


def _record(index: int, *, gt=("rotate_arm", "left"), pred=None, error=None) -> dict:
    pred = pred if pred is not None else gt
    hit = error is None and pred == gt
    return {
        "sample_index": index,
        "sample_key": f"test.jsonl#{index}",
        "gt_name": gt[0],
        "gt_action": gt[1],
        "pred_name": pred[0],
        "pred_action": pred[1],
        "all_right": hit,
        "completion": "",
        "parse_error": None,
        "error": error,
        "elapsed_sec": 0.1,
    }


def test_summarize_counts_only_valid_samples_in_denominator():
    records = [
        _record(0),
        _record(1),
        _record(2, pred=("lift_arm", "left")),
        _record(3, error="HTTP 500"),
    ]
    summary = summarize(records)
    assert summary.sample_count_total == 4
    assert summary.sample_count_valid == 3
    assert summary.sample_count_error == 1
    assert summary.correct == 2
    assert summary.incorrect == 1
    assert summary.accuracy_percent == pytest.approx(66.6667)
    assert summary.accuracy == pytest.approx(0.666667)


def test_summarize_groups_by_tool_and_action():
    records = [
        _record(0, gt=("rotate_arm", "left")),
        _record(1, gt=("rotate_arm", "left"), pred=("rotate_arm", "right")),
        _record(2, gt=("lift_arm", "up")),
    ]
    summary = summarize(records)
    assert summary.by_tool["rotate_arm"] == [1, 2]
    assert summary.by_tool["lift_arm"] == [1, 1]
    assert summary.by_action["left"] == [1, 2]
    assert summary.by_action["up"] == [1, 1]


def test_overlap_not_performed_is_distinguishable_from_performed_with_empty_set():
    """审查者要求的核心：``None``（没做）与空集（做了、重叠为零）**必须可区分**。

    单看数值：没做 ⇒ 字段 None；做了且为空 ⇒ 字段 0.0 且与主口径同值。
    一旦有人只比对数值就会误读，因此加显式布尔 + 训练切片标识。
    """
    records = [_record(0), _record(1, pred=("lift_arm", "left"))]

    not_performed = summarize(records, overlap_indices=None)
    assert not_performed.overlap_analysis_performed is False
    assert not_performed.accuracy_percent_excluding_overlap is None
    assert not_performed.overlap_excluded_count is None

    empty_set = summarize(records, overlap_indices=set())
    assert empty_set.overlap_analysis_performed is True
    assert empty_set.accuracy_percent_excluding_overlap == not_performed.accuracy_percent
    assert empty_set.overlap_excluded_count == 0

    # 两者数值上的差别只体现在"有没有数"上；布尔字段让"做了且为零"无从误读
    assert (not_performed.accuracy_percent_excluding_overlap is None) != (
        empty_set.accuracy_percent_excluding_overlap is None
    )


def test_summary_contract_rejects_number_without_analysis_flag():
    """契约自检：声称没做分析却报了数值 → 构造失败（§2.1 契约即防呆）。"""
    from graspo.eval.schema import EvalSummary

    with pytest.raises(Exception, match="must not report a number"):
        EvalSummary(
            sample_count_total=1,
            sample_count_valid=1,
            sample_count_error=0,
            correct=1,
            incorrect=0,
            accuracy=1.0,
            accuracy_percent=100.0,
            overlap_analysis_performed=False,
            accuracy_percent_excluding_overlap=50.0,
        )
    with pytest.raises(Exception, match="overlap_excluded_count is set"):
        EvalSummary(
            sample_count_total=1,
            sample_count_valid=1,
            sample_count_error=0,
            correct=1,
            incorrect=0,
            accuracy=1.0,
            accuracy_percent=100.0,
            overlap_excluded_count=0,
        )


def test_summarize_overlap_exclusion_changes_only_the_secondary_field():
    """剔除重叠只影响"敏感性分析"字段——主口径必须原样不动（不得偷换）。"""
    records = [
        _record(0),  # 覆盖样本，命中 → 会抬高主口径
        _record(1, pred=("lift_arm", "left")),
        _record(2, pred=("lift_arm", "left")),
    ]
    summary = summarize(records, overlap_indices={0})
    assert summary.overlap_analysis_performed is True
    assert summary.accuracy_percent == pytest.approx(33.3333)
    assert summary.accuracy_percent_excluding_overlap == pytest.approx(0.0)
    assert summary.overlap_excluded_count == 1


def test_summarize_excludes_error_samples_from_overlap_denominator_too():
    records = [_record(0), _record(1, error="boom")]
    summary = summarize(records, overlap_indices=set())
    assert summary.sample_count_valid == 1
    assert summary.accuracy_percent_excluding_overlap == pytest.approx(100.0)
    assert summary.overlap_excluded_count == 1


def test_summarize_handles_zero_records():
    summary = summarize([])
    assert summary.sample_count_valid == 0
    assert summary.accuracy_percent == 0.0


def _make_report(
    tmp_path: Path,
    *,
    role: str,
    accuracy_percent: float,
    sha: str = "a" * 64,
    valid: int = 100,
    temperature: float = 0.0,
    max_tokens: int = 128,
    criteria_version: str = ACCURACY_CRITERIA_VERSION,
) -> EvalReport:
    return EvalReport(
        run_id=f"{role}-20260918-000000",
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        elapsed_sec=1.0,
        model=EvalModel(served_name="s", role=role, path="/tmp/model"),  # type: ignore[arg-type]
        dataset=EvalDataset(
            path="data/test.jsonl",
            split="test",
            sha256=sha,
            sample_count_total=valid,
            sample_count_requested=valid,
        ),
        decoding=EvalDecoding(
            temperature=temperature, top_p=0.9, max_tokens=max_tokens, enable_thinking=False
        ),
        criteria=EvalCriteria(version=criteria_version, source="src"),
        environment=EvalEnvironmentPointer(
            visible_gpus=[0, 1], gpu_count=2, fingerprint_path="/tmp/env.json"
        ),
        summary=summarize(
            [
                _record(i)
                if i < int(valid * accuracy_percent / 100)
                else _record(i, pred=("x", "y"))
                for i in range(valid)
            ]
        ),
        samples=[],
    )


def test_compute_delta_uses_absolute_percentage_points(tmp_path):
    before = _make_report(tmp_path, role="base", accuracy_percent=30.0)
    after = _make_report(tmp_path, role="after", accuracy_percent=52.0)
    result = compute_delta(before, after, graspo_threshold_pp=20.0)
    assert result.delta_percentage_points == pytest.approx(22.0)
    assert result.graspo_pass is True
    # after 角色不是 sft → SFT 判据不适用（None，而不是 False）
    assert result.sft_pass is None


def test_compute_delta_fails_threshold_just_below_20pp(tmp_path):
    before = _make_report(tmp_path, role="base", accuracy_percent=30.0)
    after = _make_report(tmp_path, role="after", accuracy_percent=49.0)
    result = compute_delta(before, after)
    assert result.delta_percentage_points == pytest.approx(19.0)
    assert result.graspo_pass is False


def test_compute_delta_sft_role_applies_percent_threshold(tmp_path):
    before = _make_report(tmp_path, role="base", accuracy_percent=30.0)
    after = _make_report(tmp_path, role="sft", accuracy_percent=51.0)
    result = compute_delta(before, after, sft_threshold_percent=50.0)
    assert result.sft_pass is True

    below = _make_report(tmp_path, role="sft", accuracy_percent=49.5)
    assert compute_delta(before, below, sft_threshold_percent=50.0).sft_pass is False


def test_compute_delta_requires_base_as_baseline(tmp_path):
    """GRASPO 的 Δ 基线与锚点是 base 模型（用户拍板）——用别的角色必须被拒绝。"""
    sft_before = _make_report(tmp_path, role="sft", accuracy_percent=30.0)
    after = _make_report(tmp_path, role="after", accuracy_percent=60.0)
    with pytest.raises(DeltaError, match="baseline report role"):
        compute_delta(sft_before, after)

    # 显式放宽时可以通过（用于非 GRASPO 的对照实验）
    relaxed = compute_delta(sft_before, after, expect_role_before="")
    assert relaxed.delta_percentage_points == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"sha": "b" * 64}, "dataset sha256 differs"),
        ({"criteria_version": "other-1"}, "criteria version differs"),
        ({"max_tokens": 256}, "max_tokens differs"),
        ({"valid": 99}, "valid sample count differs"),
    ],
)
def test_compute_delta_refuses_incomparable_reports(tmp_path, mutation, match):
    before = _make_report(tmp_path, role="base", accuracy_percent=30.0)
    after = _make_report(tmp_path, role="after", accuracy_percent=50.0, **mutation)
    with pytest.raises(DeltaError, match=match):
        compute_delta(before, after)


def test_nonzero_temperature_report_cannot_even_be_constructed():
    """温度不为 0 的产物**连构造都不可能**——这是比"配对时拒绝"更早的防线。

    历史产物若真是 0.1 温度，反序列化进本契约时就会失败，不会被误当成可比基线。
    """
    with pytest.raises(Exception, match="temperature must be exactly 0.0"):
        EvalDecoding(temperature=0.1, top_p=0.9, max_tokens=128, enable_thinking=False)


def test_load_report_roundtrip_and_errors(tmp_path):
    report = _make_report(tmp_path, role="base", accuracy_percent=30.0)
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(report.model_dump(), ensure_ascii=False), encoding="utf-8")
    loaded = load_report(path)
    assert loaded.run_id == report.run_id
    assert loaded.summary.accuracy_percent == report.summary.accuracy_percent

    with pytest.raises(DeltaError, match="not found"):
        load_report(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(DeltaError, match="not valid JSON"):
        load_report(bad)

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text('{"run_id": "x"}', encoding="utf-8")
    with pytest.raises(DeltaError, match="does not match the eval artifact contract"):
        load_report(wrong_shape)
