"""A4 标定 rig（``scripts/a4_calibrate.py``）的契约测试（纯逻辑，不触 GPU）。

锁住三件事：
- **M1**：读数池与被判双跑共用 RUN_ROOT ⇒ 当场拒绝（防循环论证）；
- **独立性 / 卡集纪律**：run_roots 与 readings 一一对应且逐根唯一；card_set 非空；
- **元组原文可粘贴**：渲染出的字面量能逐字 `eval` 回一个等价记录（防转录错）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_RIG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "a4_calibrate.py"


def _load_rig() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_a4_calibrate_under_test", _RIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rig = _load_rig()


def _pool(**overrides):
    base = {
        "backend": "ms-swift",
        "cards": 2,
        "algorithm": "GRASPO",
        "model": "9B",
        "mode": "LoRA",
        "representative_tier": "T014",
        "card_set": [4, 5],
        "sampled_at": "2026-09-22 test",
        "split_check": "留一：子集最坏对 ≤ 全池",
        "provenance": "T014 测试用读数池",
        "run_roots": ["/p1", "/p2", "/p3", "/p4", "/p5"],
        "readings": [0.001, 0.002, 0.003, 0.0015, 0.0025],
    }
    base.update(overrides)
    return base


def test_rig_rejects_pool_sharing_a_run_root_with_the_judged_pair():
    spec = _pool(judged_pair={"run_roots": ["/p1", "/r2"], "delta": 0.001})
    with pytest.raises(rig.CalibrationInputError, match="M1 违反"):
        rig.validate_pool(spec)


def test_rig_accepts_disjoint_judged_pair_and_reports_the_verdict():
    spec = _pool(judged_pair={"run_roots": ["/r1", "/r2"], "delta": 0.001})
    calibration, derivation, _entry = rig.derive(spec)
    assert calibration.n == 5
    assert calibration.outcome == rig._judge.A4_OUTCOME_CALIBRATED
    assert "M1 已核" in derivation
    assert "通过" in derivation


def test_rig_rejects_independence_and_card_set_violations():
    with pytest.raises(rig.CalibrationInputError, match="独立性违反"):
        rig.validate_pool(_pool(run_roots=["/p1", "/p2", "/p3", "/p4"]))
    with pytest.raises(rig.CalibrationInputError, match="独立性违反"):
        rig.validate_pool(_pool(run_roots=["/p1", "/p1", "/p3", "/p4", "/p5"]))
    with pytest.raises(rig.CalibrationInputError, match="card_set"):
        rig.validate_pool(_pool(card_set=[]))
    with pytest.raises(rig.CalibrationInputError, match="缺字段"):
        bad = _pool()
        del bad["split_check"]
        rig.validate_pool(bad)


def test_rig_never_accepts_a_caller_supplied_tolerance():
    """rig 的输入里**没有**容差字段 ⇒ 容差只能由公式推导（禁止"看结果调容差"）。"""
    spec = _pool(tol=0.9)  # 多给的 tol 字段被忽略，不影响取值
    calibration, _derivation, _entry = rig.derive(spec)
    assert calibration.tol == pytest.approx(
        rig._judge.A4_WORST_PAIR_SAFETY_FACTOR * rig._judge.worst_pair(spec["readings"])
    )


def test_rig_rendered_entry_round_trips_to_an_equivalent_record():
    spec = _pool()
    calibration, _derivation, entry = rig.derive(spec)
    namespace = {"A4TierCalibration": rig._judge.A4TierCalibration}
    # 渲染的是**可粘贴进标定元组**的原文（带尾逗号）⇒ 包一层括号取第 0 项。
    restored, = eval("(" + entry + ")", namespace)  # noqa: S307 —— 测试固定输入，非用户数据
    assert restored == calibration
    # 标定表校验器必须接受它（与模块导入同一把尺子）。
    rig._judge.validate_tier_calibration_entry(restored)


def test_rig_degenerate_pool_is_labelled_not_treated_as_zero_tolerance():
    spec = _pool(readings=[0.0] * 5)
    calibration, _derivation, entry = rig.derive(spec)
    assert calibration.outcome == rig._judge.A4_OUTCOME_DEGENERATE
    assert calibration.tol == rig._judge.A4_MEASURED_BF16_FINAL_LOSS_DRIFT
    assert "退化" in calibration.note
    namespace = {"A4TierCalibration": rig._judge.A4TierCalibration}
    restored, = eval("(" + entry + ")", namespace)  # noqa: S307
    assert restored == calibration
