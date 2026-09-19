"""判定器门槛（A2 `min_optimizer_steps`）**来自清单**的回归测试（纯逻辑，不触 GPU）。

**为什么有这份测试**（宪法 §2 防呆 / §1.4 单一真相源）

`src/graspo/core/result_judge.py` 曾把 A2 门槛**硬编码**为常量 5，**不读**清单里的
`tiers[*].acceptance.formal_gate.min_optimizer_steps`。合成长序列档已在清单里登记
（并被批准）`min_optimizer_steps: 1`，却被判定器忽略 ⇒ 合成 4k/8k/16k 的 A2 被判否
（**假否定**，task-r3-bulk §⑦-6）。修法 = 门槛改由清单驱动，**缺省值仍是 5**。

本文件的每个断言都是**能真失败**的负向用例——把修复退回去，它们必须有一个变红：

1. :func:`test_manifest_threshold_is_honored` —— 清单写 1、实际 1 步 ⇒ 必须通过
   （旧实现：`optimizer step=1 < 门槛 5` ⇒ 失败）；
2. :func:`test_missing_manifest_threshold_falls_back_to_five` —— 清单未给 ⇒ 仍按 5
   （**不放宽缺省**：1 步 / 4 步都必须判否）；
3. :func:`test_manifest_default_five_still_rejects_four_steps` —— 清单显式写 5 ⇒ 4 步判否
   （**不放宽现有档位**）；
4. :func:`test_nonpositive_threshold_is_fail_closed` —— 清单写 0 / 负数 ⇒ **判否**，
   不静默回落（0 步也必须判否——若回落到 5 才会"0 步判否"，故另用"清单 0、实际 5 步"
   这条**只有不回落的实现才会判否**的用例把回落与 fail-closed 分开）；
5. :func:`test_non_integer_threshold_is_fail_closed` —— 字符串 / 浮点 / 布尔 ⇒ 判否；
6. :func:`test_collector_reads_threshold_from_manifest` —— 采集层取值口径（含"缺键"、
   "缺 acceptance 段"、"0 原样传出、不静默改写成缺省"）；
7. :func:`test_ledger_row_reports_effective_threshold` —— 台账留证：这一档按几判的。

运行：``python3 -m pytest tests/core/test_result_judge_threshold.py -q``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from graspo.core.result_judge import (
    MIN_OPTIMIZER_STEPS,
    RunEvidence,
    judge_a2,
    judge_tier,
    ledger_row,
    resolve_min_optimizer_steps,
)

_REPO = Path(__file__).resolve().parents[2]
_COLLECT = _REPO / "scripts" / "collect_results.py"


def _load_collector():
    """按路径加载 `scripts/collect_results.py`（与判定器同一份源码，不走安装）。"""
    spec = importlib.util.spec_from_file_location("_crd_threshold", _COLLECT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_evidence(**overrides) -> RunEvidence:
    base = dict(
        tier_id="T013-synth-4k",
        exit_code=0,
        timed_out=False,
        log_text="",
        tuner_type="lora",
        optimizer_steps=1,
        epochs_completed=1.0,
        weight_changed=True,
        checkpoint_reloadable=True,
        artifacts_present={
            "config_backup": True,
            "training_log": True,
            "checkpoint": True,
            "metrics": True,
        },
        losses=(1.0,),
        grad_norms=(1.0,),
    )
    base.update(overrides)
    return RunEvidence(**base)


# ── ① 清单门槛被遵守（负向：旧实现的硬编码 5 会判否）────────────────────────


def test_manifest_threshold_is_honored():
    """清单写 1、实际 1 步 ⇒ **必须通过**（合成档已批准的偏离）。"""
    result = judge_a2(make_evidence(optimizer_steps=1, min_optimizer_steps=1))

    assert result.passed, result.detail
    assert "门槛 1" in result.detail


def test_manifest_threshold_one_still_rejects_zero_steps():
    """门槛放宽到 1 **不等于**门槛消失：0 步仍必须判否。"""
    # 0 步走的是"缺少 step 证据"的更前面的分支之前……不：0 是**读到了**的读数，
    # 所以这里验证的是"实际 0 < 门槛 1"。
    assert not judge_a2(make_evidence(optimizer_steps=0, min_optimizer_steps=1)).passed


def test_manifest_threshold_is_order_independent_of_default():
    """清单写 10 ⇒ 9 步判否、10 步通过（门槛既可放宽也可收紧，值说了算）。"""
    assert not judge_a2(make_evidence(optimizer_steps=9, min_optimizer_steps=10)).passed
    assert judge_a2(make_evidence(optimizer_steps=10, min_optimizer_steps=10)).passed


# ── ② 清单未给 ⇒ 缺省 5（**不放松**）──────────────────────────────────────


def test_missing_manifest_threshold_falls_back_to_five():
    """清单未给门槛（`None`）⇒ 缺省 5；1 / 4 步都必须判否。"""
    assert MIN_OPTIMIZER_STEPS == 5
    for steps in (1, 4):
        result = judge_a2(make_evidence(optimizer_steps=steps, min_optimizer_steps=None))
        assert not result.passed
        assert "缺省（清单未给门槛）" in result.detail
    assert judge_a2(make_evidence(optimizer_steps=5, min_optimizer_steps=None)).passed


# ── ③ 清单显式写 5 ⇒ 现有档位门槛一字不动 ─────────────────────────────────


def test_manifest_default_five_still_rejects_four_steps():
    assert not judge_a2(make_evidence(optimizer_steps=4, min_optimizer_steps=5)).passed
    assert judge_a2(make_evidence(optimizer_steps=5, min_optimizer_steps=5)).passed


# ── ④ 非法门槛 ⇒ fail-closed，**不得静默回落** ─────────────────────────────


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_nonpositive_threshold_is_fail_closed(bad: int):
    """清单写 0/负数 ⇒ 判否。

    ★ 关键设计：用"实际 5 步"而不是"实际 0 步"——实际 0 步在任何实现下都会判否
    （`0 < 5` 且 `0 < 1`），**证明不了**是 fail-closed 还是"静默回落到 5"。
    只有"实际 5 步 ≥ 缺省 5、却因为门槛非法而判否"这一条，能把两种实现分开。
    """
    result = judge_a2(make_evidence(optimizer_steps=5, min_optimizer_steps=bad))

    assert not result.passed
    assert "非法" in result.detail

    # 纯函数层同口径（两种调用路径不得分叉，§1.4）。
    assert resolve_min_optimizer_steps(bad) == (None, resolve_min_optimizer_steps(bad)[1])
    assert resolve_min_optimizer_steps(bad)[0] is None


def test_non_integer_threshold_is_fail_closed():
    """字符串 / 浮点 / 布尔都不得被当成门槛（防呆：不猜配置意图）。"""
    for bad in ("1", 1.5, True, []):
        result = judge_a2(make_evidence(optimizer_steps=5, min_optimizer_steps=bad))
        assert not result.passed, f"{bad!r} 竟被当成合法门槛"
        assert "不是整数" in result.detail


def test_resolve_returns_none_reason_only_for_invalid():
    assert resolve_min_optimizer_steps(None) == (5, None)
    assert resolve_min_optimizer_steps(7) == (7, None)
    assert resolve_min_optimizer_steps(0)[0] is None
    assert resolve_min_optimizer_steps(0)[1] is not None


# ── ⑤ 采集层：从清单取门槛（原样传出，不静默改写）────────────────────────


def test_collector_reads_threshold_from_manifest():
    collector = _load_collector()
    extract = collector.extract_min_optimizer_steps

    assert extract({"acceptance": {"formal_gate": {"min_optimizer_steps": 1}}}) == 1
    # 缺键 ⇒ None（判定器回落缺省 5）
    assert extract({"acceptance": {"formal_gate": {"min_epochs": 1}}}) is None
    # 缺段 / 形状不对 ⇒ None（判定器回落缺省 5，**不猜**）
    assert extract({}) is None
    assert extract({"acceptance": None}) is None
    assert extract({"acceptance": {"formal_gate": "5"}}) is None
    # ★ 非法值**原样**传出（不在采集层静默换缺省）——裁决权在判定层。
    assert extract({"acceptance": {"formal_gate": {"min_optimizer_steps": 0}}}) == 0
    assert extract({"acceptance": {"formal_gate": {"min_optimizer_steps": -3}}}) == -3


# ── ⑥ 台账留证：这一档按几判的 ────────────────────────────────────────────


def test_ledger_row_reports_effective_threshold():
    """合法门槛 ⇒ 台账记下实际用的数字；非法门槛 ⇒ 记 `None`（不假装有门槛）。"""
    judgement = judge_tier(make_evidence(optimizer_steps=1, min_optimizer_steps=1))
    row = ledger_row(
        judgement,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="ms-swift",
        cards=1,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-19",
    )
    assert row["min_optimizer_steps"] == 1

    bad = judge_tier(make_evidence(optimizer_steps=5, min_optimizer_steps=0))
    row_bad = ledger_row(
        bad,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="ms-swift",
        cards=1,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-19",
    )
    assert row_bad["min_optimizer_steps"] is None
    assert not bad.passed
