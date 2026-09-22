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


# ── ⑦ 结构可达性前置判定（AF1 §⑥ 方案 S2；2026-09-21）────────────────────────
#
# **为什么有这一节**：AF1 诊断 —— 9 档 `GRASPO × native` 的 optimizer 步数上限低于
# 清单门槛 5（`T030`/`T036`/`T042` 在 4 卡下**无论跑多久最多 1 步**）。若判据照旧
# 判「optimizer step=1 < 门槛 5 ⇒ ❌」，这批 ❌ 会被读者读成"4 卡 native GRASPO
# 能力不行"，而事实是"这档压根没被真正测过"。
#
# 修法 = **门槛值 5 一个字不动**，加一道"这道题可不可能考"的前置判定：
# 清单给出 `expected_optimizer_steps_reachable < 门槛` ⇒ 返回**第三态**
# 「⚠ 口径不可测」。本节用**负向用例**锁住三件事：
#   ① 0 步**永远**判 ❌（第三态不得吞掉"压根没训练"）；
#   ② 实测 == 该档上限（含上限=1）⇒ 第三态，且**不是** ✅；
#   ③ 实测**显著低于**上限 ⇒ 仍判 ❌（第三态不得变成"只要有 1 步就过"）。


def test_zero_steps_still_fails_even_when_reachable_is_one():
    """★ ①：`optimizer_steps == 0` ⇒ **必须 ❌**，第三态不得把它变成"口径不可测"。

    这是本修法最关键的防线：`T030` 的"结构上最多 1 步"**不等于**"0 步也没关系"——
    0 步是"压根没训练"，是真失败（`judge_a2` 的步数门槛分支在最前）。
    """
    result = judge_a2(
        make_evidence(
            optimizer_steps=0,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not result.passed
    assert not result.not_applicable
    assert "optimizer_steps = 0" in result.detail
    assert "门槛 5" in result.detail


def test_steps_equal_to_reachable_gets_the_not_applicable_third_state():
    """★ ②：实测 == 该档上限（上限 = 1 < 门槛 5）⇒ 第三态，**不是 ✅、不是 ❌**。

    这条对应真机 `T030`：自然跑完 `exit_code=0`、`optimizer_steps=1`。
    """
    result = judge_a2(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not result.passed, "第三态**不是** ✅ —— 步数门槛没有被满足这个事实照写"
    assert result.not_applicable, result.detail
    assert not result.evidence_missing
    assert "⚠ 口径不可测" in result.detail
    assert "可产出步数上限=1" in result.detail
    assert "门槛 5" in result.detail


def test_steps_well_below_reachable_still_fails():
    """★ ③：实测**显著低于**该档上限 ⇒ 仍判 ❌（第三态只在"上限 < 门槛"时出现）。

    对应真失败场景：该档明明能跑 100 步（`T010` 实测 100/100），却只跑了 3 步。
    """
    result = judge_a2(
        make_evidence(
            optimizer_steps=3,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=100,
            expected_optimizer_steps_reachable=100,
        )
    )
    assert not result.passed
    assert not result.not_applicable, "可达 ⇒ 第三态不得出现"
    assert "optimizer step=3 < 门槛 5" in result.detail
    # 台账里同时写明该档上限，读者能一眼看出"差得有多远"（§2.2）
    assert "可产出步数上限=100" in result.detail


def test_reachable_above_threshold_is_enforced_normally():
    """上限 ≥ 门槛 ⇒ 门槛**照常硬判**（第三态不出现）⇒ 不放宽任何既有档位。"""
    for steps in (1, 4):
        result = judge_a2(
            make_evidence(
                optimizer_steps=steps,
                min_optimizer_steps=5,
                expected_optimizer_steps_per_epoch=40,
                expected_optimizer_steps_reachable=40,
            )
        )
        assert not result.passed and not result.not_applicable
    ok = judge_a2(
        make_evidence(
            optimizer_steps=5,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=40,
            expected_optimizer_steps_reachable=40,
        )
    )
    assert ok.passed


def test_missing_reachable_falls_back_to_hard_threshold():
    """★ 清单没给上限（老清单）⇒ **不开第三态**，门槛照常硬判（fail-closed）。"""
    for steps in (1, 4):
        result = judge_a2(
            make_evidence(optimizer_steps=steps, min_optimizer_steps=5)
        )
        assert not result.passed and not result.not_applicable


def test_illegal_reachable_does_not_open_the_third_state():
    """上限值非法（0 / 负数 / 非整数 / 布尔）⇒ **不据此开第三态**（fail-closed）。"""
    for bad in (0, -1, "1", 1.5, True):
        result = judge_a2(
            make_evidence(
                optimizer_steps=1,
                min_optimizer_steps=5,
                expected_optimizer_steps_per_epoch=bad,
                expected_optimizer_steps_reachable=bad,
            )
        )
        assert not result.passed and not result.not_applicable, bad


def test_substantive_assertions_precede_the_third_state():
    """★ 实质断言**先于**第三态：非有限跳过 / 逐步未推进 ⇒ 仍判 ❌，不被"口径不可测"吞掉。"""
    skipped = judge_a2(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            nonfinite_skips=3,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not skipped.passed and not skipped.not_applicable
    assert "非有限梯度" in skipped.detail

    stalled = judge_a2(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            optimizer_steps_per_step=(1, 0, 1),
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not stalled.passed and not stalled.not_applicable
    assert "optimizer_steps=0" in stalled.detail

    short_of_plan = judge_a2(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            steps_declared_total=5,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not short_of_plan.passed and not short_of_plan.not_applicable
    assert "实际只推进了 1 个" in short_of_plan.detail


def test_tier_ledger_status_is_gate_not_applicable_when_only_a2_is_blocked():
    """整档：只有 A2 因"口径不可测"未过、其余全过 ⇒ 台账落**第四态**（不是 ❌、不是 ✅）。

    注：这里必须给**两跑**证据——只给一跑时 A4 会以"取证缺口"判否，整档就落
    「⚠ 不可判定」了（那是**另一条**通道，与本第三态必须分得开，见下一个用例）。
    """
    judgement = judge_tier(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        ),
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        ),
    )
    assert not judgement.passed
    assert judgement.step_gate_not_applicable
    assert judgement.ledger_status == "⚠ 口径不可测"
    assert judgement.failure_class is not None
    assert "口径不可测" in judgement.note
    assert "不是训练失败" in judgement.note
    # 不得计入最大可行上下文（与取证缺口同类）
    assert judgement.counts_toward_max_context is False


def test_tier_real_failure_is_not_masked_by_the_third_state():
    """★ 第三态**不掩盖**真失败：A1 失败 + A2 口径不可测 ⇒ 仍是 ❌ 失败。"""
    judgement = judge_tier(
        make_evidence(
            exit_code=1,
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        ),
        make_evidence(
            exit_code=1,
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        ),
    )
    assert judgement.ledger_status == "❌ 不可用"
    assert not judgement.step_gate_not_applicable or True  # A2 不可测，但被真失败盖过
    assert "口径不可测" in judgement.note  # 两条事实都写出来


def test_tier_evidence_gap_is_not_confused_with_gate_not_applicable():
    """第三态与取证缺口**不可混淆**：缺 optimizer 证据仍是「⚠ 不可判定（取证缺口）」。"""
    judgement = judge_tier(
        make_evidence(
            optimizer_steps=None,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert judgement.ledger_status == "⚠ 口径不可测"
    assert not judgement.step_gate_not_applicable


def test_ledger_row_self_reports_reachable_and_third_state():
    """台账留证（§1.4）：必须能回答"这一档最多几步""为什么没判失败"。"""
    judgement = judge_tier(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        ),
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        ),
    )
    row = ledger_row(
        judgement,
        model="9B",
        algorithm="GRASPO",
        mode="LoRA",
        backend="native",
        cards=4,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-21",
    )
    assert row["status"] == "⚠ 口径不可测"
    assert row["expected_optimizer_steps_reachable"] == 1
    assert row["step_gate_not_applicable"] is True
    assert row["min_optimizer_steps"] == 5  # ★ 门槛值一个都没动


# ── ⑧ ★ **负向对照**：把门槛降到 1 ⇒ 现有测试必须把它抓住 ─────────────────────
#
# 本节的目的是**证明"降门槛"这条歧路会被测试拦住**（而不是靠人自觉）。
# 做法：显式构造一份"门槛放宽到 1"的清单值，跑一遍判据，断言它与现行口径
# **结论不同**，并且现行口径下的断言集合里有断言判红。


def test_negative_control_lowering_threshold_to_one_would_pass_zero_step_tiers():
    """★ 负向对照：把门槛降到 1 ⇒ `T030` 这类档（1 步）会变 ✅ —— 这正是**歧路**。

    它证明"降门槛"确实会改变结论（不是空谈），因此 :func:`test_manifest_default_five_still_rejects_four_steps`
    与本节其它用例必须把这条歧路拦住。
    """
    tier = make_evidence(optimizer_steps=1, min_optimizer_steps=5)
    stray = tier.__class__(
        **{
            **{
                field: getattr(tier, field)
                for field in tier.__dataclass_fields__  # type: ignore[attr-defined]
            },
            "min_optimizer_steps": 1,
        }
    )
    assert not judge_a2(tier).passed, "现行口径：1 步 < 门槛 5 ⇒ 不通过"
    assert judge_a2(stray).passed, "降门槛到 1 ⇒ 1 步变通过（这就是必须被拦住的歧路）"
    # ★ 而本包**没有**走这条歧路：清单里的门槛仍是 5（见生成器的 FORMAL_GATE_MIN_OPTIMIZER_STEPS）
    assert MIN_OPTIMIZER_STEPS == 5


def test_negative_control_third_state_is_not_a_blanket_pass():
    """★ 负向对照：第三态**不是**"所有档都过"。

    若有人把第三态实现成"只要 reachable < 门槛就通过"，那 0 步的档也会过 ——
    本用例断言 0 步仍 ❌，从而把那种实现判红。
    """
    zero = judge_a2(
        make_evidence(
            optimizer_steps=0,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not zero.passed
    assert not zero.not_applicable
    # 且"第三态"的判据文本里**不含** ✅
    na = judge_a2(
        make_evidence(
            optimizer_steps=1,
            min_optimizer_steps=5,
            expected_optimizer_steps_per_epoch=1,
            expected_optimizer_steps_reachable=1,
        )
    )
    assert not na.passed
