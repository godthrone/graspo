#!/usr/bin/env python3
"""合成长序列生成器的**结构性守卫**回归测试（纯标准库，无需 torch/ms-swift）。

**为什么有这份测试**（宪法 §2 防呆 / 指挥官 2026-09-19 裁定 3）

"两个视觉变体绝不可混填同一列"这条口径若只写在文档里，是一次复制粘贴就能破坏的纪律。
本文件证明它已被**机器强制**——并给出**能真失败的证据**（负向用例）：

1. :func:`test_fills_column_guard_rejects_v2_claiming_to_fill` —— 把 v2 标成"填列"⇒ 硬失败；
2. :func:`test_fills_column_guard_accepts_only_v1` —— 只有 v1 能声称填列；
3. :func:`test_copied_segments_identity_check_fires` —— 复制段被改写 ⇒ 自查硬失败；
4. :func:`test_v1_repeats_images_and_v2_keeps_one` —— 两个变体的产物**在结构上可区分**；
5. :func:`test_generator_rejects_unmeasured_calibration` —— 标定未标 `measured` ⇒ 拒绝生成
   （挡"算式冒充实测"）。

运行：``python3 -m pytest tests/e2e/rig_synth/test_synth_guards.py -q``
（也可直接 ``python3 tests/e2e/rig_synth/test_synth_guards.py`` —— 本文件自带 main。）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_synth_seq import (  # noqa: E402
    A4_DUAL_RUN_TIERS,
    A4_STATUS_FAIL,
    A4_STATUS_NOT_APPLICABLE_SINGLE_RUN,
    A4_STATUS_PASS,
    APPLICABLE_CRITERIA_DUAL_RUN,
    APPLICABLE_CRITERIA_SINGLE_RUN,
    FILLS_COLUMN_VARIANT,
    A4StatusInvariantError,
    FillsColumnInvariantError,
    SynthError,
    _assert_copied_segments_identical,
    _load_calibration,
    a4_dual_run_for,
    a4_status_for_build,
    applicable_criteria,
    assert_a4_status_invariant,
    assert_fills_column_invariant,
    build_synth_record,
)

SYSTEM = {"role": "system", "content": [{"type": "text", "text": "你是机器人。"}]}


def _base_record() -> dict:
    """一条最小 ELAM 形态基样：system + assistant(tool_calls) + tool + 末轮 user(2 图 + 文本)。"""
    return {
        "messages": [
            SYSTEM,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "../images/a_left.jpg"},
                    {"type": "image", "image": "../images/a_right.jpg"},
                    {"type": "text", "text": "捡起电池"},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "extend_arm",
                            "arguments": {"action_type": "收缩手臂"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "OK", "tool_call_id": "c1"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "../images/b_left.jpg"},
                    {"type": "image", "image": "../images/b_right.jpg"},
                    {"type": "text", "text": "继续"},
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "extend_arm"}}],
        "targets": [{"id": "expected", "output": {"content": {"action_type": "伸长手臂"}}}],
        "id": "base_0001",
        "source": "test",
    }


# --------------------------------------------------------------------------- #
# 裁定 3：fills_context_column 的结构性守卫
# --------------------------------------------------------------------------- #
def test_fills_column_guard_accepts_only_v1() -> None:
    assert_fills_column_invariant(FILLS_COLUMN_VARIANT, True)
    assert_fills_column_invariant("v2-text-only", False)


def test_fills_column_guard_rejects_v2_claiming_to_fill() -> None:
    """★ 能真失败的证据：v2 声称填列 ⇒ 必须抛 FillsColumnInvariantError。"""
    with pytest.raises(FillsColumnInvariantError, match="只允许来自 v1-image-copy"):
        assert_fills_column_invariant("v2-text-only", True)


def test_fills_column_guard_rejects_unknown_variant() -> None:
    with pytest.raises(FillsColumnInvariantError):
        assert_fills_column_invariant("v9-made-up", True)


def test_unknown_variant_rejected_by_builder() -> None:
    with pytest.raises(SynthError):
        build_synth_record(_base_record(), copies=1, variant="v9-made-up")


# --------------------------------------------------------------------------- #
# 复制身份自查（"图文内容一字不改"的可验证形式）
# --------------------------------------------------------------------------- #
def test_copied_segments_identity_check_fires() -> None:
    """★ 能真失败的证据：把复制出的某一份改写 ⇒ 自查必须硬失败。"""
    record, provenance = build_synth_record(_base_record(), copies=3, variant=FILLS_COLUMN_VARIANT)
    _assert_copied_segments_identical(record, provenance)  # 正常：通过

    per_copy = provenance["copied_block_count_per_copy"]
    record["messages"][-2]["content"][per_copy]["text"] = "被改写了"
    with pytest.raises(SynthError, match="内容被改写|不一致"):
        _assert_copied_segments_identical(record, provenance)


# --------------------------------------------------------------------------- #
# 两个变体在结构上可区分
# --------------------------------------------------------------------------- #
def test_v1_repeats_images_and_v2_keeps_one() -> None:
    base = _base_record()
    v1, prov1 = build_synth_record(base, copies=4, variant="v1-image-copy")
    v2, prov2 = build_synth_record(base, copies=4, variant="v2-text-only")

    # 基样末轮 user 有 2 个媒体块。
    assert prov1["source_media_block_count"] == 2
    assert prov1["output_media_block_count"] == 8, "v1：图像随样本等比例重复"
    assert prov2["output_media_block_count"] == 2, "v2：图像留 1 份"
    # 文本块两边都复制 N 份。
    assert prov1["output_user_block_count"] == prov1["copied_block_count_per_copy"] * 4
    # 填列口径只属于 v1。
    assert prov1["fills_context_column"] is True
    assert prov2["fills_context_column"] is False
    # 两条产物的 messages 结构都合法：末条是 assistant（口径 1）。
    for record in (v1, v2):
        assert record["messages"][-1]["role"] == "assistant"


def test_minimal_assistant_turn_present() -> None:
    """口径 1：必须补最小 assistant 轮（不得用"仅前向/仅显存"替代）。"""
    record, provenance = build_synth_record(_base_record(), copies=2, variant="v1-image-copy")
    assert record["messages"][-1] == {"role": "assistant", "content": "OK"}
    assert provenance["output_message_roles"] == ["system", "user", "assistant"]


# --------------------------------------------------------------------------- #
# 挡"算式冒充实测"
# --------------------------------------------------------------------------- #
def test_generator_rejects_unmeasured_calibration(tmp_path: Path) -> None:
    """★ 能真失败的证据：标定没标 measured=true ⇒ 拒绝生成。"""
    stub = tmp_path / "calib.json"
    stub.write_text(json.dumps({"points": [{"copies": 1, "tokens": 1941}]}), encoding="utf-8")
    with pytest.raises(SynthError, match="measured"):
        _load_calibration(stub)


def test_generator_rejects_missing_calibration(tmp_path: Path) -> None:
    with pytest.raises(SynthError, match="标定文件不存在"):
        _load_calibration(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# ★ 缺陷②：N 的取整口径 —— 必须 floor（"预测 ≤ max_prompt_length"），不是 ceil
#
# 真因（task-r3-bulk §⑦-2 实测复核）：配置 `data.max_prompt_length` 就是该档目标长度，
# ms-swift 以它为 **truncation 上限且策略 delete** ⇒ 实际行长 > 上限时**整条样本被丢弃**
# ⇒ 训练 **0 步**。旧实现 `math.ceil` 保证"预测 > 目标 = 上限" ⇒ 7 档全部超限、全部 0 步。
# 下面的每个断言都是**能真失败**的负向用例：把实现退回 ceil，它们会变红。
# --------------------------------------------------------------------------- #

#: 与容器内两次独立实测**逐位吻合**的标定（`tokens = 174 + 1767·N`；
#: 来源：task-r3-bulk 的 measure_synth_calibration_ext.json，copies=1,2,3,4,8,16,32,64）。
_MEASURED_CALIBRATION = {
    "measured": True,
    "points": [
        {"copies": 1, "tokens": 1941},
        {"copies": 2, "tokens": 3708},
        {"copies": 3, "tokens": 5475},
        {"copies": 4, "tokens": 7242},
        {"copies": 8, "tokens": 14310},
        {"copies": 16, "tokens": 28446},
        {"copies": 32, "tokens": 56718},
        {"copies": 64, "tokens": 113262},
    ],
}

#: 7 档的「上限 = 该档目标」与**实测**得到的历史 N（floor 口径的落点）。
#: 来源：task-r3-bulk 报告 §5.3 的实测表。注：本文件用**标定点插值**复算，
#: 与报告表里那批"实测 SFT 行 token"在 96k/128k 上有 1~2 token 的差别——报告用的是
#: **实测读数**，这里用的是**标定点预测**；`_copies_for` 的职责是后者（前者由
#: `measure_synth_tokens.py` 复核）。本测试只锁"选出来的 N"（逐档与实测一致）。
_EXPECTED_FLOOR_N = {"4k": 2, "8k": 4, "16k": 9, "32k": 18, "64k": 36, "96k": 55, "128k": 74}


def test_copies_for_never_exceeds_max_prompt_length() -> None:
    """★ 核心负向用例：ceil 口径下 16k 档会产出 N=10 ⇒ 超限；floor 口径必须 ≤ 上限。"""
    from build_synth_seq import TIERS, _copies_for

    for tag, target in TIERS:
        copies, predicted = _copies_for(_MEASURED_CALIBRATION, target, max_length=target)

        assert predicted <= target, f"{tag}: 预测 {predicted} > 上限 {target}（必 0 步）"
        assert copies >= 1
        # "旧 ceil 口径真的会超限"——否则本负向用例的构造前提不成立。
        # 旧实现 = 用两个最贴近目标的实测点插值后 math.ceil（与修复前逐字同法）。
        for old_pred in _old_ceil_predictions(target):
            assert old_pred > target, (
                f"{tag}: 旧 ceil 口径预测 {old_pred} 竟然没超上限 {target} —— "
                "本负向用例的构造前提不成立（标定或档位表变了？）"
            )


def _old_ceil_predictions(target: int) -> list[float]:
    """**修复前**实现的预测长度（ceil 语义；逐字复刻旧代码的选点 + 取整）。"""
    usable = sorted(_MEASURED_CALIBRATION["points"], key=lambda p: p["copies"])
    below = [p for p in usable if p["tokens"] < target]
    above = [p for p in usable if p["tokens"] > target]
    if below and above:
        low, high = below[-1], above[0]
    elif len(usable) >= 2:
        low, high = usable[-2], usable[-1]
    else:  # pragma: no cover - 标定点 >1，构造上到不了
        return []
    slope = (high["tokens"] - low["tokens"]) / (high["copies"] - low["copies"])
    copies = math.ceil(low["copies"] + (target - low["tokens"]) / slope)
    # 用同一份标定预测 ceil 后的 N（旧实现的 bug：**从不用预测值回查上限**）
    if copies >= usable[-1]["copies"]:
        low2, high2 = usable[-2], usable[-1]
    else:
        low2 = max((p for p in usable if p["copies"] <= copies), key=lambda p: p["copies"])
        high2 = min((p for p in usable if p["copies"] >= copies), key=lambda p: p["copies"])
    if low2["copies"] == high2["copies"]:
        return [float(low2["tokens"])]
    slope2 = (high2["tokens"] - low2["tokens"]) / (high2["copies"] - low2["copies"])
    return [low2["tokens"] + slope2 * (copies - low2["copies"])]


def test_copies_for_matches_measured_historical_n() -> None:
    """floor 口径的结果必须与 r3-bulk **实测**得到的 N 逐档一致（不得另起一套口径）。"""
    from build_synth_seq import TIERS, _copies_for

    for tag, target in TIERS:
        copies, _ = _copies_for(_MEASURED_CALIBRATION, target, max_length=target)
        assert copies == _EXPECTED_FLOOR_N[tag], f"{tag}: 得到 N={copies}"


def test_copies_for_gets_as_close_to_target_as_the_grid_allows() -> None:
    """软约束：合规的 N 里必须是**最大**的（贴近目标），不能无谓取小。"""
    from build_synth_seq import _copies_for

    copies, predicted = _copies_for(_MEASURED_CALIBRATION, 8192, max_length=8192)
    assert copies == 4 and predicted == 7242
    # 再大一份就超限（证明"最大"不是碰巧）：N=5 的预测
    from build_synth_seq import _predict_tokens

    usable = sorted(_MEASURED_CALIBRATION["points"], key=lambda p: p["copies"])
    assert _predict_tokens(usable, 5) > 8192


def test_copies_for_fails_closed_when_no_n_fits() -> None:
    """★ 若"贴近目标"与"≤ 上限"无法同时满足 ⇒ **报错并列出可选 N**，不得静默产出。

    构造：最小实测点 N=1 预测 1941 > 上限 1000 ⇒ 无可行 N。
    """
    from build_synth_seq import _copies_for

    with pytest.raises(SynthError) as excinfo:
        _copies_for(_MEASURED_CALIBRATION, 4096, max_length=1000)

    message = str(excinfo.value)
    assert "无可行 N" in message
    assert "N=1→1941" in message, "报错必须给出可选 N，便于人工决策"
    assert "上限 1000" in message


def test_copies_for_rejects_nonpositive_max_length() -> None:
    """没有上限（0/负数/非整数）⇒ 拒绝生成（fail-closed），不猜。"""
    from build_synth_seq import _copies_for

    for bad in (0, -1, "8192", 8192.0, True):
        with pytest.raises(SynthError, match="max_length"):
            _copies_for(_MEASURED_CALIBRATION, 4096, max_length=bad)


def test_copies_for_safety_margin_tightens_the_cap() -> None:
    """safety_margin 必须**收紧**上限（宁小一档，不可超限），且仍 ≤ 上限。"""
    from build_synth_seq import _copies_for

    copies, predicted = _copies_for(
        _MEASURED_CALIBRATION, 16384, max_length=16384, safety_margin_tokens=28
    )
    assert predicted + 28 <= 16384
    # 不扣余量时也是 9（16k 上限内最大 N）；扣掉 28 后仍是 9：16077+28=16105 ≤ 16384。
    # ★ 这条同时证明"扣余量不得把合规的 N 也一起砍掉"（否则就是无谓的过度收缩）。
    assert copies == 9 and predicted == 16077


def test_copies_for_is_deterministic() -> None:
    """同输入必须给出同输出（§6 可复现：口径里不得有隐藏状态/随机）。"""
    from build_synth_seq import _copies_for

    first = _copies_for(_MEASURED_CALIBRATION, 32768, max_length=32768)
    second = _copies_for(_MEASURED_CALIBRATION, 32768, max_length=32768)
    assert first == second
    assert first[0] == 18 and first[1] <= 32768


# --------------------------------------------------------------------------- #
# 裁定 1：A4 适用性 —— **单跑档绝不许记为 pass**
# --------------------------------------------------------------------------- #
def test_a4_dual_run_policy_matches_ruling() -> None:
    """4K–32K 双跑；64K/96K/128K 单跑（裁定 1）。"""
    assert A4_DUAL_RUN_TIERS == ("4k", "8k", "16k", "32k")
    for tag in ("4k", "8k", "16k", "32k"):
        assert a4_dual_run_for(tag) is True
    for tag in ("64k", "96k", "128k"):
        assert a4_dual_run_for(tag) is False
    with pytest.raises(SynthError):
        a4_dual_run_for("999k")


def test_single_run_tiers_get_not_applicable_at_build_time() -> None:
    """★ 单跑档在**生成期**就被定死为 `not_applicable_single_run`（不是等上机回填）。"""
    for tag in ("64k", "96k", "128k"):
        assert a4_status_for_build(tag) == A4_STATUS_NOT_APPLICABLE_SINGLE_RUN
    for tag in ("4k", "8k", "16k", "32k"):
        assert a4_status_for_build(tag) is None  # 双跑档：待跑


def test_a4_status_guard_accepts_legal_combinations() -> None:
    assert_a4_status_invariant(False, A4_STATUS_NOT_APPLICABLE_SINGLE_RUN)
    assert_a4_status_invariant(True, A4_STATUS_PASS)
    assert_a4_status_invariant(True, A4_STATUS_FAIL)
    assert_a4_status_invariant(True, None)  # 双跑档未跑完第 2 次 ⇒ 待跑
    assert_a4_status_invariant(False, None)  # 待跑


def test_a4_status_guard_rejects_single_run_recorded_as_pass() -> None:
    """★★ 裁定 1 的核心负向证据：**单跑档记为 `pass` ⇒ 硬失败**。"""
    with pytest.raises(A4StatusInvariantError, match="单跑档"):
        assert_a4_status_invariant(False, A4_STATUS_PASS)
    # `fail` 同样不许（没跑第二次，谈不上"不一致"）。
    with pytest.raises(A4StatusInvariantError, match="单跑档"):
        assert_a4_status_invariant(False, A4_STATUS_FAIL)


def test_a4_status_guard_rejects_dual_run_claiming_not_applicable() -> None:
    """反向也要挡：双跑档不得借"不适用"蒙混。"""
    with pytest.raises(A4StatusInvariantError, match="双跑档"):
        assert_a4_status_invariant(True, A4_STATUS_NOT_APPLICABLE_SINGLE_RUN)


def test_a4_status_guard_rejects_unknown_status() -> None:
    with pytest.raises(A4StatusInvariantError, match="未知 A4 状态"):
        assert_a4_status_invariant(True, "maybe")


def test_applicable_criteria_excludes_a4_for_single_run() -> None:
    """单跑档的**适用判据**不含 A4；双跑档含。"""
    assert applicable_criteria("4k") == APPLICABLE_CRITERIA_DUAL_RUN
    assert applicable_criteria("128k") == APPLICABLE_CRITERIA_SINGLE_RUN
    assert "A4" in applicable_criteria("32k")
    assert "A4" not in applicable_criteria("64k")


def test_provenance_carries_a4_fields(tmp_path: Path) -> None:
    """产物 provenance 里必须带 a4_dual_run / a4_status / a4_status_display /
    applicable_criteria。"""
    from build_synth_seq import write_one_tier

    for tag, dual, expect_status, expect_display in (
        ("4k", True, None, "待跑"),
        ("128k", False, A4_STATUS_NOT_APPLICABLE_SINGLE_RUN, "未适用（单跑）"),
    ):
        prov = write_one_tier(
            base=_base_record(),
            base_index=0,
            copies=2,
            variant=FILLS_COLUMN_VARIANT,
            tier_tag=tag,
            target_tokens=4096 if tag == "4k" else 131072,
            out_dir=tmp_path / tag,
            tier_id="T013",
        )
        assert prov["a4_dual_run"] is dual
        assert prov["a4_status"] == expect_status
        assert prov["a4_status_display"] == expect_display
        assert ("A4" in prov["applicable_criteria"]) is dual


def _persist(out_dir: Path, tier_id: str, tier_tag: str, provenance: dict) -> None:
    """把 write_one_tier 返回的 provenance 落盘（复刻 main() 的落盘路径）。

    `write_one_tier` 只**返回** dict，由 `main()` 落盘（因为 main 还要补
    `base_tokens_measured` 等回填上下文）。回填类测试需要磁盘上有文件 ⇒ 此处补上。
    """
    path = out_dir / "subsets" / f"{tier_id}-synth-{tier_tag}.provenance.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8")


def test_backfill_a4_rejects_single_run_as_pass(tmp_path: Path) -> None:
    """★★ 回填入口同样被守卫拦截：对单跑档回填 `pass` ⇒ 硬失败。"""
    from build_synth_seq import backfill_a4_status, write_one_tier

    prov = write_one_tier(
        base=_base_record(),
        base_index=0,
        copies=2,
        variant=FILLS_COLUMN_VARIANT,
        tier_tag="128k",
        target_tokens=131072,
        out_dir=tmp_path,
        tier_id="T013",
    )
    _persist(tmp_path, "T013", "128k", prov)
    with pytest.raises(A4StatusInvariantError, match="单跑档"):
        backfill_a4_status(
            out_dir=tmp_path,
            tier_id="T013",
            tier_tag="128k",
            a4_status=A4_STATUS_PASS,
            source_note="test",
        )
    # 正确填法则通过。
    doc = backfill_a4_status(
        out_dir=tmp_path,
        tier_id="T013",
        tier_tag="128k",
        a4_status=A4_STATUS_NOT_APPLICABLE_SINGLE_RUN,
        source_note="test",
    )
    assert doc["a4_status_display"] == "未适用（单跑）"


def test_backfill_a4_allows_dual_run_pass(tmp_path: Path) -> None:
    """双跑档回填 `pass` 合法（A4 适用）。"""
    from build_synth_seq import backfill_a4_status, write_one_tier

    prov = write_one_tier(
        base=_base_record(),
        base_index=0,
        copies=2,
        variant=FILLS_COLUMN_VARIANT,
        tier_tag="4k",
        target_tokens=4096,
        out_dir=tmp_path,
        tier_id="T013",
    )
    _persist(tmp_path, "T013", "4k", prov)
    doc = backfill_a4_status(
        out_dir=tmp_path,
        tier_id="T013",
        tier_tag="4k",
        a4_status=A4_STATUS_PASS,
        source_note="test",
    )
    assert doc["a4_status"] == A4_STATUS_PASS
    assert doc["a4_status_display"] == "通过"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
