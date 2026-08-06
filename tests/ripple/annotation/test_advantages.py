"""标注 → per-token advantage 测试(v0.20.0)。

覆盖(方案 token-reward-advantage-design-v0.20.md §3):
- 字段级 raw(数字容差/字符串/JSON 数组存在性)
- μ_f 组内相对化(合成组精确值 + 全对组零信号 + n=1 安全零)
- 标注驱动 broken(T04 拼错:前缀 S +1 / E −1 / D 0)
- 末尾 EOS 惩罚(乱码/缺闭合/半截 JSON;truncated_by_max 除外;完整无)
- 参数校验与输出长度
"""

import json
from pathlib import Path

import pytest

from graspo.ripple.annotation.advantages import (
    apply_tail_eos,
    compute_group_advantages,
    field_score,
    structure_incomplete,
    value_spans,
)
from graspo.ripple.annotation.labeler import AnnotationInput, annotate
from graspo.ripple.annotation.roles import CharTag

TESTSET = Path(__file__).resolve().parents[2] / "data" / "annotation_testset_v2.jsonl"

NO_FENCE_CASES = {"J02", "J18", "J20", "J21", "J22"}
THINK_CASES = {"T11", "T24", "T25"}

ROBOT = (
    "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
    "{}\n</parameter>\n<parameter=angle_deg>\n{}\n</parameter>\n</function>\n</tool_call>"
)


class FakeTokenizer:
    """每字符一个 token,offset (i, i+1)——字符级标注与 token 一一对应。"""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(c) % 1000 for c in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text: str, return_offsets_mapping: bool = False, **kwargs):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]

        class FakeEncoding:
            def __init__(self, input_ids, offset_mapping):
                self.input_ids = input_ids
                self.offset_mapping = offset_mapping

            def get(self, key, default=None):
                return getattr(self, key, default)

        return FakeEncoding(ids, offsets)


def _make_targets(format_type: str, gt: str) -> list[dict]:
    """与 test_annotation_testset.py 相同的构造逻辑。"""
    if format_type == "tool_call":
        name, rest = gt.split("(", 1)
        args_str = rest.rstrip(")").strip()
        arguments: dict = {}
        for pair in args_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split("=", 1)
            v = v.strip()
            try:
                arguments[k.strip()] = float(v)
            except ValueError:
                arguments[k.strip()] = v
        return [{"output": {"tool_calls": [{"name": name.strip(), "arguments": arguments}]}}]
    content = json.loads(gt)
    return [{"output": {"content": content}}]


def _annotate(completion: str, format_type: str, gt: str, check_json_markdown: bool = True):
    return annotate(
        AnnotationInput(
            completion_text=completion,
            targets=_make_targets(format_type, gt),
            tokenizer=None,
            format_type=format_type,
            check_json_markdown=check_json_markdown,
        )
    )


def _group_adv(
    completions: list[str],
    format_type: str,
    gt: str,
    *,
    truncated: list[bool] | None = None,
    tolerance: float = 0.2,
    check_json_markdown: bool = True,
    check_think: bool = False,
) -> list[list[float]]:
    annotations = [
        _annotate(c, format_type, gt, check_json_markdown=check_json_markdown)
        if format_type == "json"
        else annotate(
            AnnotationInput(
                completion_text=c,
                targets=_make_targets(format_type, gt),
                tokenizer=None,
                format_type=format_type,
                check_think=check_think,
            )
        )
        for c in completions
    ]
    return compute_group_advantages(
        completions=completions,
        annotations=annotations,
        targets=_make_targets(format_type, gt),
        tokenizer=FakeTokenizer(),
        format_type=format_type,
        numeric_tolerance=tolerance,
        truncated_by_max=truncated,
    )


def _v_token_indices(completion: str, format_type: str, gt: str) -> dict[str, list[int]]:
    """字段 → V token 索引(FakeTokenizer:token i 覆盖字符 [i, i+1))。"""
    ann = _annotate(completion, format_type, gt)
    out: dict[str, list[int]] = {}
    for s, e, f in value_spans(completion, ann.tags, ann.fields):
        out.setdefault(f, []).extend(range(s, e))
    return out


GT_ROBOT = "robot_atomic_control(action_type=逆时针旋转, angle_deg=16.9)"


# ── 字段级 raw ─────────────────────────────────────────────────────────────────


def test_field_score_numeric_tolerance_deadzone():
    # 15 vs 16.9:相对误差 11% < 20% 容差 → 满分(死区)
    assert field_score("15", 16.9, 0.2) == pytest.approx(1.0)
    # 40 vs 16.9:误差 137% → 1/(1+1.167) ≈ 0.462
    assert field_score("40", 16.9, 0.2) == pytest.approx(1.0 / (1.0 + (40 - 16.9) / 16.9 - 0.2))
    # 容差 0:任何误差都衰减
    assert field_score("15", 16.9, 0.0) < 1.0


def test_field_score_string_exact():
    assert field_score("逆时针旋转", "逆时针旋转", 0.2) == 1.0
    assert field_score("顺时针旋转", "逆时针旋转", 0.2) == 0.0


def test_field_score_json_array_existence():
    gt = ["1442201593053", "1442201593054"]
    assert field_score('"1442201593053"', gt, 0.2) == 1.0  # 去引号
    assert field_score('"460240401000000"', gt, 0.2) == 0.0


def test_field_score_missing_gt_zero():
    assert field_score("40.0", None, 0.2) == 0.0


# ── 组内相对化(μ_f) ────────────────────────────────────────────────────────────


def test_synthetic_group_mu_and_adv():
    """A 完美 / B 容差内 / C 值错+动作错 → μ_f 与 per-token adv 精确值。"""
    comps = [
        ROBOT.format("逆时针旋转", "16.9"),  # A:action 1.0, angle 1.0
        ROBOT.format("逆时针旋转", "15"),    # B:action 1.0, angle 1.0(容差内)
        ROBOT.format("顺时针旋转", "40"),    # C:action 0.0, angle ≈0.462
    ]
    adv = _group_adv(comps, "tool_call", GT_ROBOT)

    mu_action = 2.0 / 3.0
    angle_c = 1.0 / (1.0 + (40 - 16.9) / 16.9 - 0.2)
    mu_angle = (1.0 + 1.0 + angle_c) / 3.0

    for i, expect_action, expect_angle in [
        (0, 1.0 - mu_action, 1.0 - mu_angle),
        (1, 1.0 - mu_action, 1.0 - mu_angle),
        (2, 0.0 - mu_action, angle_c - mu_angle),
    ]:
        idx = _v_token_indices(comps[i], "tool_call", GT_ROBOT)
        for t in idx["action_type"]:
            assert adv[i][t] == pytest.approx(expect_action)
        for t in idx["angle_deg"]:
            assert adv[i][t] == pytest.approx(expect_angle)
        # S token 全 +1.0(不受相对化影响)
        assert adv[i][0] == 1.0  # <tool_call> 首字符是 S
        # 组内含相对化信号,不可能全是 +1
        assert not all(a == 1.0 for a in adv[i])


def test_single_clean_group_v_zero():
    """n=1:组内只有一条 → μ_f=raw → V adv=0(安全零),S +1。"""
    comp = ROBOT.format("逆时针旋转", "40.0")
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    idx = _v_token_indices(comp, "tool_call", GT_ROBOT)
    for t in idx["action_type"] + idx["angle_deg"]:
        assert adv[t] == 0.0
    # S token +1.0
    assert adv[0] == 1.0
    # 输出长度 == 字符数
    assert len(adv) == len(comp)


def test_all_correct_group_zero_content():
    """组内 8 条全对:μ_f=1 → V adv=0(目标态,不训练)。"""
    comp = ROBOT.format("逆时针旋转", "16.9")
    comps = [comp] * 8
    adv = _group_adv(comps, "tool_call", GT_ROBOT)
    for i in range(8):
        idx = _v_token_indices(comps[i], "tool_call", GT_ROBOT)
        for t in idx["action_type"] + idx["angle_deg"]:
            assert adv[i][t] == pytest.approx(0.0)
        assert adv[i][0] == 1.0  # 格式仍 +1(收敛后靠 ratio 退场)


# ── 标注驱动 broken ────────────────────────────────────────────────────────────


def test_broken_typo_prefix_eos_truncation():
    """T04 拼错闭合:前缀 S +1、E −1、其后 D 0。"""
    comp = (
        "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
        "逆时针旋转\n</parametr>\n<parameter=angle_deg>\n15\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    # E 存在于 </parametr> 的错字符处(首个不匹配)
    e_pos = comp.find("r", comp.find("</parametr>") + 8)  # </paramet> 后
    assert adv[e_pos] == -1.0
    # E 之前有 S +1(前缀正确)
    assert adv[0] == 1.0
    # E 之后全部 0(D 不训练)
    assert all(a == 0.0 for a in adv[e_pos + 1 :])


def test_garbage_all_waste_zero():
    """纯乱码:全 W 0 梯度 + 末尾 EOS −1。"""
    comp = "从图像5-11可以清晰地看到"
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    assert adv[-1] == -1.0  # 末尾 EOS
    assert all(a == 0.0 for a in adv[:-1])


# ── 末尾 EOS 惩罚 ──────────────────────────────────────────────────────────────


def test_tail_eos_incomplete_tool_call():
    """缺闭合标签:前缀 S/V 照常,末尾 −1。"""
    comp = (
        "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
        "逆时针旋转\n</parameter>\n<parameter=angle_deg>\n16.9\n</parameter>\n"
        "</function>"
    )
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    assert adv[-1] == -1.0  # 末尾非空白字符(>)
    assert adv[0] == 1.0


def test_tail_eos_truncated_by_max_skipped():
    """max_new_tokens 硬截断 → 不标末尾 EOS。"""
    comp = (
        "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
        "逆时针旋转\n</parameter>\n<parameter=angle_deg>\n16.9\n</parameter>\n"
        "</function>"
    )
    adv = _group_adv([comp], "tool_call", GT_ROBOT, truncated=[True])[0]
    assert -1.0 not in adv


def test_tail_eos_complete_no_penalty():
    """完整输出:无末尾 EOS。"""
    comp = ROBOT.format("逆时针旋转", "16.9")
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    assert -1.0 not in adv


def test_tail_eos_json_truncated():
    """半截 JSON:末尾 −1;完整 JSON:无。"""
    truncated = '```json\n{"故障号码":["1442201593053"]'
    complete = '```json\n{"故障号码":["1442201593053"]}\n```'
    gt = '{"故障号码": ["1442201593053"]}'
    adv_tr = _group_adv([truncated], "json", gt, truncated=[False])[0]
    assert adv_tr[-1] == -1.0
    adv_ok = _group_adv([complete], "json", gt)[0]
    assert -1.0 not in adv_ok


def test_tail_eos_already_has_error_skipped():
    """已有 E 定位(如拼错)→ 不叠加末尾 EOS。"""
    comp = (
        "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
        "逆时针旋转\n</parametr>\n<parameter=angle_deg>\n15\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    # 恰好一个 −1(E 定位处),末尾不重复
    assert adv.count(-1.0) == 1
    assert adv[-1] == 0.0


# ── structure_incomplete / apply_tail_eos 单元 ────────────────────────────────


def test_structure_incomplete():
    assert structure_incomplete("<tool_call>abc", "tool_call")
    assert structure_incomplete("纯乱码没有结构", "tool_call")
    assert not structure_incomplete("<tool_call>x</tool_call>", "tool_call")
    assert structure_incomplete('{"a": [1', "json")
    assert not structure_incomplete('{"a": [1]}', "json")
    assert not structure_incomplete("```json\n{\"a\": 1}\n```", "json")


def test_apply_tail_eos_conditions():
    tags = [CharTag.WASTE] * 5
    # 乱码 + 未截断 → 末尾 E
    out = apply_tail_eos("abcde", tags, "tool_call")
    assert out[-1] == CharTag.ERROR
    assert out[:-1] == [CharTag.WASTE] * 4
    # truncated_by_max → 不变
    out = apply_tail_eos("abcde", tags, "tool_call", truncated_by_max=True)
    assert CharTag.ERROR not in out
    # 空 → 不变
    assert apply_tail_eos("", tags, "tool_call") == tags
    # 已有 E → 不变
    tags_e = [CharTag.ERROR] + [CharTag.WASTE] * 4
    out = apply_tail_eos("abcde", tags_e, "tool_call")
    assert out == tags_e


# ── 参数校验与对齐 ─────────────────────────────────────────────────────────────


def test_group_adv_validation():
    ann = [_annotate("abc", "tool_call", GT_ROBOT)]
    with pytest.raises(ValueError, match="format_type"):
        compute_group_advantages(
            completions=["abc"],
            annotations=ann,
            targets=_make_targets("tool_call", GT_ROBOT),
            tokenizer=FakeTokenizer(),
            format_type="yaml",
        )
    with pytest.raises(ValueError, match="empty"):
        compute_group_advantages(
            completions=[],
            annotations=[],
            targets=_make_targets("tool_call", GT_ROBOT),
            tokenizer=FakeTokenizer(),
            format_type="tool_call",
        )
    with pytest.raises(ValueError, match="aligned"):
        compute_group_advantages(
            completions=["abc", "def"],
            annotations=ann,
            targets=_make_targets("tool_call", GT_ROBOT),
            tokenizer=FakeTokenizer(),
            format_type="tool_call",
        )


def test_output_length_matches_tokens():
    comp = ROBOT.format("逆时针旋转", "40.0")
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    tok = FakeTokenizer()
    n_gen = len(tok(comp, return_offsets_mapping=True).input_ids)
    assert len(adv) == n_gen


# ── 数据集驱动(35 条)───────────────────────────────────────────────────────────


def _load_testset() -> list[dict[str, str]]:
    with open(TESTSET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("row", _load_testset(), ids=lambda r: r["id"])
def test_testset_advantages(row: dict[str, str]) -> None:
    """35 条数据集:adv 长度对齐;完整 case 无 −1;不完整 case 末尾 −1。"""
    cid = row["id"]
    format_type = row["type"]
    completion = row["completion"]
    if not completion:
        return
    truncated = cid in {"T20"}  # T20 演示 max_len 截断场景(人为)
    adv = _group_adv(
        [completion],
        format_type,
        row["ground_truth"],
        truncated=[truncated],
        check_json_markdown=cid not in NO_FENCE_CASES,
        check_think=cid in THINK_CASES,
    )[0]
    assert len(adv) == len(completion)

    has_error = "E" in row["annotation"]
    if cid == "T20":
        # max_len 截断:无末尾 EOS(结构不完整但非模型选择)
        assert -1.0 not in adv
        return
    if has_error or structure_incomplete(completion, format_type):
        # 有错误定位或结构不完整(缺闭合/乱码/半截)→ 至少一个 −1(含末尾 EOS)
        assert adv.count(-1.0) >= 1
    else:
        # 完整且无错误定位:无 −1
        assert -1.0 not in adv
