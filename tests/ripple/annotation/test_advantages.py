"""标注 → per-token advantage 测试(v0.20.0)。

覆盖(方案 token-reward-advantage-design-v0.20.md §3):
- 字段级 raw(数字容差/字符串/JSON 数组存在性)
- 值 token 逐条闭合(只看自己对不对):合成组精确值 + 全对组 +1 + n=1 仍 +1
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
from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationInput, annotate

TESTSET = Path(__file__).resolve().parents[2] / "data" / "annotation_testset_v3.jsonl"

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

#: 组内同伴：结构从第一个字符就写错（S 字符数 = 0，标注为全 W）。
#: 把它放进组里，组内公共结构前缀为空 ⇒ 被测那条的 S 位置不会被 P3（组内全对的结构位置
#: 置 0）mask 掉，这些用例才能继续验证 **per-token 映射本身**（S→+1.0 / E→−1.0 / D→0）。
#: ⛔ P3 自己的行为由 ``test_group_structure_mask.py`` 验证，不在这里验证。
STRUCTURELESS = "从图像上看不出"


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
    gt = ["9999999000001", "9999999000002"]
    assert field_score('"9999999000001"', gt, 0.2) == 1.0  # 去引号
    assert field_score('"999999900000000"', gt, 0.2) == 0.0


def test_field_score_missing_gt_zero():
    assert field_score("40.0", None, 0.2) == 0.0


# ── 值 token 只看自己对不对（P2：μ_f 已删）────────────────────────────────────


def test_synthetic_group_per_token_value_advantage():
    """A 完美 / B 容差内 / C 值错+动作错 → 每个值 token 只看自己对不对（±1）。

    P2 口径变更（2026-11-18）：旧式 ``adv(V) = raw − μ_f``（组内均值）已删除。
    同一批构造在旧式下的期望值是 ``1−2/3``、``0−2/3`` 这类**随邻居变化**的数；
    现在 A/B 的值 token 恒为 ``+1.0``，C 的值 token 恒为 ``−1.0``（容差死区外的
    ``raw≈0.462`` 按「非满分即错」映射到 −1.0，规则见 ``advantages.value_advantage``）。
    """
    comps = [
        ROBOT.format("逆时针旋转", "16.9"),  # A:action 对, angle 满分
        ROBOT.format("逆时针旋转", "15"),  # B:action 对, angle 容差内 ⇒ 满分
        ROBOT.format("顺时针旋转", "40"),  # C:action 错, angle raw≈0.462(非满分)
    ]
    adv = _group_adv([*comps, STRUCTURELESS], "tool_call", GT_ROBOT)

    for i, expected in [(0, 1.0), (1, 1.0), (2, -1.0)]:
        idx = _v_token_indices(comps[i], "tool_call", GT_ROBOT)
        for token in idx["action_type"]:
            assert adv[i][token] == pytest.approx(expected), f"completion {i} 的 action_type"
        for token in idx["angle_deg"]:
            assert adv[i][token] == pytest.approx(expected), f"completion {i} 的 angle_deg"
        # 非全对的结构位置仍是 +1.0（与值评分无关；组里的 STRUCTURELESS 使前缀非全对）
        assert adv[i][0] == 1.0


def test_single_clean_completion_values_are_plus_one():
    """n=1（组内只有一条）：正确值恒为 +1.0——旧式的"安全零"正是 P2 要删的信号黑洞。

    结构 token 是另一回事：n=1 时它写对的结构前缀构成「组内全对」⇒ P3 置 0（见
    ``test_group_structure_mask.py``）。值 token 不受 P3 影响，仍是 +1.0。
    """
    comp = ROBOT.format("逆时针旋转", "16.9")  # 两个字段都与 GT 一致
    adv = _group_adv([comp], "tool_call", GT_ROBOT)[0]
    idx = _v_token_indices(comp, "tool_call", GT_ROBOT)
    for token in idx["action_type"] + idx["angle_deg"]:
        assert adv[token] == 1.0, f"单条组的正确值必须 +1.0，实测 {adv[token]}"
    assert adv[0] == 0.0, "n=1 且结构全对 ⇒ P3 把结构前缀置 0"
    assert len(adv) == len(comp)


def test_all_correct_group_value_tokens_are_plus_one():
    """组内 8 条全对：值 token 恒 +1.0（旧式 μ_f=1 ⇒ 0，真实数据里 0 组全对时也一直为 0）。"""
    comp = ROBOT.format("逆时针旋转", "16.9")
    comps = [comp] * 8
    adv = _group_adv(comps, "tool_call", GT_ROBOT)
    for i in range(8):
        idx = _v_token_indices(comps[i], "tool_call", GT_ROBOT)
        for token in idx["action_type"] + idx["angle_deg"]:
            assert adv[i][token] == 1.0, f"全对组里正确 token 必须 +1.0，实测 {adv[i][token]}"
        # 结构前缀 8 条全写对 ⇒ P3 置 0
        assert adv[i][0] == 0.0


# ── 标注驱动 broken ────────────────────────────────────────────────────────────


def test_broken_typo_prefix_eos_truncation():
    """T04 拼错闭合:前缀 S +1、E −1、其后 D 0。"""
    comp = (
        "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
        "逆时针旋转\n</parametr>\n<parameter=angle_deg>\n15\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    adv = _group_adv([comp, STRUCTURELESS], "tool_call", GT_ROBOT)[0]
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
    adv = _group_adv([comp, STRUCTURELESS], "tool_call", GT_ROBOT)[0]
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
    truncated = '```json\n{"故障号码":["9999999000001"]'
    complete = '```json\n{"故障号码":["9999999000001"]}\n```'
    gt = '{"故障号码": ["9999999000001"]}'
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
    assert not structure_incomplete('```json\n{"a": 1}\n```', "json")


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


# ── 数据集驱动的**字面**期望值（⛔ 不调用被测实现算期望）───────────────────────

#: 非值 token 的期望优势（P2 口径，S/E/T/W/D → 固定值）。
_NON_VALUE_ADV = {"S": 1.0, "E": -1.0, "T": 0.0, "W": 0.0, "D": 0.0}

#: 数据集每行「值段期望优势」，按出现顺序列出 ``(completion 里的字面值文本, 期望优势)``。
#: **来源**：人工拿 ``row["ground_truth"]`` 与这些字面值逐字段比对得出（数值按
#: ``numeric_tolerance=0.2`` 的死区判断对错），不经过 ``value_spans`` / ``field_score`` /
#: ``gt_value_for_field``——否则就是拿实现验证实现（B3）。
#: 只登记「完整且无错误定位」的行（else 分支）；带 E / 结构不完整的行走另一支。
#: ⚠ 工具名写对的行（B1，2026-11-24）也有一项：名字是 ``VALUE(__tool_name__)``，与 GT 名
#: 一致 ⇒ +1.0，排在参数值之前（模型输出的顺序）。
_VALUE_ADV: dict[str, list[tuple[str, float]]] = {
    # JSON：数组元素没有字段名 ⇒ gt_value_for_field 取不到 GT ⇒ 0.0（无信号，不是"值错"）
    "J01": [('"9999999000001"', 0.0), ('"999999900000001"', 0.0), ('"EXMPL-APN-00000"', 0.0)],
    "J02": [('"9999999000001"', 0.0), ('"999999900000001"', 0.0), ('"EXMPL-APN-00000"', 0.0)],
    "J03": [('"9999999000001"', 0.0), ('"999999900000000"', 0.0), ('"EXMPL-APN-00000"', 0.0)],
    "J04": [('"9999999000001"', 0.0)],
    "J08": [('"9999999000001"', 0.0), ('"999999900000001"', 0.0), ('"EXMPL-APN-00000"', 0.0)],
    "J12": [('"CMIOT"', 0.0), ('"IMS"', 0.0)],
    "J15": [('"9999999000001"', 0.0), ('"999999900000001"', 0.0), ('"EXMPL-APN-00000"', 0.0)],
    "J16": [('"Alice"', 0.0), ("30", 0.0)],
    # ★ JSON 语义：``true``/``false`` 与 GT 的 True/False 只有走 json.loads 才相等 ⇒ +1.0。
    #   回退掉 JSON 语义（回到"原样字符串比较"）这几项会变 −1.0，本表随即变红。
    #   ``null`` 是已知边界：GT 为 null 与「无 GT」哨兵撞车 ⇒ 0.0（复审警告 #7，未动口径）。
    "J17": [("true", 1.0), ("false", 1.0), ("null", 0.0)],
    "J18": [('"CMIOT"', 0.0), ('"IMS"', 0.0), ('"APN"', 0.0)],
    # ★ 转义字符串同理：``"hello\\nworld"``（字面反斜杠 n）经 JSON 解码后等于 GT 里的真换行。
    "J19": [('"hello\\nworld"', 1.0), ('"C:\\\\Users"', 1.0)],
    "J21": [('"CMIOT"', 0.0), ('"sub1"', 0.0), ('"sub2"', 0.0), ('"APN"', 0.0)],
    # tool_call：值字面与 GT 一致 ⇒ +1.0（数值在容差死区内也算对）
    "T01": [("rotate_arm", 1.0), ("逆时针旋转", 1.0), ("40.0", 1.0)],
    "T03": [("rotate_arm", 1.0), ("逆时针旋转", 1.0), ("40.0", 1.0)],
    "T11": [("rotate_arm", 1.0), ("顺时针旋转", 1.0), ("30", 1.0)],
    # 工具名写错（P1）⇒ 名字段整段 VALUE ⇒ 每个字符 −1.0
    "T13": [("bot_contol", -1.0)],
    "T15": [("rotate_arm", 1.0), ("顺时针旋转", 1.0), ("15", 1.0)],
    "T16": [("rotate_arm", 1.0), ("40.0", 1.0), ("逆时针旋转", 1.0)],
    # 两个字段都与 GT 不符 ⇒ 整段 −1.0
    "T18": [("rotate_arm", 1.0), ("逆时针旋转", -1.0), ("100", -1.0)],
    "T19": [("rotate_arm", 1.0), ("顺时针旋转", 1.0), ("16.9", 1.0)],
    "T27": [("rotate_arm", 1.0), ("45.0", 1.0)],
    "T30": [("get_weather", 1.0), ("Beijing", 1.0)],
    "T34": [("extend_arm", -1.0)],
    # 值本身含 '<'：与 GT 不符 ⇒ −1.0（值里的 '<' 不影响名字段的判据）
    "T39": [("rotate_arm", 1.0), ("a<b", -1.0), ("40.0", 1.0)],
    "T41": [("rotate_arm", 1.0), ("逆时针旋转", 1.0), ("40.0", 1.0)],
    # 名字写错但 '>' 在位 ⇒ P1 的名字段 V（−1.0），5 个字符
    "T46": [("think", -1.0)],
}


def _value_runs(annotation: str) -> list[tuple[int, int]]:
    """从数据集的 ``annotation`` 列取值段的字符区间（连续 V 的一段一个），按出现顺序。"""
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(annotation):
        if annotation[index] != "V":
            index += 1
            continue
        end = index
        while end < len(annotation) and annotation[end] == "V":
            end += 1
        runs.append((index, end))
        index = end
    return runs


# ── 数据集驱动(35 条)───────────────────────────────────────────────────────────


def _load_testset() -> list[dict[str, str]]:
    with open(TESTSET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("row", _load_testset(), ids=lambda r: r["id"])
def test_testset_advantages(row: dict[str, str]) -> None:
    """数据集逐行:adv 长度对齐;带 E/结构不完整 ⇒ 至少一个 −1;完整行 ⇒ 逐字符等于字面期望。

    完整行（else 分支）的期望值**全部预先写死**（tag 取自数据集的 ``annotation`` 列，
    值段的优势取自 ``_VALUE_ADV``），⛔ 不调用被测实现去算期望——理由见 B3 修复说明。
    """
    cid = row["id"]
    format_type = row["type"]
    completion = row["completion"]
    if not completion:
        return
    truncated = cid in {"T20"}  # T20 演示 max_len 截断场景(人为)
    # 组里带一条 STRUCTURELESS：公共结构前缀为空 ⇒ 本行的 S 位置不会被 P3 置 0，
    # 才能按数据集 ``annotation`` 列的字面期望逐字符比对（P3 另有专门测试）。
    adv = _group_adv(
        [completion, STRUCTURELESS],
        format_type,
        row["ground_truth"],
        truncated=[truncated, False],
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
        # 完整且无错误定位：**逐字符**断言整条优势等于预先写死的期望值。
        # 期望的两个来源都不是被测实现（B3 修复：原写法用 value_spans/field_score 现场重算，
        # 等于拿实现验证实现——把 field_score 换回旧实现后 J17/J19 被算成 −1 仍然全绿）：
        #   a) tag 序列取自数据集的 ``annotation`` 列（数据集本身就是标注规范）；
        #   b) 值段的期望优势取自 ``_VALUE_ADV``（人工按 ground_truth 比对写死的字面值）。
        # V 段先占 0.0 占位，随后整段被 ``_VALUE_ADV`` 覆盖（漏登记 ⇒ 下面的逐字符比对必红）
        expected = [_NON_VALUE_ADV.get(row["annotation"][i], 0.0) for i in range(len(completion))]
        runs = _value_runs(row["annotation"])
        literal_values = [completion[start:end] for start, end in runs]
        want = _VALUE_ADV.get(cid, [])
        assert literal_values == [text for text, _ in want], (
            f"{cid} 值段字面值变了（{literal_values} vs {[t for t, _ in want]}）——"
            "数据集或标注口径变了，必须同步更新 _VALUE_ADV"
        )
        for (start, end), (_, value_adv) in zip(runs, want, strict=True):
            for position in range(start, end):
                expected[position] = value_adv
        mismatch = [i for i, (got, exp) in enumerate(zip(adv, expected, strict=True)) if got != exp]
        assert not mismatch, (
            f"{cid} 逐字符优势与字面期望不符："
            f"{[(i, completion[i], adv[i], expected[i]) for i in mismatch[:5]]}"
        )


# ── 值文本的 JSON 语义（P2 的必要前提：正确输出不得吃 −1）──────────────────────


def test_value_tokens_of_json_literals_are_plus_one():
    """``true`` / ``false`` 与**转义字符串**按 JSON 语义判对 ⇒ 值 token 全是 +1.0。

    为什么必须单独钉这一条：标注是字符级的，值文本原样来自模型输出，所以 ``true`` 与 GT 的
    ``True``、``"第一行\\n第二行"``（字面反斜杠 n）与 GT 里真正的换行，只有再走一遍
    ``json.loads`` 才相等（见 ``advantages._json_decoded_value``）。这条语义一旦被回退成
    "原样字符串比较"，**正确输出**就会被算成"值错"吃 −1——P2 最危险的失效模式。

    已知边界（本测试不覆盖）：GT 值本身是 JSON ``null`` 时 ``gt_value_for_field`` 返回 ``None``，
    与「没有 GT」的哨兵撞车 ⇒ 该字段的值 token 目前是 0.0 而不是 +1.0（复审警告 #7 的口径问题，
    改动它属于 P2 语义，本任务未动）。
    """
    gt = json.dumps({"ok": True, "flag": False, "text": "第一行\n第二行"}, ensure_ascii=False)
    completion = '```json\n{"ok": true, "flag": false, "text": "第一行\\n第二行"}\n```'
    adv = _group_adv([completion], "json", gt)[0]

    for literal in ("true", "false", '"第一行\\n第二行"'):
        start = completion.index(literal)
        for position in range(start, start + len(literal)):
            assert adv[position] == 1.0, (
                f"字面值 {literal!r} 的 token@{position} 必须 +1.0（JSON 语义判对），实测 "
                f"{adv[position]}"
            )
