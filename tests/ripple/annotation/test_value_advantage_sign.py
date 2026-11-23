"""P2 判据：值 token 只看它自己对不对（+1 / −1），组内均值 μ_f 不再参与。

负责人对 GRASPO 的设计原则（唯一判据）：

    「按 token 算梯度，每个 token 有自己的梯度更新方向。第一个错误的 token 会有负梯度，
    后面的 token 就扔掉。如果这个 token 和前面的所有 token 都正确，那么这些都有正梯度。」

旧式 ``adv(V) = raw − μ_f`` 是 GRPO 的组内比较，违背这条原则：全对组 ``μ=1`` ⇒ 正确
token 优势 ``1−1=0``；全错组 ``μ=0`` ⇒ 错误 token 优势 ``0−0=0``。本文件把「正确必须
+1、错误必须 −1、且与组内其它条无关」钉成机核。

⛔ 本文件不验证任何组内归一化（减均值/z-score/baseline 已被否决）。
"""

from __future__ import annotations

import pytest

from graspo.ripple.annotation.advantages import compute_group_advantages, field_score
from graspo.ripple.annotation.labeler import AnnotationInput, annotate

ROBOT = (
    "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
    "{}\n</parameter>\n<parameter=angle_deg>\n{}\n</parameter>\n</function>\n</tool_call>"
)

TARGETS = [
    {
        "output": {
            "tool_calls": [
                {
                    "name": "robot_atomic_control",
                    "arguments": {"action_type": "逆时针旋转", "angle_deg": 16.9},
                }
            ]
        }
    }
]

GOOD_ACTION = "逆时针旋转"
BAD_ACTION = "顺时针旋转"


class FakeTokenizer:
    """每字符一个 token，offset ``(i, i+1)``（与既有 advantage 测试同一模式）。"""

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

            def __getitem__(self, key):
                return getattr(self, key)

        return FakeEncoding(ids, offsets)


def _advantages(completions: list[str], tolerance: float = 0.2) -> list[list[float]]:
    annotations = [
        annotate(
            AnnotationInput(
                completion_text=text,
                targets=TARGETS,
                tokenizer=None,
                format_type="tool_call",
            )
        )
        for text in completions
    ]
    return compute_group_advantages(
        completions=completions,
        annotations=annotations,
        targets=TARGETS,
        tokenizer=FakeTokenizer(),
        format_type="tool_call",
        numeric_tolerance=tolerance,
    )


def _value_positions(completion: str) -> dict[str, list[int]]:
    """字段 → 该字段值所在的字符下标（FakeTokenizer 下字符下标 == token 下标）。"""
    from graspo.ripple.annotation.advantages import value_spans

    result = annotate(
        AnnotationInput(
            completion_text=completion,
            targets=TARGETS,
            tokenizer=None,
            format_type="tool_call",
        )
    )
    out: dict[str, list[int]] = {}
    for start, end, field in value_spans(completion, result.tags, result.fields):
        out.setdefault(field, []).extend(range(start, end))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 全对组 / 全错组：旧式 μ_f 把信号清零，新式必须 ±1
# ══════════════════════════════════════════════════════════════════════════════


def test_all_correct_group_value_tokens_are_plus_one():
    """★ 8 条全对：正确 token 的优势必须是 **+1.0**（旧式 μ=1 ⇒ 0，正是 bug）。"""
    completion = ROBOT.format(GOOD_ACTION, "16.9")
    advantages = _advantages([completion] * 8)

    positions = _value_positions(completion)
    assert positions["action_type"], "构造前提：action_type 必须有值 token"
    for row in advantages:
        for token in positions["action_type"] + positions["angle_deg"]:
            assert row[token] == 1.0, f"全对组里正确 token 必须 +1.0，实测 {row[token]}"
        # 结构 token 不动：仍是 +1.0
        assert row[0] == 1.0


def test_all_wrong_group_value_tokens_are_minus_one():
    """★ 8 条全错：错误 token 的优势必须是 **−1.0**（旧式 μ=0 ⇒ 0，正是 bug）。"""
    completion = ROBOT.format(BAD_ACTION, "40")
    advantages = _advantages([completion] * 8)

    positions = _value_positions(completion)
    for row in advantages:
        for token in positions["action_type"]:
            assert row[token] == -1.0, f"全错组里错误 token 必须 −1.0，实测 {row[token]}"
        for token in positions["angle_deg"]:
            assert row[token] == -1.0


def test_value_advantage_does_not_depend_on_group_composition():
    """★ 删掉 μ_f 的判据：同一条 completion 的优势，与组内其它条无关。

    旧式 ``raw − μ_f`` 下，「全对组的正确 token」= 0，而「有对有错组里的正确 token」
    > 0 —— 同一条 completion、同一个 token，信号随邻居变。新式必须逐条闭合。
    """
    good = ROBOT.format(GOOD_ACTION, "16.9")
    mixed = [good, good, ROBOT.format(BAD_ACTION, "40"), ROBOT.format(BAD_ACTION, "40")]

    alone = _advantages([good] * 4)
    in_mixed = _advantages(mixed)

    assert alone[0] == in_mixed[0], "同一条 completion 的 per-token 优势必须与组内邻居无关"


def test_mixed_group_direction_follows_each_token_correctness():
    """★ 有对有错：正确 token +1、错误 token −1，逐条各自成立（方向判据）。"""
    good = ROBOT.format(GOOD_ACTION, "16.9")
    bad = ROBOT.format(BAD_ACTION, "40")
    advantages = _advantages([good, good, bad, bad])

    good_positions = _value_positions(good)
    bad_positions = _value_positions(bad)
    for row in advantages[:2]:
        assert all(row[t] == 1.0 for t in good_positions["action_type"])
        assert all(row[t] == 1.0 for t in good_positions["angle_deg"])
    for row in advantages[2:]:
        assert all(row[t] == -1.0 for t in bad_positions["action_type"])
        assert all(row[t] == -1.0 for t in bad_positions["angle_deg"])


# ══════════════════════════════════════════════════════════════════════════════
# 非二值 raw（数字容差）的映射规则：满分 ⇒ +1，非满分 ⇒ −1
# ══════════════════════════════════════════════════════════════════════════════


def test_numeric_raw_is_continuous_so_mapping_must_be_stated():
    """构造前提：``field_score`` 对数字**不是**二值（容差内满分、容差外连续衰减）。

    所以 ±1 的映射必须有明确规则；本测试把 ``leaf_compare_score`` 的两个取值钉住。
    """
    # 容差内（相对误差 11% < 20% 死区）⇒ 满分 1.0
    assert field_score("15", 16.9, 0.2) == pytest.approx(1.0)
    # 容差外 ⇒ 严格介于 0 与 1 之间（连续衰减，不是 0/1）
    partial = field_score("40", 16.9, 0.2)
    assert 0.0 < partial < 1.0


def test_numeric_within_tolerance_is_plus_one_beyond_is_minus_one():
    """★ 映射规则：``raw == 1.0``（含容差死区内）⇒ +1.0；``raw < 1.0`` ⇒ −1.0。"""
    within = ROBOT.format(GOOD_ACTION, "15")  # 15 vs 16.9：容差内 → raw 1.0
    beyond = ROBOT.format(GOOD_ACTION, "40")  # 40 vs 16.9：raw ≈ 0.462（非满分）

    advantages = _advantages([within] * 2 + [beyond] * 2)
    within_positions = _value_positions(within)
    beyond_positions = _value_positions(beyond)

    for row in advantages[:2]:
        for token in within_positions["angle_deg"]:
            assert row[token] == 1.0, "容差死区内的值算「对」⇒ +1.0"
    for row in advantages[2:]:
        for token in beyond_positions["angle_deg"]:
            assert row[token] == -1.0, "非满分（哪怕是 0.46）算「错」⇒ −1.0"


# ══════════════════════════════════════════════════════════════════════════════
# S / E / T / W / D 不动（负责人明确要求）
# ══════════════════════════════════════════════════════════════════════════════


def test_structure_error_and_waste_tags_keep_their_old_values():
    """★ S=+1.0 / E=−1.0 / W·T·D=0.0 三类**不参与**本次改动。"""
    # 拼错闭合标签 ⇒ 前缀 S(+1)、错字符 E(−1)、其后 D(0)
    typo = (
        "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
        "逆时针旋转\n</parametr>\n<parameter=angle_deg>\n16.9\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    row = _advantages([typo] * 2)[0]

    assert row[0] == 1.0, "结构 token 仍是 +1.0"
    error_pos = typo.find("</parametr>") + "</paramet".__len__()  # 首个不匹配字符
    assert typo[error_pos] == "r"
    assert row[error_pos] == -1.0, "E 仍是 −1.0"
    assert all(value == 0.0 for value in row[error_pos + 1 :]), "E 之后仍是 D=0.0"


# ══════════════════════════════════════════════════════════════════════════════
# JSON 格式同样适用（值 token 的判据与格式无关）
# ══════════════════════════════════════════════════════════════════════════════


def test_json_format_value_tokens_are_signed_too():
    """JSON 分支：值对 ⇒ +1.0，值错 ⇒ −1.0。"""
    targets = [{"output": {"content": {"answer": "4"}}}]
    good = '```json\n{"answer": "4"}\n```'
    bad = '```json\n{"answer": "5"}\n```'
    annotations = [
        annotate(
            AnnotationInput(
                completion_text=text, targets=targets, tokenizer=None, format_type="json"
            )
        )
        for text in (good, bad)
    ]
    advantages = compute_group_advantages(
        completions=[good, bad],
        annotations=annotations,
        targets=targets,
        tokenizer=FakeTokenizer(),
        format_type="json",
    )
    good_pos = good.index("4")
    bad_pos = bad.index("5")
    assert advantages[0][good_pos] == 1.0
    assert advantages[1][bad_pos] == -1.0


def test_json_group_composition_does_not_matter():
    """JSON 分支同样逐条闭合：同一条 completion 的优势与邻居无关。"""
    targets = [{"output": {"content": {"answer": "4"}}}]
    good = '```json\n{"answer": "4"}\n```'
    bad = '```json\n{"answer": "5"}\n```'
    annotations = [
        annotate(
            AnnotationInput(
                completion_text=text, targets=targets, tokenizer=None, format_type="json"
            )
        )
        for text in (good, bad)
    ]
    together = compute_group_advantages(
        completions=[good, bad],
        annotations=annotations,
        targets=targets,
        tokenizer=FakeTokenizer(),
        format_type="json",
    )
    alone = compute_group_advantages(
        completions=[good],
        annotations=annotations[:1],
        targets=targets,
        tokenizer=FakeTokenizer(),
        format_type="json",
    )
    assert together[0] == alone[0]
