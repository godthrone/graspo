"""P1 判据：工具名是一个**可评分字段**——写错要吃与「这段名字都错了」相称的负反馈。

改前缺陷（真实 rollout 实测）：``<function=NAME>`` 走逐字符比对，首个不匹配字符标 E(−1.0)、
其余全 D，而名字**前面**的 22 个结构 token 都是 +1.0 ⇒ 工具名写错的 completion **净得 +21
（正的）**，把工具名叫错反而拿正梯度。

设计（见 ``tool_call_labeler`` 模块 docstring 的 P1 段）：

- 名字与 GT 一致 ⇒ 整段 S(+1.0)（与改前逐字符**相同**，回归点）
- 名字写错 ⇒ 从**首个不匹配字符**起整段 ``VALUE``（field=``TOOL_NAME_FIELD``）⇒ raw 0
  ⇒ 该段每个 token −1.0；随后 ``drop_from``（「第一个错误之后扔掉」）
- 名字比 GT 短 ⇒ 模型在"名字该继续"处写了 ``>`` ⇒ 该处 E + drop（残缺名字，旧语义）

⛔ 不改「第一个错误之后 drop」；⛔ 名字写错不产生 E（保住「E 之后全部为 D」不变式）。
"""

from __future__ import annotations

import pytest

from graspo.ripple.annotation.advantages import compute_group_advantages, value_spans
from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationInput, annotate
from graspo.ripple.annotation.tool_call_labeler import TOOL_NAME_FIELD

ROBOT = (
    "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n"
    "{}\n</parameter>\n</function>\n</tool_call>"
)

TARGETS = [
    {
        "output": {
            "tool_calls": [
                {"name": "robot_atomic_control", "arguments": {"action_type": "逆时针旋转"}}
            ]
        }
    }
]

GOOD = ROBOT.format("逆时针旋转")
#: 等长、只错最后一个字母（「首个不匹配字符起整段受罚」的可比构造）
GOOD_WITH_TYPO = GOOD.replace("robot_atomic_control", "robot_atomic_controX")
#: 与 GT 完全不同的工具名（整段都错）
OTHER_TOOL = GOOD.replace("robot_atomic_control", "extend_arm")


class FakeTokenizer:
    """每字符一个 token，offset ``(i, i+1)``。"""

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


def _annotate(completion: str):
    return annotate(
        AnnotationInput(
            completion_text=completion, targets=TARGETS, tokenizer=None, format_type="tool_call"
        )
    )


def _advantage(completion: str) -> list[float]:
    """同一构造给两条（P2 起 advantage 本就逐条闭合，组内条数不影响结果）。"""
    return compute_group_advantages(
        completions=[completion, completion],
        annotations=[_annotate(completion), _annotate(completion)],
        targets=TARGETS,
        tokenizer=FakeTokenizer(),
        format_type="tool_call",
    )[0]


def _name_span(completion: str) -> tuple[int, int]:
    """「模型写出的工具名」在 completion 里的字符区间。"""
    start = completion.index("<function=") + len("<function=")
    return start, completion.index(">", start)


# ══════════════════════════════════════════════════════════════════════════════
# 判据 1：工具名写错 ⇒ 优势总和显著低于写对
# ══════════════════════════════════════════════════════════════════════════════


def test_wrong_tool_name_advantages_sum_far_below_correct_one():
    """★ 参数完全相同、只有工具名不同：错的优势总和必须**明显低于**对的。

    实测（同一构造）：名字全对 ``sum = +109``；换成 ``extend_arm``（整段名字都错）
    只有 ``+12``（22 个结构 token − 10 个名字 token），差 97。
    改前同一对构造是 ``+104`` vs ``+21``——错的条**仍是正的（+21）**，正是 P1 的缺陷。
    """
    good_adv = _advantage(GOOD)
    other_adv = _advantage(OTHER_TOOL)

    assert sum(good_adv) == pytest.approx(109.0), "构造前提：名字全对的条优势总和"
    assert sum(other_adv) == pytest.approx(12.0), "整段名字都错：（22 结构 − 10 名字）"
    assert sum(good_adv) - sum(other_adv) == pytest.approx(97.0)


def test_one_letter_typo_costs_that_letter_and_stops_the_rest():
    """一个字母写错 ⇒ 该字母 −1.0，且其后的参数/闭合不再被算成 +1（被 drop）。"""
    good_adv = _advantage(GOOD)
    typo_adv = _advantage(GOOD_WITH_TYPO)
    name_start, name_end = _name_span(GOOD_WITH_TYPO)

    assert good_adv[name_end - 1] == 1.0, "名字全对时末位是 +1.0"
    assert typo_adv[name_end - 1] == -1.0, "名字末位写错时该字符必须 −1.0"
    # 共前缀仍然正确 ⇒ 仍然 +1.0（「和前面的 token 都正确」的那部分保号）
    assert all(typo_adv[i] == 1.0 for i in range(name_start, name_end - 1))
    assert sum(typo_adv) < sum(good_adv)


def test_other_tool_name_whole_segment_minus_one():
    """★ 换成另一个工具 ⇒ 名字段**每个** token 都是 −1.0（不是只罚第一个字符）。"""
    adv = _advantage(OTHER_TOOL)
    name_start, name_end = _name_span(OTHER_TOOL)
    assert name_end - name_start == len("extend_arm")
    for position in range(name_start, name_end):
        assert adv[position] == -1.0, (
            f"工具名第 {position - name_start} 个字符的优势应为 −1.0，实测 {adv[position]}"
        )


def test_correct_tool_name_tokens_stay_structure_plus_one():
    """回归点：工具名**写对**时名字段仍是结构 S(+1.0)，与改前逐字符相同。"""
    adv = _advantage(GOOD)
    name_start, name_end = _name_span(GOOD)
    for position in range(name_start, name_end):
        assert adv[position] == 1.0
    tags = _annotate(GOOD).tags
    assert all(tags[i] == CharTag.STRUCTURE for i in range(name_start, name_end))


# ══════════════════════════════════════════════════════════════════════════════
# 判据 2：名字错 ⇒ 不产生 E 标签；错名之后的 token 被 drop（「后面扔掉」）
# ══════════════════════════════════════════════════════════════════════════════


def test_wrong_tool_name_produces_no_error_tag():
    """★ 名字写错 ⇒ 名字段是 VALUE（不是 E），整条 completion 不含任何 E 标签。

    理由：「E 之后必须全部为 D」是数据集级不变式（``annotation_testset_v3`` 断言），
    整段名字标 E 会破坏它——而错误名字段本就该按**值**评分（−1 每 token）。
    """
    result = _annotate(OTHER_TOOL)
    assert CharTag.ERROR not in result.tags, "工具名写错不得产生 E（破坏「E 之后全 D」不变式）"
    assert CharTag.VALUE in result.tags


def test_content_after_wrong_tool_name_is_dropped():
    """★ 工具名错 ⇒ 其后的参数/闭合全部 D(0)（「第一个错误之后扔掉」）。

    依据负责人原则：「第一个错误的 token 会有负梯度，后面的 token 就扔掉」。
    """
    adv = _advantage(OTHER_TOOL)
    _, name_end = _name_span(OTHER_TOOL)
    assert all(value == 0.0 for value in adv[name_end:]), (
        f"错名之后的 token 必须全 0，实测非零 {[v for v in adv[name_end:] if v != 0.0][:5]}"
    )
    assert CharTag.DROPPED in _annotate(OTHER_TOOL).tags


def test_truncated_tool_name_marks_error_at_the_missing_position():
    """名字比 GT 短（模型在此写了 ``>``）⇒ E 于该处 + drop（残缺名字，旧语义保留）。"""
    short = GOOD.replace("robot_atomic_control", "robot_atomic_contro")
    result = _annotate(short)
    adv = _advantage(short)
    name_end = short.index(">", short.index("<function=") + len("<function="))
    assert result.tags[name_end] == CharTag.ERROR
    assert adv[name_end] == -1.0
    assert all(value == 0.0 for value in adv[name_end + 1 :])


# ══════════════════════════════════════════════════════════════════════════════
# 判据 3：名字段落在 value_spans 里、字段名是单一真相源常量
# ══════════════════════════════════════════════════════════════════════════════


def test_tool_name_field_is_a_value_span_with_the_sentinel_field():
    """名字段必须出现在 ``value_spans`` 中且字段名为 ``TOOL_NAME_FIELD``。"""
    result = _annotate(OTHER_TOOL)
    spans = value_spans(OTHER_TOOL, result.tags, result.fields)
    name_start, name_end = _name_span(OTHER_TOOL)
    name_spans = [(start, end) for start, end, field in spans if field == TOOL_NAME_FIELD]
    assert name_spans == [(name_start, name_end)]
    assert OTHER_TOOL[name_start:name_end] == "extend_arm"


def test_tool_name_field_does_not_collide_with_real_parameter_names():
    """保留名不与任何真实参数名撞车（``__`` 前后包裹是保留区）。"""
    assert TOOL_NAME_FIELD.startswith("__") and TOOL_NAME_FIELD.endswith("__")
    result = _annotate(GOOD)
    assert TOOL_NAME_FIELD not in {field for field in result.fields if field}


# ══════════════════════════════════════════════════════════════════════════════
# 判据 4：名字后缺 `>`（找到的 `>` 属于后面的标签）⇒ E + drop，**不产生 V 段**
# ══════════════════════════════════════════════════════════════════════════════
#
# B1（2026-11-18 复审阻断）：`_line_local_gt` 只保证 `>` 与本行同名，不保证它属于工具名。
# `<function=robot_atomic_control<parameter=action_type>` 会拿到 `action_type>` 的 `>`，
# 名字段成了 `robot_atomic_control<parameter=action_type` ⇒ 被当成"写错的值"整段 −1
# （实测一条 28 个 −1、Σadv −4.0），把写对的工具名与后面的结构字符一起罚进负分。
# 判据：名字段含 `<` ⇒ 缺 `>` ⇒ T31/T32 口径（E 于应出现 `>` 的位置 + drop）。
#
# 以下用例的期望值全部是**字面**的：名字段每个字符 S、缺 `>` 处 1 个 E、其后全 D。

_NAME_END = 22 + len("robot_atomic_control")  # `<tool_call>\n<function=` = 22 字符


def _assert_missing_gt_shape(completion: str) -> list[float]:
    """断言「缺 ``>``」形态：名字段全 S、缺 ``>`` 处 E、其后全 D，且不含任何 V。"""
    result = _annotate(completion)
    name_start = 22
    name_end = _NAME_END
    assert all(result.tags[i] == CharTag.STRUCTURE for i in range(name_start, name_end)), (
        "名字段必须全 S（写对的名字不能被罚）"
    )
    assert result.tags[name_end] == CharTag.ERROR, "E 必须落在「应出现 > 的位置」"
    assert all(tag == CharTag.DROPPED for tag in result.tags[name_end + 1 :]), "其后必须全 D"
    assert CharTag.VALUE not in result.tags, "缺 > 不得产生 V 段（那会把结构字符当值罚）"
    assert value_spans(completion, result.tags, result.fields) == [], "不得有值 span"
    adv = _advantage(completion)
    assert adv[name_end] == -1.0
    assert adv.count(-1.0) == 1, f"只能有 1 个 −1（不得把后续标签算成值错），实测 {adv.count(-1.0)}"
    # 22 个前导结构（`<tool_call>`+换行+`<function=`）+ 20 个名字字符全 +1，只罚 1 个 E。
    assert sum(adv) == pytest.approx(_NAME_END - 1.0), "写对的 token 必须都为正"
    return adv


def test_missing_gt_before_parameter_tag_is_error_not_value():
    """名字后直接跟 ``<parameter=…``（缺 `>`）⇒ E + drop，不产生 28 个 V/−1。"""
    completion = (
        "<tool_call>\n<function=robot_atomic_control<parameter=action_type>\n"
        "逆时针旋转\n</parameter>\n</function>\n</tool_call>"
    )
    _assert_missing_gt_shape(completion)


def test_missing_gt_before_function_close_is_error_not_value():
    """名字后直接跟 ``</function>``（缺 `>`）⇒ 同上。"""
    _assert_missing_gt_shape("<tool_call>\n<function=robot_atomic_control</function>")


def test_wrong_name_with_missing_gt_is_error_not_value():
    """名字**也**写错、且缺 `>` ⇒ 仍按「缺 `>`」处理：E 于首个不匹配字符，不产生 V 段。

    为什么这样不丢信息：此时第一个错误字符之后照旧 drop（「后面扔掉」），
    而缺 `>` 本身已经是明确的格式错误——与 T45/T46 的真实数据形态一致。
    """
    completion = (
        "<tool_call>\n<function=extend_arm<parameter=action_type>\n"
        "逆时针旋转\n</parameter>\n</function>\n</tool_call>"
    )
    result = _annotate(completion)
    assert result.tags[22] == CharTag.ERROR, "E 于首个不匹配字符（名字首字符）"
    assert CharTag.VALUE not in result.tags
    assert all(tag == CharTag.DROPPED for tag in result.tags[23:])


def test_gt_name_that_contains_angle_bracket_still_matches():
    """极端情况：GT 工具名自身含 ``<``（``a<b``）时仍按「名字写对」处理，不被新判据误伤。"""
    completion = (
        "<tool_call>\n<function=a<b>\n<parameter=action_type>\n逆时针旋转\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    targets = [
        {"output": {"tool_calls": [{"name": "a<b", "arguments": {"action_type": "逆时针旋转"}}]}}
    ]
    result = annotate(
        AnnotationInput(
            completion_text=completion, targets=targets, tokenizer=None, format_type="tool_call"
        )
    )
    name_start = 22
    name_end = name_start + len("a<b")
    assert all(result.tags[i] == CharTag.STRUCTURE for i in range(name_start, name_end))
    assert result.tags[name_end] == CharTag.STRUCTURE, "名字后的 > 是结构"
