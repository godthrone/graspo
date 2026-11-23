"""Advantage 确定性验证：固定 rollout-group 的 token advantage 等期望值。

验证目标（C1/C2 验收）：
1. 已知 rollout-group 的 token advantage 等于期望值（S→+1.0, E→−1.0, T/W/D→0）
2. 同 seed 两次运行，advantage 结果一致（确定性）
3. n=1 单 completion 组：正确的值 token 仍是 +1.0（P2 起没有"安全零"）
4. 值 token 的优势只看**这条 completion 自己**对不对，与同组其它 completion 无关
   （旧式 ``raw − μ_f`` 组内相对化已随 P2 删除）

测试策略：
- 使用 FakeTokenizer（每字符一个 token，offset 一一对应）
- 构造已知结构的 completion 文本
- 使用 annotate() 获取标注，compute_group_advantages() 计算 advantage
"""

import pytest

from graspo.ripple.annotation.advantages import compute_group_advantages
from graspo.ripple.annotation.char_tag import CharTag
from graspo.ripple.annotation.labeler import AnnotationInput, annotate

# ── FakeTokenizer（与 test_advantages.py 相同模式）───────────────────────────


class FakeTokenizer:
    """每字符一个 token，offset (i, i+1)——字符级标注与 token 一一对应。"""

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


# ── 工具函数 ─────────────────────────────────────────────────────────────────


def _make_targets_tool_call(name: str, **kwargs) -> list[dict]:
    """构造 tool_call 格式的 targets。"""
    return [{"output": {"tool_calls": [{"name": name, "arguments": dict(kwargs)}]}}]


def _annotate_completion(completion: str, targets: list[dict], format_type: str = "tool_call"):
    """标注单条 completion。"""
    return annotate(
        AnnotationInput(
            completion_text=completion,
            targets=targets,
            tokenizer=None,
            format_type=format_type,
        )
    )


def _compute_advantages(
    completions: list[str],
    targets: list[dict],
    format_type: str = "tool_call",
    **kwargs,
) -> list[list[float]]:
    """计算一组 completion 的 per-token advantage。"""
    annotations = [_annotate_completion(c, targets, format_type) for c in completions]
    return compute_group_advantages(
        completions=completions,
        annotations=annotations,
        targets=targets,
        tokenizer=FakeTokenizer(),
        format_type=format_type,
        **kwargs,
    )


def _positions(text: str, literal: str) -> list[int]:
    """``literal`` 在 ``text`` 里占的字符下标（FakeTokenizer 下 token 下标 == 字符下标）。"""
    start = text.index(literal)
    return list(range(start, start + len(literal)))


# ── 正确结构：S/V token 分配 ─────────────────────────────────────────────────


_PERFECT_TOOL_CALL = (
    "<tool_call>\n"
    "<function=robot_atomic_control>\n"
    "<parameter=action_type>\n逆时针旋转\n</parameter>\n"
    "<parameter=angle_deg>\n16.9\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

_PERFECT_TARGETS = _make_targets_tool_call(
    "robot_atomic_control", action_type="逆时针旋转", angle_deg=16.9
)


def test_perfect_completion_structure_tokens_positive():
    """完美匹配的 completion：S 与 V token 都是 +1.0，整条无负值。"""
    adv = _compute_advantages([_PERFECT_TOOL_CALL], _PERFECT_TARGETS)

    assert len(adv) == 1
    # 所有 token 的 advantage 应 >= 0（完美匹配：S=+1.0, 值对 ⇒ V=+1.0, 无 E）
    for a in adv[0]:
        assert a >= 0.0, f"expected non-negative advantage, got {a}"
    # 至少有一个 S token 的 advantage=+1.0
    assert any(a == 1.0 for a in adv[0]), f"expected at least one +1.0 advantage, got {adv[0]}"


def test_perfect_completion_n1_value_tokens_are_plus_one():
    """★ n=1（组内只有一条）：正确的值 token 是 +1.0，0.0 只出现在 W（值前空白）。

    P2（2026-11-18）口径：值 token 是 ±1，只看它自己对不对 ⇒ 完美 completion 里
    **没有** 0.0 的值 token。旧 docstring 写的「n=1 安全零（V=0，μ_f=raw）」是 μ_f 时代的
    结论，已随 P2 删除；原断言实际检查的 4 个 0 是 W token（值前空白），不是值 token。
    """
    adv = _compute_advantages([_PERFECT_TOOL_CALL], _PERFECT_TARGETS)[0]
    tags = _annotate_completion(_PERFECT_TOOL_CALL, _PERFECT_TARGETS).tags

    value_positions = [i for i, tag in enumerate(tags) if tag == CharTag.VALUE]
    assert len(value_positions) == 9, "构造前提：两个值共 9 个字符标 V"
    for position in value_positions:
        assert adv[position] == 1.0, f"正确的值 token@{position} 必须 +1.0，实测 {adv[position]}"

    waste_positions = [i for i, tag in enumerate(tags) if tag == CharTag.WASTE]
    assert len(waste_positions) == 4, "构造前提：两个值前的空白共 4 个字符标 W"
    for position in waste_positions:
        assert adv[position] == 0.0, f"W token@{position} 必须 0.0，实测 {adv[position]}"

    assert -1.0 not in adv


# ── 错误结构：E token 分配 ────────────────────────────────────────────────────


_BROKEN_TOOL_CALL = (
    "<tool_call>\n"
    "<function=robot_atomic_control>\n"
    "<parameter=action_type>\nWRONG_ACTION\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_broken_completion_has_error_token():
    """拼写错误的 completion：首个不匹配点标 E（adv=-1.0），后续截断为 D（adv=0）。

    P2（2026-11-18）口径变更：值 token 的错误现在也是 −1.0，所以**不能**再用
    "第一个 −1.0 就是 E"来定位错误点——必须按字符标注（``CharTag.ERROR``）定位。
    """
    adv = _compute_advantages([_BROKEN_TOOL_CALL], _PERFECT_TARGETS)
    tags = _annotate_completion(_BROKEN_TOOL_CALL, _PERFECT_TARGETS).tags

    assert len(adv) == 1
    error_positions = [index for index, tag in enumerate(tags) if tag == CharTag.ERROR]
    assert error_positions, "构造前提：该 completion 必须有 E 定位"
    error_pos = error_positions[0]
    assert adv[0][error_pos] == -1.0, f"E 处必须是 −1.0，实测 {adv[0][error_pos]}"
    # E 之后的所有 token 应为 D（adv=0）——这是「第一个错误之后扔掉」的机核
    for index in range(error_pos + 1, len(adv[0])):
        assert adv[0][index] == 0.0, f"E 之后的 token@{index} 应为 D(0)，实测 {adv[0][index]}"


# ── 值 token 只看自己对不对（P2：μ_f 已删）───────────────────────────────────


def test_group_value_advantage_follows_each_completion():
    """★ 三条一组：值 token 的 ±1 只看**这条自己**对不对，与同组其它 completion 无关。

    P2（2026-11-18）口径变更：旧式 ``adv(V) = raw − μ_f``（组内均值）已删除。
    所以这里钉的是**方向**而不是"相对谁更大"：
    A 完美 ⇒ 值 token 全 +1.0；B 动作错 ⇒ action_type 的 token 全 −1.0（角度对，仍是 +1.0）；
    C 角度错 ⇒ angle_deg 的 token 全 −1.0（动作对，仍是 +1.0）。
    """
    # A: 完美匹配 (action_type="逆时针旋转", angle_deg=16.9)
    perfect = _PERFECT_TOOL_CALL
    # B: 动作错，角度对 (action_type="顺时针旋转", angle_deg=16.9)
    wrong_action = perfect.replace("逆时针旋转", "顺时针旋转")
    # C: 动作对，角度错 (action_type="逆时针旋转", angle_deg=40)
    wrong_angle = perfect.replace("16.9", "40")

    adv = _compute_advantages([perfect, wrong_action, wrong_angle], _PERFECT_TARGETS)
    assert len(adv) == 3

    # A：值全对 ⇒ 两个字段的 token 都是 +1.0
    for position in _positions(perfect, "逆时针旋转") + _positions(perfect, "16.9"):
        assert adv[0][position] == 1.0, f"A 的值 token@{position} 必须 +1.0"

    # B：动作错（每个字符 −1.0）＋ 角度对（+1.0）
    for position in _positions(wrong_action, "顺时针旋转"):
        assert adv[1][position] == -1.0, f"B 的 action_type@{position} 必须 −1.0"
    for position in _positions(wrong_action, "16.9"):
        assert adv[1][position] == 1.0, f"B 的 angle_deg@{position} 必须 +1.0"

    # C：动作对（+1.0）＋ 角度错（40 远超容差 ⇒ −1.0）
    for position in _positions(wrong_angle, "逆时针旋转"):
        assert adv[2][position] == 1.0, f"C 的 action_type@{position} 必须 +1.0"
    for position in _positions(wrong_angle, "40"):
        assert adv[2][position] == -1.0, f"C 的 angle_deg@{position} 必须 −1.0"

    # 逐条闭合：单条成组算出来的结果与三条成组时**完全一致**（无组内耦合）
    alone = _compute_advantages([perfect], _PERFECT_TARGETS)[0]
    assert alone == adv[0], "值 token 的优势不得依赖同组其它 completion"


# ── 确定性验证 ────────────────────────────────────────────────────────────────


def test_advantage_computation_is_deterministic():
    """同一输入两次运行，advantage 结果完全一致。"""
    completions = [_PERFECT_TOOL_CALL, _BROKEN_TOOL_CALL]
    targets = _PERFECT_TARGETS

    adv1 = _compute_advantages(completions, targets)
    adv2 = _compute_advantages(completions, targets)

    assert len(adv1) == len(adv2)
    for i in range(len(adv1)):
        assert len(adv1[i]) == len(adv2[i])
        for j in range(len(adv1[i])):
            assert adv1[i][j] == pytest.approx(adv2[i][j]), (
                f"position [{i}][{j}]: {adv1[i][j]} != {adv2[i][j]}"
            )


def test_advantage_length_matches_completion_tokens():
    """advantage 列表长度 = completion 的 token 数（FakeTokenizer 下 = 字符数）。"""
    completions = [_PERFECT_TOOL_CALL, _BROKEN_TOOL_CALL]
    targets = _PERFECT_TARGETS
    adv = _compute_advantages(completions, targets)

    for i, c in enumerate(completions):
        expected_len = len(c)  # FakeTokenizer: 每字符一个 token
        assert len(adv[i]) == expected_len, (
            f"completion {i}: expected {expected_len} advantages, got {len(adv[i])}"
        )


# ── 末尾 EOS 惩罚 ─────────────────────────────────────────────────────────────


def test_incomplete_structure_gets_tail_eos():
    """结构不完整（未闭合）且非 max_len 截断：末尾字符标 E（adv=-1.0）。"""
    incomplete = "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n逆时针旋转"
    # 没有 </tool_call> 闭合

    adv = _compute_advantages([incomplete], _PERFECT_TARGETS)

    assert len(adv) == 1
    # 末尾应有 E token
    assert adv[0][-1] == -1.0, f"expected tail EOS (-1.0), got {adv[0][-1]}"


def test_truncated_by_max_no_tail_eos():
    """max_len 硬截断：不标末尾 EOS（被外力切断，不算"没写完"）。"""
    incomplete = "<tool_call>\n<function=robot_atomic_control>\n<parameter=action_type>\n逆时针旋转"

    adv = _compute_advantages([incomplete], _PERFECT_TARGETS, truncated_by_max=[True])

    assert len(adv) == 1
    # 末尾不应有 E token（被截断保护）
    assert adv[0][-1] != -1.0, f"truncated: expected no tail EOS, got {adv[0][-1]}"


# ── 边界情况 ──────────────────────────────────────────────────────────────────


def test_empty_completion_raises():
    """空 completion 列表应报错。"""
    with pytest.raises(ValueError):
        _compute_advantages([], _PERFECT_TARGETS)


def test_mismatched_completions_and_annotations_raises():
    """completions 与 annotations 数量不匹配应报错。"""
    with pytest.raises(ValueError, match="aligned"):
        compute_group_advantages(
            completions=["a", "b"],
            annotations=[_annotate_completion("a", _PERFECT_TARGETS)],
            targets=_PERFECT_TARGETS,
            tokenizer=FakeTokenizer(),
            format_type="tool_call",
        )


def test_invalid_format_type_raises():
    """无效的 format_type 应报错。"""
    with pytest.raises(ValueError, match="format_type"):
        _compute_advantages([_PERFECT_TOOL_CALL], _PERFECT_TARGETS, format_type="invalid")


def test_think_tokens_have_zero_advantage():
    """THINK token 的 advantage 为 0。"""
    # 构造带 think 的 completion
    think_completion = "thinking\nLet me analyze the request.\nresponse\n" + _PERFECT_TOOL_CALL

    adv = _compute_advantages([think_completion], _PERFECT_TARGETS)

    assert len(adv) == 1
    # think 区域的 token 应全部为 0
    # think 标签 `thinking` 后的内容直到 `response` 都是 THINK
    # 验证至少有一些 advantage 为 0（THINK 区域）
    assert any(a == 0.0 for a in adv[0]), "expected some zero-advantage THINK tokens"
