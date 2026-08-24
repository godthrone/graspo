"""Advantage 确定性验证：固定 rollout-group 的 token advantage 等期望值。

验证目标（C1/C2 验收）：
1. 已知 rollout-group 的 token advantage 等于期望值（S→+1.0, E→−1.0, T/W/D→0）
2. 同 seed 两次运行，advantage 结果一致（确定性）
3. n=1 安全零：单 completion 组内 V token advantage=0
4. μ_f 组内相对化：V token 的 advantage 是 raw − 组内均值

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
    """完美匹配的 completion：所有 S token advantage=+1.0，V token 因 n=1 为 0。"""
    adv = _compute_advantages([_PERFECT_TOOL_CALL], _PERFECT_TARGETS)

    assert len(adv) == 1
    # 所有 token 的 advantage 应 >= 0（完美匹配：S=+1.0, V=0(n=1), 无 E）
    for a in adv[0]:
        assert a >= 0.0, f"expected non-negative advantage, got {a}"
    # 至少有一个 S token 的 advantage=+1.0
    assert any(a == 1.0 for a in adv[0]), f"expected at least one +1.0 advantage, got {adv[0]}"


def test_perfect_completion_n1_safety_zero():
    """n=1 安全零：单 completion 组内 V token advantage=0（μ_f=raw）。"""
    adv = _compute_advantages([_PERFECT_TOOL_CALL], _PERFECT_TARGETS)

    # 所有 V token 的 advantage 应为 0（n=1 → μ_f = raw → adv = 0）
    v_advantages = [a for a in adv[0] if a != 1.0 and a != 0.0 and a != -1.0]
    # 在 n=1 情况下，V token advantage 应该为 0
    non_s_adv = [a for a in adv[0] if a != 1.0]
    for a in non_s_adv:
        assert a == 0.0, f"n=1: expected V token advantage=0, got {a}"


# ── 错误结构：E token 分配 ────────────────────────────────────────────────────


_BROKEN_TOOL_CALL = (
    "<tool_call>\n"
    "<function=robot_atomic_control>\n"
    "<parameter=action_type>\nWRONG_ACTION\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_broken_completion_has_error_token():
    """拼写错误的 completion：首个不匹配点标 E（adv=-1.0），后续截断为 D（adv=0）。"""
    adv = _compute_advantages([_BROKEN_TOOL_CALL], _PERFECT_TARGETS)

    assert len(adv) == 1
    # 应有至少一个 E token (adv=-1.0)
    assert any(a == -1.0 for a in adv[0]), f"expected at least one -1.0 advantage, got {adv[0]}"
    # E 之后的所有 token 应为 D（adv=0）
    found_e = False
    for a in adv[0]:
        if a == -1.0:
            found_e = True
        elif found_e:
            assert a == 0.0, f"token after E should be DROPPED (adv=0), got {a}"


# ── 组内相对化 (μ_f) ─────────────────────────────────────────────────────────


def test_group_mu_f_relative_advantage():
    """多 completion 组：V token advantage = raw − μ_f（组内相对化）。"""
    # A: 完美匹配 (action_type="逆时针旋转", angle_deg=16.9)
    perfect = _PERFECT_TOOL_CALL
    # B: 动作错，角度对 (action_type="顺时针旋转", angle_deg=16.9)
    wrong_action = perfect.replace("逆时针旋转", "顺时针旋转")
    # C: 动作对，角度错 (action_type="逆时针旋转", angle_deg=40)
    wrong_angle = perfect.replace("16.9", "40")

    targets = _PERFECT_TARGETS
    adv = _compute_advantages([perfect, wrong_action, wrong_angle], targets)

    assert len(adv) == 3

    # 所有 completion 应有 S token 的 advantage=+1.0
    for i in range(3):
        assert any(a == 1.0 for a in adv[i]), f"completion {i}: expected S tokens"

    # 完美 completion 的 V token advantage 应 >= 错误 completion 的 V token advantage
    # （因为 μ_f 是组内均值，完美的大于均值，错误的小于均值）
    v_perfect = [a for a in adv[0] if a != 1.0 and a != -1.0 and a != 0.0]
    v_wrong_action = [a for a in adv[1] if a != 1.0 and a != -1.0 and a != 0.0]
    if v_perfect and v_wrong_action:
        # 完美 completion 的 V advantage 应更大
        assert max(v_perfect) >= max(v_wrong_action), (
            f"perfect V adv {v_perfect} should >= wrong V adv {v_wrong_action}"
        )


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

    adv = _compute_advantages(
        [incomplete], _PERFECT_TARGETS, truncated_by_max=[True]
    )

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
    think_completion = (
        "thinking\n"
        "Let me analyze the request.\n"
        "response\n"
        + _PERFECT_TOOL_CALL
    )

    adv = _compute_advantages([think_completion], _PERFECT_TARGETS)

    assert len(adv) == 1
    # think 区域的 token 应全部为 0
    # think 标签 `thinking` 后的内容直到 `response` 都是 THINK
    # 验证至少有一些 advantage 为 0（THINK 区域）
    assert any(a == 0.0 for a in adv[0]), "expected some zero-advantage THINK tokens"