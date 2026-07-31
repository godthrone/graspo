"""Tests for GRASPO-Ripple token-level reward computation."""

import pytest

from graspo.core.completion import ParsedCompletion
from graspo.core.token_reward import (
    compute_token_advantages,
    compute_token_rewards,
    _is_format_valid,
)


# ── Fake tokenizer for testing ────────────────────────────────────────────────


class FakeTokenizer:
    """Minimal tokenizer that simulates Qwen tool-call tokenization.

    Each "token" is a single character for simple testing.  Offset mapping
    maps each character to itself.
    """

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return fake token IDs: each char gets an ID based on ord()."""
        ids = [ord(c) % 1000 for c in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text: str, return_offsets_mapping: bool = False, **kwargs):
        """Simulate HF tokenizer call with offset mapping."""
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]

        class FakeEncoding:
            def __init__(self, input_ids, offset_mapping):
                self.input_ids = input_ids
                self.offset_mapping = offset_mapping

        return FakeEncoding(ids, offsets)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_parsed(
    tool_calls: list[dict] | None = None,
    parse_errors: list[str] | None = None,
) -> ParsedCompletion:
    return ParsedCompletion(
        raw_text="<tool_call>...</tool_call>",
        tool_calls=tool_calls or [],
        parse_errors=parse_errors or [],
        parser_name="qwen_xml_tool_call",
    )


def _make_target(tool_calls: list[dict]) -> list[dict]:
    return [{"output": {"tool_calls": tool_calls}}]


# ── Tests: format validity ─────────────────────────────────────────────────────


def test_format_valid_empty_errors_no_calls():
    parsed = _make_parsed(tool_calls=[], parse_errors=[])
    targets = _make_target([{"name": "test", "arguments": {"x": 1}}])
    assert not _is_format_valid(parsed, targets)


def test_format_valid_with_parse_errors():
    parsed = _make_parsed(
        tool_calls=[{"name": "f", "arguments": {}}],
        parse_errors=["bad tool call"],
    )
    targets = _make_target([{"name": "f", "arguments": {}}])
    assert not _is_format_valid(parsed, targets)


def test_format_valid_too_many_tool_calls():
    parsed = _make_parsed(
        tool_calls=[
            {"name": "f1", "arguments": {}},
            {"name": "f2", "arguments": {}},
        ]
    )
    targets = _make_target([{"name": "f1", "arguments": {}}])
    assert not _is_format_valid(parsed, targets)


def test_format_valid_ok():
    parsed = _make_parsed(
        tool_calls=[{"name": "f", "arguments": {"x": 1}}],
    )
    targets = _make_target([{"name": "f", "arguments": {"x": 1}}])
    assert _is_format_valid(parsed, targets)


# ── Tests: compute_token_rewards (format correct path) ──────────────────────────


def test_format_correct_all_tokens_reward_one_for_format():
    """When format is correct and all tokens are format tokens, all get 1.0."""
    tokenizer = FakeTokenizer()
    # Simple completion with no content spans (no tool calls with arguments)
    parsed = _make_parsed(
        tool_calls=[{"name": "move", "arguments": {}}],
    )
    targets = _make_target([{"name": "move", "arguments": {}}])

    text = "<tool_call>\n<function=move>\n</function>\n</tool_call>"
    gen_ids = tokenizer.encode(text)

    rewards = compute_token_rewards(
        generated_token_ids=gen_ids,
        completion_text=text,
        parsed=parsed,
        targets=targets,
        tokenizer=tokenizer,
    )

    assert len(rewards) == len(gen_ids)
    assert all(r == 1.0 for r in rewards), f"All format tokens should be 1.0, got {rewards}"


def test_format_correct_content_tokens_get_field_score():
    """Content tokens (parameter values) get field scores, not 1.0."""
    tokenizer = FakeTokenizer()
    parsed = _make_parsed(
        tool_calls=[{"name": "move", "arguments": {"action": "forward"}}],
    )
    targets = _make_target([{"name": "move", "arguments": {"action": "left"}}])

    text = "<tool_call>\n<function=move>\n<parameter=action>\nforward\n</parameter>\n</function>\n</tool_call>"
    gen_ids = tokenizer.encode(text)

    rewards = compute_token_rewards(
        generated_token_ids=gen_ids,
        completion_text=text,
        parsed=parsed,
        targets=targets,
        tokenizer=tokenizer,
    )

    # The content token "forward" should NOT be 1.0 (action type mismatch)
    # Find the position of "forward" in the text and check its reward
    forward_pos = text.index("forward")
    # With FakeTokenizer (char-level), the token at position forward_pos
    # should have a non-1.0 score
    assert rewards[forward_pos] < 1.0, (
        f"Content token 'forward' should have field score < 1.0, got {rewards[forward_pos]}"
    )


# ── Tests: compute_token_rewards (format broken path) ───────────────────────────


def test_format_broken_token_by_token_match():
    """Format broken: token-by-token comparison with ground truth."""
    tokenizer = FakeTokenizer()
    parsed = _make_parsed(tool_calls=[], parse_errors=["no tool call found"])
    targets = _make_target([{"name": "move", "arguments": {"action": "left"}}])

    # This is the ground truth format
    gt_text = "<tool_call>\n<function=move>\n<parameter=action>\nleft\n</parameter>\n</function>\n</tool_call>"

    # Completion that matches ground truth partially
    completion_text = "<tool_call>\n<function=move>\n<parameter=wrong>\nright\n</parameter>\n</function>\n</tool_call>"
    gen_ids = tokenizer.encode(completion_text)

    rewards = compute_token_rewards(
        generated_token_ids=gen_ids,
        completion_text=completion_text,
        parsed=parsed,
        targets=targets,
        tokenizer=tokenizer,
    )

    assert len(rewards) == len(gen_ids)

    # First few tokens match ground truth → 1.0
    # The text "<tool_call>" matches (characters before the mismatch)
    # Find where the texts diverge
    for i in range(min(len(completion_text), len(gt_text))):
        if completion_text[i] != gt_text[i]:
            break
        assert rewards[i] == 1.0, f"Token {i} ('{completion_text[i]}') should match, got {rewards[i]}"

    # After divergence, tokens should be 0.0
    first_diff = next(
        i for i in range(min(len(completion_text), len(gt_text)))
        if completion_text[i] != gt_text[i]
    )
    assert rewards[first_diff] == 0.0, (
        f"Token at first divergence should be 0.0, got {rewards[first_diff]}"
    )


def test_format_broken_extra_tokens_zero():
    """Extra generated tokens beyond ground truth get 0.0."""
    tokenizer = FakeTokenizer()
    parsed = _make_parsed(tool_calls=[], parse_errors=["no tool call found"])
    targets = _make_target([{"name": "move", "arguments": {}}])

    gt_text = "<tool_call>\n<function=move>\n</function>\n</tool_call>"
    # Completion is longer than ground truth
    completion_text = gt_text + "extra garbage"
    gen_ids = tokenizer.encode(completion_text)

    rewards = compute_token_rewards(
        generated_token_ids=gen_ids,
        completion_text=completion_text,
        parsed=parsed,
        targets=targets,
        tokenizer=tokenizer,
    )

    # First len(gt_text) tokens match → 1.0
    for i in range(len(gt_text)):
        assert rewards[i] == 1.0, f"Token {i} should match ground truth"

    # Extra tokens → 0.0
    for i in range(len(gt_text), len(completion_text)):
        assert rewards[i] == 0.0, f"Extra token {i} should be 0.0, got {rewards[i]}"


def test_format_broken_shorter_completion():
    """Completion shorter than ground truth: missing tokens get no reward."""
    tokenizer = FakeTokenizer()
    parsed = _make_parsed(tool_calls=[], parse_errors=["truncated"])
    targets = _make_target([{"name": "move", "arguments": {"action": "left"}}])

    gt_text = "<tool_call>\n<function=move>\n<parameter=action>\nleft\n</parameter>\n</function>\n</tool_call>"
    # Truncated completion
    completion_text = "<tool_call>\n<function=move>\n<para"
    gen_ids = tokenizer.encode(completion_text)

    rewards = compute_token_rewards(
        generated_token_ids=gen_ids,
        completion_text=completion_text,
        parsed=parsed,
        targets=targets,
        tokenizer=tokenizer,
    )

    assert len(rewards) == len(gen_ids)
    # Matching prefix gets 1.0
    for i in range(len(gen_ids)):
        assert rewards[i] == 1.0, f"All tokens in matching prefix should be 1.0"


# ── Helpers for building parallel format/content masks ────────────────────────


def _fmt(rewards: list[float]) -> "tuple[list[bool], list[str | None]]":
    """Build is_format and field_keys for a completion where all tokens are format."""
    return [True] * len(rewards), [None] * len(rewards)


def _ct(rewards: list[float], fk: str) -> "tuple[list[bool], list[str | None]]":
    """Build is_format and field_keys for a completion where all tokens are content
    belonging to field *fk*."""
    return [False] * len(rewards), [fk] * len(rewards)


def _mixed(
    rewards: list[float],
    is_fmt: list[bool],
    fk: "list[str | None]",
) -> "tuple[list[bool], list[str | None]]":
    """Explicit format/content masks."""
    return is_fmt, fk


# ── Tests: compute_token_advantages (new format/content API) ────────────────────


def test_format_tokens_fixed_plus_minus_one():
    """Format tokens get +1.0 (clean) or -1.0 (broken), independent of group."""
    rewards = [
        [1.0, 1.0, 1.0],   # clean
        [1.0, 1.0, 1.0],   # clean
        [0.0, 0.0, 0.0],   # broken
    ]
    # All tokens are format tokens
    is_fmt = [
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ]
    fk = [
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ]
    advantages = compute_token_advantages(rewards, is_fmt, fk)
    # Clean → +1.0
    for t in range(3):
        assert advantages[0][t] == 1.0
        assert advantages[1][t] == 1.0
    # Broken → -1.0
    for t in range(3):
        assert advantages[2][t] == -1.0


def test_content_tokens_field_level_z_score():
    """Content tokens get field-level advantage (z-score across clean only)."""
    # 2 clean, 1 broken. Content tokens for field "a.p".
    rewards = [
        [1.0, 0.8, 0.0],    # clean, field scores for "a.p": 1.0, 0.8
        [0.0, 0.0, 0.0],    # broken — excluded from content stats
        [1.0, 0.0, 0.0],    # clean, field score for "a.p": 1.0
    ]
    # position 0: format; positions 1-2: content for "a.p"
    is_fmt = [
        [True, False, False],
        [True, True, True],    # broken: all format
        [True, False, False],
    ]
    fk = [
        [None, "a.p", "a.p"],
        [None, None, None],
        [None, "a.p", "a.p"],
    ]
    advantages = compute_token_advantages(rewards, is_fmt, fk)

    # Format tokens at position 0: clean +1.0, broken -1.0
    assert advantages[0][0] == 1.0
    assert advantages[1][0] == -1.0
    assert advantages[2][0] == 1.0

    # Content tokens at position 1 (field "a.p"):
    # clean scores: [0.8, 0.0], mean=0.4, std=0.5657
    # idx 0: (0.8-0.4)/0.5657 ≈ 0.707
    # idx 2: (0.0-0.4)/0.5657 ≈ -0.707
    assert advantages[0][1] > 0  # above mean
    assert advantages[2][1] < 0  # below mean
    # broken content token → 0
    assert advantages[1][1] == 0.0


def test_content_tokens_broken_completion_gets_zero():
    """Broken completion content tokens get A=0 (no meaningful content)."""
    rewards = [
        [1.0, 0.5],
        [0.0, 0.0],    # broken
    ]
    is_fmt = [
        [True, False],
        [True, True],
    ]
    fk = [
        [None, "a.p"],
        [None, None],
    ]
    advantages = compute_token_advantages(rewards, is_fmt, fk)
    # Broken content token at pos 1: A=0
    assert advantages[1][1] == 0.0


def test_all_clean_format_tokens_no_variance():
    """All clean format tokens: all get +1.0 (harmless — no gradient difference)."""
    rewards = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    is_fmt = [
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ]
    fk = [
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ]
    advantages = compute_token_advantages(rewards, is_fmt, fk)
    # All +1.0 → no variance → no gradient for format tokens (already learned).
    for i in range(3):
        for t in range(3):
            assert advantages[i][t] == 1.0


def test_compute_token_rewards_returns_triple():
    """compute_token_rewards now returns (rewards, is_format, field_keys)."""
    tokenizer = FakeTokenizer()
    targets = _make_targets([("robot_atomic_control", {"action_type": "逆时针旋转", "distance_cm": 10, "angle_deg": 30})])

    # Format-correct completion
    completion_text = (
        "<tool_call>\n"
        "<function=robot_atomic_control>\n"
        "<parameter=action_type>\n逆时针旋转\n</parameter>\n"
        "<parameter=distance_cm>\n10\n</parameter>\n"
        "<parameter=angle_deg>\n30\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    gen_ids = [ord(c) % 1000 for c in completion_text]
    parsed = ParsedCompletion(
        raw_text=completion_text,
        tool_calls=[{
            "name": "robot_atomic_control",
            "arguments": {"action_type": "逆时针旋转", "distance_cm": "10", "angle_deg": "30"},
        }],
    )
    rewards, is_format, field_keys = compute_token_rewards(
        gen_ids, completion_text, parsed, targets, tokenizer,
        numeric_tolerance=10,
    )
    assert len(rewards) == len(is_format) == len(field_keys)
    # At least some format tokens
    assert any(is_format)
    # At least some content tokens
    assert any(not f for f in is_format)
    # Content tokens have non-None field keys
    for idx, is_fmt in enumerate(is_format):
        if not is_fmt:
            assert field_keys[idx] is not None, f"Content token at {idx} must have field_key"
    # Format tokens have None field keys
    for idx, is_fmt in enumerate(is_format):
        if is_fmt:
            assert field_keys[idx] is None, f"Format token at {idx} must have None field_key"

    # Format-broken completion
    broken_text = "唱跳演唱演唱會"
    broken_ids = [ord(c) % 1000 for c in broken_text]
    broken_parsed = ParsedCompletion(
        raw_text=broken_text,
        parse_errors=["parse error"],
        tool_calls=[],
    )
    rewards_b, is_fmt_b, fk_b = compute_token_rewards(
        broken_ids, broken_text, broken_parsed, targets, tokenizer,
        numeric_tolerance=10,
    )
    # All tokens should be format tokens for broken
    assert all(is_fmt_b)
    assert all(f is None for f in fk_b)
