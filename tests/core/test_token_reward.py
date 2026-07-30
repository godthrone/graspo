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


# ── Tests: compute_token_advantages ─────────────────────────────────────────────


def test_token_advantages_all_same():
    """When all completions have the same token rewards, advantage = 0."""
    rewards = [
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 0.5],
    ]
    advantages = compute_token_advantages(rewards)
    assert len(advantages) == 3
    for adv in advantages:
        assert all(abs(a) < 1e-6 for a in adv), f"Advantage should be ~0, got {adv}"


def test_token_advantages_mixed():
    """Mixed rewards produce non-zero advantages with correct direction."""
    rewards = [
        [1.0, 1.0, 1.0],  # perfect
        [1.0, 0.0, 0.5],  # format ok, content wrong at pos 1
        [0.0, 0.0, 0.0],  # format broken
    ]
    advantages = compute_token_advantages(rewards)

    # Position 0: two 1.0, one 0.0 → 1.0 has positive advantage.
    # 0.0 has advantage clamped to 0 (non-negative, reward-only).
    assert advantages[0][0] > 0  # 1.0 should have positive advantage
    assert advantages[2][0] == 0.0  # 0.0 clamped to 0 (non-negative)

    # Position 1: one 1.0, two 0.0 → 1.0 has positive advantage (best in group).
    # 0.0 has advantage clamped to 0.
    assert advantages[0][1] > 0
    assert advantages[1][1] == 0.0  # clamped to 0


def test_token_advantages_ragged_lengths():
    """Completions with different lengths are padded to max length."""
    rewards = [
        [1.0, 1.0, 1.0],
        [0.0, 0.0],  # shorter
        [1.0, 0.0],
    ]
    advantages = compute_token_advantages(rewards)

    # All padded to max length
    assert len(advantages[0]) == 3
    assert len(advantages[1]) == 3  # padded to 3
    assert len(advantages[2]) == 3  # padded to 3

    # Position 2: completion 1 doesn't have it, filled with 0
    assert advantages[1][2] == 0.0
