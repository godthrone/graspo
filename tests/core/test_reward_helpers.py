"""Tests for reward helper functions — BADGE §11.1."""

import pytest

from graspo.core.reward_helpers import (
    is_valid_json,
    normalize_targets,
    normalize_tool_calls,
)


# ── is_valid_json ────────────────────────────────────────────────────────────


def test_is_valid_json_true_for_valid_object():
    assert is_valid_json('{"key": "value"}') is True


def test_is_valid_json_true_for_valid_array():
    assert is_valid_json('[1, 2, 3]') is True


def test_is_valid_json_false_for_empty_string():
    assert is_valid_json("") is False


def test_is_valid_json_false_for_plain_text():
    assert is_valid_json("not json at all") is False


def test_is_valid_json_false_for_unclosed_brace():
    assert is_valid_json('{"key": "value"') is False


def test_is_valid_json_false_for_none():
    # None fails in json.loads() with TypeError, which is caught → False
    # Note: is_valid_json catches both TypeError and ValueError internally
    assert is_valid_json(None) is False


# ── normalize_targets ────────────────────────────────────────────────────────


def test_normalize_targets_content_target():
    targets = [{"id": "t1", "output": {"content": {"name": "test"}}}]
    result = normalize_targets(targets)
    assert len(result) == 1
    assert result[0]["output"]["content"] == {"name": "test"}


def test_normalize_targets_tool_call_target():
    targets = [{"id": "t1", "output": {"tool_calls": [{"name": "search", "arguments": {"q": "x"}}]}}]
    result = normalize_targets(targets)
    assert len(result) == 1
    assert result[0]["output"]["tool_calls"] == [{"name": "search", "arguments": {"q": "x"}}]


def test_normalize_targets_multiple_targets():
    targets = [
        {"id": "t1", "output": {"content": {"a": 1}}},
        {"id": "t2", "output": {"content": {"b": 2}}},
    ]
    result = normalize_targets(targets)
    assert len(result) == 2


def test_normalize_targets_raises_on_empty_list():
    with pytest.raises(ValueError, match="non-empty list"):
        normalize_targets([])


def test_normalize_targets_raises_on_non_list():
    with pytest.raises(ValueError, match="non-empty list"):
        normalize_targets({"not": "a list"})


def test_normalize_targets_raises_on_non_dict_item():
    with pytest.raises(ValueError, match="must be a JSON object"):
        normalize_targets(["not a dict"])


def test_normalize_targets_raises_on_missing_output():
    with pytest.raises(ValueError, match="output must be a JSON object"):
        normalize_targets([{"id": "t1"}])


def test_normalize_targets_raises_on_output_not_dict():
    with pytest.raises(ValueError, match="output must be a JSON object"):
        normalize_targets([{"id": "t1", "output": "not a dict"}])


def test_normalize_targets_raises_on_empty_output():
    with pytest.raises(ValueError, match="output must contain content"):
        normalize_targets([{"id": "t1", "output": {}}])


def test_normalize_targets_id_is_optional():
    targets = [{"output": {"content": {"key": "value"}}}]
    result = normalize_targets(targets)
    assert result[0]["id"] is None


def test_normalize_targets_raises_on_non_string_id():
    with pytest.raises(ValueError, match="id must be a string"):
        normalize_targets([{"id": 123, "output": {"content": {"key": "value"}}}])


# ── normalize_tool_calls ─────────────────────────────────────────────────────


def test_normalize_tool_calls_valid():
    calls = [{"name": "search", "arguments": {"q": "test"}}]
    result = normalize_tool_calls(calls)
    assert len(result) == 1
    assert result[0]["name"] == "search"
    assert result[0]["arguments"] == {"q": "test"}


def test_normalize_tool_calls_multiple():
    calls = [
        {"name": "search", "arguments": {"q": "a"}},
        {"name": "fetch", "arguments": {"url": "b"}},
    ]
    result = normalize_tool_calls(calls)
    assert len(result) == 2


def test_normalize_tool_calls_raises_on_empty():
    with pytest.raises(ValueError, match="non-empty list"):
        normalize_tool_calls([])


def test_normalize_tool_calls_raises_on_non_list():
    with pytest.raises(ValueError, match="non-empty list"):
        normalize_tool_calls({"not": "a list"})


def test_normalize_tool_calls_raises_on_non_dict_call():
    with pytest.raises(ValueError, match="JSON object"):
        normalize_tool_calls(["not a dict"])


def test_normalize_tool_calls_raises_on_empty_name():
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        normalize_tool_calls([{"name": "", "arguments": {}}])


def test_normalize_tool_calls_raises_on_missing_name():
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        normalize_tool_calls([{"arguments": {}}])


def test_normalize_tool_calls_raises_on_non_dict_arguments():
    with pytest.raises(ValueError, match="arguments must be a JSON object"):
        normalize_tool_calls([{"name": "search", "arguments": "not a dict"}])
