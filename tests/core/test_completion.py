"""Tests for ParsedCompletion — BADGE §11.1."""

import pytest

from graspo.core.completion import ParsedCompletion, raw_parsed_completion

# ── raw_parsed_completion ───────────────────────────────────────────────────


def test_raw_parsed_completion_stores_raw_text():
    pc = raw_parsed_completion("hello world")
    assert pc.raw_text == "hello world"
    assert pc.answer_text == "hello world"


def test_raw_parsed_completion_uses_raw_parser_by_default():
    pc = raw_parsed_completion("test")
    assert pc.parser_name == "raw"


def test_raw_parsed_completion_custom_parser_name():
    pc = raw_parsed_completion("test", parser_name="custom")
    assert pc.parser_name == "custom"


def test_raw_parsed_completion_empty_extra_text():
    pc = raw_parsed_completion("hello")
    assert pc.extra_text == ""


def test_raw_parsed_completion_empty_tool_calls():
    pc = raw_parsed_completion("hello")
    assert pc.tool_calls == []


def test_raw_parsed_completion_empty_think_text():
    pc = raw_parsed_completion("hello")
    assert pc.think_text == ""


def test_raw_parsed_completion_no_parse_errors():
    pc = raw_parsed_completion("hello")
    assert pc.parse_errors == []


# ── ParsedCompletion dataclass ───────────────────────────────────────────────


def test_parsed_completion_defaults():
    pc = ParsedCompletion(raw_text="test")
    assert pc.raw_text == "test"
    assert pc.think_text == ""
    assert pc.tool_calls == []
    assert pc.answer_text == ""
    assert pc.parser_name == "raw"
    assert pc.parse_errors == []
    assert pc.extra_text == ""


def test_parsed_completion_full_fields():
    pc = ParsedCompletion(
        raw_text="<think>hmm</think>```json\n{}\n```",
        think_text="hmm",
        tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
        answer_text="{}",
        parser_name="qwen",
        parse_errors=["missing_closing_fence"],
        extra_text="extra stuff",
    )
    assert pc.think_text == "hmm"
    assert pc.tool_calls == [{"name": "search", "arguments": {"q": "test"}}]
    assert pc.answer_text == "{}"
    assert pc.parser_name == "qwen"
    assert pc.parse_errors == ["missing_closing_fence"]
    assert pc.extra_text == "extra stuff"


def test_parsed_completion_to_dict_includes_all_fields():
    pc = ParsedCompletion(
        raw_text="x",
        think_text="t",
        answer_text="a",
        extra_text="e",
    )
    d = pc.to_dict()
    assert d["raw_text"] == "x"
    assert d["think_text"] == "t"
    assert d["answer_text"] == "a"
    assert d["extra_text"] == "e"
    assert "tool_calls" in d
    assert "parser_name" in d


def test_parsed_completion_slots_prevents_new_attributes():
    """ParsedCompletion uses slots=True — cannot add arbitrary attributes."""
    pc = ParsedCompletion(raw_text="test")
    with pytest.raises(AttributeError):
        pc.nonexistent_field = "value"


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_raw_parsed_completion_empty_string():
    pc = raw_parsed_completion("")
    assert pc.raw_text == ""
    assert pc.answer_text == ""


def test_raw_parsed_completion_unicode():
    pc = raw_parsed_completion("你好世界 🌍")
    assert pc.raw_text == "你好世界 🌍"


def test_parsed_completion_with_parse_error_and_tool_calls():
    pc = ParsedCompletion(
        raw_text="some text",
        tool_calls=[{"name": "func", "arguments": {}}],
        parse_errors=["missing_think_tag"],
    )
    assert len(pc.tool_calls) == 1
    assert len(pc.parse_errors) == 1
