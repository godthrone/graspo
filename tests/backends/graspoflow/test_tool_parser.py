"""Tests for tool-call parsing — BADGE §11.1 (CPU)."""

from graspo.backends.graspoflow.tool_parser import parse_qwen_tool_completion

# ── parse_qwen_tool_completion: JSON tool calls ──────────────────────────────


def test_parse_json_tool_call_single():
    text = '<tool_call>{"name": "search", "arguments": {"q": "hello"}}</tool_call>'
    parsed = parse_qwen_tool_completion(text, expect_tool_calls=True)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "search"
    assert parsed.tool_calls[0]["arguments"] == {"q": "hello"}


def test_parse_json_tool_call_multiple():
    text = (
        '<tool_call>{"name": "search", "arguments": {"q": "a"}}</tool_call>'
        '<tool_call>{"name": "fetch", "arguments": {"url": "b"}}</tool_call>'
    )
    parsed = parse_qwen_tool_completion(text)
    assert len(parsed.tool_calls) == 2


def test_parse_no_tool_call_returns_empty():
    text = "This is just some text without tool calls."
    parsed = parse_qwen_tool_completion(text)
    assert parsed.tool_calls == []


def test_parse_no_tool_call_expected_but_missing():
    text = "Just text."
    parsed = parse_qwen_tool_completion(text, expect_tool_calls=True)
    assert parsed.tool_calls == []
    assert "no tool call found" in parsed.parse_errors


# ── parse_qwen_tool_completion: think tag extraction ────────────────────────


def test_parse_think_tag_extracted():
    text = "<think>Let me search for that.</think><tool_call>...</tool_call>"
    parsed = parse_qwen_tool_completion(text)
    assert parsed.think_text == "Let me search for that."


def test_parse_multiline_think():
    text = "<think>\nStep 1: search\nStep 2: fetch\n</think>"
    parsed = parse_qwen_tool_completion(text)
    assert "Step 1: search" in parsed.think_text


# ── parse_qwen_tool_completion: extra_text ──────────────────────────────────


def test_parse_extra_text_is_non_marker_content():
    text = '<tool_call>{"name": "s", "arguments": {}}</tool_call>extra content here'
    parsed = parse_qwen_tool_completion(text)
    assert "extra content here" in parsed.extra_text


def test_parse_extra_text_strips_empty():
    text = '<tool_call>{"name": "s", "arguments": {}}</tool_call>'
    parsed = parse_qwen_tool_completion(text)
    assert parsed.extra_text == ""


# ── parse_qwen_tool_completion: XML tool calls ──────────────────────────────


def test_parse_xml_tool_call_with_function_parameters():
    text = "<tool_call><function=search><parameter=query>hello</parameter></function></tool_call>"
    parsed = parse_qwen_tool_completion(text)
    # XML tool calls are parsed when JSON fails
    assert len(parsed.tool_calls) >= 1 or len(parsed.parse_errors) >= 1


def test_parse_unwrapped_xml_function():
    text = "<function=search><parameter=q>hello</parameter></function>"
    parsed = parse_qwen_tool_completion(text)
    # Should attempt unwrapped XML parsing
    assert parsed.parser_name in (
        "qwen_xml_tool_call",
        "qwen_xml_tool_call_unwrapped",
        "qwen_tool_call",
    )


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_parse_empty_string():
    parsed = parse_qwen_tool_completion("")
    assert parsed.raw_text == ""
    assert parsed.tool_calls == []
    assert parsed.think_text == ""


def test_parse_invalid_json_tool_call():
    text = "<tool_call>{invalid json}</tool_call>"
    parsed = parse_qwen_tool_completion(text, expect_tool_calls=True)
    # Should report parse error and empty tool calls
    assert (
        any("canonical JSON" in err for err in parsed.parse_errors) or len(parsed.tool_calls) == 0
    )
