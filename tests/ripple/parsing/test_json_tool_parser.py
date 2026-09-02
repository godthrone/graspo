"""通用 tool call 解析的测试（CPU 零依赖）。

Qwen XML 格式的测试在 ``tests/backends/native/models/common/
test_qwen_tool_parser.py``（Qwen 家族共享层解析器）。
"""

from graspo.ripple.parsing.json_tool_parser import canonical_tool_call, try_parse_json_tool_call

# ── try_parse_json_tool_call ────────────────────────────────────────────────


def test_parse_json_tool_call_single():
    text = '{"name": "search", "arguments": {"q": "hello"}}'
    parsed = try_parse_json_tool_call(text)
    assert parsed == [{"name": "search", "arguments": {"q": "hello"}}]


def test_parse_json_tool_call_list():
    text = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
    parsed = try_parse_json_tool_call(text)
    assert len(parsed) == 2


def test_parse_invalid_json_returns_none():
    assert try_parse_json_tool_call("{invalid json}") is None


def test_parse_non_object_json_returns_none():
    assert try_parse_json_tool_call("42") is None
    assert try_parse_json_tool_call('"str"') is None
    assert try_parse_json_tool_call("null") is None


def test_parse_json_without_arguments_returns_none():
    assert try_parse_json_tool_call('{"name": "search"}') is None


# ── canonical_tool_call ─────────────────────────────────────────────────────


def test_canonical_tool_call_normalizes():
    assert canonical_tool_call({"name": "s", "arguments": {"q": 1}}) == {
        "name": "s",
        "arguments": {"q": 1},
    }


def test_canonical_tool_call_rejects_bad_shapes():
    assert canonical_tool_call("not a dict") is None
    assert canonical_tool_call({"name": 42, "arguments": {}}) is None
    assert canonical_tool_call({"name": "s"}) is None
    assert canonical_tool_call({"arguments": {}}) is None
