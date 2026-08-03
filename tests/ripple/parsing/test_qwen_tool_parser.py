"""Qwen XML tool call 解析器的测试（CPU 零依赖）。

覆盖防呆回归（v0.14 崩溃根因）：
- 双 <tool_call> 开标记的"作弊模板"必须报 parse_error（曾静默接受）
- required 数值参数缺失必须报 parse_error（曾静默接受 → 0.886 reward hacking）
- 未闭合标签、嵌套标签、杂散文本必须报 parse_error
"""

from graspo.ripple.parsing.qwen_tool_parser import (
    parse_qwen_tool_completion,
)

# 工具 schema 样例：action_type 和 angle_deg 都是 required
TOOLS = [
    {
        "function": {
            "name": "robot_atomic_control",
            "description": "控制机器人动作",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "description": "动作类型"},
                    "angle_deg": {"type": "number", "description": "旋转角度"},
                },
                "required": ["action_type", "angle_deg"],
            },
        }
    }
]


def _call(text: str):
    return parse_qwen_tool_completion(text, expect_tool_calls=True, tools=TOOLS)


# ── 合法输出：必须通过 ──────────────────────────────────────────────────────


def test_parse_valid_xml_full_parameters():
    text = (
        "<tool_call>\n"
        "<function=robot_atomic_control>\n"
        "<parameter=action_type>\n顺时针旋转\n</parameter>\n"
        "<parameter=angle_deg>\n50\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parsed = _call(text)
    assert parsed.parse_errors == []
    assert parsed.tool_calls == [
        {
            "name": "robot_atomic_control",
            "arguments": {"action_type": "顺时针旋转", "angle_deg": 50.0},
        }
    ]
    assert parsed.parser_name == "qwen_xml_tool_call"


def test_parse_valid_xml_multiple_functions():
    text = (
        "<tool_call><function=robot_atomic_control>"
        "<parameter=action_type>伸长手臂</parameter><parameter=angle_deg>15</parameter>"
        "</function><function=robot_atomic_control>"
        "<parameter=action_type>逆时针旋转</parameter><parameter=angle_deg>30</parameter>"
        "</function></tool_call>"
    )
    parsed = _call(text)
    assert parsed.parse_errors == []
    assert len(parsed.tool_calls) == 2


def test_parse_valid_xml_number_coercion():
    text = (
        "<tool_call><function=robot_atomic_control>"
        "<parameter=action_type>旋转</parameter><parameter=angle_deg>7.4</parameter>"
        "</function></tool_call>"
    )
    parsed = _call(text)
    assert parsed.parse_errors == []
    assert parsed.tool_calls[0]["arguments"]["angle_deg"] == 7.4


# ── 防呆回归：v0.14 崩溃的作弊模板 ─────────────────────────────────────────


def test_cheat_template_double_tool_call_is_parse_error():
    """v0.14 崩溃根因：双 <tool_call> 开标记 + 缺数值参数，曾静默通过拿 0.886 分。

    该模板两个开标记配一个闭标记，正则曾把第二个开标记吞进 body 并解析成功。
    严格 XML 解析必须拦截：body 里残留未闭合的 <tool_call> 标签 → ParseError。
    """
    text = (
        "<tool_call>\n"
        "<tool_call>\n"
        "<function=robot_atomic_control>\n"
        "<parameter=action_type>\n顺时针旋转\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parsed = _call(text)
    assert len(parsed.parse_errors) >= 1


def test_missing_required_parameter_is_parse_error():
    """缺 required 数值参数（angle_deg）→ parse_error。

    v0.14 中此输出 base_content_score=1.0、all_right=True、拿 0.886 分——
    模型靠省略数值参数 hack 奖励。required 校验必须在 parser 层拦截。
    """
    text = (
        "<tool_call>\n"
        "<function=robot_atomic_control>\n"
        "<parameter=action_type>\n顺时针旋转\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parsed = _call(text)
    assert any("angle_deg" in err and "required" in err for err in parsed.parse_errors)


def test_missing_all_required_parameters_is_parse_error():
    text = "<tool_call><function=robot_atomic_control></function></tool_call>"
    parsed = _call(text)
    assert len(parsed.parse_errors) >= 2


def test_unclosed_function_tag_is_parse_error():
    text = "<tool_call><function=robot_atomic_control>"
    parsed = _call(text)
    assert len(parsed.parse_errors) >= 1


def test_unclosed_tool_call_is_parse_error():
    text = "<tool_call><function=robot_atomic_control>"
    parsed = _call(text)
    assert len(parsed.parse_errors) >= 1


def test_nested_function_is_parse_error():
    text = (
        "<tool_call><function=robot_atomic_control>"
        "<function=other><parameter=action_type>旋转</parameter></function>"
        "</function></tool_call>"
    )
    parsed = _call(text)
    assert len(parsed.parse_errors) >= 1


def test_stray_text_in_tool_call_is_parse_error():
    text = (
        "<tool_call>some random text"
        "<function=robot_atomic_control><parameter=action_type>旋转</parameter>"
        "<parameter=angle_deg>10</parameter></function>"
        "</tool_call>"
    )
    parsed = _call(text)
    assert len(parsed.parse_errors) >= 1


def test_wrong_parameter_type_is_parse_error():
    """angle_deg 是 number，传非数字 → coerce 报错。"""
    text = (
        "<tool_call><function=robot_atomic_control>"
        "<parameter=action_type>旋转</parameter><parameter=angle_deg>abc</parameter>"
        "</function></tool_call>"
    )
    parsed = _call(text)
    assert any("angle_deg" in err and "expected number" in err for err in parsed.parse_errors)


# ── 兼容行为：JSON 兜底 / unwrapped / think ────────────────────────────────


def test_json_tool_call_still_parsed():
    text = '<tool_call>{"name": "search", "arguments": {"q": "hello"}}</tool_call>'
    parsed = parse_qwen_tool_completion(text, expect_tool_calls=True)
    assert parsed.tool_calls == [{"name": "search", "arguments": {"q": "hello"}}]
    assert parsed.parse_errors == []


def test_unwrapped_xml_function():
    text = "<function=search><parameter=q>hello</parameter></function>"
    parsed = parse_qwen_tool_completion(text)
    assert parsed.parser_name in ("qwen_xml_tool_call", "qwen_xml_tool_call_unwrapped")


def test_think_tag_still_extracted():
    text = "<think>Let me search for that.</think><tool_call>...</tool_call>"
    parsed = parse_qwen_tool_completion(text)
    assert parsed.think_text == "Let me search for that."
