"""ripple/parsing/xml.py 的单元测试：SFT target 文本构建与 XML 工具调用格式化。"""

import pytest

from graspo.ripple.parsing.xml import build_sft_target_text, tool_calls_to_xml


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        (
            [{"name": "robot_atomic_control", "arguments": {"action": "左转", "distance_cm": 5}}],
            '<tool_call>\n<function=robot_atomic_control>\n<parameter=action>\n左转\n</parameter>\n'
            '<parameter=distance_cm>\n5\n</parameter>\n</function>\n</tool_call>',
        ),
        (
            [
                {"name": "move", "arguments": {"action": "left"}},
                {"name": "inspect", "arguments": {"object": "panel"}},
            ],
            '<tool_call>\n<function=move>\n<parameter=action>\nleft\n</parameter>\n</function>\n'
            '</tool_call>\n<tool_call>\n<function=inspect>\n<parameter=object>\npanel\n</parameter>\n'
            "</function>\n</tool_call>",
        ),
    ],
)
def test_tool_calls_to_xml_matches_native_format(tool_calls, expected):
    assert tool_calls_to_xml(tool_calls) == expected


def test_tool_calls_to_xml_parameter_value_own_line():
    """参数值占据独立行——与 Qwen base model 预训练格式字符级一致。"""
    text = tool_calls_to_xml(
        [{"name": "robot_atomic_control", "arguments": {"action": "逆时针旋转"}}]
    )
    assert "<parameter=action>\n逆时针旋转\n</parameter>" in text
    assert "<parameter=action>逆时针旋转</parameter>" not in text


def test_build_sft_target_text_content_uses_fenced_json():
    output = {"content": {"status": "ok", "count": 3}}
    text = build_sft_target_text(output)
    assert text.startswith("```json\n")
    assert text.endswith("\n```")
    assert '"status": "ok"' in text


def test_build_sft_target_text_tool_calls_uses_xml():
    output = {"tool_calls": [{"name": "query", "arguments": {"id": "T-1"}}]}
    text = build_sft_target_text(output)
    assert text.startswith("<tool_call>")
    assert "<function=query>" in text


def test_build_sft_target_text_does_not_use_chat_template():
    """架构不变式：target text 是裸 XML/JSON，不经 apply_chat_template。"""
    output = {"tool_calls": [{"name": "query", "arguments": {"id": "T-1"}}]}
    text = build_sft_target_text(output)
    assert "<|im_start|>" not in text
