"""ripple/parsing/classification.py 的单元测试：任务分类与 tool-call 计数判定。"""

import pytest

from graspo.ripple.parsing.classification import (
    is_pure_tool_call_task,
    tool_call_count_mismatch_count,
)


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ([{"id": "a", "output": {"tool_calls": [{"name": "x", "arguments": {}}]}}], True),
        ([{"id": "a", "output": {"content": {"k": "v"}}}], False),
        ([{"id": "a", "output": {"content": {"k": "v"}, "tool_calls": [{"name": "x"}]}}], False),
        ([], False),
        (None, False),
    ],
)
def test_is_pure_tool_call_task(targets, expected):
    assert is_pure_tool_call_task(targets) is expected


@pytest.mark.parametrize(
    ("details", "targets", "expected"),
    [
        # 与 targets 数量一致 → 无不匹配
        (
            [{"parsed_tool_calls": [{"name": "a"}]}, {"parsed_tool_calls": [{"name": "b"}]}],
            [{"id": "t", "output": {"tool_calls": [{"name": "a"}]}}],
            0,
        ),
        # 数量不一致（2 个 vs 1 个）→ 1 条不匹配
        (
            [{"parsed_tool_calls": [{"name": "a"}, {"name": "b"}]}],
            [{"id": "t", "output": {"tool_calls": [{"name": "a"}]}}],
            1,
        ),
        # 解析失败的 completion（parsed_tool_calls 为 None）不计入
        (
            [{"parsed_tool_calls": None}, {"parsed_tool_calls": [{"name": "a"}]}],
            [{"id": "t", "output": {"tool_calls": [{"name": "a"}]}}],
            0,
        ),
        # 多 target 多候选：任一数量匹配即合规
        (
            [{"parsed_tool_calls": [{"name": "a"}, {"name": "b"}]}],
            [
                {"id": "one", "output": {"tool_calls": [{"name": "a"}]}},
                {"id": "two", "output": {"tool_calls": [{"name": "a"}, {"name": "b"}]}},
            ],
            0,
        ),
        # targets 非列表（异常输入）→ 视为期望 1 个调用
        (
            [{"parsed_tool_calls": [{"name": "a"}, {"name": "b"}]}],
            None,
            1,
        ),
    ],
)
def test_tool_call_count_mismatch_count(details, targets, expected):
    assert tool_call_count_mismatch_count(details, targets) == expected
