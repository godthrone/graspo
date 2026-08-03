"""通用 tool call 解析：跨模型标准 JSON 格式（<tool_call>{...}</tool_call>）。

职责边界（宪法 §1.1）：
- 本模块解析 **跨模型通用** 的 JSON tool call 格式（OpenAI 风格
  ``{"name": ..., "arguments": {...}}``），任何模型族都可使用。
- Qwen 特有的 XML tool call 格式（``<function=NAME>`` 语法）在
  ``models/common/qwen_tool_parser.py``（Qwen 家族共享层），本模块不包含
  XML 解析——v0.14 崩溃根因是正则解析 XML 被作弊模板（双 ``<tool_call>``
  开标记）绕过，XML 路径已改为规范化 + ElementTree 严格解析，见该模块。
"""

from __future__ import annotations

import json
from typing import Any


def try_parse_json_tool_call(text: str) -> list[dict[str, Any]] | None:
    """尝试把文本解析为 JSON tool call（列表或单对象）。

    :param text: ``<tool_call>`` 的 body 内容
    :return: 解析成功返回规范化 tool call 列表；不是合法 JSON 结构返回 None
    """
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    values = value if isinstance(value, list) else [value]
    if not isinstance(values, list):
        return None
    calls: list[dict[str, Any]] = []
    for item in values:
        call = canonical_tool_call(item)
        if call is None:
            return None
        calls.append(call)
    return calls


def canonical_tool_call(value: Any) -> dict[str, Any] | None:
    """规范化单个 JSON tool call：校验 name/arguments 结构。"""
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": dict(arguments)}


__all__ = ["canonical_tool_call", "try_parse_json_tool_call"]
