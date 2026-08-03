"""Qwen 家族共享的 tool call 解析器：XML 规范化 + 严格解析 + required 校验。

职责边界（宪法 §1.1）：
- 本模块解析 **Qwen 特有的 XML tool call 格式**（``<function=NAME>`` 语法），
  这是 qwen3 / qwen3.5 / qwen3.6 家族共享的输出格式，其他模型族（如
  JSON 系模型）不依赖本模块。
- 通用 JSON tool call 解析（``<tool_call>{...}</tool_call>``）属于跨模型
  标准格式，保留在 ``backends/graspoflow/tool_parser.py``（通用层），
  本模块从中 import。

防呆设计（v0.14 训练崩溃根因修复，§2.3 防线）：
- 旧实现用正则 ``<tool_call>(.*?)</tool_call>`` 提取 body 后直接正则提取
  ``<function=`` 标签——双 ``<tool_call>`` 开标记（一个闭标记）的"作弊
  模板"会被静默吞掉多余开标记并解析成功，配合 required 参数缺失零惩罚，
  模型学会省略数值参数拿 0.886 分（reward hacking）。
- 新实现先把 ``<function=X>``/``<parameter=X>`` 规范化为合法 XML
  （``<function name="X">``），再用 ``xml.etree.ElementTree`` **严格解析**：
  未配对标签、嵌套标签、杂散文本、非法字符都会抛 ParseError →
  parse_error。随后对照 tools schema 校验 required 参数，缺失 → parse_error。
- 有 parse_error 的组由 ``classify_group`` 的 ``best_completion_has_parse_error``
  拦截为 RETRY/INVALID，从源头阻止脏格式进入训练。

依赖：仅 Python 标准库（``xml.etree.ElementTree``），零第三方依赖。
"""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree as ET

from graspo.backends.graspoflow.tool_parser import try_parse_json_tool_call
from graspo.core.completion import ParsedCompletion

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)

# Qwen 类 XML 语法：<function=NAME> / <parameter=NAME> 不是合法 XML 元素名
# （元素名不允许 "="）。规范化前先匹配这种带 "=" 的语法。
_FUNCTION_EQ_RE = re.compile(r"<function=([^>\s]+)>")
_PARAMETER_EQ_RE = re.compile(r"<parameter=([^>\s]+)>")

# 规范化后的合法 XML 结构：
#   <function name="NAME"><parameter name="NAME">value</parameter></function>
_NORMALIZED_FUNCTION_TAG = '<function name="{name}">'
_NORMALIZED_PARAMETER_TAG = '<parameter name="{name}">'


def _normalize_qwen_xml(text: str) -> str:
    """把 Qwen 类 XML 语法规范化为合法 XML（<function=X> → <function name="X">）。"""
    text = _FUNCTION_EQ_RE.sub(lambda m: _NORMALIZED_FUNCTION_TAG.format(name=m.group(1)), text)
    text = _PARAMETER_EQ_RE.sub(
        lambda m: _NORMALIZED_PARAMETER_TAG.format(name=m.group(1)), text
    )
    return text


def _required_parameters(
    tools: list[dict[str, Any]] | None,
    tool_name: str,
) -> list[str]:
    """查 tools schema 中指定 tool 的 required 参数列表（无 schema/无 required → []）。"""
    if not tools:
        return []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return []
        required = parameters.get("required")
        if not isinstance(required, list):
            return []
        return [str(item) for item in required if isinstance(item, str)]
    return []


def _coerce_tool_argument(
    tool_name: str,
    param_name: str,
    raw_value: str,
    *,
    tools: list[dict[str, Any]] | None,
) -> tuple[Any, str | None]:
    """按 tools schema 的类型信息把参数值转为 int/float/bool（与旧实现等价）。"""
    schema_type: str | None = None
    if tools:
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or function.get("name") != tool_name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                break
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                break
            spec = properties.get(param_name)
            if not isinstance(spec, dict):
                break
            raw_type = spec.get("type")
            if isinstance(raw_type, str):
                schema_type = raw_type
            elif isinstance(raw_type, list):
                for item in raw_type:
                    if isinstance(item, str) and item != "null":
                        schema_type = item
                        break
            break
    if schema_type == "integer":
        if re.fullmatch(r"[+-]?\d+", raw_value):
            return int(raw_value), None
        return raw_value, "expected integer"
    if schema_type == "number":
        try:
            value = float(raw_value)
        except ValueError:
            return raw_value, "expected number"
        if value != value or value in (float("inf"), float("-inf")):
            return raw_value, "expected finite number"
        return value, None
    if schema_type == "boolean":
        lowered = raw_value.lower()
        if lowered == "true":
            return True, None
        if lowered == "false":
            return False, None
        return raw_value, "expected boolean"
    return raw_value, None


def try_parse_qwen_xml_tool_call(
    text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    error_prefix: str = "tool_call",
) -> tuple[list[dict[str, Any]], list[str]]:
    """严格解析 Qwen XML tool call：规范化 → ElementTree 校验 → 提取参数。

    :param text: 单个 ``<tool_call>`` 的 body（不含外层标签）
    :param tools: 工具 schema（用于 required 校验和类型强制转换）
    :param error_prefix: 错误消息前缀（定位到第几个 tool_call）
    :return: ``(calls, parse_errors)``。格式不合法时 calls 可能为空，
        parse_errors 必非空——由调用方决定重试或丢弃。
    """
    calls: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    normalized = _normalize_qwen_xml(text).strip()
    if not normalized:
        parse_errors.append(f"{error_prefix}: empty tool call body")
        return calls, parse_errors
    try:
        # 包一层 <root> 容纳多个 <function> 元素；单个 function 也合法
        root = ET.fromstring(f"<root>{normalized}</root>")
    except ET.ParseError as exc:
        # 未配对标签、嵌套标签、非法字符 → 全部拦截。
        # 这是 v0.14 作弊模板（双 <tool_call> 开标记）的死法：规范化后
        # body 里残留未闭合的 <tool_call> 标签，ElementTree 直接拒绝。
        parse_errors.append(f"{error_prefix}: malformed XML ({exc})")
        return calls, parse_errors

    # 杂散文本检查：root 直接子文本必须是空白（model 输出夹带说明文字即拒绝）
    if root.text and root.text.strip():
        parse_errors.append(
            f"{error_prefix}: unexpected text content before <function>"
        )
    if root.tail and root.tail.strip():
        parse_errors.append(f"{error_prefix}: unexpected text content after <function>")

    for function_el in root:
        if function_el.tag != "function":
            parse_errors.append(
                f"{error_prefix}: unexpected element <{function_el.tag}> inside <tool_call>"
            )
            continue
        name = function_el.get("name")
        if not name:
            parse_errors.append(f"{error_prefix}: <function> missing name attribute")
            continue
        arguments: dict[str, Any] = {}
        for param_el in function_el:
            if param_el.tag != "parameter":
                parse_errors.append(
                    f"{error_prefix}.{name}: unexpected element <{param_el.tag}> "
                    "inside <function>"
                )
                continue
            param_name = param_el.get("name")
            if not param_name:
                parse_errors.append(f"{error_prefix}.{name}: <parameter> missing name attribute")
                continue
            raw_value = (param_el.text or "").strip()
            value, error = _coerce_tool_argument(name, param_name, raw_value, tools=tools)
            arguments[param_name] = value
            if error:
                parse_errors.append(f"{error_prefix}.arguments.{param_name} {error}")
        for required in _required_parameters(tools, name):
            if required not in arguments:
                parse_errors.append(
                    f"{error_prefix}.arguments.{required}: missing required parameter"
                )
        if name and arguments:
            calls.append({"name": name, "arguments": arguments})
    return calls, parse_errors


def parse_qwen_tool_completion(
    text: str,
    *,
    expect_tool_calls: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> ParsedCompletion:
    """解析 Qwen 系模型的完整输出：think 提取 + JSON/XML tool call 双路径。

    先试跨模型标准 JSON 格式（``<tool_call>{...}</tool_call>``），失败后走
    Qwen 特有 XML 格式的严格解析（``try_parse_qwen_xml_tool_call``）。
    XML 格式不合法（作弊模板、未配对标签、缺 required 参数）时，
    parse_errors 非空，由上层 ``classify_group`` 拦为 RETRY/INVALID。
    """
    think_parts = [match.group(1).strip() for match in _THINK_RE.finditer(text)]
    tool_calls: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    parser_names: list[str] = []
    for idx, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        body = match.group(1).strip()
        parsed_json = try_parse_json_tool_call(body)
        if parsed_json is not None:
            tool_calls.extend(parsed_json)
            parser_names.append("qwen_json_tool_call")
            continue
        parsed_xml, xml_errors = try_parse_qwen_xml_tool_call(
            body,
            tools=tools,
            error_prefix=f"tool_call[{idx}]",
        )
        if parsed_xml:
            tool_calls.extend(parsed_xml)
            parse_errors.extend(xml_errors)
            parser_names.append("qwen_xml_tool_call")
            continue
        parse_errors.extend(xml_errors)
        if not xml_errors:
            parse_errors.append(f"tool_call[{idx}] is neither canonical JSON nor Qwen XML")
    if not tool_calls and "<function=" in text:
        parsed_xml, xml_errors = try_parse_qwen_xml_tool_call(
            _THINK_RE.sub("", text),
            tools=tools,
            error_prefix="tool_call",
        )
        if parsed_xml:
            tool_calls.extend(parsed_xml)
            parse_errors.extend(xml_errors)
            parser_names.append("qwen_xml_tool_call_unwrapped")
    if expect_tool_calls and not tool_calls:
        parse_errors.append("no tool call found")
    extra_text = _FUNCTION_RE.sub("", _TOOL_CALL_RE.sub("", _THINK_RE.sub("", text))).strip()
    parser_name = "+".join(sorted(set(parser_names))) if parser_names else "qwen_tool_call"
    return ParsedCompletion(
        raw_text=text,
        think_text="\n\n".join(part for part in think_parts if part),
        tool_calls=tool_calls,
        answer_text=extra_text or text,
        parser_name=parser_name,
        parse_errors=parse_errors,
        extra_text=extra_text,
    )


__all__ = ["parse_qwen_tool_completion", "try_parse_qwen_xml_tool_call"]
