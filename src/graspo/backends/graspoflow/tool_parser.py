from __future__ import annotations

import json
import math
import re
from typing import Any

from graspo.core.completion import ParsedCompletion

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def parse_qwen_tool_completion(
    text: str,
    *,
    expect_tool_calls: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> ParsedCompletion:
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


def try_parse_json_tool_call(text: str) -> list[dict[str, Any]] | None:
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


def try_parse_qwen_xml_tool_call(
    text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    error_prefix: str = "tool_call",
) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for match in _FUNCTION_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in _PARAMETER_RE.finditer(body):
            param_name = param_match.group(1).strip()
            raw_value = param_match.group(2).strip()
            value, error = coerce_xml_tool_argument(
                name,
                param_name,
                raw_value,
                tools=tools,
            )
            arguments[param_name] = value
            if error:
                parse_errors.append(f"{error_prefix}.arguments.{param_name} {error}")
        if name and arguments:
            calls.append({"name": name, "arguments": arguments})
    return calls, parse_errors


def coerce_xml_tool_argument(
    tool_name: str,
    param_name: str,
    raw_value: str,
    *,
    tools: list[dict[str, Any]] | None,
) -> tuple[Any, str | None]:
    schema_type = tool_argument_schema_type(tools, tool_name, param_name)
    if schema_type == "integer":
        if re.fullmatch(r"[+-]?\d+", raw_value):
            return int(raw_value), None
        return raw_value, "expected integer"
    if schema_type == "number":
        try:
            value = float(raw_value)
        except ValueError:
            return raw_value, "expected number"
        if not math.isfinite(value):
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


def tool_argument_schema_type(
    tools: list[dict[str, Any]] | None,
    tool_name: str,
    param_name: str,
) -> str | None:
    if not tools:
        return None
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return None
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return None
        spec = properties.get(param_name)
        if not isinstance(spec, dict):
            return None
        raw_type = spec.get("type")
        if isinstance(raw_type, str):
            return raw_type
        if isinstance(raw_type, list):
            for item in raw_type:
                if isinstance(item, str) and item != "null":
                    return item
    return None


def canonical_tool_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": dict(arguments)}
