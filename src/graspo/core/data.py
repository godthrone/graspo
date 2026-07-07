import json
import re
from pathlib import Path
from typing import Any

from graspo.core.reward_helpers import normalize_targets
from graspo.core.schema import Sample

# Matches raw Qwen XML / tool-call markers that should not appear in content.
_TOOL_CALL_MARKER_RE = re.compile(r"<(?:tool_call|function=|parameter=)")


def sample_from_record(record: dict[str, Any]) -> Sample:
    removed_input_fields = {"prompt", "image", "images", "video", "videos"}
    present_removed = sorted(field for field in removed_input_fields if field in record)
    if present_removed:
        raise ValueError(
            "removed input field(s): "
            + ", ".join(present_removed)
            + "; use messages + optional tools + targets JSONL"
        )
    if "ground_truth" in record:
        raise ValueError("record field 'ground_truth' was removed; use targets[].output")
    messages = _validate_messages(record.get("messages"))

    if "targets" not in record:
        raise ValueError("record must contain 'targets'")
    tools = _validate_tools(record.get("tools"))
    targets = normalize_targets(record["targets"])
    if tools is not None:
        _validate_tool_call_targets(targets, tools)
    media = _messages_media(messages)
    metadata = {
        key: value for key, value in record.items() if key not in {"messages", "targets", "tools"}
    }
    return Sample(
        messages=messages,
        targets=targets,
        tools=tools,
        metadata=metadata,
        media=media,
    )


def _validate_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("record must contain a non-empty 'messages' list")
    messages: list[dict[str, Any]] = []
    for idx, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        role = str(message.get("role") or "").strip()
        if not role:
            raise ValueError(f"messages[{idx}].role is required")
        if "content" not in message:
            raise ValueError(f"messages[{idx}].content is required")
        messages.append(dict(message))
    if str(messages[-1].get("role") or "").lower() == "assistant":
        raise ValueError(
            "messages must be prompt/context only; final assistant messages leak the target"
        )
    _validate_assistant_tool_calls(messages)
    return messages


def _validate_assistant_tool_calls(messages: list[dict[str, Any]]) -> None:
    """Validate assistant messages use structured tool_calls, not raw text in content.

    Raises ValueError if any assistant message embeds raw tool-call markers
    (``<tool_call>``, ``<function=``) in its content field, or if a
    ``tool_calls`` field does not conform to canonical JSON format.
    """
    for idx, message in enumerate(messages):
        if str(message.get("role") or "").lower() != "assistant":
            continue

        # ---------- 1. raw tool-call markers in content ----------
        content = message.get("content", "")
        content_texts: list[str] = []

        if isinstance(content, str):
            content_texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and str(block.get("type") or "").lower() == "text":
                    content_texts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    content_texts.append(block)

        for text in content_texts:
            if _TOOL_CALL_MARKER_RE.search(text):
                snippet = text.strip()[:200]
                raise ValueError(
                    f"messages[{idx}] (assistant) has raw tool-call text in content. "
                    f"Use structured 'tool_calls' field instead:\n"
                    f'  {{"tool_calls": [{{"name": "...", "arguments": {{...}}}}]}}\n'
                    f"  Found in content: {snippet!r}"
                )

        # ---------- 2. validate tool_calls field format ----------
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            _require_canonical_tool_calls(tool_calls, path=f"messages[{idx}].tool_calls")


def _require_canonical_tool_calls(value: Any, *, path: str = "tool_calls") -> None:
    """Validate *value* is a canonical tool-call list.  Raises ``ValueError``.

    Canonical form::

        [{"name": "<non-empty-str>", "arguments": {<dict>}}, ...]

    ``arguments`` values must not contain raw tool-call markers
    (avoids XML smuggled inside JSON string values).
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")

    for idx, call in enumerate(value):
        if not isinstance(call, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object")

        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}[{idx}].name must be a non-empty string")

        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{path}[{idx}].arguments must be a JSON object")

        # Reject raw tool-call markers smuggled inside argument values.
        for arg_key, arg_val in arguments.items():
            if isinstance(arg_val, str) and _TOOL_CALL_MARKER_RE.search(arg_val):
                raise ValueError(
                    f"{path}[{idx}].arguments.{arg_key} contains raw tool-call markers "
                    f"in its string value; use structured JSON values instead: "
                    f"{arg_val!r}"
                )


def _validate_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("record 'tools' must be a list of JSON objects")
    tools: list[dict[str, Any]] = []
    for idx, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{idx}] must be an object")
        tools.append(dict(tool))
    return tools


def _validate_tool_call_targets(targets: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
    tool_names = {
        str(function.get("name"))
        for tool in tools
        if isinstance((function := tool.get("function")), dict) and function.get("name")
    }
    for target_index, target in enumerate(targets):
        calls = target["output"].get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            name = str(call["name"])
            if tool_names and name not in tool_names:
                raise ValueError(
                    f"targets[{target_index}].output.tool_calls name {name!r}"
                    f" is not declared in tools"
                )
            declaration = _tool_declaration_by_name(tools, name)
            if declaration is not None:
                _validate_tool_arguments_against_declaration(
                    call["arguments"], declaration, target_index=target_index
                )


def _tool_declaration_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return function
    return None


def _validate_tool_arguments_against_declaration(
    arguments: dict[str, Any], declaration: dict[str, Any], *, target_index: int
) -> None:
    parameters = declaration.get("parameters")
    if not isinstance(parameters, dict):
        return
    required = parameters.get("required")
    if isinstance(required, list):
        missing = [str(key) for key in required if key not in arguments]
        if missing:
            raise ValueError(
                f"targets[{target_index}].output.tool_calls missing required argument(s): "
                + ", ".join(missing)
            )
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            raise ValueError(
                f"targets[{target_index}].output.tool_calls argument {key!r} "
                f"value {value!r} is not in enum"
            )


def _messages_media(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for message in messages:
        _, content_media = _content_to_text_and_media(message.get("content", ""))
        media.extend(content_media)
    return _dedupe_media(media)


def _content_to_text_and_media(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content or ""), []
    parts: list[str] = []
    media: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type == "text":
            parts.append(str(item.get("text") or ""))
        elif item_type in {"image", "image_url"}:
            path = item.get("image") or item.get("path") or item.get("url")
            if isinstance(item.get("image_url"), dict):
                path = item["image_url"].get("url") or path
            if path:
                media.append({"type": "image", "path": str(path)})
            parts.append("<image>")
        elif item_type in {"video", "video_url"}:
            path = item.get("video") or item.get("path") or item.get("url")
            if isinstance(item.get("video_url"), dict):
                path = item["video_url"].get("url") or path
            if path:
                media.append({"type": "video", "path": str(path)})
            parts.append("<video>")
        else:
            raise ValueError(f"unsupported multimodal content type: {item_type!r}")
    return "\n".join(part for part in parts if part), media


def _multimodal_row_from_sample(sample: Any) -> dict[str, Any]:
    """从 Sample 构建多模态行数据（纯数据转换，无设施依赖）。"""
    row: dict[str, Any] = {
        "messages": [dict(message) for message in sample.messages],
        "media": _media_counts(sample.media or []),
    }
    tools = getattr(sample, "tools", None)
    if tools is not None:
        row["tools"] = [dict(tool) for tool in tools]
    return row


def _media_counts(media: list[dict[str, Any]]) -> dict[str, int]:
    """统计多模态媒体类型计数（纯数据转换）。"""
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown") if isinstance(item, dict) else "unknown"
        counts[media_type] = counts.get(media_type, 0) + 1
    return counts


def _dedupe_media(media: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in media:
        media_type = str(item.get("type") or "")
        path = str(item.get("path") or "")
        key = (media_type, path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def load_jsonl(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                samples.append(sample_from_record(record))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid JSONL record at {path}:{line_no}: {exc}") from exc
    return samples


def build_sft_target_text(target_output: dict[str, Any]) -> str:
    """将 target output 转换为模型应该生成的文本。

    - tool_calls: 转为 Qwen XML 格式
      ``<tool_call><function=name><parameter=key>value</parameter></function></tool_call>``
    - content: 转为 markdown-fenced JSON
    """
    tool_calls = target_output.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return _tool_calls_to_xml(tool_calls)
    content = target_output.get("content")
    if isinstance(content, dict):
        import json as _json

        return "```json\n" + _json.dumps(content, ensure_ascii=False, indent=2) + "\n```"
    return ""


def _tool_calls_to_xml(tool_calls: list[dict[str, Any]]) -> str:
    """将 canonical tool_calls 转为 Qwen XML 格式。"""
    parts: list[str] = []
    for call in tool_calls:
        name = call["name"]
        arguments = call["arguments"]
        params = "".join(
            f"<parameter={key}>{_format_xml_param_value(value)}</parameter>"
            for key, value in arguments.items()
        )
        parts.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
    return "\n".join(parts)


def _format_xml_param_value(value: Any) -> str:
    """格式化参数值为 XML 文本。"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value == int(value) and not (value != value):  # 非 NaN 的整数值
            return str(int(value))
        return str(value)
    return str(value)


def write_jsonl(samples: list[Sample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")


def sft_tokenize(
    sample: Sample,
    tokenizer: Any,
    *,
    max_seq_length: int = 4096,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将一条 GRASPO JSONL 样本转换为 SFT 训练所需的 token 序列。

    关键对齐逻辑：
    - 用 ``apply_chat_template(enable_thinking=False, add_generation_prompt=True)``
      生成 prompt，与 GRASPO 推理时的 prompt 前缀**完全一致**
    - 从 primary target 提取 ground truth 作为 response
    - labels mask 掉 prompt 部分，只对 response 部分计算 loss

    注意：多模态样本（含 image/video）当前仅支持纯文本 tokenization；
    视觉特征通过 processor 单独编码，需在 trainer 层处理。
    """
    import torch

    # 1. 提取 primary target 并构建 response 文本
    primary = sample.targets[0]
    target_text = build_sft_target_text(primary["output"])
    if not target_text:
        raise ValueError("primary target has no content or tool_calls")

    # 2. 生成 prompt（与 GRASPO 推理完全一致）
    template_kwargs = dict(chat_template_kwargs or {})
    template_kwargs.setdefault("enable_thinking", False)
    tools = sample.tools if sample.expects_tool_calls else None
    if tools is not None:
        template_kwargs["tools"] = tools
    prompt = tokenizer.apply_chat_template(
        sample.messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    # 3. 拼接完整序列
    full_text = prompt + target_text + tokenizer.eos_token

    # 4. Tokenize
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # 5. 截断
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]

    prompt_len = min(len(prompt_ids), len(full_ids))

    # 6. 构建 labels（mask prompt 部分）
    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    result: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "prompt_len": prompt_len,
        "metadata": sample.metadata if sample.metadata else {},
    }

    # 多模态：保存原始 messages 和 media 信息，由 trainer 层的 processor 编码
    if sample.media:
        result["_multimodal_row"] = _multimodal_row_from_sample(sample)
        result["_target_text"] = target_text

    return result
