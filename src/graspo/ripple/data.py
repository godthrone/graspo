"""训练数据校验与编码：SFT tokenize、样本校验。

算法层（ripple）数据变换：Sample 构造依赖 targets 归一化（reward/normalize）、
SFT target 文本构建（parsing/xml）、多模态路径解析（multimodal/rows）——
因此本文件位于 ripple 层，只依赖 core 的 Sample 契约（单向依赖）。
JSONL 文件 I/O 已迁入 flow/data_io.py（设施层）。
"""

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from graspo.core.schema import Sample
from graspo.ripple.multimodal.rows import resolve_messages_media_paths
from graspo.ripple.parsing.xml import build_sft_target_text
from graspo.ripple.reward.normalize import normalize_targets

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
    _validate_tool_messages(messages)
    return messages


def _validate_tool_messages(messages: list[dict[str, Any]]) -> None:
    """Validate ``tool`` role messages conform to OpenAI standard.

    Each ``tool`` message must have a non-empty ``tool_call_id`` string.
    """
    for idx, message in enumerate(messages):
        if str(message.get("role") or "").lower() != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError(
                f"messages[{idx}] (tool) must have a non-empty 'tool_call_id' "
                f"(OpenAI standard)"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"messages[{idx}] (tool) content must be a string, got {type(content).__name__}"
            )


def _validate_assistant_tool_calls(messages: list[dict[str, Any]]) -> None:
    """Validate assistant messages use OpenAI-standard structured tool_calls.

    Raises ValueError if any assistant message embeds raw tool-call markers
    (``<tool_call>``, ``<function=``) in its content field, or if a
    ``tool_calls`` field does not conform to OpenAI canonical JSON format.
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
                    f"Use OpenAI-standard structured 'tool_calls' field instead:\n"
                    f'  {{"tool_calls": [{{"id": "call_0000", "type": "function",'
                    f' "function": {{"name": "...", "arguments": {{...}}}}}}]}}\n'
                    f"  Found in content: {snippet!r}"
                )

        # ---------- 2. validate tool_calls field format ----------
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if content is not None:
                raise ValueError(
                    f"messages[{idx}] (assistant) has tool_calls but content is not null; "
                    f"OpenAI standard requires content: null when tool_calls are present"
                )
            _require_canonical_tool_calls(tool_calls, path=f"messages[{idx}].tool_calls")


def _require_canonical_tool_calls(value: Any, *, path: str = "tool_calls") -> None:
    """Validate *value* is an OpenAI-standard tool-call list.  Raises ``ValueError``.

    Only the OpenAI canonical form is accepted::

        [{"id": "<non-empty-str>", "type": "function",
          "function": {"name": "<non-empty-str>", "arguments": {<dict>}}}, ...]

    ``arguments`` values must not contain raw tool-call markers
    (avoids XML smuggled inside JSON string values).
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")

    for idx, call in enumerate(value):
        if not isinstance(call, dict):
            raise ValueError(f"{path}[{idx}] must be a JSON object")

        # OpenAI standard: id + type + function wrapper
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError(
                f"{path}[{idx}].id must be a non-empty string "
                f"(OpenAI standard requires 'id' field)"
            )

        call_type = call.get("type")
        if call_type != "function":
            raise ValueError(
                f"{path}[{idx}].type must be 'function' "
                f"(OpenAI standard), got {call_type!r}"
            )

        func = call.get("function")
        if not isinstance(func, dict):
            raise ValueError(
                f"{path}[{idx}].function must be a JSON object "
                f"(OpenAI standard requires 'function' wrapper)"
            )

        name = func.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}[{idx}].function.name must be a non-empty string")

        arguments = func.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{path}[{idx}].function.arguments must be a JSON object")

        # Reject raw tool-call markers smuggled inside argument values.
        for arg_key, arg_val in arguments.items():
            if isinstance(arg_val, str) and _TOOL_CALL_MARKER_RE.search(arg_val):
                raise ValueError(
                    f"{path}[{idx}].function.arguments.{arg_key} contains raw tool-call "
                    f"markers in its string value; use structured JSON values instead: "
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


@dataclass(frozen=True, slots=True)
class MultimodalDeferred:
    """多模态延迟编码数据，由 ``_collate_sft_multimodal_batch`` 消费。

    与 RL 的 ``generate_sample_groups`` 对齐：不在数据加载阶段预编码图像，
    等到训练时通过 ``_encode_multimodal_rows`` 一次性完成 tokenize + 视觉编码。
    避免 ``sft_tokenize`` 中的 processor 调用和训练循环中的 processor 调用
    产生不一致的 ``input_ids`` / ``pixel_values`` 配对。
    """

    prompt_messages: list[dict[str, Any]]  # 路径已解析为绝对路径
    target_text: str  # 模型应生成的文本（tool_call XML 或 JSON）
    tools: list[dict[str, Any]] | None


@dataclass(frozen=True, slots=True)
class SFTTokenized:
    """``sft_tokenize_text`` / ``sft_tokenize_multimodal`` 的统一输出。

    纯文本路径：``input_ids`` / ``labels`` / ``attention_mask`` 有效，
    ``deferred_multimodal`` 为 ``None``。
    多模态路径：``deferred_multimodal`` 有效，tensor 字段为 ``None``。
    """

    metadata: dict[str, Any]
    input_ids: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    prompt_len: int = 0
    deferred_multimodal: MultimodalDeferred | None = None


def sft_tokenize_text(
    sample: Sample,
    tokenizer: Any,
    *,
    max_seq_length: int = 4096,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> SFTTokenized:
    """将纯文本样本转换为 SFT 训练所需的 token 序列（预 tokenize）。

    与 GRASPO 推理对齐：用 ``apply_chat_template`` 生成 prompt，
    labels mask 掉 prompt 部分，只对 response 部分计算 loss。
    """
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
    full_text = prompt + target_text + tokenizer.eos_token
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # 3. 截断
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]

    prompt_len = min(len(prompt_ids), len(full_ids))

    # 4. 构建 labels（mask prompt 部分）
    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len

    return SFTTokenized(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        attention_mask=torch.ones(len(full_ids), dtype=torch.long),
        prompt_len=prompt_len,
        metadata=sample.metadata if sample.metadata else {},
    )


def sft_tokenize_multimodal(
    sample: Sample,
    tokenizer: Any,
    *,
    data_dir: str | Path | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> SFTTokenized:
    """为多模态样本准备延迟编码数据。

    与 RL 的 ``generate_sample_groups`` 对齐：**不在数据加载阶段调 processor**，
    只解析路径、构建 target text，将原始数据传递给训练循环。
    训练循环通过 ``_encode_multimodal_rows`` 一次性完成 tokenize + 视觉编码，
    确保 ``input_ids`` 和 ``pixel_values`` 来自同一次 processor 调用。

    关键对齐点（SFT → RL pipeline）：
    1. target text 用 ``build_sft_target_text`` 生成纯 XML（不经过 chat template）
    2. 图像路径用 ripple/multimodal/rows.py 的 resolve_messages_media_paths 解析
    3. 编码时由 ``_encode_multimodal_rows`` 统一处理，RL 侧 rows 构建同样
       通过 ``data_dir`` 参数解析路径

    .. warning::
        ``build_sft_target_text`` 的调用**不传** tokenizer——它直接生成 XML，
        不经过 ``apply_chat_template``。详见 ``build_sft_target_text`` 的 docstring。
    """
    # 1. 提取 primary target 并构建 response 文本
    primary = sample.targets[0]
    target_text = build_sft_target_text(primary["output"])
    if not target_text:
        raise ValueError("primary target has no content or tool_calls")

    # 2. 深拷贝 messages 并解析相对路径 → 绝对路径
    messages_for_processor = copy.deepcopy(sample.messages)
    resolve_messages_media_paths(messages_for_processor, data_dir)

    tools = sample.tools if sample.expects_tool_calls else None

    return SFTTokenized(
        metadata=sample.metadata if sample.metadata else {},
        deferred_multimodal=MultimodalDeferred(
            prompt_messages=messages_for_processor,
            target_text=target_text,
            tools=tools,
        ),
    )


def sft_tokenize(
    sample: Sample,
    tokenizer: Any,
    *,
    max_seq_length: int = 4096,
    chat_template_kwargs: dict[str, Any] | None = None,
    data_dir: str | Path | None = None,
    processor: Any = None,
) -> SFTTokenized:
    """将一条样本转换为 SFT 训练所需的数据（统一入口，自动分派）。

    纯文本样本 → ``sft_tokenize_text``（预 tokenize）
    多模态样本 → ``sft_tokenize_multimodal``（延迟编码，对齐 RL 路径）
    """
    if sample.media and processor is not None:
        return sft_tokenize_multimodal(
            sample,
            tokenizer,
            data_dir=data_dir,
            chat_template_kwargs=chat_template_kwargs,
        )
    return sft_tokenize_text(
        sample,
        tokenizer,
        max_seq_length=max_seq_length,
        chat_template_kwargs=chat_template_kwargs,
    )
