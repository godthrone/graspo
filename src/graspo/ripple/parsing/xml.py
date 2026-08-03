"""Qwen XML 工具调用格式构建 —— 算法层（ripple.parsing），零设施依赖。

单一真相源：模型侧（flow）与奖励侧（ripple）都从本模块取 XML 构建函数，
防止格式串在多处重建导致漂移（v0.14 时代的已知风险）。

- build_sft_target_text: 将 target output 转为模型应生成的文本（XML 或 JSON）
- tool_calls_to_xml: canonical tool_calls → Qwen XML
- format_xml_param_value: 参数值 → XML 文本（bool/float 规范化）
"""

from typing import Any


def format_xml_param_value(value: Any) -> str:
    """格式化参数值为 XML 文本。"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value == int(value) and not (value != value):  # 非 NaN 的整数值
            return str(int(value))
        return str(value)
    return str(value)


def tool_calls_to_xml(tool_calls: list[dict[str, Any]]) -> str:
    """将 canonical tool_calls 转为 Qwen XML 格式。

    格式与 Qwen3.5 预训练中学会的原生 tool-call 输出严格对齐：参数值
    位于独立行，标签之间用换行分隔。这是 base model 的默认输出风格，
    SFT 不应改变它——只需教会模型输出正确的参数名和数值。
    """
    parts: list[str] = []
    for call in tool_calls:
        name = call["name"]
        arguments = call["arguments"]
        params = "\n".join(
            f"<parameter={key}>\n{format_xml_param_value(value)}\n</parameter>"
            for key, value in arguments.items()
        )
        parts.append(f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
    return "\n".join(parts)


def build_sft_target_text(target_output: dict[str, Any]) -> str:
    """将 target output 转换为模型应该生成的文本。

    - tool_calls: 转为 Qwen XML 格式，与 base model 原生输出格式完全一致
    - content: 转为 markdown-fenced JSON

    **格式对齐：** Qwen3.5 预训练中学会的 XML 工具调用格式是参数值占据独立行
    （``<parameter=key>\\nvalue\\n</parameter>``），不是内联紧凑格式。SFT target
    必须匹配这一原生格式，否则 LoRA 需要同时改写格式和内容，在参数量有限时导致
    灾难性干扰——模型在新旧格式间摇摆，输出结构崩溃。

    .. warning::
        **DO NOT** refactor this to use ``tokenizer.apply_chat_template``.
        Qwen's chat template inserts ``\\n response\\n`` before assistant tool_calls
        when ``enable_thinking`` is not set, but RL inference runs with
        ``enable_thinking=false`` whose assistant prefix is ``<|im_start|>assistant\\n``.
        The two prefixes mismatch, causing SFT to teach the wrong format and the
        model to output garbage like ``\\n\\nfunction\\n response\\n\\n response\\n``.
        Direct XML generation ensures the target text matches the model's actual
        inference output character-for-character.

    Architecture invariant: the SFT target text MUST be the exact text the model
    is expected to generate during inference.  For Qwen models this means bare
    ``<tool_call>...</tool_call>`` XML without any prefix or suffix.
    """
    tool_calls = target_output.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls_to_xml(tool_calls)
    content = target_output.get("content")
    if isinstance(content, dict):
        import json as _json

        return "```json\n" + _json.dumps(content, ensure_ascii=False, indent=2) + "\n```"
    return ""
