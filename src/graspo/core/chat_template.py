"""Chat template 渲染与 tokenizer 准备工具 —— 纯计算层，零设施依赖。

该模块仅依赖 tokenizer 的公开接口（``apply_chat_template``、``pad_token`` 等），
不依赖 GPU、分布式或任何模型实现，可在 CPU 上独立测试。
"""

from typing import Any


def ensure_tokenizer_ready(tokenizer: Any) -> None:
    """确保 tokenizer 具有 pad_token_id 且 padding_side 为 left。"""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id")
    tokenizer.padding_side = "left"


def render_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """将 messages 渲染为文本字符串。

    优先使用 tokenizer 的 chat_template，回退到纯文本拼接。
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    return "\n\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
    )
