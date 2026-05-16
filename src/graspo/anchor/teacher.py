from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from graspo.anchor.bank import AnsweredAnchor, AnchorPrompt, read_anchor_prompts, write_jsonl
from graspo.trainer.generation import ensure_tokenizer_ready


def render_messages(tokenizer: Any, messages: list[dict[str, str]], chat_template_kwargs: dict[str, Any] | None = None) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    return "\n\n".join(message.get("content", "") for message in messages)


@torch.no_grad()
def answer_anchor_prompts(
    model_path: str,
    input_path: str | Path,
    output_path: str | Path,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    max_new_tokens: int = 512,
    max_prompt_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    chat_template_kwargs: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[AnsweredAnchor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    ensure_tokenizer_ready(tokenizer)
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[torch_dtype.lower()]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    prompts = read_anchor_prompts(input_path)
    if limit is not None:
        prompts = prompts[:limit]

    answered: list[AnsweredAnchor] = []
    for item in prompts:
        rendered = render_messages(tokenizer, item.messages, chat_template_kwargs)
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
        answered.append(
            AnsweredAnchor(
                id=item.id,
                messages=item.messages,
                teacher_answer=answer,
                teacher_model=model_path,
                anchor_meta=item.anchor_meta,
            )
        )

    write_jsonl(answered, output_path)
    return answered
