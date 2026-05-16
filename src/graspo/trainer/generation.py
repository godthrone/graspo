from __future__ import annotations

from typing import Any

import torch


def ensure_tokenizer_ready(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id")
    tokenizer.padding_side = "left"


def render_prompt(tokenizer: Any, prompt: str, chat_template_kwargs: dict[str, Any] | None = None) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    return prompt


@torch.no_grad()
def generate_group(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    device: torch.device,
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    synced_gpus: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], int]:
    text = render_prompt(tokenizer, prompt, chat_template_kwargs)
    inputs = tokenizer(
        [text] * group_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    sequences = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        synced_gpus=synced_gpus,
        use_cache=True,
    )
    completions = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
    attention_mask = sequences.ne(tokenizer.pad_token_id)
    action_mask = torch.zeros_like(sequences[:, 1:], dtype=torch.bool)
    action_mask[:, max(prompt_len - 1, 0) :] = True
    action_mask = action_mask & sequences[:, 1:].ne(tokenizer.pad_token_id)
    return sequences, attention_mask, action_mask, completions, prompt_len
