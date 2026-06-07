from __future__ import annotations

from typing import Any

import torch


def ensure_tokenizer_ready(tokenizer: Any) -> None:
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


@torch.no_grad()
def generate_group(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    group_size: int,
    device: torch.device,
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    synced_gpus: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], int]:
    text = render_messages(tokenizer, messages, chat_template_kwargs)
    prompt_len = 0

    # Micro-batch generation to avoid OOM from KV cache on large groups.
    # Each micro-batch stores its own KV cache; with 8xA800 80GB, batch_size=1
    # is safe (~54 GB model + ~6 GB KV cache/seq = ~60 GB < 80 GB).
    micro_batch_size = 1
    all_sequences = []
    all_completions = []
    max_seq_len = 0

    # gradient_checkpointing_enable() sets config.use_cache=False globally;
    # temporarily re-enable it so generate() actually uses KV cache.
    saved_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True

    for start in range(0, group_size, micro_batch_size):
        end = min(start + micro_batch_size, group_size)
        inputs = tokenizer(
            [text] * (end - start),
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
        all_sequences.append(sequences)
        all_completions.extend(completions)
        if sequences.shape[1] > max_seq_len:
            max_seq_len = sequences.shape[1]

    # Pad to uniform length before concatenating (micro-batches may differ in length)
    padded = []
    for seq in all_sequences:
        if seq.shape[1] < max_seq_len:
            pad = torch.full(
                (seq.shape[0], max_seq_len - seq.shape[1]),
                tokenizer.pad_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat([seq, pad], dim=1)
        padded.append(seq)
    model.config.use_cache = saved_use_cache

    sequences = torch.cat(padded, dim=0)
    attention_mask = sequences.ne(tokenizer.pad_token_id)
    action_mask = torch.zeros_like(sequences[:, 1:], dtype=torch.bool)
    action_mask[:, max(prompt_len - 1, 0) :] = True
    action_mask = action_mask & sequences[:, 1:].ne(tokenizer.pad_token_id)
    return sequences, attention_mask, action_mask, all_completions, prompt_len
