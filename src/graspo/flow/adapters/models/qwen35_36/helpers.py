"""Qwen3.5/3.6 适配器的纯函数工具（不依赖 self，受 mypy 检查）。

从 training_sft.py 拆出（mixin 豁免面收窄）：SFT batch collation 系列函数。
依赖通过参数注入（adapter、device），无模块级状态。
"""

from typing import Any

import torch

from graspo.ripple.data import SFTTokenized


def collate_sft_batch(
    items: list[SFTTokenized],
    device: torch.device,
    *,
    adapter: Any,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """将多个 ``SFTTokenized`` 样本拼接为 micro-batch（统一入口，自动分派）。

    纯文本 → ``collate_sft_text_batch``（预 tokenized 张量 padding）
    多模态 → ``collate_sft_multimodal_batch``（复用 RL 的 ``_encode_multimodal_rows``）
    """
    if any(item.deferred_multimodal is not None for item in items):
        if any(item.deferred_multimodal is None for item in items):
            raise ValueError(
                "Cannot mix multimodal and text-only samples in one micro-batch. "
                "Ensure all samples in a batch have the same type."
            )
        return collate_sft_multimodal_batch(
            items, device, adapter=adapter, max_seq_length=max_seq_length
        )
    return collate_sft_text_batch(items, device)


def collate_sft_text_batch(items: list[SFTTokenized], device: torch.device) -> dict[str, Any]:
    """纯文本 SFT batch：pad 预 tokenized 张量。"""
    from torch.nn.utils.rnn import pad_sequence

    # 纯文本路径不变式：tensor 字段必非 None（多模态路径走 collate_sft_multimodal_batch）
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    attention_list: list[torch.Tensor] = []
    for item in items:
        assert item.input_ids is not None and item.labels is not None
        assert item.attention_mask is not None
        input_ids_list.append(item.input_ids)
        labels_list.append(item.labels)
        attention_list.append(item.attention_mask)

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(device)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(device)
    attention_mask = (
        pad_sequence(attention_list, batch_first=True, padding_value=0).bool().to(device)
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "metadata": [item.metadata for item in items],
    }


def collate_sft_multimodal_batch(
    items: list[SFTTokenized],
    device: torch.device,
    *,
    adapter: Any,
    max_seq_length: int = 4096,
) -> dict[str, Any]:
    """多模态 SFT batch：调用 ``_encode_multimodal_rows`` 一次性编码，复用 RL 路径。

    RL 的 ``generate_sample_groups`` → ``_encode_multimodal_rows`` 是同一次调用产出
    ``input_ids`` + ``pixel_values`` + ``image_grid_thw``。SFT 对齐：在 collate 阶段
    一次性编码，而非在 ``sft_tokenize`` 中预编码、训练时再二次编码。
    """
    tokenizer = adapter.tokenizer

    # 1. 构建 rows（与 RL 的 ripple.multimodal.rows.multimodal_row_from_sample 格式一致）
    rows: list[dict[str, Any]] = []
    target_texts: list[str] = []
    for item in items:
        assert item.deferred_multimodal is not None
        deferred = item.deferred_multimodal
        media_types = count_media_types_from_messages(deferred.prompt_messages)
        row: dict[str, Any] = {
            "messages": deferred.prompt_messages,
            "media": media_types,
        }
        tools = deferred.tools
        if tools is not None:
            row["tools"] = tools
        rows.append(row)
        target_texts.append(deferred.target_text)

    # 2. 单次编码 prompt（复用 RL 路径，一次性产出 input_ids + pixel_values）
    # SFT 禁用 thinking：与 sft_tokenize_text 的 setdefault("enable_thinking", False) 一致
    chat_template_kwargs = dict(adapter.config.model.chat_template_kwargs or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    encoded = adapter._encode_multimodal_rows(
        rows,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_ids = encoded["input_ids"].to(device)  # (batch, padded_prompt_len)
    prompt_mask = encoded["attention_mask"].to(device)  # (batch, padded_prompt_len)
    multimodal_inputs = adapter._multimodal_inputs_to_device(encoded)

    # 3. 编码 target text（纯文本，tokenizer 即可）
    eos_id = int(tokenizer.eos_token_id)
    response_ids_list = [
        tokenizer.encode(text, add_special_tokens=False) + [eos_id] for text in target_texts
    ]

    # 4. 拼接 prompt + response，构建 input_ids / labels / attention_mask
    batch_size = len(items)
    prompt_len_padded = int(prompt_ids.shape[1])
    max_response_len = max(len(ids) for ids in response_ids_list)
    total_len = prompt_len_padded + max_response_len

    input_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        actual_prompt_len = int(prompt_mask[i].sum().item())
        # 复制 prompt tokens（仅有效部分，padding 区域保持 0）
        input_ids[i, :actual_prompt_len] = prompt_ids[i, :actual_prompt_len]
        attention_mask[i, :actual_prompt_len] = True
        # 追加 response tokens
        r_ids = response_ids_list[i]
        r_len = len(r_ids)
        input_ids[i, actual_prompt_len : actual_prompt_len + r_len] = torch.tensor(
            r_ids, dtype=torch.long, device=device
        )
        labels[i, actual_prompt_len : actual_prompt_len + r_len] = torch.tensor(
            r_ids, dtype=torch.long, device=device
        )
        attention_mask[i, actual_prompt_len : actual_prompt_len + r_len] = True

    # 5. 截断（防呆，宪法 §2）：多模态样本的视觉占位符被截断会破坏
    #    placeholder 数 == image_grid_thw 特征数的对齐，forward 时抛出难懂的
    #    RuntimeError（model.py 的 masked_scatter 检查）。这里提前显式报错，
    #    并给出可操作的修复建议（调大 data.max_prompt_length）。
    if total_len > max_seq_length:
        image_token_id = int(getattr(adapter.model.config, "image_token_id", -1))
        if image_token_id >= 0:
            cut = input_ids[:, max_seq_length:]
            cut_placeholders = int((cut == image_token_id).sum().item())
            if cut_placeholders > 0:
                raise ValueError(
                    "max_prompt_length 截断了视觉占位符："
                    f"max_seq_length={max_seq_length}, input_ids 全长={total_len}, "
                    f"被截断的视觉占位符={cut_placeholders} 个。"
                    "多模态样本必须完整容纳视觉 token（占位符与视觉特征一一对应，"
                    "截断会导致 forward 报 'Image features and image placeholder tokens "
                    "do not match'）。请调大 data.max_prompt_length（建议 ≥ total_len，"
                    "Qwen3.5 720P 双目图建议 8192）。"
                )
        input_ids = input_ids[:, :max_seq_length]
        labels = labels[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "multimodal_inputs": multimodal_inputs,
        "metadata": [item.metadata for item in items],
    }


def count_media_types_from_messages(
    messages: list[dict[str, Any]],
) -> dict[str, int]:
    """从 messages 中统计媒体类型数量（与 ``ripple.multimodal.rows.media_counts`` 格式一致）。"""
    counts: dict[str, int] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            media_type = block.get("type")
            if media_type in ("image", "image_url"):
                counts["image"] = counts.get("image", 0) + 1
            elif media_type in ("video", "video_url"):
                counts["video"] = counts.get("video", 0) + 1
    return counts
