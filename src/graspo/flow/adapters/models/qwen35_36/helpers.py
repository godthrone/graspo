"""Qwen3.5/3.6 适配器的纯函数工具（不依赖 self，受 mypy 检查）。

从 training_sft.py 拆出（mixin 豁免面收窄）：SFT batch collation 系列函数。
依赖通过参数注入（adapter、device），无模块级状态。

另承载**rollout/生成边界**的两条共用口径（单一真相源 §1.4）：
``resolve_stop_token_ids``（停止 token）与 ``rollout_chat_template_kwargs``
（chat 模板参数，默认关闭 thinking）。
"""

from typing import Any

import torch

from graspo.ripple.data import SFTTokenized

#: 停止 token 的**唯一来源**：chat 回合结束符。
#: Qwen3 系列 chat 模板以 ``<|im_end|>`` 收尾；某些模型它的 id 与
#: ``tokenizer.eos_token_id`` 不同（那时只比较后者会「停不下来」）。
CHAT_TURN_END_TOKEN = "<|im_end|>"


def resolve_stop_token_ids(tokenizer: Any) -> list[int]:
    """解析本次生成必须停下的 token id 列表（**单一真相源**）。

    判据（为什么既不停不下来、也不停得太早）：

    - 取 ``tokenizer.eos_token_id``（可能是单个 int，也可能是 list——HF 允许
      多结束符，如 ``[151643, 151645]``），两种形态都接受；
    - 再补上 chat 模板的回合结束符 ``<|im_end|>``，**但必须是词表里的真 token**
      （用 ``convert_ids_to_tokens`` 反向核对，避免 tokenizer 把它映射到
      unk/0 而误停）；
    - 二者相同时去重（本矩阵的 Qwen3.5-9B 实测：``eos_token='<|im_end|>'`` ⇒
      248046 == 248046，故行为与旧实现逐字相同）；
    - **不**把 ``pad_token``（本模型是 ``<|endoftext|>``=248044）当结束符：把它
      当结束符会在左 padding 行上立刻「停」（停得太早），pad 与 eos 的语义必须
      分开（见 generation.py 的 ``pad_token_id`` 解析）；
    - 解析不出任何 id ⇒ fail-closed 报错（§2.3），绝不静默退化成「永不停止」。
    """
    raw = getattr(tokenizer, "eos_token_id", None)
    ids: list[int] = []
    if isinstance(raw, (list, tuple)):
        ids.extend(int(value) for value in raw if value is not None)
    elif raw is not None:
        ids.append(int(raw))
    candidate = tokenizer.convert_tokens_to_ids(CHAT_TURN_END_TOKEN)
    if (
        isinstance(candidate, int)
        and candidate >= 0
        and tokenizer.convert_ids_to_tokens(candidate) == CHAT_TURN_END_TOKEN
        and candidate not in ids
    ):
        ids.append(candidate)
    if not ids:
        raise ValueError(
            "无法解析停止 token：tokenizer.eos_token_id 为空且词表中没有 "
            f"{CHAT_TURN_END_TOKEN}；请检查模型目录的 tokenizer_config.json。"
        )
    return ids


def stop_ids_tensor(tokenizer: Any, device: torch.device | str) -> torch.Tensor:
    """把 :func:`resolve_stop_token_ids` 的结果固化为 1-D 张量（decode 热路径只比一次）。"""
    return torch.tensor(resolve_stop_token_ids(tokenizer), dtype=torch.long, device=device)


def apply_stop_mask(
    finished: torch.Tensor,
    next_token: torch.Tensor,
    stop_token_ids: int | list[int] | torch.Tensor,
) -> torch.Tensor:
    """把「本轮采样是否命中任一停止 token」并入 ``finished``（**唯一的判据实现**）。

    单 id（int）与多 id（list / 1-D 张量）两种形态都支持；返回新的 ``finished``，
    不原地改写，避免调用点各自写一份比较逻辑（§1.4 单一真相源）。
    """
    if isinstance(stop_token_ids, torch.Tensor):
        if stop_token_ids.ndim == 0:
            return finished | next_token.eq(stop_token_ids)
        return finished | (next_token.unsqueeze(1) == stop_token_ids.reshape(1, -1)).any(dim=1)
    if isinstance(stop_token_ids, (list, tuple)):
        comparator = torch.tensor(
            [int(value) for value in stop_token_ids],
            dtype=next_token.dtype,
            device=next_token.device,
        )
        return finished | (next_token.unsqueeze(1) == comparator.reshape(1, -1)).any(dim=1)
    return finished | next_token.eq(int(stop_token_ids))


def rollout_chat_template_kwargs(
    chat_template_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """rollout / SFT 共用的 chat 模板参数（**单一真相源**）。

    默认 ``enable_thinking=False``：Qwen3.5 的 chat 模板在 assistant 回合默认渲染
    ``'<|im_start|>assistant\\n thinking\\n'`` ⇒ 模型先产出一大段 CoT，在
    ``training.max_new_tokens`` 上限处被生生切断，``<|im_end|>`` 永不出现：
    decode「停不下来」、每条 completion 都被截断、解析失败 ⇒ rollout 空转。
    （228 实测 T028：11,544 次 forward / 674 s 未走完一个 rollout queue；
    E1 复现：32-token 上限下 completion 全部撞上限、全部判 retry。）

    SFT 侧早已如此（``collate_sft_multimodal_batch`` 与 ``sft_tokenize_text``
    都 ``setdefault("enable_thinking", False)``），rollout 侧此前漏了这一步
    ⇒ 两侧口径不一致。本函数把这条口径收敛到一处。

    配置显式给出 ``enable_thinking`` 时以配置为准（透明退路 §3.2，不静默覆盖用户值）。
    """
    kwargs = dict(chat_template_kwargs or {})
    kwargs.setdefault("enable_thinking", False)
    return kwargs


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

    # 2. 逐个样本编码 prompt（复用 RL 路径，逐个产出 input_ids + pixel_values）
    # 修复：不再一次性批量编码所有样本，避免 processor.apply_chat_template
    # 在 4 张 720P 图（batch=2 双目）时 CPU 预处理阻塞 → NCCL 超时。
    # 逐个编码后拼接 pixel_values/image_grid_thw，vision encoder forward 仍批量处理。
    # SFT 禁用 thinking：与 rollout 共用同一条口径（rollout_chat_template_kwargs，§1.4 单一真相源）
    from torch.nn.utils.rnn import pad_sequence as _pad_sequence

    chat_template_kwargs = rollout_chat_template_kwargs(adapter.config.model.chat_template_kwargs)

    all_prompt_ids: list[torch.Tensor] = []
    all_prompt_masks: list[torch.Tensor] = []
    all_multimodal: list[dict[str, torch.Tensor]] = []

    for row in rows:
        encoded = adapter._encode_multimodal_rows(
            [row],
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        all_prompt_ids.append(encoded["input_ids"].to(device))  # (1, L_i)
        all_prompt_masks.append(encoded["attention_mask"].to(device))  # (1, L_i)
        all_multimodal.append(adapter._multimodal_inputs_to_device(encoded))

    # Pad prompt_ids / attention_mask 到统一长度后 stack
    prompt_ids = _pad_sequence(
        [ids.squeeze(0) for ids in all_prompt_ids],
        batch_first=True,
        padding_value=0,
    )  # (batch, max_prompt_len)
    prompt_mask = _pad_sequence(
        [mask.squeeze(0) for mask in all_prompt_masks],
        batch_first=True,
        padding_value=0,
    ).bool()  # (batch, max_prompt_len)
    max_prompt_len = int(prompt_ids.shape[1])

    # 拼接 multimodal inputs（pixel_values / image_grid_thw 沿 batch 维 cat）
    multimodal_inputs: dict[str, torch.Tensor] = {}
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        tensors = [m[key] for m in all_multimodal if key in m]
        if tensors:
            multimodal_inputs[key] = torch.cat(tensors, dim=0)
    # mm_token_type_ids 与 input_ids 对齐，需 pad 到 max_prompt_len 后 stack
    mm_token_tensors = [m["mm_token_type_ids"] for m in all_multimodal if "mm_token_type_ids" in m]
    if mm_token_tensors:
        padded_mm: list[torch.Tensor] = []
        for t in mm_token_tensors:
            if t.shape[1] < max_prompt_len:
                t = torch.nn.functional.pad(t, (0, max_prompt_len - t.shape[1]), value=0)
            padded_mm.append(t)
        multimodal_inputs["mm_token_type_ids"] = torch.cat(padded_mm, dim=0)

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
