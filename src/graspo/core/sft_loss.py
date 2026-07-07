"""SFT 训练 loss 函数 —— 纯计算层，零设施依赖。

该模块可在 CPU 上独立测试，不依赖 GPU、分布式或任何模型实现。
"""

import torch
from torch.nn import functional as F  # noqa: N812


def sft_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """标准 SFT cross-entropy loss，自动跳过 mask 掉的 token。

    Args:
        logits: 模型输出 (batch, seq_len, vocab_size)
        labels: 目标 token ids (batch, seq_len)，prompt 部分设为 ``ignore_index``
        ignore_index: labels 中需要跳过计算 loss 的 token id，默认 -100

    Returns:
        标量 loss (scalar tensor)
    """
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
