"""GRASPO 训练 loss 集合 —— 算法层（ripple），零设施依赖。

- GRASPORippleLoss: PPO-clip loss（RL 训练）
- sft_cross_entropy_loss: 标准 SFT cross-entropy loss
- masked_mean / sequence_log_probs*: log-prob 工具函数

全部为纯 torch 张量计算，可在 CPU 上独立测试（ripple 层边界内）。
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor | None, dim: int | None = None
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1)
    return (tensor * mask).sum(dim=dim) / denom


def sequence_log_probs_from_logits(logits: torch.Tensor, output_ids: torch.Tensor) -> torch.Tensor:
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: nn.Module,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    output = model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    # HF 模型输出兼容：部分模型返回对象，部分返回字典
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    return sequence_log_probs_from_logits(
        logits=logits[:, :-1].float(),
        output_ids=sequence_ids[:, 1:],
    )


class GRASPORippleLoss(nn.Module):
    def __init__(self, policy_ratio_clip_eps: float = 0.2) -> None:
        super().__init__()
        self.policy_ratio_clip_eps = policy_ratio_clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        ratio = ratio.clamp(0.1, 10.0)
        surr1 = ratio * advantages
        surr2 = (
            ratio.clamp(1 - self.policy_ratio_clip_eps, 1 + self.policy_ratio_clip_eps) * advantages
        )
        loss = -torch.min(surr1, surr2)
        return masked_mean(loss, action_mask, dim=-1).mean()


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
