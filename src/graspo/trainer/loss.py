from __future__ import annotations

import torch
from torch import nn


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor | None, dim: int | None = None) -> torch.Tensor:
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
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    return sequence_log_probs_from_logits(
        logits=logits[:, :-1].float(),
        output_ids=sequence_ids[:, 1:],
    )


class GRASPOLoss(nn.Module):
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
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.policy_ratio_clip_eps, 1 + self.policy_ratio_clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        return masked_mean(loss, action_mask, dim=-1).mean()
