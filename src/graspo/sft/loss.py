from __future__ import annotations

import torch


def weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    per_sample = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def anchor_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    is_anchor: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if not is_anchor.any():
        return student_logits.new_tensor(0.0)
    mask = labels[:, 1:].ne(-100) & is_anchor[:, None]
    if not mask.any():
        return student_logits.new_tensor(0.0)
    s = student_logits[:, :-1, :] / temperature
    t = teacher_logits[:, :-1, :] / temperature
    log_s = torch.nn.functional.log_softmax(s, dim=-1)
    prob_t = torch.nn.functional.softmax(t, dim=-1)
    kl = torch.nn.functional.kl_div(log_s, prob_t, reduction="none").sum(dim=-1)
    return (kl * mask).sum() / mask.sum().clamp_min(1)

