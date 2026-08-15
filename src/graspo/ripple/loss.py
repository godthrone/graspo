"""GRASPO 训练 loss 集合 —— 算法层（ripple），零设施依赖。

- GRASPORippleLoss: PPO-clip loss（RL 训练）
- masked_token_log_probs_from_hidden: 从 hidden states 算指定 token 的 log-prob
  （分块 logsumexp，不物化 (B,S,V)）—— **RL 与 SFT 共用的唯一实现**（宪法 §1.4）
- masked_mean: 带 mask 的均值工具

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


def masked_token_log_probs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    ignore_index: int = -100,
    vocab_chunk_size: int = 32768,
) -> torch.Tensor:
    """从 hidden states 计算指定 token 位置的 log-prob（分块 LSE，不物化 (B,S,V)）。

    RL 与 SFT 在数学上是同一个运算（logit_t - logsumexp(全词表)），此处为唯一实现：
    - RL: 传 ``output_ids``（无 ignore_index），直接得到每个 token 的 log_prob
    - SFT: 传 ``labels``（-100 为忽略位），masked 位置返回 0，配合 ``masked_mean``
      即得 cross-entropy loss（等价于 F.cross_entropy + ignore_index 的全局均值语义）

    分块遍历词表累积 logsumexp，峰值张量 = 一块 (B,S,vocab_chunk_size)，与 TP 规模无关；
    切勿改成物化完整 (B,S,V) logits（历史上 SFT 因此 OOM，见 v0.24.0 修复）。
    """
    valid = token_ids != ignore_index
    ids = token_ids.clamp(min=0)
    selected = lm_head_weight.index_select(0, ids.reshape(-1)).view(
        *ids.shape,
        hidden_states.shape[-1],
    )
    selected_logits = (hidden_states * selected).sum(dim=-1)
    logsumexp: torch.Tensor | None = None
    for start in range(0, lm_head_weight.shape[0], vocab_chunk_size):
        chunk = lm_head_weight[start : start + vocab_chunk_size]
        logits = F.linear(hidden_states, chunk)
        chunk_lse = torch.logsumexp(logits, dim=-1)
        logsumexp = chunk_lse if logsumexp is None else torch.logaddexp(logsumexp, chunk_lse)
    assert logsumexp is not None
    log_probs = selected_logits - logsumexp
    return log_probs.masked_fill(~valid, 0.0)


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
