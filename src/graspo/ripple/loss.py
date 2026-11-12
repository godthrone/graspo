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

    **为什么这么算（旧实现坏在哪）**：旧实现把目标 token 的 logit 另算一遍（手写点积），
    并且 ``logsumexp`` 直接在 **bf16 logits** 上求。后者是主导项：``log_p = L_t - lse``
    因此被量化到 lse 的 bf16 网格上（``|lse| ≈ 12`` 时步长 **0.0625**），真实 CE 掉到网格
    以下时**读数塌成 ±0**。确定性构造实测（fp64 参考，``tests/ripple/test_loss_bf16.py``
    里锁死）：真值 CE ``0.0024`` 一档四个 token，旧读数 **4/4 都是 0.0**，现读数
    ``0.00241 / 0.00264 / 0.00227 / 0.00278``（误差 ≤ 3.3e-5）。

    真机形状下的变化量（H=5120 / V=151936 / seq=1024 / chunk=32768 / bf16 /
    torch 2.11.0+cu130 / A800-SXM4-80GB，同一输入新旧对比）：``max|Δ| = 0.125``、
    ``mean|Δ| = 0.0396``、**639/1024** 个 token 的读数发生变化；``max|Δ|`` 正好是
    ``2 ×`` lse 在该量级的 bf16 步长 ⇒ 量化项主导，不是"半个 ULP 的抵消"那么小。

    **计算走 fp32，返回值回转输入 dtype。** 分块 ``F.linear`` 的输出先按累加精度
    （``acc_dtype``，见下）转过去，``logsumexp`` 与 ``logaddexp`` 累加都在它上面做；目标
    token 的 logit **从同一块 logits 里取**（``gather``），不是另算一遍——因此
    ``log_probs = L_t - logsumexp`` 的两项用的是**同一个数**，``logsumexp ≥ L_t`` 恒成立，
    ``log_probs ≤ 0`` 由构造保证，不需要任何 ``clamp``。

    两种"改法"的实测对照（同上真机形状，1024 个 token，fp64 参考）——**只记录在这里，
    别处只放指针**：旧实现 0 个正 log_p、逐点 max|err| 5e-5…2.5e-3；**只把两条路径升成
    fp32（目标 logit 仍另算一遍）473…504 个正 log_p、最大 +0.025（更糟）**；连 GEMM 都用
    fp32 则 5…632 个正 log_p（≤1.5e-4），代价是 loss 前向 42 ms → 367 ms（8.7×）；本实现
    0 个正 log_p。

    返回时 ``.to(hidden_states.dtype)`` 回转输入 dtype（RL/SFT 共用本函数，下游 PPO
    ratio 等依赖原 dtype）。bf16 舍入是 round-to-nearest **单调**的，符号因此保持：一个非
    正数回转后**仍是负数或 ``-0.0``**（可能更靠近 0），不会翻成正数。

    精度口径：``acc_dtype`` = fp32（输入为 bf16/fp16 时），否则等于输入 dtype —— ⛔ 不能
    一律写 ``.float()``：那会把 fp64 输入静默降成 fp32，破坏双精度对拍与
    ``autograd.gradcheck``（实测会打挂仓库里两条 gradcheck）。SFT metrics 的
    ``negative_loss_count`` 保留作**防御性计数**：它一旦非零，说明这里的非负性前提被破坏
    了，值得查（口径见 ``_updated_negative_loss_count``）。
    """
    valid = token_ids != ignore_index
    ids = token_ids.clamp(min=0)
    # 防呆（fail-fast）：token_ids 越界时旧实现 `index_select` 会直接 IndexError；本实现把
    # id 映射成块内下标后，越界会被下面那个 clamp **静默**掩盖成"取到同块内别的 token 的
    # logit"（头部非负性质不破，但错输入进来了没人知道）。这里把它立刻暴露出来。
    assert not ids.numel() or int(ids.max()) < lm_head_weight.shape[0], (
        f"token_ids 越界：max={int(ids.max())} ≥ 词表大小 {lm_head_weight.shape[0]}"
    )
    # 累加精度：bf16/fp16 输入升到 fp32；fp32/fp64 输入按原精度算（⛔ 不可写成
    # ``.float()``——那会把 fp64 输入静默降成 fp32，破坏双精度对拍与 gradcheck）。
    acc_dtype = (
        torch.float32
        if hidden_states.dtype in (torch.bfloat16, torch.float16)
        else hidden_states.dtype
    )
    selected_logits: torch.Tensor | None = None
    logsumexp: torch.Tensor | None = None
    for start in range(0, lm_head_weight.shape[0], vocab_chunk_size):
        chunk = lm_head_weight[start : start + vocab_chunk_size]
        logits = F.linear(hidden_states, chunk).to(acc_dtype)
        relative = ids - start
        in_chunk = (relative >= 0) & (relative < logits.shape[-1])
        picked = logits.gather(
            -1, relative.clamp(min=0, max=logits.shape[-1] - 1).unsqueeze(-1)
        ).squeeze(-1)
        selected_logits = (
            picked if selected_logits is None else torch.where(in_chunk, picked, selected_logits)
        )
        chunk_lse = torch.logsumexp(logits, dim=-1)
        logsumexp = chunk_lse if logsumexp is None else torch.logaddexp(logsumexp, chunk_lse)
    assert logsumexp is not None and selected_logits is not None
    log_probs = selected_logits - logsumexp
    return log_probs.masked_fill(~valid, 0.0).to(hidden_states.dtype)


#: ★ P6：F-3 诊断插桩的**默认开关** —— 默认「开」与现状逐字一致（行为不变）。
#: 这是**开发期诊断**，不是算法语义；关闭只影响那一行 stdout，不影响任何数学。
#: 单一真相源在配置层（``training.f3_diag_enabled``），本常量只是无配置时的回退值。
F3_DIAG_ENABLED: bool = True


class GRASPORippleLoss(nn.Module):
    def __init__(
        self,
        policy_ratio_clip_eps: float = 0.2,
        *,
        diag_enabled: bool | None = None,
    ) -> None:
        super().__init__()
        self.policy_ratio_clip_eps = policy_ratio_clip_eps
        #: ``None`` ⇒ 用模块级默认（默认 True = 与现状一致）
        self.diag_enabled = F3_DIAG_ENABLED if diag_enabled is None else bool(diag_enabled)

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ★ F-3（诊断插桩，2026-10-05 指挥官批准）：**只读统计，⛔ 不改下面的数学**。
        #   目的：判定候选 F（`exp` 无差值保护 ⇒ 前向被 `clamp` 掩盖、反向在 inf 处爆炸）。
        #   ★ 零语义影响：统计在 `no_grad` 下计算、仅写入实例属性并向 stdout 打印一行
        #     （stdout 由 runner 重定向到 `/out/stdout.log`，可直接 grep）。
        #   ★ 全程 try/except 包裹：诊断**绝不**影响训练，也不触碰计算图。
        _diag = None
        try:
            with torch.no_grad():
                _flat = (log_probs - old_log_probs).detach().float().flatten()
                if _flat.numel():
                    _diag = {
                        "max": float(_flat.max().cpu()),
                        "p99": float(torch.quantile(_flat, 0.99).cpu()),
                        "gt20": int((_flat > 20).sum().cpu()),
                        "gt88": int((_flat > 88).sum().cpu()),
                        "numel": int(_flat.numel()),
                    }
                else:
                    _diag = {"max": 0.0, "p99": 0.0, "gt20": 0, "gt88": 0, "numel": 0}
        except Exception as _exc:  # noqa: BLE001 诊断绝不影响训练
            _diag = {"error": repr(_exc)}
        self.diag_diff = _diag
        self.diag_calls = int(getattr(self, "diag_calls", 0)) + 1
        # ★ P6：改**配置开关**（默认开 = 与现状一致）；关掉只少这一行 stdout。
        if self.diag_enabled:
            print(f"[F3-DIAG] call={self.diag_calls} {_diag}", flush=True)
        ratio = (log_probs - old_log_probs).exp()
        ratio = ratio.clamp(0.1, 10.0)
        surr1 = ratio * advantages
        surr2 = (
            ratio.clamp(1 - self.policy_ratio_clip_eps, 1 + self.policy_ratio_clip_eps) * advantages
        )
        loss = -torch.min(surr1, surr2)
        return masked_mean(loss, action_mask, dim=-1).mean()
