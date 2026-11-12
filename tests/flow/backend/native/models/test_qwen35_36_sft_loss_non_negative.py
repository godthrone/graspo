"""SFT loss 的非负性：``_compute_sft_loss`` 的输出必须 ``>= 0``（真前向，CPU 可跑）。

口径：``loss = -(log_probs * mask).sum() / mask.sum().clamp_min(1)``，而
``masked_token_log_probs_from_hidden`` 保证 ``log_probs <= 0``（目标 logit 与 logsumexp
同源 ⇒ ``logsumexp >= L_t``）⇒ **loss >= 0 由构造保证**，不需要任何 clamp、不靠容差。

这里走的是**生产路径**（``_Qwen35SFTTrainingMethods._compute_sft_loss``：``model.norm`` →
因果 shift → 共享 loss 实现 → masked 均值），不是手搓的等价式。

**判别力说明（如实记）**：这条断言在 CPU 上对**修复前**的代码不红——CPU 上两条 bf16
路径对同一个 logit 恰好给出同一个 bf16 值，舍入自行抵消（所以该缺陷长期只在真机出现）。
它对"只把两条路径升成 fp32、但目标 logit 仍另算一遍"的改法**是红的**（真机实测 47% 的
位置变成 ``log_p > 0``，见 ``tests/ripple/test_loss_bf16.py`` 模块 docstring）。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="SFT loss 测试需要 torch")

from graspo.flow.backend.native.models.qwen35_36.training_sft import (  # noqa: E402
    _Qwen35SFTTrainingMethods,
)

HIDDEN = 32
VOCAB = 256


class _IdentityNorm:
    """占位 ``model.norm``：恒等映射（本用例只考察 loss 的数值性质）。"""

    def __call__(self, hidden):
        return hidden


class _StubModel:
    """占位模型：``_compute_sft_loss`` 只需要可调用的 ``norm`` 与 ``lm_head.weight``。"""

    def __init__(self, lm_head) -> None:
        self.norm = _IdentityNorm()
        self.lm_head = lm_head


def _build_methods(dtype=torch.bfloat16):
    """把 mixin 的 ``_compute_sft_loss`` 绑到占位模型上（不实例化 adapter）。"""
    generator = torch.Generator().manual_seed(0)
    # 目标 token 的 logit 量级做大（~20）：真实训练末段 LM head 的 logit 就是这档，
    # CE 会掉到 1e-3 以下 —— 修复前正是在这个区间出现负读数。
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=generator) / HIDDEN**0.5)
    lm_head = lm_head.to(dtype)

    class _SUT(_Qwen35SFTTrainingMethods):
        def __init__(self) -> None:
            self.model = _StubModel(lm_head)

    return _SUT()


def _make_batch(dtype=torch.bfloat16, *, rows: int = 2, seq: int = 6):
    """构造 (hidden, labels)：prompt 位置 -100，response 位置为真 token。"""
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(rows, seq, HIDDEN, generator=generator).to(dtype)
    labels = torch.full((rows, seq), -100, dtype=torch.long)
    labels[:, seq // 2 :] = torch.randint(0, VOCAB, (rows, seq - seq // 2), generator=generator)
    return hidden, labels


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compute_sft_loss_is_non_negative(dtype):
    """真前向：``_compute_sft_loss`` 的输出 ``>= 0``（允许 ``-0.0``）。"""
    methods = _build_methods(dtype)
    hidden, labels = _make_batch(dtype)
    loss = methods._compute_sft_loss(hidden, labels)

    assert torch.isfinite(loss).all()
    pseudo = loss.detach().clone()
    assert pseudo.item() >= 0.0, f"loss={pseudo.item()!r} 为负"
    # 非平凡：这个 loss 确实算出了正的 CE（不是 0 蒙过去）
    assert pseudo.item() > 0.0


def test_every_token_log_prob_is_non_positive():
    """逐 token 口径：masked 后每个有效位置的 log-prob 都 ``<= 0``（无容差）。"""
    from graspo.ripple.loss import masked_token_log_probs_from_hidden

    methods = _build_methods(torch.bfloat16)
    hidden, labels = _make_batch(torch.bfloat16)
    normalized = methods.model.norm(hidden)
    log_probs = masked_token_log_probs_from_hidden(
        normalized[:, :-1].contiguous(),
        methods.model.lm_head.weight,
        labels[:, 1:].contiguous(),
        ignore_index=-100,
    )
    mask = labels[:, 1:] != -100
    assert bool((log_probs[mask] <= 0).all()), f"正 log_p：{log_probs[mask][log_probs[mask] > 0]}"
