"""``masked_token_log_probs_from_hidden`` 的 bf16 路径覆盖（CPU 可跑，需 torch）。

**为什么要这个文件**：这个函数是 RL 与 SFT 共用的唯一实现，训练时输入是 bf16；
而修复前仓库里的用例全部走 fp32（参考实现显式 ``.float()``），**bf16 路径零覆盖**。

**修的是什么（不是"把 bf16 升成 fp32"）**：修复前目标 token 的 logit 走手写点积
``(hidden * selected).sum(-1)``，全词表 logsumexp 走分块 ``F.linear`` —— **同一个 logit
被两条路径各算一遍**，两条路径各自的舍入在相减时抵消不掉。现在的实现把目标 logit
**从同一块 logits 里 gather**，于是 ``logsumexp ≥ L_t`` 恒成立、``log_p ≤ 0`` 由构造保证。

**⚠ 一条实测记录（别再走回头路）**：只把两条路径"升成 fp32"是**不够**的——目标 logit
若仍另算一遍，lse 那条路径里的 logit 仍是 bf16 舍入值，±0.5 ULP 在 fp32 减法里照样
抵消不掉。真机实测（卡 2，H=5120/V=151936/bf16，1024 个 token，全 fp32 参考）：

    pre（两条 bf16 路径）      n_pos = 0/1024,   逐点 max|err| 5e-5 … 2.5e-3
    A（两条都升 fp32 但各算一遍）n_pos = 473…504/1024, 逐点 max_pos 最高 +0.025  ← 更糟
    B（连 GEMM 都用 fp32）      n_pos = 5…632,   max_pos ≤ 1.5e-4，但 loss 前向 42 ms → 367 ms
    C（目标 logit 与 lse 同源）  n_pos = 0/1024,   逐点 max|err| 1.3e-4        ← 本实现

**误差界依据**（下面各用例的 atol）：返回值是 bf16，本就带**半个 ULP 的输出舍入**；
再加上目标 logit 在 bf16 网格上的半个 ULP。故界取 ``2 ULP(返回值的最大绝对值)``，
实测落在 0.55…1.09 ULP，界留一倍余量。fp32 累加本身的误差（相对 1e-7）比这更小。

**关于"改前必红"**：本文件的 `test_bf16_log_probs_are_non_positive` 在 **CPU 上对修复前
的代码无判别力**——CPU 上两条 bf16 路径对同一个 logit 恰好给出同一个 bf16 值，舍入自己
抵消掉了（这正是该缺陷长期只在真机出现的原因）。它对**上面那条 A 方案**是红的
（实测 47% 的位置出现 ``log_p > 0``）。
"""

import math

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from torch.nn import functional as F  # noqa: E402, N812

from graspo.ripple.loss import masked_token_log_probs_from_hidden  # noqa: E402

#: bf16 的尾数位 ⇒ ULP = 2^(⌊log2|x|⌋ - 8)
_BF16_MANTISSA_BITS = 8


def _bf16_ulp(value: float) -> float:
    """``value`` 量级上的 bf16 ULP（返回值的舍入步长）。"""
    magnitude = abs(value)
    if magnitude == 0.0:
        return float(torch.finfo(torch.bfloat16).eps)
    return 2.0 ** (math.floor(math.log2(magnitude)) - _BF16_MANTISSA_BITS)


def _reference_log_probs_fp64(hidden, lm_head, token_ids, chunk):
    """全 fp64 参考实现（对拍基准）：完整 logits → log_softmax → gather。"""
    hidden64 = hidden.double()
    selected = (
        lm_head.double()
        .index_select(0, token_ids.reshape(-1))
        .view(*token_ids.shape, hidden.shape[-1])
    )
    selected_logits = (hidden64 * selected).sum(dim=-1)
    logsumexp = None
    for start in range(0, lm_head.shape[0], chunk):
        chunk_lse = torch.logsumexp(
            F.linear(hidden64, lm_head[start : start + chunk].double()), dim=-1
        )
        logsumexp = chunk_lse if logsumexp is None else torch.logaddexp(logsumexp, chunk_lse)
    return selected_logits - logsumexp


def _make_case(*, hidden_dim: int, vocab: int, seq: int, scale: float, seed: int):
    """随机 hidden/lm_head/id（vocab 跨多个 chunk 边界）。"""
    generator = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(2, seq, hidden_dim, generator=generator) * scale).to(torch.bfloat16)
    lm_head = (torch.randn(vocab, hidden_dim, generator=generator) * hidden_dim**-0.5).to(
        torch.bfloat16
    )
    token_ids = torch.randint(0, vocab, (2, seq), generator=generator)
    return hidden, lm_head, token_ids


def test_bf16_log_probs_are_non_positive():
    """``log_p <= 0`` 由构造保证（⛔ 断言里不许有 clamp/容差）。"""
    hidden, lm_head, token_ids = _make_case(hidden_dim=64, vocab=256, seq=6, scale=1.0, seed=0)
    log_probs = masked_token_log_probs_from_hidden(hidden, lm_head, token_ids, vocab_chunk_size=64)

    assert log_probs.dtype is torch.bfloat16
    assert bool((log_probs <= 0).all()), f"出现 log_p > 0：{log_probs[log_probs > 0]}"
    # 非平凡：确实算出了负的 log-prob（不是全 0 蒙过去）
    assert float(log_probs.min()) < -1.0


@pytest.mark.parametrize(
    ("hidden_dim", "vocab", "seq", "scale", "chunk"),
    [
        (64, 256, 6, 1.0, 64),  # 常规量级
        (512, 1024, 8, 1.0, 128),  # 更大 hidden/vocab，chunk 跨边界
        (128, 512, 4, 16.0, 64),  # 大 hidden 量级（真机 RMSNorm 输出可到这档）
    ],
)
def test_bf16_matches_fp64_reference(hidden_dim, vocab, seq, scale, chunk):
    """逐点对拍：bf16 路径 vs 全 fp64 参考，误差在 ``2 ULP`` 内。"""
    hidden, lm_head, token_ids = _make_case(
        hidden_dim=hidden_dim, vocab=vocab, seq=seq, scale=scale, seed=11
    )
    actual = masked_token_log_probs_from_hidden(hidden, lm_head, token_ids, vocab_chunk_size=chunk)
    expected = _reference_log_probs_fp64(hidden, lm_head, token_ids, chunk)

    atol = 2.0 * _bf16_ulp(float(expected.abs().max()))
    max_abs_err = float((actual.double() - expected).abs().max())
    assert max_abs_err <= atol, (
        f"max|err|={max_abs_err:.6g} 超出界 {atol:.6g}（{max_abs_err / atol:.2f}× 界）"
    )


def test_returns_input_dtype():
    """返回 dtype 必须等于输入 dtype（RL/SFT 下游依赖原 dtype）。"""
    hidden, lm_head, token_ids = _make_case(hidden_dim=16, vocab=64, seq=3, scale=1.0, seed=5)
    for dtype in (torch.bfloat16, torch.float32, torch.float64, torch.float16):
        out = masked_token_log_probs_from_hidden(
            hidden.to(dtype), lm_head.to(dtype), token_ids, vocab_chunk_size=16
        )
        assert out.dtype is dtype, f"{dtype} → {out.dtype}"


def test_masked_positions_are_zero_in_bf16():
    """ignore_index 位置在 bf16 下仍返回 0（SFT 的 prompt 部分）。"""
    hidden, lm_head, token_ids = _make_case(hidden_dim=32, vocab=128, seq=4, scale=1.0, seed=7)
    token_ids = token_ids.clone()
    token_ids[:, :2] = -100
    log_probs = masked_token_log_probs_from_hidden(
        hidden, lm_head, token_ids, ignore_index=-100, vocab_chunk_size=32
    )
    assert bool((log_probs[:, :2] == 0.0).all())
    assert bool((log_probs[:, 2:] < 0.0).all())


def _grad_of_hidden(dtype):
    hidden, lm_head, token_ids = _make_case(hidden_dim=16, vocab=64, seq=4, scale=1.0, seed=3)
    hidden = hidden.to(dtype).clone().requires_grad_(True)
    loss = masked_token_log_probs_from_hidden(
        hidden, lm_head.to(dtype), token_ids, vocab_chunk_size=16
    ).sum()
    loss.backward()
    assert hidden.grad is not None
    return hidden.grad.detach().double()


def test_bf16_gradient_matches_fp64_reference():
    """梯度对拍：bf16 对 hidden 的梯度 vs fp64 参考（本仓库原本的空白）。

    目标 logit 现在走**分块 GEMM + gather**（修复前走手写点积），反传路径变了，必须对拍。
    逐元素相对误差在近零分量上会被放大，故断言用**整张量的相对范数**与**余弦相似度**。
    """
    grad_bf16 = _grad_of_hidden(torch.bfloat16)
    grad_fp64 = _grad_of_hidden(torch.float64)

    rel_norm = float((grad_bf16 - grad_fp64).norm() / grad_fp64.norm())
    cosine = float(F.cosine_similarity(grad_bf16.flatten(), grad_fp64.flatten(), dim=0))
    assert rel_norm <= 2e-2, f"梯度相对范数误差 {rel_norm:.3e}"
    assert cosine >= 0.999, f"梯度余弦相似度 {cosine:.6f}"
    # 非平凡：梯度不全为 0
    assert float(grad_fp64.abs().max()) > 0.0
