"""``masked_token_log_probs_from_hidden`` 的 bf16 路径覆盖（CPU 可跑，需 torch）。

**为什么要这个文件**：这个函数是 RL 与 SFT 共用的唯一实现，训练时输入是 bf16；
而修复前仓库里的用例全部走 fp32（参考实现显式 ``.float()``），**bf16 路径零覆盖**。

**修的是什么（不是"把 bf16 升成 fp32"）**：旧实现把目标 token 的 logit 另算一遍（手写
点积），并且 ``logsumexp`` 直接在 **bf16 logits** 上求。后者是主导项——``log_p`` 因此被
量化到 lse 的 bf16 网格上（``|lse| ≈ 12`` 时步长 0.0625），真实 CE 掉到网格以下时**读数
塌成 ±0**（下面 `test_small_ce_readings_do_not_collapse_to_zero` 把它锁死）。现在的实现
让目标 logit **与 lse 同源**（从同一块 logits 里 ``gather``），于是 ``logsumexp ≥ L_t``
恒成立、``log_p ≤ 0`` 由构造保证、且累加在 fp32 上做。

**四种改法的实测对照**（真机形状、fp64 参考）：**单一真相源是
``graspo/ripple/loss.py`` 里 ``masked_token_log_probs_from_hidden`` 的 docstring**，
这里不再抄第二份——要动"再升个 fp32 试试"的念头前先读那一份。

**误差界依据**（下面各用例的 atol）：返回值是 bf16，本身就带半个 ULP 的输出舍入，再叠加
目标 logit 在 bf16 网格上的半个 ULP；故界取 ``2 ULP(参考值的最大绝对值)``，留一倍余量。
fp32 累加本身的误差（相对 1e-7）比这更小。

**"改前红不红"如实记**（都在隔离环境里实测过，别凭印象改）：
- 改前（旧实现）**红三条**：parity 的**第 1 组参数**（`max|err| = 0.039011 > atol =
  0.03125`）、`test_small_ce_readings_do_not_collapse_to_zero`（**4/4 返回 `0.0`**）、
  `test_target_token_in_first_chunk`（`max|err| = 0.033184 > 0.03125`）。这三条在 CPU 与
  GPU（A800-SXM4-80GB / torch 2.11.0+cu130）上**数值逐位相同**。
- 非负性用例（本文件那条与 SFT 层那条）**改前不红**，原因**不是**"CPU 上两条路径给出同一个
  bf16 值"（**那是错的**：真机形状 H=5120/V=151936 下 CPU 上有 **418/1024** 个 token 两条
  路径给出**不同**的 bf16 值），而是**用例本身的 CE ≈ 7**（`min = -7.219`），远大于 bf16
  网格步长，读数量化不到塌。梯度对拍对改前也不红。
- 本文件的**旧用例对"A 只在两条路径上各升 fp32"那种改法全绿**（实测）；真正挡住它的是
  `test_small_ce_readings_do_not_collapse_to_zero`——A 在那条上给出 `+0.0057` 的正 log_p。
- ⚠ 验"改前"时**必须把测试文件拷到仓外隔离跑**：仓内 `tests/conftest.py` 会把仓库 `src`
  注入 `sys.path`，`PYTHONPATH=<旧树>` 指过去也照样被盖掉，结论全是假的。
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


def _converged_case(seed: int = 3, *, seq: int = 4, hidden_dim: int = 32, vocab: int = 256):
    """构造"已收敛"的 bf16 输入：目标 token 的 logit ≈ 12，真实 CE 落在 2e-3 一档。

    为什么要专门构造：只有 CE 掉到 **lse 的 bf16 网格步长以下**（``|lse| ≈ 12`` 时 0.0625）
    才会暴露旧实现"读数塌成 0"的缺陷；普通随机用例的 CE 是 O(1…10)，怎么算都塌不了。
    """
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, seq, hidden_dim, generator=generator).to(torch.bfloat16)
    lm_head = (torch.randn(vocab, hidden_dim, generator=generator) / hidden_dim**0.5).to(
        torch.bfloat16
    )
    token_ids = torch.randint(0, vocab, (1, seq), generator=generator)
    for position in range(seq):
        row = hidden[0, position].float()
        # 让该 token 成为 argmax 且 logit 恰为 12：logit = h·w = 12
        lm_head[token_ids[0, position]] = (row / row.norm() * (12.0 / row.norm())).to(
            torch.bfloat16
        )
    return hidden, lm_head, token_ids


def test_small_ce_readings_do_not_collapse_to_zero():
    """锁定真实缺陷：CE 小到 bf16 网格以下时，读数**不许塌成 0、更不许翻正**。

    改前（目标 logit 另算 + lse 在 bf16 logits 上求）：四个 token 全返回 ``0.0``（真值
    0.0022…0.0028 被量化掉了）。改后：``0.002274…0.002777``，误差 ≤ 3.3e-5。
    """
    hidden, lm_head, token_ids = _converged_case()
    log_probs = masked_token_log_probs_from_hidden(hidden, lm_head, token_ids, vocab_chunk_size=64)
    expected = _reference_log_probs_fp64(hidden, lm_head, token_ids, 64)

    # 真值确实是"小 CE 一档"（不是 0，也不是 O(1)）——断言的前提在这里
    assert bool((expected < -1e-3).all()) and bool((expected > -1e-2).all())
    # 读数不许塌成 0（改前 4/4 都返回 0.0）
    assert bool((log_probs <= -1e-3).all()), f"读数塌了：{log_probs.tolist()}"
    # 也不许翻正
    assert bool((log_probs <= 0).all())
    # 数值上贴着 fp64 参考
    assert float((log_probs.double() - expected).abs().max()) <= 1e-4


def test_target_token_in_first_chunk():
    """目标 token 全落在**第 0 块**：覆盖 ``selected_logits`` 的初始化分支。

    旧实现没有这个分支（目标 logit 一次算完），新实现是"逐块 where 覆盖"，第 0 块必须能
    正常初始化；现有用例（gradcheck 的 33/128、parity 的 64/256）都会跨到后面的块。
    """
    hidden, lm_head = _make_case(hidden_dim=32, vocab=128, seq=4, scale=1.0, seed=13)[:2]
    generator = torch.Generator().manual_seed(13)
    token_ids = torch.randint(0, 64, (2, 4), generator=generator)  # 全 < chunk=64 ⇒ 只在第 0 块

    log_probs = masked_token_log_probs_from_hidden(hidden, lm_head, token_ids, vocab_chunk_size=64)
    expected = _reference_log_probs_fp64(hidden, lm_head, token_ids, 64)

    assert bool((token_ids < 64).all())
    atol = 2.0 * _bf16_ulp(float(expected.abs().max()))
    assert float((log_probs.double() - expected).abs().max()) <= atol
    assert bool((log_probs <= 0).all())


def test_out_of_vocabulary_token_ids_fail_fast():
    """``token_ids >= V`` 必须立刻报错，而不是被块内下标 clamp 静默掩盖。

    旧实现走 ``index_select`` ⇒ 直接 ``IndexError``；新实现把 id 映射成块内下标，越界会被
    那个 ``clamp`` 悄悄变成"取到同块内别的 token 的 logit"——非负性质不破，但错输入进来了
    没人知道，所以显式 fail-fast。
    """
    hidden, lm_head, token_ids = _make_case(hidden_dim=8, vocab=16, seq=2, scale=1.0, seed=17)
    token_ids = token_ids.clone()
    token_ids[0, 0] = 16  # 恰好等于词表大小 ⇒ 越界

    with pytest.raises(AssertionError, match="token_ids 越界"):
        masked_token_log_probs_from_hidden(hidden, lm_head, token_ids, vocab_chunk_size=8)
