"""F-13' 回归测试：SFT 分块 loss 的**累积权重口径**必须等于整批 token 均值（CPU 可跑）。

缺陷（2026-09-19，静态确认于 `task-h1-localize/report.md` 未完成项 5）：
``_pipeline_train_batch_sft`` 每个 chunk 调 ``_compute_sft_loss`` 得到**该 chunk 的
token 均值** ``m_i``，却用**样本数**占比加权：

    weight = mb["input_ids"].shape[0] / full_batch_size        # 旧（错误）
    scaled = Σ_i (n_i / N) · m_i

正确口径是整批 token 均值 ``Σ_i S_i / Σ_i L_i``（``L_i`` = 第 i chunk 的有效
label token 数）⇒ 权重必须是 ``L_i / Σ_j L_j``。

两者等价 ⟺ **所有 chunk 的有效 token 数相等**（micro_batch_size=1 时权重恰好是均匀
1/C，仍不等价）；偏差上界 = ``max(L_i) / min(L_i)``，方向 = 长 chunk 被高估。

本文件对应工作包验收：
1. **负向用例**：各 chunk 有效 token 数不等 ⇒ 旧口径与整批 token 均值不相等
   （`test_legacy_sample_count_weighting_is_not_the_global_token_mean`），
   而新口径必须相等（`test_unequal_valid_token_counts_weighted_sum_...`）。
2. 边界：单 chunk、各 chunk 等长（均应等价）。
3. 边界：某 chunk 有效 token 为 0 ⇒ 明确定义（权重 0、不除零、不静默）。

**"修复前必须失败"的实测证据**见同工作包 `report.md` §④：对修复前的代码
（`HEAD=613dbc5` 的 `training_sft.py:481`，把样本数权重复原进本测试）运行本文件，
`test_unequal_valid_token_counts_weighted_sum_matches_global_token_mean` 失败。

依赖：torch。本机（开发机无 torch）不会收集这些用例；须在目标容器内跑：
    docker run ... graspo-msswift:4.5.3 python -m pytest \\
      tests/flow/adapters/models/test_qwen35_36_sft_accumulation.py -q -p no:cacheprovider
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="F-13' 测试需要 torch（目标容器内跑）")

from graspo.flow.adapters.models.qwen35_36.training_sft import (  # noqa: E402
    _count_sft_valid_tokens,
    _normalized_accumulation_weights,
)

VOCAB = 7  # 小词表，便于让"短 chunk"与"长 chunk"的 token 均值明显不同
HIDDEN = 4


class _FakeNorm(torch.nn.Module):
    """占位 `model.norm`：恒等映射（`masked_token_log_probs_from_hidden` 只做线性代数）。"""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _FakeModel(torch.nn.Module):
    """占位模型：只提供 `_compute_sft_loss` 需要的 `norm` + `lm_head.weight`。"""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.norm = _FakeNorm()
        generator = torch.Generator().manual_seed(seed)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=generator))


def _build_methods():
    """把 mixin 的 `_compute_sft_loss` 绑定到占位模型上（不实例化 adapter）。"""
    from graspo.flow.adapters.models.qwen35_36.training_sft import _Qwen35SFTTrainingMethods

    class _SUT(_Qwen35SFTTrainingMethods):
        def __init__(self, seed: int = 0) -> None:
            self.model = _FakeModel(seed)

    return _SUT


def _make_chunk(
    *, label_lengths: list[int], hidden_values: list[float], seq_len: int | None = None
):
    """构造一个 micro-batch（= PP 的一个 chunk）。

    ★ 关键：`_compute_sft_loss` 内部先做因果 shift ``labels[:, 1:]``，因此有效 label
    必须落在下标 ``1 .. label_length``；下标 0 的 label 会被 shift 丢掉。这里写
    ``labels[row, 1:1+label_lengths[row]] = 0``，与真实 collate（prompt 位置 -100、
    response 位置为真 token）在 shift 后的有效位置一致。

    - ``hidden`` 每行是常量 ``hidden_values[row]``：让"每个 token 的 loss"取不同的值，
      这样**各 chunk 的 token 均值才会不同**——否则（全 0 hidden）逐 token loss 恒为
      ``log(V)``，两种口径会恰好数值相同，负向测试就失去意义。

    返回 ``(hidden, labels)``；有效 token 数由 `_count_sft_valid_tokens` 统一给出
    （**不在这里手算**，避免测试自己引入口径分歧）。
    """
    rows = len(label_lengths)
    assert len(hidden_values) == rows
    length = seq_len if seq_len is not None else (max(label_lengths) + 2 if label_lengths else 2)
    labels = torch.full((rows, length), -100, dtype=torch.long)
    hidden = torch.zeros(rows, length, HIDDEN)
    for row, (label_length, hidden_value) in enumerate(
        zip(label_lengths, hidden_values, strict=True)
    ):
        labels[row, 1 : 1 + label_length] = 0
        hidden[row, :] = float(hidden_value)
    return hidden, labels


def _chunk_losses(methods, chunks) -> list[torch.Tensor]:
    """逐 chunk 调**真实** `_compute_sft_loss`（= 生产路径每 chunk 得到的东西）。"""
    return [methods._compute_sft_loss(hidden, labels) for hidden, labels in chunks]


def _global_token_mean(methods, chunks) -> torch.Tensor:
    """整批 token 均值：把各 chunk 的行**拼成一整批**后一次性算 loss。

    这就是 "整批 token 均值" 的唯一真相源——`_compute_sft_loss` 内部
    ``(log_probs * mask).sum() / mask.sum()`` 在跨 chunk 的大批上求的正是这个量。
    各 chunk 的 seq_len 可能不同：hidden 补 0、labels 补 -100（不进分子也不进分母）。
    """
    max_seq = max(labels.shape[1] for _, labels in chunks)
    hidden_rows = []
    label_rows = []
    for hidden, labels in chunks:
        pad = max_seq - labels.shape[1]
        hidden_rows.append(torch.nn.functional.pad(hidden, (0, 0, 0, pad)))
        label_rows.append(torch.nn.functional.pad(labels, (0, pad), value=-100))
    return methods._compute_sft_loss(torch.cat(hidden_rows, dim=0), torch.cat(label_rows, dim=0))


#: 最近一次 ``_counts()`` 的原始读数（修复前的变体会把权重换成样本数口径，但**统计
#: 本身**仍然真实——这样断言 "各 chunk token 数不等" 仍然基于真实读数）。
_LAST_VALID_TOKEN_COUNTS: list[int] = []


def _counts(chunks) -> list[int]:
    """统计并记录各 chunk 的有效 token 数（同时也刷新 `_LAST_VALID_TOKEN_COUNTS`）。"""
    global _LAST_VALID_TOKEN_COUNTS
    _LAST_VALID_TOKEN_COUNTS = [_count_sft_valid_tokens(labels) for _, labels in chunks]
    return list(_LAST_VALID_TOKEN_COUNTS)


def _weights_from_last_counts() -> list[float]:
    """按上一次 `_counts()` 的读数求累积权重（生产路径的等价物）。"""
    return _normalized_accumulation_weights(list(_LAST_VALID_TOKEN_COUNTS))


def _weighted_sum(losses, weights) -> float:
    return sum(float(loss.detach()) * weight for loss, weight in zip(losses, weights, strict=True))


# 负向用例的固定构造：短 chunk 的 token 均值明显低于长 chunk。
#   chunk0: 1 个有效 token、hidden 0.0
#   chunk1: 1 个有效 token、hidden 0.0
#   chunk2: 5 个有效 token、hidden 1.0
_UNEQUAL_CHUNKS = (
    dict(label_lengths=[1], hidden_values=[0.0]),
    dict(label_lengths=[1], hidden_values=[0.0]),
    dict(label_lengths=[5], hidden_values=[1.0]),
)


def _unequal_chunks():
    return [_make_chunk(**kwargs) for kwargs in _UNEQUAL_CHUNKS]


# ── 负向测试：各 chunk 有效 token 数不等 ─────────────────────────────────────


def test_unequal_valid_token_counts_weighted_sum_matches_global_token_mean():
    """★ 修复后必须通过：分块加权和 == 整批 token 均值（各 chunk 有效 token 数不等）。

    修复前此断言失败（旧口径给出 ``Σ (n_i/N)·m_i``，与整批 token 均值不等）。
    """
    methods = _build_methods()()
    chunks = _unequal_chunks()
    counts = _counts(chunks)
    assert len(set(counts)) > 1, f"用例不满足'各 chunk 有效 token 数不等'：{counts}"

    losses = _chunk_losses(methods, chunks)
    weights = _weights_from_last_counts()
    weighted_sum = _weighted_sum(losses, weights)
    global_mean = float(_global_token_mean(methods, chunks).detach())

    assert weighted_sum == pytest.approx(global_mean, abs=1e-5), (
        f"分块加权和 {weighted_sum} != 整批 token 均值 {global_mean}；"
        f"counts={counts}, weights={weights}, chunk_losses={[float(x) for x in losses]}"
    )


def test_legacy_sample_count_weighting_is_not_the_global_token_mean():
    """★ 负向测试：旧口径（样本数加权）在"各 chunk 有效 token 数不等"时**必然**偏离。

    本用例把**修复前那一行**的公式逐字重算（``weight = 样本数 / 全批样本数``），并把
    "旧口径 ≠ 整批 token 均值"钉成断言 —— 这是缺陷真实存在的活证据，与修复后的实现
    无关（所以修复前后都通过）。若有人把权重改回样本数，本用例仍会通过，而
    `test_unequal_valid_token_counts_weighted_sum_matches_global_token_mean` 会失败。
    """
    methods = _build_methods()()
    chunks = _unequal_chunks()
    counts = _counts(chunks)
    losses = _chunk_losses(methods, chunks)

    # 旧公式（training_sft.py 修复前第 481 行）：weight = mb["input_ids"].shape[0] / full_batch_size
    sample_counts = [hidden.shape[0] for hidden, _ in chunks]
    legacy_weights = [count / sum(sample_counts) for count in sample_counts]
    legacy_weighted_sum = _weighted_sum(losses, legacy_weights)

    global_mean = float(_global_token_mean(methods, chunks).detach())
    relative_bias = abs(legacy_weighted_sum - global_mean) / global_mean

    assert legacy_weighted_sum != pytest.approx(global_mean, abs=1e-5), (
        "旧口径在本用例下与整批 token 均值相等 ⇒ 用例构造不满足'各 chunk 不等'，"
        f"需重构造（counts={counts}）"
    )
    # 偏差必须落在理论界 max(L_i)/min(L_i) 之内（相对偏差 ≤ (max/min - 1) 量级）。
    max_over_min = max(counts) / min(counts)
    assert relative_bias <= max_over_min, (
        f"相对偏差 {relative_bias:.3f} 超出理论界 max(L)/min(L)={max_over_min}"
    )
    assert relative_bias > 0.1, f"偏差太小、负向用例不显著：{relative_bias:.3f}"


# ── 边界：单 chunk / 各 chunk 等长（都应等价） ────────────────────────────────


def test_single_chunk_weight_is_one_and_matches_global_token_mean():
    """单 chunk（chunk_count=1）：权重必须为 1.0，与整批 token 均值逐位等价。"""
    methods = _build_methods()()
    chunks = [_make_chunk(label_lengths=[3, 1], hidden_values=[0.0, 1.0])]
    counts = _counts(chunks)
    assert _weights_from_last_counts() == [1.0]

    losses = _chunk_losses(methods, chunks)
    weighted_sum = _weighted_sum(losses, [1.0])
    assert weighted_sum == pytest.approx(
        float(_global_token_mean(methods, chunks).detach()), abs=1e-6
    )


def test_equal_valid_token_counts_make_token_and_sample_weighting_equivalent():
    """各 chunk 有效 token 数相等：两种口径**都**等于整批 token 均值（等价条件）。"""
    methods = _build_methods()()
    chunks = [
        _make_chunk(label_lengths=[2], hidden_values=[0.0]),
        _make_chunk(label_lengths=[2], hidden_values=[1.0]),
        _make_chunk(label_lengths=[2], hidden_values=[0.5]),
    ]
    counts = _counts(chunks)
    assert len(set(counts)) == 1, f"用例构造错误：{counts}"

    losses = _chunk_losses(methods, chunks)
    sample_counts = [hidden.shape[0] for hidden, _ in chunks]
    global_mean = float(_global_token_mean(methods, chunks).detach())

    for weights in (
        _weights_from_last_counts(),
        [count / sum(sample_counts) for count in sample_counts],
    ):
        assert _weighted_sum(losses, weights) == pytest.approx(global_mean, abs=1e-5)


def test_equal_token_counts_but_unequal_sample_counts_token_weighting_is_exact():
    """反例边界：**样本数不等**但**有效 token 数相等**时，token 权重仍与整批均值逐位相等。

    ⇒ "各 chunk 样本数不等"本身不是缺陷的充要条件，**有效 token 数不等**才是。
    ⇒ 反过来，**样本数权重在任何"有效 token 数不等"的多 chunk 情形都是错的**
      （见下一条用例），而 token 权重恒为 ``L_i/ΣL``。
    """
    methods = _build_methods()()
    chunks = [
        _make_chunk(label_lengths=[1], hidden_values=[0.0]),  # 1 样本、1 有效 token
        _make_chunk(label_lengths=[1, 0], hidden_values=[0.0, 0.0]),  # 2 样本、1 有效 token
        _make_chunk(label_lengths=[1], hidden_values=[1.0]),  # 1 样本、1 有效 token
    ]
    counts = _counts(chunks)
    sample_counts = [hidden.shape[0] for hidden, _ in chunks]
    assert counts == [1, 1, 1], f"用例构造错误（各 chunk 有效 token 数应相等）：{counts}"
    assert len(set(sample_counts)) > 1, f"用例构造错误：样本数应不等，实为 {sample_counts}"

    losses = _chunk_losses(methods, chunks)
    global_mean = float(_global_token_mean(methods, chunks).detach())
    assert _weighted_sum(losses, _weights_from_last_counts()) == pytest.approx(
        global_mean, abs=1e-5
    )


def test_sample_count_weighting_is_wrong_even_when_token_counts_are_equal():
    """★ 负向用例（第二条）：**有效 token 数相等**时，样本数权重反而是错的。

    上一条用例证明了"token 数相等 ⇒ 两种口径等价"**不成立**——它只在"每个样本的
    有效 token 数也相等"（即 chunk 内样本数也相等）时才成立。本用例用"chunk2 有 2 个
    样本但只有 1 个有效 token"把这个边界钉死：样本数权重给出 ``0.25·m0+0.5·m1+0.25·m2``，
    而整批 token 均值是三者等权平均（每个 chunk 恰好 1 个有效 token）。

    结论：**累积权重必须来自有效 label token 数**（`_count_sft_valid_tokens`），
    不能来自 `input_ids.shape[0]`（样本数）——后者在 RL 路径正确，在本路径错误。
    """
    methods = _build_methods()()
    chunks = [
        _make_chunk(label_lengths=[1], hidden_values=[0.0]),
        _make_chunk(label_lengths=[1, 0], hidden_values=[0.0, 0.0]),
        _make_chunk(label_lengths=[1], hidden_values=[1.0]),
    ]
    counts = _counts(chunks)
    sample_counts = [hidden.shape[0] for hidden, _ in chunks]
    assert counts == [1, 1, 1]
    assert sample_counts == [1, 2, 1]

    losses = _chunk_losses(methods, chunks)
    global_mean = float(_global_token_mean(methods, chunks).detach())
    legacy_weights = [count / sum(sample_counts) for count in sample_counts]
    assert _weighted_sum(losses, legacy_weights) != pytest.approx(global_mean, abs=1e-5), (
        "样本数权重在本用例下竟与整批 token 均值相等 ⇒ 用例构造失效"
    )


# ── 边界：某 chunk 有效 token 为 0 ───────────────────────────────────────────


def test_zero_valid_token_chunk_gets_zero_weight_without_division_by_zero():
    """某 chunk 有效 token 为 0 ⇒ 权重**明确定义**为 0.0，不除零、不静默。"""
    assert _normalized_accumulation_weights([0, 3]) == [0.0, 1.0]

    methods = _build_methods()()
    empty_chunk = _make_chunk(label_lengths=[0], hidden_values=[1.0])
    assert _counts([empty_chunk]) == [0]
    # 真实 `_compute_sft_loss` 对全 -100 输入返回 0.0（`mask.sum().clamp_min(1)` 防除零）
    empty_loss = methods._compute_sft_loss(*empty_chunk)
    assert float(empty_loss.detach()) == 0.0

    nonempty_chunk = _make_chunk(label_lengths=[3], hidden_values=[1.0])
    nonempty_loss = methods._compute_sft_loss(*nonempty_chunk)
    _counts([empty_chunk, nonempty_chunk])
    weights = _weights_from_last_counts()
    # 0 权重 chunk 的贡献恒为 0（把 0/0 带进 loss 的路径被权重掐断）
    assert float(empty_loss.detach()) * weights[0] == 0.0

    # 反向仍然安全：Linear 的参数梯度图存在，不会报 "does not require grad"；
    # 且 0 权重 chunk 不污染其它 chunk 的梯度。
    before = methods.model.lm_head.weight.detach().clone()
    methods.model.lm_head.weight.grad = None
    (empty_loss * weights[0] + nonempty_loss * weights[1]).backward()
    grad = methods.model.lm_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.norm()) > 0.0, "非零权重 chunk 必须真的产生梯度"
    # loss.backward() 只写 .grad，不改参数本身
    assert torch.equal(before, methods.model.lm_head.weight.detach())


def test_all_zero_valid_tokens_is_rejected_not_silently_zeroed():
    """整批有效 token 全为 0 ⇒ 归一化必须**显式拒绝**（返回全 0 权重）。

    调用方（`_pipeline_train_batch_sft`）在此情形 `raise RuntimeError`，**不得**以
    0 loss 静默推进 optimizer。此处覆盖纯函数契约：不得除零（不得产生 NaN）。
    """
    weights = _normalized_accumulation_weights([0, 0])
    assert weights == [0.0, 0.0]
    assert all(weight == weight for weight in weights), "不得产生 NaN"
    assert sum(weights) == 0.0


def test_negative_valid_token_count_is_rejected():
    """防呆（§2.3）：统计口径写错（负数）时立即暴露，不得静默归一。"""
    with pytest.raises(ValueError, match="不得为负"):
        _normalized_accumulation_weights([3, -1])


def test_weights_always_sum_to_one_when_tokens_exist():
    """不变量：只要存在有效 token，各 chunk 权重之和恒为 1.0（C=1 时为 [1.0]）。"""
    for counts in ([1], [5, 5], [0, 7], [1, 1, 5], [0, 0, 2, 3]):
        weights = _weights_from_last_counts()
        assert sum(weights) == pytest.approx(1.0)
        assert all(weight >= 0.0 for weight in weights)
