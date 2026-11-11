"""负 loss 读数计数（``negative_loss_count``）的回归测试（CPU 可跑，需 torch）。

缺陷口径（两个既有检查都覆盖不到）：native SFT 的 CE 读数在 bf16 下**可能为负**
（实测 ``runs/diag-lr-L2/M047``：400 步里 3 步，step31/225 = −0.008606、
step246 = −0.004181）。成因是同一 logit 被两条 bf16 路径各算一遍（``ripple/loss.py``
的手写点积 vs cublas ``F.linear`` matmul）再经 5 次 chunk ``logaddexp`` 累加，两者各自的
±0.5 ULP 相减时抵消不掉；真实 CE 降到 ULP 量级时符号可翻。这是**合法舍入结果**，不是
符号错、不是累加错。而 ``skipped_nonfinite`` 的判据只有 ``not torch.isfinite(loss)``、
A6「数值健康」也只查 NaN/Inf ⇒ 负读数落在两个检查的口径之外，从来没被读到过。

本文件的要求：**只计数，不阻断**（把"读数不雅"升级成"训练失败"是错的判据）。

为什么测累加函数而不是端到端构造负 loss：
负读数由**机器相关的 1 ULP 数值抵消**产生，无法用固定种子在 CPU 上稳定构造——同一组
张量在另一种 kernel/另一台机器上可能回到 0 或正数，写成端到端断言会变成偶发失败
（而"不许改期望值"就意味着这种偶发失败必须永久背着）。因此：
1. 直接测生产用的累加函数 ``_updated_negative_loss_count``（非 PP 与 PP 两个训练循环
   都调它，累加逻辑本身即生产路径）；
2. 用**真实** ``_compute_sft_loss`` 的正常 batch 确认它不会误报（正向分支走真前向）。
未覆盖的是"训练循环把函数返回值接进 metrics"这段接线，由最后一条用例
（``test_build_sft_metrics_reports_negative_loss_count``）补上。

依赖：torch。本机（开发机无 torch）不会收集这些用例；须在带 torch 的环境/目标容器内跑：:

    python -m pytest tests/flow/backend/native/models/test_qwen35_36_sft_negative_loss_count.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="本测试需要 torch（目标容器内跑）")

from graspo.flow.backend.native.models.qwen35_36.training_sft import (  # noqa: E402
    _Qwen35SFTTrainingMethods,
    _updated_negative_loss_count,
)

VOCAB = 7
HIDDEN = 4

#: 实测负读数（``runs/diag-lr-L2/M047``）+ 正常正读数，用于累加测试。
_OBSERVED_NEGATIVE_VALUES = (-0.008606, -0.004181)


class _FakeNorm(torch.nn.Module):
    """占位 `model.norm`：恒等映射（`_compute_sft_loss` 只做线性代数）。"""

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


class _FakeConfig:
    effective_tuner_type = "lora"


class _MetricsHost(_Qwen35SFTTrainingMethods):
    """只提供 `_build_sft_metrics` 需要的属性的宿主（不实例化 adapter）。"""

    def __init__(self) -> None:
        self.config = _FakeConfig()

    def _current_lr(self) -> float:
        return 1e-4

    def _aggregate_rank_metrics(self, metrics: dict) -> dict:
        return metrics


def test_negative_loss_increments_count() -> None:
    """负读数 +1（实测值原样用，避免"只测 0 附近"的空断言）。"""
    for value in _OBSERVED_NEGATIVE_VALUES:
        assert _updated_negative_loss_count(0, value) == 1


def test_positive_loss_does_not_increment_count() -> None:
    assert _updated_negative_loss_count(0, 0.4321) == 0
    assert _updated_negative_loss_count(7, 0.4321) == 7


def test_zero_and_negative_zero_do_not_increment_count() -> None:
    """0.0 与 -0.0 都不是"负读数"（全 -100 的 batch 返回 -0.0，不得记成负）。"""
    assert _updated_negative_loss_count(0, 0.0) == 0
    assert _updated_negative_loss_count(0, -0.0) == 0


def test_count_accumulates_only_negative_micro_batches() -> None:
    """按 M047 的读数形状逐 micro-batch 累加：3 个负读数 ⇒ count=3。"""
    count = 0
    for value in (0.4123, -0.008606, 0.2044, -0.004181, 0.0098, -0.000012, 0.0031):
        count = _updated_negative_loss_count(count, value)
    assert count == 3


def test_real_compute_sft_loss_on_normal_batch_is_not_counted_negative() -> None:
    """正向分支走真前向：正常 batch 的 loss 为正 ⇒ 新增计数不得误报。"""

    class _SUT(_Qwen35SFTTrainingMethods):
        def __init__(self) -> None:
            self.model = _FakeModel(seed=0)

    methods = _SUT()
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[:, 1:5] = 0  # 有效 token 落在 shift 之后的位置
    hidden = torch.zeros(2, 6, HIDDEN)

    loss = methods._compute_sft_loss(hidden, labels)
    loss_value = float(loss.detach().cpu())

    assert torch.isfinite(loss)
    assert loss_value > 0.0
    assert _updated_negative_loss_count(0, loss_value) == 0


def test_build_sft_metrics_reports_negative_loss_count() -> None:
    """计数必须随 metrics 打出来（与 ``skipped_nonfinite`` 同一层、同一命名风格）。"""
    metrics = _Qwen35SFTTrainingMethods._build_sft_metrics(
        _MetricsHost(),
        sft_batch_count=1,
        optimizer_steps=1,
        skipped_nonfinite=0,
        negative_loss_count=2,
        loss_sum=0.5,
        micro_batch_count=4,
        grad_norm_sum=1.5,
        nonzero_grad_count=3,
        norm_before=1.0,
        norm_after=1.1,
        train_batch_started_at=0.0,
        micro_batch_forward_sec=0.1,
        backward_sec=0.2,
        optimizer_step_sec=0.3,
    )

    assert metrics["negative_loss_count"] == 2
    assert metrics["skipped_nonfinite"] == 0
