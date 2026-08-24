"""Tests for ``_build_scheduler`` — lr_scheduler.decay_steps 衰减语义。

v0.23.0：max_steps 已移除，训练长度只由 max_epochs 控制；decay_steps 是
纯调度参数（warmup 后的衰减跨度，optimizer-step 粒度），衰减完成后 lr
保持 min_lr 不变。
"""

import pytest
import torch

from graspo.core.schema import TrainingConfig
from graspo.flow.adapters.transformer_adapter import TransformerAdapter


class _FakeGraspoFlowConfig:
    """最小桩：只提供 _build_scheduler 依赖的 dp_size。"""

    dp_size: int = 1


class _FakeRootConfig:
    """包装 TrainingConfig 成 ``config.training`` 形态（_build_scheduler 的访问路径）。"""

    def __init__(self, training: TrainingConfig) -> None:
        self.training = training
        self.graspoflow = _FakeGraspoFlowConfig()


class _StubAdapter:
    """最小桩：只提供 _build_scheduler 依赖的 config 与 optimizer。"""

    def __init__(self, config: _FakeRootConfig, optimizer: torch.optim.Optimizer | None) -> None:
        self.config = config
        self.optimizer = optimizer


def _make_scheduler(learning_rate: float = 5e-7, **sched_kwargs):
    training = TrainingConfig(
        max_epochs=100,
        learning_rate=learning_rate,
        lr_scheduler=sched_kwargs,
    )
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([param], lr=learning_rate)
    stub = _StubAdapter(_FakeRootConfig(training), optimizer)
    return TransformerAdapter._build_scheduler(stub), optimizer


def _lr_after_steps(scheduler, optimizer, n: int) -> float:
    """手动推进 n 次 scheduler.step() 后读取 optimizer 当前 lr。"""
    for _ in range(n):
        scheduler.step()
    return float(optimizer.param_groups[0]["lr"])


def test_constant_returns_none():
    scheduler, _ = _make_scheduler(type="constant")
    assert scheduler is None


def test_cosine_requires_decay_steps():
    with pytest.raises(ValueError, match="decay_steps"):
        _make_scheduler(type="cosine", decay_steps=0)


def test_cosine_decays_to_min_lr_then_holds():
    """cosine：decay_steps 步内衰减到 min_lr，之后保持（progress clamp）。"""
    scheduler, optimizer = _make_scheduler(
        learning_rate=5e-7,
        type="cosine", decay_steps=170, min_lr_ratio=0.2,
    )
    base, min_lr = 5e-7, 1e-7

    # 衰减中途：lr 介于 base 与 min 之间且单调下降
    mid = _lr_after_steps(scheduler, optimizer, 85)
    assert min_lr < mid < base

    # 衰减终点：正好到达 min_lr
    end = _lr_after_steps(scheduler, optimizer, 85)  # 累计 170 步
    assert end == pytest.approx(min_lr, rel=1e-6)

    # 衰减后保持：再多走 50 步仍为 min_lr
    held = _lr_after_steps(scheduler, optimizer, 50)
    assert held == pytest.approx(min_lr, rel=1e-6)


def test_cosine_warmup_then_decay():
    """warmup 阶段线性爬升，之后才开始衰减。"""
    scheduler, optimizer = _make_scheduler(
        learning_rate=5e-7,
        type="cosine", warmup_steps=10, decay_steps=100, min_lr_ratio=0.0,
    )
    # warmup 中段：接近 base_lr（线性 50%）
    mid_warmup = _lr_after_steps(scheduler, optimizer, 5)
    assert mid_warmup == pytest.approx(2.5e-7, rel=1e-6)
    # warmup 终点：到达 base_lr
    end_warmup = _lr_after_steps(scheduler, optimizer, 5)  # 累计 10 步
    assert end_warmup == pytest.approx(5e-7, rel=1e-6)


def test_linear_decays_to_min_lr():
    """linear：decay_steps 后到达 min_lr 并保持。"""
    scheduler, optimizer = _make_scheduler(
        learning_rate=5e-7,
        type="linear", decay_steps=100, min_lr_ratio=0.2,
    )
    end = _lr_after_steps(scheduler, optimizer, 100)
    assert end == pytest.approx(1e-7, rel=1e-6)
    held = _lr_after_steps(scheduler, optimizer, 30)
    assert held == pytest.approx(1e-7, rel=1e-6)


def test_scheduler_does_not_read_max_epochs():
    """调度跨度与训练长度（max_epochs）完全解耦：不同 max_epochs 同曲线。"""
    def _decay_curve(max_epochs: int) -> float:
        training = TrainingConfig(
            max_epochs=max_epochs,
            learning_rate=5e-7,
            lr_scheduler={"type": "cosine", "decay_steps": 100, "min_lr_ratio": 0.2},
        )
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.AdamW([param], lr=5e-7)
        scheduler = TransformerAdapter._build_scheduler(
            _StubAdapter(_FakeRootConfig(training), optimizer)
        )
        for _ in range(50):
            scheduler.step()
        return float(optimizer.param_groups[0]["lr"])

    assert _decay_curve(max_epochs=10) == _decay_curve(max_epochs=1000)
