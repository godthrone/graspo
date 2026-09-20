"""Tests for ``apply_config_optimizer_hyperparams`` — resume 配置优先覆盖逻辑。

回归：resume 时 ``optimizer.load_state_dict`` 会把 checkpoint 保存的
param_groups 整体替换（含 lr/weight_decay），配置的 learning_rate 被静默
覆盖不生效。修复后配置优先 + 返回被覆盖项供 WARNING 告知。
"""

from graspo.flow.adapters.transformer_adapter import apply_config_optimizer_hyperparams


class _FakeOptimizer:
    """轻量 optimizer 桩：只保留 param_groups 超参部分。"""

    def __init__(self, groups: list[dict]) -> None:
        self.param_groups = groups


class _FakeScheduler:
    """轻量 LambdaLR 桩：base_lrs 锚定曲线、_last_lr 缓存最近 lr。"""

    def __init__(self, base_lrs: list[float], last_lr: list[float]) -> None:
        self.base_lrs = list(base_lrs)
        self._last_lr = list(last_lr)


def test_lr_conflict_overridden_to_config():
    """lr 冲突（checkpoint 5e-06 vs config 5e-07）→ 覆盖为配置值并返回描述。"""
    optimizer = _FakeOptimizer([{"lr": 5e-06, "weight_decay": 0.01}])

    overridden = apply_config_optimizer_hyperparams(
        optimizer, None, learning_rate=5e-07, weight_decay=0.01
    )

    assert optimizer.param_groups[0]["lr"] == 5e-07
    assert optimizer.param_groups[0]["weight_decay"] == 0.01  # 一致不受影响
    assert len(overridden) == 1
    assert "learning_rate" in overridden[0]
    assert "5e-06" in overridden[0] and "5e-07" in overridden[0]


def test_lr_consistent_no_override():
    """lr 一致（checkpoint 值 == 配置值）→ 不覆盖、不返回。"""
    optimizer = _FakeOptimizer([{"lr": 5e-06, "weight_decay": 0.01}])

    overridden = apply_config_optimizer_hyperparams(
        optimizer, None, learning_rate=5e-06, weight_decay=0.01
    )

    assert optimizer.param_groups[0]["lr"] == 5e-06
    assert overridden == []


def test_weight_decay_conflict_overridden_to_config():
    """weight_decay 冲突 → 覆盖为配置值并返回描述。"""
    optimizer = _FakeOptimizer([{"lr": 5e-06, "weight_decay": 0.0}])

    overridden = apply_config_optimizer_hyperparams(
        optimizer, None, learning_rate=5e-06, weight_decay=0.01
    )

    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert len(overridden) == 1
    assert "weight_decay" in overridden[0]


def test_multiple_param_groups_all_overridden():
    """多 param_group（TP 分片等场景）→ 每个组都按配置覆盖。"""
    optimizer = _FakeOptimizer(
        [{"lr": 5e-06, "weight_decay": 0.01}, {"lr": 5e-06, "weight_decay": 0.01}]
    )

    overridden = apply_config_optimizer_hyperparams(
        optimizer, None, learning_rate=5e-07, weight_decay=0.01
    )

    assert all(g["lr"] == 5e-07 for g in optimizer.param_groups)
    assert len(overridden) == 2


def test_scheduler_base_lrs_anchored_to_config():
    """scheduler（LambdaLR）base_lrs 锚定配置 lr，_last_lr 缓存同步刷新。"""
    optimizer = _FakeOptimizer([{"lr": 5e-06, "weight_decay": 0.01}])
    scheduler = _FakeScheduler(base_lrs=[5e-06, 5e-06], last_lr=[5e-06, 5e-06])

    overridden = apply_config_optimizer_hyperparams(
        optimizer, scheduler, learning_rate=5e-07, weight_decay=0.01
    )

    assert scheduler.base_lrs == [5e-07, 5e-07]
    assert scheduler._last_lr == [5e-07, 5e-07]
    # lr + 两个 base_lrs 均记为覆盖（scheduler 未单独计数，但 lr 必须报告）
    assert any("learning_rate" in item for item in overridden)


def test_scheduler_consistent_no_change():
    """scheduler base_lrs 与配置一致 → 不动。"""
    scheduler = _FakeScheduler(base_lrs=[5e-07], last_lr=[5e-07])

    overridden = apply_config_optimizer_hyperparams(
        None, scheduler, learning_rate=5e-07, weight_decay=0.01
    )

    assert scheduler.base_lrs == [5e-07]
    assert overridden == []


def test_none_optimizer_and_scheduler_safe():
    """optimizer 与 scheduler 均为 None（constant 调度 + 无 trainable）→ 安全返回空。"""
    overridden = apply_config_optimizer_hyperparams(
        None, None, learning_rate=5e-07, weight_decay=0.01
    )

    assert overridden == []


def test_scheduler_without_base_lrs_attribute_safe():
    """无 base_lrs 属性的调度器（非 LambdaLR）→ 跳过，不报错。"""

    class _PlainScheduler:  # noqa: N801 测试桩
        pass

    overridden = apply_config_optimizer_hyperparams(
        None, _PlainScheduler(), learning_rate=5e-07, weight_decay=0.01
    )

    assert overridden == []
