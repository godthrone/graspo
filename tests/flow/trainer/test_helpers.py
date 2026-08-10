"""Tests for ``graspo.flow.trainer.helpers`` — pure functions."""

import pytest

from graspo.flow.trainer.helpers import (
    epoch_stats_from_dict,
    epoch_stats_to_dict,
    train_stats_from_dict,
    train_stats_to_dict,
)
from graspo.ripple.monitoring.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
)


def test_train_stats_roundtrip_preserves_values():
    """train_stats_to_dict → train_stats_from_dict produces equivalent stats."""
    original = GraspoFlowTrainStats()
    original.total_groups = 16
    original.perfect_skipped = 3
    original.retries = 2
    original.invalid = 1
    original.trainable = 10
    original.optimized_steps = 5

    restored = train_stats_from_dict(train_stats_to_dict(original))

    assert restored.total_groups == 16
    assert restored.perfect_skipped == 3
    assert restored.retries == 2
    assert restored.invalid == 1
    assert restored.trainable == 10
    assert restored.optimized_steps == 5


def test_epoch_stats_roundtrip_preserves_values():
    """epoch_stats_to_dict → epoch_stats_from_dict produces equivalent stats."""
    original = GraspoFlowEpochStats()
    original.epoch = 2
    original.samples_seen = 100
    original.attempt_groups = 40
    original.trainable = 25
    original.perfect_skipped = 5

    restored = epoch_stats_from_dict(epoch_stats_to_dict(original))

    assert restored.epoch == 2
    assert restored.samples_seen == 100
    assert restored.attempt_groups == 40
    assert restored.trainable == 25
    assert restored.perfect_skipped == 5


def test_epoch_stats_from_dict_empty_returns_defaults():
    """epoch_stats_from_dict({}) returns stats with sensible defaults."""
    stats = epoch_stats_from_dict({})
    assert stats.epoch == 0
    assert stats.samples_seen == 0


def test_resume_config_snapshot_mismatch_raises():
    """防呆：checkpoint config_snapshot 与当前配置不一致 → 拒绝恢复。

    恢复训练用旧超参数语义会导致分布漂移而不自知。
    """
    from graspo.flow.trainer.checkpoint import CheckpointMixin

    class _Stub(CheckpointMixin):
        def __init__(self) -> None:
            self.backend_name = "graspoflow"
            self.config = _FakeConfig()

    class _FakeConfig:
        class Training:  # noqa: N801 测试桩（小写匹配 config.training）
            rollout_group_size = 8
            optimize_prompt_batch_size = 8
            rollout_max_retries = 5
            max_new_tokens = 2048

        training = Training()

    stub = _Stub()
    mismatched = {
        "format": "graspoflow-trainer-state",
        "config_snapshot": {
            "backend": "graspoflow",
            "rollout_group_size": 4,  # 与当前 8 不一致
            "optimize_prompt_batch_size": 8,
            "optimize_iterations_per_step": 1,
            "rollout_max_retries": 5,
            "max_new_tokens": 2048,
        },
    }
    with pytest.raises(RuntimeError, match="rollout_group_size"):
        stub._assert_resume_config_consistent(mismatched)

    consistent = dict(mismatched)
    consistent["config_snapshot"] = {
        "backend": "graspoflow",
        "rollout_group_size": 8,
        "optimize_prompt_batch_size": 8,
        "optimize_iterations_per_step": 1,
        "rollout_max_retries": 5,
        "max_new_tokens": 2048,
    }
    stub._assert_resume_config_consistent(consistent)  # 不抛错


def test_resume_config_snapshot_lr_mismatch_not_rejected():
    """防呆分级：lr 属于软键（可覆盖超参），不一致不拒绝启动。

    硬键（结构/分布语义）不一致 → 拒绝（见上面测试）；软键（lr/weight_decay）
    不一致 → 由 transformer_adapter 按配置覆盖并 WARNING 告知，此处不 raise。
    """
    from graspo.flow.trainer.checkpoint import CheckpointMixin

    class _Stub(CheckpointMixin):
        def __init__(self) -> None:
            self.backend_name = "graspoflow"
            self.config = _FakeConfig()

    class _FakeConfig:
        class Training:  # noqa: N801 测试桩（小写匹配 config.training）
            rollout_group_size = 8
            optimize_prompt_batch_size = 8
            rollout_max_retries = 5
            max_new_tokens = 2048
            learning_rate = 5.0e-07
            weight_decay = 0.01

        training = Training()

    stub = _Stub()
    state = {
        "format": "graspoflow-trainer-state",
        "config_snapshot": {
            "backend": "graspoflow",
            "rollout_group_size": 8,
            "optimize_prompt_batch_size": 8,
            "optimize_iterations_per_step": 1,
            "rollout_max_retries": 5,
            "max_new_tokens": 2048,
            # 软键：lr 与当前配置不一致，但不拒绝（配置优先 + WARNING）
            "learning_rate": 5.0e-06,
            "weight_decay": 0.01,
        },
    }
    stub._assert_resume_config_consistent(state)  # 不抛错


def test_checkpoint_trainer_state_snapshot_includes_lr_and_weight_decay():
    """config_snapshot 保存训练数值超参（lr/weight_decay），供 resume 审计。"""
    from graspo.flow.trainer.checkpoint import CheckpointMixin

    class _Stub(CheckpointMixin):
        def __init__(self) -> None:
            self.backend_name = "graspoflow"
            self.config = _FakeConfig()
            self.global_step = 24
            self.sample_index = 0
            self.total_samples = 322
            self.replay_buffer: list = []
            self.stats = GraspoFlowTrainStats()
            self.current_epoch_stats = GraspoFlowEpochStats()

    class _FakeConfig:
        class Training:  # noqa: N801 测试桩（小写匹配 config.training）
            rollout_group_size = 8
            optimize_prompt_batch_size = 8
            rollout_max_retries = 5
            max_new_tokens = 2048
            learning_rate = 5.0e-07
            weight_decay = 0.01

        training = Training()

    stub = _Stub()
    snapshot = stub._checkpoint_trainer_state(epoch=2)["config_snapshot"]

    assert snapshot["learning_rate"] == 5.0e-07
    assert snapshot["weight_decay"] == 0.01
