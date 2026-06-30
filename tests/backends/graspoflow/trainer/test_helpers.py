"""Tests for ``graspo.backends.graspoflow.trainer.helpers`` — pure functions."""

from graspo.backends.graspoflow.trainer.helpers import (
    epoch_stats_from_dict,
    epoch_stats_to_dict,
    train_stats_from_dict,
    train_stats_to_dict,
)
from graspo.backends.graspoflow.trainer.stats import (
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
