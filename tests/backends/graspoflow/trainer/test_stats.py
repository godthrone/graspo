"""Tests for training stats data structures — BADGE §11.1."""

import pytest

from graspo.backends.graspoflow.trainer.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
    _AttemptRecord,
    _QueuedSample,
)
from graspo.core.schema import Sample


def _make_sample(msg: str = "hello"):
    return Sample(
        messages=[{"role": "user", "content": msg}],
        targets=[{"id": "t1", "output": {"content": {"key": "value"}}}],
    )


# ── GraspoFlowTrainStats ──────────────────────────────────────────────────


def test_train_stats_defaults():
    stats = GraspoFlowTrainStats()
    assert stats.total_groups == 0
    assert stats.perfect_skipped == 0
    assert stats.retries == 0
    assert stats.invalid == 0
    assert stats.invalid_no_preference_gap == 0
    assert stats.trainable == 0
    assert stats.optimized_steps == 0


def test_train_stats_can_be_updated():
    stats = GraspoFlowTrainStats()
    stats.total_groups += 10
    stats.trainable += 3
    stats.optimized_steps += 5
    assert stats.total_groups == 10
    assert stats.trainable == 3
    assert stats.optimized_steps == 5


def test_train_stats_perfect_skip_counting():
    stats = GraspoFlowTrainStats(
        total_groups=100,
        perfect_skipped=30,
        trainable=50,
        invalid=15,
        invalid_no_preference_gap=5,
    )
    # Verify decomposition: 30 + 50 + 15 + 5 = 100
    assert (
        stats.perfect_skipped + stats.trainable + stats.invalid + stats.invalid_no_preference_gap
        == 100
    )


# ── GraspoFlowEpochStats ──────────────────────────────────────────────────


def test_epoch_stats_defaults():
    stats = GraspoFlowEpochStats()
    assert stats.epoch == 0
    assert stats.samples_seen == 0
    assert stats.best_reward == 0.0


def test_epoch_stats_accumulate_reward():
    stats = GraspoFlowEpochStats()
    stats.reward_mean_sum += 0.8
    stats.reward_mean_sum += 0.9
    stats.content_mean_sum += 0.7
    assert stats.reward_mean_sum == pytest.approx(1.7)
    assert stats.content_mean_sum == pytest.approx(0.7)


def test_epoch_stats_tracks_best_reward():
    stats = GraspoFlowEpochStats()
    stats.best_reward = 0.0
    stats.best_reward = max(stats.best_reward, 0.5)
    stats.best_reward = max(stats.best_reward, 0.9)
    assert stats.best_reward == 0.9


def test_epoch_stats_counts_increment():
    stats = GraspoFlowEpochStats()
    stats.attempt_groups += 20
    stats.completion_count += 160  # G=8
    assert stats.attempt_groups == 20
    assert stats.completion_count == 160


# ── _QueuedSample ──────────────────────────────────────────────────────────


def test_queued_sample_holds_reference():
    sample = _make_sample()
    queued = _QueuedSample(sample=sample, retry_count=0)
    assert queued.sample is sample
    assert queued.retry_count == 0
    assert queued.attempts == []


def test_queued_sample_tracks_retry():
    queued = _QueuedSample(sample=_make_sample(), retry_count=3)
    assert queued.retry_count == 3


# ── _AttemptRecord ────────────────────────────────────────────────────────


def test_attempt_record_stores_rewards():
    record = _AttemptRecord(
        sample=_make_sample(),
        generation=None,
        parsed_completions=[],
        rewards=[0.8, 0.9, 0.5, 1.0],
        content_scores=[0.8, 0.9, 0.5, 1.0],
        all_right=[False, True, False, True],
        reward_details=[],
        decision="trainable_max_correct",
        retry_count=0,
        readable={},
        timing={},
    )
    assert len(record.rewards) == 4
    assert record.rewards[1] == 0.9
    assert record.all_right[1] is True
    assert record.all_right[0] is False


def test_attempt_record_decision():
    record = _AttemptRecord(
        sample=_make_sample(),
        generation=None,
        parsed_completions=[],
        rewards=[1.0, 1.0, 1.0, 1.0],
        content_scores=[1.0, 1.0, 1.0, 1.0],
        all_right=[True, True, True, True],
        reward_details=[],
        decision="perfect_skip",
        retry_count=0,
        readable={},
        timing={},
    )
    assert record.decision == "perfect_skip"
