"""Tests for config schema validation — BADGE §11.1."""

import pytest
from pydantic import ValidationError

from graspo.core.schema import (
    GraspoConfig,
    Sample,
    TrainingConfig,
    _generate_run_name,
)

# ── TrainingConfig extra="forbid" ─────────────────────────────────────────────


def test_training_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TrainingConfig(nonexistent_field=42)


def test_training_config_rejects_removed_legacy_field():
    """pydantic ``extra="forbid"`` 自动拒绝任意旧字段，无需手写黑名单。"""
    with pytest.raises(ValidationError, match="total_epochs"):
        TrainingConfig(total_epochs=10)


def test_graspo_config_rejects_removed_legacy_field_in_nested_section():
    with pytest.raises(ValidationError, match="total_epochs"):
        GraspoConfig.from_dict({"training": {"total_epochs": 10}})


# ── GraspoConfig output_dir / run_name 默认值 ────────────────────────────────


def test_from_dict_empty_output_dir_generates_default():
    cfg = GraspoConfig.from_dict({})
    assert cfg.training.output_dir.startswith("outputs/graspo_")
    assert cfg.training.run_name.startswith("graspo_")


def test_from_dict_explicit_output_dir_without_run_name():
    cfg = GraspoConfig.from_dict({"training": {"output_dir": "outputs/my_run"}})
    assert cfg.training.output_dir == "outputs/my_run"
    assert cfg.training.run_name == "my_run"


def test_from_dict_explicit_run_name_without_output_dir():
    cfg = GraspoConfig.from_dict({"training": {"run_name": "experiment_42"}})
    assert cfg.training.output_dir == "outputs/experiment_42"
    assert cfg.training.run_name == "experiment_42"


def test_from_dict_both_explicit_output_dir_and_run_name():
    cfg = GraspoConfig.from_dict(
        {"training": {"output_dir": "/tmp/custom", "run_name": "custom_name"}}
    )
    assert cfg.training.output_dir == "/tmp/custom"
    assert cfg.training.run_name == "custom_name"


def test_run_name_cache_is_consistent():
    name1 = _generate_run_name()
    name2 = _generate_run_name()
    assert name1 == name2


# ── Default values ───────────────────────────────────────────────────────────


def test_training_config_default_seed_is_42():
    cfg = TrainingConfig()
    assert cfg.seed == 42


def test_training_config_default_rollout_group_size():
    cfg = TrainingConfig()
    assert cfg.rollout_group_size == 8


def test_training_config_replay_buffer_threshold_is_derived():
    cfg = TrainingConfig(optimize_prompt_batch_size=4, rollout_group_size=8)
    assert cfg.replay_buffer_optimize_threshold == 32


# ── Sample ───────────────────────────────────────────────────────────────────


def test_sample_expects_tool_calls_when_targets_have_tool_calls():
    sample = Sample(
        messages=[{"role": "user", "content": "test"}],
        targets=[{"id": "t1", "output": {"tool_calls": [{"name": "search", "arguments": {}}]}}],
    )
    assert sample.expects_tool_calls is True


def test_sample_expects_tool_calls_false_for_content_targets():
    sample = Sample(
        messages=[{"role": "user", "content": "test"}],
        targets=[{"id": "t1", "output": {"content": {"key": "value"}}}],
    )
    assert sample.expects_tool_calls is False


def test_sample_is_frozen_after_creation():
    sample = Sample(
        messages=[{"role": "user", "content": "test"}],
        targets=[{"id": "t1", "output": {"content": {"key": "value"}}}],
    )
    with pytest.raises(ValidationError):
        sample.messages = []


def test_sample_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Sample(
            messages=[{"role": "user", "content": "test"}],
            targets=[{"id": "t1", "output": {"content": {"key": "value"}}}],
            bogus_field="should_fail",
        )


def test_sample_metadata_default_is_empty():
    sample = Sample(
        messages=[{"role": "user", "content": "test"}],
        targets=[{"id": "t1", "output": {"content": {"key": "value"}}}],
    )
    assert sample.metadata == {}


def test_sample_media_default_is_empty():
    sample = Sample(
        messages=[{"role": "user", "content": "test"}],
        targets=[{"id": "t1", "output": {"content": {"key": "value"}}}],
    )
    assert sample.media == []


# ── Backward compat: backend_config.graspoflow ────────────────────────────────


def test_from_dict_accepts_backend_config_graspoflow_format():
    cfg = GraspoConfig.from_dict(
        {
            "backend_config": {"graspoflow": {"tp_size": 4, "pp_size": 2}},
        }
    )
    assert cfg.graspoflow.tp_size == 4
    assert cfg.graspoflow.pp_size == 2


def test_from_dict_top_level_graspoflow_takes_priority():
    cfg = GraspoConfig.from_dict(
        {
            "graspoflow": {"tp_size": 8},
            "backend_config": {"graspoflow": {"tp_size": 4}},
        }
    )
    assert cfg.graspoflow.tp_size == 8
