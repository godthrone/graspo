"""Tests for config schema validation — BADGE §11.1."""

import pytest
from pydantic import ValidationError

from graspo.core.schema import (
    GraspoConfig,
    Sample,
    TrainingConfig,
    _check_removed_fields,
    _generate_run_name,
)


# ── TrainingConfig 校验 ──────────────────────────────────────────────────────


def test_training_config_rejects_removed_field_total_epochs():
    with pytest.raises(ValueError, match="已废弃的 training 配置字段"):
        GraspoConfig.from_dict({"training": {"total_epochs": 10}})


def test_training_config_rejects_removed_field_group_size():
    with pytest.raises(ValueError, match="已废弃的 training 配置字段"):
        GraspoConfig.from_dict({"training": {"group_size": 8}})


def test_training_config_rejects_removed_field_train_batch_size():
    with pytest.raises(ValueError, match="已废弃的 training 配置字段"):
        GraspoConfig.from_dict({"training": {"train_batch_size": 32}})


def test_training_config_rejects_removed_field_max_retry():
    with pytest.raises(ValueError, match="已废弃的 training 配置字段"):
        GraspoConfig.from_dict({"training": {"max_retry": 3}})


def test_training_config_rejects_multiple_removed_fields():
    with pytest.raises(ValueError, match="已废弃的 training 配置字段"):
        GraspoConfig.from_dict({"training": {"total_epochs": 10, "group_size": 8}})


def test_training_config_rejects_replay_buffer_threshold():
    with pytest.raises(ValueError, match="replay_buffer_optimize_threshold"):
        GraspoConfig.from_dict({"training": {"replay_buffer_optimize_threshold": 64}})


def test_training_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TrainingConfig(nonexistent_field=42)


# ── DataConfig 校验 ──────────────────────────────────────────────────────────


def test_data_config_rejects_removed_field_prompt_field():
    with pytest.raises(ValueError, match="已废弃的 data 配置字段"):
        GraspoConfig.from_dict({"data": {"prompt_field": "text"}})


def test_data_config_rejects_removed_field_messages_field():
    with pytest.raises(ValueError, match="已废弃的 data 配置字段"):
        GraspoConfig.from_dict({"data": {"messages_field": "msgs"}})


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


# ── _check_removed_fields ────────────────────────────────────────────────────


def test_check_removed_fields_present_raises():
    with pytest.raises(ValueError, match="test_section.field_a"):
        _check_removed_fields({"field_a": 1, "field_b": 2, "ok": 3}, "test_section", {"field_a", "field_b"})


def test_check_removed_fields_none_does_not_raise():
    _check_removed_fields(None, "test_section", {"field_a"})


def test_check_removed_fields_empty_does_not_raise():
    _check_removed_fields({}, "test_section", {"field_a"})


def test_check_removed_fields_no_match_does_not_raise():
    _check_removed_fields({"valid_field": 1}, "test_section", {"field_a", "field_b"})


# ── Backward compat: backend_config.graspoflow ────────────────────────────────


def test_from_dict_accepts_backend_config_graspoflow_format():
    cfg = GraspoConfig.from_dict({
        "backend_config": {"graspoflow": {"tp_size": 4, "pp_size": 2}},
    })
    assert cfg.graspoflow.tp_size == 4
    assert cfg.graspoflow.pp_size == 2


def test_from_dict_top_level_graspoflow_takes_priority():
    cfg = GraspoConfig.from_dict({
        "graspoflow": {"tp_size": 8},
        "backend_config": {"graspoflow": {"tp_size": 4}},
    })
    assert cfg.graspoflow.tp_size == 8
