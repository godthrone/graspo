"""Tests for config schema validation —"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

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
    # 阈值由 rollout queue 决定（默认 8），与训练微批 optimize_prompt_batch_size 解耦
    cfg = TrainingConfig(optimize_prompt_batch_size=4, rollout_group_size=8)
    assert cfg.replay_buffer_optimize_threshold == 64
    cfg_small_queue = TrainingConfig(rollout_queue_batch_size=4, rollout_group_size=8)
    assert cfg_small_queue.replay_buffer_optimize_threshold == 32
    assert cfg_small_queue.optimize_prompt_batch_size == 8


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


# ── 废弃格式拒绝（backend_config shim 已删除，技术债务清理）───────────────


def test_from_dict_rejects_deprecated_backend_config_format():
    with pytest.raises(ValidationError):
        GraspoConfig.from_dict(
            {
                "backend_config": {"graspoflow": {"tp_size": 4, "pp_size": 2}},
            }
        )


def test_top_level_unknown_field_is_rejected_not_silently_dropped():
    """P0-1 防呆回归：顶层拼错字段名（train_methodd）必须报错。

    曾用手动挑键导致 `extra="forbid"` 形同虚设——拼错字段被静默忽略，
    用户拿默认值训练而不自知。现在直通 model_validate，未知键即拒绝。
    """
    with pytest.raises(ValidationError, match="train_methodd"):
        GraspoConfig.from_dict({"train_methodd": "sft"})
    with pytest.raises(ValidationError, match="backendd"):
        GraspoConfig.from_dict({"backendd": "graspoflow"})


def test_top_level_none_section_treated_as_default():
    """显式 `section: null` 等价于缺省（合法），不触发拒绝。"""
    cfg = GraspoConfig.from_dict({"training": None, "model": None})
    assert cfg.training.seed == 42
    assert cfg.graspoflow.tp_size == 2


def _field_paths(model_type: type[BaseModel], prefix: str = "") -> set[str]:  # noqa: F821
    """递归收集 pydantic 模型的全部字段路径（含嵌套模型）。"""
    paths: set[str] = set()
    for name, field in model_type.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        paths.add(path)
        nested = getattr(field.annotation, "model_fields", None)
        if nested:
            paths |= _field_paths(field.annotation, path)
    return paths


def _nested_get(mapping: dict, path: str) -> bool:
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def test_config_example_covers_all_schema_fields():
    """模板即文档防呆：config_example.yaml 必须覆盖 schema 全部字段。

    新字段发布后若忘记同步模板，此测试直接失败——防第三次脱节。
    """
    import yaml

    example = yaml.safe_load(
        Path("samples/configs/config_example.yaml").read_text(encoding="utf-8")
    )
    missing = sorted(path for path in _field_paths(GraspoConfig) if not _nested_get(example, path))
    assert not missing, f"config_example.yaml missing schema fields: {missing}"
