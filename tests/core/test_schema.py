"""Tests for config schema validation —"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from graspo.core.schema import (
    GraspoConfig,
    GraspoFlowConfig,
    MsSwiftConfig,
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
    # 阈值由 rollout queue 决定（默认 8），与训练微批 gradient_accumulation_micro_batches 解耦
    cfg = TrainingConfig(gradient_accumulation_micro_batches=4, rollout_group_size=8)
    assert cfg.replay_buffer_optimize_threshold == 64
    cfg_small_queue = TrainingConfig(rollout_queue_batch_size=4, rollout_group_size=8)
    assert cfg_small_queue.replay_buffer_optimize_threshold == 32
    # 解耦的断言：改 rollout_queue_batch_size 不会顺手改训练微批——它保持**自己的默认值**。
    # 这里曾硬编码 8（旧字段名 optimize_prompt_batch_size 的默认值）；42bfba4（v0.26.0）
    # 把该字段重命名为 gradient_accumulation_micro_batches 并把默认值改为 4，
    # 断言未同步，于是本用例从该提交起一直失败。断言改为对齐默认值本身，重命名/改默认
    # 都不会再让它脱节。
    assert (
        cfg_small_queue.gradient_accumulation_micro_batches
        == TrainingConfig().gradient_accumulation_micro_batches
    )
    assert cfg_small_queue.gradient_accumulation_micro_batches == 4


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
                "backend_config": {"native": {"tp_size": 4, "pp_size": 2}},
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
        GraspoConfig.from_dict({"backendd": "native"})


def test_top_level_none_section_treated_as_default():
    """显式 `section: null` 等价于缺省（合法），不触发拒绝。"""
    cfg = GraspoConfig.from_dict({"training": None, "model": None})
    assert cfg.training.seed == 42
    # 这里曾硬编码 2（tp_size 的旧默认值）；c9c1907 把它改成 1
    # （"P0: default caused 2x world_size"），断言未同步，于是本用例从该提交起一直失败。
    # 断言改为对齐默认值本身：本条要证明的是"None 段落到默认值"，不是某个具体数值。
    assert cfg.native.tp_size == GraspoFlowConfig().tp_size
    assert cfg.native.tp_size == 1


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


def test_msswift_config_example_covers_all_msswift_fields():
    """后端专属模板按**该后端的字段集合**要求，而不是按整个 GraspoConfig 要求。

    为什么单列一条（裁定：方案 A）：`config_example_msswift.yaml` 是与
    `config_example.yaml` **并列的第二个声明点**——同一件事写两处，就必须有测试
    同时守着两处。主模板那条测试只检查 `config_example.yaml`，于是本轮出现了
    "`max_pixels` / `freeze_vit` / `freeze_aligner` 只补了主模板、专属模板漏三个"
    的分叉。

    边界（刻意如此）：

    - **只断言 `MsSwiftConfig` 的直属字段齐全**（`:attr:`msswift` 段的键）；
    - **不要求**专属模板覆盖 `GraspoConfig` 的全量字段——那会把"后端专属模板"逼成
      "全字段模板"，与"主力模板覆盖全集"的设计冲突；
    - **不要求**嵌套模型（`msswift.megatron.*`）逐字段展开——既有的 `megatron: {}`
      等价于全部用默认值，合法且够用（主模板那份逐字段 null 属文档式全集）。
      但若模板里写了 `megatron.*` 键，必须是 schema 认识的键（防错别字，见下）。
    """
    import yaml

    example = yaml.safe_load(
        Path("samples/configs/config_example_msswift.yaml").read_text(encoding="utf-8")
    )
    msswift_section = example.get("msswift")
    assert isinstance(msswift_section, dict), "config_example_msswift.yaml 缺少 msswift 段"

    # ① 直属字段齐全（26 项，以 MsSwiftConfig 为单一真相源）
    missing = sorted(set(MsSwiftConfig.model_fields) - set(msswift_section))
    assert not missing, f"config_example_msswift.yaml msswift 段 missing fields: {missing}"

    # ② 嵌套 megatron 段若写了键，必须是 schema 认识的键（防错别字导致的静默失效）
    megatron = msswift_section.get("megatron")
    if isinstance(megatron, dict) and megatron:
        known = {
            path.split(".", 1)[1]
            for path in _field_paths(MsSwiftConfig)
            if path.startswith("megatron.")
        }
        unknown = sorted(set(megatron) - known)
        assert not unknown, f"config_example_msswift.yaml megatron 段含未知字段: {unknown}"


# ── max_steps 移除 + lr_scheduler.decay_steps（v0.23.0）───────────────────────


def test_removed_max_steps_rejected_with_clear_message():
    """max_steps 已移除：配置中出现时给出明确迁移提示而非模糊 extra 报错。"""
    with pytest.raises(ValidationError, match="max_steps has been removed"):
        TrainingConfig(max_steps=-1)


def test_scheduler_cosine_requires_decay_steps():
    """非 constant 调度必须显式 decay_steps > 0（加载即校验，§7.2）。"""
    with pytest.raises(ValidationError, match="decay_steps must be > 0"):
        TrainingConfig(
            max_epochs=10,
            learning_rate=5e-7,
            lr_scheduler={"type": "cosine", "min_lr_ratio": 0.2},
        )


def test_scheduler_constant_defaults_ok_without_decay_steps():
    """constant 调度默认不需要 decay_steps（向后兼容默认配置）。"""
    config = TrainingConfig(max_epochs=10, learning_rate=5e-6)
    assert config.lr_scheduler.type == "constant"
    assert config.lr_scheduler.decay_steps == 0


def test_scheduler_cosine_with_decay_steps_ok():
    """cosine + 显式 decay_steps 通过校验。"""
    config = TrainingConfig(
        max_epochs=10,
        learning_rate=5e-7,
        lr_scheduler={"type": "cosine", "decay_steps": 170, "min_lr_ratio": 0.2},
    )
    assert config.lr_scheduler.decay_steps == 170


def test_scheduler_linear_requires_decay_steps():
    """linear 调度同样要求 decay_steps > 0。"""
    with pytest.raises(ValidationError, match="decay_steps must be > 0"):
        TrainingConfig(
            max_epochs=10,
            lr_scheduler={"type": "linear", "decay_steps": 0},
        )


# ── 所有示例配置加载验证（C7 验收）──────────────────────────────────────────


def _all_example_configs() -> list[Path]:
    """收集 samples/configs/ 下所有 YAML 配置。"""
    configs_dir = Path("samples/configs")
    if not configs_dir.is_dir():
        return []
    return sorted(configs_dir.glob("*.yaml"))


@pytest.mark.parametrize("config_path", _all_example_configs())
def test_all_example_configs_loadable(config_path: Path, monkeypatch):
    """所有 samples/configs/*.yaml 可被 GraspoConfig.from_dict() 加载。

    验证目标（C7 验收）：配置模板与 schema 同步，不会因字段变更而脱节。

    部署事实从配置注入（2026-09-28 裁定，见 ``core.gpu_guard`` 模块头）：样例里的
    ``eval.gpus`` 用 GPU0–1，因此本用例显式声明"允许 0–5"。
    """
    import yaml

    monkeypatch.setenv("GRASPO_ALLOWED_GPU_INDICES", "0,1,2,3,4,5")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # 使用 from_dict 而非 model_validate，因为示例配置可能包含 None section
    cfg = GraspoConfig.from_dict(data)
    assert cfg.train_method in ("graspo", "sft")
    assert cfg.training.seed >= 0  # seed 必须是非负整数


# ── extra="forbid" 防呆回归（C7 验收）─────────────────────────────────────────


def test_graspo_config_rejects_multiple_unknown_fields():
    """多个未知字段同时出现时，ValidationError 列出所有错误。"""
    with pytest.raises(ValidationError) as exc_info:
        GraspoConfig.from_dict(
            {
                "training": {"bad_field_1": 1, "bad_field_2": "x"},
                "model": {"unknown_model_param": 999},
            }
        )
    # 至少报告第一个未知字段
    assert "bad_field_1" in str(exc_info.value) or "unknown_model_param" in str(exc_info.value)


def test_training_config_rejects_wrong_type():
    """字段类型错误时 pydantic 给出明确错误（非静默转换）。"""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate="not_a_float")


def test_graspo_config_from_dict_none_sections_preserved():
    """显式 None 的 section 使用默认值。"""
    cfg = GraspoConfig.from_dict({"training": None, "model": None, "data": None, "lora": None})
    assert cfg.training.seed == 42
    assert cfg.training.max_epochs == 100  # TrainingConfig 默认值
    assert cfg.model.model_path == ""
