"""``graspo.eval.merged_export`` 的单测：checkpoint 形态识别与合并前置守卫。

补齐的缺口是"ms-swift 产物（标准 PEFT 目录）→ merged-hf"。这里测的是**判定与
拒绝路径**——真正的权重合并在 GPU 容器里跑，本轮不验证（见工位 report.md）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graspo.eval.merged_export import (
    CheckpointKind,
    ExportError,
    assert_base_matches,
    classify_checkpoint,
    existing_merged_output,
    prepare_output_directory,
    read_adapter_base,
    resolve_peft_adapter_dir,
)


def _write(path: Path, payload: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _base_model(root: Path) -> Path:
    _write(root / "config.json", '{"model_type": "qwen3"}')
    (root / "model.safetensors").write_bytes(b"\x00\x00")
    return root


def _peft_adapter(root: Path, base: str | None = "/models/Qwen3.5-9B") -> Path:
    payload: dict[str, object] = {"r": 128, "lora_alpha": 256}
    if base is not None:
        payload["base_model_name_or_path"] = base
    _write(root / "adapter_config.json", json.dumps(payload))
    (root / "adapter_model.safetensors").write_bytes(b"\x00\x00")
    return root


def test_classify_base_model(tmp_path):
    result = classify_checkpoint(_base_model(tmp_path / "base"))
    assert result.kind is CheckpointKind.BASE_MODEL
    assert any("config.json" in item for item in result.evidence)


def test_classify_peft_adapter(tmp_path):
    result = classify_checkpoint(_peft_adapter(tmp_path / "adapter"))
    assert result.kind is CheckpointKind.PEFT_ADAPTER


def test_classify_native_graspo_checkpoint(tmp_path):
    root = tmp_path / "native"
    _write(root / "metadata.json")
    (root / "shard_0.safetensors").write_bytes(b"\x00\x00")
    result = classify_checkpoint(root)
    assert result.kind is CheckpointKind.GRASPO_NATIVE


def test_classify_rejects_unknown_layout_with_listing(tmp_path):
    root = tmp_path / "weird"
    root.mkdir()
    (root / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ExportError, match="cannot classify checkpoint"):
        classify_checkpoint(root)


def test_classify_rejects_missing_directory(tmp_path):
    with pytest.raises(ExportError, match="not found"):
        classify_checkpoint(tmp_path / "nope")


def test_resolve_peft_adapter_finds_nested_ms_swift_output(tmp_path):
    root = tmp_path / "run" / "v3-20260901-101010" / "checkpoint-500"
    _peft_adapter(root)
    assert resolve_peft_adapter_dir(tmp_path / "run") == root.resolve()


def test_resolve_peft_adapter_returns_self_when_direct(tmp_path):
    adapter = _peft_adapter(tmp_path / "direct")
    assert resolve_peft_adapter_dir(adapter) == adapter


def test_resolve_peft_adapter_returns_none_when_absent(tmp_path):
    (tmp_path / "empty").mkdir()
    assert resolve_peft_adapter_dir(tmp_path / "empty") is None


def test_resolve_peft_adapter_refuses_ambiguous_layout(tmp_path):
    _peft_adapter(tmp_path / "out" / "a")
    _peft_adapter(tmp_path / "out" / "b")
    with pytest.raises(ExportError, match="multiple PEFT adapters found"):
        resolve_peft_adapter_dir(tmp_path / "out")


def test_read_adapter_base_and_mismatch_rejection(tmp_path):
    adapter = _peft_adapter(tmp_path / "adapter", base="/models/Qwen3.5-9B")
    assert read_adapter_base(adapter) == "/models/Qwen3.5-9B"
    assert_base_matches(adapter, "/models/Qwen3.5-9B")  # 同名即通过

    with pytest.raises(ExportError, match="adapter base mismatch"):
        assert_base_matches(adapter, "/models/Other-9B")


def test_assert_base_matches_rejects_adapter_without_recorded_base(tmp_path):
    """拿错底座会合出一个"看起来很正常"的错模型——所以无法确认时必须拒绝。"""
    adapter = _peft_adapter(tmp_path / "adapter", base=None)
    with pytest.raises(ExportError, match="does not record base_model_name_or_path"):
        assert_base_matches(adapter, "/models/Qwen3.5-9B")


def test_existing_merged_output_requires_config_and_weights(tmp_path):
    assert existing_merged_output(tmp_path) is False
    _write(tmp_path / "config.json")
    assert existing_merged_output(tmp_path) is False
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    assert existing_merged_output(tmp_path) is True


def test_prepare_output_directory_creates_when_absent(tmp_path):
    target = tmp_path / "new" / "dir"
    assert prepare_output_directory(target) == target
    assert target.is_dir()


def test_prepare_output_directory_refuses_nonempty_without_optin(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.bin").write_bytes(b"x")
    with pytest.raises(ExportError, match="refusing to overwrite"):
        prepare_output_directory(target)
    assert (target / "old.bin").exists()  # 拒绝时不动任何东西


def test_prepare_output_directory_overwrite_replaces_contents(tmp_path):
    target = tmp_path / "out"
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "old.bin").write_bytes(b"x")
    (target / "old2.bin").write_bytes(b"x")
    prepare_output_directory(target, overwrite=True)
    assert list(target.iterdir()) == []


def test_merge_checks_filesystem_before_importing_heavy_deps(tmp_path):
    """合并入口的顺序契约：先做路径/底座校验，再 import torch/peft。

    顺序有实际意义——路径写错、底座不匹配这类用户级错误，应该在开始加载权重
    （几分钟）之前就被报出来。这里用一个"不是 PEFT 目录"的路径证明它先报文件系统错。
    """
    from graspo.eval.merged_export import merge_peft_checkpoint

    with pytest.raises(ExportError, match="is not a PEFT adapter"):
        merge_peft_checkpoint(tmp_path / "nope", "/models/base", tmp_path / "out")


def test_merge_refuses_base_mismatch_before_importing_heavy_deps(tmp_path):
    """底座不匹配必须在加载权重前拒绝——否则会合出一个"看起来很正常"的错模型。"""
    from graspo.eval.merged_export import merge_peft_checkpoint

    adapter = _peft_adapter(tmp_path / "adapter", base="/models/Qwen3.5-9B")
    with pytest.raises(ExportError, match="adapter base mismatch"):
        merge_peft_checkpoint(adapter, "/models/Other-9B", tmp_path / "out")


# ── 多模态语言层命名空间的归一（Qwen3.5 多模态 adapter 的合并前置）─────────────
#
# 真实 T013 产物的 `target_modules` 是 peft 0.19.1 写的**正则字符串**，每个分支都带
# `model.language_model` 前缀（ms-swift 在多模态 `Qwen3_5ForConditionalGeneration`
# 上训练，这个前缀是对的）。而本模块用 `AutoModelForCausalLM`（纯文本类）加载底座，
# 语言层在 `model.layers.{i}` 下 ⇒ 不归一就一条都匹配不上（ValueError）。
#
# 逐字取自 `.local/.../task-r3-effect/evidence/light_evidence/T013/adapter_config.json`。

_T013_REGEX = (
    r"^(model\.language_model(?=\.).*\.(v_proj|in_proj_a|in_proj_qkv|up_proj|in_proj_z|"
    r"in_proj_b|o_proj|gate_proj|down_proj|q_proj|out_proj|k_proj))$"
)


def test_normalize_regex_string_drops_only_the_namespace_prefix():
    """正则形态：str 进 str 出，且除前缀外一字不动（`^`、`(?=\\.)`、`.*`、分支顺序）。"""
    from graspo.eval.merged_export import normalize_language_model_targets

    normalized = normalize_language_model_targets(_T013_REGEX)
    assert isinstance(normalized, str)
    assert normalized == _T013_REGEX.replace(r"model\.language_model", "model")
    assert normalized.startswith("^(model(?=\\.)")
    # 分支与顺序原样保留
    assert "in_proj_qkv" in normalized and "gate_proj" in normalized
    assert normalized.endswith(r"\.(v_proj|in_proj_a|in_proj_qkv|up_proj|in_proj_z|in_proj_b|o_proj|gate_proj|down_proj|q_proj|out_proj|k_proj))$")


def test_normalize_regex_matches_text_namespace_and_not_vision():
    """归一后的正则必须匹配纯文本语言层，且不再把视觉塔拉进来。"""
    import re

    from graspo.eval.merged_export import normalize_language_model_targets

    normalized = normalize_language_model_targets(_T013_REGEX)
    assert isinstance(normalized, str)
    assert re.fullmatch(normalized, "model.layers.3.self_attn.q_proj")
    assert re.fullmatch(normalized, "model.layers.0.mlp.gate_proj")
    # 视觉塔不在 T013 的目标集合里，归一后也不该出现
    assert not re.fullmatch(normalized, "model.visual.blocks.0.attn.qkv")


def test_normalize_list_strips_literal_prefix_and_preserves_order():
    """list 形态：剥掉字面前缀，顺序保留、去重。"""
    from graspo.eval.merged_export import normalize_language_model_targets

    result = normalize_language_model_targets(
        [
            "model.language_model.layers.*.self_attn.q_proj",
            "model.language_model.layers.*.mlp.gate_proj",
            "model.language_model.layers.*.self_attn.q_proj",  # 重复
            "q_proj",  # 本就没有前缀
        ]
    )
    assert result == [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.mlp.gate_proj",
        "q_proj",
    ]


def test_normalize_is_identity_for_non_multimodal_targets():
    """既有可用路径必须不受影响：不带前缀的入参**原样**返回。"""
    from graspo.eval.merged_export import normalize_language_model_targets

    plain_regex = _T013_REGEX.replace(r"model\.language_model", "model")
    assert normalize_language_model_targets(plain_regex) == plain_regex
    assert normalize_language_model_targets(["q_proj", "v_proj"]) == ["q_proj", "v_proj"]
    assert normalize_language_model_targets([]) == []


def test_normalize_rejects_unsupported_type():
    """不做猜测式转换（§2.3）：非 str / 非序列一律拒绝。"""
    from graspo.eval.merged_export import normalize_language_model_targets

    with pytest.raises(ExportError, match="unsupported target_modules type"):
        normalize_language_model_targets(123)  # type: ignore[arg-type]


def test_read_adapter_targets_keeps_shape(tmp_path):
    """str 进 str 出 / list 进 list 出 —— 形态决定 peft 的匹配算法，不能悄悄统一。"""
    from graspo.eval.merged_export import read_adapter_targets

    regex_dir = tmp_path / "regex"
    _write(regex_dir / "adapter_config.json", json.dumps({"target_modules": _T013_REGEX}))
    assert read_adapter_targets(regex_dir) == _T013_REGEX

    list_dir = tmp_path / "list"
    _write(list_dir / "adapter_config.json", json.dumps({"target_modules": ["q_proj", "v_proj"]}))
    assert read_adapter_targets(list_dir) == ["q_proj", "v_proj"]

    absent = tmp_path / "absent"
    _write(absent / "adapter_config.json", json.dumps({"r": 8}))
    assert read_adapter_targets(absent) is None


def test_resolve_injection_targets_normalizes_multimodal_adapter(tmp_path):
    """端到端（无 torch）：多模态正则进 → PeftConfig.target_modules 已是纯文本命名空间。

    本用例只在 peft 安装时执行——它验证的正是"文件名里的值会覆盖调用方 kwargs"
    那个坑的绕行方式（显式构造 PeftConfig 再改字段）。
    """
    peft = pytest.importorskip("peft")
    assert peft  # 仅用于显式表达依赖

    from graspo.eval.merged_export import _resolve_injection_targets

    adapter = tmp_path / "adapter"
    _write(adapter / "adapter_config.json", json.dumps({"peft_type": "LORA", "target_modules": _T013_REGEX}))
    config, targets = _resolve_injection_targets(adapter)
    assert config is not None
    assert targets == _T013_REGEX.replace(r"model\.language_model", "model")
    assert config.target_modules == targets
