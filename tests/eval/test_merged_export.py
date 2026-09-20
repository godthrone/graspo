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


# ── 多模态语言层命名空间的归一（**仅纯文本加载器路径**）─────────────────────────
#
# 真实 T013 产物的 `target_modules` 是 peft 0.19.1 写的**正则字符串**，每个分支都带
# `model.language_model` 前缀（ms-swift 在多模态 `Qwen3_5ForConditionalGeneration`
# 上训练，这个前缀是对的）。
#
# ★ 缺陷 D 之后的语义：归一化**只在纯文本加载器**（语言层在 `model.layers.{i}`）下成立。
# 多模态加载器（语言层在 `model.language_model.{i}`）下目标名本来就对，归一化反而会
# **放宽**正则（见 `test_normalize_regex_broadens_...`）⇒ 多模态路径禁止调用它。
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
    assert normalized.endswith(
        r"\.(v_proj|in_proj_a|in_proj_qkv|up_proj|in_proj_z|in_proj_b|o_proj|gate_proj|down_proj|q_proj|out_proj|k_proj))$"
    )


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
    _write(
        adapter / "adapter_config.json",
        json.dumps({"peft_type": "LORA", "target_modules": _T013_REGEX}),
    )
    config, targets = _resolve_injection_targets(adapter)
    assert config is not None
    assert targets == _T013_REGEX.replace(r"model\.language_model", "model")
    assert config.target_modules == targets


# ── 缺陷 D：加载器类别必须按底座实际配置判定 ────────────────────────────────
#
# 事故形态：固定用 `AutoModelForCausalLM`（纯文本类）加载多模态底座 ⇒ 权重"能载入"
# （视觉塔被静默忽略），`save_pretrained` 落出的 config.json 变成
# `architectures=['Qwen3_5ForCausalLM'] / model_type=qwen3_5_text / 无 vision_config`
# ⇒ 产物在、退出码 0、**视觉塔已丢**，vLLM 拒绝，或被吞掉异常后在错模型上刷出假准确率。
#
# 下面这组用例是**负向（能真失败）**的：把加载器类别改回固定纯文本类，它们会红。

#: 真实多模态底座 config 的**最小**形态（字段名/取值取自 Qwen3.5 系列：有 vision_config）。
_MULTIMODAL_BASE_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "has_vision_config": True,
    "vision_config": {"depth": 27, "hidden_size": 1152},
    "image_token_id": 151655,
    "text_config": {"model_type": "qwen3_5_text"},
}

#: 纯文本底座 config 的最小形态（无任何视觉标记）。
_TEXT_BASE_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "text_config": {"model_type": "qwen3_5_text"},
}


def _base_with_config(root: Path, config: dict[str, object]) -> Path:
    _write(root / "config.json", json.dumps(config))
    (root / "model.safetensors").write_bytes(b"\x00\x00")
    return root


def test_detect_multimodal_base_reads_vision_markers(tmp_path):
    """多模态底座必须被认出来，并给出命中判据（判据是回显也是证据，§2.2）。"""
    from graspo.eval.merged_export import detect_multimodal_base

    shape = detect_multimodal_base(_base_with_config(tmp_path / "mm", _MULTIMODAL_BASE_CONFIG))
    assert shape.multimodal is True
    assert any("vision_config" in item for item in shape.evidence)
    assert any("ConditionalGeneration" in item for item in shape.evidence)


def test_detect_multimodal_base_classifies_text_base_as_text(tmp_path):
    """纯文本底座不得被误判成多模态（否则会去要一个不存在的视觉塔）。"""
    from graspo.eval.merged_export import detect_multimodal_base

    shape = detect_multimodal_base(_base_with_config(tmp_path / "txt", _TEXT_BASE_CONFIG))
    assert shape.multimodal is False
    assert shape.evidence


def test_detect_multimodal_base_refuses_a_base_without_config(tmp_path):
    """没有 config.json 就判不了类别 ⇒ fail-closed，不许"猜一个类"（§2.3）。"""
    from graspo.eval.merged_export import detect_multimodal_base

    (tmp_path / "bare").mkdir()
    with pytest.raises(ExportError, match="cannot determine the loader class"):
        detect_multimodal_base(tmp_path / "bare")


def test_has_vision_config_flag_wins_even_when_false(tmp_path):
    """`has_vision_config=False` 是**显式声明无视觉**，不是"未提供"（§2.2 空值语义）。"""
    from graspo.eval.merged_export import is_multimodal_base_config

    multimodal, evidence = is_multimodal_base_config({"has_vision_config": False})
    assert multimodal is False
    assert evidence == ()


def test_image_token_id_zero_is_still_a_marker(tmp_path):
    """`image_token_id = 0` 是合法 id，不能用 `if value` 判空（§2.2）。"""
    from graspo.eval.merged_export import is_multimodal_base_config

    multimodal, _ = is_multimodal_base_config({"image_token_id": 0})
    assert multimodal is True


class _FakeAutoClass:
    def __init__(self, name: str) -> None:
        self.name = name

    def from_pretrained(self, *_args, **_kwargs):  # pragma: no cover - 仅作属性存在性占位
        raise AssertionError("not called in these tests")


def _fake_transformers(**names):
    class _NS:
        __version__ = "5.12.1-fake"

    ns = _NS()
    for name in names:
        setattr(ns, name, _FakeAutoClass(name))
    return ns


def test_resolve_auto_model_class_uses_multimodal_class_for_multimodal_base():
    """★ 负向用例核心：多模态底座**必须**选多模态 auto 类，**不得**是纯文本类。"""
    from graspo.eval.merged_export import _resolve_auto_model_class

    ns = _fake_transformers(
        AutoModelForImageTextToText=_FakeAutoClass("AutoModelForImageTextToText"),
        AutoModel=_FakeAutoClass("AutoModel"),
        AutoModelForCausalLM=_FakeAutoClass("AutoModelForCausalLM"),
    )
    name, cls = _resolve_auto_model_class(ns, multimodal=True)
    assert name == "AutoModelForImageTextToText"
    assert name != "AutoModelForCausalLM"
    assert cls is ns.AutoModelForImageTextToText


def test_resolve_auto_model_class_uses_text_class_for_text_base():
    """纯文本底座仍走既有 `AutoModelForCausalLM`（既有可用路径不受影响）。"""
    from graspo.eval.merged_export import _resolve_auto_model_class

    ns = _fake_transformers(
        AutoModelForImageTextToText=_FakeAutoClass("AutoModelForImageTextToText"),
        AutoModelForCausalLM=_FakeAutoClass("AutoModelForCausalLM"),
    )
    name, _ = _resolve_auto_model_class(ns, multimodal=False)
    assert name == "AutoModelForCausalLM"


def test_resolve_auto_model_class_falls_back_to_automodel_only_for_multimodal():
    """老版 transformers 没有 ImageTextToText 时，多模态回落到 `AutoModel`（仍是按
    `config.architectures` 解析的通用类），**绝不**回落到纯文本类。"""
    from graspo.eval.merged_export import _resolve_auto_model_class

    ns = _fake_transformers(
        AutoModel=_FakeAutoClass("AutoModel"),
        AutoModelForCausalLM=_FakeAutoClass("AutoModelForCausalLM"),
    )
    name, _ = _resolve_auto_model_class(ns, multimodal=True)
    assert name == "AutoModel"


def test_resolve_auto_model_class_fails_closed_when_no_multimodal_class_exists():
    """一个多模态候选都没有 ⇒ 必须报错停止，**不得**回落成纯文本类（那正是缺陷 D）。"""
    from graspo.eval.merged_export import _resolve_auto_model_class

    ns = _fake_transformers(AutoModelForCausalLM=_FakeAutoClass("AutoModelForCausalLM"))
    with pytest.raises(ExportError, match="refusing to fall back to a text-only loader"):
        _resolve_auto_model_class(ns, multimodal=True)


class _FakeBase:
    """模拟 "loaded model" 的结构：只有挂载层级，不加载任何权重。"""

    def __init__(self, *, language_model: bool) -> None:
        inner = type("_Inner", (), {})()
        if language_model:
            inner.language_model = object()
        self.model = inner


def test_assert_loader_class_matches_base_accepts_multimodal_shape(tmp_path):
    """多模态底座 + 有 `model.language_model` 子树的模型 ⇒ 通过。"""
    from graspo.eval.merged_export import BaseModelShape, _assert_loader_class_matches_base

    shape = BaseModelShape(tmp_path, True, ("vision_config present",))
    _assert_loader_class_matches_base(_FakeBase(language_model=True), shape)


def test_assert_loader_class_matches_base_rejects_text_model_on_multimodal_base(tmp_path):
    """★ 负向用例核心：多模态底座被纯文本类装进来（无 language_model 子树）⇒
    必须当场拒绝，而不是落出一份丢掉视觉塔的产物。"""
    from graspo.eval.merged_export import BaseModelShape, _assert_loader_class_matches_base

    shape = BaseModelShape(tmp_path, True, ("vision_config present",))
    with pytest.raises(ExportError, match="vision tower would be dropped"):
        _assert_loader_class_matches_base(_FakeBase(language_model=False), shape)


def test_assert_loader_class_matches_base_rejects_multimodal_model_on_text_base(tmp_path):
    """反向也要防：纯文本底座装出了带 language_model 的模型 ⇒ 类别选错了。"""
    from graspo.eval.merged_export import BaseModelShape, _assert_loader_class_matches_base

    shape = BaseModelShape(tmp_path, False, ("no vision marker",))
    with pytest.raises(ExportError, match="chosen for the wrong shape"):
        _assert_loader_class_matches_base(_FakeBase(language_model=True), shape)


def test_normalize_regex_broadens_and_would_also_match_the_multimodal_namespace():
    """★ 复核硬证据：归一化**不是改名，是放宽**。

    归一后 `^(model(?=\\.).*\\.(...))$` 对**多模态**全名 `model.language_model.layers.3.
    self_attn.q_proj` 依旧 fullmatch（`(model(?=\\.).*` 贪婪吃掉了 `.language_model`）——
    它只是一个**超集**。这就是"归一化只在纯文本加载器下成立"的机制原因：纯文本基座里
    根本不存在 `model.language_model.*` 这些名字，超集才"恰好"选对层。
    """
    import re

    from graspo.eval.merged_export import normalize_language_model_targets

    normalized = normalize_language_model_targets(_T013_REGEX)
    assert isinstance(normalized, str)
    # 超集证据：多模态全名仍然被匹配（误伤面）
    assert re.fullmatch(normalized, "model.language_model.layers.3.self_attn.q_proj")
    # 原正则对多模态全名也匹配（ms-swift 写的就是对的）
    assert re.fullmatch(_T013_REGEX, "model.language_model.layers.3.self_attn.q_proj")
    # 原正则对纯文本全名不匹配（这才是当初出 ValueError 的原因）
    assert not re.fullmatch(_T013_REGEX, "model.layers.3.self_attn.q_proj")


def test_normalize_regex_is_a_superset_over_model_prefixed_modules():
    """放宽的量化：归一后凡是 `model.<任意非空段>....<x>_proj` 都可能命中，
    所以它**不能**用在多模态加载器上（视觉塔下同名投影层会被误伤）。"""
    import re

    from graspo.eval.merged_export import normalize_language_model_targets

    normalized = normalize_language_model_targets(_T013_REGEX)
    assert isinstance(normalized, str)
    # 一个**不属于**原始目标集合的模块名：归一后被误纳（视觉塔下的投影层）
    injected = "model.visual.blocks.0.mlp.down_proj"
    assert not re.fullmatch(_T013_REGEX, injected)
    assert re.fullmatch(normalized, injected)


def test_resolve_injection_targets_preserves_targets_for_multimodal_loader(tmp_path):
    """★ 多模态加载器下 `target_modules` **原样保留**（不归一、不改写）。"""
    pytest.importorskip("peft")

    from graspo.eval.merged_export import _resolve_injection_targets

    adapter = tmp_path / "adapter"
    _write(
        adapter / "adapter_config.json",
        json.dumps({"peft_type": "LORA", "target_modules": _T013_REGEX}),
    )
    config, targets = _resolve_injection_targets(adapter, multimodal=True)
    assert config is not None
    assert targets == _T013_REGEX
    assert config.target_modules == _T013_REGEX


def test_resolve_injection_targets_refuses_plain_text_targets_on_multimodal_base(tmp_path):
    """多模态底座 + 不含 `model.language_model` 的目标名 ⇒ fail-closed，不硬凑。"""
    pytest.importorskip("peft")

    from graspo.eval.merged_export import _resolve_injection_targets

    adapter = tmp_path / "adapter"
    _write(
        adapter / "adapter_config.json",
        json.dumps({"peft_type": "LORA", "target_modules": ["q_proj", "v_proj"]}),
    )
    with pytest.raises(ExportError, match="do not\\s+reference"):
        _resolve_injection_targets(adapter, multimodal=True)


def test_merge_detects_shape_before_importing_heavy_deps(tmp_path):
    """加载器类别判定发生在 import torch **之前**：底座 config 缺失时，报的是
    "判不了类别"而不是 ModuleNotFoundError（顺序契约，与既有两条前置校验一致）。"""
    from graspo.eval.merged_export import merge_peft_checkpoint

    adapter = _peft_adapter(tmp_path / "adapter", base="/models/Qwen3.5-9B")
    bare_base = tmp_path / "Qwen3.5-9B"
    bare_base.mkdir()
    with pytest.raises(ExportError, match="cannot determine the loader class"):
        merge_peft_checkpoint(adapter, bare_base, tmp_path / "out")
