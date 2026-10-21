"""LoRA 目标解析（``graspo.core.lora``）——**统一正则口径**的单元测试。

本轮契约（主席 2026-10-21 裁定）：``lora.target_modules`` 是**一条真实 HF 模块名
正则**，native 与 ms-swift 两个后端共用（``LORA_TARGET_MODULES_REGEX``）；任何旧形态
（preset 名、preset 名列表、裸模块名）一律 fail-closed 拒绝，⛔ 不留别名。
"""

import pytest

from graspo.core.lora import (
    LORA_TARGET_MODULES_REGEX,
    VISUAL_MODULE_PROBES,
    compile_lora_target_regex,
    looks_like_lora_target_regex,
    lora_target_modules_cover_visual,
    resolve_lora_target_modules,
)

#: 真实 HF 命名空间下**9B 那一档的缩样**（逐字沿用 modeling_qwen3_5.py 的属性名）：
#: full_attention 层是 ``self_attn.{q,k,v,o}_proj``；linear_attention 层是
#: ``linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}``；两侧都有
#: ``mlp.{gate,up,down}_proj``；视觉塔是 ``model.visual.{merger.linear_fc*,blocks.N.*}``。
#: 语言子树前缀 ``model.language_model``（native ``key_prefix``）。
_AVAILABLE = (
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.1.linear_attn.in_proj_qkv",
    "model.language_model.layers.1.linear_attn.in_proj_z",
    "model.language_model.layers.1.linear_attn.in_proj_b",
    "model.language_model.layers.1.linear_attn.in_proj_a",
    "model.language_model.layers.1.linear_attn.out_proj",
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
)


def test_shared_regex_resolves_to_real_hf_module_names_on_both_sides():
    """同一条正则：命中语言侧 12 项（含 linear_attn）+ 视觉侧 6 项，且都是真实 HF 名。"""
    resolved = resolve_lora_target_modules(LORA_TARGET_MODULES_REGEX, available=_AVAILABLE).resolved
    assert resolved == tuple(sorted(n for n in _AVAILABLE if n not in (
        "model.language_model.layers.1.linear_attn.in_proj_b",
        "model.language_model.layers.1.linear_attn.in_proj_a",
    )))
    assert "model.visual.blocks.0.attn.qkv" in resolved
    assert "model.language_model.layers.1.linear_attn.in_proj_qkv" in resolved


def test_shared_regex_matches_full_attention_and_linear_attention_variants():
    """两族层的属性名都覆盖：``self_attn.*`` 与 ``linear_attn.*``（不是只认一族）。"""
    pattern = compile_lora_target_regex(LORA_TARGET_MODULES_REGEX)
    for name in (
        "model.language_model.layers.7.self_attn.q_proj",
        "model.language_model.layers.8.linear_attn.in_proj_z",
        "model.language_model.layers.8.linear_attn.out_proj",
        "model.language_model.layers.8.linear_attn.in_proj_qkv",
        "model.visual.blocks.26.mlp.linear_fc2",
    ):
        assert pattern.fullmatch(name), name


# ── ★ 阳性对照 ①（Q1：正则必须已去掉 in_proj_a / in_proj_b）──────────────────
# 能让本测试变红的精确读法：在 `src/graspo/core/lora.py` 的
# `LORA_TARGET_MODULES_REGEX` 语言侧枚举里把 `|in_proj_qkv` 改回
# `|in_proj_qkv|in_proj_a|in_proj_b`（即撤销 Q1）⇒ 下面第一条断言立刻红。
def test_positive_control_ab_not_in_shared_regex():
    """in_proj_a/in_proj_b **不得**被同一条正则命中（否则两后端权重矩阵集合 358 vs 310）。"""
    pattern = compile_lora_target_regex(LORA_TARGET_MODULES_REGEX)
    assert "in_proj_a" not in LORA_TARGET_MODULES_REGEX
    assert "in_proj_b" not in LORA_TARGET_MODULES_REGEX
    assert pattern.fullmatch("model.language_model.layers.1.linear_attn.in_proj_a") is None
    assert pattern.fullmatch("model.language_model.layers.1.linear_attn.in_proj_b") is None
    # 去掉的只有 a/b：融合的 qkv 与 z/out_proj 仍在（否则就是砍多了）
    resolved = resolve_lora_target_modules(LORA_TARGET_MODULES_REGEX, available=_AVAILABLE).resolved
    assert "model.language_model.layers.1.linear_attn.in_proj_qkv" in resolved
    assert "model.language_model.layers.1.linear_attn.in_proj_z" in resolved
    assert "model.language_model.layers.1.linear_attn.out_proj" in resolved


def test_shared_regex_does_not_match_unrelated_modules():
    """不误伤：embedding / lm_head / norm / 视觉 patch_embed 都不在目标里。"""
    pattern = compile_lora_target_regex(LORA_TARGET_MODULES_REGEX)
    for name in (
        "model.language_model.embed_tokens",
        "lm_head",
        "model.language_model.norm",
        "model.visual.patch_embed.proj",
        "model.visual.blocks.0.norm1",
    ):
        assert pattern.fullmatch(name) is None, name


def test_lora_target_modules_cover_visual_uses_real_visual_names():
    """视觉覆盖判据：共享正则覆盖视觉塔；语言-only 正则不覆盖（fail-closed）。"""
    assert lora_target_modules_cover_visual(LORA_TARGET_MODULES_REGEX) is True
    assert lora_target_modules_cover_visual(r"^(?:model\.)?language_model\..*\.q_proj$") is False
    # 非正则 / 空值一律判否，而不是抛异常或默认放行
    assert lora_target_modules_cover_visual(None) is False
    assert lora_target_modules_cover_visual("language_all_linear") is False
    assert lora_target_modules_cover_visual(["vision_common"]) is False
    assert all(n.startswith("model.visual.") for n in VISUAL_MODULE_PROBES)


# ── ★ 阳性对照 ②（fail-closed：旧形态必须被拒，不得静默兼容）─────────────────
# 能让本测试变红的精确读法：在 `resolve_lora_target_modules` 里把
# `if target_modules is None:` 分支改回"回退到某个 preset"（或让
# `compile_lora_target_regex` 接受 list / 裸名字）⇒ 对应的 pytest.raises 立刻红。
def test_positive_control_old_forms_are_rejected_fail_closed():
    """旧 preset 名 / preset 名列表 / None / 不可编译正则 —— 全部报错，⛔ 不兜底。"""
    with pytest.raises(ValueError, match="must be a single real HF module-name regex"):
        compile_lora_target_regex(["language_all_linear", "vision_common"])
    with pytest.raises(ValueError, match="bare name"):
        compile_lora_target_regex("language_all_linear")
    with pytest.raises(ValueError, match="bare name"):
        compile_lora_target_regex("vision_common")
    with pytest.raises(ValueError, match="bare name"):
        compile_lora_target_regex("model.visual.merger.linear_fc1")  # 漏写锚点/元字符
    with pytest.raises(ValueError, match="not a compilable regex"):
        compile_lora_target_regex("^model.language_model.*(q_proj$")
    with pytest.raises(ValueError, match="lora.target_modules is required"):
        resolve_lora_target_modules(None, available=_AVAILABLE)
    with pytest.raises(ValueError, match="must be a single real HF module-name regex"):
        resolve_lora_target_modules(["language_all_linear"], available=_AVAILABLE)  # type: ignore[arg-type]


def test_resolve_raises_when_regex_matches_nothing():
    """命中 0 项（或只覆盖另一族命名空间）必须报错，不得静默建出零 LoRA 的模型。"""
    with pytest.raises(ValueError, match="matched none of the model's LoRA-able modules"):
        resolve_lora_target_modules(r"^language\..*\.q_proj$", available=_AVAILABLE)
    with pytest.raises(ValueError, match="matched none of the model's LoRA-able modules"):
        resolve_lora_target_modules(r"^totally\.unrelated$", available=_AVAILABLE)


def test_resolved_targets_signature_records_the_single_regex():
    resolved = resolve_lora_target_modules(LORA_TARGET_MODULES_REGEX, available=_AVAILABLE)
    assert resolved.requested == (LORA_TARGET_MODULES_REGEX,)
    assert resolved.signature["requested"] == [LORA_TARGET_MODULES_REGEX]


def test_looks_like_lora_target_regex_distinguishes_regex_from_bare_name():
    assert looks_like_lora_target_regex(LORA_TARGET_MODULES_REGEX) is True
    assert looks_like_lora_target_regex("q_proj") is False
    assert looks_like_lora_target_regex(None) is False  # type: ignore[arg-type]
    assert looks_like_lora_target_regex(["^q_proj$"]) is False  # type: ignore[arg-type]
