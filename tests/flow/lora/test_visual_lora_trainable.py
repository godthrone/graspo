"""native 视觉塔 LoRA 可训性：配置期防线 + 真模块替换证据（阻断 1 的回归测试）。

**背景（实测 T028，2026-09-19）**：清单档 T028（9B / GRASPO / LoRA / native / 1 卡）
写 ``lora.target_preset: language_safe``，而 ``language_safe`` 只含 ``language.*``
模式（``core/lora.py:11-14``）。``build_qwen35_visual_tower`` 先把视觉塔全部参数
``requires_grad=False``（``model_builders.py:161-162``），随后
``_replace_visual_lora_modules`` 只替换**命中 target 的**线性层 ⇒ 视觉塔零可训参数，
运行期预检抛

    RuntimeError: multimodal preflight failed: no trainable visual parameters found.
    vision_common LoRA target may not be registered. Refusing to start training.

本文件锁两件事（都不需要 GPU/真权重）：

1. **机制归因**：用真 ``_replace_visual_lora_modules`` + 真 ``LoRALinear`` 在一个
   与 Qwen3.5 视觉塔同构的模块树上跑两遍——``language_safe`` ⇒ 0 个可训视觉参数
   （复现 T028 失败态），``vision_common`` ⇒ 非零（证明**代码侧注册链路是好的**，
   坏的只是"配置没选视觉预设"）；
2. **前置防线**：``assert_lora_vision_targets_trainable`` 把同一判据提前到配置期，
   报错点名该改的键（``lora.target_preset``）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from graspo.core.lora import LORA_TARGET_PRESETS, resolve_lora_target_modules
from graspo.flow.lora.lora_helpers import native_qwen_lora_available_targets
from graspo.flow.lora.lora_linear import LoRALinear, _replace_visual_lora_modules
from graspo.flow.trainer.preflight import assert_lora_vision_targets_trainable

_VISION_DEPTH = 3


class _FakeVisionTower(nn.Module):
    """与 Qwen3.5 视觉塔同构的最小模块树（merger + depth×blocks）。

    路径与 ``_replace_visual_lora_modules`` 的 ``target_to_path`` 一一对应；
    各线性层形状取真实 ViT 的典型值，但不需要任何权重文件。
    """

    def __init__(self, depth: int = _VISION_DEPTH) -> None:
        super().__init__()
        merger = nn.Module()
        merger.linear_fc1 = nn.Linear(8, 16)
        merger.linear_fc2 = nn.Linear(16, 8)
        self.merger = merger
        blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Module()
            attn = nn.Module()
            attn.qkv = nn.Linear(8, 24)
            attn.proj = nn.Linear(8, 8)
            block.attn = attn
            mlp = nn.Module()
            mlp.linear_fc1 = nn.Linear(8, 16)
            mlp.linear_fc2 = nn.Linear(16, 8)
            block.mlp = mlp
            blocks.append(block)
        self.blocks = blocks


def _hf_config_stub(depth: int = _VISION_DEPTH):
    """``native_qwen_lora_available_targets`` 只读这几个属性（避免加载真 config）。"""

    class _Cfg:
        family = "qwen3_5_text"
        has_vision_config = True
        vision_config = {"depth": depth}
        num_hidden_layers = 4
        layer_types: list[str] = []

    return _Cfg()


def _build_tower_with_targets(target_preset: str) -> nn.Module:
    """按 **真实链路** 装配视觉塔：解析 target → 冻结 → 替换 LoRA。

    返回 **模型根**（``visual`` 作为子模块挂在它下面），因为真实参数名前缀是
    ``model.visual.…`` / ``visual.…``——预检判据 ``"visual" in name`` 依赖这个前缀。
    """
    available = native_qwen_lora_available_targets(_hf_config_stub())
    resolved = resolve_lora_target_modules([target_preset], available=available)
    visual = _FakeVisionTower()
    for param in visual.parameters():  # model_builders.py:161-162 的同一步
        param.requires_grad = False
    _replace_visual_lora_modules(
        visual,
        lora_targets=set(resolved.resolved),
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
    )
    root = nn.Module()
    root.visual = visual  # 复刻 `build_native_qwen_model` 把视觉塔挂在 model.visual
    return root


def _trainable_visual_params(module: nn.Module) -> list[str]:
    """运行期预检的同一判据（preflight.py:118-122）。"""
    return [
        name
        for name, param in module.named_parameters()
        if "visual" in name and param.requires_grad
    ]


class TestMechanismAttribution:
    """真模块替换证据：失败态能复现、修好态确实可训。"""

    def test_language_safe_preset_leaves_visual_tower_untrainable(self) -> None:
        """T028 失败态复现：language_safe ⇒ 0 个可训视觉参数（= 预检判据触发）。"""
        visual = _build_tower_with_targets("language_safe")
        assert _trainable_visual_params(visual) == []

    def test_vision_common_preset_makes_visual_tower_trainable(self) -> None:
        """修好态：vision_common ⇒ 视觉 LoRA 参数真的可训（代码链路本身没问题）。"""
        visual = _build_tower_with_targets("vision_common")
        trainable = _trainable_visual_params(visual)
        assert trainable, "vision_common 没产出可训视觉参数 —— 注册链路坏了"
        # merger(2) + depth × (attn.qkv + attn.proj + mlp.fc1 + mlp.fc2) 各 lora_a/lora_b
        assert len(trainable) == (2 + 4 * _VISION_DEPTH) * 2

    def test_vision_common_target_names_match_available_targets(self) -> None:
        """预设模式必须能被真模型给出的可用目标名解析出来（否则预检必炸）。"""
        available = native_qwen_lora_available_targets(_hf_config_stub())
        resolved = set(resolve_lora_target_modules(["vision_common"], available=available).resolved)
        visual = resolved & {name for name in available if name.startswith("visual.")}
        assert len(visual) == 2 + 4 * _VISION_DEPTH
        # 预设里写的是 glob（visual.blocks.*.attn.*），必须真的展开到逐层精确名
        assert "visual.blocks.0.attn.qkv" in visual
        assert "visual.merger.linear_fc1" in visual

    def test_visual_presets_are_declared(self) -> None:
        """防线用的预设表与 core/lora.py 是同一份（不另立判据，§1.4）。"""
        assert set(LORA_TARGET_PRESETS) >= {"vision_common", "vision_merger", "language_safe"}
        assert all(
            pattern.startswith("visual.")
            for name in ("vision_common", "vision_merger")
            for pattern in LORA_TARGET_PRESETS[name]
        )

    def test_full_param_shape_has_no_lora_matrix_at_all(self) -> None:
        """★ 全参（``lora_r=0``）的机制事实：LoRA 矩阵**根本不存在**，可训载体是参数自身。

        这是 B3 的实证：``build_native_qwen_model`` 在全参下传 ``lora_r=0`` ⇒
        ``LoRALinear.lora_enabled=False``、``lora_a/lora_b=None``（``lora_linear.py:123-133``）
        ⇒ ``enabled_lora_target_names()`` 结构性恒空，用"LoRA 目标覆盖视觉塔"判全参必误报。
        随后 ``build_qwen35_visual_tower`` 的 ``full_param`` 分支（``model_builders.py:173-177``）
        把塔内参数统一放开——此时"可训视觉参数"判据仍然非空（防线强度不倒退）。
        """
        visual = _FakeVisionTower()
        for param in visual.parameters():
            param.requires_grad = False
        _replace_visual_lora_modules(
            visual,
            lora_targets={"visual.merger.linear_fc1"},
            lora_r=0,  # ← 全参模式的形状（model.py:63）
            lora_alpha=16,
            lora_dropout=0.0,
            device=torch.device("cpu"),
            torch_dtype=torch.float32,
        )
        root = nn.Module()
        root.visual = visual

        replaced = visual.merger.linear_fc1
        assert isinstance(replaced, LoRALinear)
        assert replaced.lora_enabled is False, "r=0 时不得启用 LoRA"
        assert replaced.lora_a is None and replaced.lora_b is None
        assert [n for n, _ in root.named_parameters() if "lora_" in n] == [], (
            "全参模式不得存在任何 lora_* 参数 ⇒ LoRA 目标判据无对象可判"
        )
        assert _trainable_visual_params(root) == []  # 全参替换后、放开 requires_grad 前

        for param in visual.parameters():  # build_qwen35_visual_tower 的 full_param 分支
            param.requires_grad = True
        assert _trainable_visual_params(root), "全参放开后视觉塔必须可训（防线不倒退）"


class TestConfigTimeGuard:
    """配置期 fail-closed：报错点名该改的键，且不误伤合法组合。"""

    IMAGE_TOKEN_ID = 248056  # Qwen3.5-9B 实测值（T028 stdout）

    def test_language_preset_with_vision_model_raises(self) -> None:
        with pytest.raises(ValueError, match="select no visual module"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                lora_target_preset="language_safe",
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )

    def test_default_preset_none_with_vision_model_raises(self) -> None:
        """不写 lora 段（target_preset=None）也是 language_safe 默认 ⇒ 同样拦。"""
        with pytest.raises(ValueError, match="select no visual module"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                lora_target_preset=None,
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )

    def test_explicit_language_only_modules_raise(self) -> None:
        with pytest.raises(ValueError, match="select no visual module"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=["language.full_attn.q_proj"],
                lora_target_preset=None,
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )

    def test_error_message_names_the_config_key(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                lora_target_preset="language_safe",
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )
        message = str(excinfo.value)
        for needle in (
            "lora.target_preset",
            "vision_common",
            "no trainable visual parameters found",
            "requires_grad=False",
        ):
            assert needle in message, needle

    @pytest.mark.parametrize("preset", ["vision_common", "vision_merger"])
    def test_vision_presets_pass(self, preset: str) -> None:
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            lora_target_preset=preset,
            image_token_id=self.IMAGE_TOKEN_ID,
            model_name="Qwen3.5-9B",
        )

    def test_full_mode_language_preset_is_not_judged(self) -> None:
        """★ 全参（``tuner_type="full"``）下"LoRA 目标未覆盖视觉塔"不适用 ⇒ 放行。

        这是 T017/T018/T035/T036 的失败态回归：全参档不写 ``lora`` 段，
        ``target_preset`` 只是 schema 默认值（``core/schema.py``），
        在 LoRA 模式下必报错、在全参模式下必须放行。
        """
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            lora_target_preset="language_safe",
            image_token_id=self.IMAGE_TOKEN_ID,
            model_name="Qwen3.5-9B",
            tuner_type="full",
        )

    def test_full_mode_still_rejects_missing_vision_token(self) -> None:
        """模式分支不得吞掉与模式无关的配置不一致校验（仍 fail-closed）。"""
        with pytest.raises(ValueError, match="no usable image token id"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                lora_target_preset="vision_common",
                image_token_id=None,
                model_name="Qwen3.5-9B",
                has_vision_config=True,
                tuner_type="full",
            )

    def test_explicit_visual_modules_pass(self) -> None:
        assert_lora_vision_targets_trainable(
            lora_target_modules=["visual.merger.linear_fc1"],
            lora_target_preset=None,
            image_token_id=self.IMAGE_TOKEN_ID,
            model_name="Qwen3.5-9B",
        )

    def test_text_only_model_is_not_judged(self) -> None:
        """模型没有视觉塔（image_token_id=None）⇒ 语言-only 预设完全合法。"""
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            lora_target_preset="language_safe",
            image_token_id=None,
            model_name="Qwen3-8B",
        )
