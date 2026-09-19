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
from graspo.flow.lora.lora_linear import _replace_visual_lora_modules
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
