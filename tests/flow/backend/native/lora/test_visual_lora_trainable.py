"""native 视觉塔 LoRA 可训性：配置期防线 + 真模块替换证据（阻断 1 的回归测试）。

**背景（实测 T028，2026-09-19）**：清单档 T028（9B / GRASPO / LoRA / native / 1 卡）
的 LoRA 目标只覆盖语言侧，而 ``build_qwen35_visual_tower`` 先把视觉塔全部参数
``requires_grad=False``（``model_builders.py:161-162``），随后
``_replace_visual_lora_modules`` 只替换**命中 target 的**线性层 ⇒ 视觉塔零可训参数，
运行期预检抛

    RuntimeError: multimodal preflight failed: no trainable visual parameters found.
    Refusing to start training.

**本轮（2026-10-21）契约变更**：目标写法统一为**一条真实 HF 模块名正则**
（``LORA_TARGET_MODULES_REGEX``），因此本文件里的"目标"全部改成 **HF 模块名**
（``model.visual.*``）——判据域与 ms-swift/PEFT 逐字相同；旧 preset 名
（``language_safe`` / ``vision_common``）**不再被接受**（fail-closed）。

本文件锁两件事（都不需要 GPU/真权重）：

1. **机制归因**：用真 ``_replace_visual_lora_modules`` + 真 ``LoRALinear`` 在一个
   与 Qwen3.5 视觉塔同构的模块树上跑两遍——语言-only 正则 ⇒ 0 个可训视觉参数
   （复现 T028 失败态），共享正则 ⇒ 非零（证明**代码侧注册链路是好的**，
   坏的只是"正则没选到视觉"）；
2. **前置防线**：``assert_lora_vision_targets_trainable`` 把同一判据提前到配置期，
   报错点名该改的键（``lora.target_modules``）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from graspo.core.lora import (
    LORA_TARGET_MODULES_REGEX,
    LORA_TARGET_PRESETS,
    resolve_lora_target_modules,
)
from graspo.flow.backend.native.lora.lora_helpers import (
    _lora_target_enabled,
    native_qwen_lora_available_targets,
)
from graspo.flow.backend.native.lora.lora_linear import LoRALinear, _replace_visual_lora_modules
from graspo.flow.backend.native.trainer.preflight import assert_lora_vision_targets_trainable

_VISION_DEPTH = 3

#: 语言-only 正则（能编、能命中语言侧，但**训不到视觉塔**）= T028 的失败态。
_LANGUAGE_ONLY_REGEX = r"^(?:model\.)?language_model\..*\.(q_proj|v_proj)$"

#: 只覆盖视觉塔的自定义正则（证明"不一定非用共享那条"）。
_VISUAL_ONLY_REGEX = r"^(?:model\.)?visual\..*$"


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
    """``native_qwen_lora_available_targets`` 只读这几个属性（避免加载真 config）。

    ★ ``layer_types`` 按 transformers 的真实派生规则给出
    （``configuration_qwen3_5.py:106-109``：``(i+1) % full_attention_interval``），
    长度必须等于 ``num_hidden_layers`` —— 不再用空列表糊弄（新实现对此 fail-closed
    报错：不知道每层注意力种类就无法枚举真实模块名，⛔ 不猜）。
    """

    class _Cfg:
        family = "qwen3_5_text"
        key_prefix = "model.language_model"
        has_vision_config = True
        vision_config = {"depth": depth}
        num_hidden_layers = 4
        # interval=4 ⇒ 第 4 层（i+1=4）是 full_attention，其余是 linear_attention
        layer_types = [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]

    return _Cfg()


def _build_tower_with_targets(target_modules: str) -> nn.Module:
    """按 **真实链路** 装配视觉塔：解析 target → 冻结 → 替换 LoRA。

    返回 **模型根**（``visual`` 作为子模块挂在它下面），因为真实参数名前缀是
    ``model.visual.…``——预检判据 ``"visual" in name`` 与 LoRA 判据
    （``model.visual.`` 前缀）都依赖这个前缀。
    """
    available = native_qwen_lora_available_targets(_hf_config_stub())
    resolved = resolve_lora_target_modules(target_modules, available=available)
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
    """运行期预检的同一判据（preflight 的 ``no trainable visual parameters``）。"""
    return [
        name
        for name, param in module.named_parameters()
        if "visual" in name and param.requires_grad
    ]


class TestMechanismAttribution:
    """真模块替换证据：失败态能复现、修好态确实可训。"""

    def test_language_only_regex_leaves_visual_tower_untrainable(self) -> None:
        """T028 失败态复现：语言-only 正则 ⇒ 0 个可训视觉参数（= 预检判据触发）。"""
        root = _build_tower_with_targets(_LANGUAGE_ONLY_REGEX)
        assert _trainable_visual_params(root) == []

    # ── ★ 阳性对照 ③（判据域必须与正则同域：真实 HF 名 ``model.visual.*``）──────
    # 能让本测试变红的精确读法：把 `src/graspo/flow/backend/native/lora/lora_linear.py`
    # 里 `_lora_target_enabled(lora_targets, hf_module_path)` 改回用 native 规范名
    # `target_name`（即 `visual.merger.linear_fc1` 这种）⇒ 与正则的匹配域错位、
    # 一个视觉模块都命中不了 ⇒ 下面两条断言立刻红。
    def test_positive_control_visual_judgement_is_in_hf_name_domain(self) -> None:
        """共享正则在**真链路**上确实训到视觉塔（判据域 = 真实 HF 模块名）。"""
        root = _build_tower_with_targets(LORA_TARGET_MODULES_REGEX)
        trainable = _trainable_visual_params(root)
        assert trainable, "共享正则没产出可训视觉参数 —— 判据域与正则不匹配（注册链路坏了）"
        # merger(2) + depth × (attn.qkv + attn.proj + mlp.fc1 + mlp.fc2) 各 lora_a/lora_b
        assert len(trainable) == (2 + 4 * _VISION_DEPTH) * 2

    def test_visual_only_regex_makes_visual_tower_trainable(self) -> None:
        """不必非用共享那条：任何能命中真实视觉模块名的正则都行（不硬编码某一条）。"""
        root = _build_tower_with_targets(_VISUAL_ONLY_REGEX)
        assert len(_trainable_visual_params(root)) == (2 + 4 * _VISION_DEPTH) * 2

    def test_resolved_visual_targets_match_available_visual_targets(self) -> None:
        """共享正则解析出的视觉目标 == 可用视觉目标全集（逐层精确名，无 glob 残留）。"""
        available = native_qwen_lora_available_targets(_hf_config_stub())
        resolved = set(
            resolve_lora_target_modules(LORA_TARGET_MODULES_REGEX, available=available).resolved
        )
        visual = resolved & {name for name in available if name.startswith("model.visual.")}
        assert len(visual) == 2 + 4 * _VISION_DEPTH
        assert "model.visual.blocks.0.attn.qkv" in visual
        assert "model.visual.merger.linear_fc1" in visual
        assert all(name.startswith("model.visual.") for name in visual)

    def test_full_param_shape_has_no_lora_matrix_at_all(self) -> None:
        """★ 全参（``lora_r=0``）的机制事实：LoRA 矩阵**根本不存在**，可训载体是参数自身。

        这是 B3 的实证：``build_native_qwen_model`` 在全参下传 ``lora_r=0`` ⇒
        ``LoRALinear.lora_enabled=False``、``lora_a/lora_b=None``（``lora_linear.py``）
        ⇒ ``enabled_lora_target_names()`` 结构性恒空，用"LoRA 目标覆盖视觉塔"判全参必误报。
        随后 ``build_qwen35_visual_tower`` 的 ``full_param`` 分支（``model_builders.py:173-177``）
        把塔内参数统一放开——此时"可训视觉参数"判据仍然非空（防线强度不倒退）。
        """
        visual = _FakeVisionTower()
        for param in visual.parameters():
            param.requires_grad = False
        _replace_visual_lora_modules(
            visual,
            lora_targets={"model.visual.merger.linear_fc1"},
            lora_r=0,  # ← 全参模式的形状
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

    def test_lora_target_enabled_has_no_leaf_name_alias(self) -> None:
        """判据是**精确 HF 名**：叶子名 / native 规范名都不是别名（⛔ 不给隐式兼容）。"""
        targets = {"model.visual.merger.linear_fc1"}
        assert _lora_target_enabled(targets, "model.visual.merger.linear_fc1") is True
        assert _lora_target_enabled(targets, "linear_fc1") is False
        assert _lora_target_enabled(targets, "visual.merger.linear_fc1") is False

    def test_legacy_preset_table_is_inert_not_an_alias(self) -> None:
        """``LORA_TARGET_PRESETS`` 仍在表里，但**解析器不再消费它**（preset 名不是别名）。

        该表保留的唯一理由是**冻结产物契约**（``samples/configs/matrix54/`` 与
        ``tests/e2e/generate_matrix.py`` 仍按 ``lora.target_preset`` 校验），
        不是"新路径的兼容层"：所以 preset 名喂给解析器必须报错。
        """
        available = native_qwen_lora_available_targets(_hf_config_stub())
        assert set(LORA_TARGET_PRESETS) >= {"vision_common", "vision_merger", "language_safe"}
        for preset_name in ("vision_common", "vision_merger", "language_safe"):
            with pytest.raises(ValueError):
                resolve_lora_target_modules(preset_name, available=available)
            with pytest.raises(ValueError):
                resolve_lora_target_modules([preset_name], available=available)  # type: ignore[arg-type]


class TestConfigTimeGuard:
    """配置期 fail-closed：报错点名该改的键，且不误伤合法组合。"""

    IMAGE_TOKEN_ID = 248056  # Qwen3.5-9B 实测值（T028 stdout）

    def test_language_only_regex_with_vision_model_raises(self) -> None:
        with pytest.raises(ValueError, match="select no visual module"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=_LANGUAGE_ONLY_REGEX,
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )

    def test_missing_target_modules_with_vision_model_raises(self) -> None:
        """不写 ``lora.target_modules`` ⇒ 判否（fail-closed，不再有"默认预设"兜底）。"""
        with pytest.raises(ValueError, match="select no visual module"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )

    def test_old_preset_name_and_list_are_rejected(self) -> None:
        """旧 preset 名 / preset 名列表 **在这里也是 fail-closed**（⛔ 不静默放行）。

        ★ 实际上它们到不了这里——``core/schema.py::LoRAConfig`` 在配置加载期就拒了
        （``extra="forbid"`` 之外还有形态校验）；本测试锁的是"防线不会因为形态不认识
        而默默放行"。
        """
        for bad in ("language_safe", "vision_common", ["language_all_linear", "vision_common"]):
            with pytest.raises(ValueError, match="select no visual module"):
                assert_lora_vision_targets_trainable(
                    lora_target_modules=bad,  # type: ignore[arg-type]
                    image_token_id=self.IMAGE_TOKEN_ID,
                    model_name="Qwen3.5-9B",
                )

    def test_error_message_names_the_config_key(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            assert_lora_vision_targets_trainable(
                lora_target_modules=_LANGUAGE_ONLY_REGEX,
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )
        message = str(excinfo.value)
        for needle in (
            "lora.target_modules",
            "model.visual.merger.linear_fc1",
            "no trainable visual parameters found",
            "requires_grad=False",
        ):
            assert needle in message, needle

    def test_shared_regex_and_visual_regex_pass(self) -> None:
        for good in (LORA_TARGET_MODULES_REGEX, _VISUAL_ONLY_REGEX):
            assert_lora_vision_targets_trainable(
                lora_target_modules=good,
                image_token_id=self.IMAGE_TOKEN_ID,
                model_name="Qwen3.5-9B",
            )

    def test_full_mode_language_only_regex_is_not_judged(self) -> None:
        """★ 全参（``tuner_type="full"``）下"LoRA 目标未覆盖视觉塔"不适用 ⇒ 放行。

        这是 T017/T018/T035/T036 的失败态回归：全参档不写 ``lora`` 段，
        在 LoRA 模式下必报错、在全参模式下必须放行。
        """
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            image_token_id=self.IMAGE_TOKEN_ID,
            model_name="Qwen3.5-9B",
            tuner_type="full",
        )

    def test_full_mode_still_rejects_missing_vision_token(self) -> None:
        """模式分支不得吞掉与模式无关的配置不一致校验（仍 fail-closed）。"""
        with pytest.raises(ValueError, match="no usable image token id"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=LORA_TARGET_MODULES_REGEX,
                image_token_id=None,
                model_name="Qwen3.5-9B",
                has_vision_config=True,
                tuner_type="full",
            )

    def test_text_only_model_is_not_judged(self) -> None:
        """模型没有视觉塔（image_token_id=None）⇒ 无目标也不判（不误伤纯文本档）。"""
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            image_token_id=None,
            model_name="Qwen3-8B",
        )
