"""防线/守卫缺陷修复的负向用例（纯逻辑，不触 GPU、不加载权重、不连网络）。

覆盖缺陷②与③：

- ②``adapter.py`` 的 fail-open 豁免：**修复前**"模型声明了视觉、本 rank 又持有
  embedding、视觉塔却没建出来"被静默放过；**修复后** fail-closed 报错。
  合法情形（模型无视觉 / 视觉塔在别的 PP stage）必须**不误伤**。
- ③native SFT 缺运行期预检：**修复前** SFT 在"视觉塔冻死"下静默开训；
  **修复后**启动即报错。纯文本模型 / 语言-only 配置必须**不误伤**。

这些用例在修复前的 HEAD 上会真失败（见 ``test_b2_*`` / ``test_c3_*`` 的
"修复前一直为真"断言与 ``TestSFTPreflightWiring``），不是假绿。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from graspo.flow.trainer.preflight import (
    assert_lora_vision_targets_trainable,
    assert_vision_tower_trainable,
)
from graspo.flow.trainer.sft_trainer import SFTTrainer


class _FakeVisionModel:
    """最小原生多模态模型鸭子类型：只有 ``visual`` 与 ``enabled_lora_target_names``。

    ``visual=None`` 模拟"本 rank 上视觉塔没建出来"（缺陷②的静默降级形态）。
    """

    def __init__(
        self,
        *,
        has_visual_tower: bool,
        enabled_visual_targets: tuple[str, ...] = ("visual.merger.linear_fc1",),
    ) -> None:
        self.visual = object() if has_visual_tower else None
        self.config = SimpleNamespace(
            has_vision_config=True,
            image_token_id=151655,
            vision_config={"depth": 27},
        )
        self._enabled = enabled_visual_targets

    def enabled_lora_target_names(self) -> tuple[str, ...]:
        return ("language.full_attn.q_proj", *self._enabled)


class _FakeTextModel:
    """纯文本模型：``has_vision_config=False``，连 ``visual`` 属性都没有。"""

    def __init__(self) -> None:
        self.config = SimpleNamespace(has_vision_config=False, image_token_id=None)

    def enabled_lora_target_names(self) -> tuple[str, ...]:
        return ("language.full_attn.q_proj",)


# ── 缺陷②：adapter fail-open 豁免的判据 ──────────────────────────────────────


class TestVisionTowerTrainable:
    def test_b1_declared_vision_without_tower_on_embedding_rank_raises(self) -> None:
        """★ 负向（修复后必须报错）：声明视觉 + 持有 embedding + 无视觉塔。"""
        model = _FakeVisionModel(has_visual_tower=False)
        with pytest.raises(RuntimeError, match="visual tower was not built"):
            assert_vision_tower_trainable(
                model, model.config, owns_embeddings=True, model_name="Qwen3.5-9B"
            )

    def test_b1_pre_fix_condition_would_have_silently_exempted(self) -> None:
        """★ 鉴别力证据：修复前那条内联豁免条件**正是为这个情形**返回"豁免"。

        旧实现（HEAD ``adapter.py:98``）：
        ``not (target.startswith("visual.") and getattr(self.model, "visual", None) is None)``
        ⇒ 对 ``visual.*`` + ``visual is None`` 求值 False ⇒ 该 target **被从
        ``missing_lora_targets`` 里剔除** ⇒ 不报错、静默继续。
        """
        model = _FakeVisionModel(has_visual_tower=False)
        target = "visual.merger.linear_fc1"
        old_suppressed = not (
            target.startswith("visual.") and getattr(model, "visual", None) is None
        )
        assert old_suppressed is False, "旧豁免条件应把该 target 剔除（=fail-open）"
        # 修复后的判据在同一情形下必须报错——两者结论相反。
        with pytest.raises(RuntimeError):
            assert_vision_tower_trainable(
                model, model.config, owns_embeddings=True, model_name="Qwen3.5-9B"
            )

    def test_b2_declared_vision_with_tower_but_no_trainable_visual_lora_raises(self) -> None:
        """★ 负向：视觉塔在，但没有任何可训视觉 LoRA target（视觉塔冻死）。"""
        model = _FakeVisionModel(has_visual_tower=True, enabled_visual_targets=())
        with pytest.raises(RuntimeError, match="no trainable visual LoRA target"):
            assert_vision_tower_trainable(
                model, model.config, owns_embeddings=True, model_name="Qwen3.5-9B"
            )

    def test_b3_text_only_model_is_not_harmed(self) -> None:
        """放行：纯文本模型不得被这条判据误伤。"""
        model = _FakeTextModel()
        assert_vision_tower_trainable(
            model, model.config, owns_embeddings=True, model_name="Qwen3-8B"
        )

    def test_b4_vision_tower_on_another_pp_stage_is_not_harmed(self) -> None:
        """放行：模型有视觉但本 rank 不是 embedding stage（PP>1）⇒ 无塔属设计正常。

        若把这条判据收紧成"无塔即报错"，pp_size>1 的视觉训练会被误伤致死。
        """
        model = _FakeVisionModel(has_visual_tower=False)
        assert_vision_tower_trainable(
            model, model.config, owns_embeddings=False, model_name="Qwen3.5-9B"
        )

    def test_b5_healthy_vision_model_passes(self) -> None:
        model = _FakeVisionModel(has_visual_tower=True)
        assert_vision_tower_trainable(
            model, model.config, owns_embeddings=True, model_name="Qwen3.5-9B"
        )


class TestVisionTokenConsistency:
    """`assert_lora_vision_targets_trainable` 的 fail-open 收紧（默认行为不变）。"""

    def test_b6_default_has_vision_config_none_keeps_legacy_semantics(self) -> None:
        """★ 不放松也不改变既有语义：不传 `has_vision_config` 时，无视觉 token ⇒ 放行。"""
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            lora_target_preset="language_safe",
            image_token_id=None,
            model_name="Qwen3-8B",
        )

    def test_b7_declared_vision_without_image_token_raises(self) -> None:
        """★ 负向（修复后必须报错）：声明了视觉塔却没有可用视觉占位 token。"""
        with pytest.raises(ValueError, match="no usable image token id"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                lora_target_preset="vision_common",
                image_token_id=None,
                model_name="Qwen3.5-9B",
                has_vision_config=True,
            )

    def test_b8_language_only_preset_on_vision_model_still_raises(self) -> None:
        """既有判据未放松：有视觉塔 + 语言-only 预设 ⇒ 照旧报错。"""
        with pytest.raises(ValueError, match="select no visual module"):
            assert_lora_vision_targets_trainable(
                lora_target_modules=None,
                lora_target_preset="language_safe",
                image_token_id=151655,
                model_name="Qwen3.5-9B",
                has_vision_config=True,
            )

    def test_b9_vision_preset_on_vision_model_passes(self) -> None:
        assert_lora_vision_targets_trainable(
            lora_target_modules=None,
            lora_target_preset="vision_common",
            image_token_id=151655,
            model_name="Qwen3.5-9B",
            has_vision_config=True,
        )


# ── 缺陷③：native SFT 的运行期预检接线 ───────────────────────────────────────


class _FakeSFTConfig:
    def __init__(self, *, target_preset: str | None, target_modules: list[str] | None) -> None:
        self.lora = SimpleNamespace(target_modules=target_modules, target_preset=target_preset)
        self.model = SimpleNamespace(model_path="/models/Qwen3.5-9B")


class _FakeRuntime:
    def __init__(self, adapter: object) -> None:
        self._adapter = adapter


class _FakeAdapter:
    def __init__(self, model: object) -> None:
        self.model = model


def _make_sft_trainer(*, model: object, target_preset: str | None) -> SFTTrainer:
    trainer = SFTTrainer.__new__(SFTTrainer)  # 绕过 __init__（避免构造 runtime）
    trainer.config = _FakeSFTConfig(target_preset=target_preset, target_modules=None)
    trainer.runtime = _FakeRuntime(_FakeAdapter(model))
    return trainer


class TestSFTPreflightWiring:
    def test_c1_wiring_exists(self) -> None:
        """★ 修复前失败：HEAD 的 SFTTrainer 根本没有 `_preflight_multimodal`。"""
        assert hasattr(SFTTrainer, "_preflight_multimodal")

    def test_c2_wiring_called_in_train_before_data_load(self) -> None:
        """★ 修复前失败：train() 源码里既无该方法名，也无任何 preflight 导入。"""
        src = inspect.getsource(SFTTrainer.train)
        assert "self._preflight_multimodal()" in src
        assert src.index("self._preflight_multimodal()") < src.index("load_jsonl(")

    def test_c3_frozen_vision_tower_now_raises(self) -> None:
        """★ 核心负向：视觉塔不可训 + 配置声明要训视觉 ⇒ 修复前静默、修复后报错。"""
        model = _FakeVisionModel(has_visual_tower=True, enabled_visual_targets=())
        trainer = _make_sft_trainer(model=model, target_preset="language_safe")
        with pytest.raises(ValueError, match="select no visual module"):
            trainer._preflight_multimodal()

    def test_c3_pre_fix_sft_had_no_preflight_at_all(self) -> None:
        """★ 鉴别力证据：修复前 SFT 训练入口**没有任何** preflight 引用。

        HEAD 的 ``sft_trainer.py`` 全文不含 "from graspo.flow.trainer.preflight
        import" ⇒ SFT 在"视觉塔冻死"下静默训练；GRASPO 路径的预检只在
        ``trainer.py`` 里。
        """
        import graspo.flow.trainer.sft_trainer as sft_module

        preflight_import = "from graspo.flow.trainer.preflight import"
        # train() 只调用该方法，不自行导入第二套判据（单一真相源 §1.4）。
        assert preflight_import in inspect.getsource(SFTTrainer._preflight_multimodal)
        assert preflight_import not in inspect.getsource(SFTTrainer.train)
        # 修复前整个模块都没有这条导入；现在只应出现在 _preflight_multimodal 里。
        assert preflight_import in inspect.getsource(sft_module)

    def test_c4_text_only_model_is_not_harmed(self) -> None:
        """放行：纯文本模型（has_vision_config=False）零误伤、零开销。"""
        trainer = _make_sft_trainer(model=_FakeTextModel(), target_preset="language_safe")
        trainer._preflight_multimodal()  # 不抛异常

    def test_c5_vision_model_with_vision_preset_passes(self) -> None:
        model = _FakeVisionModel(has_visual_tower=True)
        trainer = _make_sft_trainer(model=model, target_preset="vision_common")
        trainer._preflight_multimodal()  # 不抛异常

    def test_c6_declared_vision_without_image_token_raises(self) -> None:
        model = _FakeVisionModel(has_visual_tower=True)
        model.config.image_token_id = None  # 配置不一致
        trainer = _make_sft_trainer(model=model, target_preset="vision_common")
        with pytest.raises(ValueError, match="no usable image token id"):
            trainer._preflight_multimodal()

    def test_c7_missing_visual_tower_raises(self) -> None:
        model = _FakeVisionModel(has_visual_tower=False)
        trainer = _make_sft_trainer(model=model, target_preset="vision_common")
        with pytest.raises(RuntimeError, match="visual tower was not built"):
            trainer._preflight_multimodal()

    def test_c8_non_native_model_is_not_harmed(self) -> None:
        """放行：非原生 LoRA 模型（无 model.config）不受这条判据影响。"""
        trainer = _make_sft_trainer(model=SimpleNamespace(), target_preset="language_safe")
        trainer._preflight_multimodal()  # 不抛异常

    def test_c9_adapter_not_loaded_raises(self) -> None:
        trainer = _make_sft_trainer(model=_FakeTextModel(), target_preset="language_safe")
        trainer.runtime = _FakeRuntime(None)
        with pytest.raises(RuntimeError, match="adapter not loaded"):
            trainer._preflight_multimodal()
