"""全参多模态编码上下文补丁（``_inference_mode_compat``）的契约。

**本文件回答四个问题**

1. **红→绿**：ms-swift ``Template.encode`` 在 ``torch.inference_mode()`` 下产出的
   ``pixel_values`` 是 inference tensor；把它喂给**可训练**的视觉塔 ``patch_embed``
   （conv3d，与 Qwen VL 同型）**必然**抛
   ``RuntimeError: Inference tensors cannot be saved for backward``——补丁装上以后
   同一个张量能反传，且 ``patch_embed.weight.grad`` 非零（证明视觉塔真的在被训练，
   不是被这段包装顺手冻住了）。
2. **等价性（逐位）**：同一模板、同一批"样本"，打补丁前后的编码产物
   （``input_ids`` / ``pixel_values`` / ``image_grid_thw`` / ``labels``）**逐位相同**
   ——补丁只换上下文，不动编码逻辑。
3. **作用面**：``tuner_type="lora"`` 时**连包装都不装**（类属性对象同一性 + 计数器
   双重断言）；只有 ``full`` 且模板带视觉塔时才改写；纯文本模型的 ``full`` 直通。
4. **升级韧性**：上游去掉装饰器 ⇒ 无害直通（``installed=False``）；装饰器链换形态
   导致改写无效 ⇒ **报错**（不静默通过）；上游符号缺失 ⇒ **报错并说清缺什么**。

**为什么不用 ms-swift 也能测**

被测的是**graspo 自己的接线与判定**，上游只需"同形的 ``encode``"：本文件用真
``torch`` + 一个同形替身模板类（``model_meta`` / ``@torch.inference_mode() encode``），
真实 ms-swift 上的红→绿证据在工位报告里（真 ``Template.encode`` + 真 processor，CPU）。
"""

from __future__ import annotations

import sys

import pytest
import torch

from graspo.flow.backend.msswift import _inference_mode_compat as compat

# ── 替身：真 torch 的"视觉塔"与"同形模板" ─────────────────────────────────


class _VisionTower(torch.nn.Module):
    """与 Qwen VL 视觉塔同型的极小替身：conv3d patch_embed → merger。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Conv3d(3, 8, kernel_size=2)
        self.merger = torch.nn.Linear(8, 8)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embed(pixel_values).flatten(2).mean(-1)
        return self.merger(hidden)


class _ModelArch:
    def __init__(self, vision_tower: object) -> None:
        self.vision_tower = vision_tower


class _ModelMeta:
    def __init__(self, *, is_multimodal: bool = True, vision_tower: object = ("model.visual",)):
        self.is_multimodal = is_multimodal
        self.model_arch = _ModelArch(vision_tower)


_MM = _ModelMeta()


def _encoded(n: int = 2) -> dict[str, torch.Tensor]:
    """确定性编码产物（同一输入两次调用逐位相同，便于等价性断言）。"""
    pixel_values = torch.arange(n * 24, dtype=torch.float32).reshape(n, 3, 2, 2, 2) / 97.0
    return {
        "input_ids": torch.arange(n * 3),
        "labels": torch.arange(n * 3),
        "pixel_values": pixel_values,
        "image_grid_thw": torch.tensor([[1, 2, 2]] * n),
    }


def _template_cls(*, model_meta: object = _MM, encode=None) -> type:
    """同形模板类：``encode`` 被 ``torch.inference_mode()`` 装饰（复刻 swfit base.py:652）。"""

    class _Template:
        def __init__(self, meta: object = None) -> None:
            self.model_meta = model_meta if meta is None else meta

    if encode is None:

        def encode(self, n: int = 2) -> dict[str, torch.Tensor]:
            return _encoded(n)

    else:
        encode = encode

    _Template.encode = torch.inference_mode()(encode)
    return _Template


@pytest.fixture(autouse=True)
def _clean_stats():
    compat.reset_encode_patch_stats()
    yield
    compat.reset_encode_patch_stats()


# ── ① 红→绿：inference tensor 不能进可训练视觉塔 ────────────────────────────


def test_red_inference_pixel_values_cannot_be_saved_for_backward():
    """★ 红：未打补丁的编码产物喂给可训练视觉塔 ⇒ 原文 RuntimeError。"""
    tower = _VisionTower()
    with torch.inference_mode():
        pixel_values = _encoded(2)["pixel_values"]
    assert pixel_values.is_inference() is True

    with pytest.raises(RuntimeError, match="Inference tensors cannot be saved for backward"):
        tower(pixel_values).sum().backward()


def test_detach_does_not_clear_inference_flag():
    """实测坑：``detach()`` 不解除 inference 属性（所以修法不能是"包一层 detach"）。"""
    tower = _VisionTower()
    with torch.inference_mode():
        pixel_values = _encoded(2)["pixel_values"]

    detached = pixel_values.detach()
    assert detached.is_inference() is True
    with pytest.raises(RuntimeError, match="Inference tensors cannot be saved for backward"):
        tower(detached).sum().backward()

    with torch.no_grad():
        detached_no_grad = pixel_values.detach()
    assert detached_no_grad.is_inference() is True
    with pytest.raises(RuntimeError, match="Inference tensors cannot be saved for backward"):
        tower(detached_no_grad).sum().backward()


def test_green_full_patch_makes_backward_work_and_vision_grads_flow():
    """★ 绿：``tuner_type=full`` 下同一个视觉塔能反传，且 ``patch_embed.weight.grad`` 非零。"""
    cls = _template_cls()
    template = cls(_MM)
    tower = _VisionTower()

    with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
        encoded = template.encode()
        assert encoded["pixel_values"].is_inference() is False
        loss = tower(encoded["pixel_values"]).sum()
    loss.backward()

    grad = tower.patch_embed.weight.grad
    assert grad is not None, "vision tower patch_embed 没拿到梯度 —— 等于被顺手冻住了"
    assert float(grad.abs().sum()) > 0.0
    assert compat.encode_patch_stats()["rewritten_calls"] == 1


def test_teacher_side_params_stay_frozen_while_student_trains():
    """反向断言：教师参数零梯度（GKD 结构：教师 logits 是 no_grad 目标，不是被训练方）。"""
    cls = _template_cls()
    template = cls(_MM)
    student = _VisionTower()
    teacher = _VisionTower()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
        pixel_values = template.encode()["pixel_values"]
        with torch.no_grad():
            teacher_logits = teacher(pixel_values)
        student_logits = student(pixel_values)
        loss = torch.nn.functional.kl_div(
            torch.log_softmax(student_logits, dim=-1),
            torch.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
    loss.backward()

    assert float(student.patch_embed.weight.grad.abs().sum()) > 0.0
    assert all(p.grad is None for p in teacher.parameters()), "教师侧有梯度 ⇒ 修法把教师也训了"


# ── ② 等价性：补丁前后编码产物逐位相同 ─────────────────────────────────────


@pytest.mark.parametrize("key", ["input_ids", "labels", "pixel_values", "image_grid_thw"])
def test_rewrite_is_bitwise_identical_to_unpatched_encode(key: str):
    """★ 同一模板同一批"样本"：打补丁前后逐位相同（1 ulp 都不许差）。"""
    cls = _template_cls()
    template = cls(_MM)

    unpatched = template.encode()  # 上游行为（inference_mode 下）
    assert unpatched[key].is_inference() is True

    with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
        patched = template.encode()
    assert patched[key].is_inference() is False

    assert torch.equal(unpatched[key], patched[key]), f"{key} 在补丁前后不一致"
    assert patched[key].dtype == unpatched[key].dtype
    assert patched[key].shape == unpatched[key].shape


def test_rewrite_calls_the_upstream_function_object_itself():
    """补丁不许自造编码逻辑：被调用的必须是上游剥掉装饰器后的那个函数对象。"""
    calls: list[int] = []

    def encode(self, n: int = 2):
        calls.append(n)
        return _encoded(n)

    cls = _template_cls(encode=encode)
    upstream_inner = cls.encode.__wrapped__
    assert upstream_inner is encode

    with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
        assert cls.encode.__wrapped__ is upstream_inner
        cls(_MM).encode(3)
    assert calls == [3]


# ── ③ 作用面：LoRA 不装、纯文本直通 ───────────────────────────────────────


def test_lora_does_not_install_the_patch():
    """★ LoRA 档：包装**不装**，上游对象同一性不变，产物仍是 inference tensor（逐字保持）。"""
    cls = _template_cls()
    original = cls.encode
    template = cls(_MM)

    with compat.graph_safe_encode(tuner_type="lora", template_cls=cls):
        assert cls.encode is original, "LoRA 档被动过了 —— 公共路径不许被这段代码污染"
        assert template.encode()["pixel_values"].is_inference() is True

    stats = compat.encode_patch_stats()
    assert stats["installed"] is False
    assert stats["rewritten_calls"] == 0
    assert stats["passthrough_calls"] == 0
    assert cls.encode is original


def test_full_text_only_template_passes_through():
    """``full`` 但模型没有视觉塔：没有消费者要保存这个输入 ⇒ 直通上游，不改写。"""
    text_only = _ModelMeta(is_multimodal=False, vision_tower=())
    cls = _template_cls(model_meta=text_only)
    template = cls(text_only)

    with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
        encoded = template.encode()
        assert encoded["pixel_values"].is_inference() is True
        assert compat.encode_patch_stats()["passthrough_calls"] == 1
    assert compat.encode_patch_stats()["rewritten_calls"] == 0


def test_patch_is_restored_on_exit():
    """作用域退出即还原（异常安全）：类属性回到上游对象，产物回到 inference tensor。"""
    cls = _template_cls()
    original = cls.encode
    template = cls(_MM)

    with pytest.raises(ValueError):
        with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
            assert cls.encode is not original
            raise ValueError("boom")

    assert cls.encode is original
    assert compat.encode_patch_stats()["installed"] is False
    assert template.encode()["pixel_values"].is_inference() is True


# ── ④ 升级韧性：不静默通过 ─────────────────────────────────────────────────


def test_undecorated_encode_degrades_to_passthrough():
    """上游已去掉 ``inference_mode`` 装饰器（已修）⇒ 无害直通，并记下原因。"""

    class _Undecorated:
        def __init__(self, meta: object = _MM) -> None:
            self.model_meta = meta

        def encode(self, n: int = 2) -> dict[str, torch.Tensor]:
            return _encoded(n)

    with compat.graph_safe_encode(tuner_type="full", template_cls=_Undecorated):
        stats = compat.encode_patch_stats()
        assert stats["installed"] is False
        assert "already fixed" in str(stats["degraded_reason"])
        assert _Undecorated(_MM).encode()["pixel_values"].is_inference() is False


def test_decorator_chain_change_fails_closed():
    """剥一层后仍是 inference tensor ⇒ 报错点名残留的键（绝不"看起来修了"）。"""

    def encode(self, n: int = 2):
        with torch.inference_mode():  # 装饰器链换了形态：内层又开了 inference_mode
            return _encoded(n)

    cls = _template_cls(encode=encode)
    with pytest.raises(RuntimeError, match="still contains inference tensors"):
        with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
            cls(_MM).encode()


def test_missing_encode_symbol_fails_closed():
    """上游符号缺失（改版/未安装）⇒ 报错并说清缺什么。"""
    with pytest.raises(RuntimeError, match="swift.template.base.Template.encode"):
        with compat.graph_safe_encode(tuner_type="full", template_cls=object()):
            pytest.fail("不该走到这里")


def test_missing_swift_package_fails_closed(monkeypatch):
    """ms-swift 不可导入 ⇒ 同一份可操作提示（不是静默降级）。"""
    monkeypatch.setitem(sys.modules, "swift", None)
    with pytest.raises(RuntimeError, match="swift.template.base.Template.encode"):
        compat._require_swift_template()


def test_missing_model_meta_fails_closed():
    """模板实例没有 ``model_meta`` ⇒ 无法判定有没有视觉塔 ⇒ fail-closed。"""
    cls = _template_cls()
    template = cls(_MM)
    template.model_meta = None
    with pytest.raises(RuntimeError, match="model_meta"):
        with compat.graph_safe_encode(tuner_type="full", template_cls=cls):
            template.encode()


# ── ⑤ 通道级接线：真 MsSwiftOpdTrainer.train()（只把 ms-swift 换成替身）────────


def _opd_config(tmp_path, *, tuner_type: str | None = None):
    """``train_method="opd"`` + ``backend="msswift"`` 的最小真配置。"""
    from graspo.core.schema import GraspoConfig

    sample = tmp_path / "ard.jsonl"
    sample.write_text(
        '{"id":"s1","source":"unit","data_source":"ard_text","schema_version":"3.0.0",'
        '"messages":[{"role":"user","content":"抬起手臂"}],'
        '"targets":[{"id":null,"output":{"content":"lift_arm({})","reasoning":null}}]}\n',
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "train_method": "opd",
        "backend": "msswift",
        "model": {"model_path": str(tmp_path / "Qwen3.5-9B")},
        "data": {"train_path": str(sample)},
        "training": {
            "output_dir": str(tmp_path / "out"),
            "run_name": "unit",
            "overwrite_output_dir": True,
            "seed": 1234,
        },
        "distill": {"teacher_model_path": str(tmp_path / "Qwen3.8-27B")},
    }
    if tuner_type is not None:
        payload["tuner_type"] = tuner_type
    return GraspoConfig.model_validate(payload)


def _run_opd_train(tmp_path, monkeypatch, *, tuner_type, inside):
    """跑真 ``MsSwiftOpdTrainer.train()``；``inside()`` 在替身 ms-swift 入口内部执行。"""
    import types

    from graspo.flow.backend.msswift.opd_trainer import MsSwiftOpdTrainer

    def _fake_rlhf_main(argv):
        inside()

    pipelines_mod = types.ModuleType("swift.pipelines")
    pipelines_mod.rlhf_main = _fake_rlhf_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swift", types.ModuleType("swift"))
    monkeypatch.setitem(sys.modules, "swift.pipelines", pipelines_mod)
    monkeypatch.setattr(
        "graspo.flow.backend.msswift._config_mapping.graspo_to_ms_swift_argv",
        lambda *a, **k: ["--rlhf_type", "gkd"],
    )
    monkeypatch.setattr(
        "graspo.flow.backend.msswift.dataset.prepare_ms_swift_dataset",
        lambda *a, **k: str(tmp_path / "opd.jsonl"),
    )
    MsSwiftOpdTrainer(_opd_config(tmp_path, tuner_type=tuner_type), None).train(smoke=True)


def test_opd_channel_full_installs_patch_end_to_end(tmp_path, monkeypatch):
    """★ 全参 OPD 通道：训练期间 ``Template.encode`` 已被换成 ``no_grad`` 版本。"""
    import types

    cls = _template_cls()
    template_mod = types.ModuleType("swift.template.base")
    template_mod.Template = cls  # type: ignore[attr-defined]
    template_pkg = types.ModuleType("swift.template")
    template_pkg.base = template_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swift.template", template_pkg)
    monkeypatch.setitem(sys.modules, "swift.template.base", template_mod)

    seen: dict[str, object] = {}

    def _inside() -> None:
        seen["installed"] = compat.encode_patch_stats()["installed"]
        seen["is_inference"] = cls(_MM).encode()["pixel_values"].is_inference()

    _run_opd_train(tmp_path, monkeypatch, tuner_type="full", inside=_inside)

    assert seen["installed"] is True, "全参 OPD 通道没装编码上下文补丁 —— 接线没通"
    assert seen["is_inference"] is False, "训练期编码仍产出 inference tensor"
    assert compat.encode_patch_stats()["installed"] is False, "训练结束未还原"


def test_opd_channel_lora_never_touches_the_patch(tmp_path, monkeypatch):
    """★ LoRA OPD 通道：连 ``swift.template`` 都不导入，包装不装（公共路径零污染）。"""
    seen: dict[str, object] = {}

    def _inside() -> None:
        seen["template_imported"] = "swift.template.base" in sys.modules
        seen["installed"] = compat.encode_patch_stats()["installed"]

    _run_opd_train(tmp_path, monkeypatch, tuner_type=None, inside=_inside)

    assert seen["template_imported"] is False, "LoRA 档碰了模板模块 —— 作用面没收窄"
    assert seen["installed"] is False
    assert compat.encode_patch_stats()["rewritten_calls"] == 0
