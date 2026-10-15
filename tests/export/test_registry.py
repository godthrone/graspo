"""注册表不变量（T12）：每个 ``CheckpointForm`` 都有处置、唯一分派点、未登记即 fail-closed。"""

from __future__ import annotations

import pytest

from graspo.export import registry
from graspo.export.errors import ExportError
from graspo.export.schema import CheckpointForm, ExportTarget, UnsupportedForm


def test_every_form_has_a_registered_disposition():
    registered = set(registry.registered_form_names())
    missing = [form.value for form in CheckpointForm if form.value not in registered]
    assert not missing, f"forms without a disposition (T12 violation): {missing}"


def test_positive_control_dropping_a_registration_is_detected(monkeypatch):
    """阳性对照：摘掉一个登记项 ⇒ 覆盖度不变量与分派点都必须报错。"""
    real = dict(registry._registry())
    incomplete = {key: value for key, value in real.items() if key != "native-lora-rank"}
    monkeypatch.setattr(registry, "_discover", lambda group: incomplete)
    registered = set(registry.registered_form_names())
    assert CheckpointForm.NATIVE_LORA_RANK.value not in registered

    with pytest.raises(ExportError, match="no export disposition registered"):
        registry.get_disposition(CheckpointForm.NATIVE_LORA_RANK)


def test_resolve_dispatches_native_lora_to_its_exporter():
    from graspo.export.exporters.native_lora import NativeLoraRankExporter

    assert (
        registry.resolve(CheckpointForm.NATIVE_LORA_RANK, ExportTarget.PEFT_ADAPTER)
        is NativeLoraRankExporter
    )
    assert (
        registry.resolve(CheckpointForm.NATIVE_LORA_RANK, ExportTarget.MERGED_HF)
        is NativeLoraRankExporter
    )


def test_native_full_param_resolves_to_its_real_exporter():
    """F7（P5 落地后）：全参单卡不再是声明式拒绝，而由真实导出器承接。

    保留"死亡形态不得静默缺登记"的判据：它必须是**已登记处置**（T12 覆盖度不变量），
    且 ``merged-hf`` 可解析、``peft-adapter`` 明确拒绝（全参模型无 LoRA 载体）。
    """
    from graspo.export.exporters.native_full_param import NativeFullParamRankExporter

    assert not isinstance(
        registry.get_disposition(CheckpointForm.NATIVE_FULL_PARAM_RANK), UnsupportedForm
    )
    assert (
        registry.resolve(CheckpointForm.NATIVE_FULL_PARAM_RANK, ExportTarget.MERGED_HF)
        is NativeFullParamRankExporter
    )
    with pytest.raises(ExportError, match="cannot be exported as 'peft-adapter'"):
        registry.resolve(CheckpointForm.NATIVE_FULL_PARAM_RANK, ExportTarget.PEFT_ADAPTER)


def test_peft_adapter_is_declared_unsupported_until_p3():
    """P1 交付时 PEFT 合并属 P3 ⇒ 必须**显式声明**（T12 覆盖度），⛔ 不退化成"未知形态"。"""
    disposition = registry.get_disposition(CheckpointForm.PEFT_ADAPTER)
    assert isinstance(disposition, UnsupportedForm)
    assert disposition.detector is not None
    with pytest.raises(ExportError, match="explicitly unsupported"):
        registry.resolve(CheckpointForm.PEFT_ADAPTER, ExportTarget.MERGED_HF)


def test_base_model_exporter_supports_only_merged_hf():
    with pytest.raises(ExportError, match="cannot be exported as 'peft-adapter'"):
        registry.resolve(CheckpointForm.HF_BASE_MODEL, ExportTarget.PEFT_ADAPTER)


def test_unknown_form_is_fail_closed():
    with pytest.raises(ExportError, match="no export disposition registered"):
        registry.get_disposition("not-a-form")
