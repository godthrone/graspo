from __future__ import annotations

import pytest

from graspo.backends.selector import looks_like_large_model, select_backend
from graspo.core.schema import GraspoConfig


def test_backend_auto_uses_hf_reference_without_runtime(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 0)
    monkeypatch.setattr("graspo.backends.selector._has_module", lambda name: False)
    config = GraspoConfig()
    selection = select_backend(config)
    assert selection.name == "hf-reference"


def test_backend_auto_fails_for_large_model_without_megatron(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 8)
    monkeypatch.setattr("graspo.backends.selector._has_native_megatron_runtime", lambda: False)
    monkeypatch.setattr("graspo.backends.selector._has_module", lambda name: False)
    config = GraspoConfig()
    config.model.model_path = "/models/Qwen3.6-27B"
    with pytest.raises(RuntimeError, match="megatron-native backend is required"):
        select_backend(config)


def test_backend_auto_uses_native_megatron_when_runtime_detected(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 8)
    monkeypatch.setattr("graspo.backends.selector._has_native_megatron_runtime", lambda: True)
    monkeypatch.setattr("graspo.backends.selector._has_module", lambda name: False)
    config = GraspoConfig()
    selection = select_backend(config)
    assert selection.name == "megatron-native"


def test_backend_rejects_legacy_megatron_vllm():
    config = GraspoConfig()
    with pytest.raises(ValueError, match="Unsupported backend"):
        select_backend(config, requested="megatron-vllm")


def test_backend_explicit_hf_reference_ignores_megatron(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 8)
    monkeypatch.setattr("graspo.backends.selector._has_native_megatron_runtime", lambda: False)
    config = GraspoConfig()
    selection = select_backend(config, requested="hf-reference")
    assert selection.name == "hf-reference"


def test_large_model_name_detection():
    assert looks_like_large_model("/data/Qwen3.6-27B")
    assert looks_like_large_model("/data/model-32b")
    assert not looks_like_large_model("/data/tiny-1b")
