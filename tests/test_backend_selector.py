from __future__ import annotations

import pytest

from graspo.backends.selector import looks_like_large_model, select_backend
from graspo.core.schema import GraspoConfig


def test_backend_auto_uses_hf_reference_for_small_local_model_without_gpus(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 0)
    config = GraspoConfig()
    selection = select_backend(config)
    assert selection.name == "hf-reference"


def test_backend_auto_uses_native_tp_for_large_model_without_external_runtime(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 0)
    config = GraspoConfig()
    config.model.model_path = "/models/Qwen3.6-27B"
    selection = select_backend(config)
    assert selection.name == "native-tp"
    assert selection.native_tp_available is True


def test_backend_auto_uses_native_tp_when_multiple_gpus_detected(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 8)
    config = GraspoConfig()
    selection = select_backend(config)
    assert selection.name == "native-tp"


def test_backend_rejects_legacy_megatron_vllm():
    config = GraspoConfig()
    with pytest.raises(ValueError, match="Unsupported backend"):
        select_backend(config, requested="megatron-vllm")


def test_backend_rejects_removed_megatron_native_name():
    config = GraspoConfig()
    with pytest.raises(ValueError, match="Unsupported backend"):
        select_backend(config, requested="megatron-native")


def test_backend_explicit_hf_reference_works_on_multi_gpu(monkeypatch):
    monkeypatch.setattr("graspo.backends.selector.detect_gpu_count", lambda: 8)
    config = GraspoConfig()
    selection = select_backend(config, requested="hf-reference")
    assert selection.name == "hf-reference"


def test_large_model_name_detection():
    assert looks_like_large_model("/data/Qwen3.6-27B")
    assert looks_like_large_model("/data/model-32b")
    assert not looks_like_large_model("/data/tiny-1b")
