"""Tests for ``graspo.backends.graspoflow.lora`` — native LoRA operations."""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)


def test_lora_module_is_importable():
    """Verify the native LoRA module is importable."""
    from graspo.backends.graspoflow import lora  # noqa: F401
