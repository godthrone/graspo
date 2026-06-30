"""Tests for ``graspo.trainer.checkpoint`` — LoRA adapter save utilities."""

from pathlib import Path
from unittest import mock

import pytest

from graspo.trainer.checkpoint import save_lora_adapter


def test_save_lora_adapter_delegates_to_save_pretrained(tmp_path):
    """save_lora_adapter calls model.save_pretrained and tokenizer.save_pretrained."""
    model = mock.MagicMock()
    # MagicMock has a .module attribute by default, so hasattr(model, "module")==True.
    # Remove it so the model is treated as a plain (non-DDP) model.
    del model.module
    tokenizer = mock.MagicMock()
    output_dir = tmp_path / "lora_adapter"

    save_lora_adapter(model, tokenizer, output_dir)

    assert output_dir.is_dir()
    model.save_pretrained.assert_called_once_with(Path(output_dir))
    tokenizer.save_pretrained.assert_called_once_with(Path(output_dir))


def test_save_lora_adapter_unwraps_ddp_module(tmp_path):
    """When model has .module (DDP wrapper), save_pretrained is called on the inner model."""
    inner = mock.MagicMock()
    outer = mock.MagicMock()
    outer.module = inner
    # Remove .module from inner so it is used directly
    del inner.module
    tokenizer = mock.MagicMock()
    output_dir = tmp_path / "ddp_lora"

    save_lora_adapter(outer, tokenizer, output_dir)

    inner.save_pretrained.assert_called_once_with(Path(output_dir))
    # outer (DDP wrapper) should NOT be asked to save
    outer.save_pretrained.assert_not_called()


def test_save_lora_adapter_tokenizer_none_skips_tokenizer(tmp_path):
    """save_lora_adapter with tokenizer=None only saves the model."""
    model = mock.MagicMock()
    del model.module
    output_dir = tmp_path / "model_only"

    save_lora_adapter(model, None, output_dir)

    model.save_pretrained.assert_called_once_with(Path(output_dir))
