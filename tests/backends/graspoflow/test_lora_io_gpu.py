"""GPU tests for LoRA weight I/O roundtrip — BADGE §11.1 (CUDA required)."""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError, reason="torch required")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for LoRA I/O GPU tests", allow_module_level=True)


from graspo.backends.graspoflow.lora import LoRALinear  # noqa: E402
from graspo.backends.graspoflow.lora_io import (  # noqa: E402
    load_lora_weights,
    save_lora_weights,
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path / "lora_test"


def test_save_and_load_lora_weights_roundtrip(temp_dir):
    """LoRA weights survive a save→load roundtrip."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    # Create a LoRALinear
    linear = LoRALinear.from_hf(
        torch.randn(64, 64),
        bias=None,
        shard="none",
        tp_rank=0,
        tp_size=1,
        lora_enabled=True,
        target_name="test.layer",
        hf_module_path="test.layer",
        r=8,
        alpha=16,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
    )

    # Set known values for lora_a and lora_b
    linear.lora_a.weight.data.fill_(0.1)
    linear.lora_b.weight.data.fill_(0.2)

    # Save
    save_lora_weights(linear, temp_dir)

    # Check files exist
    assert (temp_dir / "adapter_model.safetensors").exists()

    # Load into a new LoRALinear
    linear2 = LoRALinear.from_hf(
        torch.randn(64, 64),
        bias=None,
        shard="none",
        tp_rank=0,
        tp_size=1,
        lora_enabled=True,
        target_name="test.layer",
        hf_module_path="test.layer",
        r=8,
        alpha=16,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
    )
    load_lora_weights(linear2, temp_dir, device=device)

    # Verify weights match
    assert torch.allclose(linear2.lora_a.weight, linear.lora_a.weight)
    assert torch.allclose(linear2.lora_b.weight, linear.lora_b.weight)


def test_save_lora_weights_creates_config_json(temp_dir):
    """Saving LoRA weights creates expected metadata files."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    linear = LoRALinear.from_hf(
        torch.randn(32, 32),
        bias=None,
        shard="none",
        tp_rank=0,
        tp_size=1,
        lora_enabled=True,
        target_name="q_proj",
        hf_module_path="model.q_proj",
        r=4,
        alpha=8,
        dropout=0.1,
        device=device,
        dtype=torch.float32,
    )

    save_lora_weights(linear, temp_dir)
    assert (temp_dir / "adapter_config.json").exists()
