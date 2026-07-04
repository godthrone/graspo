"""GPU tests for Qwen3 model layers — BADGE §11.1 (CUDA required)."""

import pytest

torch = pytest.importorskip("torch", reason="torch required")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Qwen3 layer tests", allow_module_level=True)

from graspo.backends.graspoflow.models.common.layers_qwen3 import (  # noqa: E402
    QwenRMSNorm,
    TensorParallelQwenAttention,
    TensorParallelQwenDecoderLayer,
    TensorParallelQwenMLP,
)


class FakeSafetensorIndex:
    """A minimal fake weight loader for testing layer construction."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        intermediate_size: int = 256,
        num_layers: int = 1,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.head_dim = hidden_size // num_heads

    def get(self, key: str):
        shape = self._shape_for(key)
        return torch.randn(*shape) * 0.02

    def get_optional(self, key: str):
        if "bias" in key or "q_norm" in key or "k_norm" in key:
            return torch.randn(self.hidden_size) * 0.02
        return None

    def _shape_for(self, key: str):
        if "q_proj" in key or "k_proj" in key or "v_proj" in key or "o_proj" in key:
            return (self.hidden_size, self.hidden_size)
        if "gate_proj" in key or "up_proj" in key:
            return (self.intermediate_size, self.hidden_size)
        if "down_proj" in key:
            return (self.hidden_size, self.intermediate_size)
        if "layernorm" in key.lower() or "norm" in key.lower():
            return (self.hidden_size,)
        return (self.hidden_size, self.hidden_size)


class FakeHFConfig:
    def __init__(
        self,
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        head_dim=None,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.head_dim = head_dim


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def config():
    return FakeHFConfig()


@pytest.fixture
def loader():
    return FakeSafetensorIndex()


# ── QwenRMSNorm ────────────────────────────────────────────────────────────


def test_rms_norm_output_shape(device):
    norm = QwenRMSNorm(hidden_size=64, eps=1e-6, device=device, dtype=torch.float32)
    x = torch.randn(2, 10, 64, device=device)
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_rms_norm_zero_mean_unit_variance_approx(device):
    norm = QwenRMSNorm(hidden_size=128, eps=1e-6, device=device, dtype=torch.float64)
    x = torch.randn(4, 20, 128, device=device, dtype=torch.float64)
    out = norm(x)
    # RMS should be approximately 1.0
    rms = torch.sqrt((out.float() ** 2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


# ── TensorParallelQwenAttention ────────────────────────────────────────────


def test_attention_output_shape(device, config, loader):
    attn = TensorParallelQwenAttention(
        prefix="model.layers.0.self_attn",
        hf_config=config,
        loader=loader,
        tp_rank=0,
        tp_size=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=set(),
        torch_dtype=torch.float32,
        device=device,
    )
    x = torch.randn(2, 16, 64, device=device)
    pos_ids = torch.arange(16, device=device).unsqueeze(0).expand(2, -1)
    out = attn(x, pos_ids, attention_mask=None)
    assert out.shape == (2, 16, 64)


# ── TensorParallelQwenMLP ──────────────────────────────────────────────────


def test_mlp_output_shape(device, config, loader):
    mlp = TensorParallelQwenMLP(
        prefix="model.layers.0.mlp",
        hf_config=config,
        loader=loader,
        tp_rank=0,
        tp_size=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=set(),
        torch_dtype=torch.float32,
        device=device,
    )
    x = torch.randn(2, 16, 64, device=device)
    out = mlp(x)
    assert out.shape == (2, 16, 64)


# ── TensorParallelQwenDecoderLayer ─────────────────────────────────────────


def test_decoder_layer_output_shape(device, config, loader):
    layer = TensorParallelQwenDecoderLayer(
        layer_idx=0,
        key_prefix="model",
        hf_config=config,
        loader=loader,
        tp_rank=0,
        tp_size=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_targets=set(),
        torch_dtype=torch.float32,
        device=device,
    )
    x = torch.randn(2, 16, 64, device=device)
    pos_ids = torch.arange(16, device=device).unsqueeze(0).expand(2, -1)
    out = layer(x, pos_ids, attention_mask=None)
    assert out.shape == (2, 16, 64)
