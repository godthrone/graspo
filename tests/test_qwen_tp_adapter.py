from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from graspo.backends.megatron_native.qwen_tp_adapter import (  # noqa: E402
    LoRALinear,
    QwenMegatronNativeAdapter,
    TensorParallelQwenForCausalLM,
    collate_experiences,
)
from graspo.backends.megatron_native.runtime import MegatronNativeRuntime  # noqa: E402
from graspo.core.buffer import Experience  # noqa: E402
from graspo.core.schema import GraspoConfig  # noqa: E402


def test_runtime_uses_builtin_qwen_adapter_by_default(monkeypatch):
    monkeypatch.delenv("GRASPO_MEGATRON_NATIVE_ADAPTER", raising=False)
    monkeypatch.setattr(MegatronNativeRuntime, "validate", lambda self: None)
    monkeypatch.setattr(
        "graspo.backends.megatron_native.qwen_tp_adapter.QwenMegatronNativeAdapter.setup",
        lambda self: None,
    )

    runtime = MegatronNativeRuntime(GraspoConfig())
    runtime.setup()

    assert runtime._adapter.__class__.__name__ == "QwenMegatronNativeAdapter"


def test_lora_linear_shards_hf_weights_by_tensor_parallel_rank():
    weight = torch.arange(32, dtype=torch.float32).view(8, 4)

    out_shard = LoRALinear.from_hf(
        weight,
        bias=None,
        shard="out",
        tp_rank=1,
        tp_size=2,
        lora_enabled=True,
        r=2,
        alpha=4,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    in_shard = LoRALinear.from_hf(
        weight,
        bias=None,
        shard="in",
        tp_rank=1,
        tp_size=2,
        lora_enabled=False,
        r=0,
        alpha=1,
        dropout=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert out_shard.weight.shape == (4, 4)
    assert in_shard.weight.shape == (8, 2)
    assert out_shard.lora_a.shape == (2, 4)
    assert out_shard.lora_b.shape == (4, 2)


def test_collate_experiences_pads_sequence_and_action_tensors():
    items = [
        Experience(
            sequences=torch.tensor([1, 2, 3]),
            old_log_probs=torch.tensor([0.1, 0.2]),
            advantages=torch.tensor([1.0, 1.0]),
            attention_mask=torch.tensor([True, True, True]),
            action_mask=torch.tensor([False, True]),
            rewards=torch.tensor(1.0),
        ),
        Experience(
            sequences=torch.tensor([1, 4]),
            old_log_probs=torch.tensor([0.3]),
            advantages=torch.tensor([-1.0]),
            attention_mask=torch.tensor([True, True]),
            action_mask=torch.tensor([True]),
            rewards=torch.tensor(0.0),
        ),
    ]

    batch = collate_experiences(items, torch.device("cpu"))

    assert batch.sequences.tolist() == [[1, 2, 3], [1, 4, 0]]
    assert batch.old_log_probs.shape == (2, 2)
    assert batch.action_mask.tolist() == [[False, True], [True, False]]


def test_qwen_cache_forward_matches_full_forward_for_next_token():
    torch.manual_seed(1234)
    config = SimpleNamespace(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    model = TensorParallelQwenForCausalLM(
        hf_config=config,
        loader=_TinyQwenLoader(config),
        tp_rank=0,
        tp_size=1,
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_targets=set(),
        gradient_checkpointing=False,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        full_logits = model(input_ids, attention_mask=attention_mask)
        prefill_logits, past_key_values = model(
            input_ids[:, :3],
            attention_mask=attention_mask[:, :3],
            use_cache=True,
        )
        cached_logits, next_past = model(
            input_ids[:, 3:],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    assert torch.allclose(prefill_logits, full_logits[:, :3], atol=1e-5, rtol=1e-5)
    assert torch.allclose(cached_logits[:, -1], full_logits[:, 3], atol=1e-5, rtol=1e-5)
    assert len(next_past) == config.num_hidden_layers
    assert next_past[0][0].shape[2] == input_ids.shape[1]


def test_qwen_activation_checkpointing_only_wraps_training_forward(monkeypatch):
    torch.manual_seed(1234)
    config = SimpleNamespace(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    model = TensorParallelQwenForCausalLM(
        hf_config=config,
        loader=_TinyQwenLoader(config),
        tp_rank=0,
        tp_size=1,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_targets={"q_proj"},
        gradient_checkpointing=True,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    calls = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(
        "graspo.backends.megatron_native.qwen_tp_adapter.activation_checkpoint",
        fake_checkpoint,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    model.train()
    _ = model(input_ids, attention_mask=attention_mask)

    assert len(calls) == config.num_hidden_layers
    assert calls[0]["use_reentrant"] is False
    assert calls[0]["preserve_rng_state"] is True

    calls.clear()
    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
        _ = model(input_ids, attention_mask=attention_mask, use_cache=True)

    assert calls == []


def test_qwen_kv_cache_estimate_is_per_tp_rank():
    config = SimpleNamespace(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    model = TensorParallelQwenForCausalLM(
        hf_config=config,
        loader=_TinyQwenLoader(config),
        tp_rank=0,
        tp_size=2,
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_targets=set(),
        gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert model.estimate_kv_cache_bytes(batch_size=8, sequence_len=2048) == 8 * 3 * 2 * 1 * 4 * 2048 * 2


def test_adapter_training_indices_are_stable_without_mutating_global_random_state():
    config = GraspoConfig.from_dict({"training": {"seed": 123}})
    adapter = QwenMegatronNativeAdapter(config)
    before = torch.random.get_rng_state().clone()

    first = adapter._shared_training_indices(8, optimize_round=0)
    second = adapter._shared_training_indices(8, optimize_round=1)
    adapter._train_batch_call_index += 1
    third = adapter._shared_training_indices(8, optimize_round=0)

    assert sorted(first) == list(range(8))
    assert sorted(second) == list(range(8))
    assert sorted(third) == list(range(8))
    assert first != second
    assert first != third
    assert torch.equal(before, torch.random.get_rng_state())


def test_adapter_generation_micro_batch_uses_shared_dispatch_when_distributed(monkeypatch):
    config = GraspoConfig.from_dict(
        {
            "backend_config": {
                "megatron_native": {
                    "generation_micro_batch_size": 1,
                    "use_kv_cache_for_rollout": True,
                }
            }
        }
    )
    adapter = QwenMegatronNativeAdapter(config)
    adapter.rank = 1
    adapter.device = torch.device("cuda")
    monkeypatch.setattr(
        adapter,
        "_resolve_generation_micro_batch_size",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "graspo.backends.megatron_native.qwen_tp_adapter.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "graspo.backends.megatron_native.qwen_tp_adapter.dist.is_initialized",
        lambda: True,
    )

    def fake_broadcast_object_list(payload, src):
        assert src == 0
        assert payload == [None]
        payload[0] = 4

    monkeypatch.setattr(
        "graspo.backends.megatron_native.qwen_tp_adapter.dist.broadcast_object_list",
        fake_broadcast_object_list,
    )

    assert (
        adapter._shared_generation_micro_batch_size(
            prompt_len=128,
            rollout_group_size=8,
            max_new_tokens=2048,
            use_kv_cache=True,
        )
        == 4
    )


class _TinyQwenLoader:
    def __init__(self, config) -> None:
        self.config = config
        self.tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size) * 0.02,
            "model.norm.weight": torch.ones(config.hidden_size),
            "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size) * 0.02,
        }
        for idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{idx}"
            self.tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(config.hidden_size)
            self.tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(config.hidden_size)
            self.tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size) * 0.02
            self.tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size) * 0.02
            self.tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size) * 0.02
            self.tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(config.hidden_size, config.hidden_size) * 0.02
            self.tensors[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(config.head_dim)
            self.tensors[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(config.head_dim)
            self.tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(config.intermediate_size, config.hidden_size) * 0.02
            self.tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(config.intermediate_size, config.hidden_size) * 0.02
            self.tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(config.hidden_size, config.intermediate_size) * 0.02

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def get_optional(self, name: str) -> torch.Tensor | None:
        return self.tensors.get(name)
