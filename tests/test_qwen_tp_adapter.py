from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from graspo.backends.native_tp.qwen_tp_adapter import (  # noqa: E402
    LoRALinear,
    NativeTPCausalLMBase,
    QwenNativeTPAdapter,
    Qwen35HybridTextModel,
    Qwen3DenseModel,
    NativeQwenConfig,
    TensorParallelQwen35LinearAttention,
    TensorParallelQwenForCausalLM,
    TensorParallelQwen35TextForCausalLM,
    collate_experiences,
    load_native_qwen_config,
    native_qwen_lora_available_targets,
    _add_pipeline_stage_timing,
    _messages_from_multimodal_row,
    _new_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
)
from graspo.backends.native_tp.runtime import NativeTPRuntime  # noqa: E402
from graspo.backends.native_tp.placement import build_placement_plan  # noqa: E402
from graspo.core.buffer import Experience  # noqa: E402
from graspo.core.schema import GraspoConfig  # noqa: E402


def test_runtime_uses_builtin_qwen_adapter_by_default(monkeypatch):
    monkeypatch.delenv("GRASPO_NATIVE_TP_ADAPTER", raising=False)
    monkeypatch.setattr(NativeTPRuntime, "validate", lambda self: None)
    monkeypatch.setattr(
        "graspo.backends.native_tp.qwen_tp_adapter.QwenNativeTPAdapter.setup",
        lambda self: None,
    )

    runtime = NativeTPRuntime(GraspoConfig())
    runtime.setup()

    assert runtime._adapter.__class__.__name__ == "QwenNativeTPAdapter"


def test_multimodal_row_messages_are_preserved_for_processor_template():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "images/a.png"},
                {"type": "text", "text": "q2"},
            ],
        },
    ]

    assert _messages_from_multimodal_row({"messages": messages}) == messages


def test_qwen_format_messages_passes_tools_to_chat_template():
    class Tokenizer:
        chat_template = "template"

        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "rendered"

    tokenizer = Tokenizer()
    adapter = QwenNativeTPAdapter(GraspoConfig())
    adapter.tokenizer = tokenizer
    messages = [{"role": "user", "content": "query status"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "query_device_status", "parameters": {"type": "object"}},
        }
    ]

    rendered = adapter._format_messages(messages, {"enable_thinking": False}, tools=tools)

    assert rendered == "rendered"
    assert tokenizer.calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "tools": tools,
            },
        )
    ]


def test_native_qwen_lora_available_targets_cover_text_and_visual():
    qwen3 = NativeQwenConfig(
        {"num_hidden_layers": 2},
        family="qwen3",
        key_prefix="model",
    )
    qwen35 = NativeQwenConfig(
        {
            "num_hidden_layers": 2,
            "has_vision_config": True,
            "vision_config": {"depth": 2},
        },
        family="qwen3_5_text",
        key_prefix="model.language_model",
    )

    assert "language.self_attn.q_proj" in native_qwen_lora_available_targets(qwen3)
    qwen35_targets = native_qwen_lora_available_targets(qwen35)
    assert "language.linear_attn.in_proj_z" in qwen35_targets
    assert "visual.merger.linear_fc1" in qwen35_targets
    assert "visual.blocks.1.attn.qkv" in qwen35_targets


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


def test_pipeline_stage_timing_keeps_train_backward_breakdown(monkeypatch):
    counter = {"now": 100.0}

    def fake_monotonic():
        counter["now"] += 0.25
        return counter["now"]

    monkeypatch.setattr("graspo.backends.native_tp.qwen_tp_adapter.time.monotonic", fake_monotonic)
    timing = _new_pipeline_stage_timing()

    for key in (
        "pipeline_norm_sec",
        "pipeline_lm_head_sec",
        "pipeline_loss_sec",
        "pipeline_backward_autograd_sec",
        "pipeline_grad_clip_sec",
        "pipeline_optimizer_step_sec",
    ):
        _add_pipeline_stage_timing(timing, key, fake_monotonic())

    rounded = _round_pipeline_stage_timing(timing)

    assert rounded["pipeline_norm_sec"] == 0.25
    assert rounded["pipeline_lm_head_sec"] == 0.25
    assert rounded["pipeline_loss_sec"] == 0.25
    assert rounded["pipeline_backward_autograd_sec"] == 0.25
    assert rounded["pipeline_grad_clip_sec"] == 0.25
    assert rounded["pipeline_optimizer_step_sec"] == 0.25


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
        "graspo.backends.native_tp.qwen_tp_adapter.activation_checkpoint",
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

    assert (
        model.estimate_kv_cache_bytes(batch_size=8, sequence_len=2048)
        == 8 * 3 * 2 * 1 * 4 * 2048 * 2
    )


def test_native_qwen_registry_selects_qwen3_dense_text(tmp_path):
    (tmp_path / "config.json").write_text(
        (
            '{"model_type":"qwen3","vocab_size":17,"hidden_size":8,'
            '"num_hidden_layers":1,"num_attention_heads":2}'
        ),
        encoding="utf-8",
    )

    config = load_native_qwen_config(tmp_path)

    assert config.family == "qwen3"
    assert config.key_prefix == "model"
    assert config.model_type == "qwen3"


def test_native_qwen_registry_selects_qwen35_text_only(tmp_path):
    (tmp_path / "config.json").write_text(
        (
            '{"model_type":"qwen3_5","text_config":{'
            '"model_type":"qwen3_5_text","vocab_size":17,"hidden_size":8,'
            '"num_hidden_layers":2,"layer_types":["linear_attention","full_attention"]'
            "}}"
        ),
        encoding="utf-8",
    )

    config = load_native_qwen_config(tmp_path)

    assert config.family == "qwen3_5_text"
    assert config.key_prefix == "model.language_model"
    assert config.model_type == "qwen3_5_text"


def test_qwen35_hybrid_text_model_builds_and_disables_kv_cache():
    torch.manual_seed(1234)
    config = _tiny_qwen35_config()
    model = TensorParallelQwen35TextForCausalLM(
        hf_config=config,
        loader=_TinyQwen35Loader(config),
        tp_rank=0,
        tp_size=1,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_targets={"q_proj", "v_proj"},
        gradient_checkpointing=False,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    logits = model(input_ids, attention_mask=attention_mask)

    assert logits.shape == (2, 3, config.vocab_size)
    assert model.supports_kv_cache is True
    assert isinstance(model, NativeTPCausalLMBase)
    assert any("token_mixer.q_proj.lora_" in name for name, _ in model.named_parameters())
    assert any("token_mixer.v_proj.lora_" in name for name, _ in model.named_parameters())

    cached_logits, cache = model(input_ids, attention_mask=attention_mask, use_cache=True)
    next_mask = torch.ones((2, 4), dtype=torch.bool)
    next_logits, next_cache = model(
        torch.tensor([[7], [8]]), attention_mask=next_mask, past_key_values=cache, use_cache=True
    )

    assert cached_logits.shape == logits.shape
    assert len(cache) == config.num_hidden_layers
    assert next_logits.shape == (2, 1, config.vocab_size)
    assert len(next_cache) == config.num_hidden_layers


def test_qwen35_pipeline_stage_loads_only_local_layers_and_boundary_modules():
    torch.manual_seed(1234)
    config = _tiny_qwen35_config(
        num_hidden_layers=4, layer_types=["linear_attention", "full_attention"] * 2
    )
    first_plan = build_placement_plan(
        strategy="qwen36_pp8_static",
        model_family="qwen3_5_text",
        num_hidden_layers=4,
        tp_size=1,
        pp_size=2,
        tp_rank=0,
        pp_rank=0,
        layer_types=config.layer_types,
    )
    last_plan = build_placement_plan(
        strategy="qwen36_pp8_static",
        model_family="qwen3_5_text",
        num_hidden_layers=4,
        tp_size=1,
        pp_size=2,
        tp_rank=0,
        pp_rank=1,
        layer_types=config.layer_types,
    )

    first = TensorParallelQwen35TextForCausalLM(
        hf_config=config,
        loader=_TinyQwen35Loader(config),
        tp_rank=0,
        tp_size=1,
        placement=first_plan,
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_targets=set(),
        gradient_checkpointing=False,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    last = TensorParallelQwen35TextForCausalLM(
        hf_config=config,
        loader=_TinyQwen35Loader(config),
        tp_rank=0,
        tp_size=1,
        placement=last_plan,
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_targets=set(),
        gradient_checkpointing=False,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert first.local_layer_indices
    assert last.local_layer_indices
    assert (*first.local_layer_indices, *last.local_layer_indices) == (0, 1, 2, 3)
    assert first.local_layer_indices[-1] + 1 == last.local_layer_indices[0]
    assert first.embed_tokens is not None
    assert first.lm_head is None
    assert last.embed_tokens is None
    assert last.lm_head is not None

    hidden = first.forward_stage(
        None,
        torch.tensor([[1, 2, 3]]),
        torch.ones((1, 3), dtype=torch.bool),
        use_cache=False,
    )
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (1, 3, config.hidden_size)


def test_native_qwen_model_classes_expose_layered_contract(tmp_path):
    qwen3_config = NativeQwenConfig(
        {
            "model_type": "qwen3",
            "vocab_size": 17,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
        },
        family="qwen3",
        key_prefix="model",
    )
    qwen3 = TensorParallelQwenForCausalLM(
        hf_config=qwen3_config,
        loader=_TinyQwenLoader(qwen3_config),
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
    qwen35 = TensorParallelQwen35TextForCausalLM(
        hf_config=_tiny_qwen35_config(),
        loader=_TinyQwen35Loader(_tiny_qwen35_config()),
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

    assert isinstance(qwen3, Qwen3DenseModel)
    assert isinstance(qwen3, NativeTPCausalLMBase)
    assert isinstance(qwen35, Qwen35HybridTextModel)
    assert isinstance(qwen35, NativeTPCausalLMBase)


def test_qwen35_hybrid_cache_logits_match_full_forward_prefix():
    torch.manual_seed(91011)
    config = _tiny_qwen35_config()
    model = TensorParallelQwen35TextForCausalLM(
        hf_config=config,
        loader=_TinyQwen35Loader(config),
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
    prefix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    next_token = torch.tensor([[9], [10]])
    full = torch.cat([prefix, next_token], dim=1)
    prefix_mask = torch.ones_like(prefix, dtype=torch.bool)
    full_mask = torch.ones_like(full, dtype=torch.bool)

    with torch.no_grad():
        _, cache = model(prefix, attention_mask=prefix_mask, use_cache=True)
        cached_next, _ = model(
            next_token, attention_mask=full_mask, past_key_values=cache, use_cache=True
        )
        full_logits = model(full, attention_mask=full_mask)

    assert torch.allclose(cached_next[:, -1], full_logits[:, -1], atol=5e-4, rtol=5e-4)


def test_qwen35_linear_attention_matches_transformers_torch_fallback():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers.models, "qwen3_5"):
        pytest.skip("qwen3_5 reference implementation is not available")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    torch.manual_seed(5678)
    reference_config = Qwen3_5TextConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    )
    reference = Qwen3_5GatedDeltaNet(reference_config, layer_idx=0)
    reference.eval()
    config = _tiny_qwen35_config(linear_conv_kernel_dim=3)
    loader = _TinyQwen35LinearLoader(reference.state_dict())
    candidate = TensorParallelQwen35LinearAttention(
        prefix="model.language_model.layers.0.linear_attn",
        hf_config=config,
        loader=loader,
        tp_rank=0,
        tp_size=1,
        lora_r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        lora_targets=set(),
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    candidate.eval()
    hidden_states = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        expected = reference(hidden_states, attention_mask=attention_mask)
        actual = candidate(
            hidden_states, position_ids=torch.arange(5).expand(2, 5), attention_mask=attention_mask
        )

    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)


def test_selected_token_log_probs_match_full_vocab_log_softmax():
    torch.manual_seed(1234)
    hidden = torch.randn(2, 3, 5)
    weight = torch.randn(11, 5)
    output_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    selected = _selected_token_log_probs_from_hidden(hidden, weight, output_ids, vocab_chunk_size=4)
    full = torch.log_softmax(torch.matmul(hidden, weight.t()), dim=-1)
    expected = full.gather(-1, output_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(selected, expected, atol=1e-6, rtol=1e-6)


def test_adapter_training_indices_are_stable_without_mutating_global_random_state():
    config = GraspoConfig.from_dict({"training": {"seed": 123}})
    adapter = QwenNativeTPAdapter(config)
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
                "native_tp": {
                    "generation_micro_batch_size": 1,
                    "use_kv_cache_for_rollout": True,
                }
            }
        }
    )
    adapter = QwenNativeTPAdapter(config)
    adapter.rank = 1
    adapter.device = torch.device("cuda")
    monkeypatch.setattr(
        adapter,
        "_resolve_generation_micro_batch_size",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "graspo.backends.native_tp.qwen_tp_adapter.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "graspo.backends.native_tp.qwen_tp_adapter.dist.is_initialized",
        lambda: True,
    )

    def fake_broadcast_object_list(payload, src):
        assert src == 0
        assert payload == [None]
        payload[0] = 4

    monkeypatch.setattr(
        "graspo.backends.native_tp.qwen_tp_adapter.dist.broadcast_object_list",
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
            self.tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
                config.hidden_size
            )
            self.tensors[f"{prefix}.self_attn.q_proj.weight"] = (
                torch.randn(config.hidden_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.self_attn.k_proj.weight"] = (
                torch.randn(config.hidden_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.self_attn.v_proj.weight"] = (
                torch.randn(config.hidden_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.self_attn.o_proj.weight"] = (
                torch.randn(config.hidden_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(config.head_dim)
            self.tensors[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(config.head_dim)
            self.tensors[f"{prefix}.mlp.gate_proj.weight"] = (
                torch.randn(config.intermediate_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.mlp.up_proj.weight"] = (
                torch.randn(config.intermediate_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.mlp.down_proj.weight"] = (
                torch.randn(config.hidden_size, config.intermediate_size) * 0.02
            )

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def get_optional(self, name: str) -> torch.Tensor | None:
        return self.tensors.get(name)


def _tiny_qwen35_config(**overrides):
    values = {
        "model_type": "qwen3_5_text",
        "vocab_size": 17,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
        "layer_types": ["linear_attention", "full_attention"],
        "hidden_act": "silu",
    }
    values.update(overrides)
    return NativeQwenConfig(values, family="qwen3_5_text", key_prefix="model.language_model")


class _TinyQwen35Loader:
    def __init__(self, config) -> None:
        self.tensors: dict[str, torch.Tensor] = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size, config.hidden_size
            )
            * 0.02,
            "model.language_model.norm.weight": torch.zeros(config.hidden_size),
            "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size) * 0.02,
        }
        for idx, layer_type in enumerate(config.layer_types):
            prefix = f"model.language_model.layers.{idx}"
            self.tensors[f"{prefix}.input_layernorm.weight"] = torch.zeros(config.hidden_size)
            self.tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.zeros(
                config.hidden_size
            )
            self.tensors[f"{prefix}.mlp.gate_proj.weight"] = (
                torch.randn(config.intermediate_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.mlp.up_proj.weight"] = (
                torch.randn(config.intermediate_size, config.hidden_size) * 0.02
            )
            self.tensors[f"{prefix}.mlp.down_proj.weight"] = (
                torch.randn(config.hidden_size, config.intermediate_size) * 0.02
            )
            if layer_type == "full_attention":
                attn = f"{prefix}.self_attn"
                self.tensors[f"{attn}.q_proj.weight"] = (
                    torch.randn(
                        config.num_attention_heads * config.head_dim * 2, config.hidden_size
                    )
                    * 0.02
                )
                self.tensors[f"{attn}.k_proj.weight"] = (
                    torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size)
                    * 0.02
                )
                self.tensors[f"{attn}.v_proj.weight"] = (
                    torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size)
                    * 0.02
                )
                self.tensors[f"{attn}.o_proj.weight"] = (
                    torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim)
                    * 0.02
                )
                self.tensors[f"{attn}.q_norm.weight"] = torch.zeros(config.head_dim)
                self.tensors[f"{attn}.k_norm.weight"] = torch.zeros(config.head_dim)
            else:
                attn = f"{prefix}.linear_attn"
                key_dim = config.linear_num_key_heads * config.linear_key_head_dim
                value_dim = config.linear_num_value_heads * config.linear_value_head_dim
                conv_dim = key_dim * 2 + value_dim
                self.tensors[f"{attn}.in_proj_qkv.weight"] = (
                    torch.randn(conv_dim, config.hidden_size) * 0.02
                )
                self.tensors[f"{attn}.conv1d.weight"] = (
                    torch.randn(conv_dim, 1, config.linear_conv_kernel_dim) * 0.02
                )
                self.tensors[f"{attn}.in_proj_z.weight"] = (
                    torch.randn(value_dim, config.hidden_size) * 0.02
                )
                self.tensors[f"{attn}.in_proj_b.weight"] = (
                    torch.randn(config.linear_num_value_heads, config.hidden_size) * 0.02
                )
                self.tensors[f"{attn}.in_proj_a.weight"] = (
                    torch.randn(config.linear_num_value_heads, config.hidden_size) * 0.02
                )
                self.tensors[f"{attn}.dt_bias"] = torch.randn(config.linear_num_value_heads) * 0.02
                self.tensors[f"{attn}.A_log"] = torch.randn(config.linear_num_value_heads) * 0.02
                self.tensors[f"{attn}.norm.weight"] = torch.ones(config.linear_value_head_dim)
                self.tensors[f"{attn}.out_proj.weight"] = (
                    torch.randn(config.hidden_size, value_dim) * 0.02
                )

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def get_optional(self, name: str) -> torch.Tensor | None:
        return self.tensors.get(name)


class _TinyQwen35LinearLoader:
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.tensors = {
            f"model.language_model.layers.0.linear_attn.{name}": tensor.detach().clone()
            for name, tensor in state_dict.items()
        }

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def get_optional(self, name: str) -> torch.Tensor | None:
        return self.tensors.get(name)
