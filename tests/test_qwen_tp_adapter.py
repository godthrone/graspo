from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch

torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8  # type: ignore[attr-defined]

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
    _parse_qwen_tool_completion,
    _processor_chat_messages,
    _qwen35_mrope_embeddings,
    _round_pipeline_stage_timing,
    _selected_token_log_probs_from_hidden,
    _slice_multimodal_inputs,
)
from graspo.backends.native_tp.runtime import NativeTPRuntime  # noqa: E402
from graspo.backends.native_tp.placement import build_placement_plan  # noqa: E402
from graspo.core.buffer import Experience  # noqa: E402
from graspo.core.reward import GraspoReward, RewardConfig  # noqa: E402
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


def test_processor_chat_messages_wrap_text_content_blocks():
    messages = [
        {"role": "system", "content": "s"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "images/a.png"},
                {"type": "text", "text": "q2"},
            ],
        },
    ]

    assert _processor_chat_messages(messages) == [
        {"role": "system", "content": [{"type": "text", "text": "s"}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "images/a.png"},
                {"type": "text", "text": "q2"},
            ],
        },
    ]
    assert messages[0]["content"] == "s"


def test_qwen_format_messages_passes_tools_to_chat_template():
    class Tokenizer:
        chat_template = "template"

        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []

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


def test_qwen_parser_extracts_xml_tool_call_and_think():
    parsed = _parse_qwen_tool_completion(
        "<think>look</think>\n"
        "<tool_call>\n"
        "<function=robot_atomic_control>\n"
        "<parameter=action>\n"
        "向下\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>",
        expect_tool_calls=True,
    )

    assert parsed.think_text == "look"
    assert parsed.tool_calls == [{"name": "robot_atomic_control", "arguments": {"action": "向下"}}]
    assert parsed.parse_errors == []
    assert parsed.extra_text == ""


def test_qwen_parser_coerces_xml_integer_argument_from_tool_schema():
    tools = [_robot_tool_schema({"action": {"type": "string"}, "distance_cm": {"type": "integer"}})]
    parsed = _parse_qwen_tool_completion(
        "<tool_call><function=robot_atomic_control>"
        "<parameter=action>left</parameter>"
        "<parameter=distance_cm>6</parameter>"
        "</function></tool_call>",
        expect_tool_calls=True,
        tools=tools,
    )

    assert parsed.tool_calls == [
        {"name": "robot_atomic_control", "arguments": {"action": "left", "distance_cm": 6}}
    ]
    assert parsed.parse_errors == []


def test_qwen_parser_coerces_xml_number_argument_from_tool_schema():
    tools = [_robot_tool_schema({"distance_cm": {"type": "number"}})]
    parsed = _parse_qwen_tool_completion(
        "<tool_call><function=robot_atomic_control>"
        "<parameter=distance_cm>6.5</parameter>"
        "</function></tool_call>",
        expect_tool_calls=True,
        tools=tools,
    )

    assert parsed.tool_calls == [
        {"name": "robot_atomic_control", "arguments": {"distance_cm": 6.5}}
    ]
    assert parsed.parse_errors == []


def test_qwen_parser_reports_xml_integer_schema_mismatch():
    tools = [_robot_tool_schema({"distance_cm": {"type": "integer"}})]
    parsed = _parse_qwen_tool_completion(
        "<tool_call><function=robot_atomic_control>"
        "<parameter=distance_cm>6.5</parameter>"
        "</function></tool_call>",
        expect_tool_calls=True,
        tools=tools,
    )
    result = GraspoReward(RewardConfig(check_json_markdown=False)).score_parsed(
        parsed,
        [
            {
                "id": "expected",
                "output": {
                    "tool_calls": [
                        {
                            "name": "robot_atomic_control",
                            "arguments": {"distance_cm": 6},
                        }
                    ]
                },
            }
        ],
        is_tool_call=True,
    )

    assert parsed.tool_calls == [
        {"name": "robot_atomic_control", "arguments": {"distance_cm": "6.5"}}
    ]
    assert parsed.parse_errors == ["tool_call[0].arguments.distance_cm expected integer"]
    assert result.all_right is False


def test_qwen_parser_leaves_xml_argument_string_without_schema():
    parsed = _parse_qwen_tool_completion(
        "<tool_call><function=robot_atomic_control>"
        "<parameter=distance_cm>6</parameter>"
        "</function></tool_call>",
        expect_tool_calls=True,
    )

    assert parsed.tool_calls == [
        {"name": "robot_atomic_control", "arguments": {"distance_cm": "6"}}
    ]
    assert parsed.parse_errors == []


def test_qwen_parser_extracts_json_tool_call():
    parsed = _parse_qwen_tool_completion(
        '<tool_call>{"name":"search","arguments":{"q":"apn"}}</tool_call>',
        expect_tool_calls=True,
    )

    assert parsed.tool_calls == [{"name": "search", "arguments": {"q": "apn"}}]
    assert parsed.parser_name == "qwen_json_tool_call"


def test_qwen_parser_preserves_json_tool_call_argument_types_with_schema():
    parsed = _parse_qwen_tool_completion(
        '<tool_call>{"name":"robot_atomic_control","arguments":{"distance_cm":6}}</tool_call>',
        expect_tool_calls=True,
        tools=[_robot_tool_schema({"distance_cm": {"type": "integer"}})],
    )

    assert parsed.tool_calls == [{"name": "robot_atomic_control", "arguments": {"distance_cm": 6}}]
    assert parsed.parse_errors == []


def test_qwen_parser_preserves_multiple_tool_call_order():
    parsed = _parse_qwen_tool_completion(
        '<tool_call>{"name":"first","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"second","arguments":{"y":2}}</tool_call>',
        expect_tool_calls=True,
    )

    assert parsed.tool_calls == [
        {"name": "first", "arguments": {"x": 1}},
        {"name": "second", "arguments": {"y": 2}},
    ]


def test_qwen_parser_reports_bad_tool_call():
    parsed = _parse_qwen_tool_completion("<tool_call><function=bad></function></tool_call>")

    assert parsed.tool_calls == []
    assert parsed.parse_errors


def _robot_tool_schema(properties):
    return {
        "type": "function",
        "function": {
            "name": "robot_atomic_control",
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }


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


def test_qwen35_multimodal_rope_index_matches_transformers_reference():
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    config = _tiny_qwen35_config(
        image_token_id=15,
        video_token_id=16,
        vision_config={"spatial_merge_size": 2},
    )
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
    input_ids = torch.tensor(
        [
            [1, 2, 15, 15, 15, 15, 3, 4],
            [5, 15, 15, 15, 15, 6, 7, 8],
        ]
    )
    mm_token_type_ids = torch.where(input_ids == 15, torch.ones_like(input_ids), 0)
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    actual_position_ids, actual_deltas = model.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )
    reference = Qwen3_5Model.__new__(Qwen3_5Model)
    reference.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    expected_position_ids, expected_deltas = Qwen3_5Model.get_rope_index(
        reference,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )

    assert torch.equal(actual_position_ids, expected_position_ids)
    assert torch.equal(actual_deltas, expected_deltas)


def test_qwen35_mrope_embeddings_match_transformers_reference():
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    config = Qwen3_5TextConfig(
        vocab_size=17,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=12,
        intermediate_size=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 2, 2],
            "mrope_interleaved": True,
        },
        layer_types=["full_attention"],
    )
    reference = Qwen3_5TextRotaryEmbedding(config)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3], [2, 3, 4, 5]],
            [[0, 0, 1, 1], [2, 2, 3, 3]],
            [[0, 1, 0, 1], [3, 4, 3, 4]],
        ]
    )
    hidden_states = torch.zeros((2, 4, 24), dtype=torch.float32)

    expected_cos, expected_sin = reference(hidden_states, position_ids)
    actual_cos, actual_sin = _qwen35_mrope_embeddings(
        position_ids,
        12,
        10000.0,
        (2, 2, 2),
        True,
        torch.device("cpu"),
        torch.float32,
    )

    assert torch.allclose(actual_cos, expected_cos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_sin, expected_sin, atol=1e-5, rtol=1e-5)


def test_qwen35_text_prefill_clears_stale_mrope_delta():
    torch.manual_seed(1234)
    config = _tiny_qwen35_config(
        image_token_id=15,
        vision_config={"spatial_merge_size": 2},
    )
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
    model.rope_deltas = torch.tensor([[7], [7]])
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    position_ids = model.compute_multimodal_position_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        multimodal_inputs=None,
        past_key_values=None,
        query_len=input_ids.shape[1],
    )

    assert model.rope_deltas is None
    assert torch.equal(position_ids, torch.tensor([[0, 1, 2], [0, 1, 2]]))


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
                    "forward_batch_size": 8,
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


# ---------------------------------------------------------------------------
# _slice_multimodal_inputs tests
# ---------------------------------------------------------------------------


class TestSliceMultimodalInputs:
    """Tests for _slice_multimodal_inputs -- the helper that slices multimodal
    tensors (pixel_values, image_grid_thw, etc.) by batch row range under the
    equal-per-row assumption that holds during multimodal rollout."""

    @staticmethod
    def _make_inputs(
        B: int = 4,
        images_per_row: int = 2,
        patches_per_row: int = 6,
        include_video: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Build a plausible multimodal_inputs dict for B identical rows."""
        total_images = B * images_per_row
        total_patches = B * patches_per_row
        inputs: dict[str, torch.Tensor] = {
            "pixel_values": torch.randn(total_patches, 3, 16, 16),
            "image_grid_thw": torch.randint(1, 4, (total_images, 3)),
            "mm_token_type_ids": torch.zeros(B, 128, dtype=torch.long),
        }
        if include_video:
            inputs["pixel_values_videos"] = torch.randn(total_patches, 3, 16, 16)
            inputs["video_grid_thw"] = torch.randint(1, 4, (total_images, 3))
        return inputs

    def test_full_batch_slice_same_as_input(self):
        """Slicing [0, B) should return the full tensors unchanged."""
        B = 4
        inputs = self._make_inputs(B=B, images_per_row=2, patches_per_row=6)
        sliced = _slice_multimodal_inputs(
            inputs,
            0,
            B,
            images_per_row=2,
            patches_per_row=6,
        )
        assert torch.equal(sliced["pixel_values"], inputs["pixel_values"])
        assert torch.equal(sliced["image_grid_thw"], inputs["image_grid_thw"])
        assert torch.equal(sliced["mm_token_type_ids"], inputs["mm_token_type_ids"])

    def test_slice_first_two_rows(self):
        """Slice rows [0, 2) out of B=4."""
        B = 4
        img_per = 2
        patch_per = 6
        inputs = self._make_inputs(B=B, images_per_row=img_per, patches_per_row=patch_per)
        sliced = _slice_multimodal_inputs(
            inputs,
            0,
            2,
            images_per_row=img_per,
            patches_per_row=patch_per,
        )
        assert sliced["pixel_values"].shape == (2 * patch_per, 3, 16, 16)
        assert sliced["image_grid_thw"].shape == (2 * img_per, 3)
        assert sliced["mm_token_type_ids"].shape == (2, 128)
        # Verify pixel_values content: first 12 patches
        assert torch.equal(sliced["pixel_values"], inputs["pixel_values"][:12])
        # Verify image_grid_thw content: first 4 images
        assert torch.equal(sliced["image_grid_thw"], inputs["image_grid_thw"][:4])

    def test_slice_last_row_partial(self):
        """Slice a single row from the end: [3, 4)."""
        B = 4
        img_per = 2
        patch_per = 6
        inputs = self._make_inputs(B=B, images_per_row=img_per, patches_per_row=patch_per)
        sliced = _slice_multimodal_inputs(
            inputs,
            3,
            4,
            images_per_row=img_per,
            patches_per_row=patch_per,
        )
        assert sliced["pixel_values"].shape == (6, 3, 16, 16)
        assert sliced["image_grid_thw"].shape == (2, 3)
        # Last row's pixel_values: patches [18, 24)
        assert torch.equal(sliced["pixel_values"], inputs["pixel_values"][18:24])
        # Last row's image_grid_thw: images [6, 8)
        assert torch.equal(sliced["image_grid_thw"], inputs["image_grid_thw"][6:8])

    def test_no_images_returns_empty_dict(self):
        """When images_per_row=0 and patches_per_row=0, no image keys should appear."""
        inputs = {"mm_token_type_ids": torch.zeros(4, 128, dtype=torch.long)}
        sliced = _slice_multimodal_inputs(
            inputs,
            0,
            2,
            images_per_row=0,
            patches_per_row=0,
        )
        assert "pixel_values" not in sliced
        assert "image_grid_thw" not in sliced
        # mm_token_type_ids is always included if present
        assert sliced["mm_token_type_ids"].shape == (2, 128)

    def test_mm_token_type_ids_always_sliced(self):
        """mm_token_type_ids is sliced by simple row indexing regardless of images."""
        inputs = self._make_inputs(B=4)
        sliced = _slice_multimodal_inputs(
            inputs,
            1,
            3,
            images_per_row=2,
            patches_per_row=6,
        )
        assert torch.equal(sliced["mm_token_type_ids"], inputs["mm_token_type_ids"][1:3])

    def test_video_keys_sliced_when_present(self):
        """Video tensors are sliced when video_per_row > 0."""
        B = 3
        vid_per = 1
        vid_patch_per = 4
        inputs = {
            "pixel_values_videos": torch.randn(B * vid_patch_per, 3, 16, 16),
            "video_grid_thw": torch.randint(1, 4, (B * vid_per, 3)),
        }
        sliced = _slice_multimodal_inputs(
            inputs,
            0,
            2,
            videos_per_row=vid_per,
            video_patches_per_row=vid_patch_per,
        )
        assert sliced["pixel_values_videos"].shape == (2 * vid_patch_per, 3, 16, 16)
        assert sliced["video_grid_thw"].shape == (2 * vid_per, 3)

    def test_missing_keys_not_in_output(self):
        """A key not present in inputs should not appear in output."""
        inputs = {"mm_token_type_ids": torch.zeros(4, 128, dtype=torch.long)}
        sliced = _slice_multimodal_inputs(
            inputs,
            0,
            2,
            images_per_row=2,
            patches_per_row=6,
        )
        # images_per_row > 0 but key missing → not added
        assert "pixel_values" not in sliced
        assert "image_grid_thw" not in sliced
