from __future__ import annotations

import warnings

warnings.warn(
    "graspo.backends.native_tp.qwen_tp_adapter is deprecated. "
    "Import from graspo.backends.native_tp.models.qwen instead for adapter classes "
    "and from graspo.backends.native_tp.tool_parser / .multimodal / .tensor_utils "
    "for utility functions.",
    DeprecationWarning,
    stacklevel=2,
)

# ---------------------------------------------------------------------------
# Re-export everything from the new locations under both old and new names
# so existing code and tests continue to work.
# ---------------------------------------------------------------------------

# -- Adapter --
from graspo.backends.native_tp.models.qwen.adapter import (  # noqa: E402, F401
    QwenNativeTPAdapter,
    _patch_transformers_float8_import_compat,
    _set_tensor_parallel_group,
    _TENSOR_PARALLEL_GROUP,
    _TENSOR_PARALLEL_SIZE,
)

# -- Config --
from graspo.backends.native_tp.models.qwen.config import NativeQwenConfig  # noqa: E402, F401

# -- Tool parser (public names) --
from graspo.backends.native_tp.tool_parser import (  # noqa: E402, F401
    _THINK_RE as THINK_RE,
    _TOOL_CALL_RE as TOOL_CALL_RE,
    _FUNCTION_RE as FUNCTION_RE,
    _PARAMETER_RE as PARAMETER_RE,
    parse_qwen_tool_completion,
    try_parse_json_tool_call,
    try_parse_qwen_xml_tool_call,
    coerce_xml_tool_argument,
    tool_argument_schema_type,
    canonical_tool_call,
)

# -- Tool parser (legacy _-prefixed aliases for test compat) --
_parse_qwen_tool_completion = parse_qwen_tool_completion
_try_parse_json_tool_call = try_parse_json_tool_call
_try_parse_qwen_xml_tool_call = try_parse_qwen_xml_tool_call
_coerce_xml_tool_argument = coerce_xml_tool_argument
_tool_argument_schema_type = tool_argument_schema_type
_canonical_tool_call = canonical_tool_call
_THINK_RE = THINK_RE
_TOOL_CALL_RE = TOOL_CALL_RE
_FUNCTION_RE = FUNCTION_RE
_PARAMETER_RE = PARAMETER_RE

# -- Multimodal --
from graspo.backends.native_tp.multimodal import (  # noqa: E402, F401
    _multimodal_row_from_sample,
    _messages_from_multimodal_row,
    _processor_chat_messages,
    _tools_from_multimodal_row,
    _normalize_tool_batches,
    _tools_for_chat_template,
    _multimodal_rows_from_metadata,
    _media_counts,
    _slice_multimodal_inputs,
    _slice_multimodal_inputs_offset,
    _compute_multimodal_offset_tables,
)

# -- Tensor utils --
from graspo.backends.native_tp.tensor_utils import (  # noqa: E402, F401
    SafetensorIndex,
    CollatedExperience,
    collate_experiences,
    _resolve_dtype,
    _dtype_size,
    _shard_tensor,
    _shard_bounds,
    _select_head_rows,
    _head_row_indices,
    _TensorParallelAllReduce,
    _all_reduce_tp,
    _selected_token_log_probs_from_hidden,
    _mean_present,
    _rollout_timing_summary,
    _scale_rollout_timings,
    _new_pipeline_stage_timing,
    _add_pipeline_stage_timing,
    _round_pipeline_stage_timing,
    _cuda_memory_snapshot,
    _jsonable,
    _left_pad_token_rows,
    _position_ids,
    _rope_cache,
    _rotate_half,
    _apply_rope,
    _apply_rope_partial,
    _causal_attention_mask,
    _apply_mask_to_padding_states,
    _left_pad_last_dim,
    _l2norm,
    _torch_causal_conv1d_update,
    _torch_chunk_gated_delta_rule,
    _torch_recurrent_gated_delta_rule,
    _next_token_from_logits,
    _broadcast_and_pad_finished,
    _sample_next_token,
)

# -- LoRA / model classes (from qwen models) --
from graspo.backends.native_tp.models.qwen.lora import (  # noqa: E402, F401
    LoRALinear,
    native_qwen_lora_available_targets,
    _lora_target_enabled,
    _replace_visual_lora_modules,
    _module_parent_and_attr,
)
from graspo.backends.native_tp.models.qwen.modeling import (  # noqa: E402, F401
    NativeTPCausalLMBase,
    QwenFamilyBase,
    Qwen3DenseModel,
    TensorParallelQwenForCausalLM,
    load_native_qwen_config,
    build_native_qwen_model,
    _build_qwen35_visual_tower,
)
from graspo.backends.native_tp.models.qwen.modeling_hybrid import (  # noqa: E402, F401
    Qwen35HybridTextModel,
    TensorParallelQwen35TextForCausalLM,
)
from graspo.backends.native_tp.models.qwen.layers import (  # noqa: E402, F401
    TensorParallelQwenDecoderLayer,
    TensorParallelQwen35DecoderLayer,
    TensorParallelQwen35FullAttention,
    TensorParallelQwen35LinearAttention,
    TensorParallelQwenAttention,
    TensorParallelQwenMLP,
    QwenRMSNorm,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
    _checkpoint_decoder_layer_forward,
    _checkpoint_qwen35_decoder_layer_forward,
    _qwen35_mrope_embeddings,
    _qwen35_apply_mrope_layout,
    _qwen35_cache_sequence_len,
)
