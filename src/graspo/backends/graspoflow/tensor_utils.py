from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils.rnn import pad_sequence

from graspo.core.buffer import Experience

_TENSOR_PARALLEL_GROUP: dist.ProcessGroup | None = None
_TENSOR_PARALLEL_SIZE: int = 1


def _set_tensor_parallel_group(group: dist.ProcessGroup | None, size: int) -> None:
    """Set the global tensor-parallel group used by _all_reduce_tp."""
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_SIZE
    _TENSOR_PARALLEL_GROUP = group
    _TENSOR_PARALLEL_SIZE = int(size)


class SafetensorIndex:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.weight_map: dict[str, str] = {}
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(payload["weight_map"])
        else:
            files = sorted(model_path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
            for file in files:
                for key in load_file(str(file), device="cpu").keys():
                    self.weight_map[key] = file.name
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, name: str) -> torch.Tensor:
        value = self.get_optional(name)
        if value is None:
            raise KeyError(f"Tensor not found in HF checkpoint: {name}")
        return value

    def get_optional(self, name: str) -> torch.Tensor | None:
        filename = self.weight_map.get(name)
        if filename is None:
            return None
        if filename not in self._cache:
            self._cache[filename] = load_file(str(self.model_path / filename), device="cpu")
        return self._cache[filename][name]


class CollatedExperience:
    def __init__(self, items: Iterable[Experience], device: torch.device) -> None:
        items = list(items)
        self.sequences = pad_sequence([item.sequences for item in items], batch_first=True).to(
            device
        )
        self.old_log_probs = pad_sequence(
            [item.old_log_probs for item in items], batch_first=True
        ).to(device)
        self.advantages = pad_sequence(
            [item.advantages for item in items], batch_first=True, padding_value=0.0
        ).to(device)
        self.attention_mask = (
            pad_sequence([item.attention_mask for item in items], batch_first=True)
            .bool()
            .to(device)
        )
        self.action_mask = (
            pad_sequence([item.action_mask for item in items], batch_first=True).bool().to(device)
        )
        self.metadata = [item.metadata for item in items]


def collate_experiences(items: list[Experience], device: torch.device) -> CollatedExperience:
    return CollatedExperience(items, device)


def _resolve_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _dtype_size(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    return 4


def _shard_tensor(tensor: torch.Tensor, *, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tensor.shape[dim] % tp_size != 0:
        raise ValueError(
            f"Cannot shard tensor shape {tuple(tensor.shape)} on dim={dim} by TP={tp_size}"
        )
    return tensor.chunk(tp_size, dim=dim)[tp_rank].contiguous()


def _shard_bounds(size: int, *, tp_rank: int, tp_size: int) -> tuple[int, int]:
    if int(size) % int(tp_size) != 0:
        raise ValueError(f"Cannot shard size {size} by TP={tp_size}")
    chunk = int(size) // int(tp_size)
    start = int(tp_rank) * chunk
    return start, start + chunk


def _select_head_rows(
    tensor: torch.Tensor,
    *,
    head_indices: Iterable[int],
    head_width: int,
) -> torch.Tensor:
    indices = _head_row_indices(head_indices, head_width)
    return tensor.index_select(
        0, torch.tensor(indices, dtype=torch.long, device=tensor.device)
    ).contiguous()


def _head_row_indices(head_indices: Iterable[int], head_width: int) -> list[int]:
    indices: list[int] = []
    for head in head_indices:
        start = int(head) * int(head_width)
        indices.extend(range(start, start + int(head_width)))
    return indices


class _TensorParallelAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        del ctx
        output = tensor.contiguous()
        if dist.is_available() and dist.is_initialized() and _TENSOR_PARALLEL_SIZE > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=_TENSOR_PARALLEL_GROUP)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (grad_output,)


def _all_reduce_tp(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and _TENSOR_PARALLEL_SIZE > 1:
        return _TensorParallelAllReduce.apply(tensor)
    return tensor


def _selected_token_log_probs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    output_ids: torch.Tensor,
    *,
    vocab_chunk_size: int = 32768,
) -> torch.Tensor:
    selected = lm_head_weight.index_select(0, output_ids.reshape(-1)).view(
        *output_ids.shape,
        hidden_states.shape[-1],
    )
    selected_logits = (hidden_states * selected).sum(dim=-1)
    logsumexp: torch.Tensor | None = None
    for start in range(0, lm_head_weight.shape[0], vocab_chunk_size):
        chunk = lm_head_weight[start : start + vocab_chunk_size]
        logits = F.linear(hidden_states, chunk)
        chunk_lse = torch.logsumexp(logits, dim=-1)
        logsumexp = chunk_lse if logsumexp is None else torch.logaddexp(logsumexp, chunk_lse)
    assert logsumexp is not None
    return selected_logits - logsumexp


def _mean_present(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _rollout_timing_summary(
    tokenize_sec: float, chunk_timings: list[dict[str, float | int]]
) -> dict[str, Any]:
    summary = {
        "tokenize_sec": round(float(tokenize_sec), 6),
        "prefill_sec": round(
            sum(float(item.get("prefill_sec") or 0.0) for item in chunk_timings), 6
        ),
        "decode_sec": round(sum(float(item.get("decode_sec") or 0.0) for item in chunk_timings), 6),
        "sampling_sec": round(
            sum(float(item.get("sampling_sec") or 0.0) for item in chunk_timings), 6
        ),
        "stop_check_sec": round(
            sum(float(item.get("stop_check_sec") or 0.0) for item in chunk_timings), 6
        ),
        "decode_tokens": sum(int(item.get("decode_tokens") or 0) for item in chunk_timings),
    }
    for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS:
        value = sum(float(item.get(key) or 0.0) for item in chunk_timings)
        if value:
            summary[key] = round(value, 6)
    calls = sum(int(item.get("pipeline_forward_calls") or 0) for item in chunk_timings)
    if calls:
        summary["pipeline_forward_calls"] = calls
    return summary


def _scale_rollout_timings(
    chunk_timings: list[dict[str, float | int]],
    divisor: int,
) -> list[dict[str, float | int]]:
    divisor = max(1, int(divisor))
    scaled: list[dict[str, float | int]] = []
    for item in chunk_timings:
        payload: dict[str, float | int] = {
            "prefill_sec": float(item.get("prefill_sec") or 0.0) / divisor,
            "decode_sec": float(item.get("decode_sec") or 0.0) / divisor,
            "sampling_sec": float(item.get("sampling_sec") or 0.0) / divisor,
            "stop_check_sec": float(item.get("stop_check_sec") or 0.0) / divisor,
            "decode_tokens": int(item.get("decode_tokens") or 0),
        }
        for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS:
            if key in item:
                payload[key] = float(item.get(key) or 0.0) / divisor
        if "pipeline_forward_calls" in item:
            payload["pipeline_forward_calls"] = int(item.get("pipeline_forward_calls") or 0)
        scaled.append(payload)
    return scaled


_PIPELINE_STAGE_TIMING_FLOAT_KEYS = (
    "pipeline_recv_sec",
    "pipeline_send_sec",
    "pipeline_stage_compute_sec",
    "pipeline_norm_sec",
    "pipeline_lm_head_sec",
    "pipeline_loss_sec",
    "pipeline_sample_compute_sec",
    "pipeline_token_broadcast_sec",
    "pipeline_backward_autograd_sec",
    "pipeline_grad_recv_sec",
    "pipeline_grad_send_sec",
    "pipeline_grad_clip_sec",
    "pipeline_optimizer_step_sec",
)


def _new_pipeline_stage_timing() -> dict[str, float | int]:
    payload: dict[str, float | int] = {key: 0.0 for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS}
    payload["pipeline_forward_calls"] = 0
    return payload


def _add_pipeline_stage_timing(
    timing: dict[str, float | int] | None,
    key: str,
    started_at: float,
) -> None:
    if timing is None:
        return
    timing[key] = float(timing.get(key) or 0.0) + (time.monotonic() - started_at)


def _round_pipeline_stage_timing(timing: dict[str, float | int]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key in _PIPELINE_STAGE_TIMING_FLOAT_KEYS:
        value = float(timing.get(key) or 0.0)
        if value:
            payload[key] = round(value, 6)
    calls = int(timing.get("pipeline_forward_calls") or 0)
    if calls:
        payload["pipeline_forward_calls"] = calls
    return payload


def _cuda_memory_snapshot(device: torch.device) -> dict[str, float | int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "max_allocated_mib": 0.0,
            "max_reserved_mib": 0.0,
        }
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    max_allocated = int(torch.cuda.max_memory_allocated(device))
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    mib = 1024.0 * 1024.0
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": max_allocated,
        "max_reserved_bytes": max_reserved,
        "allocated_mib": allocated / mib,
        "reserved_mib": reserved / mib,
        "max_allocated_mib": max_allocated / mib,
        "max_reserved_mib": max_reserved / mib,
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert to JSON-serializable types using duck-typing for tensor capabilities."""
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    # 鸭子类型检测 tensor 能力，非接口契约探测
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _left_pad_token_rows(
    rows: Iterable[Iterable[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    values = [list(row) for row in rows]
    if not values:
        return torch.empty((0, 0), dtype=torch.long, device=device), []
    lengths = [len(row) for row in values]
    width = max(max(lengths), 1)
    output = torch.full((len(values), width), int(pad_token_id), dtype=torch.long, device=device)
    for idx, row in enumerate(values):
        if not row:
            continue
        output[idx, width - len(row) :] = torch.tensor(row, dtype=torch.long, device=device)
    return output, lengths


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _qwen35_apply_mrope_layout(
    freqs: torch.Tensor,
    mrope_section: tuple[int, ...],
    mrope_interleaved: bool,
) -> torch.Tensor:
    if len(mrope_section) != 3 or sum(mrope_section) != freqs.shape[-1]:
        return freqs[0].clone()
    if not mrope_interleaved:
        chunks: list[torch.Tensor] = []
        offset = 0
        for dim, section in enumerate(mrope_section):
            chunks.append(freqs[dim, ..., offset : offset + section])
            offset += section
        return torch.cat(chunks, dim=-1)
    output = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        output[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
    return output


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim == 4:
        # mRoPE: cos already position-aware from _qwen35_mrope_embeddings,
        # shape (1, B, query_len, head_dim). Squeeze the mrope dim (always 1)
        # and unsqueeze head dim to match ndim=3 pattern: (B, 1, S, D).
        cos = cos.squeeze(0).unsqueeze(1)
        sin = sin.squeeze(0).unsqueeze(1)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _apply_rope_partial(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim == 4:
        # mRoPE: cos already position-aware from _qwen35_mrope_embeddings,
        # shape (1, B, query_len, head_dim). Squeeze the mrope dim (always 1)
        # and unsqueeze head dim to match ndim=3 pattern: (B, 1, S, D).
        cos = cos.squeeze(0).unsqueeze(1)
        sin = sin.squeeze(0).unsqueeze(1)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rot = (query_rot * cos) + (_rotate_half(query_rot) * sin)
    key_rot = (key_rot * cos) + (_rotate_half(key_rot) * sin)
    return torch.cat([query_rot, query_pass], dim=-1), torch.cat([key_rot, key_pass], dim=-1)


def _causal_attention_mask(
    attention_mask: torch.Tensor | None,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(key_len - query_len, key_len, device=device).unsqueeze(1)
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    causal = key_positions <= query_positions
    if attention_mask is None:
        return causal.view(1, 1, query_len, key_len)
    # During incremental decode the attention mask may be longer than
    # key_len (the caller appends positions for new query tokens).
    # Truncate to the last key_len positions so the mask length aligns
    # with the KV cache.
    key_mask = attention_mask[:, None, None, -key_len:].bool()
    return causal.view(1, 1, query_len, key_len) & key_mask


def _apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        query_mask = attention_mask[:, -hidden_states.shape[1] :]
        return (hidden_states * query_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


def _left_pad_last_dim(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.shape[-1] >= width:
        return value[:, :, -width:].contiguous()
    return F.pad(value, (width - value.shape[-1], 0))


def _l2norm(value: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.vector_norm(value, dim=dim, keepdim=True), min=eps)


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    output = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias=None, padding=0, groups=hidden_size
    )
    if activation == "silu":
        output = F.silu(output)
    return output[:, :, -seq_len:].to(hidden_states.dtype)


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        item.transpose(1, 2).contiguous().to(torch.float32) for item in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        item.reshape(item.shape[0], item.shape[1], -1, chunk_size, item.shape[-1])
        for item in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for idx in range(1, chunk_size):
        row = attn[..., idx, :idx].clone()
        sub = attn[..., :idx, :idx].clone()
        attn[..., idx, :idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    for idx in range(0, total_sequence_length // chunk_size):
        query_i, key_i, value_i = query[:, :, idx], key[:, :, idx], value[:, :, idx]
        attn = query_i @ key_i.transpose(-1, -2) * decay_mask[:, :, idx]
        value_prime = (k_cumdecay[:, :, idx]) @ last_recurrent_state
        value_new = value_i - value_prime
        attn_inter = (query_i * g[:, :, idx, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, idx] = attn_inter + attn @ value_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, idx, -1, None, None].exp()
            + (key_i * (g[:, :, idx, -1, None] - g[:, :, idx]).exp()[..., None]).transpose(-1, -2)
            @ value_new
        )
    if not output_final_state:
        last_recurrent_state = None  # type: ignore[assignment]
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        item.transpose(1, 2).contiguous().to(torch.float32) for item in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    query = query * (1 / (query.shape[-1] ** 0.5))
    core_attn_out = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        value_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    for idx in range(sequence_length):
        query_t = query[:, :, idx]
        key_t = key[:, :, idx]
        value_t = value[:, :, idx]
        g_t = g[:, :, idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, idx].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_memory = (last_recurrent_state * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - kv_memory) * beta_t
        last_recurrent_state = last_recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, idx] = (last_recurrent_state * query_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None  # type: ignore[assignment]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


def _next_token_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature > 0:
        logits = logits / max(float(temperature), 1e-6)
        return _sample_next_token(logits, top_p=top_p)
    return logits.argmax(dim=-1)


def _broadcast_and_pad_finished(
    next_token: torch.Tensor,
    finished: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(next_token, src=0)
    return torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)


def _sample_next_token(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
