"""Qwen3 模型层实现（从 layers.py 按模型族拆分，宪法 §8.4）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from graspo.backends.graspoflow.tensor_utils import SafetensorIndex

from graspo.backends.graspoflow.lora import LoRALinear
from graspo.backends.graspoflow.lora_helpers import _lora_target_enabled
from graspo.backends.graspoflow.tensor_utils import (
    _all_reduce_tp,
    _apply_rope,
    _causal_attention_mask,
    _rope_cache,
)


class TensorParallelQwenDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        key_prefix: str,
        hf_config: Any,
        loader: SafetensorIndex,
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        prefix = f"{key_prefix}.layers.{layer_idx}"
        self.input_layernorm = QwenRMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.post_attention_layernorm = QwenRMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.input_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.input_layernorm.weight").to(device=device, dtype=torch_dtype)
        )
        self.post_attention_layernorm.weight.data.copy_(
            loader.get(f"{prefix}.post_attention_layernorm.weight").to(
                device=device, dtype=torch_dtype
            )
        )
        self.self_attn = TensorParallelQwenAttention(
            prefix=f"{prefix}.self_attn",
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.mlp = TensorParallelQwenMLP(
            prefix=f"{prefix}.mlp",
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            torch_dtype=torch_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_output = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present = None
        if use_cache:
            attn_output, present = attn_output
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if use_cache:
            assert present is not None
            return hidden_states, present
        return hidden_states


def _checkpoint_decoder_layer_forward(
    layer: TensorParallelQwenDecoderLayer,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    return layer(hidden_states, position_ids, attention_mask)


class TensorParallelQwenAttention(nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        hf_config: Any,
        loader: SafetensorIndex,
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(
            getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        )
        if self.num_heads % tp_size != 0 or self.num_kv_heads % tp_size != 0:
            raise ValueError("Qwen attention heads and kv heads must be divisible by TP size")
        self.local_heads = self.num_heads // tp_size
        self.local_kv_heads = self.num_kv_heads // tp_size
        self.hidden_size = int(hf_config.hidden_size)
        self.rope_theta = float(getattr(hf_config, "rope_theta", 1000000.0))
        self.q_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.q_proj.weight"),
            bias=loader.get_optional(f"{prefix}.q_proj.bias"),
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.q_proj"),
            target_name="language.self_attn.q_proj",
            hf_module_path=f"{prefix}.q_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.k_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.k_proj.weight"),
            bias=loader.get_optional(f"{prefix}.k_proj.bias"),
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.k_proj"),
            target_name="language.self_attn.k_proj",
            hf_module_path=f"{prefix}.k_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.v_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.v_proj.weight"),
            bias=loader.get_optional(f"{prefix}.v_proj.bias"),
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.v_proj"),
            target_name="language.self_attn.v_proj",
            hf_module_path=f"{prefix}.v_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.q_norm = QwenRMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.k_norm = QwenRMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        q_norm_weight = loader.get_optional(f"{prefix}.q_norm.weight")
        k_norm_weight = loader.get_optional(f"{prefix}.k_norm.weight")
        if q_norm_weight is not None:
            self.q_norm.weight.data.copy_(q_norm_weight.to(device=device, dtype=torch_dtype))
        if k_norm_weight is not None:
            self.k_norm.weight.data.copy_(k_norm_weight.to(device=device, dtype=torch_dtype))
        self.o_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.o_proj.weight"),
            bias=loader.get_optional(f"{prefix}.o_proj.bias"),
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.self_attn.o_proj"),
            target_name="language.self_attn.o_proj",
            hf_module_path=f"{prefix}.o_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_len, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, query_len, self.local_heads, self.head_dim)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(batch, query_len, self.local_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value = (
            self.v_proj(hidden_states)
            .view(batch, query_len, self.local_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        past_len = int(past_key_value[0].shape[2]) if past_key_value is not None else 0
        key_len = past_len + query_len
        cos, sin = _rope_cache(
            key_len, self.head_dim, self.rope_theta, hidden_states.device, hidden_states.dtype
        )
        query, key = _apply_rope(query, key, cos, sin, position_ids)
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        present = (key, value)
        if self.local_kv_heads != self.local_heads:
            repeat = self.local_heads // self.local_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attn_mask = _causal_attention_mask(attention_mask, query_len, key_len, hidden_states.device)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch, query_len, self.local_heads * self.head_dim)
        )
        output = self.o_proj(attn)
        output = _all_reduce_tp(output)
        if use_cache:
            return output, present
        return output


class TensorParallelQwenMLP(nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        hf_config: Any,
        loader: SafetensorIndex,
        tp_rank: int,
        tp_size: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.gate_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.gate_proj.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.mlp.gate_proj"),
            target_name="language.mlp.gate_proj",
            hf_module_path=f"{prefix}.gate_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.up_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.up_proj.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.mlp.up_proj"),
            target_name="language.mlp.up_proj",
            hf_module_path=f"{prefix}.up_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.down_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.down_proj.weight"),
            bias=None,
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.mlp.down_proj"),
            target_name="language.mlp.down_proj",
            hf_module_path=f"{prefix}.down_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return _all_reduce_tp(output)


class QwenRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return hidden_states * self.weight
