from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

if TYPE_CHECKING:
    from graspo.backends.graspoflow.tensor_utils import SafetensorIndex

from graspo.backends.graspoflow.lora import LoRALinear
from graspo.backends.graspoflow.lora_helpers import _lora_target_enabled
from graspo.backends.graspoflow.models.common.layers_qwen3 import TensorParallelQwenMLP
from graspo.backends.graspoflow.tensor_utils import (
    _all_reduce_tp,
    _apply_mask_to_padding_states,
    _apply_rope_partial,
    _causal_attention_mask,
    _head_row_indices,
    _left_pad_last_dim,
    _rope_cache,
    _select_head_rows,
    _shard_tensor,
    _torch_causal_conv1d_update,
    _torch_chunk_gated_delta_rule,
    _torch_recurrent_gated_delta_rule,
)


class TensorParallelQwen35DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        layer_type: str,
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
        self.layer_type = layer_type
        self.input_layernorm = Qwen35RMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
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
        if layer_type == "linear_attention":
            self.token_mixer = TensorParallelQwen35LinearAttention(
                prefix=f"{prefix}.linear_attn",
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
        elif layer_type == "full_attention":
            self.token_mixer = TensorParallelQwen35FullAttention(  # type: ignore[assignment]
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
        else:
            raise ValueError(f"Unsupported qwen3_5_text layer type: {layer_type}")
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
        past_key_value: Any | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        mixer_output = self.token_mixer(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present = None
        if use_cache:
            mixer_output, present = mixer_output
        hidden_states = hidden_states + mixer_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if use_cache:
            return hidden_states, present
        return hidden_states


def _checkpoint_qwen35_decoder_layer_forward(
    layer: TensorParallelQwen35DecoderLayer,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    return layer(hidden_states, position_ids, attention_mask)


class TensorParallelQwen35FullAttention(nn.Module):
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
        if self.num_heads % tp_size != 0:
            raise ValueError("Qwen3.5 full-attention query heads must be divisible by TP size")
        self.local_heads = self.num_heads // tp_size
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = float(
            (getattr(hf_config, "rope_parameters", {}) or {}).get("rope_theta", 1000000.0)
        )
        rope_parameters = getattr(hf_config, "rope_parameters", {}) or {}
        partial = float(rope_parameters.get("partial_rotary_factor", 1.0))
        self.rotary_dim = int(self.head_dim * partial)
        self.mrope_section = tuple(int(value) for value in rope_parameters.get("mrope_section", ()))
        self.mrope_interleaved = bool(
            rope_parameters.get("mrope_interleaved", bool(self.mrope_section))
        )
        self.local_q_head_start = tp_rank * self.local_heads
        self.local_q_head_stop = self.local_q_head_start + self.local_heads
        self.local_kv_indices = sorted(
            {
                head // self.num_key_value_groups
                for head in range(self.local_q_head_start, self.local_q_head_stop)
            }
        )

        local_q_heads = range(self.local_q_head_start, self.local_q_head_stop)
        q_bias = loader.get_optional(f"{prefix}.q_proj.bias")
        if q_bias is not None:
            q_bias = _select_head_rows(
                q_bias, head_indices=local_q_heads, head_width=self.head_dim * 2
            )
        self.q_proj = LoRALinear(
            _select_head_rows(
                loader.get(f"{prefix}.q_proj.weight"),
                head_indices=local_q_heads,
                head_width=self.head_dim * 2,
            ),
            q_bias,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.q_proj"),
            target_name="language.full_attn.q_proj",
            hf_module_path=f"{prefix}.q_proj",
            shard_kind="rows",
            row_indices=_head_row_indices(local_q_heads, self.head_dim * 2),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        k_bias = loader.get_optional(f"{prefix}.k_proj.bias")
        if k_bias is not None:
            k_bias = _select_head_rows(
                k_bias, head_indices=self.local_kv_indices, head_width=self.head_dim
            )
        self.k_proj = LoRALinear(
            _select_head_rows(
                loader.get(f"{prefix}.k_proj.weight"),
                head_indices=self.local_kv_indices,
                head_width=self.head_dim,
            ),
            k_bias,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.k_proj"),
            target_name="language.full_attn.k_proj",
            hf_module_path=f"{prefix}.k_proj",
            shard_kind="rows",
            row_indices=_head_row_indices(self.local_kv_indices, self.head_dim),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        v_bias = loader.get_optional(f"{prefix}.v_proj.bias")
        if v_bias is not None:
            v_bias = _select_head_rows(
                v_bias, head_indices=self.local_kv_indices, head_width=self.head_dim
            )
        self.v_proj = LoRALinear(
            _select_head_rows(
                loader.get(f"{prefix}.v_proj.weight"),
                head_indices=self.local_kv_indices,
                head_width=self.head_dim,
            ),
            v_bias,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.v_proj"),
            target_name="language.full_attn.v_proj",
            hf_module_path=f"{prefix}.v_proj",
            shard_kind="rows",
            row_indices=_head_row_indices(self.local_kv_indices, self.head_dim),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.q_norm = Qwen35RMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.k_norm = Qwen35RMSNorm(
            self.head_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.q_norm.weight.data.copy_(
            loader.get(f"{prefix}.q_norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.k_norm.weight.data.copy_(
            loader.get(f"{prefix}.k_norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.o_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.o_proj.weight"),
            bias=loader.get_optional(f"{prefix}.o_proj.bias"),
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.full_attn.o_proj"),
            target_name="language.full_attn.o_proj",
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
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_len, _ = hidden_states.shape
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch, query_len, self.local_heads, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch, query_len, self.local_heads * self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, query_len, len(self.local_kv_indices), self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, query_len, len(self.local_kv_indices), self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        past_len = int(past_key_value[0].shape[2]) if past_key_value is not None else 0
        key_len = past_len + query_len
        if position_ids.ndim == 3:
            cos, sin = _qwen35_mrope_embeddings(
                position_ids,
                self.rotary_dim,
                self.rope_theta,
                self.mrope_section,
                self.mrope_interleaved,
                hidden_states.device,
                hidden_states.dtype,
            )
        else:
            cos, sin = _rope_cache(
                key_len,
                self.rotary_dim,
                self.rope_theta,
                hidden_states.device,
                hidden_states.dtype,
            )
        query, key = _apply_rope_partial(query, key, cos, sin, position_ids)
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        present = (key, value)
        local_kv_for_heads = [
            self.local_kv_indices.index(head // self.num_key_value_groups)
            for head in range(self.local_q_head_start, self.local_q_head_stop)
        ]
        key = key[:, local_kv_for_heads]
        value = value[:, local_kv_for_heads]
        attn_mask = _causal_attention_mask(attention_mask, query_len, key_len, hidden_states.device)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch, query_len, self.local_heads * self.head_dim)
        )
        output = self.o_proj(attn * torch.sigmoid(gate))
        output = _all_reduce_tp(output)
        if use_cache:
            return output, present
        return output


class TensorParallelQwen35LinearAttention(nn.Module):
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
        self.num_v_heads = int(hf_config.linear_num_value_heads)
        self.num_k_heads = int(hf_config.linear_num_key_heads)
        self.head_k_dim = int(hf_config.linear_key_head_dim)
        self.head_v_dim = int(hf_config.linear_value_head_dim)
        if self.num_k_heads % tp_size != 0 or self.num_v_heads % tp_size != 0:
            raise ValueError(
                "Qwen3.5 linear-attention key/value heads must be divisible by TP size"
            )
        self.local_k_heads = self.num_k_heads // tp_size
        self.local_v_heads = self.num_v_heads // tp_size
        self.local_key_dim = self.local_k_heads * self.head_k_dim
        self.local_value_dim = self.local_v_heads * self.head_v_dim
        self.conv_kernel_size = int(hf_config.linear_conv_kernel_dim)
        self.local_k_start = tp_rank * self.local_key_dim
        self.local_k_stop = self.local_k_start + self.local_key_dim
        self.local_v_start = tp_rank * self.local_value_dim
        self.local_v_stop = self.local_v_start + self.local_value_dim
        key_dim = self.num_k_heads * self.head_k_dim

        qkv_weight = loader.get(f"{prefix}.in_proj_qkv.weight")
        self.q_proj = LoRALinear(
            qkv_weight[self.local_k_start : self.local_k_stop],
            None,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.q_proj"),
            target_name="language.linear_attn.q_proj",
            hf_module_path=f"{prefix}.in_proj_qkv",
            base_weight_name=f"{prefix}.in_proj_qkv.weight",
            shard_kind="rows",
            row_start=self.local_k_start,
            row_stop=self.local_k_stop,
            peft_exportable=False,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.k_proj = LoRALinear(
            qkv_weight[key_dim + self.local_k_start : key_dim + self.local_k_stop],
            None,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.k_proj"),
            target_name="language.linear_attn.k_proj",
            hf_module_path=f"{prefix}.in_proj_qkv",
            base_weight_name=f"{prefix}.in_proj_qkv.weight",
            shard_kind="rows",
            row_start=key_dim + self.local_k_start,
            row_stop=key_dim + self.local_k_stop,
            peft_exportable=False,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.v_proj = LoRALinear(
            qkv_weight[2 * key_dim + self.local_v_start : 2 * key_dim + self.local_v_stop],
            None,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.v_proj"),
            target_name="language.linear_attn.v_proj",
            hf_module_path=f"{prefix}.in_proj_qkv",
            base_weight_name=f"{prefix}.in_proj_qkv.weight",
            shard_kind="rows",
            row_start=2 * key_dim + self.local_v_start,
            row_stop=2 * key_dim + self.local_v_stop,
            peft_exportable=False,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        conv_indices = torch.tensor(
            [
                *range(self.local_k_start, self.local_k_stop),
                *range(key_dim + self.local_k_start, key_dim + self.local_k_stop),
                *range(2 * key_dim + self.local_v_start, 2 * key_dim + self.local_v_stop),
            ],
            dtype=torch.long,
        )
        self.conv1d_weight = nn.Parameter(
            loader.get(f"{prefix}.conv1d.weight")
            .index_select(0, conv_indices)
            .to(device=device, dtype=torch_dtype),
            requires_grad=False,
        )
        self.in_proj_z = LoRALinear.from_hf(
            loader.get(f"{prefix}.in_proj_z.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.in_proj_z"),
            target_name="language.linear_attn.in_proj_z",
            hf_module_path=f"{prefix}.in_proj_z",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        self.in_proj_b = LoRALinear.from_hf(
            loader.get(f"{prefix}.in_proj_b.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=False,
            target_name="language.linear_attn.in_proj_b",
            hf_module_path=f"{prefix}.in_proj_b",
            r=0,
            alpha=1,
            dropout=0.0,
            device=device,
            dtype=torch_dtype,
        )
        self.in_proj_a = LoRALinear.from_hf(
            loader.get(f"{prefix}.in_proj_a.weight"),
            bias=None,
            shard="out",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=False,
            target_name="language.linear_attn.in_proj_a",
            hf_module_path=f"{prefix}.in_proj_a",
            r=0,
            alpha=1,
            dropout=0.0,
            device=device,
            dtype=torch_dtype,
        )
        self.dt_bias = nn.Parameter(
            _shard_tensor(
                loader.get(f"{prefix}.dt_bias"), dim=0, tp_rank=tp_rank, tp_size=tp_size
            ).to(device=device, dtype=torch_dtype),
            requires_grad=False,
        )
        self.A_log = nn.Parameter(
            _shard_tensor(
                loader.get(f"{prefix}.A_log"), dim=0, tp_rank=tp_rank, tp_size=tp_size
            ).to(device=device, dtype=torch_dtype),
            requires_grad=False,
        )
        self.norm = Qwen35RMSNormGated(
            self.head_v_dim, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.norm.weight.data.copy_(
            loader.get(f"{prefix}.norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.out_proj = LoRALinear.from_hf(
            loader.get(f"{prefix}.out_proj.weight"),
            bias=None,
            shard="in",
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_enabled=_lora_target_enabled(lora_targets, "language.linear_attn.out_proj"),
            target_name="language.linear_attn.out_proj",
            hf_module_path=f"{prefix}.out_proj",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        del position_ids
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        mixed_qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
        conv_state = past_key_value[0] if past_key_value is not None else None
        recurrent_state = past_key_value[1] if past_key_value is not None else None
        if conv_state is not None and seq_len == 1:
            mixed_qkv = _torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d_weight.squeeze(1),
                activation="silu",
            )
            next_conv_state = conv_state
        else:
            if conv_state is not None:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if use_cache:
                next_conv_state = _left_pad_last_dim(mixed_qkv, self.conv_kernel_size)
            else:
                next_conv_state = None
            mixed_qkv = F.silu(
                F.conv1d(
                    mixed_qkv,
                    self.conv1d_weight,
                    bias=None,
                    padding=self.conv_kernel_size - 1,
                    groups=mixed_qkv.shape[1],
                )[:, :, : mixed_qkv.shape[-1]]
            )
            if conv_state is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.local_key_dim, self.local_key_dim, self.local_value_dim],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, self.local_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.local_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.local_v_heads, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(
            batch, seq_len, self.local_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        if self.local_v_heads // self.local_k_heads > 1:
            repeat = self.local_v_heads // self.local_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        if recurrent_state is not None and seq_len == 1:
            core_attn_out, next_recurrent_state = _torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, next_recurrent_state = _torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch, seq_len, self.local_value_dim)
        output = self.out_proj(core_attn_out)
        output = _all_reduce_tp(output)
        if use_cache:
            assert next_conv_state is not None
            assert next_recurrent_state is not None
            return output, (next_conv_state, next_recurrent_state)
        return output


class Qwen35RMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        output = output * (1.0 + self.weight.float())
        return output.to(hidden_states.dtype)


class Qwen35RMSNormGated(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


def _qwen35_mrope_embeddings(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    mrope_section: tuple[int, ...],
    mrope_interleaved: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].to(device=device, dtype=torch.float32)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
    freqs = _qwen35_apply_mrope_layout(freqs, mrope_section, mrope_interleaved)
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


def _qwen35_cache_sequence_len(layer_cache: Any) -> int:
    if layer_cache is None:
        return 0
    if (
        len(layer_cache) >= 2
        # 鸭子类型检测 tensor shape（GPU/CPU tensor 均支持）
        and hasattr(layer_cache[0], "shape")
        and len(layer_cache[0].shape) == 4
    ):
        return int(layer_cache[0].shape[2])
    return 0
