from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

if TYPE_CHECKING:
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex

from graspo.backends.native_tp.models.qwen.config import NativeQwenConfig
from graspo.backends.native_tp.models.qwen.layers import (
    Qwen35RMSNorm,
    TensorParallelQwen35DecoderLayer,
    _checkpoint_qwen35_decoder_layer_forward,
    _qwen35_cache_sequence_len,
)
from graspo.backends.native_tp.models.qwen.modeling import QwenFamilyBase
from graspo.backends.native_tp.placement import NativePlacementPlan
from graspo.backends.native_tp.tensor_utils import (
    _dtype_size,
    _position_ids,
    _selected_token_log_probs_from_hidden,
)


class Qwen35HybridTextModel(QwenFamilyBase):
    def __init__(
        self,
        *,
        hf_config: NativeQwenConfig,
        loader: "SafetensorIndex",
        tp_rank: int,
        tp_size: int,
        placement: NativePlacementPlan | None = None,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: set[str],
        gradient_checkpointing: bool,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = hf_config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.placement = placement
        self.device_ref = device
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.supports_kv_cache = True
        self.lora_targets = set(lora_targets)
        self.key_prefix = str(getattr(hf_config, "key_prefix", "model.language_model"))
        self.rope_deltas: torch.Tensor | None = None
        layer_types = list(getattr(hf_config, "layer_types", []) or [])
        if len(layer_types) != int(hf_config.num_hidden_layers):
            raise ValueError("qwen3_5_text layer_types length must match num_hidden_layers")

        include_embeddings = placement.include_embeddings if placement is not None else True
        include_lm_head = placement.include_lm_head if placement is not None else True
        local_layer_indices = (
            list(placement.local_layer_indices)
            if placement is not None
            else list(range(hf_config.num_hidden_layers))
        )
        self.embed_tokens = (
            nn.Embedding(
                hf_config.vocab_size, hf_config.hidden_size, device=device, dtype=torch_dtype
            )
            if include_embeddings
            else None
        )
        if self.embed_tokens is not None:
            self.embed_tokens.weight.data.copy_(
                loader.get(f"{self.key_prefix}.embed_tokens.weight").to(
                    device=device, dtype=torch_dtype
                )
            )
        from graspo.backends.native_tp.models.qwen.modeling import _build_qwen35_visual_tower

        self.visual = (
            _build_qwen35_visual_tower(
                hf_config=hf_config,
                loader=loader,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_targets=lora_targets,
                torch_dtype=torch_dtype,
                device=device,
            )
            if include_embeddings and bool(getattr(hf_config, "has_vision_config", False))
            else None
        )
        self.local_layer_indices = tuple(local_layer_indices)
        self.layers = nn.ModuleList(
            [
                TensorParallelQwen35DecoderLayer(
                    layer_idx=idx,
                    layer_type=layer_types[idx],
                    key_prefix=self.key_prefix,
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
                for idx in local_layer_indices
            ]
        )
        self.norm = (
            Qwen35RMSNorm(
                hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
            )
            if include_lm_head
            else None
        )
        if self.norm is not None:
            self.norm.weight.data.copy_(
                loader.get(f"{self.key_prefix}.norm.weight").to(device=device, dtype=torch_dtype)
            )
        self.lm_head = (
            nn.Linear(
                hf_config.hidden_size,
                hf_config.vocab_size,
                bias=False,
                device=device,
                dtype=torch_dtype,
            )
            if include_lm_head
            else None
        )
        if self.lm_head is not None:
            lm_head = loader.get_optional("lm_head.weight")
            if lm_head is None:
                lm_head = loader.get(f"{self.key_prefix}.embed_tokens.weight")
            self.lm_head.weight.data.copy_(lm_head.to(device=device, dtype=torch_dtype))
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[Any, ...] | None = None,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, ...]]:
        hidden_states = self._forward_hidden(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            multimodal_inputs=multimodal_inputs,
            use_cache=use_cache,
        )
        if self.lm_head is None:
            raise RuntimeError("This Qwen3.5 stage does not own lm_head")
        if use_cache:
            hidden_states, present_key_values = hidden_states
            return self.lm_head(hidden_states), present_key_values
        assert isinstance(hidden_states, torch.Tensor)
        return self.lm_head(hidden_states)

    def sequence_log_probs(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("This Qwen3.5 stage does not own lm_head")
        hidden_states = self._forward_hidden(
            sequences,
            attention_mask=attention_mask,
            multimodal_inputs=multimodal_inputs,
        )
        assert isinstance(hidden_states, torch.Tensor)
        return _selected_token_log_probs_from_hidden(
            hidden_states[:, :-1].float(),
            self.lm_head.weight.float(),
            sequences[:, 1:],
        )

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[Any, ...] | None = None,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, ...]]:
        hidden_states = self.embed_inputs(input_ids, multimodal_inputs=multimodal_inputs)
        if attention_mask is None:
            past_len = _qwen35_cache_sequence_len(past_key_values[0]) if past_key_values else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_len + input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        position_ids = self.compute_multimodal_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            multimodal_inputs=multimodal_inputs,
            past_key_values=past_key_values,
            query_len=int(input_ids.shape[1]),
        )
        present_key_values: list[Any] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present)
            elif self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    _checkpoint_qwen35_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        if self.norm is None:
            raise RuntimeError("This Qwen3.5 stage does not own final norm")
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states

    def embed_inputs(
        self,
        input_ids: torch.Tensor,
        *,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("This Qwen3.5 stage does not own embeddings")
        hidden_states = self.embed_tokens(input_ids)
        if multimodal_inputs is None:
            return hidden_states
        if self.visual is None:
            raise RuntimeError(
                "Qwen3.5 multimodal inputs require visual tower on the embedding stage"
            )
        image_features = self._visual_features(multimodal_inputs, kind="image")
        if image_features is not None:
            image_token_id = int(getattr(self.config, "image_token_id"))
            image_mask = input_ids.eq(image_token_id).unsqueeze(-1).expand_as(hidden_states)
            if int(image_mask.sum().item()) != int(image_features.numel()):
                raise RuntimeError(
                    "Image features and image placeholder tokens do not match: "
                    f"tokens={int(image_mask.sum().item())}, features={int(image_features.numel())}"
                )
            hidden_states = hidden_states.masked_scatter(
                image_mask, image_features.to(hidden_states.dtype)
            )
        video_features = self._visual_features(multimodal_inputs, kind="video")
        if video_features is not None:
            video_token_id = int(getattr(self.config, "video_token_id"))
            video_mask = input_ids.eq(video_token_id).unsqueeze(-1).expand_as(hidden_states)
            if int(video_mask.sum().item()) != int(video_features.numel()):
                raise RuntimeError(
                    "Video features and video placeholder tokens do not match: "
                    f"tokens={int(video_mask.sum().item())}, features={int(video_features.numel())}"
                )
            hidden_states = hidden_states.masked_scatter(
                video_mask, video_features.to(hidden_states.dtype)
            )
        return hidden_states

    def compute_multimodal_position_ids(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        multimodal_inputs: dict[str, torch.Tensor] | None,
        past_key_values: tuple[Any, ...] | None,
        query_len: int,
    ) -> torch.Tensor:
        past_len = _qwen35_cache_sequence_len(past_key_values[0]) if past_key_values else 0
        has_multimodal = bool(
            multimodal_inputs is not None
            and (
                multimodal_inputs.get("image_grid_thw") is not None
                or multimodal_inputs.get("video_grid_thw") is not None
            )
        )
        if has_multimodal and input_ids is not None and past_len == 0:
            assert multimodal_inputs is not None
            mm_token_type_ids = self._multimodal_token_type_ids(input_ids, multimodal_inputs)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=multimodal_inputs.get("image_grid_thw"),
                video_grid_thw=multimodal_inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids[:, :, -query_len:]
        if self.rope_deltas is not None and (past_len > 0 or input_ids is None):
            position_ids = _position_ids(attention_mask).view(
                1, attention_mask.shape[0], attention_mask.shape[1]
            )
            deltas = self.rope_deltas.to(device=attention_mask.device)
            if deltas.shape[0] != attention_mask.shape[0]:
                repeat = max(1, attention_mask.shape[0] // max(int(deltas.shape[0]), 1))
                deltas = deltas.repeat_interleave(repeat, dim=0)
            position_ids = position_ids.expand(3, -1, -1) + deltas[: attention_mask.shape[0]].view(
                1, -1, 1
            )
            return position_ids[:, :, -query_len:]
        return _position_ids(attention_mask)[:, -query_len:]

    def _multimodal_token_type_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        provided = multimodal_inputs.get("mm_token_type_ids")
        if isinstance(provided, torch.Tensor) and tuple(provided.shape) == tuple(input_ids.shape):
            return provided.to(device=input_ids.device, dtype=torch.long)
        token_types = torch.zeros_like(input_ids, dtype=torch.long)
        image_token_id = getattr(self.config, "image_token_id", None)
        if image_token_id is not None:
            token_types = token_types.masked_fill(input_ids.eq(int(image_token_id)), 1)
        video_token_id = getattr(self.config, "video_token_id", None)
        if video_token_id is not None:
            token_types = token_types.masked_fill(input_ids.eq(int(video_token_id)), 2)
        return token_types

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        *,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: torch.device,
    ) -> torch.Tensor:
        llm_grid_t = int(grid_thw[0].item()) // int(temp_merge_size)
        llm_grid_h = int(grid_thw[1].item()) // int(spatial_merge_size)
        llm_grid_w = int(grid_thw[2].item()) // int(spatial_merge_size)
        position_temporal = torch.arange(llm_grid_t, device=device) * int(time_interval)
        position_height = torch.arange(llm_grid_h, device=device) + int(start_position)
        position_width = torch.arange(llm_grid_w, device=device) + int(start_position)
        position_width = position_width.repeat(llm_grid_h * llm_grid_t)
        position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
        position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + int(
            start_position
        )
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = int(
            (getattr(self.config, "vision_config", {}) or {}).get("spatial_merge_size", 1)
        )
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters: dict[int, Any] = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        mrope_position_deltas: list[torch.Tensor] = []
        for batch_idx in range(input_ids.shape[0]):
            input_token_type = mm_token_type_ids[batch_idx]
            current_input_ids = input_ids[batch_idx]
            current_attention_mask = (
                attention_mask[batch_idx].bool() if attention_mask is not None else None
            )
            if current_attention_mask is not None:
                current_input_ids = current_input_ids[current_attention_mask]
                input_token_type = input_token_type[current_attention_mask]
            groups: list[tuple[int, int, int]] = []
            token_types = [int(value) for value in input_token_type.tolist()]
            if token_types:
                start_idx = 0
                current_type = token_types[0]
                for idx, token_type in enumerate(token_types[1:], start=1):
                    if token_type != current_type:
                        groups.append((current_type, start_idx, idx))
                        start_idx = idx
                        current_type = token_type
                groups.append((current_type, start_idx, len(token_types)))
            current_pos = 0
            positions: list[torch.Tensor] = []
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    positions.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                    continue
                grid_iter = grid_iters.get(modality_type)
                if grid_iter is None:
                    raise RuntimeError(
                        f"Missing grid_thw for multimodal token type {modality_type}"
                    )
                grid_thw = next(grid_iter)
                vision_position_ids = self.get_vision_position_ids(
                    current_pos,
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    device=input_ids.device,
                )
                positions.append(vision_position_ids)
                current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // max(
                    spatial_merge_size, 1
                )
            llm_positions = (
                torch.cat(positions, dim=1).reshape(3, -1)
                if positions
                else torch.empty(3, 0, dtype=input_ids.dtype, device=input_ids.device)
            )
            if current_attention_mask is not None:
                position_ids[:, batch_idx, current_attention_mask] = llm_positions.to(
                    position_ids.device
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            if llm_positions.numel():
                mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
            else:
                mrope_position_deltas.append(torch.tensor(0, device=input_ids.device))
        return position_ids, torch.stack(mrope_position_deltas).to(input_ids.device).unsqueeze(1)

    def _visual_features(
        self,
        multimodal_inputs: dict[str, torch.Tensor],
        *,
        kind: str,
    ) -> torch.Tensor | None:
        if kind == "image":
            pixel_values = multimodal_inputs.get("pixel_values")
            grid_thw = multimodal_inputs.get("image_grid_thw")
        else:
            pixel_values = multimodal_inputs.get("pixel_values_videos")
            grid_thw = multimodal_inputs.get("video_grid_thw")
        if pixel_values is None:
            return None
        if grid_thw is None:
            raise RuntimeError(f"{kind} pixel values were provided without grid_thw")
        assert self.visual is not None
        dtype = next(self.visual.parameters()).dtype
        output = self.visual(pixel_values.to(dtype=dtype), grid_thw=grid_thw)
        features = output.pooler_output if hasattr(output, "pooler_output") else output[1]
        return features.to(device=pixel_values.device)

    def forward_stage(
        self,
        hidden_states: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        *,
        past_key_values: tuple[Any, ...] | None = None,
        use_cache: bool = False,
        apply_lm_head: bool = False,
        multimodal_inputs: dict[str, torch.Tensor] | None = None,
        position_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[Any, ...]]:
        if hidden_states is None:
            if input_ids is None or self.embed_tokens is None:
                raise RuntimeError("Pipeline stage requires input_ids on the embedding stage")
            hidden_states = self.embed_inputs(input_ids, multimodal_inputs=multimodal_inputs)
            query_len = int(input_ids.shape[1])
        else:
            query_len = int(hidden_states.shape[1])
        position_ids = self.compute_multimodal_position_ids(
            input_ids=position_input_ids if position_input_ids is not None else input_ids,
            attention_mask=attention_mask,
            multimodal_inputs=multimodal_inputs,
            past_key_values=past_key_values,
            query_len=query_len,
        )
        present_key_values: list[Any] = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present)
            elif self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    _checkpoint_qwen35_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        if apply_lm_head:
            if self.norm is None or self.lm_head is None:
                raise RuntimeError("Pipeline final stage requires norm and lm_head")
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states

    def estimate_kv_cache_bytes(self, *, batch_size: int, sequence_len: int) -> int:
        dtype_source = (
            self.embed_tokens.weight if self.embed_tokens is not None else next(self.parameters())
        )
        dtype_size = _dtype_size(dtype_source.dtype)
        total = 0
        layer_types = list(getattr(self.config, "layer_types", []) or [])
        full_head_dim = int(
            getattr(
                self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
            )
        )
        local_layers = set(getattr(self, "local_layer_indices", tuple(range(len(layer_types)))))
        for idx, layer_type in enumerate(layer_types):
            if idx not in local_layers:
                continue
            if layer_type == "full_attention":
                local_kv_heads = max(
                    1, math.ceil(int(self.config.num_key_value_heads) / int(self.tp_size))
                )
                total += (
                    int(batch_size)
                    * 2
                    * local_kv_heads
                    * full_head_dim
                    * int(sequence_len)
                    * dtype_size
                )
            elif layer_type == "linear_attention":
                local_v_heads = int(self.config.linear_num_value_heads) // int(self.tp_size)
                local_k_heads = int(self.config.linear_num_key_heads) // int(self.tp_size)
                key_dim = local_k_heads * int(self.config.linear_key_head_dim)
                value_dim = local_v_heads * int(self.config.linear_value_head_dim)
                conv_dim = 2 * key_dim + value_dim
                recurrent = (
                    int(batch_size)
                    * local_v_heads
                    * int(self.config.linear_key_head_dim)
                    * int(self.config.linear_value_head_dim)
                )
                conv = int(batch_size) * conv_dim * int(self.config.linear_conv_kernel_dim)
                total += (recurrent + conv) * dtype_size
        return int(total)


class TensorParallelQwen35TextForCausalLM(Qwen35HybridTextModel):
    """Compatibility alias for older tests/imports."""
