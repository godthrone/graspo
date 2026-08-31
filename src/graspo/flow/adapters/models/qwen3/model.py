"""Qwen3 模型适配器实现——设施层，模型特定逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

if TYPE_CHECKING:
    from graspo.flow.parallel.placement_plan import NativePlacementPlan
    from graspo.flow.parallel.tensor_utils import SafetensorIndex

from graspo.flow.adapters.models.common.base import QwenFamilyBase
from graspo.flow.adapters.models.common.layers_qwen3 import (
    QwenRMSNorm,
    TensorParallelQwenDecoderLayer,
    _checkpoint_decoder_layer_forward,
)
from graspo.flow.parallel.tensor_utils import (
    _dtype_size,
    _position_ids,
    _selected_token_log_probs_from_hidden,
)


class Qwen3DenseModel(QwenFamilyBase):
    def __init__(
        self,
        *,
        hf_config: Any,
        loader: SafetensorIndex,
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
        self.key_prefix = str(getattr(hf_config, "key_prefix", "model"))
        self.embed_tokens = nn.Embedding(
            hf_config.vocab_size, hf_config.hidden_size, device=device, dtype=torch_dtype
        )
        self.embed_tokens.weight.data.copy_(
            loader.get(f"{self.key_prefix}.embed_tokens.weight").to(
                device=device, dtype=torch_dtype
            )
        )
        self.layers = nn.ModuleList(
            [
                TensorParallelQwenDecoderLayer(
                    layer_idx=idx,
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
                for idx in range(hf_config.num_hidden_layers)
            ]
        )
        self.norm = QwenRMSNorm(
            hf_config.hidden_size, eps=hf_config.rms_norm_eps, device=device, dtype=torch_dtype
        )
        self.norm.weight.data.copy_(
            loader.get(f"{self.key_prefix}.norm.weight").to(device=device, dtype=torch_dtype)
        )
        self.lm_head = nn.Linear(
            hf_config.hidden_size,
            hf_config.vocab_size,
            bias=False,
            device=device,
            dtype=torch_dtype,
        )
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
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            past_len = int(past_key_values[0][0].shape[2]) if past_key_values else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_len + input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        position_ids = _position_ids(attention_mask)[:, -input_ids.shape[1] :]
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
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
                    _checkpoint_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, tuple(present_key_values)
        return logits

    def sequence_log_probs(
        self, sequences: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self._forward_hidden(sequences, attention_mask=attention_mask)
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
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            past_len = int(past_key_values[0][0].shape[2]) if past_key_values else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_len + input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        position_ids = _position_ids(attention_mask)[:, -input_ids.shape[1] :]
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
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
                    _checkpoint_decoder_layer_forward,
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden_states = layer(hidden_states, position_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states

    def estimate_kv_cache_bytes(self, *, batch_size: int, sequence_len: int) -> int:
        dtype_size = _dtype_size(self.embed_tokens.weight.dtype)
        local_kv_heads = int(self.config.num_key_value_heads) // int(self.tp_size)
        head_dim = int(
            getattr(
                self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
            )
        )
        return (
            int(batch_size)
            * int(self.config.num_hidden_layers)
            * 2
            * local_kv_heads
            * head_dim
            * int(sequence_len)
            * dtype_size
        )
