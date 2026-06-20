from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

if TYPE_CHECKING:
    from graspo.backends.native_tp.placement import NativePlacementPlan
    from graspo.backends.native_tp.tensor_utils import SafetensorIndex

from graspo.backends.native_tp.models.qwen.config import NativeQwenConfig
from graspo.backends.native_tp.models.qwen.layers import (
    QwenRMSNorm,
    TensorParallelQwenDecoderLayer,
    _checkpoint_decoder_layer_forward,
)
from graspo.backends.native_tp.models.qwen.lora import (
    LoRALinear, _replace_visual_lora_modules,
)
from graspo.backends.native_tp.tensor_utils import (
    _dtype_size,
    _position_ids,
    _selected_token_log_probs_from_hidden,
)

class NativeTPCausalLMBase(nn.Module):
    """Shared contract for repository-native tensor-parallel causal LMs.

    Unknown models should fail closed in the registry instead of inheriting this
    class and silently attempting an unsafe best-effort sharding scheme.
    """

    supports_kv_cache = False

    def sequence_log_probs(
        self, sequences: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def estimate_kv_cache_bytes(self, *, batch_size: int, sequence_len: int) -> int:
        raise NotImplementedError

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu() for name, param in self.named_parameters() if "lora_" in name
        }

    def lora_tensor_metadata(self) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        for module_name, module in self.named_modules():
            if not isinstance(module, LoRALinear) or not module.lora_enabled:
                continue
            metadata.append(module.lora_metadata(module_name))
        return metadata

    def lora_parameter_norm(self) -> float:
        total = 0.0
        for name, param in self.named_parameters():
            if "lora_" in name:
                total += float(param.detach().float().pow(2).sum().cpu())
        return math.sqrt(total)

    def nonzero_lora_grad_count(self) -> int:
        return sum(
            int(param.grad is not None and bool(param.grad.detach().abs().sum().cpu() > 0))
            for name, param in self.named_parameters()
            if "lora_" in name
        )

    def enabled_lora_target_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for _, module in self.named_modules():
            if isinstance(module, LoRALinear) and module.lora_enabled:
                names.add(str(module.lora_target_name))
        return tuple(sorted(names))

    def lora_target_signature(self) -> dict[str, object]:
        return {
            "resolved": list(self.enabled_lora_target_names()),
            "parameter_count": sum(
                param.numel() for name, param in self.named_parameters() if "lora_" in name
            ),
        }


class QwenFamilyBase(NativeTPCausalLMBase):
    """Common Qwen native-TP helpers shared by Qwen generations."""


def load_native_qwen_config(model_path: Path) -> NativeQwenConfig:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "")
    if model_type == "qwen3":
        return NativeQwenConfig(config, family="qwen3", key_prefix="model")
    text_config = dict(config.get("text_config") or {})
    if model_type == "qwen3_5" and text_config.get("model_type") == "qwen3_5_text":
        text_config["has_vision_config"] = "vision_config" in config
        text_config["vision_config"] = dict(config.get("vision_config") or {})
        text_config["image_token_id"] = config.get("image_token_id")
        text_config["video_token_id"] = config.get("video_token_id")
        text_config["root_model_type"] = model_type
        return NativeQwenConfig(
            text_config, family="qwen3_5_text", key_prefix="model.language_model"
        )
    raise ValueError(
        f"native-tp supports text-only qwen3 and qwen3_5_text models; got model_type={model_type!r}"
    )


def build_native_qwen_model(
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
) -> nn.Module:
    if hf_config.family == "qwen3":
        return Qwen3DenseModel(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            placement=placement,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device=device,
        )
    if hf_config.family == "qwen3_5_text":
        from graspo.backends.native_tp.models.qwen.modeling_hybrid import Qwen35HybridTextModel
        return Qwen35HybridTextModel(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            placement=placement,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported native Qwen family: {hf_config.family}")


def _build_qwen35_visual_tower(
    *,
    hf_config: NativeQwenConfig,
    loader: "SafetensorIndex",
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: set[str],
    torch_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qwen3.5-family multimodal training requires transformers Qwen3.5 vision classes"
        ) from exc

    vision_values = dict(getattr(hf_config, "vision_config", {}) or {})
    if not vision_values:
        raise RuntimeError("Qwen3.5-family config has no vision_config")
    vision_config = Qwen3_5VisionConfig(**vision_values)
    if hasattr(vision_config, "_attn_implementation"):
        setattr(vision_config, "_attn_implementation", "sdpa")
    visual = Qwen3_5VisionModel(vision_config).to(device=device, dtype=torch_dtype)  # type: ignore[call-arg]
    state: dict[str, torch.Tensor] = {}
    prefix = "model.visual."
    for key in loader.weight_map:
        if key.startswith(prefix):
            state[key[len(prefix) :]] = loader.get(key).to(device=device, dtype=torch_dtype)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load Qwen3.5 visual tower weights: "
            f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    for param in visual.parameters():
        param.requires_grad = False
    _replace_visual_lora_modules(
        visual,
        lora_targets=lora_targets,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
        torch_dtype=torch_dtype,
    )
    return visual


class Qwen3DenseModel(QwenFamilyBase):
    def __init__(
        self,
        *,
        hf_config: Any,
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


class TensorParallelQwenForCausalLM(Qwen3DenseModel):
    """Compatibility alias for older tests/imports."""


