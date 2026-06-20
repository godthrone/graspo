from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from graspo.backends.native_tp.models.qwen.config import NativeQwenConfig
from graspo.backends.native_tp.tensor_utils import _shard_bounds, _shard_tensor

def native_qwen_lora_available_targets(hf_config: NativeQwenConfig) -> tuple[str, ...]:
    language_mlp = (
        "language.mlp.gate_proj",
        "language.mlp.up_proj",
        "language.mlp.down_proj",
    )
    if hf_config.family == "qwen3":
        return (
            "language.self_attn.q_proj",
            "language.self_attn.k_proj",
            "language.self_attn.v_proj",
            "language.self_attn.o_proj",
            *language_mlp,
        )
    if hf_config.family == "qwen3_5_text":
        targets: tuple[str, ...] = (
            "language.full_attn.q_proj",
            "language.full_attn.k_proj",
            "language.full_attn.v_proj",
            "language.full_attn.o_proj",
            "language.linear_attn.q_proj",
            "language.linear_attn.k_proj",
            "language.linear_attn.v_proj",
            "language.linear_attn.in_proj_z",
            "language.linear_attn.out_proj",
            *language_mlp,
        )
        if bool(getattr(hf_config, "has_vision_config", False)):
            depth = int((getattr(hf_config, "vision_config", {}) or {}).get("depth") or 0)
            visual_block_targets = tuple(
                target
                for idx in range(depth)
                for target in (
                    f"visual.blocks.{idx}.attn.qkv",
                    f"visual.blocks.{idx}.attn.proj",
                    f"visual.blocks.{idx}.mlp.linear_fc1",
                    f"visual.blocks.{idx}.mlp.linear_fc2",
                )
            )
            targets = (
                *targets,
                "visual.merger.linear_fc1",
                "visual.merger.linear_fc2",
                *visual_block_targets,
            )
        return targets
    return ()


def _lora_target_enabled(lora_targets: set[str], canonical_name: str) -> bool:
    return canonical_name in lora_targets or canonical_name.rsplit(".", 1)[-1] in lora_targets


def _replace_visual_lora_modules(
    visual: nn.Module,
    *,
    lora_targets: set[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    target_to_path = {
        "visual.merger.linear_fc1": "merger.linear_fc1",
        "visual.merger.linear_fc2": "merger.linear_fc2",
    }
    depth = len(getattr(visual, "blocks", []))
    for idx in range(depth):
        target_to_path.update(
            {
                f"visual.blocks.{idx}.attn.qkv": f"blocks.{idx}.attn.qkv",
                f"visual.blocks.{idx}.attn.proj": f"blocks.{idx}.attn.proj",
                f"visual.blocks.{idx}.mlp.linear_fc1": f"blocks.{idx}.mlp.linear_fc1",
                f"visual.blocks.{idx}.mlp.linear_fc2": f"blocks.{idx}.mlp.linear_fc2",
            }
        )
    for target_name, module_path in target_to_path.items():
        if not _lora_target_enabled(lora_targets, target_name):
            continue
        parent, attr = _module_parent_and_attr(visual, module_path)
        linear = getattr(parent, attr)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"visual LoRA target {target_name} is not an nn.Linear")
        replacement = LoRALinear(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            lora_enabled=True,
            target_name=target_name,
            hf_module_path=f"model.visual.{module_path}",
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            device=device,
            dtype=torch_dtype,
        )
        setattr(parent, attr, replacement)


def _module_parent_and_attr(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent: nn.Module = module
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if part.isdigit() and isinstance(parent, nn.ModuleList)
            else getattr(parent, part)
        )
    return parent, parts[-1]


class LoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        lora_enabled: bool,
        r: int,
        alpha: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        target_name: str | None = None,
        hf_module_path: str | None = None,
        base_weight_name: str | None = None,
        shard_kind: str = "none",
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
        row_indices: Iterable[int] | None = None,
        peft_exportable: bool = True,
    ) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        self.bias = (
            nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
            if bias is not None
            else None
        )
        self.lora_target_name = str(target_name or "unknown")
        self.hf_module_path = str(hf_module_path) if hf_module_path else None
        self.base_weight_name = (
            str(base_weight_name)
            if base_weight_name
            else (f"{self.hf_module_path}.weight" if self.hf_module_path else None)
        )
        self.lora_shard_kind = str(shard_kind or "none")
        self.lora_row_start = row_start
        self.lora_row_stop = row_stop
        self.lora_col_start = col_start
        self.lora_col_stop = col_stop
        self.lora_row_indices = tuple(int(idx) for idx in row_indices) if row_indices else None
        self.peft_exportable = bool(peft_exportable)
        self.lora_enabled = bool(lora_enabled and r > 0)
        self.lora_r = int(r)
        self.lora_alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)
        if self.lora_enabled:
            self.lora_a = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @classmethod
    def from_hf(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        shard: str,
        tp_rank: int,
        tp_size: int,
        lora_enabled: bool,
        r: int,
        alpha: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        target_name: str | None = None,
        hf_module_path: str | None = None,
        base_weight_name: str | None = None,
        peft_exportable: bool = True,
    ) -> "LoRALinear":
        dim = 0 if shard == "out" else 1
        row_start = row_stop = col_start = col_stop = None
        if shard == "out":
            row_start, row_stop = _shard_bounds(weight.shape[0], tp_rank=tp_rank, tp_size=tp_size)
        elif shard == "in":
            col_start, col_stop = _shard_bounds(weight.shape[1], tp_rank=tp_rank, tp_size=tp_size)
        weight = _shard_tensor(weight, dim=dim, tp_rank=tp_rank, tp_size=tp_size)
        if bias is not None and shard == "out":
            bias = _shard_tensor(bias, dim=0, tp_rank=tp_rank, tp_size=tp_size)
        elif shard == "in":
            bias = None
        return cls(
            weight,
            bias,
            lora_enabled=lora_enabled,
            target_name=target_name,
            hf_module_path=hf_module_path,
            base_weight_name=base_weight_name,
            shard_kind=shard,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            col_stop=col_stop,
            peft_exportable=peft_exportable,
            r=r,
            alpha=alpha,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

    def lora_metadata(self, module_name: str) -> dict[str, Any]:
        if self.hf_module_path is None or self.base_weight_name is None:
            raise RuntimeError(f"Enabled LoRA module {module_name} is missing HF module metadata")
        return {
            "module_name": module_name,
            "lora_a_name": f"{module_name}.lora_a",
            "lora_b_name": f"{module_name}.lora_b",
            "target_name": self.lora_target_name,
            "hf_module_path": self.hf_module_path,
            "base_weight_name": self.base_weight_name,
            "shard_kind": self.lora_shard_kind,
            "row_start": self.lora_row_start,
            "row_stop": self.lora_row_stop,
            "col_start": self.lora_col_start,
            "col_stop": self.lora_col_stop,
            "row_indices": list(self.lora_row_indices)
            if self.lora_row_indices is not None
            else None,
            "peft_exportable": self.peft_exportable,
            "r": self.lora_r,
            "alpha": self.lora_alpha,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        if self.lora_enabled:
            lora = F.linear(F.linear(self.dropout(hidden_states), self.lora_a), self.lora_b)
            output = output + lora * self.scaling
        return output


