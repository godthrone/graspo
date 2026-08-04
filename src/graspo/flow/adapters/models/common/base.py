"""Qwen 家族共享模型基类 —— 设施层（flow.adapters.models.common）。

- GraspoFlowCausalLMBase: 所有原生 TP 因果 LM 的公共基类（防呆：未知模型
  应 fail-closed，而非继承本类做未校验的切分）
- QwenFamilyBase: Qwen 家族公共 helper（TP LoRA 状态、KV cache 估算签名）

注意：模型构建函数（load_native_qwen_config / build_native_qwen_model /
build_qwen35_visual_tower）在 model_builders.py，不在本文件。
"""

import math
from typing import Any

import torch
from torch import nn

from graspo.flow.lora.lora_linear import LoRALinear


class GraspoFlowCausalLMBase(nn.Module):
    """Shared base class for all native tensor-parallel causal LMs.

    Unknown models should fail-closed in the registry, not inherit from this
    class and attempt unchecked sharding.
    """

    supports_kv_cache = False
    lora_targets: set[str] = set()
    placement: Any = None

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


class QwenFamilyBase(GraspoFlowCausalLMBase):
    """Common Qwen native-TP helpers shared by Qwen generations."""
