"""Qwen 家族共享模型基类 —— 设施层（flow.adapters.models.common）。

- GraspoFlowCausalLMBase: 所有原生 TP 因果 LM 的公共基类（防呆：未知模型
  应 fail-closed，而非继承本类做未校验的切分）
- QwenFamilyBase: Qwen 家族公共 helper（TP LoRA 状态、KV cache 估算签名）

注意：模型构建函数（load_native_qwen_config / build_native_qwen_model /
build_qwen35_visual_tower）在 model_builders.py，不在本文件。
"""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from graspo.flow.lora.lora_linear import LoRALinear

#: 权重范数分块大小（元素数）。单块 fp32 临时张量 = 4 × 该值 ≈ 32 MiB。
#: 取值理由是显存安全：``embed_tokens``/``lm_head`` 各 1.017e9 参数，一次性
#: ``.float()`` 会产生 ≈4 GiB 峰值，而 native 单卡全参正处在显存边界附近。
_WEIGHT_NORM_CHUNK_ELEMENTS = 8 << 20


def _chunked_parameter_norm(parameters: Iterable[torch.Tensor]) -> float:
    """分块求全部参数的平方和，再开方（fp32 累加，单块临时量有上限）。

    累加器留在参数所在设备上，只在最后做**一次** ``.cpu()`` 同步。
    """
    total: torch.Tensor | None = None
    for param in parameters:
        flat = param.detach().reshape(-1)
        for start in range(0, flat.numel(), _WEIGHT_NORM_CHUNK_ELEMENTS):
            chunk = flat[start : start + _WEIGHT_NORM_CHUNK_ELEMENTS].float()
            partial = chunk.pow(2).sum()
            total = partial if total is None else total + partial
    if total is None:
        return 0.0
    return math.sqrt(float(total.cpu()))


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

    # ── 训练进度指标（模式感知；全参入口的诚实性修复）────────────────────────

    def trainable_parameter_norm(self) -> float:
        """全部 ``requires_grad`` 参数的 L2 范数（全参模式下即"权重范数"）。

        **分块累加**：单块 fp32 临时张量上限 ≈ ``_WEIGHT_NORM_CHUNK_ELEMENTS × 4``
        字节。不能对整参数直接 ``.float()``——``embed_tokens``/``lm_head`` 各
        1.017e9 参数，一次性 fp32 副本 ≈ 4 GiB，而 native 单卡全参正处在显存边界上
        （见 `task-i2-fullparam/report.md` §3），指标不能把训练推过界。
        """
        return _chunked_parameter_norm(
            param for param in self.parameters() if param.requires_grad
        )

    def training_progress_norm(self) -> float:
        """本 rank 的训练进度权重范数（**模式感知的唯一入口**）。

        - lora（默认）：等于 :meth:`lora_parameter_norm`（只统计 ``lora_`` 矩阵，原语义）。
        - full：等于 :meth:`trainable_parameter_norm`（全部可训参数）。全参下
          ``lora_parameter_norm()`` 结构性恒为 0（不存在 lora 参数），直接产出会被
          误读成"权重没变"——见 ``flow/progress_metrics.py``。
        """
        if bool(getattr(self, "full_param", False)):
            return self.trainable_parameter_norm()
        return self.lora_parameter_norm()

    def grad_populated_count(self) -> int:
        """已收到梯度（``grad is not None``）的可训参数个数。

        廉价（O(1)/参数，不做任何张量归约）：全参模式下判定"backward 是否真的到达了
        每一个可训参数"。不要在这里做 ``grad.abs().sum()``——那会对每个参数产生一个
        同尺寸临时张量（1.017e9 参数的参数会产生 ~2 GiB bf16 峰值）。
        """
        return sum(
            1
            for param in self.parameters()
            if param.requires_grad and param.grad is not None
        )

    def training_progress_grad_count(self) -> int:
        """梯度计数（**模式感知的唯一入口**），语义由
        ``flow/progress_metrics.grad_count_event`` 的 ``grad_count_metric`` 声明。"""
        if bool(getattr(self, "full_param", False)):
            return self.grad_populated_count()
        return self.nonzero_lora_grad_count()

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
