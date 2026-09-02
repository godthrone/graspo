"""Qwen 家族模型构建器 —— 设施层（flow.adapters.models.common）。

- load_native_qwen_config: 从 config.json 加载并规范化 Qwen 家族配置
- build_native_qwen_model: 按 family 分派构建原生 TP 模型（qwen3 / qwen3_5_text）
- build_qwen35_visual_tower: 构建 Qwen3.5 视觉塔（HF weights + LoRA 替换 + RoPE 精度修复）

从 qwen3/model.py 拆出：修复 qwen35_36 家族反向依赖 qwen3 的错位
（视觉塔/配置加载是家族共享能力，不属于任何单一家族）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from graspo.flow.adapters.models.common.native_qwen_config import NativeQwenConfig
from graspo.flow.lora.lora_linear import _replace_visual_lora_modules

if TYPE_CHECKING:
    from graspo.flow.parallel.placement_plan import NativePlacementPlan
    from graspo.flow.parallel.tensor_utils import SafetensorIndex


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
        f"native supports text-only qwen3 and qwen3_5_text models; "
        f"got model_type={model_type!r}"
    )


def build_native_qwen_model(
    *,
    hf_config: NativeQwenConfig,
    loader: SafetensorIndex,
    tp_rank: int,
    tp_size: int,
    use_sp: bool = False,
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
        # 函数内延迟导入：common 层不依赖具体家族（防家族反向依赖）
        from graspo.flow.adapters.models.qwen3.model import Qwen3DenseModel

        return Qwen3DenseModel(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_sp=use_sp,
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
        from graspo.flow.adapters.models.qwen35_36.model import Qwen35HybridTextModel

        return Qwen35HybridTextModel(
            hf_config=hf_config,
            loader=loader,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_sp=use_sp,
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


def build_qwen35_visual_tower(
    *,
    hf_config: NativeQwenConfig,
    loader: SafetensorIndex,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: set[str],
    gradient_checkpointing: bool = False,
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
    # HF Transformers 版本兼容：某些版本不会自动设置 _attn_implementation
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
    # Fix: .to(dtype=torch_dtype) casts the inv_freq buffer from float32 to the
    # model dtype (e.g. bfloat16), losing ~3 significant digits.  That tiny error
    # compounds across 27 ViT layers and destroys image features, which then
    # propagates through the decoder and corrupts tool-call generation.
    # Recompute inv_freq in float32 explicitly to match HF precision.
    for _mod in visual.modules():
        # 仅为 RoPE 模块重新计算 inv_freq（HF 模块内部探测）
        if hasattr(_mod, "inv_freq"):
            _inv_freq = 1.0 / (
                _mod.theta
                ** (torch.arange(0, _mod.dim, 2, dtype=torch.float, device=device) / _mod.dim)
            )
            _mod.register_buffer("inv_freq", _inv_freq, persistent=False)
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
    if gradient_checkpointing:
        # 视觉塔 27 层 ViT 默认不启用 checkpointing，每层激活值全部保留。
        # 高分辨率/多图场景下视觉塔激活值可达 27 GB（实测），启用后降至 ~1 GB。
        # HF PretrainedModel 标准接口，内部对每个 ViT layer 包 checkpoint。
        visual.gradient_checkpointing_enable()
    return visual
