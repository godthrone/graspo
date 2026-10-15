"""LoRA 权重 IO：PEFT adapter → native 模型的 warm-start 注入。

**职责边界（本次拆分后）**：本模块只负责"把一个标准 PEFT/LoRA adapter 的权重注入
native 模型"（warm-start / 续训初始化）。

**已迁出**：checkpoint → ``merged-hf`` / ``peft-adapter`` 的**导出**子集已搬入
``src/graspo/export/exporters/native_lora.py``（导出是跨后端的标准功能，不属单后端目录）；
``prepare_output_dir`` 的**唯一实现**升格为 ``graspo/export/paths.py``。
"""

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def load_peft_adapter_into_native_model(
    model: torch.nn.Module,
    adapter_path: str | Path,
    *,
    base_model_path: str,
) -> None:
    adapter_dir = Path(adapter_path)
    config, tensors, graspo_metadata = _load_peft_adapter(adapter_dir)
    expected_base = str(config.get("base_model_name_or_path") or "")
    if expected_base and expected_base != str(base_model_path):
        raise ValueError(
            "PEFT adapter base_model_name_or_path does not match model.model_path: "
            f"adapter={expected_base}, runtime={base_model_path}"
        )

    modules = dict(model.named_modules())
    grouped = _group_peft_tensors(tensors)
    consumed: set[str] = set()
    metadata = _model_lora_metadata(model)
    if not metadata:
        return

    for record in metadata:
        hf_module = str(record["hf_module_path"])
        if not bool(record.get("peft_exportable", True)):
            lora_a, lora_b = _slice_graspo_peft_tensor(
                grouped,
                graspo_metadata,
                record,
            )
            module = modules[str(record["module_name"])]
            _copy_parameter(module, "lora_a", lora_a, hf_module)
            _copy_parameter(module, "lora_b", lora_b, hf_module)
            consumed.add(hf_module)
            continue
        adapter_r = _peft_module_int(config, hf_module, "rank_pattern", "r")
        if adapter_r is not None and adapter_r != int(record["r"]):
            raise ValueError(
                f"PEFT adapter r={adapter_r} does not match native LoRA r={record['r']}"
            )
        adapter_alpha = _peft_module_int(config, hf_module, "alpha_pattern", "lora_alpha")
        if adapter_alpha is not None and adapter_alpha != int(record["alpha"]):
            raise ValueError(
                "PEFT adapter lora_alpha="
                f"{adapter_alpha} does not match native LoRA alpha={record['alpha']}"
            )
        pair = grouped.get(hf_module)
        if pair is None:
            raise ValueError(f"PEFT adapter is missing LoRA tensors for {hf_module}")
        module = modules[str(record["module_name"])]
        lora_a = _slice_lora_a(pair["A"], record)
        lora_b = _slice_lora_b(pair["B"], record)
        _copy_parameter(module, "lora_a", lora_a, hf_module)
        _copy_parameter(module, "lora_b", lora_b, hf_module)
        consumed.add(hf_module)

    extra = sorted(set(grouped) - consumed)
    placement = getattr(model, "placement", None)
    is_pipeline_stage = bool(getattr(placement, "is_pipeline", False))
    if extra and not is_pipeline_stage:
        raise ValueError("PEFT adapter contains unsupported LoRA target(s): " + ", ".join(extra))


def _model_lora_metadata(model: torch.nn.Module) -> list[dict[str, Any]]:
    getter = getattr(model, "lora_tensor_metadata", None)
    if not callable(getter):
        raise RuntimeError("Native model does not expose LoRA tensor metadata")
    return list(getter())


def _load_peft_adapter(
    adapter_dir: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    config_path = adapter_dir / "adapter_config.json"
    tensor_path = adapter_dir / "adapter_model.safetensors"
    graspo_metadata_path = adapter_dir / "graspo_adapter_metadata.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter_config.json: {config_path}")
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter_model.safetensors: {tensor_path}")
    graspo_metadata: dict[str, Any] = {}
    if graspo_metadata_path.exists():
        graspo_metadata = json.loads(graspo_metadata_path.read_text(encoding="utf-8"))
    return (
        json.loads(config_path.read_text(encoding="utf-8")),
        load_file(str(tensor_path), device="cpu"),
        graspo_metadata,
    )


def _group_peft_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        module_path, which = _parse_peft_lora_key(key)
        grouped.setdefault(module_path, {})[which] = tensor
    incomplete = [name for name, pair in grouped.items() if set(pair) != {"A", "B"}]
    if incomplete:
        raise ValueError("PEFT adapter has incomplete LoRA A/B pairs: " + ", ".join(incomplete))
    return grouped


def _parse_peft_lora_key(key: str) -> tuple[str, str]:
    value = key
    if value.startswith("base_model.model."):
        value = value[len("base_model.model.") :]
    for marker, which in (
        (".lora_A.default.weight", "A"),
        (".lora_B.default.weight", "B"),
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
    ):
        if value.endswith(marker):
            return value[: -len(marker)], which
    raise ValueError(f"Unsupported PEFT LoRA tensor key: {key}")


def _copy_parameter(
    module: torch.nn.Module,
    parameter_name: str,
    value: torch.Tensor,
    hf_module: str,
) -> None:
    param = getattr(module, parameter_name)
    if param is None:
        raise ValueError(f"Native module {hf_module} has no {parameter_name}")
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(
            f"Shape mismatch for {hf_module}.{parameter_name}: "
            f"adapter={tuple(value.shape)}, native={tuple(param.shape)}"
        )
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def _peft_module_int(
    config: dict[str, Any],
    hf_module: str,
    pattern_key: str,
    default_key: str,
) -> int | None:
    pattern = config.get(pattern_key)
    if isinstance(pattern, dict):
        for key in (
            hf_module,
            f"base_model.model.{hf_module}",
            hf_module.rsplit(".", 1)[-1],
        ):
            if key in pattern:
                return int(pattern[key])
    if default_key in config:
        return int(config[default_key])
    return None


def _slice_lora_a(tensor: torch.Tensor, record: dict[str, Any]) -> torch.Tensor:
    if record["shard_kind"] in {"in", "cols"}:
        return tensor[:, int(record["col_start"]) : int(record["col_stop"])].contiguous()
    return tensor.contiguous()


def _slice_lora_b(tensor: torch.Tensor, record: dict[str, Any]) -> torch.Tensor:
    if record["shard_kind"] == "out":
        return tensor[int(record["row_start"]) : int(record["row_stop"])].contiguous()
    if record["shard_kind"] == "rows":
        indices = record.get("row_indices")
        if indices is not None:
            return tensor.index_select(0, torch.tensor(indices, dtype=torch.long)).contiguous()
        return tensor[int(record["row_start"]) : int(record["row_stop"])].contiguous()
    return tensor.contiguous()


def _slice_graspo_peft_tensor(
    grouped: dict[str, dict[str, torch.Tensor]],
    graspo_metadata: dict[str, Any],
    record: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    hf_module = str(record["hf_module_path"])
    modules = dict(graspo_metadata.get("modules") or {})
    slices = modules.get(hf_module)
    if not isinstance(slices, list):
        raise ValueError(
            "PEFT adapter warm-start cannot initialize native fused/split LoRA target "
            f"{record['target_name']}; missing graspo_adapter_metadata.json slice metadata"
        )
    matched = [
        item
        for item in slices
        if isinstance(item, dict)
        and str(item.get("target_name")) == str(record["target_name"])
        and str(item.get("base_weight_name")) == str(record["base_weight_name"])
    ]
    if len(matched) != 1:
        raise ValueError(
            "PEFT adapter warm-start cannot uniquely map native fused/split LoRA target "
            f"{record['target_name']} in {hf_module}"
        )
    pair = grouped.get(hf_module)
    if pair is None:
        raise ValueError(f"PEFT adapter is missing LoRA tensors for {hf_module}")
    item = matched[0]
    rank_start = int(item["rank_start"])
    rank_stop = int(item["rank_stop"])
    lora_a = pair["A"][rank_start:rank_stop].contiguous()
    lora_b = pair["B"][:, rank_start:rank_stop].contiguous()
    adapter_b_scale = float(item.get("adapter_b_scale") or 1.0)
    if adapter_b_scale == 0.0:
        raise ValueError(f"Invalid zero adapter_b_scale for {hf_module}")
    lora_b = lora_b / adapter_b_scale
    return _slice_lora_a(lora_a, record), _slice_lora_b(lora_b, record)


def save_lora_adapter(model: torch.nn.Module, tokenizer: Any, output_dir: str | Path) -> None:
    """保存 LoRA adapter 和 tokenizer（兼容 DDP 包装和 HF 接口）。

    ★ 输出目录防呆的唯一实现在 ``graspo.export.paths``（本模块不再自带一份，§1.4）；
    在函数体内延迟 import，避免训练栈模块级依赖导出包。
    """
    from graspo.export.paths import prepare_output_dir

    path = prepare_output_dir(output_dir)
    unwrapped = model
    # DDP 包装检测：DistributedDataParallel 将模型放在 .module 属性中
    if hasattr(model, "module"):
        unwrapped = model.module
    # HF 标准接口：save_pretrained 是 HuggingFace 模型的约定
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(path)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)
