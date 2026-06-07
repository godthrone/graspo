from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def load_peft_adapter_into_native_model(
    model: torch.nn.Module,
    adapter_path: str | Path,
    *,
    base_model_path: str,
) -> None:
    adapter_dir = Path(adapter_path)
    config, tensors = _load_peft_adapter(adapter_dir)
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
        if not bool(record.get("peft_exportable", True)):
            raise ValueError(
                "PEFT adapter warm-start cannot initialize native fused/split LoRA target "
                f"{record['target_name']}; use a GRASPO native checkpoint or export merged-hf"
            )
        if "r" in config and int(config["r"]) != int(record["r"]):
            raise ValueError(
                f"PEFT adapter r={config['r']} does not match native LoRA r={record['r']}"
            )
        if "lora_alpha" in config and int(config["lora_alpha"]) != int(record["alpha"]):
            raise ValueError(
                "PEFT adapter lora_alpha="
                f"{config['lora_alpha']} does not match native LoRA alpha={record['alpha']}"
            )
        hf_module = str(record["hf_module_path"])
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
    if extra:
        raise ValueError("PEFT adapter contains unsupported LoRA target(s): " + ", ".join(extra))


def export_peft_adapter_from_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model_path: str | Path | None = None,
) -> None:
    payloads = _load_native_payloads(checkpoint_dir)
    config = _payload_config(payloads[0])
    lora_config = dict(config.get("lora", {}) or {})
    base_model = str(base_model_path or (config.get("model", {}) or {}).get("model_path") or "")
    tensors = _reconstruct_peft_tensors(payloads)
    output = _prepare_output_dir(output_dir)
    save_file(tensors, str(output / "adapter_model.safetensors"))
    (output / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model,
                "bias": lora_config.get("bias", "none"),
                "fan_in_fan_out": False,
                "inference_mode": True,
                "lora_alpha": int(lora_config.get("alpha", _first_metadata(payloads)["alpha"])),
                "lora_dropout": float(lora_config.get("dropout", 0.0)),
                "peft_type": "LORA",
                "r": int(lora_config.get("r", _first_metadata(payloads)["r"])),
                "target_modules": sorted(
                    {
                        str(record["hf_module_path"]).rsplit(".", 1)[-1]
                        for record in _all_metadata(payloads)
                        if bool(record.get("peft_exportable", True))
                    }
                ),
                "task_type": lora_config.get("task_type", "CAUSAL_LM"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def export_merged_hf_from_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model_path: str | Path,
) -> None:
    base = Path(base_model_path)
    if _is_relative_to(Path(output_dir).resolve(), base.resolve()):
        raise ValueError("merged-hf output directory must not be inside the base model directory")
    payloads = _load_native_payloads(checkpoint_dir, require_metadata=False)
    deltas = _collect_weight_deltas(payloads, base_model_path=base)
    output = _prepare_output_dir(output_dir)
    _copy_hf_sidecar_files(base, output)

    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = dict(index["weight_map"])
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = sorted(file.name for file in base.glob("*.safetensors"))
        if not filenames:
            raise FileNotFoundError(f"No safetensors files found in {base}")
        weight_map = {}
        for filename in filenames:
            for key in load_file(str(base / filename), device="cpu").keys():
                weight_map[key] = filename

    total_size = 0
    for filename in filenames:
        state = load_file(str(base / filename), device="cpu")
        updated: dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            merged = tensor
            if name in deltas:
                merged = _apply_deltas(tensor, deltas[name]).to(dtype=tensor.dtype).contiguous()
            updated[name] = merged
            total_size += int(updated[name].numel() * updated[name].element_size())
        save_file(updated, str(output / filename))

    if index_path.exists():
        index["metadata"] = dict(index.get("metadata") or {})
        index["metadata"]["total_size"] = total_size
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def export_from_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    export_format: str,
    base_model_path: str | Path | None = None,
) -> None:
    if export_format == "peft-adapter":
        export_peft_adapter_from_checkpoint(
            checkpoint_dir, output_dir, base_model_path=base_model_path
        )
        return
    if export_format == "merged-hf":
        if base_model_path is None:
            config = _payload_config(_load_native_payloads(checkpoint_dir)[0])
            base_model_path = (config.get("model", {}) or {}).get("model_path")
        if not base_model_path:
            raise ValueError("--base-model is required when checkpoint config has no model_path")
        export_merged_hf_from_checkpoint(
            checkpoint_dir, output_dir, base_model_path=base_model_path
        )
        return
    raise ValueError(f"Unsupported export format: {export_format}")


def _model_lora_metadata(model: torch.nn.Module) -> list[dict[str, Any]]:
    getter = getattr(model, "lora_tensor_metadata", None)
    if not callable(getter):
        raise RuntimeError("Native model does not expose LoRA tensor metadata")
    return list(getter())


def _load_peft_adapter(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    config_path = adapter_dir / "adapter_config.json"
    tensor_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter_config.json: {config_path}")
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter_model.safetensors: {tensor_path}")
    return (
        json.loads(config_path.read_text(encoding="utf-8")),
        load_file(str(tensor_path), device="cpu"),
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


def _load_native_payloads(
    checkpoint_dir: str | Path,
    *,
    require_metadata: bool = True,
) -> list[dict[str, Any]]:
    directory = Path(checkpoint_dir)
    paths = sorted(directory.glob("rank_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No native GRASPO checkpoint rank shards found in {directory}")
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if require_metadata and "lora_tensor_metadata" not in payload:
            raise ValueError(
                f"Checkpoint shard {path} has no lora_tensor_metadata; "
                "rerun training with a newer GRASPO checkpoint or export from a newer final checkpoint"
            )
        payloads.append(payload)
    return payloads


def _payload_config(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("config") or {})


def _all_metadata(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        for record in payload.get("lora_tensor_metadata") or []:
            records.append(dict(record))
    return records


def _first_metadata(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    records = _all_metadata(payloads)
    if not records:
        raise ValueError("Native checkpoint contains no enabled LoRA tensors")
    return records[0]


def _reconstruct_peft_tensors(payloads: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    grouped = _records_with_tensors(payloads)
    output: dict[str, torch.Tensor] = {}
    for hf_module, items in grouped.items():
        if any(not bool(record.get("peft_exportable", True)) for record, _ in items):
            target_names = sorted({str(record["target_name"]) for record, _ in items})
            raise ValueError(
                "Cannot export fused/split native LoRA target(s) as strict PEFT adapter: "
                + ", ".join(target_names)
                + ". Use --format merged-hf instead."
            )
        lora_a, lora_b = _reconstruct_peft_pair(hf_module, items)
        output[f"base_model.model.{hf_module}.lora_A.weight"] = lora_a.contiguous()
        output[f"base_model.model.{hf_module}.lora_B.weight"] = lora_b.contiguous()
    return output


def _records_with_tensors(
    payloads: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, torch.Tensor]]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, torch.Tensor]]]] = {}
    for payload in payloads:
        state = dict(payload["lora_state_dict"])
        for record in payload.get("lora_tensor_metadata") or []:
            rec = dict(record)
            pair = {
                "A": state[str(rec["lora_a_name"])],
                "B": state[str(rec["lora_b_name"])],
            }
            grouped.setdefault(str(rec["hf_module_path"]), []).append((rec, pair))
    return grouped


def _ensure_merge_metadata(
    payloads: list[dict[str, Any]],
    *,
    base_model_path: Path,
) -> None:
    if all(payload.get("lora_tensor_metadata") for payload in payloads):
        return
    base_shapes = _base_safetensor_shapes(base_model_path)
    for payload in payloads:
        if payload.get("lora_tensor_metadata"):
            continue
        payload["lora_tensor_metadata"] = _infer_merge_metadata(payload, base_shapes)


def _base_safetensor_shapes(base: Path) -> dict[str, tuple[int, ...]]:
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = sorted(set(dict(index["weight_map"]).values()))
    else:
        filenames = sorted(file.name for file in base.glob("*.safetensors"))
    if not filenames:
        raise FileNotFoundError(f"No safetensors files found in {base}")
    shapes: dict[str, tuple[int, ...]] = {}
    for filename in filenames:
        with safe_open(str(base / filename), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[str(key)] = tuple(int(v) for v in handle.get_slice(key).get_shape())
    return shapes


def _infer_merge_metadata(
    payload: dict[str, Any],
    base_shapes: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    state = dict(payload.get("lora_state_dict") or {})
    if not state:
        raise ValueError("Native checkpoint contains no lora_state_dict")
    config = _payload_config(payload)
    lora_config = dict(config.get("lora", {}) or {})
    default_r = int(lora_config.get("r", 0) or 0)
    default_alpha = int(lora_config.get("alpha", default_r or 1))
    records: list[dict[str, Any]] = []
    for a_name in sorted(k for k in state if k.endswith(".lora_a")):
        module_name = a_name[: -len(".lora_a")]
        b_name = f"{module_name}.lora_b"
        if b_name not in state:
            raise ValueError(f"Native checkpoint has {a_name} without matching {b_name}")
        record = _infer_merge_record(
            module_name,
            state[a_name],
            state[b_name],
            payload,
            base_shapes,
            default_r=default_r,
            default_alpha=default_alpha,
        )
        record["lora_a_name"] = a_name
        record["lora_b_name"] = b_name
        records.append(record)
    if not records:
        raise ValueError("Native checkpoint contains no enabled LoRA tensors")
    return records


def _infer_merge_record(
    module_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    payload: dict[str, Any],
    base_shapes: dict[str, tuple[int, ...]],
    *,
    default_r: int,
    default_alpha: int,
) -> dict[str, Any]:
    tp_rank = int(payload.get("tp_rank", 0))
    tp_size = int(payload.get("tp_size", 1))
    base_module_name = _globalize_legacy_module_name(module_name, payload)
    candidates = _legacy_merge_candidates(base_module_name, lora_a, lora_b, tp_rank, tp_size)
    matches = [
        candidate
        for candidate in candidates
        if _candidate_matches_base(candidate, lora_a, lora_b, base_shapes)
    ]
    if len(matches) != 1:
        detail = ", ".join(str(c["base_weight_name"]) for c in candidates)
        if not matches:
            raise ValueError(
                f"Cannot infer LoRA merge target for legacy checkpoint tensor {module_name}; "
                f"tried: {detail}"
            )
        raise ValueError(
            f"Ambiguous LoRA merge target for legacy checkpoint tensor {module_name}: "
            + ", ".join(str(c["base_weight_name"]) for c in matches)
        )
    record = matches[0]
    base_name = str(record["base_weight_name"])
    record = _resolve_legacy_dynamic_offsets(record, int(base_shapes[base_name][0]))
    record["module_name"] = module_name
    record["r"] = int(default_r or lora_a.shape[0])
    record["alpha"] = int(default_alpha)
    return record


def _globalize_legacy_module_name(module_name: str, payload: dict[str, Any]) -> str:
    parts = module_name.split(".")
    if len(parts) < 2 or parts[0] != "layers" or not parts[1].isdigit():
        return module_name
    local_index = int(parts[1])
    placement = dict(payload.get("placement") or {})
    local_layers = placement.get("local_layer_indices")
    if not isinstance(local_layers, list) or local_index >= len(local_layers):
        return module_name
    global_index = int(local_layers[local_index])
    return ".".join(["layers", str(global_index), *parts[2:]])


def _legacy_merge_candidates(
    module_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> list[dict[str, Any]]:
    parts = module_name.split(".")
    if len(parts) < 4 or parts[0] != "layers" or not parts[1].isdigit():
        return []
    layer = int(parts[1])
    suffix = ".".join(parts[2:])
    prefixes = [
        f"model.language_model.layers.{layer}",
        f"model.layers.{layer}",
    ]
    records: list[dict[str, Any]] = []
    if suffix.startswith("token_mixer."):
        name = suffix[len("token_mixer.") :]
        for prefix in prefixes:
            if name in {"q_proj", "k_proj", "v_proj"}:
                records.append(
                    _record_template(
                        target_name=f"language.full_attn.{name}",
                        hf_module_path=f"{prefix}.self_attn.{name}",
                        shard_kind="rows",
                        row_start=tp_rank * int(lora_b.shape[0]),
                        row_stop=(tp_rank + 1) * int(lora_b.shape[0]),
                    )
                )
                records.append(
                    _linear_qkv_record(
                        prefix=f"{prefix}.linear_attn",
                        name=name,
                        lora_b=lora_b,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                    )
                )
            elif name == "o_proj":
                records.append(
                    _record_template(
                        target_name="language.full_attn.o_proj",
                        hf_module_path=f"{prefix}.self_attn.o_proj",
                        shard_kind="in",
                        col_start=tp_rank * int(lora_a.shape[1]),
                        col_stop=(tp_rank + 1) * int(lora_a.shape[1]),
                    )
                )
            elif name in {"in_proj_z", "out_proj"}:
                shard = "out" if name == "in_proj_z" else "in"
                kwargs: dict[str, int] = {}
                if shard == "out":
                    kwargs = {
                        "row_start": tp_rank * int(lora_b.shape[0]),
                        "row_stop": (tp_rank + 1) * int(lora_b.shape[0]),
                    }
                else:
                    kwargs = {
                        "col_start": tp_rank * int(lora_a.shape[1]),
                        "col_stop": (tp_rank + 1) * int(lora_a.shape[1]),
                    }
                records.append(
                    _record_template(
                        target_name=f"language.linear_attn.{name}",
                        hf_module_path=f"{prefix}.linear_attn.{name}",
                        shard_kind=shard,
                        **kwargs,
                    )
                )
        return records
    if suffix.startswith("mlp."):
        name = suffix[len("mlp.") :]
        if name not in {"gate_proj", "up_proj", "down_proj"}:
            return []
        for prefix in prefixes:
            shard = "in" if name == "down_proj" else "out"
            kwargs = (
                {
                    "col_start": tp_rank * int(lora_a.shape[1]),
                    "col_stop": (tp_rank + 1) * int(lora_a.shape[1]),
                }
                if shard == "in"
                else {
                    "row_start": tp_rank * int(lora_b.shape[0]),
                    "row_stop": (tp_rank + 1) * int(lora_b.shape[0]),
                }
            )
            records.append(
                _record_template(
                    target_name=f"language.mlp.{name}",
                    hf_module_path=f"{prefix}.mlp.{name}",
                    shard_kind=shard,
                    **kwargs,
                )
            )
    return records


def _linear_qkv_record(
    *,
    prefix: str,
    name: str,
    lora_b: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> dict[str, Any]:
    local_rows = int(lora_b.shape[0])
    # The exact q/k/v offset is validated later against the base fused weight
    # shape. Value rows are placed after the full q and k blocks.
    if name == "q_proj":
        row_start = tp_rank * local_rows
    elif name == "k_proj":
        row_start = local_rows * tp_size + tp_rank * local_rows
    else:
        row_start = -1
    record = _record_template(
        target_name=f"language.linear_attn.{name}",
        hf_module_path=f"{prefix}.in_proj_qkv",
        base_weight_name=f"{prefix}.in_proj_qkv.weight",
        shard_kind="rows",
        row_start=row_start,
        row_stop=row_start + local_rows if row_start >= 0 else -1,
        peft_exportable=False,
    )
    if name == "v_proj":
        record["_legacy_linear_qkv_value_rows"] = local_rows * tp_size
        record["_legacy_linear_qkv_local_rows"] = local_rows
        record["_legacy_linear_qkv_tp_rank"] = tp_rank
    return record


def _record_template(
    *,
    target_name: str,
    hf_module_path: str,
    shard_kind: str,
    base_weight_name: str | None = None,
    row_start: int | None = None,
    row_stop: int | None = None,
    col_start: int | None = None,
    col_stop: int | None = None,
    peft_exportable: bool = True,
) -> dict[str, Any]:
    return {
        "target_name": target_name,
        "hf_module_path": hf_module_path,
        "base_weight_name": base_weight_name or f"{hf_module_path}.weight",
        "shard_kind": shard_kind,
        "row_start": row_start,
        "row_stop": row_stop,
        "col_start": col_start,
        "col_stop": col_stop,
        "row_indices": None,
        "peft_exportable": peft_exportable,
    }


def _candidate_matches_base(
    candidate: dict[str, Any],
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    base_shapes: dict[str, tuple[int, ...]],
) -> bool:
    base_name = str(candidate["base_weight_name"])
    shape = base_shapes.get(base_name)
    if shape is None or len(shape) != 2:
        return False
    rows, cols = shape
    candidate = _resolve_legacy_dynamic_offsets(candidate, rows)
    kind = str(candidate["shard_kind"])
    if kind in {"out", "rows"}:
        start = int(candidate["row_start"])
        stop = int(candidate["row_stop"])
        return (
            stop <= rows and stop - start == int(lora_b.shape[0]) and cols == int(lora_a.shape[1])
        )
    if kind in {"in", "cols"}:
        start = int(candidate["col_start"])
        stop = int(candidate["col_stop"])
        return (
            rows == int(lora_b.shape[0]) and stop <= cols and stop - start == int(lora_a.shape[1])
        )
    if kind == "none":
        return rows == int(lora_b.shape[0]) and cols == int(lora_a.shape[1])
    return False


def _resolve_legacy_dynamic_offsets(candidate: dict[str, Any], base_rows: int) -> dict[str, Any]:
    if "_legacy_linear_qkv_value_rows" not in candidate:
        return candidate
    record = dict(candidate)
    value_rows = int(record.pop("_legacy_linear_qkv_value_rows"))
    local_rows = int(record.pop("_legacy_linear_qkv_local_rows"))
    tp_rank = int(record.pop("_legacy_linear_qkv_tp_rank"))
    key_rows_twice = base_rows - value_rows
    if key_rows_twice <= 0 or key_rows_twice % 2 != 0:
        return record
    row_start = key_rows_twice + tp_rank * local_rows
    record["row_start"] = row_start
    record["row_stop"] = row_start + local_rows
    return record


def _reconstruct_peft_pair(
    hf_module: str,
    items: list[tuple[dict[str, Any], dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    shard_kinds = {str(record["shard_kind"]) for record, _ in items}
    if shard_kinds == {"none"}:
        return items[0][1]["A"].cpu(), items[0][1]["B"].cpu()
    if shard_kinds <= {"out", "rows"}:
        a = _require_replicated(hf_module, [pair["A"] for _, pair in items], "lora_A")
        rows = _row_count(items)
        b = torch.zeros(rows, a.shape[0], dtype=items[0][1]["B"].dtype)
        filled = torch.zeros(rows, dtype=torch.bool)
        for record, pair in items:
            _place_rows(b, filled, pair["B"], record)
        _require_all_filled(hf_module, filled)
        return a.cpu(), b.cpu()
    if shard_kinds <= {"in", "cols"}:
        b = _require_replicated(hf_module, [pair["B"] for _, pair in items], "lora_B")
        cols = max(int(record["col_stop"]) for record, _ in items)
        a = torch.zeros(b.shape[1], cols, dtype=items[0][1]["A"].dtype)
        filled = torch.zeros(cols, dtype=torch.bool)
        for record, pair in items:
            start = int(record["col_start"])
            stop = int(record["col_stop"])
            a[:, start:stop] = pair["A"]
            filled[start:stop] = True
        _require_all_filled(hf_module, filled)
        return a.cpu(), b.cpu()
    raise ValueError(f"Cannot reconstruct mixed LoRA shard kinds for {hf_module}: {shard_kinds}")


def _require_replicated(
    hf_module: str,
    tensors: list[torch.Tensor],
    label: str,
) -> torch.Tensor:
    first = tensors[0].cpu()
    for tensor in tensors[1:]:
        if tuple(tensor.shape) != tuple(first.shape) or not torch.allclose(
            tensor.cpu(), first, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(
                f"Native TP shards for {hf_module} have non-identical replicated {label}; "
                "strict PEFT adapter export is not representable. Use --format merged-hf instead."
            )
    return first


def _row_count(items: list[tuple[dict[str, Any], dict[str, torch.Tensor]]]) -> int:
    max_row = 0
    for record, _ in items:
        indices = record.get("row_indices")
        if indices is not None:
            max_row = max(max_row, max(int(idx) for idx in indices) + 1)
        else:
            max_row = max(max_row, int(record["row_stop"]))
    return max_row


def _place_rows(
    output: torch.Tensor,
    filled: torch.Tensor,
    rows: torch.Tensor,
    record: dict[str, Any],
) -> None:
    indices = record.get("row_indices")
    if indices is not None:
        index = torch.tensor(indices, dtype=torch.long)
        output.index_copy_(0, index, rows)
        filled[index] = True
        return
    start = int(record["row_start"])
    stop = int(record["row_stop"])
    output[start:stop] = rows
    filled[start:stop] = True


def _require_all_filled(hf_module: str, filled: torch.Tensor) -> None:
    if not bool(filled.all()):
        raise ValueError(f"Native TP shards for {hf_module} do not cover the full tensor")


def _collect_weight_deltas(
    payloads: list[dict[str, Any]],
    *,
    base_model_path: Path,
) -> dict[str, list[tuple[dict[str, Any], torch.Tensor]]]:
    _ensure_merge_metadata(payloads, base_model_path=base_model_path)
    grouped: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for items in _records_with_tensors(payloads).values():
        for record, pair in items:
            delta = pair["B"].float().matmul(pair["A"].float())
            delta = delta * (float(record["alpha"]) / float(record["r"]))
            grouped.setdefault(str(record["base_weight_name"]), []).append((record, delta.cpu()))
    return grouped


def _apply_deltas(
    tensor: torch.Tensor,
    deltas: list[tuple[dict[str, Any], torch.Tensor]],
) -> torch.Tensor:
    merged = tensor.float().clone()
    for record, delta in deltas:
        kind = str(record["shard_kind"])
        if kind in {"none"}:
            if tuple(delta.shape) != tuple(merged.shape):
                raise ValueError(
                    f"LoRA delta shape {tuple(delta.shape)} does not match {record['base_weight_name']} {tuple(merged.shape)}"
                )
            merged += delta
        elif kind == "out":
            merged[int(record["row_start"]) : int(record["row_stop"])] += delta
        elif kind == "rows":
            indices = record.get("row_indices")
            if indices is not None:
                merged.index_add_(0, torch.tensor(indices, dtype=torch.long), delta)
            else:
                merged[int(record["row_start"]) : int(record["row_stop"])] += delta
        elif kind == "in":
            merged[:, int(record["col_start"]) : int(record["col_stop"])] += delta
        elif kind == "cols":
            merged[:, int(record["col_start"]) : int(record["col_stop"])] += delta
        else:
            raise ValueError(f"Unsupported LoRA shard kind for merge: {kind}")
    return merged


def _prepare_output_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _copy_hf_sidecar_files(base: Path, output: Path) -> None:
    skip_suffixes = {".safetensors"}
    skip_names = {"model.safetensors.index.json"}
    for item in base.iterdir():
        if item.is_dir():
            if item.name in {".git", "__pycache__"}:
                continue
            shutil.copytree(item, output / item.name, dirs_exist_ok=True)
            continue
        if item.name in skip_names or item.suffix in skip_suffixes:
            continue
        shutil.copy2(item, output / item.name)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
