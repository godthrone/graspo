import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


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


def export_peft_adapter_from_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model_path: str | Path | None = None,
) -> None:
    payloads = _load_native_payloads(checkpoint_dir, require_metadata=True)
    config = _payload_config(payloads[0])
    lora_config = dict(config.get("lora", {}) or {})
    base_model = str(base_model_path or (config.get("model", {}) or {}).get("model_path") or "")
    tensors, module_specs, graspo_metadata = _reconstruct_peft_export(payloads)
    if not module_specs:
        raise ValueError("Native checkpoint contains no enabled LoRA tensors")
    first_spec = next(iter(module_specs.values()))
    target_modules = sorted({module.rsplit(".", 1)[-1] for module in module_specs})
    rank_pattern = {
        module: int(spec["r"])
        for module, spec in sorted(module_specs.items())
        if int(spec["r"]) != int(first_spec["r"])
    }
    alpha_pattern = {
        module: int(spec["alpha"])
        for module, spec in sorted(module_specs.items())
        if int(spec["alpha"]) != int(first_spec["alpha"])
    }
    adapter_config: dict[str, Any] = {
        "base_model_name_or_path": base_model,
        "bias": lora_config.get("bias", "none"),
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": int(first_spec["alpha"]),
        "lora_dropout": float(lora_config.get("dropout", 0.0)),
        "peft_type": "LORA",
        "r": int(first_spec["r"]),
        "target_modules": target_modules,
        "task_type": lora_config.get("task_type", "CAUSAL_LM"),
    }
    if rank_pattern:
        adapter_config["rank_pattern"] = rank_pattern
    if alpha_pattern:
        adapter_config["alpha_pattern"] = alpha_pattern
    output = _prepare_output_dir(output_dir)
    save_file(tensors, str(output / "adapter_model.safetensors"))
    (output / "adapter_config.json").write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if graspo_metadata["modules"]:
        (output / "graspo_adapter_metadata.json").write_text(
            json.dumps(graspo_metadata, ensure_ascii=False, indent=2) + "\n",
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
    payloads = _load_native_payloads(checkpoint_dir, require_metadata=True)
    deltas = _collect_weight_deltas(payloads)
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
                "rerun training with a newer GRASPO checkpoint "
                "or export from a newer final checkpoint"
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
    tensors, _, _ = _reconstruct_peft_export(payloads)
    return tensors


def _reconstruct_peft_export(
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]], dict[str, Any]]:
    grouped = _records_with_tensors(payloads)
    output: dict[str, torch.Tensor] = {}
    module_specs: dict[str, dict[str, int]] = {}
    graspo_metadata: dict[str, Any] = {
        "format": "graspo-peft-adapter-metadata",
        "version": 1,
        "modules": {},
    }
    for hf_module, items in grouped.items():
        if any(not bool(record.get("peft_exportable", True)) for record, _ in items):
            lora_a, lora_b, slices = _reconstruct_combined_peft_pair(hf_module, items)
            output[f"base_model.model.{hf_module}.lora_A.weight"] = lora_a.contiguous()
            output[f"base_model.model.{hf_module}.lora_B.weight"] = lora_b.contiguous()
            module_specs[hf_module] = {"r": int(lora_a.shape[0]), "alpha": int(lora_a.shape[0])}
            graspo_metadata["modules"][hf_module] = slices
            continue
        lora_a, lora_b = _reconstruct_peft_pair(hf_module, items)
        output[f"base_model.model.{hf_module}.lora_A.weight"] = lora_a.contiguous()
        output[f"base_model.model.{hf_module}.lora_B.weight"] = lora_b.contiguous()
        record = items[0][0]
        module_specs[hf_module] = {
            "r": int(record["r"]),
            "alpha": int(record["alpha"]),
        }
    return output, module_specs, graspo_metadata


def _reconstruct_combined_peft_pair(
    hf_module: str,
    items: list[tuple[dict[str, Any], dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    target_groups: dict[str, list[tuple[dict[str, Any], dict[str, torch.Tensor]]]] = {}
    for record, pair in items:
        key = str(record["target_name"])
        target_groups.setdefault(key, []).append((record, pair))

    rows = _row_count(items)
    cols = _column_count(items)
    target_pairs: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, float]] = []
    for target_name, target_items in sorted(target_groups.items()):
        shard_kinds = {str(record["shard_kind"]) for record, _ in target_items}
        if not shard_kinds <= {"out", "rows", "none"}:
            raise ValueError(
                f"Cannot export mixed/column-sharded fused LoRA target {target_name} "
                f"for {hf_module} as a PEFT adapter"
            )
        a = _require_replicated(
            hf_module,
            [pair["A"] for _, pair in target_items],
            f"{target_name}.lora_A",
        )
        if int(a.shape[1]) != cols:
            raise ValueError(
                f"Native LoRA A for {target_name} has {a.shape[1]} columns, "
                f"but {hf_module} expects {cols}"
            )
        b = torch.zeros(rows, a.shape[0], dtype=target_items[0][1]["B"].dtype)
        for record, pair in target_items:
            if str(record["shard_kind"]) == "none":
                if tuple(pair["B"].shape) != tuple(b.shape):
                    raise ValueError(
                        f"Native LoRA B for {target_name} cannot be embedded into {hf_module}: "
                        f"{tuple(pair['B'].shape)} != {tuple(b.shape)}"
                    )
                b += pair["B"]
            else:
                _place_rows(b, torch.zeros(rows, dtype=torch.bool), pair["B"], record)
        representative = target_items[0][0]
        native_scale = float(representative["alpha"]) / float(representative["r"])
        target_pairs.append((representative, a.cpu(), b.cpu() * native_scale, native_scale))

    total_rank = sum(int(a.shape[0]) for _, a, _, _ in target_pairs)
    if total_rank <= 0:
        raise ValueError(f"Native checkpoint contains no enabled LoRA tensors for {hf_module}")
    combined_a = torch.zeros(total_rank, cols, dtype=target_pairs[0][1].dtype)
    combined_b = torch.zeros(rows, total_rank, dtype=target_pairs[0][2].dtype)
    slices: list[dict[str, Any]] = []
    rank_offset = 0
    for record, a, b, native_scale in target_pairs:
        rank_stop = rank_offset + int(a.shape[0])
        combined_a[rank_offset:rank_stop] = a
        combined_b[:, rank_offset:rank_stop] = b
        slices.append(
            {
                "target_name": str(record["target_name"]),
                "base_weight_name": str(record["base_weight_name"]),
                "rank_start": rank_offset,
                "rank_stop": rank_stop,
                "native_r": int(record["r"]),
                "native_alpha": int(record["alpha"]),
                "adapter_b_scale": native_scale,
            }
        )
        rank_offset = rank_stop
    return combined_a.cpu(), combined_b.cpu(), slices


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
                "strict PEFT adapter export is not representable. "
                "Set export.export_format: merged-hf in your config instead."
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


def _column_count(items: list[tuple[dict[str, Any], dict[str, torch.Tensor]]]) -> int:
    max_col = 0
    for record, pair in items:
        kind = str(record["shard_kind"])
        if kind in {"in", "cols"}:
            max_col = max(max_col, int(record["col_stop"]))
        else:
            max_col = max(max_col, int(pair["A"].shape[1]))
    return max_col


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
) -> dict[str, list[tuple[dict[str, Any], torch.Tensor]]]:
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
                    f"LoRA delta shape {tuple(delta.shape)} does not match "
                    f"{record['base_weight_name']} {tuple(merged.shape)}"
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


def save_lora_adapter(model, tokenizer, output_dir: str | Path) -> None:
    """保存 LoRA adapter 和 tokenizer（兼容 DDP 包装和 HF 接口）。"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = model
    # DDP 包装检测：DistributedDataParallel 将模型放在 .module 属性中
    if hasattr(model, "module"):
        unwrapped = model.module
    # HF 标准接口：save_pretrained 是 HuggingFace 模型的约定
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(path)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(path)
