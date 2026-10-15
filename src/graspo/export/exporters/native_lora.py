"""native LoRA rank 分片（``manifest.json`` + ``rank_*.pt``）→ ``merged-hf`` / ``peft-adapter``。

**形态**：F3–F6（``dp``/``tp``/``pp`` 全形态）。

**职责**：把 backend 侧 ``flow/backend/native/lora/lora_io.py`` 的**导出子集**
（``:86-213`` / ``:366-742``）**语义等价地**搬入本文件，并加上 fail-closed 边界：
``manifest.json`` 是拓扑唯一真相源（V3/V4）、分片完备性（V5）、PP 覆盖（V6）、
DP 副本逐键唯一性（V7）、LoRA 语义防线（V8）。

**本文件不负责**：不碰 PEFT 目录、不碰全参、不做 warm-start 注入（留在 native）。

⛔ 与旧实现的**唯一行为差异**：旧 ``_load_native_payloads`` 对 ``dp>1`` **静默**取
``dp_rank==0`` 单副本；本文件改为"逐键比对一致才合、分歧即 fail-closed"（设计 §4.1 V7）。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from graspo.export import verify
from graspo.export.base import Exporter, LoadedShards
from graspo.export.errors import ExportError
from graspo.export.schema import (
    CheckpointClassification,
    CheckpointForm,
    ExportTarget,
    NativeTopology,
    PlacementRecord,
    ShardPlan,
    ShardRef,
)
from graspo.export.semantics import require_lora_semantics

_FORMAT_LORA = "native-lora"
_MANIFEST_NAME = "manifest.json"


# ══════════════════════════════════════════════════════════════════════════════
# 组装的中间产物（纯数据，不含 IO）
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PeftAdapterAssembly:
    """待落盘的 PEFT adapter：三份文件的内容。"""

    tensors: dict[str, Any]
    adapter_config: dict[str, Any]
    graspo_metadata: dict[str, Any]
    #: 回读器（由 ``_write`` 钩子注入 ``safetensors.torch.load_file``）——
    #: 让回读自校验不必再 import 重依赖（C3：重依赖只在 ``_load``/``_write``）。
    reader: Any = None


@dataclass(slots=True)
class MergedHfAssembly:
    """待落盘的 merged-hf：按基座权重名聚合的 LoRA delta。"""

    deltas: dict[str, list[tuple[dict[str, Any], Any]]]
    #: 头读取器（由 ``_write`` 钩子注入 ``safetensors.safe_open``），同上。
    header_reader: Any = None


# ══════════════════════════════════════════════════════════════════════════════
# manifest / 拓扑（纯 JSON，无 torch）
# ══════════════════════════════════════════════════════════════════════════════


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest = root / _MANIFEST_NAME
    if not manifest.is_file():
        raise ExportError(
            f"native checkpoint is missing manifest.json (the topology truth source): {root}; "
            "⛔ refusing to infer topology from filenames/shard presence"
        )
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{manifest} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ExportError(f"{manifest}: expected a JSON object, got {type(data).__name__}")
    return data


def _parse_placement(raw: Any, *, where: str) -> PlacementRecord | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ExportError(f"{where}: placement must be a JSON object, got {type(raw).__name__}")
    known = set(PlacementRecord.model_fields)
    return PlacementRecord(**{key: value for key, value in raw.items() if key in known})


def _parse_topology(data: dict[str, Any], *, where: str) -> NativeTopology:
    fmt = str(data.get("format") or "")
    if fmt != _FORMAT_LORA:
        raise ExportError(
            f"{where}: manifest format={fmt!r} does not match native LoRA form "
            f"{CheckpointForm.NATIVE_LORA_RANK.value!r} (V4); refusing to continue"
        )
    try:
        return NativeTopology(
            format=fmt,  # type: ignore[arg-type]
            tuner_type=str(data.get("tuner_type") or ""),
            dp_size=int(data.get("dp_size") or 0),
            tp_size=int(data.get("tp_size") or 0),
            pp_size=int(data.get("pp_size") or 0),
            world_size=(int(data["world_size"]) if data.get("world_size") is not None else None),
            checkpoint_type=(
                str(data["checkpoint_type"]) if data.get("checkpoint_type") is not None else None
            ),
            has_trainer_state=bool(data.get("has_trainer_state", False)),
            placement=_parse_placement(data.get("placement"), where=where),
            lora_target_signature=dict(data.get("lora_target_signature") or {}),
        )
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{where}: manifest topology fields are invalid: {exc}") from None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: F7（native 全参）的处置**不在本模块**：P5 落地后由
#: :mod:`graspo.export.exporters.native_full_param` 提供真实导出器
#: （``NativeFullParamRankExporter``，单卡单分片）。


# ══════════════════════════════════════════════════════════════════════════════
# 导出器
# ══════════════════════════════════════════════════════════════════════════════


class NativeLoraRankExporter(Exporter):
    """native LoRA ``.pt`` 分片 → ``merged-hf`` / ``peft-adapter``。"""

    form = CheckpointForm.NATIVE_LORA_RANK
    supported_targets = frozenset({ExportTarget.MERGED_HF, ExportTarget.PEFT_ADAPTER})

    # ── 形态检测（纯 FS）──────────────────────────────────────────────
    @classmethod
    def _matches(cls, root: Path) -> tuple[bool, tuple[str, ...]]:
        manifest = root / _MANIFEST_NAME
        if not manifest.is_file():
            return False, ()
        shards = sorted(root.glob("rank_*.pt"))
        if not shards:
            return False, ()
        try:
            data = _read_manifest(root)
        except ExportError:
            return False, ()
        if str(data.get("format") or "") != _FORMAT_LORA:
            return False, ()
        return True, (
            f"found manifest.json format={_FORMAT_LORA}",
            f"found {len(shards)} .pt rank shard(s)",
        )

    # ── 规划（纯 JSON）────────────────────────────────────────────────
    def _plan(
        self,
        config: Any,
        classification: CheckpointClassification,
        *,
        base_model_path: str | None = None,
    ) -> ShardPlan:
        root = Path(classification.path)
        data = _read_manifest(root)
        topology = _parse_topology(data, where=str(root / _MANIFEST_NAME))
        target = ExportTarget.parse(str(getattr(config, "export_format", "")))
        return ShardPlan(
            form=self.form,
            target=target,
            checkpoint_name=root.name,
            source_dir=str(root),
            topology=topology,
            base_model_path=(str(base_model_path) if base_model_path else None),
        )

    def _assert_plan_is_sound(self, plan: ShardPlan) -> None:
        super()._assert_plan_is_sound(plan)
        # V10：基座路径**唯一来源** = 调用方传入的 config.model.model_path，
        # ⛔ 不回落读 checkpoint 内嵌 config（那是第二个真相源）。
        if not plan.base_model_path:
            raise ExportError(
                f"native export to {plan.target.value!r} requires an explicit base model path "
                "(config.model.model_path); refusing to fall back to the checkpoint-embedded "
                "model_path (V10 single truth source)"
            )

    # ── 加载（唯一允许 import torch 的钩子之一）──────────────────────────
    def _load(self, plan: ShardPlan) -> LoadedShards:
        import torch

        root = Path(plan.source_dir)
        shard_paths = sorted(root.glob("rank_*.pt"))
        if not shard_paths:
            raise ExportError(f"No native GRASPO checkpoint rank shards found in {root}")

        payloads: list[dict[str, Any]] = []
        refs: list[ShardRef] = []
        for path in shard_paths:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # 旧版 torch 不支持 weights_only
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise ExportError(f"Checkpoint shard {path} did not load into a mapping")
            if "lora_tensor_metadata" not in payload:
                raise ExportError(
                    f"Checkpoint shard {path} has no lora_tensor_metadata; rerun training with "
                    "a newer GRASPO checkpoint or export from a newer final checkpoint"
                )
            refs.append(
                ShardRef(
                    name=path.name,
                    dp_rank=int(payload.get("dp_rank", 0)),
                    tp_rank=int(payload.get("tp_rank", 0)),
                    pp_rank=int(payload.get("pp_rank", 0)),
                    sha256=_file_sha256(path),
                )
            )
            payloads.append(payload)
        return LoadedShards(payloads=payloads, shard_refs=refs, paths=shard_paths)

    # ── 分片级校验（V5/V7 在基类，V6/V8 在本形态）──────────────────────
    def _assert_shards_are_sound(self, plan: ShardPlan, loaded: LoadedShards):
        replica_check = super()._assert_shards_are_sound(plan, loaded)
        # V8：全参 shard 不得走 LoRA 语义导出（逐字保留的既有防线）。
        try:
            require_lora_semantics(loaded.payloads, operation=plan.target.value)
        except ValueError as exc:
            raise ExportError(str(exc)) from None
        self._assert_pp_coverage(plan, loaded)
        return replica_check

    def _assert_pp_coverage(self, plan: ShardPlan, loaded: LoadedShards) -> None:
        """V6：``pp>1`` 时各 stage 的 ``local_layer_indices`` 并集必须覆盖全部层且不重，
        且 ``include_lm_head`` / ``include_embeddings`` 各恰有一个 stage 为真。

        ``manifest.placement`` 只由 rank0 写、只反映它自己那段；因此覆盖校验按
        **各 shard payload 自带的 placement** 汇总（与 ``transformer_model_adapter`` 的
        写出事实一致）。
        """
        topology = plan.topology
        if topology is None or topology.pp_size <= 1:
            return

        layers_by_stage: dict[int, set[int]] = {}
        lm_head_stages: set[int] = set()
        embedding_stages: set[int] = set()
        for ref, payload in zip(loaded.shard_refs, loaded.payloads, strict=True):
            placement = payload.get("placement")
            if not isinstance(placement, dict):
                raise ExportError(
                    f"pp_size={topology.pp_size} requires each shard to carry its placement "
                    f"(shard {ref.name} has none); ⛔ refusing to merge without PP coverage proof"
                )
            indices = {int(item) for item in (placement.get("local_layer_indices") or [])}
            layers_by_stage.setdefault(ref.pp_rank, set()).update(indices)
            if bool(placement.get("include_lm_head")):
                lm_head_stages.add(ref.pp_rank)
            if bool(placement.get("include_embeddings")):
                embedding_stages.add(ref.pp_rank)

        flattened: list[int] = []
        for stage in sorted(layers_by_stage):
            flattened.extend(sorted(layers_by_stage[stage]))
        if len(flattened) != len(set(flattened)):
            raise ExportError(
                f"PP stages overlap in local_layer_indices for {plan.checkpoint_name}: "
                f"{sorted(layers_by_stage)} — refusing to merge"
            )
        if flattened and flattened != list(range(flattened[-1] + 1)):
            raise ExportError(
                f"PP stages do not cover a contiguous layer range for {plan.checkpoint_name}: "
                f"got {flattened} — refusing to merge (a missing PP shard would silently keep "
                "base weights for those layers)"
            )
        if len(lm_head_stages) != 1:
            raise ExportError(
                f"expected exactly one PP stage to own lm_head, got {sorted(lm_head_stages)} "
                f"for {plan.checkpoint_name}"
            )
        if len(embedding_stages) != 1:
            raise ExportError(
                f"expected exactly one PP stage to own embeddings, got {sorted(embedding_stages)} "
                f"for {plan.checkpoint_name}"
            )

    # ── 组装（纯张量变换）──────────────────────────────────────────────
    def _assemble(self, loaded: LoadedShards, plan: ShardPlan) -> Any:
        payloads = _dedupe_dp_replicas(loaded, plan)
        if plan.target is ExportTarget.PEFT_ADAPTER:
            tensors, module_specs, graspo_metadata = _reconstruct_peft_export(payloads)
            if not module_specs:
                raise ExportError("Native checkpoint contains no enabled LoRA tensors")
            adapter_config = _build_adapter_config(payloads, module_specs, plan.base_model_path)
            return PeftAdapterAssembly(
                tensors=tensors, adapter_config=adapter_config, graspo_metadata=graspo_metadata
            )
        if plan.target is ExportTarget.MERGED_HF:
            return MergedHfAssembly(deltas=_collect_weight_deltas(payloads))
        raise ExportError(f"unsupported target for native LoRA exporter: {plan.target.value!r}")

    # ── 落盘（唯一允许写盘的钩子）──────────────────────────────────────
    def _write(self, destination: Path, assembled: Any, plan: ShardPlan) -> tuple[str, ...]:
        if isinstance(assembled, PeftAdapterAssembly):
            return self._write_peft_adapter(destination, assembled)
        if isinstance(assembled, MergedHfAssembly):
            return self._write_merged_hf(destination, assembled, plan)
        raise ExportError(f"unknown assembly type: {type(assembled).__name__}")

    @staticmethod
    def _write_peft_adapter(destination: Path, assembled: PeftAdapterAssembly) -> tuple[str, ...]:
        from safetensors.torch import load_file, save_file

        save_file(assembled.tensors, str(destination / "adapter_model.safetensors"))
        assembled.reader = load_file
        written = ["adapter_model.safetensors"]
        (destination / "adapter_config.json").write_text(
            json.dumps(assembled.adapter_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append("adapter_config.json")
        if assembled.graspo_metadata["modules"]:
            (destination / "graspo_adapter_metadata.json").write_text(
                json.dumps(assembled.graspo_metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append("graspo_adapter_metadata.json")
        return tuple(written)

    @staticmethod
    def _write_merged_hf(
        destination: Path, assembled: MergedHfAssembly, plan: ShardPlan
    ) -> tuple[str, ...]:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        base = Path(str(plan.base_model_path))
        assembled.header_reader = safe_open
        written = _copy_hf_sidecar_files(base, destination)

        index_path = base / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = dict(index["weight_map"])
            filenames = sorted(set(weight_map.values()))
        else:
            filenames = sorted(file.name for file in base.glob("*.safetensors"))
            if not filenames:
                raise ExportError(f"No safetensors files found in the base model directory {base}")
            weight_map = {}
            for filename in filenames:
                for key in load_file(str(base / filename), device="cpu").keys():
                    weight_map[key] = filename

        total_size = 0
        for filename in filenames:
            state = load_file(str(base / filename), device="cpu")
            updated: dict[str, Any] = {}
            for name, tensor in state.items():
                merged = tensor
                if name in assembled.deltas:
                    merged = (
                        _apply_deltas(tensor, assembled.deltas[name])
                        .to(dtype=tensor.dtype)
                        .contiguous()
                    )
                updated[name] = merged
                total_size += int(updated[name].numel() * updated[name].element_size())
            save_file(updated, str(destination / filename))
            written.append(filename)

        if index_path.exists():
            index["metadata"] = dict(index.get("metadata") or {})
            index["metadata"]["total_size"] = total_size
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append("model.safetensors.index.json")
        return tuple(written)

    # ── 回读自校验（V14）──────────────────────────────────────────────
    def _verify_produced_state(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ) -> int:
        if isinstance(assembled, PeftAdapterAssembly):
            state = assembled.reader(
                str(destination / "adapter_model.safetensors"), device="cpu"
            )
            if set(state) != set(assembled.tensors):
                missing = sorted(set(assembled.tensors) - set(state))[:3]
                extra = sorted(set(state) - set(assembled.tensors))[:3]
                raise ExportError(
                    f"peft-adapter self-check failed: produced key set differs from the assembled "
                    f"set (missing={missing}, extra={extra})"
                )
            bad = verify.non_finite_keys(state)
            if bad:
                raise ExportError(
                    f"peft-adapter self-check failed: non-finite tensors in {bad[:3]}"
                )
            return len(state)

        if isinstance(assembled, MergedHfAssembly):
            for name, items in assembled.deltas.items():
                for _record, delta in items:
                    if not bool(delta.isfinite().all()):
                        raise ExportError(
                            f"merged-hf self-check failed: non-finite LoRA delta for {name}"
                        )
            base_keys = _safetensors_keys(Path(str(plan.base_model_path)), assembled.header_reader)
            produced_keys = _safetensors_keys(destination, assembled.header_reader)
            if base_keys != produced_keys:
                missing = sorted(base_keys - produced_keys)[:3]
                extra = sorted(produced_keys - base_keys)[:3]
                raise ExportError(
                    "merged-hf self-check failed: produced key set differs from the base model "
                    f"(missing={missing}, extra={extra}) — refusing to leave a half-merged product"
                )
            orphan = sorted(set(assembled.deltas) - base_keys)[:3]
            if orphan:
                raise ExportError(
                    "merged-hf self-check failed: LoRA deltas target weights that the base model "
                    f"does not contain (orphan={orphan}) — the base does not match this "
                    "checkpoint; refusing to merge"
                )
            return len(produced_keys)

        raise ExportError(f"unknown assembly type for self-check: {type(assembled).__name__}")


# ══════════════════════════════════════════════════════════════════════════════
# 以下函数自 lora_io.py 导出子集**语义等价地**搬入（除 V5/V7 边界外逐字保留）
# ══════════════════════════════════════════════════════════════════════════════


def _dedupe_dp_replicas(loaded: LoadedShards, plan: ShardPlan) -> list[dict[str, Any]]:
    """把已通过 V7 的 DP 副本化简为每个 ``(tp,pp)`` 一份代表。

    ⛔ 这不是"静默取副本"：只有当 :meth:`Exporter._check_replica_uniqueness` 已用
    ``Tensor.equal``（位模式等价，语义边界见 ``verify.states_identical``）证明同一
    ``(tp,pp)`` 的所有 DP 副本一致（且证据写入产物清单）时才到达这里；
    任何分歧都会在到达本函数之前 fail-closed。取代表是等价化简，不是选择。
    """
    topology = plan.topology
    if topology is None or topology.dp_size <= 1:
        return list(loaded.payloads)
    representatives: dict[tuple[int, int], dict[str, Any]] = {}
    for ref, payload in zip(loaded.shard_refs, loaded.payloads, strict=True):
        representatives.setdefault((ref.tp_rank, ref.pp_rank), payload)
    expected = topology.tp_size * topology.pp_size
    if len(representatives) != expected:
        raise ExportError(
            f"expected {expected} (tp,pp) representative(s) after replica reduction, found "
            f"{len(representatives)} — refusing to merge"
        )
    return [representatives[position] for position in sorted(representatives)]


def _safetensors_keys(root: Path, safe_open: Any) -> set[str]:
    keys: set[str] = set()
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            keys |= set(handle.keys())
    return keys


def _build_adapter_config(
    payloads: list[dict[str, Any]],
    module_specs: dict[str, dict[str, int]],
    base_model_path: str | None,
) -> dict[str, Any]:
    config = dict(payloads[0].get("config") or {})
    lora_config = dict(config.get("lora", {}) or {})
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
        "base_model_name_or_path": str(base_model_path or ""),
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
    return adapter_config


def _reconstruct_peft_export(
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], dict[str, Any]]:
    grouped = _records_with_tensors(payloads)
    output: dict[str, Any] = {}
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
    items: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    target_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for record, pair in items:
        key = str(record["target_name"])
        target_groups.setdefault(key, []).append((record, pair))

    rows = _row_count(items)
    cols = _column_count(items)
    target_pairs: list[tuple[dict[str, Any], Any, Any, float]] = []
    for target_name, target_items in sorted(target_groups.items()):
        shard_kinds = {str(record["shard_kind"]) for record, _ in target_items}
        if not shard_kinds <= {"out", "rows", "none"}:
            raise ExportError(
                f"Cannot export mixed/column-sharded fused LoRA target {target_name} "
                f"for {hf_module} as a PEFT adapter"
            )
        a = _require_replicated(
            hf_module,
            [pair["A"] for _, pair in target_items],
            f"{target_name}.lora_A",
        )
        if int(a.shape[1]) != cols:
            raise ExportError(
                f"Native LoRA A for {target_name} has {a.shape[1]} columns, "
                f"but {hf_module} expects {cols}"
            )
        b = target_items[0][1]["B"].new_zeros((rows, int(a.shape[0])))
        for record, pair in target_items:
            if str(record["shard_kind"]) == "none":
                if tuple(pair["B"].shape) != tuple(b.shape):
                    raise ExportError(
                        f"Native LoRA B for {target_name} cannot be embedded into {hf_module}: "
                        f"{tuple(pair['B'].shape)} != {tuple(b.shape)}"
                    )
                b += pair["B"]
            else:
                _place_rows(b, b.new_zeros((rows,)).bool(), pair["B"], record)
        representative = target_items[0][0]
        native_scale = float(representative["alpha"]) / float(representative["r"])
        target_pairs.append((representative, a.cpu(), b.cpu() * native_scale, native_scale))

    total_rank = sum(int(a.shape[0]) for _, a, _, _ in target_pairs)
    if total_rank <= 0:
        raise ExportError(f"Native checkpoint contains no enabled LoRA tensors for {hf_module}")
    combined_a = target_pairs[0][1].new_zeros((total_rank, cols))
    combined_b = target_pairs[0][2].new_zeros((rows, total_rank))
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
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
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
    items: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[Any, Any]:
    shard_kinds = {str(record["shard_kind"]) for record, _ in items}
    if shard_kinds == {"none"}:
        return items[0][1]["A"].cpu(), items[0][1]["B"].cpu()
    if shard_kinds <= {"out", "rows"}:
        a = _require_replicated(hf_module, [pair["A"] for _, pair in items], "lora_A")
        rows = _row_count(items)
        b = items[0][1]["B"].new_zeros((rows, int(a.shape[0])))
        filled = b.new_zeros((rows,)).bool()
        for record, pair in items:
            _place_rows(b, filled, pair["B"], record)
        _require_all_filled(hf_module, filled)
        return a.cpu(), b.cpu()
    if shard_kinds <= {"in", "cols"}:
        b = _require_replicated(hf_module, [pair["B"] for _, pair in items], "lora_B")
        cols = max(int(record["col_stop"]) for record, _ in items)
        a = items[0][1]["A"].new_zeros((int(b.shape[1]), cols))
        filled = a.new_zeros((cols,)).bool()
        for record, pair in items:
            start = int(record["col_start"])
            stop = int(record["col_stop"])
            a[:, start:stop] = pair["A"]
            filled[start:stop] = True
        _require_all_filled(hf_module, filled)
        return a.cpu(), b.cpu()
    raise ExportError(f"Cannot reconstruct mixed LoRA shard kinds for {hf_module}: {shard_kinds}")


def _require_replicated(
    hf_module: str,
    tensors: list[Any],
    label: str,
) -> Any:
    first = tensors[0].cpu()
    for tensor in tensors[1:]:
        if tuple(tensor.shape) != tuple(first.shape) or not bool(
            tensor.cpu().isclose(first, rtol=1e-5, atol=1e-6).all()
        ):
            raise ExportError(
                f"Native TP shards for {hf_module} have non-identical replicated {label}; "
                "strict PEFT adapter export is not representable. "
                "Set export.export_format: merged-hf in your config instead."
            )
    return first


def _row_count(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
    max_row = 0
    for record, _ in items:
        indices = record.get("row_indices")
        if indices is not None:
            max_row = max(max_row, max(int(idx) for idx in indices) + 1)
        else:
            max_row = max(max_row, int(record["row_stop"]))
    return max_row


def _column_count(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
    max_col = 0
    for record, pair in items:
        kind = str(record["shard_kind"])
        if kind in {"in", "cols"}:
            max_col = max(max_col, int(record["col_stop"]))
        else:
            max_col = max(max_col, int(pair["A"].shape[1]))
    return max_col


def _place_rows(output: Any, filled: Any, rows: Any, record: dict[str, Any]) -> None:
    indices = record.get("row_indices")
    if indices is not None:
        # 用张量自身构造索引（等价于 ``torch.tensor(indices, dtype=torch.long)``），
        # 从而本函数不需 import torch（C3：重依赖只在 _load/_write）。
        index = output.new_tensor(indices, dtype=output.long().dtype)
        output.index_copy_(0, index, rows)
        filled[index] = True
        return
    start = int(record["row_start"])
    stop = int(record["row_stop"])
    output[start:stop] = rows
    filled[start:stop] = True


def _require_all_filled(hf_module: str, filled: Any) -> None:
    if not bool(filled.all()):
        raise ExportError(f"Native TP shards for {hf_module} do not cover the full tensor")


def _collect_weight_deltas(
    payloads: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], Any]]]:
    grouped: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    for items in _records_with_tensors(payloads).values():
        for record, pair in items:
            delta = pair["B"].float().matmul(pair["A"].float())
            delta = delta * (float(record["alpha"]) / float(record["r"]))
            grouped.setdefault(str(record["base_weight_name"]), []).append((record, delta.cpu()))
    return grouped


def _merge_axis(
    *,
    kind: str,
    base_name: Any,
    axis: int,
    base_dim: int,
    start: int,
    stop: int,
    delta_dim: int,
    merged_shape: tuple[int, ...],
    delta_shape: tuple[int, ...],
) -> None:
    """★ 合并前**显式校验切片**（B12）——防"基座与 delta 不同源"被静默错并。

    逐字保留自 ``lora_io.py``（该函数记录了 2026-10-12 T022 27B 的真实事故）。
    """
    if stop > base_dim:
        raise ExportError(
            f"LoRA shard slice out of range while merging {base_name}: "
            f"kind={kind} axis={axis} slice=[{start}:{stop}] but base tensor "
            f"{merged_shape} has {base_dim} on that axis; delta is {delta_shape}. "
            "The base model does not match the checkpoint that produced this adapter "
            "(different model/size?) — refusing to merge."
        )
    if stop - start != delta_dim:
        raise ExportError(
            f"LoRA shard slice width does not match the delta while merging {base_name}: "
            f"kind={kind} axis={axis} slice=[{start}:{stop}] width {stop - start} "
            f"but delta {delta_shape} has {delta_dim} on that axis; base tensor is "
            f"{merged_shape}. The base model does not match the adapter — refusing to merge."
        )


def _apply_deltas(
    tensor: Any,
    deltas: list[tuple[dict[str, Any], Any]],
) -> Any:
    merged = tensor.float().clone()
    for record, delta in deltas:
        kind = str(record["shard_kind"])
        base_name = record.get("base_weight_name")
        if kind in {"none"}:
            if tuple(delta.shape) != tuple(merged.shape):
                raise ExportError(
                    f"LoRA delta shape {tuple(delta.shape)} does not match "
                    f"{record['base_weight_name']} {tuple(merged.shape)}"
                )
            merged += delta
        elif kind == "out":
            start, stop = int(record["row_start"]), int(record["row_stop"])
            _merge_axis(
                kind=kind, base_name=base_name, axis=0, base_dim=int(merged.shape[0]),
                start=start, stop=stop, delta_dim=int(delta.shape[0]),
                merged_shape=tuple(merged.shape), delta_shape=tuple(delta.shape),
            )
            merged[start:stop] += delta
        elif kind == "rows":
            indices = record.get("row_indices")
            if indices is not None:
                index_tensor = merged.new_tensor(indices, dtype=merged.long().dtype)
                if int(index_tensor.numel()) != int(delta.shape[0]):
                    raise ExportError(
                        f"LoRA row_indices count {int(index_tensor.numel())} does not match "
                        f"delta {tuple(delta.shape)} while merging {base_name} — refusing to merge."
                    )
                if index_tensor.numel() and int(index_tensor.max()) >= int(merged.shape[0]):
                    raise ExportError(
                        f"LoRA row_indices max {int(index_tensor.max())} exceeds base tensor "
                        f"{tuple(merged.shape)} while merging {base_name}; the base model does "
                        "not match the adapter — refusing to merge."
                    )
                merged.index_add_(0, index_tensor, delta)
            else:
                start, stop = int(record["row_start"]), int(record["row_stop"])
                _merge_axis(
                    kind=kind, base_name=base_name, axis=0, base_dim=int(merged.shape[0]),
                    start=start, stop=stop, delta_dim=int(delta.shape[0]),
                    merged_shape=tuple(merged.shape), delta_shape=tuple(delta.shape),
                )
                merged[start:stop] += delta
        elif kind in {"in", "cols"}:
            start, stop = int(record["col_start"]), int(record["col_stop"])
            _merge_axis(
                kind=kind, base_name=base_name, axis=1, base_dim=int(merged.shape[1]),
                start=start, stop=stop, delta_dim=int(delta.shape[1]),
                merged_shape=tuple(merged.shape), delta_shape=tuple(delta.shape),
            )
            merged[:, start:stop] += delta
        else:
            raise ExportError(f"Unsupported LoRA shard kind for merge: {kind}")
    return merged


def _copy_hf_sidecar_files(base: Path, output: Path) -> list[str]:
    """复制基座的旁路文件（配置/分词器等），跳过权重与索引。返回写出的相对名。"""
    written: list[str] = []
    skip_suffixes = {".safetensors"}
    skip_names = {"model.safetensors.index.json"}
    for item in base.iterdir():
        if item.is_dir():
            if item.name in {".git", "__pycache__"}:
                continue
            shutil.copytree(item, output / item.name, dirs_exist_ok=True)
            written.append(item.name)
            continue
        if item.name in skip_names or item.suffix in skip_suffixes:
            continue
        shutil.copy2(item, output / item.name)
        written.append(item.name)
    return written


__all__ = [
    "MergedHfAssembly",
    "NativeLoraRankExporter",
    "PeftAdapterAssembly",
]
