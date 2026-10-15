"""native **全参** rank 分片（``manifest.json`` + ``rank_*.pt``）→ ``merged-hf``（F7，P5）。

**形态**：``format=native-full-param`` 的 native checkpoint。本阶段的**实现范围**是
**单卡单分片自足**（``dp=1 ∧ tp=1 ∧ pp=1``，``local_layer_indices`` 覆盖全层、
``include_embeddings``/``include_lm_head`` 均为真）；``dp/tp/pp > 1`` 继续 **fail-closed**
（PP 拼装属阶段 2，⛔ 不在此提前实现）。

**它防的是什么**：全参 checkpoint 的 ``lora_state_dict`` 是**空字典**（全参模式不建 LoRA
矩阵），权重在 ``full_param_state_dict`` 里。若把它塞进 LoRA 通道，会算出零个 delta、
把**未训练的基座**原样写成"导出成功"的产物——这是最危险的静默失败（见
:mod:`graspo.export.semantics`）。本模块走的是**整张权重替换**通道，并在落盘后做
"产物 == ckpt 值"的逐元素自校验，⛔ 不允许用基座值兜底。

**native → HF 命名映射的唯一权威来源**（⛔ 不猜、不手抄成第二份并行表）：

- 前缀：native 参数名没有前缀；HF 侧文本塔是 ``model.language_model.``（**两层**）、
  视觉塔 ``model.visual.``、``lm_head.weight`` 在**顶层**。
- ``layers.N.token_mixer`` → ``layers.N.linear_attn``（线attention层）或 ``layers.N.self_attn``
  （full-attention 层）。**按 HF 侧键名分派**即可（HF 键里已写明 ``linear_attn`` /
  ``self_attn``），因此本模块**不需要** ``layer_types`` 表。
- ``token_mixer.conv1d_weight`` ↔ ``linear_attn.conv1d.weight``（改名，``tp=1`` 时恒等置换）。
- **融合**：``linear_attn.in_proj_qkv.weight = cat([q_proj, k_proj, v_proj], dim=0)``
  —— 由 native **载入侧**的逐行切片契约定义（生产者/消费者同一份代码）：
  ``flow/backend/native/models/common/layers.py`` 的 ``TensorParallelQwen35LinearAttention``
  用 ``qkv_weight[0:key_dim]`` / ``[key_dim:2*key_dim]`` / ``[2*key_dim : 2*key_dim+value_dim]``
  切给 q/k/v。本模块取其**逐行逆变换**（拼接），行序因此逐位可证。
  ★ **顺序错误名称覆盖查不出**（目标名存在、形状对、"产物 == ckpt" 也全过）⇒ 另做一条
  **独立判据**：每个 native 块必须**最近邻**落在基座 ``in_proj_qkv`` 的对应行段上
  （见 :func:`fused_block_assignment`），顺序写反即 fail-closed。
- ``mtp.*``（HF 侧 MTP 头）：native 模型不建、ckpt 里没有 ⇒ **显式从基座逐张量复制**
  并逐项记入产物清单（⛔ 不静默丢弃、⛔ 不用随机值）。

**输出布局**：沿用基座的 safetensors 分片 + ``model.safetensors.index.json`` 布局与
sidecar 文件（复用 ``native_lora._copy_hf_sidecar_files``，⛔ 不写第二份落盘逻辑）。

**dtype**：产物 dtype = **基座对应键的 dtype**。本 run 里唯一需要回落的是 48 个
``A_log`` / ``linear_attn.norm.weight``（基座 F32、ckpt bf16）；回落项逐项记入清单。
其余 native dtype 与基座不一致（或非"F32←BF16"的已知回落）⇒ fail-closed。

**不负责**：不碰 PEFT 目录；不做 PP/TP/DP 拼装（>1 一律拒绝）；不读 LoRA 语义。

⚠ **已登记的元数据缺口**（H9 结构说明 §5.2 建议①）：native 侧 ``LoRALinear`` 运行时带
``hf_module_path`` / ``base_weight_name`` / ``shard_kind`` 等自描述元数据，但它们
**不出现在 full-param ckpt 里**（``lora_tensor_metadata = []``、``lora_state_dict = {}``）。
要复用它们必须实例化 native 模型 ⇒ ``graspo.export`` 依赖 ``graspo.flow``，
撞设计 T30（``tests/export/test_boundaries.py`` 机检）。因此本模块按 §5.1 的
(A)+(B)+(B-fuse) 规则实现**唯一一份**导出侧映射，并用"满覆盖 + 孤儿键 fail-closed"
机检它（不是猜测）。
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from graspo.export.base import Exporter, LoadedShards
from graspo.export.errors import ExportError
from graspo.export.exporters.native_lora import _copy_hf_sidecar_files
from graspo.export.schema import (
    CheckpointClassification,
    CheckpointForm,
    ExportTarget,
    NativeTopology,
    PlacementRecord,
    ShardPlan,
    ShardRef,
)

_FORMAT_FULL_PARAM = "native-full-param"
_MANIFEST_NAME = "manifest.json"

#: HF 文本塔前缀（**两层**）与视觉塔前缀（H9 结构说明 §5.1(A)）。
_LM_PREFIX = "model.language_model"
_VIS_PREFIX = "model.visual"

#: HF 侧顶层、native 侧没有、且**显式声明为"从基座复制"**的键前缀（MTP 头）。
_BASE_COPY_PREFIXES = ("mtp.",)

#: safetensors dtype 字符串 → torch dtype 名（仅用于比对，不 import torch）。
_SAFETENSORS_DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}

#: 允许的 dtype 回落：基座 F32 ← ckpt bf16（native 前向本身 ``A_log.float()``）。
_ALLOWED_DTYPE_FALLBACKS = {("float32", "bfloat16")}


# ══════════════════════════════════════════════════════════════════════════════
# native → HF 命名映射（纯函数：无 IO、无 torch）
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """一个 HF 键的来源：native 键列表（空 ⇒ 显式从基座复制）与是否融合。"""

    native_keys: tuple[str, ...] = ()
    fused: bool = False
    from_base: bool = False


#: connv1d 与 linear_attn 内部改名表（其余 ``linear_attn.*`` 逐字同名）。
_LINEAR_ATTN_RENAMES = {
    "conv1d.weight": "conv1d_weight",
}


def hf_source_plan(hf_key: str) -> SourcePlan | None:
    """HF state_dict 键 → 它的 native 来源（``None`` ⇒ **未登记**，调用方必须 fail-closed）。

    规则与权威来源见模块 docstring。本函数**只做命名变换**，不做形状/存在性判断
    （那些在 :meth:`NativeFullParamRankExporter._assemble` 里按真实张量做）。
    """
    if hf_key.startswith(_BASE_COPY_PREFIXES):
        return SourcePlan(from_base=True)

    if hf_key == "lm_head.weight":
        return SourcePlan(("lm_head.weight",))

    if hf_key.startswith(_VIS_PREFIX + "."):
        return SourcePlan(("visual." + hf_key[len(_VIS_PREFIX) + 1 :],))

    if not hf_key.startswith(_LM_PREFIX + "."):
        return None

    sub = hf_key[len(_LM_PREFIX) + 1 :]
    if sub in ("embed_tokens.weight", "norm.weight"):
        return SourcePlan((sub,))

    if not sub.startswith("layers."):
        return None

    prefix, index_text, rest = sub.split(".", 2)
    del prefix  # 恒为 "layers"
    native_base = f"layers.{index_text}.token_mixer."

    if rest == "linear_attn.in_proj_qkv.weight":
        # ★ 融合：载入侧切片 [q | k | v]，导出取逐行逆变换 cat([q, k, v], dim=0)。
        return SourcePlan(
            (
                native_base + "q_proj.weight",
                native_base + "k_proj.weight",
                native_base + "v_proj.weight",
            ),
            fused=True,
        )

    if rest.startswith("linear_attn."):
        tail = rest[len("linear_attn.") :]
        return SourcePlan((native_base + _LINEAR_ATTN_RENAMES.get(tail, tail),))

    if rest.startswith("self_attn."):
        tail = rest[len("self_attn.") :]
        return SourcePlan((native_base + tail,))

    # input_layernorm / post_attention_layernorm / mlp.{gate,up,down}_proj：同名
    return SourcePlan((f"layers.{index_text}." + rest,))


def relative_l1(tensor: Any, reference: Any) -> float:
    """相对 L1 偏差 ``|t - r|_1 / |r|_1``（纯张量方法，本函数不 import torch）。"""
    diff = float((tensor.float() - reference.float()).abs().sum())
    denom = float(reference.float().abs().sum())
    return diff / denom if denom > 0.0 else diff


def fused_block_assignment(
    base_weight: Any, native_blocks: list[Any]
) -> tuple[list[int], list[list[float | None]]]:
    """**融合顺序**的逐块最近邻判别（⛔ 名称覆盖查不出顺序错误）。

    前提（已实测、写入报告）：native 的 q/k/v 三块本来就是从基座 ``in_proj_qkv`` 的
    对应行段切出来再训练的 ⇒ 每块与**自己那段**的相对 L1 偏差落在"训练漂移"量级，
    而与其它段相差一个信号量级。

    只把**行数相同**的段纳入候选（不同行数的张量无法逐元素相减；``None`` 表示不可比）。
    因此它专门覆盖"行数相同的块被换序"（真实 H9 里 q/k 同为 2048 行、v 为 4096 行）。

    Returns:
        ``(assignment, deviations)`` —— ``assignment[i]`` = 第 i 个 native 块最接近的
        基座行段下标；``deviations[i][j]`` = 第 i 个块对第 j 段的相对 L1 偏差（``None`` 不可比）。
    """
    sizes = [int(block.shape[0]) for block in native_blocks]
    rows = int(base_weight.shape[0])
    if sum(sizes) != rows:
        raise ExportError(
            f"fused block sizes {sizes} sum to {sum(sizes)} but the base fused weight has "
            f"{rows} rows — the q/k/v split does not match the base layout"
        )
    segments: list[Any] = []
    offset = 0
    for size in sizes:
        segments.append(base_weight[offset : offset + size])
        offset += size
    deviations: list[list[float | None]] = [
        [
            (relative_l1(block, segment) if int(block.shape[0]) == int(segment.shape[0]) else None)
            for segment in segments
        ]
        for block in native_blocks
    ]
    assignment = [
        min((index for index, value in enumerate(row) if value is not None), key=row.__getitem__)
        for row in deviations
    ]
    return assignment, deviations


def fused_permutation_deviations(
    base_weight: Any, native_blocks: list[Any]
) -> dict[tuple[int, ...], float]:
    """**全体排列**判据：声明顺序必须是**偏差最小**的那个排列。

    比逐块最近邻更强 —— 它覆盖"跨不同行数的段被换序"（例如 ``[q, v, k]`` 这种把 v
    插到中间的错序），因为任何排列的拼接总行数都等于基座行数，故偏差始终可算。

    返回 ``{排列: 相对 L1 偏差}``（分母 = 基座整张的 L1）。
    """
    sizes = [int(block.shape[0]) for block in native_blocks]
    rows = int(base_weight.shape[0])
    if sum(sizes) != rows:
        raise ExportError(
            f"fused block sizes {sizes} sum to {sum(sizes)} but the base fused weight has "
            f"{rows} rows — the q/k/v split does not match the base layout"
        )
    denom = float(base_weight.float().abs().sum())
    results: dict[tuple[int, ...], float] = {}
    for permutation in itertools.permutations(range(len(native_blocks))):
        offset = 0
        numerator = 0.0
        for index in permutation:
            size = sizes[index]
            reference = base_weight[offset : offset + size]
            numerator += float(
                (native_blocks[index].float() - reference.float()).abs().sum()
            )
            offset += size
        results[permutation] = numerator / denom if denom > 0.0 else numerator
    return results


def match_native_full_param(root: Path) -> tuple[bool, tuple[str, ...]]:
    """检测器：``manifest.json`` 声明 ``native-full-param`` 且存在 ``rank_*.pt``（纯 FS）。"""
    if not (root / _MANIFEST_NAME).is_file():
        return False, ()
    shards = sorted(root.glob("rank_*.pt"))
    if not shards:
        return False, ()
    try:
        data = _read_manifest(root)
    except ExportError:
        return False, ()
    if str(data.get("format") or "") != _FORMAT_FULL_PARAM:
        return False, ()
    return True, (
        f"found manifest.json format={_FORMAT_FULL_PARAM}",
        f"found {len(shards)} .pt rank shard(s)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# manifest / 拓扑 / 基座布局（纯 JSON + safetensors 头部，无权重加载）
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
    if fmt != _FORMAT_FULL_PARAM:
        raise ExportError(
            f"{where}: manifest format={fmt!r} does not match native full-param form "
            f"{CheckpointForm.NATIVE_FULL_PARAM_RANK.value!r} (V4); refusing to continue"
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


def _read_base_layout(
    base: Path,
) -> tuple[dict[str, str], list[str], dict[str, Any] | None, dict[str, Any]]:
    """读基座 safetensors 的**头部**（键/形状/dtype），不加载权重。

    Returns:
        ``(weight_map, files, index, meta)``，其中
        ``meta[key] = {"file": <分片名>, "dtype": <torch dtype 名>, "shape": (…) }``。
    """
    from safetensors import safe_open

    index_path = base / "model.safetensors.index.json"
    index: dict[str, Any] | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = dict(index["weight_map"])
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(path.name for path in base.glob("*.safetensors"))
        if not files:
            raise ExportError(
                f"no safetensors weights found in base model directory {base}; "
                "cannot build a merged-hf product (V10 requires an explicit base)"
            )
        weight_map = {}

    meta: dict[str, Any] = {}
    for filename in files:
        with safe_open(str(base / filename), framework="pt") as handle:
            for key in handle.keys():
                if index is None:
                    weight_map[key] = filename
                slice_ = handle.get_slice(key)
                raw_dtype = str(slice_.get_dtype())
                meta[key] = {
                    "file": filename,
                    "dtype": _SAFETENSORS_DTYPE_NAMES.get(raw_dtype, raw_dtype.lower()),
                    "shape": tuple(int(dim) for dim in slice_.get_shape()),
                }
    missing = sorted(set(weight_map) - set(meta))
    if missing:
        raise ExportError(
            f"base model index refers to keys absent from its safetensors shards: {missing[:5]}"
        )
    return weight_map, files, index, meta


# ══════════════════════════════════════════════════════════════════════════════
# 加载 / 组装中间产物
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class FullParamLoaded(LoadedShards):
    """:class:`LoadedShards` + 基座布局（键→分片/dtype/shape），供 ``_assemble`` 用。"""

    base_weight_map: dict[str, str] = field(default_factory=dict)
    base_files: list[str] = field(default_factory=list)
    base_index: dict[str, Any] | None = None
    base_meta: dict[str, Any] = field(default_factory=dict)
    #: 融合键的基座整张权重（仅这些键会被读入），供**顺序/行段**独立判别。
    fused_base_rows: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MergedHfAssembly:
    """待落盘的 merged-hf：每个 HF 键的来源计划 + native 权重引用。"""

    sources: dict[str, SourcePlan]
    native_state: dict[str, Any]
    base_weight_map: dict[str, str]
    base_files: list[str]
    base_index: dict[str, Any] | None
    base_meta: dict[str, Any]
    fusions: dict[str, list[str]]
    copied_from_base: list[str]
    dtype_casts: dict[str, str]
    #: 回读器（由 ``_write`` 注入 ``safetensors.safe_open``），供自校验回读产物用。
    reader: Any = None


class NativeFullParamRankExporter(Exporter):
    """native 全参 ``.pt`` 分片 → ``merged-hf``（单卡单分片，P5）。"""

    form = CheckpointForm.NATIVE_FULL_PARAM_RANK
    supported_targets = frozenset({ExportTarget.MERGED_HF})

    # ── 形态检测（纯 FS）──────────────────────────────────────────────
    @classmethod
    def _matches(cls, root: Path) -> tuple[bool, tuple[str, ...]]:
        return match_native_full_param(root)

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
        # V10：基座路径**唯一来源** = 调用方传入的 config.model.model_path。
        if not plan.base_model_path:
            raise ExportError(
                f"native export to {plan.target.value!r} requires an explicit base model path "
                "(config.model.model_path); refusing to fall back to the checkpoint-embedded "
                "model_path (V10 single truth source)"
            )
        topology = plan.topology
        assert topology is not None  # native 形态必有拓扑
        if topology.dp_size > 1 or topology.tp_size > 1 or topology.pp_size > 1:
            raise ExportError(
                f"native full-param checkpoint {plan.checkpoint_name} declares "
                f"dp={topology.dp_size}, tp={topology.tp_size}, pp={topology.pp_size}; this "
                "exporter supports only a single self-contained shard (dp=1 ∧ tp=1 ∧ pp=1). "
                "DP replica reconciliation / TP row reassembly / PP layer assembly are NOT "
                "implemented here and MUST stay fail-closed (no silent replica or partial-shard "
                "export) — they are a separate stage. Refusing to export."
            )

    # ── 加载（唯一允许 import torch 的钩子之一）──────────────────────────
    def _load(self, plan: ShardPlan) -> FullParamLoaded:
        import torch

        root = Path(plan.source_dir)
        shard_paths = sorted(root.glob("rank_*.pt"))
        if len(shard_paths) != 1:
            raise ExportError(
                f"native full-param exporter expects exactly one self-contained shard "
                f"(dp=tp=pp=1), found {len(shard_paths)} in {root} — refusing to guess which "
                "shard to read"
            )

        path = shard_paths[0]
        try:
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:  # 旧版 torch 不支持 mmap/weights_only
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ExportError(f"Checkpoint shard {path} did not load into a mapping")

        state = payload.get("full_param_state_dict")
        if not isinstance(state, dict) or not state:
            raise ExportError(
                f"Checkpoint shard {path} has no non-empty full_param_state_dict; this exporter "
                "handles native full-parameter checkpoints only (LoRA checkpoints have an empty "
                "full_param_state_dict and must go through the native LoRA exporter)"
            )
        for name in ("dp_rank", "tp_rank", "pp_rank"):
            value = int(payload.get(name, -1))
            if value != 0:
                raise ExportError(
                    f"Checkpoint shard {path} declares {name}={value}; the full-param exporter "
                    "requires a single shard at rank 0 (topology truth source = payload fields)"
                )
        if str(payload.get("tuner_type") or "") != "full":
            raise ExportError(
                f"Checkpoint shard {path} declares tuner_type="
                f"{payload.get('tuner_type')!r}; expected 'full' (V4 format/semantics gate)"
            )

        ref = ShardRef(
            name=path.name,
            dp_rank=int(payload.get("dp_rank", 0)),
            tp_rank=int(payload.get("tp_rank", 0)),
            pp_rank=int(payload.get("pp_rank", 0)),
            sha256=_file_sha256(path),
        )

        base = Path(str(plan.base_model_path))
        weight_map, files, index, meta = _read_base_layout(base)

        # 融合键的基座整张权重：**顺序/行段独立判别**的唯一参照（名称覆盖查不出顺序错）。
        from safetensors import safe_open

        fused_base_rows: dict[str, Any] = {}
        for hf_key, base_info in meta.items():
            source = hf_source_plan(hf_key)
            if source is not None and source.fused:
                with safe_open(str(base / base_info["file"]), framework="pt") as handle:
                    fused_base_rows[hf_key] = handle.get_tensor(hf_key)
        return FullParamLoaded(
            payloads=[payload],
            shard_refs=[ref],
            paths=[path],
            base_weight_map=weight_map,
            base_files=files,
            base_index=index,
            base_meta=meta,
            fused_base_rows=fused_base_rows,
        )

    # ── 组装（纯张量变换 + 名称映射校验）──────────────────────────────
    def _assemble(self, loaded: FullParamLoaded, plan: ShardPlan) -> MergedHfAssembly:
        native_state = dict(loaded.payloads[0]["full_param_state_dict"])
        sources: dict[str, SourcePlan] = {}
        fusions: dict[str, list[str]] = {}
        copied_from_base: list[str] = []
        dtype_casts: dict[str, str] = {}
        consumed: set[str] = set()

        for hf_key, base_info in loaded.base_meta.items():
            source = hf_source_plan(hf_key)
            if source is None:
                raise ExportError(
                    f"no native→HF mapping is registered for base key {hf_key!r} "
                    f"(shape={tuple(base_info['shape'])}, dtype={base_info['dtype']}); "
                    "⛔ refusing to emit a product that silently drops or guesses a weight"
                )
            if source.from_base:
                copied_from_base.append(hf_key)
                sources[hf_key] = source
                continue

            missing = [key for key in source.native_keys if key not in native_state]
            if missing:
                raise ExportError(
                    f"checkpoint {plan.checkpoint_name} is missing native parameter(s) {missing} "
                    f"required by base key {hf_key!r}; the shard is not self-contained "
                    "(refusing to leave that weight untrained/base-valued)"
                )
            self._check_shape_and_dtype(
                hf_key, base_info, source, native_state, dtype_casts
            )
            consumed.update(source.native_keys)
            if source.fused:
                self._verify_fused_block_order(hf_key, source, native_state, loaded)
                fusions[hf_key] = list(source.native_keys)
            sources[hf_key] = source

        orphan = sorted(set(native_state) - consumed)
        if orphan:
            raise ExportError(
                f"checkpoint {plan.checkpoint_name} contains {len(orphan)} native parameter(s) "
                f"that map to no base key (first: {orphan[:5]}); the base model does not match "
                "this checkpoint — refusing to merge (no silent drop)"
            )

        return MergedHfAssembly(
            sources=sources,
            native_state=native_state,
            base_weight_map=dict(loaded.base_weight_map),
            base_files=list(loaded.base_files),
            base_index=loaded.base_index,
            base_meta=loaded.base_meta,
            fusions=fusions,
            copied_from_base=sorted(copied_from_base),
            dtype_casts=dtype_casts,
        )

    @staticmethod
    def _check_shape_and_dtype(
        hf_key: str,
        base_info: dict[str, Any],
        source: SourcePlan,
        native_state: dict[str, Any],
        dtype_casts: dict[str, str],
    ) -> None:
        base_shape = tuple(int(dim) for dim in base_info["shape"])
        native_shapes = [
            tuple(int(dim) for dim in native_state[key].shape) for key in source.native_keys
        ]
        if source.fused:
            if len({shape[1:] for shape in native_shapes}) != 1:
                raise ExportError(
                    f"fused source for {hf_key!r} has inconsistent trailing dims: {native_shapes}"
                )
            merged_shape = (sum(shape[0] for shape in native_shapes),) + native_shapes[0][1:]
        else:
            merged_shape = native_shapes[0]
        if merged_shape != base_shape:
            raise ExportError(
                f"shape mismatch for {hf_key!r}: base has {base_shape}, native source "
                f"{list(source.native_keys)} yields {merged_shape}; the base model does not "
                "match this checkpoint — refusing to merge"
            )

        base_dtype = str(base_info["dtype"])
        native_dtypes = {
            str(native_state[key].dtype).replace("torch.", "") for key in source.native_keys
        }
        if len(native_dtypes) != 1:
            raise ExportError(
                f"native source for {hf_key!r} has mixed dtypes {sorted(native_dtypes)}"
            )
        native_dtype = native_dtypes.pop()
        if native_dtype != base_dtype:
            if (base_dtype, native_dtype) in _ALLOWED_DTYPE_FALLBACKS:
                dtype_casts[hf_key] = f"{native_dtype}->{base_dtype}"
            else:
                raise ExportError(
                    f"dtype mismatch for {hf_key!r}: base is {base_dtype}, native is "
                    f"{native_dtype}; only the declared fallback "
                    f"{sorted(_ALLOWED_DTYPE_FALLBACKS)} is allowed (refusing to guess)"
                )

    @staticmethod
    def _verify_fused_block_order(
        hf_key: str,
        source: SourcePlan,
        native_state: dict[str, Any],
        loaded: FullParamLoaded,
    ) -> None:
        """融合的**顺序/行段**独立判据（⛔ 名称覆盖 + "产物==ckpt" 都查不出顺序错）。

        两条判据都必须通过：

        1. 每个 native 块**最近邻**落在基座 ``in_proj_qkv`` 的对应行段上
           （:func:`fused_block_assignment`）；
        2. **声明顺序的整张拼接偏差必须是所有排列里最小的**
           （:func:`fused_permutation_deviations`）——它额外覆盖"跨不同行数的段被换序"。

        顺序写反 ⇒ 当场 fail-closed，而不是产出一个"能加载、语义错"的权重。
        """
        base_weight = loaded.fused_base_rows.get(hf_key)
        if base_weight is None:
            raise ExportError(
                f"fused verification could not read the base weight for {hf_key!r}; refusing to "
                "fuse q/k/v without an independent order reference"
            )
        native_keys = list(source.native_keys)
        blocks = [native_state[key] for key in native_keys]
        expected = list(range(len(blocks)))

        assignment, deviations = fused_block_assignment(base_weight, blocks)
        permutation_deviations = fused_permutation_deviations(base_weight, blocks)
        declared = permutation_deviations[tuple(expected)]
        best_alternative = min(
            value
            for permutation, value in permutation_deviations.items()
            if permutation != tuple(expected)
        )
        matrix = ", ".join(
            f"{native_keys[index]}->{row}" for index, row in enumerate(deviations)
        )
        if assignment != expected or declared >= best_alternative:
            raise ExportError(
                f"fused block-order check FAILED for {hf_key!r}: nearest-segment assignment="
                f"{assignment} (expected {expected}); declared permutation deviation="
                f"{declared:.6g} vs best alternative={best_alternative:.6g} "
                "(declared MUST be the smallest). Relative-L1 matrix "
                f"[{matrix}]. The q/k/v split or its concatenation order does "
                "NOT match the base in_proj_qkv layout. Names, shapes and 'produced == checkpoint' "
                "all pass in this case, so this is the only check that can catch it — refusing to "
                "write a semantically wrong fused weight."
            )

    # ── 落盘（唯一允许写盘的钩子）──────────────────────────────────────
    def _write(self, destination: Path, assembled: Any, plan: ShardPlan) -> tuple[str, ...]:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        assert isinstance(assembled, MergedHfAssembly)
        base = Path(str(plan.base_model_path))
        assembled.reader = safe_open

        written = list(_copy_hf_sidecar_files(base, destination))
        total_size = 0
        for filename in assembled.base_files:
            base_state = load_file(str(base / filename), device="cpu")
            updated: dict[str, Any] = {}
            for key, base_tensor in base_state.items():
                source = assembled.sources[key]
                if source.from_base:
                    produced = base_tensor
                elif source.fused:
                    produced = torch.cat(
                        [
                            assembled.native_state[item].to(base_tensor.dtype)
                            for item in source.native_keys
                        ],
                        dim=0,
                    )
                else:
                    produced = assembled.native_state[source.native_keys[0]].to(base_tensor.dtype)
                updated[key] = produced.contiguous()
                total_size += int(updated[key].numel() * updated[key].element_size())
            save_file(updated, str(destination / filename))
            written.append(filename)

        if assembled.base_index is not None:
            index = json.loads(json.dumps(assembled.base_index))
            index["metadata"] = dict(index.get("metadata") or {})
            index["metadata"]["total_size"] = total_size
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            written.append("model.safetensors.index.json")
        return tuple(written)

    # ── 回读自校验（V14 + 本形态的"产物 == ckpt 值"强校验）────────────
    def _expected_tensor(
        self, assembled: MergedHfAssembly, key: str, target_dtype: Any, base_tensor: Any
    ) -> Any:
        import torch

        source = assembled.sources[key]
        if source.from_base:
            return base_tensor
        if source.fused:
            return torch.cat(
                [
                    assembled.native_state[item].to(target_dtype)
                    for item in source.native_keys
                ],
                dim=0,
            )
        return assembled.native_state[source.native_keys[0]].to(target_dtype)

    def _verify_produced_state(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ) -> int:
        from safetensors import safe_open
        from safetensors.torch import load_file

        assert isinstance(assembled, MergedHfAssembly)
        base = Path(str(plan.base_model_path))

        produced_keys: set[str] = set()
        for filename in assembled.base_files:
            with safe_open(str(destination / filename), framework="pt") as produced:
                produced_file_keys = set(produced.keys())
                base_state = load_file(str(base / filename), device="cpu")
                expected_file_keys = set(base_state)
                if produced_file_keys != expected_file_keys:
                    missing = sorted(expected_file_keys - produced_file_keys)[:3]
                    extra = sorted(produced_file_keys - expected_file_keys)[:3]
                    raise ExportError(
                        f"merged-hf self-check failed: {filename} key set differs from the base "
                        f"(missing={missing}, extra={extra})"
                    )
                for key in sorted(produced_file_keys):
                    tensor = produced.get_tensor(key)
                    expected = self._expected_tensor(
                        assembled, key, base_state[key].dtype, base_state[key]
                    )
                    # ① 逐元素一致：产物必须**逐位**等于 ckpt 张量（经声明过的融合/dtype 回落），
                    #    ⛔ 不是"近似相等"，也⛔不允许悄悄写基座值。
                    if tuple(tensor.shape) != tuple(expected.shape):
                        raise ExportError(
                            f"merged-hf self-check failed: {key} shape {tuple(tensor.shape)} != "
                            f"expected {tuple(expected.shape)}"
                        )
                    if not bool(tensor.equal(expected)):
                        raise ExportError(
                            f"merged-hf self-check failed: {key} is NOT element-wise equal to the "
                            "checkpoint value after the declared mapping. This is the exact "
                            "failure mode of silently exporting an untrained/base model; "
                            "refusing to leave a wrong product."
                        )
                    if not bool(tensor.isfinite().all()):
                        raise ExportError(
                            f"merged-hf self-check failed: non-finite values in {key}"
                        )
                    produced_keys.add(key)

        expected_keys = set(assembled.base_meta)
        if produced_keys != expected_keys:
            missing = sorted(expected_keys - produced_keys)[:3]
            extra = sorted(produced_keys - expected_keys)[:3]
            raise ExportError(
                "merged-hf self-check failed: produced key set does not cover the base "
                f"state_dict (missing={missing}, extra={extra})"
            )
        return len(produced_keys)

    def _verify_extra(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ) -> None:
        """V16：产物目录内不得残留 native ``.pt`` 分片。"""
        residue = sorted(path.name for path in destination.glob("*.pt"))
        if residue:
            raise ExportError(
                f"merged-hf self-check failed: native shard residue in {destination}: {residue}"
            )
        produced = set(written)
        for name in assembled.base_files:
            if name not in produced:
                raise ExportError(
                    f"merged-hf self-check failed: declared weight shard {name!r} was not written"
                )

    def _verify(
        self,
        destination: Path,
        plan: ShardPlan,
        assembled: Any,
        written: tuple[str, ...],
    ):
        manifest = super()._verify(destination, plan, assembled, written)
        return manifest.model_copy(
            update={
                "base_copied_keys": list(assembled.copied_from_base),
                "mapping_fusions": {key: list(value) for key, value in assembled.fusions.items()},
                "dtype_casts": dict(assembled.dtype_casts),
            }
        )


__all__ = [
    "MergedHfAssembly",
    "NativeFullParamRankExporter",
    "SourcePlan",
    "fused_block_assignment",
    "fused_permutation_deviations",
    "hf_source_plan",
    "match_native_full_param",
    "relative_l1",
]
