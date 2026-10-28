"""native 分片检查点（tp / pp / dp）→ 唯一完整 LoRA 的合并器。

**职责**：把一个 ``final/``（``manifest.json`` ＋ ``rank_*_tp_*_pp_*.pt``）里的
**分片** LoRA 张量，按分片自带的元数据**严格**重组成"整模型那一份" LoRA，
产出与 ``dp=tp=pp=1`` 通道**同格式**的 PEFT 键值字典（后续复用
``merged_export.merge_peft_checkpoint`` 合到基座 + 起 vLLM）。

**本文件不负责**：合到基座（``merged_export``）、起 vLLM、评测。

## 为什么必须"按元数据拼"，而不是"猜层归属"

分片脱水时每个 LoRA 模块都带 ``lora_tensor_metadata`` 记录（见
``flow/backend/native/lora/lora_linear.py::LoRALinear.lora_metadata``）：

===================  ==========================================  ==============================
``shard_kind``       本 rank 持有的 ``lora_a``                       本 rank 持有的 ``lora_b``
===================  ==========================================  ==============================
``none``             完整（复制）                                 完整（复制）
``out``              完整（复制）                                 ``[row_start, row_stop)`` 行
``rows``             完整（复制）                                 ``row_indices`` 指定的行（**非连续**）
``in``               ``[col_start, col_stop)`` 列                  完整（复制）
===================  ==========================================  ==============================

（这张表的权威来源是 ``lora_io.py::_slice_lora_a`` / ``_slice_lora_b`` —— 加载
PEFT adapter 进 native 模型时做的正是它的**逆变换**；本模块做的就是"反向拼回去"。）

**层归属不在这里推断**：PP 各 stage 持有哪些层、谁持 embedding / lm_head，一律
读分片自带 ``placement`` 记录，并用 ``parallel/placement_plan.build_placement_plan``
（该语义的唯一真相源）**复算对照**；不一致 ⇒ fail-closed。

## 三道防线（任一不符即拒，⛔ 绝不产出"看着像"的权重）

1. **分片齐备**：``manifest`` 声明的 ``tp×dp×pp`` 与实际文件名集合**逐一对应**，
   缺一个/多一个都列出名字后拒。
2. **切片铺满**：每种切片（列块 / 行块 / ``row_indices`` 行集）在各 rank 上必须
   **恰好铺满完整维度**——0 空洞、0 重叠、0 重复，否则拒（附维度与缺失索引）。
3. **复制一致**：声明为"复制"的张量（见上表右列/左列的"完整"项）在各 rank 上必须
   是**同一个值**。判定口径见 :class:`ReplicaPolicy`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: native 分片的文件命名（``transformer_model_adapter.save_checkpoint`` 是唯一产出者）。
SHARD_GLOB = "rank_*_tp_*_pp_*.pt"

#: ``rank_<全局 rank 5 位>_tp_<tp_rank 2 位>_pp_<pp_rank 2 位>.pt``。
#: ⛔ **文件名不含 dp_rank**（DP 副本靠 rank 段区分，dp_rank 只能从 payload 里读）。
_SHARD_RE = re.compile(r"^rank_(\d{5,})_tp_(\d{2})_pp_(\d{2})\.pt$")

#: PEFT 对 transformers 模型的固定外层（``base_model.model.`` + 模型内模块路径）。
_PEFT_ROOT = "base_model.model."

#: 元数据里没有 ``hf_module_path`` 时的兜底前缀（仅 ``layers.`` 与 ``visual.`` 两塔）。
_TOWER_PREFIX = {
    "layers.": "base_model.model.model.language_model.",
    "visual.": "base_model.model.model.",
}

#: 声明为「复制」的矩阵角色（其余为切片）。
REPLICATED_MATRIX = {"none": ("a", "b"), "out": ("a",), "rows": ("a",), "in": ("b",)}

#: 声明为「切片」的矩阵角色。
SLICED_MATRIX = {"none": (), "out": ("b",), "rows": ("b",), "in": ("a",)}

_MATRIX_SUFFIX = {"a": ".lora_a", "b": ".lora_b"}
_PEFT_SUFFIX = {"a": ".lora_A.weight", "b": ".lora_B.weight"}


class ShardMergeError(Exception):
    """分片不可合并（fail-closed）。消息里必须说清**缺什么 / 哪一类不一致**。"""


# ── 纯文件系统 / 纯字符串层（不 import torch，可无 GPU 单测）────────────────────


@dataclass(frozen=True, slots=True)
class ShardRef:
    """一个分片文件及其**文件名里**的坐标。``dp_rank`` 不在文件名里。"""

    path: Path
    rank: int
    tp_rank: int
    pp_rank: int


def parse_shard_name(name: str) -> tuple[int, int, int]:
    """从分片文件名解析 ``(rank, tp_rank, pp_rank)``；不合命名规则 ⇒ fail-closed。"""
    m = _SHARD_RE.match(str(name))
    if m is None:
        raise ShardMergeError(
            f"分片文件名不合 native 命名规则 rank_<rank>_tp_<tp>_pp_<pp>.pt：{name!r}"
        )
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def discover_shards(root: str | Path) -> tuple[ShardRef, ...]:
    """列举 ``root`` 下全部分片（按文件名排序）；一个都没有 ⇒ fail-closed。"""
    base = Path(root)
    refs = []
    for path in sorted(base.glob(SHARD_GLOB)):
        rank, tp_rank, pp_rank = parse_shard_name(path.name)
        refs.append(ShardRef(path=path, rank=rank, tp_rank=tp_rank, pp_rank=pp_rank))
    if not refs:
        raise ShardMergeError(f"{base} 下没有任何 native 分片（{SHARD_GLOB}）")
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ParallelLayout:
    """``manifest.json`` 声明的并行拓扑。"""

    tp_size: int
    dp_size: int
    pp_size: int

    @property
    def world_size(self) -> int:
        return self.tp_size * self.dp_size * self.pp_size

    def __str__(self) -> str:  # 便于错误消息与回执
        return f"tp={self.tp_size} dp={self.dp_size} pp={self.pp_size}"

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> ParallelLayout:
        sizes: dict[str, int] = {}
        for key in ("tp_size", "dp_size", "pp_size"):
            raw = manifest.get(key)
            if raw is None:
                raise ShardMergeError(f"manifest.json 缺 {key} ⇒ 无法判定分片拓扑（fail-closed）")
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ShardMergeError(f"manifest.json 的 {key}={raw!r} 不是整数") from exc
            if value < 1:
                raise ShardMergeError(f"manifest.json 的 {key}={value} < 1 ⇒ 拓扑非法")
            sizes[key] = value
        return cls(tp_size=sizes["tp_size"], dp_size=sizes["dp_size"], pp_size=sizes["pp_size"])


def read_manifest(root: str | Path) -> dict[str, Any]:
    """读 ``manifest.json``；缺失/非对象 ⇒ fail-closed。"""
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise ShardMergeError(f"native 分片目录缺 manifest.json：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShardMergeError(f"{path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ShardMergeError(f"{path} 顶层不是对象（{type(payload).__name__}）")
    return payload


def expected_rank_filenames(layout: ParallelLayout) -> dict[str, tuple[int, int, int]]:
    """该拓扑下**应有**的全部分片文件名 → 文件名声明的 ``(rank, tp_rank, pp_rank)``。

    全局 rank 分解（``world = ((dp_rank * pp_size) + pp_rank) * tp_size + tp_rank``，
    tp 最快变化）与 ``transformer_model_adapter.save_checkpoint`` 的落盘命名一致；
    实际 ``dp_rank`` 由 :func:`merge_lora_shards` 从 payload 读出后**再校验一次**。
    """
    expected: dict[str, tuple[int, int, int]] = {}
    for dp_rank in range(layout.dp_size):
        for pp_rank in range(layout.pp_size):
            for tp_rank in range(layout.tp_size):
                rank = ((dp_rank * layout.pp_size) + pp_rank) * layout.tp_size + tp_rank
                name = f"rank_{rank:05d}_tp_{tp_rank:02d}_pp_{pp_rank:02d}.pt"
                expected[name] = (rank, tp_rank, pp_rank)
    return expected


def assert_shard_set_complete(refs: tuple[ShardRef, ...], layout: ParallelLayout) -> None:
    """分片集合必须与 ``tp×dp×pp`` **逐一对应**；缺/多都拒并列出名字。"""
    expected = expected_rank_filenames(layout)
    got = {ref.path.name: (ref.rank, ref.tp_rank, ref.pp_rank) for ref in refs}
    duplicates = sorted({name for name in got if [r.path.name for r in refs].count(name) > 1})
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    wrong = sorted(
        f"{name}（文件名声明 rank/tp/pp={got[name]}，按拓扑应为 {expected[name]}）"
        for name in set(got) & set(expected)
        if got[name] != expected[name]
    )
    problems: list[str] = []
    if missing:
        problems.append(f"缺 {len(missing)} 个分片：{missing[:8]}")
    if extra:
        problems.append(f"多出 {len(extra)} 个不属于该拓扑的分片：{extra[:8]}")
    if duplicates:
        problems.append(f"重复分片名：{duplicates[:8]}")
    if wrong:
        problems.append(f"文件名坐标与拓扑不符：{wrong[:4]}")
    if problems:
        raise ShardMergeError(
            f"分片集合不完整（{layout}，应有 {layout.world_size} 个分片）：" + "；".join(problems)
        )


@dataclass(frozen=True, slots=True)
class ReplicaPolicy:
    """「声明为复制」的张量如何判定"是同一个值"。

    ★ 这是**过渡期口径**，不是终态验收标准：训练侧修复"复制张量跨 rank 归约末位
    不一致 + 视觉塔 LoRA 无同步"（缺陷①②）之后，应切回 ``require_bitwise=True``。

    ``ulp_tolerance`` 按**该元素自身量级的 dtype 分辨率**判定（不是全局绝对值/相对值
    阈值）：``|Δ| <= ulp_tolerance × ulp(x)`` 视为"同一存储精度下的同一个值"。
    bf16 的 ulp = ``2**(exponent(x) - 8)``。
    """

    require_bitwise: bool = False
    ulp_tolerance: int = 1

    def __post_init__(self) -> None:
        if self.ulp_tolerance < 0:
            raise ShardMergeError(f"ulp_tolerance={self.ulp_tolerance} 非法（须 >= 0）")

    def describe(self) -> str:
        if self.require_bitwise:
            return "require_bitwise（逐位相同，训练侧修复后的终态口径）"
        return f"ulp<={self.ulp_tolerance}（过渡口径：同一 bf16 分辨率下的同一个值）"


@dataclass(slots=True)
class ReplicaDrift:
    """一个复制矩阵的跨 rank 一致性读数（回执用）。"""

    module: str
    matrix: str
    shard_kind: str
    n_ranks: int
    bitwise_equal: bool
    max_abs: float
    max_ulp: float
    n_beyond_ulp: int
    n_elements: int
    ulp_p50: float = 0.0
    ulp_p90: float = 0.0
    ulp_p99: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "matrix": self.matrix,
            "shard_kind": self.shard_kind,
            "n_ranks": self.n_ranks,
            "bitwise_equal": self.bitwise_equal,
            "max_abs": self.max_abs,
            "max_ulp": self.max_ulp,
            "n_beyond_ulp": self.n_beyond_ulp,
            "n_elements": self.n_elements,
            "ulp_p50": self.ulp_p50,
            "ulp_p90": self.ulp_p90,
            "ulp_p99": self.ulp_p99,
        }


@dataclass(slots=True)
class MergeReceipt:
    """合并回执：形状支持矩阵与漂移证据的唯一产出点（写进交付物用）。"""

    layout: ParallelLayout
    shards: list[str] = field(default_factory=list)
    dp_replicas_dropped: list[str] = field(default_factory=list)
    dp_replicas_bitwise_equal: bool = True
    stages: list[dict[str, Any]] = field(default_factory=list)
    modules: int = 0
    keys: int = 0
    rank: int = 0
    lora_r: int | None = None
    lora_alpha: int | None = None
    replica_drift: list[ReplicaDrift] = field(default_factory=list)
    policy: ReplicaPolicy = field(default_factory=ReplicaPolicy)

    @property
    def max_replica_ulp(self) -> float:
        return max((d.max_ulp for d in self.replica_drift), default=0.0)

    @property
    def n_replica_tensors_not_bitwise(self) -> int:
        return sum(1 for d in self.replica_drift if not d.bitwise_equal)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": {
                "tp_size": self.layout.tp_size,
                "dp_size": self.layout.dp_size,
                "pp_size": self.layout.pp_size,
            },
            "policy": self.policy.describe(),
            "shards": self.shards,
            "dp_replicas_dropped": self.dp_replicas_dropped,
            "dp_replicas_bitwise_equal": self.dp_replicas_bitwise_equal,
            "stages": self.stages,
            "modules": self.modules,
            "keys": self.keys,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "rank": self.rank,
            "replica_drift_summary": {
                "n_replicated_tensors": len(self.replica_drift),
                "n_not_bitwise": self.n_replica_tensors_not_bitwise,
                "max_ulp": self.max_replica_ulp,
            },
            "replica_drift": [d.as_dict() for d in self.replica_drift],
        }


# ── 迁移/映射（纯字符串，单测可覆盖）──────────────────────────────────────────


def native_module_to_peft_key(
    module_name: str, matrix: str, hf_module_path: str | None = None
) -> str:
    """``layers.<N>.<mod>`` + ``a|b`` → ``base_model.model.<HF 模块路径>.lora_A/B.weight``。

    **优先用分片元数据里的 ``hf_module_path``**（落盘时由 ``LoRALinear.lora_metadata``
    记录，是该模块在基座里的**真实路径**，唯一真相源）；只有元数据没带它时才按
    ``layers.`` / ``visual.`` 两个塔前缀兜底推导，其余前缀一律拒（不猜）。

    ★ 为什么要优先 ``hf_module_path``：旧通道（``convert_native_rank_to_peft``）按
    ``"visual." -> "…model.visual."`` 拼，视觉塔键会变成 ``…model.visual.visual.blocks…``
    （多一个 ``visual``），对不上基座 ⇒ peft 加载时**静默丢键**、视觉 LoRA 无声失效。
    分片元数据里现成有 ``model.visual.blocks.0.attn.qkv`` ⇒ 直接用，不再推导。
    """
    if matrix not in _MATRIX_SUFFIX:
        raise ShardMergeError(f"矩阵角色只允许 a/b，收到 {matrix!r}")
    if hf_module_path:
        path = str(hf_module_path)
        # ★ 末段模块名必须与 native 名同尾（不要求字面相同：native 有内部别名，
        #   如 `layers.0.token_mixer.in_proj_qkv` ↔ `…layers.0.linear_attn.in_proj_qkv`）。
        #   层号一致性已由 `resolve_module_identity` 结合 stage 偏移校验过。
        tail = str(module_name).rsplit(".", 1)[-1]
        if not path.endswith("." + tail) and not path.endswith(tail):
            raise ShardMergeError(
                f"元数据 hf_module_path={path!r} 末段与 module_name={module_name!r} 不同尾 ⇒ fail-closed"
            )
        return f"{_PEFT_ROOT}{path}{_PEFT_SUFFIX[matrix]}"
    tower = next(
        (
            peft_prefix
            for prefix, peft_prefix in _TOWER_PREFIX.items()
            if str(module_name).startswith(prefix)
        ),
        None,
    )
    if tower is None:
        raise ShardMergeError(
            f"源键前缀未知且无 hf_module_path（不猜前缀）⇒ fail-closed：{module_name}"
        )
    return f"{tower}{module_name}{_PEFT_SUFFIX[matrix]}"


def module_layer_index(module_name: str) -> int | None:
    """``layers.<N>.…`` ⇒ ``N``；无层号（如 ``visual.…``）⇒ ``None``。"""
    parts = str(module_name).split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        return int(parts[1])
    return None


def resolve_module_identity(
    module_name: str,
    hf_module_path: str | None,
    local_layer_indices: tuple[int, ...],
) -> tuple[str, int | None]:
    """给出模块的**全局唯一身份**与**全局层号**。

    ★ 真件实测（pp=2, tp=1）：``pp_rank=1`` 的分片里 ``module_name`` 用的是**本 stage 内
    的局部编号**（``layers.0.…``…``layers.14.…``），而 ``hf_module_path`` 是**全局**的
    （``model.language_model.layers.17.…``…``layers.31.…``）。两个 stage 因此会出现
    **同名不同层**的 `layers.0.mlp.down_proj`，若按下标名聚合会把它们并成一个 ⇒ 静默拼错。
    故：
      - 身份（键）= ``hf_module_path``（全局、唯一）；缺它时退化为
        ``f"{offset}:{module_name}"``（offset=本 stage 最小全局层号），⛔ 绝不用裸名当键；
      - 全局层号 = ``hf_module_path`` 里的 ``layers.<N>.``；缺它时 = ``offset + 局部编号``。
    """
    offset = min(local_layer_indices) if local_layer_indices else 0
    local = module_layer_index(module_name)
    if hf_module_path:
        path = str(hf_module_path)
        identity = path
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", path)
        if match is not None:
            global_layer = int(match.group(1))
            if local is not None and global_layer != offset + local:
                raise ShardMergeError(
                    f"模块 {module_name} 的局部层号 {local} + stage 偏移 {offset} "
                    f"≠ hf_module_path 的全局层号 {global_layer}（{path}）⇒ 元数据自相矛盾"
                )
            return identity, global_layer
        # 无层号的模块（视觉塔等）：用路径当身份、层号留空
        return identity, None
    return f"{offset}:{module_name}", (offset + local if local is not None else None)


def assert_layer_ownership(
    stage_layers: dict[int, tuple[int, ...]],
    module_layers: dict[int, tuple[int | None, ...]],
) -> None:
    """PP 各 stage 的层归属必须①铺满 ②无重叠 ③模块落在本 stage 的层里。

    ``stage_layers``: ``pp_rank -> local_layer_indices``（分片自带 placement 记录）。
    ``module_layers``: ``pp_rank -> 该 stage 实际出现模块的**全局**层号``（见
    :func:`resolve_module_identity`；``None`` = 无层号模块如视觉塔）。
    """
    if not stage_layers:
        raise ShardMergeError("没有任何 pp stage 的 placement 记录 ⇒ 无法判定层归属")
    seen: dict[int, int] = {}
    for pp_rank, layers in sorted(stage_layers.items()):
        if not layers:
            raise ShardMergeError(f"pp stage {pp_rank} 的 local_layer_indices 为空")
        for idx in layers:
            if idx in seen:
                raise ShardMergeError(
                    f"层 {idx} 同时出现在 pp stage {seen[idx]} 与 {pp_rank} ⇒ 层归属重叠"
                )
            seen[idx] = pp_rank
    n_layers = len(seen)
    missing = sorted(set(range(n_layers)) - set(seen))
    if missing:
        raise ShardMergeError(f"层归属有空洞：缺层 {missing[:8]}（共 {n_layers} 层）")
    for pp_rank, layers_present in sorted(module_layers.items()):
        allowed = set(stage_layers.get(pp_rank, ()))
        stray = sorted(idx for idx in layers_present if idx is not None and idx not in allowed)
        if stray:
            raise ShardMergeError(
                f"pp stage {pp_rank} 带着不属于它的层 {stray[:8]} 的模块"
                f"（该 stage 声明持有层 {sorted(allowed)[:8]}…，共 {len(allowed)} 层）"
                f" ⇒ 分片与 placement 不符"
            )


# ── 张量层（需要 torch）──────────────────────────────────────────────────────


def _torch():
    import torch  # noqa: PLC0415  设施层延迟导入：本模块的纯逻辑部分要能在无 torch 环境 import

    return torch


def _load_payload(path: Path) -> dict[str, Any]:
    """``mmap=True`` 懒加载：**不把整个分片读进内存**（轴 7 的流式策略起点）。

    ``mmap`` 让 pickle 里的张量按需映射，只有真正取用的张量才会缺页入 rss。
    """
    torch = _torch()
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:  # torch < 2.0 无 mmap 形参
        return torch.load(path, map_location="cpu", weights_only=False)


def dtype_ulp(tensor: Any):
    """逐元素"该 dtype 在这个量级上的分辨率" ``eps × 2**floor(log2|x|)``。

    ``eps`` 取该 dtype 的 machine epsilon（bf16 = 2^-7，fp16 = 2^-10，fp32 = 2^-23），
    因此这把尺子**随 dtype 变**，不是写死的阈值：1 ulp 就是"这个 dtype 在 x 这个
    量级上能表示的最小一步"，差 1 ulp ⇔ 两个副本在存储精度下是同一个值。
    """
    torch = _torch()
    magnitude = tensor.detach().float().abs().clamp_min(1e-38)
    exponent = torch.floor(torch.log2(magnitude))
    spacing = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent)
    return spacing * float(torch.finfo(tensor.dtype).eps)


def compare_replica(
    reference: Any, other: Any
) -> tuple[float, float, int, bool, tuple[float, float, float]]:
    """比较两份"声明为复制"的张量。

    Returns:
        ``(max_abs, max_ulp, n_beyond_ulp, bitwise, (ulp_p50, ulp_p90, ulp_p99))``
        —— ★ 给**逐元素分布**而不是一个 Frobenius 总和：同一条曲线里，1 ulp 级
        差异与 1e5 ulp 级差异是完全不同的两件事，混成一个数就看不出来了。
    """
    torch = _torch()
    if tuple(reference.shape) != tuple(other.shape):
        raise ShardMergeError(
            f"复制张量形状不一致：{tuple(reference.shape)} vs {tuple(other.shape)} ⇒ fail-closed"
        )
    a = reference.detach().float()
    b = other.detach().float()
    bitwise = bool(torch.equal(reference, other))
    delta = (a - b).abs()
    ulp = dtype_ulp(reference)
    units = (delta / ulp).flatten()
    quantiles = (
        (0.0, 0.0, 0.0)
        if not units.numel()
        else (
            float(units.quantile(0.5)),
            float(units.quantile(0.9)),
            float(units.quantile(0.99)),
        )
    )
    return (
        float(delta.max()) if delta.numel() else 0.0,
        float(units.max()) if units.numel() else 0.0,
        int((delta > ulp * 1.0000001).sum()),
        bitwise,
        quantiles,
    )


def _matrix_of(key: str) -> str:
    for role, suffix in _MATRIX_SUFFIX.items():
        if key.endswith(suffix):
            return role
    raise ShardMergeError(f"分片内出现既非 .lora_a 也非 .lora_b 的键：{key}")


def _module_of(key: str) -> str:
    return key[: -len(_MATRIX_SUFFIX[_matrix_of(key)])]


_AUDIT_ONLY = False


def _check_policy(drift: ReplicaDrift, policy: ReplicaPolicy) -> None:
    if _AUDIT_ONLY:
        return
    if policy.require_bitwise and not drift.bitwise_equal:
        raise ShardMergeError(
            f"复制张量跨 rank 非逐位相同（require_bitwise）：{drift.module}.lora_{drift.matrix}"
            f"（shard_kind={drift.shard_kind}，max|Δ|={drift.max_abs:.3e}，"
            f"max={drift.max_ulp:.3g} ulp，{drift.n_beyond_ulp}/{drift.n_elements} 个元素超限）"
            f" ⇒ fail-closed：该 checkpoint 不编码唯一一个模型（训练侧缺陷①②）"
        )
    if drift.n_beyond_ulp > 0:
        raise ShardMergeError(
            f"复制张量跨 rank 超出 {policy.ulp_tolerance} ulp：{drift.module}.lora_{drift.matrix}"
            f"（shard_kind={drift.shard_kind}，max|Δ|={drift.max_abs:.3e}，"
            f"max={drift.max_ulp:.3g} ulp，{drift.n_beyond_ulp}/{drift.n_elements} 个元素超限）"
            f" ⇒ fail-closed：该 checkpoint 不编码唯一一个模型（训练侧缺陷①②）"
        )


def _scatter_rows(dst: Any, src: Any, rows: list[int], label: str) -> None:
    torch = _torch()
    index = torch.tensor(rows, dtype=torch.long)
    if int(index.numel()) != int(src.shape[0]):
        raise ShardMergeError(f"{label}: row_indices 数 {index.numel()} != 切片行数 {src.shape[0]}")
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= dst.shape[0]):
        raise ShardMergeError(f"{label}: row_indices 越界（dst 行数 {dst.shape[0]}）")
    dst.index_copy_(0, index, src.to(dst.dtype))


def _replicated(
    module: str,
    role: str,
    kind: str,
    copies: list[tuple[tuple[int, int], Any]],
    policy: ReplicaPolicy,
    drift_out: list[ReplicaDrift],
) -> Any:
    """复制矩阵：逐份比对后取 canonical（(pp_rank, tp_rank) 最小者）。

    ★ 复制张量**既跨 tp rank 复制，也可能跨 pp stage 复制**（视觉塔只在 embedding
    stage 上，故通常只有一份）；两种情形都按同一把尺子判——必须是同一个值。
    """
    reference = copies[0][1]
    worst_abs = worst_ulp = 0.0
    beyond = 0
    all_bitwise = True
    quantiles = (0.0, 0.0, 0.0)
    for _position, other in copies[1:]:
        max_abs, max_ulp, n_beyond, bitwise, quantiles = compare_replica(reference, other)
        worst_abs = max(worst_abs, max_abs)
        worst_ulp = max(worst_ulp, max_ulp)
        beyond += n_beyond
        all_bitwise = all_bitwise and bitwise
    drift = ReplicaDrift(
        module=module,
        matrix=role,
        shard_kind=kind,
        n_ranks=len(copies),
        bitwise_equal=all_bitwise,
        max_abs=worst_abs,
        max_ulp=worst_ulp,
        n_beyond_ulp=beyond,
        n_elements=int(reference.numel()),
        ulp_p50=quantiles[0],
        ulp_p90=quantiles[1],
        ulp_p99=quantiles[2],
    )
    drift_out.append(drift)
    if len(copies) > 1:
        _check_policy(drift, policy)
    return reference


def _gather_slices(
    module: str,
    role: str,
    kind: str,
    pieces: list[tuple[int, Any, dict[str, Any]]],
    full_dim: int,
    drift_out: list[ReplicaDrift],
    policy: ReplicaPolicy,
) -> Any:
    """切片矩阵：按元数据把各 rank 的切片**铺回**完整张量，并校验铺满且不重叠。"""
    torch = _torch()
    sample = pieces[0][1]
    dtype = sample.dtype
    if role == "a":
        if int(sample.shape[0]) <= 0:
            raise ShardMergeError(f"{module}.lora_a 形状非法：{tuple(sample.shape)}")
        dst = torch.zeros((int(sample.shape[0]), full_dim), dtype=torch.float32)
    else:
        dst = torch.zeros((full_dim, int(sample.shape[1])), dtype=torch.float32)
    covered: list[int] = []
    for tp_rank, tensor, meta in sorted(pieces, key=lambda item: item[0]):
        if role == "a":
            if kind != "in":
                raise ShardMergeError(f"{module}: shard_kind={kind} 的 lora_a 不声明为切片")
            start, stop = meta.get("col_start"), meta.get("col_stop")
            if start is None or stop is None:
                raise ShardMergeError(f"{module}.lora_a: shard_kind=in 但缺 col_start/col_stop")
            start, stop = int(start), int(stop)
            if stop - start != int(tensor.shape[1]):
                raise ShardMergeError(
                    f"{module}.lora_a: col 范围 [{start},{stop}) 与切片列数 {tensor.shape[1]} 不符"
                )
            dst[:, start:stop] += tensor.float()
            covered.extend(range(start, stop))
        else:
            if kind == "out" or (kind == "rows" and meta.get("row_indices") is None):
                start, stop = meta.get("row_start"), meta.get("row_stop")
                if start is None or stop is None:
                    raise ShardMergeError(
                        f"{module}.lora_b: shard_kind={kind} 但缺 row_start/row_stop"
                    )
                start, stop = int(start), int(stop)
                if stop - start != int(tensor.shape[0]):
                    raise ShardMergeError(
                        f"{module}.lora_b: row 范围 [{start},{stop}) 与切片行数 {tensor.shape[0]} 不符"
                    )
                dst[start:stop] += tensor.float()
                covered.extend(range(start, stop))
            elif kind == "rows":
                rows = [int(i) for i in (meta.get("row_indices") or [])]
                if not rows:
                    raise ShardMergeError(f"{module}.lora_b: shard_kind=rows 但 row_indices 为空")
                _scatter_rows(dst, tensor.float(), rows, f"{module}.lora_b(tp{tp_rank})")
                covered.extend(rows)
            else:
                raise ShardMergeError(f"{module}: shard_kind={kind} 的 lora_b 不声明为切片")
        del tp_rank
    seen: dict[int, int] = {}
    duplicated: list[int] = []
    for idx in covered:
        if idx in seen:
            duplicated.append(idx)
        seen[idx] = 1
    missing = sorted(set(range(full_dim)) - set(seen))
    if duplicated or missing:
        dim_name = "列" if role == "a" else "行"
        raise ShardMergeError(
            f"{module}.lora_{role}: 切片未恰好铺满完整{dim_name}（full={full_dim}）："
            f"重叠 {len(duplicated)} 个（{sorted(set(duplicated))[:5]}），"
            f"空洞 {len(missing)} 个（{missing[:5]}） ⇒ fail-closed"
        )
    drift_out.append(
        ReplicaDrift(
            module=module,
            matrix=role,
            shard_kind=kind,
            n_ranks=len(pieces),
            bitwise_equal=True,  # 切片类不做"复制一致性"判定（每片本就不同）
            max_abs=0.0,
            max_ulp=0.0,
            n_beyond_ulp=0,
            n_elements=int(dst.numel()),
        )
    )
    del policy
    return dst.to(dtype)


def _full_dim_of(
    module: str,
    role: str,
    kind: str,
    pieces: list[tuple[int, Any, dict[str, Any]]],
) -> int:
    """由元数据推出完整维度：不猜、不取基座权重、只读分片声明的坐标。"""
    if role == "a":
        stops = [int(m.get("col_stop") or 0) for _, _, _, m in pieces]
        if kind != "in" or not stops or max(stops) <= 0:
            raise ShardMergeError(f"{module}.lora_a: shard_kind={kind} 无有效 col_stop")
        return max(stops)
    row_stops = [int(m.get("row_stop") or 0) for _, _, _, m in pieces]
    indices = [i for _, _, _, m in pieces for i in (m.get("row_indices") or [])]
    if indices:
        return max(max(indices) + 1, max(row_stops, default=0))
    if kind == "rows":
        raise ShardMergeError(f"{module}.lora_b: shard_kind=rows 但各 rank 都没有 row_indices")
    if max(row_stops, default=0) <= 0:
        raise ShardMergeError(f"{module}.lora_b: shard_kind={kind} 无有效 row_stop")
    return max(row_stops)


# ── 主入口 ──────────────────────────────────────────────────────────────────


def merge_lora_shards(
    checkpoint_dir: str | Path,
    *,
    policy: ReplicaPolicy | None = None,
    audit_only: bool = False,
) -> tuple[dict[str, Any], MergeReceipt]:
    """把 ``final/`` 的分片 LoRA 重组成**整模型那一份**，返回 (PEFT 键值字典, 回执)。

    键名/形状/dtype 与 ``dp=tp=pp=1`` 通道**完全一致**（``lora_A/B.weight`` 全模块路径），
    因此可直接交给 ``merged_export.merge_peft_checkpoint`` 合到基座。

    Args:
        audit_only: ★ 仅用于**诊断**：复制张量超限只记录、不抛异常，用于产出
            "这个分片集到底哪些类漂了多少"的完整报告（见 :func:`audit_shards`）。
            ⛔ 绝不作为放行开关——``audit_only=True`` 的返回值不得用于导出。

    Raises:
        ShardMergeError: 分片不齐 / 切片不铺满 / 复制不一致 / 元数据缺失 / 拓扑不支持。
    """
    torch = _torch()
    root = Path(checkpoint_dir)
    policy = policy or ReplicaPolicy()
    manifest = read_manifest(root)
    layout = ParallelLayout.from_manifest(manifest)
    refs = discover_shards(root)
    assert_shard_set_complete(refs, layout)
    global _AUDIT_ONLY
    _AUDIT_ONLY = audit_only
    receipt = MergeReceipt(layout=layout, shards=[r.path.name for r in refs], policy=policy)

    fmt = str(manifest.get("format") or "")
    if fmt != "native-lora":
        raise ShardMergeError(
            f"manifest.format={fmt!r} 不是 native-lora ⇒ 本合并器只支持 LoRA 分片"
            f"（全参分片 full_param_state_dict 不含 lora_tensor_metadata，无法按元数据拼回，"
            f"见 DELIVERY.md「未做与已知限制」）"
        )

    # ── ① 逐 stage 归组：payload 的 tp/dp/pp 必须与文件名和 manifest 三方一致 ──
    by_stage: dict[int, dict[int, dict[int, dict[str, Any]]]] = {}
    for ref in refs:
        payload = _load_payload(ref.path)
        for key, expect in (
            ("tp_size", layout.tp_size),
            ("dp_size", layout.dp_size),
            ("pp_size", layout.pp_size),
        ):
            if int(payload.get(key, -1)) != expect:
                raise ShardMergeError(
                    f"{ref.path.name} 的 {key}={payload.get(key)!r} 与 manifest 声明的 {expect} 不符"
                )
        for key, expect in (("tp_rank", ref.tp_rank), ("pp_rank", ref.pp_rank), ("rank", ref.rank)):
            if int(payload.get(key, -1)) != expect:
                raise ShardMergeError(
                    f"{ref.path.name} 的 {key}={payload.get(key)!r} 与文件名声明的 {expect} 不符"
                    f"（落盘契约破损）"
                )
        dp_rank = int(payload.get("dp_rank", -1))
        if not 0 <= dp_rank < layout.dp_size:
            raise ShardMergeError(
                f"{ref.path.name} 的 dp_rank={dp_rank} 越界（dp_size={layout.dp_size}）"
            )
        stage = by_stage.setdefault(ref.pp_rank, {})
        tp_slot = stage.setdefault(ref.tp_rank, {})
        if dp_rank in tp_slot:
            raise ShardMergeError(
                f"pp{ref.pp_rank}/tp{ref.tp_rank} 上出现两个 dp_rank={dp_rank} 的分片"
            )
        tp_slot[dp_rank] = payload

    # ── ② DP 副本必须逐位相同（训练侧 dpfix 之后的事实），取 dp_rank=0 ──
    canonical: dict[int, dict[int, dict[str, Any]]] = {}
    for pp_rank, stage in sorted(by_stage.items()):
        slot: dict[int, dict[str, Any]] = {}
        for tp_rank, copies in sorted(stage.items()):
            if len(copies) != layout.dp_size:
                raise ShardMergeError(
                    f"pp{pp_rank}/tp{tp_rank} 只有 {len(copies)} 个 dp 副本，应为 {layout.dp_size}"
                )
            base = copies[0]["lora_state_dict"]
            base_meta = copies[0]["lora_tensor_metadata"]
            for dp_rank in sorted(copies)[1:]:
                other = copies[dp_rank]["lora_state_dict"]
                if set(base) != set(other):
                    raise ShardMergeError(
                        f"pp{pp_rank}/tp{tp_rank} 的 dp_rank=0 与 dp_rank={dp_rank} 键集合不同"
                    )
                differing = [k for k in base if not torch.equal(base[k], other[k])]
                receipt.dp_replicas_bitwise_equal &= not differing
                receipt.dp_replicas_dropped.append(
                    f"{copies[dp_rank].get('rank')}@pp{pp_rank}/tp{tp_rank}/dp{dp_rank}"
                )
                if differing:
                    raise ShardMergeError(
                        f"DP 副本不一致（pp{pp_rank}/tp{tp_rank}，dp0 vs dp{dp_rank}）："
                        f"{len(differing)} 个键不同，例如 {differing[:4]} ⇒ fail-closed"
                        f"（DP 副本必须逐位相同，见 _sync_dp_lora_grads 的 AVG 同步）"
                    )
                if copies[dp_rank]["lora_tensor_metadata"] != base_meta:
                    raise ShardMergeError(
                        f"DP 副本元数据不一致（pp{pp_rank}/tp{tp_rank}，dp{dp_rank}）⇒ fail-closed"
                    )
            slot[tp_rank] = copies[0]
        if len(slot) != layout.tp_size:
            raise ShardMergeError(f"pp{pp_rank} 只有 {len(slot)} 个 tp rank，应为 {layout.tp_size}")
        canonical[pp_rank] = slot

    # ── ③ 层归属：读分片 placement，用 placement_plan（唯一真相源）复算对照 ──
    from graspo.flow.backend.native.parallel.placement_plan import build_placement_plan

    stage_layers: dict[int, tuple[int, ...]] = {}
    module_layers: dict[int, tuple[int | None, ...]] = {}
    for pp_rank, slot in sorted(canonical.items()):
        base_payload = slot[0]
        placement = base_payload.get("placement")
        if not isinstance(placement, dict):
            raise ShardMergeError(f"pp{pp_rank} 的分片缺 placement 记录 ⇒ 无法判定层归属")
        layers = tuple(int(i) for i in (placement.get("local_layer_indices") or ()))
        if not layers:
            raise ShardMergeError(f"pp{pp_rank} 的 placement.local_layer_indices 为空")
        stage_layers[pp_rank] = layers
        metas = base_payload.get("lora_tensor_metadata")
        if not isinstance(metas, list) or not metas:
            raise ShardMergeError(f"pp{pp_rank} 的分片缺 lora_tensor_metadata ⇒ 无法按元数据拼回")
        identities = {
            resolve_module_identity(str(m["module_name"]), m.get("hf_module_path"), layers)[0]
            for m in metas
        }
        module_layers[pp_rank] = tuple(
            resolve_module_identity(str(m["module_name"]), m.get("hf_module_path"), layers)[1]
            for m in metas
        )
        names = tuple(sorted(identities))
        receipt.stages.append(
            {
                "pp_rank": pp_rank,
                "local_layer_indices": list(layers),
                "include_embeddings": bool(placement.get("include_embeddings")),
                "include_lm_head": bool(placement.get("include_lm_head")),
                "modules": len(names),
                "strategy": placement.get("placement_strategy"),
            }
        )
    assert_layer_ownership(stage_layers, module_layers)

    num_hidden_layers = 1 + max(max(v) for v in stage_layers.values())
    for pp_rank, slot in sorted(canonical.items()):
        placement = slot[0]["placement"]
        strategy = str(placement.get("placement_strategy") or "auto")
        if strategy == "manual":
            # 手工区间无法从 summary 复原原始 ranges ⇒ 只校验连续性（覆盖性已由
            # assert_layer_ownership 保证），并在回执里留痕。
            receipt.stages[pp_rank]["placement_cross_check"] = "manual（自记录，仅校验连续）"
            continue
        plan = build_placement_plan(
            strategy=strategy,
            model_family=str(placement.get("model_family") or "qwen3_5_text"),
            num_hidden_layers=num_hidden_layers,
            tp_size=int(placement.get("tp_size") or layout.tp_size),
            pp_size=int(placement.get("pp_size") or layout.pp_size),
            tp_rank=int(placement.get("tp_rank") or 0),
            pp_rank=int(placement.get("pp_rank") or 0),
        )
        if tuple(plan.local_layer_indices) != stage_layers[pp_rank]:
            raise ShardMergeError(
                f"pp{pp_rank} 的层归属与 placement_plan 复算不符："
                f"分片={list(stage_layers[pp_rank])[:8]}… / 复算={list(plan.local_layer_indices)[:8]}…"
                f"（strategy={strategy}）⇒ fail-closed"
            )
        if bool(plan.include_embeddings) != bool(placement.get("include_embeddings")) or bool(
            plan.include_lm_head
        ) != bool(placement.get("include_lm_head")):
            raise ShardMergeError(
                f"pp{pp_rank} 的 embeddings/lm_head 归属与 placement_plan 复算不符 ⇒ fail-closed"
            )
        receipt.stages[pp_rank]["placement_cross_check"] = "ok"

    # ── ④ 逐模块拼回（顺序 = canonical 分片里 lora_state_dict 的落盘顺序，pp 按 stage 串起来）──
    merged: dict[str, Any] = {}
    per_module: dict[str, dict[str, list[tuple[int, Any, dict[str, Any]]]]] = {}
    kinds: dict[str, str] = {}
    hf_paths: dict[str, str | None] = {}
    native_names: dict[str, str] = {}
    order: list[str] = []
    stage_offsets = {pp: (min(v) if v else 0) for pp, v in stage_layers.items()}
    for pp_rank in sorted(canonical):
        first = canonical[pp_rank][0]
        offset = stage_offsets[pp_rank]
        meta_by_name = {
            str(m["module_name"]): m for m in first["lora_tensor_metadata"]
        }
        for key in first["lora_state_dict"]:
            module = _module_of(key)
            if module not in meta_by_name:
                raise ShardMergeError(
                    f"{module} 在 lora_state_dict 里有张量但元数据里没有 ⇒ fail-closed"
                )
            meta = meta_by_name[module]
            identity = resolve_module_identity(
                module, meta.get("hf_module_path"), stage_layers[pp_rank]
            )[0]
            if identity in kinds:
                # 同一模块的 lora_a / lora_b 两个键 ⇒ 正常；但同一身份对应两个不同
                # native 名说明元数据自相矛盾，必须拒。
                if native_names.get(identity) != module:
                    raise ShardMergeError(
                        f"模块身份 {identity} 对应了多个 native 名："
                        f"{native_names.get(identity)!r} vs {module!r} ⇒ fail-closed"
                    )
                continue
            native_names[identity] = module
            kinds[identity] = str(meta["shard_kind"])
            hf_paths[identity] = meta.get("hf_module_path")
            order.append(identity)
        del offset
    ranks_seen: set[int] = set()
    alphas_seen: set[int] = set()
    for pp_rank in sorted(canonical):
        for meta in canonical[pp_rank][0]["lora_tensor_metadata"]:
            if meta.get("r") is not None:
                ranks_seen.add(int(meta["r"]))
            if meta.get("alpha") is not None:
                alphas_seen.add(int(meta["alpha"]))
    if len(ranks_seen) > 1 or len(alphas_seen) > 1:
        raise ShardMergeError(
            f"分片元数据里的 r/alpha 不一致：r={sorted(ranks_seen)} alpha={sorted(alphas_seen)}"
            f" ⇒ fail-closed"
        )
    receipt.lora_r = next(iter(ranks_seen), None)
    receipt.lora_alpha = next(iter(alphas_seen), None)
    known_kinds = set(REPLICATED_MATRIX)
    unknown = {m: k for m, k in kinds.items() if k not in known_kinds}
    if unknown:
        sample = list(unknown.items())[:4]
        raise ShardMergeError(
            f"出现未支持的分片拓扑 shard_kind={sorted({k for k in unknown.values()})}（例 {sample}）"
            f" ⇒ fail-closed：合并器只认 {sorted(known_kinds)}"
        )

    for module, kind in kinds.items():
        per_module[module] = {"a": [], "b": []}
        hf_path = hf_paths.get(module)
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.", hf_path or "")
        layer = int(layer_match.group(1)) if layer_match else None
        for pp_rank, slot in canonical.items():
            if layer is not None and layer not in stage_layers[pp_rank]:
                continue
            for tp_rank, payload in sorted(slot.items()):
                metas = {}
                for m in payload["lora_tensor_metadata"]:
                    ident = resolve_module_identity(
                        str(m["module_name"]), m.get("hf_module_path"), stage_layers[pp_rank]
                    )[0]
                    metas[ident] = m
                meta = metas.get(module)
                if meta is None:
                    if layer is not None:
                        # 该 stage 声明持有这一层，却没有该模块 ⇒ 分片不完整，不得跳过
                        raise ShardMergeError(
                            f"pp{pp_rank}/tp{tp_rank} 声明持有层 {layer}，但缺模块 {module} 的元数据"
                            f" ⇒ fail-closed"
                        )
                    # 无层号的模块（视觉塔等）允许只出现在部分 stage（真实布局只在
                    # embedding stage 建塔）——但出现几次就必须几份都相同，见 _replicated。
                    continue
                state = payload["lora_state_dict"]
                native_name = str(meta["module_name"])
                path_here = meta.get("hf_module_path")
                if path_here is not None and hf_paths.get(module) not in (None, path_here):
                    raise ShardMergeError(
                        f"{module} 的 hf_module_path 跨 rank 不一致："
                        f"{hf_paths[module]!r} vs {path_here!r} ⇒ fail-closed"
                    )
                hf_paths.setdefault(module, path_here)
                for role in ("a", "b"):
                    key = f"{native_name}{_MATRIX_SUFFIX[role]}"
                    if key not in state:
                        raise ShardMergeError(f"pp{pp_rank}/tp{tp_rank} 缺张量 {key}（{module}）")
                    per_module[module][role].append((pp_rank, tp_rank, state[key], meta))
        for role in ("a", "b"):
            pieces = per_module[module][role]
            if not pieces:
                raise ShardMergeError(
                    f"{module}.lora_{role} 在没有任何 stage/tp 上出现 ⇒ fail-closed"
                )
            tp_ranks = {tp for _, tp, _, _ in pieces}
            stages_present = {pp for pp, _, _, _ in pieces}
            if tp_ranks != set(range(layout.tp_size)):
                raise ShardMergeError(
                    f"{module}.lora_{role} 只在 tp rank {sorted(tp_ranks)} 上出现"
                    f"（应覆盖 0..{layout.tp_size - 1}）⇒ 分片缺失"
                )
            if len(stages_present) > 1 and role in SLICED_MATRIX[kind]:
                raise ShardMergeError(
                    f"{module}.lora_{role}: shard_kind={kind} 声明为切片，却出现在多个 pp stage "
                    f"{sorted(stages_present)} ⇒ fail-closed（切片只能在同一个 stage 内跨 tp 分）"
                )

    for module in order:
        kind = kinds[module]
        tensors: dict[str, Any] = {}
        for role in ("a", "b"):
            pieces = per_module[module][role]
            if role in REPLICATED_MATRIX[kind]:
                tensors[role] = _replicated(
                    module,
                    role,
                    kind,
                    [((pp, tp), tensor) for pp, tp, tensor, _ in pieces],
                    policy,
                    receipt.replica_drift,
                )
            elif role in SLICED_MATRIX[kind]:
                full = _full_dim_of(module, role, kind, pieces)
                tensors[role] = _gather_slices(
                    module,
                    role,
                    kind,
                    [(tp, tensor, meta) for _, tp, tensor, meta in pieces],
                    full,
                    receipt.replica_drift,
                    policy,
                )
            else:
                raise ShardMergeError(f"{module}.lora_{role}: shard_kind={kind} 的角色归属未知")
        a, b = tensors["a"], tensors["b"]
        if int(a.shape[0]) != int(b.shape[1]):
            raise ShardMergeError(
                f"{module}: lora_a 的 r={a.shape[0]} 与 lora_b 的 r={b.shape[1]} 不符 ⇒ fail-closed"
            )
        native_name = native_names[module]
        merged[native_module_to_peft_key(native_name, "a", hf_paths.get(module))] = a
        merged[native_module_to_peft_key(native_name, "b", hf_paths.get(module))] = b
        del per_module[module]

    receipt.modules = len(order)
    receipt.keys = len(merged)
    return merged, receipt


def merge_receipt_summary(receipt: MergeReceipt) -> str:
    """一行人类可读摘要（日志用）。"""
    return (
        f"{receipt.layout} → {receipt.modules} 模块 / {receipt.keys} 键；"
        f"dp 副本取 1 弃 {len(receipt.dp_replicas_dropped)}（逐位相同={receipt.dp_replicas_bitwise_equal}）；"
        f"复制张量 {len(receipt.replica_drift)} 个，非逐位 {receipt.n_replica_tensors_not_bitwise} 个，"
        f"max={receipt.max_replica_ulp:.3g} ulp；口径={receipt.policy.describe()}"
    )


def _drift_by_class(receipt: MergeReceipt) -> dict[str, dict[str, Any]]:
    """把复制张量漂移按 ``(shard_kind, matrix)`` 归类汇总（支持矩阵与定性报告用）。"""
    buckets: dict[str, dict[str, Any]] = {}
    for drift in receipt.replica_drift:
        key = f"{drift.shard_kind}.lora_{drift.matrix}"
        bucket = buckets.setdefault(
            key,
            {
                "shard_kind": drift.shard_kind,
                "matrix": drift.matrix,
                "n_tensors": 0,
                "n_not_bitwise": 0,
                "max_abs": 0.0,
                "max_ulp": 0.0,
                "ulp_p50_max": 0.0,
                "ulp_p90_max": 0.0,
                "ulp_p99_max": 0.0,
                "n_beyond_ulp": 0,
                "n_elements": 0,
            },
        )
        bucket["n_tensors"] += 1
        bucket["n_not_bitwise"] += 0 if drift.bitwise_equal else 1
        bucket["max_abs"] = max(bucket["max_abs"], drift.max_abs)
        bucket["max_ulp"] = max(bucket["max_ulp"], drift.max_ulp)
        bucket["ulp_p50_max"] = max(bucket["ulp_p50_max"], drift.ulp_p50)
        bucket["ulp_p90_max"] = max(bucket["ulp_p90_max"], drift.ulp_p90)
        bucket["ulp_p99_max"] = max(bucket["ulp_p99_max"], drift.ulp_p99)
        bucket["n_beyond_ulp"] += drift.n_beyond_ulp
        bucket["n_elements"] += drift.n_elements
    return dict(sorted(buckets.items()))


def audit_shards(
    checkpoint_dir: str | Path, *, policy: ReplicaPolicy | None = None
) -> dict[str, Any]:
    """**只读诊断**：给定 native 分片目录，报告拓扑、切片覆盖、复制漂移的完整体检结果。

    ⛔ 本函数**不产出任何权重**，也不代表"可以导出"——它只回答"这套分片到底哪里对不上"。
    合并放行始终由 :func:`merge_lora_shards`（不加 ``audit_only``）决定。
    """
    torch = _torch()
    receipt: MergeReceipt | None = None
    error: str | None = None
    merged: dict[str, Any] = {}
    try:
        merged, receipt = merge_lora_shards(
            checkpoint_dir, policy=policy or ReplicaPolicy(), audit_only=True
        )
    except ShardMergeError as exc:
        error = str(exc)
    finally:
        global _AUDIT_ONLY
        _AUDIT_ONLY = False
    payload: dict[str, Any] = {"ok": error is None}
    if error:
        payload["error"] = error
    if receipt is not None:
        effective = policy or ReplicaPolicy()
        violations = [
            d
            for d in receipt.replica_drift
            if (effective.require_bitwise and not d.bitwise_equal) or d.n_beyond_ulp > 0
        ]
        payload["ok"] = error is None and not violations
        payload["n_tensors_over_tolerance"] = len(violations)
        payload["first_violation"] = (
            f"{violations[0].module}.lora_{violations[0].matrix}"
            f"（max={violations[0].max_ulp:.3g} ulp，超过 {violations[0].n_beyond_ulp} 个元素）"
            if violations
            else None
        )
        payload.update(
            layout={
                "tp_size": receipt.layout.tp_size,
                "dp_size": receipt.layout.dp_size,
                "pp_size": receipt.layout.pp_size,
            },
            n_shards=len(receipt.shards),
            modules=receipt.modules,
            keys=receipt.keys,
            dp_replicas_bitwise_equal=receipt.dp_replicas_bitwise_equal,
            policy=receipt.policy.describe(),
            drift_by_class=_drift_by_class(receipt),
            drift_summary=receipt.as_dict()["replica_drift_summary"],
        )
    del merged, torch
    return payload
