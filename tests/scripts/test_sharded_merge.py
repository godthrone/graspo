"""N3c 分片检查点合并（`eval/sharded_merge.py`）的单测。

**为什么这个套件必须存在**：合并器是"把分片拼回唯一一个模型"的唯一通道，
它错了不会崩，只会**静默产出一个看着正常的错模型**（污染矩阵结论）。
因此这里逐条锁住契约：三类切片的还原（列/行/非连续行）、PP 各 stage 层拼接、
DP 副本取一、embeddings/lm_head 归属、以及**五类 fail-closed**。

**fixture 扮演"训练器/落盘方"**：按 ``lora_linear.py::from_hf`` 的切片规则
（``_shard_bounds`` / ``_slice_lora_a`` / ``_slice_lora_b`` 的逆）把"完整 LoRA"
切成各 rank 的分片，再让合并器拼回来，断言**逐位还原**。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="分片合并用例需 torch")

from graspo.eval.sharded_merge import (  # noqa: E402
    ParallelLayout,
    ReplicaPolicy,
    ShardMergeError,
    assert_shard_set_complete,
    discover_shards,
    expected_rank_filenames,
    merge_lora_shards,
    merge_receipt_summary,
    native_module_to_peft_key,
    parse_shard_name,
)
from graspo.flow.backend.native.parallel.placement_plan import (  # noqa: E402
    build_placement_plan,
)
from graspo.flow.backend.native.parallel.tensor_utils import _shard_bounds  # noqa: E402

HIDDEN = 8
INTER = 6


# ── fixture：「完整 LoRA」+ 按真实切片规则切分落盘 ────────────────────────────


def _full_lora(n_layers: int, seed: int = 0) -> dict[str, dict[str, object]]:
    """一份**完整**（未分片）LoRA：模块名 → {kind, a, b}。"""
    g = torch.Generator().manual_seed(seed)
    full: dict[str, dict[str, object]] = {}

    def add(name: str, kind: str, out: int, inn: int) -> None:
        full[name] = {
            "kind": kind,
            "a": torch.randn((2, inn), generator=g, dtype=torch.float32).to(torch.bfloat16),
            "b": torch.randn((out, 2), generator=g, dtype=torch.float32).to(torch.bfloat16),
        }

    for i in range(n_layers):
        add(f"layers.{i}.mlp.gate_proj", "out", INTER, HIDDEN)  # 行并行：B 切行
        add(f"layers.{i}.token_mixer.out_proj", "in", HIDDEN, HIDDEN)  # 列并行：A 切列
        add(f"layers.{i}.token_mixer.in_proj_qkv", "rows", 2 * HIDDEN, HIDDEN)  # 非连续行
    add("visual.blocks.0.attn.qkv", "none", HIDDEN, HIDDEN)  # 复制
    return full


def _row_indices(kind_out: int, tp_rank: int, tp_size: int, block: int = 2) -> list[int]:
    """模拟真实模型的**按头（非连续块）**行归属。"""
    n_blocks = kind_out // block
    return [
        r
        for b in range(n_blocks)
        if b % tp_size == tp_rank
        for r in range(b * block, (b + 1) * block)
    ]


def _slice_entry(
    entry: dict[str, object], kind: str, tp_rank: int, tp_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """把完整 LoRA 按 ``kind`` 切成该 rank 的分片 + 元数据（落盘方视角）。"""
    a = entry["a"]
    b = entry["b"]
    meta: dict[str, object] = {
        "shard_kind": kind,
        "row_start": None,
        "row_stop": None,
        "col_start": None,
        "col_stop": None,
        "row_indices": None,
    }
    if kind == "out":
        start, stop = _shard_bounds(int(b.shape[0]), tp_rank=tp_rank, tp_size=tp_size)
        meta["row_start"], meta["row_stop"] = start, stop
        b = b[start:stop].contiguous()
    elif kind == "in":
        start, stop = _shard_bounds(int(a.shape[1]), tp_rank=tp_rank, tp_size=tp_size)
        meta["col_start"], meta["col_stop"] = start, stop
        a = a[:, start:stop].contiguous()
    elif kind == "rows":
        idx = _row_indices(int(b.shape[0]), tp_rank, tp_size)
        meta["row_indices"] = idx
        b = b.index_select(0, torch.tensor(idx, dtype=torch.long)).contiguous()
    return a.clone(), b.clone(), meta


def _write_shards(
    root,
    full: dict[str, dict[str, object]],
    *,
    tp_size: int = 1,
    dp_size: int = 1,
    pp_size: int = 1,
    layer_ranges: dict[int, tuple[int, ...]] | None = None,
    strategy: str = "qwen3_tp",
    drop: str | None = None,
    corrupt=None,
) -> None:
    """按 (tp, dp, pp) 拓扑落盘分片；``drop``/``corrupt`` 用于造 fail-closed 场景。"""
    root.mkdir(parents=True, exist_ok=True)
    n_layers = 1 + max((int(n.split(".")[1]) for n in full if n.startswith("layers.")), default=-1)
    if layer_ranges is None:
        layer_ranges = {r: tuple(range(n_layers)) for r in range(pp_size)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "native-lora",
                "tuner_type": "lora",
                "tp_size": tp_size,
                "dp_size": dp_size,
                "pp_size": pp_size,
            }
        ),
        encoding="utf-8",
    )
    for dp_rank in range(dp_size):
        for pp_rank in range(pp_size):
            layers = set(layer_ranges[pp_rank])
            for tp_rank in range(tp_size):
                rank = ((dp_rank * pp_size) + pp_rank) * tp_size + tp_rank
                name = f"rank_{rank:05d}_tp_{tp_rank:02d}_pp_{pp_rank:02d}.pt"
                if drop == name:
                    continue
                state: dict[str, torch.Tensor] = {}
                metadata: list[dict] = []
                for module, entry in full.items():
                    if module.startswith("layers.") and int(module.split(".")[1]) not in layers:
                        continue
                    a, b, meta = _slice_entry(entry, str(entry["kind"]), tp_rank, tp_size)
                    if corrupt is not None:
                        a, b, meta = corrupt(module, a, b, meta, tp_rank)
                    module_name = module
                    state[f"{module_name}.lora_a"] = a
                    state[f"{module_name}.lora_b"] = b
                    metadata.append(
                        {"module_name": module_name, "r": 2, "alpha": 4,
                         "hf_module_path": None, **meta}
                    )
                payload = {
                    "adapter": "native",
                    "tuner_type": "lora",
                    "rank": rank,
                    "tp_rank": tp_rank,
                    "dp_rank": dp_rank,
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "pp_rank": pp_rank,
                    "pp_size": pp_size,
                    "placement": {
                        "placement_strategy": strategy,
                        "model_family": "qwen3_5_text",
                        "tp_size": tp_size,
                        "pp_size": pp_size,
                        "tp_rank": tp_rank,
                        "pp_rank": pp_rank,
                        "local_layer_indices": sorted(layers),
                        "include_embeddings": pp_rank == 0,
                        "include_lm_head": pp_rank == pp_size - 1,
                    },
                    "lora_tensor_metadata": metadata,
                    "lora_state_dict": state,
                }
                torch.save(payload, root / name)


def _assert_full_restored(merged: dict[str, torch.Tensor], full: dict[str, dict]) -> None:
    for module, entry in full.items():
        ka = native_module_to_peft_key(module, "a")
        kb = native_module_to_peft_key(module, "b")
        assert torch.equal(merged[ka].to(torch.bfloat16), entry["a"]), f"{module}.lora_a 未还原"
        assert torch.equal(merged[kb].to(torch.bfloat16), entry["b"]), f"{module}.lora_b 未还原"


# ── 1~3：三类 TP 切片的还原（列并行 / 行并行 / 非连续行）────────────────────


def test_tp_in_column_slice_restored_exactly(tmp_path):
    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2)
    merged, receipt = merge_lora_shards(tmp_path)
    assert torch.equal(
        merged[native_module_to_peft_key("layers.0.token_mixer.out_proj", "a")].to(torch.bfloat16),
        full["layers.0.token_mixer.out_proj"]["a"],
    )
    assert receipt.layout.tp_size == 2


def test_tp_out_row_slice_restored_exactly(tmp_path):
    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2)
    merged, _ = merge_lora_shards(tmp_path)
    assert torch.equal(
        merged[native_module_to_peft_key("layers.0.mlp.gate_proj", "b")].to(torch.bfloat16),
        full["layers.0.mlp.gate_proj"]["b"],
    )


def test_tp_rows_non_contiguous_indices_restored_exactly(tmp_path):
    """`rows` 是**按头的非连续行集**（真实 in_proj_qkv 就是这种），⛔ 不能用连续块近似。"""
    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2)
    merged, _ = merge_lora_shards(tmp_path)
    restored = merged[native_module_to_peft_key("layers.0.token_mixer.in_proj_qkv", "b")]
    assert torch.equal(restored.to(torch.bfloat16), full["layers.0.token_mixer.in_proj_qkv"]["b"])


# ── 4~5：PP 层拼接 / TP+PP 组合 ─────────────────────────────────────────────


def test_pp_stage_layers_concatenated(tmp_path):
    """pp=2：每 stage 只持自己那些层，合并后必须是**全层**且逐位还原。"""
    full = _full_lora(4)
    ranges = {
        r: tuple(
            build_placement_plan(
                strategy="qwen36_pp8_static",
                model_family="qwen3_5_text",
                num_hidden_layers=4,
                tp_size=1,
                pp_size=2,
                tp_rank=0,
                pp_rank=r,
            ).local_layer_indices
        )
        for r in range(2)
    }
    _write_shards(tmp_path, full, pp_size=2, layer_ranges=ranges, strategy="qwen36_pp8_static")
    merged, receipt = merge_lora_shards(tmp_path)
    _assert_full_restored(merged, full)
    assert receipt.stages[0]["placement_cross_check"] == "ok"
    assert receipt.stages[1]["include_lm_head"] is True


def test_tp_and_pp_combined_restored_exactly(tmp_path):
    """tp=2 + pp=2 同时开：四片拼回唯一一份完整 LoRA。"""
    full = _full_lora(4)
    ranges = {
        r: tuple(
            build_placement_plan(
                strategy="qwen36_pp8_static",
                model_family="qwen3_5_text",
                num_hidden_layers=4,
                tp_size=2,
                pp_size=2,
                tp_rank=0,
                pp_rank=r,
            ).local_layer_indices
        )
        for r in range(2)
    }
    _write_shards(
        tmp_path, full, tp_size=2, pp_size=2, layer_ranges=ranges, strategy="qwen36_pp8_static"
    )
    merged, receipt = merge_lora_shards(tmp_path)
    _assert_full_restored(merged, full)
    assert receipt.layout.world_size == 4
    assert len(receipt.shards) == 4


# ── 6：DP 多副本取一且内容一致 ───────────────────────────────────────────────


def test_dp_replicas_take_one_and_require_equal(tmp_path):
    full = _full_lora(2)
    _write_shards(tmp_path, full, dp_size=2)
    merged, receipt = merge_lora_shards(tmp_path)
    _assert_full_restored(merged, full)
    assert receipt.dp_replicas_bitwise_equal is True
    assert len(receipt.dp_replicas_dropped) == 1


def test_dp_replicas_divergent_fails_closed(tmp_path):
    """DP 副本不同 ⇒ 拒（DP 侧 AVG 同步之后必须逐位相同）。"""
    full = _full_lora(2)

    _write_shards(tmp_path, full, dp_size=2)
    # 改一个 DP 副本的张量（模拟 DP 同步失效）
    victim = tmp_path / "rank_00001_tp_00_pp_00.pt"
    payload = torch.load(victim, map_location="cpu", weights_only=False)
    key = "layers.0.mlp.gate_proj.lora_a"
    payload["lora_state_dict"][key] = payload["lora_state_dict"][key] + 1.0
    torch.save(payload, victim)
    with pytest.raises(ShardMergeError, match="DP 副本不一致"):
        merge_lora_shards(tmp_path)


# ── 7：embeddings / lm_head 归属（由 placement_plan 复算对照）────────────────


def test_embeddings_and_lm_head_ownership_mismatch_fails_closed(tmp_path):
    full = _full_lora(4)
    ranges = {0: (0, 1), 1: (2, 3)}
    _write_shards(tmp_path, full, pp_size=2, layer_ranges=ranges, strategy="manual")
    # 篡改 lm_head 归属 ⇒ 必须拒（手工 strategy 不比对 placement_plan，但占用/归属仍被读）
    victim = tmp_path / "rank_00001_tp_00_pp_01.pt"
    payload = torch.load(victim, map_location="cpu", weights_only=False)
    payload["placement"]["local_layer_indices"] = [0, 1]  # 与 stage0 重叠
    torch.save(payload, victim)
    with pytest.raises(ShardMergeError, match="重叠|层归属"):
        merge_lora_shards(tmp_path)


# ── 8~9：LoRA 通道可用 / 非 LoRA 通道显式拒绝 ───────────────────────────────


def test_lora_path_produces_peft_full_module_keys(tmp_path):
    full = _full_lora(2)
    _write_shards(tmp_path, full, tp_size=2)
    merged, _ = merge_lora_shards(tmp_path)
    assert all(k.startswith("base_model.model.model.") for k in merged)
    assert any("language_model.layers.1.mlp.gate_proj.lora_A.weight" in k for k in merged)
    assert "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight" in merged
    assert len(merged) == 2 * len(full)


def test_full_param_checkpoint_refused_with_reason(tmp_path):
    full = _full_lora(1)
    _write_shards(tmp_path, full)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["format"] = "native-full-param"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShardMergeError, match="native-lora|全参"):
        merge_lora_shards(tmp_path)


# ── 10~14：五类 fail-closed ────────────────────────────────────────────────


def test_missing_shard_fails_closed_and_names_it(tmp_path):
    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2, drop="rank_00001_tp_01_pp_00.pt")
    with pytest.raises(ShardMergeError, match="缺 1 个分片"):
        merge_lora_shards(tmp_path)


def test_slice_gap_fails_closed(tmp_path):
    """切片留空洞（少一列）⇒ 拒，⛔ 不得把空洞当 0 静默写进权重。"""

    def corrupt(module, a, b, meta, tp_rank):
        if meta["shard_kind"] == "in" and tp_rank == 1:
            meta = dict(meta, col_start=int(meta["col_start"]) + 1)
            a = a[:, 1:].contiguous()
        return a, b, meta

    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2, corrupt=corrupt)
    with pytest.raises(ShardMergeError, match="未恰好铺满|空洞"):
        merge_lora_shards(tmp_path)


def test_slice_overlap_fails_closed(tmp_path):
    def corrupt(module, a, b, meta, tp_rank):
        if meta["shard_kind"] == "out" and tp_rank == 1:
            meta = dict(meta, row_start=int(meta["row_start"]) - 1)
        return a, b, meta

    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2, corrupt=corrupt)
    with pytest.raises(ShardMergeError, match="未恰好铺满|重叠|不符"):
        merge_lora_shards(tmp_path)


def test_replica_drift_beyond_ulp_fails_closed(tmp_path):
    """复制张量跨 rank 差得比 1 ulp 多 ⇒ 拒（该 checkpoint 不编码唯一模型）。"""

    def corrupt(module, a, b, meta, tp_rank):
        if meta["shard_kind"] == "none" and tp_rank == 1:
            a = a + 0.05
        return a, b, meta

    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2, corrupt=corrupt)
    with pytest.raises(ShardMergeError, match="超出 1 ulp|非逐位"):
        merge_lora_shards(tmp_path)


def test_compare_replica_one_ulp_counts_as_same_value():
    """1 ulp = "同一 bf16 分辨率下的同一个值" ⇒ 不算漂移；逐位口径下才算。"""
    from graspo.eval.sharded_merge import compare_replica

    a = torch.tensor([1.0, 1.5, 1.9375], dtype=torch.bfloat16)
    b = (a.view(torch.int16) + 1).view(torch.bfloat16)  # 恰好 +1 ulp
    max_abs, max_ulp, n_beyond, bitwise, ulp_q = compare_replica(a, b)
    assert bitwise is False
    assert max_ulp == pytest.approx(1.0, rel=1e-6)
    assert n_beyond == 0
    assert max_abs > 0
    assert ulp_q[2] <= 1.0 + 1e-6  # 逐元素 ulp 分位：p99 也不超过 1 ulp


def test_require_bitwise_rejects_one_ulp_difference(tmp_path):
    """``--require-bitwise``（终态口径）比 1-ulp 口径更严：连 1 ulp 也要拒。"""

    def corrupt(module, a, b, meta, tp_rank):
        if meta["shard_kind"] == "none":
            a = (a.abs() * 0.25 + 1.0).to(torch.bfloat16)  # 取值域 [1,2) ⇒ +1bit 恰为 1 ulp
            if tp_rank == 1:
                a = (a.view(torch.int16) + 1).view(torch.bfloat16)
        return a, b, meta

    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2, corrupt=corrupt)
    _, receipt = merge_lora_shards(tmp_path)
    assert receipt.n_replica_tensors_not_bitwise >= 1  # 1-ulp 口径下接受
    assert receipt.max_replica_ulp <= 1.0 + 1e-6
    with pytest.raises(ShardMergeError, match="非逐位相同"):
        merge_lora_shards(tmp_path, policy=ReplicaPolicy(require_bitwise=True))


def test_unknown_shard_kind_fails_closed(tmp_path):
    def corrupt(module, a, b, meta, tp_rank):
        if module.endswith("gate_proj"):
            meta = dict(meta, shard_kind="megatron_sequence")
        return a, b, meta

    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2, corrupt=corrupt)
    with pytest.raises(ShardMergeError, match="未支持的分片拓扑"):
        merge_lora_shards(tmp_path)


def test_missing_metadata_fails_closed(tmp_path):
    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=2)
    victim = tmp_path / "rank_00000_tp_00_pp_00.pt"
    payload = torch.load(victim, map_location="cpu", weights_only=False)
    payload["lora_tensor_metadata"] = []
    torch.save(payload, victim)
    with pytest.raises(ShardMergeError, match="lora_tensor_metadata"):
        merge_lora_shards(tmp_path)


def test_module_on_wrong_stage_fails_closed(tmp_path):
    """某 stage 带上不属于它的层 ⇒ 拒（层归属与 placement 不符）。"""
    full = _full_lora(4)
    ranges = {0: (0, 1), 1: (2, 3)}
    _write_shards(tmp_path, full, pp_size=2, layer_ranges=ranges, strategy="manual")
    victim = tmp_path / "rank_00001_tp_00_pp_01.pt"
    payload = torch.load(victim, map_location="cpu", weights_only=False)
    entry = full["layers.0.mlp.gate_proj"]
    payload["lora_state_dict"]["layers.0.mlp.gate_proj.lora_a"] = entry["a"].clone()
    payload["lora_state_dict"]["layers.0.mlp.gate_proj.lora_b"] = entry["b"].clone()
    payload["lora_tensor_metadata"].append(
        {
            "module_name": "layers.0.mlp.gate_proj",
            "shard_kind": "out",
            "row_start": 0,
            "row_stop": INTER,
            "col_start": None,
            "col_stop": None,
            "row_indices": None,
        }
    )
    torch.save(payload, victim)
    with pytest.raises(ShardMergeError, match="不属于它的层"):
        merge_lora_shards(tmp_path)


def test_manifest_missing_or_bad_fails_closed(tmp_path):
    (tmp_path / "rank_00000_tp_00_pp_00.pt").write_bytes(b"x")
    with pytest.raises(ShardMergeError, match="manifest.json"):
        merge_lora_shards(tmp_path)


def test_topology_mismatch_fails_closed(tmp_path):
    """manifest 说 tp=2，实际只有 1 个分片 ⇒ 拒。"""
    full = _full_lora(1)
    _write_shards(tmp_path, full, tp_size=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["tp_size"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShardMergeError, match="分片集合不完整"):
        merge_lora_shards(tmp_path)


# ── 15~：纯函数层（不触张量）───────────────────────────────────────────────


def test_parse_shard_name_rejects_unknown_pattern():
    assert parse_shard_name("rank_00003_tp_01_pp_00.pt") == (3, 1, 0)
    with pytest.raises(ShardMergeError, match="命名规则"):
        parse_shard_name("rank_3_tp_1_pp_0.pt")


def test_expected_rank_filenames_matches_disk_naming(tmp_path):
    layout = ParallelLayout(tp_size=2, dp_size=2, pp_size=2)
    names = expected_rank_filenames(layout)
    assert len(names) == 8 and layout.world_size == 8
    assert names["rank_00005_tp_01_pp_00.pt"] == (5, 1, 0)
    full = _full_lora(2)
    _write_shards(
        tmp_path,
        full,
        tp_size=2,
        dp_size=2,
        pp_size=2,
        layer_ranges={0: (0,), 1: (1,)},
        strategy="manual",
    )
    assert set(names) == {r.path.name for r in discover_shards(tmp_path)}
    assert_shard_set_complete(discover_shards(tmp_path), layout)


def test_native_module_to_peft_key_rejects_unknown_prefix():
    assert native_module_to_peft_key("visual.blocks.0.attn.qkv", "b").endswith(
        "model.visual.blocks.0.attn.qkv.lora_B.weight"
    )
    with pytest.raises(ShardMergeError, match="前缀未知"):
        native_module_to_peft_key("audio.tower.0", "a")


def test_receipt_is_json_serializable(tmp_path):
    full = _full_lora(2)
    _write_shards(tmp_path, full, tp_size=2)
    _, receipt = merge_lora_shards(tmp_path)
    payload = receipt.as_dict()
    assert json.dumps(payload, ensure_ascii=False)
    assert "过渡" in payload["policy"] or "ulp" in payload["policy"]
    assert merge_receipt_summary(receipt)


def test_golden_single_shard_matches_legacy_conversion(tmp_path):
    """★ 回归锁：``dp=tp=pp=1`` 且**只含语言塔**时，产物必须与旧通道**逐位相同**。

    旧通道 = ``merged_export.convert_native_rank_to_peft`` 的单分片分支。这里端到端
    跑一遍，比对 ``adapter_model.safetensors`` 的 **sha256**（字节级），锁住"旧路径
    一个字节都没动"。（视觉塔键的差异见下一个用例——那是旧通道的缺陷修正。）
    """
    from safetensors.torch import save_file

    from graspo.eval.merged_export import convert_native_rank_to_peft

    full = {k: v for k, v in _full_lora(3).items() if not k.startswith("visual.")}
    _write_shards(tmp_path / "ckpt", full)
    legacy = convert_native_rank_to_peft(
        tmp_path / "ckpt", tmp_path / "legacy", base_model_path="/models/base"
    )
    merged, _ = merge_lora_shards(tmp_path / "ckpt")
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    save_file(
        {k: v.detach().to(torch.float32).contiguous() for k, v in merged.items()},
        str(new_dir / "adapter_model.safetensors"),
    )
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert legacy["keys"] == len(merged)
    assert list(merged) == sorted(merged, key=list(merged).index)  # 键序 = 落盘顺序
    assert digest(tmp_path / "legacy" / "adapter_model.safetensors") == digest(
        new_dir / "adapter_model.safetensors"
    )


def test_visual_tower_key_uses_hf_module_path_not_double_prefix(tmp_path):
    """★ 旧通道缺陷：视觉塔键被拼成 ``…model.visual.visual.blocks…``（多一个 ``visual``）。

    分片元数据里现成有 ``hf_module_path = model.visual.blocks.0.attn.qkv``，合并器用它
    ⇒ 键名对得上基座。本用例把"旧通道错、新通道对"同时钉住，防止回退。
    """
    from graspo.eval.merged_export import convert_native_rank_to_peft

    base = _full_lora(1)
    full = {
        "visual.blocks.0.attn.qkv": base["visual.blocks.0.attn.qkv"],
        "layers.0.mlp.gate_proj": base["layers.0.mlp.gate_proj"],
    }
    _write_shards(tmp_path / "ckpt", full)
    payload = torch.load(
        tmp_path / "ckpt" / "rank_00000_tp_00_pp_00.pt", map_location="cpu", weights_only=False
    )
    for meta in payload["lora_tensor_metadata"]:
        if meta["module_name"].startswith("visual."):
            meta["hf_module_path"] = f"model.{meta['module_name']}"
    torch.save(payload, tmp_path / "ckpt" / "rank_00000_tp_00_pp_00.pt")
    merged, receipt = merge_lora_shards(tmp_path / "ckpt")
    assert "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight" in merged
    assert receipt.modules == 2
    legacy_dir = tmp_path / "legacy"
    convert_native_rank_to_peft(tmp_path / "ckpt", legacy_dir, base_model_path="/models/base")
    from safetensors.torch import load_file

    legacy_keys = set(load_file(str(legacy_dir / "adapter_model.safetensors")))
    assert any("visual.visual." in k for k in legacy_keys)  # 旧通道确实错
    assert not any("visual.visual." in k for k in merged)  # 新通道不再错


# ── 通道选择（`convert_native_checkpoint_to_peft`）────────────────────────────


def test_dispatcher_routes_single_shard_to_legacy_channel(tmp_path):
    """`tp=dp=pp=1` ⇒ 走旧通道，产物与直接调旧通道**逐字节相同**。"""
    from graspo.eval.merged_export import (
        convert_native_checkpoint_to_peft,
        convert_native_rank_to_peft,
    )

    full = {k: v for k, v in _full_lora(2).items() if not k.startswith("visual.")}
    _write_shards(tmp_path / "ckpt", full)
    info = convert_native_checkpoint_to_peft(
        tmp_path / "ckpt", tmp_path / "a1", base_model_path="/models/base"
    )
    convert_native_rank_to_peft(tmp_path / "ckpt", tmp_path / "a2", base_model_path="/models/base")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert info["channel"] == "single-shard" and info["receipt"] is None
    assert digest(tmp_path / "a1" / "adapter_model.safetensors") == digest(
        tmp_path / "a2" / "adapter_model.safetensors"
    )


def test_dispatcher_sharded_channel_and_receipt(tmp_path):
    """`tp=2` ⇒ 走分片合并通道，回执可序列化、键数守恒。"""
    from graspo.eval.merged_export import convert_native_checkpoint_to_peft

    full = _full_lora(2)
    _write_shards(tmp_path / "ckpt", full, tp_size=2)
    info = convert_native_checkpoint_to_peft(
        tmp_path / "ckpt", tmp_path / "adapter", base_model_path="/models/base"
    )
    assert info["channel"] == "sharded-merge"
    assert info["keys"] == 2 * len(full)
    assert json.dumps(info["receipt"], ensure_ascii=False)
    assert (
        "sharded"
        in json.loads((tmp_path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))[
            "_note"
        ]
        or "分片"
        in json.loads((tmp_path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))[
            "_note"
        ]
    )


def test_dispatcher_sharded_channel_fails_closed_as_export_error(tmp_path):
    """fail-closed 必须以 `ExportError` 形式冒到 CLI 层（评测链靠它中止）。"""
    from graspo.eval.merged_export import ExportError, convert_native_checkpoint_to_peft

    def corrupt(module, a, b, meta, tp_rank):
        if meta["shard_kind"] == "none" and tp_rank == 1:
            a = a + 0.5
        return a, b, meta

    full = _full_lora(1)
    _write_shards(tmp_path / "ckpt", full, tp_size=2, corrupt=corrupt)
    with pytest.raises(ExportError, match="分片不可合并"):
        convert_native_checkpoint_to_peft(
            tmp_path / "ckpt", tmp_path / "adapter", base_model_path="/models/base"
        )


def test_audit_reports_drift_by_class_without_producing_weights(tmp_path):
    """`audit_shards` 只报告、不产出权重；漂移按 (shard_kind, matrix) 归类给分位。"""

    def corrupt(module, a, b, meta, tp_rank):
        if meta["shard_kind"] == "none" and tp_rank == 1:
            a = a + 0.5
        return a, b, meta

    from graspo.eval.sharded_merge import audit_shards

    full = _full_lora(1)
    _write_shards(tmp_path / "ckpt", full, tp_size=2, corrupt=corrupt)
    report = audit_shards(tmp_path / "ckpt")
    assert report["ok"] is False  # 有超限复制张量 ⇒ 这份分片集不可导出
    bucket = report["drift_by_class"]["none.lora_a"]
    assert bucket["n_not_bitwise"] >= 1 and bucket["max_ulp"] > 1
    assert set(report["drift_by_class"]) >= {"none.lora_a", "out.lora_a", "in.lora_b"}


def test_hf_module_path_wins_over_native_alias_names():
    """native 模块名有内部别名（`token_mixer` → HF `linear_attn`）：以 `hf_module_path` 为准。"""
    key = native_module_to_peft_key(
        "layers.0.token_mixer.in_proj_qkv",
        "b",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
    )
    assert key == (
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"
    )
    with pytest.raises(ShardMergeError, match="没有层号"):
        native_module_to_peft_key(
            "layers.7.mlp.up_proj", "a", "model.language_model.layers.3.mlp.up_proj"
        )


def test_alpha_comes_from_metadata_not_derived_from_r(tmp_path):
    """★ native scaling = alpha/r（实测 r=32、alpha=64 ⇒ 2）。adapter_config 必须记真 alpha。"""
    from graspo.eval.merged_export import convert_native_checkpoint_to_peft

    full = _full_lora(1)
    _write_shards(tmp_path / "ckpt", full, tp_size=2)
    info = convert_native_checkpoint_to_peft(
        tmp_path / "ckpt", tmp_path / "adapter", base_model_path="/models/base"
    )
    cfg = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["r"] == 2 and cfg["lora_alpha"] == 4  # 元数据的 alpha=4 ≠ r=2
    assert info["receipt"]["lora_alpha"] == 4
