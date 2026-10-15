"""``tests/export/`` 的合成夹具构造器（纯 FS + 合成 ``.pt``；不碰 GPU、不读真实权重）。

夹具原则（宪法 §11.4）：每个检查都必须能用这里的夹具构造出**使其失败的阳性对照**；
因此夹具提供 ``divergent`` / 缺分片 / 缺 manifest / 重叠 PP 等可切换形态。

（``tests/export/conftest.py`` 会把本目录加入 ``sys.path``，故各用例直接 ``import helpers``。）
"""

from __future__ import annotations

import json
from pathlib import Path


def lora_records(*, n: int = 2, r: int = 2, in_features: int = 4, out_features: int = 4):
    """构造 n 个 LoRA 目标记录（字段与 ``transformer_model_adapter`` 写出的一致）。"""
    records = []
    for index in range(n):
        name = f"model.layers.{index}.mlp.down_proj"
        records.append(
            {
                "module_name": f"layers.{index}.mlp.down_proj",
                "lora_a_name": f"layers.{index}.mlp.down_proj.lora_a",
                "lora_b_name": f"layers.{index}.mlp.down_proj.lora_b",
                "target_name": f"layers.{index}.mlp.down_proj",
                "hf_module_path": name,
                "base_weight_name": f"{name}.weight",
                "shard_kind": "none",
                "row_start": None,
                "row_stop": None,
                "col_start": None,
                "col_stop": None,
                "row_indices": None,
                "peft_exportable": True,
                "r": r,
                "alpha": 2 * r,
            }
        )
    return records


def lora_state(records, *, offset: float = 0.0):
    """构造与 records 对应的 ``lora_state_dict``（B@A 非零 ⇒ delta 非平凡）。"""
    import torch

    state = {}
    for index, record in enumerate(records):
        r = int(record["r"])
        a = torch.arange(r * 4, dtype=torch.float32).reshape(r, 4) / 10.0 + float(offset)
        b = torch.ones(4, r, dtype=torch.float32) * (1.0 + 0.1 * index)
        state[record["lora_a_name"]] = a
        state[record["lora_b_name"]] = b
    return state


def placement_for(*, pp_rank: int, pp_size: int, layer_count: int = 4, drop_interior: bool = False):
    per_stage = max(layer_count // pp_size, 1)
    start = pp_rank * per_stage
    stop = min(start + per_stage, layer_count)
    indices = list(range(start, stop))
    # 造出一个**中间层缺口**（stage0 少掉自己的最后一层）——V6 的并集连续性判据可检出。
    if drop_interior and pp_rank == 0 and len(indices) > 1:
        indices = indices[:-1]
    return {
        "placement_strategy": "test_tp",
        "model_family": "test",
        "tp_size": 1,
        "tp_rank": 0,
        "pp_size": pp_size,
        "pp_rank": pp_rank,
        "local_layer_indices": indices,
        "include_embeddings": pp_rank == 0,
        "include_lm_head": pp_rank == pp_size - 1,
    }


def write_native_checkpoint(
    root: Path,
    *,
    dp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    fmt: str = "native-lora",
    tuner_type: str = "lora",
    divergent: bool = False,
    gap_in_pp: bool = False,
    manifest_placement: dict | None = None,
    write_manifest: bool = True,
    write_shards: bool = True,
    n_records: int = 2,
) -> Path:
    """写一个合成 native checkpoint（``manifest.json`` + ``rank_*_tp_*_pp_*.pt``）。"""
    import torch

    root.mkdir(parents=True, exist_ok=True)
    records = lora_records(n=n_records)
    if write_manifest:
        manifest = {
            "format": fmt,
            "tuner_type": tuner_type,
            "dp_size": dp_size,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "world_size": dp_size * tp_size * pp_size,
            "placement": manifest_placement
            if manifest_placement is not None
            else placement_for(pp_rank=0, pp_size=pp_size),
            "lora_target_signature": {},
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if not write_shards:
        return root

    rank = 0
    for dp in range(dp_size):
        for tp in range(tp_size):
            for pp in range(pp_size):
                offset = 0.0 if not divergent or dp == 0 else 0.01 * dp
                payload = {
                    "adapter": "native",
                    "tuner_type": tuner_type,
                    "rank": rank,
                    "dp_rank": dp,
                    "dp_size": dp_size,
                    "tp_rank": tp,
                    "tp_size": tp_size,
                    "pp_rank": pp,
                    "pp_size": pp_size,
                    "placement": placement_for(
                        pp_rank=pp, pp_size=pp_size, drop_interior=gap_in_pp
                    ),
                    "lora_target_signature": {},
                    "lora_tensor_metadata": records,
                    "lora_state_dict": lora_state(records, offset=offset),
                    "full_param_state_dict": (
                        {record["base_weight_name"]: None for record in records}
                        if tuner_type == "full"
                        else None
                    ),
                    "config": {"model": {"model_path": "models/test-base"}, "lora": {}},
                }
                torch.save(payload, root / f"rank_{rank:05d}_tp_{tp:02d}_pp_{pp:02d}.pt")
                rank += 1
    return root


def write_hf_base(root: Path, *, n_records: int = 2) -> Path:
    """写一个合成 HF 基座目录（``config.json`` + ``model.safetensors``，形状与 delta 一致）。"""
    import json as _json

    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    records = lora_records(n=n_records)
    (root / "config.json").write_text(
        _json.dumps({"model_type": "test", "architectures": ["TestForCausalLM"]}),
        encoding="utf-8",
    )
    state = {record["base_weight_name"]: torch.zeros(4, 4) for record in records}
    save_file(state, str(root / "model.safetensors"))
    return root


def make_export_config(
    checkpoint: Path, output: Path, *, fmt: str = "peft-adapter", allow_overwrite=False
):
    from graspo.core.schema import ExportConfig

    return ExportConfig(
        checkpoint_path=str(checkpoint),
        export_output=str(output),
        export_format=fmt,
        allow_overwrite=allow_overwrite,
    )


# ══════════════════════════════════════════════════════════════════════════════
# F7：native 全参单卡夹具（合成小模型；覆盖前缀/改名/融合/dtype 回落/mtp 复制）
# ══════════════════════════════════════════════════════════════════════════════


def full_param_states(
    *,
    hidden: int = 4,
    vocab: int = 6,
    key_dim: int = 2,
    value_dim: int = 4,
    head_dim: int = 2,
    visual_dim: int = 3,
    linear_layers: tuple[int, ...] = (0,),
    full_layers: tuple[int, ...] = (1,),
    seed: int = 0,
):
    """构造自洽的一对 ``(native full-param state, HF base state)`` 小夹具。

    覆盖全部四类变换：
    (A) 前缀两层（``model.language_model.`` / ``model.visual.`` / 顶层 ``lm_head``）；
    (B) ``token_mixer`` → ``linear_attn``（linear 层）/ ``self_attn``（full 层）
        + ``conv1d_weight`` 改名；
    (B-fuse) ``in_proj_qkv = cat([q, k, v], dim=0)``；
    (C) 48 个 dtype 回落的最小代表：``A_log`` / ``linear_attn.norm.weight`` 基座 F32、native bf16；
    外加 ``mtp.*``（基座独有 ⇒ 必须显式复制）。

    ★ native 值与基座值**故意不同** ⇒ 可用于证明"导出写的是 ckpt 值，不是基座值"。
    """
    import torch

    gen = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen)

    def bf(tensor):
        return tensor.to(torch.bfloat16)

    native: dict = {}
    base: dict = {}

    def put(native_key, base_key, native_tensor, base_tensor):
        native[native_key] = native_tensor
        base[base_key] = base_tensor

    put(
        "embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        bf(rn(vocab, hidden)),
        bf(rn(vocab, hidden)),
    )
    put(
        "visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.weight",
        bf(rn(visual_dim, visual_dim, 2, 2)),
        bf(rn(visual_dim, visual_dim, 2, 2)),
    )
    put(
        "visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        bf(rn(3 * visual_dim, visual_dim)),
        bf(rn(3 * visual_dim, visual_dim)),
    )

    for index in sorted(set(linear_layers) | set(full_layers)):
        native_prefix = f"layers.{index}."
        hf_prefix = f"model.language_model.layers.{index}."
        put(
            native_prefix + "input_layernorm.weight",
            hf_prefix + "input_layernorm.weight",
            bf(rn(hidden)),
            bf(rn(hidden)),
        )
        put(
            native_prefix + "post_attention_layernorm.weight",
            hf_prefix + "post_attention_layernorm.weight",
            bf(rn(hidden)),
            bf(rn(hidden)),
        )
        for proj, shape in (
            ("gate_proj", (2 * hidden, hidden)),
            ("up_proj", (2 * hidden, hidden)),
            ("down_proj", (hidden, 2 * hidden)),
        ):
            put(
                native_prefix + f"mlp.{proj}.weight",
                hf_prefix + f"mlp.{proj}.weight",
                bf(rn(*shape)),
                bf(rn(*shape)),
            )

        mixer = native_prefix + "token_mixer."
        if index in linear_layers:
            # 融合三块（故意的 [q|k|v] 顺序）。基座侧是**整张** in_proj_qkv，且构造为
            # "native 各块 + 小扰动" —— 模拟"基座是 ckpt 的来源、ckpt = 基座 + 训练漂移"，
            # 从而 `fused_block_assignment` 的最近邻判据在本夹具上**唯一且稳健**。
            q_block = bf(rn(key_dim, hidden))
            k_block = bf(rn(key_dim, hidden))
            v_block = bf(rn(value_dim, hidden))
            native[mixer + "q_proj.weight"] = q_block
            native[mixer + "k_proj.weight"] = k_block
            native[mixer + "v_proj.weight"] = v_block
            drift_scale = 0.05
            base[hf_prefix + "linear_attn.in_proj_qkv.weight"] = torch.cat(
                [
                    q_block + drift_scale * bf(rn(key_dim, hidden)),
                    k_block + drift_scale * bf(rn(key_dim, hidden)),
                    v_block + drift_scale * bf(rn(value_dim, hidden)),
                ],
                dim=0,
            )
            put(
                mixer + "conv1d_weight",
                hf_prefix + "linear_attn.conv1d.weight",
                bf(rn(2 * key_dim + value_dim, 1, 2)),
                bf(rn(2 * key_dim + value_dim, 1, 2)),
            )
            put(
                mixer + "in_proj_z.weight",
                hf_prefix + "linear_attn.in_proj_z.weight",
                bf(rn(hidden, hidden)),
                bf(rn(hidden, hidden)),
            )
            put(
                mixer + "in_proj_b.weight",
                hf_prefix + "linear_attn.in_proj_b.weight",
                bf(rn(value_dim, hidden)),
                bf(rn(value_dim, hidden)),
            )
            put(
                mixer + "in_proj_a.weight",
                hf_prefix + "linear_attn.in_proj_a.weight",
                bf(rn(value_dim, hidden)),
                bf(rn(value_dim, hidden)),
            )
            put(
                mixer + "dt_bias",
                hf_prefix + "linear_attn.dt_bias",
                bf(rn(value_dim)),
                bf(rn(value_dim)),
            )
            # ★ dtype 回落：基座 F32、native bf16（与真实 H9 ckpt 的 48 个键同形）。
            put(
                mixer + "A_log",
                hf_prefix + "linear_attn.A_log",
                bf(rn(value_dim)),
                rn(value_dim),
            )
            put(
                mixer + "norm.weight",
                hf_prefix + "linear_attn.norm.weight",
                bf(rn(head_dim)),
                rn(head_dim),
            )
            put(
                mixer + "out_proj.weight",
                hf_prefix + "linear_attn.out_proj.weight",
                bf(rn(hidden, value_dim)),
                bf(rn(hidden, value_dim)),
            )
        if index in full_layers:
            for name, shape in (
                ("q_proj", (2 * hidden, hidden)),
                ("k_proj", (key_dim, hidden)),
                ("v_proj", (key_dim, hidden)),
                ("o_proj", (hidden, 2 * hidden)),
                ("q_norm", (head_dim,)),
                ("k_norm", (head_dim,)),
            ):
                put(
                    mixer + f"{name}.weight",
                    hf_prefix + f"self_attn.{name}.weight",
                    bf(rn(*shape)),
                    bf(rn(*shape)),
                )

    put("norm.weight", "model.language_model.norm.weight", bf(rn(hidden)), bf(rn(hidden)))
    put("lm_head.weight", "lm_head.weight", bf(rn(vocab, hidden)), bf(rn(vocab, hidden)))

    # mtp.*：native 不建 ⇒ 基座独有，导出必须**显式复制**。
    base["mtp.fc.weight"] = bf(rn(hidden, 2 * hidden))
    base["mtp.norm.weight"] = bf(rn(hidden))
    return native, base


def write_native_full_param_checkpoint(
    root: Path,
    native_state: dict | None = None,
    *,
    dp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    layer_indices: tuple[int, ...] = (0, 1),
    **_state_kwargs,
) -> Path:
    """写一个合成 native **全参** checkpoint（``manifest.json`` + 单个 ``rank_*.pt``）。"""
    import torch

    if native_state is None:
        native_state, _ = full_param_states(**_state_kwargs)

    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "native-full-param",
        "tuner_type": "full",
        "tp_size": tp_size,
        "dp_size": dp_size,
        "pp_size": pp_size,
        "placement": {
            "placement_strategy": "test_tp",
            "model_family": "qwen3_5_text",
            "tp_size": tp_size,
            "tp_rank": 0,
            "pp_size": pp_size,
            "pp_rank": 0,
            "local_layer_indices": list(layer_indices),
            "include_embeddings": True,
            "include_lm_head": True,
        },
        "lora_target_signature": {"resolved": [], "parameter_count": 0},
        "world_size": dp_size * tp_size * pp_size,
        "checkpoint_type": "recoverable_full_param_training_state",
        "has_trainer_state": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "adapter": "native",
        "tuner_type": "full",
        "rank": 0,
        "dp_rank": 0,
        "dp_size": dp_size,
        "tp_rank": 0,
        "tp_size": tp_size,
        "pp_rank": 0,
        "pp_size": pp_size,
        "placement": manifest["placement"],
        "lora_target_signature": manifest["lora_target_signature"],
        "lora_tensor_metadata": [],
        "lora_state_dict": {},
        "full_param_state_dict": native_state,
        "optimizer_state_dict": None,
        "config": {"model": {"model_path": "models/test-base"}, "lora": {}},
    }
    torch.save(payload, root / "rank_00000_tp_00_pp_00.pt")
    return root


def write_hf_full_param_base(
    root: Path, base_state: dict | None = None, *, shards: int = 1, **_state_kwargs
) -> Path:
    """写一个合成 HF 基座目录（``config.json`` + safetensors 分片 + index，可选）。"""
    import torch  # noqa: F401  (仅确保 torch 可用)
    from safetensors.torch import save_file

    if base_state is None:
        _, base_state = full_param_states(**_state_kwargs)

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
        ),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text("{}\n", encoding="utf-8")

    keys = sorted(base_state)
    if shards <= 1:
        save_file({key: base_state[key] for key in keys}, str(root / "model.safetensors"))
        return root

    per_shard = {}
    for index, key in enumerate(keys):
        per_shard.setdefault(index % shards, {})[key] = base_state[key]
    weight_map = {}
    for index in range(shards):
        filename = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        save_file(per_shard[index], str(root / filename))
        for key in per_shard[index]:
            weight_map[key] = filename
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": weight_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return root
