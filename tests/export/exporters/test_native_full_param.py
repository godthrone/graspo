"""F7（native 全参单卡）→ ``merged-hf``：端到端 + 三类自校验 + **融合顺序独立判据**。

覆盖设计 P5 的三项自校验：
① 产物与 ckpt 张量**逐元素一致**（含融合与 dtype 回落）；
② 产物键集合**覆盖基座 state_dict**；
③（实验阶段在 228 上另做）产物可被 ``transformers`` 实际加载。

★ 每条检查都带**能失败的阳性对照**（宪法 §11.4）—— 尤其是融合顺序：
   名称覆盖 / 形状 / "产物 == ckpt" 三项**都查不出** ``cat([k,q,v])`` 这类语义错配，
   只有"native 块对基座行段的最近邻指派"能查出来，因此它必须有独立的负例。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import (
    full_param_states,
    make_export_config,
    write_hf_full_param_base,
    write_native_full_param_checkpoint,
)

from graspo.export import export_checkpoint
from graspo.export.errors import ExportError
from graspo.export.exporters import native_full_param as nfp

_CONV1D = "model.language_model.layers.0.linear_attn.conv1d.weight"
_QKV = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
_A_LOG = "model.language_model.layers.0.linear_attn.A_log"
_LAYER_NORM = "model.language_model.layers.0.linear_attn.norm.weight"
_MTP = ("mtp.fc.weight", "mtp.norm.weight")


def _stage(tmp_path, *, mutate_native=None, mutate_base=None, shards=1, **state_kwargs):
    """构造 (native ckpt, HF base, native_state, base_state) 四元组。"""
    native_state, base_state = full_param_states(**state_kwargs)
    if mutate_native is not None:
        mutate_native(native_state)
    if mutate_base is not None:
        mutate_base(base_state)
    checkpoint = write_native_full_param_checkpoint(tmp_path / "final", native_state)
    base = write_hf_full_param_base(tmp_path / "base", base_state, shards=shards)
    return checkpoint, base, native_state, base_state


def _read_produced(directory: Path) -> dict:
    from safetensors.torch import load_file

    state: dict = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        state.update(load_file(str(path), device="cpu"))
    return state


def _export(tmp_path, checkpoint, base, **kwargs):
    output = tmp_path / "merged"
    config = make_export_config(checkpoint, output, fmt="merged-hf", **kwargs)
    return export_checkpoint(config, base_model_path=str(base)), output


# ══════════════════════════════════════════════════════════════════════════════
# 端到端 + 自校验 ① / ②
# ══════════════════════════════════════════════════════════════════════════════


def test_full_param_export_produces_a_merged_hf(tmp_path):
    checkpoint, base, native_state, base_state = _stage(tmp_path)
    result, output = _export(tmp_path, checkpoint, base)

    assert (output / "config.json").is_file()
    assert (output / "generation_config.json").is_file(), "base sidecar files must be copied"
    assert (output / "export_manifest.json").is_file()
    assert len(result.consumed_shards) == 1

    produced = _read_produced(output)
    # ② 覆盖基座 state_dict（含 mtp.*）。
    assert set(produced) == set(base_state)

    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["form"] == "native-full-param-rank"
    assert manifest["produced_keys"] == len(base_state)
    assert manifest["base_copied_keys"] == sorted(_MTP), (
        "the 15/2 mtp.* tensors are absent from the native checkpoint and MUST be recorded as "
        "explicitly copied from the base"
    )
    assert manifest["mapping_fusions"] == {
        _QKV: [
            "layers.0.token_mixer.q_proj.weight",
            "layers.0.token_mixer.k_proj.weight",
            "layers.0.token_mixer.v_proj.weight",
        ]
    }
    assert manifest["dtype_casts"] == {
        _A_LOG: "bfloat16->float32",
        _LAYER_NORM: "bfloat16->float32",
    }
    assert "/home/" not in json.dumps(manifest)


def test_exported_weights_equal_checkpoint_values_not_base(tmp_path):
    """① 逐元素一致；且证明**不是**把基座值写进去（夹具 base != ckpt）。"""
    import torch

    checkpoint, base, native_state, base_state = _stage(tmp_path)
    _, output = _export(tmp_path, checkpoint, base)
    produced = _read_produced(output)

    for hf_key, base_tensor in base_state.items():
        source = nfp.hf_source_plan(hf_key)
        if source is None or source.from_base or source.fused:
            continue  # 融合项与"从基座复制"项单独覆盖
        native_key = source.native_keys[0]
        expected = native_state[native_key].to(produced[hf_key].dtype)
        assert torch.equal(produced[hf_key], expected), hf_key

    # 阳性对照：夹具的基座值与 ckpt 值确实不同 ⇒ "写基座值"这条错路会被上面抓住。
    assert not torch.equal(produced[_CONV1D], base_state[_CONV1D])
    assert not torch.equal(produced[_CONV1D].float(), base_state[_CONV1D].float())


def test_fused_in_proj_qkv_is_q_then_k_then_v(tmp_path):
    import torch

    checkpoint, base, native_state, _ = _stage(tmp_path)
    _, output = _export(tmp_path, checkpoint, base)
    produced = _read_produced(output)

    prefix = "layers.0.token_mixer."
    expected = torch.cat(
        [
            native_state[prefix + "q_proj.weight"],
            native_state[prefix + "k_proj.weight"],
            native_state[prefix + "v_proj.weight"],
        ],
        dim=0,
    )
    assert torch.equal(produced[_QKV], expected)


def test_dtype_fallback_upcasts_to_base_dtype(tmp_path):
    import torch

    checkpoint, base, native_state, _ = _stage(tmp_path)
    _, output = _export(tmp_path, checkpoint, base)
    produced = _read_produced(output)

    assert produced[_A_LOG].dtype == torch.float32
    assert torch.equal(produced[_A_LOG], native_state["layers.0.token_mixer.A_log"].float())
    assert produced[_LAYER_NORM].dtype == torch.float32


def test_sharded_base_keeps_its_sharding_layout(tmp_path):
    checkpoint, base, _, base_state = _stage(tmp_path, shards=3)
    _, output = _export(tmp_path, checkpoint, base)

    assert (output / "model.safetensors.index.json").is_file()
    assert len(list(output.glob("model-*.safetensors"))) == 3
    assert set(_read_produced(output)) == set(base_state)


# ══════════════════════════════════════════════════════════════════════════════
# ★ 融合顺序 / 行段切分的**独立**判据（含能失败的负例）
# ══════════════════════════════════════════════════════════════════════════════


def test_fused_block_order_checker_detects_wrong_assignment():
    """直接单测判别函数：同尺寸块换序 ⇒ 逐块最近邻与排列判据**都**变红。"""
    import torch

    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    c = torch.randn(2, 4)
    base_true = torch.cat([a, b, c], dim=0) + 0.01 * torch.randn(6, 4)

    assignment, _ = nfp.fused_block_assignment(base_true, [a, b, c])
    assert assignment == [0, 1, 2]
    assert nfp.fused_permutation_deviations(base_true, [a, b, c])[(0, 1, 2)] == pytest.approx(
        min(nfp.fused_permutation_deviations(base_true, [a, b, c]).values())
    )

    assignment, deviations = nfp.fused_block_assignment(base_true, [b, a, c])
    assert assignment == [1, 0, 2], "swapping q/k rows must move the nearest-segment assignment"
    permutation = nfp.fused_permutation_deviations(base_true, [b, a, c])
    assert permutation[(1, 0, 2)] < permutation[(0, 1, 2)]
    # 同尺寸块（a vs c，均为 2 行）之间的偏差可分辨，不是碰巧。
    assert deviations[0][1] is not None and deviations[0][2] is not None
    assert deviations[0][1] < deviations[0][2]


def test_fused_permutation_catches_cross_size_misorder():
    """排列判据额外覆盖"跨不同行数的段被换序"（``[q, v, k]``）——逐块最近邻查不出。"""
    import torch

    q = torch.randn(2, 4)
    k = torch.randn(2, 4)
    v = torch.randn(4, 4)
    base_true = torch.cat([q, k, v], dim=0) + 0.01 * torch.randn(8, 4)

    permutation = nfp.fused_permutation_deviations(base_true, [q, v, k])
    assert permutation[(0, 1, 2)] > permutation[(0, 2, 1)], (
        "declared [q, v, k] must NOT be the minimum-deviation permutation"
    )


def test_positive_control_swapped_qk_blocks_fail_closed(tmp_path):
    """★ 负例：只把 native 的 q/k 两块**换序**（名称、形状、覆盖、"产物==ckpt" 全部照旧）。

    这种情况下唯一能红的检查就是融合顺序判据；导出必须 fail-closed。
    """

    def swap_qk(native_state):
        prefix = "layers.0.token_mixer."
        q = native_state[prefix + "q_proj.weight"]
        k = native_state[prefix + "k_proj.weight"]
        assert q.shape == k.shape
        native_state[prefix + "q_proj.weight"], native_state[prefix + "k_proj.weight"] = k, q

    checkpoint, base, _, _ = _stage(tmp_path, mutate_native=swap_qk)
    with pytest.raises(ExportError, match="block-order check FAILED"):
        _export(tmp_path, checkpoint, base)


def test_fused_check_reported_matrix_is_part_of_the_error(tmp_path):
    """失败消息必须带偏差矩阵（可人工复核），而不是一句笼统的"失败"。"""

    def swap_qk(native_state):
        prefix = "layers.0.token_mixer."
        a = native_state[prefix + "q_proj.weight"]
        b = native_state[prefix + "k_proj.weight"]
        native_state[prefix + "q_proj.weight"], native_state[prefix + "k_proj.weight"] = b, a

    checkpoint, base, _, _ = _stage(tmp_path, mutate_native=swap_qk)
    with pytest.raises(ExportError) as excinfo:
        _export(tmp_path, checkpoint, base)
    message = str(excinfo.value)
    assert "Relative-L1 matrix" in message and "expected [0, 1, 2]" in message


# ══════════════════════════════════════════════════════════════════════════════
# 边界与阳性对照
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("shape_kwargs", [{"dp_size": 2}, {"tp_size": 2}, {"pp_size": 2}])
def test_multi_shard_topologies_stay_fail_closed(tmp_path, shape_kwargs):
    native_state, base_state = full_param_states()
    checkpoint = write_native_full_param_checkpoint(
        tmp_path / "final", native_state, **shape_kwargs
    )
    base = write_hf_full_param_base(tmp_path / "base", base_state)
    with pytest.raises(ExportError, match="single self-contained shard"):
        _export(tmp_path, checkpoint, base)


def test_unknown_base_key_is_fail_closed(tmp_path):
    def add_unknown(base_state):
        base_state["mystery.weight"] = base_state["model.language_model.norm.weight"].clone()

    checkpoint, base, _, _ = _stage(tmp_path, mutate_base=add_unknown)
    with pytest.raises(ExportError, match="no native→HF mapping"):
        _export(tmp_path, checkpoint, base)


def test_orphan_native_key_is_fail_closed(tmp_path):
    def add_orphan(native_state):
        native_state["layers.0.token_mixer.bogus.weight"] = native_state[
            "layers.0.token_mixer.out_proj.weight"
        ]

    checkpoint, base, _, _ = _stage(tmp_path, mutate_native=add_orphan)
    with pytest.raises(ExportError, match="map to no base key"):
        _export(tmp_path, checkpoint, base)


def test_shape_mismatch_is_fail_closed(tmp_path):
    def break_shape(base_state):
        base_state[_QKV] = base_state[_QKV][:-1]

    checkpoint, base, _, _ = _stage(tmp_path, mutate_base=break_shape)
    with pytest.raises(ExportError, match="shape mismatch|does not match the base layout"):
        _export(tmp_path, checkpoint, base)


def test_undeclared_dtype_change_is_fail_closed(tmp_path):
    import torch

    def break_dtype(native_state):
        native_state["layers.0.token_mixer.dt_bias"] = native_state[
            "layers.0.token_mixer.dt_bias"
        ].float()

    checkpoint, base, _, _ = _stage(tmp_path, mutate_native=break_dtype)
    assert torch is not None
    with pytest.raises(ExportError, match="dtype mismatch"):
        _export(tmp_path, checkpoint, base)


def test_positive_control_base_substitution_is_caught(tmp_path, monkeypatch):
    """★ 负例：把某个已写出的权重**换成基座值**（模拟"静默导出未训练基座"）。

    "产物 == ckpt 值"这条自校验必须把它抓住。
    """
    from safetensors.torch import load_file, save_file

    original_write = nfp.NativeFullParamRankExporter._write

    def tampering_write(self, destination, assembled, plan):
        written = original_write(self, destination, assembled, plan)
        base_dir = Path(str(plan.base_model_path))
        produced_path = destination / assembled.base_files[0]
        base_path = base_dir / assembled.base_files[0]
        state = dict(load_file(str(produced_path), device="cpu"))
        state[_CONV1D] = load_file(str(base_path), device="cpu")[_CONV1D]
        save_file(state, str(produced_path))
        return written

    monkeypatch.setattr(nfp.NativeFullParamRankExporter, "_write", tampering_write)
    checkpoint, base, _, _ = _stage(tmp_path)
    with pytest.raises(ExportError, match="NOT element-wise equal"):
        _export(tmp_path, checkpoint, base)


def test_native_shard_residue_is_refused(tmp_path, monkeypatch):
    """V16：产物目录内出现 native ``.pt`` 残留 ⇒ 拒绝（positive control 由人为放入驱动）。"""
    original_write = nfp.NativeFullParamRankExporter._write

    def residue_write(self, destination, assembled, plan):
        written = original_write(self, destination, assembled, plan)
        (destination / "rank_00000_tp_00_pp_00.pt").write_bytes(b"residue")
        return written

    monkeypatch.setattr(nfp.NativeFullParamRankExporter, "_write", residue_write)
    checkpoint, base, _, _ = _stage(tmp_path)
    with pytest.raises(ExportError, match="native shard residue"):
        _export(tmp_path, checkpoint, base)


def test_peft_adapter_target_is_not_supported(tmp_path):
    from graspo.export.registry import resolve
    from graspo.export.schema import CheckpointForm, ExportTarget

    with pytest.raises(ExportError, match="cannot be exported as 'peft-adapter'"):
        resolve(CheckpointForm.NATIVE_FULL_PARAM_RANK, ExportTarget.PEFT_ADAPTER)
