"""TP 下 LoRA 初始化的切分语义（缺陷③的回归钉子）。

**背景（2026-10-28 实测）**：``LoRALinear.__init__`` 用**本地形状**跑
``nn.init.kaiming_uniform_``，而各 rank 用的是**同一个 init seed**（``_set_model_init_seed``，
与 rank 无关）⇒

* 列切分（``shard_kind="in"``）的 ``lora_a`` 在每个 rank 上都抽到**同一份** ``[r, in/tp]``
  ⇒ 逻辑矩阵是 ``[X | X]``：**有效秩砍半**（kaiming 界还随 fan_in 变，与单卡也不同）；
* 由于每层的抽样量随分片宽度变化，RNG 流与 tp=1 从第一个 ``in`` 层起就错位
  ⇒ **tp>1 与 tp=1 从 step 0 起训的不是同一个矩阵**（实测 ``max|Δ|`` 3.6e-01~4.1e-01）。

本文件锁的修法与验收（都不需要 GPU）：

1. ``rank0_shard != rank1_shard``（不再是重复抽样）；
2. ``concat(shards, dim=col_dim)`` 与 **tp=1 的整块矩阵逐位相同**（同一 seed、同一 fan_in 语义）；
3. 连续多层的 RNG 流**不因 TP 而错位**（全形状抽样 ⇒ 消费量与 tp=1 一致）；
4. 复制类（``none`` / ``out`` / ``rows`` 的 lora_a）在 rank 间仍逐位相同；
5. tp=1 路径**逐字节不变**（golden）：``tp_size=1`` 时抽样量与旧实现完全一致。
"""

from __future__ import annotations

import torch
from torch import nn

from graspo.flow.backend.native.lora.lora_linear import LoRALinear
from graspo.flow.backend.native.trainer.helpers import _set_model_init_seed

_OUT, _IN, _R, _SEED = 64, 32, 4, 42
_CPU = torch.device("cpu")


def _build(shard: str, *, tp_rank: int = 0, tp_size: int = 1, layers: int = 1) -> list[LoRALinear]:
    """用真 ``from_hf`` 建 layers 个 LoRA 层；每层同一 seed（模拟 ``_set_model_init_seed``）。"""
    modules: list[LoRALinear] = []
    for idx in range(layers):
        _set_model_init_seed(_SEED)
        modules.append(
            LoRALinear.from_hf(
                torch.zeros(_OUT, _IN),
                None,
                shard=shard,
                tp_rank=tp_rank,
                tp_size=tp_size,
                lora_enabled=True,
                r=_R,
                alpha=2 * _R,
                dropout=0.0,
                device=_CPU,
                dtype=torch.float32,
                target_name="out_proj",
                hf_module_path=f"model.language_model.layers.{idx}.out_proj",
            )
        )
    return modules


def test_in_shard_lora_a_is_a_partition_not_a_copy() -> None:
    """验收①②：两 rank 的列分片**不同**，且拼接后与 tp=1 整块**逐位相同**。"""
    full = _build("in", tp_size=1)[0].lora_a.detach()
    rank0 = _build("in", tp_rank=0, tp_size=2)[0].lora_a.detach()
    rank1 = _build("in", tp_rank=1, tp_size=2)[0].lora_a.detach()

    assert rank0.shape == rank1.shape == (_R, _IN // 2)
    assert full.shape == (_R, _IN)
    assert not torch.equal(rank0, rank1), "两 rank 抽到了同一份初始化（逻辑矩阵秩退化）"
    assert torch.equal(torch.cat([rank0, rank1], dim=1), full), "拼接后必须与单卡整块逐位相同"


def test_in_shard_stream_stays_aligned_across_layers() -> None:
    """验收③：连续多层（含前面的 in 层）后，RNG 流仍与 tp=1 对齐。"""
    full = _build("in", tp_size=1, layers=3)
    part0 = _build("in", tp_rank=0, tp_size=2, layers=3)
    part1 = _build("in", tp_rank=1, tp_size=2, layers=3)

    for idx, (f, p0, p1) in enumerate(zip(full, part0, part1, strict=True)):
        assert torch.equal(torch.cat([p0.lora_a, p1.lora_a], dim=1), f.lora_a), f"layer{idx} 流错位"


def _build_direct(shard_kind: str, *, out_rows: int = _OUT) -> LoRALinear:
    """按生产侧对 ``rows`` / ``none`` 的方式**直接构造**（不经 ``from_hf`` 的行/列切分）。"""
    _set_model_init_seed(_SEED)
    return LoRALinear(
        torch.zeros(out_rows, _IN),
        None,
        lora_enabled=True,
        r=_R,
        alpha=2 * _R,
        dropout=0.0,
        device=_CPU,
        dtype=torch.float32,
        target_name="q_proj",
        hf_module_path="model.language_model.layers.0.q_proj",
        shard_kind=shard_kind,
        row_indices=tuple(range(out_rows)) if shard_kind == "rows" else None,
    )


def test_replicated_lora_a_is_identical_on_all_ranks() -> None:
    """复制类的 ``lora_a`` 仍必须 rank 间逐位相同、且与 tp=1 的同一构造逐位相同。

    - ``out`` 走 ``from_hf``：三者的 ``lora_a`` 都是整宽 ``[r, in]``（全形状抽样 ⇒ 同值）；
    - ``rows`` / ``none`` 生产侧是**直接构造**（行按 head 选、视觉塔整块复制）⇒ 同 seed
      同形状 ⇒ 逐位相同，不受列切分逻辑影响。
    """
    full_out = _build("out", tp_size=1)[0].lora_a.detach()
    rank0_out = _build("out", tp_rank=0, tp_size=2)[0].lora_a.detach()
    rank1_out = _build("out", tp_rank=1, tp_size=2)[0].lora_a.detach()
    assert torch.equal(rank0_out, rank1_out)
    assert torch.equal(rank0_out, full_out)

    for shard_kind in ("rows", "none"):
        a = _build_direct(shard_kind).lora_a.detach()
        b = _build_direct(shard_kind, out_rows=_OUT // 2).lora_a.detach()
        assert torch.equal(a, b), f"{shard_kind}: 复制类 lora_a 必须与行数无关、逐位相同"


def test_tp1_init_is_byte_identical_to_legacy_path() -> None:
    """golden：``tp_size=1`` 的抽样结果必须与"老实现（直接对本地形状抽样）"逐位相同。"""
    _set_model_init_seed(_SEED)
    legacy_tensor = torch.empty(_R, _IN, device=_CPU, dtype=torch.float32)
    nn.init.kaiming_uniform_(legacy_tensor, a=5**0.5)

    module = _build("in", tp_size=1)[0]
    assert torch.equal(module.lora_a.detach(), legacy_tensor)


def test_lora_b_is_still_zero_initialised() -> None:
    """``lora_b`` 仍零初始化（LoRA 起点 δ=0，前向与基座逐位一致），不得被切分逻辑影响。"""
    for shard in ("in", "out", "rows", "none"):
        for tp_rank in (0, 1):
            module = _build(shard, tp_rank=tp_rank, tp_size=2)[0]
            assert torch.count_nonzero(module.lora_b) == 0, f"{shard}: lora_b 必须全零"
