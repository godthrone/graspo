"""TP 复制类 LoRA 的梯度同步 / 范数域（缺陷①②的回归钉子）。

**背景（2026-10-28 实测，远端机上 tp=2 单步 smoke）**：

* 缺陷①：``shard_kind="none"``（整块复制的视觉塔 LoRA）**没有任何跨 rank 同步**——
  ``_sync_nonsharded_lora_grads`` 只有 ``rows``/``out``（同步 lora_a）与 ``in``
  （同步 lora_b）两个分支，``none`` 被 if/elif 静默漏掉；``dp_size=1`` 时
  ``_sync_dp_lora_grads`` 又整体短路 ⇒ 两个 rank 各走各的 Adam。
  实测：检查点里 ``kind=none lora_b`` **0/110 逐位相同**，``max|Δ|=4.03e-05``，
  ``min cos=-0.065``（探测点证明两 rank 拿到的是**互不相同的部分梯度**）。
* 缺陷②：``clip_grad_norm_`` 用的是**本 rank 分片**的范数（复制类各算一份、切片类各占
  一半）⇒ 两个 rank 得到**不同标量**并乘到**同一份**（已 SUM 归约、逐位相同的）梯度上。
  实测：``clip_total_norm`` rank0=0.707031250 / rank1=1.078125000 ⇒ coef 1.0 vs 0.927535；
  逐元素验证 ``kind=in lora_b`` 的 rank1 梯度 = bf16(rank0 × 标量) 命中 131072/131072。

本文件锁的是**机制**（都不需要 GPU / 真权重）：用假 ``all_reduce`` 模拟 tp=2 的
SUM 语义——"每个 rank 只持有部分梯度，求和才是全量"——验证

1. 覆盖范围由 :data:`TP_REPLICATED_LORA_MATRICES` 单点决定，``none`` 类 a、b 都进同步；
2. 未登记的 ``shard_kind`` fail-closed（⛔ 不静默跳过）；
3. tp 组为 None 时 0 次集合（单卡行为不变）；
4. DP 组同步的覆盖范围不因本修复收窄（仍覆盖全部 LoRA 矩阵，含 none 类）；
5. 梯度范数分桶：复制类只计一次、切片类跨 TP 求和 ⇒ 两个 rank 拿到**同一个**全模型
   范数 ⇒ clip 系数逐 rank 相同（缺陷② 的修法）；
6. tp=1 侧行为不变的**数值**证据：老 ``clip_grad_norm_`` 与新不变范数相对差 ≤ 2e-3
   （同一个全模型范数，只差 bf16 累加量化；逐位相同做不到）。
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch.distributed as dist
from torch import nn

from graspo.flow.backend.native.lora.lora_linear import (
    TP_REPLICATED_LORA_MATRICES,
    LoRALinear,
    _iter_lora_modules,
    _replace_visual_lora_modules,
    _sync_dp_lora_grads,
    _sync_nonsharded_lora_grads,
    _tp_replicated_lora_params,
    clip_grads_tp_invariant,
    lora_grad_norm_sq_parts,
    tp_invariant_grad_norm_sq,
)

_CPU = torch.device("cpu")


def _mk_lora(
    kind: str,
    *,
    out_features: int = 8,
    in_features: int = 6,
    r: int = 2,
    hf_module_path: str = "model.language_model.layers.0.q_proj",
) -> LoRALinear:
    """造一个真 ``LoRALinear``（CPU/float32，无 dropout），只关心 shard_kind 语义。"""
    weight = torch.randn(out_features, in_features)
    return LoRALinear(
        weight,
        None,
        lora_enabled=True,
        r=r,
        alpha=2 * r,
        dropout=0.0,
        device=_CPU,
        dtype=torch.float32,
        target_name=hf_module_path.rsplit(".", 1)[-1],
        hf_module_path=hf_module_path,
        shard_kind=kind,
    )


class _Tower(nn.Module):
    """最小模型树：每种 shard_kind 各一层 + 一个非 LoRA 线性层（范数分桶用）。"""

    def __init__(self, *, kinds: tuple[str, ...] = ("none", "out", "rows", "in")) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _mk_lora(kind, hf_module_path=f"model.language_model.layers.{idx}.q_proj")
                for idx, kind in enumerate(kinds)
            ]
        )
        self.visual_head = nn.Linear(6, 4, bias=False)


def _fill_grads(model: nn.Module, *, seed: int) -> None:
    """给每个可训参数灌一份"部分梯度"（不同 seed ⇒ 模拟不同 rank 的不同部分）。"""
    generator = torch.Generator().manual_seed(seed)
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn(param.shape, generator=generator)


class _FakeTpSum:
    """假 ``all_reduce``：第 i 次集合把对端 rank 的同序张量加进来（= tp SUM 语义）。"""

    def __init__(self, peers: list[torch.Tensor], *, op: object = None) -> None:
        self.peers = list(peers)
        self.calls: list[tuple[tuple[int, ...], object]] = []
        self.op = op

    def __call__(self, tensor: torch.Tensor, op: object = None, group: object = None) -> None:
        index = len(self.calls)
        assert index < len(self.peers), f"集合调用次数超出预期（第 {index + 1} 次）"
        self.calls.append((tuple(tensor.shape), op))
        with torch.no_grad():
            tensor += self.peers[index]


@pytest.fixture
def fake_sum(monkeypatch):
    """安装假 all_reduce；返回 (install, 记录器)。

    ``install(peers)`` 装配对端张量并返回本次调用记录器。
    """

    def install(peers: list[torch.Tensor]) -> _FakeTpSum:
        fake = _FakeTpSum(peers)
        monkeypatch.setattr(dist, "all_reduce", fake)
        return fake

    return install


def _params_in_sync_order(model: nn.Module) -> list[torch.Tensor]:
    """按 ``_sync_nonsharded_lora_grads`` 的遍历顺序列出应参与集合的参数（本 rank 侧）。"""
    ordered: list[torch.Tensor] = []
    for module in _iter_lora_modules(model):
        for _letter, param in _tp_replicated_lora_params(module):
            ordered.append(param)
    return ordered


# ── ① 覆盖范围（唯一判据）────────────────────────────────────────────────────


def test_registry_covers_all_four_shard_kinds() -> None:
    """``TP_REPLICATED_LORA_MATRICES`` 必须登记全部已知 kind（新增 kind 必须同源登记）。"""
    assert set(TP_REPLICATED_LORA_MATRICES) == {"none", "out", "rows", "in"}
    assert TP_REPLICATED_LORA_MATRICES["none"] == ("a", "b")
    assert TP_REPLICATED_LORA_MATRICES["out"] == ("a",)
    assert TP_REPLICATED_LORA_MATRICES["rows"] == ("a",)
    assert TP_REPLICATED_LORA_MATRICES["in"] == ("b",)


def test_none_kind_lora_a_and_b_are_both_tp_synced(fake_sum) -> None:
    """缺陷①：``none`` 类的 a **和** b 都必须进 tp 集合（修复前 0 次、全被漏掉）。"""
    model = _Tower(kinds=("none",))
    _fill_grads(model, seed=1)
    module = model.blocks[0]
    peers = [
        torch.full_like(module.lora_a, 10.0),
        torch.full_like(module.lora_b, 20.0),
    ]
    recorder = fake_sum(peers)

    _sync_nonsharded_lora_grads(model, tp_group=object())

    assert [shape for shape, _op in recorder.calls] == [
        tuple(module.lora_a.shape),
        tuple(module.lora_b.shape),
    ]
    assert all(op == dist.ReduceOp.SUM for _shape, op in recorder.calls)
    # 求和不平均：部分梯度相加才是全量（AVG 会把量级砍半）。
    assert torch.all(module.lora_a.grad > 5.0)
    assert torch.all(module.lora_b.grad > 10.0)


def test_rows_out_sync_only_lora_a_and_in_syncs_only_lora_b(fake_sum) -> None:
    """切片类的口径不变：``rows``/``out`` 只同步 a，``in`` 只同步 b。"""
    model = _Tower(kinds=("out", "rows", "in"))
    _fill_grads(model, seed=2)
    out_mod, rows_mod, in_mod = model.blocks[0], model.blocks[1], model.blocks[2]
    before = {
        "out_b": out_mod.lora_b.grad.clone(),
        "rows_b": rows_mod.lora_b.grad.clone(),
        "in_a": in_mod.lora_a.grad.clone(),
    }
    peers = [torch.zeros_like(p) for p in _params_in_sync_order(model)]
    fake_sum(peers)

    _sync_nonsharded_lora_grads(model, tp_group=object())

    # 被同步的三个矩阵（out.a / rows.a / in.b）都变了；未被同步的三个逐位不动。
    assert torch.equal(out_mod.lora_b.grad, before["out_b"])
    assert torch.equal(rows_mod.lora_b.grad, before["rows_b"])
    assert torch.equal(in_mod.lora_a.grad, before["in_a"])


def test_unknown_shard_kind_fails_closed(fake_sum) -> None:
    """未登记的 shard_kind ⇒ 报错（修复前是静默跳过，正是缺陷① 的成因）。"""
    model = _Tower(kinds=("none",))
    model.blocks[0].lora_shard_kind = "column-parallel-v2"
    _fill_grads(model, seed=3)
    fake_sum([])  # 报错前不允许发生任何集合

    with pytest.raises(RuntimeError, match="TP_REPLICATED_LORA_MATRICES"):
        _sync_nonsharded_lora_grads(model, tp_group=object())


def test_tp_group_none_is_noop_with_zero_collectives(fake_sum) -> None:
    """tp 组为 None ⇒ 0 次集合（单卡路径不许多任何通信）。"""
    model = _Tower(kinds=("none", "rows"))
    _fill_grads(model, seed=4)
    recorder = fake_sum([])

    _sync_nonsharded_lora_grads(model, tp_group=None)

    assert recorder.calls == []


def test_two_ranks_converge_bitwise_after_none_sync(fake_sum) -> None:
    """两个 rank 各持一份部分梯度，各自 SUM 归约后必须**逐位相同**（缺陷① 的钉子）。"""
    rank0 = _Tower(kinds=("none",))
    rank1 = _Tower(kinds=("none",))
    _fill_grads(rank0, seed=13)
    _fill_grads(rank1, seed=14)
    pre0 = [p.grad.clone() for p in _params_in_sync_order(rank0)]
    pre1 = [p.grad.clone() for p in _params_in_sync_order(rank1)]
    assert len(pre0) == len(pre1) == 2

    fake_sum([p.clone() for p in pre1])
    _sync_nonsharded_lora_grads(rank0, tp_group=object())
    fake_sum([p.clone() for p in pre0])
    _sync_nonsharded_lora_grads(rank1, tp_group=object())

    for param0, param1 in zip(
        _params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True
    ):
        assert torch.equal(param0.grad, param1.grad)
        assert bool(param0.grad.isfinite().all())


def test_missing_grad_zero_fills_but_keeps_grad_none(fake_sum) -> None:
    """本 rank 无梯度 ⇒ 仍用同形状零张量参与集合（集合次数/形状必须两 rank 对齐）。"""
    model = _Tower(kinds=("none",))
    module = model.blocks[0]
    module.lora_a.grad = torch.ones_like(module.lora_a)
    module.lora_b.grad = None
    recorder = fake_sum([torch.zeros_like(module.lora_a), torch.ones_like(module.lora_b)])

    _sync_nonsharded_lora_grads(model, tp_group=object())

    assert len(recorder.calls) == 2
    assert module.lora_b.grad is None
    assert torch.all(module.lora_a.grad == 1.0)


def test_dp_group_sync_still_covers_every_lora_matrix(fake_sum) -> None:
    """DP 组同步的覆盖范围不因本修复收窄：仍是**全部** LoRA 矩阵（含 none 类），算子 AVG。"""
    model = _Tower(kinds=("none", "out", "rows", "in"))
    _fill_grads(model, seed=5)
    expected = sum(2 for _module in _iter_lora_modules(model))
    recorder = fake_sum([torch.zeros_like(p.grad) for p in _all_lora_params(model)])

    _sync_dp_lora_grads(model, dp_group=object())

    assert len(recorder.calls) == expected == 8
    assert all(op == dist.ReduceOp.AVG for _shape, op in recorder.calls)


def _all_lora_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in _iter_lora_modules(model):
        for param in (module.lora_a, module.lora_b):
            if param is not None:
                params.append(param)
    return params


def test_visual_tower_modules_are_registered_as_replicated() -> None:
    """缺陷① 的发生地：真 ``_replace_visual_lora_modules`` 产出的模块必须是 none 类
    （a、b 都复制 ⇒ 两个矩阵都要进 tp 同步）。"""

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.attn.qkv = nn.Linear(6, 8, bias=False)

    class _Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block()])
            self.merger = nn.Module()
            self.merger.linear_fc1 = nn.Linear(6, 8, bias=False)

    visual = _Visual()
    _replace_visual_lora_modules(
        visual,
        lora_targets={"model.visual.blocks.0.attn.qkv"},
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        device=_CPU,
        torch_dtype=torch.float32,
    )
    module = visual.blocks[0].attn.qkv
    assert isinstance(module, LoRALinear)
    assert module.lora_shard_kind == "none"
    assert [letter for letter, _param in _tp_replicated_lora_params(module)] == ["a", "b"]


# ── ② 范数域（clip 系数必须跨 rank 相同）────────────────────────────────────


def test_norm_sq_parts_split_replicated_and_sliced() -> None:
    """复制桶 = none a/b + out/rows a + in b；切片桶 = 其余 + 非 LoRA 权重。"""
    model = _Tower(kinds=("none", "out", "rows", "in"))
    _fill_grads(model, seed=6)
    trainable = [p for p in model.parameters() if p.requires_grad]

    replicated_sq, sliced_sq = lora_grad_norm_sq_parts(model, trainable)

    replicated_ids = {id(p) for p in _params_in_sync_order(model)}
    expected_replicated = sum(
        float(p.grad.detach().float().pow(2).sum()) for p in trainable if id(p) in replicated_ids
    )
    expected_sliced = sum(
        float(p.grad.detach().float().pow(2).sum())
        for p in trainable
        if id(p) not in replicated_ids
    )
    assert replicated_sq.item() == pytest.approx(expected_replicated, rel=1e-6)
    assert sliced_sq.item() == pytest.approx(expected_sliced, rel=1e-6)
    assert sliced_sq.item() > 0.0 and replicated_sq.item() > 0.0


def test_tp_invariant_norm_counts_replicated_once_and_sums_sliced(fake_sum) -> None:
    """复制类只计一次、切片类跨 TP 求和 ⇒ 两个 rank 得到**同一个**全模型范数平方。"""
    rank0 = _Tower(kinds=("none", "in"))
    rank1 = _Tower(kinds=("none", "in"))
    _fill_grads(rank0, seed=7)
    _fill_grads(rank1, seed=8)
    # 复制类：模拟"已 SUM 归约完成"⇒ 两 rank 逐位相同。
    for param0, param1 in zip(
        _params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True
    ):
        param1.grad = param0.grad.clone()
    trainable0 = [p for p in rank0.parameters() if p.requires_grad]
    trainable1 = [p for p in rank1.parameters() if p.requires_grad]

    rep0, sliced0 = lora_grad_norm_sq_parts(rank0, trainable0)
    rep1, sliced1 = lora_grad_norm_sq_parts(rank1, trainable1)
    assert torch.equal(rep0, rep1)

    fake_sum([sliced1.clone()])
    total_rank0 = tp_invariant_grad_norm_sq(rank0, trainable0, tp_group=object())
    total_rank1 = rep1 + sliced0 + sliced1  # rank1 视角：自己 + 对端

    assert total_rank0.item() == pytest.approx(total_rank1.item(), rel=1e-6)
    # 关键：SUM 而不是"Σ 本地"（后者会把复制类重复计数 tp 次）。
    naive = (rep0 + sliced0).item() + (rep1 + sliced1).item()
    assert naive > total_rank0.item()


def test_clip_coefficient_is_identical_across_ranks(fake_sum) -> None:
    """缺陷②：复制类梯度逐位相同，clip 后仍必须逐位相同（修复前各自乘不同标量）。"""
    rank0 = _Tower(kinds=("none",))
    rank1 = _Tower(kinds=("none",))
    _fill_grads(rank0, seed=9)
    _fill_grads(rank1, seed=10)
    for param0, param1 in zip(
        _params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True
    ):
        param1.grad = param0.grad.clone()  # 复制类：已 SUM 归约 ⇒ 逐位相同
    pre_rep = [p.grad.clone() for p in _params_in_sync_order(rank0)]
    # 切片类两 rank 各持一半：故意让"本地范数"相差 30 倍（修复前的分叉条件）。
    rank0.visual_head.weight.grad = torch.full_like(rank0.visual_head.weight, 0.1)
    rank1.visual_head.weight.grad = torch.full_like(rank1.visual_head.weight, 3.0)
    trainable0 = [p for p in rank0.parameters() if p.requires_grad]
    trainable1 = [p for p in rank1.parameters() if p.requires_grad]
    _rep0, sliced0 = lora_grad_norm_sq_parts(rank0, trainable0)
    _rep1, sliced1 = lora_grad_norm_sq_parts(rank1, trainable1)

    fake_sum([sliced1.clone()])
    norm0 = clip_grads_tp_invariant(rank0, trainable0, 1e-3, tp_group=object())
    fake_sum([sliced0.clone()])
    norm1 = clip_grads_tp_invariant(rank1, trainable1, 1e-3, tp_group=object())

    # ① 两 rank 拿到同一个全模型范数（因此同一个 clip 系数）。
    assert norm0.item() == pytest.approx(norm1.item(), rel=0)
    assert norm0.item() > 1e-3  # 确实触发了裁剪
    # ② 裁剪后复制类梯度仍逐位相同（同标量 × 同输入）。
    for param0, param1 in zip(
        _params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True
    ):
        assert torch.equal(param0.grad, param1.grad)
    # ③ 乘数 = 同一个标量（对裁剪前梯度的比值逐元素一致）。
    coef = min(1.0, 1e-3 / (norm0.item() + 1e-6))
    for param, before in zip(_params_in_sync_order(rank0), pre_rep, strict=True):
        assert torch.allclose(param.grad, before * coef, rtol=1e-6, atol=0.0)


def test_clip_is_noop_when_norm_below_threshold(fake_sum) -> None:
    """阈值语义不变：全局范数小于 max_grad_norm ⇒ 系数 1.0、梯度逐位不动。"""
    model = _Tower(kinds=("none",))
    _fill_grads(model, seed=11)
    for param in model.parameters():
        if param.requires_grad:
            param.grad.mul_(1e-6)
    trainable = [p for p in model.parameters() if p.requires_grad]
    snapshot = {id(p): p.grad.clone() for p in trainable}
    _rep, sliced = lora_grad_norm_sq_parts(model, trainable)
    fake_sum([sliced.clone()])

    norm = clip_grads_tp_invariant(model, trainable, 1.0, tp_group=object())

    assert norm.item() < 1.0
    for param in trainable:
        assert torch.equal(param.grad, snapshot[id(param)])


def test_tp_invariant_norm_sq_without_group_is_local(fake_sum) -> None:
    """tp 组为 None ⇒ 纯本地范数、0 次集合（供单卡复用；训练器 tp=1 不走本函数）。"""
    model = _Tower(kinds=("none", "in"))
    _fill_grads(model, seed=12)
    trainable = [p for p in model.parameters() if p.requires_grad]
    recorder = fake_sum([])

    total = tp_invariant_grad_norm_sq(model, trainable, tp_group=None)

    assert recorder.calls == []
    expected = sum(float(p.grad.detach().float().pow(2).sum()) for p in trainable)
    assert total.item() == pytest.approx(expected, rel=1e-6)


def test_tp1_legacy_clip_and_invariant_clip_agree_within_bf16_tolerance() -> None:
    """tp=1 侧"行为不变"的**数值**证据：老 `clip_grad_norm_` 与新不变范数只差累加精度。

    两者算的是**同一个**全模型范数，只是累加 dtype 不同（老路径 bf16 foreach，不变量走 fp32）。
    判据（指挥官裁定）：相对差 ≤ 2e-3；逐位相同做不到，因为 bf16 累范数本身有 ~1e-3 量化。
    """
    model = _Tower(kinds=("none", "out", "rows", "in"))
    _fill_grads(model, seed=21)
    model_bf16 = model.to(torch.bfloat16)
    params_bf16 = [p for p in model_bf16.parameters() if p.requires_grad]

    legacy = torch.nn.utils.clip_grad_norm_(params_bf16, 1e9)  # 阈值取大 ⇒ 只测范数、不动梯度
    invariant = clip_grads_tp_invariant(model_bf16, params_bf16, 1e9, tp_group=None)

    rel = abs(float(legacy) - float(invariant)) / float(invariant)
    assert rel <= 2e-3, (
        f"tp=1 老/新范数相对差 {rel:.3e} 超过 2e-3（legacy={legacy} new={invariant}）"
    )


# ── 对照实验（永远绿）：用修复前的口径跑同一对 rank，把缺陷钉成可执行证据 ─────


def _prefix_sync_nonsharded(model: nn.Module, tp_group: object) -> None:
    """照抄修复前（HEAD 41e96d4）的实现：只有 rows/out→a、in→b，``none`` 被静默漏掉。"""
    for module in _iter_lora_modules(model):
        kind = module.lora_shard_kind
        if kind in ("rows", "out"):
            dist.all_reduce(module.lora_a.grad, op=dist.ReduceOp.SUM, group=tp_group)
        elif kind == "in":
            dist.all_reduce(module.lora_b.grad, op=dist.ReduceOp.SUM, group=tp_group)


def test_prefix_sync_leaves_none_kind_divergent(fake_sum) -> None:
    """缺陷① 对照：旧实现下两个 rank 的 ``none`` 类梯度**不会**收敛（本修复的靶子）。"""
    rank0 = _Tower(kinds=("none",))
    rank1 = _Tower(kinds=("none",))
    _fill_grads(rank0, seed=21)
    _fill_grads(rank1, seed=22)
    fake_sum([])  # 旧实现：0 次集合
    _prefix_sync_nonsharded(rank0, tp_group=object())

    for param0, param1 in zip(
        _params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True
    ):
        assert not torch.equal(param0.grad, param1.grad)


def test_prefix_local_clip_diverges_where_fixed_clip_agrees(fake_sum) -> None:
    """缺陷② 对照：旧口径（各 rank 用本地 ``clip_grad_norm_``）分叉，新口径收敛。"""

    def build(local_slice_value: float):
        model = _Tower(kinds=("none",))
        _fill_grads(model, seed=31)
        model.visual_head.weight.grad = torch.full_like(model.visual_head.weight, local_slice_value)
        return model

    rank0 = build(0.1)
    rank1 = build(3.0)
    for param0, param1 in zip(
        _params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True
    ):
        param1.grad = param0.grad.clone()  # 复制类：已 SUM 归约 ⇒ 逐位相同

    # 旧口径：各自按本 rank 范数裁剪 ⇒ 同一个梯度被乘上两个不同标量。
    prefix0 = [p.grad.clone() for p in _params_in_sync_order(rank0)]
    torch.nn.utils.clip_grad_norm_([p for p in rank0.parameters() if p.requires_grad], 1e-3)
    torch.nn.utils.clip_grad_norm_([p for p in rank1.parameters() if p.requires_grad], 1e-3)
    diverged = [
        not torch.equal(p0.grad, p1.grad)
        for p0, p1 in zip(_params_in_sync_order(rank0), _params_in_sync_order(rank1), strict=True)
    ]
    assert any(diverged), "旧口径下两 rank 的复制类梯度应已分叉"

    # 新口径：全局不变量范数 ⇒ 同标量 ⇒ 逐位相同。
    rank0b, rank1b = build(0.1), build(3.0)
    for param0, param1 in zip(
        _params_in_sync_order(rank0b), _params_in_sync_order(rank1b), strict=True
    ):
        param1.grad = param0.grad.clone()
    trainable0 = [p for p in rank0b.parameters() if p.requires_grad]
    trainable1 = [p for p in rank1b.parameters() if p.requires_grad]
    _r0, sliced0 = lora_grad_norm_sq_parts(rank0b, trainable0)
    _r1, sliced1 = lora_grad_norm_sq_parts(rank1b, trainable1)
    fake_sum([sliced1.clone()])
    clip_grads_tp_invariant(rank0b, trainable0, 1e-3, tp_group=object())
    fake_sum([sliced0.clone()])
    clip_grads_tp_invariant(rank1b, trainable1, 1e-3, tp_group=object())
    for param0, param1 in zip(
        _params_in_sync_order(rank0b), _params_in_sync_order(rank1b), strict=True
    ):
        assert torch.equal(param0.grad, param1.grad)
    assert len(prefix0) == 2


# ── 调用点钉子：tp>1 必须走不变量裁剪，tp=1 必须保持旧路径（单卡逐字节不变）────


_SRC = (
    pathlib.Path(__file__).resolve().parents[5] / "src" / "graspo" / "flow" / "backend" / "native"
)


def _find_tp_guarded_calls(path: pathlib.Path, call_name: str) -> list[int]:
    """返回「在 test 里含 ``_TENSOR_PARALLEL_SIZE > 1`` 的 if 分支内调用 call_name」的行号。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "_TENSOR_PARALLEL_SIZE" not in test_src:
            continue
        body_only = ast.Module(body=node.body, type_ignores=[])
        for inner in ast.walk(body_only):
            if isinstance(inner, ast.Call):
                func = inner.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == call_name:
                    hits.append(inner.lineno)
    return hits


@pytest.mark.parametrize(
    ("relative_path", "call_name"),
    [
        ("models/qwen35_36/training_sft.py", "clip_grads_tp_invariant"),
        ("models/qwen35_36/training.py", "clip_grads_tp_invariant"),
        ("models/qwen35_36/training.py", "tp_invariant_grad_norm_sq"),
        ("models/qwen35_36/training_sft.py", "_sync_nonsharded_lora_grads"),
    ],
)
def test_tp_gt_1_call_sites_use_tp_invariant_paths(relative_path: str, call_name: str) -> None:
    """三个 clip 调用点＋两处同步调用点都必须落在 ``_TENSOR_PARALLEL_SIZE > 1`` 的守卫内。"""
    path = _SRC / relative_path
    assert _find_tp_guarded_calls(path, call_name), (
        f"{relative_path} 缺少 tp>1 守卫内的 {call_name}"
    )


def test_tp_eq_1_keeps_legacy_clip_path() -> None:
    """tp=1 分支必须仍是原 ``clip_grad_norm_``（golden：单卡产物逐字节不变）。"""
    for relative_path in ("models/qwen35_36/training_sft.py", "models/qwen35_36/training.py"):
        path = _SRC / relative_path
        source = path.read_text(encoding="utf-8")
        assert "clip_grad_norm_" in source
        assert _find_tp_guarded_calls(path, "clip_grad_norm_") == [], (
            f"{relative_path}: clip_grad_norm_ 不应出现在 tp>1 的守卫分支里"
        )
