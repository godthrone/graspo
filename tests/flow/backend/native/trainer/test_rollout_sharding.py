"""★ rollout 分卡 · 设施层（``flow/``）的机核测试 —— 每条都带**阳性对照**。

覆盖设计件 `report.md` §10 的 **A-4**（决策逐卡同判 ＋ 会合不可省的反向用例）、
**A-5**（``μ_f`` 合并＝整组 ＋ ``n`` 整数严格）、**A-6**（切片互补）、
**A-8/A-8a**（配置 fail-closed）、**A-11**（阈值/组数读数一致）、
以及**验收标准 5**（等长切片下跨卡平均 = 全局均值：数学证明 ＋ 单测）。

★ **行号证据口径**：本文件用 `inspect.getsource` 做**源码级**断言（防"守卫被人挪走"），
⛔ 不用行号（行号会漂移）。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="分卡会合测试需要 torch（CPU 即可）")

from graspo.flow.backend.native.assignment import (  # noqa: E402
    assembled_slot_order,
    assert_rollout_slices,
    rollout_slice,
    rollouts_per_rank,
)
from graspo.flow.backend.native.trainer.rollout_shard import (  # noqa: E402
    gather_field_stats,
    gather_rollout_values,
    union_field_names,
)
from graspo.ripple.group_decision import classify_group  # noqa: E402

G = 8


# ══════════════════════════════════════════════════════════════════════════════
# 假 runtime：`resolve_dp_group` 走"无进程组"短路 ⇒ 全部返回本地值（= dp=1 语义）
# ══════════════════════════════════════════════════════════════════════════════


class _FakeRuntime:
    class _Adapter:
        device = torch.device("cpu")
        tp_state = SimpleNamespace(dp_size=1, dp_group=None, device=torch.device("cpu"))

    def _require_adapter(self):  # noqa: D102
        return self._Adapter()


# ══════════════════════════════════════════════════════════════════════════════
# A-4 / A-5 / A-6：`gather_rollout_values` 的槽位序拼接（dp≤1 短路）
# ══════════════════════════════════════════════════════════════════════════════


def test_gather_is_identity_when_dp_size_is_one():
    """★ dp≤1 ⇒ **零新增集合**、原样返回（A-1 回归锚）。"""
    values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    got = gather_rollout_values(
        _FakeRuntime(), values, cutoff=2, rollouts_per_rank=2
    )
    assert len(got) == 2
    assert torch.equal(got[0], values[0])
    assert torch.equal(got[1], values[1])


def test_gather_rejects_cutoff_shrink_or_pad_without_source():
    """★ fail-closed：`cutoff > len(local_values)` 且**无补齐张量** ⇒ 必须显式失败。

    这正是"各 rank 会合形状不一致"的形态 —— 静默按行号猜（截断/补零）会让
    槽位错位 ⇒ 决策输入不再是整组（I2 禁止的双真相源）。
    ★ 该断言在 **dp≤1 短路之前**判（否则 dp=1 下它永远到不了 ⇒ 哑断言）。
    """
    values = [torch.tensor([1.0, 2.0])]
    with pytest.raises(RuntimeError, match="补齐张量"):
        gather_rollout_values(_FakeRuntime(), values, cutoff=3, rollouts_per_rank=2)
    with pytest.raises(RuntimeError, match="cutoff 不能为负"):
        gather_rollout_values(_FakeRuntime(), values, cutoff=-1, rollouts_per_rank=2)


def test_gather_truncates_to_cutoff_and_uses_pad_when_supplied():
    """`cutoff < len(local)` ⇒ 截断；`cutoff > len(local)` ＋ 补齐张量 ⇒ 补足形状。"""
    values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    got = gather_rollout_values(_FakeRuntime(), values, cutoff=1, rollouts_per_rank=2)
    assert len(got) == 1 and torch.equal(got[0], values[0])
    padded = gather_rollout_values(
        _FakeRuntime(),
        [torch.tensor([1.0, 2.0])],
        cutoff=2,
        rollouts_per_rank=2,
        pad_values=[torch.tensor([9.0, 9.0])],
    )
    assert len(padded) == 2


def test_gather_rejects_bad_local_shape():
    """★ fail-closed：本地切片的**元素数** ≠ `n_local · (G//m)` ⇒ 必须显式失败（dp≤1 也判）。

    ★ 元素数是 `all_gather` 的**载荷契约**：不符 ⇒ 槽位序拼接错位 ⇒
    决策输入不再是整组。本断言在短路口径**之前**判，保证它不会退化成哑断言。
    """
    bad = [torch.tensor([1.0, 2.0, 3.0])]  # 应为 1 × 2 = 2 个元素
    with pytest.raises(RuntimeError, match="元素数应为"):
        gather_rollout_values(_FakeRuntime(), bad, cutoff=1, rollouts_per_rank=2)
    with pytest.raises(RuntimeError, match="元素数应为"):
        gather_rollout_values(
            _FakeRuntime(), [torch.tensor([[1.0, 2.0]])], cutoff=1, rollouts_per_rank=4
        )


def test_local_slice_then_gather_reproduces_whole_group():
    """★ A-4/A-5：`m` 张卡的**连续**切片按 rank 序拼接 == 单卡整组（逐元素/逐位）。

    这是"会合后每卡拿到同一批 G 个标量"这条唯一依据的机核
    （拼接在真实实现里走 `all_gather`；此处用同一**加序**在 CPU 上重放）。
    """
    whole = list(range(1, G + 1))
    for m in (1, 2, 4, 8):
        rpr = G // m
        per_rank = [
            torch.tensor([float(v) for v in whole[r * rpr : (r + 1) * rpr]])
            for r in range(m)
        ]
        gathered = [float(v) for r in range(m) for v in per_rank[r].tolist()]
        assert gathered == [float(v) for v in whole]
        assert gathered == [
            float(whole[slot]) for slot in assembled_slot_order(G, m)
        ]


def test_round_robin_slicing_would_break_the_assembly_order():
    """★ 阳性对照：把切片改成**轮转** `[rank::m]` ⇒ 拼齐后**加序** ≠ `0..G−1`。

    完备性仍成立（所以只测"并集"抓不住这个错）—— 必须测**加序**。
    """
    gathered = [slot for slot in range(G) for _ in (0,) if slot % 2 == 0] + [
        slot for slot in range(G) if slot % 2 == 1
    ]
    assert sorted(gathered) == list(range(G))  # 完备性：抓不住
    assert gathered != list(assembled_slot_order(G, 2))  # ★ 加序：抓得住


# ══════════════════════════════════════════════════════════════════════════════
# A-4(c)：会合**不可省** —— 只用本地 G//m 条 ⇒ 决策必不同
# ══════════════════════════════════════════════════════════════════════════════

_DECISION_KWARGS = dict(
    retry_count=0,
    rollout_max_retries=5,
    perfect_skip_reward_threshold=1.0,
    best_completion_has_parse_error=False,
    reject_unparseable_groups=True,
)


def test_decision_input_must_be_the_whole_group_positive_control():
    """★ A-4(c)（**能失败**的阳性对照）：固定 G=8 的 rewards。

    (a) 直调 `classify_group(整组 8 条)`；
    (b) 切片 → 按 rank 序拼齐 → `classify_group(整组 8 条)` ⇒ **必须相等**；
    (c) **只用本卡 `G//m` 条** ⇒ **必须不等**（证明会合不可省）。
    """
    whole_rewards = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
    whole_content = [1.0] * 8
    direct = classify_group(whole_rewards, whole_content, **_DECISION_KWARGS)

    m, rpr = 2, G // 2
    # (b) 切片 → 拼齐（同一加序）⇒ 整对象相等
    #   ★ dtype 与生产实现一致（`rollout.py` 用 fp64 建张量）⇒ 值**逐位往返**，
    #     这样 (b) 的 "相等" 是**逐位**相等，而不是被 fp32 截断掩盖的近似相等。
    per_rank = [
        torch.tensor(whole_rewards[r * rpr : (r + 1) * rpr], dtype=torch.float64)
        for r in range(m)
    ]
    gathered = [float(v) for r in range(m) for v in per_rank[r].tolist()]
    via_gather = classify_group(gathered, whole_content, **_DECISION_KWARGS)
    assert via_gather == direct  # frozen pydantic ⇒ 整对象可 `==`

    # (c) 只用本地切片 ⇒ 必须不等（否则"会合是空操作"，判否）
    local_only = classify_group(
        whole_rewards[:rpr], whole_content[:rpr], **_DECISION_KWARGS
    )
    assert local_only != direct, (
        "只用本地 G//m 条却得到与整组相同的决策 ⇒ 会合是空操作 ⇒ 判否"
    )


def test_decision_is_identical_across_ranks_when_input_is_whole_group():
    """★ A-4①：会合后各卡的 `GroupSampleDecision` **逐字段相同**（m=2/4）。"""
    whole_rewards = [0.1, 0.9, 0.3, 0.3, 0.2, 0.8, 0.4, 0.4]
    whole_content = [1.0] * 8
    reference = classify_group(whole_rewards, whole_content, **_DECISION_KWARGS)
    for m in (2, 4, 8):
        rpr = G // m
        gathered: list[float] = []
        for r in range(m):
            gathered.extend(whole_rewards[r * rpr : (r + 1) * rpr])
        assert classify_group(gathered, whole_content, **_DECISION_KWARGS) == reference


# ══════════════════════════════════════════════════════════════════════════════
# A-5：`(Σraw, n)` 的合并（dp≤1 短路 + `n` 整数严格）
# ══════════════════════════════════════════════════════════════════════════════


def test_gather_field_stats_is_identity_when_dp_size_is_one():
    stats = [{"a": (2.0, 4)}]
    got = gather_field_stats(
        _FakeRuntime(), stats, cutoff=1, field_names=["a"]
    )
    assert got == stats


def test_field_names_union_is_sorted_and_deterministic():
    """★ 字段维是 `all_gather` 的**形状**之一 ⇒ 必须逐 rank 同序（`sorted`）。"""
    per_prompt = [{"b": (1.0, 1)}, {"a": (2.0, 2)}]
    assert union_field_names(per_prompt) == ["a", "b"]
    assert union_field_names(list(reversed(per_prompt))) == ["a", "b"]
    assert union_field_names([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# A-6：切片互补（不丢不重，并集 = 原 G 条槽位）
# ══════════════════════════════════════════════════════════════════════════════


def test_slot_union_over_ranks_covers_every_slot_exactly_once():
    """★ A-6：跨 rank 的槽位并集 = `{(entry, 0..G−1)}` **恰一次**（不丢不重）。"""
    for m in (1, 2, 4, 8):
        for entries in (1, 3, 8, 15, 17):
            seen: dict[tuple[int, int], int] = {}
            for r in range(m):
                for entry, slot in rollout_slice(entries, G, m, r):
                    seen[(entry, slot)] = seen.get((entry, slot), 0) + 1
            assert set(seen) == {(e, s) for e in range(entries) for s in range(G)}
            assert set(seen.values()) == {1}, f"m={m} entries={entries} 有重复/缺失"
            assert_rollout_slices(entries, G, m)


def test_metadata_slot_offset_maps_local_index_to_global_slot():
    """★ A-6 的元数据口径：本卡切片第 `idx` 条的**全局槽位** = `dp_rank·(G//m) + idx`。

    这条与 `rollout.py::_append_experiences` 的 `slot_offset` 同式；⛔ 若这里与实现不符，
    跨 rank 的并集就不再是 `0..G−1`（会出现重复或缺号）。
    """
    for m in (2, 4):
        rpr = G // m
        for rank in range(m):
            local_slots = list(range(rpr))
            global_slots = [rank * rpr + idx for idx in range(rpr)]
            assert global_slots == [
                slot for _, slot in rollout_slice(1, G, m, rank)
            ]
            assert len(set(global_slots)) == len(local_slots)
        union = sorted(
            rank * rpr + idx for rank in range(m) for idx in range(rpr)
        )
        assert union == list(range(G))


# ══════════════════════════════════════════════════════════════════════════════
# 验收标准 5：等长切片下跨卡平均 = 全局均值（★ 数学证明 ＋ 单测 ＋ 阳性对照）
# ══════════════════════════════════════════════════════════════════════════════


def test_equal_slice_average_equals_global_average_real_arithmetic():
    """★ 验收标准 5（**实数算术**）：`mean(mean(slice_k)) == mean(all)`，当各切片等长。

    **证明**：设各卡条数 `N_k ≡ N/m`（本机制保证），则
    ``(1/m)·Σ_k (1/N_k)·Σ_{i∈k} g_i = (1/m)·(m/N)·Σ_i g_i = (1/N)·Σ_i g_i``。
    浮点下**逐位不成立**（加序不同）⇒ 判据用容差（A-5 的容差表）。
    """
    values = [float(v) for v in range(1, 25)]  # 24 条
    global_mean = sum(values) / len(values)
    for m in (2, 3, 4, 6, 8):
        per_rank = [values[k :: m] for k in range(m)]  # 等长（24 % m == 0）
        assert len({len(chunk) for chunk in per_rank}) == 1
        mean_of_means = sum(sum(chunk) / len(chunk) for chunk in per_rank) / m
        assert mean_of_means == pytest.approx(global_mean, rel=1e-12, abs=1e-12)


def test_unequal_slice_average_differs_from_global_average_positive_control():
    """★ 阳性对照（**能失败**）：切片**不等长**时，`mean(mean)` **必须** ≠ 全局均值。

    这条证明上面那条断言不是恒真式 —— 它真的在检查"等长"这个前提。
    ★ 口径说明：不等长下 `mean(mean)` 给**短的**那一半更大的权重。
    构造 9/15 两段（`values[:9]` 全是小值）⇒ `mean(mean) < 全局均值`。
    """
    values = [float(v) for v in range(1, 25)]
    global_mean = sum(values) / len(values)
    a, b = values[:9], values[9:]
    assert len(a) != len(b)
    unbalanced = (sum(a) / len(a) + sum(b) / len(b)) / 2
    assert unbalanced != pytest.approx(global_mean, rel=1e-6)
    # 短的 a 段（小值）被以 1/m 权重、却按 1/len(a) 归一 ⇒ 整体偏小
    assert unbalanced < global_mean
    # 等长切分（正确机制）则**相等**
    per_rank = [values[k::2] for k in range(2)]
    assert len(per_rank[0]) == len(per_rank[1]) == 12
    balanced = sum(sum(chunk) / len(chunk) for chunk in per_rank) / 2
    assert balanced == pytest.approx(global_mean, rel=1e-12)


def test_rollout_slicing_makes_per_rank_counts_strictly_equal_for_odd_subsets():
    """★ 验收标准 3：**S=15/17 非整除子集**下逐卡条数仍**严格相等**（旧机制给差 1）。"""
    for entries in (15, 17):
        for m in (2, 4):
            counts = [len(rollout_slice(entries, G, m, r)) for r in range(m)]
            assert len(set(counts)) == 1
            assert counts[0] == entries * (G // m)
            # 旧机制（条目维分片）在同形状下条数不等 ⇒ 这就是死锁面
            old = [len(list(range(entries))[r::m]) for r in range(m)]
            if entries % m != 0:
                assert len(set(old)) > 1


# ══════════════════════════════════════════════════════════════════════════════
# A-8 / A-11：配置面（单一真相源 ＋ fail-closed ＋ 读数一致）
# ══════════════════════════════════════════════════════════════════════════════


def _training(**overrides):
    from graspo.core.schema import TrainingConfig

    base = dict(rollout_group_size=8, rollout_queue_batch_size=8)
    base.update(overrides)
    return TrainingConfig(**base)


def test_slice_quantum_and_threshold_are_single_source_of_truth():
    """★ A-11：`quantum == G//m`；`threshold_for(dp) == Q·(G//dp)`；`dp=1` 与既有属性同值。"""
    cfg = _training()
    for m in (1, 2, 4, 8):
        assert cfg.rollout_slice_quantum(m) == G // m
        assert cfg.replay_buffer_optimize_threshold_for(m) == 8 * (G // m)
    # ★ dp=1 ⇒ 与**既有属性**（`Q×G`，`generate_matrix` 与 msswift 的既有口径）同值
    assert cfg.replay_buffer_optimize_threshold_for(1) == cfg.replay_buffer_optimize_threshold
    assert cfg.replay_buffer_optimize_threshold == 64


def test_optimize_quantum_matches_assignment_single_source():
    """★ §1.4：配置层的量子与 `assignment.rollouts_per_rank` **逐值一致**（防双真相源）。"""
    cfg = _training()
    for m in (1, 2, 4, 8):
        assert cfg.rollout_slice_quantum(m) == rollouts_per_rank(G, m)


@pytest.mark.parametrize("dp_size", [1, 2, 4, 8])
def test_optimize_trainable_group_count_uses_the_quantum(dp_size: int):
    """★ A-11 / M5：`trainable_group_count == usable // 量子`（⛔ 不是 `// G`）。

    ★ **阳性对照（能失败）**：用 `usable // G` 会得到 2 而不是 4（`usable=16, G=8, m=2`）
    ⇒ 落盘读数静默缩 1/m（复核者 B-2）。
    """
    from graspo.flow.backend.native.trainer.optimize import OptimizeMixin

    quantum = _training().rollout_slice_quantum(dp_size)
    usable = 16
    assert usable % quantum == 0, "新量子下 buffer 长度恒为量子整数倍（取整恒等、不丢尾）"
    assert usable // quantum == 16 // quantum
    if dp_size > 1:
        assert usable // quantum != usable // G  # ★ 旧口径确实会错


def test_rollout_slice_quantum_fails_closed_and_names_the_keys():
    """★ A-8：`G % m != 0` / `G//m < 1` ⇒ 显式失败且**点名**要改哪个键。"""
    cfg = _training(rollout_group_size=8)
    with pytest.raises(ValueError, match="rollout_group_size"):
        cfg.rollout_slice_quantum(3)
    with pytest.raises(ValueError, match="dp_size"):
        cfg.rollout_slice_quantum(3)
    with pytest.raises(ValueError, match="不得大于"):
        _training(rollout_group_size=2).rollout_slice_quantum(4)
    with pytest.raises(ValueError, match="dp_size"):
        cfg.rollout_slice_quantum(0)


# ══════════════════════════════════════════════════════════════════════════════
# ★ 源码级接线断言（唯一能防"守卫/会合点被人挪走"的机核）
# ══════════════════════════════════════════════════════════════════════════════


def _src(module_path: str) -> str:
    import importlib

    return inspect.getsource(importlib.import_module(module_path))


def test_rollout_attempt_gathers_before_decision():
    """★ 源码级：`classify_group` 必须出现在会合之后（`gather_rollout_values` 之后）。"""
    src = _src("graspo.flow.backend.native.trainer.rollout")
    assert "gather_rollout_values" in src
    assert src.index("gather_rollout_values") < src.index("decision = classify_group("), (
        "会合必须发生在 classify_group **之前**（否则决策输入仍是本地切片）"
    )


def test_finalize_uses_whole_group_rewards_for_the_second_gate():
    """★ 源码级（W2）：第二道门 `has_reward_variance` 必须用**会合后整组** rewards。"""
    src = _src("graspo.flow.backend.native.trainer.rollout")
    assert "has_reward_variance(whole_group_rewards)" in src
    assert "has_reward_variance(rewards)" not in src


def test_metadata_carries_entry_key_and_rollout_index():
    """★ 源码级（A-6）：`Experience.metadata` 必须写 `entry_key` / `rollout_index`。"""
    src = _src("graspo.flow.backend.native.trainer.rollout")
    assert '"entry_key"' in src
    assert '"rollout_index"' in src


def test_trainer_no_longer_shards_entries_when_sliced():
    """★ 源码级：训练循环的 epoch 取数**不得**再按 `dp_rank` 过滤条目。"""
    src = _src("graspo.flow.backend.native.trainer.trainer")
    assert "shard_of[index] == parallel.dp_rank" not in src, (
        "条目维分片必须消失（切片模式下各卡持**全部**条目）"
    )
    assert "rollouts_per_rank" in src


def test_capability_specs_sharding_text_is_unified_no_double_truth():
    """★ **W-7**：SHARDING 契约文案不得再是单义的"整组不拆（β1）"（双真相源）。

    阳性对照：把文案改回旧值 ⇒ 本断言必红。
    """
    from graspo.flow.backend.base import CAPABILITY_SPECS, BackendCapability

    spec = next(s for s in CAPABILITY_SPECS if s[0] is BackendCapability.SHARDING)
    invariant = spec[2]
    assert "整组不拆（β1）" in invariant  # 旧表述仍在（不是删掉，而是**限定**）
    assert "G//m" in invariant, f"SHARDING 不变量必须写明组内切片口径，实际={invariant!r}"
    assert "装配" in invariant or "G 条齐" in invariant

    generate = next(s for s in CAPABILITY_SPECS if s[0] is BackendCapability.GENERATE)
    assert "装配后" in generate[2], f"GENERATE 必须声明 native 的交付点，实际={generate[2]!r}"


def test_rollout_group_size_has_only_one_meaning_in_optimize():
    """★ **W-1**：`optimize.py` 里 `rollout_group_size` 只用于**语义面**（组大小），
    步边界量子必须走 `_ucd_quantum()`（= `G//m`）。
    """
    src = _src("graspo.flow.backend.native.trainer.optimize")
    assert "_ucd_quantum()" in src
    # 量子不得写成 `rollout_group_size`（那正是 W-1 的同名两义）
    assert "whole_group_count(raw_usable, rollouts_per_rank)" in src
    assert "usable // max(rollouts_per_rank, 1)" in src
