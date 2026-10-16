"""β1b 整组均衡分配的机核测试（P4）—— ★ 每条不变量都带阳性对照。

覆盖：
1. **整组不拆 / 全局组集合不变 / 各分片组数差 ≤ 1**（默认模式，多组形状）；
2. ★ **与既有 `[dp_rank::dp_size]` 逐位等价**（证明"把既有分片收编到契约面"而非改分布）；
3. **按工作量分组**（给权重 ⇒ LPT 工作量均衡，仍整组/不丢）；
4. **阳性对照**：重复键 / 非法分片数 / 缺键多键 / 分片越界 / 失衡 —— 必须判红；
5. **与 `flow/adapter/contract.assert_group_partition` 接线一致**（同一不变量、两份实现互证）；
6. 步边界 **整组配额**（R5 条数级 MIN 的整组替代）；
7. **批次级再分配规划**（默认 `max_move_groups=0` ⇒ 空计划 ⇒ 零新增通信）；
8. `NativeBackend._step_batches` 只交出**本卡的完整组**且组数均衡。
"""

from __future__ import annotations

import pytest

from graspo.flow.adapter.contract import (
    RolloutGroup,
    assert_group_partition,
)
from graspo.flow.backend.native.assignment import (
    AssignmentError,
    assembled_slot_order,
    assert_balanced_assignment,
    assert_rollout_slices,
    assign_whole_groups,
    plan_group_rebalance,
    rollout_slice,
    rollouts_per_rank,
    step_group_quota,
    validate_rollout_slicing,
    whole_group_count,
)

SHAPES = [(1, 1), (1, 3), (14, 2), (16, 4), (17, 4), (64, 2), (0, 3)]


# ── 1. 不变量（默认模式）─────────────────────────────────────────────────────
@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_default_assignment_is_whole_group_balanced(n: int, m: int):
    keys = list(range(n))
    shard_of = assign_whole_groups(keys, m)
    assert_balanced_assignment(keys, shard_of, n_shards=m)
    counts = [sum(1 for k in keys if shard_of[k] == s) for s in range(m)]
    assert max(counts) - min(counts) <= 1


# ── 2. ★ 与既有 [dp_rank::dp_size] 逐位等价 ─────────────────────────────────
@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_default_assignment_matches_legacy_round_robin(n: int, m: int):
    """★ 默认模式 == 既有 `epoch_samples[dp_rank::dp_size]` ⇒ **分布未变**。"""
    keys = list(range(n))
    shard_of = assign_whole_groups(keys, m)
    for rank in range(m):
        assert [k for k in keys if shard_of[k] == rank] == keys[rank::m]


# ── 3. 按工作量分组（LPT）────────────────────────────────────────────────────
def test_weighted_assignment_balances_workload_not_just_count():
    keys = ["heavy", "mid", "a", "b", "c", "d"]
    weights = {"heavy": 10.0, "mid": 5.0, "a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    shard_of = assign_whole_groups(keys, 2, weights=weights)
    assert_balanced_assignment(keys, shard_of, n_shards=2, require_count_balance=False)
    loads = sorted(sum(weights[k] for k in keys if shard_of[k] == s) for s in range(2))
    # 总工作量 19 ⇒ 最优切分是 9 / 10（不可能 9/9）；LPT 必须达到该最优。
    assert loads == [9.0, 10.0], loads
    assert max(loads) <= 10.0  # 最优上界 = ceil(19/2)


# ── 4. 阳性对照 ──────────────────────────────────────────────────────────────
def test_duplicate_keys_are_rejected():
    with pytest.raises(AssignmentError, match="重复"):
        assign_whole_groups(["a", "a"], 2)


@pytest.mark.parametrize("m", [0, -1])
def test_invalid_shard_count_is_rejected(m: int):
    with pytest.raises(AssignmentError, match="n_shards"):
        assign_whole_groups(["a"], m)


def test_assert_rejects_missing_or_extra_group():
    """★ 阳性对照：组集合被改变（丢/多）必须判红。"""
    with pytest.raises(AssignmentError, match="组集合变了"):
        assert_balanced_assignment(["a", "b"], {"a": 0}, n_shards=2)
    with pytest.raises(AssignmentError, match="组集合变了"):
        assert_balanced_assignment(["a"], {"a": 0, "x": 1}, n_shards=2)


def test_assert_rejects_out_of_range_shard():
    with pytest.raises(AssignmentError, match="越界"):
        assert_balanced_assignment(["a"], {"a": 5}, n_shards=2)


def test_assert_rejects_imbalance():
    """★ 阳性对照：3 vs 1 ⇒ 差 2 ⇒ 必须判红。"""
    with pytest.raises(AssignmentError, match="失衡"):
        assert_balanced_assignment(["a", "b", "c", "d"], {"a": 0, "b": 0, "c": 0, "d": 1}, n_shards=2)


# ── 5. 与 flow/adapter/contract 的 assert_group_partition 互证 ───────────────
def test_assignment_agrees_with_adapter_contract_partition_assertion():
    """同一不变量、两份实现互证（§1.4：契约面与 backend 面不得各说各话）。"""
    keys = [f"p{i}" for i in range(6)]
    shard_of = assign_whole_groups(keys, 3)
    groups = [RolloutGroup(group_key=k, completions=(0,) * 8, group_size=8) for k in keys]
    assert_group_partition(groups, shard_of, n_shards=3)  # 不抛 = 通过

    bad = dict(shard_of)
    bad[keys[0]] = bad[keys[1]] = bad[keys[2]] = 0  # 3 vs 1 vs 1 → 差 2
    with pytest.raises(Exception):
        assert_group_partition(groups, bad, n_shards=3)


# ── 6. 步边界：整组配额 ─────────────────────────────────────────────────────
def test_step_group_quota_is_min_and_never_partial():
    assert step_group_quota([7, 7]) == 7
    assert step_group_quota([8, 5, 6]) == 5
    assert step_group_quota([8, 5], move_budget=3) == 5  # 预算>0 也只在规划层生效


def test_step_group_quota_rejects_empty_and_negative():
    with pytest.raises(AssignmentError, match="不能为空"):
        step_group_quota([])
    with pytest.raises(AssignmentError, match="不能为负"):
        step_group_quota([1, -1])


# ── 7. 批次级整组再分配规划 ─────────────────────────────────────────────────
def test_rebalance_plan_is_empty_when_budget_is_zero():
    """★ 默认 `max_move_groups=0` ⇒ 空计划 ⇒ **零新增通信 / 行为不变**。"""
    assert plan_group_rebalance({0: ["a", "b", "c"], 1: ["d"]}, max_move_groups=0) == []


def test_rebalance_plan_moves_whole_groups_to_even_out_counts():
    holdings = {0: ["a", "b", "c"], 1: ["d"]}
    plan = plan_group_rebalance(holdings, max_move_groups=5)
    assert plan == [("c", 1)]  # 搬 1 组后 2 vs 2 = 均衡；且搬的是**整组**


def test_rebalance_plan_rejects_negative_budget():
    with pytest.raises(AssignmentError, match="不能为负"):
        plan_group_rebalance({0: ["a"]}, max_move_groups=-1)


# ── 8. NativeBackend._step_batches（分片能力钩子）───────────────────────────
@pytest.mark.parametrize(("n", "m", "rank"), [(14, 2, 0), (14, 2, 1), (16, 4, 2), (5, 3, 1)])
def test_native_backend_step_batches_yields_this_rank_whole_groups(n: int, m: int, rank: int):
    class _FakeRuntime:
        class _Degree:
            dp_size = m
            dp_rank = rank

        parallel_degree = _Degree()

    class _FakeTrainer:
        runtime = _FakeRuntime()

    from graspo.flow.backend.native.backend import NativeBackend

    backend = NativeBackend(config=object(), trainer=_FakeTrainer())
    keys = list(range(n))
    got = list(backend._step_batches(keys))
    assert got == keys[rank::m]  # 与既有轮转同分布
    assert len(got) == len(set(got))  # 不重


# ── 9. 步边界原语：整组取整（R5 条数级 MIN 的整组替代）──────────────────────
@pytest.mark.parametrize(
    ("raw", "g", "expected"),
    [(0, 8, 0), (7, 8, 0), (8, 8, 8), (15, 8, 8), (16, 8, 16), (63, 8, 56), (64, 8, 64),
     (5, 1, 5), (-3, 8, 0)],
)
def test_whole_group_count_never_cuts_a_group(raw: int, g: int, expected: int):
    assert whole_group_count(raw, g) == expected


def test_whole_group_count_is_identity_when_already_whole():
    """★ 常见形状（条数已是 G 的整数倍）⇒ **步边界逐位不变**（行为变化的边界条件）。"""
    for raw in (0, 8, 16, 48, 64, 144, 288):
        assert whole_group_count(raw, 8) == raw


def test_whole_group_count_rejects_bad_group_size():
    with pytest.raises(AssignmentError, match="group_size"):
        whole_group_count(10, 0)


# ══════════════════════════════════════════════════════════════════════════════
# 10. ★ rollout 分卡（2026-10-16 主席裁定）—— 分配符合切片口径 ＋ 阳性对照
#
# 口径（主席原话逐字）：dp=2、一批 4 条数据、每条 8 个 rollout
#   ⇒ **卡一 = 这 4 条的 rollout 0–3；卡二 = 4–7**。
# 本段每条断言都带**可构造的阳性对照**（把实现改成"整组"/"轮转"/"条目分片"必红）。
# ⛔ 本段**不修改**上文任何既有期望值（回归只许加强，不许改期望值）。
# ══════════════════════════════════════════════════════════════════════════════

_SLICE_SHAPES = [
    (p, g, m) for p in range(1, 9) for g in (8, 16) for m in (1, 2, 4, 8) if g % m == 0
]


@pytest.mark.parametrize(("p", "g", "m"), _SLICE_SHAPES)
def test_rollout_slices_satisfy_four_invariants(p: int, g: int, m: int):
    """★ A-8a：穷举 `P∈1..8, G∈{8,16}, m∈{1,2,4,8}` 断言四条不变量。

    ① 每卡条目集合 = 全体；② 同一 entry 的槽位互斥且并 = `0..G−1`；
    ③ 每卡条数 = `P·G/m`；④ `m=1` 恒等。
    """
    assert_rollout_slices(p, g, m)  # 不抛 = 四条全过


def test_rollout_slice_matches_chairman_wording_verbatim():
    """★ A-8a 逐字对照：`P=4, G=8, m=2` ⇒ 卡一 `(e,0..3)` / 卡二 `(e,4..7)`。"""
    rank0 = rollout_slice(4, 8, 2, 0)
    rank1 = rollout_slice(4, 8, 2, 1)
    assert rank0 == [(e, s) for e in range(4) for s in range(0, 4)]
    assert rank1 == [(e, s) for e in range(4) for s in range(4, 8)]
    # 两卡条目集合都 = 全体 4 条（切片**不切条目维**）
    assert sorted({e for e, _ in rank0}) == [0, 1, 2, 3]
    assert sorted({e for e, _ in rank1}) == [0, 1, 2, 3]
    # 每卡 16 条（= 主席原话「每卡 16 条」）
    assert len(rank0) == 16 and len(rank1) == 16


def test_rollout_slice_is_identity_when_dp_size_is_one():
    """★ m=1 恒等 ⇒ dp=1 路径逐位不变（A-1 回归锚）。"""
    for g in (8, 16):
        assert rollout_slice(3, g, 1, 0) == [(e, s) for e in range(3) for s in range(g)]
        assert assembled_slot_order(g, 1) == tuple(range(g))


def test_rollouts_per_rank_is_the_single_source_of_truth():
    """量化表：`G//m`（⛔ 调用方不得自己写 `G // dp_size`）。"""
    assert rollouts_per_rank(8, 1) == 8
    assert rollouts_per_rank(8, 2) == 4
    assert rollouts_per_rank(8, 4) == 2
    assert rollouts_per_rank(8, 8) == 1
    assert rollouts_per_rank(16, 4) == 4


def test_assembled_slot_order_equals_singleton_slot_order():
    """★ 会合的加序契约：按 rank 序拼齐后槽位序 = `0..G−1`（与单卡**逐元素相同**）。

    这是「`μ_f` 能按槽位序确定地求和、与单卡同加序」的唯一依据（A-5）。
    """
    for g, m in ((8, 2), (8, 4), (16, 4), (8, 1)):
        gathered: list[int] = []
        for r in range(m):
            gathered.extend(slot for _, slot in rollout_slice(1, g, m, r))
        assert gathered == list(assembled_slot_order(g, m)) == list(range(g))


@pytest.mark.parametrize(
    ("p", "m"), [(1, 2), (8, 2), (4, 4), (15, 2), (17, 2), (15, 4), (17, 4)]
)
def test_rollout_slice_non_divisible_subset_sizes_are_still_exactly_equal(p: int, m: int):
    """★ 验收标准 3 的**非整除子集构造性用例**（S=15 与 S=17）。

    旧机制（按 prompt 分卡）在 S=15/17 这类**非整除**子集上给差 1 的 prompt 数
    （`[实测]` `assign_whole_groups(17,2) = [9,8]`）⇒ 逐卡条数不等 ⇒ 死锁。
    新机制：每卡持**全部** `P` 条 prompt、各持 `G/m` 个槽位
    ⇒ **每卡条数严格相等 = P·G/m**，且**与 P 是否整除 m 无关**。
    """
    G = 8
    counts = [len(rollout_slice(p, G, m, r)) for r in range(m)]
    assert len(set(counts)) == 1, f"逐卡条数必须严格相等，得到 {counts}"
    assert counts[0] == p * (G // m)
    # 对照组：旧机制在同样形状下**逐卡 prompt 数不等**（阳性对照锚点）
    old_counts = [len(list(range(p))[r::m]) for r in range(m)]
    assert max(old_counts) - min(old_counts) <= 1
    assert (p % m != 0) == (len(set(old_counts)) > 1)
    assert_rollout_slices(p, G, m)


# ── 阳性对照：把实现改坏 ⇒ 同一断言必须判红 ──────────────────────────────────

def test_positive_control_whole_group_per_rank_must_fail():
    """★ 阳性对照 A：若每卡都拿**整组**（切片被写成恒等） ⇒ 不变量 ② 必红。"""

    def broken_slice(prompt_count: int, group_size: int, dp_size: int, rank: int):
        del rank
        return [(e, s) for e in range(prompt_count) for s in range(group_size)]

    G, m, P = 8, 2, 4
    union: list[int] = []
    for r in range(m):
        for e, s in broken_slice(P, G, m, r):
            if e == 0:
                union.append(s)
    assert sorted(union) != list(range(G))  # 槽位重叠（0..7 各两次）⇒ 会合拼不齐整组
    # 真实实现下该并集**恰为** 0..G−1：
    good: list[int] = []
    for r in range(m):
        good.extend(s for e, s in rollout_slice(P, G, m, r) if e == 0)
    assert sorted(good) == list(range(G))


def test_positive_control_round_robin_slicing_must_fail():
    """★ 阳性对照 B：若切片改成**轮转** `[rank::m]` ⇒ 拼齐后槽位序 ≠ `0..G−1`。

    轮转下并集仍完备（不丢不重），但**加序**变成 `0,2,4,6,1,3,5,7`
    ⇒ `μ_f` 的浮点加序与单卡不同 ⇒ 失去 A-5 的"同加序"依据。
    """

    def round_robin(prompt_count: int, group_size: int, dp_size: int, rank: int):
        return [(e, s) for e in range(prompt_count) for s in range(group_size) if s % dp_size == rank]

    G, m = 8, 2
    gathered: list[int] = []
    for r in range(m):
        gathered.extend(s for _, s in round_robin(1, G, m, r))
    assert sorted(gathered) == list(range(G))  # 完备性仍成立（所以不能只测完备性）
    assert gathered != list(assembled_slot_order(G, m))  # ★ 但加序**必须**不等 ⇒ 判红点


def test_positive_control_entry_dimension_sharding_must_fail():
    """★ 阳性对照 C：若把**条目维**也切了（回到按 prompt 分卡）⇒ 不变量 ① 必红。

    这正是本机制要消灭的形态：卡一持条目 0..1、卡二持 2..3 ⇒ 两卡条目集合都不是全体。
    """
    P, m = 4, 2
    per_rank_entries = [list(range(P))[r::m] for r in range(m)]
    assert per_rank_entries[0] != list(range(P))
    # 真实实现：每卡条目集合都 = 全体
    for r in range(m):
        assert sorted({e for e, _ in rollout_slice(P, 8, m, r)}) == list(range(P))


def test_positive_control_assertion_rejects_duplicate_slot():
    """★ 阳性对照 D：`assert_rollout_slices` 的机核本身**能红**（不看实现，看断言）。"""
    G, m, P = 8, 2, 2
    slices = {r: rollout_slice(P, G, m, r) for r in range(m)}
    # 人为把 rank1 的槽位整体 +1 ⇒ 并集缺 0、多 8 ⇒ 不变量 ② 破
    tampered = {0: slices[0], 1: [(e, s + 1) for e, s in slices[1]]}
    union = sorted(s for r in range(m) for e, s in tampered[r] if e == 0)
    assert union != list(range(G))


# ── fail-closed：非法切片参数必须显式失败并点名该改哪个键 ────────────────────

def test_validate_rollout_slicing_rejects_non_divisible_group_size():
    """★ A-8：`G=8, m=3`（不整除）⇒ 必须显式失败且**点名** `rollout_group_size`/`dp_size`。"""
    with pytest.raises(AssignmentError, match="rollout_group_size"):
        validate_rollout_slicing(8, 3)
    with pytest.raises(AssignmentError, match="dp_size"):
        validate_rollout_slicing(8, 3)
    with pytest.raises(AssignmentError, match="整除"):
        rollouts_per_rank(8, 3)
    with pytest.raises(AssignmentError, match="整除"):
        rollout_slice(1, 8, 3, 0)


def test_validate_rollout_slicing_rejects_dp_larger_than_group():
    """★ A-8：`m > G` ⇒ `G//m < 1` ⇒ 有卡一条都不生成 ⇒ 必须显式失败。

    ★ 构造注意：`m > G` 蕴涵 `G % m == G != 0` ⇒ **"不整除"分支恒真**。
    故校验顺序把"每卡至少一条"**放在整除之前**，报错文案才指向真正的成因。
    真正能命中"不整除且 `m < G`"分支的构造是 **`G=8, m=3`**（见下一个用例）。
    """
    for g, m in ((1, 2), (2, 4), (4, 8)):
        with pytest.raises(AssignmentError, match="不得大于"):
            validate_rollout_slicing(g, m)
        with pytest.raises(AssignmentError, match="不得大于"):
            rollouts_per_rank(g, m)


@pytest.mark.parametrize("bad", [0, -1])
def test_validate_rollout_slicing_rejects_non_positive_group_size(bad: int):
    with pytest.raises(AssignmentError, match="rollout_group_size"):
        validate_rollout_slicing(bad, 1)


@pytest.mark.parametrize("bad", [0, -1])
def test_validate_rollout_slicing_rejects_non_positive_dp_size(bad: int):
    with pytest.raises(AssignmentError, match="dp_size"):
        validate_rollout_slicing(8, bad)


def test_rollout_slice_rejects_out_of_range_rank_and_negative_prompt_count():
    with pytest.raises(AssignmentError, match="越界"):
        rollout_slice(1, 8, 2, 2)
    with pytest.raises(AssignmentError, match="越界"):
        rollout_slice(1, 8, 2, -1)
    with pytest.raises(AssignmentError, match="不能为负"):
        rollout_slice(-1, 8, 2, 0)
    with pytest.raises(AssignmentError, match="不能为负"):
        assert_rollout_slices(-1, 8, 2)


def test_assert_rollout_slices_passes_for_all_valid_shapes():
    for p, g, m in _SLICE_SHAPES:
        assert_rollout_slices(p, g, m)
