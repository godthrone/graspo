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
    assign_whole_groups,
    assert_balanced_assignment,
    plan_group_rebalance,
    step_group_quota,
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
