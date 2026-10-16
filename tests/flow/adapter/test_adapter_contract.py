"""``flow.adapter.contract`` 的契约测试 —— ★ 每条断言都带**阳性对照**（能失败）。

依据：``task-arch-design/backend-interface.md`` §3.2（I1/I2/I3）与 §3.4（可测断言）；
验证台 ``task-equiv-harness/adapter-contract.md`` 的 C1–C6 是同一批不变量在**在役实现**上的
端到端对照，本文件是**契约模块自身**的可失败机核（⛔ 不接受恒 PASS 的检查）。
"""

from __future__ import annotations

import pytest

from graspo.flow.adapter.contract import (
    ASSEMBLY_DELIVERY_POINTS,
    ROLLOUT_SLICING_ASSEMBLY,
    AlignmentError,
    AssemblyError,
    ContractError,
    IncompleteGroupError,
    PartitionError,
    RolloutGroup,
    assert_aligned,
    assert_assembled_group_complete,
    assert_complete_group,
    assert_group_partition,
    assert_no_partial_group,
    sample_key,
)

G = 8


def _group(key: str = "p1", n: int = G, *, aligned: bool = True) -> RolloutGroup:
    completions = tuple(f"{key}-c{i}" for i in range(n))
    rewards = {sample_key(key, i): float(i) for i in range(n)} if aligned else None
    return RolloutGroup(group_key=key, completions=completions, group_size=G, rewards=rewards)


# ── 形状：sample_key 与位置无关 ──────────────────────────────────────────────
def test_sample_key_is_position_independent():
    """键只由 ``(group_key, index)`` 决定 —— 与"排在第几位"无关（禁行号回退的根）。"""
    assert sample_key("p1", 3) == sample_key("p1", 3)
    assert sample_key("p1", 3) != sample_key("p1", 4)
    assert sample_key("p1", 3) != sample_key("p2", 3)


@pytest.mark.parametrize(
    ("key", "idx"),
    [("", 0), (None, 0), ("p1", -1), ("p1", "0"), ("p1", 1.0)],  # type: ignore[list-item]
)
def test_sample_key_rejects_invalid_inputs(key, idx):
    """阳性对照：非法入参必须显式失败（不得静默生成一个键）。"""
    with pytest.raises(ContractError):
        sample_key(key, idx)


def test_group_size_must_be_positive():
    with pytest.raises(ContractError):
        RolloutGroup(group_key="p1", completions=(), group_size=0)


# ── I1：组完整性 = G 条齐 ────────────────────────────────────────────────────
def test_i1_complete_group_passes():
    assert_complete_group(_group())  # 不抛 = 通过


@pytest.mark.parametrize("n", [0, 1, G - 1, G + 1])
def test_i1_partial_or_overfull_group_fails(n: int):
    """★ 阳性对照：非 G 条**必须**显式失败（含多于 G 条）。"""
    with pytest.raises(IncompleteGroupError):
        assert_complete_group(_group(n=n))


# ── I2：奖励 ↔ 样本按键对齐 ──────────────────────────────────────────────────
def test_i2_aligned_rewards_pass():
    assert_aligned(_group())


def test_i2_missing_rewards_fail():
    """★ 阳性对照：未评分 ≠ 零奖励 ⇒ 必须显式失败。"""
    with pytest.raises(AlignmentError):
        assert_aligned(_group(aligned=False))


def test_i2_key_mismatch_fails_instead_of_positional_fallback():
    """★ 阳性对照（对应验证台 V2）：长度/键不匹配**不得**回退按行号配对。"""
    good = _group()
    # 模拟"上游只按行号给奖励"的错位：键少一个、且多了一个组外键
    broken = RolloutGroup(
        group_key=good.group_key,
        completions=good.completions,
        group_size=G,
        rewards={"p1#0": 0.0, sample_key("other", 0): 1.0},
    )
    with pytest.raises(AlignmentError):
        assert_aligned(broken)


# ── I3：不完整组必须显式失败（是防线，不是过滤）──────────────────────────────
def test_i3_no_partial_groups_passes():
    assert_no_partial_group([_group("p1"), _group("p2")])


def test_i3_partial_group_fails_rather_than_being_filtered():
    """★ 阳性对照：清单里出现不完整组 ⇒ **抛错**，而不是静默过滤掉它。"""
    groups = [_group("p1"), _group("p2", n=3), _group("p3")]
    with pytest.raises(IncompleteGroupError):
        assert_no_partial_group(groups)
    # 反证"不是过滤"：调用方拿到的是异常，不是"少了一个元素的清单"
    assert len(groups) == 3


# ── β1：整组不拆 / 全局组集合不变 / 各分片组数差 ≤ 1 ─────────────────────────
def test_beta1_balanced_partition_passes():
    groups = [_group(f"p{i}") for i in range(4)]
    assert_group_partition(groups, {f"p{i}": i % 2 for i in range(4)}, n_shards=2)


def test_beta1_imbalanced_partition_fails():
    """★ 阳性对照：各分片组数差 2 ⇒ 必须失败。"""
    groups = [_group(f"p{i}") for i in range(4)]
    assignment = {f"p{i}": (0 if i < 3 else 1) for i in range(4)}  # 3 vs 1
    with pytest.raises(PartitionError):
        assert_group_partition(groups, assignment, n_shards=2)


def test_beta1_dropped_or_split_group_fails():
    """★ 阳性对照：漏掉一个组 / 引入组外键 ⇒ 必须失败（全局组集合不变）。"""
    groups = [_group(f"p{i}") for i in range(4)]
    with pytest.raises(PartitionError):
        assert_group_partition(groups, {"p0": 0, "p1": 1, "p2": 0}, n_shards=2)
    with pytest.raises(PartitionError):
        assert_group_partition(
            groups, {"p0": 0, "p1": 1, "p2": 0, "p3": 1, "pX": 0}, n_shards=2
        )


def test_beta1_shard_index_out_of_range_fails():
    groups = [_group("p0")]
    with pytest.raises(PartitionError):
        assert_group_partition(groups, {"p0": 2}, n_shards=2)


# ══════════════════════════════════════════════════════════════════════════════
# ★ 装配点契约（rollout 分卡 / U-C）：`assert_assembled_group_complete`
#
# 语义（`report.md` §7.4 / X-09）：**I1 一个字都没放宽**（上面 I1/I3 的用例一字未改），
# 新增的是"**G 条齐在哪一层被断言**"—— 交给算法层**之前**必须跨卡拼齐。
# ⛔ 本段只**新增**，不改任何既有期望值（回归只许加强）。
# ══════════════════════════════════════════════════════════════════════════════


def test_assembly_point_accepts_sliced_slots_gathered_from_all_ranks():
    """★ 正例：`m=2, G=8` 的**连续**切片按 rank 序拼齐 ⇒ 通过。"""
    slots = [(0, s) for s in range(4)] + [(1, s) for s in range(4, 8)]
    assert_assembled_group_complete("p1", slots, group_size=8, dp_size=2)
    # 乱序拼齐也必须通过（键语义天然支持乱序/跨卡）
    assert_assembled_group_complete(
        "p1", list(reversed(slots)), group_size=8, dp_size=2
    )


def test_assembly_point_accepts_m1_identity():
    """`m=1` ⇒ 恒等（槽位 0..G−1 全来自 rank 0）。"""
    assert_assembled_group_complete(
        "p1", [(0, s) for s in range(8)], group_size=8, dp_size=1
    )


@pytest.mark.parametrize("n", [0, 1, 7])
def test_assembly_point_rejects_missing_slots(n: int):
    """★ 阳性对照 A：**只交 `G/m` 条**（或不完整）⇒ **必须**抛 :class:`AssemblyError`。

    这正是"分卡后不做会合"的形态 ⇒ 组决策会在部分组上做（`groups_seen ≡ 1`）。
    """
    slots = [(0, s) for s in range(n)]
    if n > 4:
        slots += [(1, s) for s in range(4, n)]
    with pytest.raises(AssemblyError, match="装配点契约被违反"):
        assert_assembled_group_complete("p1", slots, group_size=8, dp_size=2)


def test_assembly_point_rejects_duplicate_slot():
    """★ 阳性对照 B：槽位重复（有重）⇒ 必须抛。"""
    slots = [(0, s) for s in range(4)] + [(1, s) for s in range(3, 8)]
    with pytest.raises(AssemblyError, match="重复槽位"):
        assert_assembled_group_complete("p1", slots, group_size=8, dp_size=2)


def test_assembly_point_rejects_slot_from_wrong_rank():
    """★ 阳性对照 C：槽位**齐**但**来源 rank 与切片口径不符** ⇒ 必须抛。

    这条比"条数齐"更严：它抓的是"拼接顺序/切片口径不一致"（例如把连续切片误写成
    轮转 `[rank::m]`）。
    """
    wrong = [(1, s) for s in range(4)] + [(0, s) for s in range(4, 8)]
    with pytest.raises(AssemblyError, match="切片口径不符"):
        assert_assembled_group_complete("p1", wrong, group_size=8, dp_size=2)


def test_assembly_point_rejects_out_of_range_source_rank():
    with pytest.raises(AssemblyError, match="rank 越界"):
        assert_assembled_group_complete(
            "p1", [(2, s) for s in range(4)] + [(1, s) for s in range(4, 8)],
            group_size=8, dp_size=2,
        )


def test_assembly_point_rejects_non_divisible_group_and_dp():
    """★ fail-closed：`G` 不被 `m` 整除 ⇒ **任何装配都配不出整组** ⇒ 必须显式失败。"""
    with pytest.raises(AssemblyError, match="整除"):
        assert_assembled_group_complete("p1", [], group_size=8, dp_size=3)


@pytest.mark.parametrize(("g", "m"), [(0, 1), (8, 0)])
def test_assembly_point_rejects_non_positive_sizes(g: int, m: int):
    with pytest.raises(AssemblyError):
        assert_assembled_group_complete("p1", [], group_size=g, dp_size=m)


def test_assembly_delivery_points_are_named_and_differ():
    """★ 交付点声明：两个 backend 的**交付点不同** ⇒ 必须具名登记（§7.4）。"""
    assert set(ASSEMBLY_DELIVERY_POINTS) == {"native", "msswift"}
    assert ASSEMBLY_DELIVERY_POINTS["native"] != ASSEMBLY_DELIVERY_POINTS["msswift"]
    assert "装配" in ASSEMBLY_DELIVERY_POINTS["native"]
    assert ROLLOUT_SLICING_ASSEMBLY == "rollout_slicing_assembly"
