"""``flow.adapter.contract`` 的契约测试 —— ★ 每条断言都带**阳性对照**（能失败）。

依据：``task-arch-design/backend-interface.md`` §3.2（I1/I2/I3）与 §3.4（可测断言）；
验证台 ``task-equiv-harness/adapter-contract.md`` 的 C1–C6 是同一批不变量在**在役实现**上的
端到端对照，本文件是**契约模块自身**的可失败机核（⛔ 不接受恒 PASS 的检查）。
"""

from __future__ import annotations

import pytest

from graspo.flow.adapter.contract import (
    AlignmentError,
    ContractError,
    IncompleteGroupError,
    PartitionError,
    RolloutGroup,
    assert_aligned,
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
