"""P5 机核：msswift 的 G1 整组分片 / G2 按键对齐 / retry —— ★ 每条带阳性对照。

依据：`option-a-override-plan.md` §2/§3（Option A）与不变量 V1–V5。
本文件只测 **纯函数**（CPU、零 ms-swift）：多卡实测部分标 **待跑次阶段**。
"""

from __future__ import annotations

import pytest

from graspo.flow.adapter.contract import AlignmentError
from graspo.flow.backend.msswift.alignment import (
    advantages_for_samples,
    align_by_key,
    completion_keys,
    group_reward_rows,
    keyed_rows_from_table,
)
from graspo.flow.backend.msswift.group_sharding import (
    ShardingError,
    assert_group_contiguous,
    build_group_contiguous_batches,
    groups_per_rank_for,
    local_batch_rows,
    steps_per_generation_for,
)
from graspo.flow.backend.msswift.retry import (
    RetryError,
    assert_replacement_preserves_size,
    group_slice_for_rank,
    is_resample_needed,
    plan_resample,
)
from graspo.ripple.group_decision import GroupDecision


# ══ G1：steps_per_generation 新映射（option-a §3 的表）══════════════════════
@pytest.mark.parametrize(
    ("per_device", "world", "g", "expected_steps", "expected_rows", "expected_k"),
    [
        (1, 2, 8, 8, 8, 1),   # T044：旧 ceil(G/(1×2))=4 → 4 ❌ 切组；新 8
        (1, 4, 8, 8, 8, 1),   # T045：旧 2 → 4 ❌；新 8
        (2, 2, 8, 4, 8, 1),
        (8, 1, 8, 1, 8, 1),   # 侥幸档：旧 1 恰好 8 ✅
    ],
)
def test_new_steps_mapping_gives_whole_groups(
    per_device: int, world: int, g: int, expected_steps: int, expected_rows: int, expected_k: int
):
    del world  # world 不参与 steps 推导（这正是修法：旧式错在把 world 乘进去）
    steps = steps_per_generation_for(per_device, g)
    assert steps == expected_steps
    assert local_batch_rows(per_device, g, steps) == expected_rows
    assert groups_per_rank_for(per_device, g, steps) == expected_k
    assert expected_rows % g == 0  # ★ 每进程条数 % G == 0（V1）


def test_old_mapping_would_have_cut_a_group():
    """★ 反证：旧式 `ceil(G/(per_device×world))` 在 T044 给 4 < G=8 ⇒ 必然切组。"""
    import math

    old = max(1, math.ceil(8 / (1 * 2)))
    assert old == 4 and old % 8 != 0
    assert steps_per_generation_for(1, 8) == 8  # 新式必整除


def test_groups_per_rank_rejects_non_divisible():
    with pytest.raises(ShardingError, match="不整除"):
        groups_per_rank_for(1, 8, 3)  # 1×3=3 不是 8 的倍数


@pytest.mark.parametrize(("p", "g"), [(0, 8), (1, 0)])
def test_steps_mapping_rejects_bad_inputs(p: int, g: int):
    with pytest.raises(ShardingError, match="必须 ≥ 1"):
        steps_per_generation_for(p, g)


# ══ G1：group-contiguous 批（V1/V2/V3）══════════════════════════════════════
def test_batches_are_whole_groups_and_contiguous():
    batches = build_group_contiguous_batches(64, 8, groups_per_rank=2, world_size=4)
    assert len(batches) == 4  # 64 / (2×8)
    assert len(batches) % 4 == 0  # V2
    assert_group_contiguous(batches, 8, groups_per_rank=2)
    assert batches[0] == list(range(0, 16))
    assert batches[3] == list(range(48, 64))


def test_batches_reject_uneven_world_split():
    """★ 阳性对照：批数不能被 world 整除 ⇒ V2 违反 ⇒ 必须显式失败。"""
    with pytest.raises(ShardingError, match="整除"):
        build_group_contiguous_batches(48, 8, groups_per_rank=1, world_size=4)  # 6 批 % 4 != 0


def test_drop_last_trims_tail_batches():
    batches = build_group_contiguous_batches(48, 8, groups_per_rank=1, world_size=4, drop_last=True)
    assert len(batches) == 4 and len(batches) % 4 == 0


def test_assert_group_contiguous_catches_half_group():
    """★ 阳性对照：切在半个组上必须判红。"""
    with pytest.raises(ShardingError, match="组边界|连续"):
        assert_group_contiguous([[4, 5, 6, 7, 8, 9, 10, 11]], 8)
    with pytest.raises(ShardingError, match="条数"):
        assert_group_contiguous([[0, 1, 2]], 8)


def test_assert_group_contiguous_catches_duplicate_rows():
    with pytest.raises(ShardingError, match="重复"):
        assert_group_contiguous([[0, 0, 0, 0, 0, 0, 0, 0]], 8)


# ══ G2：按键对齐（I2）═══════════════════════════════════════════════════════
def test_align_by_key_orders_by_expected_keys():
    values = {"p1#1": 11, "p1#0": 10}
    assert align_by_key(["p1#0", "p1#1"], values) == [10, 11]


def test_align_by_key_fails_instead_of_positional_pairing():
    """★ 阳性对照（对应验证台 S1）：键不匹配 ⇒ `AlignmentError`，**绝不回退行号配对**。"""
    with pytest.raises(AlignmentError, match="未按键对齐"):
        align_by_key(["p1#0", "p1#1"], {"p1#0": 1.0, "other#0": 2.0})


def test_keyed_rows_rejects_duplicate_keys():
    with pytest.raises(AlignmentError, match="重复键"):
        keyed_rows_from_table([1, 2], ["k", "k"])


def test_keyed_rows_rejects_length_mismatch():
    with pytest.raises(AlignmentError, match="长度与键数不符"):
        keyed_rows_from_table([1, 2], ["k"])


def test_group_rewards_has_no_positional_fallback():
    """★ 阳性对照（对应验证台 S1/C2）：`_group_rewards` 的旧 positional fallback 已不复存在。"""
    global_rows = [[0.5], [0.25]]
    global_keys = ["p0#0", "p0#1"]
    assert group_reward_rows(global_rows, global_keys, ["p0#1", "p0#0"]) == [[0.25], [0.5]]
    # 键缺一个 ⇒ 显式失败（旧实现会按行号回退并打警告）
    with pytest.raises(AlignmentError):
        group_reward_rows(global_rows, global_keys, ["p0#0", "p0#2"])


def test_completion_keys_and_advantages_writeback():
    keys = completion_keys("p7", 3)
    assert keys == ["p7#0", "p7#1", "p7#2"]
    assert advantages_for_samples(keys, {"p7#2": 3, "p7#0": 1, "p7#1": 2}) == [1, 2, 3]
    with pytest.raises(AlignmentError):
        advantages_for_samples(keys, {"p7#0": 1})


def test_completion_keys_reject_bad_group_size():
    with pytest.raises(AlignmentError, match="group_size"):
        completion_keys("p1", 0)


# ══ retry（V5 + 谓词替换 + 按组切片）════════════════════════════════════════
@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (GroupDecision.RETRY, True),
        (GroupDecision.PERFECT_SKIP, False),
        (GroupDecision.TRAINABLE_MAX_CORRECT, False),
        (GroupDecision.INVALID, False),
    ],
)
def test_predicate_is_graspo_retry_not_upstream_std(decision: GroupDecision, expected: bool):
    """★ 谓词换成 graspo 的 `GroupDecision.RETRY`（不是上游的 reward std==0）。"""
    assert is_resample_needed(decision) is expected


def test_plan_resample_replaces_within_budget():
    plan = plan_resample(
        [GroupDecision.RETRY, GroupDecision.PERFECT_SKIP, GroupDecision.RETRY],
        [0, 0, 2],
        max_attempts=4,
    )
    assert plan.replace == (0, 2) and plan.keep == (1,) and plan.exhausted == ()


def test_plan_resample_marks_exhausted_at_budget():
    plan = plan_resample([GroupDecision.RETRY], [3], max_attempts=4)
    assert plan.exhausted == (0,) and plan.replace == ()


def test_plan_resample_rejects_length_mismatch():
    with pytest.raises(RetryError, match="长度不符"):
        plan_resample([GroupDecision.RETRY], [], max_attempts=2)


def test_replacement_must_preserve_size():
    """★ 阳性对照（V5）：追加式重采样（条数变了）必须判红。"""
    assert_replacement_preserves_size(8, 8)
    with pytest.raises(RetryError, match="替换而非追加"):
        assert_replacement_preserves_size(8, 16)


def test_group_slice_is_by_group_not_by_position():
    """★ 按组切片（替代上游 `slice(idx*len(samples), …)` 的位置切片）。"""
    assert group_slice_for_rank(16, 1, 0, world_size=4) == slice(0, 1)
    assert group_slice_for_rank(16, 1, 3, world_size=4) == slice(3, 4)
    with pytest.raises(RetryError, match="不可能严格相等"):
        group_slice_for_rank(2, 1, 0, world_size=4)


def test_group_slice_rejects_bad_rank():
    with pytest.raises(RetryError, match="不在"):
        group_slice_for_rank(16, 1, 4, world_size=4)
