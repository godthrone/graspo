"""Tests for layer placement planning — BADGE §11.1."""

import pytest

from graspo.backends.graspoflow.placement import (
    NativePlacementPlan,
    _minimax_contiguous_ranges,
    _validate_manual_ranges,
    build_placement_plan,
    placement_summary,
)

# ── build_placement_plan: single GPU (pp_size=1) ────────────────────────────


def test_pp_size_1_returns_all_layers_on_rank_0():
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3",
        num_hidden_layers=36,
        tp_size=1,
        pp_size=1,
        tp_rank=0,
        pp_rank=0,
    )
    assert plan.pp_size == 1
    assert plan.pp_rank == 0
    assert plan.local_layer_indices == tuple(range(36))
    assert plan.include_embeddings is True
    assert plan.include_lm_head is True


def test_pp_size_1_tp_multiple_rank():
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3",
        num_hidden_layers=36,
        tp_size=4,
        pp_size=1,
        tp_rank=2,
        pp_rank=0,
    )
    assert plan.tp_size == 4
    assert plan.tp_rank == 2
    assert plan.local_layer_indices == tuple(range(36))


# ── build_placement_plan: manual ranges ─────────────────────────────────────


def test_manual_ranges_valid_partition():
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3",
        num_hidden_layers=6,
        tp_size=1,
        pp_size=3,
        tp_rank=0,
        pp_rank=1,
        manual_ranges=[[0, 2], [2, 4], [4, 6]],
    )
    assert plan.strategy == "manual"
    assert plan.local_layer_indices == (2, 3)
    assert plan.include_embeddings is False  # middle stage
    assert plan.include_lm_head is False


def test_manual_ranges_first_stage_has_embeddings():
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3",
        num_hidden_layers=6,
        tp_size=1,
        pp_size=3,
        tp_rank=0,
        pp_rank=0,
        manual_ranges=[[0, 2], [2, 4], [4, 6]],
    )
    assert plan.include_embeddings is True
    assert plan.include_lm_head is False


def test_manual_ranges_last_stage_has_lm_head():
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3",
        num_hidden_layers=6,
        tp_size=1,
        pp_size=3,
        tp_rank=0,
        pp_rank=2,
        manual_ranges=[[0, 2], [2, 4], [4, 6]],
    )
    assert plan.include_embeddings is False
    assert plan.include_lm_head is True


# ── _validate_manual_ranges ───────────────────────────────────────────────────


def test_validate_manual_ranges_valid():
    _validate_manual_ranges([[0, 3], [3, 6]], num_hidden_layers=6, pp_size=2)


def test_validate_manual_ranges_missing_layers():
    with pytest.raises(ValueError, match="Missing"):
        _validate_manual_ranges([[0, 2], [3, 6]], num_hidden_layers=6, pp_size=2)


def test_validate_manual_ranges_extra_layers():
    # Overlapping ranges [[0,4],[2,6]] are detected first as non-contiguous
    # (ranges[0][1]=4 != ranges[1][0]=2) before the overlap/extra analysis
    with pytest.raises(ValueError, match="contiguous"):
        _validate_manual_ranges([[0, 4], [2, 6]], num_hidden_layers=6, pp_size=2)


def test_validate_manual_ranges_gap():
    # A gap between ranges means missing layers are detected
    with pytest.raises(ValueError, match="Missing"):
        _validate_manual_ranges([[0, 2], [4, 6]], num_hidden_layers=6, pp_size=2)


def test_validate_manual_ranges_invalid_start_end():
    with pytest.raises(ValueError, match="start must be < end"):
        _validate_manual_ranges([[3, 1], [1, 6]], num_hidden_layers=6, pp_size=2)


# ── _minimax_contiguous_ranges ──────────────────────────────────────────────


def test_minimax_contiguous_ranges_equal_costs():
    ranges = _minimax_contiguous_ranges([1.0] * 10, [0.0] * 2)
    assert len(ranges) == 2
    # With equal costs, should be roughly balanced
    start1, end1 = ranges[0]
    start2, end2 = ranges[1]
    assert start1 == 0
    assert end1 == start2
    assert end2 == 10
    # With 10 layers and 2 equal stages: 5+5
    assert end1 == 5 or end1 == 4


def test_minimax_contiguous_ranges_single_stage():
    ranges = _minimax_contiguous_ranges([1.0] * 5, [0.0])
    assert ranges == [(0, 5)]


def test_minimax_contiguous_ranges_overhead_affects_split():
    overheads_heavy_last = [0.0, 0.0, 0.0, 0.0, 10.0]
    ranges = _minimax_contiguous_ranges([1.0] * 10, overheads_heavy_last)
    assert len(ranges) == 5
    # Last stage should have fewer layers due to heavy overhead
    last_range_size = ranges[4][1] - ranges[4][0]
    first_range_size = ranges[0][1] - ranges[0][0]
    assert last_range_size <= first_range_size


# ── placement_summary ──────────────────────────────────────────────────────


def test_placement_summary_includes_all_keys():
    plan = build_placement_plan(
        strategy="auto",
        model_family="qwen3",
        num_hidden_layers=12,
        tp_size=2,
        pp_size=1,
        tp_rank=0,
        pp_rank=0,
    )
    summary = placement_summary(plan)
    assert summary["placement_strategy"] == "qwen3_tp"
    assert summary["tp_size"] == 2
    assert summary["local_layer_indices"] == list(range(12))


# ── Frozen dataclass ───────────────────────────────────────────────────────


def test_native_placement_plan_is_frozen():
    plan = NativePlacementPlan(
        strategy="test",
        model_family="qwen3",
        tp_size=1,
        pp_size=1,
        pp_rank=0,
        tp_rank=0,
        local_layer_indices=(0, 1),
        include_embeddings=True,
        include_lm_head=True,
    )
    with pytest.raises(Exception):
        plan.strategy = "modified"
