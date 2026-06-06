from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NativePlacementPlan:
    strategy: str
    model_family: str
    tp_size: int
    pp_size: int
    pp_rank: int
    tp_rank: int
    local_layer_indices: tuple[int, ...]
    include_embeddings: bool
    include_lm_head: bool

    @property
    def is_pipeline(self) -> bool:
        return self.pp_size > 1


def build_placement_plan(
    *,
    strategy: str,
    model_family: str,
    num_hidden_layers: int,
    tp_size: int,
    pp_size: int,
    tp_rank: int,
    pp_rank: int,
    layer_types: list[str] | tuple[str, ...] | None = None,
) -> NativePlacementPlan:
    requested = (strategy or "auto").strip()
    if requested == "auto":
        requested = "qwen36_pp8_static" if model_family == "qwen3_5_text" and pp_size > 1 else "qwen3_tp"
    if pp_size == 1:
        return NativePlacementPlan(
            strategy=requested,
            model_family=model_family,
            tp_size=int(tp_size),
            pp_size=1,
            pp_rank=0,
            tp_rank=int(tp_rank),
            local_layer_indices=tuple(range(int(num_hidden_layers))),
            include_embeddings=True,
            include_lm_head=True,
        )
    supported_pipeline_strategies = {"qwen36_pp8_static", "qwen36_pp8_lm_head_only_final"}
    if requested not in supported_pipeline_strategies:
        raise ValueError(f"Unsupported pipeline placement strategy: {requested}")
    if model_family != "qwen3_5_text":
        raise ValueError(f"{requested} placement requires qwen3_5_text model family")
    if int(tp_size) != 1:
        raise ValueError(f"{requested} v1 requires tensor_model_parallel_size=1")
    if requested == "qwen36_pp8_lm_head_only_final":
        ranges = _qwen36_lm_head_only_final_ranges(
            num_hidden_layers=int(num_hidden_layers),
            pp_size=int(pp_size),
            layer_types=tuple(layer_types or ()),
        )
    else:
        ranges = _balanced_qwen36_layer_ranges(
            num_hidden_layers=int(num_hidden_layers),
            pp_size=int(pp_size),
            layer_types=tuple(layer_types or ()),
        )
    local_layers = tuple(range(*ranges[int(pp_rank)]))
    return NativePlacementPlan(
        strategy=requested,
        model_family=model_family,
        tp_size=1,
        pp_size=int(pp_size),
        pp_rank=int(pp_rank),
        tp_rank=0,
        local_layer_indices=local_layers,
        include_embeddings=int(pp_rank) == 0,
        include_lm_head=int(pp_rank) == int(pp_size) - 1,
    )


def _balanced_qwen36_layer_ranges(
    *,
    num_hidden_layers: int,
    pp_size: int,
    layer_types: tuple[str, ...],
) -> list[tuple[int, int]]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if pp_size <= 0:
        raise ValueError("pipeline_model_parallel_size must be positive")
    if len(layer_types) != num_hidden_layers:
        layer_types = tuple("linear_attention" for _ in range(num_hidden_layers))
    costs = [1.18 if layer_type == "full_attention" else 1.0 for layer_type in layer_types]
    stage_overheads = [0.0 for _ in range(pp_size)]
    stage_overheads[0] = 1.5
    # The final stage owns final norm and lm_head. It consistently has much
    # higher memory pressure and backward cost, so keep fewer decoder layers
    # there than a pure layer-count balance would choose.
    stage_overheads[-1] = 8.0
    return _minimax_contiguous_ranges(costs, stage_overheads)


def _qwen36_lm_head_only_final_ranges(
    *,
    num_hidden_layers: int,
    pp_size: int,
    layer_types: tuple[str, ...],
) -> list[tuple[int, int]]:
    if pp_size < 2:
        raise ValueError("lm-head-only final placement requires at least two pipeline stages")
    if len(layer_types) != num_hidden_layers:
        layer_types = tuple("linear_attention" for _ in range(num_hidden_layers))
    costs = [1.18 if layer_type == "full_attention" else 1.0 for layer_type in layer_types]
    stage_overheads = [0.0 for _ in range(pp_size - 1)]
    stage_overheads[0] = 1.5
    return [*_minimax_contiguous_ranges(costs, stage_overheads), (num_hidden_layers, num_hidden_layers)]


def _minimax_contiguous_ranges(costs: list[float], stage_overheads: list[float]) -> list[tuple[int, int]]:
    num_layers = len(costs)
    pp_size = len(stage_overheads)
    if pp_size > num_layers:
        raise ValueError("pipeline_model_parallel_size cannot exceed num_hidden_layers")
    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + float(cost))

    best: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}

    def solve(layer_start: int, stage_idx: int) -> tuple[float, list[tuple[int, int]]]:
        key = (layer_start, stage_idx)
        if key in best:
            return best[key]
        if stage_idx == pp_size - 1:
            load = prefix[num_layers] - prefix[layer_start] + stage_overheads[stage_idx]
            result = (load, [(layer_start, num_layers)])
            best[key] = result
            return result
        remaining_stages = pp_size - stage_idx - 1
        best_value = float("inf")
        best_ranges: list[tuple[int, int]] | None = None
        max_stop = num_layers - remaining_stages
        for layer_stop in range(layer_start + 1, max_stop + 1):
            local_load = prefix[layer_stop] - prefix[layer_start] + stage_overheads[stage_idx]
            rest_load, rest_ranges = solve(layer_stop, stage_idx + 1)
            value = max(local_load, rest_load)
            if value < best_value:
                best_value = value
                best_ranges = [(layer_start, layer_stop), *rest_ranges]
        assert best_ranges is not None
        result = (best_value, best_ranges)
        best[key] = result
        return result

    return solve(0, 0)[1]


def placement_summary(plan: NativePlacementPlan) -> dict[str, Any]:
    return {
        "placement_strategy": plan.strategy,
        "model_family": plan.model_family,
        "tensor_model_parallel_size": plan.tp_size,
        "pipeline_model_parallel_size": plan.pp_size,
        "tp_rank": plan.tp_rank,
        "pp_rank": plan.pp_rank,
        "local_layer_indices": list(plan.local_layer_indices),
        "include_embeddings": plan.include_embeddings,
        "include_lm_head": plan.include_lm_head,
    }
