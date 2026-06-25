"""Layer 2 — OptimizePipeline: 1F1B training with memory-budgeted backpressure.

The OptimizePipeline orchestrates the forward → backward cycle for training.
It uses the 1F1B scheduler to interleave forwards and backwards, and the
memory budget to cap the number of in-flight microbatches.

Each rank runs exactly one Op (the layers it owns).  The scheduler's plan
is a sequence of forward/backward indices.  The pipeline executes them in order.
"""

from __future__ import annotations

import time
from typing import Any

from graspo.backends.graspoflow.graph import PipelineConfig, PipelineGraph
from graspo.backends.graspoflow.memory import (
    compute_max_inflight,
    get_gpu_free_memory_bytes,
)
from graspo.backends.graspoflow.operator import ComputeOperator, Microbatch
from graspo.backends.graspoflow.schedule import (
    PipelineAction,
    PipelineScheduler,
    SchedulerStep,
)


class OptimizePipeline(PipelineGraph):
    """Training pipeline: 1F1B schedule with memory-aware backpressure.

    Usage::

        pipeline = OptimizePipeline(ops, scheduler, config=...)
        results = pipeline.train_chunk(chunk_microbatches)
    """

    def __init__(
        self,
        ops: list[ComputeOperator],
        scheduler: PipelineScheduler,
        *,
        config: PipelineConfig | None = None,
        gradient_checkpointing: bool = True,
        dtype_bytes: int = 2,
    ) -> None:
        super().__init__(ops, scheduler, config=config)
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype_bytes = dtype_bytes
        # Per-microbatch activation estimate (lazy-computed on first use)
        self._per_mb_activation_bytes: int | None = None
        # Microbatches stored by index during forward → used by backward
        self._forward_store: dict[int, Microbatch] = {}

    # ── memory budget ──────────────────────────────────────────────────────

    def _compute_memory_budget(
        self, batch_size: int, seq_len: int, hidden_size: int
    ) -> int:
        """Compute max_inflight from current GPU free memory."""
        free_bytes = get_gpu_free_memory_bytes()
        if free_bytes <= 0:
            # No GPU available — use config default
            return self.config.max_inflight
        return compute_max_inflight(
            gpu_memory_free_bytes=free_bytes,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            dtype_bytes=self.dtype_bytes,
            gradient_checkpointing=self.gradient_checkpointing,
            safety_factor=0.8,
        )

    # ── training ───────────────────────────────────────────────────────────

    def train_chunk(
        self,
        chunk_microbatches: list[Microbatch],
        *,
        hidden_size: int = 2048,
    ) -> dict[str, Any]:
        """Execute one training chunk (optimizer step) through the pipeline.

        Args:
            chunk_microbatches: List of microbatches for this chunk.

        Returns:
            Metrics dict with forward_sec, backward_sec, fill_sec, etc.
        """
        chunk_count = len(chunk_microbatches)
        if chunk_count == 0:
            return _empty_metrics()

        # Determine memory budget from the first microbatch
        first_mb = chunk_microbatches[0]
        batch_size = first_mb.batch_size
        seq_len = first_mb.seq_len

        max_inflight = self._compute_memory_budget(batch_size, seq_len, hidden_size)

        # Build the scheduler plan (in phases for fill/steady/drain timing)
        phases = self.scheduler.plan_phases(chunk_count, max_inflight=max_inflight)

        # Pre-load microbatches into the entry buffer
        self._forward_store.clear()
        for mb in chunk_microbatches:
            self._forward_store[mb.idx] = mb

        # Execute the plan
        timing: dict[str, float] = {
            "fill_sec": 0.0,
            "steady_sec": 0.0,
            "drain_sec": 0.0,
            "forward_sec": 0.0,
            "backward_sec": 0.0,
            "optimizer_step_sec": 0.0,
        }
        finite = True
        loss_values: list[float] = []

        # Fill phase
        t_fill = time.monotonic()
        for step in phases.get("fill", []):
            self._exec_step(step, timing)
        timing["fill_sec"] = time.monotonic() - t_fill

        # Steady phase
        t_steady = time.monotonic()
        for step in phases.get("steady", []):
            self._exec_step(step, timing)
        timing["steady_sec"] = time.monotonic() - t_steady

        # Drain phase
        t_drain = time.monotonic()
        for step in phases.get("drain", []):
            self._exec_step(step, timing)
        timing["drain_sec"] = time.monotonic() - t_drain

        return {
            "finite": finite,
            "loss_value": sum(loss_values),
            "forward_sec": timing["forward_sec"],
            "backward_sec": timing["backward_sec"],
            "fill_sec": timing["fill_sec"],
            "steady_sec": timing["steady_sec"],
            "drain_sec": timing["drain_sec"],
            "chunk_count": chunk_count,
            "max_inflight": max_inflight,
        }

    def _exec_step(self, step: SchedulerStep, timing: dict[str, float]) -> None:
        """Execute one scheduler step (forward or backward)."""
        mb = self._forward_store.get(step.microbatch_idx)
        if mb is None:
            # Fallback: create an empty microbatch for backward
            # (this happens when forward was executed on a different rank
            # and the microbatch was sent via NCCL, not stored here)
            mb = Microbatch(idx=step.microbatch_idx)
            self._forward_store[step.microbatch_idx] = mb

        op = self.ops[0]  # single Op per rank

        if step.action == PipelineAction.FORWARD:
            t0 = time.monotonic()
            result_mb = op.forward(mb)
            timing["forward_sec"] += time.monotonic() - t0
            if result_mb is not None:
                self._forward_store[step.microbatch_idx] = result_mb
        else:
            t0 = time.monotonic()
            op.backward(mb)
            timing["backward_sec"] += time.monotonic() - t0


def _empty_metrics() -> dict[str, Any]:
    return {
        "finite": True,
        "loss_value": 0.0,
        "forward_sec": 0.0,
        "backward_sec": 0.0,
        "fill_sec": 0.0,
        "steady_sec": 0.0,
        "drain_sec": 0.0,
        "chunk_count": 0,
        "max_inflight": 0,
    }
