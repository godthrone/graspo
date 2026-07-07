"""Layer 2 — RolloutPipeline: Flink-style streaming forward-only pipeline.

During rollout (generation), data flows in one direction — no backward pass.
Each microbatch is independent, making this a natural fit for Flink-style
operator chaining with backpressure.

Key design decisions:
  - Microbatches are queued at the entry buffer; backpressure is triggered
    when a downstream buffer is full.
  - Each microbatch is a group of sequences (batch_size = forward_batch_size).
  - KV-cache management is internal to each operator (not exposed to the scheduler).
  - The decode loop (token-by-token) runs INSIDE the final operator, not in
    the pipeline graph — this avoids per-token scheduling overhead.
"""

import time
from typing import Any

import torch

from graspo.backends.graspoflow.graph import PipelineConfig, PipelineGraph
from graspo.backends.graspoflow.operator import ComputeOperator, Microbatch
from graspo.backends.graspoflow.schedule import PipelineScheduler


class RolloutPipeline(PipelineGraph):
    """Forward-only pipeline for autoregressive generation.

    Microbatches enter at stage 0 and stream through to stage N-1.
    Backpressure is applied when any inter-stage buffer is full.
    """

    def __init__(
        self,
        ops: list[ComputeOperator],
        scheduler: PipelineScheduler,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        super().__init__(ops, scheduler, config=config)
        # Rollout-specific: each Op needs to support KV cache.
        # The final Op samples tokens and broadcasts them.

    def generate(
        self,
        microbatches: list[Microbatch],
        *,
        max_new_tokens: int = 512,
        eos_token_id: int | None = None,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Generate completions for a batch of microbatches.

        This is a simplified CPU-side orchestration.  The actual generation
        happens inside the operators via NCCL P2P.  The pipeline graph's role
        is to manage the flow of microbatches and apply backpressure.

        Returns:
            (generated_sequences, timing_info)
        """
        timing: dict[str, float] = {}
        t_start = time.monotonic()

        generated: list[torch.Tensor] = []

        for mb in microbatches:
            # Push into entry buffer; wait if full (backpressure)
            while not self.push_input(mb):
                self._drain_one()

        # Drain remaining microbatches
        while self._has_inflight():
            self._drain_one()

        timing["total_sec"] = time.monotonic() - t_start
        return generated, timing

    def _has_inflight(self) -> bool:
        """Check whether any microbatch is still in-flight."""
        for op in self.ops:
            if op.input_buffer is not None and not op.input_buffer.is_empty:
                return True
            if op.output_buffer is not None and not op.output_buffer.is_empty:
                return True
        return False

    def _drain_one(self) -> None:
        """Process one unit of work across all operators.

        This is the Flink-style event loop: each operator checks its input
        buffer and, if it has data and downstream has capacity, processes
        one microbatch.
        """
        for op in self.ops:
            if op.input_buffer is None or op.input_buffer.is_empty:
                continue
            if op.output_buffer is not None and op.output_buffer.is_full:
                continue  # backpressure: downstream not ready
            mb = op.input_buffer.pop()
            if mb is None:
                continue
            result = op.forward(mb)
            if op.output_buffer is not None:
                op.output_buffer.push(result)
            # If this is the terminal Op and has no output buffer,
            # the result (token) has been broadcast internally via NCCL.
