"""Layer 2 — PipelineGraph (GraspoFlow): assembles operators into an executable pipeline.

PipelineGraph wires inter-stage buffers, holds a scheduler, and exposes
two execution modes:

  execute_rollout()   — Flink-style streaming forward (no backward)
  execute_optimize()  — 1F1B training with memory-budgeted backpressure

TP awareness:
  - Each ComputeOperator has a tp_size field.
  - When tp_size > 1: multiple ranks share the same pp_rank.  Only one of
    them (typically tp_rank=0) should be wired to the inter-stage buffer.
    The other tp_rank s participate via TP all-reduce within the stage.
  - In single-Op-per-rank mode (the current design), each rank has exactly
    one Op and the buffer wiring is 1:1 regardless of tp_size.  Inter-stage
    P2P uses rank ± tp_size neighbours, which is handled inside the Op.
"""


from dataclasses import dataclass

from graspo.backends.graspoflow.operator import ComputeOperator, Microbatch, OpBuffer
from graspo.backends.graspoflow.schedule import (
    PipelineAction,
    PipelineScheduler,
    SchedulerStep,
)


@dataclass
class PipelineConfig:
    """Pipeline-level configuration."""

    max_inflight: int = 8  # max concurrent in-flight microbatches
    synchronize_cuda_timing: bool = False


class PipelineGraph:
    """A linear pipeline DAG: Op[0] → Op[1] → … → Op[N-1].

    The graph wires inter-stage OpBuffers.  Each Op reads from its input buffer
    (upstream) and writes to its output buffer (downstream).  The first Op
    receives microbatches directly (via ``push``), and the last Op emits results
    (via ``collect``).

    The scheduler determines *when* each forward/backward happens.  The graph
    only knows *who* (which Op) to invoke.
    """

    def __init__(
        self,
        ops: list[ComputeOperator],
        scheduler: PipelineScheduler,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        if len(ops) == 0:
            raise ValueError("PipelineGraph requires at least one operator")
        self.ops = ops
        self.scheduler = scheduler
        self.config = config or PipelineConfig()
        self._wire_buffers()

    # ── buffer wiring ─────────────────────────────────────────────────────

    def _wire_buffers(self) -> None:
        """Create and wire inter-stage OpBuffers."""
        buf_size = max(1, self.config.max_inflight)
        for i in range(len(self.ops) - 1):
            buf = OpBuffer(max_slots=buf_size, name=f"stage_{i}_to_{i + 1}")
            self.ops[i].attach_buffers(input_buffer=self.ops[i].input_buffer, output_buffer=buf)
            self.ops[i + 1].attach_buffers(
                input_buffer=buf, output_buffer=self.ops[i + 1].output_buffer
            )
        # First op may have no input buffer yet — will receive via push_input
        if self.ops[0].input_buffer is None:
            self.ops[0].attach_buffers(
                input_buffer=OpBuffer(max_slots=buf_size, name="entry"),
                output_buffer=self.ops[0].output_buffer,
            )

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def first_op(self) -> ComputeOperator:
        return self.ops[0]

    @property
    def last_op(self) -> ComputeOperator:
        return self.ops[-1]

    def push_input(self, mb: Microbatch) -> bool:
        """Push a microbatch into the pipeline entry buffer.

        Returns False if the entry buffer is full (backpressure signal).
        """
        buf = self.first_op.input_buffer
        if buf is None:
            raise RuntimeError("Pipeline entry buffer not wired")
        return buf.push(mb)

    def collect_output(self) -> Microbatch | None:
        """Collect a finished microbatch from the pipeline.

        For rollout, the last Op writes to its output_buffer (if any), or
        the result is obtained by executing the last Op's forward and returning.
        For simplicity, the caller tracks results via the execution methods.
        """
        return None  # override in subclasses

    def execute_plan(self, plan: list[SchedulerStep]) -> dict[int, Microbatch]:
        """Execute a pre-computed scheduler plan.

        Returns a dict mapping microbatch_idx → final Microbatch after all
        forwards and backwards have been applied.
        """
        results: dict[int, Microbatch] = {}

        for step in plan:
            # In a real pipeline, each step runs on a specific PP stage.
            # Here, the graph runs on ONE rank — each Op corresponds to the
            # layers owned by this rank.  The scheduler's plan is for *this
            # rank's* Op(s).
            if step.action == PipelineAction.FORWARD:
                # Find the appropriate Op for this microbatch
                # (in single-Op-per-rank mode, there's exactly one Op)
                mb = self._do_forward(step.microbatch_idx)
                if mb is not None:
                    results[step.microbatch_idx] = mb
            else:
                self._do_backward(step.microbatch_idx)

        return results

    def _do_forward(self, microbatch_idx: int) -> Microbatch | None:
        """Execute forward for one microbatch on this rank's Op."""
        # In the current design, each rank has exactly one Op
        # that handles all microbatches.  The microbatch data flows
        # through NCCL P2P inside the Op.
        op = self.ops[0]  # single Op per rank

        # The microbatch comes from the entry buffer
        buf = op.input_buffer
        if buf is None:
            return None
        mb = buf.pop()
        if mb is None:
            return None

        result = op.forward(mb)
        # If this op has an output buffer, the result was already sent
        # downstream. Otherwise (last rank), save it.
        if op.output_buffer is None:
            return result
        return None

    def _do_backward(self, microbatch_idx: int) -> None:
        """Execute backward for one microbatch on this rank's Op."""
        op = self.ops[0]
        # The microbatch must be retrieved from the Op's internal state
        # (stored during forward as _stage_output / _stage_input).
        # In practice, backward() is called by the scheduler after forward().
        # We track microbatches by their index.
        op.backward(Microbatch(idx=microbatch_idx))

    def shutdown(self) -> None:
        """Release all operator resources."""
        for op in self.ops:
            op.done()
