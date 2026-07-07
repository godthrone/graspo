"""Layer 1 — Pipeline schedulers (pluggable scheduling strategies).

A scheduler is a pure function: given a list of microbatches and pipeline
topology, it returns an *ordered plan* of (action, microbatch_idx) steps.
The caller then executes each step by invoking the appropriate operator's
forward() or backward().

This is the Flink-inspired separation: the scheduler knows *when* to execute
*which* microbatch, but it does NOT know *how* to compute or communicate.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class PipelineAction(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True, slots=True)
class SchedulerStep:
    """One atomic step the pipeline should execute."""

    action: PipelineAction
    microbatch_idx: int  # 0-based index in the chunk

    def __repr__(self) -> str:
        return f"{self.action.value}({self.microbatch_idx})"


class PipelineScheduler(ABC):
    """Abstract pipeline scheduling strategy.

    Subclasses implement specific schedules (1F1B, GPipe, etc.).
    """

    def __init__(self, *, pp_size: int, pp_rank: int) -> None:
        if pp_size < 1:
            raise ValueError("pp_size must be >= 1")
        if pp_rank < 0 or pp_rank >= pp_size:
            raise ValueError(f"pp_rank {pp_rank} out of range [0, {pp_size})")
        self.pp_size = pp_size
        self.pp_rank = pp_rank

    @property
    def is_pipeline(self) -> bool:
        return self.pp_size > 1

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @abstractmethod
    def plan(self, chunk_count: int, *, max_inflight: int | None = None) -> list[SchedulerStep]:
        """Build an ordered execution plan for a chunk of microbatches.

        Args:
            chunk_count: Number of microbatches in this chunk.
            max_inflight: Optional cap on concurrent in-flight microbatches.
                When None, uses pp_size as the natural bound.

        Returns:
            Ordered list of steps that the pipeline should execute.
        """

    def plan_phases(
        self, chunk_count: int, *, max_inflight: int | None = None
    ) -> dict[str, list[SchedulerStep]]:
        """Like plan() but returns steps grouped by phase name.

        Useful for logging / monitoring (e.g. fill_sec, steady_sec, drain_sec).
        """
        return {"all": self.plan(chunk_count, max_inflight=max_inflight)}


# ── GPipe Scheduler ───────────────────────────────────────────────────────────


class GPipeScheduler(PipelineScheduler):
    """Naive GPipe: all forwards, then all backwards.

    Simplest schedule, maximal memory pressure (all activations are kept
    until backward begins).
    """

    def plan(self, chunk_count: int, *, max_inflight: int | None = None) -> list[SchedulerStep]:
        plan: list[SchedulerStep] = []
        # All forwards
        for i in range(chunk_count):
            plan.append(SchedulerStep(PipelineAction.FORWARD, i))
        # All backwards (reverse order)
        for i in reversed(range(chunk_count)):
            plan.append(SchedulerStep(PipelineAction.BACKWARD, i))
        return plan

    def plan_phases(
        self, chunk_count: int, *, max_inflight: int | None = None
    ) -> dict[str, list[SchedulerStep]]:
        fwd = [SchedulerStep(PipelineAction.FORWARD, i) for i in range(chunk_count)]
        bwd = [SchedulerStep(PipelineAction.BACKWARD, i) for i in reversed(range(chunk_count))]
        return {"forward": fwd, "backward": bwd}


# ── 1F1B Scheduler ────────────────────────────────────────────────────────────


class OneFOneBScheduler(PipelineScheduler):
    """Standard 1F1B (one-forward-one-backward) pipeline schedule.

    Three phases:
      fill:   forward 0 … warmup-1              (build the pipeline)
      steady: forward k+warmup then backward k  (interleave)
      drain:  backward remaining … chunk-1      (flush the pipeline)

    warmup = min(pp_size - pp_rank - 1, chunk_count, max_inflight)

    Reference: Huang et al. "GPipe", Narayanan et al. "PipeDream" (1F1B variant).
    """

    def _warmup_count(self, chunk_count: int, *, max_inflight: int | None = None) -> int:
        natural = max(0, min(self.pp_size - self.pp_rank - 1, chunk_count))
        if max_inflight is not None and max_inflight > 0:
            return min(natural, max_inflight)
        return natural

    def plan(self, chunk_count: int, *, max_inflight: int | None = None) -> list[SchedulerStep]:
        warmup = self._warmup_count(chunk_count, max_inflight=max_inflight)
        plan: list[SchedulerStep] = []

        # fill
        for i in range(warmup):
            plan.append(SchedulerStep(PipelineAction.FORWARD, i))

        # steady
        remaining = chunk_count - warmup
        for offset in range(remaining):
            plan.append(SchedulerStep(PipelineAction.FORWARD, offset + warmup))
            plan.append(SchedulerStep(PipelineAction.BACKWARD, offset))

        # drain
        for i in range(remaining, chunk_count):
            plan.append(SchedulerStep(PipelineAction.BACKWARD, i))

        return plan

    def plan_phases(
        self, chunk_count: int, *, max_inflight: int | None = None
    ) -> dict[str, list[SchedulerStep]]:
        warmup = self._warmup_count(chunk_count, max_inflight=max_inflight)
        remaining = chunk_count - warmup

        fill = [SchedulerStep(PipelineAction.FORWARD, i) for i in range(warmup)]

        steady: list[SchedulerStep] = []
        for offset in range(remaining):
            steady.append(SchedulerStep(PipelineAction.FORWARD, offset + warmup))
            steady.append(SchedulerStep(PipelineAction.BACKWARD, offset))

        drain = [SchedulerStep(PipelineAction.BACKWARD, i) for i in range(remaining, chunk_count)]

        return {"fill": fill, "steady": steady, "drain": drain}


# ── Async 1F1B (placeholder) ──────────────────────────────────────────────────


class AsyncOneFOneBScheduler(OneFOneBScheduler):
    """1F1B with explicit forward-backward pairing for async P2P overlap.

    Currently identical to OneFOneBScheduler; the async variant will emit
    paired (forward, backward) steps that the executor can overlap using
    CUDA streams.  This class exists as an extension point.
    """

    def plan(self, chunk_count: int, *, max_inflight: int | None = None) -> list[SchedulerStep]:
        # Same schedule; the executor is responsible for async overlap.
        return super().plan(chunk_count, max_inflight=max_inflight)


# ── Scheduler registry ────────────────────────────────────────────────────────


_SCHEDULER_REGISTRY: dict[str, type[PipelineScheduler]] = {
    "simple": GPipeScheduler,
    "gpipe": GPipeScheduler,
    "one_f_one_b": OneFOneBScheduler,
    "1f1b": OneFOneBScheduler,
    "async_1f1b": AsyncOneFOneBScheduler,
}


def get_scheduler(name: str, *, pp_size: int, pp_rank: int) -> PipelineScheduler:
    """Create a scheduler by name."""
    name = name.lower().strip()
    cls = _SCHEDULER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_SCHEDULER_REGISTRY))
        raise ValueError(f"Unknown pipeline scheduler: {name!r}. Available: {available}")
    return cls(pp_size=pp_size, pp_rank=pp_rank)
