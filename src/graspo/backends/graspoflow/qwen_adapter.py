"""Layer 3 — Qwen-specific training adapter for graspoflow.

This is the model-specialized layer.  It knows about:
  - Qwen tokenizer, chat template, multimodal encoding
  - How to build QwenStageOp s from a placement plan
  - How to wire GRPO training (rollout + reward + optimize) through the pipeline

For now this is a skeleton — the full integration with Trainer will be done
in Phase 4 when we add ``backend: pipeline-v2`` support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graspo.backends.graspoflow.graph import PipelineConfig, PipelineGraph
from graspo.backends.graspoflow.operator import Microbatch
from graspo.backends.graspoflow.optimize import OptimizePipeline
from graspo.backends.graspoflow.rollout import RolloutPipeline
from graspo.backends.graspoflow.schedule import PipelineScheduler, get_scheduler


@dataclass
class QwenPPAdapterConfig:
    """Configuration for the Qwen graspoflow training adapter.

    This is a simplified config — in production the values come from
    GraspoConfig (core/schema.py).
    """

    # ── Model ──
    model_path: str = ""
    torch_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    # ── Pipeline ──
    tp_size: int = 1
    pp_size: int = 1
    placement_strategy: str = "qwen36_pp8_static"
    pp_schedule: str = "one_f_one_b"
    forward_batch_size: int = 8
    max_inflight: int = 8

    # ── Training ──
    rollout_group_size: int = 8
    optimize_prompt_batch_size: int = 4
    optimize_times_per_step: int = 3
    max_new_tokens: int = 512
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    policy_ratio_clip_eps: float = 0.2

    # ── Memory ──
    empty_cache_after_rollout_split: bool = False
    empty_cache_before_train: bool = False


class QwenPPTrainingAdapter:
    """Model-specific training adapter for Qwen PP training.

    This sits at Layer 3 of the architecture:
      Layer 1 (operators) → Layer 2 (pipeline) → Layer 3 (this adapter)

    Responsibilities:
      - Load model with correct placement
      - Build QwenStageOp s
      - Create PipelineGraph with appropriate scheduler
      - Orchestrate rollout → reward → optimize cycles
    """

    def __init__(self, config: QwenPPAdapterConfig) -> None:
        self.config = config
        self._model: Any = None
        self._pipeline: PipelineGraph | None = None
        self._scheduler: PipelineScheduler | None = None

    # ── initialization ────────────────────────────────────────────────────

    def initialize(self, rank: int, world_size: int, device: Any = None) -> None:
        """Initialize the adapter: load model, build ops, wire pipeline."""

        self._scheduler = get_scheduler(
            self.config.pp_schedule,
            pp_size=self.config.pp_size,
            pp_rank=0,  # placeholder; real value from NativeTPState
        )

        pipeline_config = PipelineConfig(
            max_inflight=self.config.max_inflight,
        )

        # Skeleton: ops will be built from the loaded model
        ops: list = []  # placeholder — populated after model load

        self._pipeline = OptimizePipeline(
            ops, self._scheduler, config=pipeline_config
        )

    # ── training cycle ─────────────────────────────────────────────────────

    def train_step(
        self, experiences: list[Any]
    ) -> dict[str, Any]:
        """Run one training step: forward → backward → optimizer step.

        Args:
            experiences: List of Experience objects (collated from replay buffer).

        Returns:
            Training metrics.
        """
        if self._pipeline is None or not isinstance(self._pipeline, OptimizePipeline):
            raise RuntimeError("Adapter not initialized for training")

        # Convert experiences to microbatches
        microbatches = self._experiences_to_microbatches(experiences)

        # Execute through the optimize pipeline
        metrics = self._pipeline.train_chunk(microbatches)
        return metrics

    def rollout(self, samples: list[Any]) -> list[Any]:
        """Generate completions for a batch of samples.

        Uses the RolloutPipeline for Flink-style streaming generation.
        """
        if self._pipeline is None or self._scheduler is None:
            raise RuntimeError("Adapter not initialized")

        # Tokenize samples into microbatches
        microbatches = self._samples_to_microbatches(samples)

        # Create rollout pipeline and generate
        rollout = RolloutPipeline(
            self._pipeline.ops, self._scheduler,
            config=PipelineConfig(max_inflight=self.config.forward_batch_size),
        )
        generated, timing = rollout.generate(
            microbatches,
            max_new_tokens=self.config.max_new_tokens,
        )
        return generated

    # ── microbatch construction ────────────────────────────────────────────

    def _experiences_to_microbatches(
        self, experiences: list[Any]
    ) -> list[Microbatch]:
        """Convert training experiences into pipeline microbatches."""
        mbs: list[Microbatch] = []
        for i, exp in enumerate(experiences):
            mb = Microbatch(
                idx=i,
                input_ids=getattr(exp, "sequences", None),
                attention_mask=getattr(exp, "attention_mask", None),
                old_log_probs=getattr(exp, "old_log_probs", None),
                advantages=getattr(exp, "advantages", None),
                action_mask=getattr(exp, "action_mask", None),
            )
            mbs.append(mb)
        return mbs

    def _samples_to_microbatches(self, samples: list[Any]) -> list[Microbatch]:
        """Convert raw training samples into pipeline microbatches."""
        mbs: list[Microbatch] = []
        for i, sample in enumerate(samples):
            mb = Microbatch(idx=i, multimodal_inputs=None)
            mbs.append(mb)
        return mbs

    # ── cleanup ────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._pipeline is not None:
            self._pipeline.shutdown()
