from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from graspo.backends.native_tp.checkpoint import save_native_checkpoint
from graspo.backends.native_tp.logger import NativeRolloutLogger
from graspo.backends.native_tp.runtime import (
    NativeTPRuntime,
    NativeTPRuntimeProtocol,
    NativeGeneration,
    validate_native_runtime_config,
)
from graspo.core.advantage import group_advantages
from graspo.core.buffer import Experience, ReplayBuffer
from graspo.core.completion import ParsedCompletion, raw_parsed_completion
from graspo.core.data import load_jsonl
from graspo.core.graspo_parity import classify_group, has_reward_variance, lower_median
from graspo.core.reward import GraspoReward
from graspo.core.schema import GraspoConfig, Sample


@dataclass(slots=True)
class NativeTrainStats:
    total_groups: int = 0
    perfect_skipped: int = 0
    retries: int = 0
    invalid: int = 0
    invalid_no_preference_gap: int = 0
    trainable: int = 0
    trainable_max_correct: int = 0
    trainable_not_correct: int = 0
    optimized_steps: int = 0


@dataclass(slots=True)
class NativeEpochStats:
    epoch: int = 0
    samples_seen: int = 0
    attempt_groups: int = 0
    completion_count: int = 0
    perfect_skipped: int = 0
    retries: int = 0
    invalid: int = 0
    invalid_no_preference_gap: int = 0
    trainable: int = 0
    trainable_max_correct: int = 0
    trainable_not_correct: int = 0
    reward_mean_sum: float = 0.0
    content_mean_sum: float = 0.0
    best_reward: float = 0.0


@dataclass(slots=True)
class _QueuedSample:
    sample: Sample
    retry_count: int = 0
    attempts: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class _AttemptRecord:
    sample: Sample
    generation: NativeGeneration
    parsed_completions: list[ParsedCompletion]
    rewards: list[float]
    content_scores: list[float]
    all_right: list[bool]
    reward_details: list[dict[str, Any]]
    decision: Any
    retry_count: int
    readable: dict[str, Any]
    timing: dict[str, Any]


class NativeTPGraspoTrainer:
    """Self-owned GRASPO loop backed by native TP tensor parallel runtime."""

    def __init__(
        self,
        config: GraspoConfig,
        selection: Any | None = None,
        runtime: NativeTPRuntimeProtocol | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.runtime = runtime or NativeTPRuntime.from_config(config)
        self.reward = GraspoReward(config.reward)
        self.replay_buffer = ReplayBuffer()
        self.stats = NativeTrainStats()
        self.backend_name = "native-tp"
        self.global_step = 0
        self.sample_index = 0
        self.total_samples = 0
        self.started_at = time.monotonic()
        self.current_epoch_stats = NativeEpochStats()
        self.recent_groups: deque[dict[str, Any]] = deque(maxlen=50)
        self.pending_batch_attempts: list[dict[str, Any]] = []
        self.pending_batch_timings: list[dict[str, Any]] = []
        self.resume_info: dict[str, Any] | None = None
        native = self.config.native_tp
        self.logger = NativeRolloutLogger(
            self.config.training.output_dir,
            readable_enabled=native.readable_log_enabled,
            raw_enabled=native.raw_log_enabled,
        )

    def train(self) -> None:
        validate_native_runtime_config(self.config)
        self.runtime.validate()
        self.runtime.setup()
        # Force CUDA expandable segments to prevent fragmentation when
        # empty_cache_after_rollout_split is disabled.  Without this,
        # the allocator holds free-but-reserved KV-cache blocks that
        # fragment and cause OOM during the next optimize phase.
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")
        conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        if "expandable_segments:True" not in conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                conf + ("," if conf else "") + "expandable_segments:True"
            )
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "backend_selected",
                "backend": self.backend_name,
                "reason": self.selection.reason if self.selection is not None else "configured",
                "dependency_boundary": (
                    "PyTorch distributed TP/PP only; no NeMo/vLLM/Ray/DeepSpeed/FSDP/DDP/Accelerate"
                ),
                "model_path": self.config.model.model_path,
                "train_path": self.config.data.train_path,
                "completion_parser": self._completion_parser_name(),
                "tp_size": self.config.native_tp.tp_size,
            }
        )

        samples = load_jsonl(self.config.data.train_path)
        self.total_samples = len(samples)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._resume_if_requested()
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "run_start",
                "backend": self.backend_name,
                "samples_total": self.total_samples,
                "resume": self.resume_info,
                "config": {
                    "rollout_group_size": self.config.training.rollout_group_size,
                    "optimize_prompt_batch_size": self.config.training.optimize_prompt_batch_size,
                    "optimize_times_per_step": self.config.training.optimize_times_per_step,
                    "replay_buffer_optimize_threshold": self.config.training.replay_buffer_optimize_threshold,
                    "rollout_max_retry_times": self.config.training.rollout_max_retry_times,
                    "training_epoch_count": self.config.training.training_epoch_count,
                    "max_steps": self.config.training.max_steps,
                    "max_new_tokens": self.config.training.max_new_tokens,
                    "save_steps": self.config.training.save_steps,
                    "activation_checkpointing_enabled": bool(
                        self.config.model.gradient_checkpointing
                    ),
                    "lora_target_modules": list(
                        self.config.lora.target_modules or [self.config.lora.target_preset]
                    ),
                    "forward_batch_size": self.config.native_tp.forward_batch_size,
                    "empty_cache_after_rollout_split": self.config.native_tp.empty_cache_after_rollout_split,
                    "synchronize_cuda_timing": self.config.native_tp.synchronize_cuda_timing,
                },
            }
        )

        try:
            start_epoch = int(self.current_epoch_stats.epoch)
            if (
                self.total_samples
                and int(self.current_epoch_stats.samples_seen) >= self.total_samples
            ):
                start_epoch += 1
            for epoch in range(start_epoch, self.config.training.training_epoch_count):
                if epoch != start_epoch or self.current_epoch_stats.samples_seen == 0:
                    self.current_epoch_stats = NativeEpochStats(epoch=epoch)
                epoch_samples = list(samples)
                random.Random(int(self.config.training.seed) + epoch).shuffle(epoch_samples)
                resume_sample_offset = (
                    int(self.current_epoch_stats.samples_seen) if epoch == start_epoch else 0
                )
                pending_samples = epoch_samples[resume_sample_offset:]
                # Process samples in groups that match the replay buffer threshold.
                # Each group of optimize_prompt_batch_size samples produces enough
                # completions to trigger one optimize step, interleaving rollout
                # generation with training for better GPU utilisation and lower
                # time-to-first-step latency.
                if not pending_samples:
                    continue
                queue_size = max(1, int(self.config.training.optimize_prompt_batch_size))
                for start in range(0, len(pending_samples), queue_size):
                    sample_queue = pending_samples[start : start + queue_size]
                    if self._sample_queue(sample_queue, epoch=epoch):
                        self._save_checkpoint(output_dir / "final", epoch=epoch)
                        return
                self._print_json(
                    {
                        "timestamp": _timestamp(),
                        "event": "epoch_summary",
                        "elapsed_sec": round(time.monotonic() - self.started_at, 3),
                        "epoch": self._epoch_summary(),
                        "run": self._run_summary(),
                    }
                )
                if self.config.training.save_epoch_checkpoint:
                    if len(self.replay_buffer) > 0:
                        self._maybe_optimize(epoch=epoch, force=True)
                    self._save_checkpoint(output_dir / f"epoch_{epoch}", epoch=epoch)
            if len(self.replay_buffer) > 0:
                self._maybe_optimize(
                    epoch=self.config.training.training_epoch_count - 1, force=True
                )
            self._save_checkpoint(
                output_dir / "final", epoch=self.config.training.training_epoch_count - 1
            )
        finally:
            self.runtime.close()

    def _sample_one(self, sample: Sample, *, epoch: int) -> bool:
        return self._sample_queue([sample], epoch=epoch)

    def _sample_queue(self, samples: list[Sample], *, epoch: int) -> bool:
        active = [_QueuedSample(sample=sample) for sample in samples]
        finished: list[_QueuedSample] = []
        max_attempts = self.config.training.rollout_max_retry_times + 1
        while active:
            attempt_records = self._rollout_queue_attempt(active, epoch=epoch)
            next_active: list[_QueuedSample] = []
            for state, record in zip(active, attempt_records, strict=True):
                if record.decision.should_retry:
                    state.attempts.append(record)
                    state.retry_count += 1
                    if state.retry_count >= max_attempts:
                        raise RuntimeError("GRASPO retry state exceeded configured max attempts")
                    next_active.append(state)
                else:
                    state.attempts.append(record)
                    finished.append(state)
            active = next_active

        stop_requested = False
        for state in finished:
            if self._finalize_sample(state, epoch=epoch):
                stop_requested = True
        return stop_requested

    def _rollout_queue_attempt(
        self, active: list[_QueuedSample], *, epoch: int
    ) -> list[_AttemptRecord]:
        rollout_started_at = time.monotonic()
        generations = self._generate_sample_groups([state.sample for state in active])
        rollout_sec = time.monotonic() - rollout_started_at
        rollout_sec_per_prompt = rollout_sec / max(len(generations), 1)
        records: list[_AttemptRecord] = []
        for state, generation in zip(active, generations, strict=True):
            attempt_started_at = time.monotonic()
            reward_started_at = time.monotonic()
            parsed_completions = [
                self._parse_completion(text, state.sample) for text in generation.completions
            ]
            results = [
                self.reward.score_parsed(
                    parsed,
                    state.sample.targets,
                    is_tool_call=state.sample.expects_tool_calls,
                )
                for parsed in parsed_completions
            ]
            reward_cpu_sec = time.monotonic() - reward_started_at
            rewards = [float(result.reward) for result in results]
            content_scores = [float(result.content_score) for result in results]
            all_right = [bool(result.all_right) for result in results]
            reward_details = [_reward_detail(result) for result in results]
            decision_started_at = time.monotonic()
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            if best_idx < len(parsed_completions):
                best_parsed = parsed_completions[best_idx]
                has_parse_error = bool(best_parsed.parse_errors)
                # Also catch tool_call_count_mismatch: the best completion produced
                # valid XML/JSON but the wrong number of tool calls.  The parser
                # does not flag this as a parse error, so we check it explicitly.
                has_count_mismatch = False
                if state.sample.expects_tool_calls and state.sample.targets and not has_parse_error:
                    first_target = state.sample.targets[0]
                    tc = first_target.get("output", {}).get("tool_calls")
                    expected_count = len(tc) if isinstance(tc, list) else 0
                    actual_count = len(best_parsed.tool_calls)
                    has_count_mismatch = expected_count > 0 and actual_count != expected_count
                best_has_parse_error = has_parse_error or has_count_mismatch
            else:
                best_has_parse_error = False
            decision = classify_group(
                rewards,
                content_scores,
                retry_count=state.retry_count,
                rollout_max_retry_times=self.config.training.rollout_max_retry_times,
                perfect_skip_reward_threshold=self.config.training.perfect_skip_reward_threshold,
                best_completion_has_parse_error=best_has_parse_error,
                skip_format_broken_groups=self.config.training.skip_format_broken_groups,
            )
            decision_sec = time.monotonic() - decision_started_at
            self.stats.total_groups += 1
            if decision.should_retry:
                self.stats.retries += 1

            readable = self._group_payload(
                sample=state.sample,
                epoch=epoch,
                generation=generation,
                rewards=rewards,
                content_scores=content_scores,
                all_right=all_right,
                reward_details=reward_details,
                parsed_completions=parsed_completions,
                decision=decision,
                retry_count=state.retry_count,
            )
            if self._is_primary():
                self.logger.write_readable(readable)
            timing = {
                "rollout_sec": rollout_sec_per_prompt,
                "reward_cpu_sec": reward_cpu_sec,
                "decision_sec": decision_sec,
                "old_logprob_sec": 0.0,
                "replay_append_sec": 0.0,
                "attempt_total_sec": time.monotonic() - attempt_started_at,
                "decision": decision.decision.value,
                "retry_count": state.retry_count,
                "reward_max_median_gap": decision.reward_max_median_gap,
                "completion_count": len(generation.completions),
                "sequence_len": int(getattr(generation.sequences, "shape", [0, 0])[1]),
                "generated_tokens_max": max(_generated_token_counts(generation), default=0),
                **_scalar_generation_timing(generation.metadata or {}),
            }

            records.append(
                _AttemptRecord(
                    sample=state.sample,
                    generation=generation,
                    parsed_completions=parsed_completions,
                    rewards=rewards,
                    content_scores=content_scores,
                    all_right=all_right,
                    reward_details=reward_details,
                    decision=decision,
                    retry_count=state.retry_count,
                    readable=readable,
                    timing=timing,
                )
            )
        return records

    def _generate_groups(self, samples: list[Sample]) -> list[NativeGeneration]:
        message_batches = [sample.messages for sample in samples]
        tool_batches = [sample.tools for sample in samples]
        generate_groups = getattr(self.runtime, "generate_groups", None)
        if callable(generate_groups):
            generations = generate_groups(
                message_batches=message_batches,
                tool_batches=tool_batches,
                rollout_group_size=self.config.training.rollout_group_size,
                max_new_tokens=self.config.training.max_new_tokens,
                max_prompt_length=self.config.data.max_prompt_length,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            if len(generations) != len(message_batches):
                raise RuntimeError(
                    f"native-tp generate_groups returned {len(generations)} groups for {len(message_batches)} prompts"
                )
            return generations
        return [
            self.runtime.generate_group(
                messages=messages,
                tools=tools,
                rollout_group_size=self.config.training.rollout_group_size,
                max_new_tokens=self.config.training.max_new_tokens,
                max_prompt_length=self.config.data.max_prompt_length,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            for messages, tools in zip(message_batches, tool_batches, strict=True)
        ]

    def _generate_sample_groups(self, samples: list[Sample]) -> list[NativeGeneration]:
        if not any(sample.media for sample in samples):
            return self._generate_groups(samples)
        generate_sample_groups = getattr(self.runtime, "generate_sample_groups", None)
        if not callable(generate_sample_groups):
            raise RuntimeError(
                "Input samples contain image/video media, but the selected native runtime "
                "does not implement multimodal generate_sample_groups"
            )
        generations = generate_sample_groups(
            samples=samples,
            rollout_group_size=self.config.training.rollout_group_size,
            max_new_tokens=self.config.training.max_new_tokens,
            max_prompt_length=self.config.data.max_prompt_length,
            temperature=self.config.training.temperature,
            top_p=self.config.training.top_p,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        if len(generations) != len(samples):
            raise RuntimeError(
                f"native-tp generate_sample_groups returned {len(generations)} "
                f"groups for {len(samples)} samples"
            )
        return generations

    def _parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion:
        parse_completion = getattr(self.runtime, "parse_completion", None)
        if callable(parse_completion):
            return parse_completion(completion, sample)
        return raw_parsed_completion(completion)

    def _completion_parser_name(self) -> str:
        adapter = getattr(self.runtime, "_adapter", None)
        if adapter is not None:
            parser_name = getattr(adapter, "completion_parser_name", None)
            if parser_name:
                return str(parser_name)
            if hasattr(adapter, "parse_completion"):
                return adapter.__class__.__name__
        if hasattr(self.runtime, "parse_completion"):
            return self.runtime.__class__.__name__
        return "raw"

    def _finalize_sample(self, state: _QueuedSample, *, epoch: int) -> bool:
        if not state.attempts:
            raise RuntimeError("Cannot finalize queued sample without rollout attempts")
        record = state.attempts[-1]
        generation = record.generation
        rewards = record.rewards
        decision = record.decision
        readable = record.readable
        timing = record.timing
        if not decision.should_train:
            if decision.decision.value == "perfect_skip":
                self.stats.perfect_skipped += 1
            elif decision.decision.value == "invalid_no_preference_gap":
                self.stats.invalid_no_preference_gap += 1
            else:
                self.stats.invalid += 1
            if self._is_primary():
                self.logger.write_raw({**readable, "raw": self._raw_generation(generation)})
            self._commit_sample_attempts(state, epoch=epoch)
            return self._finish_sample_and_maybe_optimize(epoch=epoch)
        if not has_reward_variance(rewards):
            self.stats.invalid += 1
            if self._is_primary():
                self.logger.write_raw({**readable, "raw": self._raw_generation(generation)})
            timing["decision"] = "invalid"
            self._commit_sample_attempts(state, epoch=epoch)
            return self._finish_sample_and_maybe_optimize(epoch=epoch)

        old_logprob_started_at = time.monotonic()
        if isinstance(generation.metadata, dict) and "_multimodal_rows" in generation.metadata:
            old_log_probs = self.runtime.sequence_log_probs(
                generation.sequences,
                generation.attention_mask,
                metadata=generation.metadata,
            )
        else:
            old_log_probs = self.runtime.sequence_log_probs(
                generation.sequences,
                generation.attention_mask,
            )
        timing["old_logprob_sec"] = time.monotonic() - old_logprob_started_at
        replay_started_at = time.monotonic()
        advantages = _expand_advantages_like(rewards, old_log_probs)
        self._append_experiences(generation, rewards, old_log_probs, advantages)
        timing["replay_append_sec"] = time.monotonic() - replay_started_at
        timing["attempt_total_sec"] = (
            float(timing.get("rollout_sec") or 0.0)
            + float(timing.get("reward_cpu_sec") or 0.0)
            + float(timing.get("decision_sec") or 0.0)
            + float(timing.get("old_logprob_sec") or 0.0)
            + float(timing.get("replay_append_sec") or 0.0)
        )
        self.stats.trainable += 1
        if decision.decision.value == "trainable_max_correct":
            self.stats.trainable_max_correct += 1
        else:
            self.stats.trainable_not_correct += 1
        if self._is_primary():
            self.logger.write_raw(
                {
                    **readable,
                    "raw": {
                        **self._raw_generation(generation),
                        "old_log_probs": old_log_probs,
                        "advantages": advantages,
                    },
                }
            )
        self._commit_sample_attempts(state, epoch=epoch)
        return self._finish_sample_and_maybe_optimize(epoch=epoch)

    def _commit_sample_attempts(self, state: _QueuedSample, *, epoch: int) -> None:
        for record in state.attempts:
            self.recent_groups.append(_monitor_group(record.readable))
            self.pending_batch_attempts.append(record.readable)
            self._record_epoch_attempt(record.readable)
            self._record_attempt_timing(
                epoch=epoch, retry_count=record.retry_count, timing=record.timing
            )

    def _finish_sample_and_maybe_optimize(self, *, epoch: int) -> bool:
        self.current_epoch_stats.samples_seen += 1
        self.sample_index += 1
        return (
            self._maybe_optimize(epoch=epoch)
            and 0 < self.config.training.max_steps <= self.global_step
        )

    def _maybe_optimize(self, *, epoch: int, force: bool = False) -> bool:
        threshold = self.config.training.replay_buffer_optimize_threshold
        if len(self.replay_buffer) < threshold and not force:
            return False
        if len(self.replay_buffer) == 0:
            return False

        usable = len(self.replay_buffer) if force else threshold
        data = self.replay_buffer.take(usable)
        optimize_started_at = time.monotonic()
        metrics = self.runtime.train_batch(
            data,
            policy_ratio_clip_eps=self.config.training.policy_ratio_clip_eps,
            optimize_times_per_step=self.config.training.optimize_times_per_step,
            max_grad_norm=self.config.training.max_grad_norm,
        )
        optimize_sec = time.monotonic() - optimize_started_at
        attempts = list(self.pending_batch_attempts)
        timings = list(self.pending_batch_timings)
        reward_batch = _reward_batch_summary(
            attempts,
            rollout_group_size=self.config.training.rollout_group_size,
            optimize_prompt_batch_size=self.config.training.optimize_prompt_batch_size,
        )
        metrics["replay_buffer_optimize_threshold"] = threshold
        metrics["replay_buffer_trainable_completion_count"] = usable
        metrics["replay_buffer_trainable_group_count"] = usable / max(
            int(self.config.training.rollout_group_size), 1
        )
        metrics["optimize_prompt_batch_size"] = self.config.training.optimize_prompt_batch_size
        metrics["optimize_times_per_step"] = self.config.training.optimize_times_per_step
        metrics["force_flush"] = bool(force)
        self.replay_buffer.clear()
        self.pending_batch_attempts.clear()
        self.pending_batch_timings.clear()
        self.global_step += 1
        self.stats.optimized_steps += 1
        checkpoint_sec = 0.0
        checkpoint_dir: Path | None = None
        if (
            self.config.training.save_steps > 0
            and self.global_step % self.config.training.save_steps == 0
        ):
            checkpoint_dir = Path(self.config.training.output_dir) / f"step_{self.global_step}"
            checkpoint_started_at = time.monotonic()
            self._save_checkpoint(checkpoint_dir, epoch=epoch)
            checkpoint_sec = time.monotonic() - checkpoint_started_at
        reward_window = _reward_window_summary(self.recent_groups)
        health = _training_health(metrics, reward_batch, reward_window)
        optimize = _compact_optimize_metrics(metrics)
        batch = _compact_batch_summary(reward_batch)
        timing = _compact_timing_summary(
            timings,
            optimize_sec=optimize_sec,
            checkpoint_sec=checkpoint_sec,
            metrics=metrics,
        )
        if self._is_primary():
            self.logger.write_train_batch_readable(
                {
                    "backend": self.backend_name,
                    "epoch": epoch,
                    "step": self.global_step,
                    "timestamp": _timestamp(),
                    "batch": batch,
                    "optimize": optimize,
                    "health": health,
                    "timing": timing,
                    "attempts": attempts,
                }
            )
            self.logger.write_timing_event(
                self._timing_event(
                    phase="optimize",
                    duration_sec=optimize_sec + checkpoint_sec,
                    epoch=epoch,
                    details={
                        **timing,
                        "force_flush": bool(force),
                        "replay_buffer_trainable_completion_count": usable,
                    },
                )
            )
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "train_step",
                "backend": self.backend_name,
                "step": self.global_step,
                "elapsed_sec": round(time.monotonic() - self.started_at, 3),
                "run": self._run_summary(),
                "epoch": self._epoch_summary(),
                "batch": batch,
                "optimize": optimize,
                "timing": timing,
                "health": health,
            }
        )
        if checkpoint_dir is not None:
            self._print_json(
                {
                    "timestamp": _timestamp(),
                    "event": "checkpoint_saved",
                    "step": self.global_step,
                    "path": str(checkpoint_dir),
                    "checkpoint_save_sec": round(checkpoint_sec, 6),
                }
            )
        return True

    def _save_checkpoint(self, path: Path, *, epoch: int) -> None:
        save_native_checkpoint(
            self.runtime,
            path,
            trainer_state=self._checkpoint_trainer_state(epoch=epoch),
        )
        if path.name == "final" and self.config.export.final_formats and self.runtime.is_primary():
            self._export_final_checkpoint(path)

    def _export_final_checkpoint(self, checkpoint_dir: Path) -> None:
        from graspo.backends.native_tp.lora_io import export_from_checkpoint

        for export_format in self.config.export.final_formats:
            output_dir = checkpoint_dir / str(export_format)
            export_from_checkpoint(
                checkpoint_dir,
                output_dir,
                export_format=str(export_format),
                base_model_path=self.config.model.model_path,
            )
            self._print_json(
                {
                    "timestamp": _timestamp(),
                    "event": "checkpoint_exported",
                    "checkpoint": str(checkpoint_dir),
                    "format": str(export_format),
                    "output": str(output_dir),
                }
            )

    def _resume_if_requested(self) -> None:
        checkpoint = self.config.training.resume_from_checkpoint
        if not checkpoint:
            return
        checkpoint_dir = Path(checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"training.resume_from_checkpoint does not exist: {checkpoint_dir}"
            )
        loader = getattr(self.runtime, "load_checkpoint", None)
        if not callable(loader):
            raise RuntimeError("Selected runtime does not support checkpoint resume")
        trainer_state = loader(checkpoint_dir)
        if trainer_state is None:
            raise RuntimeError(
                "GRASPO checkpoint is missing trainer_state; latest-only resume requires "
                "a current recoverable native checkpoint"
            )
        if trainer_state.get("format") != "graspo-native-tp-trainer-state":
            raise RuntimeError(
                "Unsupported trainer_state format: "
                f"{trainer_state.get('format')!r}; latest-only resume requires current GRASPO"
            )
        self._restore_trainer_state(trainer_state)
        self.resume_info = {
            "checkpoint": str(checkpoint_dir),
            "global_step": self.global_step,
            "epoch": self.current_epoch_stats.epoch,
            "samples_seen": self.current_epoch_stats.samples_seen,
        }
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "checkpoint_resumed",
                **self.resume_info,
            }
        )

    def _checkpoint_trainer_state(self, *, epoch: int) -> dict[str, Any]:
        if len(self.replay_buffer) > 0:
            raise RuntimeError("Cannot save recoverable checkpoint while ReplayBuffer is non-empty")
        return {
            "format": "graspo-native-tp-trainer-state",
            "version": 1,
            "global_step": self.global_step,
            "sample_index": self.sample_index,
            "total_samples": self.total_samples,
            "epoch": epoch,
            "run_stats": _train_stats_to_dict(self.stats),
            "epoch_stats": _epoch_stats_to_dict(self.current_epoch_stats),
            "config_snapshot": {
                "backend": self.backend_name,
                "rollout_group_size": self.config.training.rollout_group_size,
                "optimize_prompt_batch_size": self.config.training.optimize_prompt_batch_size,
                "optimize_times_per_step": self.config.training.optimize_times_per_step,
                "rollout_max_retry_times": self.config.training.rollout_max_retry_times,
                "max_new_tokens": self.config.training.max_new_tokens,
            },
        }

    def _restore_trainer_state(self, state: dict[str, Any]) -> None:
        self.global_step = int(state["global_step"])
        self.sample_index = int(state.get("sample_index") or 0)
        self.stats = _train_stats_from_dict(state.get("run_stats") or {})
        self.current_epoch_stats = _epoch_stats_from_dict(state.get("epoch_stats") or {})
        if self.current_epoch_stats.epoch < 0:
            self.current_epoch_stats.epoch = int(state.get("epoch") or 0)
        self.stats.optimized_steps = max(self.stats.optimized_steps, self.global_step)
        self.replay_buffer.clear()
        self.pending_batch_attempts.clear()
        self.pending_batch_timings.clear()

    def _record_attempt_timing(
        self, *, epoch: int, retry_count: int, timing: dict[str, Any]
    ) -> None:
        self.pending_batch_timings.append(timing)
        if self._is_primary():
            self.logger.write_timing_event(
                self._timing_event(
                    phase="rollout_attempt",
                    duration_sec=float(timing.get("attempt_total_sec") or 0.0),
                    epoch=epoch,
                    sample_index=self.sample_index,
                    attempt_index=retry_count + 1,
                    retry_count=retry_count,
                    details=timing,
                )
            )

    def _timing_event(
        self,
        *,
        phase: str,
        duration_sec: float,
        epoch: int,
        details: dict[str, Any],
        sample_index: int | None = None,
        attempt_index: int | None = None,
        retry_count: int | None = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": _timestamp(),
            "elapsed_sec": round(time.monotonic() - self.started_at, 6),
            "phase": phase,
            "duration_sec": round(float(duration_sec), 6),
            "step": self.global_step,
            "epoch": epoch,
            "sample_index": sample_index,
            "attempt_index": attempt_index,
            "retry_count": retry_count,
            "rank": int(getattr(self.runtime, "rank", 0)),
            "tp_rank": int(getattr(self.runtime, "tp_rank", 0)),
            "details": _round_timing_details(details),
        }

    def _append_experiences(
        self,
        generation: NativeGeneration,
        rewards: list[float],
        old_log_probs: Any,
        advantages: Any,
    ) -> None:
        import torch

        reward_tensor = torch.tensor(
            rewards,
            dtype=old_log_probs.dtype,
            device=old_log_probs.device,
        )
        items: list[Experience] = []
        for idx in range(len(rewards)):
            items.append(
                Experience(
                    sequences=generation.sequences[idx].detach().cpu(),
                    old_log_probs=old_log_probs[idx].detach().cpu(),
                    advantages=advantages[idx].detach().cpu(),
                    attention_mask=generation.attention_mask[idx].detach().cpu(),
                    action_mask=generation.action_mask[idx].detach().cpu(),
                    rewards=reward_tensor[idx].detach().cpu(),
                    metadata=_experience_metadata_for_row(generation.metadata, idx),
                )
            )
        self.replay_buffer.append_many(items)

    def _group_payload(
        self,
        *,
        sample: Sample,
        epoch: int,
        generation: NativeGeneration,
        rewards: list[float],
        content_scores: list[float],
        all_right: list[bool],
        reward_details: list[dict[str, Any]],
        parsed_completions: list[ParsedCompletion],
        decision: Any,
        retry_count: int,
    ) -> dict[str, Any]:
        payload = {
            "event": "graspo_group",
            "backend": self.backend_name,
            "epoch": epoch,
            "step": self.global_step,
            "sample_index": self.sample_index,
            "messages": sample.messages,
            "tools": sample.tools,
            "prompt_preview": sample.prompt_preview,
            "targets": sample.targets,
            "metadata": _safe_sample_metadata(sample),
            "completions": generation.completions,
            "parsed_completions": [parsed.to_dict() for parsed in parsed_completions],
            "rewards": rewards,
            "content_scores": content_scores,
            "all_right": all_right,
            "reward_details": reward_details,
            "generated_tokens": _generated_token_counts(generation),
            "decision": decision.decision.value,
            "attempt_index": retry_count + 1,
            "max_attempts": self.config.training.rollout_max_retry_times + 1,
            "retry_count": retry_count,
            "group_stats": _group_stats(rewards),
            "reward_max_median_gap": decision.reward_max_median_gap,
            "generation_metadata": _public_generation_metadata(generation.metadata or {}),
        }
        if decision.decision.value == "invalid_no_preference_gap":
            payload["invalid_reason"] = "no_preference_gap"
        return payload

    @staticmethod
    def _raw_generation(generation: NativeGeneration) -> dict[str, Any]:
        return {
            "sequences": generation.sequences,
            "attention_mask": generation.attention_mask,
            "action_mask": generation.action_mask,
            "prompt_len": generation.prompt_len,
        }

    def _record_epoch_attempt(self, payload: dict[str, Any]) -> None:
        rewards = [float(value) for value in payload.get("rewards", [])]
        content_scores = [float(value) for value in payload.get("content_scores", [])]
        decision = str(payload.get("decision"))
        self.current_epoch_stats.attempt_groups += 1
        self.current_epoch_stats.completion_count += len(rewards)
        if rewards:
            self.current_epoch_stats.reward_mean_sum += sum(rewards) / len(rewards)
            self.current_epoch_stats.best_reward = max(
                self.current_epoch_stats.best_reward, max(rewards)
            )
        if content_scores:
            self.current_epoch_stats.content_mean_sum += sum(content_scores) / len(content_scores)
        if decision == "retry":
            self.current_epoch_stats.retries += 1
        elif decision == "perfect_skip":
            self.current_epoch_stats.perfect_skipped += 1
        elif decision == "invalid":
            self.current_epoch_stats.invalid += 1
        elif decision == "invalid_no_preference_gap":
            self.current_epoch_stats.invalid_no_preference_gap += 1
        elif decision == "trainable_max_correct":
            self.current_epoch_stats.trainable += 1
            self.current_epoch_stats.trainable_max_correct += 1
        elif decision == "trainable_not_correct":
            self.current_epoch_stats.trainable += 1
            self.current_epoch_stats.trainable_not_correct += 1

    def _run_summary(self) -> dict[str, Any]:
        return {
            "attempt_groups": self.stats.total_groups,
            "completions": self.stats.total_groups * int(self.config.training.rollout_group_size),
            "decisions": _compact_decisions(
                perfect_skip=self.stats.perfect_skipped,
                trainable_max_correct=self.stats.trainable_max_correct,
                trainable_not_correct=self.stats.trainable_not_correct,
                invalid=self.stats.invalid,
                invalid_no_preference_gap=self.stats.invalid_no_preference_gap,
                retry_attempts=self.stats.retries,
            ),
            "optimized_steps": self.stats.optimized_steps,
        }

    def _epoch_summary(self) -> dict[str, Any]:
        stats = self.current_epoch_stats
        attempt_groups = max(stats.attempt_groups, 1)
        return {
            "epoch": stats.epoch,
            "samples_seen": stats.samples_seen,
            "samples_total": self.total_samples,
            "progress": stats.samples_seen / self.total_samples if self.total_samples else 0.0,
            "attempt_groups": stats.attempt_groups,
            "completions": stats.completion_count,
            "decisions": _compact_decisions(
                perfect_skip=stats.perfect_skipped,
                trainable_max_correct=stats.trainable_max_correct,
                trainable_not_correct=stats.trainable_not_correct,
                invalid=stats.invalid,
                invalid_no_preference_gap=stats.invalid_no_preference_gap,
                retry_attempts=stats.retries,
            ),
            "reward_mean": stats.reward_mean_sum / attempt_groups if stats.attempt_groups else 0.0,
            "content_mean": stats.content_mean_sum / attempt_groups
            if stats.attempt_groups
            else 0.0,
            "best_reward": stats.best_reward,
        }

    def _print_json(self, payload: dict[str, Any]) -> None:
        if self._is_primary():
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _is_primary(self) -> bool:
        is_primary = getattr(self.runtime, "is_primary", None)
        if callable(is_primary):
            return bool(is_primary())
        return int(getattr(self.runtime, "rank", 0)) == 0


def _expand_advantages_like(rewards: list[float], old_log_probs: Any) -> Any:
    import torch

    values = torch.tensor(
        group_advantages(rewards),
        dtype=old_log_probs.dtype,
        device=old_log_probs.device,
    ).unsqueeze(1)
    return values.expand_as(old_log_probs)


def _train_stats_to_dict(stats: NativeTrainStats) -> dict[str, Any]:
    return {
        "total_groups": stats.total_groups,
        "perfect_skipped": stats.perfect_skipped,
        "retries": stats.retries,
        "invalid": stats.invalid,
        "invalid_no_preference_gap": stats.invalid_no_preference_gap,
        "trainable": stats.trainable,
        "trainable_max_correct": stats.trainable_max_correct,
        "trainable_not_correct": stats.trainable_not_correct,
        "optimized_steps": stats.optimized_steps,
    }


def _epoch_stats_to_dict(stats: NativeEpochStats) -> dict[str, Any]:
    return {
        "epoch": stats.epoch,
        "samples_seen": stats.samples_seen,
        "attempt_groups": stats.attempt_groups,
        "completion_count": stats.completion_count,
        "perfect_skipped": stats.perfect_skipped,
        "retries": stats.retries,
        "invalid": stats.invalid,
        "invalid_no_preference_gap": stats.invalid_no_preference_gap,
        "trainable": stats.trainable,
        "trainable_max_correct": stats.trainable_max_correct,
        "trainable_not_correct": stats.trainable_not_correct,
        "reward_mean_sum": stats.reward_mean_sum,
        "content_mean_sum": stats.content_mean_sum,
        "best_reward": stats.best_reward,
    }


def _train_stats_from_dict(raw: dict[str, Any]) -> NativeTrainStats:
    return NativeTrainStats(
        total_groups=int(raw.get("total_groups") or raw.get("attempt_groups") or 0),
        perfect_skipped=int(raw.get("perfect_skipped") or 0),
        retries=int(raw.get("retries") or 0),
        invalid=int(raw.get("invalid") or 0),
        invalid_no_preference_gap=int(raw.get("invalid_no_preference_gap") or 0),
        trainable=int(raw.get("trainable") or 0),
        trainable_max_correct=int(raw.get("trainable_max_correct") or 0),
        trainable_not_correct=int(raw.get("trainable_not_correct") or 0),
        optimized_steps=int(raw.get("optimized_steps") or 0),
    )


def _epoch_stats_from_dict(raw: dict[str, Any]) -> NativeEpochStats:
    return NativeEpochStats(
        epoch=int(raw.get("epoch") or 0),
        samples_seen=int(raw.get("samples_seen") or 0),
        attempt_groups=int(raw.get("attempt_groups") or 0),
        completion_count=int(raw.get("completion_count") or raw.get("completions") or 0),
        perfect_skipped=int(raw.get("perfect_skipped") or 0),
        retries=int(raw.get("retries") or 0),
        invalid=int(raw.get("invalid") or 0),
        invalid_no_preference_gap=int(raw.get("invalid_no_preference_gap") or 0),
        trainable=int(raw.get("trainable") or 0),
        trainable_max_correct=int(raw.get("trainable_max_correct") or 0),
        trainable_not_correct=int(raw.get("trainable_not_correct") or 0),
        reward_mean_sum=float(raw.get("reward_mean_sum") or 0.0),
        content_mean_sum=float(raw.get("content_mean_sum") or 0.0),
        best_reward=float(raw.get("best_reward") or 0.0),
    )


def _group_stats(rewards: list[float]) -> dict[str, float | int]:
    if not rewards:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0, "range": 0.0}
    minimum = min(rewards)
    maximum = max(rewards)
    return {
        "count": len(rewards),
        "min": minimum,
        "median": lower_median(rewards),
        "max": maximum,
        "mean": sum(rewards) / len(rewards),
        "range": maximum - minimum,
    }


def _reward_detail(result: Any) -> dict[str, Any]:
    extracted = dict(result.extracted)
    valid_extracted_json = None
    if "answer" in extracted:
        try:
            answer = extracted["answer"]
            if isinstance(answer, str) and answer.strip():
                json.loads(answer.strip())
                valid_extracted_json = True
        except (TypeError, ValueError):
            valid_extracted_json = False
    return {
        "raw_score": float(result.raw_score),
        "max_score": float(result.max_score),
        "extracted": extracted,
        "parsed_tool_calls": extracted.get("tool_calls"),
        "parser": extracted.get("parser"),
        "parse_errors": extracted.get("parse_errors"),
        "extra_text": extracted.get("extra_text"),
        "matched_target_index": result.matched_target_index,
        "matched_target_id": result.matched_target_id,
        "target_scores": result.target_scores,
        "useless_text_length": len(result.useless_text),
        "valid_extracted_json": valid_extracted_json,
    }


def _generated_token_counts(generation: NativeGeneration) -> list[int]:
    try:
        return [int(value) for value in generation.action_mask.detach().sum(dim=1).cpu().tolist()]
    except Exception:
        return []


def _safe_sample_metadata(sample: Sample) -> dict[str, Any]:
    redacted = {
        key: value
        for key, value in sample.metadata.items()
        if key not in {"image", "images", "video", "videos"}
    }
    if sample.media:
        counts: dict[str, int] = {}
        for item in sample.media:
            media_type = str(item.get("type") or "unknown")
            counts[media_type] = counts.get(media_type, 0) + 1
        redacted["media"] = {
            "count": len(sample.media),
            "types": counts,
        }
    return redacted


def _public_generation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in metadata.items() if not str(key).startswith("_")}
    private_rows = metadata.get("_multimodal_rows")
    if isinstance(private_rows, list) and private_rows:
        media_counts: dict[str, int] = {}
        for row in private_rows:
            if not isinstance(row, dict):
                continue
            for item in row.get("media") or []:
                if not isinstance(item, dict):
                    continue
                media_type = str(item.get("type") or "unknown")
                media_counts[media_type] = media_counts.get(media_type, 0) + 1
        public["multimodal"] = {
            "row_count": len(private_rows),
            "media_counts": media_counts,
        }
    return public


def _experience_metadata_for_row(
    metadata: dict[str, Any] | None, row_index: int
) -> dict[str, Any] | None:
    if not metadata:
        return None
    rows = metadata.get("_multimodal_rows")
    if isinstance(rows, list):
        if row_index >= len(rows):
            raise RuntimeError(
                f"generation multimodal metadata has {len(rows)} rows, cannot index row {row_index}"
            )
        return {"_multimodal_rows": [rows[row_index]]}
    return dict(metadata)


def _is_pure_tool_call_task(targets: Any) -> bool:
    """Return True when targets only use tool_calls (no output.content entries)."""
    if not isinstance(targets, list) or not targets:
        return False
    has_content = any(
        isinstance(t, dict) and isinstance(t.get("output"), dict) and "content" in t["output"]
        for t in targets
    )
    has_tool_calls = any(
        isinstance(t, dict) and isinstance(t.get("output"), dict) and "tool_calls" in t["output"]
        for t in targets
    )
    return has_tool_calls and not has_content


def _monitor_group(payload: dict[str, Any]) -> dict[str, Any]:
    rewards = [float(value) for value in payload.get("rewards", [])]
    content_scores = [float(value) for value in payload.get("content_scores", [])]
    details = payload.get("reward_details", [])
    completions = payload.get("completions", [])
    pure_tool_call = _is_pure_tool_call_task(payload.get("targets"))
    return {
        "decision": payload.get("decision"),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_range": max(rewards) - min(rewards) if rewards else 0.0,
        "content_mean": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "content_all_zero": bool(content_scores) and all(value == 0.0 for value in content_scores),
        "content_all_one": bool(content_scores) and all(value == 1.0 for value in content_scores),
        "missing_json_marker_count": (
            0 if pure_tool_call else sum(1 for text in completions if "```json" not in text)
        ),
        "unclosed_json_fence_count": (
            0
            if pure_tool_call
            else sum(1 for text in completions if "```json" in text and text.count("```") < 2)
        ),
        "invalid_extracted_json_count": (
            0
            if pure_tool_call
            else sum(1 for detail in details if detail.get("valid_extracted_json") is False)
        ),
        "likely_truncated_json_count": sum(
            1
            for text, detail in zip(completions, details, strict=False)
            if _likely_truncated_json(text, detail)
        ),
        "tool_call_parse_error_count": sum(1 for detail in details if detail.get("parse_errors")),
        "tool_call_count_mismatch_count": _tool_call_count_mismatch_count(
            details, payload.get("targets")
        ),
    }


def _reward_window_summary(groups: deque[dict[str, Any]]) -> dict[str, Any]:
    items = list(groups)
    if not items:
        return {
            "count": 0,
            "decision_counts": {},
            "reward_mean_avg": 0.0,
            "reward_max_avg": 0.0,
            "nonzero_range_rate": 0.0,
            "content_mean_avg": 0.0,
            "content_all_zero_rate": 0.0,
            "content_all_one_rate": 0.0,
            "missing_json_marker_count": 0,
            "unclosed_json_fence_count": 0,
            "invalid_extracted_json_count": 0,
            "likely_truncated_json_count": 0,
            "tool_call_parse_error_count": 0,
            "tool_call_count_mismatch_count": 0,
        }
    decision_counts: dict[str, int] = {}
    for item in items:
        decision = str(item.get("decision"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    count = len(items)
    return {
        "count": count,
        "decision_counts": decision_counts,
        "reward_mean_avg": sum(float(item["reward_mean"]) for item in items) / count,
        "reward_max_avg": sum(float(item["reward_max"]) for item in items) / count,
        "nonzero_range_rate": sum(float(item["reward_range"]) > 0.0 for item in items) / count,
        "content_mean_avg": sum(float(item["content_mean"]) for item in items) / count,
        "content_all_zero_rate": sum(bool(item["content_all_zero"]) for item in items) / count,
        "content_all_one_rate": sum(bool(item["content_all_one"]) for item in items) / count,
        "missing_json_marker_count": sum(int(item["missing_json_marker_count"]) for item in items),
        "unclosed_json_fence_count": sum(int(item["unclosed_json_fence_count"]) for item in items),
        "invalid_extracted_json_count": sum(
            int(item["invalid_extracted_json_count"]) for item in items
        ),
        "likely_truncated_json_count": sum(
            int(item["likely_truncated_json_count"]) for item in items
        ),
        "tool_call_parse_error_count": sum(
            int(item.get("tool_call_parse_error_count") or 0) for item in items
        ),
        "tool_call_count_mismatch_count": sum(
            int(item.get("tool_call_count_mismatch_count") or 0) for item in items
        ),
    }


def _reward_batch_summary(
    attempts: list[dict[str, Any]],
    *,
    rollout_group_size: int,
    optimize_prompt_batch_size: int,
) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    rewards: list[float] = []
    content_scores: list[float] = []
    group_ranges: list[float] = []
    group_max_median_gaps: list[float] = []
    missing_json_marker_count = 0
    unclosed_json_fence_count = 0
    invalid_extracted_json_count = 0
    likely_truncated_json_count = 0
    tool_call_parse_error_count = 0
    tool_call_count_mismatch_count = 0

    for attempt in attempts:
        decision = str(attempt.get("decision"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        attempt_rewards = [float(value) for value in attempt.get("rewards", [])]
        attempt_content = [float(value) for value in attempt.get("content_scores", [])]
        rewards.extend(attempt_rewards)
        content_scores.extend(attempt_content)
        if attempt_rewards:
            group_ranges.append(max(attempt_rewards) - min(attempt_rewards))
            group_max_median_gaps.append(max(attempt_rewards) - lower_median(attempt_rewards))
        completions = attempt.get("completions", [])
        details = attempt.get("reward_details", [])
        pure_tool_call = _is_pure_tool_call_task(attempt.get("targets"))
        if not pure_tool_call:
            missing_json_marker_count += sum(1 for text in completions if "```json" not in text)
            unclosed_json_fence_count += sum(
                1 for text in completions if "```json" in text and text.count("```") < 2
            )
            invalid_extracted_json_count += sum(
                1 for detail in details if detail.get("valid_extracted_json") is False
            )
        if not pure_tool_call:
            likely_truncated_json_count += sum(
                1
                for text, detail in zip(completions, details, strict=False)
                if _likely_truncated_json(text, detail)
            )
        tool_call_parse_error_count += sum(1 for detail in details if detail.get("parse_errors"))
        tool_call_count_mismatch_count += _tool_call_count_mismatch_count(
            details, attempt.get("targets")
        )

    attempt_group_count = len(attempts)
    trainable_group_count = decision_counts.get("trainable_max_correct", 0) + decision_counts.get(
        "trainable_not_correct", 0
    )
    retry_group_count = decision_counts.get("retry", 0)
    invalid_group_count = decision_counts.get("invalid", 0)
    invalid_no_preference_gap_group_count = decision_counts.get("invalid_no_preference_gap", 0)
    perfect_skip_group_count = decision_counts.get("perfect_skip", 0)
    return {
        "unit": "batch_attempt",
        "rollout_group_size": int(rollout_group_size),
        "optimize_prompt_batch_size": int(optimize_prompt_batch_size),
        "attempt_group_count": attempt_group_count,
        "completion_count": attempt_group_count * int(rollout_group_size),
        "observed_completion_count": len(rewards),
        "trainable_group_count": trainable_group_count,
        "trainable_completion_count": trainable_group_count * int(rollout_group_size),
        "retry_group_count": retry_group_count,
        "retry_completion_count": retry_group_count * int(rollout_group_size),
        "perfect_skip_group_count": perfect_skip_group_count,
        "perfect_skip_completion_count": perfect_skip_group_count * int(rollout_group_size),
        "invalid_group_count": invalid_group_count,
        "invalid_completion_count": invalid_group_count * int(rollout_group_size),
        "invalid_no_preference_gap_group_count": invalid_no_preference_gap_group_count,
        "invalid_no_preference_gap_completion_count": (
            invalid_no_preference_gap_group_count * int(rollout_group_size)
        ),
        "decision_counts": decision_counts,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_median": lower_median(rewards),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_nonzero_range_group_count": sum(value > 0.0 for value in group_ranges),
        "reward_nonzero_range_rate": (
            sum(value > 0.0 for value in group_ranges) / len(group_ranges) if group_ranges else 0.0
        ),
        "reward_max_median_gap_mean": (
            sum(group_max_median_gaps) / len(group_max_median_gaps)
            if group_max_median_gaps
            else 0.0
        ),
        "content_mean": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "content_all_zero_group_count": sum(
            bool(attempt.get("content_scores"))
            and all(float(value) == 0.0 for value in attempt.get("content_scores", []))
            for attempt in attempts
        ),
        "content_all_one_group_count": sum(
            bool(attempt.get("content_scores"))
            and all(float(value) == 1.0 for value in attempt.get("content_scores", []))
            for attempt in attempts
        ),
        "missing_json_marker_count": missing_json_marker_count,
        "unclosed_json_fence_count": unclosed_json_fence_count,
        "invalid_extracted_json_count": invalid_extracted_json_count,
        "likely_truncated_json_count": likely_truncated_json_count,
        "tool_call_parse_error_count": tool_call_parse_error_count,
        "tool_call_count_mismatch_count": tool_call_count_mismatch_count,
    }


def _likely_truncated_json(text: str, detail: dict[str, Any]) -> bool:
    has_json = "```json" in text
    if has_json and text.count("```") < 2:
        return True
    if detail.get("valid_extracted_json") is False and has_json:
        stripped = text.rstrip()
        return not (stripped.endswith("```") or stripped.endswith("}"))
    return False


def _compact_batch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    decisions = dict(summary.get("decision_counts") or {})
    return {
        "attempt_groups": int(summary.get("attempt_group_count") or 0),
        "completions": int(summary.get("completion_count") or 0),
        "decisions": _compact_decisions(
            perfect_skip=int(decisions.get("perfect_skip") or 0),
            trainable_max_correct=int(decisions.get("trainable_max_correct") or 0),
            trainable_not_correct=int(decisions.get("trainable_not_correct") or 0),
            invalid=int(decisions.get("invalid") or 0),
            invalid_no_preference_gap=int(decisions.get("invalid_no_preference_gap") or 0),
            retry_attempts=int(decisions.get("retry") or 0),
        ),
        "reward": {
            "min": float(summary.get("reward_min") or 0.0),
            "median": float(summary.get("reward_median") or 0.0),
            "mean": float(summary.get("reward_mean") or 0.0),
            "max": float(summary.get("reward_max") or 0.0),
            "nonzero_range_rate": float(summary.get("reward_nonzero_range_rate") or 0.0),
            "max_median_gap_mean": float(summary.get("reward_max_median_gap_mean") or 0.0),
        },
        "content": {
            "mean": float(summary.get("content_mean") or 0.0),
            "all_zero_groups": int(summary.get("content_all_zero_group_count") or 0),
            "all_one_groups": int(summary.get("content_all_one_group_count") or 0),
        },
        "debug": {
            "missing_json_marker": int(summary.get("missing_json_marker_count") or 0),
            "unclosed_json_fence": int(summary.get("unclosed_json_fence_count") or 0),
            "invalid_json": int(summary.get("invalid_extracted_json_count") or 0),
            "truncated_json": int(summary.get("likely_truncated_json_count") or 0),
            "tool_call_parse_error": int(summary.get("tool_call_parse_error_count") or 0),
            "tool_call_count_mismatch": int(summary.get("tool_call_count_mismatch_count") or 0),
        },
    }


def _tool_call_count_mismatch_count(details: list[dict[str, Any]], targets: Any) -> int:
    target_counts = _target_tool_call_counts(targets)
    return sum(
        1
        for detail in details
        if detail.get("parsed_tool_calls") is not None
        and len(detail.get("parsed_tool_calls") or []) not in target_counts
    )


def _target_tool_call_counts(targets: Any) -> set[int]:
    counts: set[int] = set()
    if not isinstance(targets, list):
        return {1}
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        calls = output.get("tool_calls") if isinstance(output, dict) else None
        if isinstance(calls, list):
            counts.add(len(calls))
    return counts or {1}


def _compact_decisions(
    *,
    perfect_skip: int,
    trainable_max_correct: int,
    trainable_not_correct: int,
    invalid: int,
    invalid_no_preference_gap: int,
    retry_attempts: int,
) -> dict[str, Any]:
    trainable_total = int(trainable_max_correct) + int(trainable_not_correct)
    terminal_total = (
        int(perfect_skip) + trainable_total + int(invalid) + int(invalid_no_preference_gap)
    )
    return {
        "rollout_attempts": {
            "total": terminal_total + int(retry_attempts),
            "retry": int(retry_attempts),
            "terminal": terminal_total,
        },
        "terminal": {
            "perfect_skip": int(perfect_skip),
            "trainable": trainable_total,
            "invalid": int(invalid),
            "invalid_no_preference_gap": int(invalid_no_preference_gap),
            "total": terminal_total,
        },
        "trainable": {
            "max_correct": int(trainable_max_correct),
            "not_correct": int(trainable_not_correct),
            "total": trainable_total,
        },
    }


def _compact_timing_summary(
    attempt_timings: list[dict[str, Any]],
    *,
    optimize_sec: float,
    checkpoint_sec: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rollout_sec = _sum_timing(attempt_timings, "rollout_sec")
    reward_cpu_sec = _sum_timing(attempt_timings, "reward_cpu_sec")
    decision_sec = _sum_timing(attempt_timings, "decision_sec")
    old_logprob_sec = _sum_timing(attempt_timings, "old_logprob_sec")
    replay_append_sec = _sum_timing(attempt_timings, "replay_append_sec")
    total = rollout_sec + reward_cpu_sec + decision_sec + old_logprob_sec + replay_append_sec
    total += float(optimize_sec) + float(checkpoint_sec)
    requested_queue_size = max(
        [int(item.get("rollout_prompt_queue_batch_size") or 1) for item in attempt_timings],
        default=1,
    )
    effective_queue_size = max(
        [int(item.get("rollout_prompt_queue_effective_size") or 1) for item in attempt_timings],
        default=1,
    )
    return {
        "attempt_count": len(attempt_timings),
        "rollout_prompt_queue_batch_size": requested_queue_size,
        "rollout_prompt_queue_effective_size": effective_queue_size,
        "rollout_prompt_queue_fallback_count": sum(
            bool(item.get("rollout_prompt_queue_fallback")) for item in attempt_timings
        ),
        "rollout_sec": round(rollout_sec, 6),
        "reward_cpu_sec": round(reward_cpu_sec, 6),
        "decision_sec": round(decision_sec, 6),
        "old_logprob_sec": round(old_logprob_sec, 6),
        "replay_append_sec": round(replay_append_sec, 6),
        "optimize_sec": round(float(optimize_sec), 6),
        "checkpoint_sec": round(float(checkpoint_sec), 6),
        "prefill_sec": round(_sum_timing(attempt_timings, "prefill_sec"), 6),
        "decode_sec": round(_sum_timing(attempt_timings, "decode_sec"), 6),
        "sampling_sec": round(_sum_timing(attempt_timings, "sampling_sec"), 6),
        "stop_check_sec": round(_sum_timing(attempt_timings, "stop_check_sec"), 6),
        "train_batch_total_sec": round(
            float(metrics.get("train_batch_total_sec") or optimize_sec), 6
        ),
        "optimize_round_sec_sum": round(float(metrics.get("optimize_round_sec_sum") or 0.0), 6),
        "micro_batch_forward_sec": round(float(metrics.get("micro_batch_forward_sec") or 0.0), 6),
        "backward_sec": round(float(metrics.get("backward_sec") or 0.0), 6),
        "optimizer_step_sec": round(float(metrics.get("optimizer_step_sec") or 0.0), 6),
        "micro_batch_count": int(
            metrics.get("micro_batch_count") or metrics.get("optimizer_steps") or 0
        ),
        "decode_tokens": int(_sum_timing(attempt_timings, "decode_tokens")),
        "rollout_generation_split_count": int(
            _sum_timing(attempt_timings, "rollout_generation_split_count")
        ),
        "total_observed_sec": round(total, 6),
    }


def _sum_timing(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item.get(key) or 0.0) for item in items)


def _round_timing_details(details: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        elif isinstance(value, (str, int, bool)) or value is None:
            rounded[key] = value
    return rounded


def _scalar_generation_timing(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "tokenize_sec",
        "prefill_sec",
        "decode_sec",
        "sampling_sec",
        "stop_check_sec",
        "decode_tokens",
        "rollout_elapsed_sec",
        "rollout_use_kv_cache",
        "rollout_generation_micro_batch_size",
        "rollout_generation_split_count",
        "rollout_empty_cache_after_split",
        "prefill_len",
        "generated_tokens_max",
        "rollout_prompt_queue_batch_size",
        "rollout_prompt_queue_effective_size",
        "rollout_prompt_queue_fallback",
    }
    return {key: value for key, value in metadata.items() if key in keys}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _compact_optimize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    optimizer_steps_per_rank = int(metrics.get("optimizer_steps") or 0)
    global_optimizer_steps = int(
        metrics.get("global_optimizer_steps_sum") or optimizer_steps_per_rank
    )
    return {
        "optimized": bool(metrics.get("optimized")),
        "replay_buffer_trainable_completion_count": int(
            metrics.get("replay_buffer_trainable_completion_count") or 0
        ),
        "replay_buffer_trainable_group_count": float(
            metrics.get("replay_buffer_trainable_group_count") or 0.0
        ),
        "replay_buffer_optimize_threshold": int(
            metrics.get("replay_buffer_optimize_threshold") or 0
        ),
        "optimize_prompt_batch_size": int(metrics.get("optimize_prompt_batch_size") or 0),
        "optimize_times_per_step": int(metrics.get("optimize_times_per_step") or 0),
        "optimizer_steps_per_rank": optimizer_steps_per_rank,
        "global_optimizer_steps_sum": global_optimizer_steps,
        "loss_mean": _metric_float(metrics, "global_loss_mean", "loss_mean"),
        "grad_norm_mean": _metric_float(metrics, "global_grad_norm_mean", "grad_norm_mean"),
        "lora_delta_mean": _metric_float(metrics, "global_lora_norm_delta_mean", "lora_norm_delta"),
        "skipped_nonfinite": int(metrics.get("skipped_nonfinite") or 0),
        "force_flush": bool(metrics.get("force_flush")),
    }


def _metric_float(metrics: dict[str, Any], preferred: str, fallback: str) -> float:
    value = metrics.get(preferred)
    if value is None:
        value = metrics.get(fallback)
    return float(value or 0.0)


def _training_health(
    metrics: dict[str, Any],
    reward_batch: dict[str, Any],
    reward_window: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(metrics.get("skipped_nonfinite") or 0) > 0:
        reasons.append("nonfinite_loss_or_grad")
    lora_delta = _metric_float(metrics, "global_lora_norm_delta_mean", "lora_norm_delta")
    if metrics.get("optimized") and lora_delta == 0.0:
        reasons.append("zero_lora_delta")
    if int(reward_batch.get("attempt_group_count") or 0) > 0:
        if float(reward_batch.get("reward_mean") or 0.0) == 0.0:
            reasons.append("batch_reward_all_zero")
        if int(reward_batch.get("likely_truncated_json_count") or 0) > 0:
            reasons.append("batch_json_truncation_detected")
    if int(reward_window.get("count") or 0) >= 10:
        if float(reward_window.get("reward_mean_avg") or 0.0) == 0.0:
            reasons.append("reward_all_zero_window")
        if float(reward_window.get("nonzero_range_rate") or 0.0) == 0.0:
            reasons.append("no_group_reward_variance_window")
        if float(reward_window.get("content_all_zero_rate") or 0.0) >= 0.8:
            reasons.append("content_score_all_zero_window")
    return {"ok": not reasons, "early_stop_recommended": bool(reasons), "reasons": reasons}
