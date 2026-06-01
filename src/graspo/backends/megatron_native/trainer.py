from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from graspo.backends.megatron_native.checkpoint import save_native_checkpoint
from graspo.backends.megatron_native.logger import NativeRolloutLogger
from graspo.backends.megatron_native.runtime import (
    MegatronNativeRuntime,
    MegatronNativeRuntimeProtocol,
    NativeGeneration,
    validate_native_runtime_config,
)
from graspo.core.advantage import group_advantages
from graspo.core.buffer import Experience, ReplayBuffer
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


class MegatronNativeGraspoTrainer:
    """Self-owned GRASPO loop backed by native Megatron tensor parallel runtime."""

    def __init__(
        self,
        config: GraspoConfig,
        selection: Any | None = None,
        runtime: MegatronNativeRuntimeProtocol | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.runtime = runtime or MegatronNativeRuntime.from_config(config)
        self.reward = GraspoReward(config.reward)
        self.replay_buffer = ReplayBuffer()
        self.stats = NativeTrainStats()
        self.backend_name = "megatron-native"
        self.global_step = 0
        self.sample_index = 0
        self.total_samples = 0
        self.started_at = time.monotonic()
        self.current_epoch_stats = NativeEpochStats()
        self.recent_groups: deque[dict[str, Any]] = deque(maxlen=50)
        self.pending_batch_attempts: list[dict[str, Any]] = []
        native = self.config.megatron_native
        self.logger = NativeRolloutLogger(
            self.config.training.output_dir,
            readable_enabled=native.readable_log_enabled,
            raw_enabled=native.raw_log_enabled,
        )

    def train(self) -> None:
        validate_native_runtime_config(self.config)
        self.runtime.validate()
        self.runtime.setup()
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "backend_selected",
                "backend": self.backend_name,
                "reason": self.selection.reason if self.selection is not None else "configured",
                "dependency_boundary": (
                    "Megatron Core/L.M. TP only; no NeMo/vLLM/Ray/DeepSpeed/FSDP/DDP/Accelerate"
                    if self.backend_name == "megatron-native"
                    else "single-process Hugging Face reference; not a production multi-card backend"
                ),
                "model_path": self.config.model.model_path,
                "train_path": self.config.data.train_path,
                "tensor_model_parallel_size": self.config.megatron_native.tensor_model_parallel_size,
            }
        )

        samples = load_jsonl(
            self.config.data.train_path,
            prompt_field=self.config.data.prompt_field,
            ground_truth_field=self.config.data.ground_truth_field,
            messages_field=self.config.data.messages_field,
        )
        self.total_samples = len(samples)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "run_start",
                "backend": self.backend_name,
                "samples_total": self.total_samples,
                "config": {
                    "rollout_group_size": self.config.training.rollout_group_size,
                    "optimize_completion_batch_size": self.config.training.optimize_completion_batch_size,
                    "optimize_times_per_step": self.config.training.optimize_times_per_step,
                    "replay_buffer_optimize_threshold": self.config.training.replay_buffer_optimize_threshold,
                    "rollout_max_retry_times": self.config.training.rollout_max_retry_times,
                    "training_epoch_count": self.config.training.training_epoch_count,
                    "max_steps": self.config.training.max_steps,
                    "max_new_tokens": self.config.training.max_new_tokens,
                    "save_steps": self.config.training.save_steps,
                    "activation_checkpointing_enabled": bool(self.config.model.gradient_checkpointing),
                    "lora_target_modules": list(self.config.lora.target_modules or ["q_proj", "v_proj"]),
                    "rollout_kv_cache_max_reserved_fraction": self.config.megatron_native.rollout_kv_cache_max_reserved_fraction,
                    "empty_cache_after_rollout_split": self.config.megatron_native.empty_cache_after_rollout_split,
                    "legacy_config_alias_used": bool(self.config.training.legacy_config_aliases),
                },
            }
        )

        try:
            for epoch in range(self.config.training.training_epoch_count):
                self.current_epoch_stats = NativeEpochStats(epoch=epoch)
                epoch_samples = list(samples)
                random.Random(int(self.config.training.seed) + epoch).shuffle(epoch_samples)
                for sample in epoch_samples:
                    self._sample_one(sample, epoch=epoch)
                    if self._maybe_optimize(epoch=epoch):
                        if 0 < self.config.training.max_steps <= self.global_step:
                            save_native_checkpoint(self.runtime, output_dir / "final")
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
            if len(self.replay_buffer) > 0:
                self._maybe_optimize(epoch=self.config.training.training_epoch_count - 1, force=True)
            save_native_checkpoint(self.runtime, output_dir / "final")
        finally:
            self.runtime.close()

    def _sample_one(self, sample: Sample, *, epoch: int) -> None:
        for retry_count in range(self.config.training.rollout_max_retry_times + 1):
            generation = self.runtime.generate_group(
                prompt=sample.prompt,
                rollout_group_size=self.config.training.rollout_group_size,
                max_new_tokens=self.config.training.max_new_tokens,
                max_prompt_length=self.config.data.max_prompt_length,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            results = [self.reward.score(text, sample.ground_truth) for text in generation.completions]
            rewards = [float(result.reward) for result in results]
            content_scores = [float(result.content_score) for result in results]
            all_right = [bool(result.all_right) for result in results]
            reward_details = [_reward_detail(result) for result in results]
            decision = classify_group(
                rewards,
                content_scores,
                retry_count=retry_count,
                rollout_max_retry_times=self.config.training.rollout_max_retry_times,
                perfect_skip_reward_threshold=self.config.training.perfect_skip_reward_threshold,
            )
            self.stats.total_groups += 1
            if decision.should_retry:
                self.stats.retries += 1

            readable = self._group_payload(
                sample=sample,
                epoch=epoch,
                generation=generation,
                rewards=rewards,
                content_scores=content_scores,
                all_right=all_right,
                reward_details=reward_details,
                decision=decision,
                retry_count=retry_count,
            )
            if self._is_primary():
                self.logger.write_readable(readable)
            self.recent_groups.append(_monitor_group(readable))
            self.pending_batch_attempts.append(readable)
            self._record_epoch_attempt(readable)

            if decision.should_retry:
                continue
            if not decision.should_train:
                if decision.decision.value == "perfect_skip":
                    self.stats.perfect_skipped += 1
                elif decision.decision.value == "invalid_no_preference_gap":
                    self.stats.invalid_no_preference_gap += 1
                else:
                    self.stats.invalid += 1
                if self._is_primary():
                    self.logger.write_raw({**readable, "raw": self._raw_generation(generation)})
                break
            if not has_reward_variance(rewards):
                self.stats.invalid += 1
                if self._is_primary():
                    self.logger.write_raw({**readable, "raw": self._raw_generation(generation)})
                break

            old_log_probs = self.runtime.sequence_log_probs(
                generation.sequences,
                generation.attention_mask,
            )
            advantages = _expand_advantages_like(rewards, old_log_probs)
            self._append_experiences(generation, rewards, old_log_probs, advantages)
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
            break
        self.current_epoch_stats.samples_seen += 1
        self.sample_index += 1

    def _maybe_optimize(self, *, epoch: int, force: bool = False) -> bool:
        threshold = self.config.training.replay_buffer_optimize_threshold
        if len(self.replay_buffer) < threshold and not force:
            return False
        if len(self.replay_buffer) == 0:
            return False

        usable = len(self.replay_buffer) if force else threshold
        data = self.replay_buffer.take(usable)
        metrics = self.runtime.train_batch(
            data,
            policy_ratio_clip_eps=self.config.training.policy_ratio_clip_eps,
            optimize_times_per_step=self.config.training.optimize_times_per_step,
            max_grad_norm=self.config.training.max_grad_norm,
        )
        attempts = list(self.pending_batch_attempts)
        reward_batch = _reward_batch_summary(
            attempts,
            rollout_group_size=self.config.training.rollout_group_size,
            optimize_completion_batch_size=self.config.training.optimize_completion_batch_size,
        )
        metrics["replay_buffer_optimize_threshold"] = threshold
        metrics["replay_buffer_trainable_completion_count"] = usable
        metrics["replay_buffer_trainable_group_count"] = usable / max(int(self.config.training.rollout_group_size), 1)
        metrics["optimize_completion_batch_size"] = self.config.training.optimize_completion_batch_size
        metrics["optimize_times_per_step"] = self.config.training.optimize_times_per_step
        metrics["force_flush"] = bool(force)
        self.replay_buffer.clear()
        self.pending_batch_attempts.clear()
        self.global_step += 1
        self.stats.optimized_steps += 1
        reward_window = _reward_window_summary(self.recent_groups)
        health = _training_health(metrics, reward_batch, reward_window)
        optimize = _compact_optimize_metrics(metrics)
        batch = _compact_batch_summary(reward_batch)
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
                    "attempts": attempts,
                }
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
                "health": health,
            }
        )
        if self.global_step % self.config.training.save_steps == 0:
            checkpoint_dir = Path(self.config.training.output_dir) / f"step_{self.global_step}"
            save_native_checkpoint(
                self.runtime,
                checkpoint_dir,
            )
            self._print_json(
                {
                    "timestamp": _timestamp(),
                    "event": "checkpoint_saved",
                    "step": self.global_step,
                    "path": str(checkpoint_dir),
                }
            )
        return True

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
        decision: Any,
        retry_count: int,
    ) -> dict[str, Any]:
        payload = {
            "event": "graspo_group",
            "backend": self.backend_name,
            "epoch": epoch,
            "step": self.global_step,
            "sample_index": self.sample_index,
            "prompt": sample.prompt,
            "ground_truth": sample.ground_truth,
            "metadata": sample.metadata,
            "completions": generation.completions,
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
            "generation_metadata": generation.metadata or {},
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
            self.current_epoch_stats.best_reward = max(self.current_epoch_stats.best_reward, max(rewards))
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
            "content_mean": stats.content_mean_sum / attempt_groups if stats.attempt_groups else 0.0,
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
            json.loads(str(extracted["answer"]).strip())
            valid_extracted_json = True
        except (TypeError, ValueError):
            valid_extracted_json = False
    return {
        "raw_score": float(result.raw_score),
        "max_score": float(result.max_score),
        "extracted": extracted,
        "useless_text_length": len(result.useless_text),
        "valid_extracted_json": valid_extracted_json,
    }


def _generated_token_counts(generation: NativeGeneration) -> list[int]:
    try:
        return [int(value) for value in generation.action_mask.detach().sum(dim=1).cpu().tolist()]
    except Exception:
        return []


def _monitor_group(payload: dict[str, Any]) -> dict[str, Any]:
    rewards = [float(value) for value in payload.get("rewards", [])]
    content_scores = [float(value) for value in payload.get("content_scores", [])]
    details = payload.get("reward_details", [])
    completions = payload.get("completions", [])
    return {
        "decision": payload.get("decision"),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_range": max(rewards) - min(rewards) if rewards else 0.0,
        "content_mean": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "content_all_zero": bool(content_scores) and all(value == 0.0 for value in content_scores),
        "content_all_one": bool(content_scores) and all(value == 1.0 for value in content_scores),
        "missing_json_marker_count": sum(1 for text in completions if "```json" not in text),
        "unclosed_json_fence_count": sum(1 for text in completions if "```json" in text and text.count("```") < 2),
        "invalid_extracted_json_count": sum(1 for detail in details if detail.get("valid_extracted_json") is False),
        "likely_truncated_json_count": sum(
            1
            for text, detail in zip(completions, details, strict=False)
            if _likely_truncated_json(text, detail)
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
        "invalid_extracted_json_count": sum(int(item["invalid_extracted_json_count"]) for item in items),
        "likely_truncated_json_count": sum(int(item["likely_truncated_json_count"]) for item in items),
    }


def _reward_batch_summary(
    attempts: list[dict[str, Any]],
    *,
    rollout_group_size: int,
    optimize_completion_batch_size: int,
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
        missing_json_marker_count += sum(1 for text in completions if "```json" not in text)
        unclosed_json_fence_count += sum(
            1 for text in completions if "```json" in text and text.count("```") < 2
        )
        invalid_extracted_json_count += sum(
            1 for detail in details if detail.get("valid_extracted_json") is False
        )
        likely_truncated_json_count += sum(
            1
            for text, detail in zip(completions, details, strict=False)
            if _likely_truncated_json(text, detail)
        )

    attempt_group_count = len(attempts)
    trainable_group_count = (
        decision_counts.get("trainable_max_correct", 0)
        + decision_counts.get("trainable_not_correct", 0)
    )
    retry_group_count = decision_counts.get("retry", 0)
    invalid_group_count = decision_counts.get("invalid", 0)
    invalid_no_preference_gap_group_count = decision_counts.get("invalid_no_preference_gap", 0)
    perfect_skip_group_count = decision_counts.get("perfect_skip", 0)
    return {
        "unit": "batch_attempt",
        "rollout_group_size": int(rollout_group_size),
        "optimize_completion_batch_size": int(optimize_completion_batch_size),
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
            sum(group_max_median_gaps) / len(group_max_median_gaps) if group_max_median_gaps else 0.0
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
        },
    }


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
    terminal_total = int(perfect_skip) + trainable_total + int(invalid) + int(invalid_no_preference_gap)
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


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _compact_optimize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    optimizer_steps_per_rank = int(metrics.get("optimizer_steps") or 0)
    global_optimizer_steps = int(metrics.get("global_optimizer_steps_sum") or optimizer_steps_per_rank)
    return {
        "optimized": bool(metrics.get("optimized")),
        "replay_buffer_trainable_completion_count": int(metrics.get("replay_buffer_trainable_completion_count") or 0),
        "replay_buffer_trainable_group_count": float(metrics.get("replay_buffer_trainable_group_count") or 0.0),
        "replay_buffer_optimize_threshold": int(metrics.get("replay_buffer_optimize_threshold") or 0),
        "optimize_completion_batch_size": int(metrics.get("optimize_completion_batch_size") or 0),
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
