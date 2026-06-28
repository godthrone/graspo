"""GraspoFlowTrainer 优化步骤的 mixin。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from graspo.backends.graspoflow.trainer.helpers import (
    compact_batch_summary,
    compact_optimize_metrics,
    compact_timing_summary,
    reward_batch_summary,
    reward_window_summary,
    training_health,
)


class OptimizeMixin:
    """优化步骤、checkpoint 保存、统计摘要的 mixin。"""

    config: Any
    runtime: Any
    replay_buffer: Any
    stats: Any
    current_epoch_stats: Any
    recent_groups: Any
    pending_batch_attempts: Any
    pending_batch_timings: Any
    logger: Any
    global_step: int
    backend_name: str

    def _maybe_optimize(self, *, epoch: int, force: bool = False) -> bool:
        """当 replay buffer 达到阈值时触发优化步骤。"""
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
        reward_batch = reward_batch_summary(
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
        checkpoint_dir = None
        if (
            self.config.training.save_steps > 0
            and self.global_step % self.config.training.save_steps == 0
        ):
            checkpoint_dir = Path(self.config.training.output_dir) / f"step_{self.global_step}"
            checkpoint_started_at = time.monotonic()
            self._save_checkpoint(checkpoint_dir, epoch=epoch)
            checkpoint_sec = time.monotonic() - checkpoint_started_at
        reward_window = reward_window_summary(self.recent_groups)
        health = training_health(metrics, reward_batch, reward_window)
        optimize = compact_optimize_metrics(metrics)
        batch = compact_batch_summary(reward_batch)
        timing = compact_timing_summary(
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
                    "timestamp": self._timestamp(),
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
                "timestamp": self._timestamp(),
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
                    "timestamp": self._timestamp(),
                    "event": "checkpoint_saved",
                    "step": self.global_step,
                    "path": str(checkpoint_dir),
                    "checkpoint_save_sec": round(checkpoint_sec, 6),
                }
            )
        return True

    def _run_summary(self) -> dict[str, Any]:
        """生成全局运行摘要。"""
        from graspo.backends.graspoflow.trainer.helpers import compact_decisions

        return {
            "attempt_groups": self.stats.total_groups,
            "completions": self.stats.total_groups * int(self.config.training.rollout_group_size),
            "decisions": compact_decisions(
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
        """生成当前 epoch 摘要。"""
        from graspo.backends.graspoflow.trainer.helpers import compact_decisions

        stats = self.current_epoch_stats
        attempt_groups = max(stats.attempt_groups, 1)
        return {
            "epoch": stats.epoch,
            "samples_seen": stats.samples_seen,
            "samples_total": self.total_samples,
            "progress": stats.samples_seen / self.total_samples if self.total_samples else 0.0,
            "attempt_groups": stats.attempt_groups,
            "completions": stats.completion_count,
            "decisions": compact_decisions(
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
