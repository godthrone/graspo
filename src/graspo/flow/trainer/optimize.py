"""GraspoFlowTrainer 优化步骤的 mixin。"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from graspo.ripple.monitoring.summary import (
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
    _last_checkpoint_time: float  # wall-clock monotonic seconds for time-based save

    def _maybe_optimize(self, *, epoch: int, force: bool = False) -> bool:
        """当 replay buffer 达到阈值时触发优化步骤。

        集结算子说明（RL-1 cross-PG deadlock 修复）：是否进入训练由所有 rank 共同决定，
        而不是单个 dp_rank 各自判断。若任意 rank 需要训练，则全部 rank（包括本地缓冲区
        为空、无 trainable experience 的 dp_rank）都进入 ``train_batch``，以参与
        dp_group 上 ``_shared_training_indices`` 的 ``all_reduce`` 和梯度同步等跨 rank
        集结算子。否则 dp_rank=0 卡在 dp_group 的 all_reduce，而 dp_rank=1 在 teardown
        的 WORLD barrier 上等待 dp_rank=0（两个不同进程组互等），即 RL-1 死锁。
        """
        threshold = self.config.training.replay_buffer_optimize_threshold
        # 本地是否具备训练条件（保留原语义：不达阈值不训练，force 强制刷新）。
        local_wants_train = (force and len(self.replay_buffer) > 0) or (
            not force and len(self.replay_buffer) >= threshold
        )
        # 集结算子（WORLD，group=None）：所有 rank 同时决定是否进入训练集体。
        if dist.is_available() and dist.is_initialized():
            want = torch.tensor(
                [1 if local_wants_train else 0],
                dtype=torch.int,
                device=self.runtime._require_adapter().device,
            )
            dist.all_reduce(want, op=dist.ReduceOp.MAX)
            global_wants_train = bool(want.item())
        else:
            global_wants_train = local_wants_train
        if not global_wants_train:
            return False

        if len(self.replay_buffer) == 0:
            # 空缓冲区 rank 仍进入训练集体：以空数据参与 dp_group 集结算子，
            # 贡献零梯度，但保证跨 rank 通信不被撕裂（否则 MIN 归零，本 rank 无训练）。
            usable = 0
            data: list[Any] = []
        else:
            usable = (
                len(self.replay_buffer) if force else min(threshold, len(self.replay_buffer))
            )
            data = self.replay_buffer.take(usable)
        optimize_started_at = time.monotonic()
        metrics = self.runtime.train_batch(
            data,
            policy_ratio_clip_eps=self.config.training.policy_ratio_clip_eps,
            max_grad_norm=self.config.training.max_grad_norm,
        )
        optimize_sec = time.monotonic() - optimize_started_at
        attempts = list(self.pending_batch_attempts)
        timings = list(self.pending_batch_timings)
        reward_batch = reward_batch_summary(
            attempts,
            rollout_group_size=self.config.training.rollout_group_size,
            effective_batch_size=(
                int(self.config.native.micro_batch_size)
                * int(self.config.training.gradient_accumulation_micro_batches)
            ),
        )
        metrics["replay_buffer_optimize_threshold"] = threshold
        metrics["replay_buffer_trainable_completion_count"] = usable
        metrics["replay_buffer_trainable_group_count"] = usable / max(
            int(self.config.training.rollout_group_size), 1
        )
        metrics["effective_batch_size"] = (
            int(self.config.native.micro_batch_size)
            * int(self.config.training.gradient_accumulation_micro_batches)
        )
        metrics["optimize_iterations_per_step"] = 1
        metrics["force_flush"] = bool(force)
        self.replay_buffer.clear()
        self.pending_batch_attempts.clear()
        self.pending_batch_timings.clear()
        self.global_step += 1
        self.stats.optimized_steps += 1
        checkpoint_sec = 0.0
        checkpoint_dir = None
        save_period_min = int(self.config.training.save_checkpoint_time_period_minutes)
        if (
            self.config.training.save_steps > 0
            and self.global_step % self.config.training.save_steps == 0
        ):
            checkpoint_dir = Path(self.config.training.output_dir) / f"step_{self.global_step}"
            checkpoint_started_at = time.monotonic()
            self._save_checkpoint(checkpoint_dir, epoch=epoch)
            checkpoint_sec = time.monotonic() - checkpoint_started_at
            self._last_checkpoint_time = time.monotonic()
        elif save_period_min > 0:
            elapsed = time.monotonic() - self._last_checkpoint_time
            if elapsed >= save_period_min * 60:
                checkpoint_dir = (
                    Path(self.config.training.output_dir) / f"time_{self._timestamp()}"
                )
                checkpoint_started_at = time.monotonic()
                self._save_checkpoint(checkpoint_dir, epoch=epoch)
                checkpoint_sec = time.monotonic() - checkpoint_started_at
                self._last_checkpoint_time = time.monotonic()
                logging.getLogger("graspo.trainer").info(
                    "RL: time-based checkpoint saved at epoch=%d step=%d "
                    "(period=%d min, elapsed=%.1f min)",
                    epoch, self.global_step, save_period_min, elapsed / 60.0,
                )
        reward_window = reward_window_summary(self.recent_groups)
        health = training_health(metrics, reward_batch, reward_window)
        if not health["ok"]:
            logging.getLogger("graspo.trainer").warning(
                "training health degraded: %s", ", ".join(health["reasons"])
            )
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
                "epoch": epoch,
                "elapsed_sec": round(time.monotonic() - self.started_at, 3),
                # 口径自证（v20 教训）：run_cumulative 是 run 全程累计（永不清零），
                # epoch_cumulative 是本 epoch 累计，batch 是本 step 单步。
                "run_cumulative": self._run_summary(),
                "epoch_cumulative": self._epoch_summary(),
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
                    # §10.1：日志不含宿主机路径（config 备份为真相源），只留目录名
                    "path": checkpoint_dir.name,
                    "checkpoint_save_sec": round(checkpoint_sec, 6),
                }
            )
        return True

    def _run_summary(self) -> dict[str, Any]:
        """生成全局运行摘要（run 全程累计口径）。"""
        from graspo.ripple.monitoring.summary import compact_decisions

        return {
            "step": self.global_step,
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
        from graspo.ripple.monitoring.summary import compact_decisions

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
            "base_content_mean": stats.base_content_mean_sum / attempt_groups
            if stats.attempt_groups
            else 0.0,
            "best_reward": stats.best_reward,
        }
