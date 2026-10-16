"""GraspoFlowTrainer 优化步骤的 mixin（★ P4：R5 已退役，步边界改用 β1b 整组配额）。

**R5 退役登记（D3 裁定 + 指挥部 2026-10-11 P4 口径）**

旧实现（R5，2026-10-08 在役本地补丁）在设施层做了三件**改变"喂给 Ripple 的输入"**的事：
① `_align_dp_usable`：把本 rank 的待训条数对齐到 dp 组**裸 MIN**（逐 rank 同值）；
② 对齐塌缩为 0 时 **defer**（本步不训、缓冲区不清）；
③ 终端 force 步的**清余数**（`_flush_terminal_force_remainder`）。
这三件合起来 **改了每步样本集合与步数** ⇒ 违反设施层契约（可重排，但不得改喂给
Ripple 的样本集合/步数）⇒ **退役**（`baseline-by-layer.md` §6 D3 裁定）。

**替代方案 = β1b 整组均衡分配**（`flow/backend/native/assignment.py`）：

- ★ **组不拆**：步边界按**整组**取数（`usable = (raw // G) * G`）⇒ Ripple 只见完整组
  （宪法 §1.1；I1 组完整性）；
- ★ **按工作量分组**：分片在 `trainer.py` 由 `assignment.assign_whole_groups` 完成
  （各分片组数差 ≤ 1，`assert_group_partition` 可机核）；
- ★ **不新增会合**：本文件**不再**引入任何跨 rank 归约（R5 的 `all_reduce(MIN)` 已删，
  既有 WORLD `all_reduce(MAX)` 进入门保留不变）。

⚠ **本阶段唯一预期行为变化即在此处**：**每步样本集合与步数改变**（见 `phase-4-report.md`）。
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from graspo.flow.backend.native.assignment import whole_group_count
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

    def _ucd_threshold(self) -> int:
        """★ replay 触发阈值（**切片口径** = ``Q × (G//m)``；dp≤1 与既有属性同值）。

        单一真相源 = ``core.schema.TrainingConfig.replay_buffer_optimize_threshold_for``。
        ⛔ 本方法**不复制**该算式：它只是把"配置对象可能没有该方法"的两种非生产形态
        （单测里的 ``SimpleNamespace`` / 旧配置对象）回落成**语义等价**的取值。
        """
        training = self.config.training
        compute = getattr(training, "replay_buffer_optimize_threshold_for", None)
        if callable(compute):
            return int(compute(self._ucd_dp_size()))
        # 回落：既有的 `Q × G` 属性（dp≤1 语义），再按量子折算（等价变换）。
        base = int(training.replay_buffer_optimize_threshold)
        group_size = max(1, int(training.rollout_group_size))
        quantum = self._ucd_quantum()
        if quantum == group_size or group_size % max(quantum, 1) != 0:
            return base
        return base * quantum // group_size

    def _ucd_quantum(self) -> int:
        """★ 步边界**量子**（= 切片后每条 prompt 对 replay buffer 的贡献条数）。

        ``rollout 分卡`` 下 = ``G // dp_size``；``dp_size <= 1`` ⇒ ``= G``（**恒等**，
        逐位不变）。取值优先用 epoch 循环落下的 ``_ucd_rollouts_per_rank``
        （与**生成入参 / 读数除数**同源）；未进 ``train()``（单测 / 诊断）⇒ 现算一次。

        ⛔ **不得**在这里重写 ``G // dp_size`` 算式：单一真相源是
        ``core.schema.TrainingConfig.rollout_slice_quantum``（§1.4）。
        """
        preset = getattr(self, "_ucd_rollouts_per_rank", None)
        if preset is not None:
            return max(1, int(preset))
        # 回落：配置对象可能没有该方法（单测的 `SimpleNamespace` / 旧配置对象）
        # ⇒ 用 `G // m` 的**同一算式**（仍由 §1.4 的 `rollout_slice_quantum` 定义）。
        compute = getattr(self.config.training, "rollout_slice_quantum", None)
        dp_size = self._ucd_dp_size()
        if callable(compute):
            return max(1, int(compute(dp_size)))
        return max(1, int(self.config.training.rollout_group_size) // dp_size)

    def _ucd_dp_size(self) -> int:
        """切片口径下的 ``m``（= 实际 ``dp_size``）；取不到或 ≤1 ⇒ 1（**切片关闭**）。"""
        degree = getattr(self, "parallel_degree", None)
        return max(1, int(getattr(degree, "dp_size", 1) or 1))

    def _maybe_optimize(self, *, epoch: int, force: bool = False) -> bool:
        """当 replay buffer 达到阈值时触发优化步骤。

        集结算子说明（RL-1 cross-PG deadlock 修复）：是否进入训练由所有 rank 共同决定，
        而不是单个 dp_rank 各自判断。若任意 rank 需要训练，则全部 rank（包括本地缓冲区
        为空、无 trainable experience 的 dp_rank）都进入 ``train_batch``，以参与
        dp_group 上 ``_shared_training_indices`` 的 ``all_reduce`` 和梯度同步等跨 rank
        集结算子。否则 dp_rank=0 卡在 dp_group 的 all_reduce，而 dp_rank=1 在 teardown
        的 WORLD barrier 上等待 dp_rank=0（两个不同进程组互等），即 RL-1 死锁。
        """
        # ★ 阈值取**切片口径**（= 配置层单一真相源；dp≤1 与既有属性**同值**）。
        threshold = self._ucd_threshold()
        # ★ **两个量分开命名**（W-1「一名两义」的落点，两处都在函数**入口**求值，
        #   避免"只有某些分支才定义"的 NameError 与口径漂移）：
        #   - `group_size` = **算法组大小 G**（语义面：`reward_batch_summary` 的"每组 G 条"）；
        #   - `rollouts_per_rank` = **步边界量子**（切片后 = `G//m`；dp≤1 ⇒ = G）。
        group_size = max(1, int(self.config.training.rollout_group_size))
        rollouts_per_rank = self._ucd_quantum()
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
            # ★ 量子已在函数入口求值（见上）；此处只做**取整**，不再重算。
            raw_usable = (
                len(self.replay_buffer) if force else min(threshold, len(self.replay_buffer))
            )
            # ★ β1b（P4）：按**整组**配额取数 —— 组不拆，Ripple 只见完整组。
            #   R5 的"对齐到 dp 组裸 MIN"已删：跨 rank 均衡改由分片保证
            #   （`assignment.assign_whole_groups`，各分片组数差 ≤1），此处不再引入归约。
            #   ★ 取整原语是单一真相源（可在 CPU 直接单测）：assignment.whole_group_count。
            #   ★ U-C：量子由 `G` 改为 `rollouts_per_rank`（切片下每条 prompt 只贡献 G/m 条）。
            #     正向副产品：新量子下每个条目恰好贡献 0 或 `G/m` 条 ⇒ buffer 长度永远是该
            #     量子的整数倍 ⇒ **取整变成恒等、不再丢尾**（前提：离散决策逐卡同判 + 无本地
            #     drop + `ReplayBuffer.limit == 0`，启动期已 fail-closed 校验）。
            usable = whole_group_count(raw_usable, rollouts_per_rank)
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
        # ★ M5 / 复核者 B-2：**除数必须随量子改**（切片后 = `G//m`）。
        #   不改 ⇒「可训组数」静默缩到 1/m（可能为 0），污染健康判据与矩阵对账。
        metrics["replay_buffer_trainable_group_count"] = usable // max(rollouts_per_rank, 1)
        metrics["effective_batch_size"] = int(self.config.native.micro_batch_size) * int(
            self.config.training.gradient_accumulation_micro_batches
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
                checkpoint_dir = Path(self.config.training.output_dir) / f"time_{self._timestamp()}"
                checkpoint_started_at = time.monotonic()
                self._save_checkpoint(checkpoint_dir, epoch=epoch)
                checkpoint_sec = time.monotonic() - checkpoint_started_at
                self._last_checkpoint_time = time.monotonic()
                logging.getLogger("graspo.trainer").info(
                    "RL: time-based checkpoint saved at epoch=%d step=%d "
                    "(period=%d min, elapsed=%.1f min)",
                    epoch,
                    self.global_step,
                    save_period_min,
                    elapsed / 60.0,
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
