"""Layer 2 — GraspoFlowTrainer: GRASPO 训练循环主类。

采用类改目录模式：统计、rollout、优化、checkpoint 分别驻留在独立文件中，
通过 mixin 组合到主类。外部使用者只 import 类名，完全不感知内部拆分。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from graspo.backends.graspoflow.logger import NativeRolloutLogger
from graspo.backends.graspoflow.runtime import (
    GraspoFlowRuntime,
    GraspoFlowRuntimeProtocol,
    validate_graspoflow_runtime_config,
)
from graspo.backends.graspoflow.trainer.checkpoint import CheckpointMixin
from graspo.backends.graspoflow.trainer.helpers import round_timing_details
from graspo.backends.graspoflow.trainer.optimize import OptimizeMixin
from graspo.backends.graspoflow.trainer.rollout import RolloutMixin
from graspo.backends.graspoflow.trainer.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
    _QueuedSample,
)
from graspo.core.buffer import ReplayBuffer
from graspo.core.data import load_jsonl
from graspo.core.logging import setup_logging
from graspo.core.reward import GraspoReward
from graspo.core.schema import GraspoConfig


class GraspoFlowTrainer(RolloutMixin, OptimizeMixin, CheckpointMixin):
    """GRASPO 训练循环，由 GraspoFlow 分布式运行时驱动。

    使用 mixin 组合：RolloutMixin（生成+评分）、OptimizeMixin（优化步骤）、
    CheckpointMixin（保存+恢复）。
    """

    def __init__(
        self,
        config: GraspoConfig,
        selection: Any | None = None,
        runtime: GraspoFlowRuntimeProtocol | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.runtime = runtime or GraspoFlowRuntime.from_config(config)
        self.reward = GraspoReward(config.reward)
        self.replay_buffer = ReplayBuffer()
        self.stats = GraspoFlowTrainStats()
        self.backend_name = "graspoflow"
        self.global_step = 0
        self.sample_index = 0
        self.total_samples = 0
        self.started_at = time.monotonic()
        self.current_epoch_stats = GraspoFlowEpochStats()
        self.recent_groups: deque[dict[str, Any]] = deque(maxlen=50)
        self.pending_batch_attempts: list[dict[str, Any]] = []
        self.pending_batch_timings: list[dict[str, Any]] = []
        self.resume_info: dict[str, Any] | None = None
        gf = self.config.graspoflow
        self.logger = NativeRolloutLogger(
            self.config.training.output_dir,
            readable_enabled=gf.readable_log_enabled,
            raw_enabled=gf.raw_log_enabled,
        )

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """GRASPO 训练主入口。"""
        validate_graspoflow_runtime_config(self.config)
        self.runtime.validate()
        self.runtime.setup()
        # 初始化标准 Python logging 通道（宪法 §13.2）
        rank = int(getattr(self.runtime, "rank", 0))
        setup_logging(self.config.training.output_dir, rank=rank)
        _log = logging.getLogger("graspo.trainer")
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
                "tp_size": self.config.graspoflow.tp_size,
            }
        )
        _log.info("Training started: backend=%s model=%s samples=%d",
                   self.backend_name, self.config.model.model_path, self.total_samples)

        samples = load_jsonl(self.config.data.train_path)
        self.total_samples = len(samples)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._resume_if_requested()
        # 将当前配置备份到输出目录，确保可完整复现
        if self.resume_info is None:
            _backup_config(self.config, output_dir)
        _log.info("Run config: rollout_group_size=%d optimize_prompt_batch_size=%d "
                  "training_epoch_count=%d max_new_tokens=%d",
                  self.config.training.rollout_group_size,
                  self.config.training.optimize_prompt_batch_size,
                  self.config.training.training_epoch_count,
                  self.config.training.max_new_tokens)
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
                    "forward_batch_size": self.config.graspoflow.forward_batch_size,
                    "empty_cache_after_rollout_split": self.config.graspoflow.empty_cache_after_rollout_split,
                    "synchronize_cuda_timing": self.config.graspoflow.synchronize_cuda_timing,
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
                    self.current_epoch_stats = GraspoFlowEpochStats(epoch=epoch)
                epoch_samples = list(samples)
                random.Random(int(self.config.training.seed) + epoch).shuffle(epoch_samples)
                resume_sample_offset = (
                    int(self.current_epoch_stats.samples_seen) if epoch == start_epoch else 0
                )
                pending_samples = epoch_samples[resume_sample_offset:]
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
                _log.info("Epoch %d/%d finished: elapsed=%.1fs samples_seen=%d",
                          epoch + 1, self.config.training.training_epoch_count,
                          time.monotonic() - self.started_at,
                          self.current_epoch_stats.samples_seen)
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

    # ── 样本队列调度 ──────────────────────────────────────────────────────────

    def _sample_one(self, sample: Any, *, epoch: int) -> bool:
        """处理单个样本（仅用于测试）。"""
        return self._sample_queue([sample], epoch=epoch)

    def _sample_queue(self, samples: list[Any], *, epoch: int) -> bool:
        """处理一批样本：rollout → 评分 → 重试 → 最终化。"""
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

    # ── 日志输出辅助 ──────────────────────────────────────────────────────────

    def _print_json(self, payload: dict[str, Any]) -> None:
        """主 rank 输出 JSON 日志。"""
        if self._is_primary():
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _is_primary(self) -> bool:
        """判断当前 rank 是否为主 rank（负责日志 I/O）。"""
        is_primary = getattr(self.runtime, "is_primary", None)
        if callable(is_primary):
            return bool(is_primary())
        return int(getattr(self.runtime, "rank", 0)) == 0

    def _timestamp(self) -> str:
        """返回当前时区的 ISO 格式时间戳。"""
        return _timestamp()

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
        """构建 timing 事件记录。"""
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
            "details": round_timing_details(details),
        }


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _backup_config(config: Any, output_dir: Path) -> None:
    """将当前配置写入输出目录，确保事后可完整复现。"""
    import yaml

    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(config.model_dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
