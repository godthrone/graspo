"""Layer 2 — GraspoFlowTrainer: GRASPO 训练循环主类。

采用类改目录模式：统计、rollout、优化、checkpoint 分别驻留在独立文件中，
通过 mixin 组合到主类。外部使用者只 import 类名，完全不感知内部拆分。
"""

import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from graspo.core.data import load_jsonl
from graspo.core.schema import GraspoConfig
from graspo.flow.logger.logger import NativeRolloutLogger
from graspo.flow.logging import setup_logging
from graspo.flow.runtime import (
    GraspoFlowRuntime,
    GraspoFlowRuntimeProtocol,
    validate_graspoflow_runtime_config,
)
from graspo.flow.trainer.checkpoint import CheckpointMixin
from graspo.flow.trainer.helpers import (
    _backup_config,
    _set_random_seed,
    _timestamp,
)
from graspo.flow.trainer.optimize import OptimizeMixin
from graspo.flow.trainer.rollout import RolloutMixin
from graspo.ripple.buffer import ReplayBuffer
from graspo.ripple.monitoring.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
    _QueuedSample,
)
from graspo.ripple.monitoring.summary import round_timing_details
from graspo.ripple.reward.reward import GraspoReward


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
        _set_random_seed(int(self.config.training.seed), rank=rank)
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
        _log.info(
            "Training started: backend=%s model=%s samples=%d",
            self.backend_name,
            self.config.model.model_path,
            self.total_samples,
        )

        samples = load_jsonl(self.config.data.train_path)
        self.total_samples = len(samples)
        # 多模态训练启动预检（防线 §2.3）：数据含图时验证视觉链路完整、
        # visual LoRA 可训练，失败即拒绝启动，避免 v13 式静默丢图空跑。
        self._preflight_multimodal(samples)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._resume_if_requested()
        # 将当前配置备份到输出目录，确保可完整复现
        if self.resume_info is None:
            _backup_config(self.config, output_dir)
        _log.info(
            "Run config: rollout_group_size=%d optimize_prompt_batch_size=%d "
            "max_epochs=%d max_new_tokens=%d",
            self.config.training.rollout_group_size,
            self.config.training.optimize_prompt_batch_size,
            self.config.training.max_epochs,
            self.config.training.max_new_tokens,
        )
        self._print_json(
            {
                "timestamp": _timestamp(),
                "event": "run_start",
                "backend": self.backend_name,
                "samples_total": self.total_samples,
                "resume": self.resume_info,
                "config": {
                    "rollout_group_size": self.config.training.rollout_group_size,
                    "rollout_queue_batch_size": self.config.training.rollout_queue_batch_size,
                    "optimize_prompt_batch_size": self.config.training.optimize_prompt_batch_size,
                    "optimize_iterations_per_step": 1,
                    "replay_buffer_optimize_threshold": (
                        self.config.training.replay_buffer_optimize_threshold
                    ),
                    "rollout_max_retries": self.config.training.rollout_max_retries,
                    "max_epochs": self.config.training.max_epochs,
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
                    "empty_cache_after_rollout_split": (
                        self.config.graspoflow.empty_cache_after_rollout_split
                    ),
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
                self.current_epoch_stats.epoch = start_epoch
                self.current_epoch_stats.samples_seen = 0
            for epoch in range(start_epoch, self.config.training.max_epochs):
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
                queue_size = max(1, int(self.config.training.rollout_queue_batch_size))
                for start in range(0, len(pending_samples), queue_size):
                    sample_queue = pending_samples[start : start + queue_size]
                    if self._sample_queue(sample_queue, epoch=epoch):
                        # max_steps 提前终止：先 flush replay buffer 再保存，
                        # 否则 _checkpoint_trainer_state 会因 buffer 非空而
                        # 拒绝保存（防线 §2.3 生效，但提前终止路径漏了 flush）。
                        if len(self.replay_buffer) > 0:
                            self._maybe_optimize(epoch=epoch, force=True)
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
                _log.info(
                    "Epoch %d/%d finished: elapsed=%.1fs samples_seen=%d",
                    epoch + 1,
                    self.config.training.max_epochs,
                    time.monotonic() - self.started_at,
                    self.current_epoch_stats.samples_seen,
                )
                if self.config.training.save_checkpoint_every_epoch:
                    if len(self.replay_buffer) > 0:
                        self._maybe_optimize(epoch=epoch, force=True)
                    self._save_checkpoint(output_dir / f"epoch_{epoch}", epoch=epoch)
            if len(self.replay_buffer) > 0:
                self._maybe_optimize(epoch=self.config.training.max_epochs - 1, force=True)
            self._save_checkpoint(output_dir / "final", epoch=self.config.training.max_epochs - 1)
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
        max_attempts = self.config.training.rollout_max_retries + 1
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

    def _preflight_multimodal(self, samples: list[Any]) -> None:
        """多模态训练启动预检（防线 §2.3）。

        数据含图时验证 encode → attach → resolve 链路完整、visual LoRA
        可训练（fake 1-step 前向梯度非零），失败即拒绝启动训练——
        避免 v13 式的静默丢图空跑 19.5 小时。纯文本数据直接跳过。
        """
        from graspo.flow.trainer.preflight import (
            run_multimodal_preflight,
        )

        model = getattr(self.runtime, "_require_adapter", lambda: None)()
        model_config = getattr(getattr(model, "model", None), "config", None)
        image_token_id = getattr(model_config, "image_token_id", None)
        if image_token_id is not None and self._is_primary():
            _log = logging.getLogger("graspo.trainer")
            _log.info(
                "multimodal training detected (image_token_id=%s); running "
                "visual-link preflight before training starts",
                image_token_id,
            )
        run_multimodal_preflight(
            self.runtime,
            samples,
            data_dir=str(Path(self.config.data.train_path).parent),
            image_token_id=image_token_id,
            model_name=str(self.config.model.model_path),
        )

    def _print_json(self, payload: dict[str, Any]) -> None:
        """主 rank 通过 logging 输出结构化 JSON 日志。"""
        if self._is_primary():
            logging.getLogger("graspo.trainer").info(json.dumps(payload, ensure_ascii=False))

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
