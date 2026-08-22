"""Layer 2 — GraspoFlowTrainer：GRASPO 训练循环主类。

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

from graspo.core.schema import GraspoConfig
from graspo.flow.logger.native_rollout_logger import NativeRolloutLogger
from graspo.flow.logging import setup_logging
from graspo.flow.runtime import (
    GraspoFlowRuntime,
    GraspoFlowRuntimeBase,
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
from graspo.ripple.data import load_jsonl
from graspo.ripple.monitoring.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
    QueuedSample,
)
from graspo.ripple.monitoring.summary import round_timing_details
from graspo.ripple.reward.reward import create_reward


class GraspoFlowTrainer(RolloutMixin, OptimizeMixin, CheckpointMixin):
    """GRASPO 训练循环，由 GraspoFlow 分布式运行时驱动。

    使用 mixin 组合：RolloutMixin（生成+评分）、OptimizeMixin（优化步骤）、
    CheckpointMixin（保存+恢复）。
    """

    def __init__(
        self,
        config: GraspoConfig,
        selection: Any | None = None,
        runtime: GraspoFlowRuntimeBase | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.runtime = runtime or GraspoFlowRuntime.from_config(config)
        self.reward = create_reward(config.reward)
        self.replay_buffer = ReplayBuffer()
        self._smoke_boundary = False  # 冒烟运行边界：首轮 optimize 后停止（见 train(smoke=)）
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
        self._last_checkpoint_time = 0.0  # 墙钟时间周期保存用（monotonic 秒）
        gf = self.config.graspoflow
        self.logger = NativeRolloutLogger(
            self.config.training.output_dir,
            readable_enabled=gf.readable_log_enabled,
            raw_enabled=gf.raw_log_enabled,
        )

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    @property
    def adapter(self) -> Any:
        """当前运行时已 setup 的模型适配器（DP 分片等逻辑访问）。

        与 SFT 路径（``self.runtime._adapter``）一致；此属性让 mixin 无需各自
        探测 runtime 的私有字段。
        """
        return self.runtime._require_adapter()

    def train(self, *, smoke: bool = False) -> None:
        """GRASPO 训练主入口。

        :param smoke: 冒烟运行边界（运行边界参数，不改变训练语义）——
            运行到首轮 optimize 后停止，**不修改任何配置值**。
        """
        self._smoke_boundary = bool(smoke)
        validate_graspoflow_runtime_config(self.config)
        self.runtime.validate()
        self.runtime.setup()
        # 初始化标准 Python logging 通道
        rank = self.runtime.rank
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
                # §10.1：日志不含宿主机路径（config.yaml 备份为唯一真相源），只留文件名
                "model_path": Path(self.config.model.model_path).name,
                "train_path": Path(self.config.data.train_path).name,
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
        # 多模态训练启动预检（防线）：数据含图时验证视觉链路完整、
        # visual LoRA 可训练，失败即拒绝启动，避免 v13 式静默丢图空跑。
        self._preflight_multimodal(samples)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._resume_if_requested()
        # 初始化墙钟时间周期 checkpoint 计时器
        self._last_checkpoint_time = time.monotonic()
        save_period_min = int(self.config.training.save_checkpoint_time_period_minutes)
        if save_period_min > 0:
            _log.info(
                "RL: time-based checkpoint save enabled (every %d min)",
                save_period_min,
            )
        # 将当前配置备份到输出目录，确保可完整复现。resume 时也刷新——
        # 否则输出目录内的 config.yaml 停留在旧配置，误导复现（§1.4 单一真相源）
        _backup_config(self.config, output_dir)
        _log.info(
            "Run config: rollout_group_size=%d micro_batch_size=%d "
            "gradient_accumulation_micro_batches=%d max_epochs=%d max_new_tokens=%d",
            self.config.training.rollout_group_size,
            self.config.graspoflow.micro_batch_size,
            self.config.training.gradient_accumulation_micro_batches,
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
                # §1.4 单一真相源：config 全量以落盘备份 config.yaml 为准（§8.5），
                # 事件只引用文件名，不再内嵌人工子集（防子集漂移）
                "config_backup": "config.yaml",
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
                # DP: 每个 DP rank 处理不同的数据分片
                if self.adapter is not None and self.adapter.dp_size > 1:
                    epoch_samples = epoch_samples[self.adapter.dp_rank :: self.adapter.dp_size]
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
                        # smoke 提前终止：先 flush replay buffer 再保存，
                        # 否则 _checkpoint_trainer_state 会因 buffer 非空而
                        # 拒绝保存（防线生效，但提前终止路径漏了 flush）。
                        if len(self.replay_buffer) > 0:
                            self._maybe_optimize(epoch=epoch, force=True)
                        self._save_checkpoint(output_dir / "final", epoch=epoch)
                        return
                self._print_json(
                    {
                        "timestamp": _timestamp(),
                        "event": "epoch_summary",
                        "epoch": self.current_epoch_stats.epoch,
                        "elapsed_sec": round(time.monotonic() - self.started_at, 3),
                        "epoch_cumulative": self._epoch_summary(),
                        "run_cumulative": self._run_summary(),
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
                    self._last_checkpoint_time = time.monotonic()
            if len(self.replay_buffer) > 0:
                self._maybe_optimize(epoch=self.config.training.max_epochs - 1, force=True)
            self._save_checkpoint(output_dir / "final", epoch=self.config.training.max_epochs - 1)
            self._last_checkpoint_time = time.monotonic()
        finally:
            self.runtime.close()

    # ── 样本队列调度 ──────────────────────────────────────────────────────────

    def _sample_one(self, sample: Any, *, epoch: int) -> bool:
        """处理单个样本（仅用于测试）。"""
        return self._sample_queue([sample], epoch=epoch)

    def _sample_queue(self, samples: list[Any], *, epoch: int) -> bool:
        """处理一批样本：rollout → 评分 → 重试 → 最终化。"""
        active = [QueuedSample(sample=sample) for sample in samples]
        finished: list[QueuedSample] = []
        max_attempts = self.config.training.rollout_max_retries + 1
        while active:
            attempt_records = self._rollout_queue_attempt(active, epoch=epoch)
            next_active: list[QueuedSample] = []
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
        """多模态训练启动预检（防线）。

        数据含图时验证 encode → attach → resolve 链路完整、visual LoRA
        可训练（fake 1-step 前向梯度非零），失败即拒绝启动训练——
        避免 v13 式的静默丢图空跑 19.5 小时。纯文本数据直接跳过。
        """
        from graspo.flow.trainer.preflight import (
            run_multimodal_preflight,
        )

        model = self.runtime._require_adapter()
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
            pp_size=int(self.config.graspoflow.pp_size),
        )

    def _print_json(self, payload: dict[str, Any]) -> None:
        """主 rank 输出结构化 JSON 事件（§13.3 易读/原始分离）。

        落盘到 ``logs/events.jsonl``（机器分析通道），同时打印到 stdout
        （部署端 nohup.out 兼容）；training.log 只保留人类可读文本。
        """
        if self._is_primary():
            self.logger.write_event(payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _is_primary(self) -> bool:
        """判断当前 rank 是否为主 rank（负责日志 I/O）。"""
        return self.runtime.is_primary()

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
        attempt_number: int | None = None,
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
            "attempt_number": attempt_number,
            "retry_count": retry_count,
            "rank": self.runtime.rank,
            "tp_rank": self.runtime.tp_rank,
            "details": round_timing_details(details),
        }
