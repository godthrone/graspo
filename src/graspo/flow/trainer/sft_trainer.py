"""SFT 训练循环，复用 GraspoFlow 基础设施（TP/PP、模型加载、LoRA、checkpoint）。

不依赖任何 RL 模块（reward/advantage/buffer/rollout）。
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from graspo.core.schema import GraspoConfig
from graspo.flow.logging import setup_logging
from graspo.flow.runtime import (
    GraspoFlowRuntime,
    GraspoFlowRuntimeBase,
    validate_native_runtime_config,
)
from graspo.flow.trainer.helpers import _backup_config, _set_random_seed, _timestamp
from graspo.flow.data_io import load_jsonl
from graspo.ripple.data import SFTTokenized, sft_tokenize


class SFTTrainer:
    """SFT (Supervised Fine-Tuning) 训练器。

    复用 :class:`GraspoFlowRuntime` 的全部基础设施：
    - 分布式初始化（TP/PP）
    - 模型加载 + LoRA 注入
    - Optimizer + LR scheduler
    - Checkpoint 存取
    - 多模态编码

    只新增 SFT 特有的训练循环：tokenize → forward → cross-entropy loss → backward。
    """

    def __init__(
        self,
        config: GraspoConfig,
        runtime: GraspoFlowRuntimeBase | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or GraspoFlowRuntime.from_config(config)
        self.started_at = time.monotonic()
        self.global_step = 0
        self.total_samples = 0
        self._resume_epoch = 0
        self._resume_batch_idx = 0

    def train(self, *, smoke: bool = False) -> None:
        """SFT 训练主入口。
        """
        validate_native_runtime_config(self.config)
        from graspo.flow.lora.lora_io import prepare_output_dir
        output_dir = prepare_output_dir(
            self.config.training.output_dir,
            overwrite=self.config.training.overwrite_output_dir,
        )
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.runtime.validate()
        self.runtime.setup()
        rank = self.runtime.rank
        setup_logging(self.config.training.output_dir, rank=rank)
        _set_random_seed(int(self.config.training.seed), rank=rank)
        _log = logging.getLogger("graspo.sft_trainer")

        # 加载并 tokenize 数据（所有 rank 各自执行，因为 train_batch_sft 在所有 rank 上调用）
        samples = load_jsonl(self.config.data.train_path)
        self.total_samples = len(samples)
        # DP: 每个 DP rank 处理不同的数据分片
        adapter = self.runtime._adapter
        if adapter is not None and adapter.dp_size > 1:
            samples = samples[adapter.dp_rank :: adapter.dp_size]
            self.total_samples = len(samples)
        _log.info("SFT: loaded %d samples from %s", self.total_samples, self.config.data.train_path)

        if self._is_primary():
            _backup_config(self.config, output_dir)

        _log.info("SFT: tokenizing %d samples...", self.total_samples)
        data_dir = str(Path(self.config.data.train_path).parent)
        adapter = self.runtime._adapter
        if adapter is None:
            raise RuntimeError("runtime adapter not loaded")
        processor = getattr(adapter, "processor", None)
        tokenized: list[SFTTokenized] = [
            sft_tokenize(
                s,
                adapter.tokenizer,
                max_seq_length=self.config.data.max_prompt_length,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
                data_dir=data_dir,
                processor=processor,
            )
            for s in samples
        ]
        # 纯文本样本：直接统计；多模态样本：延迟编码，token 数在 collate 阶段确定
        text_tokenized = [t for t in tokenized if t.deferred_multimodal is None]
        multimodal_count = len(tokenized) - len(text_tokenized)
        if text_tokenized:
            _log.info(
                "SFT: tokenization complete, text=%d (avg tokens=%.0f), multimodal=%d (deferred)",
                len(text_tokenized),
                sum(len(t.input_ids) for t in text_tokenized if t.input_ids is not None)
                / max(len(text_tokenized), 1),
                multimodal_count,
            )
        else:
            _log.info(
                "SFT: tokenization complete, all %d samples multimodal (deferred encoding)",
                multimodal_count,
            )

        # 有效 batch = micro_batch_size × gradient_accumulation_micro_batches
        # SFT 训练器按此大小创建批次，train_batch_sft 内部拆分为 micro-batch
        mb = max(1, int(self.config.native.micro_batch_size))
        ga = max(1, int(self.config.training.gradient_accumulation_micro_batches))
        effective_batch_size = mb * ga
        max_grad_norm = float(self.config.training.max_grad_norm)
        save_steps = int(self.config.training.save_steps)
        save_period_min = int(self.config.training.save_checkpoint_time_period_minutes)

        _log.info(
            "SFT config: micro_batch_size=%d gradient_accumulation_micro_batches=%d "
            "effective_batch=%d max_epochs=%d lr=%.1e max_seq_len=%d",
            mb, ga, effective_batch_size,
            self.config.training.max_epochs,
            self.config.training.effective_learning_rate(
                dp_size=self.config.native.dp_size
            ),
            self.config.data.max_prompt_length,
        )

        # Resume: 恢复 LoRA 权重、优化器、调度器、RNG、数据位置
        self._resume_if_requested()

        if save_period_min > 0:
            _log.info(
                "SFT: time-based checkpoint save enabled (every %d min)",
                save_period_min,
            )

        try:
            _last_checkpoint_time = time.monotonic()
            for epoch in range(self._resume_epoch, self.config.training.max_epochs):
                random.Random(int(self.config.training.seed) + epoch).shuffle(tokenized)
                batches = [
                    tokenized[start : start + effective_batch_size]
                    for start in range(0, len(tokenized), effective_batch_size)
                ]
                # SFT-2: drop last incomplete batch to prevent DP deadlock.
                # When DP distributes samples unevenly across ranks, the last
                # batch may differ in size.  Dropping it ensures every rank
                # processes the same number of equally-sized batches so that
                # collective ops (all-reduce) never deadlock on a rank that
                # exited its loop early.
                if batches and len(batches[-1]) < effective_batch_size:
                    batches = batches[:-1]
                # SFT-2 (additional): cross-DP-rank sync of min batch count.
                # When effective_batch_size=1, every batch is "complete" (size=1)
                # so the drop-last above never triggers.  But DP ranks can still
                # get different numbers of samples, leading to different batch
                # counts.  all_reduce(MIN) finds the globally smallest count and
                # truncates every rank to it, eliminating the deadlock root cause.
                if adapter.dp_size > 1:
                    import torch
                    import torch.distributed as dist
                    tp_state = adapter.tp_state
                    if tp_state is not None and tp_state.dp_group is not None:
                        n_batches = torch.tensor([len(batches)], device=tp_state.device)
                        dist.all_reduce(n_batches, op=dist.ReduceOp.MIN, group=tp_state.dp_group)
                        batches = batches[: int(n_batches.item())]

                _log.info(
                    "SFT epoch %d/%d: %d batches",
                    epoch + 1,
                    self.config.training.max_epochs,
                    len(batches),
                )

                start_batch = self._resume_batch_idx if epoch == self._resume_epoch else 0
                for batch_idx in range(start_batch, len(batches)):
                    batch = batches[batch_idx]
                    batch_started_at = time.monotonic()
                    metrics = self.runtime.train_batch_sft(
                        batch,
                        max_grad_norm=max_grad_norm,
                    )
                    self.global_step += 1
                    if smoke:
                        # 冒烟边界：1 个 batch 后停止（基础设施参数，不改配置）
                        _log.info("SFT smoke: stopping after step 1 (boundary reached)")
                        self.runtime.save_checkpoint(
                            output_dir / "final",
                            trainer_state=self._sft_trainer_state(epoch=epoch, batch_idx=batch_idx),
                        )
                        return
                    if self._is_primary():
                        batch_sec = time.monotonic() - batch_started_at
                        _log.info(
                            "SFT step %d: loss=%.6f grad_norm=%.4f lr=%.2e batch_sec=%.2f",
                            self.global_step,
                            metrics.get("loss_mean") or 0.0,
                            metrics.get("grad_norm_mean") or 0.0,
                            metrics.get("current_lr") or 0.0,
                            batch_sec,
                        )
                        self._print_json(
                            {
                                "timestamp": _timestamp(),
                                "event": "sft_step",
                                "step": self.global_step,
                                "epoch": epoch,
                                "batch": batch_idx,
                                "loss": metrics.get("loss_mean"),
                                "grad_norm": metrics.get("grad_norm_mean"),
                                "lr": metrics.get("current_lr"),
                                "batch_sec": round(batch_sec, 3),
                                "elapsed_sec": round(time.monotonic() - self.started_at, 1),
                            }
                        )

                    if save_steps > 0 and self.global_step % save_steps == 0:
                        self.runtime.save_checkpoint(
                            output_dir / f"step_{self.global_step}",
                            trainer_state=self._sft_trainer_state(epoch=epoch, batch_idx=batch_idx),
                        )
                        _last_checkpoint_time = time.monotonic()

                    # 墙钟时间周期保存：每隔 N 分钟在优化步之间保存完整 checkpoint
                    if save_period_min > 0:
                        elapsed_since_last = time.monotonic() - _last_checkpoint_time
                        if elapsed_since_last >= save_period_min * 60:
                            self.runtime.save_checkpoint(
                                output_dir / f"time_{_timestamp()}",
                                trainer_state=self._sft_trainer_state(
                                    epoch=epoch, batch_idx=batch_idx
                                ),
                            )
                            _last_checkpoint_time = time.monotonic()
                            _log.info(
                                "SFT: time-based checkpoint saved at epoch=%d batch=%d "
                                "step=%d (period=%d min)",
                                epoch, batch_idx, self.global_step, save_period_min,
                            )

                # epoch 结束 checkpoint
                if self.config.training.save_checkpoint_every_epoch:
                    self.runtime.save_checkpoint(
                        output_dir / f"epoch_{epoch}",
                        trainer_state=self._sft_trainer_state(
                            epoch=epoch, batch_idx=len(batches) - 1, is_epoch_end=True,
                        ),
                    )
                    _last_checkpoint_time = time.monotonic()

            # final checkpoint
            self.runtime.save_checkpoint(
                output_dir / "final",
                trainer_state=self._sft_trainer_state(
                    epoch=self.config.training.max_epochs - 1,
                    batch_idx=0,
                    is_epoch_end=True,
                ),
            )
            _log.info(
                "SFT complete: steps=%d elapsed=%.1fs",
                self.global_step,
                time.monotonic() - self.started_at,
            )
        finally:
            self.runtime.close()

    def _sft_trainer_state(
        self, *, epoch: int, batch_idx: int, is_epoch_end: bool = False,
    ) -> dict[str, Any]:
        """构建 SFT trainer state 字典，用于 checkpoint 保存。

        与 RL 的 ``_checkpoint_trainer_state`` 保持一致的 payload 结构：
        ``format`` 标记区分 SFT/RL 格式，``global_step`` / ``epoch`` 为共有字段，
        ``batch_idx`` 为 SFT 特有（epoch 内恢复位置）。

        ``is_epoch_end`` 标记 checkpoint 是否为 epoch 结束保存：
        - ``True``：epoch 结束 checkpoint → resume 跳到下一 epoch batch=0
        - ``False``：epoch 内 checkpoint（step/time）→ resume 从 batch_idx+1 续
        """
        return {
            "format": "native-sft-trainer-state",
            "version": 1,
            "global_step": self.global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "total_samples": self.total_samples,
            "is_epoch_end": is_epoch_end,
        }

    def _resume_if_requested(self) -> None:
        """从配置指定的 checkpoint 恢复 SFT 训练状态。

        恢复内容：
        - LoRA 权重（适配器 ``load_checkpoint``）
        - 优化器 / LR scheduler 状态
        - RNG（各 DP rank 不同 seed 恢复）
        - 数据位置：当前 epoch、batch_idx

        兼容 v0.26.1 旧格式（``{"step": ..., "epoch": ...}``，无 ``format`` 标记）：
        旧格式只能从 epoch 边界恢复（从 epoch+1 开始）。
        """
        checkpoint = self.config.training.resume_from_checkpoint
        if not checkpoint:
            return
        checkpoint_dir = Path(checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"training.resume_from_checkpoint does not exist: {checkpoint_dir}"
            )
        _log = logging.getLogger("graspo.sft_trainer")
        trainer_state = self.runtime.load_checkpoint(checkpoint_dir)
        if trainer_state is None:
            raise RuntimeError(
                "SFT checkpoint is missing trainer_state; resume requires a "
                "recoverable checkpoint"
            )
        fmt = trainer_state.get("format")
        if fmt is None:
            # v0.26.1 旧格式：{"step": ..., "epoch": ...}，无 format 标记
            _log.warning(
                "Resuming from legacy SFT checkpoint (v0.26.1 format, no 'format' marker); "
                "only epoch-boundary resume is supported — starting from epoch=%d. "
                "Future checkpoints will use the new format automatically.",
                int(trainer_state.get("epoch") or 0) + 1,
            )
            self.global_step = int(trainer_state.get("step") or 0)
            saved_epoch = int(trainer_state.get("epoch") or 0)
            self._resume_epoch = saved_epoch + 1
            self._resume_batch_idx = 0
        elif fmt == "native-sft-trainer-state":
            self.global_step = int(trainer_state["global_step"])
            saved_epoch = int(trainer_state["epoch"])
            saved_batch_idx = int(trainer_state.get("batch_idx") or 0)
            is_epoch_end = bool(trainer_state.get("is_epoch_end") or False)
            self.total_samples = int(
                trainer_state.get("total_samples") or self.total_samples
            )
            if is_epoch_end:
                # epoch 结束 checkpoint：该 epoch 已全部处理完，跳到下一 epoch
                self._resume_epoch = saved_epoch + 1
                self._resume_batch_idx = 0
            else:
                # epoch 内 checkpoint（step/time）：从 batch_idx+1 续，不重跑已保存的 batch
                self._resume_epoch = saved_epoch
                self._resume_batch_idx = saved_batch_idx + 1
            _log.info(
                "SFT resume: checkpoint=%s step=%d epoch=%d batch_idx=%d "
                "is_epoch_end=%s → resume_epoch=%d resume_batch_idx=%d",
                checkpoint_dir.name,
                self.global_step,
                saved_epoch,
                saved_batch_idx,
                is_epoch_end,
                self._resume_epoch,
                self._resume_batch_idx,
            )
        else:
            raise RuntimeError(
                f"Unsupported SFT trainer_state format: {fmt!r}; "
                f"expected 'native-sft-trainer-state' or legacy (no format marker)"
            )

    def _is_primary(self) -> bool:
        return self.runtime.is_primary()

    def _print_json(self, payload: dict[str, Any]) -> None:
        if self._is_primary():
            logging.getLogger("graspo.sft_trainer").info(json.dumps(payload, ensure_ascii=False))


def create_native_sft_trainer(config: GraspoConfig, selection: Any = None) -> SFTTrainer:
    """native 后端的 SFT 训练器工厂（供 ``graspo.sft_backends`` 注册表发现）。

    调用契约：``factory(config, selection) -> 含 train(smoke=bool) 的训练器``。

    Args:
        config: GraspoConfig 实例。
        selection: BackendSelection 实例（native SFT 不需要它做分派，
            保留该参数以与其它后端工厂签名一致，保证注册表可统一调用）。

    Returns:
        ``SFTTrainer`` 实例；``train(smoke=...)`` 由 ``cli/train_worker.py`` 驱动。
    """
    if config.train_method != "sft":
        raise ValueError(
            f"create_native_sft_trainer requires train_method='sft', got {config.train_method!r}"
        )
    return SFTTrainer(config)
