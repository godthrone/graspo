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
    validate_graspoflow_runtime_config,
)
from graspo.flow.trainer.helpers import _backup_config, _set_random_seed, _timestamp
from graspo.ripple.data import SFTTokenized, load_jsonl, sft_tokenize


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

    def train(self, *, smoke: bool = False) -> None:
        """SFT 训练主入口。
        """
        validate_graspoflow_runtime_config(self.config)
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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

        optimize_prompt_batch_size = max(1, int(self.config.training.optimize_prompt_batch_size))
        max_grad_norm = float(self.config.training.max_grad_norm)
        save_steps = int(self.config.training.save_steps)

        _log.info(
            "SFT config: batch_size=%d max_epochs=%d lr=%.1e max_seq_len=%d",
            optimize_prompt_batch_size,
            self.config.training.max_epochs,
            self.config.training.learning_rate,
            self.config.data.max_prompt_length,
        )

        try:
            for epoch in range(self.config.training.max_epochs):
                random.Random(int(self.config.training.seed) + epoch).shuffle(tokenized)
                batches = [
                    tokenized[start : start + optimize_prompt_batch_size]
                    for start in range(0, len(tokenized), optimize_prompt_batch_size)
                ]

                _log.info(
                    "SFT epoch %d/%d: %d batches",
                    epoch + 1,
                    self.config.training.max_epochs,
                    len(batches),
                )

                for batch_idx, batch in enumerate(batches):
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
                            trainer_state={"step": self.global_step, "epoch": epoch},
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
                            trainer_state={"step": self.global_step, "epoch": epoch},
                        )

                # epoch 结束 checkpoint
                if self.config.training.save_checkpoint_every_epoch:
                    self.runtime.save_checkpoint(
                        output_dir / f"epoch_{epoch}",
                        trainer_state={"step": self.global_step, "epoch": epoch},
                    )

            # final checkpoint
            self.runtime.save_checkpoint(
                output_dir / "final",
                trainer_state={"step": self.global_step, "epoch": self.config.training.max_epochs},
            )
            _log.info(
                "SFT complete: steps=%d elapsed=%.1fs",
                self.global_step,
                time.monotonic() - self.started_at,
            )
        finally:
            self.runtime.close()

    def _is_primary(self) -> bool:
        return self.runtime.is_primary()

    def _print_json(self, payload: dict[str, Any]) -> None:
        if self._is_primary():
            logging.getLogger("graspo.sft_trainer").info(json.dumps(payload, ensure_ascii=False))
