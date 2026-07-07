"""SFT 训练循环，复用 GraspoFlow 基础设施（TP/PP、模型加载、LoRA、checkpoint）。

不依赖任何 RL 模块（reward/advantage/buffer/rollout）。
"""


import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graspo.backends.graspoflow.runtime import (
    GraspoFlowRuntime,
    GraspoFlowRuntimeProtocol,
    validate_graspoflow_runtime_config,
)
from graspo.core.data import load_jsonl, sft_tokenize
from graspo.core.logging import setup_logging
from graspo.core.schema import GraspoConfig


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
        runtime: GraspoFlowRuntimeProtocol | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or GraspoFlowRuntime.from_config(config)
        self.started_at = time.monotonic()
        self.global_step = 0
        self.total_samples = 0

    def train(self) -> None:
        """SFT 训练主入口。"""
        validate_graspoflow_runtime_config(self.config)
        self.runtime.validate()
        self.runtime.setup()
        rank = int(getattr(self.runtime, "rank", 0))
        setup_logging(self.config.training.output_dir, rank=rank)
        _set_random_seed(int(self.config.training.seed), rank=rank)
        _log = logging.getLogger("graspo.sft_trainer")

        # 加载并 tokenize 数据（所有 rank 各自执行，因为 train_batch_sft 在所有 rank 上调用）
        samples = load_jsonl(self.config.data.train_path)
        self.total_samples = len(samples)
        _log.info("SFT: loaded %d samples from %s", self.total_samples, self.config.data.train_path)

        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._is_primary():
            _backup_config(self.config, output_dir)

        _log.info("SFT: tokenizing %d samples...", self.total_samples)
        tokenized = [
            sft_tokenize(
                s,
                self.runtime._adapter.tokenizer,
                max_seq_length=self.config.data.max_prompt_length,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            for s in samples
        ]
        _log.info(
            "SFT: tokenization complete, avg tokens=%.0f",
            sum(len(t["input_ids"]) for t in tokenized) / max(len(tokenized), 1),
        )

        optimize_prompt_batch_size = max(1, int(self.config.training.optimize_prompt_batch_size))
        optimize_iterations_per_step = max(
            1, int(self.config.training.optimize_iterations_per_step)
        )
        max_grad_norm = float(self.config.training.max_grad_norm)
        save_steps = int(self.config.training.save_steps)

        _log.info(
            "SFT config: batch_size=%d grad_accum=%d max_epochs=%d lr=%.1e max_seq_len=%d",
            optimize_prompt_batch_size,
            optimize_iterations_per_step,
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
                        optimize_iterations_per_step=optimize_iterations_per_step,
                        max_grad_norm=max_grad_norm,
                    )
                    self.global_step += 1
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
        is_primary = getattr(self.runtime, "is_primary", None)
        if callable(is_primary):
            return bool(is_primary())
        return int(getattr(self.runtime, "rank", 0)) == 0

    def _print_json(self, payload: dict[str, Any]) -> None:
        if self._is_primary():
            logging.getLogger("graspo.sft_trainer").info(
                json.dumps(payload, ensure_ascii=False)
            )


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _set_random_seed(seed: int, *, rank: int = 0) -> None:
    """设置所有随机数生成器的种子，确保可复现性（宪法 §6）。"""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _backup_config(config: Any, output_dir: Path) -> None:
    import yaml

    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(config.model_dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
