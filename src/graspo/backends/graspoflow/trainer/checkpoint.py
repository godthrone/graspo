"""GraspoFlowTrainer checkpoint 保存与恢复的 mixin。"""


from pathlib import Path
from typing import Any

from graspo.backends.graspoflow.checkpoint import save_native_checkpoint
from graspo.backends.graspoflow.trainer.helpers import (
    epoch_stats_from_dict,
    epoch_stats_to_dict,
    train_stats_from_dict,
    train_stats_to_dict,
)


class CheckpointMixin:
    """Checkpoint 保存、恢复、导出、trainer state 序列化的 mixin。"""

    config: Any
    runtime: Any
    stats: Any
    current_epoch_stats: Any
    replay_buffer: Any
    pending_batch_attempts: Any
    pending_batch_timings: Any
    global_step: int
    sample_index: int
    total_samples: int
    backend_name: str
    resume_info: dict[str, Any] | None

    def _save_checkpoint(self, path: Path, *, epoch: int) -> None:
        """保存可恢复的 native checkpoint。"""
        save_native_checkpoint(
            self.runtime,
            path,
            trainer_state=self._checkpoint_trainer_state(epoch=epoch),
        )
        if path.name == "final" and self.config.export.final_formats and self.runtime.is_primary():
            self._export_final_checkpoint(path)

    def _export_final_checkpoint(self, checkpoint_dir: Path) -> None:
        """导出最终 checkpoint 为可部署格式。"""
        from graspo.backends.graspoflow.lora_io import export_from_checkpoint

        for export_format in self.config.export.final_formats:
            output_dir = checkpoint_dir / str(export_format)
            export_from_checkpoint(
                checkpoint_dir,
                output_dir,
                export_format=str(export_format),
                base_model_path=self.config.model.model_path,
            )
            self._print_json(
                {
                    "timestamp": self._timestamp(),
                    "event": "checkpoint_exported",
                    "checkpoint": str(checkpoint_dir),
                    "format": str(export_format),
                    "output": str(output_dir),
                }
            )

    def _resume_if_requested(self) -> None:
        """从配置指定的 checkpoint 恢复训练状态。"""
        checkpoint = self.config.training.resume_from_checkpoint
        if not checkpoint:
            return
        checkpoint_dir = Path(checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"training.resume_from_checkpoint does not exist: {checkpoint_dir}"
            )
        loader = getattr(self.runtime, "load_checkpoint", None)
        if not callable(loader):
            raise RuntimeError("Selected runtime does not support checkpoint resume")
        trainer_state = loader(checkpoint_dir)
        if trainer_state is None:
            raise RuntimeError(
                "GRASPO checkpoint is missing trainer_state; latest-only resume requires "
                "a current recoverable checkpoint"
            )
        if trainer_state.get("format") != "graspoflow-trainer-state":
            raise RuntimeError(
                "Unsupported trainer_state format: "
                f"{trainer_state.get('format')!r}; latest-only resume requires current GRASPO"
            )
        self._restore_trainer_state(trainer_state)
        self.resume_info = {
            "checkpoint": str(checkpoint_dir),
            "global_step": self.global_step,
            "epoch": self.current_epoch_stats.epoch,
            "samples_seen": self.current_epoch_stats.samples_seen,
        }
        self._print_json(
            {
                "timestamp": self._timestamp(),
                "event": "checkpoint_resumed",
                **self.resume_info,
            }
        )

    def _checkpoint_trainer_state(self, *, epoch: int) -> dict[str, Any]:
        """序列化 trainer 状态用于 checkpoint。"""
        if len(self.replay_buffer) > 0:
            raise RuntimeError("Cannot save recoverable checkpoint while ReplayBuffer is non-empty")
        return {
            "format": "graspoflow-trainer-state",
            "version": 1,
            "global_step": self.global_step,
            "sample_index": self.sample_index,
            "total_samples": self.total_samples,
            "epoch": epoch,
            "run_stats": train_stats_to_dict(self.stats),
            "epoch_stats": epoch_stats_to_dict(self.current_epoch_stats),
            "config_snapshot": {
                "backend": self.backend_name,
                "rollout_group_size": self.config.training.rollout_group_size,
                "optimize_prompt_batch_size": self.config.training.optimize_prompt_batch_size,
                "optimize_iterations_per_step": self.config.training.optimize_iterations_per_step,
                "rollout_max_retries": self.config.training.rollout_max_retries,
                "max_new_tokens": self.config.training.max_new_tokens,
            },
        }

    def _restore_trainer_state(self, state: dict[str, Any]) -> None:
        """从 checkpoint 恢复 trainer 状态。"""
        self.global_step = int(state["global_step"])
        self.sample_index = int(state.get("sample_index") or 0)
        self.stats = train_stats_from_dict(state.get("run_stats") or {})
        self.current_epoch_stats = epoch_stats_from_dict(state.get("epoch_stats") or {})
        if self.current_epoch_stats.epoch < 0:
            self.current_epoch_stats.epoch = int(state.get("epoch") or 0)
        self.stats.optimized_steps = max(self.stats.optimized_steps, self.global_step)
        self.replay_buffer.clear()
        self.pending_batch_attempts.clear()
        self.pending_batch_timings.clear()
