"""rollout JSONL 领域日志：NativeRolloutLogger（结构化事件日志）。"""

import datetime
import json
from pathlib import Path
from typing import Any

from graspo.flow.logging import append_jsonl_segment, run_log_dir


def _timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class NativeRolloutLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        readable_enabled: bool = True,
        raw_enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.logs_dir = run_log_dir(self.output_dir)
        self.readable_enabled = readable_enabled
        self.raw_enabled = raw_enabled
        # run_id = 输出目录名（config 派生，可复现）——跨文件关联键（§13.3）
        self.run_id = Path(output_dir).name
        self.readable_path = self.logs_dir / "rollouts.readable.jsonl"
        self.raw_path = self.logs_dir / "rollouts.raw.jsonl"
        self.train_batches_readable_path = self.logs_dir / "train_batches.readable.jsonl"
        self.timing_path = self.logs_dir / "timing_events.jsonl"
        self.events_path = self.logs_dir / "events.jsonl"

    def _with_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        """注入 timestamp + run_id 关联键（各事件文件跨文件对齐时间线）。"""
        if "timestamp" not in payload:
            payload = {"timestamp": _timestamp(), **payload}
        payload.setdefault("run_id", self.run_id)
        return payload

    def write_event(self, payload: dict[str, Any]) -> None:
        """结构化事件流（train_step/epoch_summary/checkpoint/group_decision 等）。"""
        self._append(self.events_path, self._with_context(payload))

    def write_readable(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.readable_path, readable_payload(self._with_context(payload)))

    def write_raw(self, payload: dict[str, Any]) -> None:
        if self.raw_enabled:
            self._append(self.raw_path, _to_jsonable(self._with_context(payload)))

    def write_train_batch_readable(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(
                self.train_batches_readable_path,
                train_batch_readable_payload(self._with_context(payload)),
            )

    def write_timing_event(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.timing_path, timing_event_payload(self._with_context(payload)))

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        append_jsonl_segment(path, payload)


from graspo.flow.logger.logger_helpers import (  # noqa: E402, F401
    _get_index,
    _target_tool_call_counts,
    _to_jsonable,
    group_debug_summary,
    readable_payload,
    summarize_think,
    timing_event_payload,
    train_batch_attempt_summary,
    train_batch_readable_payload,
)
