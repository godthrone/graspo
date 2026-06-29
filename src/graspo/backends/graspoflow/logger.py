import json
from pathlib import Path
from typing import Any


class NativeRolloutLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        readable_enabled: bool = True,
        raw_enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.readable_enabled = readable_enabled
        self.raw_enabled = raw_enabled
        self.readable_path = self.logs_dir / "rollouts.readable.jsonl"
        self.raw_path = self.logs_dir / "rollouts.raw.jsonl"
        self.train_batches_readable_path = self.logs_dir / "train_batches.readable.jsonl"
        self.timing_path = self.logs_dir / "timing_events.jsonl"

    def write_readable(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.readable_path, readable_payload(payload))

    def write_raw(self, payload: dict[str, Any]) -> None:
        if self.raw_enabled:
            self._append(self.raw_path, _to_jsonable(payload))

    def write_train_batch_readable(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.train_batches_readable_path, train_batch_readable_payload(payload))

    def write_timing_event(self, payload: dict[str, Any]) -> None:
        if self.readable_enabled:
            self._append(self.timing_path, timing_event_payload(payload))

    def write_error(self, payload: dict[str, Any]) -> None:
        """Write an ERROR-level event to the common error log."""
        self._append(self.logs_dir / "error.log", _to_jsonable(payload))

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

from graspo.backends.graspoflow.logger_helpers import (  # noqa: E402, F401
    _get_index,
    _is_pure_tool_call_task,
    _target_tool_call_counts,
    _to_jsonable,
    group_debug_summary,
    likely_truncated_json,
    readable_payload,
    summarize_json_markers,
    summarize_think,
    timing_event_payload,
    train_batch_attempt_summary,
    train_batch_readable_payload,
)
