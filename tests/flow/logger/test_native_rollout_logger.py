"""Tests for ``graspo.flow.logger.native_rollout_logger`` — 事件流与关联键。"""

import json

from graspo.flow.logger.native_rollout_logger import NativeRolloutLogger
from graspo.flow.logging import run_log_dir


def test_write_event_writes_events_jsonl(tmp_path):
    """write_event 落盘到 logs/<run_id>/events.jsonl（每次重启一文件夹，v0.21 结构化事件流）。"""
    logger = NativeRolloutLogger(tmp_path)
    logger.write_event({"event": "train_step", "step": 1})
    rows = [
        json.loads(line)
        for line in (run_log_dir(tmp_path) / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["event"] == "train_step"
    assert rows[0]["step"] == 1


def test_all_event_files_get_timestamp_and_run_id(tmp_path):
    """各事件文件统一注入 timestamp + run_id 关联键（§13.3 跨文件对齐时间线）。"""
    logger = NativeRolloutLogger(tmp_path / "my_run")
    logger.write_event({"event": "run_start"})
    logger.write_readable({"event": "graspo_group", "step": 1, "decision": "retry"})
    logger.write_raw({"event": "graspo_group", "step": 1, "raw": {"x": 1}})
    logger.write_train_batch_readable({"event": "graspo_train_batch", "step": 1})
    logger.write_timing_event({"event": "timing_event", "phase": "rollout_attempt"})

    for name in (
        "events.jsonl",
        "rollouts.readable.jsonl",
        "rollouts.raw.jsonl",
        "train_batches.readable.jsonl",
        "timing_events.jsonl",
    ):
        rows = [
            json.loads(line)
            for line in (run_log_dir(tmp_path / "my_run") / name)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows, name
        for row in rows:
            assert row["run_id"] == "my_run", f"{name}: {row}"
            assert "timestamp" in row, f"{name}: missing timestamp"


def test_write_event_preserves_existing_timestamp(tmp_path):
    """已有 timestamp 的事件不被覆盖。"""
    logger = NativeRolloutLogger(tmp_path)
    logger.write_event({"event": "train_step", "timestamp": "2026-08-07T00:00:00+08:00"})
    row = json.loads(
        (run_log_dir(tmp_path) / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["timestamp"] == "2026-08-07T00:00:00+08:00"
