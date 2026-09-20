"""Tests for ``graspo.flow.logger.native_rollout_logger`` — 事件流与关联键。"""

import json

import pytest

import graspo.flow.logging as graspo_logging
from graspo.flow.logger.native_rollout_logger import NativeRolloutLogger
from graspo.flow.logging import run_log_dir


@pytest.fixture
def _reset_run_id(monkeypatch: pytest.MonkeyPatch):
    """每个用 run_id 的用例都从"干净进程"开始，避免模块级缓存串味。

    与 ``tests/compliance/test_oss_config_boundaries.py`` 的同名 fixture 同一做法
    （§1.4 单一真相源：run_id 只在 ``graspo.flow.logging`` 里缓存一份）。
    """
    monkeypatch.setattr(graspo_logging, "_run_id", None)
    yield
    monkeypatch.setattr(graspo_logging, "_run_id", None)


def test_write_event_writes_events_jsonl(tmp_path):
    """write_event 落盘到 logs/<run_id>/events.jsonl（每次重启一文件夹，v0.21 结构化事件流）。"""
    logger = NativeRolloutLogger(tmp_path)
    logger.write_event({"event": "train_step", "step": 1})
    rows = [
        json.loads(line)
        for line in (run_log_dir(tmp_path) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["event"] == "train_step"
    assert rows[0]["step"] == 1


def test_all_event_files_get_timestamp_and_run_id(tmp_path, _reset_run_id):
    """各事件文件统一注入 timestamp + run_id 关联键（§13.3 跨文件对齐时间线）。

    ``run_id`` 的唯一真相源是 config（``set_run_id`` ← ``training.run_name``，§1.4）；
    这里绑定后，logger 必须把这**同一个**值注入全部事件文件——而不是沿用
    "输出目录名"（那个口径在 ``d6e2aa8`` 已随 ``GRASPO_RUN_ID`` 一起删除，§18.1）。
    """
    graspo_logging.set_run_id("my_run")
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
