"""阻断 B 的负向用例：建日志目录与清输出目录的顺序冲突 ⇒ ENOENT。

T028 实测（native 后端，0 个 optimizer step）：
``FileNotFoundError: '/out/T028/logs/T028/events.jsonl'`` → ``ChildFailedError``。

真因链（本文件逐段复现，全部是仓内真实代码，零 GPU）：
1. ``NativeRolloutLogger.__init__`` → ``run_log_dir(out)`` 建出 ``{out}/logs/<run_id>/``；
2. ``train()`` 里 ``prepare_output_dir(out, overwrite=True)``
   （``overwrite_output_dir: true``）执行 ``shutil.rmtree(out)`` ⇒ **连 logs 子树一起删**；
3. 随后写 ``{out}/logs/<run_id>/events.jsonl`` ⇒ ``open(..., "a")`` 抛 ENOENT。

修复（选项 a：**顺序对齐**）= ``prepare_output_dir`` 提到日志目录创建**之前**，
再重建日志器。本文件用"旧顺序 / 新顺序"两种调用序列做负向对照。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from graspo.flow.logger.native_rollout_logger import NativeRolloutLogger
from graspo.flow.logging import run_log_dir
from graspo.flow.lora.lora_io import prepare_output_dir

_TRAINER = (
    Path(__file__).resolve().parents[3] / "src" / "graspo" / "flow" / "trainer" / "trainer.py"
)


def _train_source() -> str:
    return _TRAINER.read_text(encoding="utf-8")


# ── 1. ★ 旧顺序真失败：先建日志目录，再 prepare_output_dir(overwrite=True) ──


def test_old_order_loses_log_dir_and_events_write_fails(tmp_path: Path):
    """修复前的顺序：``run_log_dir`` 先建，``rmtree`` 后删 ⇒ 写 events.jsonl 必炸。"""
    out = tmp_path / "run"
    out.mkdir()
    logger = NativeRolloutLogger(out)  # ← 建出 out/logs/<run_id>/
    logs_dir = logger.logs_dir  # 注意：不要用 run_log_dir() 复核——它会 mkdir 出目录
    assert logs_dir.exists(), "前置：日志目录已建出"

    prepare_output_dir(out, overwrite=True)  # ← rmtree 整棵 out（含 logs）

    assert not logs_dir.exists(), "前置：prepare_output_dir(overwrite=True) 确实把 logs 子树删掉了"
    with pytest.raises(FileNotFoundError) as excinfo:
        logger.write_event({"event": "run_start"})
    # 与 T028 真机报错同型（同一路径形态：{out}/logs/<run_id>/events.jsonl）
    assert "events.jsonl" in str(excinfo.value), str(excinfo.value)


def test_old_order_second_prepare_call_after_log_recreation_fails(tmp_path: Path):
    """旧代码的第二次调用形态：先重建 logs，再 prepare_output_dir(overwrite=True)。"""
    out = tmp_path / "run"
    out.mkdir()
    NativeRolloutLogger(out)
    prepare_output_dir(out, overwrite=True)
    logger = NativeRolloutLogger(out)  # 旧代码把重建放在 rmtree 之后……
    prepare_output_dir(out, overwrite=True)  # ……但 prepare_output_dir 又会删一次
    with pytest.raises(FileNotFoundError):
        logger.write_event({"event": "train_step", "step": 1})


# ── 2. ★ 新顺序：prepare_output_dir 先，日志目录后 ⇒ events.jsonl 正常落盘 ──


def test_new_order_writes_events_jsonl(tmp_path: Path):
    """修复后的顺序：``prepare_output_dir`` → 重建日志器 → 写事件成功。"""
    out = tmp_path / "run"
    out.mkdir()
    # 上一次运行的遗留产物（含上一次的 logs）——overwrite=true 应当清掉它们
    (out / "config.yaml").write_text("old: true", encoding="utf-8")
    NativeRolloutLogger(out).write_event({"event": "run_start"})  # 上次的日志

    prepare_output_dir(out, overwrite=True)  # ← 先清（连上次的 logs 一起）
    logger = NativeRolloutLogger(out)  # ← 后建本轮的 logs
    logger.write_event({"event": "run_start", "resume": None})
    logger.write_event({"event": "train_step", "step": 1})

    events = run_log_dir(out) / "events.jsonl"
    assert events.exists(), f"events.jsonl 未落盘：{events}"
    lines = events.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines
    assert '"event": "train_step"' in lines[1]
    # 上一次的产物已被清掉（overwrite 语义未变）
    assert not (out / "config.yaml").exists()


# ── 3. overwrite_output_dir=false 的对照：语义逐字不变（fail-closed）────────


def test_overwrite_false_still_refuses_nonempty_dir(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    NativeRolloutLogger(out)  # 目录变非空（logs/）
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_dir(out, overwrite=False)


def test_overwrite_false_on_empty_dir_creates_and_logs(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir()
    prepare_output_dir(out, overwrite=False)  # 空目录：建出来，不报错
    logger = NativeRolloutLogger(out)
    logger.write_event({"event": "run_start"})
    assert (run_log_dir(out) / "events.jsonl").exists()


# ── 4. 接线顺序断言（结构判据，防回归）─────────────────────────────────────


def test_trainer_prepares_output_dir_before_building_logger():
    """``train()`` 里 ``prepare_output_dir`` 必须出现在 ``_build_rollout_logger`` 之前。"""
    source = _train_source()
    body = source[source.index("    def train(self, *, smoke: bool = False)") :]
    end = body.index("\n    def ", 10)
    train_body = body[:end]
    prepare_pos = train_body.index("prepare_output_dir(")
    logger_pos = train_body.index("_build_rollout_logger()")
    assert prepare_pos < logger_pos, (
        "train() 里日志器重建出现在 prepare_output_dir 之前 ⇒ 阻断 B 复发"
    )


def test_trainer_does_not_call_prepare_output_dir_twice():
    """``prepare_output_dir`` 在 ``train()`` 里只能出现一次（第二次会删掉本轮 logs）。"""
    source = _train_source()
    body = source[source.index("    def train(self, *, smoke: bool = False)") :]
    end = body.index("\n    def ", 10)
    train_body = body[:end]
    assert len(re.findall(r"prepare_output_dir\(", train_body)) == 1, (
        "train() 里出现了多次 prepare_output_dir 调用 —— 第二次会 rmtree 掉本轮 logs"
    )
