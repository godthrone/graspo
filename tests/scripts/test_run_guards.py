"""`scripts/run_guards.py` 的用例（§7 形态谓词 / §8 新鲜度 / §10 Mamba 并发）。

每族各含**正例 + 负例**；只创建临时目录，不碰 GPU / 容器 / 真实产物。
"""

import os
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import run_guards as g  # noqa: E402


# ---------- §7 形态谓词 ----------
def test_form_accepts_full_hf(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00004.safetensors").write_text("x")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok and form == "full", msg


def test_form_accepts_peft_adapter_without_config_json(tmp_path: Path):
    """★ 关键回归：adapter 目录**没有** config.json（T008 曾被误拒）⇒ 必须通过。"""
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_text("x")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok and form == "adapter", msg


def test_form_accepts_native_rank_shards(tmp_path: Path):
    (tmp_path / "rank_00000_tp_00_pp_00.pt").write_text("x")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok and form == "native_rank", msg


def test_form_rejects_missing_directory(tmp_path: Path):
    ok, form = g.detect_form_and_validate(str(tmp_path / "nope"))[1], None
    assert ok is False


def test_form_rejects_unknown_layout(tmp_path: Path):
    (tmp_path / "readme.txt").write_text("nothing useful")
    form, ok, msg = g.detect_form_and_validate(str(tmp_path))
    assert ok is False and form == "unknown", msg


# ---------- §8 产物新鲜度 ----------
def test_fresh_pre_passes_when_absent(tmp_path: Path):
    ok, msg = g.check_fresh(str(tmp_path / "merged"), "pre")
    assert ok, msg


def test_fresh_pre_rejects_stale_directory(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model-00001-of-00018.safetensors").write_text("stale")
    ok, msg = g.check_fresh(str(d), "pre")
    assert ok is False and "陈旧" in msg, msg


def test_fresh_post_passes_for_new_dir_with_weights(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model.safetensors").write_text("x")
    ok, msg = g.check_fresh(str(d), "post", since_epoch=time.time() - 60)
    assert ok, msg


def test_fresh_post_rejects_dir_without_weights(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text("{}")
    ok, msg = g.check_fresh(str(d), "post")
    assert ok is False and "无权重" in msg, msg


def test_fresh_post_rejects_older_than_batch(tmp_path: Path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "model.safetensors").write_text("x")
    ok, msg = g.check_fresh(str(d), "post", since_epoch=time.time() + 3600)
    assert ok is False and "疑似陈旧" in msg, msg


# ---------- §10 Mamba 并发 ----------
def test_mamba_accepts_within_blocks():
    ok, msg = g.check_mamba(192, 232)
    assert ok, msg


def test_mamba_accepts_equal_bound():
    ok, msg = g.check_mamba(232, 232)
    assert ok, msg


def test_mamba_rejects_default_over_blocks():
    """★ 关键回归：缺省 1024 > 232 blocks（T024 实测事故）。"""
    ok, msg = g.check_mamba(1024, 232)
    assert ok is False and "Engine core initialization failed" in msg, msg
