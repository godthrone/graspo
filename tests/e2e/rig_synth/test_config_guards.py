#!/usr/bin/env python3
"""配置生成器的**白名单守卫**回归测试（纯标准库，无需 torch/ms-swift）。

**为什么有这份测试**（宪法 §2 防呆 / 指挥官 2026-09-19 裁定 3）

配套配置是**派生文件**：它必须只偏离基线在**恰三个**白名单键上
（`data.max_prompt_length` / `training.run_name` / `training.save_steps`），
多改一个键就会污染"长度是唯一自变量"的对照。本文件证明这条被**机器强制**：

1. :func:`test_patch_config_changes_exactly_three_whitelisted_keys` —— 改写恰好 3 个键；
2. :func:`test_diff_guard_rejects_fourth_key` —— **能真失败的证据**：多改一个键 ⇒ 硬失败；
3. :func:`test_diff_guard_accepts_all_three_whitelisted_keys` —— 3 个键都被接受；
4. :func:`test_save_steps_override_is_one` —— 合成档 `save_steps` 覆盖为 1（裁定 3）；
5. :func:`test_patch_config_rejects_missing_key` —— 基线缺键即失败（不静默产出）。

运行：``python3 -m pytest tests/e2e/rig_synth/test_config_guards.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_configs import (  # noqa: E402
    ALLOWED_KEYS,
    ALLOWED_PATCH_KEY,
    ALLOWED_RUN_NAME_KEY,
    ALLOWED_SAVE_STEPS_KEY,
    SYNTH_SAVE_STEPS,
    TIERS,
    ConfigEmitError,
    _assert_only_allowed_diff,
    _patch_config,
)

#: 一份最小基线（结构与仓内 T013.yaml 的关键行一致）。
BASE = """\
train_method: sft
backend: msswift
data:
  train_path: /work/elam-v5/subsets/T013.jsonl
  max_prompt_length: 8192
training:
  output_dir: /out/T013
  run_name: T013
  save_steps: 100
lora:
  r: 8
"""


def _patched(**kwargs) -> tuple[str, list[str]]:
    defaults = {
        "max_prompt_length": 131072,
        "run_name": "T013-synth-x-128k",
        "save_steps": SYNTH_SAVE_STEPS,
    }
    defaults.update(kwargs)
    return _patch_config(BASE, **defaults)


# --------------------------------------------------------------------------- #
# 白名单恰三个键
# --------------------------------------------------------------------------- #
def test_whitelist_is_exactly_three_keys() -> None:
    """白名单**恰三个**键（多一个都是越界；这是"长度是唯一自变量"的结构保证）。"""
    assert len(ALLOWED_KEYS) == 3, ALLOWED_KEYS
    assert set(ALLOWED_KEYS) == {
        ALLOWED_PATCH_KEY,
        ALLOWED_RUN_NAME_KEY,
        ALLOWED_SAVE_STEPS_KEY,
    }
    assert ALLOWED_KEYS == (
        ALLOWED_PATCH_KEY,
        ALLOWED_RUN_NAME_KEY,
        ALLOWED_SAVE_STEPS_KEY,
    )


def test_patch_config_changes_exactly_three_whitelisted_keys() -> None:
    new_text, changes = _patched()
    assert len(changes) == 3, changes
    assert any("max_prompt_length" in c for c in changes)
    assert any("run_name" in c for c in changes)
    assert any("save_steps" in c for c in changes)
    # 实际内容也对。
    assert "  max_prompt_length: 131072\n" in new_text
    assert "  run_name: T013-synth-x-128k\n" in new_text
    assert f"  save_steps: {SYNTH_SAVE_STEPS}\n" in new_text


def test_save_steps_override_is_one() -> None:
    """裁定 3：合成档 A3 保障 ⇒ save_steps 覆盖为 1。"""
    assert SYNTH_SAVE_STEPS == 1
    new_text, _ = _patched()
    assert "  save_steps: 1\n" in new_text
    assert "  save_steps: 100\n" not in new_text


def test_diff_guard_accepts_all_three_whitelisted_keys() -> None:
    new_text, _ = _patched()
    _assert_only_allowed_diff(BASE, new_text)  # 不抛即通过


# --------------------------------------------------------------------------- #
# ★ 能真失败的证据
# --------------------------------------------------------------------------- #
def test_diff_guard_rejects_fourth_key() -> None:
    """★★ 多改一个键（此处 `learning_rate`）⇒ 必须抛 ConfigEmitError。"""
    new_text, _ = _patched()
    tampered = new_text.replace("  save_steps: 1\n", "  save_steps: 1\n  learning_rate: 0.999\n")
    with pytest.raises(ConfigEmitError, match="白名单之外"):
        _assert_only_allowed_diff(BASE, tampered)


def test_diff_guard_rejects_modified_value_of_non_whitelisted_key() -> None:
    """改**非白名单键的取值**（不是新增行）同样必须被拒。"""
    new_text, _ = _patched()
    tampered = new_text.replace("  r: 8", "  r: 64")
    with pytest.raises(ConfigEmitError, match="白名单之外"):
        _assert_only_allowed_diff(BASE, tampered)


def test_patch_config_rejects_missing_key() -> None:
    """基线缺 `save_steps` ⇒ 失败（不静默产出缺键配置）。"""
    bad = BASE.replace("  save_steps: 100\n", "")
    with pytest.raises(ConfigEmitError, match="save_steps"):
        _patch_config(bad, max_prompt_length=4096, run_name="x", save_steps=1)


def test_patch_config_rejects_duplicate_key() -> None:
    """同一键出现两次 ⇒ 失败（单一真相源被破坏）。"""
    dup = BASE + "  save_steps: 7\n"
    with pytest.raises(ConfigEmitError, match="多个 save_steps"):
        _patch_config(dup, max_prompt_length=4096, run_name="x", save_steps=1)


def test_tiers_cover_all_seven() -> None:
    assert set(TIERS) == {"4k", "8k", "16k", "32k", "64k", "96k", "128k"}
    assert TIERS["128k"] == 131072


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
