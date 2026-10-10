"""matrix54 v2 清单生成器：T041 卡位覆盖（资源调度例外）的回归覆盖。

**为什么必须有这个文件**：运行清单 `matrix54_manifest_source.json` 是**生成物**。
只手改清单文件的话，下次重生成会按 `GPU_POOL_BY_CARDS` 把 T041 静默改回 `[4,5]`
（回退陷阱），而当时的测试**不会报警**（既有 `test_generate_matrix.py` 断言的是
**v1 生成器**的产物，不校 v2 清单文件本身）。

本文件把「T041 覆盖 = `[2,3]`、`cards` 仍为 2、其它档逐字不变」固化成断言：
  ① 2026-10-05 主席令的默认映射 `GPU_POOL_BY_CARDS` **逐字未变**；
  ② 覆盖表只点名 T041，卡位 `[2,3]`；
  ③ `patch_tier` 对 T041 的产出：`gpus=[2,3]`、`cards=2`、**除 `gpus` 外逐字段不变**；
  ④ 未点名的档走默认映射（逐档与 `GPU_POOL_BY_CARDS` 相等）；
  ⑤ 覆盖表卡数错配 ⇒ fail-closed（不"将就"）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_SPEC = importlib.util.spec_from_file_location(
    "generate_matrix_manifest_source_uut", _HERE.parent / "generate_matrix_manifest_source.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gmv2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gmv2)

_V1_TIERS = json.loads(gmv2.V1_MANIFEST_PATH.read_text(encoding="utf-8"))["tiers"]
_V1_BY_ID = {t["tier_id"]: t for t in _V1_TIERS}


def _patched(tier_id: str) -> dict:
    return gmv2.patch_tier(_V1_BY_ID[tier_id], "/data-root", "/pregen-dir")


def test_default_pool_map_is_verbatim():
    """2026-10-05 主席令的卡数→卡位映射**逐字未变**（覆盖表不得顶替它）。"""
    assert gmv2.GPU_POOL_BY_CARDS == {1: [4], 2: [4, 5], 3: [4, 5, 6], 4: [4, 5, 6, 7]}


def test_tier_gpu_override_is_exactly_t041():
    """覆盖表**只**点名 T041，卡位 `[2,3]`（用户 2026-10-09 指令的资源调度例外）。"""
    assert gmv2.TIER_GPU_OVERRIDE == {"T041": [2, 3]}


def test_t041_is_moved_to_2_3_and_cards_unchanged():
    """★ 核心断言：T041 的 `gpus` 覆盖为 `[2,3]`，而 `cards`（卡数）**仍为 2**。"""
    out = _patched("T041")
    assert out["gpus"] == [2, 3]
    assert out["cards"] == 2
    assert len(out["gpus"]) == out["cards"]


def test_t041_override_changes_nothing_but_gpus(monkeypatch):
    """★ 覆盖只改 `gpus`：把覆盖表清空后逐字段对比，除 `gpus` 外**一字不改**。"""
    monkeypatch.setattr(gmv2, "TIER_GPU_OVERRIDE", {})
    default = _patched("T041")
    assert default["gpus"] == [4, 5]
    monkeypatch.undo()
    overridden = _patched("T041")
    assert overridden["gpus"] == [2, 3]
    assert set(default) == set(overridden)
    for key, value in default.items():
        if key == "gpus":
            continue
        assert overridden[key] == value, key


def test_all_other_tiers_follow_the_default_map():
    """⛔ 覆盖表不得波及其它档：未点名档的 `gpus` == `GPU_POOL_BY_CARDS[cards]`。"""
    for tier in _V1_TIERS:
        tier_id = tier["tier_id"]
        if tier_id in gmv2.TIER_GPU_OVERRIDE:
            continue
        out = _patched(tier_id)
        assert out["gpus"] == list(gmv2.GPU_POOL_BY_CARDS[int(tier["cards"])]), tier_id
        assert out["cards"] == int(tier["cards"]), tier_id


def test_override_card_count_mismatch_is_fail_closed(monkeypatch):
    """覆盖表卡数 ≠ `cards` ⇒ fail-closed（不静默"将就"成错卡数）。"""
    monkeypatch.setattr(gmv2, "TIER_GPU_OVERRIDE", {"T041": [2, 3, 4]})
    with pytest.raises(AssertionError, match="卡数"):
        _patched("T041")


def test_override_duplicate_gpus_is_fail_closed(monkeypatch):
    """覆盖表含重复卡 ⇒ fail-closed（重复卡会让"卡数"名不副实）。"""
    monkeypatch.setattr(gmv2, "TIER_GPU_OVERRIDE", {"T041": [2, 2]})
    with pytest.raises(AssertionError, match="重复"):
        _patched("T041")
