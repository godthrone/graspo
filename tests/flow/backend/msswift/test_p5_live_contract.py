"""★ P6②：把 msswift 的**正确性检查落进仓库 tests/**（原只在 `.local/` 验证台）。

这些检查**不需要 GPU / ms-swift**：要么是活源码的静态判据，要么是纯函数的行为判据。
与验证台 `task-equiv-harness/msswift_p5_live_check.py` 同判据（同一真相源，§1.4）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graspo.flow.adapter.contract import AlignmentError, IncompleteGroupError
from graspo.flow.backend.msswift.alignment import align_by_key
from graspo.flow.backend.msswift.trainer import _group_rewards, _rollout_groups

TRAINER = (
    Path(__file__).resolve().parents[4] / "src/graspo/flow/backend/msswift/trainer.py"
)


def _code_lines() -> list[str]:
    """只取**可执行行**（退役注释里提到旧警告文本属历史登记，不算残留）。"""
    return [
        line
        for line in TRAINER.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]


class _Sample:
    def __init__(self, prompt_id: str, key: str) -> None:
        self.prompt_id = prompt_id
        self.request_id = key


# ── 静态判据（活源码）────────────────────────────────────────────────────────
def test_live_source_has_no_hardcoded_retry_count_zero():
    """★ 旧实现 `retry_count=0` 硬编码 ⇒ `GroupDecision.RETRY` 退化为"本批不训"。"""
    assert all("retry_count=0" not in line for line in _code_lines())


def test_live_source_has_no_positional_pairing_fallback():
    """★ 旧实现在长度不匹配时按行号配对（S1 现象）。"""
    assert all("falling back to positional pairing" not in line for line in _code_lines())


def test_live_source_does_not_use_get_even_process_data():
    """★ G2：位置写回的入口（`get_even_process_data`）不得再被调用。"""
    assert all("get_even_process_data" not in line for line in _code_lines())


# ── 行为判据（纯函数 / 活函数）───────────────────────────────────────────────
def test_rollout_groups_fails_explicitly_on_incomplete_slice():
    """★ G1/I3：本进程切片**不足一组** ⇒ `IncompleteGroupError`（旧实现当作整组）。"""
    samples = [_Sample("p", f"p#{i}") for i in range(4)]  # 只有半个 8 条组
    with pytest.raises(IncompleteGroupError):
        _rollout_groups(samples, 8)


def test_rollout_groups_returns_complete_groups():
    samples = [_Sample("p", f"p#{i}") for i in range(8)]
    groups = _rollout_groups(samples, 8)
    assert len(groups) == 1 and len(groups[0]) == 8


def test_group_rewards_fails_instead_of_positional_fallback():
    """★ G2/I2：奖励表与样本数不匹配 ⇒ `AlignmentError`（不再按行号配对）。"""
    samples = [_Sample("p", f"p#{i}") for i in range(4)]

    import torch

    table = torch.zeros(8, 1)  # 表有 8 行，本进程只有 4 条样本
    with pytest.raises(AlignmentError):
        _group_rewards(samples, table, None)


def test_key_alignment_is_fail_closed():
    with pytest.raises(AlignmentError, match="未按键对齐"):
        align_by_key(["a", "b"], {"a": 1.0})
