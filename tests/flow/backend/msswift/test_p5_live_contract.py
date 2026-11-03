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


# ── 组键语义：批内占位 `prompt_id` ≠ 全局组身份（dp≥2 / 每步多组）───────────────
# 机制（上游 ms-swift 4.5.3，均只读核过）：
#   * `rollout_mixin.py:1242-1258`：`prompt_id` 是**每批内从 0 重编号**的占位符
#     （`prompt_0`…），每次 `_generate_completions` 都从 0 重来；
#   * `grpo_trainer.py:2270-2286`：DAPO 重采样在 `valid < gbs` 时从 **seed+1 的另一个
#     迭代器**取新 prompt 批并**追加**到 `valid_samples`（`:666-673`）；
#   * `grpo_trainer.py:704`：上游按**位置** `rewards.view(-1, num_generations)` 分组。
# ⇒ 「同键 2G 条」是**两个完整的 G 组**，不是「1 组 2G 条」。我方旧实现把批内占位键
#   当全局组身份 ⇒ 误报 `IncompleteGroupError`（A3：`prompt_0` 16 条 vs 契约 8 条）。
def test_rollout_groups_splits_repeated_placeholder_key_into_blocks():
    """★ 同键 >G 条 ⇒ 按连续 G 块切分（与上游按位置分组一致），不报错。"""
    samples = [_Sample("prompt_0", f"a#{i}") for i in range(8)] + [
        _Sample("prompt_0", f"b#{i}") for i in range(8)
    ]
    groups = _rollout_groups(samples, 8)
    assert [len(group) for group in groups] == [8, 8]
    # 顺序切块：第一块=前 G 条、第二块=后 G 条（**不重排**，与上游位置顺序一致）
    assert [s.request_id for s in groups[0]] == [f"a#{i}" for i in range(8)]
    assert [s.request_id for s in groups[1]] == [f"b#{i}" for i in range(8)]


def test_rollout_groups_still_fails_on_truly_incomplete_repeated_key():
    """★ 回归保护：真正的「同键 G-1 / G+1 条」仍必须显式失败（不许把校验改松）。"""
    short = [_Sample("prompt_0", f"s#{i}") for i in range(7)]
    with pytest.raises(IncompleteGroupError):
        _rollout_groups(short, 8)
    long = [_Sample("prompt_0", f"l#{i}") for i in range(9)]
    with pytest.raises(IncompleteGroupError):
        _rollout_groups(long, 8)
    # 2G+1：末尾余 1 条 ⇒ 仍失败（切块不得掩盖残块）
    ragged = [_Sample("prompt_0", f"r#{i}") for i in range(17)]
    with pytest.raises(IncompleteGroupError):
        _rollout_groups(ragged, 8)


def test_rollout_groups_keeps_distinct_keys_as_separate_groups():
    """★ 合法情形不变：不同键各 G 条 ⇒ 各自成组（顺序与条数逐字不变）。"""
    samples = [_Sample("prompt_0", f"x#{i}") for i in range(8)] + [
        _Sample("prompt_1", f"y#{i}") for i in range(8)
    ]
    groups = _rollout_groups(samples, 8)
    assert [len(group) for group in groups] == [8, 8]
    assert [s.request_id for s in groups[0]] == [f"x#{i}" for i in range(8)]
    assert [s.request_id for s in groups[1]] == [f"y#{i}" for i in range(8)]
