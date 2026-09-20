"""TP 内 next_token 广播的边界（228 实测 `T030` DP=4/TP=1 复现，纯 CPU）。

缺陷：``_broadcast_and_pad_finished`` 写死 ``src=0`` 且不判 TP 组大小。

- TP=1（DP>1 的每一档）时 ``tp_group`` 只有本 rank 一个成员，``src=0`` 直接抛
  ``ValueError: Global rank 0 is not part of group``（T030 rank2 逐字堆栈，
  发生在第一个 decode step）⇒ 4 卡 native GRASPO 根本进不了 rollout 第二步；
- TP>1 且 DP>1 时，只有 global rank 0 所在的 DP 副本包含 rank 0，其余副本同样抛错
  ⇒ src 必须是"组内 rank0 的全局 rank"。

本文件锁定两条契约：**TP=1 不广播**、**TP>1 广播到组内 rank0 的全局 rank**。
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from graspo.flow.parallel.tensor_utils import _broadcast_and_pad_finished


class _Group:
    """进程组替身（只需要被 get_world_size / get_global_rank 认出来）。"""


def _install(monkeypatch: pytest.MonkeyPatch, *, size: int, group_rank0: int, calls: list):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: size)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: group_rank0 + rank)

    def broadcast(tensor, *, src=0, group=None):
        calls.append((src, group))
        tensor.fill_(0)  # 源 rank 的 token 覆盖到本 rank

    monkeypatch.setattr(dist, "broadcast", broadcast, raising=False)


def test_tp_size_one_does_not_broadcast(monkeypatch):
    """TP=1：没有可广播的对端 ⇒ 不得调用集合算子（否则 DP>1 直接 ValueError）。"""
    calls: list = []
    group = _Group()
    _install(monkeypatch, size=1, group_rank0=2, calls=calls)

    next_token = torch.tensor([7, 8])
    out = _broadcast_and_pad_finished(
        next_token, torch.zeros(2, dtype=torch.bool), 99, tp_group=group
    )
    assert calls == [], "TP=1 时不得广播"
    assert out.tolist() == [7, 8]


def test_tp_size_above_one_broadcasts_from_group_rank0(monkeypatch):
    """TP>1：src = 组内 rank0 的**全局 rank**（不是写死的 0）。"""
    calls: list = []
    group = _Group()
    _install(monkeypatch, size=2, group_rank0=3, calls=calls)

    next_token = torch.tensor([7, 8])
    out = _broadcast_and_pad_finished(
        next_token, torch.zeros(2, dtype=torch.bool), 99, tp_group=group
    )
    assert calls == [(3, group)], calls
    assert out.tolist() == [0, 0], "广播后本 rank 的 token 来自源 rank"


def test_no_group_is_a_noop(monkeypatch):
    """tp_group=None（单卡 / PP 路径）⇒ 不广播，只做 finished 填充。"""
    calls: list = []
    _install(monkeypatch, size=1, group_rank0=0, calls=calls)
    out = _broadcast_and_pad_finished(
        torch.tensor([7, 8]), torch.tensor([True, False]), 99, tp_group=None
    )
    assert calls == []
    assert out.tolist() == [99, 8], "已结束的行填 pad，未结束的行保持原 token"


def test_not_initialized_is_a_noop(monkeypatch):
    calls: list = []
    _install(monkeypatch, size=2, group_rank0=0, calls=calls)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    out = _broadcast_and_pad_finished(
        torch.tensor([7]), torch.tensor([False]), 99, tp_group=_Group()
    )
    assert calls == []
    assert out.tolist() == [7]
