"""T032 多卡奖励对齐缺陷的回归机核 —— ``_group_rewards`` 的**全局键序**（契约 I2）。

## 缺陷（探针已定案，见 ``task-t032-reward-probe/REPORT.md``）

``rewards_per_func`` 是 ``[N_global, …]``（上游 ``_compute_rewards_per_func`` gather 过），
而 ``_compute_advantages`` 拿到的 ``samples`` 只是**本进程切片**。旧实现只对本地列表
排序，于是：

- **单卡**：本地 == 全局 ⇒ 8 == 8 通过（T031 能跑通的原因）；
- **多卡**：全局 **16** 行 vs 本地 **8** 条 ⇒ 抛 I2（实测 T032 两 rank 是互补切片：
  ``|rank0|=|rank1|=8``、交集 ∅、并集 16 = 表行数）。

修法：新增 :func:`_gather_sample_order`，用 ``all_gather_object`` 把各 rank 的键收齐后
按同一键序排序 ⇒ **"比长度"的两侧都在全局作用域**。⛔ 不恢复按行号配对。

本文件只测**纯函数 + 注入的分布式缝**（CPU、零 torch 进程组、零 ms-swift）。
"""

from __future__ import annotations

import types

import pytest
import torch

import graspo.flow.backend.msswift.trainer as trainer
from graspo.flow.adapter.contract import AlignmentError
from graspo.flow.backend.msswift.trainer import (
    _gather_sample_order,
    _global_sample_order,
    _group_rewards,
    _sample_key,
)


def _smp(request_id: str, **extra) -> types.SimpleNamespace:
    return types.SimpleNamespace(request_id=request_id, **extra)


def _legacy_single_card_mapping(samples, weighted) -> dict[str, float]:
    """**修复前**的单卡算法（独立重写，用作"逐字不变"的对照基准）。

    旧实现：本地排序 → 与表行**按行号**一一对应。
    """
    ordered = sorted(samples, key=lambda sample: str(sample.request_id))
    return {_sample_key(sample): float(weighted[index]) for index, sample in enumerate(ordered)}


class _FakeDist:
    """把 ``torch.distributed`` 的三个入口换成"每 rank 一份键"的假实现。

    ``other_ranks`` = **其它 rank** 的 payload 列表（不含本 rank）。真实
    ``all_gather_object`` 的语义是"本 rank 的 payload 也进结果"，所以本 fake 把被测代码
    传入的 payload 放到 ``out[0]``、其余按 ``other_ranks`` 顺序放到 ``out[1:]``。
    世界大小 = ``1 + len(other_ranks)``。
    """

    def __init__(self, monkeypatch, other_ranks: list[list[tuple[str, bool]]]) -> None:
        self._others = other_ranks
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1 + len(other_ranks))
        monkeypatch.setattr(torch.distributed, "all_gather_object", self._gather)

    def _gather(self, out: list, payload) -> None:
        out[0] = list(payload)
        for index, part in enumerate(self._others, start=1):
            out[index] = list(part)


# ══ ① 单卡：修复前后**逐字不变** ═════════════════════════════════════════════


def test_single_card_is_byte_identical_to_the_pre_fix_mapping():
    """★单卡逐字不变：新实现的返回值与**修复前算法**在多个样本数/乱序输入上完全相等。"""
    for size in (1, 2, 6):
        samples = [_smp(f"chatcmpl-{i:04d}") for i in range(size)]
        shuffled = list(reversed(samples))
        weighted = [index * 0.5 for index in range(size)]
        expected = _legacy_single_card_mapping(shuffled, weighted)

        got = _group_rewards(shuffled, torch.tensor(weighted).reshape(size, 1), None)

        assert got == expected, f"size={size}"
        assert list(got) == [_sample_key(s) for s in sorted(shuffled, key=_sample_key)]


def test_single_card_key_order_matches_the_old_helper():
    """★``_gather_sample_order`` 在单卡下 == ``[_sample_key(s) for s in _global_sample_order(...)]``。"""
    samples = [_smp("c"), _smp("a"), _smp("b")]
    assert _gather_sample_order(samples) == [_sample_key(s) for s in _global_sample_order(samples)]
    assert _gather_sample_order(samples) == ["a", "b", "c"]


def test_single_card_without_request_id_keeps_original_order():
    """★缺 ``request_id`` 时旧语义是"保持原序"——单卡必须逐字保留（不排序、不报错）。"""
    samples = [types.SimpleNamespace(prompt_id="p2"), types.SimpleNamespace(prompt_id="p1")]
    order = _gather_sample_order(samples)
    assert order == [_sample_key(sample) for sample in samples] == ["prompt:p2", "prompt:p1"]

    weighted = torch.tensor([[1.0], [2.0]])
    assert _group_rewards(samples, weighted, None) == {"prompt:p2": 1.0, "prompt:p1": 2.0}


# ══ ② 多卡（2 rank）：对齐通过 + 每个 rank 只取自己的键 ═══════════════════════


def test_multi_rank_each_rank_reads_only_its_own_keys(monkeypatch):
    """★T032 复现：全局表 6 行 / rank0 本地 3 条（互补切片）⇒ 通过且只取自己的键。"""
    rank0 = [_smp(f"chatcmpl-{i:02d}") for i in range(3)]  # 00,01,02
    rank1_keys = [f"chatcmpl-{i:02d}" for i in range(3, 6)]  # 03,04,05
    _FakeDist(monkeypatch, [[(key, True) for key in rank1_keys]])
    weighted = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    got = _group_rewards(rank0, weighted, None)

    assert set(got) == {"chatcmpl-00", "chatcmpl-01", "chatcmpl-02"}
    # ★ pairing 正确性：值取的是**全局表里对应行**（按键），不是行号前 3 个以外的任何错位
    assert got == {"chatcmpl-00": 0.0, "chatcmpl-01": 1.0, "chatcmpl-02": 2.0}


def test_multi_rank_key_based_pairing_survives_shuffled_table(monkeypatch):
    """★按键而非按行号：把本地顺序与全局键序都打乱，值仍必须跟着键走。"""
    rank0 = [_smp("chatcmpl-c"), _smp("chatcmpl-a")]
    rank1_keys = ["chatcmpl-b", "chatcmpl-d"]
    _FakeDist(monkeypatch, [[(key, True) for key in rank1_keys]])
    # 全局键序（升序）= a,b,c,d ⇒ 值 10,20,30,40
    weighted = torch.tensor([[10.0], [20.0], [30.0], [40.0]])

    got = _group_rewards(rank0, weighted, None)

    assert got == {"chatcmpl-c": 30.0, "chatcmpl-a": 10.0}


def test_multi_rank_rank1_also_reads_only_its_own_keys(monkeypatch):
    """对称性：rank1 视角同样只取自己的键（两 rank 合起来正好覆盖全局表）。"""
    rank1 = [_smp(f"chatcmpl-{i:02d}") for i in range(3, 6)]
    rank0_keys = [f"chatcmpl-{i:02d}" for i in range(3)]
    _FakeDist(monkeypatch, [[(key, True) for key in rank0_keys]])
    weighted = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    got = _group_rewards(rank1, weighted, None)

    assert got == {"chatcmpl-03": 3.0, "chatcmpl-04": 4.0, "chatcmpl-05": 5.0}


# ══ ③ 不一致场景：仍 fail-closed ═══════════════════════════════════════════


def test_length_mismatch_still_fails_closed(monkeypatch):
    """★全局键数 ≠ 表行数 ⇒ ``AlignmentError``（⛔ 不得回退按行号配对）。"""
    rank0 = [_smp("chatcmpl-a"), _smp("chatcmpl-b")]
    _FakeDist(monkeypatch, [[]])
    weighted = torch.arange(5, dtype=torch.float32).reshape(5, 1)  # 5 行 vs 2 键

    with pytest.raises(AlignmentError, match="未按键对齐"):
        _group_rewards(rank0, weighted, None)


def test_cross_rank_key_collision_fails_closed_even_when_lengths_match(monkeypatch):
    """★长度**对得上**但键跨 rank 撞车（rank0=[a,b]、rank1=[a,c]、表 4 行）⇒ 仍必须失败。

    这是与"长度不等"**独立**的一条防线：长度相等**不等于**键一一对应。
    """
    rank0 = [_smp("chatcmpl-a"), _smp("chatcmpl-b")]
    _FakeDist(monkeypatch, [[("chatcmpl-a", True), ("chatcmpl-c", True)]])
    weighted = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 4 行 == 4 个扁平键

    with pytest.raises(AlignmentError, match="跨 rank 样本键重复"):
        _group_rewards(rank0, weighted, None)


def test_duplicate_local_keys_fail_closed(monkeypatch):
    """★本地键重复 ⇒ ``AlignmentError``（键必须一一对应，旧口径同样禁止）。"""
    rank0 = [_smp("chatcmpl-a"), _smp("chatcmpl-a")]
    _FakeDist(monkeypatch, [[]])
    weighted = torch.tensor([[1.0]])

    with pytest.raises(AlignmentError):
        _group_rewards(rank0, weighted, None)


@pytest.mark.parametrize("world", [2, 4])
def test_missing_request_id_across_ranks_fails_closed(monkeypatch, world):
    """★跨 rank 采样缺 ``request_id`` ⇒ 无法确定全局键序 ⇒ fail-closed（不猜）。"""
    rank0 = [types.SimpleNamespace(prompt_id="p1")]
    _FakeDist(monkeypatch, [[] for _ in range(world - 1)])

    with pytest.raises(AlignmentError, match="缺少 request_id"):
        _gather_sample_order(rank0)


# ══ ④ 上游复制场景（交集 = 全集）：保持 fail-closed ═════════════════════════


def test_upstream_copy_scenario_is_still_fail_closed(monkeypatch):
    """★★上游"两 rank 同键"（复制而非切分）⇒ **必须失败**，⛔ 不得静默放行。

    探针已实测本仓是**互补切片**；此分支是为"上游改成按副本分发"准备的防呆：
    那种情形下全局表行数（16）与去重后的样本数（8）无法一一对应，静默放行会产出
    按行号错配的奖励 ⇒ 显式 ``AlignmentError``。
    """
    rank0 = [_smp(f"chatcmpl-{i:02d}") for i in range(3)]
    entries = [(_sample_key(s), True) for s in rank0]
    _FakeDist(monkeypatch, [entries])  # 其它 rank == 本 rank（同键）
    weighted = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    with pytest.raises(AlignmentError, match="跨 rank 样本键重复"):
        _group_rewards(rank0, weighted, None)


def test_upstream_copy_scenario_never_silently_passes(monkeypatch):
    """★反向对照：确认上面那条不是"恰好因为行数不匹配"才失败——键重复**独立**触发。"""
    rank0 = [_smp(f"chatcmpl-{i:02d}") for i in range(3)]
    entries = [(_sample_key(s), True) for s in rank0]
    _FakeDist(monkeypatch, [entries])

    with pytest.raises(AlignmentError, match="跨 rank 样本键重复"):
        _gather_sample_order(rank0)


# ══ 缝的自证：真实 ``_all_gather_sample_entries`` 走 ``all_gather_object`` ═══


def test_all_gather_entries_calls_the_real_distributed_entry(monkeypatch):
    """★自证：``_all_gather_sample_entries`` 确实调 ``torch.distributed.all_gather_object``。"""
    calls: list[object] = []

    def fake_all_gather(out: list, payload) -> None:  # noqa: ANN001
        calls.append(payload)
        out[0] = list(payload)
        out[1] = [("from-rank1", True)]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)

    out = trainer._all_gather_sample_entries([("local", True)])

    assert calls == [[("local", True)]]
    assert out == [("local", True), ("from-rank1", True)]


def test_world_size_is_one_without_a_process_group():
    """未初始化进程组 ⇒ ``world_size == 1``（单卡语义；不新增失败面）。"""
    assert trainer._process_group_world_size() == 1
