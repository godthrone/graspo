"""E0 修复机核：**dp≥2 的组奖励键序必须与 ms-swift 行序同源（rank-major，⛔ 不是字典序）**。

## 缺陷（直证见 ``task-e0-directproof/E0-DIRECTPROOF.md``）

``trainer._gather_sample_order`` 末尾原为 ``return sorted(seen)``：把**整个 batch** 的样本键
（dp=2 时两组共 16 条）按**字典序**重排；``alignment.keyed_rows_from_table`` 再按**位置**
把行配给键（只校验长度/重复键）。而 ms-swift 的 ``rewards_per_func`` 行序是
``all_gather`` 的 **rank 序拼接**（rank0 的行在前、rank1 的在后）⇒ 键在字典序里的
**名次**决定它拿到哪个 rank 的行 ⇒ dp≥2 时某个键拿到**另一个组**的 reward。

后果：真实组内零方差被制造成**假组内方差** ⇒ ``classify_group`` 由 ``retry`` 翻成
``trainable_not_correct`` ⇒ 错配的 advantage 被注入。

## 本文件的口径

纯 CPU、零 ms-swift、零进程组：分布式缝用 monkeypatch 的假 ``all_gather_object``，
其语义与 ``_all_gather_sample_entries`` 一致——**按 rank 序拼接**，且各 rank 的行序
= ``_global_sample_order``（按 request_id 升序，即被测代码自己的前提）。

三条防线：
1. dp=2 + 两组各 8 条、组内奖励**常不同**（0.385714 / 0.719048）⇒ 改前必须红（跨组），
   改后必须绿（0 跨组 / std=0 / retry）。
2. 反向对照：把键序**人为换回字典序**（= 修复前的 ``sorted(seen)``）⇒ 同一批数据立刻跨组
   （证明第 1 条的断言**不是恒真**，能失败）。
3. 回归：dp=1 逐字不变（只做组内置换）；k=1/k=2 且 gather 序恰好 == 字典序时行为与
   修复前的"行号基准"逐字一致。
"""

from __future__ import annotations

import types

import pytest
import torch

import graspo.flow.backend.msswift.trainer as trainer_module
from graspo.flow.backend.msswift.trainer import (
    _gather_sample_order,
    _global_sample_order,
    _group_rewards,
    _sample_key,
)
from graspo.ripple.group_decision import classify_group

# 组内奖励**各自为常数**（真实日志里两组各是一个常数：E0 A-2 的 0.385714 / 1.004762）。
GROUP_REWARD = {0: 0.385714, 1: 0.719048}

# 两组的 request_id 在**字典序上刻意交错**（真实 uuid4 就是随机的；E0 A-2 的实测形状）。
HEX_GROUP0 = [
    "0a11f0c4d9e2418e9b0f6f2a1c3d5e70",
    "1c8b3d2e5f6041a79c8d1e0b2a4f6c81",
    "3e5f7a9b1c2d43058e6f7a8b9c0d1e24",
    "4f6a8b0c1d2e43519f7a8b9c0d1e2f35",
    "5a7b9c0d1e2f43720a8b9c0d1e2f3046",
    "6b8c0d1e2f3049391b9c0d1e2f304157",
    "7c9d1e2f3041045a2c0d1e2f30415268",
    "8dae2f304152036b3d1e2f3041526379",
]
HEX_GROUP1 = [
    "0b22e1d5eaf352909c1a7a3b2d4e6f81",
    "2d9c4e3f6a7152b8ad9e2f1c3b5a7d92",
    "3f6a8b9c0d1e4521a9f7b8c9d0e1f236",
    "5b8c9d0e1f2a45330b9c0d1e2f304257",
    "6c9d0e1f2a3045441c0d1e2f30415269",
    "7dae1f2a304105552d1e2f3041526370",
    "9ebf2f30415274663e1f2f3041526378",
    "afc030415274677f401f304152637480",
]


def _smp(request_id: str, prompt_id: str = "p0") -> types.SimpleNamespace:
    return types.SimpleNamespace(request_id=request_id, prompt_id=prompt_id)


def _group0_samples() -> list[types.SimpleNamespace]:
    return [_smp(f"chatcmpl-{hex_}", "prompt_0") for hex_ in HEX_GROUP0]


def _group1_samples() -> list[types.SimpleNamespace]:
    return [_smp(f"chatcmpl-{hex_}", "prompt_1") for hex_ in HEX_GROUP1]


def _tensor(rows: list[float]):
    """``[N]`` → ``[N, 1]`` 的 ``rewards_per_func``（权重 None ⇒ 单列 nansum 即该值）。"""
    return torch.tensor([[value] for value in rows], dtype=torch.float64)


class _RankMajorGather:
    """把 ``torch.distributed`` 换成"按 **rank 序** 拼接各 rank 行块"的确定性等价物。

    ``per_rank[i]`` = rank *i* 本地的样本列表；每个 rank 的行序由被测代码自己的前提给出
    （``_global_sample_order``：按 request_id 升序）。``caller_rank`` 用来复核被测代码
    传进 ``all_gather_object`` 的 payload 确实是"本 rank 本地排序后的键"。
    """

    def __init__(self, monkeypatch, per_rank: list[list[types.SimpleNamespace]], *, caller_rank: int):
        self._per_rank = per_rank
        self._caller_rank = caller_rank
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(per_rank))
        monkeypatch.setattr(torch.distributed, "all_gather_object", self._gather)

    def _entries(self, rank: int) -> list[tuple[str, bool]]:
        return [(_sample_key(sample), True) for sample in _global_sample_order(self._per_rank[rank])]

    def _gather(self, out: list, payload) -> None:  # noqa: ANN001 - 与 dist 同签名
        assert list(payload) == self._entries(self._caller_rank), "payload 必须是本 rank 本地行序"
        for rank in range(len(self._per_rank)):
            out[rank] = self._entries(rank)  # ★ rank 序：out[0] 永远是 rank0 的行块


def _rank_major_rows(per_rank: list[list[types.SimpleNamespace]], own: dict[str, float]) -> list[float]:
    """行表 ``rewards_per_func`` 的加权行：**rank-major**，rank 内按本地行序。"""
    rows: list[float] = []
    for samples in per_rank:
        rows += [own[_sample_key(sample)] for sample in _global_sample_order(samples)]
    return rows


def _legacy_mapping(samples, weighted) -> dict[str, float]:
    """**修复前** dp=1 / 无错位场景的算法（独立重写）：本地排序 → 按行号配对。"""
    ordered = sorted(samples, key=lambda sample: str(sample.request_id))
    return {_sample_key(sample): float(weighted[index]) for index, sample in enumerate(ordered)}


def _values(got: dict[str, float], samples) -> list[float]:
    return [got[_sample_key(sample)] for sample in samples]


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _decide(values: list[float]):
    return classify_group(values, [1.0] * len(values), retry_count=0, rollout_max_retries=2)


# ══ ① dp=2：改前红（跨组）/ 改后绿（0 跨组、std=0、retry）════════════════════


def test_gather_sample_order_is_rank_major_not_lexicographic(monkeypatch):
    """★直接钉死键序：gather 拼出来的是 ``rank0 行块 + rank1 行块``，⛔ 不是全局字典序。"""
    per_rank = [_group0_samples(), _group1_samples()]
    _RankMajorGather(monkeypatch, per_rank, caller_rank=0)

    order = _gather_sample_order(per_rank[0])

    expected = [_sample_key(sample) for samples in per_rank for sample in _global_sample_order(samples)]
    assert order == expected
    # 反向对照的钉死：字典序**不同**于行序（否则本用例没有区分力）
    assert order != sorted(order)


@pytest.mark.parametrize("rank", [0, 1])
def test_dp2_group_rewards_never_cross_group(monkeypatch, rank):
    """★★dp=2 主用例：两组各 8 条、组内奖励常数不同；每条键必须拿到**本组**的 reward。

    改前（``sorted(seen)``）：组内出现另一组的常数 ⇒ 第 1 条断言红；
    改后（rank-major）：0 跨组、std=0、``classify_group`` 回落 ``retry``（不注入优势）。
    """
    per_rank = [_group0_samples(), _group1_samples()]
    own = {_sample_key(sample): GROUP_REWARD[index] for index, samples in enumerate(per_rank) for sample in samples}
    _RankMajorGather(monkeypatch, per_rank, caller_rank=rank)

    got = _group_rewards(per_rank[rank], _tensor(_rank_major_rows(per_rank, own)), None)

    # ① 0 跨组：每个键拿到的是**它自己**那组的常数
    assert got == {_sample_key(sample): GROUP_REWARD[rank] for sample in per_rank[rank]}
    values = _values(got, per_rank[rank])
    # ② 真实组内零方差没有被制造成假方差
    assert _std(values) == 0.0
    # ③ 下游：不翻档、不注入
    decision = _decide(values)
    assert decision.decision.value == "retry"
    assert decision.should_train is False


def test_reverse_control_lexicographic_key_order_still_mismatches(monkeypatch):
    """★反向对照：把键序**人为换回字典序**（= 修复前的 ``sorted(seen)``）⇒ 同一批数据立刻跨组。

    这条保证 ① 的断言**能失败**：只要键序与行序不同源，它就会红（⛔ 不是恒真式）。
    """
    per_rank = [_group0_samples(), _group1_samples()]
    own = {
        _sample_key(sample): GROUP_REWARD[index]
        for index, samples in enumerate(per_rank)
        for sample in samples
    }
    _RankMajorGather(monkeypatch, per_rank, caller_rank=0)
    all_keys = sorted(_sample_key(sample) for samples in per_rank for sample in samples)
    monkeypatch.setattr(trainer_module, "_gather_sample_order", lambda samples: list(all_keys))

    rows = _tensor(_rank_major_rows(per_rank, own))
    got = {rank: _group_rewards(per_rank[rank], rows, None) for rank in (0, 1)}

    crossed = {
        rank: [key for key in got[rank] if abs(got[rank][key] - GROUP_REWARD[rank]) > 1e-9]
        for rank in (0, 1)
    }
    # ① 键序被换成字典序 ⇒ 两个 rank 都出现跨组取值（键的名次决定它拿哪一行）
    assert crossed[0] and crossed[1]
    # ② 假方差被制造出来
    for rank in (0, 1):
        assert _std(_values(got[rank], per_rank[rank])) > 1e-6
    # ③ 下游至少一侧翻成 trainable（优势可被注入）：lower_median 取向决定只翻一侧
    assert any(_decide(_values(got[rank], per_rank[rank])).should_train for rank in (0, 1))


# ══ ② dp=1：行为**逐字不变**（只有组内置换，多重集不变）════════════════════════


@pytest.mark.parametrize("size", [1, 2, 6, 8])
def test_dp1_is_byte_identical_to_the_pre_fix_mapping(size):
    """★dp=1 不加 gather、不排序新逻辑：返回值与修复前算法**逐字相等**（含 dict 键序）。"""
    samples = [_smp(f"chatcmpl-{index:04d}") for index in range(size)]
    shuffled = list(reversed(samples))
    weighted = [index * 0.125 for index in range(size)]

    got = _group_rewards(shuffled, _tensor(weighted), None)

    assert got == _legacy_mapping(shuffled, weighted)
    assert list(got) == [_sample_key(sample) for sample in sorted(shuffled, key=_sample_key)]


def test_dp1_repeated_group_reward_is_only_a_permutation(monkeypatch):
    """★dp=1 的两组各自常数：组内多重集 = 本组自身，std=0，``classify_group`` 不翻档。"""
    per_rank = [_group0_samples(), _group1_samples()]
    own = {_sample_key(sample): GROUP_REWARD[index] for index, samples in enumerate(per_rank) for sample in samples}
    for rank in (0, 1):
        rows = _tensor([own[_sample_key(sample)] for sample in _global_sample_order(per_rank[rank])])
        got = _group_rewards(per_rank[rank], rows, None)
        values = _values(got, per_rank[rank])
        assert sorted(values) == [GROUP_REWARD[rank]] * 8
        assert _std(values) == 0.0
        assert _decide(values).decision.value == "retry"


# ══ ③ 合法情形（k=1 / k=2、gather 序 == 字典序）：行为与修复前逐字一致 ═══════════


@pytest.mark.parametrize("k", [1, 2])
def test_k_groups_per_rank_without_misalignment_is_unchanged(monkeypatch, k):
    """★k=1 / k=2：每 rank **k 个整组**，且各 rank 的键块在字典序上前后分离
    （于是 gather 序 == 全局字典序）⇒ 修复前后逐字相同，且每组取回本组常数。
    """
    per_rank: list[list[types.SimpleNamespace]] = []
    own: dict[str, float] = {}
    cursor = 0
    for rank in range(2):
        samples = []
        for group_in_rank in range(k):
            group_index = rank * k + group_in_rank
            reward = 0.1 * (group_index + 1)
            for _ in range(8):
                sample = _smp(f"chatcmpl-{cursor:04d}", f"prompt_{group_index}")
                cursor += 1
                samples.append(sample)
                own[_sample_key(sample)] = reward
        per_rank.append(samples)
    _RankMajorGather(monkeypatch, per_rank, caller_rank=0)

    rows = _tensor(_rank_major_rows(per_rank, own))
    got = _group_rewards(per_rank[0], rows, None)

    # 修复前后的"行号基准"逐字一致（键序恰好相同，所以 sorted(seen) 无害）
    assert got == _legacy_mapping(per_rank[0], rows)
    assert list(got) == sorted(got)
    # 每个键仍拿到本组常数
    assert got == {_sample_key(sample): own[_sample_key(sample)] for sample in per_rank[0]}
