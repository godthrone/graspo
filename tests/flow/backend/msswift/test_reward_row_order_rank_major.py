"""E0/裁决修复机核：**键序 = 行序**（rank-major + 每 rank 本地 rollout 序），含块内。

## 缺陷（裁决结论 + 审查实测，均已是事实基础）

行序真值 = **rank-major + 每 rank 的本地 rollout 序**：

- 上游 ``score_completions(samples)`` 对**本地 samples 列表**逐条出一行（行 i ↔ ``samples[i]``）；
- ``accelerate.gather`` → ``all_gather_into_tensor`` ⇒ 全局行序 = ``rank0 行块 + rank1 行块 …``；
- 本地 ``samples`` 的顺序来自 ``get_even_process_data`` 的**连续切片**（rollout 生成序），
  逐样本 ``advantages`` 也是 `samples[i].advantages = local_advantages[i]`。

而修复前 ``_gather_sample_order`` 的 entries 走 ``_global_sample_order``
（``sorted(str(request_id))``），``request_id`` 是**逐样本 uuid4**
（上游 ``rollout_mixin.py`` ``chatcmpl-{uuid4().hex}``）⇒ "字典序名次"与"行序"是两个
独立随机置换，**块内**是随机置换（a1c30b2 只修好了 rank 块边界）；dp=1 走同一字典序路径。
绑定行→键的步骤是 ``trainer.py`` 的 ``keyed_rows_from_table(weighted, global_keys)``
（``alignment.py`` 纯 ``zip``，按位置）⇒ 键序一错，reward 就挂到**别的样本**上：
组统计量可能不变，但 **``best_idx`` / 逐样本归属**错，`content_scores` 逐样本时连
``classify_group`` 的 verdict 都会翻（``trainable_not_correct`` → ``retry``）。

## 本文件的口径（对块内残差有灵敏度）

1. **夹具现实**：request_id 是伪随机 uuid4 形状，**行序 ≠ 字典序**（夹具构造时断言）；
   组内奖励**非常数**（逐样本递增）；覆盖 **每 rank 1 组与 2 组**、world=1 与 world=2。
2. **行序 oracle 独立于被测代码**：行表由夹具的**插入序**（= rollout 序）直接给出，
   ⛔ 不再用 ``_global_sample_order`` 生成"每 rank 的行序"（旧夹具的循环论证）。
3. **逐样本正确性**：每个键拿到的 reward 必须等于它自己的行；``best_idx`` 与其
   ``content_scores`` 一起断言；``classify_group`` 的 verdict 与夹具真值一致。
4. **可证伪**：① "字典序映射"对照 ⇒ 每个夹具都断言它会产生错配/翻档（否则本文件无区分力）；
   ② 端到端把 entries 换回字典序 ⇒ ``AlignmentError``（契约校验红）；③ 直接对契约
   断言函数做红/绿两侧。

纯 CPU、零 ms-swift、零真实进程组（分布式缝用 monkeypatch 的 ``all_gather_object``，
语义 = ``out[i] = rank i 的 payload``）。
"""

from __future__ import annotations

import random
import types
from dataclasses import dataclass

import pytest
import torch

import graspo.flow.backend.msswift.trainer as trainer_module
from graspo.flow.adapter.contract import AlignmentError
from graspo.flow.backend.msswift.trainer import (
    _assert_key_order_matches_rows,
    _assert_paired_rows_match_source,
    _gather_sample_blocks,
    _gather_sample_order,
    _group_rewards,
    _local_sample_entries,
    _sample_key,
)
from graspo.ripple.group_decision import classify_group

GROUP_SIZE = 8
SEED = 20261103


def _uuid4_hex(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(32))


def _smp(request_id: str, prompt_id: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(request_id=request_id, prompt_id=prompt_id)


@dataclass(frozen=True)
class Group:
    """一个 rollout group：本地插入序（= rollout 序）、逐样本奖励、逐样本内容分。"""

    prompt_id: str
    samples: list[types.SimpleNamespace]
    rewards: list[float]
    content_scores: list[float]

    @property
    def keys(self) -> list[str]:
        return [_sample_key(sample) for sample in self.samples]

    @property
    def best_index(self) -> int:
        values = self.rewards
        return max(range(len(values)), key=lambda index: values[index])


def _make_group(rng: random.Random, prompt_id: str, base: float) -> Group:
    """造一个组：uuid4 形状的 id（**行序 ≠ 字典序**）、组内奖励**非常数**。

    ★ 刻意让本组**字典序最大**的那条落在本地第 0 位：这样"字典序键 ↔ 行序"的错配会把
    组内最高奖励发给第 0 条，而第 0 条恰好是**解析失败**（``content_scores=0``）的那条 ⇒
    ``best_completion_has_parse_error`` 翻转 ⇒ verdict 从 ``trainable_not_correct`` 翻成
    ``retry``。这不是人造怪相：uuid4 的字典序与 rollout 序本来就无关，且 E0 实测的
    组内奖励本身就是"大小差异"而不是常数。
    """
    ids = [_uuid4_hex(rng) for _ in range(GROUP_SIZE)]
    largest = max(ids)
    rng.shuffle(ids)
    # 把字典序最大者挪到本地第 0 位（uuid4 ⇒ 位置任意，属现实形状）
    ids.remove(largest)
    ids.insert(0, largest)
    samples = [_smp(f"chatcmpl-{hex_}", prompt_id) for hex_ in ids]
    # 组内奖励**非常数**（逐样本递增；组内最高分在本地最后一位 = content 1 的那条）
    rewards = [round(base + index * 0.05, 6) for index in range(GROUP_SIZE)]
    content_scores = [0.0] + [1.0] * (GROUP_SIZE - 1)
    group = Group(
        prompt_id=prompt_id, samples=samples, rewards=rewards, content_scores=content_scores
    )
    # ── 夹具自检（防止又造出"行序 == 字典序 / 组内常数"的零灵敏度夹具）──────────
    assert sorted(group.keys) != group.keys, "夹具退化：行序 == 字典序，对块内错配零灵敏度"
    assert len(set(rewards)) == GROUP_SIZE, "夹具退化：组内奖励是常数，对逐样本错配零灵敏度"
    return group


def _make_rank_groups(rng: random.Random, rank: int, groups_per_rank: int) -> list[Group]:
    return [
        _make_group(
            rng, f"prompt_{rank}_{index}", base=0.10 + 0.05 * (rank * groups_per_rank + index)
        )
        for index in range(groups_per_rank)
    ]


def _build(world: int, groups_per_rank: int) -> list[list[Group]]:
    """每个 rank 的组列表（组在 rank 内**连续**，与 ``get_even_process_data`` 连续切片一致）。"""
    rng = random.Random(SEED + world * 10 + groups_per_rank)
    return [_make_rank_groups(rng, rank, groups_per_rank) for rank in range(world)]


def _local_samples(rank_groups: list[Group]) -> list[types.SimpleNamespace]:
    return [sample for group in rank_groups for sample in group.samples]


def _truth(rank_groups: list[Group]) -> dict[str, float]:
    return {
        _sample_key(sample): reward
        for group in rank_groups
        for sample, reward in zip(group.samples, group.rewards, strict=True)
    }


def _content(rank_groups: list[Group]) -> dict[str, float]:
    return {
        _sample_key(sample): score
        for group in rank_groups
        for sample, score in zip(group.samples, group.content_scores, strict=True)
    }


def _tensor(values: list[float]):
    return torch.tensor([[value] for value in values], dtype=torch.float64)


def _row_order_rows(per_rank: list[list[Group]]) -> list[float]:
    """★行表：**rank-major + 每 rank 本地插入序**（夹具真值，与被测代码无关）。"""
    rows: list[float] = []
    for rank_groups in per_rank:
        truth = _truth(rank_groups)
        rows += [truth[_sample_key(sample)] for sample in _local_samples(rank_groups)]
    return rows


class _AllGatherObject:
    """``torch.distributed`` 的确定性等价物：``out[i] = rank i 的 payload``（真实语义）。

    非本 rank 的 payload 由夹具用**插入序**直接给出（独立于被测代码）；本 rank 的位置
    放被测代码实际传入的 payload —— 于是"被测代码是否按本地 rollout 序构造 entries"
    会被真实检验，而不是拿它自己的假设当 oracle。
    """

    def __init__(self, monkeypatch, per_rank: list[list[Group]], *, caller_rank: int) -> None:
        self._per_rank = per_rank
        self._caller_rank = caller_rank
        self.payloads: list[object] = []
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(per_rank))
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: caller_rank)
        monkeypatch.setattr(torch.distributed, "all_gather_object", self._gather)

    def expected_entries(self, rank: int) -> list[tuple[str, bool]]:
        return [(_sample_key(sample), True) for sample in _local_samples(self._per_rank[rank])]

    def _gather(self, out: list, payload) -> None:  # noqa: ANN001 - 与 dist 同签名
        self.payloads.append(list(payload))
        for rank in range(len(self._per_rank)):
            out[rank] = list(payload) if rank == self._caller_rank else self.expected_entries(rank)


def _install(monkeypatch, per_rank, *, caller_rank: int) -> _AllGatherObject:
    fake = _AllGatherObject(monkeypatch, per_rank, caller_rank=caller_rank)
    return fake


SHAPES = [
    pytest.param(1, 1, id="world1-gpr1"),
    pytest.param(1, 2, id="world1-gpr2"),
    pytest.param(2, 1, id="world2-gpr1"),
    pytest.param(2, 2, id="world2-gpr2"),
]


# ══ ① 键序 = 行序（world=1 与 world>1 同一条构造）════════════════════════════════


@pytest.mark.parametrize(("world", "groups_per_rank"), SHAPES)
@pytest.mark.parametrize("rank", [0, 1])
def test_gather_sample_order_is_local_rollout_order(monkeypatch, world, groups_per_rank, rank):
    """★键序逐位等于 "rank-major + 每 rank 本地 rollout 序"（⛔ 不是字典序）。

    world=1 与 world>1 **走同一条 entries 构造**（`_local_sample_entries`），所以本用例
    对两条路径同时生效；``rank`` 超过 world-1 时跳过（只为让 1 卡形状也跑 rank0）。
    """
    if rank >= world:
        pytest.skip("world 小于该 rank，无意义")
    per_rank = _build(world, groups_per_rank)
    fake = _install(monkeypatch, per_rank, caller_rank=rank)

    order = _gather_sample_order(_local_samples(per_rank[rank]))

    expected = [key for rank_groups in per_rank for key in _truth(rank_groups)]
    assert order == expected
    # 反向对照的钉死：字典序**不同**于行序（否则本用例对块内残差零灵敏度）
    assert order != sorted(order)
    if world > 1:
        # entries 是**本地 rollout 序**，不是本地字典序
        assert fake.payloads == [fake.expected_entries(rank)]


@pytest.mark.parametrize(("world", "groups_per_rank"), SHAPES)
def test_gather_sample_blocks_shape_and_local_order(monkeypatch, world, groups_per_rank):
    """★分块形状：``blocks[i]`` == rank i 的本地 rollout 序条目（world=1 ⇒ 单块）。"""
    per_rank = _build(world, groups_per_rank)
    _install(monkeypatch, per_rank, caller_rank=0)

    blocks = _gather_sample_blocks(_local_samples(per_rank[0]))

    assert len(blocks) == world
    for rank, block in enumerate(blocks):
        assert block == [(_sample_key(sample), True) for sample in _local_samples(per_rank[rank])]


def test_world_size_one_and_multi_share_the_same_local_order_construction(monkeypatch):
    """★同一处病、同一条修法：world≤1 不 gather，但 entries 与 world>1 **同一个函数**给出。"""
    per_rank = _build(2, 1)
    local = _local_samples(per_rank[0])

    single = _gather_sample_blocks(local)  # 未初始化进程组 ⇒ world=1
    assert single == [_local_sample_entries(local)]
    assert [key for key, _ in single[0]] == [_sample_key(s) for s in local]

    _install(monkeypatch, per_rank, caller_rank=0)
    multi = _gather_sample_blocks(local)
    assert multi[0] == single[0], "world≤1 与 world>1 的本地块必须逐位相同"


# ══ ② 逐样本正确性（reward 归属 / best_idx / verdict）══════════════════════════


@pytest.mark.parametrize(("world", "groups_per_rank"), SHAPES)
@pytest.mark.parametrize("rank", [0, 1])
def test_group_rewards_are_per_sample_correct(monkeypatch, world, groups_per_rank, rank):
    """★★每条键拿到的 reward 必须等于**它自己**的那一行（不只是组内多重集不变）。"""
    if rank >= world:
        pytest.skip("world 小于该 rank，无意义")
    per_rank = _build(world, groups_per_rank)
    _install(monkeypatch, per_rank, caller_rank=rank)
    local = _local_samples(per_rank[rank])

    got = _group_rewards(local, _tensor(_row_order_rows(per_rank)), None)

    truth = _truth(per_rank[rank])
    assert got == {_sample_key(sample): truth[_sample_key(sample)] for sample in local}
    # dict 键序也是可观测行为：必须是本地 rollout 序（⛔ 不是字典序）
    assert list(got) == [_sample_key(sample) for sample in local]
    assert list(got) != sorted(got)


@pytest.mark.parametrize(("world", "groups_per_rank"), SHAPES)
def test_best_index_and_group_verdict_follow_per_sample_rewards(
    monkeypatch, world, groups_per_rank
):
    """★★``best_idx`` / ``content_scores[best]`` / ``classify_group`` 逐样本一致。

    修复前（字典序键）：组内奖励是**同一个多重集**的置换 ⇒ 组统计量看不出问题，但
    ``best_idx`` 落到第 0 条（解析失败那条）⇒ ``best_completion_has_parse_error=True``
    ⇒ verdict 从 ``trainable_not_correct`` 翻成 ``retry``（优势被错误注入/错误拦下）。
    本用例把"真值 verdict"与"字典序映射的 verdict"都断言出来 ⇒ 双向可证伪。
    """
    per_rank = _build(world, groups_per_rank)
    _install(monkeypatch, per_rank, caller_rank=0)
    local = _local_samples(per_rank[0])

    got = _group_rewards(local, _tensor(_row_order_rows(per_rank)), None)
    content = _content(per_rank[0])

    for group in per_rank[0]:
        keys = group.keys
        values = [got[key] for key in keys]
        scores = [content[key] for key in keys]
        best = max(range(len(keys)), key=lambda index: values[index])
        assert values == group.rewards, "逐样本 reward 归属必须与行序一致"
        assert best == group.best_index
        decision = classify_group(
            values,
            scores,
            retry_count=0,
            rollout_max_retries=2,
            best_completion_has_parse_error=scores[best] == 0.0,
        )
        assert decision.decision.value == "trainable_not_correct"
        assert decision.should_train is True

        # ── 字典序映射（= 修复前的键序）的对照：必须**不同**（否则本用例无区分力）─────
        legacy = _legacy_mapping(keys, group.rewards)
        legacy_values = [legacy[key] for key in keys]
        legacy_best = max(range(len(keys)), key=lambda index: legacy_values[index])
        legacy_decision = classify_group(
            legacy_values,
            scores,
            retry_count=0,
            rollout_max_retries=2,
            best_completion_has_parse_error=scores[legacy_best] == 0.0,
        )
        assert legacy_values != values
        assert legacy_best != best
        assert legacy_decision.decision.value == "retry"
        assert legacy_decision.should_train is False


@pytest.mark.parametrize(("world", "groups_per_rank"), SHAPES)
def test_reward_weights_do_not_break_per_sample_attribution(monkeypatch, world, groups_per_rank):
    """★多奖励函数 + 权重：加权后逐样本归属仍必须正确（reduction 不影响键序契约）。"""
    per_rank = _build(world, groups_per_rank)
    _install(monkeypatch, per_rank, caller_rank=0)
    local = _local_samples(per_rank[0])
    base = _row_order_rows(per_rank)
    rows = [[value, value + 1.0] for value in base]
    weights = torch.tensor([2.0, 0.5], dtype=torch.float64)

    got = _group_rewards(local, torch.tensor(rows, dtype=torch.float64), weights)

    truth = _truth(per_rank[0])
    for sample in local:
        key = _sample_key(sample)
        assert got[key] == pytest.approx(2.0 * truth[key] + 0.5 * (truth[key] + 1.0))


# ══ ③ 可证伪：契约断言 + 端到端"字典序即红"══════════════════════════════════════


def _legacy_mapping(keys: list[str], rewards: list[float]) -> dict[str, float]:
    """**修复前**的键序算法（独立重写）：把键按字典序排好后与行**按位置**配对。"""
    ordered = sorted(keys)
    return {key: float(rewards[index]) for index, key in enumerate(ordered)}


def test_key_order_contract_assertion_is_green_for_rollout_order(monkeypatch):
    """契约断言的正向：键序 = 行序 ⇒ 不抛。"""
    per_rank = _build(2, 1)
    _install(monkeypatch, per_rank, caller_rank=0)
    local = _local_samples(per_rank[0])
    blocks = _gather_sample_blocks(local)
    global_keys = [key for block in blocks for key, _ in block]
    local_keys = [_sample_key(sample) for sample in local]

    _assert_key_order_matches_rows(global_keys, local_keys, blocks)
    table = {
        key: reward for key, reward in zip(global_keys, _row_order_rows(per_rank), strict=True)
    }
    _assert_paired_rows_match_source(_row_order_rows(per_rank), table, local_keys, blocks)


def test_key_order_contract_assertion_is_red_for_lexicographic_order(monkeypatch):
    """★契约断言的反向：把键序换回字典序 ⇒ **必须红**（证明它不是恒真式）。"""
    per_rank = _build(2, 1)
    _install(monkeypatch, per_rank, caller_rank=0)
    local = _local_samples(per_rank[0])
    blocks = _gather_sample_blocks(local)
    lexicographic = sorted(key for block in blocks for key, _ in block)

    with pytest.raises(AlignmentError, match="键序"):
        _assert_key_order_matches_rows(lexicographic, [_sample_key(s) for s in local], blocks)


def test_paired_rows_assertion_is_red_for_mispaired_table(monkeypatch):
    """★配对后复核的反向：表把行配错（模拟错位绑定）⇒ **必须红**。"""
    per_rank = _build(2, 1)
    _install(monkeypatch, per_rank, caller_rank=0)
    local = _local_samples(per_rank[0])
    blocks = _gather_sample_blocks(local)
    global_keys = [key for block in blocks for key, _ in block]
    rows = _row_order_rows(per_rank)
    local_keys = [_sample_key(sample) for sample in local]
    truth = _truth(per_rank[0])
    # 排序后的表：键 → 行按字典序位置配对（= 修复前的绑定方式）
    sorted_keys = sorted(global_keys)
    mispaired = {key: rows[index] for index, key in enumerate(sorted_keys)}

    with pytest.raises(AlignmentError, match="配对后校验失败"):
        _assert_paired_rows_match_source(rows, mispaired, local_keys, blocks)
    assert mispaired != {key: truth[key] for key in local_keys}


@pytest.mark.parametrize(("world", "groups_per_rank"), SHAPES)
def test_lexicographic_entries_fail_closed_end_to_end(monkeypatch, world, groups_per_rank):
    """★端到端红：把 entries 换回字典序（= 修复前的 ``_global_sample_order``）⇒ fail-closed。

    这是"断言能失败"的端到端证据：修复前的路径在本用例里**不是**产出错值，而是被
    契约断言拦下（``AlignmentError``）——静默错配不再可能。
    """
    per_rank = _build(world, groups_per_rank)
    _install(monkeypatch, per_rank, caller_rank=0)
    local = _local_samples(per_rank[0])

    def legacy_entries(samples):
        ordered = sorted(samples, key=lambda sample: str(sample.request_id))
        return [(_sample_key(sample), True) for sample in ordered]

    monkeypatch.setattr(trainer_module, "_local_sample_entries", legacy_entries)

    with pytest.raises(AlignmentError, match="键序来源不一致"):
        _group_rewards(local, _tensor(_row_order_rows(per_rank)), None)
