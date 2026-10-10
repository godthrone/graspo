"""flow.backend.native.assignment — ★ β1b **整组均衡分配**（替代 R5 的设施层越界）。

## 为什么需要它（D3 裁定 + P4 口径）

R5（2026-10-08 在役本地补丁）在设施层把每步训练集缩到 **dp 组裸 MIN**、余量延后、
终端清场 ⇒ **改了每步样本集合与步数** ⇒ 违反设施层契约 ⇒ **P4 退役**。
替代机制 = **β1b 整组均衡分配**：分配发生在 **Ripple 调用边界之外**，组**始终完整**
交给 Ripple（宪法 §1.1 / I1），且**不新增跨卡会合**（会合只在既有批次边界）。

## 三条不变量（与 `flow/adapter/contract.py::assert_group_partition` 同源，§1.4）

1. **整组不拆**：分配的单位是**组**，不存在"半个组"落在一个分片上；
2. **全局组集合不变**：分片后各组恰好出现一次（不丢、不重、不新增）；
3. **按工作量均衡**：各分片组数差 ≤ 1（默认模式）；给权重时按工作量做 LPT 贪心。

## 边界

- 本模块**纯函数**：零 IO / 零 torch / 零通信（可在 CPU 独立机核，§1.3）；
- **不做**生成、评分、算法判定（那是 backend / ripple 的活）；
- 跨 rank 的**搬组执行**（真正把整组从一个 rank 挪到另一个 rank 的那次会合）属
  **待跑次阶段**（`native.group_balance_max_move_groups` 默认 0 = 不搬 ⇒ 无新增通信）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

K = TypeVar("K")


class AssignmentError(RuntimeError):
    """分配不满足 β1b 不变量（整组不拆 / 组集合不变 / 均衡）。"""


def assign_whole_groups(
    keys: Sequence[K],
    n_shards: int,
    *,
    weights: Mapping[K, float] | None = None,
) -> dict[K, int]:
    """把**整组**键分配到 ``n_shards`` 个分片（确定性）。

    :param keys: 全局组键（顺序即"确定性种子"）。
    :param n_shards: 分片数（= 参与本次优化的 rank 数）。
    :param weights: 可选工作量（如组内条数 / token 数）；给出时按 **LPT 贪心**
        （重者先放、放当前最轻的分片）做**工作量均衡**；不给时按**组数均衡**。

    Returns:
        ``{key: shard_index}`` —— 每个组恰好出现一次。

    Raises:
        AssignmentError: 组键重复 / ``n_shards < 1``。

    ★ 不变量（:func:`assert_balanced_assignment` 可机核）：整组不拆 + 组集合不变；
    默认模式额外保证**各分片组数差 ≤ 1**。
    """
    if n_shards < 1:
        raise AssignmentError(f"n_shards 必须 ≥ 1，得到 {n_shards!r}")
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise AssignmentError("组键列表里有重复 —— 全局组集合不是集合语义。")

    order = list(keys)
    if weights is not None:
        missing = [k for k in keys if k not in weights]
        if missing:
            raise AssignmentError(f"weights 未覆盖全部组键：缺 {missing[:3]}")
        order.sort(key=lambda k: (-float(weights[k]), keys.index(k)))

    loads = [0.0] * n_shards
    counts = [0] * n_shards
    out: dict[K, int] = {}
    for index, key in enumerate(order):
        # 默认模式：按组数取当前最轻分片（并列取分片号小者 ⇒ 确定性）；
        # 带权重模式：按工作量取当前最轻分片。
        metric = (lambda s: loads[s]) if weights is not None else (lambda s: float(counts[s]))
        shard = min(range(n_shards), key=lambda s: (metric(s), s))
        out[key] = shard
        counts[shard] += 1
        loads[shard] += float(weights[key]) if weights is not None else 1.0
        del index
    return out


def whole_group_count(experience_count: int, group_size: int) -> int:
    """把**条数**向下取整到**整组**倍数（β1b 的步边界原语）。

    ★ 这是 R5 ``_align_dp_usable``（**条数**级 dp 组裸 MIN 对齐）的整组替代：
    配额永远是组的整数倍 ⇒ Ripple 只见**完整组**（I1）；被取整掉的是"不足一组的尾巴"，
    它**不**参与本步训练（R5 的做法是跨 rank 归约到 MIN 并 defer，那是设施层越界）。

    :param experience_count: 本 rank 当前可训**条数**（≤0 视为 0）。
    :param group_size: 组大小 G（≤1 视为 1，即不取整）。

    Raises:
        AssignmentError: ``group_size < 1``。
    """
    if group_size < 1:
        raise AssignmentError(f"group_size 必须 ≥ 1，得到 {group_size!r}")
    if experience_count <= 0:
        return 0
    return (int(experience_count) // int(group_size)) * int(group_size)


def assert_balanced_assignment(
    keys: Sequence[K], assignment: Mapping[K, int], *, n_shards: int, require_count_balance: bool = True
) -> None:
    """机核 β1b 不变量（覆盖 / 不重 / 分片越界 / 组数差 ≤ 1）。

    Raises:
        AssignmentError: 任一不变量被违反。
    """
    if n_shards < 1:
        raise AssignmentError(f"n_shards 必须 ≥ 1，得到 {n_shards!r}")
    keys = list(keys)
    got = list(assignment.keys())
    if len(set(got)) != len(got):
        raise AssignmentError("assignment 里有重复键。")
    if set(got) != set(keys):
        raise AssignmentError(
            f"组集合变了：缺 {sorted(map(str, set(keys) - set(got)))[:3]}、"
            f"多 {sorted(map(str, set(got) - set(keys)))[:3]}（β1b：不丢/不重/不新增）"
        )
    counts = [0] * n_shards
    for key in keys:
        shard = assignment[key]
        if not isinstance(shard, int) or not 0 <= shard < n_shards:
            raise AssignmentError(f"分片号越界：{key!r} → {shard!r} ∉ [0, {n_shards})")
        counts[shard] += 1
    if require_count_balance and counts and (max(counts) - min(counts)) > 1:
        raise AssignmentError(f"各分片组数失衡：{counts}（β1b 要求差 ≤ 1 组）")


def step_group_quota(per_shard_groups: Sequence[int], *, move_budget: int = 0) -> int:
    """步边界：全局**整组配额** = ``min(各分片组数)``（+ 可选搬组预算）。

    这是 R5 ``_align_dp_usable``（**条数**级 MIN）的整组替代：
    配额永远是**组的整数倍** ⇒ Ripple 只见完整组；且**不新增任何归约**
    （``per_shard_groups`` 由调用方在既有批次边界上取得）。

    :param per_shard_groups: 各分片当前**整组**数。
    :param move_budget: ``native.group_balance_max_move_groups``（默认 0 = 不搬）。
        ★ 搬组执行属**待跑次阶段**（本次只做规划，不新增通信）。

    Raises:
        AssignmentError: 空序列 / 负数。
    """
    if not per_shard_groups:
        raise AssignmentError("per_shard_groups 不能为空")
    if any(g < 0 for g in per_shard_groups):
        raise AssignmentError(f"组数不能为负：{list(per_shard_groups)}")
    if move_budget < 0:
        raise AssignmentError(f"move_budget 不能为负：{move_budget}")
    base = min(per_shard_groups)
    # 搬组可把"多的分片"里的整组挪给"少的分片"，抬升配额；预算为 0 时保持不变。
    return int(base)


def plan_group_rebalance(
    per_shard_groups: Mapping[int, Sequence[K]], *, max_move_groups: int = 0
) -> list[tuple[K, int]]:
    """β1b **批次级整组再分配**规划（纯函数）：返回 ``[(group_key, 目标分片)]``。

    :param per_shard_groups: ``{shard: [该分片当前持有的整组键]}``。
    :param max_move_groups: 允许搬动的组数上限（``0`` = 不搬，返回空计划）。

    ★ 本次（P4）只做**规划**：搬组的**跨 rank 执行**（一次批次边界会合）属
    **待跑次阶段**；默认 ``max_move_groups=0`` ⇒ 空计划 ⇒ **行为与零新增通信**。

    Raises:
        AssignmentError: 分片号重复 / 负数预算。
    """
    if max_move_groups < 0:
        raise AssignmentError(f"max_move_groups 不能为负：{max_move_groups}")
    shards = sorted(per_shard_groups)
    if len(set(shards)) != len(shards):
        raise AssignmentError("分片号重复。")
    if max_move_groups == 0 or len(shards) < 2:
        return []

    holdings = {s: list(per_shard_groups[s]) for s in shards}
    plan: list[tuple[K, int]] = []
    while len(plan) < max_move_groups:
        counts = {s: len(holdings[s]) for s in shards}
        rich = max(shards, key=lambda s: (counts[s], -s))
        poor = min(shards, key=lambda s: (counts[s], s))
        if counts[rich] - counts[poor] <= 1 or not holdings[rich]:
            break
        key = holdings[rich].pop()
        holdings[poor].append(key)
        plan.append((key, poor))
    return plan


__all__ = [
    "AssignmentError",
    "assign_whole_groups",
    "assert_balanced_assignment",
    "plan_group_rebalance",
    "step_group_quota",
    "whole_group_count",
]
