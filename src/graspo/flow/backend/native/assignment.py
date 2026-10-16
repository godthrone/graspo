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

## ★ 新增：**rollout 分卡**（2026-10-16 主席裁定「必须做 rollout 分卡…只有这个才能治本」）

第二条分配单位——**组内 rollout 维的切片**（本模块后半部分：`rollouts_per_rank` /
`rollout_slice` / `assembled_slot_order` / `assert_rollout_slices`）。
β1b 的「整组不拆」在这里**不被放宽**：β1b 管的是"组整体落在哪个分片"，
切片管的是"**同一条 prompt 的各 rollout 由哪张卡生成/训练**"；
交给算法层之前仍必须**拼齐整组 G 条**（装配点契约，
`flow/adapter/contract.py::assert_assembled_group_complete`）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

from graspo.flow.backend.native.error import AssignmentError

K = TypeVar("K")


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


# ── ★ rollout 分卡（2026-10-16 主席裁定「必须做 rollout 分卡」）────────────────
#
# 分配单位从「整组（一条 prompt 的 G 条）」改成「**(条目, rollout 槽位)**」：
# 卡 `r` 持**该批全部条目**，但每条只生成/训练槽位 `[r·(G/m), (r+1)·(G/m))`。
#
# **为什么这不是 β2 复活**：β2 是「每卡只看本卡的 `G/m` 条就做组决策」；
# 本机制在**决策之前**把整组凑齐（一次 ~300 B 的 `all_gather`）⇒ **算法可见输入仍是整组 G 条**。
# 决定性的差别在**会合**，不在切片本身（`report.md` §4-②）。
#
# **切片是"连续"口径**（= 主席原话「卡一 0–3、卡二 4–7」），⛔ 不是 `[rank::m]` 轮转：
# 连续切片才让「按 rank 序拼接」与「单卡槽位序」逐元素对齐（`assembled_slot_order`）。


def validate_rollout_slicing(group_size: int, dp_size: int) -> None:
    """切片参数合法性（**fail-closed**，消息点名该改哪个键 —— 宪法 §2.3 防呆）。

    两条：① ``G % m == 0``（否则各卡切片并集 ≠ 整组 G 条）；
    ② ``G // m >= 1``（否则有卡一条都不生成 ⇒ 该卡恒空、日程必然不等）。

    Raises:
        AssignmentError: 任一不成立。
    """
    if int(group_size) < 1:
        raise AssignmentError(
            f"training.rollout_group_size (G) 必须 >= 1，得到 {group_size!r}；"
            "请改 training.rollout_group_size"
        )
    if int(dp_size) < 1:
        raise AssignmentError(
            f"native.dp_size (m) 必须 >= 1，得到 {dp_size!r}；请改 native.dp_size"
        )
    # ★ 顺序有讲究：`m > G` 时 `G % m == G != 0` **恒真** ⇒ 若先判整除，
    #   下面这条"每卡至少一条"的分支就永远到不了（`m > G ⇒ G//m == 0`）。
    #   先判它 ⇒ 报错文案才指向**真正的**成因（"有卡一条都不生成"）。
    if int(group_size) // int(dp_size) < 1:
        raise AssignmentError(
            f"native.dp_size (m={int(dp_size)}) 不得大于 training.rollout_group_size "
            f"(G={int(group_size)})：G // m 必须 >= 1，否则有 dp rank 一条 rollout 都不生成。"
            "请改 G 或 dp_size"
        )
    if int(group_size) % int(dp_size) != 0:
        raise AssignmentError(
            f"training.rollout_group_size (G={int(group_size)}) 必须能被 "
            f"native.dp_size (m={int(dp_size)}) 整除（否则各卡 rollout 切片并集 "
            "≠ 整组 G 条 ⇒ 决策输入不再是整组）。请改 G 或 dp_size"
        )


def rollouts_per_rank(group_size: int, dp_size: int) -> int:
    """每卡对**每条 prompt**生成/训练的 rollout 数 ``G // m``（★ 切片量子）。

    ★ **单一真相源**（§1.4）：生成入参（``rollout.py``）、步边界量子（``optimize.py``）、
    落盘读数除数（``optimize.py``）、配置阈值口径（``schema.py``）**全部**从这里取，
    ⛔ 任何调用方**不得**自己写 ``G // dp_size``。

    ``m == 1`` ⇒ 返回 ``G``（**恒等**）⇒ dp=1 全路径逐位不变（A-1 回归锚）。

    Raises:
        AssignmentError: ``G % m != 0``（切片拼不齐整组）或 ``G // m < 1``。
    """
    validate_rollout_slicing(group_size, dp_size)
    return int(group_size) // int(dp_size)


def rollout_slice(
    prompt_count: int, group_size: int, dp_size: int, rank: int
) -> list[tuple[int, int]]:
    """卡 ``rank`` 应负责的 ``(条目序号, rollout 槽位)`` 清单（确定性、纯函数）。

    槽位区间 = **连续** ``[rank·(G/m), (rank+1)·(G/m))``；条目维**不切**
    （卡 ``rank`` 对 ``0..prompt_count-1`` **每一条**都持有一段槽位）。

    返回顺序：**先条目、后槽位**（与 ``generate_groups`` 的
    ``(chunk_prompt_count, rollout_group_size)`` 行序一致）—— 故
    ``[slot for _, slot in rollout_slice(1, G, m, r)]`` 就是该卡的槽位序列。

    Raises:
        AssignmentError: 切片参数非法 / ``prompt_count < 0`` / ``rank`` 越界。
    """
    validate_rollout_slicing(group_size, dp_size)
    if int(prompt_count) < 0:
        raise AssignmentError(f"prompt_count 不能为负，得到 {prompt_count!r}")
    m = int(dp_size)
    if not 0 <= int(rank) < m:
        raise AssignmentError(f"rank 越界：{rank!r} ∉ [0, {m})")
    rpr = int(group_size) // m
    lo = int(rank) * rpr
    hi = lo + rpr
    return [(entry, slot) for entry in range(int(prompt_count)) for slot in range(lo, hi)]


def assembled_slot_order(group_size: int, dp_size: int) -> tuple[int, ...]:
    """跨卡**按 rank 序**拼齐后的槽位序列（★ 会合的加序契约）。

    机制：``all_gather`` 按 rank 序返回各卡切片，沿**槽位维**拼接 ⇒ 槽位序列
    恰为 ``(0, 1, …, G−1)`` —— 与**单卡整组**的槽位序**逐元素相同**。

    ★ 这条等式是「``μ_f`` 能按槽位序确定地求和、与单卡同加序」的**唯一依据**
    （`report.md` §4-③ / A-5）。⛔ 若切片改成轮转 ``[rank::m]``，本等式**不成立**。

    Raises:
        AssignmentError: 切片参数非法。
    """
    validate_rollout_slicing(group_size, dp_size)
    return tuple(range(int(group_size)))


def assert_rollout_slices(prompt_count: int, group_size: int, dp_size: int) -> None:
    """机核四条切片不变量（A-8a / A-6）—— 违反即 :class:`AssignmentError`。

    1. **条目集合不变**：每卡的条目集合都 = 全体条目（切片**不切条目维**）；
    2. **槽位互斥且完备**：同一 entry 的各卡槽位两两不交，并集 = ``0..G−1``；
    3. **等长**：每卡条数 = ``prompt_count · (G/m)``（⇒ 逐卡条数**严格相等**）；
    4. **``m == 1`` 恒等**：单卡时槽位序列 = ``0..G−1``（逐位不变）。

    ★ 这是「能失败的断言」：把 `rollout_slice` 改成"整组给每卡"或"轮转切片"，
    #2 或 #4 立即判红（阳性对照见 ``tests/flow/backend/native/test_assignment.py``）。

    Raises:
        AssignmentError: 任一不变量被违反。
    """
    validate_rollout_slicing(group_size, dp_size)
    P, G, m = int(prompt_count), int(group_size), int(dp_size)
    if P < 0:
        raise AssignmentError(f"prompt_count 不能为负，得到 {prompt_count!r}")
    rpr = G // m
    slices = {r: rollout_slice(P, G, m, r) for r in range(m)}

    # 1. 条目集合不变 + 3. 等长
    for r in range(m):
        entries = sorted({e for e, _ in slices[r]})
        if entries != list(range(P)):
            raise AssignmentError(
                f"切片改变了条目集合：rank={r} 的条目 {entries[:3]}… ≠ 全体 0..{P - 1}"
                "（切片只切 rollout 维，⛔ 不切条目维）"
            )
        if len(slices[r]) != P * rpr:
            raise AssignmentError(
                f"各卡条数不等：rank={r} 有 {len(slices[r])} 条，应为 P·(G/m)"
                f" = {P}·{rpr} = {P * rpr}"
            )

    # 2. 槽位互斥且完备
    for entry in range(P):
        union: list[int] = []
        for r in range(m):
            got = [slot for e, slot in slices[r] if e == entry]
            if len(set(got)) != len(got):
                raise AssignmentError(f"rank={r} entry={entry} 的槽位有重复：{got}")
            union.extend(got)
        if sorted(union) != list(range(G)):
            raise AssignmentError(
                f"entry={entry} 的跨卡槽位并集 ≠ 0..{G - 1}：得到 {sorted(union)}"
                "（切片必须互斥且完备，否则组凑不齐 G 条）"
            )

    # 4. m == 1 恒等
    if m == 1 and [slot for _, slot in slices[0]] != list(range(G)) * P:
        raise AssignmentError("m == 1 时切片必须恒等（槽位序列 = 0..G−1）")


__all__ = [
    "AssignmentError",
    "assembled_slot_order",
    "assert_balanced_assignment",
    "assert_rollout_slices",
    "assign_whole_groups",
    "plan_group_rebalance",
    "rollout_slice",
    "rollouts_per_rank",
    "step_group_quota",
    "validate_rollout_slicing",
    "whole_group_count",
]
