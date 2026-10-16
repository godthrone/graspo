"""flow.adapter.contract — ★ 算法 ↔ 后端的**对接契约**（形状 + 三条不变量）。

**一名一物**（P0 消歧后）：本模块所在的 ``flow/adapter/`` 专指「算法与后端之间的对接
契约」；**不是**「模型 / LoRA 适配」（``flow/backend/native/models/<m>/model_adapter.py``），
**不是**「并行度」（``flow/backend/parallel_degree.py``）。

## 契约是什么（§1.2 接口边界）

后端（native / msswift）必须交出**同型**的 :class:`RolloutGroup`；算法层（``ripple/``）
只认这个形状。契约**不做**生成、通信、分片，也**不做**算法判定。

## 三条硬不变量（§2.1 契约即防呆 / §2.3 边界校验即防呆）

| # | 不变量 | 违反时 |
|:--:|---|---|
| **I1** | **组完整性 = G 条齐**：一个组的 ``completions`` 条数必须 ``== group_size``，且同属一个 ``group_key`` | **显式失败**（:class:`IncompleteGroupError`），不得静默丢弃 / 填充 / 切片 |
| **I2** | **奖励 ↔ 样本按键对齐**：``reward[i]`` 属于 ``completion[i]``，键 = :func:`sample_key`；**禁用按行号回退** | **显式失败**（:class:`AlignmentError`） |
| **I3** | **不完整组必须显式失败**：组清单里不允许出现 ``len < group_size`` 的组（**不是过滤，是失败** —— 是防线不是退路，§3.4） | **显式失败**（:class:`IncompleteGroupError`） |

另有一条 **β1 专用**断言 :func:`assert_group_partition`（整组不拆 / 全局组集合不变 /
各分片组数差 ≤ 1），供 P4 的均衡分配接线使用。

★ 以及一条 **rollout 分卡专用**的**装配点契约** :func:`assert_assembled_group_complete`
（`report.md` §7.4 / X-09）：分卡把**组内 rollout 维**切成 ``G//m`` ⇒ 在把组交给算法层
**之前**必须跨卡拼齐 G 条。⛔ 它**不放宽 I1** —— I1 的原文与机核
（:func:`assert_complete_group`）**一字不动**；变的只是"G 条齐在哪一层被断言"。
交付点由 :data:`ASSEMBLY_DELIVERY_POINTS` 具名登记（native = 装配后；msswift = 本进程整组）。

## 边界

- 本模块**只做形状与判定**，零 IO、零 torch、零通信（便于 CPU 独立单测）；
- 具体 backend 的接线与这些断言的调用，属 P5（msswift G1/G2）与 P4（β1）；
- 断言的**可机核验证**由验证台/测试负责（``tests/flow/adapter/test_contract.py``）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


# ── 失败语义（全部 fail-closed）───────────────────────────────────────────────
class ContractError(RuntimeError):
    """对接契约被违反（基类）。调用方必须让它终止该 step —— 不要 catch 后继续跑。"""


class IncompleteGroupError(ContractError):
    """**I1 / I3**：组不完整（条数 < G，或跨组混装）。

    这是**防线**而不是**退路**（§3.4）：不完整的分片**不得**被当成一个完整的组
    继续喂给算法层 —— 那会让组决策在部分组上做（实测后果：``groups_seen ≡ 1``）。
    """


class AlignmentError(ContractError):
    """**I2**：奖励与样本的键集合不一致（或缺失），**禁止**回退为按行号配对。"""


class PartitionError(ContractError):
    """β1：分配不满足"整组不拆 / 全局组集合不变 / 各分片组数差 ≤ 1"。"""


class AssemblyError(ContractError):
    """★ **装配点契约**：交给算法层的组没有从同一 dp 组的各成员处拼齐 G 条。

    这不是 β1 的"分片"问题，也不是 I1 的"组形状"问题 —— 它说的是**交付点**：
    rollout 分卡（``G → G//m`` 切片）之后，组内 G 条**分散在 m 张卡上**，
    ⇒ 在把组交给算法层**之前**必须先把它们拼齐。拼不齐 = **显式失败**
    （⛔ 绝不允许算法层在"部分组"上做决策 —— 那正是
    :class:`IncompleteGroupError` docstring 点名的事故 ``groups_seen ≡ 1``）。
    """


#: ★ 装配点契约的**具名标识**（能力矩阵/文档引用它，避免"装配"一词散落成隐式约定）。
ROLLOUT_SLICING_ASSEMBLY: str = "rollout_slicing_assembly"

#: ★ 交付点声明（`report.md` §7.4 / X-09）：**"G 条齐"这条不变量只有一份**（I1），
#: 变的只是"它在哪一层被断言"。两个 backend 的交付点**不同**，故必须具名登记。
ASSEMBLY_DELIVERY_POINTS: dict[str, str] = {
    "native": "装配后（决策前一次 dp_group all_gather 把各卡切片拼成整组 G 条）",
    "msswift": "本进程整组（ms-swift 侧不切 rollout 维）",
}

# ── 形状 ─────────────────────────────────────────────────────────────────────
def sample_key(group_key: str, completion_index: int) -> str:
    """奖励 ↔ 样本的**唯一键**（I2 的单一真相源，§1.4）。

    键由 ``(group_key, completion_index)`` 决定，与"这一条在本地切成什么样、
    排在列表第几位"**无关** —— 这正是禁止按行号回退的原因。
    """
    if not isinstance(group_key, str) or not group_key:
        raise ContractError(f"group_key 必须是非空字符串，得到 {group_key!r}")
    if not isinstance(completion_index, int) or completion_index < 0:
        raise ContractError(f"completion_index 必须是非负整数，得到 {completion_index!r}")
    return f"{group_key}#{completion_index}"


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    """一个 rollout 组：**同属一个 group_key 的 G 条 completion** + 按键奖励表。

    :param group_key: 组标识（同一 prompt 的 G 条共享）。
    :param completions: 本组的 completion（后端产出的原始对象 / 解析结果）。
    :param group_size: 契约要求的组大小 G（= rollout_group_size）。
    :param rewards: ``{sample_key(group_key, i): reward}``；``None`` = 尚未评分。
    """

    group_key: str
    completions: tuple[Any, ...]
    group_size: int
    rewards: Mapping[str, float] | None = None

    def keys(self) -> tuple[str, ...]:
        """本组每个 completion 的契约键（顺序与 ``completions`` 一致）。"""
        return tuple(sample_key(self.group_key, i) for i in range(len(self.completions)))

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ContractError(f"group_size 必须 ≥ 1，得到 {self.group_size!r}")


# ── I1：组完整性 = G 条齐 ────────────────────────────────────────────────────
def assert_complete_group(group: RolloutGroup) -> None:
    """**I1**：``len(completions) == group_size``，否则显式失败。

    Raises:
        IncompleteGroupError: 条数不符（含 0 条 / 多于 G 条）。
    """
    actual = len(group.completions)
    if actual != group.group_size:
        raise IncompleteGroupError(
            f"组不完整：group_key={group.group_key!r} 有 {actual} 条 completion，"
            f"契约要求 {group.group_size} 条（I1）。不完整的分片不得当成完整组使用。"
        )


# ── I2：奖励 ↔ 样本按键对齐 ──────────────────────────────────────────────────
def assert_aligned(group: RolloutGroup) -> None:
    """**I2**：奖励键集合 == completion 键集合（一一对应），否则显式失败。

    禁止"长度不匹配就按行号配对"——那是双真相源（§1.4），实测会造成奖励错位。

    Raises:
        AlignmentError: 奖励缺失、键集合不等、或出现组外键。
    """
    if group.rewards is None:
        raise AlignmentError(
            f"奖励缺失：group_key={group.group_key!r} 尚未评分（rewards=None）。"
            "未评分不等于零奖励 —— 不得据此推算。"
        )
    expected = set(group.keys())
    actual = set(group.rewards.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AlignmentError(
            f"奖励与样本未按键对齐：group_key={group.group_key!r} "
            f"期望键 {len(expected)} 个、实际 {len(actual)} 个；"
            f"缺 {missing[:3]}、多 {extra[:3]}（I2）。禁止回退为按行号配对。"
        )


# ── I3：组清单里不得出现不完整组 ─────────────────────────────────────────────
def assert_no_partial_group(groups: Iterable[RolloutGroup]) -> None:
    """**I3**：逐个校验组清单，任一不完整即显式失败（**不是过滤**）。

    Raises:
        IncompleteGroupError: 任一组的条数 != 其 ``group_size``。
    """
    for group in groups:
        assert_complete_group(group)


# ── β1（P4 接线）：整组不拆 / 全局组集合不变 / 各分片组数差 ≤ 1 ────────────────
def assert_group_partition(
    groups: Sequence[RolloutGroup],
    assignment: Mapping[str, int],
    *,
    n_shards: int,
) -> None:
    """β1 专用：校验"组 → 分片"的分配不变量。

    :param groups: 全局组清单（分片前的口径）。
    :param assignment: ``{group_key: shard_index}``（键 = 组，**不拆组**由形状保证）。
    :param n_shards: 分片数（= 参与本次优化的 rank 数）。

    Raises:
        PartitionError: 组集合不齐 / 分片越界 / 各分片组数差 > 1。
    """
    if n_shards < 1:
        raise PartitionError(f"n_shards 必须 ≥ 1，得到 {n_shards!r}")
    keys = [g.group_key for g in groups]
    if len(set(keys)) != len(keys):
        raise PartitionError("组清单里有重复 group_key —— 全局组集合不是集合语义。")
    if set(assignment) != set(keys):
        missing = sorted(set(keys) - set(assignment))
        extra = sorted(set(assignment) - set(keys))
        raise PartitionError(
            f"分配未覆盖全局组集合：缺 {missing[:3]}、多 {extra[:3]}（β1）。组不得被拆或丢弃。"
        )
    counts = [0] * n_shards
    for shard in assignment.values():
        if not 0 <= shard < n_shards:
            raise PartitionError(f"分片号越界：{shard!r} ∉ [0, {n_shards})")
        counts[shard] += 1
    if counts and (max(counts) - min(counts)) > 1:
        raise PartitionError(f"各分片组数失衡：{counts}（β1 要求差 ≤ 1 组）")


# ── ★ 装配点契约（rollout 分卡 / U-C）：交付给算法层之前必须拼齐整组 ──────────
def assert_assembled_group_complete(
    group_key: str,
    slots: Iterable[tuple[int, int]],
    *,
    group_size: int,
    dp_size: int,
) -> None:
    """**装配点契约**：跨卡拼齐的槽位必须**恰为** ``0..G−1`` 各一次。

    :param group_key: 组标识（仅用于报错定位）。
    :param slots: 跨卡收集到的 ``(dp_rank, rollout_slot)`` 二元组（**顺序无关** ——
        键语义天然支持乱序/跨卡，:func:`sample_key` 同源）。
    :param group_size: 算法组大小 ``G``（= ``training.rollout_group_size``）。
    :param dp_size: 切片数 ``m``（= ``native.dp_size``）。

    语义（`report.md` §7.4 / X-09）：**"G 条齐"只有一份不变量（I1）**，本函数断言的是
    "它**在哪一层**被断言" —— 必须在**把组交给算法层之前**。

    Raises:
        AssemblyError: 槽位有缺 / 有重 / 越界 / ``G`` 与 ``m`` 不自洽。
    """
    if int(group_size) < 1:
        raise AssemblyError(f"group_size 必须 >= 1，得到 {group_size!r}")
    if int(dp_size) < 1:
        raise AssemblyError(f"dp_size 必须 >= 1，得到 {dp_size!r}")
    if int(group_size) % int(dp_size) != 0:
        raise AssemblyError(
            f"group_size (G={int(group_size)}) 必须能被 dp_size (m={int(dp_size)}) 整除 —— "
            "否则任何装配都不可能得到整组 G 条（切片本身就配不出整组）"
        )
    rows = [(int(rank), int(slot)) for rank, slot in slots]
    ranks = [rank for rank, _ in rows]
    if any(not 0 <= rank < int(dp_size) for rank in ranks):
        raise AssemblyError(
            f"槽位来源 rank 越界（应在 [0, {int(dp_size)})）：group_key={group_key!r} "
            f"实际 {sorted(set(ranks))}"
        )
    got = sorted(slot for _, slot in rows)
    expected = list(range(int(group_size)))
    if got != expected:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise AssemblyError(
            f"装配点契约被违反：group_key={group_key!r} 拼齐后应有 {int(group_size)} 条"
            f"（槽位 0..{int(group_size) - 1} 各一次），实际 {len(got)} 条；"
            f"缺 {missing[:5]}、多 {extra[:5]}（重复槽位 {len(got) - len(set(got))} 个）。"
            f"来源 rank 分布={ {rank: ranks.count(rank) for rank in sorted(set(ranks))} }。"
            "⛔ 不完整的组不得交给算法层（否则组决策在部分组上做，groups_seen ≡ 1）。"
        )
    # 每条 slot 的来源 rank 也必须在**切片口径**下自洽（每个 slot 只有一个出处）。
    expected_rank_of_slot = {
        slot: slot // (int(group_size) // int(dp_size)) for slot in expected
    }
    wrong = [
        (rank, slot) for rank, slot in rows if expected_rank_of_slot[slot] != rank
    ]
    if wrong:
        raise AssemblyError(
            f"槽位来源与切片口径不符：group_key={group_key!r} 的 "
            f"{wrong[:3]} 不属于其 rank 的连续槽位区间 "
            f"（G//m = {int(group_size) // int(dp_size)}）—— "
            "说明装配时拼接顺序/切片口径不一致。"
        )


__all__ = [
    "ASSEMBLY_DELIVERY_POINTS",
    "AlignmentError",
    "AssemblyError",
    "ContractError",
    "IncompleteGroupError",
    "PartitionError",
    "ROLLOUT_SLICING_ASSEMBLY",
    "RolloutGroup",
    "assert_aligned",
    "assert_assembled_group_complete",
    "assert_complete_group",
    "assert_group_partition",
    "assert_no_partial_group",
    "sample_key",
]
