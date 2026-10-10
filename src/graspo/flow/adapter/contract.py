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


__all__ = [
    "AlignmentError",
    "ContractError",
    "IncompleteGroupError",
    "PartitionError",
    "RolloutGroup",
    "assert_aligned",
    "assert_complete_group",
    "assert_group_partition",
    "assert_no_partial_group",
    "sample_key",
]
