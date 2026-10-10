"""flow.backend.msswift.group_sharding — ★ G1 **整组分片**（切在哪里，不是拼回来）。

依据：`task-arch-design/option-a-override-plan.md` §2.1/§3（Option A）、`s1-s2-findings.md` §1–§3。

## 要保的不变量（option-a §1）

| # | 不变量 | 判据 |
|:--:|---|---|
| **V1** | 每进程本地批 = **恰好 k 个完整组** | `len(local) == k×G` 且每 G 条同属一个组 |
| **V2** | 各进程组数**严格相等** | 逐 rank 批长度相同（`all_gather_into_tensor` 要求 `numel` 相等） |
| **V3** | 全局拼接后**组连续** | 按 rank 序拼接，每 G 行一组 |
| **V5** | retry **替换而非追加** | 重采样后本地条数不变 |

## ★ 新 `steps_per_generation` 映射（构造性保证 V1/V2）

上游约束：``per_device_train_batch_size × steps_per_generation ≡ 0 (mod G)``；
**最小解** ``steps_per_generation = G / gcd(per_device, G)``（旧式 ``ceil(G/(per_device×world))``
在 T044 给 4 < 8 ⇒ **必然切组**，正是 6 档降级的机制根）。

| per_device | world | G | 旧 → 每进程条数 | **新 steps** | 每进程条数 / 组数 |
|---:|---:|---:|---|---:|---|
| 1 | 2 (T044) | 8 | 4 ❌ 切组 | **8** | 8 / **1** |
| 1 | 4 (T045) | 8 | 4 ❌ 切组 | **8** | 8 / **1** |
| 2 | 2 | 8 | 4 ❌ | **4** | 8 / 1 |
| 8 | 1 | 8 | 8 ✅（侥幸） | **1** | 8 / 1 |

## 边界

- 本模块**纯函数**：零 IO / 零 torch / 零 ms-swift 依赖 ⇒ CPU 可机核（§1.3）；
- **不做**生成/评分/算法判定；`even_batches=False` 是**必须**的（否则尾部复制整批 ⇒ 同一组训两次，违反 V1 的内容面）；
- 多卡实测（各 rank 组数相等）⇒ **待跑次阶段**。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd

from graspo.flow.backend.msswift.error import ShardingError


def steps_per_generation_for(per_device_train_batch_size: int, group_size: int) -> int:
    """★ 新映射：``G / gcd(per_device, G)``（构造性保证每进程整除 ⇒ V1/V2）。

    Raises:
        ShardingError: 任一参数 < 1。
    """
    if per_device_train_batch_size < 1 or group_size < 1:
        raise ShardingError(
            f"per_device/G 必须 ≥ 1，得到 {per_device_train_batch_size!r}/{group_size!r}"
        )
    return group_size // gcd(per_device_train_batch_size, group_size)


def groups_per_rank_for(
    per_device_train_batch_size: int, group_size: int, steps_per_generation: int
) -> int:
    """每进程组数 ``k = per_device × steps / G``（由新映射保证为整数）。

    Raises:
        ShardingError: 不整除（⇒ 调用方必须先修 `steps`，不得继续跑）。
    """
    if group_size < 1:
        raise ShardingError(f"group_size 必须 ≥ 1，得到 {group_size!r}")
    product = per_device_train_batch_size * steps_per_generation
    if product % group_size != 0:
        raise ShardingError(
            f"每进程条数不整除：per_device×steps={product} 不是 G={group_size} 的整数倍 "
            "⇒ 必然切组（G1 禁止）"
        )
    k = product // group_size
    if k < 1:
        raise ShardingError(f"每进程组数 k 必须 ≥ 1，得到 {k}")
    return k


def local_batch_rows(
    per_device_train_batch_size: int, group_size: int, steps_per_generation: int
) -> int:
    """每进程本地批的**行数**（= k×G）——V1/V2 的构造性保证面。"""
    return groups_per_rank_for(per_device_train_batch_size, group_size, steps_per_generation) * group_size


def build_group_contiguous_batches(
    n_rows: int,
    group_size: int,
    *,
    groups_per_rank: int = 1,
    world_size: int = 1,
    drop_last: bool = False,
) -> list[list[int]]:
    """把数据集行按"每 G 行一组"**连续**切成批，每批 = ``groups_per_rank`` 个整组。

    :param n_rows: 数据集行数（同一组的 G 行必须**连续**存放 —— 上游 `prompt_id` 序）。
    :param group_size: 组大小 G。
    :param groups_per_rank: 每批组数 k（≥1）。
    :param world_size: 进程数；★ V2 要求 ``len(batches) % world_size == 0``。
    :param drop_last: 不足一批时是否丢弃（``False`` ⇒ 显式失败，不静默产生半批）。

    Returns:
        批列表，每批是**行索引**列表，长度恒为 ``k×G``。

    Raises:
        ShardingError: 参数非法、或 ``len(batches) % world_size != 0`` 且 ``drop_last=False``。
    """
    if group_size < 1 or groups_per_rank < 1 or world_size < 1:
        raise ShardingError(
            f"G/k/world 必须 ≥ 1，得到 {group_size}/{groups_per_rank}/{world_size}"
        )
    if n_rows < 0:
        raise ShardingError(f"n_rows 不能为负：{n_rows}")

    per_batch = groups_per_rank * group_size
    n_batches = n_rows // per_batch
    if n_rows % per_batch and not drop_last:
        # 余数不足一批：只允许"丢尾"（显式配置）或显式失败 —— 绝不产生半批。
        pass  # 由下面的 world 整除检查统一处理（余数行不进入任何批）
    batches = [
        list(range(index * per_batch, (index + 1) * per_batch)) for index in range(n_batches)
    ]
    if drop_last:
        n_batches = (n_batches // world_size) * world_size
        batches = batches[:n_batches]
    if batches and len(batches) % world_size != 0:
        raise ShardingError(
            f"批数 {len(batches)} 不能被 world_size={world_size} 整除 ⇒ 各进程组数不相等 "
            "（违反 V2）：请设 `msswift.dataloader_drop_last=true` 或调整 `groups_per_rank`"
        )
    return batches


def assert_group_contiguous(
    batches: Sequence[Sequence[int]], group_size: int, *, groups_per_rank: int = 1
) -> None:
    """机核 V1/V3：每批恰好 ``k`` 个**完整组**，且组边界对齐（无跨批半组）。

    Raises:
        ShardingError: 任一违反。
    """
    per_batch = groups_per_rank * group_size
    for index, batch in enumerate(batches):
        if len(batch) != per_batch:
            raise ShardingError(
                f"第 {index} 批条数 {len(batch)} != k×G = {per_batch}（违反 V1）"
            )
        if len(set(batch)) != len(batch):
            raise ShardingError(f"第 {index} 批有重复行（同一条训两次，违反 V1 内容面）")
        if batch[0] % group_size != 0:
            raise ShardingError(
                f"第 {index} 批起点 {batch[0]} 不在组边界上（切了半个组，违反 V1/V3）"
            )
        if list(batch) != list(range(batch[0], batch[0] + per_batch)):
            raise ShardingError(f"第 {index} 批不是连续区间（违反 V3 组连续）")


__all__ = [
    "ShardingError",
    "assert_group_contiguous",
    "build_group_contiguous_batches",
    "groups_per_rank_for",
    "local_batch_rows",
    "steps_per_generation_for",
]
