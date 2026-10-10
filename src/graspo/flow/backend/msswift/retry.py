"""flow.backend.msswift.retry — ★ retry：复用上游 resample 机制，换谓词 + 替换而非追加。

依据：`task-arch-design/option-a-override-plan.md` §2.5（Option A）与不变量 **V5**。

## 现状与修法

在役实现把 `retry_count=0` **硬编码**（`msswift/trainer.py:448`）⇒ `GroupDecision.RETRY`
退化为"本批不训"，**不重新 rollout**（native 侧有真重采样循环，`flow/backend/native/trainer/trainer.py`）。

修法（Option A §2.5）：
1. **复用**上游已有的 DAPO resample（`dynamic_sample` / `_prepare_resample_data_iterator` /
   `_dynamic_sampling`）；
2. **换谓词**：从上游的"组内 reward std==0"换成 graspo 的 :class:`GroupDecision.RETRY`；
3. ★ **替换而不是追加** ⇒ 重采样后本地条数**不变**（V5/V2）；
4. 删 `retry_count=0` 硬编码，传入**真实计数**；
5. ★ 上游 `_dynamic_sampling` 的 `process_slice = slice(idx*len(samples), (idx+1)*len(samples))`
   是**位置切片**，同样假设各 rank 等长 ⇒ 覆写时必须一并**按组切片**。

## 边界

纯函数：零 IO / 零 torch / 零 ms-swift；本模块只出**判定与计划**，执行仍在 trainer 覆写里
（多卡实测 ⇒ **待跑次阶段**）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from graspo.ripple.group_decision import GroupDecision


class RetryError(RuntimeError):
    """retry 计划不满足 V5（替换而非追加）或超出重试预算。"""


@dataclass(frozen=True, slots=True)
class ResamplePlan:
    """一批的重采样计划：**替换**哪些位置、保留哪些位置。"""

    keep: tuple[int, ...]
    replace: tuple[int, ...]
    exhausted: tuple[int, ...]

    @property
    def is_noop(self) -> bool:
        return not self.replace and not self.exhausted


def is_resample_needed(decision: Any) -> bool:
    """★ 谓词：graspo 的 ``GroupDecision.RETRY``（**不是**上游的"reward std==0"）。"""
    if isinstance(decision, GroupDecision):
        return decision is GroupDecision.RETRY
    should_retry = getattr(decision, "should_retry", None)
    if should_retry is not None:
        return bool(should_retry)
    value = getattr(decision, "decision", decision)
    return str(value) == str(GroupDecision.RETRY)


def plan_resample(
    decisions: Sequence[Any],
    retry_counts: Sequence[int],
    *,
    max_attempts: int,
) -> ResamplePlan:
    """出重采样计划：``RETRY`` 且未超预算 ⇒ 进 ``replace``；超预算 ⇒ 进 ``exhausted``。

    :param decisions: 本批每条的组决策。
    :param retry_counts: 本批每条**已重试次数**（真实值，不再是硬编码 0）。
    :param max_attempts: 每样本最多尝试次数（含首跑；对应 ``training.rollout_max_retries + 1``）。

    Raises:
        RetryError: 长度不符 / ``max_attempts < 1``。
    """
    if len(decisions) != len(retry_counts):
        raise RetryError(
            f"decisions 与 retry_counts 长度不符：{len(decisions)} vs {len(retry_counts)}"
        )
    if max_attempts < 1:
        raise RetryError(f"max_attempts 必须 ≥ 1，得到 {max_attempts}")
    keep: list[int] = []
    replace: list[int] = []
    exhausted: list[int] = []
    for index, decision in enumerate(decisions):
        if not is_resample_needed(decision):
            keep.append(index)
        elif int(retry_counts[index]) + 1 >= max_attempts:
            exhausted.append(index)
        else:
            replace.append(index)
    return ResamplePlan(tuple(keep), tuple(replace), tuple(exhausted))


def assert_replacement_preserves_size(
    old_size: int, new_size: int, *, plan: ResamplePlan | None = None
) -> None:
    """机核 **V5**：重采样是**替换**而不是追加 ⇒ 本地条数必须不变。

    Raises:
        RetryError: 条数变了。
    """
    if old_size != new_size:
        extra = f"（plan={plan}）" if plan is not None else ""
        raise RetryError(
            f"重采样改变了本地条数：{old_size} → {new_size}{extra}；"
            "V5 要求**替换而非追加**（否则各 rank 不等长 ⇒ all_gather_into_tensor 炸）"
        )


def group_slice_for_rank(
    n_groups_total: int, groups_per_rank: int, rank: int, *, world_size: int
) -> slice:
    """★ **按组切片**（替代上游的位置切片 ``slice(idx*len(samples), …)``）。

    :param n_groups_total: 全局**组**数（不是条数）。
    :param groups_per_rank: 每进程组数 k。
    :param rank: 本进程的 **DP** 序（整组分片后每进程拿第 ``rank`` 段整组）。
    :param world_size: 进程数。

    Raises:
        RetryError: 参数非法或越界。
    """
    if groups_per_rank < 1 or world_size < 1:
        raise RetryError(f"k/world 必须 ≥ 1，得到 {groups_per_rank}/{world_size}")
    if not 0 <= rank < world_size:
        raise RetryError(f"rank={rank} 不在 [0, {world_size}) 内")
    total_needed = groups_per_rank * world_size
    if n_groups_total < total_needed:
        raise RetryError(
            f"全局组数 {n_groups_total} < k×world = {total_needed} ⇒ 各进程组数不可能严格相等"
        )
    start = rank * groups_per_rank
    return slice(start, start + groups_per_rank)


__all__ = [
    "ResamplePlan",
    "RetryError",
    "assert_replacement_preserves_size",
    "group_slice_for_rank",
    "is_resample_needed",
    "plan_resample",
]
