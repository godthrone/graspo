"""flow.backend.msswift.alignment — ★ G2 **奖励 ↔ 样本按键对齐**（禁止按行号回退）。

依据：`task-arch-design/option-a-override-plan.md` §2.2/§2.3（Option A）；契约 I2
（`flow/adapter/contract.py`）；`backend-interface.md` §3.3 的 V1/V2。

## 为什么要显式按键（而不是"位置恰好对得上"）

位置写回的正确性依赖三条**隐式**前提：① 批大小整除；② `gather` 序 = rank 序；
③ 组连续。任一条被破坏即**静默错位**（实测 I2 违反：`rewards_per_func has N rows but M samples`
⇒ 回退按行号配对，T032/T033/T038/T039/T044/T045 每跑次 40/80 次错位）。
契约 I2 要求**显式按键 + 不匹配即失败**（§2.2 显式即防呆）——本模块是那条契约在
msswift 上的落地口径，**单一真相源**（§1.4）。

## 边界

纯函数：零 IO / 零 torch / 零 ms-swift；失败一律用契约的 :class:`AlignmentError`（fail-closed）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from graspo.flow.adapter.contract import AlignmentError, sample_key

V = TypeVar("V")


def keyed_rows_from_table(
    rows: Sequence[V], keys: Sequence[str], *, what: str = "表"
) -> dict[str, V]:
    """把 ``(rows, keys)`` 的表转成 ``{key: row}``，**键缺失/重复即失败**。

    Raises:
        AlignmentError: 长度不符 / 键重复。
    """
    if len(rows) != len(keys):
        raise AlignmentError(
            f"{what}的长度与键数不符：rows={len(rows)} keys={len(keys)}（I2：禁止按行号配对）"
        )
    out: dict[str, V] = {}
    duplicates: list[str] = []
    for row, key in zip(rows, keys, strict=True):
        if key in out:
            duplicates.append(key)
        out[key] = row
    if duplicates:
        raise AlignmentError(f"{what}里有重复键：{sorted(set(duplicates))[:3]}（I2：键必须一一对应）")
    return out


def align_by_key(
    expected_keys: Sequence[str], values_by_key: Mapping[str, V], *, what: str = "奖励"
) -> list[V]:
    """按 ``expected_keys`` 的顺序取出 ``values_by_key`` 的值，**键集合必须相等**。

    这是 `_postprocess_batch`（advantages 写回）与 `_group_rewards`（组奖励取行）的**共同口径**。

    Raises:
        AlignmentError: 键集合不等（缺/多）—— **不得**回退为按行号配对。
    """
    expected = list(expected_keys)
    if len(set(expected)) != len(expected):
        raise AlignmentError(f"expected_keys 里有重复键（I2）")
    missing = [key for key in expected if key not in values_by_key]
    extra = [key for key in values_by_key if key not in set(expected)]
    if missing or extra:
        raise AlignmentError(
            f"{what}与样本未按键对齐：期望 {len(expected)} 个键、实际 {len(values_by_key)} 个；"
            f"缺 {missing[:3]}、多 {extra[:3]}（I2）。禁止回退为按行号配对。"
        )
    return [values_by_key[key] for key in expected]


def group_reward_rows(
    global_rows: Sequence[V], global_keys: Sequence[str], local_keys: Sequence[str], *, what: str = "组奖励表"
) -> list[V]:
    """从全局组奖励表里**按键**取出本地样本对应的行（`_group_rewards` 的落地口径）。

    旧实现（在役 `msswift/trainer.py:901-943`）在长度不匹配时按行号回退并打
    `falling back to positional pairing` 警告 —— **本函数没有那条退路**：不匹配即 `AlignmentError`。

    Raises:
        AlignmentError: 表键与本地键集合不等。
    """
    table = keyed_rows_from_table(global_rows, global_keys, what=what)
    return align_by_key(local_keys, table, what=what)


def completion_keys(group_key: str, group_size: int) -> list[str]:
    """一组的 G 个 completion 键（``sample_key`` 的唯一构造入口，§1.4）。"""
    if group_size < 1:
        raise AlignmentError(f"group_size 必须 ≥ 1，得到 {group_size!r}")
    return [sample_key(group_key, index) for index in range(group_size)]


def advantages_for_samples(
    local_keys: Sequence[str], advantages_by_key: Mapping[str, Any], *, what: str = "advantages"
) -> list[Any]:
    """`_postprocess_batch` 的写回口径：**按键**取 advantages（不再调 `get_even_process_data`）。"""
    return align_by_key(local_keys, advantages_by_key, what=what)


__all__ = [
    "advantages_for_samples",
    "align_by_key",
    "completion_keys",
    "group_reward_rows",
    "keyed_rows_from_table",
]
