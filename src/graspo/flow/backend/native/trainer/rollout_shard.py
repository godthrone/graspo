"""flow.backend.native.trainer.rollout_shard — ★ rollout 分卡的**决策前会合**（设施层）。

## 这个模块做什么

rollout 分卡把"分配单位"从**整组**改成 **(条目, rollout 槽位)**：卡 `r` 生成槽位
`[r·(G/m), (r+1)·(G/m))`（见 ``assignment.rollout_slice``）。但**算法层的可见输入必须仍是整组
G 条** —— 否则组决策 / retry / perfect-skip 会在部分组上做（`IncompleteGroupError` 点名的事故）。
本模块因此提供**决策前把整组凑齐**的一次小会合：

| 会合 | 载荷 | 频次 |
|---|---|---|
| :func:`gather_rollout_values`（决策输入） | 每卡 `(P, G/m)` 的 ``rewards`` / ``content_scores`` / ``parse_error`` 标志 | 每 attempt 1 次（≈288 B/rank，P=8,G=8,m=2） |
| :func:`gather_field_stats`（``μ_f``） | 每卡 `(P, F, 2)` 的 ``(Σraw, n)`` | 每**可训** attempt 1 次（≈640 B/rank） |

★ **载荷里没有文本、没有标注**：只有标量与其计数 ⇒ 「文本/标注不出卡」这条红线由**载荷类型**保证，
不靠调用方自觉（β2 当年被否的理由正是"载荷是序列化 completion/标注"）。

## 加序契约（`μ_f` 的确定性和）

``all_gather`` 按 **rank 序**返回各卡切片，沿**槽位维**拼接
⇒ 槽位序列恰为 ``(0, …, G−1)``（``assignment.assembled_slot_order`` 机核）
⇒ ``μ_f`` 的累加序与**单卡整组**相同（仍非逐位——浮点累加本身有误差，见 X-04）。

## 边界与短路口径（★ dp=1 逐位不变）

- 本模块**只在 dp≥2 时做集合**；``dp_size <= 1`` / 无 ``dp_group`` / 进程组未初始化
  ⇒ 原样返回本地值，**0 次新增集合**（A-1 回归锚，与 ``training.py`` 的
  ``_dp_consistent_has_valid_micro_batches`` 短路口径同族）。
- ⛔ 本模块**不做**生成、评分、算法判定；也**不**改任何判定语义。
- 会合域 = ``tp_state.dp_group``（**与 K-会合同域**；WORLD 面由 U-A 单独负责）。

## 防御（fail-closed）

集合前后的形状全部机核：``cut`` 越界、切片长度 ≠ ``G/m``、gather 结果长度 ≠
``n·m`` ⇒ **显式 RuntimeError**。⛔ 不用"长度不对就按行号猜"的退路（那正是 I2 禁止的双真相源）。
"""

from __future__ import annotations

from typing import Any


def resolve_dp_group(runtime: Any) -> tuple[Any, int, Any]:
    """取 ``(dp_group, dp_size, device)``；任一不可用 ⇒ ``(None, 1, None)``。

    短路口径与 ``trainer.py::_world_gate_alignment_enabled`` 同源：
    **只有 dp_size > 1 且进程组已初始化**才做集合。
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None, 1, None
    adapter = runtime._require_adapter()
    tp_state = getattr(adapter, "tp_state", None)
    if tp_state is None:
        return None, 1, None
    dp_group = getattr(tp_state, "dp_group", None)
    dp_size = int(getattr(tp_state, "dp_size", 1) or 1)
    if dp_group is None or dp_size <= 1:
        return None, 1, None
    return dp_group, dp_size, getattr(tp_state, "device", None)


def _assert_local_shape(value: Any, n_local: int, rollouts_per_rank: int, *, what: str) -> None:
    """机核本地切片**元素数** ``== n_local · (G/m)``（fail-closed，⛔ 不静默截断/补零）。

    ★ 为什么判**元素数**而不是"二维形状"：``stack`` 前会 ``reshape(-1)``，拼接的**唯一**
    契约就是"每条条目恰好贡献 `G/m` 个槽位"。判元素数既覆盖 `(G/m,)` 与 `(1, G/m)`
    两种写法（生产实现与单测都自然），又**仍然**抓得住"多一条/少一条槽位"这类真错。
    """
    shape = tuple(getattr(value, "shape", ()))
    size = int(getattr(value, "numel", lambda: -1)())
    expected = int(n_local) * int(rollouts_per_rank)
    if size != expected or len(shape) == 0:
        raise RuntimeError(
            f"[rollout 分卡] {what} 的元素数应为 n_local × rollouts_per_rank = "
            f"{n_local} × {rollouts_per_rank} = {expected}，实际 shape={shape} "
            f"（元素数 {size}）—— 形状不符会让 all_gather 的槽位序拼接错位"
            "（决策输入不再是整组）。"
        )


def gather_rollout_values(
    runtime: Any,
    local_values: list[Any],
    *,
    cutoff: int,
    rollouts_per_rank: int,
    pad_values: list[Any] | None = None,
    what: str = "rollout 值",
) -> list[Any]:
    """把每卡 ``(n_local, G/m)`` 的标量切片按 **rank 序沿槽位维**拼成各自的整组 ``(n, G)``。

    :param local_values: 本卡每条条目的标量张量，形状 ``(n_local, G/m)``。
    :param cutoff: 参与本次会合的条目数 —— **必须逐 rank 相同**（调用方用
        ``_collective_any_pending`` 的 MAX 计数保证），否则不同 rank 会发不同形状的集合。
    :param pad_values: 长度 ≥ ``cutoff`` 的"补齐用"张量（形状同 ``local_values`` 元素）。
        当 ``len(local_values) < cutoff`` 时用它们补足**形状**（值不参与真实判定 ——
        ``classify_group`` 对每条 prompt 独立，补齐行的结果被调用方丢弃）。
        ``None`` ⇒ 不允许补齐（``len(local_values) != cutoff`` 即显式失败）。
    :returns: 长度 ``cutoff`` 的列表，每个是本条目**整组**的 ``(G,)`` 张量（槽位序 = 0..G−1）。

    Raises:
        RuntimeError: 形状不符 / ``cutoff < 0`` / 需要补齐但未提供 / gather 结果长度不符。
    """
    import torch
    import torch.distributed as dist

    if cutoff < 0:
        raise RuntimeError(f"[rollout 分卡] cutoff 不能为负：{cutoff}")
    if pad_values is None and len(local_values) < cutoff:
        raise RuntimeError(
            f"[rollout 分卡] {what}：本卡只有 {len(local_values)} 条、需要 {cutoff} 条，"
            "但未提供补齐张量 ⇒ 各 rank 的 all_gather 形状会不一致（集合失配）。"
        )
    # ★ 形状归一到 `cutoff` **先于** dp 短路：这样 dp≤1 与 dp≥2 的**返回长度语义一致**
    #   （否则同一个函数在两种并行度下返回值类型/长度不同 ⇒ 调用方要写两份防御）。
    if len(local_values) > cutoff:
        local_values = list(local_values[:cutoff])
    while len(local_values) < cutoff:
        local_values.append(pad_values[len(local_values) % len(pad_values)])
    # ★ 形状机核也**先于** dp 短路：形状是 `all_gather` 的**载荷契约**，
    #   dp≤1 下同样必须是 `(cutoff, G/m)` —— 否则同一份实现会有两套形状口径。
    for index, value in enumerate(local_values):
        _assert_local_shape(value, 1, rollouts_per_rank, what=f"{what}[{index}]")

    group, dp_size, device = resolve_dp_group(runtime)
    if group is None:
        # ★ dp ≤ 1：一个集合都不发（A-1 回归锚），原样返回本地值。
        return list(local_values)
    if device is None:
        device = torch.device("cpu")

    # ★ 会合成员数取自**组本身**（`dist.get_world_size(group)`），⛔ 不用调用方传的
    #   `dp_size` 当长度：两处一旦不一致，`all_gather` 的 `tensor_list` 长度就错
    #   （fail-closed，但报错点离真因很远）。组成员数才是唯一的载荷形状真相源。
    members = int(dist.get_world_size(group))
    if members != dp_size:
        raise RuntimeError(
            f"[rollout 分卡] {what}：会合组有 {members} 个成员，但 tp_state.dp_size="
            f"{dp_size} —— 两者不一致（切片宽度与会合域不符）⇒ 载荷形状无法确定。"
        )
    stacked = torch.stack([value.reshape(-1) for value in local_values], dim=0).to(device)
    gathered = [torch.empty_like(stacked) for _ in range(members)]
    dist.all_gather(gathered, stacked, group=group)
    # ★ 加序契约：`gathered` 按 **rank 序**；沿槽位维拼接 ⇒ 槽位序 = 0..G−1。
    whole = torch.cat(gathered, dim=1)
    if tuple(whole.shape) != (cutoff, rollouts_per_rank * members):
        raise RuntimeError(
            f"[rollout 分卡] {what}：会合后形状应为 "
            f"({cutoff}, {rollouts_per_rank}×{members})，实际 {tuple(whole.shape)}。"
        )
    return [whole[index] for index in range(cutoff)]


def gather_field_stats(
    runtime: Any,
    per_prompt_stats: list[dict[str, tuple[float, int]]],
    *,
    cutoff: int,
    field_names: list[str],
    what: str = "μ_f 统计",
) -> list[dict[str, tuple[float, int]]]:
    """把每卡 ``(P, F, 2)`` 的 ``(Σraw, n)`` 按 rank 序沿字段维拼齐后**按槽位序求和**。

    :param per_prompt_stats: 本卡每个条目的 ``{field: (Σraw, n)}``（= 该卡切片的
        ``group_field_stats`` 结果）。长度可 **小于 ``cutoff``**：缺的位置按**空统计**
        （全 0）参与会合 —— 与 :func:`gather_rollout_values` 的"补齐行"同义
        （值不参与真实判定，结果被调用方丢弃）。
    :param field_names: **全局并集**的字段名（⛔ 调用方必须让逐 rank 一致的 ``sorted`` 并集，
        否则字段维对不上）。
    :returns: 长度 ``cutoff`` 的列表，每项是该条目**整组**的 ``{field: (Σraw, n)}``
        （``n`` 为整数，且 = 跨卡整数之和 ⇒ 可用**整数严格**判据）。

    Raises:
        RuntimeError: 形状不符 / gather 长度不符。
    """
    import torch
    import torch.distributed as dist

    if cutoff < 0:
        raise RuntimeError(f"[rollout 分卡] cutoff 不能为负：{cutoff}")
    group, dp_size, device = resolve_dp_group(runtime)
    if group is None:
        return list(per_prompt_stats)
    if device is None:
        device = torch.device("cpu")

    rows: list[Any] = []
    for index in range(cutoff):
        stats = per_prompt_stats[index] if index < len(per_prompt_stats) else {}
        row = [
            # ★ fp64：`(Σraw, n)` 的载荷只有 `P·F·2·8` B（P=8,F=5 ⇒ 640 B），
            #   用 fp64 换取"本地切片 → 会合 → 整组"的**逐位可对拍**
            #   （fp32 会把 `raw` 截断，导致 A-5 的"与单卡一致"只能靠容差）。
            [float(stats[name][0]), float(stats[name][1])] if name in stats else [0.0, 0.0]
            for name in field_names
        ]
        rows.append(torch.tensor(row, dtype=torch.float64, device=device))
    field_count = max(1, len(field_names))
    # ★ 成员数同样取自**组本身**（与 `gather_rollout_values` 同口径）
    members = int(dist.get_world_size(group))
    if members != dp_size:
        raise RuntimeError(
            f"[rollout 分卡] {what}：会合组有 {members} 个成员，但 tp_state.dp_size="
            f"{dp_size} —— 两者不一致（字段维求和宽度与会合域不符）。"
        )
    stacked = torch.stack(rows, dim=0).reshape(cutoff, field_count, 2)
    gathered = [torch.empty_like(stacked) for _ in range(members)]
    dist.all_gather(gathered, stacked, group=group)
    if tuple(gathered[0].shape) != (cutoff, field_count, 2):
        raise RuntimeError(
            f"[rollout 分卡] {what}：会合后单卡形状应为 ({cutoff}, {field_count}, 2)，"
            f"实际 {tuple(gathered[0].shape)}。"
        )

    out: list[dict[str, tuple[float, int]]] = []
    for index in range(cutoff):
        merged: dict[str, tuple[float, int]] = {}
        for position, name in enumerate(field_names):
            raw_sum = 0.0
            count = 0
            for rank in range(members):  # ★ 按槽位序（rank 序）累加 ⇒ 加序确定
                cell = gathered[rank][index][position]
                raw_sum += float(cell[0].item())
                count += int(round(float(cell[1].item())))
            if count > 0 or raw_sum != 0.0:
                merged[name] = (raw_sum, count)
        out.append(merged)
    return out


def union_field_names(per_prompt_stats: list[dict[str, tuple[float, int]]]) -> list[str]:
    """全局字段名并集（**``sorted`` ⇒ 逐 rank 确定相同**）。

    ★ 为什么必须"逐 rank 相同"：字段维是 ``all_gather`` 的**形状**之一，
    两卡的字段名集合/顺序不同 ⇒ 集合失配。``sorted`` 是与 ``dict`` 插入序无关的确定序。
    """
    names: set[str] = set()
    for stats in per_prompt_stats:
        names.update(stats)
    return sorted(names)


__all__ = [
    "gather_field_stats",
    "gather_rollout_values",
    "resolve_dp_group",
    "union_field_names",
]
