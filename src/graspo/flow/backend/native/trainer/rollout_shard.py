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
| `gather_field_names`（字段维） | 每卡切片字段名标识符列表 | 每可训 attempt 1 次（dp≥2） |

★ **载荷里没有文本、没有标注**：只有标量与其计数 ⇒ 「文本/标注不出卡」这条红线由**载荷类型**保证，
不靠调用方自觉（β2 当年被否的理由正是"载荷是序列化 completion/标注"）。

★★ **边界澄清（U2）**：本节这条红线的**原意**是「**内容**不出卡」——即 completion 文本、
标注对象（``AnnotationResult`` / per-token 标签）不序列化过卡。唯一额外过卡的是
:func:`gather_field_names` 为求**全局字段并集**而交换的**字段名标识符列表**（短字符串，来自标注
的字段名，如 ``a`` / ``action_type``）：它**不是** completion 文本、**不是**标注对象，且**必须**
过卡——字段维是会合载荷的**形状**之一，各卡字段集不同时只有**按名对齐**才能让分卡路径与 dp=1
逐值相同（按槽位位置对齐会把对端字段的统计记到本卡字段名下 ⇒ 静默错 μ）。故本段把上一段
「只有标量与其计数」的**字面范围**明确扩展为「标量、计数、字段名标识符」——原红线不被违反，
边界在此具名，不靠调用方自觉。

★ **为什么不改用"预定义 canonical 字段序 ＋ 定长 bitmap"（无字符串）**：不存在可用的全局字段
词表。标注字段名**不是** targets 的子集——``json_labeler._check_key`` 在 GT 上下文为 ``None`` 时
**接受任意键**（``{"a": {"zzz": 1.0}}`` 对 targets ``{"a": 1.0}`` 会产出字段 ``zzz``；机核见本战役
工位 ``evidence/canonical_vocab_infeasible.txt``）。用 targets 派生词表做 bitmap 会**静默丢字段**
（⇒ 改 μ，禁止）；对未知名 fail-closed 则会让 dp≥2 拒绝 dp=1 能处理的合法输入（⇒ 违反
"分卡 == dp=1"）；改用名字哈希则碰撞即静默错值。⇒ 交换**短标识符**是本通道唯一无损且保契约的
方案。

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

    # ── ★ 空表边角（同族：`torch.stack` 吃空输入 ⇒ fail-closed ＋ 点名诊断）──────
    # `cutoff == 0` ⇒ 下面的 `torch.stack([...])` 只会抛一条与真因无关的
    # "stack expects a non-empty TensorList"。生产路径上 `cutoff` 由
    # `_collective_any_pending` 的 MAX 保证 ≥ 1（`trainer.py` 的 `if active:` 门）
    # ⇒ 这是**不变量违例**，⛔ 不用"返回空"糊过去（本函数的返回值长度语义是 `cutoff`，
    # 但会合载荷的行数同时是 `all_gather` 的**形状** —— 静默返回空等于把形状失配藏起来）。
    # ★ 守卫放在 dp≤1 短路**之后**：dp≤1 的行为（含 `cutoff == 0` 返回 `[]`）逐位不变。
    if cutoff == 0:
        raise RuntimeError(
            f"[rollout 分卡] {what}：cutoff=0（本次会合一个条目都没有）⇒ 会合载荷的行数为 0，"
            "`torch.stack` 无法构造（只会报 'stack expects a non-empty TensorList'）。"
            "生产路径上 cutoff 由 `_collective_any_pending` 的 MAX 保证 ≥ 1，故这是不变量违例。"
        )

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


def gather_field_names(
    runtime: Any,
    local_field_names: list[str],
    *,
    what: str = "μ_f 字段名",
) -> list[str]:
    """把各卡**本卡切片**的字段名并集成**逐 rank 一致的 ``sorted`` 全局并集**。

    ★ 缺陷背景（U2）：调用方原先只把 ``union_field_names(per_prompt)``（= **本卡切片**的并集）
    直接传给 :func:`gather_field_stats`，违反后者 ``:param field_names:`` 声明的「全局并集」
    契约。同基数、字段集不同的两卡（``rank0={a}`` vs ``rank1={b}``）会**静默算错** μ
    （对端字段被记到本卡字段名下）；基数不同时 ``all_gather`` 直接形状失配。本函数在
    **并集的求取处**做一次跨卡对象会合，恢复声明契约（⇒ 分卡路径 == dp=1 单卡路径）。

    :param local_field_names: 本卡切片内出现过的字段名（顺序任意，内部 ``sorted`` 去重）。
    :returns: **组内每个成员都相同**的 ``sorted`` 全局并集。``dp_size <= 1`` / 无 ``dp_group`` /
        进程组未初始化 ⇒ 直接返回本卡 ``sorted`` 并集（**0 次新增集合**，与
        :func:`resolve_dp_group` 的其余短路口径同族 ⇒ dp=1 路径逐位不变）。

    ★ **为什么保证"每 rank 同一个 sorted 全局并集"**：组内每个 rank 恰好发一次同一个集合
    （载荷 = 短字段名**标识符**列表，⛔ 不含 completion 文本、不含标注对象），
    ``all_gather_object`` 按 **rank 序**返回同一份成员列表；``set`` 并集与 ``sorted`` 都与
    dict 插入序无关、无随机性 ⇒ 各 rank 的结果**逐字节相同**。

    ★ **具名登记（为什么必须传名字／传的是什么／载荷多大）**：

    ① **为什么必须传名字**：字段维是会合载荷的**形状**之一；各卡字段集不同时，按**槽位位置**
       对齐会把对端字段的统计记到本卡字段名下（同基数不同名 ⇒ 静默错 μ；基数不同 ⇒
       ``all_gather`` 形状失配）。只有拿到**同一个 sorted 全局并集**才能让分卡路径与 dp=1
       单卡路径**逐值相同** ⇒ 别无选择地要交换"名字"。
    ② **传的是什么**：仅**短字段名标识符**（来自标注的字段名，如 ``a`` / ``action_type``）；
       ⛔ 不含 completion 文本、⛔ 不含标注对象（``AnnotationResult`` / per-token 标签）。
       模块 docstring「载荷里没有文本、没有标注」这条红线的**原意**是"**内容**不出卡"，本会合
       不违反它；边界已在模块 docstring 的「★★ 边界澄清（U2）」段具名登记，⛔ 不默默绕过。
    ③ **载荷多大**：每卡一个 ``list[str]``，元素数 = 本卡切片出现过的字段数（≪ `(P,F,2)` 主载荷，
       量级 O(F)），单元素为短标识符 ⇒ 与 ≈640 B/rank 的 ``field_stats`` 主载荷同量级、可忽略。

    ★ **为什么不改用"预定义 canonical 字段序 ＋ 定长 bitmap"（无字符串）**：不存在可用的全局字段
    词表 —— 标注字段名**不是** targets 的子集（``json_labeler._check_key`` 在 GT 上下文为 ``None``
    时接受任意键；机核 ``evidence/canonical_vocab_infeasible.txt``：``{"a": {"zzz": 1.0}}`` 对
    targets ``{"a": 1.0}`` 产出字段 ``zzz``）。用 targets 派生词表做 bitmap 会**静默丢字段**
    （⇒ 改 μ，禁止）；对未知名 fail-closed 会让 dp≥2 拒绝 dp=1 能处理的合法输入（⇒ 违反
    "分卡 == dp=1"）；用名字哈希则碰撞即静默错值。

    ★ **为什么不是"局部并集＋补零到固定宽度"/"按位置拼接不同名"**：前者需要魔数宽度且会静默
    丢字段，后者把"对端第 ``position`` 个字段"当成本卡第 ``position`` 个字段（字段名不同即
    算错）—— 两者都会得到**第三种错值**。本函数按**字段名**对齐，宽度真相源就是全局并集本身。

    Raises:
        RuntimeError: 会合组成员数（``dist.get_world_size(group)``）与 ``tp_state.dp_size``
            不一致（全局并集的求取域与会合域不符），或会合返回份数不符。
    """
    import torch.distributed as dist

    local_sorted = sorted({str(name) for name in local_field_names})
    group, dp_size, _device = resolve_dp_group(runtime)
    if group is None:
        # ★ dp ≤ 1：一个集合都不发（与 `gather_rollout_values` / `gather_field_stats` 同锚）。
        return local_sorted

    members = int(dist.get_world_size(group))
    if members != dp_size:
        raise RuntimeError(
            f"[rollout 分卡] {what}：会合组有 {members} 个成员，但 tp_state.dp_size="
            f"{dp_size} —— 两者不一致（全局字段并集的求取域与会合域不符）。"
        )
    gathered: list[list[str]] = [[] for _ in range(members)]
    dist.all_gather_object(gathered, local_sorted, group=group)
    if len(gathered) != members:
        raise RuntimeError(
            f"[rollout 分卡] {what}：全局字段名会合返回 {len(gathered)} 份，应为 {members} 份。"
        )
    names: set[str] = set()
    for peer_names in gathered:
        names.update(str(name) for name in peer_names)
    return sorted(names)


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
        否则字段维对不上）。★ **空**字段集 ⇒ 见下方 Raises：fail-closed，⛔ 不返回空统计。
    :returns: 长度 ``cutoff`` 的列表，每项是该条目**整组**的 ``{field: (Σraw, n)}``
        （``n`` 为整数，且 = 跨卡整数之和 ⇒ 可用**整数严格**判据）。

    Raises:
        RuntimeError: 形状不符 / gather 长度不符。
        RuntimeError: ``cutoff == 0``（空行列表）或 ``field_names`` 为空（空字段集）——
            两者都会让会合载荷无法构造，且都**点名**真因（⛔ 不静默返回空）。
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

    # ── ★ 空表边角（fail-closed ＋ 点名诊断，⛔ 不静默返回空）────────────────────
    # ★ 守卫放在 dp≤1 短路**之后**：dp≤1 时"本地值就是整组"、一个载荷都不构造
    #   ⇒ 空字段集在 dp≤1 下**原样返回**（A-1 回归锚，逐位不变）。这正是本缺陷
    #   只在 dp≥2 档中招、dp=1 档（T028/T040）不受影响的原因。
    if cutoff == 0:
        raise RuntimeError(
            f"[rollout 分卡] {what}：cutoff=0（本次会合一个条目都没有）⇒ 会合载荷的行数为 0，"
            "`torch.stack` 无法构造（只会报 'stack expects a non-empty TensorList'）。"
            "生产路径上 cutoff 由 `_collective_any_pending` 的 MAX 保证 ≥ 1，故这是不变量违例。"
        )
    if not field_names:
        # ★ T029 现场：rank0 走到的就是这一支。旧代码把行宽 0 用 `max(1, len(...))`
        #   掩盖成 1 ⇒ 报出与真因无关的 `shape '[1, 1, 2]' is invalid for input of size 0`。
        raise RuntimeError(
            f"[rollout 分卡] {what}：rank={int(dist.get_rank(group))} 本次 attempt"
            f"（cutoff={cutoff}，本卡 {len(per_prompt_stats)} 条统计）的**字段名并集为空**"
            " ⇒ 每个 row 是 0 元素张量、真行宽 = 0，与会合载荷的形状语义矛盾"
            "（旧代码用 `max(1, len(field_names))` 把行宽 0 掩盖成 1，故现场报的是"
            " 'shape \\'[1, 1, 2]\\' is invalid for input of size 0'）。"
            "成因：本卡切片内**一条 completion 都没有**被标出 VALUE span"
            "（`ripple.annotation.json_labeler` 的「乱码无结构起点」分支会把整条标成 WASTE"
            " ⇒ `group_field_stats` 返回 {}）；而 `field_names` 由**本卡切片**的"
            " `union_field_names(per_prompt)` 求并集（`rollout.py`）⇒ 各 rank 的字段集"
            "**不保证一致**。"
            "⛔ 本函数不按「空统计」糊过去：字段维同时是 `all_gather` 的**形状**之一，"
            "本 rank 空而对端非空必让会合失配；静默返回空还会把「本轮无字段」这条事实藏起来"
            "（宪法点名反对的静默跳过形态）。"
        )

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
    # ★ 形状真相源 = **行宽本身**：`field_names` 已由上面的守卫保证非空，
    #   ⛔ 不再用 `max(1, len(field_names))`（那会把"空字段集"掩盖成宽度 1 ⇒
    #   报出一条与真因无关的 reshape 错误，正是 T029 的现场）。
    field_count = len(field_names)
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
    "gather_field_names",
    "gather_field_stats",
    "gather_rollout_values",
    "resolve_dp_group",
    "union_field_names",
]
