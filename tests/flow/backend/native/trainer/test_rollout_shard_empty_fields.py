"""★ T029（GRASPO/LoRA/native/9B/dp_size=2）空表缺陷的**能失败的机核**。

现场（rank0 traceback 收敛到一行）：

    src/graspo/flow/backend/native/trainer/rollout_shard.py:208
    stacked = torch.stack(rows, dim=0).reshape(cutoff, field_count, 2)
    RuntimeError: shape '[1, 1, 2]' is invalid for input of size 0

真因不是"`rows` 列表为空"，而是 **`field_names` 为空**：每个 `row` 成了 0 元素张量，
而旧代码 `field_count = max(1, len(field_names))` 把"空字段集"**掩盖成宽度 1** ⇒
`reshape(1, 1, 2)` 需要 2 个元素、实际 0 个。`field_names` 由**本卡切片**的
`union_field_names(per_prompt)` 求并集（`rollout.py`），而 `group_field_stats` 只为
**标出了 VALUE span** 的字段记账 ⇒ 本卡切片内一条 completion 都没有 VALUE span 时并集为空
（`json_labeler` 的"乱码无结构起点"分支把整条标成 WASTE）。dp≤1 时
`gather_field_stats` 走短路（根本不构造载荷）⇒ 这就是"只有 dp≥2 档中招"的原因。

本文件钉住四条不变量（★ 每条都**能失败**，⛔ 无 skip/xfail）：

1. 空字段集 ⇒ **fail-closed 且点名诊断**（⛔ 不是静默返回空、也不再是旧 reshape 报错）；
2. `cutoff=0` ⇒ 同族 fail-closed（旧行为是 `stack expects a non-empty TensorList`，
   同样与真因无关）；
3. dp≤1 短路 **先于** 空表守卫 ⇒ 空字段集在 dp=1 下仍**原样返回本地值**（A-1 回归锚）；
4. **正向对照**：非空输入的合并结果**逐值不变**（dp=1 短路 + dp=2 脚本化对端）。

★ 空字段集的**来源**另有 `test_field_names_go_empty_when_no_completion_has_a_value_span`：
它直接跑真实 `annotate` → `group_field_stats` → `union_field_names` 链，把"为什么空"钉成机核，
而不是靠注释。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from graspo.flow.backend.native.trainer.rollout_shard import (
    gather_field_stats,
    gather_rollout_values,
    union_field_names,
)

# ★ 守卫必须点名的标记（⛔ 改动诊断文案时同步改这里，防"文案漂移 ⇒ 断言变哑"）
_EMPTY_FIELDS_MARK = "字段名并集为空"
# ★ dp≤1 短路的源码锚点（对齐 `test_rollout_sharding_gloo.py` 的锚点写法）
_DP_SHORTCUT_ANCHOR = "if dp_group is None or dp_size <= 1:"


class _Adapter:
    def __init__(self, dp_size: int) -> None:
        self.tp_state = SimpleNamespace(
            dp_size=dp_size, dp_group=object(), device=torch.device("cpu")
        )


class _Runtime:
    def __init__(self, dp_size: int) -> None:
        self._adapter = _Adapter(dp_size)

    def _require_adapter(self):
        return self._adapter


def _fake_dp(monkeypatch, *, dp_size: int, rank: int) -> None:
    """把 `torch.distributed` 伪装成"已初始化的 m 卡组"（不真起进程组）。

    `resolve_dp_group` 只需要 `is_available/is_initialized` 为真 ＋ `tp_state.dp_group`
    非空 ＋ `dp_size > 1`；本文件只在**空表**路径上用假进程组（真 gloo 双 rank 的
    加序/计数对照见 `test_rollout_sharding_gloo.py`）。
    """
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: dp_size)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 空字段集：fail-closed ＋ 点名诊断
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("rank", [0, 1])
def test_empty_field_names_fails_closed_with_a_named_diagnosis(monkeypatch, rank: int) -> None:
    """★ dp≥2 ＋ 空字段集 ⇒ 必须抛**点名空字段并集**的 RuntimeError。

    ★ 改前必红：旧实现抛的是 `shape '[1, 1, 2]' is invalid for input of size 0`
    （与真因无关）⇒ 断言 `_EMPTY_FIELDS_MARK in msg` 失败。
    """
    _fake_dp(monkeypatch, dp_size=2, rank=rank)
    with pytest.raises(RuntimeError) as excinfo:
        gather_field_stats(_Runtime(2), [{}], cutoff=1, field_names=[], what="μ_f 的 (Σraw, n)")

    msg = str(excinfo.value)
    # ★ 判别式 = "本模块自己发的点名诊断"（裸 torch 报错从不会带 `[rollout 分卡]` 前缀）。
    #   ⛔ 不用 `"invalid for input of size" not in msg`：新诊断**故意引用**旧现场报错串
    #   （便于把老 traceback 与本守卫对上），那样写会误判。
    assert msg.startswith("[rollout 分卡]"), f"必须是本模块的点名诊断，实际：{msg}"
    assert _EMPTY_FIELDS_MARK in msg, f"诊断必须点名「字段名并集为空」，实际：{msg}"
    assert "μ_f" in msg, f"诊断必须点名 what（哪一处会合），实际：{msg}"
    assert f"rank={rank}" in msg, f"诊断必须点名分片（rank），实际：{msg}"
    assert "VALUE span" in msg, f"诊断必须点名「为什么空」，实际：{msg}"


def test_empty_field_names_guard_is_not_a_silent_empty_return(monkeypatch) -> None:
    """★ ⛔ 不许"静默返回空统计"糊过去 —— 字段维是会合的形状之一。"""
    _fake_dp(monkeypatch, dp_size=2, rank=0)
    with pytest.raises(RuntimeError):
        gather_field_stats(_Runtime(2), [{}], cutoff=2, field_names=[], what="μ_f")
    # 反向断言：绝不能是"返回了空统计列表"
    try:
        got = gather_field_stats(_Runtime(2), [{}], cutoff=2, field_names=[], what="μ_f")
    except RuntimeError:
        got = None
    assert got is None, f"空字段集必须 fail-closed，⛔ 不许返回 {got!r}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 空行列表（cutoff=0）：同族 fail-closed
# ══════════════════════════════════════════════════════════════════════════════


def test_cutoff_zero_fails_closed_with_a_named_diagnosis(monkeypatch) -> None:
    """★ `torch.stack` 也吃空输入：`cutoff=0` 必须是**点名**的 fail-closed。

    ★ 改前必红：旧实现的报错是 `stack expects a non-empty TensorList`（与真因无关）。
    """
    _fake_dp(monkeypatch, dp_size=2, rank=0)
    with pytest.raises(RuntimeError) as excinfo:
        gather_field_stats(_Runtime(2), [], cutoff=0, field_names=["a"], what="μ_f 的 (Σraw, n)")
    msg = str(excinfo.value)
    assert msg.startswith("[rollout 分卡]"), f"必须是本模块的点名诊断，实际：{msg}"
    assert "cutoff=0" in msg, f"诊断必须点名 cutoff=0，实际：{msg}"


def test_gather_rollout_values_cutoff_zero_fails_closed_too(monkeypatch) -> None:
    """★ 同类（空表喂 `torch.stack`）在 `gather_rollout_values` 上同口径 fail-closed。"""
    _fake_dp(monkeypatch, dp_size=2, rank=0)
    with pytest.raises(RuntimeError) as excinfo:
        gather_rollout_values(_Runtime(2), [], cutoff=0, rollouts_per_rank=2, what="rewards")
    msg = str(excinfo.value)
    assert msg.startswith("[rollout 分卡]"), f"必须是本模块的点名诊断，实际：{msg}"
    assert "cutoff=0" in msg, f"诊断必须点名 cutoff=0，实际：{msg}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 放置约束：dp≤1 短路**先于**空表守卫（A-1 回归锚，逐位不变）
# ══════════════════════════════════════════════════════════════════════════════


def test_dp1_shortcut_still_returns_local_values_when_field_names_are_empty() -> None:
    """★ 空表守卫必须住在 dp≤1 短路**之后** ⇒ dp=1 行为逐位不变（含空字段集）。"""
    runtime = _Runtime(dp_size=1)
    runtime._adapter.tp_state.dp_group = None
    stats: list[dict[str, tuple[float, int]]] = [{}]
    assert gather_field_stats(runtime, stats, cutoff=1, field_names=[], what="μ_f") == stats
    assert gather_field_stats(_Runtime(dp_size=1), stats, cutoff=1, field_names=[]) == stats


def test_field_stats_dp_shortcut_anchor_is_still_present() -> None:
    """★ 防呆：上一条依赖的短路锚点必须仍在（否则它退化成哑断言）。"""
    import inspect

    from graspo.flow.backend.native.trainer import rollout_shard as _rs

    assert _DP_SHORTCUT_ANCHOR in inspect.getsource(_rs.resolve_dp_group), (
        "`resolve_dp_group` 的 dp≤1 锚点漂移 ⇒ 本文件的空表放置约束断言会静默失效"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. 正向对照：非空输入**逐值不变**
# ══════════════════════════════════════════════════════════════════════════════


def test_nonempty_field_stats_dp1_is_value_identical() -> None:
    """★ dp=1 短路：返回值 == 本地统计（逐值，含 `n` 的整数值）。"""
    stats = [{"a": (2.0, 4)}, {"b": (1.5, 3)}]
    got = gather_field_stats(_Runtime(dp_size=1), stats, cutoff=2, field_names=["a", "b"])
    assert got == stats


def test_nonempty_field_stats_dp2_merge_is_value_identical(monkeypatch) -> None:
    """★ dp=2 ＋ 非空字段集：合并结果**逐值**等于"两卡槽位序相加"的期望值。

    载荷用**脚本化对端**喂（rank0 = 本地，rank1 = 固定载荷）⇒ 期望值是硬编码的整数/浮点，
    改动形状或加序都会让这条判红。
    """
    _fake_dp(monkeypatch, dp_size=2, rank=0)
    peers = {
        # (Σraw, n)：本卡 a=(1.0, 2)；对端 b=(3.0, 1)
        0: torch.tensor([[[1.0, 2.0], [0.0, 0.0]]], dtype=torch.float64),
        1: torch.tensor([[[0.0, 0.0], [3.0, 1.0]]], dtype=torch.float64),
    }

    def scripted_all_gather(tensor_list, tensor, group=None):
        for index, target in enumerate(tensor_list):
            target.copy_(peers[index])

    monkeypatch.setattr(dist, "all_gather", scripted_all_gather)
    got = gather_field_stats(
        _Runtime(2), [{"a": (1.0, 2)}], cutoff=1, field_names=["a", "b"], what="μ_f"
    )
    assert got == [{"a": (1.0, 2), "b": (3.0, 1)}]
    # ★ `n` 必须是**整数**（跨卡整数之和 ⇒ 可用整数严格判据）
    assert isinstance(got[0]["a"][1], int) and isinstance(got[0]["b"][1], int)


# ══════════════════════════════════════════════════════════════════════════════
# 5. "为什么空"：空字段名来自"本卡切片一条 VALUE span 都没有"（真实标注链）
# ══════════════════════════════════════════════════════════════════════════════


def test_field_names_go_empty_when_no_completion_has_a_value_span() -> None:
    """★ 真因机核：真实 `annotate` → `group_field_stats` → `union_field_names`。

    乱码/无 JSON 结构起点的 completion ⇒ `json_labeler` 整条标 WASTE（J13 分支）⇒
    `group_field_stats` 返回 `{}` ⇒ 字段并集为空。合法围栏 JSON ⇒ 非空。
    ⇒ "本卡切片恰好全是没有字段的 completion"就是空字段集的**合法成因**，与
    replay buffer 阈值 / 生成侧派生式无关（见工位 report.md §7）。
    """
    from graspo.ripple.annotation.advantages import group_field_stats
    from graspo.ripple.annotation.labeler import AnnotationInput, annotate

    targets = [{"output": {"content": {"a": 1.0}}}]
    garbage = "sorry i cannot help with that"  # 无 JSON 结构起点 ⇒ 整条 WASTE
    good = '```json\n{"a": 1.0}\n```'

    def field_names_for(text: str) -> list[str]:
        ann = annotate(
            AnnotationInput(
                completion_text=text, targets=targets, tokenizer=None, format_type="json"
            )
        )
        stats = group_field_stats(
            completions=[text],
            annotations=[ann],
            targets=targets,
            tokenizer=None,
            format_type="json",
        )
        return union_field_names([stats])

    assert field_names_for(garbage) == [], "乱码 completion 必须不贡献任何字段（空字段集的来源）"
    assert field_names_for(good) == ["a"], "合法围栏 JSON 必须贡献字段（对照，证明断言不是哑的）"


# ══════════════════════════════════════════════════════════════════════════════
# 6. 阳性对照：**撤掉空字段集守卫** ⇒ 本文件第 1 节的断言必红
# ══════════════════════════════════════════════════════════════════════════════

# ★ 变体靠**源码文本锚点**切片构造（对齐 `test_rollout_sharding_gloo.py` 的
#   `broken_shortcut` 手法）：⛔ 不在生产源码里插测试标记，⛔ 不碰工作树。
_GUARD_START = "    if not field_names:"
_ROWS_ANCHOR = "\n    rows: list[Any] = []"
_WIDTH_LINE = "field_count = len(field_names)"


def _load_variant(tmp_path, name: str, source: str):
    """把变体落在**仓外临时目录**（⛔ 不在 `src/` 留未跟踪文件），再按路径加载。"""
    import importlib.util

    target = tmp_path / f"_{name}_rollout_shard.py"
    target.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_positive_control_removing_the_guard_brings_back_the_t029_crash(
    monkeypatch, tmp_path
) -> None:
    """★ 阳性对照：撤掉空字段集守卫 ⇒ 第 1 节的 fail-closed 断言**必红**。

    两个变体各自证明一件事：

    - **A（只撤守卫，保留诚实行宽）**：空字段集**静默返回 `[{}]`** ⇒ 第 1 节的两条
      "必须抛点名诊断" 断言变红 —— 这正是⛔不许出现的"静默返回空"形态，也说明
      "把 `max(1, ·)` 改成 `len(·)`" 与"加守卫"**必须成对出现**。
    - **B（撤守卫 ＋ 恢复旧行宽 `max(1, ·)`）**：报回 T029 现场原串
      `shape '[1, 1, 2]' is invalid for input of size 0` ⇒ 证明本修补与现场报错的映射。
    """
    from pathlib import Path

    from graspo.flow.backend.native.trainer import rollout_shard as _rs

    source = Path(_rs.__file__).read_text(encoding="utf-8")
    # ★ 锚点防呆：锚点漂移 ⇒ 变体替换会静默失效、阳性对照退化成哑断言。
    assert _GUARD_START in source, "守卫锚点漂移 ⇒ 阳性对照无效"
    assert _ROWS_ANCHOR in source, "行构造锚点漂移 ⇒ 阳性对照无效"
    assert _WIDTH_LINE in source, "行宽锚点漂移 ⇒ 阳性对照无效"

    start = source.index(_GUARD_START)
    end = source.index(_ROWS_ANCHOR, start)
    without_guard = source[:start] + source[end:]
    assert without_guard != source and _GUARD_START not in without_guard

    _fake_dp(monkeypatch, dp_size=2, rank=0)
    # 变体 A 会真的走到 `all_gather`（假 group 无法真会合）⇒ 用一个"两侧同载荷"的替身，
    # 让"静默返回空"这条退化路径能被观察到。
    def symmetric_all_gather(tensor_list, tensor, group=None):
        for target in tensor_list:
            target.copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", symmetric_all_gather)

    # ── 变体 A：只撤守卫 ⇒ 静默返回空（第 1 节断言在此必红）──
    variant_a = _load_variant(tmp_path, "variantA", without_guard)
    got = variant_a.gather_field_stats(_Runtime(2), [{}], cutoff=1, field_names=[], what="μ_f")
    assert got == [{}], f"撤掉守卫后必须退化成静默返回空（阳性对照的前提），实际 {got!r}"

    # ── 变体 B：撤守卫 ＋ 恢复 `max(1, ·)` ⇒ 复现 T029 现场原串 ──
    with_old_width = without_guard.replace(_WIDTH_LINE, "field_count = max(1, len(field_names))")
    assert with_old_width != without_guard
    variant_b = _load_variant(tmp_path, "variantB", with_old_width)
    with pytest.raises(RuntimeError) as excinfo:
        variant_b.gather_field_stats(_Runtime(2), [{}], cutoff=1, field_names=[], what="μ_f")
    assert "shape '[1, 1, 2]' is invalid for input of size 0" in str(excinfo.value), (
        f"阳性对照 B 必须复现现场原串，实际：{excinfo.value}"
    )

