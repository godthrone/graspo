"""U-B（`train_batch` 入口 fail-closed 守卫）的零 GPU 判别力测试。

**背景**（真值链，逐行证据见工位 `plan.md` §0）：决定死锁与否的是 **日程**
``len(indices)`` → ``num_micro_batches``（``training.py`` 内）→ ``_dp_sync_segment_cuts``
的 ``cuts``；而 ``dp_group`` 上的段末集合、R-B、step 门都按 ``cuts`` 的**迭代序号**发。
逐 rank 日程不同 ⇒ 集合错位 ⇒ 静默自旋（实测死锁对照：
``task-graspo-dp-seed-fix/evidence/deadlock_control_stdout.log:477,484``，
rank0 ``n_micro=1 len_indices=0`` vs rank1 ``n_micro=2 len_indices=2``）。

本文件锁死四条契约（全部零 GPU）：

1. **短路**：``tp_state is None`` / ``dp_group is None`` / ``dp_size <= 1`` / 进程组未初始化
   ⇒ **0 次新增集合**（单卡行为逐位不变的前提）；
2. **相等放行**：日程三量逐 rank 相同 ⇒ 不抛，集合恰为 ``MIN``+``MAX`` 两次；
3. **不等显式失败**：把"静默自旋"变成"一行报错"，且文案含**逐 rank 数值**；
4. ★ **NB-2**：断言对象必须是 ``[num_micro_batches, K, batch_size]`` 三元，**不是**
   ``len(indices)`` 代理量——给出"``len(indices)`` 相同、K 不同"的反例
   （此时 ``cuts`` 不同 ⇒ 会挂，而代理量守卫会**假通过**）。
   另加真进程组（gloo 双 rank）e2e：不等时**两侧一起报错**（而不是一侧报错、一侧自旋）。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="U-B 契约测试需要 torch（CPU 即可）")

import graspo.flow.backend.native.models.qwen35_36.training as training_module  # noqa: E402
from graspo.flow.backend.native.models.qwen35_36.training import (  # noqa: E402
    _DP_SCHEDULE_FIELDS,
    _assert_dp_schedule_consistent,
    _dp_sync_segment_cuts,
)

_REPO = Path(__file__).resolve().parents[5]
_TRAINING_SRC = (
    _REPO
    / "src"
    / "graspo"
    / "flow"
    / "backend"
    / "native"
    / "models"
    / "qwen35_36"
    / "training.py"
)


_MISSING = object()


class _FakeState:
    def __init__(self, *, dp_size: int = 2, dp_group: object | None = _MISSING) -> None:
        self.dp_size = dp_size
        # 不传 dp_group ⇒ 给一个非 None 的哑句柄；显式传 None ⇒ 保持 None（短路口径用例）
        self.dp_group = object() if dp_group is _MISSING else dp_group
        self.device = torch.device("cpu")


class _ScriptedDist:
    """按"逐 rank 行"脚本模拟 ``dp_group`` 上的 MIN/MAX/all_gather。

    ⛔ 不是恒真桩：``all_reduce`` 真的按 ``rows`` 做逐元素 min/max（与真实集合语义同构），
    因此"相等 ⇒ 放行 / 不等 ⇒ 判红"是被**算出来**的，不是写死的。
    """

    def __init__(
        self, rows: list[list[int]], *, my_index: int = 0, initialized: bool = True
    ) -> None:
        self.rows = [list(row) for row in rows]
        self.my_index = my_index
        self.calls: list[tuple[str, object]] = []
        self.ReduceOp = SimpleNamespace(MIN="MIN", MAX="MAX")
        self._initialized = initialized

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    def all_reduce(self, tensor, op=None, group=None) -> None:  # noqa: ANN001
        self.calls.append(("all_reduce", op))
        columns = list(zip(*self.rows, strict=True))
        if op == "MIN":
            values = [min(column) for column in columns]
        elif op == "MAX":
            values = [max(column) for column in columns]
        else:  # pragma: no cover - 防御
            raise AssertionError(f"unexpected op {op!r}")
        tensor.copy_(torch.tensor(values, dtype=tensor.dtype))

    def all_gather(self, tensor_list, tensor, group=None) -> None:  # noqa: ANN001
        self.calls.append(("all_gather", None))
        for slot, row in zip(tensor_list, self.rows, strict=True):
            slot.copy_(torch.tensor(row, dtype=slot.dtype))


def _guard(tp_state, **schedule) -> None:
    _assert_dp_schedule_consistent(
        tp_state,
        num_micro_batches=schedule.get("num_micro_batches", 8),
        dp_sync_segments=schedule.get("dp_sync_segments", 4),
        micro_batch_size=schedule.get("micro_batch_size", 8),
    )


# ── 1) 短路：单卡 / 无 dp_group / 未初始化 ⇒ 零新增集合 ─────────────────────


@pytest.mark.parametrize("dp_size", [1, 0])
def test_dp_size_le_1_short_circuits_with_zero_collectives(
    dp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★ 单卡（含 ``dp_size == 1`` 的 PP 档）⇒ 直接返回、**0 次新增集合**。

    即便"逐 rank 数值不同"也不得判红（此时根本不该有集合）。
    """
    fake = _ScriptedDist([[8, 4, 8], [9, 2, 4]], my_index=0)
    monkeypatch.setattr(training_module, "dist", fake)
    _guard(_FakeState(dp_size=dp_size))
    assert fake.calls == []


def test_missing_tp_state_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedDist([[8, 4, 8], [9, 2, 4]], my_index=0)
    monkeypatch.setattr(training_module, "dist", fake)
    _guard(None)
    assert fake.calls == []


def test_missing_dp_group_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedDist([[8, 4, 8], [9, 2, 4]], my_index=0)
    monkeypatch.setattr(training_module, "dist", fake)
    _guard(_FakeState(dp_size=2, dp_group=None))
    assert fake.calls == []


def test_uninitialized_process_group_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedDist([[8, 4, 8], [9, 2, 4]], my_index=0, initialized=False)
    monkeypatch.setattr(training_module, "dist", fake)
    _guard(_FakeState(dp_size=2))
    assert fake.calls == []


# ── 2) 相等 ⇒ 放行（不误拦）────────────────────────────────────────────────


def test_equal_schedule_passes_with_exactly_two_scalar_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _ScriptedDist([[8, 4, 8], [8, 4, 8]], my_index=0)
    monkeypatch.setattr(training_module, "dist", fake)
    _guard(_FakeState(dp_size=2))
    assert fake.calls == [("all_reduce", "MIN"), ("all_reduce", "MAX")]


def test_equal_schedule_holds_even_when_ranks_differ_in_other_ways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对照：只要**日程三量**相同就放行（守卫不越界去管别的量）。"""
    fake = _ScriptedDist([[1, 1, 1], [1, 1, 1], [1, 1, 1]], my_index=1)
    monkeypatch.setattr(training_module, "dist", fake)
    _guard(_FakeState(dp_size=3), num_micro_batches=1)
    assert [kind for kind, _ in fake.calls] == ["all_reduce", "all_reduce"]


# ── 3) 不等 ⇒ 显式失败 + 打印逐 rank 数值 ───────────────────────────────────


def test_unequal_num_micro_batches_raises_with_per_rank_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ A-2 的核心：人为让两 rank 条数不等（死锁对照的真实形态）⇒ **报错而不是自旋**。"""
    fake = _ScriptedDist([[1, 4, 8], [2, 4, 8]], my_index=0)
    monkeypatch.setattr(training_module, "dist", fake)
    with pytest.raises(RuntimeError) as excinfo:
        _guard(_FakeState(dp_size=2), num_micro_batches=1)
    message = str(excinfo.value)
    assert "日程不一致" in message
    for field in _DP_SCHEDULE_FIELDS:
        assert field in message
    assert "(1, 4, 8)" in message and "(2, 4, 8)" in message, message
    assert [kind for kind, _ in fake.calls] == ["all_reduce", "all_reduce", "all_gather"]


def test_unequal_micro_batch_size_also_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """日程三量里任一分量不等都要判红（同一个 `micro_batch_size` 也可能逐 rank 配置漂移）。"""
    fake = _ScriptedDist([[8, 4, 8], [8, 4, 16]], my_index=0)
    monkeypatch.setattr(training_module, "dist", fake)
    with pytest.raises(RuntimeError, match="日程不一致"):
        _guard(_FakeState(dp_size=2))


# ── 4) ★ NB-2：代理量（len(indices)）会假通过，三元不会 ─────────────────────


def test_k_divergence_is_caught_even_though_len_indices_is_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ NB-2 反例：``len(indices)`` 相同（64, ``micro_batch_size``=8 ⇒ ``nmb``=8）
    但 **K 不同** ⇒ ``cuts`` 不同 ⇒ 真会挂；而"只钉 ``len(indices)``"的守卫会**假通过**。

    本用例同时把两件事钉住：
    ① 三元守卫判红；② 日程本身真的不同（``cuts`` 不等）⇒ 不是形式主义。
    """
    lhs_cuts = _dp_sync_segment_cuts(8, 4)
    rhs_cuts = _dp_sync_segment_cuts(8, 2)
    assert lhs_cuts != rhs_cuts, "构造失效：K 不同必须给出不同 cut"
    assert len(lhs_cuts) == 3 and len(rhs_cuts) == 1

    fake = _ScriptedDist([[8, 4, 8], [8, 2, 8]], my_index=0)  # len(indices)=64 两侧相同
    monkeypatch.setattr(training_module, "dist", fake)
    with pytest.raises(RuntimeError, match="日程不一致"):
        _guard(_FakeState(dp_size=2))


def test_schedule_triple_is_what_decides_the_deadlock_not_len_indices() -> None:
    """★ 纯函数机核（无集合）：``cuts`` 只由 ``nmb``/K 决定，**不由** ``len(indices)`` 决定。

    构造：``len(indices)`` 都是 64（``nmb``=8）但 K 不同 ⇒ ``cuts`` 不同；
    反过来 ``len(indices)`` 不同（0 vs 1）但 ``nmb`` 相同（``max(1, ·)`` 钳到 1）⇒ ``cuts`` 相同。
    ⇒ 守卫的对象必须是日程三量（本包已按此实现），不能是条数。
    """
    batch_size = 8
    assert max(1, 64 // batch_size) == max(1, 64 // batch_size) == 8
    assert _dp_sync_segment_cuts(8, 4) != _dp_sync_segment_cuts(8, 2)
    assert max(1, 0 // batch_size) == max(1, 1 // batch_size) == 1
    assert _dp_sync_segment_cuts(1, 4) == _dp_sync_segment_cuts(1, 4) == ()


def test_guard_is_stricter_than_exact_cut_equality_by_design() -> None:
    """⚠ 口径登记（fail-closed 的代价）：守卫钉的是**日程三量**，比"集合日程真正相等"更严。

    反例（**按指挥口径保留为已知代价**）：``nmb=7`` vs ``8``、``K=4`` ⇒ ``cuts`` 相同
    ``(2, 4, 6)`` ⇒ 段末 `dp_seg_sync` 的次数与位置其实**能对上**（不会挂），
    但三量不等 ⇒ 守卫判红。

    为什么仍然这么钉：三量相等是**充分**条件（保证集合日程必然相同），fail-closed 的方向
    是"宁可显式失败，不要静默自旋"；而"直接断言 ``cuts`` 相等"虽然更精确，却要额外处理
    变长元组的集合化与 K/nmb 来源分叉问题（指挥口径明确要求钉三量）。
    """
    assert _dp_sync_segment_cuts(7, 4) == _dp_sync_segment_cuts(8, 4) == (2, 4, 6)
    assert max(1, 7 * 8 // 8) != max(1, 8 * 8 // 8)  # 三量确实不等 ⇒ 守卫会判红


# ── 5) 接线：守卫真的在 `train_batch` 里、且在 `cuts` 之前 ──────────────────


def _guard_call_block(source: str) -> str:
    needle = "_assert_dp_schedule_consistent(\n            self.tp_state,"
    assert needle in source, "`train_batch` 里没找到守卫调用（接线被删/挪了）"
    start = source.index(needle)
    return source[start : start + 400]


def test_train_batch_wires_the_guard_between_indices_and_cuts() -> None:
    """源码文本级接线断言：守卫必须在 ``num_micro_batches`` 之后、``cuts`` 之前，
    且以日程三量为实参（风格同仓库既有的源码文本级 parity 断言）。"""
    source = _TRAINING_SRC.read_text(encoding="utf-8")
    nmb_at = source.index("num_micro_batches = max(1, len(indices) // batch_size)")
    guard_at = source.index("_assert_dp_schedule_consistent(\n            self.tp_state,")
    cuts_at = source.index(
        "_dp_k_cuts = _dp_sync_segment_cuts(num_micro_batches, _dp_sync_segments)"
    )
    assert nmb_at < guard_at < cuts_at
    block = _guard_call_block(source)
    for kwarg in (
        "num_micro_batches=num_micro_batches",
        "dp_sync_segments=_dp_sync_segments",
        "micro_batch_size=batch_size",
    ):
        assert kwarg in block, f"守卫实参缺少 {kwarg}"
    # K 必须是**同一个**来源：上提的局部量同时喂给守卫与 cuts（不重复求值、不会两处分叉）
    assert "_resolve_dp_sync_segments()" in source[nmb_at:guard_at]
    assert source.count("_dp_sync_segment_cuts(num_micro_batches,") == 1


def test_guard_uses_dp_group_and_dp_size_le_1_shortcut_like_existing_helper() -> None:
    """短路口径与既有 `_dp_consistent_has_valid_micro_batches` **同源**（复用同一写法）。"""
    source = _TRAINING_SRC.read_text(encoding="utf-8")
    guard_src = source[
        source.index("def _assert_dp_schedule_consistent") : source.index(
            "class _Qwen35TrainingMethods"
        )
    ]
    assert 'getattr(tp_state, "dp_group", None) is None' in guard_src
    assert 'int(getattr(tp_state, "dp_size", 1)) <= 1' in guard_src
    assert "group=tp_state.dp_group" in guard_src
    assert "op=dist.ReduceOp.MIN" in guard_src and "op=dist.ReduceOp.MAX" in guard_src


# ── 6) 真进程组（gloo 双 rank）：两侧一起报错，而不是一侧报错一侧自旋 ─────────

_GLOO_CHILD = textwrap.dedent(
    '''
    """gloo 双 rank 子进程：真进程组上调同一个 fail-closed 守卫。"""
    import json
    import sys
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from graspo.flow.backend.native.models.qwen35_36.training import (
        _assert_dp_schedule_consistent,
    )

    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    port = sys.argv[3]
    mode = sys.argv[4]
    out_path = sys.argv[5]
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}"
    )
    schedule = {"num_micro_batches": 8, "dp_sync_segments": 4, "micro_batch_size": 8}
    if mode == "unequal_nmb" and rank == 1:
        schedule["num_micro_batches"] = 9
    if mode == "k_divergence" and rank == 1:
        schedule["dp_sync_segments"] = 2
    tp_state = SimpleNamespace(
        dp_group=dist.group.WORLD, dp_size=2, device=torch.device("cpu")
    )
    raised = False
    message = ""
    try:
        _assert_dp_schedule_consistent(tp_state, **schedule)
    except RuntimeError as exc:
        raised = True
        message = str(exc)
    # ★ 关键：两侧都活着走到收口 barrier。若判据是"只看本地"的，一侧会在这里静默自旋。
    dist.barrier()
    dist.destroy_process_group()
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"rank": rank, "mode": mode, "raised": raised, "message": message})
            + "\\n"
        )
    '''
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_gloo_pair(tmp_path: Path, mode: str) -> list[dict]:
    script = tmp_path / "gloo_child_ub.py"
    script.write_text(_GLOO_CHILD, encoding="utf-8")
    out_path = tmp_path / f"ub_{mode}.jsonl"
    out_path.unlink(missing_ok=True)  # 复跑同一 tmp 目录（如证据脚本）时不得累加旧行
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    procs = [
        subprocess.Popen(  # noqa: S603 - 固定 argv，无 shell
            [sys.executable, str(script), str(rank), "2", str(port), mode, str(out_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    try:
        for proc in procs:
            _, stderr = proc.communicate(timeout=120)
            assert proc.returncode == 0, stderr
    finally:
        for proc in procs:
            if proc.poll() is None:  # pragma: no cover - 兜底，避免留下子进程
                proc.kill()
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2, rows
    return sorted(rows, key=lambda row: row["rank"])


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="torch.distributed gloo 不可用"
)
def test_gloo_pair_equal_schedule_passes_on_both_ranks(tmp_path: Path) -> None:
    rows = _run_gloo_pair(tmp_path, "equal")
    assert [row["raised"] for row in rows] == [False, False], rows


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="torch.distributed gloo 不可用"
)
def test_gloo_pair_unequal_schedule_fails_on_both_ranks(tmp_path: Path) -> None:
    """★ A-2 阳性对照（真进程组）：人为让条数不等 ⇒ **两侧一起显式失败**（不是自旋）。"""
    rows = _run_gloo_pair(tmp_path, "unequal_nmb")
    assert [row["raised"] for row in rows] == [True, True], rows
    assert rows[0]["message"] == rows[1]["message"], "两侧报错文案必须一致（同一次归约）"
    assert "(8, 4, 8)" in rows[0]["message"] and "(9, 4, 8)" in rows[0]["message"]


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="torch.distributed gloo 不可用"
)
def test_gloo_pair_k_divergence_fails_on_both_ranks(tmp_path: Path) -> None:
    """★ NB-2 的真进程组侧：``len(indices)`` 相同、K 不同 ⇒ 两侧一起判红。"""
    rows = _run_gloo_pair(tmp_path, "k_divergence")
    assert [row["raised"] for row in rows] == [True, True], rows
    assert "(8, 4, 8)" in rows[0]["message"] and "(8, 2, 8)" in rows[0]["message"]
