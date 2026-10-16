"""U-A（WORLD 门节奏对齐）的零 GPU 判别力测试。

**背景**（真值链，逐行证据见工位 `plan.md` §0）：``_maybe_optimize`` 在
``local_wants_train`` 判定**之前**无条件发一次 WORLD ``all_reduce(MAX)``
（``optimize.py:75-82``，无 ``group=`` ⇒ ``default_pg``，与 K-会合的 ``dp_group``
**不是同一个 communicator**），而它是**每终结一条 prompt 调用一次**
（``trainer.py`` 的 ``for state in finished:`` → ``_finalize_sample`` →
``rollout.py:434``）。各 rank 分到的 prompt 数可差 1（``assign_whole_groups`` 只保证
**组数**差 ≤ 1）⇒ 门次数不等 ⇒ 集合调用次数不等 ⇒ **静默自旋**。

本文件锁死 U-A 的三条契约（全部零 GPU）：

1. **频率不变量**：一次 ``_sample_queue(8 条)`` ⇒ ``_maybe_optimize`` 被调用 **8 次**
   （= 复核裁定的"CPU 单测替代真机判别实验"；若实现成"每队列一次"会看到 1 次）；
   ★ 并且用**源码变体**（把门改成"每队列一次"）**真的跑红一次**，证明这条断言能失败。
2. **dp ≤ 1 零新增集合**：``dp_size == 1`` 时**不得**新增任何集合/补门
   （计数证明：``_collective_any_queues_pending`` 调用数 = 0、总门数 = prompt 数）。
3. **dp = 2 门次数逐 rank 相等**（真进程组 gloo 双 rank，零 GPU）：
   不等分片（``n=9`` vs ``n=8``、``Q=8``）下，逐轮门数序列逐 rank 相等、轮数相等、
   ``global_step``/``optimized_steps`` 相等；并给出**旧节奏**（不补门）的双进程
   **静默自旋**阳性对照（父进程按超时判红）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch", reason="U-A 判别测试需要 torch（CPU 即可）")

from graspo.flow.backend.native.trainer.trainer import GraspoFlowTrainer  # noqa: E402
from graspo.ripple.buffer import ReplayBuffer  # noqa: E402
from graspo.ripple.monitoring.stats import (  # noqa: E402
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
)

_REPO = Path(__file__).resolve().parents[5]
_TRAINER_SRC = (
    _REPO / "src" / "graspo" / "flow" / "backend" / "native" / "trainer" / "trainer.py"
)


# ── 假协作者（绕过 __init__：不建 runtime、不碰 GPU、不建真进程组）──────────────


class _FakeDegree:
    def __init__(self, dp_size: int, dp_rank: int = 0) -> None:
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.tp_size = 1
        self.pp_size = 1


class _FakeAdapter:
    def __init__(self, dp_size: int) -> None:
        self.device = torch.device("cpu")
        self.tp_state = SimpleNamespace(dp_size=dp_size, dp_group=None, device=self.device)


class _FakeRuntime:
    """最小鸭子类型：只提供 trainer 取数的字段与 `train_batch`。"""

    def __init__(self, *, dp_size: int = 1, dp_rank: int = 0) -> None:
        self.parallel_degree = _FakeDegree(dp_size, dp_rank)
        self.rank = dp_rank
        self.tp_rank = 0
        self._adapter = _FakeAdapter(dp_size)
        self.train_batch_calls: list[list[object]] = []

    def _require_adapter(self) -> _FakeAdapter:
        return self._adapter

    def is_primary(self) -> bool:
        return False

    def train_batch(self, data: list[object], **kwargs: object) -> dict[str, object]:
        # 只替换"训练本体"，门的判定/取数/清缓冲/自增全部走真代码（optimize.py 未改）。
        self.train_batch_calls.append(list(data))
        return {"loss_mean": 0.0, "grad_norm_mean": 0.0, "optimized": False}

    def close(self) -> None:  # pragma: no cover - 本文件不走到
        return None


def _fake_config(
    *, queue_size: int = 8, group_size: int = 8, threshold: int = 64, max_retries: int = 0
) -> SimpleNamespace:
    training = SimpleNamespace(
        rollout_queue_batch_size=queue_size,
        rollout_group_size=group_size,
        replay_buffer_optimize_threshold=threshold,
        rollout_max_retries=max_retries,
        gradient_accumulation_micro_batches=1,
        save_steps=0,
        save_checkpoint_time_period_minutes=0,
        output_dir="/tmp/graspo-ua-unused",
    )
    return SimpleNamespace(training=training, native=SimpleNamespace(micro_batch_size=group_size))


def _make_trainer(
    *,
    dp_size: int = 1,
    dp_rank: int = 0,
    queue_size: int = 8,
    threshold: int = 64,
    max_retries: int = 0,
) -> GraspoFlowTrainer:
    """仿 `tests/flow/backend/native/test_assignment.py:145-153` 的假 runtime 风格。

    绕过 `__init__`（不建 runtime / 不建日志目录 / 不碰 GPU），只填训练循环真正读的字段。
    """
    trainer = GraspoFlowTrainer.__new__(GraspoFlowTrainer)
    trainer.config = _fake_config(
        queue_size=queue_size, threshold=threshold, max_retries=max_retries
    )
    trainer.runtime = _FakeRuntime(dp_size=dp_size, dp_rank=dp_rank)
    trainer.replay_buffer = ReplayBuffer()
    trainer.stats = GraspoFlowTrainStats()
    trainer.current_epoch_stats = GraspoFlowEpochStats(epoch=0)
    trainer.recent_groups = deque(maxlen=50)
    trainer.pending_batch_attempts = []
    trainer.pending_batch_timings = []
    trainer.global_step = 0
    trainer.sample_index = 0
    trainer.total_samples = 0
    trainer.started_at = 0.0
    trainer._smoke_boundary = False
    trainer.backend_name = "native"
    trainer.logger = None
    return trainer


def _record(*, decision: str = "perfect_skip", retry: bool = False) -> SimpleNamespace:
    """`AttemptRecord` 的最小鸭子类型（走 `_finalize_sample` 的 perfect-skip 真分支）。"""
    return SimpleNamespace(
        decision=SimpleNamespace(
            should_retry=retry,
            should_train=False,
            decision=SimpleNamespace(value=decision),
        ),
        generation=None,
        rewards=[0.0] * 8,
        readable={
            "decision": decision,
            "rewards": [0.0] * 8,
            "content_scores": [0.0] * 8,
            "base_content_scores": [0.0] * 8,
        },
        timing={},
        retry_count=0,
    )


def _drive_real_sample_queue(trainer: GraspoFlowTrainer, *, n_prompts: int) -> mock.Mock:
    """跑**真** `_sample_queue`（含真 `_finalize_sample`/`_finish_sample_and_maybe_optimize`），
    只把 rollout 与重试轮次判定换成假件；返回被 `mock.patch.object` 包住的 `_maybe_optimize`。"""
    samples = [object() for _ in range(n_prompts)]
    records = [_record() for _ in samples]
    gate = mock.Mock(name="_maybe_optimize", return_value=False)
    with (
        mock.patch.object(trainer, "_rollout_queue_attempt", return_value=records),
        mock.patch.object(trainer, "_collective_any_pending", return_value=False),
        mock.patch.object(trainer, "_maybe_optimize", gate),
    ):
        trainer._sample_queue(samples, epoch=0)
    return gate


# ── 1) 频率不变量：每终结一条 prompt 一次门 ─────────────────────────────────


def test_gate_is_issued_once_per_finished_prompt() -> None:
    """★ CPU 单测替代真机判别实验（A-13）：8 条 prompt ⇒ 8 次 `_maybe_optimize`。

    **失败时看到什么**：调用数 = 1 ⇒ "每队列一次"读法成立（此时才需要真机 S=15 实验）。
    真实链条：`trainer.py:501-505 for state in finished:` → `_finalize_sample` →
    `rollout.py:434`（三条返回路径同一终点）→ `optimize.py:81` 的 WORLD 集合。
    """
    trainer = _make_trainer()
    gate = _drive_real_sample_queue(trainer, n_prompts=8)
    assert gate.call_count == 8, f"门次数 = {gate.call_count}（每个 prompt 恰好一次门）"


@pytest.mark.parametrize("n_prompts", [1, 2, 7, 8, 9])
def test_gate_count_equals_prompt_count(n_prompts: int) -> None:
    """门次数 == 本队列 prompt 数（补门算术 `Q − len(queue)` 依托的不变式）。"""
    trainer = _make_trainer()
    gate = _drive_real_sample_queue(trainer, n_prompts=n_prompts)
    assert gate.call_count == n_prompts


def test_retry_does_not_add_or_lose_gates() -> None:
    """★ 重试**不**改门次数：8 条里 2 条先 retry 一次再终结 ⇒ 仍是 8 次门。

    这同时锁死 `_sample_queue` 收尾新增的 `len(finished) == len(samples)` fail-closed 校验
    在"有重试"形态下成立（补门次数按 `len(queue)` 推算的前提）。
    """
    trainer = _make_trainer(max_retries=1)
    samples = [object() for _ in range(8)]
    first_pass = [_record(retry=(index < 2)) for index in range(8)]
    second_pass = [_record() for _ in range(2)]
    gate = mock.Mock(name="_maybe_optimize", return_value=False)
    with (
        mock.patch.object(
            trainer, "_rollout_queue_attempt", side_effect=[first_pass, second_pass]
        ),
        mock.patch.object(trainer, "_collective_any_pending", side_effect=[True, False]),
        mock.patch.object(trainer, "_maybe_optimize", gate),
    ):
        trainer._sample_queue(samples, epoch=0)
    assert gate.call_count == 8
    assert trainer.current_epoch_stats.samples_seen == 8


class _OverReportingList(list):
    """构造用容器：`__len__` 谎报 `+1`（模拟"本队列有条目凭空多出/丢失"的坏形态）。"""

    def __len__(self) -> int:  # type: ignore[override]
        return list.__len__(self) + 1


def test_finished_count_mismatch_fails_closed() -> None:
    """阳性对照：`finished` 与 `len(samples)` 不等 ⇒ **显式失败**（不是继续发门 ⇒ 集合错位）。

    `len(finished) == len(samples)` 是补门次数算术（`Q − len(queue)`）的依托；构造一个
    谎报长度的容器把它打破 ⇒ 必须在**任何门之前**判红（而不是默默把 WORLD 集合发错位）。
    """
    trainer = _make_trainer()
    samples = _OverReportingList([object() for _ in range(3)])
    records = [_record() for _ in range(3)]
    gate = mock.Mock(name="_maybe_optimize", return_value=False)
    with (
        mock.patch.object(trainer, "_rollout_queue_attempt", return_value=records),
        mock.patch.object(trainer, "_collective_any_pending", return_value=False),
        mock.patch.object(trainer, "_maybe_optimize", gate),
    ):
        with pytest.raises(RuntimeError, match="finished-count mismatch"):
            trainer._sample_queue(samples, epoch=0)
    assert gate.call_count == 0, "必须在发出第一个门之前失败（否则集合已错位）"


# ── 2) ★ 变异测试：把门改成"每队列一次"⇒ 主断言必须**真的红** ────────────────

_PER_PROMPT_LOOP = """        stop_requested = False
        for state in finished:
            if self._finalize_sample(state, epoch=epoch):
                stop_requested = True
        return stop_requested"""

_PER_QUEUE_LOOP = """        _real_gate = self._maybe_optimize
        self._maybe_optimize = lambda **kwargs: False
        for state in finished:
            self._finalize_sample(state, epoch=epoch)
        self._maybe_optimize = _real_gate
        return self._maybe_optimize(epoch=epoch)"""


def _load_per_queue_variant(tmp_path: Path):
    """把源码里的"逐 prompt 门"改成"每队列一次门"（复核者上一轮的错误读法）并加载成模块。

    变异是**源码文本级**的：保留 `_finalize_sample` 真链，只把"每终结一条 prompt 发一次门"
    换成"整队列发一次门"。⇒ 若主断言（== 8）写成"能失败的断言"，它在变体上必须判红。
    """
    source = _TRAINER_SRC.read_text(encoding="utf-8")
    assert _PER_PROMPT_LOOP in source, (
        "变异锚点未找到 —— `_sample_queue` 的收尾结构变了；本阳性对照必须同步重做"
        "（否则它会退化成永远通过的哑断言）"
    )
    mutated = source.replace(_PER_PROMPT_LOOP, _PER_QUEUE_LOOP, 1)
    path = tmp_path / "trainer_per_queue_variant.py"
    path.write_text(mutated, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_graspo_trainer_per_queue_variant", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_per_queue_implementation_makes_the_frequency_assertion_red(tmp_path: Path) -> None:
    """★ **阳性对照的真实红证**：错误的"每队列一次"实现 ⇒ 门计数 = 1（而非 8）。

    这条用例本身就是"主断言能失败"的机核证明：
    `assert count == 8` 在错误实现下 **确实** 判红（此处把红的结果固化为 `== 1`）。
    """
    variant = _load_per_queue_variant(tmp_path)
    trainer = variant.GraspoFlowTrainer.__new__(variant.GraspoFlowTrainer)
    trainer.config = _fake_config()
    trainer.runtime = _FakeRuntime(dp_size=1)
    trainer.replay_buffer = ReplayBuffer()
    trainer.stats = GraspoFlowTrainStats()
    trainer.current_epoch_stats = GraspoFlowEpochStats(epoch=0)
    trainer.recent_groups = deque(maxlen=50)
    trainer.pending_batch_attempts = []
    trainer.pending_batch_timings = []
    trainer.global_step = 0
    trainer.sample_index = 0
    trainer.total_samples = 0
    trainer.started_at = 0.0
    trainer._smoke_boundary = False
    trainer.backend_name = "native"
    trainer.logger = None
    samples = [object() for _ in range(8)]
    records = [_record() for _ in samples]
    gate = mock.Mock(name="_maybe_optimize", return_value=False)
    with (
        mock.patch.object(trainer, "_rollout_queue_attempt", return_value=records),
        mock.patch.object(trainer, "_collective_any_pending", return_value=False),
        mock.patch.object(trainer, "_maybe_optimize", gate),
    ):
        trainer._sample_queue(samples, epoch=0)
    # ⛔ 错误实现下的读数 = 1 ⇒ 主断言 `== 8` 会红（本行是红证的固定形态）
    assert gate.call_count == 1, f"错误实现下门次数 = {gate.call_count}"


# ── 3) dp ≤ 1：零新增集合 / 零补门（计数证明）───────────────────────────────


class _RecordingDist:
    """记录被调用的集合通信入口（只用于 dp ≤ 1 的"零新增"计数证明）。"""

    def __init__(self, *, initialized: bool = True) -> None:
        self.calls: list[str] = []
        self._initialized = initialized
        self.ReduceOp = SimpleNamespace(MAX="MAX", MIN="MIN")

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    def all_reduce(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        self.calls.append("all_reduce")

    def barrier(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        self.calls.append("barrier")


@pytest.mark.parametrize("dp_size", [1, 0])
def test_alignment_is_disabled_and_adds_no_collective_at_dp_le_1(
    dp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★ A-1a：`dp_size <= 1` ⇒ 对齐路径**关闭**、`_collective_any_queues_pending` **0 次集合**。"""
    import graspo.flow.backend.native.trainer.trainer as trainer_module

    fake_dist = _RecordingDist()
    monkeypatch.setattr(trainer_module, "dist", fake_dist)
    trainer = _make_trainer(dp_size=dp_size)
    assert trainer._world_gate_alignment_enabled() is False
    assert trainer._collective_any_queues_pending(True) is True
    assert trainer._collective_any_queues_pending(False) is False
    assert fake_dist.calls == [], f"dp≤1 不得新增集合，实际：{fake_dist.calls}"


def test_uninitialized_dist_also_disables_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """进程组未初始化（CPU 单测常态）⇒ 对齐关闭、本地回落。"""
    import graspo.flow.backend.native.trainer.trainer as trainer_module

    fake_dist = _RecordingDist(initialized=False)
    monkeypatch.setattr(trainer_module, "dist", fake_dist)
    trainer = _make_trainer(dp_size=8)
    assert trainer._world_gate_alignment_enabled() is False
    assert trainer._collective_any_queues_pending(True) is True
    assert fake_dist.calls == []


def test_dp2_enables_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """对照：`dp_size > 1` 且进程组已初始化 ⇒ 对齐路径开启（否则本单元是空操作）。"""
    import graspo.flow.backend.native.trainer.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "dist", _RecordingDist())
    assert _make_trainer(dp_size=2)._world_gate_alignment_enabled() is True


# ── 4) ⛔ 禁止写法模型：按本 rank 欠步数发门 ⇒ global_step 逐 rank 分叉 ───────


def test_owed_steps_variant_diverges_global_step() -> None:
    """★ A-4 的反向证明（模型级）：若按"本 rank 应欠的步数"计步 ⇒ `global_step` 分叉。

    场景取**真实触发形态** `S=17, m=2, Q=8, G=8, threshold=64`（`assign_whole_groups`
    给出 `[9,8]`）。两种实现的**算术**：

    - 正确（本包）：每个 rank 每 epoch 恰好 `ceil(max(len)/Q) * Q = 16` 次门，
      布尔由同一次 WORLD MAX 产生 ⇒ `global_step` 逐 rank 相等；
    - ⛔ 禁止（按欠步数）：每个 rank 只把自己应欠的步数计进 `global_step`
      ⇒ `9` vs `8` ⇒ **分叉**，污染 time-based checkpoint 门（`optimize.py:148-155`）、
      周期保存与 smoke 边界。

    ⚠ 口径声明：这是**算术模型**（不是真实现的运行），用于把"为什么禁止"变成可判红的数；
    真实现侧的相等性由下节的 gloo 双进程 e2e 给出。
    """
    shards = [9, 8]  # assign_whole_groups(17, 2)（本包实测）
    queue_size = 8
    fixed_gates_per_rank = -(-max(shards) // queue_size) * queue_size
    owed_gates_per_rank = [(-(-n // queue_size)) * queue_size for n in shards]
    # 正确实现：次数相同 ⇒ 同一次数目的 MAX 判决 ⇒ global_step 相同
    assert [fixed_gates_per_rank] * len(shards) == [16, 16]
    # ⛔ 禁止写法：分岔
    assert owed_gates_per_rank == [16, 8]
    assert len(set(owed_gates_per_rank)) == 2, "按欠步数发门必然让两端步数不同 ⇒ 判据红"


# ── 5) 真进程组（gloo 双 rank）：门次数逐 rank 相等 + 旧节奏静默自旋 ──────────

_GLOO_CHILD = textwrap.dedent(
    '''
    """gloo 双 rank 子进程：跑**真** `_run_aligned_sample_queues`（U-A 代码路径）。

    只替换 rollout 本体（`_sample_queue` → 逐 prompt 终结 + 逐 prompt 门），
    门的判定/取数/清缓冲/global_step 自增走**真** `_maybe_optimize`（optimize.py 未改）。
    """
    import json
    import sys
    from collections import deque
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from graspo.flow.backend.native.trainer.trainer import GraspoFlowTrainer
    from graspo.ripple.buffer import ReplayBuffer
    from graspo.ripple.monitoring.stats import GraspoFlowEpochStats, GraspoFlowTrainStats

    Q, G = 8, 8
    THRESHOLD = Q * G  # = core/schema.py:449-451 的 replay_buffer_optimize_threshold


    class _Degree:
        def __init__(self, dp_size, dp_rank):
            self.dp_size = dp_size
            self.dp_rank = dp_rank
            self.tp_size = 1
            self.pp_size = 1


    class _Adapter:
        def __init__(self, dp_size):
            self.device = torch.device("cpu")
            self.tp_state = SimpleNamespace(dp_size=dp_size, dp_group=None, device=self.device)


    class _Runtime:
        def __init__(self, dp_size, dp_rank):
            self.parallel_degree = _Degree(dp_size, dp_rank)
            self.rank = dp_rank
            self.tp_rank = 0
            self._adapter = _Adapter(dp_size)

        def _require_adapter(self):
            return self._adapter

        def is_primary(self):
            return False

        def train_batch(self, data, **kwargs):
            return {"loss_mean": 0.0, "grad_norm_mean": 0.0, "optimized": False}


    def _build(dp_size, dp_rank):
        training = SimpleNamespace(
            rollout_queue_batch_size=Q,
            rollout_group_size=G,
            replay_buffer_optimize_threshold=THRESHOLD,
            rollout_max_retries=0,
            gradient_accumulation_micro_batches=1,
            policy_ratio_clip_eps=0.2,
            max_grad_norm=1.0,
            save_steps=0,
            save_checkpoint_time_period_minutes=0,
            output_dir="/tmp/graspo-ua-e2e",
        )
        trainer = GraspoFlowTrainer.__new__(GraspoFlowTrainer)
        trainer.config = SimpleNamespace(
            training=training, native=SimpleNamespace(micro_batch_size=G)
        )
        trainer.runtime = _Runtime(dp_size, dp_rank)
        trainer.replay_buffer = ReplayBuffer()
        trainer.stats = GraspoFlowTrainStats()
        trainer.current_epoch_stats = GraspoFlowEpochStats(epoch=0)
        trainer.recent_groups = deque(maxlen=50)
        trainer.pending_batch_attempts = []
        trainer.pending_batch_timings = []
        trainer.global_step = 0
        trainer.sample_index = 0
        trainer.total_samples = 0
        trainer.started_at = 0.0
        trainer._smoke_boundary = False
        trainer.backend_name = "native"
        trainer.logger = None
        return trainer


    def main():
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        port = sys.argv[3]
        mode = sys.argv[4]
        out_path = sys.argv[5]
        n_prompts = int(sys.argv[6])
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}"
        )
        dp_size = 1 if mode == "dp1" else 2
        trainer = _build(dp_size, rank)
        pending = list(range(n_prompts))
        events = []
        if mode != "dp1":
            # rank1 前 2 条 prompt 记为 perfect_skip（不入 buffer）⇒ 缓冲增长节奏逐 rank 不同
            skip_first = 2 if rank == 1 else 0
        else:
            skip_first = 0

        real_gate = trainer._maybe_optimize

        def gate(*, epoch, force=False):
            trained = real_gate(epoch=epoch, force=force)
            events.append(("gate", bool(trained)))
            return trained

        trainer._maybe_optimize = gate

        def stub_sample_queue(samples, *, epoch):
            events.append(("queue", len(samples)))
            for index in range(len(samples)):
                if index >= skip_first:
                    trainer.replay_buffer.append_many([object()] * G)
                trainer._maybe_optimize(epoch=epoch)
            return False

        trainer._sample_queue = stub_sample_queue
        real_vote = trainer._collective_any_queues_pending

        def vote(has_more):
            events.append(("vote", bool(has_more)))
            return real_vote(has_more)

        trainer._collective_any_queues_pending = vote

        if mode in ("aligned", "dp1"):
            if mode == "dp1":
                for start in range(0, len(pending), Q):
                    stub_sample_queue(pending[start : start + Q], epoch=0)
            else:
                trainer._run_aligned_sample_queues(pending, epoch=0, queue_size=Q)
        elif mode == "owed":
            # ⛔ 禁止写法模型：只发"本 rank 自己欠的"门（不补齐到全局 max）⇒ 次数不等
            for start in range(0, len(pending), Q):
                stub_sample_queue(pending[start : start + Q], epoch=0)
        elif mode == "legacy":
            # 改动前的逐字结构：不补门、不投轮次判定
            for start in range(0, len(pending), Q):
                stub_sample_queue(pending[start : start + Q], epoch=0)
        else:  # pragma: no cover
            raise SystemExit(f"unknown mode {mode}")

        # 收口（真实代码：train() 正常退出 ⇒ `_teardown_after_train_loop` 的 WORLD barrier）。
        # legacy/owed 形态下这一步就是"静默自旋"的现场：一侧堵在多余的门上、另一侧堵在这里。
        dist.barrier()
        payload = {
            "rank": rank,
            "mode": mode,
            "aligned_enabled": trainer._world_gate_alignment_enabled(),
            "global_step": trainer.global_step,
            "optimized_steps": trainer.stats.optimized_steps,
            "events": [[kind, value] for kind, value in events],
        }
        dist.destroy_process_group()
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\\n")


    main()
    '''
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_gloo_pair(
    tmp_path: Path, *, mode: str, n_prompts: list[int]
) -> tuple[list, Path]:
    script = tmp_path / "gloo_child_ua.py"
    script.write_text(_GLOO_CHILD, encoding="utf-8")
    out_path = tmp_path / f"ua_{mode}_{n_prompts[0]}_{n_prompts[1]}.jsonl"
    out_path.unlink(missing_ok=True)  # 复跑同一 tmp 目录（如证据脚本）时不得累加旧行
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    procs = [
        subprocess.Popen(  # noqa: S603 - 固定 argv，无 shell
            [
                sys.executable,
                str(script),
                str(rank),
                "2",
                str(port),
                mode,
                str(out_path),
                str(n_prompts[rank]),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    return procs, out_path


def _collect_ok(procs: list, out_path: Path, *, timeout: int = 120) -> list[dict]:
    try:
        for proc in procs:
            _, stderr = proc.communicate(timeout=timeout)
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


def _gates_per_round(events: list) -> list[int]:
    """把 `[queue, gate..., vote]` 事件流切成逐轮门数（含补门）。

    轮边界 = `queue`（本 rank 本轮的真门区间）与 `vote`（轮次判定）；
    dp ≤ 1 形态没有 `vote` ⇒ 以相邻 `queue` 为边界（便于证明"没有补门"）。
    """
    per_round: list[int] = []
    current = 0
    open_round = False
    for kind, _value in events:
        if kind == "queue":
            if open_round:
                per_round.append(current)
                current = 0
            open_round = True
        elif kind == "vote":
            if open_round:
                per_round.append(current)
                current = 0
                open_round = False
        elif kind == "gate":
            current += 1
    if open_round:
        per_round.append(current)
    return per_round


def _real_gates_per_round(events: list) -> list[int]:
    per_round: list[int] = []
    for kind, value in events:
        if kind == "queue":
            per_round.append(int(value))
    return per_round


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="torch.distributed gloo 不可用"
)
def test_gloo_pair_unequal_shards_aligns_rhythm_and_global_step(tmp_path: Path) -> None:
    """★ A-2/A-3/A-4：不等分片（9 vs 8）下，对齐路径必须**逐 rank 同构且跑通**。

    断言（全部逐 rank 对比）：
    1. 轮数相等（`go_on` 判定生效）；
    2. **逐轮门数序列逐项相等**（= 复核建议"逐队列终结数序列逐个相等"在补门设计下的
       等价形式：每轮恰好 `Q` 次门）；
    3. `global_step` / `optimized_steps` 相等（每次门的布尔来自同一次 WORLD MAX）；
    4. 同时打印**真门数**序列（`[8,1]` vs `[8,0]`）——**它本身不可能逐 rank 相等**，
       这正是必须补门的原因。
    """
    procs, out_path = _launch_gloo_pair(tmp_path, mode="aligned", n_prompts=[9, 8])
    rows = _collect_ok(procs, out_path)
    per_rank_gates = [_gates_per_round(row["events"]) for row in rows]
    per_rank_reals = [_real_gates_per_round(row["events"]) for row in rows]
    assert per_rank_gates[0] == per_rank_gates[1] == [8, 8], per_rank_gates
    assert per_rank_reals == [[8, 1], [8, 0]], per_rank_reals
    assert rows[0]["global_step"] == rows[1]["global_step"] > 0
    assert rows[0]["optimized_steps"] == rows[1]["optimized_steps"] > 0
    assert rows[0]["aligned_enabled"] is rows[1]["aligned_enabled"] is True


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="torch.distributed gloo 不可用"
)
def test_gloo_pair_legacy_rhythm_deadlocks(tmp_path: Path) -> None:
    """★ A-2c 阳性对照（**真实失败**）：旧节奏（不补门）在不等分片下 ⇒ 静默自旋。

    一侧堵在多余的第 9 次门上、另一侧堵在收口 barrier 上 ⇒ 两侧**都不退出**
    （父进程按超时判红；这正是本单元要消灭的形态）。
    """
    procs, _out_path = _launch_gloo_pair(tmp_path, mode="legacy", n_prompts=[9, 8])
    try:
        time.sleep(15)
        alive = [proc.poll() is None for proc in procs]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for proc in procs:
            proc.communicate()
    # 机制预测：rank0 堵在第 9 次门上、rank1 堵在收口 barrier 上 ⇒ **两侧都活着**（互等）。
    assert alive == [True, True], f"旧节奏的存活形态与机制预测不符：alive={alive}"


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="torch.distributed gloo 不可用"
)
def test_gloo_pair_dp1_adds_no_alignment_collective(tmp_path: Path) -> None:
    """★ A-1a（真进程组侧）：进程组已初始化但 `dp_size == 1` ⇒ **零新增集合、零补门**。

    计数证据：总门数 == prompt 数（9，逐轮 `[8,1]`，没有补到 `Q`），
    且轮次判定 `_collective_any_queues_pending` **0 次**。
    """
    procs, out_path = _launch_gloo_pair(tmp_path, mode="dp1", n_prompts=[9, 9])
    rows = _collect_ok(procs, out_path)
    for row in rows:
        assert row["aligned_enabled"] is False
        assert _gates_per_round(row["events"]) == [8, 1], row["events"]
        assert _real_gates_per_round(row["events"]) == [8, 1]
        assert sum(1 for kind, _ in row["events"] if kind == "vote") == 0
        assert sum(1 for kind, _ in row["events"] if kind == "gate") == 9
