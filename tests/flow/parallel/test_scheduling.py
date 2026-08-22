"""PP 调度策略单元测试 — 验证 1F1B 的 fill/steady/drain 时序与回调顺序。

纯 CPU 逻辑测试：用 mock forward/backward 回调记录调用顺序，不触 GPU/分布式。
验证目标：
1. 各 stage 的 forward/backward 调用数正确（forward=num_chunks，backward=num_chunks）。
2. fill/steady/drain 触发的前向/反向先后顺序符合 1F1B（上游 stage 先 forward）。
3. backward 只在前向完成对应 chunk 后调用（依赖顺序正确）。
"""

from __future__ import annotations

from graspo.flow.parallel.scheduling import OneFOneBScheduler, build_scheduler


class _Recorder:
    """记录 forward/backward 调用的模拟器。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def forward(self, chunk_idx: int) -> None:
        self.calls.append(("forward", chunk_idx))

    def backward(self, chunk_idx: int) -> None:
        self.calls.append(("backward", chunk_idx))


def test_scheduler_one_f_one_b_pp2_rank0():
    """pp=2, pp_rank=0：fill 1 个 forward，steady 交错，drain 收尾。"""
    rec = _Recorder()
    sched = OneFOneBScheduler(
        pp_rank=0, pp_size=2, num_chunks=4,
        forward=rec.forward, backward=rec.backward,
    )
    stats = sched.run()

    forwards = [c for c in rec.calls if c[0] == "forward"]
    backwards = [c for c in rec.calls if c[0] == "backward"]
    assert [f[1] for f in forwards] == [0, 1, 2, 3], forwards
    assert [b[1] for b in backwards] == [0, 1, 2, 3], backwards
    # fill 在前
    assert rec.calls[0] == ("forward", 0), rec.calls[0]
    assert stats["pp_schedule"] == "one_f_one_b"
    assert stats["pipeline_fill_sec"] >= 0.0


def test_scheduler_one_f_one_b_pp2_rank1():
    """pp=2, pp_rank=1（末 stage）：fill 0 个 forward，steady 全量交错。"""
    rec = _Recorder()
    sched = OneFOneBScheduler(
        pp_rank=1, pp_size=2, num_chunks=4,
        forward=rec.forward, backward=rec.backward,
    )
    sched.run()

    forwards = [c for c in rec.calls if c[0] == "forward"]
    backwards = [c for c in rec.calls if c[0] == "backward"]
    assert [f[1] for f in forwards] == [0, 1, 2, 3], forwards
    assert [b[1] for b in backwards] == [0, 1, 2, 3], backwards
    # 末 stage：第一个调用就是 forward(0)（fill 为空）
    assert rec.calls[0] == ("forward", 0), rec.calls[0]


def test_scheduler_one_f_one_b_pp3():
    """pp=3：warmup 依 stage 递减（stage0=2, stage1=1, stage2=0）。"""
    for rank in range(3):
        rec = _Recorder()
        sched = OneFOneBScheduler(
            pp_rank=rank, pp_size=3, num_chunks=5,
            forward=rec.forward, backward=rec.backward,
        )
        sched.run()
        forwards = [c for c in rec.calls if c[0] == "forward"]
        backwards = [c for c in rec.calls if c[0] == "backward"]
        assert [f[1] for f in forwards] == list(range(5)), rec.calls
        assert [b[1] for b in backwards] == list(range(5)), rec.calls
        # 每个 stage 都是 forward 总数 = chunk_count；backward 总数 = chunk_count
        assert len(forwards) == 5
        assert len(backwards) == 5


def test_scheduler_factory_name():
    """工厂：按名构建 OneFOneB。"""
    from graspo.flow.parallel.scheduling import build_scheduler

    rec = _Recorder()
    sched = build_scheduler(
        "1f1b",
        pp_rank=0, pp_size=2, num_chunks=1,
        forward=rec.forward, backward=rec.backward,
    )
    assert isinstance(sched, OneFOneBScheduler)


def test_scheduler_factory_unknown_raises():
    """工厂：未知调度名报错。"""
    from graspo.flow.parallel.scheduling import build_scheduler

    rec = _Recorder()
    try:
        build_scheduler(
            "zero_bubble",
            pp_rank=0, pp_size=2, num_chunks=1,
            forward=rec.forward, backward=rec.backward,
        )
        raise AssertionError("expected ValueError for unknown scheduler")
    except ValueError:
        pass


def test_scheduler_factory_default_is_one_f_one_b():
    """工厂：默认构建 1F1B（OneFOneB，PP 的唯一调度策略）。"""
    rec = _Recorder()
    sched = build_scheduler(
        None,
        pp_rank=0, pp_size=2, num_chunks=2,
        forward=rec.forward, backward=rec.backward,
    )
    assert isinstance(sched, OneFOneBScheduler)
    sched.run()
    assert len(rec.calls) == 4  # 2 forward + 2 backward
