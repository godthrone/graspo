"""调度器插件扩展验证：通过注册表添加自定义 PipelineScheduler。

验证目标（C5 验收）：
1. 可以通过 _SCHEDULERS 注册表添加新的调度器实现
2. 新调度器通过 build_scheduler() 按名称构建
3. 不改 flow core 即可注册并运行
"""

import pytest

from graspo.flow.parallel.scheduling.base import PipelineScheduler
from graspo.flow.parallel.scheduling.factory import _SCHEDULERS, build_scheduler


class _MockPipelineScheduler(PipelineScheduler):
    """测试用自定义调度器：记录调用并返回固定统计。

    实现 PipelineScheduler ABC 的最小接口：仅需 run() 方法。
    """

    def run(self) -> dict:
        return {
            "pp_schedule": "mock",
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "num_chunks": self.num_chunks,
            "mock_called": True,
        }


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """测试后清理注册表，避免污染其他测试。"""
    yield
    _SCHEDULERS.pop("mock", None)


def _recorder():
    calls: list[tuple[str, int]] = []

    def forward(chunk_idx: int) -> None:
        calls.append(("forward", chunk_idx))

    def backward(chunk_idx: int) -> None:
        calls.append(("backward", chunk_idx))

    return calls, forward, backward


def test_register_mock_scheduler():
    """通过注册表添加自定义调度器。"""
    _SCHEDULERS["mock"] = _MockPipelineScheduler
    assert "mock" in _SCHEDULERS
    assert _SCHEDULERS["mock"] is _MockPipelineScheduler


def test_build_mock_scheduler_by_name():
    """build_scheduler("mock") 返回自定义调度器实例。"""
    _SCHEDULERS["mock"] = _MockPipelineScheduler
    calls, forward, backward = _recorder()

    sched = build_scheduler(
        "mock", pp_rank=0, pp_size=2, num_chunks=3, forward=forward, backward=backward
    )

    assert isinstance(sched, _MockPipelineScheduler)
    assert sched.pp_rank == 0
    assert sched.pp_size == 2
    assert sched.num_chunks == 3


def test_mock_scheduler_run_returns_stats():
    """自定义调度器 run() 返回的统计信息可被调用方读取。"""
    _SCHEDULERS["mock"] = _MockPipelineScheduler
    calls, forward, backward = _recorder()

    sched = build_scheduler(
        "mock", pp_rank=1, pp_size=4, num_chunks=5, forward=forward, backward=backward
    )
    stats = sched.run()

    assert stats["pp_schedule"] == "mock"
    assert stats["pp_rank"] == 1
    assert stats["pp_size"] == 4
    assert stats["num_chunks"] == 5
    assert stats["mock_called"] is True


def test_mock_scheduler_does_not_affect_builtin():
    """注册自定义调度器不影响内置调度器（one_f_one_b 仍可用）。"""
    _SCHEDULERS["mock"] = _MockPipelineScheduler
    calls, forward, backward = _recorder()

    # 内置调度器仍正常工作
    sched = build_scheduler(
        "one_f_one_b", pp_rank=0, pp_size=2, num_chunks=2, forward=forward, backward=backward
    )
    from graspo.flow.parallel.scheduling.one_f_one_b import OneFOneBScheduler

    assert isinstance(sched, OneFOneBScheduler)
    stats = sched.run()
    assert stats["pp_schedule"] == "one_f_one_b"


def test_unregistered_scheduler_name_raises():
    """未注册的调度器名报错（不注册 mock 时）。"""
    calls, forward, backward = _recorder()

    with pytest.raises(ValueError, match="mock"):
        build_scheduler(
            "mock", pp_rank=0, pp_size=2, num_chunks=1, forward=forward, backward=backward
        )
