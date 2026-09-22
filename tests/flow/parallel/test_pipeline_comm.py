"""PipelineComm 单元测试 — 验证双向通道/校验/背压（纯 CPU mock，不触 GPU 与真进程组）。

设计目标（§1.3 计算与设施分离、§2.1 契约即防呆）：
1. forward-hidden 与 backward-grad 分别走 fwd_group / bwd_group，且 tag 区间正确。
2. tag 越界立即报错（而非 NCCL 静默挂起）。
3. max_inflight>0 时背压生效：超过窗口则等待最旧 send work。
4. recv 在无 CUDA stream 时退化为底层 work.wait()。
5. **有界等待（缺陷 P6 的 a1）**：``wait_timeout_s`` 未配置时逐字旧行为（
   ``work.wait()`` 不带参）；配置后把 ``timedelta`` 透传下去，超时抛具名
   ``PipelineP2PTimeoutError``（含 direction/peer/shape）。**真进程组的端到端
   版本**在 ``tests/flow/parallel/test_pp_rendezvous_e2e.py``（gloo/NCCL）。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
import torch

from graspo.flow.parallel.pipeline_comm import (
    PipelineComm,
    PipelineP2PTimeoutError,
    bounded_broadcast,
    pp_rendezvous_timeout,
)


class _FakeWork:
    """模拟 dist.isend/irecv 返回的 work。"""

    def __init__(self, *, on_wait: Any | None = None) -> None:
        self.wait_calls = 0
        self.wait_args: list[Any] = []
        self._on_wait = on_wait

    def wait(self, timeout: Any = None) -> Any:
        self.wait_calls += 1
        self.wait_args.append(timeout)
        if self._on_wait is not None:
            return self._on_wait()
        return True


class _FakeGroup:
    """模拟 dist.ProcessGroup。"""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def fake_dist(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Patch dist.isend/irecv 为记录调用 + 返回 _FakeWork，返回 send/recv 调用列表。"""
    import graspo.flow.parallel.pipeline_comm as pc

    send_calls: list[dict[str, Any]] = []
    recv_calls: list[dict[str, Any]] = []

    def fake_isend(tensor: torch.Tensor, dst: int, group: Any, tag: int) -> _FakeWork:
        send_calls.append({"tensor": tensor, "dst": dst, "group": group, "tag": tag})
        return _FakeWork()

    def fake_irecv(tensor: torch.Tensor, src: int, group: Any, tag: int) -> _FakeWork:
        recv_calls.append({"tensor": tensor, "src": src, "group": group, "tag": tag})
        return _FakeWork()

    monkeypatch.setattr(pc.dist, "isend", fake_isend)
    monkeypatch.setattr(pc.dist, "irecv", fake_irecv)
    return send_calls, recv_calls


def _comm(
    chunk_count: int = 2, max_inflight: int = 0, wait_timeout_s: float | None = None
) -> PipelineComm:
    return PipelineComm(
        device=torch.device("cpu"),
        fwd_group=_FakeGroup("fwd"),
        bwd_group=_FakeGroup("bwd"),
        chunk_count=chunk_count,
        max_inflight=max_inflight,
        wait_timeout_s=wait_timeout_s,
    )


def test_directions_use_correct_group_and_tag(fake_dist) -> None:
    send_calls, recv_calls = fake_dist
    comm = _comm(chunk_count=3)

    # forward-hidden：fwd_group，tag∈[0,3)
    h = torch.randn(1, 4, 8)
    comm.fwd_send(h, dst=1, tag=2)
    out_recv = torch.empty_like(h)
    comm.fwd_recv(out_recv, src=0, tag=1)

    # backward-grad：bwd_group，tag∈[3,6)
    g = torch.randn(1, 4, 8)
    comm.bwd_send(g, dst=0, tag=4)
    comm.bwd_recv(out_recv, src=1, tag=3)

    fwd_group = send_calls[0]["group"]
    bwd_group = send_calls[1]["group"]
    assert fwd_group.name == "fwd"
    assert bwd_group.name == "bwd"
    # send direction-group mapping
    assert recv_calls[0]["group"].name == "fwd"
    assert recv_calls[1]["group"].name == "bwd"
    # tag values forwarded as-is
    assert send_calls[0]["tag"] == 2
    assert send_calls[1]["tag"] == 4
    assert recv_calls[0]["tag"] == 1
    assert recv_calls[1]["tag"] == 3


def test_tag_out_of_range_raises(fake_dist) -> None:
    _ = fake_dist
    comm = _comm(chunk_count=2)
    h = torch.randn(1, 4, 8)
    # fwd tag must be in [0,2)
    with pytest.raises(RuntimeError, match="fwd tag out of range"):
        comm.fwd_send(h, dst=1, tag=2)
    # bwd tag must be in [2,4)
    with pytest.raises(RuntimeError, match="bwd tag out of range"):
        comm.bwd_send(h, dst=0, tag=4)  # 4 在 [2,4) 之外


def test_max_inflight_backpressure_waits_oldest(fake_dist) -> None:
    send_calls, _ = fake_dist
    comm = _comm(chunk_count=4, max_inflight=1)
    h = torch.randn(1, 4, 8)
    # 第一个 send 进入窗口
    first = comm.fwd_send(h, dst=1, tag=0)
    assert first.wait_calls == 0
    # 第二个 send 触发背压：等待最旧的 first 完成
    second = comm.fwd_send(h, dst=1, tag=1)
    assert first.wait_calls == 1
    assert second.wait_calls == 0


def test_recv_wait_uses_work_wait_on_cpu(fake_dist) -> None:
    _, recv_calls = fake_dist
    comm = _comm(chunk_count=2)
    h = torch.randn(1, 4, 8)
    out = torch.empty_like(h)
    handle = comm.fwd_recv(out, src=0, tag=0)
    comm.wait(handle)
    # CPU（无 CUDA stream）下退化为底层 irecv work.wait()
    assert recv_calls[0] is not None
    assert handle._work.wait_calls == 1


def test_comm_supports_chunk_count_zero_no_validation(fake_dist) -> None:
    send_calls, _ = fake_dist
    comm = _comm(chunk_count=0)
    h = torch.randn(1, 4, 8)
    # chunk_count=0：跳过 tag 区间校验（兼容旧调用）
    comm.fwd_send(h, dst=1, tag=999)
    assert send_calls[0]["tag"] == 999


# ── 有界等待（缺陷 P6 的 a1）────────────────────────────────────────────────


def test_timeout_conversion_is_the_single_source_of_truth() -> None:
    """``秒 → timedelta`` 的转换只有一个入口；0/None ⇒ 不设超时。"""
    assert pp_rendezvous_timeout(None) is None
    assert pp_rendezvous_timeout(0) is None
    assert pp_rendezvous_timeout(-1) is None
    assert pp_rendezvous_timeout(5) == timedelta(seconds=5)
    assert pp_rendezvous_timeout(2.5) == timedelta(seconds=2.5)


def test_unconfigured_timeout_keeps_legacy_wait_byte_identical(fake_dist) -> None:
    """**pp_size=1 与既有档位的保护**：未配置超时时 ``work.wait()`` 不带参。"""
    _, recv_calls = fake_dist
    comm = _comm(chunk_count=1, wait_timeout_s=None)
    send_work = comm.fwd_send(torch.randn(1, 4, 8), dst=1)
    comm.wait(send_work)
    assert send_work.wait_calls == 1
    assert send_work.wait_args == [None], "旧行为要求 wait() 不收任何参数"
    # recv 侧同样退化到 work.wait()（CPU 无 stream）
    out = torch.empty(1, 4, 8)
    handle = comm.fwd_recv(out, src=0)
    comm.wait(handle, label="legacy.recv")
    assert handle._work.wait_args == [None]
    assert recv_calls[0] is not None


def test_configured_timeout_is_forwarded_as_timedelta(fake_dist) -> None:
    comm = _comm(chunk_count=1, wait_timeout_s=7)
    work = comm.fwd_send(torch.randn(1, 4, 8), dst=1)
    comm.wait(work, label="bounded.send")
    assert work.wait_args == [timedelta(seconds=7)]


def test_send_timeout_raises_named_error_with_context(fake_dist) -> None:
    """超时必须是**具名异常**，且带上 direction/peer/shape（判据侧要能分类）。"""
    comm = _comm(chunk_count=1, wait_timeout_s=3)
    work = comm.fwd_send(torch.randn(2, 3), dst=1)

    def timed_out(timeout: Any = None) -> Any:
        raise RuntimeError(f"NCCL work timed out after {timeout}")

    work._on_wait = timed_out
    with pytest.raises(PipelineP2PTimeoutError) as info:
        comm.wait(work, label="pp_rollout.prefill.send")
    err = info.value
    assert err.label == "pp_rollout.prefill.send"
    assert err.direction == "fwd_send"
    assert err.peer == "dst=1"
    assert err.shape == (2, 3)
    assert err.timeout_s == 3
    assert "PP rendezvous timeout" in str(err)


def test_wait_returning_false_is_a_timeout(fake_dist) -> None:
    """torch 的 ``Work.wait(timeout)`` 也可能以返回值表达未完成 ⇒ 同样必须报错。"""
    comm = _comm(chunk_count=1, wait_timeout_s=1)
    work = comm.fwd_send(torch.randn(1, 4, 8), dst=1)
    work._on_wait = lambda: False
    with pytest.raises(PipelineP2PTimeoutError, match="fwd_send"):
        comm.wait(work, label="bounded.false")


def test_recv_timeout_raises_named_error_on_cpu_path(fake_dist) -> None:
    comm = _comm(chunk_count=1, wait_timeout_s=2)
    out = torch.empty(1, 4, 8)
    handle = comm.fwd_recv(out, src=0)

    def timed_out(timeout: Any = None) -> Any:
        raise RuntimeError("recv timed out")

    handle._work._on_wait = timed_out
    with pytest.raises(PipelineP2PTimeoutError) as info:
        comm.wait(handle, label="pp_rollout.prefill.recv")
    assert info.value.direction == "fwd_recv"
    assert info.value.peer == "src=0"
    assert info.value.timeout_s == 2


def test_non_timeout_exception_is_not_masked(fake_dist) -> None:
    """只有"确认是超时"的异常才翻译成具名类型；其它错误原样抛出（不掩盖首错）。"""
    comm = _comm(chunk_count=1, wait_timeout_s=5)
    work = comm.fwd_send(torch.randn(1, 4, 8), dst=1)

    def boom(timeout: Any = None) -> Any:
        raise ValueError("shape mismatch, not a timeout")

    work._on_wait = boom
    with pytest.raises(ValueError, match="not a timeout"):
        comm.wait(work, label="bounded.other")


def test_bounded_broadcast_uses_async_op_and_raises(monkeypatch) -> None:
    """a2：有界广播用 ``async_op=True`` + 有界 wait；未配置超时时逐字旧行为。"""
    import graspo.flow.parallel.pipeline_comm as pc

    calls: list[dict[str, Any]] = []

    class _Handle:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail

        def wait(self, timeout: Any = None) -> Any:
            calls.append({"phase": "wait", "timeout": timeout})
            if self.fail:
                raise RuntimeError(f"broadcast timed out ({timeout})")
            return True

    def fake_broadcast(tensor: Any, *, src: int, group: Any, async_op: bool = False) -> Any:
        calls.append({"phase": "broadcast", "src": src, "async_op": async_op})
        return _Handle(fail=async_op)

    monkeypatch.setattr(pc.dist, "broadcast", fake_broadcast)
    group = _FakeGroup("pp")

    # 未配置超时 ⇒ 同步调用，逐字旧行为
    t = torch.zeros(2, dtype=torch.long)
    bounded_broadcast(t, src=1, group=group, timeout_s=0, label="pp.token")
    assert calls[-1] == {"phase": "broadcast", "src": 1, "async_op": False}

    calls.clear()
    with pytest.raises(PipelineP2PTimeoutError) as info:
        bounded_broadcast(t, src=1, group=group, timeout_s=4, label="pp.token.step=1")
    assert calls[0] == {"phase": "broadcast", "src": 1, "async_op": True}
    assert calls[1]["timeout"] == timedelta(seconds=4)
    assert info.value.direction == "pp_group_broadcast"
    assert info.value.peer == "src=1"
    assert info.value.label == "pp.token.step=1"
