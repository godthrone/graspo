"""PipelineComm 单元测试 — 验证双向通道/校验/背压（纯 CPU mock，不触 GPU 与真进程组）。

设计目标（§1.3 计算与设施分离、§2.1 契约即防呆）：
1. forward-hidden 与 backward-grad 分别走 fwd_group / bwd_group，且 tag 区间正确。
2. tag 越界立即报错（而非 NCCL 静默挂起）。
3. max_inflight>0 时背压生效：超过窗口则等待最旧 send work。
4. recv 在无 CUDA stream 时退化为底层 work.wait()。
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from graspo.flow.parallel.pipeline_comm import PipelineComm


class _FakeWork:
    """模拟 dist.isend/irecv 返回的 work。"""

    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1


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


def _comm(chunk_count: int = 2, max_inflight: int = 0) -> PipelineComm:
    return PipelineComm(
        device=torch.device("cpu"),
        fwd_group=_FakeGroup("fwd"),
        bwd_group=_FakeGroup("bwd"),
        chunk_count=chunk_count,
        max_inflight=max_inflight,
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
