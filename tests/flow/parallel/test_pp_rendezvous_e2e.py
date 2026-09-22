"""PP 会合序列的**真进程组**端到端测试（缺陷 P6 的判别力证据 + 冒烟）。

与 ``test_pipeline_comm.py``（纯 CPU mock）不同，本文件用**两个真进程**跑
``PipelineComm`` 的真实 send/recv 与 ``bounded_broadcast``，覆盖三件事：

1. **PP rollout 会合的快乐路径**：多步 ``fwd_send``/``fwd_recv`` + ``pp_group``
   广播都正确完成（这是 T035/T036 走的那条序列的最小复现）；
2. **有界性（本包的核心交付）**：故意制造"对端不发匹配 recv"的错配 ⇒
   必须在一个 timeout 内抛具名 ``PipelineP2PTimeoutError``；
   修复前这里会**无界挂死**，所以父进程设了硬墙钟——超时即判失败并杀进程，
   绝不把 CI 挂住；
3. 后端参数化：``gloo`` 在**无 GPU 也能跑**（本机即可验证机制），
   ``nccl`` 在 ≥2 卡时自动启用（228 容器内的真后端覆盖）。**需要 2 卡的用例
   已在 skipif 里标清。**

为什么用 ``subprocess`` 而不是 ``multiprocessing.spawn``：本机沙箱禁用
posix semaphore（``PermissionError: [Errno 13]``），而 ``subprocess`` 不受影响；
且子进程独立退出更贴合"俩 rank 一个容器"的真实形态。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PP rendezvous e2e needs torch")
if not torch.distributed.is_available():  # pragma: no cover - 环境性
    pytest.skip("torch.distributed unavailable", allow_module_level=True)

#: 有界等待的测试超时（秒）。取小值让测试快，同时远大于真机上的会合耗时。
BOUNDED_TIMEOUT_SEC = 5.0

#: 父进程墙钟余量（秒）：超过 ``BOUNDED_TIMEOUT_SEC + 这个值`` 仍未退出 ⇒ 判无界挂死。
_PARENT_SLACK_SEC = 60.0

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"

#: 子进程脚本（两个 rank 都跑同一段，用 argv 区分）。
_CHILD_SCRIPT = r"""
import os, sys, time

import torch
import torch.distributed as dist

from graspo.flow.parallel.pipeline_comm import PipelineComm, bounded_broadcast

rank = int(sys.argv[1])
mode = sys.argv[2]
backend = sys.argv[3]
out_dir = sys.argv[4]
timeout_s = float(sys.argv[5])

device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
if backend == "nccl":
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=2, device_id=device)
else:
    dist.init_process_group(backend, rank=rank, world_size=2)

fwd = dist.new_group([0, 1])
bwd = dist.new_group([0, 1])
pp = dist.new_group([0, 1])
comm = PipelineComm(
    device=device,
    fwd_group=fwd,
    bwd_group=bwd,
    chunk_count=1,
    wait_timeout_s=timeout_s,
)
results = []
try:
    if mode == "happy":
        for step in range(3):
            payload = torch.arange(4.0) + rank
            if rank == 0:
                work = comm.fwd_send(payload, dst=1, tag=0)
                comm.wait(work, label=f"pp_rollout.send.step={step}")
            else:
                buf = torch.zeros(4)
                handle = comm.fwd_recv(buf, src=0, tag=0)
                comm.wait(handle, label=f"pp_rollout.recv.step={step}")
                results.append(("recv_data", buf.tolist()))
            token = torch.tensor([rank + 1], dtype=torch.long)
            bounded_broadcast(
                token, src=1, group=pp, timeout_s=timeout_s, label=f"pp.token.step={step}"
            )
            results.append(("bcast", token.tolist()))
    elif mode == "mismatch":
        if rank == 0:
            payload = torch.arange(4.0)
            work = comm.fwd_send(payload, dst=1, tag=0)
            started = time.monotonic()
            try:
                comm.wait(work, label="mismatch.send")
                results.append(("NO_ERROR", "bounded wait did not fire"))
            except Exception as exc:  # noqa: BLE001 - 测试要记录类型与耗时
                results.append(
                    (type(exc).__name__, round(time.monotonic() - started, 1), str(exc))
                )
        else:
            # 故意**不**投递匹配的 recv：这正是"一侧 stage 失败/缺席"的形态。
            time.sleep(timeout_s + 10)
            results.append(("rank1_idle", "no recv posted"))
    else:
        raise SystemExit(f"unknown mode: {mode}")
finally:
    with open(os.path.join(out_dir, f"rank{rank}.txt"), "w", encoding="utf-8") as handle:
        handle.write(repr(results))
    dist.destroy_process_group()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_two_ranks(*, mode: str, backend: str) -> dict[int, str]:
    """跑 2 个 rank，返回 ``{rank: 结果文件内容}``；超墙钟即判无界挂死。"""
    out_dir = tempfile.mkdtemp(prefix="pp-e2e-")
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(_free_port()),
        PYTHONPATH=str(_SRC_ROOT),
        # NCCL 在测试里不需要高带宽；显式关掉 IB 让容器内不依赖网卡。
        NCCL_IB_DISABLE="1",
    )
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD_SCRIPT,
                str(rank),
                mode,
                backend,
                out_dir,
                str(BOUNDED_TIMEOUT_SEC),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + BOUNDED_TIMEOUT_SEC + _PARENT_SLACK_SEC
    for proc in procs:
        remaining = max(1.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            for other in procs:
                other.kill()
            pytest.fail(
                "有界等待失效：两个 rank 在 "
                f"{BOUNDED_TIMEOUT_SEC + _PARENT_SLACK_SEC:.0f}s 内没有退出"
                "（这正是修复前的无界挂死形态；旧码上本用例必然超墙钟）"
            )
    results: dict[int, str] = {}
    for rank in range(2):
        path = Path(out_dir) / f"rank{rank}.txt"
        results[rank] = path.read_text(encoding="utf-8") if path.exists() else ""
        if not results[rank]:
            stream = procs[rank].stderr
            stderr = stream.read().decode("utf-8", "replace") if stream else ""
            pytest.fail(f"rank{rank} 没有产出结果文件；stderr 尾部：\n{stderr[-2000:]}")
    return results


# ── gloo：本机（无 GPU）即可跑，先把机制钉死 ─────────────────────────────────


def test_pp_rollout_rendezvous_sequence_gloo() -> None:
    """快乐路径：多步 P2P + pp_group 广播都完成（PP rollout 的最小复现）。"""
    results = _run_two_ranks(mode="happy", backend="gloo")
    assert "recv_data" in results[1], results[1]
    assert "('recv_data', [0.0, 1.0, 2.0, 3.0])" in results[1]
    # pp_group 广播：末 stage（rank1）采样值必须到达两端
    assert results[0].count("('bcast', [2])") == 3
    assert results[1].count("('bcast', [2])") == 3


def test_bounded_wait_turns_unbounded_hang_into_named_error_gloo() -> None:
    """**判别力用例**：缺匹配 recv ⇒ 有界失败并抛具名异常（旧码上会永久挂死）。"""
    results = _run_two_ranks(mode="mismatch", backend="gloo")
    text = results[0]
    assert "PipelineP2PTimeoutError" in text, text
    assert "NO_ERROR" not in text, text
    assert "fwd_send peer=dst=1" in text, text
    assert "mismatch.send" in text, text
    # 耗时应落在 timeout 附近（而不是"立刻返回"或"永不返回"）
    elapsed = float(text.split("'PipelineP2PTimeoutError', ")[1].split(",")[0])
    assert BOUNDED_TIMEOUT_SEC * 0.8 <= elapsed <= BOUNDED_TIMEOUT_SEC + 20, text


# ── nccl：需要 2 卡（228 容器内生效；本机自动跳过）─────────────────────────


_requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="真 NCCL 的 PP 会合测试需要 ≥2 张可见 GPU（容器内用 NVIDIA_VISIBLE_DEVICES 限定）",
)


@pytest.mark.gpu
@_requires_two_gpus
def test_pp_rollout_rendezvous_sequence_nccl() -> None:
    """真 NCCL 下的 PP rollout 会合序列（PP+rollout 从 0 覆盖到 1 的关键一项）。"""
    results = _run_two_ranks(mode="happy", backend="nccl")
    assert "recv_data" in results[1], results[1]
    assert "('recv_data', [0.0, 1.0, 2.0, 3.0])" in results[1]
    assert results[0].count("('bcast', [2])") == 3
    assert results[1].count("('bcast', [2])") == 3


@pytest.mark.gpu
@_requires_two_gpus
def test_bounded_wait_named_error_nccl() -> None:
    """真 NCCL 下，缺失匹配 recv 必须是**有界**失败（NCCL watchdog 看不见它）。"""
    results = _run_two_ranks(mode="mismatch", backend="nccl")
    assert "PipelineP2PTimeoutError" in results[0], results[0]
