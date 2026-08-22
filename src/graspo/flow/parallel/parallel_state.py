"""TP/DP/PP 分布式状态容器：GraspoFlowState 数据类及进程组管理（设施层）。

rank 拓扑（3D: dp × tp × pp）::

    rank = dp_rank × (tp_size × pp_size) + pp_rank × tp_size + tp_rank
    dp_rank  = rank // (tp_size × pp_size)
    tp_rank  = (rank // pp_size) % tp_size
    pp_rank  = (rank % (tp_size × pp_size)) // tp_size
    world_size = dp_size × tp_size × pp_size

进程组:
    - tp_group: 同 dp、同 pp 的所有 rank → all_reduce(SUM)
    - dp_group: 同 tp、同 pp 的所有 rank → all_reduce(AVG)
    - pp_group: 同 dp、同 tp 的所有 rank → send/recv
    - pp_group_fwd / pp_group_bwd: 与 pp_group 同 rank 序列的两个独立进程组，
      分别承担 forward-hidden 与 backward-grad 的 P2P，隔离 1F1B 交错下的
      peer-pair 单 FIFO 错配。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(slots=True)
class GraspoFlowState:
    rank: int
    local_rank: int
    world_size: int
    tp_size: int
    tp_rank: int
    dp_size: int
    dp_rank: int
    pp_size: int
    pp_rank: int
    tp_group: dist.ProcessGroup | None
    dp_group: dist.ProcessGroup | None
    pp_group: dist.ProcessGroup | None
    # 双向 PP 通信组：forward-hidden 与 backward-grad 分属独立进程组（各自独立
    # NCCL P2P channel），隔离 1F1B 交错下同一 peer-pair 上双向消息的单 FIFO 错配。
    pp_group_fwd: dist.ProcessGroup | None
    pp_group_bwd: dist.ProcessGroup | None
    prev_pp_rank: int | None
    next_pp_rank: int | None
    device: torch.device

    @classmethod
    def initialize(
        cls, tp_size: int, pp_size: int = 1, dp_size: int = 1
    ) -> GraspoFlowState:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tp_size = int(tp_size)
        pp_size = int(pp_size)
        dp_size = int(dp_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        if world_size > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if backend == "nccl":
                # device_id=local_rank 让每个 rank 的 NCCL 通信组只绑定到其本地卡，
                # 避免"每 rank 对全卡各建上下文"的浪费（内存/通信组开销）。
                dist.init_process_group(backend=backend, device_id=local_rank)
            else:
                dist.init_process_group(backend=backend)
        expected_world_size = dp_size * tp_size * pp_size
        if world_size != expected_world_size:
            raise RuntimeError(
                "native placement requires WORLD_SIZE == dp_size * tp_size * "
                f"pp_size ({world_size} != {dp_size} * {tp_size} * {pp_size})"
            )
        # 3D rank mapping: dp × tp × pp
        pp_tp_size = tp_size * pp_size
        dp_rank = rank // pp_tp_size
        local_rank_in_dp = rank % pp_tp_size
        pp_rank = local_rank_in_dp // tp_size
        tp_rank = local_rank_in_dp % tp_size

        tp_group = None
        dp_group = None
        pp_group = None
        pp_group_fwd = None
        pp_group_bwd = None
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            # TP groups: ranks with same (dp_rank, pp_rank)
            for dp_idx in range(dp_size):
                for stage_idx in range(pp_size):
                    base = dp_idx * pp_tp_size + stage_idx * tp_size
                    ranks = list(range(base, base + tp_size))
                    group = dist.new_group(ranks=ranks)
                    if rank in ranks:
                        tp_group = group
            # PP groups: ranks with same (dp_rank, tp_rank)
            for dp_idx in range(dp_size):
                for shard_idx in range(tp_size):
                    ranks = [
                        dp_idx * pp_tp_size + stage_idx * tp_size + shard_idx
                        for stage_idx in range(pp_size)
                    ]
                    group = dist.new_group(ranks=ranks)
                    if rank in ranks:
                        pp_group = group
                    # 每个 PP group 另建 fwd/bwd 两个独立进程组（同 rank 序列）：
                    # 1F1B 的 forward-hidden 与 backward-grad 在交错时序下共享同一
                    # peer-pair 的单 FIFO 会错配死锁；拆成独立 group 各得一条独立
                    # NCCL P2P channel，从机制上隔离（§2.1 契约即防呆）。
                    # 仅在 pp_size>1 时创建（pp=1 走非 PP 路径，无需双通道）。
                    if pp_size > 1:
                        fwd_group = dist.new_group(ranks=ranks)
                        bwd_group = dist.new_group(ranks=ranks)
                        if rank in ranks:
                            pp_group_fwd = fwd_group
                            pp_group_bwd = bwd_group
            # DP groups: ranks with same (tp_rank, pp_rank)
            for pp_idx in range(pp_size):
                for tp_idx in range(tp_size):
                    ranks = [
                        d * pp_tp_size + pp_idx * tp_size + tp_idx
                        for d in range(dp_size)
                    ]
                    group = dist.new_group(ranks=ranks)
                    if rank in ranks:
                        dp_group = group

        prev_pp_rank = rank - tp_size if pp_rank > 0 else None
        next_pp_rank = rank + tp_size if pp_rank < pp_size - 1 else None
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            tp_group=tp_group,
            dp_group=dp_group,
            pp_group=pp_group,
            pp_group_fwd=pp_group_fwd,
            pp_group_bwd=pp_group_bwd,
            prev_pp_rank=prev_pp_rank,
            next_pp_rank=next_pp_rank,
            device=device,
        )


def destroy_parallel_state() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()
