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
    pp_size: int
    pp_rank: int
    tp_group: dist.ProcessGroup | None
    pp_group: dist.ProcessGroup | None
    prev_pp_rank: int | None
    next_pp_rank: int | None
    device: torch.device

    @classmethod
    def initialize(cls, tp_size: int, pp_size: int = 1) -> GraspoFlowState:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tp_size = int(tp_size)
        pp_size = int(pp_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        if world_size > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        expected_world_size = tp_size * pp_size
        if world_size != expected_world_size:
            raise RuntimeError(
                "native placement requires WORLD_SIZE == tp_size * "
                f"pp_size ({world_size} != {tp_size} * {pp_size})"
            )
        pp_rank = rank // tp_size
        tp_rank = rank % tp_size
        tp_group = None
        pp_group = None
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            for stage_idx in range(pp_size):
                ranks = list(range(stage_idx * tp_size, (stage_idx + 1) * tp_size))
                group = dist.new_group(ranks=ranks)
                if rank in ranks:
                    tp_group = group
            for shard_idx in range(tp_size):
                ranks = [stage_idx * tp_size + shard_idx for stage_idx in range(pp_size)]
                group = dist.new_group(ranks=ranks)
                if rank in ranks:
                    pp_group = group
        prev_pp_rank = rank - tp_size if pp_rank > 0 else None
        next_pp_rank = rank + tp_size if pp_rank < pp_size - 1 else None
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            tp_group=tp_group,
            pp_group=pp_group,
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
