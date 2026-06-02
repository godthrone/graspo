from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(slots=True)
class NativeTPState:
    rank: int
    local_rank: int
    world_size: int
    tp_size: int
    tp_rank: int
    device: torch.device

    @classmethod
    def initialize(cls, tp_size: int) -> "NativeTPState":
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        if world_size > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        if world_size != int(tp_size):
            raise RuntimeError(
                "native-tp v1 requires WORLD_SIZE == tensor_model_parallel_size "
                f"({world_size} != {tp_size})"
            )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            tp_size=int(tp_size),
            tp_rank=rank,
            device=device,
        )


def destroy_native_tp() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()
