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

import logging
import os
import shutil
import subprocess
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
            # 防呆（§2.1 契约即防呆 / §2.3 边界校验即防呆）：PCIe 拓扑下强制
            # NCCL_P2P_DISABLE=1，防止 NCCL 子通信组 hang（参照 run.sh 的
            # _p2p_disable_required() 逻辑）。
            if backend == "nccl" and os.environ.get("NCCL_P2P_DISABLE") != "1":
                _ensure_nccl_p2p_disabled()
            # 防呆（§2.3 边界校验即防呆）：init_process_group 不传 device_id 时，
            # NCCL 世界通信组不会按 rank 绑定到各自卡；容器暴露全部 GPU 时会在每个
            # 可见设备上建 CUDA/NCCL context，默认组缓冲集中到 cuda:0，造成卡间显存
            # 不均衡（GP0 多扛 ~2.7GB×N）。传 device_id=local_rank 让每个 rank 的
            # 默认组显式绑定到本卡。仅当 local_rank 在可见设备范围内才传（防
            # nproc_per_node > GPU 数时 device_id 越界、NCCL init 失败）。
            if backend == "nccl":
                if local_rank < torch.cuda.device_count():
                    dist.init_process_group(
                        backend=backend, device_id=torch.device(f"cuda:{local_rank}")
                    )
                else:
                    # 防呆（§2.3 边界校验 / §3.4 不做静默坏退路）：请求的 rank 数超过
                    # 可见 GPU 数时，绝不静默回退到不带 device_id 的初始化——那会让
                    # NCCL 默认组缓冲集中到 cuda:0（显存不均）并让多个 rank 挤同一卡
                    # （每 GPU 多进程）。这是配置/资源边界错误，应在边界拒绝而非带病运行。
                    raise RuntimeError(
                        f"local_rank={local_rank} exceeds visible GPU count "
                        f"{torch.cuda.device_count()}; launcher requested more ranks "
                        "than GPUs. Set launch.nproc_per_node (or dp_size*tp_size*pp_size) "
                        "to the number of GPUs exposed via --gpus, then restart."
                    )
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


def _parse_gpu_topo(topo: str) -> dict[int, dict[int, str]]:
    """解析 ``nvidia-smi topo -m`` 输出为 ``{gpu_i: {gpu_j: link}}``。

    仅保留合法数据行（行标 ``GPU<i>`` 且对角线为 ``X``），自动排除表头行。
    链接类型值（``NV#`` / ``PXB`` / ``PHB`` / ``SYS`` / ``X`` 等）按列位置存入。
    """
    matrix: dict[int, dict[int, str]] = {}
    for line in topo.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("GPU"):
            continue
        try:
            gi = int(parts[0][3:])
        except ValueError:
            continue
        # 表头行（如 ``GPU0  GPU1 ...``）的 $1 也以 GPU 开头，但它没有
        # ``X`` 对角线；用 ``parts[gi + 1] == "X"`` 把表头与数据行区分开。
        if gi + 1 >= len(parts) or parts[gi + 1] != "X":
            continue
        row: dict[int, str] = {}
        for col, token in enumerate(parts[1:], start=0):
            row[col] = token
        matrix[gi] = row
    return matrix


def _p2p_disable_required() -> bool:
    """返回当前可见 GPU 拓扑是否必须禁用 NCCL P2P（对齐 ``run.sh`` 判定语义）。

    - 任意两 GPU 路径含 ``PXB`` / ``PHB`` / ``SYS``（跨 PCIe bridge / 跨 NUMA）→ True；
    - 全 NVLink（``NV#``）→ False；
    - ``nvidia-smi`` 不可用、拓扑解析失败或无 GPU 行 → True（安全优先，避免 hang）。
    """
    if shutil.which("nvidia-smi") is None:
        logging.getLogger(__name__).warning(
            "nvidia-smi 不可用，无法检测 GPU 拓扑；安全回退为禁用 NCCL P2P（避免 hang）"
        )
        return True
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        logging.getLogger(__name__).warning(
            "无法读取 GPU 拓扑（nvidia-smi topo -m 失败）；安全回退为禁用 NCCL P2P（避免 hang）"
        )
        return True
    matrix = _parse_gpu_topo(topo)
    gpus = sorted(matrix)
    if len(gpus) == 0:
        # 输出为空或格式异常，解析不出任何 GPU 行 → 安全回退为禁用（避免 hang）
        logging.getLogger(__name__).warning(
            "无法从 nvidia-smi topo -m 输出解析出 GPU 拓扑行；安全回退为禁用 NCCL P2P（避免 hang）"
        )
        return True
    if len(gpus) < 2:
        return False
    for gi in gpus:
        for gj in gpus:
            if gi >= gj:
                continue
            link = matrix[gi].get(gj, "")
            if any(pattern in link for pattern in ("PXB", "PHB", "SYS")):
                return True
    return False


def _ensure_nccl_p2p_disabled() -> None:
    """PCIe 拓扑下强制 ``NCCL_P2P_DISABLE=1``，防止 NCCL 子通信组 hang（防呆 §2.1）。

    必须在 ``dist.init_process_group`` 之前调用：NCCL 在初始化时读取该环境变量，
    因此需在此之前设好。逻辑对齐 ``run.sh::_p2p_disable_required()``：

    - 以 ``nvidia-smi topo -m`` 检测可见 GPU 拓扑，任意两 GPU 路径含
      ``PXB`` / ``PHB`` / ``SYS``（跨 PCIe bridge / 跨 NUMA）→ 强制 ``=1``；
    - ``nvidia-smi`` 不可用或拓扑解析失败 → 安全回退为禁用（避免 hang）；
    - 全 NVLink（``NV#``）→ 不干预（保留 P2P，避免非对称显存 / 效率损耗）。

    透明退路说明（§3.2）：禁用 P2P 不改变训练结果，仅引入 <0.1% 的效率损耗，
    并以 WARNING 告知用户；它不是 §2.3 中"静默降级为更差结果"的坏退路。
    """
    if _p2p_disable_required():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        logging.getLogger(__name__).warning(
            "GPU 拓扑含跨 PCIe bridge/NUMA 路径（PXB/PHB/SYS），强制 "
            "NCCL_P2P_DISABLE=1 以防 NCCL 子通信组 hang（参照 run.sh "
            "_p2p_disable_required() 逻辑）"
        )


def destroy_parallel_state() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()
