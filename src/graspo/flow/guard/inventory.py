"""flow.guard.inventory — 容器内**实测可见卡**的探测（设施操作）。

★ 一名一物（P2 落位）：探测是**卡守卫的设施实现**，故住 `flow/guard/`，
而不是 `cli/gpu_monitor.py`。这样 `flow/guard/lock.py` 不再需要 import `cli`
—— 那正是 `tests/test_ast_boundary.py` 里最后 1 条跨层循环 `cli ↔ flow` 的根因。

口径（F-1/F-2 实测确证）：
- 用 ``nvidia-smi -L``：它天然只列**可见**卡，行首序号 = 容器命名空间里的本地 index；
- runtime 重编号后宿主卡号在这里看不到，因此**不得**拿宿主卡号去查容器内的卡。
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from graspo.core.gpu_guard import GpuInventory


def parse_visible_device_indices(text: str) -> tuple[int, ...]:
    """从 ``nvidia-smi -L`` 输出解析**容器内本地序号**（探测可见卡的唯一入口）。

    ``-L`` 每个可见卡一行（``GPU 0: NVIDIA ... (UUID: GPU-xxx)``）。行首序号即
    容器命名空间里的本地 index——runtime 重编号后，宿主卡号在这里是看不到的
    （这正是 F-2 的根因：不能拿宿主卡号去查容器内的卡）。

    :raises RuntimeError: 行结构与 ``GPU <n>:`` 不符——宁可 fail-closed，
        也不要猜出一组卡号去查。
    """
    indices: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("GPU ") or ":" not in stripped:
            raise RuntimeError(f"无法解析 nvidia-smi -L 行：{line!r}")
        head = stripped[len("GPU ") :].split(":", 1)[0].strip()
        if not head.isdigit():
            raise RuntimeError(f"无法解析 nvidia-smi -L 卡号：{line!r}")
        indices.append(int(head))
    return tuple(indices)


def probe_gpu_inventory(
    runner: Callable[[list[str]], str] | None = None,
) -> GpuInventory:
    """实测容器内**可见卡**（F-1/F-2 的设施层入口）。

    用 ``nvidia-smi -L``（结构稳定：每个可见卡一行）数卡并取容器内本地序号。
    ``-L`` 不接受 ``-i``，但它天然只列**可见**卡——这正是我们要的口径
    （不要 ``--query-gpu`` 的全卡枚举语义）。带 ``-i`` 的查询仍由
    :func:`build_gpu_query_command` 唯一构造（采样路径）。

    :param runner: 命令执行器（接收 argv、返回 stdout）；默认走 ``subprocess``。
    :raises RuntimeError: ``nvidia-smi`` 不在或执行失败——调用方必须 fail-closed。
    """
    command = ["nvidia-smi", "-L"]
    if runner is None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"nvidia-smi 不可用：{exc}") from None
        if completed.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi -L 失败（rc={completed.returncode}）：{completed.stderr.strip()[:200]}"
            )
        stdout = completed.stdout
    else:
        stdout = runner(command)
    indices = parse_visible_device_indices(stdout)
    return GpuInventory(
        source="nvidia-smi -L",
        count=len(indices),
        indices=indices,
    )
