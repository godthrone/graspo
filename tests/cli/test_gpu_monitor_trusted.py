"""可信显存采样的负向测试——"可见 2 卡但机上 8 卡"。

历史事故：采样命令不带 ``-i``，峰值混入生产 GPU6/7。本测试构造最坏场景
（nvidia-smi 无视 ``-i``、返回全部 8 卡），断言采样结果只含可见的 2 卡。
纯逻辑测试，不触 GPU（runner 注入）。
"""

import pytest

from graspo.cli.gpu_monitor import (
    build_gpu_query_command,
    query_gpu_rows,
    resolve_sample_gpus,
)
from graspo.core.gpu_guard import GpuLockError

#: 机上 8 卡（含生产卡 6/7）的 nvidia-smi 输出：index,uuid,used,free,total,util,temp,power
_EIGHT_CARD_OUTPUT = "".join(
    f"{idx}, GPU-{idx:03d}, {100 * idx}, {81920 - 100 * idx}, 81920, {idx * 5}, 40, 100.0\n"
    for idx in range(8)
)


class _NoisyRunner:
    """最坏情况模拟器：无视 argv，总是返回全部 8 卡。"""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        return _EIGHT_CARD_OUTPUT


def test_query_only_returns_visible_two_cards():
    """可见 2 卡（0,3）但机上 8 卡 → 采样结果只含 0 和 3。"""
    runner = _NoisyRunner()

    rows = query_gpu_rows(["0", "3"], runner=runner)

    assert [row["gpu_index"] for row in rows] == [0, 3]
    assert len(rows) == 2
    sampled = {int(row["gpu_index"]) for row in rows}
    assert 6 not in sampled, "生产卡 GPU6 不得出现在样本中"
    assert 7 not in sampled, "生产卡 GPU7 不得出现在样本中"


def test_query_command_always_carries_dash_i():
    """查询命令必须带 ``-i <目标卡>``——不带 -i 的全卡查询在结构上不可能发生。"""
    command = build_gpu_query_command(["0", "3"])

    assert "-i" in command
    assert command[command.index("-i") + 1] == "0,3"
    assert "all" not in command


def test_query_command_rejects_empty_targets():
    """没有目标卡就不构造命令（不给"全卡查询"留缺口）。"""
    with pytest.raises(ValueError):
        build_gpu_query_command([])


def test_resolve_sample_gpus_uses_visible_set():
    """默认采样目标 = 可见卡；显式取值必须是可见集子集。"""
    visible = {"NVIDIA_VISIBLE_DEVICES": "0,3"}

    assert resolve_sample_gpus(None, visible=visible["NVIDIA_VISIBLE_DEVICES"]) == ["0", "3"]
    assert resolve_sample_gpus("3", visible=visible["NVIDIA_VISIBLE_DEVICES"]) == ["3"]


@pytest.mark.parametrize(
    "gpus,visible",
    [
        (None, None),  # 未锁卡
        (None, "all"),  # 全卡
        (None, "0,1,2,3,6"),  # 含生产卡
        ("0,6", "0,3"),  # 显式越出可见集
    ],
)
def test_resolve_sample_gpus_is_fail_closed(gpus, visible):
    """任一边界不满足 → 拒绝采样，绝不回退到全卡。"""
    with pytest.raises(GpuLockError):
        resolve_sample_gpus(gpus, visible=visible)
