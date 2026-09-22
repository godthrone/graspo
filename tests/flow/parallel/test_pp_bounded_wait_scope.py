"""**裁定 3 的作用域围栏**（缺陷 P6）：有界超时只准留在 rollout/生成路径。

背景（指挥官 2026-09-22 裁定 3）：torch 的 ``Work.wait(timeout)`` **在设了 timeout 时
会阻塞 CPU 线程**（官方文档原文 "if timeout is set, it will block the CPU thread until
the NCCL work is completed or timed out"；不设 timeout 时只是"让当前流挂完成事件"）。
把它放到 1F1B 的训练热路径（``training*.py`` 的 ``wait_all(send_works)``）上，
就是**每步末尾多一次 CPU 同步、损失跨 step 重叠** —— 用一个健康路径上永不触发的
超时换热路径性能，属 §18 留债。

训练路径的有界性另有来源：它的等待对象是"**已入队的 NCCL work**"，由 **NCCL 自己的
600s watchdog** 计时；P6 的无界形态恰恰是 NCCL 看不见的"流/事件依赖 + 尚未入队的
会合"，只出现在 rollout 的紧会合序列里（这一点由 ``pp_debug`` 探针在 r3 实测钉住）。

本文件是**接线守卫**：源码级断言，防止以后有人"顺手"把超时加回热路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER_DIR = _REPO_ROOT / "src" / "graspo" / "flow" / "adapters" / "models" / "qwen35_36"

#: 训练热路径：**禁止**出现有界等待。
_HOT_PATH_FILES = ("training.py", "training_sft.py", "logprobs.py")
#: rollout/生成路径：**唯一**允许使用有界等待的地方。
_ROLLOUT_FILE = "generation.py"


def _source(name: str) -> str:
    return (_ADAPTER_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", _HOT_PATH_FILES)
def test_training_hot_path_has_no_bounded_wait(name: str) -> None:
    source = _source(name)
    assert "wait_timeout_s" not in source, (
        f"{name} 里出现了有界等待（wait_timeout_s）——裁定 3 禁止："
        "Work.wait(timeout) 会阻塞 CPU，1F1B 每步末同步一次，损失跨 step 重叠"
    )
    assert "comm.wait_all(" not in source, (
        f"{name} 应使用模块级 `wait_all(...)`（异步、不设超时），"
        "而不是带超时的 `comm.wait_all(...)`"
    )
    assert "wait_all(" in source, f"{name} 既没有 comm.wait_all 也没有 wait_all —— 等待点不见了？"


@pytest.mark.parametrize("name", _HOT_PATH_FILES)
def test_training_hot_path_does_not_consume_the_rollout_timeout_key(name: str) -> None:
    assert "pp_rollout_p2p_timeout_sec" not in _source(name), (
        f"{name} 引用了 native.pp_rollout_p2p_timeout_sec —— 该键的作用域是"
        "**PP rollout / 生成路径**（裁定 3），训练热路径不得消费它"
    )


def test_rollout_path_is_the_only_consumer_of_the_bounded_wait() -> None:
    """正向断言：rollout 路径**必须**仍然有界（否则 P6 的无界挂死会回来）。"""
    source = _source(_ROLLOUT_FILE)
    assert "wait_timeout_s=int(self.config.native.pp_rollout_p2p_timeout_sec)" in source
    assert "comm.wait_all([send_work], label=" in source
    consumers = [
        name
        for name in (*_HOT_PATH_FILES, _ROLLOUT_FILE)
        if "pp_rollout_p2p_timeout_sec" in _source(name)
    ]
    assert consumers == [_ROLLOUT_FILE], f"有界超时的消费点必须唯一，实际：{consumers}"
