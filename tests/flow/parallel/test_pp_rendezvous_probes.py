"""会合点探针（缺陷 P6 的 a3）测试 —— 打点必须落盘、必须喂心跳、且**零行为改动**。

为什么这是本包最有价值的一项：P6 的实测形态是"``pp_debug.log`` 只有 3 行、
两卡 util 0%、>18 min 无任何 NCCL 超时"——**没有任何一行日志能指出卡在哪个会合点**。
探针把"一次挂死 = 一次无信息的白耗卡"变成"一次有界运行 ⇒ 定位到唯一一行"。
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch", reason="PP 探针在 pipeline_forward（依赖 torch）里")

from graspo.flow.adapters.models.qwen35_36 import generation, pipeline_forward  # noqa: E402
from graspo.flow.logging import get_run_id  # noqa: E402
from graspo.flow.parallel import rendezvous_watchdog  # noqa: E402


class _RecordingWatchdog:
    """只记录 beat 的假看门狗（避免真的起线程）。"""

    def __init__(self) -> None:
        self.beats: list[str] = []

    def beat(self, label: str) -> None:
        self.beats.append(label)


def test_probe_writes_a_labelled_line_and_beats_the_heartbeat(tmp_path, monkeypatch) -> None:
    fake = _RecordingWatchdog()
    monkeypatch.setattr(rendezvous_watchdog, "_WATCHDOG", fake, raising=False)
    monkeypatch.setattr(pipeline_forward, "rendezvous_watchdog", fake, raising=False)

    pipeline_forward._pp_probe(  # noqa: SLF001
        str(tmp_path), "recv_enqueue", stage=1, tag=0, src=0, shape=(64, 2382, 4096)
    )

    log = tmp_path / "logs" / get_run_id() / "pp_debug.log"
    assert log.exists(), "探针必须走既有 pp_debug.log 落盘通路"
    text = log.read_text(encoding="utf-8")
    assert "probe=recv_enqueue" in text
    assert "stage=1" in text and "src=0" in text and "shape=(64, 2382, 4096)" in text
    assert fake.beats == ["recv_enqueue"], "每个探针必须同时喂一次看门狗心跳"


def test_probe_never_raises_even_if_the_log_dir_is_unwritable(monkeypatch) -> None:
    """**零行为改动**：调试日志落盘失败绝不允许中断训练/生成主流程。"""
    fake = _RecordingWatchdog()
    monkeypatch.setattr(pipeline_forward, "rendezvous_watchdog", fake, raising=False)
    pipeline_forward._pp_probe("/proc/definitely/not/writable", "send_enqueue", stage=0)  # noqa: SLF001
    assert fake.beats == ["send_enqueue"]


def test_probe_returns_none_and_touches_no_tensor() -> None:
    """探针的签名只有 (output_dir, label, **fields) —— 不接收/不返回任何张量。"""
    signature = inspect.signature(pipeline_forward._pp_probe)  # noqa: SLF001
    assert list(signature.parameters) == ["output_dir", "label", "fields"]


@pytest.mark.parametrize(
    "label",
    ["fwd_enter", "recv_enqueue", "recv_ready", "send_enqueue"],
)
def test_pipeline_forward_wires_every_rendezvous_probe(label: str) -> None:
    """接线守卫：PP stage forward 的四个会合点都必须真的打点（少一个就留盲区）。"""
    source = inspect.getsource(pipeline_forward)
    assert f'"{label}"' in source


@pytest.mark.parametrize(
    "label",
    [
        "prefill_done",
        "decode_step_enter",
        "bcast_enter",
        "bcast_done",
        "decode_fwd_enter",
        "send_wait_enter",
        "send_wait_done",
        "chunk_done",
    ],
)
def test_generation_wires_every_rollout_probe(label: str) -> None:
    """接线守卫：PP rollout 的每个会合点（含 C1 的 bcast、C2 的 send_wait）都必须打点。"""
    assert f'"{label}"' in inspect.getsource(generation)


def test_pp_rollout_entry_points_arm_and_disarm_the_watchdog() -> None:
    """a4 接线：两个 PP 入口都必须 arm，并在 finally 里 disarm（不留后台线程）。"""
    source = inspect.getsource(generation)
    assert source.count("self._arm_pp_rollout_watchdog()") == 2
    assert source.count("rendezvous_watchdog.disarm()") == 2
