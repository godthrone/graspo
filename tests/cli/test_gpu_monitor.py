"""graspo.cli.gpu_monitor 的单元测试（v0.23.0 从 tests/scripts/ 迁移）。

record_gpu_memory 按宪法 §10.2 从 scripts/ 提升为 CLI 命令
（graspo record-gpu-memory），测试随逻辑迁移。
"""

from graspo.cli.gpu_monitor import (
    parse_gpu_query,
    parse_process_query,
    summarize_samples,
)


def test_record_gpu_memory_parses_nvidia_smi_gpu_rows():
    rows = parse_gpu_query("6, GPU-abc, 1024, 80896, 81920, 75, 42, 302.5\n")

    assert rows == [
        {
            "gpu_index": 6,
            "gpu_uuid": "GPU-abc",
            "memory_used_mib": 1024.0,
            "memory_free_mib": 80896.0,
            "memory_total_mib": 81920.0,
            "utilization_gpu_pct": 75.0,
            "temperature_gpu_c": 42.0,
            "power_draw_w": 302.5,
        }
    ]


def test_record_gpu_memory_filters_and_summarizes_samples():
    process_rows = parse_process_query(
        "GPU-abc, 1234, /usr/bin/python, 2048\nGPU-def, 4567, /usr/bin/other, 1024\n"
    )
    samples = [
        {"gpu_index": 6, "memory_used_mib": 100.0, "utilization_gpu_pct": 10.0},
        {"gpu_index": 6, "memory_used_mib": 300.0, "utilization_gpu_pct": 30.0},
        {"gpu_index": 7, "memory_used_mib": 200.0, "utilization_gpu_pct": 20.0},
    ]

    summary = summarize_samples(samples, samples[-2:])

    assert process_rows[0]["pid"] == 1234
    assert process_rows[0]["process_name"] == "/usr/bin/python"
    assert summary["per_gpu"]["6"]["memory_used_mib_peak"] == 300.0
    assert summary["per_gpu"]["6"]["memory_used_mib_mean"] == 200.0
    assert summary["max_peak_memory_gap_mib"] == 100.0
    assert summary["recent_samples"] == samples[-2:]
