"""GPU 显存/利用率采样记录（运维监控工具，**只采可见卡**）。

v0.23.0 起从 ``scripts/record_gpu_memory.py`` 提升为 CLI 命令
（``graspo record-gpu-memory``）：被测试固化的运维工具按宪法 §10.2
必须制度化，不能留在 scripts/（scripts/ 已删除，防止内网路径/凭据
随脚本入库）。

职责：按固定间隔采样 ``nvidia-smi`` 的 GPU 显存/利用率/温度/功耗与
进程占用，追加写入 JSONL，结束时输出摘要 JSON。纯诊断工具，不参与
训练产物。

**可信采样（防呆，§2）——历史事故的修复：**

历史采样用不带 ``-i`` 的 ``nvidia-smi --query-gpu``，峰值混入生产
GPU6/7 与他人并发作业，显存数据不可作依据。本工具现在三重防护：

1. 采样目标由 ``NVIDIA_VISIBLE_DEVICES`` 唯一决定（``--gpus`` 只能是
   它的子集，越界即拒绝）——守卫逻辑在 ``graspo.core.gpu_guard``；
2. 查询命令**永远**带 ``-i <目标卡>``，不提供"全卡查询"入口；
3. 返回行再做一次白名单过滤——即使 ``nvidia-smi`` 忽略 ``-i``，
   也不会有可见集之外的行进入样本。
"""

import argparse
import json
import os
import signal
import subprocess
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graspo.core.gpu_guard import select_sample_targets

GPU_FIELDS = (
    "index",
    "uuid",
    "memory.used",
    "memory.free",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
)
PROCESS_FIELDS = ("gpu_uuid", "pid", "process_name", "used_memory")


def parse_float(value: str) -> float:
    cleaned = value.strip()
    if cleaned in {"", "[N/A]", "N/A", "Not Supported"}:
        return 0.0
    return float(cleaned)


def math_floor(value: float) -> int:
    return int(value // 1)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, math_floor(q * (len(ordered) - 1))))
    return ordered[idx]


def parse_gpu_query(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            raise ValueError(f"Unexpected nvidia-smi GPU row: {line}")
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "gpu_uuid": parts[1],
                "memory_used_mib": parse_float(parts[2]),
                "memory_free_mib": parse_float(parts[3]),
                "memory_total_mib": parse_float(parts[4]),
                "utilization_gpu_pct": parse_float(parts[5]),
                "temperature_gpu_c": parse_float(parts[6]),
                "power_draw_w": parse_float(parts[7]),
            }
        )
    return rows


def parse_process_query(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=len(PROCESS_FIELDS) - 1)]
        if len(parts) != len(PROCESS_FIELDS):
            continue
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": parse_float(parts[3]),
            }
        )
    return rows


def build_gpu_query_command(gpu_indices: Sequence[str]) -> list[str]:
    """构造**只查指定卡**的 nvidia-smi 命令——``-i`` 是不可省略的防线。

    单一入口：采样永远经由本函数构造命令，因此"忘了带 -i"在结构上不可能发生
    （宪法 §2 防呆：不靠调用方自觉）。
    """
    if not gpu_indices:
        raise ValueError("build_gpu_query_command requires at least one GPU index")
    return [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(str(item) for item in gpu_indices),
    ]


def query_gpu_rows(
    gpu_indices: list[str],
    *,
    runner: Callable[[list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    """查询给定卡的显存行；结果再按白名单过滤一次。

    :param gpu_indices: 目标卡号（字符串，来自可信采样目标解析）。
    :param runner: 命令执行器（接收 argv，返回 stdout）；默认走 ``subprocess``。
        注入点让负向测试能构造"可见 2 卡但机上 8 卡"的场景。
    """
    command = build_gpu_query_command(gpu_indices)
    if runner is None:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        stdout = completed.stdout
    else:
        stdout = runner(command)
    requested = {int(item) for item in gpu_indices}
    # 白名单过滤：即使 nvidia-smi 忽略 -i 返回了全卡，也不让可见集之外的行进入样本。
    return [row for row in parse_gpu_query(stdout) if int(row["gpu_index"]) in requested]


def query_process_rows(pid_filters: list[str]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-compute-apps={','.join(PROCESS_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    rows = parse_process_query(result.stdout)
    if not pid_filters:
        return rows
    filtered = []
    for row in rows:
        haystack = f"{row.get('pid', '')} {row.get('process_name', '')}".lower()
        if any(item in haystack for item in pid_filters):
            filtered.append(row)
    return filtered


def summarize_samples(
    samples: list[dict[str, Any]], recent_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_gpu[int(sample["gpu_index"])].append(sample)
    per_gpu = {}
    for gpu_index, rows in by_gpu.items():
        used = [float(row["memory_used_mib"]) for row in rows]
        util = [float(row["utilization_gpu_pct"]) for row in rows]
        per_gpu[str(gpu_index)] = {
            "samples": len(rows),
            "memory_used_mib_peak": max(used),
            "memory_used_mib_mean": sum(used) / len(used),
            "memory_used_mib_p95": percentile(used, 0.95),
            "utilization_gpu_pct_mean": sum(util) / len(util),
            "last": rows[-1],
        }
    peak_values = [float(str(item["memory_used_mib_peak"])) for item in per_gpu.values()]
    return {
        "per_gpu": per_gpu,
        "max_peak_memory_gap_mib": max(peak_values) - min(peak_values)
        if len(peak_values) >= 2
        else 0.0,
        "recent_samples": recent_samples,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def resolve_sample_gpus(gpus: str | None, *, visible: str | None = None) -> list[str]:
    """确定采样目标卡号：只允许落在 ``NVIDIA_VISIBLE_DEVICES`` 之内。

    fail-closed：可见集未设置 / 含生产卡 6,7 / 超过 4 卡 → 抛
    ``GpuLockError``，不返回任何"全卡"退路（宪法 §2.3 边界校验）。
    """
    if visible is None:
        visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    return [str(device) for device in select_sample_targets(gpus, visible)]


def record_gpu_memory(
    *,
    gpus: str | None,
    interval_sec: float,
    output_dir: str,
    tag: str,
    pid_filter: str,
    duration_sec: float | None,
    recent_limit: int,
) -> int:
    """采样循环：按固定间隔记录 GPU 状态到 JSONL，结束时写摘要。

    ``gpus=None`` 表示"全部可见卡"；显式取值必须是可见集的子集。
    """
    gpu_indices = resolve_sample_gpus(gpus)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / "gpu_memory.jsonl"
    process_path = out_dir / "gpu_processes.jsonl"
    summary_path = out_dir / "gpu_memory_summary.json"
    memory_path.touch()
    process_path.touch()
    stop = {"requested": False}

    def _stop(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    pid_filters = [item.strip().lower() for item in pid_filter.split(",") if item.strip()]
    samples: list[dict[str, Any]] = []
    recent_samples: deque[dict[str, Any]] = deque(maxlen=recent_limit)
    started = time.monotonic()

    try:
        while not stop["requested"]:
            timestamp = utc_timestamp()
            gpu_rows = query_gpu_rows(gpu_indices)
            # 进程行按可见卡的 UUID 过滤：--query-compute-apps 不支持 -i，
            # 生产进程会出现在原始输出里，必须在进入样本前剔除。
            visible_uuids = {str(row["gpu_uuid"]) for row in gpu_rows}
            process_rows = [
                row
                for row in query_process_rows(pid_filters)
                if str(row.get("gpu_uuid", "")) in visible_uuids
            ]
            for row in gpu_rows:
                row["timestamp"] = timestamp
                row["tag"] = tag
                samples.append(row)
                recent_samples.append(row)
                append_jsonl(memory_path, row)
            for row in process_rows:
                row["timestamp"] = timestamp
                row["tag"] = tag
                append_jsonl(process_path, row)
            if duration_sec is not None and time.monotonic() - started >= duration_sec:
                break
            time.sleep(interval_sec)
    finally:
        summary = summarize_samples(samples, list(recent_samples))
        summary.update(
            {
                "tag": tag,
                "gpus": gpu_indices,
                "interval_sec": interval_sec,
                "started_monotonic": started,
                "finished_at": utc_timestamp(),
                "sample_count": len(samples),
            }
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def build_gpu_monitor_parser(subparsers: Any) -> None:
    """注册 ``record-gpu-memory`` 子命令（由 cli.app.build_parser 调用）。"""
    gpu = subparsers.add_parser(
        "record-gpu-memory",
        help="Record nvidia-smi GPU memory/utilization to JSONL (visible GPUs only).",
    )
    gpu.add_argument(
        "--output-dir", required=True, help="Directory for gpu_memory.jsonl and summary."
    )
    gpu.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated GPU indices; must be a subset of NVIDIA_VISIBLE_DEVICES. "
            "Default (None) = all visible GPUs. 'all' is rejected (fail-closed)."
        ),
    )
    gpu.add_argument(
        "--interval-sec", type=float, default=1.0, help="Sampling interval in seconds."
    )
    gpu.add_argument("--tag", default="", help="Optional run tag written into each row.")
    gpu.add_argument(
        "--pid-filter",
        default="",
        help=(
            "Comma-separated substrings matched against process_name or pid. "
            "Empty records all GPU processes."
        ),
    )
    gpu.add_argument(
        "--duration-sec", type=float, default=None, help="Optional duration for smoke/dry runs."
    )
    gpu.add_argument(
        "--recent-limit", type=int, default=120, help="Recent GPU rows copied into summary."
    )
    gpu.set_defaults(func=cmd_record_gpu_memory)


def cmd_record_gpu_memory(args: argparse.Namespace) -> int:
    """``record-gpu-memory`` 命令入口。锁卡守卫拒绝时打印原因并返回 1。"""
    import sys

    from graspo.core.gpu_guard import GpuLockError

    try:
        return record_gpu_memory(
            gpus=args.gpus,
            interval_sec=args.interval_sec,
            output_dir=args.output_dir,
            tag=args.tag,
            pid_filter=args.pid_filter,
            duration_sec=args.duration_sec,
            recent_limit=args.recent_limit,
        )
    except GpuLockError as exc:
        print(str(exc), file=sys.stderr)
        return 1
