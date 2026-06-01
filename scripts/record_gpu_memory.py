#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_path = output_dir / "gpu_memory.jsonl"
    process_path = output_dir / "gpu_processes.jsonl"
    summary_path = output_dir / "gpu_memory_summary.json"
    memory_path.touch()
    process_path.touch()
    stop = {"requested": False}

    def _stop(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    gpu_indices = [item.strip() for item in args.gpus.split(",") if item.strip()]
    pid_filters = [item.strip().lower() for item in args.pid_filter.split(",") if item.strip()]
    samples: list[dict[str, Any]] = []
    recent_samples: deque[dict[str, Any]] = deque(maxlen=args.recent_limit)
    started = time.monotonic()

    try:
        while not stop["requested"]:
            timestamp = utc_timestamp()
            gpu_rows = query_gpu_rows(gpu_indices)
            process_rows = query_process_rows(pid_filters)
            for row in gpu_rows:
                row["timestamp"] = timestamp
                row["tag"] = args.tag
                samples.append(row)
                recent_samples.append(row)
                append_jsonl(memory_path, row)
            for row in process_rows:
                row["timestamp"] = timestamp
                row["tag"] = args.tag
                append_jsonl(process_path, row)
            if args.duration_sec is not None and time.monotonic() - started >= args.duration_sec:
                break
            time.sleep(args.interval_sec)
    finally:
        summary = summarize_samples(samples, list(recent_samples))
        summary.update(
            {
                "tag": args.tag,
                "gpus": gpu_indices,
                "interval_sec": args.interval_sec,
                "started_monotonic": started,
                "finished_at": utc_timestamp(),
                "sample_count": len(samples),
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record nvidia-smi GPU memory/utilization to JSONL.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices, e.g. 6,7.")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--output-dir", required=True, help="Directory for gpu_memory.jsonl and summary.")
    parser.add_argument("--tag", default="", help="Optional run tag written into each row.")
    parser.add_argument(
        "--pid-filter",
        default="",
        help="Comma-separated substrings matched against process_name or pid. Empty records all GPU processes.",
    )
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration for smoke/dry runs.")
    parser.add_argument("--recent-limit", type=int, default=120, help="Recent GPU rows copied into summary.")
    return parser.parse_args()


def query_gpu_rows(gpu_indices: list[str]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(gpu_indices),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_gpu_query(result.stdout)


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


def summarize_samples(samples: list[dict[str, Any]], recent_samples: list[dict[str, Any]]) -> dict[str, Any]:
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
    peak_values = [item["memory_used_mib_peak"] for item in per_gpu.values()]
    return {
        "per_gpu": per_gpu,
        "max_peak_memory_gap_mib": max(peak_values) - min(peak_values) if len(peak_values) >= 2 else 0.0,
        "recent_samples": recent_samples,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, math_floor(q * (len(ordered) - 1))))
    return ordered[idx]


def math_floor(value: float) -> int:
    return int(value // 1)


def parse_float(value: str) -> float:
    cleaned = value.strip()
    if cleaned in {"", "[N/A]", "N/A", "Not Supported"}:
        return 0.0
    return float(cleaned)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
