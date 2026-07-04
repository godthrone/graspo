#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

TIMING_KEYS = (
    "total_observed_sec",
    "rollout_sec",
    "prefill_sec",
    "decode_sec",
    "sampling_sec",
    "stop_check_sec",
    "old_logprob_sec",
    "optimize_sec",
    "train_batch_total_sec",
    "micro_batch_forward_sec",
    "backward_sec",
    "optimizer_step_sec",
    "checkpoint_sec",
    "decode_tokens",
    "rollout_generation_split_count",
)


def main() -> int:
    args = parse_args()
    summaries = [
        summarize_run(Path(path), skip_warmup_steps=args.skip_warmup_steps)
        for path in args.run_dirs
    ]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        print_table(summaries)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize GRASPO native placement profiling outputs."
    )
    parser.add_argument("run_dirs", nargs="+", help="One or more GRASPO output directories.")
    parser.add_argument(
        "--skip-warmup-steps", type=int, default=1, help="Train steps skipped for mean timing."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a compact table.")
    return parser.parse_args()


def summarize_run(run_dir: Path, *, skip_warmup_steps: int = 1) -> dict[str, Any]:
    train_steps = _read_train_steps(run_dir)
    measured = (
        train_steps[skip_warmup_steps:] if len(train_steps) > skip_warmup_steps else train_steps
    )
    latest = train_steps[-1] if train_steps else {}
    timing_rows = [dict(step.get("timing") or {}) for step in measured]
    gpu_summary = _read_gpu_summary(run_dir)
    rank_summary = _read_rank_summary(run_dir)
    decisions = latest.get("batch", {}).get("decisions", {}) if latest else {}
    latest_timing = latest.get("timing", {}) if latest else {}
    latest_reward = latest.get("batch", {}).get("reward_mean")
    if latest_reward is None:
        latest_reward = latest.get("epoch", {}).get("reward_mean")
    latest_content = latest.get("batch", {}).get("content_mean")
    if latest_content is None:
        latest_content = latest.get("epoch", {}).get("content_mean")
    total_sec = _mean_key(timing_rows, "total_observed_sec")
    decode_tokens = _sum_key(timing_rows, "decode_tokens")
    rollout_sec = _sum_key(timing_rows, "rollout_sec")
    trainable_groups = _sum_latest_or_batch(
        train_steps, measured, ("batch", "decisions", "trainable", "total")
    )
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "step_count": len(train_steps),
        "measured_step_count": len(measured),
        "latest_step": latest.get("run", {}).get("step")
        or latest.get("run", {}).get("optimized_steps"),
        "latest_epoch": latest.get("epoch", {}).get("index")
        or latest.get("epoch", {}).get("epoch"),
        "latest_reward_mean": latest_reward,
        "latest_content_mean": latest_content,
        "latest_trainable_groups": decisions.get("trainable", {}).get("total"),
        "latest_invalid": decisions.get("terminal", {}).get("invalid"),
        "latest_invalid_no_preference_gap": decisions.get("terminal", {}).get(
            "invalid_no_preference_gap"
        ),
        "timing_mean": {key: _mean_key(timing_rows, key) for key in TIMING_KEYS},
        "latest_timing": {
            key: latest_timing.get(key) for key in TIMING_KEYS if key in latest_timing
        },
        "decode_tokens_per_sec": decode_tokens / rollout_sec if rollout_sec > 0 else None,
        "trainable_groups_per_hour": trainable_groups * 3600.0 / (total_sec * len(measured))
        if total_sec and measured
        else None,
        "gpu": gpu_summary,
        "rank": rank_summary,
    }


def _read_train_steps(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in (run_dir / "nohup.out", run_dir / "logs" / "train.log", run_dir / "train.log"):
        if not path.exists():
            continue
        for payload in _iter_json_lines(path):
            if payload.get("event") == "train_step":
                events.append(payload)
    by_step: dict[Any, dict[str, Any]] = {}
    for event in events:
        step = event.get("run", {}).get("step")
        by_step[step if step is not None else len(by_step)] = event
    return list(by_step.values())


def _read_gpu_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "gpu_memory" / "gpu_memory_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return _compact_gpu_summary(summary)
    memory_path = run_dir / "gpu_memory" / "gpu_memory.jsonl"
    if not memory_path.exists():
        return {}
    rows = list(_iter_json_lines(memory_path))
    by_gpu: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gpu.setdefault(str(row.get("gpu_index")), []).append(row)
    per_gpu = {}
    for gpu, gpu_rows in by_gpu.items():
        memory = [float(row.get("memory_used_mib") or 0.0) for row in gpu_rows]
        util = [float(row.get("utilization_gpu_pct") or 0.0) for row in gpu_rows]
        per_gpu[gpu] = {
            "samples": len(gpu_rows),
            "memory_used_mib_peak": max(memory) if memory else None,
            "memory_used_mib_mean": mean(memory) if memory else None,
            "utilization_gpu_pct_mean": mean(util) if util else None,
        }
    return {"per_gpu": per_gpu}


def _compact_gpu_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = {"sample_count": summary.get("sample_count"), "per_gpu": {}}
    for gpu, values in (summary.get("per_gpu") or {}).items():
        compact["per_gpu"][str(gpu)] = {
            "samples": values.get("samples"),
            "memory_used_mib_peak": values.get("memory_used_mib_peak"),
            "memory_used_mib_p95": values.get("memory_used_mib_p95"),
            "memory_used_mib_mean": values.get("memory_used_mib_mean"),
            "utilization_gpu_pct_mean": values.get("utilization_gpu_pct_mean"),
        }
    return compact


def _read_rank_summary(run_dir: Path) -> dict[str, Any]:
    latest_by_rank: dict[int, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("rank_metrics.rank_*.jsonl")):
        for payload in _iter_json_lines(path):
            if payload.get("phase") != "pipeline_train_batch_after":
                continue
            metrics = payload.get("metrics") or {}
            for item in metrics.get("rank_metrics") or [metrics]:
                if "rank" in item:
                    latest_by_rank[int(item["rank"])] = item
    per_rank = {}
    for rank, metrics in sorted(latest_by_rank.items()):
        stage_timing = metrics.get("pipeline_stage_timing") or {}
        per_rank[str(rank)] = {
            "placement_strategy": metrics.get("placement_strategy"),
            "pipeline_train_schedule": metrics.get("pipeline_train_schedule"),
            "pipeline_stage_rank": metrics.get("pipeline_stage_rank"),
            "pipeline_stage_compute_sec": stage_timing.get("pipeline_stage_compute_sec"),
            "pipeline_backward_autograd_sec": stage_timing.get("pipeline_backward_autograd_sec"),
            "pipeline_send_sec": stage_timing.get("pipeline_send_sec"),
            "pipeline_recv_sec": stage_timing.get("pipeline_recv_sec"),
            "pipeline_grad_send_sec": stage_timing.get("pipeline_grad_send_sec"),
            "pipeline_grad_recv_sec": stage_timing.get("pipeline_grad_recv_sec"),
            "pipeline_norm_sec": stage_timing.get("pipeline_norm_sec"),
            "pipeline_lm_head_sec": stage_timing.get("pipeline_lm_head_sec"),
            "pipeline_loss_sec": stage_timing.get("pipeline_loss_sec"),
        }
    return {"per_rank": per_rank}


def _iter_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _sum_key(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows)


def _sum_latest_or_batch(
    all_steps: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    path: tuple[str, ...],
) -> float:
    del all_steps
    total = 0.0
    for step in measured:
        value: Any = step
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        total += float(value or 0.0)
    return total


def print_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "run",
        "steps",
        "reward",
        "total_s",
        "rollout_s",
        "opt_s",
        "decode_tok_s",
        "groups_h",
        "gpu_util",
        "gpu_peak_gib",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(item) for item in headers))
    for summary in summaries:
        timing = summary.get("timing_mean") or {}
        gpu_util, gpu_peak_gib = _gpu_rollup(summary.get("gpu") or {})
        values = [
            summary.get("name"),
            f"{summary.get('measured_step_count')}/{summary.get('step_count')}",
            _fmt(summary.get("latest_reward_mean")),
            _fmt(timing.get("total_observed_sec")),
            _fmt(timing.get("rollout_sec")),
            _fmt(timing.get("optimize_sec")),
            _fmt(summary.get("decode_tokens_per_sec")),
            _fmt(summary.get("trainable_groups_per_hour")),
            _fmt(gpu_util),
            _fmt(gpu_peak_gib),
        ]
        print(" | ".join(str(item) for item in values))


def _gpu_rollup(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    per_gpu = summary.get("per_gpu") or {}
    utils = [
        float(item["utilization_gpu_pct_mean"])
        for item in per_gpu.values()
        if item.get("utilization_gpu_pct_mean") is not None
    ]
    peaks = [
        float(item["memory_used_mib_peak"])
        for item in per_gpu.values()
        if item.get("memory_used_mib_peak") is not None
    ]
    return (mean(utils) if utils else None, max(peaks) / 1024.0 if peaks else None)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
