#!/usr/bin/env python3
"""
analyze_training.py — Extract structured stats from a GRASPO training run.

Usage:
    python scripts/analyze_training.py outputs/<run_name>/
    python scripts/analyze_training.py outputs/<run_name>/ --output my_stats.json

Output:
    A JSON file (default: training_stats.json in the run directory) containing
    epoch summaries, decision distribution, timing breakdown, GPU/memory stats,
    and error/numeric-precision analysis from the rollout logs.

Workflow:
    1. Run this script on a training output directory.
    2. Feed the output JSON to an LLM with the prompt:
       "Analyze this GRASPO training run and identify issues, trends, and
        suggestions."
    3. The LLM reads all structured facts and gives you a human-readable report.

Notes:
    - Only the rollouts.readable.jsonl is fully scanned (streaming, line by line).
    - training.log and rank_metrics are parsed for aggregate stats.
    - Large runs may take a few minutes to scan the rollout log.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone


def parse_training_log(log_path: str) -> dict:
    """Parse training.log for per-step and per-epoch stats."""
    result = {
        "steps": [],
        "epochs": {},
        "total_steps": 0,
        "total_elapsed_sec": 0,
        "last_log_timestamp": None,
    }
    if not os.path.exists(log_path):
        return result

    ep_data = defaultdict(lambda: {
        "steps": 0, "reward_means": [], "content_means": [], "best_rewards": [],
        "timing_rollout": [], "timing_optimize": [], "elapsed_times": [],
        "decisions": {}, "samples_seen": 0, "samples_total": 0,
    })
    steps = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line[line.index("{"):])
            except (ValueError, json.JSONDecodeError):
                continue
            if msg.get("event") != "train_step" or "step" not in msg:
                continue

            steps.append(msg)
            ep = msg.get("epoch", {})
            ep_num = ep.get("epoch", "?")
            d = ep_data[ep_num]
            d["steps"] += 1
            d["reward_means"].append(ep.get("reward_mean", 0))
            d["content_means"].append(ep.get("content_mean", 0))
            d["best_rewards"].append(ep.get("best_reward", 0))
            d["elapsed_times"].append(msg.get("elapsed_sec", 0))
            timing = msg.get("timing", {})
            d["timing_rollout"].append(timing.get("rollout_sec", 0))
            d["timing_optimize"].append(timing.get("optimize_sec", 0))
            d["samples_seen"] = ep.get("samples_seen", 0)
            d["samples_total"] = ep.get("samples_total", 0)
            decisions = ep.get("decisions", {})
            term = decisions.get("terminal", {})
            trn = decisions.get("trainable", {})
            d["decisions"] = {
                "perfect_skip": term.get("perfect_skip", 0),
                "trainable": term.get("trainable", 0),
                "invalid": term.get("invalid", 0),
                "max_correct": trn.get("max_correct", 0),
                "not_correct": trn.get("not_correct", 0),
            }

    result["total_steps"] = len(steps)
    if steps:
        result["total_elapsed_sec"] = steps[-1].get("elapsed_sec", 0)
        # Extract timestamp from log line
        try:
            ts_str = steps[-1]["timestamp"]
            result["last_log_timestamp"] = ts_str
        except (KeyError, TypeError):
            pass

    # Build epoch summary list sorted by epoch number
    epochs_list = []
    prev_time = 0
    for ep_num in sorted(ep_data.keys(), key=lambda x: int(x) if x != "?" else -1):
        d = ep_data[ep_num]
        if not d["reward_means"]:
            continue
        rwd = sum(d["reward_means"]) / len(d["reward_means"])
        cont = sum(d["content_means"]) / len(d["content_means"])
        best = max(d["best_rewards"])
        last_time = d["elapsed_times"][-1]
        ep_dur = (last_time - prev_time) if prev_time > 0 else last_time
        prev_time = last_time
        epochs_list.append({
            "epoch": int(ep_num) if ep_num != "?" else -1,
            "steps": d["steps"],
            "samples_seen": d["samples_seen"],
            "samples_total": d["samples_total"],
            "reward_mean": round(rwd, 4),
            "content_mean": round(cont, 4),
            "best_reward": round(best, 4),
            "duration_sec": round(ep_dur, 1),
            "duration_h": round(ep_dur / 3600, 2),
            **d["decisions"],
        })
    result["epochs"] = epochs_list

    # Timing summary from the last step
    if steps:
        last = steps[-1]
        t = last.get("timing", {})
        result["latest_step_timing"] = {
            "step": last["step"],
            "total_observed_sec": round(t.get("total_observed_sec", 0), 1),
            "rollout_sec": round(t.get("rollout_sec", 0), 1),
            "old_logprob_sec": round(t.get("old_logprob_sec", 0), 1),
            "prefill_sec": round(t.get("prefill_sec", 0), 1),
            "decode_sec": round(t.get("decode_sec", 0), 1),
            "optimize_sec": round(t.get("optimize_sec", 0), 1),
            "forward_sec": round(t.get("micro_batch_forward_sec", 0), 1),
            "backward_sec": round(t.get("backward_sec", 0), 1),
            "decode_tokens": t.get("decode_tokens", 0),
        }

    # Recent loss/grad trend
    recent = steps[-20:] if len(steps) >= 20 else steps
    result["recent_optimization_trend"] = []
    for rec in recent:
        opt = rec.get("optimize", {})
        result["recent_optimization_trend"].append({
            "step": rec["step"],
            "loss_mean": opt.get("loss_mean", 0),
            "grad_norm_mean": opt.get("grad_norm_mean", 0),
            "lora_delta_mean": opt.get("lora_delta_mean", 0),
        })

    # Per-step timing trend (last 10)
    recent_steps = steps[-20:] if len(steps) >= 20 else steps
    result["recent_step_timing"] = []
    for rec in recent_steps:
        t = rec.get("timing", {})
        result["recent_step_timing"].append({
            "step": rec["step"],
            "rollout_sec": round(t.get("rollout_sec", 0), 1),
            "optimize_sec": round(t.get("optimize_sec", 0), 1),
            "total_sec": round(t.get("total_observed_sec", 0), 1),
            "decode_tokens": t.get("decode_tokens", 0),
        })

    # Average step timing
    if steps:
        avg_rollout = sum(s.get("timing", {}).get("rollout_sec", 0) for s in steps) / len(steps)
        avg_optimize = sum(s.get("timing", {}).get("optimize_sec", 0) for s in steps) / len(steps)
        avg_total = sum(s.get("timing", {}).get("total_observed_sec", 0) for s in steps) / len(steps)
        result["avg_step_timing"] = {
            "rollout_sec": round(avg_rollout, 1),
            "optimize_sec": round(avg_optimize, 1),
            "total_sec": round(avg_total, 1),
        }

    return result


def parse_rank_metrics(output_dir: str, logs_dir: str) -> dict:
    """Parse rank_metrics.*.jsonl files for GPU/memory stats."""
    result = {"records": 0, "phases": {}}
    phase_data = defaultdict(lambda: {"alloc_mib": [], "reserved_mib": [],
                                       "max_alloc_mib": [], "max_reserved_mib": []})

    # Search both output_dir and logs_dir for rank_metrics files
    search_dirs = []
    for d in [output_dir, logs_dir]:
        if os.path.isdir(d):
            search_dirs.append(d)

    rank_files = set()
    for sd in search_dirs:
        for f in os.listdir(sd):
            if f.startswith("rank_metrics.") and f.endswith(".jsonl"):
                rank_files.add(os.path.join(sd, f))

    for rp in rank_files:
        try:
            with open(rp) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    result["records"] += 1
                    phase = rec.get("phase", "unknown")
                    mem = rec.get("memory", {})
                    if isinstance(mem, dict):
                        phase_data[phase]["alloc_mib"].append(mem.get("allocated_mib", 0))
                        phase_data[phase]["reserved_mib"].append(mem.get("reserved_mib", 0))
                        phase_data[phase]["max_alloc_mib"].append(mem.get("max_alloced_mib", mem.get("max_allocated_mib", 0)))
                        phase_data[phase]["max_reserved_mib"].append(mem.get("max_reserved_mib", 0))
        except (FileNotFoundError, PermissionError):
            continue

    for phase in sorted(phase_data.keys()):
        d = phase_data[phase]
        if not d["alloc_mib"]:
            continue
        result["phases"][phase] = {
            "count": len(d["alloc_mib"]),
            "avg_allocated_gb": round(sum(d["alloc_mib"]) / len(d["alloc_mib"]) / 1024, 2),
            "avg_reserved_gb": round(sum(d["reserved_mib"]) / len(d["reserved_mib"]) / 1024, 2),
            "peak_allocated_gb": round(max(d["max_alloc_mib"]) / 1024, 2),
            "peak_reserved_gb": round(max(d["max_reserved_mib"]) / 1024, 2),
        }

    return result


def analyze_rollouts(rollout_path: str) -> dict:
    """Scan rollouts.readable.jsonl for decision distribution and error analysis.

    This function streams the file line by line (no full load into memory).
    """
    result = {
        "decision_distribution": {},
        "parse_errors": {},
        "tool_call_issues": {},
        "numeric_precision": {},
        "completion_content_score_buckets": {},
        "groups_scanned": 0,
        "completions_scanned": 0,
        "action_type_error_count": 0,
    }

    if not os.path.exists(rollout_path):
        return result

    decision_counts = Counter()
    group_debug_flags = Counter()
    parse_error_counter = Counter()
    tool_call_issues = Counter()
    content_buckets = Counter()
    dist_diffs = []
    angle_diffs = []
    action_type_errors = 0
    groups_with_clean_flags = 0
    total_not_correct = 0
    total_groups = 0

    with open(rollout_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "graspo_group":
                continue

            total_groups += 1
            decision = rec.get("decision", "")
            decision_counts[decision] += 1
            group_debug = rec.get("group_debug", {})

            if decision != "trainable_not_correct":
                continue

            total_not_correct += 1
            targets = rec.get("targets", [])

            # Track group-level debug flags
            has_any_flag = False
            for k in ["missing_json_marker_count", "unclosed_json_fence_count",
                      "invalid_extracted_json_count", "likely_truncated_json_count",
                      "tool_call_parse_error_count", "tool_call_count_mismatch_count"]:
                val = group_debug.get(k, 0)
                if val > 0:
                    group_debug_flags[k] += 1
                    has_any_flag = True
            if not has_any_flag:
                groups_with_clean_flags += 1

            # Analyze each completion
            for c in rec.get("completions", []):
                result["completions_scanned"] += 1
                parse_errors = c.get("parse_errors", [])
                if parse_errors:
                    for err in parse_errors:
                        parse_error_counter[err] += 1

                if c.get("tool_call_count_mismatch", False):
                    tool_call_issues["count_mismatch"] += 1
                if not c.get("parsed_tool_calls"):
                    tool_call_issues["no_tool_calls"] += 1
                else:
                    matched_target_idx = c.get("matched_target_index", -1) or -1
                    if 0 <= matched_target_idx < len(targets):
                        target = targets[matched_target_idx]
                        target_tc = target.get("output", {}).get("tool_calls", [])
                        if target_tc and c.get("parsed_tool_calls"):
                            tc = c["parsed_tool_calls"][0]
                            tgt = target_tc[0]
                            tc_name = tc.get("name", "")
                            tgt_name = tgt.get("name", "")
                            tc_args = tc.get("arguments", {})
                            tgt_args = tgt.get("arguments", {})

                            if tc_name != tgt_name:
                                pass  # wrong tool name — tracked via parse_errors
                            else:
                                tc_at = tc_args.get("action_type", "")
                                tgt_at = tgt_args.get("action_type", "")
                                if tc_at != tgt_at:
                                    action_type_errors += 1
                                else:
                                    # Same action type — numeric precision
                                    for pkey in ["distance_cm", "angle_deg"]:
                                        tc_val = tc_args.get(pkey)
                                        tgt_val = tgt_args.get(pkey)
                                        if tc_val is not None and tgt_val is not None:
                                            try:
                                                diff = abs(float(tc_val) - float(tgt_val))
                                                if pkey == "distance_cm":
                                                    dist_diffs.append(diff)
                                                else:
                                                    angle_diffs.append(diff)
                                            except (ValueError, TypeError):
                                                pass

                # Content score buckets
                cs = c.get("content_score", 0)
                if cs <= 0.001:
                    content_buckets["0_to_0.001"] += 1
                elif cs < 0.5:
                    content_buckets["0.001_to_0.5"] += 1
                elif cs < 0.7:
                    content_buckets["0.5_to_0.7"] += 1
                elif cs < 0.85:
                    content_buckets["0.7_to_0.85"] += 1
                elif cs < 1.0:
                    content_buckets["0.85_to_1.0"] += 1
                else:
                    content_buckets["1.0"] += 1

    result["groups_scanned"] = total_groups
    result["decision_distribution"] = dict(decision_counts.most_common())

    if total_not_correct > 0:
        result["not_correct_group_debug_flags"] = {
            k: {"count": v, "pct": round(v / total_not_correct * 100, 1)}
            for k, v in group_debug_flags.most_common()
        }
        result["not_correct_clean_groups"] = {
            "count": groups_with_clean_flags,
            "pct": round(groups_with_clean_flags / total_not_correct * 100, 1),
        }

        result["completion_content_score_buckets"] = {
            k: {"count": v, "pct": round(v / result["completions_scanned"] * 100, 1) if result["completions_scanned"] else 0}
            for k, v in content_buckets.most_common()
        }

        result["parse_errors"] = dict(parse_error_counter.most_common(20))
        result["tool_call_issues"] = dict(tool_call_issues.most_common())
        result["action_type_error_count"] = action_type_errors

        # Numeric precision
        if dist_diffs:
            dist_diffs.sort()
            total_dist = len(dist_diffs)
            buckets = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 100)]
            dist_buckets = {}
            for lo, hi in buckets:
                cnt = sum(1 for d in dist_diffs if lo <= d < hi)
                if cnt:
                    dist_buckets[f"{lo}-{hi}cm"] = {"count": cnt, "pct": round(cnt / total_dist * 100, 1)}
            result["numeric_precision"]["distance_cm"] = {
                "samples": total_dist,
                "mean_abs_error": round(sum(dist_diffs) / total_dist, 2),
                "median_abs_error": round(dist_diffs[total_dist // 2], 2),
                "max_abs_error": round(max(dist_diffs), 2),
                "min_abs_error": round(min(dist_diffs), 2),
                "distribution": dist_buckets,
            }
        if angle_diffs:
            angle_diffs.sort()
            total_angle = len(angle_diffs)
            result["numeric_precision"]["angle_deg"] = {
                "samples": total_angle,
                "mean_abs_error": round(sum(angle_diffs) / total_angle, 2),
                "median_abs_error": round(angle_diffs[total_angle // 2], 2),
                "max_abs_error": round(max(angle_diffs), 2),
            }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured training stats from a GRASPO run.")
    parser.add_argument("output_dir", help="Training output directory (e.g. outputs/my_run/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <output_dir>/training_stats.json)")
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip("/")
    if not os.path.isdir(output_dir):
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(output_dir, "training_stats.json")
    logs_dir = os.path.join(output_dir, "logs")

    print(f"Analyzing training run: {output_dir}")
    print(f"  Logs: {logs_dir}")
    print()

    # ── Parse training.log ──
    log_path = os.path.join(logs_dir, "training.log")
    if os.path.exists(log_path):
        print("  [1/3] Parsing training.log ... ", end="", flush=True)
        training_stats = parse_training_log(log_path)
        print(f"OK ({training_stats['total_steps']} steps, "
              f"{training_stats['total_elapsed_sec']/3600:.1f}h)")
    else:
        print("  [1/3] training.log not found — skipping")
        training_stats = {}

    # ── Parse rank_metrics ──
    rank_files = []
    for d in [output_dir, logs_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "rank_metrics" in f and f.endswith(".jsonl"):
                    rank_files.append(os.path.join(d, f))
    if rank_files:
        print("  [2/3] Parsing rank_metrics ... ", end="", flush=True)
        gpu_stats = parse_rank_metrics(output_dir, logs_dir)
        print(f"OK ({gpu_stats['records']} records, {len(gpu_stats['phases'])} phases)")
    else:
        print("  [2/3] rank_metrics not found — skipping")
        gpu_stats = {}

    # ── Analyze rollout logs ──
    rollout_path = os.path.join(logs_dir, "rollouts.readable.jsonl")
    if os.path.exists(rollout_path):
        # Check file size
        size_gb = os.path.getsize(rollout_path) / (1024**3)
        print(f"  [3/3] Scanning rollouts.readable.jsonl ({size_gb:.1f} GB) ... ", end="", flush=True)
        rollout_stats = analyze_rollouts(rollout_path)
        print(f"OK ({rollout_stats['groups_scanned']} groups, "
              f"{rollout_stats['completions_scanned']} completions)")
    else:
        print("  [3/3] rollouts.readable.jsonl not found — skipping")
        rollout_stats = {}

    # ── Assemble ──
    stats = {
        "run_dir": output_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/analyze_training.py",
        "training": training_stats,
        "gpu_memory": gpu_stats,
        "rollout_analysis": rollout_stats,
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print()
    print(f"Stats written to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print()
    print("Next step: feed this JSON to an LLM with:")
    print('  "Analyze this GRASPO training run and identify issues, trends,')
    print('   and suggestions."')


if __name__ == "__main__":
    main()
