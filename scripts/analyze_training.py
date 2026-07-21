#!/usr/bin/env python3
"""
analyze_training.py — Extract structured stats from a GRASPO training run.

Usage:
    python scripts/analyze_training.py outputs/<run_name>/
    python scripts/analyze_training.py outputs/<run_name>/ --output my_stats.json

Output:
    A JSON file (default: training_stats.json in the run directory) containing:
    - Epoch and step summaries from training.log
    - GPU/memory stats from rank_metrics
    - Intra-group reward gap analysis (the core GRASPO signal)
    - Tool call count mismatch analysis (spatial perception quality)
    - Content score stratification (how close to correct each group is)
    - Sampling strategy diagnosis with concrete recommendations
    - Decision distribution, parse errors, numeric precision

What this script tells you:
    GRASPO learns by comparing completions within a group. If 8 completions from
    the same prompt all have nearly identical reward, there is no preference signal
    — the advantage is zero, and the gradient is zero. This script measures that.
    It also tracks whether the model is failing because it doesn't see objects
    (tool_call_count_mismatch) or because it sees them but gets the parameters wrong.

Workflow:
    1. Run this script on a training output directory.
    2. Read the stdout summary — it diagnoses the most important issues.
    3. Feed the output JSON to an LLM for deeper analysis if needed.

Notes:
    - Only the rollouts.readable.jsonl is fully scanned (streaming, line by line).
    - training.log and rank_metrics are parsed for aggregate stats.
    - Large runs may take a few minutes to scan the rollout log.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_config(output_dir: str) -> dict:
    """Read config.yaml from the run directory to extract sampling parameters.

    Returns {} if the config file is not found (non-fatal).
    """
    config_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_path):
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return {}

    training = cfg.get("training", {})
    return {
        "train_method": cfg.get("train_method", ""),
        "temperature": training.get("temperature"),
        "top_p": training.get("top_p"),
        "rollout_group_size": training.get("rollout_group_size"),
        "max_new_tokens": training.get("max_new_tokens"),
        "learning_rate": training.get("learning_rate"),
        "max_epochs": training.get("max_epochs"),
        "max_steps": training.get("max_steps"),
        "model_path": cfg.get("model", {}).get("model_path", ""),
        "train_data_path": cfg.get("data", {}).get("train_path", ""),
        "lora_r": cfg.get("lora", {}).get("r"),
        "lora_alpha": cfg.get("lora", {}).get("alpha"),
    }


# ---------------------------------------------------------------------------
# Training log parsing
# ---------------------------------------------------------------------------

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
        try:
            result["last_log_timestamp"] = steps[-1]["timestamp"]
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

    # Per-step timing trend (last 20)
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


# ---------------------------------------------------------------------------
# Rank metrics parsing
# ---------------------------------------------------------------------------

def parse_rank_metrics(output_dir: str, logs_dir: str) -> dict:
    """Parse rank_metrics.*.jsonl files for GPU/memory stats."""
    result = {"records": 0, "phases": {}}
    phase_data = defaultdict(lambda: {"alloc_mib": [], "reserved_mib": [],
                                       "max_alloc_mib": [], "max_reserved_mib": []})

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
                        phase_data[phase]["max_alloc_mib"].append(
                            mem.get("max_alloced_mib", mem.get("max_allocated_mib", 0)))
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


# ---------------------------------------------------------------------------
# Rollout analysis (single pass — the file is the largest artifact)
# ---------------------------------------------------------------------------

def analyze_rollouts(rollout_path: str) -> dict:
    """Scan rollouts.readable.jsonl in a single pass.

    Collects:
    - Decision distribution and parse errors (existing)
    - Intra-group reward gap per group (new — core GRASPO signal)
    - Tool call count mismatch direction (new — spatial perception)
    - High-content completion stratification (new — how close to correct)
    - Content score bucketing and numeric precision (existing)

    All per-group data is stored as scalar values (floats, ints, bools),
    so memory is safe even for multi-GB rollout files.
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
        # --- new fields ---
        "gap_analysis": {},
        "tool_call_count_mismatch": {},
        "not_correct_stratification": {},
    }

    if not os.path.exists(rollout_path):
        return result

    # --- Counters and accumulators ---
    decision_counts = Counter()
    group_debug_flags = Counter()
    parse_error_counter = Counter()
    content_buckets = Counter()
    dist_diffs = []
    angle_diffs = []
    action_type_errors = 0
    groups_with_clean_flags = 0
    total_not_correct = 0
    total_groups = 0

    # Per-group gap data: (decision, gap, max_r, min_r, mean_r, epoch, is_clean)
    group_gaps = []

    # Tool call count mismatch direction
    mismatch_too_many = 0
    mismatch_too_few = 0
    mismatch_groups = 0
    target_tc_counts = Counter()
    model_tc_counts = Counter()

    # High-content completion counts per not_correct group
    high_content_counts = []

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
            epoch = rec.get("epoch", -1)

            # --- Per-group gap ---
            completions = rec.get("completions", [])
            if completions:
                rewards = [c.get("reward", 0) for c in completions]
                max_r = max(rewards)
                min_r = min(rewards)
                gap = max_r - min_r
                mean_r = sum(rewards) / len(rewards)

                has_mismatch = group_debug.get("tool_call_count_mismatch_count", 0) > 0
                has_other_flag = any(
                    group_debug.get(k, 0) > 0
                    for k in ["missing_json_marker_count", "unclosed_json_fence_count",
                              "invalid_extracted_json_count", "likely_truncated_json_count",
                              "tool_call_parse_error_count"]
                )
                is_clean = (not has_mismatch and not has_other_flag)

                group_gaps.append({
                    "decision": decision,
                    "gap": gap,
                    "max_reward": max_r,
                    "min_reward": min_r,
                    "mean_reward": mean_r,
                    "epoch": epoch,
                    "is_clean": is_clean,
                    "has_tool_call_mismatch": has_mismatch,
                })

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

            # --- Tool call mismatch direction ---
            # completion-level "tool_call_count_mismatch" is not populated;
            # we detect mismatch by comparing parsed_tool_calls length vs target
            tc_mismatch_count = group_debug.get("tool_call_count_mismatch_count", 0)
            if tc_mismatch_count > 0:
                mismatch_groups += 1
                target_tc_count = (
                    len(targets[0].get("output", {}).get("tool_calls", []))
                    if targets else 0
                )
                target_tc_counts[target_tc_count] += 1

                for c in completions:
                    parsed = c.get("parsed_tool_calls", [])
                    model_count = len(parsed)
                    # Only count completions that actually mismatch the target
                    if model_count != target_tc_count:
                        model_tc_counts[model_count] += 1
                        if model_count > target_tc_count:
                            mismatch_too_many += 1
                        else:
                            mismatch_too_few += 1

            # --- High-content stratification ---
            high = sum(1 for c in completions if c.get("content_score", 0) >= 0.85)
            high_content_counts.append(high)

            # Analyze each completion (existing logic)
            for c in completions:
                result["completions_scanned"] += 1
                parse_errors = c.get("parse_errors", [])
                if parse_errors:
                    for err in parse_errors:
                        parse_error_counter[err] += 1

                if c.get("tool_call_count_mismatch", False):
                    if "count_mismatch" not in result["tool_call_issues"]:
                        result["tool_call_issues"]["count_mismatch"] = 0
                    # already counted under mismatch direction above
                if not c.get("parsed_tool_calls"):
                    if "no_tool_calls" not in result["tool_call_issues"]:
                        result["tool_call_issues"]["no_tool_calls"] = 0
                    result["tool_call_issues"]["no_tool_calls"] += 1
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

                            if tc_name == tgt_name:
                                tc_at = tc_args.get("action_type", "")
                                tgt_at = tgt_args.get("action_type", "")
                                if tc_at != tgt_at:
                                    action_type_errors += 1
                                else:
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

    # --- Assemble existing fields ---
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
            k: {"count": v, "pct": round(v / result["completions_scanned"] * 100, 1)
                if result["completions_scanned"] else 0}
            for k, v in content_buckets.most_common()
        }

        result["parse_errors"] = dict(parse_error_counter.most_common(20))
        result["tool_call_issues"] = dict(result["tool_call_issues"])
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

    # --- Compute new diagnostics ---
    result["gap_analysis"] = compute_gap_stats(group_gaps)
    result["tool_call_count_mismatch"] = {
        "description": (
            "Tool call count mismatch is a spatial perception issue, not a formatting issue. "
            "The model generates too many or too few tool calls because it fails to see objects "
            "in certain directions. Counts here reveal whether the model's spatial perception "
            "bias is toward over-detection or under-detection."
        ),
        "groups_with_mismatch": mismatch_groups,
        "pct_of_not_correct": round(mismatch_groups / total_not_correct * 100, 1) if total_not_correct else 0,
        "total_mismatched_completions": mismatch_too_many + mismatch_too_few,
        "too_many_tool_calls": {
            "count": mismatch_too_many,
            "description": "Model generated more tool calls than target — over-detection bias.",
        },
        "too_few_tool_calls": {
            "count": mismatch_too_few,
            "description": "Model generated fewer tool calls than target — under-detection bias.",
        },
        "target_tool_call_count_distribution": dict(target_tc_counts.most_common()),
        "model_tool_call_count_distribution": dict(model_tc_counts.most_common()),
    }
    result["not_correct_stratification"] = compute_high_content_stratification(
        high_content_counts, total_not_correct
    )

    return result


# ---------------------------------------------------------------------------
# Gap analysis (new — core GRASPO diagnostic)
# ---------------------------------------------------------------------------

def compute_gap_stats(group_gaps: list) -> dict:
    """Compute intra-group reward gap statistics from per-group data.

    The intra-group gap (max_reward - min_reward within a group of completions
    for the same prompt) is the fundamental signal GRASPO needs. If the gap is
    near zero, the advantage is near zero, and the gradient is near zero —
    training makes no progress.

    Returns a dict with:
    - overall distribution (mean, median, percentiles)
    - bucketed histogram
    - by decision type
    - by clean/flagged split
    - epoch-level trend
    - gradient signal effectiveness (what % of groups cross each threshold)
    """
    if not group_gaps:
        return {"description": "No gap data available (no rollout groups found)."}

    nc_gaps = [g["gap"] for g in group_gaps if g["decision"] == "trainable_not_correct"]
    nc_clean = [g["gap"] for g in group_gaps
                if g["decision"] == "trainable_not_correct" and g["is_clean"]]
    nc_flagged = [g["gap"] for g in group_gaps
                  if g["decision"] == "trainable_not_correct" and not g["is_clean"]]

    # Decision-type breakdown
    by_decision = {}
    for decision in set(g["decision"] for g in group_gaps):
        d_gaps = sorted(g["gap"] for g in group_gaps if g["decision"] == decision)
        if not d_gaps:
            continue
        by_decision[decision] = {
            "count": len(d_gaps),
            "mean": round(sum(d_gaps) / len(d_gaps), 4),
            "median": round(d_gaps[len(d_gaps) // 2], 4),
            "min": round(d_gaps[0], 4),
            "max": round(d_gaps[-1], 4),
        }

    # Epoch trend
    epoch_gaps = defaultdict(list)
    epoch_mismatch = defaultdict(lambda: {"total": 0, "mismatch": 0})
    for g in group_gaps:
        if g["decision"] == "trainable_not_correct":
            epoch_gaps[g["epoch"]].append(g["gap"])
            epoch_mismatch[g["epoch"]]["total"] += 1
            if g["has_tool_call_mismatch"]:
                epoch_mismatch[g["epoch"]]["mismatch"] += 1

    epoch_trend = []
    for ep in sorted(epoch_gaps.keys()):
        gaps = epoch_gaps[ep]
        if not gaps:
            continue
        gs = sorted(gaps)
        em = epoch_mismatch[ep]
        epoch_trend.append({
            "epoch": ep,
            "groups": len(gaps),
            "mean_gap": round(sum(gaps) / len(gaps), 4),
            "median_gap": round(gs[len(gs) // 2], 4),
            "p25_gap": round(gs[len(gs) // 4], 4),
            "p75_gap": round(gs[len(gs) * 3 // 4], 4),
            "mismatch_pct": round(em["mismatch"] / em["total"] * 100, 1) if em["total"] else 0,
        })

    # Bucketed histogram
    buckets = [(0, 0.001), (0.001, 0.005), (0.005, 0.01), (0.01, 0.02),
               (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0)]
    histogram = []
    for lo, hi in buckets:
        cnt = sum(1 for g in nc_gaps if lo <= g < hi)
        histogram.append({
            "range": [lo, hi],
            "count": cnt,
            "pct": round(cnt / len(nc_gaps) * 100, 1) if nc_gaps else 0,
        })

    # Gradient signal effectiveness
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
    signal_effectiveness = {}
    for t in thresholds:
        cnt = sum(1 for g in nc_gaps if g >= t)
        if t >= 0.05:
            label = "effective gradient signal"
        elif t >= 0.02:
            label = "very weak signal"
        else:
            label = "almost no signal"
        signal_effectiveness[f"gap_ge_{str(t).replace('.', '_')}"] = {
            "threshold": t,
            "groups": cnt,
            "pct": round(cnt / len(nc_gaps) * 100, 1) if nc_gaps else 0,
            "signal_quality": label,
        }

    nc_sorted = sorted(nc_gaps)

    return {
        "description": (
            "Intra-group reward gap is the core GRASPO signal. For each prompt, "
            "the model generates N completions (rollout_group_size). GRASPO learns "
            "by comparing rewards within the group — completions with higher reward "
            "get positive advantage, lower ones get negative advantage. If the gap "
            "(max - min) is near zero, advantage is zero for all completions, and "
            "the gradient is zero — the step contributes nothing to learning."
        ),
        "not_correct_total": len(nc_gaps),
        "not_correct_clean": len(nc_clean),
        "not_correct_flagged": len(nc_flagged),
        "overall": {
            "mean": round(sum(nc_gaps) / len(nc_gaps), 4) if nc_gaps else 0,
            "median": round(nc_sorted[len(nc_sorted) // 2], 4) if nc_sorted else 0,
            "p10": round(nc_sorted[len(nc_sorted) // 10], 4) if nc_sorted else 0,
            "p25": round(nc_sorted[len(nc_sorted) // 4], 4) if nc_sorted else 0,
            "p75": round(nc_sorted[len(nc_sorted) * 3 // 4], 4) if nc_sorted else 0,
            "p90": round(nc_sorted[len(nc_sorted) * 9 // 10], 4) if nc_sorted else 0,
            "min": round(nc_sorted[0], 4) if nc_sorted else 0,
            "max": round(nc_sorted[-1], 4) if nc_sorted else 0,
        },
        "histogram": histogram,
        "by_decision": by_decision,
        "clean_vs_flagged": {
            "clean": {
                "count": len(nc_clean),
                "mean": round(sum(nc_clean) / len(nc_clean), 4) if nc_clean else 0,
                "median": round(sorted(nc_clean)[len(nc_clean) // 2], 4) if nc_clean else 0,
            },
            "flagged": {
                "count": len(nc_flagged),
                "mean": round(sum(nc_flagged) / len(nc_flagged), 4) if nc_flagged else 0,
                "median": round(sorted(nc_flagged)[len(nc_flagged) // 2], 4) if nc_flagged else 0,
            },
        },
        "epoch_trend": epoch_trend,
        "gradient_signal_effectiveness": signal_effectiveness,
    }


# ---------------------------------------------------------------------------
# High-content stratification (new)
# ---------------------------------------------------------------------------

def compute_high_content_stratification(high_content_counts: list, total_not_correct: int) -> dict:
    """Analyze how many not_correct groups are "close to correct."

    A group where all 8 completions have content_score >= 0.85 means the model
    consistently produces well-formatted, semantically correct outputs — but the
    action choice is wrong. These are the most valuable training samples for GRASPO
    because they isolate the spatial reasoning problem cleanly.

    Groups with 0 high-content completions are likely format/spatial failures where
    the model couldn't even produce valid tool calls.
    """
    if not high_content_counts:
        return {"description": "No not_correct groups found."}

    hc_dist = Counter(high_content_counts)
    any_high = sum(1 for h in high_content_counts if h > 0)
    all_high = sum(1 for h in high_content_counts if h == max(high_content_counts))

    return {
        "description": (
            "For each not_correct group, counts how many of its completions have "
            "content_score >= 0.85 (well-formatted, semantically correct). "
            "Groups where all completions are high-content but still not_correct "
            "are the purest signal: the model understands the scene but gets the "
            "action parameters wrong. Groups with 0 high-content completions are "
            "likely spatial perception failures (model doesn't see the objects)."
        ),
        "total_not_correct_groups": total_not_correct,
        "groups_with_any_high_content": {
            "count": any_high,
            "pct": round(any_high / total_not_correct * 100, 1) if total_not_correct else 0,
            "description": "Groups where at least one completion has content >= 0.85.",
        },
        "groups_with_all_high_content": {
            "count": all_high,
            "pct": round(all_high / total_not_correct * 100, 1) if total_not_correct else 0,
            "description": (
                "Groups where ALL completions have content >= 0.85. "
                "These are the most valuable training samples — correct format, "
                "correct semantics, wrong action parameters."
            ),
        },
        "high_content_count_distribution": {
            str(k): {"count": v, "pct": round(v / total_not_correct * 100, 1)}
            for k, v in sorted(hc_dist.items())
        },
    }


# ---------------------------------------------------------------------------
# Sampling diagnosis (new)
# ---------------------------------------------------------------------------

def diagnose_sampling(config: dict, gap_stats: dict) -> dict:
    """Correlate sampling parameters with observed gap to diagnose root cause.

    Reads the sampling parameters from config (temperature, top_p, rollout_group_size)
    and the observed gap statistics. Produces a diagnosis with concrete recommendations.

    The key insight: if top_p is too low, the token distribution is heavily truncated,
    and all completions from the same prompt are nearly identical — killing the
    intra-group variance that GRASPO depends on.
    """
    if not config or not gap_stats:
        return {"description": "Insufficient data for sampling diagnosis."}

    temp = config.get("temperature")
    top_p = config.get("top_p")
    group_size = config.get("rollout_group_size", 8)

    sig = gap_stats.get("gradient_signal_effectiveness", {})
    effective_pct = sig.get("gap_ge_0_05", {}).get("pct", 0)

    issues = []
    recommendations = []
    severity = "ok"

    if top_p is not None and top_p < 0.7:
        issues.append(
            f"top_p={top_p} heavily truncates the token distribution — "
            f"only the top {top_p*100:.0f}% of probability mass is considered. "
            f"With {group_size} completions from the same prompt, outputs are nearly identical."
        )
        recommendations.append(
            f"Raise top_p to 0.9-1.0. This allows tail tokens and increases "
            f"completion diversity within each group."
        )
        severity = "warning"

    if top_p is not None and top_p <= 0.6 and effective_pct < 20:
        severity = "critical"

    if temp is not None and temp <= 1.0 and top_p is not None and top_p <= 0.6:
        issues.append(
            f"temperature={temp} + top_p={top_p} is a conservative sampling strategy. "
            f"Both parameters suppress diversity. For GRASPO, this is self-defeating: "
            f"without intra-group variance, the advantage is always near zero."
        )
        recommendations.append(
            "Raise temperature to 1.2-1.5 to increase token-level diversity."
        )
        if severity != "critical":
            severity = "warning"

    if effective_pct < 10:
        issues.append(
            f"Only {effective_pct}% of not_correct groups have gap >= 0.05. "
            f"This means {100-effective_pct}% of training steps contribute no "
            f"meaningful gradient. The model is spinning its wheels."
        )
        severity = "critical"

    if group_size <= 8 and effective_pct < 15:
        recommendations.append(
            f"Increase rollout_group_size (currently {group_size}) to 16. "
            f"More samples per group = higher chance of finding diverse completions."
        )

    return {
        "description": (
            "Correlates the sampling parameters (temperature, top_p, rollout_group_size) "
            "from the training config with the observed intra-group reward gap. "
            "GRASPO requires intra-group diversity to compute meaningful advantages. "
            "Conservative sampling (low temp, low top_p) kills this diversity."
        ),
        "sampling_params": {
            "temperature": temp,
            "top_p": top_p,
            "rollout_group_size": group_size,
        },
        "observed": {
            "effective_groups_pct": effective_pct,
            "description": "Percentage of not_correct groups with gap >= 0.05 (usable signal).",
        },
        "diagnosis": {
            "severity": severity,
            "issues": issues,
            "recommendations": recommendations,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ── Parse config ──
    config_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_path):
        print("  [1/4] Parsing config.yaml ... ", end="", flush=True)
        config = parse_config(output_dir)
        print("OK")
    else:
        print("  [1/4] config.yaml not found — skipping")
        config = {}

    # ── Parse training.log ──
    log_path = os.path.join(logs_dir, "training.log")
    if os.path.exists(log_path):
        print("  [2/4] Parsing training.log ... ", end="", flush=True)
        training_stats = parse_training_log(log_path)
        print(f"OK ({training_stats['total_steps']} steps, "
              f"{training_stats['total_elapsed_sec']/3600:.1f}h)")
    else:
        print("  [2/4] training.log not found — skipping")
        training_stats = {}

    # ── Parse rank_metrics ──
    rank_files = []
    for d in [output_dir, logs_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "rank_metrics" in f and f.endswith(".jsonl"):
                    rank_files.append(os.path.join(d, f))
    if rank_files:
        print("  [3/4] Parsing rank_metrics ... ", end="", flush=True)
        gpu_stats = parse_rank_metrics(output_dir, logs_dir)
        print(f"OK ({gpu_stats['records']} records, {len(gpu_stats['phases'])} phases)")
    else:
        print("  [3/4] rank_metrics not found — skipping")
        gpu_stats = {}

    # ── Analyze rollout logs ──
    rollout_path = os.path.join(logs_dir, "rollouts.readable.jsonl")
    if os.path.exists(rollout_path):
        size_gb = os.path.getsize(rollout_path) / (1024**3)
        print(f"  [4/4] Scanning rollouts.readable.jsonl ({size_gb:.1f} GB) ... ",
              end="", flush=True)
        rollout_stats = analyze_rollouts(rollout_path)
        print(f"OK ({rollout_stats['groups_scanned']} groups, "
              f"{rollout_stats['completions_scanned']} completions)")
    else:
        print("  [4/4] rollouts.readable.jsonl not found — skipping")
        rollout_stats = {}

    # ── Sampling diagnosis ──
    sampling_diag = diagnose_sampling(
        config,
        rollout_stats.get("gap_analysis", {}),
    )

    # ── Assemble JSON ──
    stats = {
        "run_dir": output_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/analyze_training.py",
        "config": config,
        "training": training_stats,
        "gpu_memory": gpu_stats,
        "rollout_analysis": rollout_stats,
        "sampling_diagnosis": sampling_diag,
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print()
    print(f"Stats written to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print()

    # ── Self-diagnosing summary ──
    _print_summary(stats)


def _print_summary(stats: dict) -> None:
    """Print a self-diagnosing human-readable summary to stdout."""
    tr = stats.get("training", {})
    cfg = stats.get("config", {})
    ra = stats.get("rollout_analysis", {})
    ga = ra.get("gap_analysis", {})
    sd = stats.get("sampling_diagnosis", {})

    run_name = os.path.basename(stats["run_dir"])
    duration_h = tr.get("total_elapsed_sec", 0) / 3600
    total_steps = tr.get("total_steps", 0)
    epochs = len(tr.get("epochs", []))
    temp = cfg.get("temperature", "?")
    top_p = cfg.get("top_p", "?")
    group_size = cfg.get("rollout_group_size", "?")

    print("=" * 70)
    print(f"  GRASPO Training Analysis: {run_name}")
    print(f"  Duration: {duration_h:.1f}h | Steps: {total_steps} | "
          f"Epochs: {epochs} | t={temp}, p={top_p}, group={group_size}")
    print()

    # --- Decision flow ---
    dd = ra.get("decision_distribution", {})
    total_groups = ra.get("groups_scanned", 0)
    if total_groups > 0:
        print("  Decision flow:")
        for decision, count in sorted(dd.items(), key=lambda x: -x[1]):
            pct = count / total_groups * 100
            marker = ""
            if decision == "trainable_not_correct":
                marker = "  <- trainable"
            elif decision == "perfect_skip":
                marker = "  <- best-case outcome"
            print(f"    {decision:<28s} {count:>6d} ({pct:5.1f}%){marker}")
        print()

    # --- Intra-group gap ---
    if ga:
        overall = ga.get("overall", {})
        sig = ga.get("gradient_signal_effectiveness", {})
        effective = sig.get("gap_ge_0_05", {}).get("pct", 0)

        print("  Intra-group reward gap (max - min within same prompt):")
        print(f"    median: {overall.get('median', 0):.4f}  "
              f"mean: {overall.get('mean', 0):.4f}  "
              f"P25: {overall.get('p25', 0):.4f}  "
              f"P75: {overall.get('p75', 0):.4f}")
        print(f"    groups with gap >= 0.05 (effective signal): {effective}%")

        diag = sd.get("diagnosis", {})
        severity = diag.get("severity", "ok")
        issues = diag.get("issues", [])
        recommendations = diag.get("recommendations", [])

        if severity == "critical":
            print(f"    !! CRITICAL: {effective}% of steps have usable gradient signal.")
            for issue in issues:
                print(f"       {issue}")
            for rec in recommendations:
                print(f"       -> {rec}")
        elif severity == "warning":
            print(f"    !  WARNING: Low gradient signal. Consider adjusting sampling params.")
            for rec in recommendations:
                print(f"       -> {rec}")
        print()

    # --- Tool call count mismatch ---
    tcmm = ra.get("tool_call_count_mismatch", {})
    if tcmm:
        mm_pct = tcmm.get("pct_of_not_correct", 0)
        too_many = tcmm.get("too_many_tool_calls", {}).get("count", 0)
        too_few = tcmm.get("too_few_tool_calls", {}).get("count", 0)
        total_mm = too_many + too_few

        print("  Tool call count mismatch (spatial perception):")
        print(f"    {mm_pct}% of not_correct groups have wrong number of tool calls.")
        if total_mm > 0:
            print(f"    too many: {too_many} ({too_many/total_mm*100:.0f}%)  "
                  f"too few: {too_few} ({too_few/total_mm*100:.0f}%)")
        print("    This is a spatial issue — the model fails to see objects in")
        print("    certain directions, not a formatting problem.")

        # Check epoch trend
        epoch_trend = ga.get("epoch_trend", [])
        if len(epoch_trend) >= 3:
            first_mm = epoch_trend[0].get("mismatch_pct", 0)
            last_mm = epoch_trend[-1].get("mismatch_pct", 0)
            if abs(last_mm - first_mm) < 5:
                print(f"    Mismatch rate has NOT improved: {first_mm}% -> {last_mm}%")
                print("    -> spatial perception is not being learned.")
        print()

    # --- Not-correct stratification ---
    ncs = ra.get("not_correct_stratification", {})
    if ncs:
        all_high = ncs.get("groups_with_all_high_content", {})
        print("  Not-correct stratification:")
        print(f"    all completions content>=0.85 but wrong action: "
              f"{all_high.get('count', 0)} ({all_high.get('pct', 0)}%)")
        print("    These are the most valuable training samples — correct format,")
        print("    correct semantics, but wrong spatial parameters.")
        print()

    # --- Content score clustering ---
    csb = ra.get("completion_content_score_buckets", {})
    if csb:
        dominant = max(csb.items(), key=lambda x: x[1].get("pct", 0))
        print("  Content score clustering:")
        print(f"    {dominant[1].get('pct', 0)}% of completions in bucket "
              f"'{dominant[0]}'")
        if dominant[1].get("pct", 0) > 60:
            print("    Heavy clustering -> reward variance collapses -> gap shrinks.")
        print()

    print("=" * 70)
    print("  Output: " + stats.get("run_dir", "") + "/training_stats.json")
    print("  Next: feed this JSON to an LLM for deeper analysis.")
    print("=" * 70)


if __name__ == "__main__":
    main()