"""GraspoFlowTrainer 监控摘要与统计压缩（宪法 §8.4 从 helpers.py 按功能域拆分）。"""

import logging
from collections import deque
from typing import Any

from graspo.backends.graspoflow.trainer.helpers import (
    is_pure_tool_call_task,
    tool_call_count_mismatch_count,
)
from graspo.core.graspo_parity import lower_median

# ── 监控与摘要 ─────────────────────────────────────────────────────────────────


def monitor_group(payload: dict[str, Any]) -> dict[str, Any]:
    """从 group payload 生成监控用的摘要数据。"""
    rewards = [float(value) for value in payload.get("rewards", [])]
    content_scores = [float(value) for value in payload.get("content_scores", [])]
    details = payload.get("reward_details", [])
    completions = payload.get("completions", [])
    pure_tool_call = is_pure_tool_call_task(payload.get("targets"))
    return {
        "decision": payload.get("decision"),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_range": max(rewards) - min(rewards) if rewards else 0.0,
        "content_mean": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "content_all_zero": bool(content_scores) and all(value == 0.0 for value in content_scores),
        "content_all_one": bool(content_scores) and all(value == 1.0 for value in content_scores),
        "missing_json_marker_count": (
            0 if pure_tool_call else sum(1 for text in completions if "```json" not in text)
        ),
        "unclosed_json_fence_count": (
            0
            if pure_tool_call
            else sum(1 for text in completions if "```json" in text and text.count("```") < 2)
        ),
        "invalid_extracted_json_count": (
            0
            if pure_tool_call
            else sum(1 for detail in details if detail.get("valid_extracted_json") is False)
        ),
        "likely_truncated_json_count": sum(
            1
            for text, detail in zip(completions, details, strict=False)
            if _likely_truncated_json(text, detail)
        ),
        "tool_call_parse_error_count": sum(1 for detail in details if detail.get("parse_errors")),
        "tool_call_count_mismatch_count": tool_call_count_mismatch_count(
            details, payload.get("targets")
        ),
    }


def reward_window_summary(groups: deque[dict[str, Any]]) -> dict[str, Any]:
    """从最近 N 个 group 的监控数据中生成滑动窗口摘要。"""
    items = list(groups)
    if not items:
        return {
            "count": 0,
            "decision_counts": {},
            "reward_mean_avg": 0.0,
            "reward_max_avg": 0.0,
            "nonzero_range_rate": 0.0,
            "content_mean_avg": 0.0,
            "content_all_zero_rate": 0.0,
            "content_all_one_rate": 0.0,
            "missing_json_marker_count": 0,
            "unclosed_json_fence_count": 0,
            "invalid_extracted_json_count": 0,
            "likely_truncated_json_count": 0,
            "tool_call_parse_error_count": 0,
            "tool_call_count_mismatch_count": 0,
        }
    decision_counts: dict[str, int] = {}
    for item in items:
        decision = str(item.get("decision"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    count = len(items)
    return {
        "count": count,
        "decision_counts": decision_counts,
        "reward_mean_avg": sum(float(item["reward_mean"]) for item in items) / count,
        "reward_max_avg": sum(float(item["reward_max"]) for item in items) / count,
        "nonzero_range_rate": sum(float(item["reward_range"]) > 0.0 for item in items) / count,
        "content_mean_avg": sum(float(item["content_mean"]) for item in items) / count,
        "content_all_zero_rate": sum(bool(item["content_all_zero"]) for item in items) / count,
        "content_all_one_rate": sum(bool(item["content_all_one"]) for item in items) / count,
        "missing_json_marker_count": sum(int(item["missing_json_marker_count"]) for item in items),
        "unclosed_json_fence_count": sum(int(item["unclosed_json_fence_count"]) for item in items),
        "invalid_extracted_json_count": sum(
            int(item["invalid_extracted_json_count"]) for item in items
        ),
        "likely_truncated_json_count": sum(
            int(item["likely_truncated_json_count"]) for item in items
        ),
        "tool_call_parse_error_count": sum(
            int(item.get("tool_call_parse_error_count") or 0) for item in items
        ),
        "tool_call_count_mismatch_count": sum(
            int(item.get("tool_call_count_mismatch_count") or 0) for item in items
        ),
    }


def reward_batch_summary(
    attempts: list[dict[str, Any]],
    *,
    rollout_group_size: int,
    optimize_prompt_batch_size: int,
) -> dict[str, Any]:
    """从一批 rollout attempts 生成 batch 级别摘要。"""
    decision_counts: dict[str, int] = {}
    rewards: list[float] = []
    content_scores: list[float] = []
    group_ranges: list[float] = []
    group_max_median_gaps: list[float] = []
    missing_json_marker_count = 0
    unclosed_json_fence_count = 0
    invalid_extracted_json_count = 0
    likely_truncated_json_count = 0
    tool_call_parse_error_count = 0
    tool_call_count_mismatch = 0

    for attempt in attempts:
        decision = str(attempt.get("decision"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        attempt_rewards = [float(value) for value in attempt.get("rewards", [])]
        attempt_content = [float(value) for value in attempt.get("content_scores", [])]
        rewards.extend(attempt_rewards)
        content_scores.extend(attempt_content)
        if attempt_rewards:
            group_ranges.append(max(attempt_rewards) - min(attempt_rewards))
            group_max_median_gaps.append(max(attempt_rewards) - lower_median(attempt_rewards))
        completions = attempt.get("completions", [])
        details = attempt.get("reward_details", [])
        pure_tool_call = is_pure_tool_call_task(attempt.get("targets"))
        if not pure_tool_call:
            missing_json_marker_count += sum(1 for text in completions if "```json" not in text)
            unclosed_json_fence_count += sum(
                1 for text in completions if "```json" in text and text.count("```") < 2
            )
            invalid_extracted_json_count += sum(
                1 for detail in details if detail.get("valid_extracted_json") is False
            )
        if not pure_tool_call:
            likely_truncated_json_count += sum(
                1
                for text, detail in zip(completions, details, strict=False)
                if _likely_truncated_json(text, detail)
            )
        tool_call_parse_error_count += sum(1 for detail in details if detail.get("parse_errors"))
        tool_call_count_mismatch += tool_call_count_mismatch_count(details, attempt.get("targets"))

    attempt_group_count = len(attempts)
    trainable_group_count = decision_counts.get("trainable_max_correct", 0) + decision_counts.get(
        "trainable_not_correct", 0
    )
    return {
        "unit": "batch_attempt",
        "rollout_group_size": int(rollout_group_size),
        "optimize_prompt_batch_size": int(optimize_prompt_batch_size),
        "attempt_group_count": attempt_group_count,
        "completion_count": attempt_group_count * int(rollout_group_size),
        "observed_completion_count": len(rewards),
        "trainable_group_count": trainable_group_count,
        "trainable_completion_count": trainable_group_count * int(rollout_group_size),
        "retry_group_count": decision_counts.get("retry", 0),
        "retry_completion_count": decision_counts.get("retry", 0) * int(rollout_group_size),
        "perfect_skip_group_count": decision_counts.get("perfect_skip", 0),
        "perfect_skip_completion_count": decision_counts.get("perfect_skip", 0)
        * int(rollout_group_size),
        "invalid_group_count": decision_counts.get("invalid", 0),
        "invalid_completion_count": decision_counts.get("invalid", 0) * int(rollout_group_size),
        "invalid_no_preference_gap_group_count": decision_counts.get(
            "invalid_no_preference_gap", 0
        ),
        "invalid_no_preference_gap_completion_count": decision_counts.get(
            "invalid_no_preference_gap", 0
        )
        * int(rollout_group_size),
        "decision_counts": decision_counts,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_median": lower_median(rewards),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_nonzero_range_group_count": sum(value > 0.0 for value in group_ranges),
        "reward_nonzero_range_rate": (
            sum(value > 0.0 for value in group_ranges) / len(group_ranges) if group_ranges else 0.0
        ),
        "reward_max_median_gap_mean": (
            sum(group_max_median_gaps) / len(group_max_median_gaps)
            if group_max_median_gaps
            else 0.0
        ),
        "content_mean": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "content_all_zero_group_count": sum(
            bool(attempt.get("content_scores"))
            and all(float(value) == 0.0 for value in attempt.get("content_scores", []))
            for attempt in attempts
        ),
        "content_all_one_group_count": sum(
            bool(attempt.get("content_scores"))
            and all(float(value) == 1.0 for value in attempt.get("content_scores", []))
            for attempt in attempts
        ),
        "missing_json_marker_count": missing_json_marker_count,
        "unclosed_json_fence_count": unclosed_json_fence_count,
        "invalid_extracted_json_count": invalid_extracted_json_count,
        "likely_truncated_json_count": likely_truncated_json_count,
        "tool_call_parse_error_count": tool_call_parse_error_count,
        "tool_call_count_mismatch_count": tool_call_count_mismatch,
    }


def _likely_truncated_json(text: str, detail: dict[str, Any]) -> bool:
    has_json = "```json" in text
    if has_json and text.count("```") < 2:
        return True
    if detail.get("valid_extracted_json") is False and has_json:
        stripped = text.rstrip()
        return not (stripped.endswith("```") or stripped.endswith("}"))
    return False


# ── 摘要压缩 ───────────────────────────────────────────────────────────────────


def compact_batch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """将 reward_batch_summary 压缩为精简格式。"""
    decisions = dict(summary.get("decision_counts") or {})
    return {
        "attempt_groups": int(summary.get("attempt_group_count") or 0),
        "completions": int(summary.get("completion_count") or 0),
        "decisions": compact_decisions(
            perfect_skip=int(decisions.get("perfect_skip") or 0),
            trainable_max_correct=int(decisions.get("trainable_max_correct") or 0),
            trainable_not_correct=int(decisions.get("trainable_not_correct") or 0),
            invalid=int(decisions.get("invalid") or 0),
            invalid_no_preference_gap=int(decisions.get("invalid_no_preference_gap") or 0),
            retry_attempts=int(decisions.get("retry") or 0),
        ),
        "reward": {
            "min": float(summary.get("reward_min") or 0.0),
            "median": float(summary.get("reward_median") or 0.0),
            "mean": float(summary.get("reward_mean") or 0.0),
            "max": float(summary.get("reward_max") or 0.0),
            "nonzero_range_rate": float(summary.get("reward_nonzero_range_rate") or 0.0),
            "max_median_gap_mean": float(summary.get("reward_max_median_gap_mean") or 0.0),
        },
        "content": {
            "mean": float(summary.get("content_mean") or 0.0),
            "all_zero_groups": int(summary.get("content_all_zero_group_count") or 0),
            "all_one_groups": int(summary.get("content_all_one_group_count") or 0),
        },
        "debug": {
            "missing_json_marker": int(summary.get("missing_json_marker_count") or 0),
            "unclosed_json_fence": int(summary.get("unclosed_json_fence_count") or 0),
            "invalid_json": int(summary.get("invalid_extracted_json_count") or 0),
            "truncated_json": int(summary.get("likely_truncated_json_count") or 0),
            "tool_call_parse_error": int(summary.get("tool_call_parse_error_count") or 0),
            "tool_call_count_mismatch": int(summary.get("tool_call_count_mismatch_count") or 0),
        },
    }


def compact_decisions(
    *,
    perfect_skip: int,
    trainable_max_correct: int,
    trainable_not_correct: int,
    invalid: int,
    invalid_no_preference_gap: int,
    retry_attempts: int,
) -> dict[str, Any]:
    """将决策计数压缩为分组格式。"""
    trainable_total = int(trainable_max_correct) + int(trainable_not_correct)
    terminal_total = (
        int(perfect_skip) + trainable_total + int(invalid) + int(invalid_no_preference_gap)
    )
    return {
        "rollout_attempts": {
            "total": terminal_total + int(retry_attempts),
            "retry": int(retry_attempts),
            "terminal": terminal_total,
        },
        "terminal": {
            "perfect_skip": int(perfect_skip),
            "trainable": trainable_total,
            "invalid": int(invalid),
            "invalid_no_preference_gap": int(invalid_no_preference_gap),
            "total": terminal_total,
        },
        "trainable": {
            "max_correct": int(trainable_max_correct),
            "not_correct": int(trainable_not_correct),
            "total": trainable_total,
        },
    }


def compact_timing_summary(
    attempt_timings: list[dict[str, Any]],
    *,
    optimize_sec: float,
    checkpoint_sec: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """将各阶段耗时压缩为 summary。"""
    rollout_sec = _sum_timing(attempt_timings, "rollout_sec")
    reward_cpu_sec = _sum_timing(attempt_timings, "reward_cpu_sec")
    decision_sec = _sum_timing(attempt_timings, "decision_sec")
    old_logprob_sec = _sum_timing(attempt_timings, "old_logprob_sec")
    replay_append_sec = _sum_timing(attempt_timings, "replay_append_sec")
    total = rollout_sec + reward_cpu_sec + decision_sec + old_logprob_sec + replay_append_sec
    total += float(optimize_sec) + float(checkpoint_sec)
    return {
        "attempt_count": len(attempt_timings),
        "rollout_prompt_queue_batch_size": max(
            [int(item.get("rollout_prompt_queue_batch_size") or 1) for item in attempt_timings],
            default=1,
        ),
        "rollout_prompt_queue_effective_size": max(
            [int(item.get("rollout_prompt_queue_effective_size") or 1) for item in attempt_timings],
            default=1,
        ),
        "rollout_prompt_queue_fallback_count": sum(
            bool(item.get("rollout_prompt_queue_fallback")) for item in attempt_timings
        ),
        "rollout_sec": round(rollout_sec, 6),
        "reward_cpu_sec": round(reward_cpu_sec, 6),
        "decision_sec": round(decision_sec, 6),
        "old_logprob_sec": round(old_logprob_sec, 6),
        "replay_append_sec": round(replay_append_sec, 6),
        "optimize_sec": round(float(optimize_sec), 6),
        "checkpoint_sec": round(float(checkpoint_sec), 6),
        "prefill_sec": round(_sum_timing(attempt_timings, "prefill_sec"), 6),
        "decode_sec": round(_sum_timing(attempt_timings, "decode_sec"), 6),
        "sampling_sec": round(_sum_timing(attempt_timings, "sampling_sec"), 6),
        "stop_check_sec": round(_sum_timing(attempt_timings, "stop_check_sec"), 6),
        "train_batch_total_sec": round(
            float(metrics.get("train_batch_total_sec") or optimize_sec), 6
        ),
        "optimize_round_sec_sum": round(float(metrics.get("optimize_round_sec_sum") or 0.0), 6),
        "micro_batch_forward_sec": round(float(metrics.get("micro_batch_forward_sec") or 0.0), 6),
        "backward_sec": round(float(metrics.get("backward_sec") or 0.0), 6),
        "optimizer_step_sec": round(float(metrics.get("optimizer_step_sec") or 0.0), 6),
        "micro_batch_count": int(
            metrics.get("micro_batch_count") or metrics.get("optimizer_steps") or 0
        ),
        "decode_tokens": int(_sum_timing(attempt_timings, "decode_tokens")),
        "rollout_generation_split_count": int(
            _sum_timing(attempt_timings, "rollout_generation_split_count")
        ),
        "total_observed_sec": round(total, 6),
    }


def _sum_timing(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item.get(key) or 0.0) for item in items)


def round_timing_details(details: dict[str, Any]) -> dict[str, Any]:
    """将 timing 详情中的浮点数四舍五入到 6 位小数。"""
    rounded: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        elif isinstance(value, (str, int, bool)) or value is None:
            rounded[key] = value
    return rounded


def scalar_generation_timing(metadata: dict[str, Any]) -> dict[str, Any]:
    """从生成元数据中提取标量 timing 指标。"""
    keys = {
        "tokenize_sec",
        "prefill_sec",
        "decode_sec",
        "sampling_sec",
        "stop_check_sec",
        "decode_tokens",
        "rollout_elapsed_sec",
        "rollout_use_kv_cache",
        "rollout_generation_micro_batch_size",
        "rollout_generation_split_count",
        "rollout_empty_cache_after_split",
        "prefill_len",
        "generated_tokens_max",
        "rollout_prompt_queue_batch_size",
        "rollout_prompt_queue_effective_size",
        "rollout_prompt_queue_fallback",
    }
    return {key: value for key, value in metadata.items() if key in keys}


def compact_optimize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """将优化指标压缩为精简格式。"""
    optimizer_steps_per_rank = int(metrics.get("optimizer_steps") or 0)
    global_optimizer_steps = int(
        metrics.get("global_optimizer_steps_sum") or optimizer_steps_per_rank
    )
    return {
        "optimized": bool(metrics.get("optimized")),
        "replay_buffer_trainable_completion_count": int(
            metrics.get("replay_buffer_trainable_completion_count") or 0
        ),
        "replay_buffer_trainable_group_count": float(
            metrics.get("replay_buffer_trainable_group_count") or 0.0
        ),
        "replay_buffer_optimize_threshold": int(
            metrics.get("replay_buffer_optimize_threshold") or 0
        ),
        "optimize_prompt_batch_size": int(metrics.get("optimize_prompt_batch_size") or 0),
        "optimize_iterations_per_step": int(metrics.get("optimize_iterations_per_step") or 0),
        "optimizer_steps_per_rank": optimizer_steps_per_rank,
        "global_optimizer_steps_sum": global_optimizer_steps,
        "loss_mean": _metric_float(metrics, "global_loss_mean", "loss_mean"),
        "grad_norm_mean": _metric_float(metrics, "global_grad_norm_mean", "grad_norm_mean"),
        "lora_delta_mean": _metric_float(metrics, "global_lora_norm_delta_mean", "lora_norm_delta"),
        "skipped_nonfinite": int(metrics.get("skipped_nonfinite") or 0),
        "force_flush": bool(metrics.get("force_flush")),
    }


def _metric_float(metrics: dict[str, Any], preferred: str, fallback: str) -> float:
    value = metrics.get(preferred)
    if value is None:
        value = metrics.get(fallback)
        logging.getLogger("graspo.trainer").warning(
            "metric %r not available, falling back to %r (transparent degradation per §3.2)",
            preferred,
            fallback,
        )
    return float(value or 0.0)


def training_health(
    metrics: dict[str, Any],
    reward_batch: dict[str, Any],
    reward_window: dict[str, Any],
) -> dict[str, Any]:
    """基于训练指标和 reward 统计判断训练健康状态。"""
    reasons: list[str] = []
    if int(metrics.get("skipped_nonfinite") or 0) > 0:
        reasons.append("nonfinite_loss_or_grad")
    lora_delta = _metric_float(metrics, "global_lora_norm_delta_mean", "lora_norm_delta")
    if metrics.get("optimized") and lora_delta == 0.0:
        reasons.append("zero_lora_delta")
    if int(reward_batch.get("attempt_group_count") or 0) > 0:
        if float(reward_batch.get("reward_mean") or 0.0) == 0.0:
            reasons.append("batch_reward_all_zero")
        if int(reward_batch.get("likely_truncated_json_count") or 0) > 0:
            reasons.append("batch_json_truncation_detected")
    if int(reward_window.get("count") or 0) >= 10:
        if float(reward_window.get("reward_mean_avg") or 0.0) == 0.0:
            reasons.append("reward_all_zero_window")
        if float(reward_window.get("nonzero_range_rate") or 0.0) == 0.0:
            reasons.append("no_group_reward_variance_window")
        if float(reward_window.get("content_all_zero_rate") or 0.0) >= 0.8:
            reasons.append("content_score_all_zero_window")
    return {"ok": not reasons, "early_stop_recommended": bool(reasons), "reasons": reasons}
