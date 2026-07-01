"""GraspoFlowTrainer 纯函数工具。

不依赖 self 状态，可独立测试（宪法 §8.4 拆分后保留数据变换工具）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from graspo.backends.graspoflow.trainer.stats import (
    GraspoFlowEpochStats,
    GraspoFlowTrainStats,
)
from graspo.core.advantage import group_advantages
from graspo.core.graspo_parity import lower_median

# ── 时间戳 ────────────────────────────────────────────────────────────────────


def _timestamp() -> str:
    """返回当前时区的 ISO 格式时间戳。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ── advantage 扩展 ─────────────────────────────────────────────────────────────


def expand_advantages_like(rewards: list[float], old_log_probs: Any) -> Any:
    """将 group 级 advantage 扩展为与 old_log_probs 同 shape 的 tensor。"""
    import torch

    values = torch.tensor(
        group_advantages(rewards),
        dtype=old_log_probs.dtype,
        device=old_log_probs.device,
    ).unsqueeze(1)
    return values.expand_as(old_log_probs)


# ── 统计序列化/反序列化 ──────────────────────────────────────────────────────


def train_stats_to_dict(stats: GraspoFlowTrainStats) -> dict[str, Any]:
    return {
        "total_groups": stats.total_groups,
        "perfect_skipped": stats.perfect_skipped,
        "retries": stats.retries,
        "invalid": stats.invalid,
        "invalid_no_preference_gap": stats.invalid_no_preference_gap,
        "trainable": stats.trainable,
        "trainable_max_correct": stats.trainable_max_correct,
        "trainable_not_correct": stats.trainable_not_correct,
        "optimized_steps": stats.optimized_steps,
    }


def epoch_stats_to_dict(stats: GraspoFlowEpochStats) -> dict[str, Any]:
    return {
        "epoch": stats.epoch,
        "samples_seen": stats.samples_seen,
        "attempt_groups": stats.attempt_groups,
        "completion_count": stats.completion_count,
        "perfect_skipped": stats.perfect_skipped,
        "retries": stats.retries,
        "invalid": stats.invalid,
        "invalid_no_preference_gap": stats.invalid_no_preference_gap,
        "trainable": stats.trainable,
        "trainable_max_correct": stats.trainable_max_correct,
        "trainable_not_correct": stats.trainable_not_correct,
        "reward_mean_sum": stats.reward_mean_sum,
        "content_mean_sum": stats.content_mean_sum,
        "best_reward": stats.best_reward,
    }


def train_stats_from_dict(raw: dict[str, Any]) -> GraspoFlowTrainStats:
    return GraspoFlowTrainStats(
        total_groups=int(raw.get("total_groups") or raw.get("attempt_groups") or 0),
        perfect_skipped=int(raw.get("perfect_skipped") or 0),
        retries=int(raw.get("retries") or 0),
        invalid=int(raw.get("invalid") or 0),
        invalid_no_preference_gap=int(raw.get("invalid_no_preference_gap") or 0),
        trainable=int(raw.get("trainable") or 0),
        trainable_max_correct=int(raw.get("trainable_max_correct") or 0),
        trainable_not_correct=int(raw.get("trainable_not_correct") or 0),
        optimized_steps=int(raw.get("optimized_steps") or 0),
    )


def epoch_stats_from_dict(raw: dict[str, Any]) -> GraspoFlowEpochStats:
    return GraspoFlowEpochStats(
        epoch=int(raw.get("epoch") or 0),
        samples_seen=int(raw.get("samples_seen") or 0),
        attempt_groups=int(raw.get("attempt_groups") or 0),
        completion_count=int(raw.get("completion_count") or raw.get("completions") or 0),
        perfect_skipped=int(raw.get("perfect_skipped") or 0),
        retries=int(raw.get("retries") or 0),
        invalid=int(raw.get("invalid") or 0),
        invalid_no_preference_gap=int(raw.get("invalid_no_preference_gap") or 0),
        trainable=int(raw.get("trainable") or 0),
        trainable_max_correct=int(raw.get("trainable_max_correct") or 0),
        trainable_not_correct=int(raw.get("trainable_not_correct") or 0),
        reward_mean_sum=float(raw.get("reward_mean_sum") or 0.0),
        content_mean_sum=float(raw.get("content_mean_sum") or 0.0),
        best_reward=float(raw.get("best_reward") or 0.0),
    )


# ── reward 辅助 ────────────────────────────────────────────────────────────────


def group_stats(rewards: list[float]) -> dict[str, float | int]:
    """计算一组 reward 的统计摘要。"""
    if not rewards:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0, "range": 0.0}
    minimum = min(rewards)
    maximum = max(rewards)
    return {
        "count": len(rewards),
        "min": minimum,
        "median": lower_median(rewards),
        "max": maximum,
        "mean": sum(rewards) / len(rewards),
        "range": maximum - minimum,
    }


def reward_detail(result: Any) -> dict[str, Any]:
    """从 RewardResult 提取可读的评分详情。"""
    extracted = dict(result.extracted)
    valid_extracted_json = None
    if "answer" in extracted:
        try:
            answer = extracted["answer"]
            if isinstance(answer, str) and answer.strip():
                json.loads(answer.strip())
                valid_extracted_json = True
        except (TypeError, ValueError):
            valid_extracted_json = False
    return {
        "raw_score": float(result.raw_score),
        "max_score": float(result.max_score),
        "extracted": extracted,
        "parsed_tool_calls": extracted.get("tool_calls"),
        "parser": extracted.get("parser"),
        "parse_errors": extracted.get("parse_errors"),
        "extra_text": extracted.get("extra_text"),
        "matched_target_index": result.matched_target_index,
        "matched_target_id": result.matched_target_id,
        "target_scores": result.target_scores,
        "useless_text_length": len(result.useless_text),
        "valid_extracted_json": valid_extracted_json,
    }


def generated_token_counts(generation: Any) -> list[int]:
    """从 generation 中提取每条 completion 的生成 token 数。

    当 action_mask 不可用时返回空列表——这是透明降级（宪法 3.2），不影响训练，
    仅影响监控日志中的 token 计数。降级原因通过 warnings 告知用户。
    """
    try:
        return [int(value) for value in generation.action_mask.detach().sum(dim=1).cpu().tolist()]
    except (AttributeError, TypeError, RuntimeError):
        logging.getLogger("graspo.trainer").warning(
            "generated_token_counts: action_mask unavailable, token counts will be empty"
        )
        return []


# ── 元数据处理 ─────────────────────────────────────────────────────────────────


def safe_sample_metadata(sample: Any) -> dict[str, Any]:
    """从 sample 中提取安全可输出的元数据（不含媒体原始数据）。"""
    redacted = {
        key: value
        for key, value in sample.metadata.items()
        if key not in {"image", "images", "video", "videos"}
    }
    if sample.media:
        counts: dict[str, int] = {}
        for item in sample.media:
            media_type = str(item.get("type") or "unknown")
            counts[media_type] = counts.get(media_type, 0) + 1
        redacted["media"] = {
            "count": len(sample.media),
            "types": counts,
        }
    return redacted


def public_generation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """从生成元数据中提取公开可输出的部分。"""
    public = {key: value for key, value in metadata.items() if not str(key).startswith("_")}
    private_rows = metadata.get("_multimodal_rows")
    if isinstance(private_rows, list) and private_rows:
        media_counts: dict[str, int] = {}
        for row in private_rows:
            if not isinstance(row, dict):
                continue
            for item in row.get("media") or []:
                if not isinstance(item, dict):
                    continue
                media_type = str(item.get("type") or "unknown")
                media_counts[media_type] = media_counts.get(media_type, 0) + 1
        public["multimodal"] = {
            "row_count": len(private_rows),
            "media_counts": media_counts,
        }
    return public


def experience_metadata_for_row(
    metadata: dict[str, Any] | None, row_index: int
) -> dict[str, Any] | None:
    """从生成元数据中提取第 row_index 条 experience 的元数据。"""
    if not metadata:
        return None
    rows = metadata.get("_multimodal_rows")
    if isinstance(rows, list):
        if row_index >= len(rows):
            raise RuntimeError(
                f"generation multimodal metadata has {len(rows)} rows, cannot index row {row_index}"
            )
        return {"_multimodal_rows": [rows[row_index]]}
    return dict(metadata)


# ── tool-call 辅助 ─────────────────────────────────────────────────────────────


def is_pure_tool_call_task(targets: Any) -> bool:
    """判断 targets 是否为纯 tool-call 任务（无 content 字段）。"""
    if not isinstance(targets, list) or not targets:
        return False
    has_content = any(
        isinstance(t, dict) and isinstance(t.get("output"), dict) and "content" in t["output"]
        for t in targets
    )
    has_tool_calls = any(
        isinstance(t, dict) and isinstance(t.get("output"), dict) and "tool_calls" in t["output"]
        for t in targets
    )
    return has_tool_calls and not has_content


def tool_call_count_mismatch_count(details: list[dict[str, Any]], targets: Any) -> int:
    """统计 tool-call 数量与 targets 不匹配的 completion 数量。"""
    target_counts = _target_tool_call_counts(targets)
    return sum(
        1
        for detail in details
        if detail.get("parsed_tool_calls") is not None
        and len(detail.get("parsed_tool_calls") or []) not in target_counts
    )


def _target_tool_call_counts(targets: Any) -> set[int]:
    counts: set[int] = set()
    if not isinstance(targets, list):
        return {1}
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        calls = output.get("tool_calls") if isinstance(output, dict) else None
        if isinstance(calls, list):
            counts.add(len(calls))
    return counts or {1}
