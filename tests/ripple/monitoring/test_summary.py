"""Tests for ``graspo.ripple.monitoring.summary`` — monitoring summaries."""

from graspo.ripple.monitoring.summary import monitor_group


def test_monitor_group_perfect_all_right():
    """monitor_group computes reward statistics for a perfect group."""
    payload = {
        "decision": "trainable",
        "rewards": [1.0, 1.0, 0.8],
        "content_scores": [0.95, 1.0, 0.9],
        "reward_details": [
            {"valid_extracted_json": True},
            {"valid_extracted_json": True},
            {"valid_extracted_json": True},
        ],
        "completions": [
            "```json\n{}\n```",
            "```json\n{}\n```",
            "```json\n{}\n```",
        ],
        "targets": [{"output": {"content": {"key": "val"}}}],
    }

    result = monitor_group(payload)

    assert result["decision"] == "trainable"
    assert result["reward_mean"] > 0.9
    assert result["reward_max"] == 1.0
    assert result["reward_range"] > 0.0
    assert result["content_all_one"] is False


def test_monitor_group_empty_completions():
    """monitor_group handles empty completions gracefully."""
    payload = {
        "decision": "invalid",
        "rewards": [],
        "content_scores": [],
        "reward_details": [],
        "completions": [],
        "targets": [],
    }

    result = monitor_group(payload)

    assert result["decision"] == "invalid"
    assert result["reward_mean"] == 0.0
    assert result["reward_max"] == 0.0
    assert result["reward_range"] == 0.0


def test_summary_module_imports_compact_functions():
    """Verify compact summary functions are importable."""
    from graspo.ripple.monitoring.summary import (  # noqa: F401
        compact_batch_summary,
        compact_optimize_metrics,
        compact_timing_summary,
        reward_batch_summary,
        reward_window_summary,
        training_health,
    )


def _group_payload(decision: str, rewards: list[float]) -> dict:
    """构造最小 group payload（终结口径测试用）。"""
    return {
        "decision": decision,
        "rewards": rewards,
        "content_scores": [r * 0.5 for r in rewards],
        "base_content_scores": [r * 0.25 for r in rewards],
        "reward_details": [{} for _ in rewards],
        "completions": ["" for _ in rewards],
        "targets": [{"output": {"tool_calls": [{"name": "f", "arguments": {}}]}}],
    }


def test_batch_summary_terminal_means_exclude_retry():
    """质量均值只统计终结 attempt（v20 教训：retry 中间态不入均值）。"""
    from graspo.ripple.monitoring.summary import reward_batch_summary

    summary = reward_batch_summary(
        [
            _group_payload("trainable_max_correct", [1.0, 0.9, 0.8]),
            _group_payload("retry", [0.0, 0.0, 0.0]),  # retry 中间态
            _group_payload("trainable_not_correct", [0.4, 0.3]),
        ],
        rollout_group_size=8,
        effective_batch_size=4,
    )
    # 均值来自两个终结 attempt 的 completion 加权拉平（[1.0,0.9,0.8,0.4,0.3]），
    # retry 的 0.0 不入
    assert abs(summary["reward_mean"] - 0.68) < 1e-9
    assert summary["reward_min"] == 0.3
    assert summary["reward_max"] == 1.0
    assert summary["reward_median"] == 0.8
    # retry 计数与比例独立成桶
    assert summary["retry_group_count"] == 1
    assert summary["terminal_group_count"] == 2
    assert abs(summary["retry_rate"] - 1 / 3) < 1e-9
    # observed_completion_count 保持全量（含 retry 的真实评分条数）
    assert summary["observed_completion_count"] == 8
    # 决策计数保留全量
    assert summary["decision_counts"]["retry"] == 1


def test_window_summary_terminal_means_exclude_retry():
    """窗口质量均值同样过滤 retry 中间态。"""
    from collections import deque

    from graspo.ripple.monitoring.summary import monitor_group, reward_window_summary

    groups = deque(
        [
            monitor_group(_group_payload("trainable_max_correct", [1.0, 1.0])),
            monitor_group(_group_payload("retry", [0.0, 0.0])),
        ],
        maxlen=50,
    )
    window = reward_window_summary(groups)
    assert window["count"] == 2  # 计数含 retry
    assert abs(window["reward_mean_avg"] - 1.0) < 1e-9  # 均值只看终结 attempt
    assert abs(window["content_mean_avg"] - 0.5) < 1e-9
    assert window["content_all_zero_rate"] == 0.0


def test_compact_decisions_ratio():
    """mc_ratio 是真实质量指标；无 trainable 组时为 None。"""
    from graspo.ripple.monitoring.summary import compact_decisions

    d = compact_decisions(
        perfect_skip=1,
        trainable_max_correct=3,
        trainable_not_correct=7,
        invalid=2,
        invalid_no_preference_gap=0,
        retry_attempts=5,
    )
    assert d["trainable"]["ratio"] == 0.3
    d2 = compact_decisions(
        perfect_skip=0,
        trainable_max_correct=0,
        trainable_not_correct=0,
        invalid=0,
        invalid_no_preference_gap=0,
        retry_attempts=0,
    )
    assert d2["trainable"]["ratio"] is None


def test_health_high_retry_rate_warns():
    """retry 积压过半触发健康告警（v16/v17 退化早信号）。"""
    from graspo.ripple.monitoring.summary import reward_batch_summary, training_health

    batch = reward_batch_summary(
        [_group_payload("retry", [0.0] * 8)],
        rollout_group_size=8,
        effective_batch_size=4,
    )
    health = training_health({"optimized": True}, batch, {"count": 0})
    assert "batch_high_retry_rate" in health["reasons"]
    assert health["ok"] is False
