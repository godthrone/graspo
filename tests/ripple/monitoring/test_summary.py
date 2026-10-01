"""Tests for ``graspo.ripple.monitoring.summary`` — monitoring summaries."""

import pytest

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


# ── 权重变化量的口径（T034 只读对照实验；缺陷 P18 的结构性恒 0）──────────────


def _full_param_metrics(**overrides) -> dict:
    """一条 native 全参档的逐步指标（键名与形状取自 run 自产的 rank_metrics JSONL）。

    全参档的关键事实：``lora_norm_*`` 是 **None**（无 lora 参数 ⇒ 指标不适用，
    见 ``flow/progress_metrics.training_norm_event``），真正有读数的是
    ``trainable_norm_*`` / ``global_trainable_norm_delta_mean``。
    """
    metrics = {
        "optimized": True,
        "tuner_type": "full",
        "norm_metric": "trainable_parameter_l2_norm",
        "lora_norm_before": None,
        "lora_norm_after": None,
        "lora_norm_delta": None,
        "trainable_norm_before": 1796.7959909563967,
        "trainable_norm_after": 1796.7959836404752,
        "trainable_norm_delta": -7.315921493500355e-06,
        "global_lora_norm_delta_mean": None,
        "global_trainable_norm_delta_mean": -7.315921493500355e-06,
    }
    metrics.update(overrides)
    return metrics


def test_full_param_health_reads_the_trainable_delta_not_the_lora_placeholder():
    """全参档必须读 ``trainable_norm_delta``：``lora_norm_*`` 在本档是 None 占位。

    修前：``_metric_float`` 先取 ``global_lora_norm_delta_mean``（None）→ 回落
    ``lora_norm_delta``（也是 None）→ ``float(None or 0.0) == 0.0`` ⇒ 必然产出
    ``zero_lora_delta`` 假告警（真机日志原文就是这个 reason）。
    """
    from graspo.ripple.monitoring.summary import training_health

    health = training_health(_full_param_metrics(), {}, {})
    assert "zero_lora_delta" not in health["reasons"]
    assert health["ok"] is True


def test_full_param_health_does_not_fabricate_zero_when_the_metric_is_missing():
    """没有读数 ⇒ **未知**，不得冒充 0（§2.2 None 语义 / §8.1 三态）。

    这是"恒 0 假告警"的根：把"取不到"当成"等于 0"。
    """
    from graspo.ripple.monitoring.summary import training_health

    health = training_health(
        _full_param_metrics(trainable_norm_delta=None, global_trainable_norm_delta_mean=None),
        {},
        {},
    )
    assert "zero_lora_delta" not in health["reasons"]


def test_full_param_health_still_flags_a_real_zero_delta():
    """真读数恰好为 0 时**仍然**要告警——短路只针对"取不到"，不放松判据。"""
    from graspo.ripple.monitoring.summary import training_health

    health = training_health(
        _full_param_metrics(trainable_norm_delta=0.0, global_trainable_norm_delta_mean=0.0),
        {},
        {},
    )
    assert "zero_lora_delta" in health["reasons"]


def test_lora_health_keeps_the_original_caliber():
    """LoRA 档的键名与数值语义逐字不变（含 0.0 ⇒ 告警）。"""
    from graspo.ripple.monitoring.summary import training_health

    nonzero = training_health(
        {
            "optimized": True,
            "tuner_type": "lora",
            "lora_norm_delta": 1.6e-06,
            "global_lora_norm_delta_mean": 1.6e-06,
        },
        {},
        {},
    )
    assert "zero_lora_delta" not in nonzero["reasons"]

    zero = training_health(
        {
            "optimized": True,
            "tuner_type": "lora",
            "lora_norm_delta": 0.0,
            "global_lora_norm_delta_mean": 0.0,
        },
        {},
        {},
    )
    assert "zero_lora_delta" in zero["reasons"]


def test_compact_optimize_metrics_reports_the_mode_aware_weight_delta():
    """readable 记录里那个"权重变化量"必须来自本档的真实口径，且带口径来源。"""
    from graspo.ripple.monitoring.summary import compact_optimize_metrics

    full = compact_optimize_metrics(_full_param_metrics())
    assert full["lora_delta_mean"] == pytest.approx(-7.315921493500355e-06)
    assert full["weight_delta_source"] == "global_trainable_norm_delta_mean"

    lora = compact_optimize_metrics(
        {"optimized": True, "lora_norm_delta": 1.6e-06, "global_lora_norm_delta_mean": 1.6e-06}
    )
    assert lora["lora_delta_mean"] == pytest.approx(1.6e-06)
    assert lora["weight_delta_source"] == "global_lora_norm_delta_mean"

    missing = compact_optimize_metrics({"optimized": True})
    assert missing["lora_delta_mean"] is None
    assert missing["weight_delta_source"] is None
