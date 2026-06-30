"""Tests for ``graspo.backends.graspoflow.trainer.summary`` — monitoring summaries."""

from graspo.backends.graspoflow.trainer.summary import monitor_group


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
        "targets": [
            {"output": {"content": {"key": "val"}}}
        ],
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
    from graspo.backends.graspoflow.trainer.summary import (  # noqa: F401
        compact_batch_summary,
        compact_optimize_metrics,
        compact_timing_summary,
        reward_batch_summary,
        reward_window_summary,
        training_health,
    )
