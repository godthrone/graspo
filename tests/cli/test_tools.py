"""CLI 工具命令实现（graspo.cli.tools）的单元测试。

覆盖：validate_reward_scores（只 print 不落盘的评分逻辑）、
evaluate_samples（config 决定输出目录的评测逻辑）、
summarize_run（analyze-profile 的日志汇总逻辑）。
"""

import json
from pathlib import Path

from graspo.cli.tools import evaluate_samples, summarize_run, validate_reward_scores
from graspo.core.schema import GraspoConfig, Sample
from graspo.ripple.parsing.completion import ParsedCompletion, raw_parsed_completion

# ── validate-reward ──────────────────────────────────────────────────────────


def test_validate_reward_scores_ground_truth_content_completion():
    sample = Sample(
        messages=[{"role": "user", "content": "answer"}],
        targets=[{"id": "ok", "output": {"content": {"status": "ok"}}}],
        metadata={},
    )
    scores = validate_reward_scores([sample])

    assert len(scores) == 1
    assert scores[0]["all_right"] is True
    assert scores[0]["reward"] > 0


def test_validate_reward_scores_explicit_completion_is_used():
    sample = Sample(
        messages=[{"role": "user", "content": "answer"}],
        targets=[{"id": "ok", "output": {"content": {"status": "ok"}}}],
        metadata={},
    )
    scores = validate_reward_scores([sample], completions=['```json\n{"status":"ok"}\n```'])

    assert scores[0]["all_right"] is True


def test_validate_reward_scores_tool_call_task():
    sample = Sample(
        messages=[{"role": "user", "content": "call"}],
        targets=[{"id": "t", "output": {"tool_calls": [{"name": "move", "arguments": {"d": 1}}]}}],
        metadata={},
    )
    scores = validate_reward_scores([sample])

    assert len(scores) == 1
    assert scores[0]["all_right"] is True


# ── evaluate-checkpoint ──────────────────────────────────────────────────────


class _Generation:
    def __init__(self, completions: list[str]) -> None:
        self.completions = completions


class _EvalRuntime:
    """评测运行时假件。

    ★ 必须实现 ``parse_completion``：``cli.tools._parse_completion`` 已按 §2.2 改为
    **直接调用**（不再 ``getattr`` 探测 + 静默退回 raw 解析器）。本假件用
    :func:`raw_parsed_completion` 复现当年那条退路的解析结果——它是**显式声明**的
    假件行为，而不是生产代码的隐式回退。
    """

    def __init__(self, completions: list[str]) -> None:
        self.completions = completions

    def is_primary(self) -> bool:
        return True

    def parse_completion(self, completion: str, sample: Sample) -> ParsedCompletion:
        return raw_parsed_completion(completion)

    def generate_sample_groups(self, **kwargs):
        assert len(kwargs["samples"]) == 1
        return [_Generation(self.completions)]

    def generate_groups(self, **kwargs):
        assert len(kwargs["message_batches"]) == 1
        return [_Generation(self.completions)]


def test_evaluate_samples_scores_groups_and_scrubs_media_paths(tmp_path: Path) -> None:
    config = GraspoConfig()
    config.training.rollout_group_size = 2
    sample = Sample(
        messages=[{"role": "user", "content": "read the panel"}],
        targets=[{"id": "ok", "output": {"content": {"status": "ok"}}}],
        metadata={"source": "synthetic"},
        media=[{"type": "image", "path": "/private/panel.png"}],
    )
    perfect = '```json\n{"status":"ok"}\n```'
    partial = '```json\n{"status":"fail"}\n```'

    summary = evaluate_samples(
        _EvalRuntime([perfect, partial]),
        config,
        [sample],
        tmp_path,
        checkpoint="/checkpoints/final",
    )

    assert summary["count"] == 1
    assert summary["completion_count"] == 2
    assert summary["reward_mean"] > 0
    assert summary["reward_range_mean"] > 0
    assert summary["checkpoint"] == "/checkpoints/final"

    rows = [
        json.loads(line)
        for line in (tmp_path / "completions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["metadata"]["media"] == [{"type": "image"}]
    assert rows[0]["targets"] == [{"id": "ok", "output": {"content": {"status": "ok"}}}]
    assert rows[0]["matched_target_id"] == "ok"
    assert "/private/panel.png" not in (tmp_path / "completions.jsonl").read_text(encoding="utf-8")


# ── analyze-profile ──────────────────────────────────────────────────────────


def test_summarize_run_reads_train_gpu_and_rank_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "gpu_memory").mkdir()
    train_step = {
        "event": "train_step",
        "epoch": 1,
        "run_cumulative": {"step": 2},
        "epoch_cumulative": {"reward_mean": 0.5, "content_mean": 0.9},
        "batch": {
            "reward_mean": 0.5,
            "content_mean": 0.9,
            "decisions": {
                "trainable": {"total": 4},
                "terminal": {"invalid": 1, "invalid_no_preference_gap": 2},
            },
        },
        "timing": {
            "total_observed_sec": 100.0,
            "rollout_total_sec": 40.0,
            "optimize_sec": 50.0,
            "decode_tokens": 200,
        },
    }
    (run_dir / "nohup.out").write_text(json.dumps(train_step) + "\n", encoding="utf-8")
    gpu_summary = {
        "sample_count": 2,
        "per_gpu": {
            "0": {
                "samples": 2,
                "memory_used_mib_peak": 2048.0,
                "memory_used_mib_p95": 2048.0,
                "memory_used_mib_mean": 1536.0,
                "utilization_gpu_pct_mean": 75.0,
            }
        },
    }
    (run_dir / "gpu_memory" / "gpu_memory_summary.json").write_text(
        json.dumps(gpu_summary),
        encoding="utf-8",
    )
    rank_event = {
        "event": "rank_memory",
        "phase": "pipeline_train_batch_after",
        "metrics": {
            "rank_metrics": [
                {
                    "rank": 0,
                    "pipeline_train_schedule": "one_f_one_b",
                    "pipeline_stage_timing": {
                        "pipeline_stage_compute_sec": 3.0,
                        "pipeline_backward_autograd_sec": 4.0,
                    },
                }
            ]
        },
    }
    (run_dir / "rank_metrics.rank_00000.jsonl").write_text(
        json.dumps(rank_event) + "\n", encoding="utf-8"
    )

    summary = summarize_run(run_dir, skip_warmup_steps=0)

    assert summary["latest_step"] == 2
    assert summary["latest_reward_mean"] == 0.5
    assert summary["decode_tokens_per_sec"] == 5.0
    assert summary["trainable_groups_per_hour"] == 144.0
    assert summary["gpu"]["per_gpu"]["0"]["memory_used_mib_peak"] == 2048.0
    assert summary["rank"]["per_rank"]["0"]["pipeline_train_schedule"] == "one_f_one_b"
