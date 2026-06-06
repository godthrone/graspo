from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_analyzer_module():
    path = Path("scripts/analyze_native_profile.py")
    spec = importlib.util.spec_from_file_location("analyze_native_profile", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_native_profile_summarizes_train_gpu_and_rank_metrics(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "gpu_memory").mkdir()
    train_step = {
        "event": "train_step",
        "run": {"step": 2},
        "epoch": {"index": 1},
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
            "rollout_sec": 40.0,
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
    (run_dir / "rank_metrics.rank_00000.jsonl").write_text(json.dumps(rank_event) + "\n", encoding="utf-8")

    summary = analyzer.summarize_run(run_dir, skip_warmup_steps=0)

    assert summary["latest_step"] == 2
    assert summary["latest_reward_mean"] == 0.5
    assert summary["decode_tokens_per_sec"] == 5.0
    assert summary["trainable_groups_per_hour"] == 144.0
    assert summary["gpu"]["per_gpu"]["0"]["memory_used_mib_peak"] == 2048.0
    assert summary["rank"]["per_rank"]["0"]["pipeline_train_schedule"] == "one_f_one_b"
