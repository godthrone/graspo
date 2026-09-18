"""结果收集器 CLI 的冒烟测试（合成运行目录，不触 GPU）。

验证：抽取退出码 / 步数 / loss 序列 / 四件套产物，产出 ledger.jsonl 与 ledger.md；
证据不足时判不通过而不是崩溃或默认通过。
"""

import json
import subprocess
import sys
from pathlib import Path

_COLLECTOR = Path(__file__).resolve().parents[2] / "scripts" / "collect_results.py"

_MANIFEST = {
    "schema_version": 1,
    "tiers": [
        {
            "tier_id": "T010",
            "model": "9B",
            "algorithm": "SFT",
            "mode": "LoRA",
            "backend": "native",
            "cards": 1,
            "gpus": [0],
            "status": "ready",
        }
    ],
}

_TRAINER_STATE = {
    "global_step": 6,
    "epoch": 1.0,
    "log_history": [
        {"loss": 1.0, "grad_norm": 1.0},
        {"loss": 0.7, "grad_norm": 0.9},
        {"loss": 0.5, "grad_norm": 0.8},
    ],
}

_STDOUT = "{'loss': 1.0, 'grad_norm': 1.0, 'learning_rate': 5e-05}\n" + (
    "{'loss': 0.5, 'grad_norm': 0.8, 'learning_rate': 5e-05}\n" * 3
)


def _make_run(root: Path, tier_id: str = "T010") -> None:
    run = root / tier_id
    run.mkdir(parents=True)
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "stdout.log").write_text(_STDOUT, encoding="utf-8")
    out = run / "outputs" / tier_id
    (out / "logs").mkdir(parents=True)
    (out / "final").mkdir(parents=True)
    (out / "config.yaml").write_text("train_method: sft\n", encoding="utf-8")
    (out / "logs" / "training.log").write_text("train ok\n", encoding="utf-8")
    (out / "logs" / "events.jsonl").write_text('{"event": "train_step"}\n', encoding="utf-8")
    (out / "trainer_state.json").write_text(json.dumps(_TRAINER_STATE), encoding="utf-8")


def _run_collector(tmp_path: Path, *extra: str) -> dict:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    out = tmp_path / "ledger"
    completed = subprocess.run(
        [
            sys.executable,
            str(_COLLECTOR),
            "--manifest",
            str(manifest),
            "--run-root",
            str(tmp_path / "runs"),
            "--out",
            str(out),
            "--context-length",
            "8192",
            "--date",
            "2026-09-18",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    records = [
        json.loads(line)
        for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    return records[0]


def test_collector_extracts_evidence_and_fails_closed_without_weight_evidence(tmp_path):
    _make_run(tmp_path / "runs")

    record = _run_collector(tmp_path)

    assert record["tier_id"] == "T010"
    # A1 通过；权重证据缺失 → A2 fail-closed；checkpoint 交付物为空 → A3 不通过。
    assert record["criteria"]["A1"] is True
    assert record["criteria"]["A2"] is False
    assert record["status"] == "❌ 失败"
    assert record["failure_class"] != "真 OOM"
    assert record["max_context"] is None
    assert (tmp_path / "ledger" / "ledger.md").exists()


def test_collector_reports_untested_when_run_dir_missing(tmp_path):
    (tmp_path / "runs").mkdir()

    record = _run_collector(tmp_path)

    assert record["status"] == "— 未测"
    assert "运行目录不存在" in record["note"]


def test_collector_pairs_rerun_for_a4(tmp_path):
    _make_run(tmp_path / "runs")
    _make_run(tmp_path / "reruns")

    record = _run_collector(tmp_path, "--rerun-root", str(tmp_path / "reruns"))

    # 双跑的步数与 loss 序列一致 → A4 通过（即使 A2/A3 因缺权重证据未通过）。
    assert record["criteria"]["A4"] is True


# ── 🟡-11：数值异常判据必须前置于"日志里出现 OOM 字样" ──────────────────────
#
# 硬要求是"只有**真 OOM** 才计入最大可行上下文"。若一次运行的日志里恰好出现
# `CUDA out of memory`（例如真 OOM 之后数值崩溃、或日志里带历史 OOM 记录），
# 而 A6 结论是 NaN/Inf，那真正的结论是**数值异常**——不能因此把该档上下文长度
# 记进"最大可行上下文"。这两条用例把该优先级锁进端到端链路（collector → judge）。

_OOM_STDOUT = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"


def test_collector_numeric_anomaly_wins_over_oom_text(tmp_path):
    _make_run(tmp_path / "runs")
    run = tmp_path / "runs" / "T010"
    (run / "stdout.log").write_text(_OOM_STDOUT, encoding="utf-8")
    state = json.loads(json.dumps(_TRAINER_STATE))
    state["log_history"][-1]["loss"] = float("nan")
    (run / "outputs" / "T010" / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    record = _run_collector(tmp_path)

    assert record["criteria"]["A6"] is False
    assert record["failure_class"] == "数值异常"
    assert record["failure_class"] != "真 OOM"
    # 关键：不得被当作"最大可行上下文"的候选。
    assert record["max_context"] is None


def test_collector_real_oom_without_numeric_anomaly_still_counts(tmp_path):
    """对照：数值健康 + 日志含 OOM ⇒ 仍是真 OOM（修复没有把这条通道关掉）。"""
    _make_run(tmp_path / "runs")
    (tmp_path / "runs" / "T010" / "stdout.log").write_text(_OOM_STDOUT, encoding="utf-8")

    record = _run_collector(tmp_path)

    assert record["criteria"]["A6"] is True
    assert record["failure_class"] == "真 OOM"
    # 台账 `max_context` 只在整档通过时填数（ledger_row 的口径）；这里锁住
    # "真 OOM 才允许成为候选"这一层：备注里必须出现真 OOM 的候选说明。
    assert "真 OOM" in record["note"]
