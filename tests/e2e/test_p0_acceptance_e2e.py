"""端到端负向测试（collector → judge），用 **F-4 实测产物**做夹具。

夹具来源：`.local/hb-workspace/20260915-152747/task-f4-nan/evidence/
probe-4c8k-same-sample/`（4 卡 PP=4 全参、同一样本 ×4、第 2 步起数值崩坏）。
这是真实发生的 run：**exit_code=0、写了 final checkpoint、stdout 看着正常**，
但权重自第 3 步起完全冻结。

本文件回答工作包必须回答的那个问题：**修后该 NaN run 是否仍判不通过？**
并且证明修复**不再依赖** ``loss: null`` 恰好被解析成 NaN 这条偶然路径。
"""

import importlib.util
import json
import sys
from pathlib import Path

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / ".local"
    / "hb-workspace"
    / "20260915-152747"
    / "task-p0-acceptance"
    / "fixtures"
)
_COLLECTOR = Path(__file__).resolve().parents[2] / "scripts" / "collect_results.py"

_MANIFEST = {
    "schema_version": 1,
    "tiers": [
        {
            "tier_id": "T018",
            "model": "9B",
            "algorithm": "SFT",
            "mode": "全量",
            "backend": "native",
            "cards": 4,
            "gpus": [0, 1, 2, 4],
            "status": "ready",
        }
    ],
}


def _load_collector():
    """按文件路径加载 collector（避免经 graspo/__init__ 拉入 pydantic/torch）。"""
    spec = importlib.util.spec_from_file_location("_p0_collector", _COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _materialize_run(
    root: Path,
    *,
    rewrite_null_losses: bool = False,
    tier_id: str = "T018",
) -> Path:
    """把 F-4 夹具铺成一个运行目录（含 rank_metrics 旁路与 stdout）。"""
    run = root / tier_id
    out = run / "outputs" / tier_id
    logs = out / "logs"
    logs.mkdir(parents=True)
    (out / "final").mkdir(parents=True)
    (out / "config.yaml").write_text(
        (_FIXTURES / "f4probe-config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / "final" / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "exit_code").write_text(
        (_FIXTURES / "f4probe-exit_code").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run / "stdout.log").write_text(
        (_FIXTURES / "f4probe-stdout.log").read_text(encoding="utf-8"), encoding="utf-8"
    )
    metrics_text = (_FIXTURES / "f4probe-rank_metrics.rank_00000.jsonl").read_text(
        encoding="utf-8"
    )
    if rewrite_null_losses:
        # ★ 注入缺陷：把 `"global_loss_mean": null` 改写成 `0.0`（人类可读行早就是这个
        #   口径）。旧判定器会因此判"数值健康"通过——本测试证明现在**不会**。
        metrics_text = metrics_text.replace('"global_loss_mean": null', '"global_loss_mean": 0.0')
    (logs / "rank_metrics.rank_00000.jsonl").write_text(metrics_text, encoding="utf-8")
    (logs / "training.log").write_text("train ok\n", encoding="utf-8")
    return run


def _run_collector(tmp_path: Path, collector) -> dict:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    out = tmp_path / "ledger"
    args = collector.build_parser().parse_args(
        [
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
        ]
    )
    assert collector.run(args) == 0
    records = [
        json.loads(line)
        for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    return records[0]


def test_f4_nan_run_still_fails_open_loop(tmp_path):
    """★ 工作包必答项：修后该 NaN run **仍判不通过**（面向调用方的直接证明）。"""
    collector = _load_collector()
    _materialize_run(tmp_path / "runs")
    record = _run_collector(tmp_path, collector)

    assert record["status"] == "❌ 失败"
    # step2 的全局 grad_norm 是真 NaN ⇒ 分类为数值异常（不得被当"真 OOM"，
    # 也不得因为 step3/4 的 null 而变成"未分类"——数值异常优先）。
    assert record["failure_class"] == "数值异常"
    assert record["max_context"] is None
    assert record["criteria"]["A2"] is False  # 训练未真推进（梯度被跳过）
    assert record["criteria"]["A6"] is False  # 数值异常
    # 逐步证据与旁路口径都必须落到台账里，便于事后复核
    assert record["optimizer_steps_per_step"] == [4, 4, 0, 0]
    assert record["nonfinite_skips"] == 2
    assert record["series_source"] == "rank_metrics"


def test_f4_nan_run_still_fails_after_null_rewritten_to_zero(tmp_path):
    """★★ 核心负向证据（任务 A④）：把 ``loss: null`` 改写成 ``0.0`` 后**仍不通过**。

    旧路径下 ``grad_norms`` 全 finite（26.25 / 1.03e7 / 0.0 / 0.0），一旦 null
    变成 0.0，A6 会判"数值健康"并让整档通过。这里证明修复不依赖那条偶然路径。
    """
    collector = _load_collector()
    _materialize_run(tmp_path / "runs", rewrite_null_losses=True)
    record = _run_collector(tmp_path, collector)

    assert record["status"] == "❌ 失败"
    assert record["criteria"]["A2"] is False
    assert record["max_context"] is None


def test_series_comes_from_rank_metrics_not_stdout(tmp_path):
    """读数口径：全局序列必须取自 rank_metrics 旁路（stdout 只是 rank0 局部值）。"""
    collector = _load_collector()
    _materialize_run(tmp_path / "runs")
    run = tmp_path / "runs" / "T018"
    output_dir = collector.find_output_dir(run, "T018")
    series = collector.extract_steps_and_series(
        output_dir, (run / "stdout.log").read_text(encoding="utf-8")
    )

    assert series.source == "rank_metrics"
    assert series.losses[0] == 0.005218505859375  # 全局 loss（stdout 的 rank0 局部值是 0.0）
    assert series.grad_norms[0] == 10.03515625
    # 第 2 步全局 grad_norm 为 NaN ⇒ 必须原样保留，不得被过滤成有限值
    assert series.grad_norms[1] != series.grad_norms[1]  # NaN
    assert series.optimizer_steps_per_step == [4, 4, 0, 0]
    assert series.nonfinite_skips == 2
    # 口径不一致必须被显式记录（rank0 局部 loss=0.0 vs 全局 loss=0.0052）
    assert any("口径不一致" in note for note in series.notes)


def test_missing_loss_is_explicit_not_nan_by_accident(tmp_path):
    """``loss: null`` 只能被**显式**识别：不得靠"恰好解析成 NaN"。

    用一个**只有 stdout**（无 rank_metrics 旁路）的合成目录验证：
      - ``loss: null`` 留下显式标记（``loss_unavailable``）；
      - 该步在序列里是 ``MISSING_SENTINEL``（**不是** NaN）⇒ "没有读数"不会被
        升格成"数值异常"这个**事实断言**。
    """
    collector = _load_collector()
    run = tmp_path / "runs" / "T018"
    out = run / "outputs" / "T018"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text("train_method: sft\n", encoding="utf-8")
    (out / "final").mkdir()
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "stdout.log").write_text(
        '\n'.join(
            [
                'SFT step 1: loss=0.0209 grad_norm=26.2500',
                '{"event": "sft_step", "step": 1, "loss": 0.0209, "grad_norm": 26.25}',
                'SFT step 3: loss=0.0 grad_norm=0.0',
                '{"event": "sft_step", "step": 3, "loss": null, "grad_norm": 0.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    series = collector.extract_steps_and_series(out, (run / "stdout.log").read_text())

    assert series.loss_unavailable is True
    assert any("loss: null" in note for note in series.notes)
    # 哨兵是 judge 层用的同一个对象（对象身份识别）
    assert any(value is collector._judge.MISSING_SENTINEL for value in series.losses)
    # 且不得被当成"真读到的 NaN"
    assert series.loss_nonfinite is False


def test_only_null_run_is_unclassified_not_numeric_anomaly(tmp_path):
    """★ 指挥官改判后的必测项：**全程只有 null、没有任何真 NaN** 的 run。

    断言：分类为 ``未分类（需人工判定）``（不是"数值异常"），且仍 ❌ 失败、
    仍不计入最大可行上下文（fail-closed）。
    """
    collector = _load_collector()
    run = tmp_path / "runs" / "T018"
    out = run / "outputs" / "T018"
    logs = out / "logs"
    logs.mkdir(parents=True)
    (out / "config.yaml").write_text("train_method: sft\n", encoding="utf-8")
    (out / "final").mkdir()
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "stdout.log").write_text("SFT complete\n", encoding="utf-8")
    # rank_metrics 旁路：每一步的全局 loss 都是 null（进程被外部杀掉/超时等，
    # 我们**根本没有读数**——不得据此断言"数值异常"），grad_norm 全 finite。
    rows = [
        {
            "event": "rank_memory",
            "phase": "pipeline_sft_train_batch_after",
            "metrics": {
                "global_loss_mean": None,
                "global_grad_norm_mean": 0.0,
                "global_optimizer_steps_sum": 0,
                "skipped_nonfinite": 1,
            },
        }
        for _ in range(3)
    ]
    (logs / "rank_metrics.rank_00000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    series = collector.extract_steps_and_series(out, (run / "stdout.log").read_text())
    assert series.loss_unavailable is True
    assert series.loss_nonfinite is False

    record = _run_collector(tmp_path, collector)
    assert record["status"] == "❌ 失败"
    assert record["failure_class"] == "未分类（需人工判定）"
    assert record["failure_class"] != "数值异常"
    assert record["max_context"] is None  # 不计入最大可行上下文
