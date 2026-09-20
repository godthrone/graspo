"""rank_metrics 旁路的 **phase 名注册表** 与「含 metrics 的未登记 phase 显式暴露」回归测试。

背景（2026-09-20 实测伪否，本模块锁死它不再复发）：

训练侧 SFT 有**两条**路径，各自 emit 不同的 phase 名——
``training_sft.py:753`` 的 PP 路径 emit ``pipeline_sft_train_batch_after``，
``training_sft.py:478`` 的普通/单卡路径 emit ``sft_train_batch_after``。
采集层 ``_rank_metric_steps`` 曾把"哪条 phase 承载逐步权威指标"写死成**一个**字面量
（只认前者），于是 T010（9B·SFT·LoRA·native·**1 卡**，100/100 步、exit=0、
loss/grad_norm 全程 finite）的 100 条逐步指标被**整批静默丢弃**，
判成"A2 缺少 optimizer step 证据 / A6 缺少 loss 序列证据"——**采集侧伪否**。

本模块三条锁：

1. **非 pipeline 的单卡档必须被正确解析**——用的是**普通路径**那个 phase 名；
2. **未知 phase 但含 metrics ⇒ 必须显式暴露**（计数 + notes + 台账字段），不得静默丢弃；
   同时**不得**因此放松 fail-closed：未登记 phase 的 metrics **不进**证据序列；
3. **注册表在纯计算层**（``graspo.core.result_judge``）且采集侧**零 phase 字面量**——
   防止下一次"训练侧加了新路径、采集侧忘了跟"重演。

不触 GPU、不需要 torch：只跑采集器的 CLI（与 ``test_collect_results.py`` 同一手法）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_COLLECTOR = _REPO / "scripts" / "collect_results.py"
_JUDGE_SOURCE = _REPO / "src" / "graspo" / "core" / "result_judge.py"


def _load_result_judge() -> ModuleType:
    """按文件路径加载纯逻辑模块（与采集器同一原因：不能经 graspo/__init__ 拉入 torch）。"""
    spec = importlib.util.spec_from_file_location("_e7_result_judge", _JUDGE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_JUDGE = _load_result_judge()

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

_STDOUT_NO_NUMBERS = "[graspo] sft step 1 ok\n"


def _make_run(root: Path, *, rank_metric_lines: list[dict], tier_id: str = "T010") -> Path:
    """造一个 native 布局的运行目录：<run>/outputs/<tier>/logs/rank_metrics.rank_00000.jsonl。"""
    run = root / tier_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "stdout.log").write_text(_STDOUT_NO_NUMBERS, encoding="utf-8")
    out = run / "outputs" / tier_id
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text("train_method: sft\n", encoding="utf-8")
    (out / "logs" / "training.log").write_text("train ok\n", encoding="utf-8")
    (out / "logs" / "rank_metrics.rank_00000.jsonl").write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in rank_metric_lines),
        encoding="utf-8",
    )
    return run


def _step_payload(
    *,
    phase: str,
    metrics: dict | None = None,
    with_metrics: bool = True,
) -> dict:
    payload: dict = {
        "event": "rank_memory",
        "timestamp": "2026-09-20T22:49:02+08:00",
        "phase": phase,
        "kind": "diagnostic",
        "reproducible": False,
        "rank": 0,
        "tp_rank": 0,
        "tp_size": 1,
        "memory": {"allocated_mib": 1.0},
    }
    if with_metrics:
        payload["metrics"] = metrics if metrics is not None else {}
    return payload


def _single_card_metrics(step: int) -> dict:
    """**修复后**单卡 emit 的 metrics 形状（``_aggregate_rank_metrics`` 非 dist 分支）。

    单卡下"局部 == 全局"平凡成立，所以 ``global_*`` 与本地键同值——
    这正是修复 ``transformer_adapter._aggregate_rank_metrics`` 要保证的契约。
    """
    local = {
        "rank": 0,
        "tp_rank": 0,
        "optimizer_steps": 1,
        "skipped_nonfinite": 0,
        "loss_mean": 1.0 / step,
        "grad_norm_mean": 1.0 * step,
        "nonzero_grad_count": 7,
        "grad_count_metric": "nonzero_lora_grads",
        "lora_norm_delta": 1e-4 * step,
        "current_lr": 5e-05,
    }
    return {
        **local,
        "rank_metrics": [local],
        "global_optimizer_steps_sum": 1,
        "global_nonzero_grad_count_sum": 7,
        "global_loss_mean": local["loss_mean"],
        "global_grad_norm_mean": local["grad_norm_mean"],
        "global_lora_norm_delta_mean": local["lora_norm_delta"],
        "global_trainable_norm_delta_mean": None,
    }


def _run_collector(tmp_path: Path, run_root: Path) -> dict:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    out = tmp_path / "ledger"
    completed = subprocess.run(
        [
            sys.executable,
            str(_COLLECTOR),
            "--manifest",
            str(manifest_path),
            "--run-root",
            str(run_root),
            "--out",
            str(out),
            "--context-length",
            "8192",
            "--date",
            "2026-09-20",
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


# ── 1. 注册表本体：两条 SFT 路径都必须在册（§1.4 单一真相源）──────────────


def test_step_metrics_phases_covers_both_sft_paths() -> None:
    """两条 SFT 路径的 phase 名都必须承载权威指标：漏一个就重演 T010 伪否。"""
    for phase in (
        "sft_train_batch_after",  # training_sft.py:478  普通路径（含全部单卡/DP/TP 档）
        "pipeline_sft_train_batch_after",  # training_sft.py:753  PP 路径
        "train_batch_after",  # training.py:204     RL/GRASPO 普通路径
        "pipeline_train_batch_after",  # training.py:440     RL/GRASPO PP 路径
    ):
        assert phase in _JUDGE.STEP_METRICS_PHASES
    # 诊断类事件**不带** metrics，不得混进注册表（否则会给空 dict 判成证据）。
    assert not (_JUDGE.DIAGNOSTIC_PHASES & _JUDGE.STEP_METRICS_PHASES)
    for phase in ("setup_after", "checkpoint_after", "logprob_after", "pipeline_logprob_after"):
        assert phase in _JUDGE.DIAGNOSTIC_PHASES


def test_collector_has_no_hardcoded_phase_literal() -> None:
    """采集侧**零 phase 字面量**：名单只在 result_judge 里定义一次（§1.4/§2.2）。"""
    source = _COLLECTOR.read_text(encoding="utf-8")
    # 只检查可执行代码：注释与 docstring 里提到 phase 名是**允许**的（而且有用）。
    import ast

    code_tokens: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            code_tokens.append(node.value)
    # docstring 也是 Constant ⇒ 排除掉那些含空格/换行的"文档串"，只留疑似标识符。
    identifiers = [t for t in code_tokens if t and "\n" not in t and " " not in t]
    offenders = [
        t for t in identifiers if t in _JUDGE.STEP_METRICS_PHASES or t in _JUDGE.DIAGNOSTIC_PHASES
    ]
    assert offenders == [], f"采集侧不得出现 phase 字面量，改用注册表：{offenders}"


# ── 2. 非 pipeline 的单卡档必须被正确解析（T010 伪否的直接回归）───────────


def test_non_pipeline_single_card_phase_is_parsed(tmp_path: Path) -> None:
    """``sft_train_batch_after`` 的逐步指标必须被采集器读成 A2/A6 证据。

    这是 T010 的确切回归：修复前 ``series_source`` 会退回 stdout 兜底、
    ``optimizer_steps_per_step`` 为空、loss 全 MISSING。
    """
    lines = [
        _step_payload(phase="sft_train_batch_after", metrics=_single_card_metrics(i))
        for i in range(1, 8)
    ]
    run_root = tmp_path / "runs"
    _make_run(run_root, rank_metric_lines=lines)
    record = _run_collector(tmp_path, run_root)

    assert record["series_source"] == "rank_metrics", "必须走权威旁路，不得退回 stdout 兜底"
    assert len(record["losses"]) == 7
    assert record["losses"][0] == 1.0  # 1/1
    assert all(isinstance(value, float) for value in record["losses"]), "不得有 MISSING 哨兵"
    # 逐步 optimizer_steps>0 ⇒ A2 的"训练步真推进"断言有证据可判。
    assert record["optimizer_steps_per_step"] == [1] * 7
    assert record["nonfinite_skips"] == 0
    # 没有任何东西被忽略——空 dict 是**显式**结论，不是"没查"。
    assert record["ignored_phase_records"] == {}
    assert not [note for note in record["series_notes"] if "忽略了" in note]


def test_unknown_phase_with_metrics_is_explicitly_exposed(tmp_path: Path) -> None:
    """未知 phase 但 payload 含 metrics ⇒ **显式曝光**，不得静默丢弃（§2.2）。

    同时**不得**把它当证据用：未登记 phase 的 metrics 不进 loss/step 序列，
    fail-closed 语义保持不变。
    """
    lines = [
        _step_payload(phase="sft_train_batch_after", metrics=_single_card_metrics(i))
        for i in range(1, 4)
    ] + [
        _step_payload(phase="brand_new_train_batch_after", metrics=_single_card_metrics(99))
        for _ in range(5)
    ]
    run_root = tmp_path / "runs"
    _make_run(run_root, rank_metric_lines=lines)
    record = _run_collector(tmp_path, run_root)

    # ① 显式计数，名字与条数都落地在台账里。
    assert record["ignored_phase_records"] == {"brand_new_train_batch_after": 5}
    # ② notes 里有一句人能读的话（不得只存在于结构化字段里）。
    assert any(
        "忽略了 5 条含 metrics 的未登记 phase 记录" in note
        and "brand_new_train_batch_after" in note
        for note in record["series_notes"]
    ), record["series_notes"]
    # ③ 未被当成证据：只有 3 条已登记行进序列（fail-closed 不放松）。
    assert len(record["losses"]) == 3
    assert record["optimizer_steps_per_step"] == [1, 1, 1]


def test_unknown_phase_without_metrics_is_not_reported(tmp_path: Path) -> None:
    """纯诊断事件（无 metrics）不算"被忽略的指标"——不得制造噪声。"""
    lines = [
        _step_payload(phase="sft_train_batch_after", metrics=_single_card_metrics(1)),
        _step_payload(phase="setup_after", with_metrics=False),
        _step_payload(phase="checkpoint_after", with_metrics=False),
        # 连 phase 都缺、又没 metrics ⇒ 也不该冒出来
        {"event": "rank_memory", "memory": {"allocated_mib": 1.0}},
    ]
    run_root = tmp_path / "runs"
    _make_run(run_root, rank_metric_lines=lines)
    record = _run_collector(tmp_path, run_root)

    assert record["ignored_phase_records"] == {}
    assert not [note for note in record["series_notes"] if "忽略了" in note]
    assert len(record["losses"]) == 1


def test_registered_phase_without_global_keys_stays_fail_closed(tmp_path: Path) -> None:
    """已登记 phase 但**缺 global_* 键**（旧 emit / 未修聚合器）⇒ 如实记取证缺口。

    这条锁死"修 phase 名不等于无脑变绿"：没有全局读数就必须 fail-closed，
    写 MISSING 哨兵而**不是** NaN（不许把"没有读数"升格成"数值异常"）。
    """
    # 只有本地键的单卡 metrics——修复前每条 rank_metrics 行的真实形状。
    stale = {
        "rank": 0,
        "tp_rank": 0,
        "optimizer_steps": 1,
        "skipped_nonfinite": 0,
        "loss_mean": 0.5,
        "grad_norm_mean": 1.0,
        "rank_metrics": [{"rank": 0, "optimizer_steps": 1}],
    }
    lines = [_step_payload(phase="sft_train_batch_after", metrics=stale) for _ in range(3)]
    run_root = tmp_path / "runs"
    _make_run(run_root, rank_metric_lines=lines)
    record = _run_collector(tmp_path, run_root)

    assert record["series_source"] == "rank_metrics"
    assert record["losses"] == ["MISSING"] * 3
    assert record["grad_norms"] == ["MISSING"] * 3
    assert record["criteria"]["A6"] is False, "没有全局读数 ⇒ A6 必须 fail-closed，不得默认通过"
    detail = record["criteria_detail"]["A6"]
    assert "取证缺口" in detail or "不可判定" in detail
