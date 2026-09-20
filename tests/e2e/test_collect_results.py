"""结果收集器 CLI 的冒烟测试（合成运行目录，不触 GPU、不需要 torch/safetensors）。

验证：抽取退出码 / 步数 / loss 序列 / 四件套产物，产出 ledger.jsonl 与 ledger.md；
证据不足时判不通过而不是崩溃或默认通过。

**F-D 追加（本模块的两条主线）**：

1. **ms-swift 产物布局可达**：ms-swift 4.5.3 实测把产物落在
   ``<output_dir>/<run_name>/v<N>-<时间戳>/``（``args.json`` + ``logging.jsonl`` +
   ``checkpoint-<step>/``），**不产** ``rank_metrics`` 旁路。旧 collector 一处都读不到
   ⇒ A2/A3/A5/A6 全部 fail-closed，把一次 exit=0、100 步、loss 正常下降的训练记成失败。
2. **取证缺口 ≠ 失败**：判定器读不到证据时台账必须写「⚠ 不可判定（取证缺口）」，
   而不是「❌ 失败」——后者会把 collector 的缺陷记成训练失败。两条方向都锁死：
   读不到**绝不**记通过，也**绝不**记失败。
"""

import json
import struct
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

_MSSWIFT_MANIFEST = {
    "schema_version": 1,
    "tiers": [
        {
            "tier_id": "T010",
            "model": "9B",
            "algorithm": "SFT",
            "mode": "LoRA",
            "backend": "ms-swift",
            "cards": 1,
            "gpus": [0],
            "status": "ready",
            "model_path": "/models/Qwen3.5-9B",
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


def _run_collector(tmp_path: Path, *extra: str, manifest: dict | None = None) -> dict:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest or _MANIFEST), encoding="utf-8")
    out = tmp_path / "ledger"
    completed = subprocess.run(
        [
            sys.executable,
            str(_COLLECTOR),
            "--manifest",
            str(manifest_path),
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
    # ★ 取证缺口 ≠ 失败：没有任何判据被证据否定 ⇒ 判"不可判定"，不得记训练失败。
    assert record["status"] == "⚠ 不可判定（取证缺口）"
    assert record["failure_class"] == "取证不足（不可判定）"
    assert record["status"] != "✅ 通过"
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


# ── 🔴-1：真 OOM 的最大可行上下文必须真的能落进台账（负向测试）──────────────
#
# 旧实现 `max_context if judgement.passed else None`（`result_judge.ledger_row`）
# 在**结构上**不可能满足硬要求"只有真 OOM 才能写入最大可行上下文"：真 OOM 的
# run 必然 exit≠0 ⇒ A1 不过 ⇒ passed=False ⇒ 该列恒为 None。下面两条用例必须
# **能真失败**（旧代码下第一条会断言失败），且非真 OOM 的失败一律不得出现数字。


def test_collector_real_oom_writes_oom_boundary_context(tmp_path):
    """真 OOM 夹具（exit=1 + OOM 报文 + 数值健康）⇒ 台账里必须出现该档上下文。"""
    _make_run(tmp_path / "runs")
    run = tmp_path / "runs" / "T010"
    (run / "exit_code").write_text("1\n", encoding="utf-8")
    (run / "stdout.log").write_text(_OOM_STDOUT, encoding="utf-8")

    record = _run_collector(tmp_path)

    assert record["criteria"]["A1"] is False  # 真 OOM 必然 exit≠0
    assert record["status"] == "❌ 失败"
    assert record["failure_class"] == "真 OOM"
    # 关键（阻断点）：该档实测上下文必须落盘，不得再是 null。
    assert record["max_context"] == 8192
    # 口径必须随数字一起落盘，否则下游会把"OOM 边界候选"读成"这个长度跑得通"。
    assert record["max_context_kind"] == "真 OOM 边界候选"
    assert "（真 OOM 边界候选）" in (tmp_path / "ledger" / "ledger.md").read_text(encoding="utf-8")


def test_collector_non_oom_failures_never_write_max_context(tmp_path):
    """非真 OOM 的失败（数据问题 / 框架未实现）⇒ 台账里**不得**出现该档上下文。"""
    for marker in ("FileNotFoundError: dataset.jsonl", "NotImplementedError: l2k"):
        case = tmp_path / marker.split(":")[0].strip().replace(" ", "_")
        _make_run(case / "runs")
        run = case / "runs" / "T010"
        (run / "exit_code").write_text("1\n", encoding="utf-8")
        (run / "stdout.log").write_text(f"RuntimeError: {marker}\n", encoding="utf-8")

        record = _run_collector(case)

        assert record["status"] == "❌ 失败"
        assert record["failure_class"] != "真 OOM"
        assert record["max_context"] is None
        assert record["max_context_kind"] is None


# ── F-D：ms-swift 产物布局必须可判（正向）+ 判据必须与 native 同语义 ─────────
#
# 夹具逐字复刻 228 上实测的 ms-swift 4.5.3 布局（T013 100 步 run 的保留产物）：
#   <run>/exit_code, stdout.log
#   <run>/T013/v1-<ts>/{args.json, logging.jsonl, checkpoint-<step>/adapter_model.safetensors}
#   <run>/T013/msswift/ms_swift_sft.jsonl
# 关键差异（旧 collector 读不到的地方）：产物根**不含 config.yaml**，
# 序列不来自 rank_metrics，checkpoint 目录名是 checkpoint-<step> 而非 final。


def _write_safetensors(path: Path, tensors: dict[str, bytes], dtype: str = "F32") -> None:
    """写一个最小但**格式合法**的 safetensors 文件（头部 + 原始数据区）。

    这是防呆测试夹具：collector 的 A2/A3 用纯 stdlib 解析头部，这里按同一份公开
    格式构造，用来验证"有 lora_B 且非零 ⇒ 权重已更新"这条判据真的能通过/能失败。
    """
    header: dict[str, object] = {}
    offset = 0
    for name, raw in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [len(raw) // 4, 1],
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    body = b"".join(tensors.values())
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + body)


def _write_msswift_run(
    run_root: Path,
    tier_id: str = "T010",
    *,
    exit_code: int = 0,
    steps: int = 6,
    declared: int = 6,
    losses: tuple[float, ...] = (1.3, 0.9, 0.6, 0.4, 0.3, 0.2),
    lora_b: bytes | None = None,
    checkpoint: bool = True,
    args_json: bool = True,
    corrupt_checkpoint: bool = False,
) -> Path:
    """造一个 ms-swift 布局的运行目录（不触 GPU、不需要 torch/safetensors）。"""
    run = run_root / tier_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")
    (run / "stdout.log").write_text("[graspo] msswift sft ok\n", encoding="utf-8")
    swift_dir = run / tier_id / "v1-20260919-003821"
    swift_dir.mkdir(parents=True, exist_ok=True)
    if args_json:
        (swift_dir / "args.json").write_text(
            json.dumps({"output_dir": str(swift_dir), "lora_rank": 8}), encoding="utf-8"
        )
    entries: list[dict] = []
    for index, loss in enumerate(losses, start=1):
        entries.append(
            {
                "loss": loss,
                "grad_norm": 1.0 / index,
                "global_step/max_steps": f"{index}/{declared}",
            }
        )
    history = [
        {"loss": loss, "grad_norm": 1.0 / index, "step": index}
        for index, loss in enumerate(losses, start=1)
    ]
    entries.append({"train_runtime": 10.0, "train_loss": losses[-1]})
    entries.append(
        {
            "model_parameter_info": "PeftModelForCausalLM: 9431M Params (21.6M Trainable)",
            "last_model_checkpoint": str(swift_dir / f"checkpoint-{steps}"),
            "global_step": steps,
            "log_history": history,
        }
    )
    (swift_dir / "logging.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    if checkpoint:
        ckpt = swift_dir / f"checkpoint-{steps}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        weights = ckpt / "adapter_model.safetensors"
        if corrupt_checkpoint:
            weights.write_bytes(b"this is not a safetensors file at all")
        else:
            payload = lora_b if lora_b is not None else struct.pack("<4f", 0.1, -0.2, 0.3, 0.0)
            _write_safetensors(
                weights,
                {
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": struct.pack(
                        "<4f", 0.0, 0.0, 0.0, 0.0
                    ),
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": payload,
                },
            )
    return run


def test_collector_msswift_layout_makes_a1_to_a6_decidable(tmp_path):
    """★F-D 正向：ms-swift 布局 + 双跑 ⇒ A1–A6 **全部可判且全部通过**。

    这条用例在旧 collector 下必须失败（旧实现读不到产物根：无 config.yaml、
    无 rank_metrics、checkpoint 叫 checkpoint-6 不叫 final ⇒ A2/A3/A5/A6 fail-closed）。
    """
    _write_msswift_run(tmp_path / "runs")
    _write_msswift_run(tmp_path / "reruns")

    record = _run_collector(
        tmp_path,
        "--rerun-root",
        str(tmp_path / "reruns"),
        manifest=_MSSWIFT_MANIFEST,
    )

    assert record["criteria"] == {
        "A1": True,
        "A2": True,
        "A3": True,
        "A4": True,
        "A5": True,
        "A6": True,
    }, record["criteria_detail"]
    assert record["status"] == "✅ 通过"
    assert record["failure_class"] is None
    # 读数口径必须自证（§1.4）：序列来自 ms-swift logging.jsonl，而不是 stdout 兜底。
    assert record["series_source"] == "msswift_logging"
    assert record["steps_declared_total"] == 6
    assert record["optimizer_steps_per_step"] == []


def test_collector_msswift_a5_artifacts_are_recognised_in_swift_layout(tmp_path):
    """A5 四件套在 ms-swift 布局下逐个可识别（含 checkpoint-<step> 目录）。"""
    _write_msswift_run(tmp_path / "runs")

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["criteria"]["A5"] is True
    assert record["criteria"]["A3"] is True
    assert "checkpoint" in record["criteria_detail"]["A5"]


def test_collector_msswift_without_evidence_is_indeterminate_never_pass(tmp_path):
    """★F-D 负向：产物读不到 ⇒ 判「不可判定」，**既不是通过、也不是失败**。

    把 args.json 去掉（ms-swift 运行目录的识别判据）后，collector 定位不到任何产物根。
    要求：① 不得默认通过；② 不得记成训练失败。
    """
    _write_msswift_run(tmp_path / "runs", args_json=False)

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["status"] == "⚠ 不可判定（取证缺口）"
    assert record["status"] != "✅ 通过"
    assert record["failure_class"] == "取证不足（不可判定）"
    assert record["max_context"] is None


def test_collector_msswift_zero_lora_weights_is_a_substantive_failure(tmp_path):
    """★F-D 负向（不得放松）：lora_B 全零 ⇒ A2 **实质**不通过（不是"不可判定"）。

    LoRA B 初值为 0；全零即"权重根本没被更新"。这条把"读到了证据并否定它"与
    "读不到证据"锁死为两种不同结论。
    """
    _write_msswift_run(tmp_path / "runs", lora_b=struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["criteria"]["A2"] is False
    assert record["status"] == "❌ 失败"
    assert record["failure_class"] != "取证不足（不可判定）"
    assert "权重未变化" in record["criteria_detail"]["A2"]


def test_collector_msswift_unfinished_steps_is_a_substantive_failure(tmp_path):
    """★F-D 负向（不得放松）：实际步数 < 计划步数 ⇒ A2 实质不通过。

    ms-swift 的 ``global_step/max_steps`` 分母就是"训练器自报的计划步数"；
    实际只推进到 4 而计划 6 ⇒ 有步没推进。这条是 native 逐步
    ``optimizer_steps>0`` 断言在 ms-swift 侧的**等价证据**（判据语义一致）。
    """
    _write_msswift_run(tmp_path / "runs", steps=4, declared=6, losses=(1.3, 1.1, 1.0, 0.9))

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["criteria"]["A2"] is False
    assert record["status"] == "❌ 失败"
    assert "计划 6" in record["criteria_detail"]["A2"]


def test_collector_corrupt_checkpoint_is_a_substantive_failure(tmp_path):
    """★F-D 负向：checkpoint 文件存在但不可解析 ⇒ A3 实质不通过（判"无法重新加载"）。"""
    _write_msswift_run(tmp_path / "runs", corrupt_checkpoint=True)

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["criteria"]["A3"] is False
    assert record["status"] == "❌ 失败"
    assert record["failure_class"] != "取证不足（不可判定）"


def test_collector_native_layout_still_works_after_msswift_support(tmp_path):
    """对照：native 布局的 A5/A6 与取证缺口语义没有被 ms-swift 支持改坏。"""
    _make_run(tmp_path / "runs")

    record = _run_collector(tmp_path)

    assert record["criteria"]["A5"] is True
    assert record["criteria"]["A6"] is True
    assert record["series_source"] == "trainer_state"


# ── A4 守卫（2026-09-19 裁定：终态容差按实测推导，但零容差子检查必须在岗）──────
#
# 这组用例走的是**完整 collector CLI**（合成 ms-swift 运行目录 → ledger.jsonl），
# 因此锁住的是端到端链路，而不只是纯逻辑函数。

_A4_BASE_LOSSES = (1.3, 0.9, 0.6, 0.4, 0.3, 0.2)


def _a4_pair(tmp_path: Path, second_losses: tuple[float, ...]) -> dict:
    _write_msswift_run(tmp_path / "runs", losses=_A4_BASE_LOSSES)
    _write_msswift_run(tmp_path / "reruns", losses=second_losses)
    return _run_collector(
        tmp_path, "--rerun-root", str(tmp_path / "reruns"), manifest=_MSSWIFT_MANIFEST
    )


def test_collector_a4_rejects_controllable_nondeterminism_at_the_first_step(tmp_path):
    """★A4 负向守卫：**可控**的非确定性（种子/数据顺序变了）⇒ A4 必须失败。

    构造：第二跑只有**首步** loss 不同（1.3 vs 1.300001），终态完全相同（0.2 == 0.2）。
    这正是"去掉种子"或"打乱数据顺序"的指纹 —— 首步只取决于种子/初始化/数据顺序/首批样本。
    如果没有零容差子检查，这条会（错误地）判通过。
    """
    record = _a4_pair(tmp_path, (1.300001, 0.9, 0.6, 0.4, 0.3, 0.2))

    assert record["criteria"]["A4"] is False, record["criteria_detail"]["A4"]
    assert "首步" in record["criteria_detail"]["A4"]
    assert record["status"] == "❌ 失败"


def test_collector_a4_accepts_measured_bf16_drift(tmp_path):
    """★A4 正向：实测 bf16 漂移量级（首步逐位相同、终态差 1.0302e-3）⇒ A4 通过。"""
    drift = 1.0302e-3
    record = _a4_pair(tmp_path, (1.3, 0.9, 0.6, 0.4, 0.3, 0.2 + drift))

    assert record["criteria"]["A4"] is True, record["criteria_detail"]["A4"]
    assert "首步 loss 逐位相同" in record["criteria_detail"]["A4"]


def test_collector_a4_still_rejects_gross_divergence(tmp_path):
    """★A4 负向：量级更大的分歧（终态 0.2 vs 0.25）仍必须被拒绝 ⇒ 容差不是空断言。"""
    record = _a4_pair(tmp_path, (1.3, 0.9, 0.6, 0.4, 0.3, 0.25))

    assert record["criteria"]["A4"] is False
    assert record["status"] == "❌ 失败"
