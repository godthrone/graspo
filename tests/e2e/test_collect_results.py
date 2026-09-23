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

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_COLLECTOR = Path(__file__).resolve().parents[2] / "scripts" / "collect_results.py"
_RESULT_JUDGE = Path(__file__).resolve().parents[2] / "src" / "graspo" / "core" / "result_judge.py"
_JUDGE_MODULE_NAME = "_ae1_result_judge"


def _load_judge():
    """按文件路径加载判据模块（**不能**经 ``graspo/__init__``：那条链会拉 torch）。

    与 ``tests/core/test_result_judge.py`` 里的加载方式同源（§1.4：同一份判定器，
    测试不得各读一份影子实现）。
    """
    existing = sys.modules.get(_JUDGE_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_JUDGE_MODULE_NAME, _RESULT_JUDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_JUDGE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module

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


_FAKE_TORCH_BLOCKER = """
import sys


class _NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"tests: torch 被显式屏蔽（模拟宿主无 torch）：{fullname}")
        return None


sys.meta_path.insert(0, _NoTorch())
"""

#: 模拟"宿主**有** torch、但 checkpoint 文件坏了"：``torch.load`` 抛
#: ``pickle.UnpicklingError``（真机实测形态：``could not find MARK``）。它走的是
#: ``_require_torch`` 成功后的反序列化分支，专门覆盖"反序列化异常必须被显式捕获"。
_FAKE_TORCH_BROKEN_BLOCKER = (
    "import _pickle, sys, types\n"
    "torch = types.ModuleType('torch')\n"
    "def _load(*a, **k):\n"
    "    raise _pickle.UnpicklingError('could not find MARK')\n"
    "torch.load = _load\n"
    "sys.modules['torch'] = torch\n"
)


def _sitecustomize_env(tmp_path: Path, source: str) -> dict[str, str]:
    """构造一个显式声明 torch 可用性的子进程环境（**不靠环境碰运气**）。

    为什么必须这样：采集器是 ``subprocess`` 跑的独立进程，``monkeypatch`` 管不到它；
    同一个用例在"宿主有 torch"（开发机 ``.local/uv-venv`` 实测 torch 2.11）与"宿主无
    torch"（228 采集机）上会走到**不同的证据来源**，断言只能保住一边。这里用
    ``PYTHONPATH`` + ``sitecustomize`` 在子进程**启动时**决定 torch 能不能 import ——
    测试自己声明它要模拟的宿主环境。
    """
    blocker_dir = tmp_path / "faketorch"
    blocker_dir.mkdir(exist_ok=True)
    (blocker_dir / "sitecustomize.py").write_text(source, encoding="utf-8")
    env = dict(os.environ)
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{blocker_dir}{os.pathsep}{previous}" if previous else str(blocker_dir)
    return env


def _run_collector(
    tmp_path: Path,
    *extra: str,
    manifest: dict | None = None,
    env: dict[str, str] | None = None,
    expected_rc: int = 0,
) -> dict:
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
        env=env,
    )
    assert completed.returncode == expected_rc, completed.stderr
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
    assert record["status"] == "⚠ 口径不可测"
    assert record["failure_class"] == "取证不足（不可判定）"
    assert record["status"] != "✅ 可用"
    assert record["max_context"] is None
    assert (tmp_path / "ledger" / "ledger.md").exists()


# ── ★ A7 取证公平性（2026-09-22 指挥官裁定，C）──────────────────────────────
# ms-swift **不上报**跳过计数（swift/trainers/mixin.py 的 patch 只 `p.grad=None`，无日志无计数）
# ⇒ 旧行为会让 A7 对它**自动通过**，而 native 却严格判 ⇒ **同一判据对不同后端不等价**。
# 现在：native 取精确计数；ms-swift 从"逐步 grad_norm 是否为 NaN"**推断**；
# 推断不出 ⇒ **口径不可测（绝不放行）**。

def test_a7_native_exact_count_fails_the_tier(tmp_path):
    """① native 有精确计数：`skipped_nonfinite > 0` ⇒ A7 不过 ⇒ ❌ 不可用。"""
    _make_native_full_run(tmp_path / "runs", deltas=_FULL_DELTAS_OK, skipped_nonfinite=2)
    record = _run_collector(
        tmp_path, manifest=_full_manifest(), env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER)
    )
    assert record["criteria"]["A7"] is False, record["criteria_detail"]
    # 逐 step 计数会被**跨步求和**（fixture 6 行 × 每行 2 次 = 12）⇒ 断言 > 0 + 来源自证
    assert record["nonfinite_skips"] and record["nonfinite_skips"] > 0
    # P0-1：跳过计数**跨全部 rank 取 MAX**（不再只信 rank0）⇒ 来源标识随之改为 all_ranks_max
    assert record["nonfinite_skips_source"] == "run_metrics:all_ranks_max"
    assert record["status"] == "❌ 不可用"


def test_a7_rank1_only_skip_is_not_a_false_pass(tmp_path):
    """★P0-1 判别力实证：**rank0 跳过=0、rank1 单独跳过=1** ⇒ 不许假通过。

    修前行为（可复现的**假通过**）：采集只读 `rank_metrics.rank_00000.jsonl`
    ⇒ 读到 0 ⇒ `A7` 通过 ⇒ 一次"rank1 权重/LR 已分叉"的运行被记成 **✅ 训练可用**。
    修后：跨**全部** rank 取 MAX ⇒ 1 ⇒ `A7` 不通过。
    （矩阵命中：T029/T030/T041/T042 —— native graspo `dp>1`。）
    """
    _make_native_full_run(
        tmp_path / "runs", deltas=_FULL_DELTAS_OK,
        skipped_nonfinite=0,            # rank0：全 0（**这正是修前唯一被读到的**）
        rank1_skipped_nonfinite=1,      # rank1：单独跳过 1 次
    )
    record = _run_collector(
        tmp_path, manifest=_full_manifest(), env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER)
    )
    assert record["nonfinite_skips"] == 1, record.get("nonfinite_skips_source")
    assert record["nonfinite_skips_source"].startswith("run_metrics:all_ranks_max")
    assert record["criteria"]["A7"] is False, record["criteria_detail"]
    assert record["status"] == "❌ 不可用"


def test_a7_msswift_infers_skips_from_nan_grad_norm(tmp_path):
    """② ms-swift 推断成功：逐步 `grad_norm` 出现 1 个 NaN ⇒ 推断跳过 1 次 ⇒ A7 不过。"""
    _write_msswift_run(
        tmp_path / "runs", grad_norms=(1.0, 0.9, float("nan"), 0.7, 0.6, 0.5)
    )
    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)
    assert record["nonfinite_skips"] == 1, record.get("nonfinite_skips_source")
    assert record["nonfinite_skips_source"] == "log_inference:nan_grad_norm_count"
    assert record["criteria"]["A7"] is False
    assert record["status"] == "❌ 不可用"


def test_a7_cannot_self_prove_every_step_logged_is_indeterminate(tmp_path):
    """★C 收紧：`logging_steps > 1` 时读数条数 < 计划步数 ⇒ **无法自证每步都有读数**
    ⇒ 跳过计数按**口径不可测**处理（不推断、不放行）。

    为什么（复核指出的漏洞）：NaN 只出现在**非 logging 步**时，逐步序列看起来"全 finite"
    ⇒ 旧推断会得出"0 次跳过" ⇒ **仍是自动通过**（假阴性）。
    """
    _write_msswift_run(tmp_path / "runs", declared=20)   # 6 条读数 < 20 计划步
    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)
    assert record["nonfinite_skips"] is None, record.get("nonfinite_skips_source")
    assert record["criteria"]["A7"] is False
    assert "口径不可测" in record["criteria_detail"]["A7"]


def test_a7_unavailable_count_is_indeterminate_never_pass(tmp_path):
    """③ 取不到 ⇒ **口径不可测**（`⚠ 口径不可测`），**绝不自动通过**。

    构造：ms-swift 档**没有任何 grad_norm 读数**（逐步序列为空）⇒ 推断不出
    ⇒ `nonfinite_skips is None` ⇒ A7 以"口径不可测"判否（而不是通过）。
    """
    _write_msswift_run(tmp_path / "runs", grad_norms=())
    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)
    assert record["nonfinite_skips"] is None, record.get("nonfinite_skips_source")
    assert record["criteria"]["A7"] is False, record["criteria_detail"]
    assert "口径不可测" in record["criteria_detail"]["A7"]
    assert record["status"] == "⚠ 口径不可测", record["note"]


def test_collector_distinguishes_blocked_not_applicable_and_untested(tmp_path):
    """★六态口径（2026-09-22 指挥官裁定）：三种"没跑"**不得**塌成一个词，且出处可查。

    · `samples/configs/matrix54/<T>.blocked.md`        ⇒ `⛔ 无配方`（**将来可能做得了**）
    · `samples/configs/matrix54/<T>.not_applicable.md` ⇒ `⛔ 不适用`（**逻辑上不适用**）
    · 都没有                                           ⇒ `— 未测`（尚未纳入跑批）
    用户专门问过这几个词的区别；塌成一个词是信息量倒退。
    """
    manifest = json.loads(json.dumps(_MANIFEST))
    manifest["tiers"] = [
        dict(manifest["tiers"][0], tier_id=tid) for tid in ("T001", "T002", "T003")
    ]
    cfg = tmp_path / "samples" / "configs" / "matrix54"
    cfg.mkdir(parents=True)
    (cfg / "T001.blocked.md").write_text("# blocked 依据\n", encoding="utf-8")
    (cfg / "T002.not_applicable.md").write_text("# not_applicable 依据\n", encoding="utf-8")

    # 三档的运行目录都不存在 ⇒ 全走 not-run 分支
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "ledger"
    completed = subprocess.run(
        [sys.executable, str(_COLLECTOR), "--manifest", str(manifest_path),
         "--run-root", str(tmp_path / "runs"), "--out", str(out),
         "--repo-root", str(tmp_path), "--date", "2026-09-22"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = {
        json.loads(line)["tier_id"]: json.loads(line)
        for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert rows["T001"]["status"] == "⛔ 无配方", rows["T001"]
    assert rows["T002"]["status"] == "⛔ 不适用", rows["T002"]
    assert rows["T003"]["status"] == "— 未测", rows["T003"]
    # 出处可查（§2.2）：读者能从台账找到依据文件
    assert rows["T001"]["status_provenance"].endswith("T001.blocked.md")
    assert rows["T002"]["status_provenance"].endswith("T002.not_applicable.md")
    assert rows["T003"]["status_provenance"] is None
    # 六态白名单：产出里不得出现第 7 种词
    allowed = {"✅ 可用", "❌ 不可用", "⚠ 口径不可测", "⛔ 无配方", "⛔ 不适用", "— 未测"}
    assert {r["status"] for r in rows.values()} <= allowed


def test_collector_reports_untested_when_run_dir_missing(tmp_path):
    (tmp_path / "runs").mkdir()

    record = _run_collector(tmp_path)

    assert record["status"] == "— 未测"
    assert "运行目录不存在" in record["note"]


def test_collector_pairs_rerun_for_a4(tmp_path):
    """双跑 + **两棵不同目录树** ⇒ A4 具备可判性（即使 A2/A3 因缺权重证据未通过）。

    ★ 2026-09-22 更新（裁定 2）：本 fixture 的**两跑权重文件逐位相同**（都写同一份
    ``lora_b``）**且**末步 loss 也逐位相同。旧实现在这种形态下一律判「双通道冲突/
    不可判定」——裁定 2 指出那会**误伤真·完美可复现**。现在的判据是：
    两通道逐位相同时，**先看能不能证明这是两次独立运行**（运行根 / 产物根
    ``st_dev:st_ino`` / 首条记录时间戳，≥2 项独立证据）；能证明 ⇒ 判 **✅ 通过**
    （完美可复现），不能证明（缺元数据/指向同一实体）⇒ 继续 fail-closed。

    本 fixture 是**两个独立目录树**（``runs/`` 与 ``reruns/``，inode 不同）⇒ 独立性成立
    ⇒ 逐位相同只能解释为确定性复现 ⇒ **A4 通过**。
    """
    # 用 ms-swift 布局（权重文件是可定位的 checkpoint-6/adapter_model.safetensors），
    # 两跑写**逐位相同**的 lora_b ⇒ 独立通道有证据，且证据是"两跑权重相同"。
    _write_msswift_run(tmp_path / "runs")
    _write_msswift_run(tmp_path / "reruns")

    record = _run_collector(
        tmp_path, "--rerun-root", str(tmp_path / "reruns"), manifest=_MSSWIFT_MANIFEST
    )

    assert record["criteria"]["A4"] is True, record["criteria_detail"]["A4"]
    assert "完美可复现" in record["criteria_detail"]["A4"]
    # 独立性证据必须落台账（读者可复核"凭什么说这是两次独立运行"）。
    assert record["a4_independence_signals"] >= 2, record["a4_independence_detail"]
    assert "运行根不同" in record["a4_independence_detail"]


# ── A6 首末 loss 走向：记录项必须进台账（2026-09-20 裁定）────────────────────
#
# 裁定把「最终 loss 不高于初始」从阻断降为**记录项**。这两条端到端用例锁死
# 降级后的两个方向：① 末点高于起点不再使 A6 判否，但走向必须在台账里**可见**
# （§2.2 显式即防呆，禁止静默丢弃）；② NaN/Inf 仍然判否，不得被"降级"带走。


def _make_increased_loss_run(root: Path, tier_id: str = "T010") -> None:
    """合成一次"末点高于起点、但全程 finite"的运行（T010 的真机形状的最小化）。"""
    _make_run(root, tier_id)
    run = root / tier_id
    (run / "stdout.log").write_text(
        "{'loss': 0.0250, 'grad_norm': 4.15}\n{'loss': 0.0603, 'grad_norm': 3.02}\n",
        encoding="utf-8",
    )
    state = {
        "global_step": 6,
        "epoch": 1.0,
        # 末点 0.0603 高于起点 0.0250，但逐步波动区间 0.0042–0.4043 ⇒ 健康。
        "log_history": [
            {"loss": 0.0250, "grad_norm": 4.15},
            {"loss": 0.4043, "grad_norm": 9.0},
            {"loss": 0.0603, "grad_norm": 3.02},
        ],
    }
    (run / "outputs" / tier_id / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )


def test_collector_records_increased_a6_loss_trend_without_blocking(tmp_path):
    """★裁定正向：末点高于起点 ⇒ A6 通过，且 ``loss_trend=increased`` 进台账。"""
    _make_increased_loss_run(tmp_path / "runs")

    record = _run_collector(tmp_path)

    assert record["criteria"]["A6"] is True, record["criteria_detail"]["A6"]
    assert record["loss_trend"] == "increased", "走向必须进台账字段，不得静默丢弃"
    detail = record["criteria_detail"]["A6"]
    assert "loss_trend=increased" in detail, detail
    assert "不阻断" in detail, "台账明细必须写明'按裁定不作阻断'，供人一眼看到"
    assert record["losses"][0] == 0.0250 and record["losses"][-1] == 0.0603
    assert "数值异常" not in record["note"], record["note"]


def test_collector_still_fails_a6_on_nan_after_the_trend_demotion(tmp_path):
    """★裁定反向：降级**没有**放松数值健康 —— 真 NaN 时 A6 仍然判否、仍记数值异常。"""
    _make_run(tmp_path / "runs")
    run = tmp_path / "runs" / "T010"
    (run / "stdout.log").write_text(
        "{'loss': 0.5, 'grad_norm': 1.0}\n{'loss': nan, 'grad_norm': 1.0}\n",
        encoding="utf-8",
    )
    state = {
        "global_step": 6,
        "epoch": 1.0,
        "log_history": [{"loss": 0.5, "grad_norm": 1.0}, {"loss": float("nan"), "grad_norm": 1.0}],
    }
    (run / "outputs" / "T010" / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    record = _run_collector(tmp_path)

    assert record["criteria"]["A6"] is False, record["criteria_detail"]["A6"]
    assert "NaN/Inf" in record["criteria_detail"]["A6"]
    assert record["failure_class"] == "数值异常"
    assert record["loss_trend"] is None, "NaN 档无从比较走向 ⇒ 台账字段为 None（§2.2）"


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
    assert record["status"] == "❌ 不可用"
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

        assert record["status"] == "❌ 不可用"
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
    grad_norms: tuple[float, ...] | None = None,   # None ⇒ 用默认 1/index；可注入 NaN
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
                # `grad_norms=None` ⇒ 默认 1/index；`grad_norms=()` ⇒ **真的没有读数**（None）
                "grad_norm": (
                    1.0 / index
                    if grad_norms is None
                    else (grad_norms[index - 1] if grad_norms else None)
                ),
                "global_step/max_steps": f"{index}/{declared}",
            }
        )
    history = [
        {
            "loss": loss,
            "grad_norm": (
                1.0 / index
                if grad_norms is None
                else (grad_norms[index - 1] if grad_norms else None)
            ),
            "step": index,
        }
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
    # ★ 两跑写**不同**的 lora_b 字节 ⇒ 末步权重指纹不同。这是"真双跑"的最小真实形状：
    # 真实训练在 §6 排除的 GPU 内核残余下权重必然分叉（AD1 §6.2 实测两跑 ckpt sha 不同）。
    # 若两跑权重逐位相同（合成 fixture 的默认行为）⇒ 新判据判「双通道冲突/不可判定」。
    _write_msswift_run(tmp_path / "runs", lora_b=struct.pack("<4f", 0.1, -0.2, 0.3, 0.0))
    _write_msswift_run(tmp_path / "reruns", lora_b=struct.pack("<4f", 0.1, -0.2, 0.3, 1e-6))

    record = _run_collector(
        tmp_path,
        "--rerun-root",
        str(tmp_path / "reruns"),
        manifest=_MSSWIFT_MANIFEST,
    )

    # A4 独立通道（五要素第 5 条）：两跑指纹必须落进台账（§1.4 单一真相源，键名在判据层）。
    assert record["final_ckpt_sha256_first"] != record["final_ckpt_sha256_second"]
    assert record["final_ckpt_sha256_agree"] is False
    # 自比防呆两列：台账必须自证"两跑来自两棵不同目录树"。
    assert record["run_root"] != record["rerun_root"]

    assert record["criteria"] == {
        "A1": True,
        "A2": True,
                # ★ 2026-09-23 假绿收口：仅 `safetensors_header_stdlib`（头部级）**不得**支撑 A3
        "A3": False,
        #   并落机器可核来源（本 fixture 只有 safetensors 头部 ⇒ safetensors_header）
        #   ⇒ 该档因证据等级不足落「不可判定」（fail-closed，不是 True 也不是 False 的缩放）

        "A4": True,
        "A5": True,
        "A6": True,
        # ★ A7（训练真推进，2026-09-22 收紧）：nonfinite 跳过 = 0 ⇒ 通过。
        "A7": True,
    }, record["criteria_detail"]
    # ★ 2026-09-23 假绿收口（消费端硬判据）：本 fixture 的 A3 只有 `safetensors_header_stdlib`
    #   （**头部级**）⇒ **不得支撑 A3=True** ⇒ 该档落「**不可判定**」：既不是 ✅（假绿），
    #   也不是 ❌（那会把"测不出来"误记成"训练失败"）。方向仍是 fail-closed。
    assert record["a3_source"] == "safetensors_header", record.get("a3_source")
    assert record["criteria"]["A3"] is False, record["criteria_detail"]
    assert record["status"] == "⚠ 口径不可测", record["note"]
    assert "A3" in record["note"] or "不可判定" in record["note"]
    # A3 证据等级不足 ⇒ 落"取证缺口"分类（不是训练失败）
    assert record["failure_class"] == "取证不足（不可判定）", record["failure_class"]
    # 读数口径必须自证（§1.4）：序列来自 ms-swift logging.jsonl，而不是 stdout 兜底。
    assert record["series_source"] == "msswift_logging"
    assert record["steps_declared_total"] == 6
    assert record["optimizer_steps_per_step"] == []


def test_collector_msswift_a5_artifacts_are_recognised_in_swift_layout(tmp_path):
    """A5 四件套在 ms-swift 布局下逐个可识别（含 checkpoint-<step> 目录）。"""
    _write_msswift_run(tmp_path / "runs")

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["criteria"]["A5"] is True
    # ★ 假绿收口：A5 齐全**不能**顺带把 A3 抬起来 —— 头部级证据 ⇒ 不得为 True
    assert record["a3_source"] == "safetensors_header", record.get("a3_source")
    assert record["criteria"]["A3"] is False, record["criteria_detail"]["A3"]
    assert "checkpoint" in record["criteria_detail"]["A5"]


def test_collector_msswift_without_evidence_is_indeterminate_never_pass(tmp_path):
    """★F-D 负向：产物读不到 ⇒ 判「不可判定」，**既不是通过、也不是失败**。

    把 args.json 去掉（ms-swift 运行目录的识别判据）后，collector 定位不到任何产物根。
    要求：① 不得默认通过；② 不得记成训练失败。
    """
    _write_msswift_run(tmp_path / "runs", args_json=False)

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["status"] == "⚠ 口径不可测"
    assert record["status"] != "✅ 可用"
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
    assert record["status"] == "❌ 不可用"
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
    assert record["status"] == "❌ 不可用"
    assert "计划 6" in record["criteria_detail"]["A2"]


def test_collector_corrupt_checkpoint_is_a_substantive_failure(tmp_path):
    """★F-D 负向：checkpoint 文件存在但不可解析 ⇒ A3 实质不通过（判"无法重新加载"）。"""
    _write_msswift_run(tmp_path / "runs", corrupt_checkpoint=True)

    record = _run_collector(tmp_path, manifest=_MSSWIFT_MANIFEST)

    assert record["criteria"]["A3"] is False
    assert record["status"] == "❌ 不可用"
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


def _a4_pair(tmp_path: Path, second_losses: tuple[float, ...], *, same_ckpt: bool = False) -> dict:
    """造一对 ms-swift 双跑。

    ``same_ckpt=True`` ⇒ 两跑写**逐位相同**的权重文件（用于专门测"双通道冲突"）；
    默认两跑权重指纹不同，模拟真实双跑（GPU 内核残余下权重必然分叉，AD1 §6.2）。
    """
    _write_msswift_run(tmp_path / "runs", losses=_A4_BASE_LOSSES)
    _write_msswift_run(
        tmp_path / "reruns",
        losses=second_losses,
        lora_b=(
            struct.pack("<4f", 0.1, -0.2, 0.3, 1e-6)
            if not same_ckpt
            else None
        ),
    )
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

    # ★ 2026-09-22 用户口径变更：**A4 降级为诊断字段，不再参与 ✅/❌**。
    #   用户原话："判定标准应该是能顺利跑完一个 epoch……看的不是效果，是训练可用"。
    #   本 fixture 的两次跑都跑完了预定步数 ⇒ 按新口径就是 **✅ 可用**；
    #   但 A4 的结论**仍必须被计算并记录**（降级 ≠ 删除），且要出现在诊断备注里。
    assert record["criteria"]["A4"] is False, record["criteria_detail"]["A4"]
    assert "首步" in record["criteria_detail"]["A4"]
    # ★ 2026-09-23 假绿收口（消费端硬判据）：本 fixture 的 A3 只有 `safetensors_header_stdlib`
    #   （**头部级**）⇒ **不得支撑 A3=True** ⇒ 该档落「**不可判定**」：既不是 ✅（假绿），
    #   也不是 ❌（那会把"测不出来"误记成"训练失败"）。方向仍是 fail-closed。
    assert record["a3_source"] == "safetensors_header", record.get("a3_source")
    assert record["criteria"]["A3"] is False, record["criteria_detail"]
    assert record["status"] == "⚠ 口径不可测", record["note"]
    assert "A3" in record["note"] or "不可判定" in record["note"]
    assert "诊断" in record["note"] and "A4" in record["note"]


def test_collector_a4_accepts_measured_bf16_drift(tmp_path):
    """★A4 正向：实测 bf16 漂移量级（首步逐位相同、终态差 1.0302e-3）⇒ A4 通过。"""
    drift = 1.0302e-3
    record = _a4_pair(tmp_path, (1.3, 0.9, 0.6, 0.4, 0.3, 0.2 + drift))

    assert record["criteria"]["A4"] is True, record["criteria_detail"]["A4"]
    assert "首步 loss 逐位相同" in record["criteria_detail"]["A4"]


def test_collector_a4_still_rejects_gross_divergence(tmp_path):
    """★A4 负向：量级更大的分歧（终态 0.2 vs 0.25）仍必须被拒绝 ⇒ 容差不是空断言。"""
    record = _a4_pair(tmp_path, (1.3, 0.9, 0.6, 0.4, 0.3, 0.25))

    # ★ 同上：A4 已降级为诊断 ⇒ 跑完就是 ✅ 可用；A4 的"量级更大的分歧"仍被如实记录。
    assert record["criteria"]["A4"] is False
    # ★ 2026-09-23 假绿收口（消费端硬判据）：本 fixture 的 A3 只有 `safetensors_header_stdlib`
    #   （**头部级**）⇒ **不得支撑 A3=True** ⇒ 该档落「**不可判定**」：既不是 ✅（假绿），
    #   也不是 ❌（那会把"测不出来"误记成"训练失败"）。方向仍是 fail-closed。
    assert record["a3_source"] == "safetensors_header", record.get("a3_source")
    assert record["criteria"]["A3"] is False, record["criteria_detail"]
    assert record["status"] == "⚠ 口径不可测", record["note"]
    assert "A3" in record["note"] or "不可判定" in record["note"]


# ── H1：A2/A3 的"环境性伪否"收口 ────────────────────────────────────────────
#
# 背景（T010 真机实测）：9B·SFT·LoRA·native·1 卡 训练完全成功（100/100 步、exit=0、
# ckpt 可重载已由容器内 torch.load 独立验证），但 228 宿主**没有 torch** ⇒ 采集器
# 拿不到 `rank_*.pt` 的反序列化结果 ⇒ A2/A3 双双被判 fail-closed「缺少证据」，
# 台账上与"训练真失败"长得一模一样。要根治的是「**明明能取到却不取**」：
#   · A2-LoRA 的正式判据（§8.1）是 `lora_norm_delta ≠ 0`，它由训练侧每步写进
#     `rank_metrics.rank_*.jsonl`——**纯 JSON，不需要 torch**；
#   · A3 的真证据在**镜像内**（镜像自带 torch），可由容器探测落盘后交采集器判定。
# 下面 6 条覆盖：就地取到证据（A2）、容器证据契约（A3）、以及**不放松**的负向
# （Δ 全零 = 实质失败；归档损坏 = 实质失败；陈旧证据 = 不采信）。

#: 6 个训练步，满足 A2 的缺省门槛 5（清单未给门槛时回落到 MIN_OPTIMIZER_STEPS）。
_LORA_DELTAS_OK = [0.0, 1.62e-4, 1.7e-4, 2.0e-4, 3.0e-4, 3.71e-4]


def _valid_torch_archive() -> bytes:
    """构造一个**结构合法**的极简 torch 归档（真 zip + ``data.pkl`` + 一个 storage）。

    用途：让"宿主无 torch"的用例停在**取证缺口**（弱证据 structural_only），而不是
    停在"归档损坏"那个实质失败上——两者必须是不同的结论，这正是本包的修复点。
    """
    import io
    import zipfile as _zipfile

    buffer = io.BytesIO()
    with _zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("rank_00000_tp_00_pp_00/data.pkl", b"\x80\x04N.")
        archive.writestr("rank_00000_tp_00_pp_00/data/0", b"\x00" * 16)
    return buffer.getvalue()


def _make_native_lora_run(
    root: Path, tier_id: str = "T010", *, deltas: list[float], rank_bytes: bytes | None = None
) -> Path:
    """重建 native 布局的运行目录：config.yaml + final/ + rank_metrics 旁路逐步指标。

    ``deltas`` 每项写成一个训练步的 ``metrics.lora_norm_delta``（与训练侧
    ``progress_metrics.lora_norm_delta_event`` 的落盘形状一致）。
    ``rank_bytes`` 非 None 时用它覆盖默认的"结构合法"归档（造坏档用）。
    """
    run = root / tier_id
    run.mkdir(parents=True)
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "stdout.log").write_text(_STDOUT, encoding="utf-8")
    # 与真机 T010 证据归档**逐字同构**的 native 布局：``config.yaml`` 在 run 根
    # ⇒ collector 的 find_output_dirs 把 run 根认作产物根，find_checkpoint_dirs
    # 在其下 rglob("final")、rglob("rank_metrics.rank_*.jsonl") 都能找到。
    (run / "config.yaml").write_text("train_method: sft\n", encoding="utf-8")
    (run / "final").mkdir(parents=True)
    (run / "final" / "manifest.json").write_text('{"format": "native-lora"}\n', encoding="utf-8")
    (run / "final" / "rank_00000_tp_00_pp_00.pt").write_bytes(_valid_torch_archive())
    metrics = run / "metrics"
    metrics.mkdir()
    lines = [
        json.dumps(
            {
                "event": "rank_metrics",
                "phase": "sft_train_batch_after",
                "kind": "diagnostic",
                "metrics": {
                    "optimizer_steps": 1,
                    # A2 的逐步断言读全局口径（单卡下与局部同值），与真机落盘形状一致。
                    "global_optimizer_steps_sum": 1,
                    "skipped_nonfinite": 0,
                    "loss_mean": 1.0 - index * 0.01,
                    "grad_norm_mean": 1.0,
                    # A6 读全局口径；与真机落盘形状一致（缺这两键会记成"数值不可得"）。
                    "global_loss_mean": 1.0 - index * 0.01,
                    "global_grad_norm_mean": 1.0,
                    "tuner_type": "lora",
                    "lora_norm_delta": delta,
                },
            }
        )
        for index, delta in enumerate(deltas)
    ]
    (metrics / "rank_metrics.rank_00000.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    if rank_bytes is not None:
        # 覆盖默认的"结构合法"归档：`_make_native_lora_run` 默认造合法归档，需要坏档的
        # 用例在这里换成任意字节。放在**最后**覆盖，避免下游用例再自己覆盖时漏掉同步。
        (run / "final" / "rank_00000_tp_00_pp_00.pt").write_bytes(rank_bytes)
    return run


def test_collector_native_lora_a2_uses_run_metrics_without_torch(tmp_path):
    """★H1 正向：宿主无 torch 也能判 A2 —— 证据取自 run 自产的 ``lora_norm_delta``。

    这是本工作包的核心修复：旧实现在 `torch is None` 时就放弃翻 checkpoint，于是
    "证据本来就在运行目录里"被判成"缺少权重变化证据"。修复后来源必须**显式**落在
    台账里（§2.2），否则下游仍分不清"采集机环境"与"训练失败"。

    ``env`` 显式屏蔽 torch：这条用例的语义就是"**宿主无 torch**"，不能靠开发机碰巧
    装没装 torch 来决定走哪条证据路径（否则同一用例在 228 与开发机上结论不同）。
    """
    _make_native_lora_run(tmp_path / "runs", deltas=_LORA_DELTAS_OK)

    record = _run_collector(tmp_path, env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER))

    assert record["criteria"]["A2"] is True, record["criteria_detail"]["A2"]
    assert record["weight_evidence_source"] == "run_metrics:lora_norm_delta"
    assert "lora_norm_delta=0.000162" in record["criteria_detail"]["A2"]


def test_collector_native_lora_all_zero_delta_is_substantive_failure(tmp_path):
    """★H1 负向（不放松）：每步 Δ 都恰好为 0 ⇒ 权重确实没被更新 ⇒ **实质失败**。

    这条守住"取证缺口 ≠ 失败"的另一侧：读到的是"零变化"这个**事实**，不是"读不到"。
    不得因为它是就地证据就放宽成通过，也不得记成取证缺口（那会诱导重跑而不是修缺陷）。
    """
    _make_native_lora_run(tmp_path / "runs", deltas=[0.0] * 6)

    record = _run_collector(tmp_path, env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER))

    assert record["criteria"]["A2"] is False
    assert "权重未变化" in record["criteria_detail"]["A2"]
    assert record["status"] == "❌ 不可用"
    assert record["failure_class"] != "取证不足（不可判定）"


def test_collector_native_without_torch_reports_structural_only_not_pass(tmp_path):
    """★H1 负向（**弱证据不放行**）：宿主无 torch 且无容器探测 ⇒ A3 仍是缺口，但成因显式。

    这条是本包与"放松判据"的分界线：stdlib 归档/CRC 校验只证明"文件完整"，**不**证明
    "torch 能反序列化" ⇒ 不得据此判 A3 通过。它唯一的作用是把"文件好但验不了"与
    "文件坏了"分开，并在台账上标明证据等级（`structural_only:zip_crc`）。

    ``env`` 显式屏蔽 torch：这条用例的语义是"宿主无 torch"（228 采集机的实测环境）。
    修前它靠"开发机恰好没装 torch"才成立，而开发机 ``.local/uv-venv`` 实测有 torch
    2.11 ⇒ 证据来源变成 ``host_torch_load``，断言只能靠环境碰运气（本包要修的
    "测试不封闭"）。现在由测试自己声明宿主环境。
    """
    _make_native_lora_run(tmp_path / "runs", deltas=_LORA_DELTAS_OK)
    env = _sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER)
    # 自证：屏蔽真的生效（否则这条用例会悄悄退回 host_torch_load 路径而"看起来通过"）。
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"], capture_output=True, text=True, check=False, env=env
    )
    assert probe.returncode != 0, "假宿主仍有 torch，用例前提不成立"

    record = _run_collector(tmp_path, env=env)

    assert record["criteria"]["A3"] is False
    assert record["reload_evidence_source"] == "structural_only:zip_crc"
    assert "structural_only" in record["criteria_detail"]["A3"]
    assert record["status"] == "⚠ 口径不可测"
    assert record["failure_class"] == "取证不足（不可判定）"


def test_collector_native_broken_rank_archive_is_substantive_failure(tmp_path):
    """★H1 负向（**加强** fail-closed）：宿主无 torch 时，坏归档必须判"无法重新加载"。

    旧实现把"没 torch"与"文件坏"一起折叠成取证缺口。修复后无 torch 也能做 stdlib
    归档/CRC 校验 ⇒ 文件真坏就是**实质失败**（与 ms-swift 的 safetensors 分支同方向）。
    """
    run = _make_native_lora_run(tmp_path / "runs", deltas=_LORA_DELTAS_OK)
    (run / "final" / "rank_00000_tp_00_pp_00.pt").write_bytes(b"definitely not a zip archive")

    record = _run_collector(tmp_path, env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER))

    assert record["criteria"]["A3"] is False
    assert "无法重新加载" in record["criteria_detail"]["A3"]
    assert record["reload_evidence_source"] == "structural_only:zip_crc"
    assert record["status"] == "❌ 不可用"


def test_collector_native_rank_torch_load_crash_is_caught(tmp_path):
    """★H2 负向（**真 bug 回归锁**）：反序列化异常不得让采集器崩掉。

    这是本包唯一一条"跑旧代码必挂"的用例，**反证过了**：把三处改动 ``git stash``
    掉、只留下这条测试跑 HEAD 版 collector，得到

        _pickle.UnpicklingError: could not find MARK
        E   assert 1 == 0   (returncode != 0)

    ——异常从 ``torch.load`` 冒到顶层，采集器 rc≠0 崩掉，**这一档连同整个台账都产不出来**。
    旧实现只捕 ``RuntimeError / OSError / AttributeError``，而 ``pickle.UnpicklingError``
    直接派生自 ``Exception``，与这三者**互不派生** ⇒ 必然漏。

    为什么必须走"宿主有 torch"这条分支：``torch.load`` 的调用点在
    ``_require_torch()`` 之后。**宿主无 torch 时该分支根本不可达**（会先走 stdlib
    归档/CRC 那条弱证据路径），所以只有显式注入一个"有 torch"的宿主才能复现。
    真机触发条件（228 采集机有 torch 2.x）：ckpt 损坏 + 这一档的 ``lora_norm_delta``
    读不到（指标缺文件 / 全非有限），于是 A2 回落到 checkpoint 兜底路径。

    修复后必须：① 采集器 rc=0 且台账照常产出；② 损坏事实**不被吞掉**——A3 判
    **实质失败**（``checkpoint_reloadable=False``，来源 ``host_torch_load``），异常类型
    与消息写进明细（§2.2 显式即防呆）。**不得**降级成 ``None``（那是把"坏文件"伪装成
    "没证据"，是放行风险）。

    宿主 torch 由 ``_FAKE_TORCH_BROKEN_BLOCKER`` 显式注入：一个只让 ``torch.load``
    抛 ``UnpicklingError("could not find MARK")`` 的假 torch，与真机实测形态逐字一致。
    """
    run = _make_native_lora_run(
        tmp_path / "runs",
        # 逐位全 0 ⇒ 逐步指标这条路上 A2 **拿不到**"权重已变化"，迫使它回落到
        # checkpoint 兜底路径（也就是真正会踩到 torch.load 的那条路）。
        deltas=[0.0] * 6,
        # 宿主机认为这是个可读文件，但反序列化必炸（假 torch 的 load 恒抛）。
        rank_bytes=b"corrupted-but-readable",
    )
    # final/ 必须存在且带 manifest.json，才会进 native 反序列化分支。
    assert (run / "final" / "manifest.json").is_file()

    record = _run_collector(tmp_path, env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BROKEN_BLOCKER))

    # ① 采集器不崩：_run_collector 内部已断言 returncode == 0，且台账有且仅有一条记录。
    # ② 损坏事实不被吞掉：A3 是在反序列化那一步炸的 ⇒ 判实质失败、来源 host_torch_load。
    assert record["criteria"]["A3"] is False
    assert "无法重新加载" in record["criteria_detail"]["A3"]
    assert record["reload_evidence_source"] == "host_torch_load"
    # 异常类型与消息必须可见，而不是被吞成一句"缺证据"。
    assert "UnpicklingError" in record["criteria_detail"]["A3"], record["criteria_detail"]["A3"]
    assert "could not find MARK" in record["criteria_detail"]["A3"]
    assert record["status"] == "❌ 不可用"


def test_collector_native_container_probe_grades_a3_without_host_torch(tmp_path):
    """★H1 正向：宿主无 torch 时，**容器内** torch 探测证据足以判 A3 通过。

    契约见 ``scripts/collect_results.py`` 的 ``read_torch_probe``：探测文件必须带
    schema 标识与**当前文件内容的 sha256**。这条同时验证"证据等级可审计"。
    """
    import hashlib

    run = _make_native_lora_run(tmp_path / "runs", deltas=_LORA_DELTAS_OK)
    rank = run / "final" / "rank_00000_tp_00_pp_00.pt"
    digest = hashlib.sha256(rank.read_bytes()).hexdigest()
    (run / "torch_probe.json").write_text(
        json.dumps(
            {
                "schema": "graspo.torch_probe.v1",
                "torch": "2.11.0+cu130",
                "all_ok": True,
                "checked": [
                    {"relpath": "final/rank_00000_tp_00_pp_00.pt", "ok": True, "sha256": digest}
                ],
            }
        ),
        encoding="utf-8",
    )

    record = _run_collector(tmp_path)

    assert record["criteria"]["A3"] is True, record["criteria_detail"]["A3"]
    assert record["reload_evidence_source"] == "container_torch_probe"


def test_collector_native_stale_torch_probe_is_not_trusted(tmp_path):
    """★H1 负向（防陈旧证据）：探测文件的 sha256 与磁盘不符 ⇒ **不采信**，退回取证缺口。

    为什么必须有这条：同一档第二次跑会覆盖 ``final/`` 下的权重；若沿用上一次的探测
    结论，就是用旧 run 的证据给新 run 作证（真实风险，不是假想）。

    ``env`` 显式屏蔽 torch：探测证据被拒后要退回 ``structural_only:zip_crc``
    （"文件好但验不了"），这只有在"宿主无 torch"的前提下才成立——不能靠环境碰运气。
    """
    run = _make_native_lora_run(tmp_path / "runs", deltas=_LORA_DELTAS_OK)
    (run / "torch_probe.json").write_text(
        json.dumps(
            {
                "schema": "graspo.torch_probe.v1",
                "torch": "2.11.0+cu130",
                "all_ok": True,
                "checked": [
                    {"relpath": "final/rank_00000_tp_00_pp_00.pt", "ok": True, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )

    record = _run_collector(tmp_path, env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER))

    assert record["criteria"]["A3"] is False
    assert record["reload_evidence_source"] == "structural_only:zip_crc"
    assert record["status"] == "⚠ 口径不可测"
    assert record["failure_class"] == "取证不足（不可判定）"
    # 探测文件被拒的原因必须出现在台账明细里（不静默丢弃）。
    assert "sha256 不符" in record["criteria_detail"]["A3"], record["criteria_detail"]["A3"]
    markdown = (tmp_path / "ledger" / "ledger.md").read_text(encoding="utf-8")
    assert markdown


# ── A4 自比结构性防呆 + 分档容差端到端守卫（2026-09-21 裁定）──────────────────
#
# AD1 §2.3 登记的边界校验缺口：collector 可以把同一棵目录树同时当 --run-root 与
# --rerun-root，此时判据层没有任何断言 ⇒ 台账里会出一堆"差 0"的假 A4 ✅。
# 下面三条锁死这个缺口被补上后的行为（边界校验 = 拒绝，不是退路，§2.3/§3.4）。


def test_collector_rejects_self_comparison_same_directory_for_both_roots(tmp_path):
    """★自比断言**能真失败**：同目录同传 ⇒ 显式报错退出，**不得**静默跑出差 0 的 A4。

    三种"同一实体目录"的伪装都必须被 `resolve()` 识破：
    同字面量、相对路径（`./runs`）、软链。
    """
    _write_msswift_run(tmp_path / "runs", lora_b=struct.pack("<4f", 0.1, -0.2, 0.3, 0.0))
    same = str(tmp_path / "runs")

    for alias in (same, str(tmp_path / "runs" / "."), str(Path(same).resolve())):
        manifest_path = tmp_path / "m.json"
        manifest_path.write_text(json.dumps(_MSSWIFT_MANIFEST), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(_COLLECTOR),
                "--manifest", str(manifest_path),
                "--run-root", same,
                "--rerun-root", alias,
                "--out", str(tmp_path / "ledger_self"),
                "--context-length", "8192",
                "--date", "2026-09-21",
            ],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 2, completed.stdout
        assert "自比" in completed.stderr
        assert not (tmp_path / "ledger_self" / "ledger.jsonl").exists(), (
            "自比必须在写台账**之前**被拒绝（否则会留下差 0 的假 A4 台账）"
        )

    # 软链别名同样必须识破。
    link = tmp_path / "runs_link"
    link.symlink_to(tmp_path / "runs")
    failed = subprocess.run(
        [
            sys.executable,
            str(_COLLECTOR),
            "--manifest", str(tmp_path / "m.json"),
            "--run-root", same,
            "--rerun-root", str(link),
            "--out", str(tmp_path / "ledger_link"),
            "--context-length", "8192",
            "--date", "2026-09-21",
        ],
        capture_output=True, text=True, check=False,
    )
    assert failed.returncode == 2, failed.stdout
    assert "自比" in failed.stderr


def test_collector_ledger_records_both_roots_so_double_run_is_self_evident(tmp_path):
    """防呆两列：台账必须落 `run_root` / `rerun_root`，且两者解析后不同。"""
    _write_msswift_run(tmp_path / "runs", lora_b=struct.pack("<4f", 0.1, -0.2, 0.3, 0.0))
    _write_msswift_run(tmp_path / "reruns", lora_b=struct.pack("<4f", 0.1, -0.2, 0.3, 1e-6))

    record = _run_collector(
        tmp_path, "--rerun-root", str(tmp_path / "reruns"), manifest=_MSSWIFT_MANIFEST
    )

    assert Path(record["run_root"]).resolve() != Path(record["rerun_root"]).resolve()
    assert Path(record["run_root"]).resolve() == (tmp_path / "runs").resolve()


def test_collector_a4_rejects_the_inflated_tolerance_variant_end_to_end(tmp_path):
    """★ADR 负向对照（端到端）：一个"容差被放大到 0.34"的变体会被现有测试拦住。

    做法：直接对判据层的**同一把尺子**（`validate_tier_tolerance`）喂 0.34 ——
    它就是 AD1 锁二点名的危险值（能放过真 bug 0.0935）。端到端层面再加一条：
    真 bug 量级的分歧（T032 的 0.0935）在**任何**档族下都必须判否。
    """
    judge = _load_judge()

    with pytest.raises(ValueError):
        judge.validate_tier_tolerance(0.34, min_true_bug_signature=judge.A4_MIN_TRUE_BUG_SIGNATURE)

    # T032 实测对（−0.0217 vs +0.0718，差 0.0935）走完整 collector 链路必须判否。
    _write_msswift_run(tmp_path / "runs", losses=(1.0, 0.8, -0.021734893321990967))
    _write_msswift_run(
        tmp_path / "reruns",
        losses=(1.0, 0.8, 0.07180161476135254),
        lora_b=struct.pack("<4f", 0.1, -0.2, 0.3, 1e-6),
    )
    record = _run_collector(
        tmp_path, "--rerun-root", str(tmp_path / "reruns"), manifest=_MSSWIFT_MANIFEST
    )
    assert record["criteria"]["A4"] is False, record["criteria_detail"]["A4"]


def test_collector_requires_dual_channel_for_identical_losses_without_fingerprints(tmp_path):
    """★矩阵采集口径（锁三，子检查④形态②）：两跑 loss 逐位相同 + 独立通道**无证据**
    ⇒ 判「不可判定」，**不得**静默退化成单通道 ✅（AD1 §④ 的 7 条退化通过就靠这条拦住）。

    造一个 native 布局（`_make_run` **不产**可定位的 checkpoint ⇒ 指纹取不到），
    双跑 loss 序列逐位相同。
    """
    _make_run(tmp_path / "runs")
    _make_run(tmp_path / "reruns")

    record = _run_collector(tmp_path, "--rerun-root", str(tmp_path / "reruns"))

    assert record["final_ckpt_sha256_first"] is None
    assert record["final_ckpt_sha256_second"] is None
    assert record["final_ckpt_sha256_agree"] is None, "取不到 ≠ 不一致（三态）"
    assert record["a4_require_dual_channel"] is True
    assert record["criteria"]["A4"] is False
    assert record["failure_class"] == "取证不足（不可判定）"
    detail = record["criteria_detail"]["A4"]
    assert "双通道未一致" in detail
    assert "不可判定" in detail

# ── ② 取证路径：native **全参 PP**（T017 真机实测形状）──────────────────────────
#
# 背景（T017 真机实测，2026-09-22）：9B·SFT·**全参**·native·pp=2 两跑各 100 步、
# ``exit_code=0``、权重每步都在变，但 A2/A4 双双被判「取证缺口」：
#   · A2：全参分支只认「checkpoint vs 基座」的 safetensors 比对，而 native 全参落的是
#     ``rank_*_tp_*_pp_*.pt``；宿主又无 torch ⇒ 读不了。
#     **但证据其实在 run 逐步指标里**（``trainable_norm_delta``，纯 JSON，
#     与 LoRA 的 ``lora_norm_delta`` 同源同强度）。
#   · A4：独立通道只找 ``adapter_model.safetensors`` 等**单文件**名，认不出 PP 分片集合。
# 下面 4 条锁死：能取到（正向）＋ 取不到仍 fail-closed（负向）＋ 指纹必须**真读权重字节**。

#: 合法的全参逐步 Δ（首步 0 是构造使然；后续非零 ⇒ 权重被更新过）
_FULL_DELTAS_OK = [0.0, 3.34e-5, 5.0e-4, 1.2e-3, 2.0e-3, 3.82e-3]


def _full_manifest(tier_id: str = "T017") -> dict:
    """把默认 T010 清单改成 **mode=全量**（``tuner_type`` 由 ``mode`` 决定，见 collector）。"""
    manifest = json.loads(json.dumps(_MANIFEST))
    manifest["tiers"][0].update({"tier_id": tier_id, "mode": "全量", "cards": 2, "gpus": [4, 5]})
    return manifest


def _make_native_full_run(
    root: Path,
    tier_id: str = "T017",
    *,
    deltas: list[float],
    skipped_nonfinite: int = 0,
    rank1_skipped_nonfinite: int | None = None,   # 只给 rank1 写（P0-1 判别力用）
    first_shard_bytes: bytes | None = None,
    drop_shards: bool = False,
) -> Path:
    """重建 **native 全参 PP** 的运行目录（与真机 T017 证据归档同构）。

    - ``final/rank_00000_tp_00_pp_00.pt`` + ``rank_00001_tp_00_pp_01.pt`` + ``manifest.json``；
      ``manifest.json`` **只写元数据**（真机实测：它不含任何内容哈希，且三处/两跑逐字节相同
      —— 所以"拿 manifest 当指纹"是静默降级，本 fixture 刻意保留这一点）。
    - ``rank_metrics.rank_00000.jsonl`` 每步带 ``trainable_norm_delta`` 与全局口径。
    """
    run = root / tier_id
    run.mkdir(parents=True)
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "stdout.log").write_text(_STDOUT, encoding="utf-8")
    (run / "config.yaml").write_text("train_method: sft\n", encoding="utf-8")
    final = run / "final"
    final.mkdir(parents=True)
    (final / "manifest.json").write_text(
        '{"format": "native-full-param", "pp_size": 2, "tuner_type": "full"}\n',
        encoding="utf-8",
    )
    if not drop_shards:
        payload = first_shard_bytes if first_shard_bytes is not None else b"PP-SHARD-A" * 8
        (final / "rank_00000_tp_00_pp_00.pt").write_bytes(payload)
        (final / "rank_00001_tp_00_pp_01.pt").write_bytes(b"PP-SHARD-B" * 8)
    metrics = run / "metrics"
    metrics.mkdir()
    lines = []
    for index, delta in enumerate(deltas):
        lines.append(
            json.dumps(
                {
                    "event": "rank_metrics",
                    "phase": "sft_train_batch_after",
                    "kind": "diagnostic",
                    "metrics": {
                        "optimizer_steps": 1,
                        "global_optimizer_steps_sum": 2,
                        "skipped_nonfinite": skipped_nonfinite,
                        "loss_mean": 0.0,          # 非末段 rank 的局部 loss 恒 0（真机形状）
                        "grad_norm_mean": 1.0,
                        "global_loss_mean": 1.0 - index * 0.01,
                        "global_grad_norm_mean": 1.0,
                        "tuner_type": "full",
                        "norm_metric": "trainable_parameter_l2_norm",
                        "trainable_norm_delta": delta,
                        "global_trainable_norm_delta_mean": delta,
                    },
                }
            )
        )
    (metrics / "rank_metrics.rank_00000.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if rank1_skipped_nonfinite is not None:
        # rank1 的旁路文件：**它才是"单独跳过"的载体**（rank0 全 0）。
        rank1_lines = []
        for index, delta in enumerate(deltas):
            rank1_lines.append(
                json.dumps(
                    {
                        "event": "rank_metrics",
                        "phase": "sft_train_batch_after",
                        "kind": "diagnostic",
                        "metrics": {
                            "optimizer_steps": 1,
                            "global_optimizer_steps_sum": 1,
                            "skipped_nonfinite": rank1_skipped_nonfinite,
                            "loss_mean": 1.0 - index * 0.01,
                            "grad_norm_mean": 1.0,
                            "tuner_type": "full",
                        },
                    }
                )
            )
        (metrics / "rank_metrics.rank_00001.jsonl").write_text(
            "\n".join(rank1_lines) + "\n", encoding="utf-8"
        )
    return run


def test_collector_native_full_a2_uses_trainable_norm_delta_without_torch(tmp_path):
    """★正向：宿主无 torch 也能判全参 A2 —— 证据取自 run 自产的 ``trainable_norm_delta``。

    它与 LoRA 的 ``lora_norm_delta`` 出自**同一个** ``training_norm_event()``，
    只是按 tuner_type 取模式感知的键名 ⇒ 证据**同类同强度**，不是代用指标。
    来源必须显式落在台账里（§2.2），否则读者分不清"环境缺 torch"与"训练没生效"。
    """
    _make_native_full_run(tmp_path / "runs", deltas=_FULL_DELTAS_OK)

    record = _run_collector(
        tmp_path, manifest=_full_manifest(), env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER)
    )

    assert record["criteria"]["A2"] is True, record["criteria_detail"]["A2"]
    assert record["weight_evidence_source"] == "run_metrics:trainable_norm_delta"
    assert "trainable_norm_delta=" in record["criteria_detail"]["A2"]


def test_collector_native_full_all_zero_trainable_delta_is_substantive_failure(tmp_path):
    """★负向（不放松）：每步 Δ 都恰好 0 ⇒ 权重确实没被更新 ⇒ **实质失败**，不是缺口。"""
    _make_native_full_run(tmp_path / "runs", deltas=[0.0] * 6)

    record = _run_collector(
        tmp_path, manifest=_full_manifest(), env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER)
    )

    assert record["criteria"]["A2"] is False
    # 文案必须点明**实际来源**（就地指标），不得写成"与基座一致"——那条路径没比过基座。
    assert record["criteria_detail"]["A2"] == (
        "权重未变化：全参（run 自产的可训练参数 L2 变化指标全零）"
        "（来源：run_metrics:trainable_norm_delta）"
    )
    assert record["status"] == "❌ 不可用"


def test_collector_native_full_pp_shard_set_is_a_content_fingerprint(tmp_path):
    """★正向（A4 独立通道）：PP 分片集合必须被认成**内容指纹**，且覆盖到分片字节。

    两跑只差``第一个分片的一个字节``⇒ 两个指纹必须不同。若实现退化成
    "用 manifest / 文件名 / 大小当指纹"，这条会失败（真机实测 manifest 两跑逐字节相同）。
    """
    _make_native_full_run(tmp_path / "runs", deltas=_FULL_DELTAS_OK)
    _make_native_full_run(
        tmp_path / "reruns", deltas=_FULL_DELTAS_OK, first_shard_bytes=b"PP-SHARD-X" * 8
    )

    record = _run_collector(
        tmp_path,
        "--rerun-root",
        str(tmp_path / "reruns"),
        manifest=_full_manifest(),
        env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER),
    )

    first = record["final_ckpt_sha256_first"]
    second = record["final_ckpt_sha256_second"]
    assert first and second, record["final_ckpt_sha256_detail"]
    assert record["final_ckpt_sha256_source"].startswith("pp_shard_set_sha256:")
    assert first != second, "分片字节不同却得到同一指纹 ⇒ 指纹没覆盖权重内容（静默降级）"
    assert "PP 分片集合" in record["final_ckpt_sha256_detail"]
    assert "2 片" in record["final_ckpt_sha256_detail"]


def test_collector_native_full_pp_shard_missing_is_fail_closed(tmp_path):
    """★负向（fail-closed 不变）：只有 ``manifest.json``、没有分片 ⇒ **不得**有指纹。

    这条守住"不许悄悄降级"：``manifest.json`` 是元数据，拿它当指纹会让两跑永远"一致"。
    读不到就继续按取证缺口处理（A4 不通过且标 evidence_missing）。
    """
    _make_native_full_run(tmp_path / "runs", deltas=_FULL_DELTAS_OK, drop_shards=True)
    _make_native_full_run(tmp_path / "reruns", deltas=_FULL_DELTAS_OK, drop_shards=True)

    record = _run_collector(
        tmp_path,
        "--rerun-root",
        str(tmp_path / "reruns"),
        manifest=_full_manifest(),
        env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER),
    )

    assert record["final_ckpt_sha256_first"] is None
    assert record["final_ckpt_sha256_source"] is None
    assert "未匹配 native 全参 PP 分片" in record["final_ckpt_sha256_detail"]
    assert record["criteria"]["A4"] is False

# ── ② b：ms-swift 全参的 **HF 分片 safetensors**（T005/T020 真机形状）───────────
# 真机实测（2026-09-22，`graspo:v0.28.11-cu130fix` 跑通 T005/T020 后）：
#   `<ckpt>/model-0000{1..4}-of-00004.safetensors`（合计 ~17.8 GiB，= 模型权重）
#   `<ckpt>/model.safetensors.index.json`（权威分片清单）
#   `<ckpt>/global_step50/`（**123 GiB** 的 DeepSpeed ZeRO 优化器状态）
# 旧实现只认单文件名 ⇒ A4 指纹恒为 None（补第二跑也判不了）。下面锁死三件事：
#   ① 分片集合被认成指纹；② 指纹**只看权重**（⇒ 回收优化器状态不影响 A4 可复算性）；
#   ③ 改动任一分片字节 ⇒ 指纹必须变。

def _write_hf_sharded_checkpoint(
    run: Path, tier_id: str, *, shards: dict[str, bytes], optimizer_bytes: bytes = b""
) -> None:
    ckpt = run / "final"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name, payload in shards.items():
        (ckpt / name).write_bytes(payload)
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {
            f"layer.{i}.weight": name for i, name in enumerate(sorted(shards))
        }}),
        encoding="utf-8",
    )
    if optimizer_bytes:
        opt = ckpt / "global_step50"
        opt.mkdir(parents=True, exist_ok=True)
        (opt / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(optimizer_bytes)


def test_collector_hf_sharded_weights_are_fingerprinted_and_ignore_optimizer_state(tmp_path):
    """★正向：HF 分片 safetensors 被认成**内容指纹**；且指纹**不随优化器状态变化**。"""
    _make_native_full_run(tmp_path / "runs", deltas=_FULL_DELTAS_OK)
    # 把 fixture 的 checkpoint 换成 HF 分片形态
    ckpt = tmp_path / "runs" / "T017" / "final"
    for stale in ckpt.glob("rank_*.pt"):
        stale.unlink()
    shards = {
        "model-00001-of-00002.safetensors": b"WEIGHTS-A" * 16,
        "model-00002-of-00002.safetensors": b"WEIGHTS-B" * 16,
    }
    _write_hf_sharded_checkpoint(
        tmp_path / "runs" / "T017", "T017", shards=shards, optimizer_bytes=b"OPT-A" * 32
    )
    _make_native_full_run(tmp_path / "reruns", deltas=_FULL_DELTAS_OK)
    rerun_ckpt = tmp_path / "reruns" / "T017" / "final"
    for stale in rerun_ckpt.glob("rank_*.pt"):
        stale.unlink()
    _write_hf_sharded_checkpoint(
        tmp_path / "reruns" / "T017", "T017", shards=shards,
        optimizer_bytes=b"OPT-DIFFERENT" * 32,      # ← 优化器状态**不同**
    )

    record = _run_collector(
        tmp_path, "--rerun-root", str(tmp_path / "reruns"), manifest=_full_manifest(),
        env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER),
    )

    first, second = record["final_ckpt_sha256_first"], record["final_ckpt_sha256_second"]
    assert first and second, record["final_ckpt_sha256_detail"]
    assert record["final_ckpt_sha256_source"].startswith("hf_shard_set_sha256:")
    assert "2 片" in record["final_ckpt_sha256_detail"]
    # ② 权重相同、优化器状态不同 ⇒ 指纹**必须相同**（⇒ 回收优化器状态不破坏 A4 可复算性）
    assert first == second, "指纹被优化器状态污染 ⇒ 回收优化器状态会让 A4 不可复算"


def test_collector_hf_sharded_fingerprint_tracks_weight_bytes(tmp_path):
    """★负向守卫：任一**权**分片改一个字节 ⇒ 指纹必须变（守住"覆盖权重字节"）。"""
    _make_native_full_run(tmp_path / "runs", deltas=_FULL_DELTAS_OK)
    base_ckpt = tmp_path / "runs" / "T017" / "final"
    for stale in base_ckpt.glob("rank_*.pt"):
        stale.unlink()
    _write_hf_sharded_checkpoint(
        tmp_path / "runs" / "T017", "T017",
        shards={"model-00001-of-00002.safetensors": b"WEIGHTS-A" * 16,
                "model-00002-of-00002.safetensors": b"WEIGHTS-B" * 16},
    )
    first = _run_collector(
        tmp_path, manifest=_full_manifest(),
        env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER),
    )["final_ckpt_sha256_first"]

    (base_ckpt / "model-00002-of-00002.safetensors").write_bytes(b"WEIGHTS-X" * 16)
    second = _run_collector(
        tmp_path, manifest=_full_manifest(),
        env=_sitecustomize_env(tmp_path, _FAKE_TORCH_BLOCKER),
    )["final_ckpt_sha256_first"]

    assert first and second and first != second, "改了权分片字节却指纹不变 ⇒ 指纹没覆盖权重"
