"""§7「实测每卡峰值显存(GiB)」列的**取数口径**回归测试（本模块锁死混口径不再复发）。

背景（本包实测确证的口径矛盾）：
- 权威口径（`.local/本期工程跟踪.md` §9.1「★ §7 各列的读取口径」表）：
  **单一口径 = 容器内 PyTorch allocator 的 rank0 `max_allocated`**；
  其它口径（OOM 报文的进程占用、宿主 `nvidia-smi` 采样）**不得混入同一列**。
- 旧实现 `collect_results._read_peak_memory` 读的却是
  `gpu/gpu_memory_summary.json` 的 `per_gpu[*].memory_used_mib_peak` —— 那是**宿主
  `nvidia-smi` 采样**，与声明不符。修好采样链路后，新跑的档会把宿主口径写进同一列
  ⇒ **同列混口径**，正是 §9.1 明文禁止的。
- 宿主口径在 228（**共享机**）上还会被他人作业污染：`T033/run2` 实测 GPU2 =
  27451 MiB 而 `max_peak_memory_gap_mib = 2388`（真值应 ≈ 25063 MiB）。
  ⇒ 现有取数路径不只是"口径不同"，而是**会给出错误答案**。

本模块四条锁：

1. `_read_peak_memory` 只认 **rank_metrics 旁路的 allocator 值**，且取全部 `rank_memory`
   行 `max_allocated_mib` 的最大值；
2. **allocator 不可得 ⇒ 返回 `None` + 台账显式标注「未取得（allocator 口径不可得）」**，
   **绝不**回退宿主采样（这是旧实现的缺陷本体）；
3. 宿主采样进**独立字段** `host_sample_peak_gib`（+ 缺口 `host_sample_peak_gap_mib`），
   **不进** `peak_memory_gib`；
4. 口径标识 `peak_memory_caliber` 与 §9.1 的 allocator 口径同名，且宿主值不得经
   `ledger_row` 的 `peak_memory_gib` 参数进入该列。

不触 GPU、不需要 torch：只跑采集器 CLI 与按文件路径加载的纯逻辑模块。
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


def _load_module(name: str, path: Path) -> ModuleType:
    """按文件路径加载（collector 与 result_judge 都**不能**经 graspo/__init__ 拉 torch）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_JUDGE = _load_module("_peak_caliber_result_judge", _JUDGE_SOURCE)
_COLLECTOR_MOD = _load_module("_peak_caliber_collector", _COLLECTOR)

#: 与 §9.1 声明逐字对齐的口径标识（allocator 口径）。
_EXPECTED_CALIBER = "rank0_max_allocated"

_MANIFEST_NATIVE = {
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

_MANIFEST_MSSWIFT = {
    "schema_version": 1,
    "tiers": [
        {
            "tier_id": "T031",
            "model": "9B",
            "algorithm": "SFT",
            "mode": "LoRA",
            "backend": "ms-swift",
            "cards": 1,
            "gpus": [0],
            "status": "ready",
        }
    ],
}


def _rank_memory_line(phase: str, max_allocated_mib: float) -> dict:
    """训练侧 `_emit_rank_memory_event` 落盘的一行（与真机证据同形）。"""
    return {
        "event": "rank_memory",
        "timestamp": "2026-09-20T22:49:02+08:00",
        "phase": phase,
        "kind": "diagnostic",
        "reproducible": False,
        "rank": 0,
        "tp_rank": 0,
        "tp_size": 1,
        "memory": {
            "allocated_mib": 1.0,
            "reserved_mib": 2.0,
            "max_allocated_mib": max_allocated_mib,
            "max_reserved_mib": 3.0,
        },
    }


def _write_rank_metrics(run: Path, tier_id: str, lines: list[dict]) -> None:
    """native 布局：`<run>/outputs/<tier>/logs/rank_metrics.rank_00000.jsonl`。"""
    logs = run / "outputs" / tier_id / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "rank_metrics.rank_00000.jsonl").write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )


def _write_host_summary(run: Path, peak_mib: float, gap_mib: float = 0.0) -> None:
    gpu_dir = run / "gpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    (gpu_dir / "gpu_memory_summary.json").write_text(
        json.dumps(
            {
                "per_gpu": {"0": {"samples": 10, "memory_used_mib_peak": peak_mib}},
                "max_peak_memory_gap_mib": gap_mib,
            }
        ),
        encoding="utf-8",
    )


def _run_collector(tmp_path: Path, run_root: Path, manifest: dict) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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
    records[0]["_ledger_md"] = (out / "ledger.md").read_text(encoding="utf-8")
    return records[0]


# ── 1. 口径标识与 §9.1 声明一致 ─────────────────────────────────────────────


def test_peak_memory_caliber_matches_the_tracking_doc_declaration() -> None:
    """§9.1 声明的是「容器内 PyTorch allocator 的 rank0 `max_allocated`」⇒ 标识必须同名。

    这是防"声明改了、代码没跟"或"代码改了、声明没跟"的单点闸门（§1.4）。
    """
    assert _JUDGE.PEAK_MEMORY_CALIBER == _EXPECTED_CALIBER
    assert _JUDGE.HOST_SAMPLE_PEAK_FIELD == "host_sample_peak_gib"
    assert "allocator" in _JUDGE.PEAK_MEMORY_UNAVAILABLE


# ── 2. 取数函数：只读 allocator，不看宿主摘要 ───────────────────────────────


def test_read_peak_memory_uses_allocator_and_ignores_host_sample(tmp_path: Path) -> None:
    """两口径同场时，取到的必须是 allocator 值（宿主采样更大也不许被选中）。"""
    run = tmp_path / "T010"
    run.mkdir(parents=True)
    _write_rank_metrics(
        run,
        "T010",
        [
            _rank_memory_line("setup_after", 19895.50),
            _rank_memory_line("sft_train_batch_after", 20851.79),
            _rank_memory_line("sft_train_batch_after", 20852.23),
        ],
    )
    _write_host_summary(run, peak_mib=21991.0)

    assert _COLLECTOR_MOD._read_peak_memory(run) == 20852.23 / 1024.0
    assert _COLLECTOR_MOD._read_host_sample_peak_memory(run) == 21991.0 / 1024.0
    assert _COLLECTOR_MOD._read_peak_memory(run) < _COLLECTOR_MOD._read_host_sample_peak_memory(run)


def test_read_peak_memory_takes_the_max_over_all_rank_memory_lines(tmp_path: Path) -> None:
    """不按 phase 白名单筛选：allocator 的 max 是累计峰值，漏一个 phase 就漏峰值。"""
    run = tmp_path / "T010"
    run.mkdir(parents=True)
    _write_rank_metrics(
        run,
        "T010",
        [
            _rank_memory_line("train_before_empty_cache", 100.0),
            _rank_memory_line("某个将来才加的新 phase", 4096.0),
            _rank_memory_line("logprob_after", 50.0),
        ],
    )
    assert _COLLECTOR_MOD._read_peak_memory(run) == 4.0


def test_read_peak_memory_returns_none_when_allocator_value_absent(tmp_path: Path) -> None:
    """**核心锁**：宿主摘要在场但无 rank_metrics ⇒ 必须 None，**不**回退宿主口径。

    这正是修复前的缺陷：旧实现此时会返回 21.47 GiB（宿主）并写进 §7 那一列。
    """
    run = tmp_path / "T031"
    run.mkdir(parents=True)
    _write_host_summary(run, peak_mib=26579.0)
    assert _COLLECTOR_MOD._read_peak_memory(run) is None
    assert _COLLECTOR_MOD._read_host_sample_peak_memory(run) is not None


def test_read_peak_memory_skips_broken_lines_without_falling_back(tmp_path: Path) -> None:
    """旁路半截写入的坏行必须被跳过，且**不**因此落到宿主口径上。"""
    run = tmp_path / "T010"
    run.mkdir(parents=True)
    logs = run / "outputs" / "T010" / "logs"
    logs.mkdir(parents=True)
    (logs / "rank_metrics.rank_00000.jsonl").write_text(
        '{"event": "rank_memory", "memory": {"max_allocated_mib": 2048.0}}\n'
        '{"event": "rank_memory", "memory": {"max_allocated_mi\n',
        encoding="utf-8",
    )
    _write_host_summary(run, peak_mib=99999.0)
    assert _COLLECTOR_MOD._read_peak_memory(run) == 2.0


def test_read_peak_memory_none_when_memory_has_no_max_allocated(tmp_path: Path) -> None:
    """rank_memory 行在、但快照退化成非 CUDA 形状（只有 0 值键）⇒ None，不猜。"""
    run = tmp_path / "T010"
    run.mkdir(parents=True)
    _write_rank_metrics(run, "T010", [{"event": "rank_memory", "memory": {"allocated_mib": 1.0}}])
    assert _COLLECTOR_MOD._read_peak_memory(run) is None


# ── 3. 台账落盘：两口径分列，空值带原因 ─────────────────────────────────────


def test_ledger_separates_calibers_and_annotates_unavailable(tmp_path: Path) -> None:
    """native 档两口径分列；ms-swift 档（不产 rank_metrics）峰值列显式标注不可得。"""
    native_root = tmp_path / "native"
    (native_root / "T010").mkdir(parents=True)
    _write_rank_metrics(
        native_root / "T010", "T010", [_rank_memory_line("sft_train_batch_after", 19895.50)]
    )
    _write_host_summary(native_root / "T010", peak_mib=21991.0)

    row = _run_collector(tmp_path / "n", native_root, _MANIFEST_NATIVE)
    assert row["peak_memory_caliber"] == _EXPECTED_CALIBER
    assert row["peak_memory_note"] == ""
    assert row["peak_memory_gib"] == 19895.50 / 1024.0
    # 宿主副读数**另存**，且**不等于**峰值列（口径分离的机器可核形式）
    assert row["host_sample_peak_gib"] == 21991.0 / 1024.0
    assert row["host_sample_peak_gib"] != row["peak_memory_gib"]
    assert row["host_sample_peak_gap_mib"] == 0.0
    assert _EXPECTED_CALIBER in row["_ledger_md"]

    # ms-swift：宿主摘要在，allocator 结构性不可得
    swift_root = tmp_path / "swift"
    (swift_root / "T031").mkdir(parents=True)
    _write_host_summary(swift_root / "T031", peak_mib=26579.0, gap_mib=2388.0)
    swift_row = _run_collector(tmp_path / "s", swift_root, _MANIFEST_MSSWIFT)
    assert swift_row["peak_memory_gib"] is None
    assert swift_row["peak_memory_note"] == _JUDGE.PEAK_MEMORY_UNAVAILABLE
    assert swift_row["host_sample_peak_gib"] == 26579.0 / 1024.0
    assert swift_row["host_sample_peak_gap_mib"] == 2388.0
    # 台账 md 里必须看得见原因，而不是一个光秃秃的 `—`
    assert _JUDGE.PEAK_MEMORY_UNAVAILABLE in swift_row["_ledger_md"]


def test_ledger_row_never_lets_host_value_reach_the_peak_column(tmp_path: Path) -> None:
    """纯逻辑层锁：宿主值只能走独立参数，`peak_memory_gib` 的语义不被污染。"""
    judgement = _JUDGE.TierJudgement(
        tier_id="T010",
        criteria=(
            _JUDGE.CriterionResult(
                criterion="A6", passed=True, detail="数值健康", evidence_missing=False
            ),
        ),
        passed=False,
        failure_class=None,
        counts_toward_max_context=False,
        note="",
    )
    row = _JUDGE.ledger_row(
        judgement,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="native",
        cards=1,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-20",
        host_sample_peak_gib=21.4755859375,
    )
    assert row["peak_memory_gib"] is None
    assert row["peak_memory_note"] == _JUDGE.PEAK_MEMORY_UNAVAILABLE
    assert row[_JUDGE.HOST_SAMPLE_PEAK_FIELD] == 21.4755859375
