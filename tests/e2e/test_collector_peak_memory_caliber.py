"""§7「实测每卡峰值显存(GiB)」列的**取数口径**回归测试（本模块锁死混口径不再复发）。

背景（口径演变）：
- 权威口径（`.local/本期工程跟踪.md` §9.1「★ §7 各列的读取口径」表，**`:420`**）：
  **二分口径，按后端** —— `native` 档 = 容器内 PyTorch allocator 的 rank0
  `max_allocated`；`ms-swift` 档 = 该进程的 `max_memory_reserved`（同进程全部可见卡取
  max）。其它口径（OOM 报文的进程占用、宿主 `nvidia-smi` 采样）**仍不得混入该列**。
- 旧实现（本包修复前）`collect_results._read_peak_memory` 读的是
  `gpu/gpu_memory_summary.json` 的 `per_gpu[*].memory_used_mib_peak` —— 那是**宿主
  `nvidia-smi` 采样**，与声明不符。修好采样链路后，新跑的档会把宿主口径写进同一列
  ⇒ **同列混口径**，正是 §9.1 明文禁止的。
- 宿主口径在 228（**共享机**）上还会被他人作业污染：`T033/run2` 实测 GPU2 =
  27451 MiB 而 `max_peak_memory_gap_mib = 2388`（真值应 ≈ 25063 MiB）。
  ⇒ 宿主采样这条路不只是"口径不同"，而是**会给出错误答案**，**任何时候都不得**回填该列。
- 旧的"单一口径"还有个结构性后果：`ms-swift` 后端**不产** `rank_metrics` 旁路
  （其自报 `memory(GiB)` 取的是 `max_memory_reserved`，三处打印点都不调
  `max_memory_allocated`）⇒ 54 档里 36 个 ms-swift 档**永久填不出值**，§7 那一列
  便无法回答用户的"我的硬件够不够"。2026-09-21 指挥官裁定「方案 A：二分口径」，
  使**全 54 档都能填出值**。

本模块的锁（混口径仍不得复发）：

1. `_read_peak_memory(run_dir, backend)` **按后端选口径并成对返回 `(值, 口径标签)`**：
   `native` ⇒ 只认 **rank_metrics 旁路的 allocator 值**（取全部 `rank_memory` 行
   `max_allocated_mib` 的最大值）；`ms-swift` ⇒ 只认自报 `memory(GiB)`（= reserved，
   **保守上界**）；
2. **本档口径不可得 ⇒ 返回 `(None, 该档口径标签)` + 台账按该口径显式标注**，
   **绝不**回退宿主采样、**也绝不**回退另一种口径；
3. 宿主采样进**独立字段** `host_sample_peak_gib`（+ 缺口 `host_sample_peak_gap_mib`），
   **不进** `peak_memory_gib`；
4. 口径标签 `peak_memory_caliber` 由**后端→口径的唯一映射**
   （`result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND`）决定；未登记后端 ⇒ 显式
   `unknown_backend_no_caliber`（不猜）；`ledger_row` 对本档推导值与采集层标签不一致
   **当场拒绝**；
5. 二分口径**不放宽**任何既有底线：宿主值仍不得经 `ledger_row` 的 `peak_memory_gib`
   参数进入该列。

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
    """§9.1 声明的是二分口径 ⇒ 两个口径标识与「后端→口径」映射必须逐字对齐。

    这是防"声明改了、代码没跟"或"代码改了、声明没跟"的单点闸门（§1.4）。
    """
    assert _JUDGE.PEAK_MEMORY_CALIBER == _EXPECTED_CALIBER
    assert _JUDGE.PEAK_MEMORY_CALIBER_BY_BACKEND == {
        "native": _EXPECTED_CALIBER,
        "ms-swift": _JUDGE.MSSWIFT_RESERVED_CALIBER,
    }
    assert _JUDGE.peak_memory_caliber_for_backend("native") == _EXPECTED_CALIBER
    assert _JUDGE.peak_memory_caliber_for_backend("ms-swift") == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert _JUDGE.HOST_SAMPLE_PEAK_FIELD == "host_sample_peak_gib"
    assert "allocator" in _JUDGE.PEAK_MEMORY_UNAVAILABLE


def test_unregistered_backend_gets_an_explicit_no_caliber_label() -> None:
    """未登记的后端 ⇒ **显式**"无口径"标签，**不猜、不默认取某一支**（§2.2）。

    为什么必须有这条：若未登记后端静默落回 allocator 口径，台账会声称一个它**没有**
    校验过的口径——这正是"这一格是什么口径"变得不可回答的通道。
    """
    assert _JUDGE.peak_memory_caliber_for_backend("教学中台") == _JUDGE.PEAK_MEMORY_CALIBER_UNKNOWN
    assert _JUDGE.PEAK_MEMORY_CALIBER_UNKNOWN not in _JUDGE.PEAK_MEMORY_CALIBER_BY_BACKEND.values()
    assert (
        _JUDGE.peak_memory_unavailable_note(_JUDGE.PEAK_MEMORY_CALIBER_UNKNOWN)
        == _JUDGE.PEAK_MEMORY_UNAVAILABLE_UNKNOWN
    )


def test_both_calibers_have_distinct_explicit_unavailable_notes() -> None:
    """**两者都不可得 ⇒ 各自显式标注且文案不同**：下游必须能分辨"缺哪个值"。

    若两种情况落同一句话，下游就无法判断该档缺的是 rank_metrics 旁路还是 ms-swift 日志，
    也就无法判断"这台机器上这个值到底能不能取到"（§2.2）。
    """
    native_note = _JUDGE.peak_memory_unavailable_note(_JUDGE.PEAK_MEMORY_CALIBER)
    swift_note = _JUDGE.peak_memory_unavailable_note(_JUDGE.MSSWIFT_RESERVED_CALIBER)
    assert native_note == _JUDGE.PEAK_MEMORY_UNAVAILABLE
    assert swift_note == _JUDGE.MSSWIFT_RESERVED_UNAVAILABLE
    assert native_note != swift_note
    assert native_note and swift_note  # 都不是空串：空串会被下游读成"没有原因"


def test_ledger_row_rejects_a_caliber_that_contradicts_the_backend(tmp_path: Path) -> None:
    """采集层标签与本档推导值不一致 ⇒ **当场拒绝**，不把错标签落盘（§1.4）。"""
    judgement = _JUDGE.TierJudgement(
        tier_id="T031",
        criteria=(),
        passed=False,
        failure_class=None,
        counts_toward_max_context=False,
        note="",
    )
    try:
        _JUDGE.ledger_row(
            judgement,
            model="9B",
            algorithm="SFT",
            mode="LoRA",
            backend="ms-swift",
            cards=1,
            max_context=None,
            peak_memory_gib=24.45,
            date="2026-09-20",
            peak_memory_caliber=_JUDGE.PEAK_MEMORY_CALIBER,  # 后端的口径是 reserved
        )
    except ValueError as error:
        assert _JUDGE.MSSWIFT_RESERVED_CALIBER in str(error)
    else:  # pragma: no cover - 反例必须抛
        raise AssertionError("口径标签与本档 backend 矛盾时，ledger_row 必须拒绝而不是落盘")


# ── 2. 取数函数：按后端选口径，只读本档口径、不看宿主摘要 ──────────────────────


def _native_peak(run: Path) -> tuple[float | None, str]:
    return _COLLECTOR_MOD._read_peak_memory(run, "native")


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

    value, caliber = _native_peak(run)
    assert value == 20852.23 / 1024.0
    assert caliber == _EXPECTED_CALIBER  # 口径标签必须与值成对返回
    assert _COLLECTOR_MOD._read_host_sample_peak_memory(run) == 21991.0 / 1024.0
    assert value < _COLLECTOR_MOD._read_host_sample_peak_memory(run)


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
    value, caliber = _native_peak(run)
    assert value == 4.0
    assert caliber == _EXPECTED_CALIBER


def test_read_peak_memory_returns_none_when_allocator_value_absent(tmp_path: Path) -> None:
    """**核心锁**：宿主摘要在场但无 rank_metrics ⇒ 必须 None，**不**回退宿主口径。

    这正是修复前的缺陷：旧实现此时会返回 21.47 GiB（宿主）并写进 §7 那一列。
    口径标签仍须返回本档口径（allocator），让"缺值"与"缺口径"可分辨。
    """
    run = tmp_path / "T031"
    run.mkdir(parents=True)
    _write_host_summary(run, peak_mib=26579.0)
    value, caliber = _native_peak(run)
    assert value is None
    assert caliber == _EXPECTED_CALIBER
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
    assert _native_peak(run)[0] == 2.0


def test_read_peak_memory_none_when_memory_has_no_max_allocated(tmp_path: Path) -> None:
    """rank_memory 行在、但快照退化成非 CUDA 形状（只有 0 值键）⇒ None，不猜。"""
    run = tmp_path / "T010"
    run.mkdir(parents=True)
    _write_rank_metrics(run, "T010", [{"event": "rank_memory", "memory": {"allocated_mib": 1.0}}])
    assert _native_peak(run)[0] is None


# ── 3. 台账落盘：两口径分列，空值带原因 ─────────────────────────────────────


def test_ledger_separates_calibers_and_annotates_unavailable(tmp_path: Path) -> None:
    """native 档取 allocator；ms-swift 档（无自报日志）峰值列**按本档口径**显式标注不可得。"""
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

    # ms-swift：宿主摘要在，且**没有** ms-swift 自报日志 ⇒ 该档口径（reserved）不可得
    swift_root = tmp_path / "swift"
    (swift_root / "T031").mkdir(parents=True)
    _write_host_summary(swift_root / "T031", peak_mib=26579.0, gap_mib=2388.0)
    swift_row = _run_collector(tmp_path / "s", swift_root, _MANIFEST_MSSWIFT)
    assert swift_row["peak_memory_gib"] is None
    # ★ 标注必须是**本档口径**那一句，不能是 allocator 那一句
    assert swift_row["peak_memory_caliber"] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert swift_row["peak_memory_note"] == _JUDGE.MSSWIFT_RESERVED_UNAVAILABLE
    assert swift_row["peak_memory_note"] != _JUDGE.PEAK_MEMORY_UNAVAILABLE
    # 宿主值在场，但**绝不**进峰值列（共享机会被他人作业污染）
    assert swift_row["host_sample_peak_gib"] == 26579.0 / 1024.0
    assert swift_row["host_sample_peak_gap_mib"] == 2388.0
    # 台账 md 里必须看得见原因，而不是一个光秃秃的 `—`
    assert _JUDGE.MSSWIFT_RESERVED_UNAVAILABLE in swift_row["_ledger_md"]


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


# ── 4. ms-swift 自报 `memory(GiB)`：reserved 口径，即 ms-swift 档的 §7 峰值口径 ──
#
# 定性结论（源码核证，见 `.local/refs/ms-swift-4.5.3/git-v4.5.3`）：
# ms-swift 的 `logs['memory(GiB)']` 取的是 `max_memory_reserved`，**不是**
# `max_memory_allocated`（`swift/trainers/patcher.py:27` → `swift/utils/torch_utils.py:416`；
# `swift/megatron/callbacks/print.py:64` 同）。reserved ≥ allocated ⇒ 它**不可与**
# native 档的 allocator 值直接互换，故 §9.1 用**二分口径**（按后端），并要求逐格口径
# 显式落台账。本节的锁是"值 + 口径标签成对 + 宿主值绝不混入 + reserved 是保守上界"。


def _write_msswift_logging(run: Path, values: list[float]) -> None:
    """ms-swift 布局：`<run>/logging.jsonl`，键逐字为 `memory(GiB)`。"""
    run.mkdir(parents=True, exist_ok=True)
    (run / "logging.jsonl").write_text(
        "".join(
            json.dumps({"loss": 0.5, "memory(GiB)": value, "train_speed(s/it)": 1.0}) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def test_msswift_caliber_is_reserved_and_distinct_from_the_peak_column() -> None:
    """口径标识必须是 reserved（不是 allocator），且与 §9.1 的列口径**不同名**。

    这条锁防的是"看着都是 PyTorch allocator 就顺手当同一口径接上"——上游用的是
    `max_memory_reserved`，`reserved >= allocated`，混填会让 §7 那一列不可回答。
    """
    assert _JUDGE.MSSWIFT_RESERVED_CALIBER == "msswift_rank0_max_reserved"
    assert _JUDGE.MSSWIFT_RESERVED_CALIBER != _JUDGE.PEAK_MEMORY_CALIBER
    assert "reserved" in _JUDGE.MSSWIFT_RESERVED_CALIBER
    # 说明文案必须自带源码依据（可核到文件:行），否则下游无法复核口径。
    assert "max_memory_reserved" in _JUDGE.MSSWIFT_RESERVED_NOTE
    assert "patcher.py" in _JUDGE.MSSWIFT_RESERVED_NOTE
    assert "allocated" in _JUDGE.MSSWIFT_RESERVED_NOTE
    # 字段名与口径标签字段名都必须是独立字段（不许复用 peak_memory_*）。
    assert _JUDGE.MSSWIFT_RESERVED_PEAK_FIELD == "msswift_reserved_peak_gib"
    assert _JUDGE.MSSWIFT_RESERVED_PEAK_FIELD != _JUDGE.PEAK_MEMORY_CALIBER
    assert _JUDGE.MSSWIFT_RESERVED_CALIBER_FIELD != _JUDGE.MSSWIFT_RESERVED_PEAK_FIELD


def test_read_msswift_reserved_peak_from_logging_jsonl(tmp_path: Path) -> None:
    """解析成功：`logging.jsonl` 的 `memory(GiB)`，逐行取**最大**（= 该 run 峰值）。

    实测对照（`task-j1-batch-msswift/evidence`）：T031 = 24.45、T032 = 23.05、
    T033 = 23.06 GiB；宿主采样峰值分别为 25.96 / 24.47 / 24.48 GiB。
    """
    run = tmp_path / "T031"
    _write_msswift_logging(run, [23.57, 23.57, 24.45])
    assert _COLLECTOR_MOD._read_msswift_reserved_peak_memory(run) == 24.45


def test_read_msswift_reserved_peak_takes_max_across_nested_logging_jsonl(tmp_path: Path) -> None:
    """多卡/多段：ms-swift 每个日志事件只打一个数（对本进程全部卡取 max），

    故"每卡峰值" = 所有行取最大；嵌套层（`outputs/.../logging.jsonl`）也要能扫到。
    """
    run = tmp_path / "T033"
    _write_msswift_logging(run / "outputs" / "T033", [20.0])
    _write_msswift_logging(run / "rerun", [23.06])
    assert _COLLECTOR_MOD._read_msswift_reserved_peak_memory(run) == 23.06


def test_read_msswift_reserved_peak_falls_back_to_stdout_dict_repr(tmp_path: Path) -> None:
    """stdout 回退：tqdm 把 dict repr 打进 stdout（实测档 `T044` 只有 stdout）。

    逐字样本：`... 'memory(GiB)': '56.15', ...`
    """
    run = tmp_path / "T044"
    run.mkdir(parents=True)
    (run / "stdout.log").write_text(
        "Train:  10%|#         | 1/10 [00:10<01:30, 1.00s/it]"
        "{'loss': 0.5, 'memory(GiB)': '56.14', 'train_speed(s/it)': 1.0}\n"
        "{'loss': 0.4, 'memory(GiB)': '56.15', 'train_speed(s/it)': 1.0}\n",
        encoding="utf-8",
    )
    assert _COLLECTOR_MOD._read_msswift_reserved_peak_memory(run) == 56.15


def test_read_msswift_reserved_peak_returns_none_and_never_falls_back(tmp_path: Path) -> None:
    """**负向锁**：无 `memory(GiB)` ⇒ None；宿主摘要在场也**绝不**回退（§9.1 禁混口径）。

    并且：native 档（有 rank_metrics、无 `memory(GiB)`）必须**不**被本函数"顺手指"到
    ——它只认 ms-swift 的那一个键，不看 rank_metrics。
    """
    run = tmp_path / "T031"
    run.mkdir(parents=True)
    _write_host_summary(run, peak_mib=26579.0)
    assert _COLLECTOR_MOD._read_msswift_reserved_peak_memory(run) is None
    assert _COLLECTOR_MOD._read_host_sample_peak_memory(run) is not None

    native = tmp_path / "T010"
    native.mkdir(parents=True)
    _write_rank_metrics(native, "T010", [_rank_memory_line("sft_train_batch_after", 20852.23)])
    assert _COLLECTOR_MOD._read_msswift_reserved_peak_memory(native) is None
    assert _COLLECTOR_MOD._read_peak_memory(native, "native")[0] == 20852.23 / 1024.0


def test_read_msswift_reserved_peak_none_on_broken_jsonl_and_unparsable_stdout(tmp_path: Path) -> None:
    """显式标注的前置条件：坏行 / 非数值 ⇒ None（不猜、不取 0）。"""
    run = tmp_path / "T031"
    run.mkdir(parents=True)
    (run / "logging.jsonl").write_text(
        '{"memory(GiB)": "not-a-number"}\n{"memory(GiB)": nu\n', encoding="utf-8"
    )
    (run / "stdout.log").write_text("{'memory(GiB)': 'n/a'}\n", encoding="utf-8")
    assert _COLLECTOR_MOD._read_msswift_reserved_peak_memory(run) is None


def test_read_peak_memory_dispatches_to_reserved_for_msswift_backend(tmp_path: Path) -> None:
    """**二分口径的取数锁**：同一个 run 目录，`backend` 决定口径。

    这是 ms-swift 档"填得出值"的机核形式：`_read_peak_memory(run, "ms-swift")` 必须
    返回（自报 reserved 值, reserved 口径标签），而不是 None（旧单一口径下的结果）。
    """
    run = tmp_path / "T031"
    _write_msswift_logging(run, [23.57, 24.45])
    _write_host_summary(run, peak_mib=26579.0)

    value, caliber = _COLLECTOR_MOD._read_peak_memory(run, "ms-swift")
    assert value == 24.45
    assert caliber == _JUDGE.MSSWIFT_RESERVED_CALIBER
    # 宿主值在场且更大，仍**不得**被选中（共享机会被他人作业污染）
    assert value != _COLLECTOR_MOD._read_host_sample_peak_memory(run)
    # 同一目录用 native 口径取 ⇒ None（结构性不产 rank_metrics）＋ allocator 口径标签。
    # ⇒ "后端→口径"确实是**决定性**的，而不是"谁有值取谁"。
    native_value, native_caliber = _COLLECTOR_MOD._read_peak_memory(run, "native")
    assert native_value is None
    assert native_caliber == _EXPECTED_CALIBER

    # 未登记后端 ⇒ 值不可得 + 显式"无口径"标签（不猜、不回落）
    unknown_value, unknown_caliber = _COLLECTOR_MOD._read_peak_memory(run, "教学中台")
    assert unknown_value is None
    assert unknown_caliber == _JUDGE.PEAK_MEMORY_CALIBER_UNKNOWN


def test_ledger_fills_the_peak_column_for_msswift_with_its_own_caliber(tmp_path: Path) -> None:
    """**核心正向锁（方案 A）**：ms-swift 档的 §7 峰值列 = 自报 reserved，口径已标注。

    同时是**负向锁**：宿主采样值（26579 MiB = 25.96 GiB，且带 2388 MiB 缺口）**没有**
    被写进峰值列——它仍只在 `host_sample_peak_gib` 独立字段里。
    """
    swift_root = tmp_path / "swift"
    _write_msswift_logging(swift_root / "T031", [23.57, 24.45])
    _write_host_summary(swift_root / "T031", peak_mib=26579.0, gap_mib=2388.0)

    row = _run_collector(tmp_path / "s", swift_root, _MANIFEST_MSSWIFT)
    # §7 峰值列：ms-swift 档取本后端口径（reserved），值来自自报 memory(GiB)
    assert row["peak_memory_gib"] == 24.45
    assert row["peak_memory_caliber"] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert row["peak_memory_note"] == ""  # 有值 ⇒ 无"不可得"标注
    # ★ 负向：宿主值绝不进峰值列（差 1.5 GiB，且含 2388 MiB 他人作业成分）
    assert row["peak_memory_gib"] != row["host_sample_peak_gib"]
    assert row["host_sample_peak_gib"] == 26579.0 / 1024.0
    assert row["host_sample_peak_gap_mib"] == 2388.0
    # 自报值的"原始读数"另存字段与峰值列同值同源（不是第二个真相源）
    assert row[_JUDGE.MSSWIFT_RESERVED_PEAK_FIELD] == row["peak_memory_gib"]
    assert row[_JUDGE.MSSWIFT_RESERVED_CALIBER_FIELD] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert row[_JUDGE.MSSWIFT_RESERVED_NOTE_FIELD] == _JUDGE.MSSWIFT_RESERVED_NOTE
    # md 里：列头写死二分口径，值可见，"保守上界"可见
    assert _JUDGE.MSSWIFT_RESERVED_CALIBER in row["_ledger_md"]
    assert _EXPECTED_CALIBER in row["_ledger_md"]
    assert "保守上界" in row["_ledger_md"]
    assert "24.45" in row["_ledger_md"]


def test_ledger_row_annotates_msswift_reserved_unavailable(tmp_path: Path) -> None:
    """未取得 ⇒ 显式写原因（**本档口径那一句**），**不是**空字符串、更不是宿主值。"""
    judgement = _JUDGE.TierJudgement(
        tier_id="T031",
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
        backend="ms-swift",
        cards=1,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-20",
        host_sample_peak_gib=25.9560546875,
        msswift_reserved_peak_gib=None,
    )
    assert row[_JUDGE.MSSWIFT_RESERVED_PEAK_FIELD] is None
    assert row[_JUDGE.MSSWIFT_RESERVED_NOTE_FIELD] == _JUDGE.MSSWIFT_RESERVED_UNAVAILABLE
    assert row[_JUDGE.MSSWIFT_RESERVED_CALIBER_FIELD] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert row["peak_memory_gib"] is None
    # 峰值列的"不可得"标注必须按**本档口径**（reserved），不是 allocator 那一句
    assert row["peak_memory_caliber"] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert row["peak_memory_note"] == _JUDGE.MSSWIFT_RESERVED_UNAVAILABLE
    # 宿主值只在独立字段里，**没有**趁 peak 为空时被回填
    assert row[_JUDGE.HOST_SAMPLE_PEAK_FIELD] == 25.9560546875
    assert row["peak_memory_gib"] != row[_JUDGE.HOST_SAMPLE_PEAK_FIELD]

    # 采集层把自报 reserved 值经 `peak_memory_gib` 传入（方案 A 的正常路径）⇒ 值落列、
    # 口径标签为 reserved、无"不可得"标注；**原始读数**字段仍配对落盘。
    row2 = _JUDGE.ledger_row(
        judgement,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="ms-swift",
        cards=1,
        max_context=None,
        peak_memory_gib=56.15,
        date="2026-09-20",
        peak_memory_caliber=_JUDGE.MSSWIFT_RESERVED_CALIBER,
        msswift_reserved_peak_gib=56.15,
    )
    assert row2["peak_memory_gib"] == 56.15
    assert row2["peak_memory_caliber"] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert row2["peak_memory_note"] == ""
    assert row2[_JUDGE.MSSWIFT_RESERVED_PEAK_FIELD] == 56.15
    assert row2[_JUDGE.MSSWIFT_RESERVED_NOTE_FIELD] == _JUDGE.MSSWIFT_RESERVED_NOTE
    # ★ 保守上界必须写进随值落库的说明里（用户要知道这个数是上界，不是实测下界）
    assert "保守上界" in _JUDGE.MSSWIFT_RESERVED_NOTE

    # 未登记后端 ⇒ 值不可得 + 显式"无口径"标签 + 专用原因文案（不猜、不回落）
    row3 = _JUDGE.ledger_row(
        judgement,
        model="9B",
        algorithm="SFT",
        mode="LoRA",
        backend="教学中台",
        cards=1,
        max_context=None,
        peak_memory_gib=None,
        date="2026-09-20",
    )
    assert row3["peak_memory_caliber"] == _JUDGE.PEAK_MEMORY_CALIBER_UNKNOWN
    assert row3["peak_memory_note"] == _JUDGE.PEAK_MEMORY_UNAVAILABLE_UNKNOWN


# ── 5. 方案 A 的验收核心：二分口径下**两种后端都填得出值** ───────────────────
#
# 旧"单一口径"的结构性后果：36/54 个 ms-swift 档永久填不出值。本节的锁是
# "同一份 manifest 里两种后端各一档 ⇒ 两档 `peak_memory_gib` **都不是 None**，
# 且各自 `peak_memory_caliber` 正确、两值互不串味"。


def test_both_backends_fill_the_peak_column_in_one_manifest(tmp_path: Path) -> None:
    """**验收锁**：native 与 ms-swift 各一档 ⇒ 两档都填出值，口径各自正确。"""
    manifest = {
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
            },
            {
                "tier_id": "T031",
                "model": "9B",
                "algorithm": "SFT",
                "mode": "LoRA",
                "backend": "ms-swift",
                "cards": 1,
                "gpus": [1],
                "status": "ready",
            },
        ],
    }
    run_root = tmp_path / "runs"
    # native 档：rank0 旁路 + 宿主摘要（后者更大，且**不得**被选中）
    _write_rank_metrics(
        run_root / "T010", "T010", [_rank_memory_line("sft_train_batch_after", 20852.23)]
    )
    _write_host_summary(run_root / "T010", peak_mib=21991.0)
    # ms-swift 档：自报 logging.jsonl + 宿主摘要（后者更大，同样不得被选中）
    _write_msswift_logging(run_root / "T031", [23.57, 24.45])
    _write_host_summary(run_root / "T031", peak_mib=26579.0, gap_mib=2388.0)

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
    rows = {
        row["tier_id"]: row
        for row in (
            json.loads(line)
            for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assert set(rows) == {"T010", "T031"}

    native_row, swift_row = rows["T010"], rows["T031"]
    # ① 两档都**填得出值**（这正是方案 A 要消除的缺口）
    assert native_row["peak_memory_gib"] is not None
    assert swift_row["peak_memory_gib"] is not None
    # ② 口径各自正确，且互不串味
    assert native_row["peak_memory_caliber"] == _EXPECTED_CALIBER
    assert swift_row["peak_memory_caliber"] == _JUDGE.MSSWIFT_RESERVED_CALIBER
    assert native_row["peak_memory_caliber"] != swift_row["peak_memory_caliber"]
    # ③ 值分别来自本档口径的落点
    assert native_row["peak_memory_gib"] == 20852.23 / 1024.0
    assert swift_row["peak_memory_gib"] == 24.45
    # ④ 负向：两档的宿主值都在场且都更大，**都没有**被选中
    assert native_row["peak_memory_gib"] < native_row["host_sample_peak_gib"]
    assert swift_row["peak_memory_gib"] < swift_row["host_sample_peak_gib"]
    # ⑤ 有值 ⇒ 无"不可得"标注
    assert native_row["peak_memory_note"] == ""
    assert swift_row["peak_memory_note"] == ""
    # ⑥ 可推导性：口径由「后端」列单值推导 ⇒ 台账 md 里两个口径名都出现
    md = (out / "ledger.md").read_text(encoding="utf-8")
    assert _EXPECTED_CALIBER in md
    assert _JUDGE.MSSWIFT_RESERVED_CALIBER in md
    assert "保守上界" in md
