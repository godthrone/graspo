#!/usr/bin/env python3
"""结果收集器：抽取一次/两次运行的证据，调用 A1–A6 判定，产出可填台账的记录。

职责：读取 54 档运行清单与运行根目录，对每档抽取（退出码、日志、optimizer step、
权重是否变化、checkpoint 是否可重载、四件套产物、loss/grad_norm 序列），调用
``graspo.core.result_judge`` 的纯逻辑判定，写出：

  - ``<out>/ledger.jsonl``  —— 每档一行机器可读记录
  - ``<out>/ledger.md``     —— 每档一行 Markdown，可直接填进 §7 台账

**硬要求**：只有"真 OOM"才计入最大可行上下文；其余失败记为 ❌ 失败 + 失败类型，
修复后重测。抽取不到的判据证据一律判不通过（fail-closed），不默认通过。

运行根目录约定（由 ``tests/e2e/run_matrix54.sh`` 产出）：
    <run_root>/<T###>/exit_code, stdout.log, gpu/, <训练输出目录>

用法：
    python3 scripts/collect_results.py --manifest tests/e2e/matrix54_manifest.json \
        --run-root .local/matrix54-runs --out .local/matrix54-ledger
    # A4 双跑：--run-root attempt1 --rerun-root attempt2
    # 全参档 A2 的权重证据（可选）：加 --base-model-root <宿主模型根目录>
    #   —— 宿主路径**必须走 CLI 参数**（§10.1）；旧环境变量
    #   GRASPO_MODELS_HOST_ROOT / GRASPO_BASE_MODEL_FOR_COMPARE 已删除（§7.1/§1.4）。
    #   不给该参数时 A2 权重证据按"取证缺口"记不可判定，不猜也不放松。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

# ── 纯逻辑模块按文件路径加载，避免经 graspo/__init__ 拉入 torch/pydantic ──────
_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_result_judge() -> ModuleType:
    source = _SRC / "graspo" / "core" / "result_judge.py"
    if source.is_file():
        spec = importlib.util.spec_from_file_location("_graspo_result_judge", source)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            # 必须先登记到 sys.modules：dataclass 装饰器在 exec 时依赖
            # cls.__module__ 能在 sys.modules 中找到对应模块。
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    import graspo.core.result_judge as installed  # noqa: PLC0415

    return installed


_judge: Any = _load_result_judge()

_LOSS_DICT = re.compile(
    r"['\"]loss['\"]\s*:\s*([-\d.eE+naif]+)[^}]*?['\"]grad_norm['\"]\s*:\s*([-\d.eE+naif]+)"
)
_LOSS_ANY = re.compile(r"['\"]loss['\"]\s*:\s*([-\d.eE+naif]+)")
_GRAD_ANY = re.compile(r"['\"]grad_norm['\"]\s*:\s*([-\d.eE+naif]+)")
_GLOBAL_STEP = re.compile(r"['\"]?global_step['\"]?\s*[:=]\s*(\d+)")

#: 显式数值语法：只承认十进制/科学计数法/NaN/Inf 字面量。
#: 为什么不复用 ``_LOSS_ANY`` 的 ``[-\d.eE+naif]+``：那个字符类**包含 ``n``**，
#: 于是 ``"loss": null`` 会匹配出 ``"n"``，再被 ``float()`` 抛错而变成 NaN。
#: F-4（2026-09-18）实测的"偶然通过"正来自这条路径——一旦某天改成 ``0.0``，
#: 判定器会把坏 run 记成"数值健康"。这里把语法收窄，null/缺失**一律显式识别**。
_NUMERIC_LITERAL = re.compile(
    r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^-?(?:nan|inf)$", re.IGNORECASE
)

#: 训练器在首个非有限梯度处硬失败时打的标记（唯一真相源：
#: ``flow/adapters/models/qwen35_36/training_sft.py``）。
#: native SFT 路径的实现痕迹（用于逐档判定 A7 的 fail-closed 保证）。
_NATIVE_SFT_PATH_MARKER = re.compile(r"train_batch_sft|_pipeline_train_batch_sft|sft_trainer\.py")

_NONFINITE_GRAD_MARKER = re.compile(r"非有限梯度|non-?finite gradient")


# ── 基础工具 ────────────────────────────────────────────────────────────────


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


#: rank0 逐 rank 指标旁路的**固定文件名**（rank 编码在文件名里，见 §9.1 的 "rank0"）。
#: 训练侧 `transformer_adapter._emit_rank_memory_event` 写的正是这一份。
_PEAK_MEMORY_RANK_FILE = "rank_metrics.rank_00000.jsonl"

#: ms-swift 自报显存峰值的**机器可读落点**与键名（见
#: :func:`_read_msswift_reserved_peak_memory` 的口径说明）。
#: 键名来自上游源码逐字：``swift/trainers/patcher.py:29`` ``logs['memory(GiB)']``。
_MSSWIFT_LOGGING_JSONL = "logging.jsonl"
_MSSWIFT_MEMORY_KEY = "memory(GiB)"

#: stdout 回退用的取值正则（tqdm 打的 dict repr：``'memory(GiB)': '24.45'``）。
#: **只**认这一个键，不认裸 ``memory``——后者只在 train_msg 出现且语义相同但无口径自证。
_MSSWIFT_STDOUT_MEMORY_RE = re.compile(r"memory\(GiB\)['\"]?\s*:\s*['\"]?(-?\d+(?:\.\d+)?)")


def _iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读 JSONL，**只**产出顶层是 dict 的记录；坏行/非 dict 行静默跳过。

    为什么单独抽出来（§2.2 显式即防呆）：rank_metrics 是**只追加**的旁路，
    半截写入的行在真实 run 里出现过；取数函数必须能容忍坏行而**不**把它读成
    "没有这个字段"从而落到别的口径上去。
    """
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _parse_numeric(raw: str | None) -> float | None:
    """把文本/JSON 值解析成 float；``None`` / 缺失 / 非法字面量一律返回 ``None``。

    **不要**把无法解析的值变成 NaN 或 0.0：那正是 F-4 实测的"坏 run 被记为成功"
    的通道。调用方（``judge_a6`` / ``judge_a2``）把 ``None`` 当作 fail-closed 证据。
    """
    if raw is None:
        return None
    text = str(raw).strip().strip("'\"")
    if not text or text.lower() in {"null", "none", "nil"}:
        return None
    if not _NUMERIC_LITERAL.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _match_numeric(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return _parse_numeric(match.group(1))


def find_output_dirs(run_dir: Path, tier_id: str) -> list[Path]:
    """定位本次运行的全部**产物根**（两种后端布局都查，与 backend 声明无关）。

    为什么按"布局"而不是按"后端"分支：判据语义必须与后端无关，变的只是证据落点。
    把两种布局都作为候选一起查，后端标签就不再参与取证——少一处"因后端不同而放松"的入口。

    已知布局（各自显式、可验证）：

    - **native**：输出目录含 ``config.yaml``（``_backup_config`` 的唯一落点）；
    - **ms-swift**：run 目录含 ``logging.jsonl`` **且** 含 ``args.json``
      （ms-swift 4.5.3 实测：``<output_dir>/<run_name>/v<N>-<时间戳>/`` 下同时有这两个文件）。
      末行 ``logging.jsonl`` 是 HF ``trainer_state`` 的等价物（含 ``global_step`` /
      ``log_history`` / ``epoch``），因此序列抽取可以复用同一条代码路径。

    抽不到任何产物根时返回 ``[]``（调用方据此把缺失判成**取证缺口**，而不是训练失败）。
    """
    found: list[Path] = []
    candidates = [
        run_dir / "outputs" / tier_id,
        run_dir / "out" / tier_id,
        run_dir / tier_id,
        run_dir / "outputs",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.yaml").exists():
            found.append(candidate)
    for child in sorted(run_dir.rglob("config.yaml")):
        if child.parent not in found:
            found.append(child.parent)
        break
    for log_path in sorted(run_dir.rglob("logging.jsonl")):
        swift_dir = log_path.parent
        if (swift_dir / "args.json").exists() and swift_dir not in found:
            found.append(swift_dir)
    return found


def find_output_dir(run_dir: Path, tier_id: str) -> Path | None:
    """兼容入口：返回第一个产物根（新代码请用 :func:`find_output_dirs`）。"""
    dirs = find_output_dirs(run_dir, tier_id)
    return dirs[0] if dirs else None


# ── 证据抽取 ────────────────────────────────────────────────────────────────


@dataclass
class SeriesEvidence:
    """loss / grad_norm 序列 + 「训练步真推进」的逐步证据。

    ``source`` 显式标注读数口径，便于报告里区分"权威旁路"与"stdout 兜底"；
    ``notes`` 记录本次抽取遇到的退化（null 字段、口径不可得），**不静默**。
    """

    steps: int | None = None
    epochs: float | None = None
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    #: 每步全局 optimizer step 数（rank_metrics 旁路 ``global_optimizer_steps_sum``）。
    optimizer_steps_per_step: list[int] = field(default_factory=list)
    #: **本次运行计划跑完的 optimizer step 数**（ms-swift ``logging.jsonl`` 的
    #: ``global_step/max_steps`` 分母；native 侧无此读数 ⇒ ``None``）。
    #: 用途：A2 的「训练步真推进」在 ms-swift 上的等价证据 —— 实际 step 数必须
    #: 达到计划数，否则说明有步没推进（等价于 native 的逐步 optimizer_steps>0 断言）。
    declared_total_steps: int | None = None
    #: ``losses[0]`` 对应的**训练步号**（A4 的零容差首步子检查用它确认"确实是首步"）。
    #: 来源：``log_history[*].step``；rank_metrics 旁路的第一行即 step 1；
    #: stdout 兜底拿不到 ⇒ ``None``（此时 A4 如实声明该子检查"未适用"）。
    first_logged_step: int | None = None
    #: 累计因非有限梯度跳过优化器步的次数。**三态**（2026-09-22 指挥官裁定）：
    #: ``int`` = 有读数（native 精确计数 / ms-swift 从 NaN grad_norm **推断**）；
    #: ``None`` = **该后端不上报且推断不出** ⇒ A7 必须记「口径不可测」而**绝不自动通过**
    #: （同一判据对不同后端必须等价）。
    nonfinite_skips: int | None = None
    #: 该计数的**来源自证**（§2.2）：`run_metrics:skipped_nonfinite`（native 精确）/
    #: `log_inference:nan_grad_norm_count` / `log_inference:all_finite_grad_norm` /
    #: ``""``（不可得）。
    nonfinite_skips_source: str = ""
    source: str = "none"
    notes: list[str] = field(default_factory=list)
    #: 是否出现过 ``loss: null`` / 字段缺失（A6 必须对此 fail-closed）。
    loss_unavailable: bool = False
    #: 是否**真读到**非有限 loss（NaN/Inf 字面量）。与 ``loss_unavailable`` 严格
    #: 区分：前者是"数值异常"这个事实断言，后者只是"没有读数"。
    loss_nonfinite: bool = False
    #: ``{未登记的 phase 名: 条数}``——payload 含 ``metrics`` 却没进
    #: ``result_judge.STEP_METRICS_PHASES`` 的记录数（§2.2：**不静默丢弃**）。
    #: 空 dict = "没有含指标却被忽略的记录"（这个结论本身也要显式落地）。
    ignored_phase_records: dict[str, int] = field(default_factory=dict)


def extract_skipped_nonfinite_all_ranks(
    output_dir: Path | None,
) -> tuple[int | None, dict[str, int], str]:
    """**跨全部 rank** 抽取"因非有限梯度跳过优化器步"的次数（P0-1 修复）。

    为什么必须遍历全部 rank（2026-09-22 复核实测的**能产出错误 ✅** 的洞）：
    ``skipped_nonfinite`` 是**逐 rank 局部**读数，而 ``_rank_metric_steps`` 只读
    ``rank_metrics.rank_00000.jsonl`` ⇒ **rank1 单独跳过、rank0 没跳时读到 0 ⇒ A7 假通过**，
    一次"权重/LR 已分叉"的运行会被记成「✅ 训练可用」。
    矩阵命中：`T029`(dp2)/`T030`(dp4)/`T041`(dp2)/`T042`(dp4)（native graspo，pp=1）。

    口径（**取 MAX 不取 rank0**）：
      · 优先读全局键 ``global_skipped_nonfinite_sum``（新落盘，若有）；
      · 否则读该 rank 的局部 ``skipped_nonfinite``；
      · 对**全部** rank 文件取 **MAX**——任一路径发生过跳过就是发生过（逐 rank 之和会
        重复计数，MAX 是"最坏 rank"的保守读数，且不会漏报）。
    返回 ``(max_count | None, {rank: count}, 人读明细)``；一个读数都没有 ⇒ ``None``（不猜）。
    """
    if output_dir is None:
        return None, {}, "无可读产物根"
    per_rank: dict[str, int] = {}
    for events in sorted(output_dir.rglob("rank_metrics.rank_*.jsonl")):
        rank_key = events.name
        best: int | None = None
        try:
            text = events.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            if not isinstance(metrics, dict):
                continue
            for key in ("global_skipped_nonfinite_sum", "skipped_nonfinite"):
                value = metrics.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    best = value if best is None else max(best, value)
        if best is not None:
            per_rank[rank_key] = best
    if not per_rank:
        return None, {}, "没有任何 rank 文件带跳过计数读数"
    worst = max(per_rank.values())
    spread = sorted(set(per_rank.values()))
    detail = "；".join(f"{name}={value}" for name, value in sorted(per_rank.items()))
    if len(per_rank) > 1 and len(spread) > 1:
        detail += " ⇒ **⚠ 各 rank 不一致（存在 rank 间跳过数不等 ⇒ 权重/LR 可能已分叉）**"
    return worst, per_rank, detail


def _rank_metric_steps(output_dir: Path | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """从 rank_metrics 旁路读「每个训练步一行」的逐步权威指标。

    哪些 phase 承载这些指标，**不在本文件判断**——唯一真相源是
    ``graspo.core.result_judge.STEP_METRICS_PHASES``（同模块里逐条注明了每个名字
    对应的代码路径与出处行号）。训练侧有**两条** SFT 路径：pipeline（``pp_size>1``）
    emit ``pipeline_sft_train_batch_after``，普通/单卡路径 emit ``sft_train_batch_after``；
    只认前者会把全部单卡/DP/TP 档的逐步指标整批丢掉（T010 实测伪否，2026-09-20）。

    ``metrics`` 里带全局聚合口径（``global_loss_mean`` / ``global_grad_norm_mean`` /
    ``global_optimizer_steps_sum``）与逐 rank 明细 ``rank_metrics``。这解决了 F-4 的
    核心观测缺陷：stdout 的 ``sft_step`` 打的是 **rank0 局部** loss（PP 下结构性恒 0.0）
    与 rank0 局部 grad_norm（有限），全局 NaN 只在旁路可见。

    返回值第二项是**显式暴露**（§2.2 显式即防呆）：``{未登记的 phase 名: 条数}``，
    只统计 payload **含 metrics** 的记录。静默丢弃正是 T010 伪否的成因，所以这里
    宁可多报一个名字，也不让"含指标却没被读"这件事无声无息。
    """
    if output_dir is None:
        return [], {}
    rows: list[dict[str, Any]] = []
    ignored_with_metrics: dict[str, int] = {}
    for events in sorted(output_dir.rglob("rank_metrics.rank_*.jsonl")):
        if events.name != "rank_metrics.rank_00000.jsonl":
            continue  # rank0 的事件已含全体 rank 的聚合与明细，避免重复计数
        try:
            text = events.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                # 纯诊断事件（显存快照等）：登记在 _judge.DIAGNOSTIC_PHASES，无指标可丢。
                continue
            phase = payload.get("phase")
            if phase not in _judge.STEP_METRICS_PHASES:
                # 含 metrics 却不在注册表里 ⇒ **显式计数**，绝不静默丢弃。
                key = str(phase) if phase is not None else "<missing>"
                ignored_with_metrics[key] = ignored_with_metrics.get(key, 0) + 1
                continue
            rows.append(metrics)
    return rows, ignored_with_metrics


_GLOBAL_STEP_SLASH = re.compile(r"(\d+)\s*/\s*(\d+)")


def _msswift_state_from_logging(swift_dir: Path) -> dict[str, Any] | None:
    """把 ms-swift 的 ``logging.jsonl`` 归一成 **HF ``trainer_state`` 同形**的 dict。

    为什么这样做（§1.4 单一真相源 + 判据语义不因后端而变）：ms-swift 4.5.3 实测的
    ``logging.jsonl`` 末行就是 trainer_state 的等价物 —— 含 ``global_step``、
    ``log_history``（逐步 ``loss`` / ``grad_norm`` / ``step``）、``epoch``、
    ``model_parameter_info``、``last_model_checkpoint``。把**形状**归一之后，
    A2/A4/A6 读的是同一段代码、同一套语义，只是数据来源不同（``source`` 字段自证）。

    返回 ``None`` 表示该目录没有可用的 ``logging.jsonl``（不是"训练失败"）。
    """
    log_path = swift_dir / "logging.jsonl"
    if not log_path.is_file():
        return None
    entries: list[dict[str, Any]] = []
    for line in _read_text(log_path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    if not entries:
        return None

    state: dict[str, Any] = {}
    final = entries[-1]
    if isinstance(final.get("global_step"), int):
        state["global_step"] = final["global_step"]
    if isinstance(final.get("epoch"), (int, float)):
        state["epoch"] = float(final["epoch"])
    history = final.get("log_history")
    if not isinstance(history, list) or not history:
        # 没有 log_history（例如被截断/旧版本）时，用逐行 loss/grad_norm 自建同形历史。
        history = [
            {"loss": entry.get("loss"), "grad_norm": entry.get("grad_norm")}
            for entry in entries
            if isinstance(entry.get("loss"), (int, float))
            or isinstance(entry.get("grad_norm"), (int, float))
        ]
    state["log_history"] = history

    # ``global_step/max_steps`` 是"计划跑多少步"的唯一读数（A2 的等价证据）。
    for entry in reversed(entries):
        raw = entry.get("global_step/max_steps")
        if isinstance(raw, str):
            match = _GLOBAL_STEP_SLASH.search(raw)
            if match is not None:
                if "global_step" not in state:
                    state["global_step"] = int(match.group(1))
                state["max_steps"] = int(match.group(2))
                break
    return state


def _trainer_state_candidates(output_dirs: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    """列出全部 trainer_state 等价物，``(source_name, state)``。

    - ``trainer_state.json``：native 与 ms-swift（HF Trainer 在每个 checkpoint 里都写）
      共用同一文件名与结构 ⇒ 一条读取路径覆盖两者；
    - ``logging.jsonl``：ms-swift 的等价旁路（checkpoint 被清掉时仍可取证）。

    ⚠ **必须收集全部再挑最优**：ms-swift 的 ``save_steps=1`` 会在每个 checkpoint 里
    留一份 ``trainer_state.json``，取到的是 ``checkpoint-1`` 那份就只有 1 步历史 ——
    那会把一次 100 步的 run 误判成"步数不够"。因此由 :func:`_best_trainer_state`
    按 ``global_step`` 取最大者。
    """
    candidates: list[tuple[str, dict[str, Any]]] = []
    for directory in output_dirs:
        for state_path in sorted(directory.rglob("trainer_state.json")):
            state = _read_json(state_path)
            if isinstance(state, dict):
                candidates.append(("trainer_state", state))
    for directory in output_dirs:
        state = _msswift_state_from_logging(directory)
        if state is not None:
            candidates.append(("msswift_logging", state))
    return candidates


def _best_trainer_state(
    candidates: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | None:
    """挑 ``global_step`` 最大的那份（同值优先 ``trainer_state.json``）。"""
    best: tuple[str, dict[str, Any]] | None = None
    best_key: tuple[int, int] = (-1, -1)
    for source, state in candidates:
        step = state.get("global_step")
        key = (step if isinstance(step, int) else -1, 1 if source == "trainer_state" else 0)
        if key > best_key:
            best_key = key
            best = (source, state)
    return best


def _cross_check_stdout(log_text: str, losses: Sequence[float | None]) -> list[str]:
    """把旁路 loss 与 stdout 的 ``sft_step`` 行做交叉核对（口径自证，§1.4）。"""
    stdout_losses = [_match_numeric(m) for m in _LOSS_DICT.finditer(log_text)]
    if not stdout_losses:
        stdout_losses = [_match_numeric(m) for m in _LOSS_ANY.finditer(log_text)]
    if not stdout_losses:
        return ["stdout 无 loss 读数（无法交叉核对）"]
    paired = list(zip(stdout_losses, losses))
    mismatched = [
        index
        for index, (left, right) in enumerate(paired)
        if left is not None and right is not None and left != right
    ]
    if mismatched:
        return [
            f"stdout 与 rank_metrics 的 loss 口径不一致（步 "
            f"{', '.join(str(i + 1) for i in mismatched[:5])}）："
            "stdout 为 rank0 局部值，全局口径以 rank_metrics 为准"
        ]
    return []


def extract_steps_and_series(output_dirs: Sequence[Path], log_text: str) -> SeriesEvidence:
    """抽取 optimizer step / epoch / loss 序列 / grad_norm 序列 / 逐步推进证据。

    来源优先级（**与后端无关**，只与证据质量有关）：
      1. ``rank_metrics.rank_*.jsonl`` 旁路的**全局**逐步指标（native PP 唯一可信口径）；
      2. ``trainer_state.json``（native RL/SFT 与 ms-swift 每个 checkpoint 共用；
         ms-swift 无 checkpoint 时由 ``logging.jsonl`` 归一同形，见
         :func:`_msswift_state_from_logging`）——**按 ``global_step`` 取最大者**，
         否则会取到 ``checkpoint-1`` 那份只有 1 步的历史；
      3. 日志中的 dict 行（stdout 兜底，口径为 rank0 局部，显式标注）。
    """
    result = SeriesEvidence()
    metrics_rows: list[dict[str, Any]] = []
    ignored_phases: dict[str, int] = {}
    rank_metrics_dir: Path | None = None
    for directory in output_dirs:
        metrics_rows, ignored_phases = _rank_metric_steps(directory)
        if metrics_rows:
            rank_metrics_dir = directory
            break
    if ignored_phases:
        # 防呆（§2.2 显式即防呆）：含 metrics 却不在 STEP_METRICS_PHASES 里的 phase
        # 记录**必须显式暴露**。T010 伪否（2026-09-20）正是"静默丢弃"造成的：
        # 100 条含 loss/grad_norm 的记录被整批跳过，台账上却看不到任何痕迹。
        result.ignored_phase_records = dict(sorted(ignored_phases.items()))
        total_ignored = sum(ignored_phases.values())
        detail = "、".join(f"{name}×{count}" for name, count in sorted(ignored_phases.items()))
        result.notes.append(
            f"忽略了 {total_ignored} 条含 metrics 的未登记 phase 记录（{detail}）"
            "——请核对 result_judge.STEP_METRICS_PHASES：训练侧可能新增了路径"
        )
    if metrics_rows:
        result.source = "rank_metrics"
        for row in metrics_rows:
            loss = _parse_numeric(row.get("global_loss_mean"))
            grad = _parse_numeric(row.get("global_grad_norm_mean"))
            if row.get("global_loss_mean") is None:
                result.loss_unavailable = True
                # 全局 loss 旁路不可得 ⇒ 用 MISSING 哨兵表达"证据缺口"。
                # **不得**用 NaN：那会让 A6 断言"数值异常"——而我们根本没读到数。
                loss = _judge.MISSING_SENTINEL
            if row.get("global_grad_norm_mean") is None:
                grad = _judge.MISSING_SENTINEL
            result.losses.append(loss if loss is not None else _judge.MISSING_SENTINEL)
            result.grad_norms.append(grad if grad is not None else _judge.MISSING_SENTINEL)
            step_total = row.get("global_optimizer_steps_sum")
            if isinstance(step_total, int):
                result.optimizer_steps_per_step.append(step_total)
            # ★ 跳过计数**不在此处逐行累加**（那只看得到 rank0）——见函数末尾的
            #   `extract_skipped_nonfinite_all_ranks`（P0-1 修复：跨全部 rank 取 MAX）。
        result.steps = len(result.losses)
        # 旁路每步一行、首行即训练步 1 ⇒ 首步 loss 就是 losses[0]（A4 零容差子检查可用）。
        result.first_logged_step = 1
        result.notes.extend(_cross_check_stdout(log_text, result.losses))
        # epoch 与"计划步数"仍从 trainer_state 家族取（旁路不含这两项）。
        best = _best_trainer_state(_trainer_state_candidates(output_dirs))
        if best is not None:
            state = best[1]
            if isinstance(state.get("epoch"), (int, float)):
                result.epochs = float(state["epoch"])
            if isinstance(state.get("max_steps"), int):
                result.declared_total_steps = state["max_steps"]
        return result

    best = _best_trainer_state(_trainer_state_candidates(output_dirs))
    if best is not None:
        source, state = best
        if isinstance(state.get("global_step"), int):
            result.steps = state["global_step"]
        if isinstance(state.get("epoch"), (int, float)):
            result.epochs = float(state["epoch"])
        if isinstance(state.get("max_steps"), int):
            result.declared_total_steps = state["max_steps"]
        loss_entries = [
            entry
            for entry in (state.get("log_history", []) or [])
            if isinstance(entry, dict) and isinstance(entry.get("loss"), (int, float))
        ]
        if loss_entries and isinstance(loss_entries[0].get("step"), int):
            # 只有**写出步号**时才认"首步"（拿不到就留 None ⇒ A4 声明该子检查未适用）。
            result.first_logged_step = int(loss_entries[0]["step"])
        for entry in state.get("log_history", []) or []:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("loss"), (int, float)):
                result.losses.append(float(entry["loss"]))
            if isinstance(entry.get("grad_norm"), (int, float)):
                result.grad_norms.append(float(entry["grad_norm"]))
        if result.losses:
            result.source = source
            if source == "msswift_logging":
                result.notes.append(
                    "loss/grad_norm 序列来自 ms-swift logging.jsonl（HF trainer_state 同形；"
                    "该目录无 rank_metrics 旁路）"
                )

    if not result.losses:
        # stdout 兜底：显式识别 null / 缺失，绝不把"读不到"变成 NaN 或 0.0。
        # ``_parse_numeric`` 对 null/缺失返回 None；无法解释的取值（语法不合法）也
        # 返回 None，但语义不同——后者要显式记录"读到了但解释不了"，不得静默丢掉。
        raw_pairs = [
            (_parse_numeric(m.group(1)), _parse_numeric(m.group(2)))
            for m in _LOSS_DICT.finditer(log_text)
        ]
        if not raw_pairs:
            null_losses = len(re.findall(r"['\"]loss['\"]\s*:\s*(?:null|None)", log_text))
            if null_losses:
                result.loss_unavailable = True
                result.notes.append(
                    f"stdout 出现 {null_losses} 个 loss: null —— 显式识别为"
                    "「数值不可得」，fail-closed 判不通过（不再沿用 null→NaN 的偶然路径）"
                )
                result.losses.extend([_judge.MISSING_SENTINEL] * null_losses)
            else:
                result.losses.extend(
                    value
                    for value in (_match_numeric(m) for m in _LOSS_ANY.finditer(log_text))
                    if value is not None
                )
                result.notes.append("loss 序列来自 stdout 兜底（无 rank_metrics 旁路）")
        else:
            for loss, grad in raw_pairs:
                result.losses.append(loss if loss is not None else _judge.MISSING_SENTINEL)
                result.grad_norms.append(grad if grad is not None else _judge.MISSING_SENTINEL)
            null_count = len(re.findall(r"['\"]loss['\"]\s*:\s*(?:null|None)", log_text))
            if null_count:
                result.loss_unavailable = True
                result.notes.append(
                    f"stdout 出现 {null_count} 个 loss: null —— 显式识别为"
                    "「数值不可得」并以 MISSING 哨兵显式表示（不依赖 null→NaN 的偶然路径）"
                )
            result.notes.append("loss/grad_norm 序列来自 stdout 兜底（rank0 局部口径）")
        if not result.grad_norms:
            result.grad_norms.extend(
                value
                for value in (_match_numeric(m) for m in _GRAD_ANY.finditer(log_text))
                if value is not None
            )
        if result.source == "none" and (result.losses or result.grad_norms):
            result.source = "stdout"

    if result.steps is None:
        matches = _GLOBAL_STEP.findall(log_text)
        if matches:
            result.steps = int(matches[-1])
    if result.steps is None and result.losses:
        result.steps = len(result.losses)
    # ── ★ A7 取证公平性（2026-09-22 指挥官裁定，C）───────────────────────────
    #   ms-swift **不上报**跳过计数：实测 `swift/trainers/mixin.py` 的 patch 是
    #   `if grad_norm.isnan(): p.grad = None` —— **不打任何日志、也不计数**
    #   （且只判 `isnan()` 不判 `inf`）。旧行为 ⇒ `nonfinite_skips` 缺省 0 ⇒ A7 **自动通过**
    #   ⇒ native 严格判、ms-swift 宽松 ⇒ **同一判据对不同后端不等价**。
    #   但从 ms-swift 的 patch 可推：跳过时它返回的 `grad_norm` **就是 NaN**，且该读数会进
    #   逐步序列 ⇒ 用"**逐步 grad_norm 是否为 NaN**"推断跳过次数（**下界**：它不判 Inf 的
    #   那种跳过不会留下 NaN 读数）。**推断不出（无 grad_norm 读数）⇒ 保持 None**。
    # ── ★ P0-1（2026-09-22 复核）：跳过计数**必须跨全部 rank**取 MAX ─────────────
    #   `skipped_nonfinite` 是逐 rank 局部读数；只读 rank0 ⇒ rank1 单独跳过时读到 0
    #   ⇒ A7 假通过（一次"权重/LR 已分叉"的运行会被记成 ✅ 可用）。
    if result.source == "rank_metrics":
        worst, per_rank, detail = extract_skipped_nonfinite_all_ranks(rank_metrics_dir)
        if worst is not None:
            result.nonfinite_skips = worst
            result.nonfinite_skips_source = "run_metrics:all_ranks_max"
            result.notes.append(f"nonfinite 跳过（跨全部 rank 取 MAX）：{detail}")
    if result.source != "rank_metrics" and not result.nonfinite_skips_source:
        # ★ 只数**数值型**读数：`MISSING_SENTINEL`（str）与空槽**不算读数**
        #   ——否则"一个读数都没有"会被误推断成"全部有限 ⇒ 0 次跳过"（假通过）。
        real = [
            value for value in result.grad_norms
            if isinstance(value, float) and math.isfinite(value)
        ]
        nonfinite = [
            value for value in result.grad_norms
            if isinstance(value, float) and not math.isfinite(value)
        ]
        # ★ C 收紧（2026-09-22 复核）：**必须自证"每一步都有 grad_norm 读数"**。
        #   否则 `logging_steps > 1` 且 NaN 只出现在**非 logging 步**时，会得到
        #   "全部 finite ⇒ 0 次跳过" —— **仍是自动通过**（假阴性）。
        #   自证条件：读数条数 ≥ 计划步数（`declared_total_steps`，拿不到就不算自证）。
        declared = result.declared_total_steps
        readings = len(real) + len(nonfinite)
        self_proven = isinstance(declared, int) and declared > 0 and readings >= declared
        if self_proven and (real or nonfinite):
            result.nonfinite_skips = len(nonfinite)
            result.nonfinite_skips_source = (
                "log_inference:nan_grad_norm_count"
                if nonfinite
                else "log_inference:all_finite_grad_norm"
            )
        elif real or nonfinite:
            result.notes.append(
                f"grad_norm 读数 {readings} 条 < 计划步数 {declared}（无法自证每步都有读数）"
                "⇒ 跳过计数按**口径不可测**处理（不推断、不放行）"
            )
        # ② Inf 也计入 nonfinite：ms-swift 只判 isnan，而 torch 在 total_norm=inf 时
        #    clip_coef=0 ⇒ `inf*0=nan` 的梯度**照常进 step**（可能污染参数）；
        #    这里用 `math.isfinite` 同时覆盖 NaN 与 ±Inf。
        # 既无 finite 也无 nonfinite 的**数值**读数 ⇒ 保持 None（口径不可测，绝不放行）
    if _NONFINITE_GRAD_MARKER.search(log_text) and not result.nonfinite_skips:
        # 硬失败标记本身就是"这一步没推进"的证据（默认 1 次，仅用于让 A2 不通过；
        # 精确次数由 rank_metrics 的 skipped_nonfinite 给出）。
        result.nonfinite_skips = max(1, int(result.nonfinite_skips or 0))
        result.nonfinite_skips_source = "log_marker:nonfinite_grad_hard_fail"
        result.notes.append("stdout 出现「非有限梯度」硬失败标记 ⇒ 训练未真推进")

    # ── 三种情形的最终判定（必须在所有来源汇合之后做）────────────────────────
    # ① 真读到非有限值：**类型上必须是 float** 才算"读到了数"。MISSING 哨兵是 str，
    #    天然被排除 ⇒ 不会把"没有读数"升格成"数值异常"这个事实断言。
    result.loss_nonfinite = any(
        isinstance(value, float) and not math.isfinite(value) for value in result.losses
    ) or any(isinstance(value, float) and not math.isfinite(value) for value in result.grad_norms)
    # ② 有"读不到值"的步（MISSING 哨兵 = 明确知道该步没有读数）
    result.loss_unavailable = result.loss_unavailable or any(
        value is _judge.MISSING_SENTINEL for value in result.losses
    )
    return result


def _require_torch() -> Any | None:
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError:
        return None


# ── 证据来源与等级（**唯一真相源**，§1.4 单一真相源 / §2.2 显式即防呆）────────
#
# 为什么必须有这一层：A2/A3 的**判据**只有一份（§8.1），但同一判据在不同环境里
# 能拿到的**证据来源**不同（宿主无 torch / 镜像内有 torch）。旧实现把"证据来源不可得"
# 静默折叠成"取值 None"，台账上只剩一句"缺少证据" —— 于是**环境性伪否**（采集机缺
# torch）与**训练真失败**在台账里长得一模一样，228 上 T010 真机踩过。
# 下面这些常量是"这条证据是哪种等级、来自哪里"的**唯一**命名处；抽取层给出标识、
# 判定层把它写进明细、台账把它落成独立列（collect_results.run 组装）。
#
# 等级语义（**不得混用**）：
#   - 强证据：真正做了"判据原文要求的那件事"（反序列化 / 权重比对）；
#   - 弱证据：只证明了**更弱的事实**（文件完整、归档结构自洽）。弱证据**不足以放行**，
#     只能在台账里把"取证缺口"的成因说清楚，并**显式**标注自己的等级。

#: A2「权重真变化」的证据来源标识（LoRA 档）。
WEIGHT_SOURCE_RUN_METRICS = "run_metrics:lora_norm_delta"
WEIGHT_SOURCE_NATIVE_LORA_TORCH = "checkpoint:torch_load(lora_b)"
WEIGHT_SOURCE_ADAPTER_SAFETENSORS = "checkpoint:safetensors_bytes(lora_b)"
#: A2「权重真变化」的证据来源标识（全参档）。
WEIGHT_SOURCE_BASE_COMPARE = "checkpoint:base_weights_bytes_compare"

#: 全参（native / ms-swift 通用）：run 自产的**可训练参数 L2 范数变化**指标。
#: 与 ``WEIGHT_SOURCE_RUN_METRICS``（LoRA 的 ``lora_norm_delta``）**同源同强度**：
#: 两者都由 ``flow/progress_metrics.py::training_norm_event()`` 在**同一步的前后**对
#: **真实参数张量**求 L2 范数再相减（``after - before``，见 ``progress_metrics.py:52``），
#: 键名模式感知（full ⇒ ``trainable_norm_*``，lora ⇒ ``lora_norm_*``）。纯 JSON、
#: 与宿主有没有 torch 无关（§6 环境可复现）。
WEIGHT_SOURCE_TRAINABLE_NORM_METRIC = "run_metrics:trainable_norm_delta"

#: A3「checkpoint 可重载」的证据等级标识。
#: 强：容器内（镜像自带 torch）跑一次真实的 ``torch.load`` 探测。
RELOAD_SOURCE_CONTAINER_PROBE = "container_torch_probe"
#: 强：宿主自带 torch 时直接反序列化。
RELOAD_SOURCE_HOST_TORCH = "host_torch_load"
#: 强：ms-swift / peft 的 safetensors 走 stdlib 头部 + data_offsets 校验（既有路径）。
RELOAD_SOURCE_SAFETENSORS_HEADER = "safetensors_header_stdlib"
#: **弱**：宿主无 torch 时对 torch 归档做 stdlib 结构 / CRC 校验。
#: 只证明"文件完整、归档结构自洽"，**不证明 torch 能反序列化** ⇒ 不得据此放行 A3。
RELOAD_SOURCE_STRUCTURAL_ONLY = "structural_only:zip_crc"


@dataclass(frozen=True)
class WeightEvidence:
    """A2「权重真变化」的抽取结果：**判据值 + 来源 + 人读明细**。

    ``value`` 的三态与 :class:`ReloadEvidence` 一致（``None`` = 取证缺口）。
    ``source`` 为 ``None`` 恒等价于 ``value is None``（没有证据就没有来源）——这条
    不变式由构造处保证，判定层据此断言台账不会出现"有来源却没证据"的怪状态。
    """

    value: bool | None
    source: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class ReloadEvidence:
    """A3「checkpoint 可重载」的抽取结果：**判据值 + 证据等级 + 人读明细**。

    ``value is None`` 表示**取证缺口**（不可判定，fail-closed 方向不变）。
    此时 ``source`` 可以是弱证据等级（``structural_only:zip_crc``），用来在台账里
    回答"明明读到文件了，为什么还判缺口"——**弱证据绝不升格为通过**。
    """

    value: bool | None
    source: str | None = None
    detail: str = ""


def _any_nonzero(tensors: Any) -> bool | None:
    torch = _require_torch()
    if torch is None:
        return None
    found = False
    for value in tensors.values() if isinstance(tensors, dict) else []:
        try:
            if bool(torch.is_tensor(value)) and bool((value != 0).any()):
                return True
            found = found or bool(torch.is_tensor(value))
        except (RuntimeError, TypeError):
            continue
    return False if found else None


#: checkpoint 目录名的**唯一真相源**：native 落 ``final/``，ms-swift（HF Trainer）
#: 落 ``checkpoint-<step>/``。两种布局用同一套抽取逻辑（判据语义不因后端而变）。
NATIVE_CHECKPOINT_DIRNAME = "final"
SWIFT_CHECKPOINT_GLOB = "checkpoint-*"


def find_checkpoint_dirs(output_dirs: Sequence[Path]) -> list[Path]:
    """发现全部 checkpoint 目录，**按训练步号降序**（最新优先）。"""
    found: list[Path] = []
    for directory in output_dirs:
        for path in directory.rglob(NATIVE_CHECKPOINT_DIRNAME):
            if path.is_dir():
                found.append(path)
        for path in directory.rglob(SWIFT_CHECKPOINT_GLOB):
            if path.is_dir():
                found.append(path)
    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)

    def step_of(path: Path) -> int:
        tail = path.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 10**9  # final = 终态，排最前

    return sorted(unique, key=step_of, reverse=True)


#: 权重文件的**候选名**（唯一真相源）：native 的 ``pytorch_model.bin`` / HF shard /
#: ms-swift(peft) 的 ``adapter_model.safetensors``。用同一名表覆盖两种布局，
#: 避免"文件名取决于后端分支"的隐式契约（§2.2）。
FINAL_CKPT_WEIGHT_FILENAMES: tuple[str, ...] = (
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",
)

#: native **全参 PP 分片**的权重文件模式（2026-09-22 T017 实测布局）：
#: ``<checkpoint>/rank_00000_tp_00_pp_00.pt`` + ``rank_00001_tp_00_pp_01.pt`` + ``manifest.json``。
#:
#: ⚠ ``manifest.json`` **只是分片布局元数据、不含任何内容哈希**——实测它在
#: ``step_50`` / ``step_100`` / ``final`` 三处**逐字节相同**，两跑之间也**逐字节相同**
#: ⇒ **绝不可**拿它（或"文件名+大小"）充当权重指纹：那会让两跑永远"指纹一致"，
#: 是**静默降级**。指纹必须覆盖分片里的**每一个权重字节**（见 extract_final_ckpt_sha256）。
PP_SHARD_GLOB = "rank_*_tp_*_pp_*.pt"


def _hf_safetensors_shards(checkpoint: Path) -> list[Path]:
    """HF 分片 safetensors 的**分片清单**（按名排序）；非分片布局返回空表。

    权威来源 = ``model.safetensors.index.json`` 的 ``weight_map``（§1.4 单一真相源，
    不靠"文件名看起来像分片"来猜）；仅有分片文件而缺 index 时才回落 glob，并在明细里显式说明。
    """
    names: list[str] = []
    index_file = checkpoint / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            payload = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            weight_map = payload.get("weight_map")
            if isinstance(weight_map, dict):
                names = sorted({str(v) for v in weight_map.values()})
    if not names:
        names = sorted(path.name for path in checkpoint.glob("model-*-of-*.safetensors"))
    return [checkpoint / name for name in names if (checkpoint / name).is_file()]


def _shard_set_fingerprint(
    shards: Sequence[Path],
    checkpoint: Path,
    prefix: str,
    label: str,
    how: str,
    source_prefix: str,
) -> tuple[str | None, str, str | None]:
    """**多文件权重指纹**：逐片 sha256 → 规范化清单再取 sha256（覆盖全部权重字节）。

    证据强度：与单文件 sha256 **同强度**（读的是每个分片的真实字节），
    **不是**"文件名/大小/manifest"那种弱指纹。读不到任一分片 ⇒ 取证缺口（``None``），
    **不得**退化成弱指纹（§2.3 fail-closed）。
    """
    digests: list[tuple[str, str, int]] = []
    for shard in shards:
        digest = _sha256(shard)
        if digest is None:
            return None, f"{shard} 读不到内容（sha256 失败）", None
        digests.append((shard.name, digest, shard.stat().st_size))
    combined = hashlib.sha256(
        "".join(f"{digest}  {name}\n" for name, digest, _ in digests).encode("utf-8")
    ).hexdigest()
    total_bytes = sum(size for _, _, size in digests)
    return (
        combined,
        f"{prefix} 的 **{label}**（{len(digests)} 片 / {total_bytes / 2**30:.1f} GiB，{how}）"
        f"组合 sha256={combined[:12]}…（逐片 sha256 的规范化清单再取 sha256 ⇒ 覆盖全部权重字节）",
        f"{source_prefix}:{checkpoint}",
    )


def extract_final_ckpt_sha256(
    output_dirs: Sequence[Path], optimizer_steps: int | None
) -> tuple[str | None, str, str | None]:
    """A4 的**独立通道**证据：末步 checkpoint 的权重文件内容 sha256。

    返回 ``(sha256 | None, 人读明细, 来源标识 | None)``（三件一起返回，调用方不可能
    只拿到指纹而不知道它取自哪个文件——§2.2 显式即防呆）。

    **"末步"的权威定义只有一个**（§1.4）：与 ``optimizer_steps`` 同名的那个 checkpoint
    （``final`` / ``checkpoint-<step>``）。取不到同名的就回落到
    :func:`find_checkpoint_dirs` 的"最新优先"排序结果（该函数已把 ``final`` 排最前），
    并在明细里**如实写明**这是回落，不假装它就是末步。

    ★ 为什么需要这条通道：loss 是**标量**，两跑 loss 相同（尤其都是 0）时它不提供
    任何鉴别力；权重文件指纹是**高维**的，能把"loss 相同但权重已分叉"这种危险组合
    暴露出来（AD1 §6.3 方案要素 5）。

    **两种形态都支持**（2026-09-22 补第二种）：① 单文件（peft ``adapter_model.safetensors`` /
    HF ``model.safetensors`` / native ``pytorch_model.bin``）；② **native 全参 PP 分片**
    （``rank_*_tp_*_pp_*.pt`` 多文件，见 :data:`PP_SHARD_GLOB`）——多文件时指纹是
    "逐片 sha256 的规范化清单再取 sha256"，覆盖全部权重字节，强度与①等价。
    """
    wanted = f"checkpoint-{optimizer_steps}" if optimizer_steps is not None else None
    candidates: list[Path] = []
    if wanted is not None:
        for directory in output_dirs:
            candidates.extend(
                path for path in directory.rglob(wanted) if path.is_dir()
            )
    fallback_used = False
    if not candidates:
        fallback_used = True
        candidates = find_checkpoint_dirs(output_dirs)
    if not candidates:
        return None, "没有任何 checkpoint 目录 ⇒ 独立通道无证据", None

    checkpoint = candidates[0]
    for name in FINAL_CKPT_WEIGHT_FILENAMES:
        weights = checkpoint / name
        if weights.is_file():
            digest = _sha256(weights)
            if digest is None:
                return None, f"{weights} 读不到内容（sha256 失败）", None
            source = f"file_sha256:{weights}"
            note = (
                f"末步 checkpoint {checkpoint.name} 的 {name}（sha256={digest[:12]}…）"
                if not fallback_used
                else f"**回落**到最新 checkpoint {checkpoint.name}（不是与 "
                f"optimizer_steps={optimizer_steps} 同名的那个）的 {name}"
                f"（sha256={digest[:12]}…）"
            )
            return digest, note, source
    # ── native 全参 PP 分片布局（多文件形态）────────────────────────────
    # 指纹 = **对"逐分片 sha256 + 分片名"的规范化清单再取一次 sha256**。
    # 强度论证（不许悄悄降级）：它覆盖了每个分片的**全部权重字节**，
    # 与单文件 sha256 是**同一强度的内容指纹**；不是"文件名/大小/manifest"这种弱指纹。
    # 代价：要读完全部分片（T017 实测 final/ 共 56.5 GB，宿主 sha256 ≈ 975 MB/s ⇒ ~1 min/跑）。
    prefix = (
        f"末步 checkpoint {checkpoint.name}"
        if not fallback_used
        else f"**回落**到最新 checkpoint {checkpoint.name}（不是与 "
        f"optimizer_steps={optimizer_steps} 同名的那个）"
    )
    shards = sorted(path for path in checkpoint.glob(PP_SHARD_GLOB) if path.is_file())
    if shards:
        return _shard_set_fingerprint(
            shards, checkpoint, prefix, "native 全参 PP 分片集合", f"`{PP_SHARD_GLOB}`",
            "pp_shard_set_sha256",
        )

    # ── HF **分片 safetensors**（ms-swift 全参布局；2026-09-22 T005/T020 实测）─────
    # 实测形态：`<ckpt>/model-00001-of-00004.safetensors` … `model-00004-of-00004.safetensors`
    # + `model.safetensors.index.json`（+ 123 GiB 的 DeepSpeed `global_step<N>/` 优化器状态）。
    # 旧实现只认单文件名 ⇒ 命中不了分片 ⇒ A4 指纹恒为 None（**即使补了第二跑也判不了**）。
    # 权威清单是 index.json 的 `weight_map`（§1.4 单一真相源）；缺 index 时才回落 glob。
    hf_shards = _hf_safetensors_shards(checkpoint)
    if hf_shards:
        return _shard_set_fingerprint(
            hf_shards, checkpoint, prefix, "HF 分片 safetensors 权重集合",
            "`model-*-of-*.safetensors`（清单取自 model.safetensors.index.json）",
            "hf_shard_set_sha256",
        )
    names = ", ".join(FINAL_CKPT_WEIGHT_FILENAMES)
    return (
        None,
        f"末步 checkpoint {checkpoint.name} 下没有已知权重文件（找过：{names}；"
        f"也未匹配 native 全参 PP 分片 `{PP_SHARD_GLOB}`）"
        " ⇒ 独立通道无证据",
        None,
    )


#: safetensors 头部长度上限（防呆：损坏/恶意文件不得让本脚本吃满内存）。
_SAFETENSORS_HEADER_LIMIT = 100 * 1024 * 1024


def _safetensors_index(path: Path) -> tuple[dict[str, tuple[int, int]], int] | None:
    """**纯 stdlib** 解析 safetensors 头部，返回 ``({张量名: (起始, 结束)}, 数据区起点)``。

    为什么不用 ``safetensors.safe_open``：collector 要在**没有 torch/safetensors 的宿主**
    上跑（228 宿主实测两者都没有）。判据需要的是"权重有没有变化 / checkpoint 能不能读"，
    而 safetensors 的格式是公开且极简的（8 字节小端头长 + JSON 头 + 原始数据），
    因此这里直接读头部并校验每个张量的 ``data_offsets`` 落在文件范围内。
    **语义与库版本一致**：库的 ``safe_open`` 也只是解析同一份头部。

    返回 ``None`` 表示"不是可解析的 safetensors 文件"。
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                return None
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len <= 0 or header_len > _SAFETENSORS_HEADER_LIMIT:
                return None
            header = handle.read(header_len)
            if len(header) != header_len:
                return None
    except OSError:
        return None
    try:
        doc = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    data_start = 8 + header_len
    index: dict[str, tuple[int, int]] = {}
    for name, spec in doc.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            return None
        offsets = spec.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            return None
        begin, end = offsets
        if not isinstance(begin, int) or not isinstance(end, int):
            return None
        if begin < 0 or end < begin or data_start + end > size:
            return None
        index[name] = (data_start + begin, data_start + end)
    return index, data_start


def _read_range(path: Path, begin: int, end: int) -> bytes | None:
    try:
        with path.open("rb") as handle:
            handle.seek(begin)
            return handle.read(end - begin)
    except OSError:
        return None


def _file_readable(path: Path) -> bool:
    """该文件对**当前进程**是否可读。

    为什么必须显式区分"读不到"与"文件坏"（228 实测踩过）：容器内以 root 身份训练，
    ms-swift/HF 落盘的 ``adapter_model.safetensors`` 是 ``root:0600``；宿主机上的
    collector 以普通用户运行时**打不开**它。若把 ``PermissionError`` 当成
    "checkpoint 无法重新加载"，就等于把"采集侧读不到"记成一次**训练失败**——
    这正是本包要根治的方向性错误。因此：不可读 ⇒ 取证缺口（返回 ``None``）。
    运行侧的正解是 runner 收尾时把产物放开读权限（见 ``run_matrix54.sh``）。
    """
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _safetensors_any_nonzero(path: Path, *, key_filter: str) -> bool | None:
    """指定关键字张量里是否有非零值（LoRA ``lora_B`` 初值为 0 ⇒ 非零即被更新过）。

    判据口径与 native 侧 ``_any_nonzero``（torch 张量 ``!= 0``）一致。实现按**字节**
    判零：本项目权重是 bf16/fp16/fp32，+0.0 的编码是全零字节，初值就是 +0.0。
    返回 ``None`` = 读不到 / 没有匹配张量（调用方按取证缺口处理，不得当成"未变化"）。
    """
    if not _file_readable(path):
        return None
    parsed = _safetensors_index(path)
    if parsed is None:
        return None
    index, _ = parsed
    keys = [key for key in index if key_filter in key.lower()]
    if not keys:
        return None
    for key in keys:
        begin, end = index[key]
        chunk = _read_range(path, begin, end)
        if chunk is None:
            return None
        if chunk.strip(b"\x00") != b"":
            return True
    return False


def extract_lora_norm_delta(output_dirs: Sequence[Path]) -> WeightEvidence:
    """A2-LoRA 的**正式判据**（§8.1 原文：「LoRA ⇒ ``lora_norm_delta ≠ 0``」）。

    为什么这条路径是必须的（本包要根治的环境性伪否）：``lora_norm_delta`` 由训练侧
    ``flow/progress_metrics.py`` 的 ``lora_norm_delta_event`` **每个训练步**算出并落进
    ``rank_metrics.rank_*.jsonl`` 的 ``metrics`` 里（键 ``lora_norm_delta``，同时有全局
    聚合口径 ``global_lora_norm_delta_mean``）。它是**纯 JSON**：不需要 torch，不需要
    safetensors，甚至不需要 checkpoint 还在——而 §8.1 把它定义为 LoRA 档的正式判据。

    旧实现的缺口（T010 真机实测）：只翻 checkpoint 文件（native 是 ``rank_*.pt``，
    需要 ``torch.load``），宿主无 torch ⇒ 直接返回 ``None`` ⇒ A2 判"缺少权重变化证据
    (fail-closed)"。这是**判据实现与判据语义不一致**：证据本来就在运行目录里躺着，
    却因为"采集机没有 torch"被判成训练失败。注意 §8.1 的 A2 判据**不含**任何
    "必须有基座模型"的要求——``base_model_root`` 只服务全参档（见
    :func:`extract_weight_changed` 的全参分支）。

    三态语义（**不放松**）：

    - 读到 finite 且非零的 Δ ⇒ ``True``（权重确实被更新过）；
    - 读到 finite 的 Δ 但**全部恰好为 0** ⇒ ``False``（权重未变化：这正是 §8.1
      用 ``≠ 0`` 要抓的事实，属**实质失败**，不是缺口）；
    - 只读到非有限 Δ、或压根没有 Δ 读数 ⇒ ``None``（不猜、不放行，按取证缺口处理）。
    """
    finite_values: list[float] = []
    first_nonzero: tuple[int, float] | None = None
    nonfinite = 0
    for directory in output_dirs:
        rows, _ = _rank_metric_steps(directory)
        for row in rows:
            raw = row.get("lora_norm_delta")
            if raw is None:
                # 全局聚合口径：单卡下与 lora_norm_delta 同值（ranks 只有 1 个），
                # 显式回落避免"键名取决于分支"的隐式契约（§2.2）。
                raw = row.get("global_lora_norm_delta_mean")
            # 全参档该键为 None（"指标不适用"），``is None`` 比对而非 ``not raw``（§2.2）。
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if not math.isfinite(value):
                nonfinite += 1
                continue
            finite_values.append(value)
            if value != 0.0 and first_nonzero is None:
                first_nonzero = (len(finite_values), value)
    total = len(finite_values) + nonfinite
    if total == 0:
        return WeightEvidence(None, None, "run 逐步指标里没有 lora_norm_delta 读数")
    if first_nonzero is not None:
        index, value = first_nonzero
        return WeightEvidence(
            True,
            WEIGHT_SOURCE_RUN_METRICS,
            f"第 {index} 步 lora_norm_delta={value:.6g} ≠ 0（共 {total} 步读数）",
        )
    if finite_values:
        return WeightEvidence(
            False,
            WEIGHT_SOURCE_RUN_METRICS,
            f"{len(finite_values)} 步 lora_norm_delta 全部恰好为 0 ⇒ 权重未被更新",
        )
    return WeightEvidence(
        None,
        None,
        f"{nonfinite} 步 lora_norm_delta 全部非有限（§8.1 要求变化量 finite）⇒ 不作变化断言",
    )


def extract_trainable_norm_delta(output_dirs: Sequence[Path]) -> WeightEvidence:
    """A2-**全参**的就地判据：run 自产的 ``trainable_norm_delta``（可训练参数 L2 变化）。

    为什么这条路径是必须的（本包要根治的第二条环境性伪否，T017 真机实测）：
    T017（9B·SFT·**全参**·native·pp=2）两跑各 100 步、``exit_code=0``、权重确实每步都在变
    （``global_trainable_norm_delta_mean`` 100/100 步非零），但 A2 仍被判
    「缺少权重变化证据」——因为全参分支只认 ``checkpoint`` 与**基座**的 safetensors 比对，
    而 native 全参落的是 ``rank_*_tp_*_pp_*.pt``（torch.save 归档），宿主又无 torch。
    **证据本来就在运行目录里躺着**（纯 JSON），却因"采集机没有 torch"被判成取证缺口。

    ★ 证据强度（**不许悄悄降级**，与 §8.1 已接受的 LoRA 正式判据逐条对齐）：
      · 同源：与 ``lora_norm_delta`` 出自**同一个** ``training_norm_event()`` 单一真相源，
        只是按 tuner_type 取模式感知的键名（源码头显式写明"lora 模式键名与数值语义逐字不变"）；
      · 同强度：都是"同一步前后、对**真实参数张量**求 L2 范数再相减"，不是梯度/代理指标；
      · 自证口径：随行带 ``norm_metric=trainable_parameter_l2_norm`` 与
        ``grad_count_metric=grad_populated_trainable_params``；
      · **边界（如实写）**：它证明"可训练参数的**整体 L2 范数**发生变化"，
        **不**逐一证明每层都变；也**不**替代 A4 的独立通道（那需要权重指纹）。

    三态语义（**与 LoRA 路径逐条一致，不放松**）：

    - finite 且非零 ⇒ ``True``（权重确实被更新过）；
    - finite 但**全部恰好为 0** ⇒ ``False``（权重未变化 = **实质失败**，不是缺口）；
    - 只有非有限值、或压根没有读数 ⇒ ``None``（不猜、不放行，按取证缺口处理）。
    """
    finite_values: list[float] = []
    first_nonzero: tuple[int, float] | None = None
    nonfinite = 0
    for directory in output_dirs:
        rows, _ = _rank_metric_steps(directory)
        for row in rows:
            raw = row.get("trainable_norm_delta")
            if raw is None:
                # 全局聚合口径：单卡下与 trainable_norm_delta 同值（ranks 只有 1 个）。
                # 显式回落，避免"键名取决于分支"的隐式契约（§2.2）。
                raw = row.get("global_trainable_norm_delta_mean")
            # LoRA 档该键为 None（"指标不适用"）⇒ ``is None`` 比对而非 ``not raw``（§2.2）。
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if not math.isfinite(value):
                nonfinite += 1
                continue
            finite_values.append(value)
            if value != 0.0 and first_nonzero is None:
                first_nonzero = (len(finite_values), value)
    total = len(finite_values) + nonfinite
    if total == 0:
        return WeightEvidence(None, None, "run 逐步指标里没有 trainable_norm_delta 读数")
    if first_nonzero is not None:
        index, value = first_nonzero
        return WeightEvidence(
            True,
            WEIGHT_SOURCE_TRAINABLE_NORM_METRIC,
            f"第 {index} 步 trainable_norm_delta={value:.6g} ≠ 0（共 {total} 步读数）",
        )
    if finite_values:
        return WeightEvidence(
            False,
            WEIGHT_SOURCE_TRAINABLE_NORM_METRIC,
            f"{len(finite_values)} 步 trainable_norm_delta 全部恰好为 0 ⇒ 可训练参数未被更新",
        )
    return WeightEvidence(
        None,
        None,
        f"{nonfinite} 步 trainable_norm_delta 全部非有限（§8.1 要求变化量 finite）⇒ 不作变化断言",
    )


def extract_weight_changed(
    output_dirs: Sequence[Path], tuner_type: str, base_model_dir: Path | None = None
) -> WeightEvidence:
    """抽取"权重是否真变化"证据，返回**判据值 + 来源 + 明细**（§2.2 显式即防呆）。

    两种后端的**判据语义完全相同**（§8.1），变的只是证据落点与可用性：

    - **LoRA**（§8.1：``lora_norm_delta ≠ 0``）：**先**读 run 自产的逐步指标
      （:func:`extract_lora_norm_delta`，纯 stdlib、**与宿主有没有 torch 无关**，
      §6 环境可复现：同一份 run 产物在任何采集机上结论一致）；拿不到才回落到
      checkpoint 里的 ``lora_b`` 字节/torch 校验（:func:`_any_nonzero` /
      :func:`_safetensors_any_nonzero`）作为**等价兜底**。**不需要**
      ``base_model_dir``——旧实现把它当成"没给就没有权重证据"的因素之一，是判据
      实现与 §8.1 语义不一致的具体表现。
    - **全参**（§8.1：可训练参数 ≈ 全参 **且** 基座某层权重 L2 delta ≠ 0）：
      **先**读 run 自产的 ``trainable_norm_delta``（:func:`extract_trainable_norm_delta`，
      与 LoRA 的 ``lora_norm_delta`` **同源同强度**、纯 stdlib、不需要 torch/基座）；
      拿不到才回落到"终态权重 vs 基座权重"的 safetensors 字节比对
      （:func:`_full_weights_differ`，需要 ``base_model_dir`` + safetensors 形态）。
      两条都拿不到 ⇒ 取证缺口（``None``）——**不放松**，也不拿 LoRA 的 Δ 指标冒充。

    抽取不到返回 ``value is None`` → A2 按**取证缺口**处理（不可判定，而不是判训练失败）。

    ★ 兜底路径的失败**必须记账后继续**（指挥官裁定 (b)，2026-09-21；见下）：旧实现
      **裸捕获** ``torch.load`` 异常并 ``continue``——失败原因被静默吞掉，台账上只剩
      "checkpoint 与 run 逐步指标里都没有权重变化证据"，读的人无法区分
      "文件坏了" / "torch 版本不兼容" / "压根没有这个文件"（§2.2 显式即防呆）。
      为什么**不能**把"载不进 ckpt"记成 ``WeightEvidence(False, ...)``：那等于断言
      "权重未变化"，即"训练没生效"——**最坏的假失败**（环境/兼容性问题被记成训练失败，
      方向性错误）。兜底路径的**唯一**职责是补证据，不是制造结论。
    """
    torch = _require_torch()
    #: 兜底路径的**失败台账**：逐条累积原因，最终并入 detail（**绝不静默丢弃**）。
    #: 只在"真的尝试过兜底且失败"时才有内容——它解释的是"为什么没拿到兜底证据"。
    fallback_notes: list[str] = []
    if tuner_type == "lora":
        metrics_evidence = extract_lora_norm_delta(output_dirs)
        if metrics_evidence.value is not None:
            return metrics_evidence
        # 正式判据（lora_norm_delta）已不可得 ⇒ 把原因记进台账，继续走兜底路径。
        fallback_notes.append(f"A2 正式判据 lora_norm_delta 不可得（{metrics_evidence.detail}）")
    else:
        # ★ 全参的**就地正式判据**（与 LoRA 同源同强度，2026-09-22 补）：先读 run 自产的
        #   `trainable_norm_delta`；拿得到就**不需要** torch、也不需要基座目录。
        full_metrics_evidence = extract_trainable_norm_delta(output_dirs)
        if full_metrics_evidence.value is not None:
            return full_metrics_evidence
        fallback_notes.append(
            f"A2 全参就地判据 trainable_norm_delta 不可得（{full_metrics_evidence.detail}）"
        )
    if torch is None:
        fallback_notes.append("宿主无 torch ⇒ checkpoint 兜底路径（rank_*.pt 反序列化）不可用")

    for checkpoint in find_checkpoint_dirs(output_dirs):
        if tuner_type == "lora":
            # native：单个 rank pt 里的 lora_state_dict
            for pt in sorted(checkpoint.glob("rank_*.pt")):
                if torch is None:
                    break
                try:
                    payload = torch.load(pt, map_location="cpu", weights_only=False)
                except (RuntimeError, OSError, AttributeError) as exc:
                    # ★ 记账后继续（裁定 (b)）：**不**把"载不进"当成"权重未变化"。
                    # 异常类型与消息原样入台账（§2.2）；继续找别的 checkpoint / 兜底源。
                    fallback_notes.append(
                        f"兜底路径 torch.load({pt.name}) 失败：{type(exc).__name__}: {exc}"
                    )
                    continue
                if not isinstance(payload, dict):
                    fallback_notes.append(f"兜底路径 {pt.name} 反序列化结果不是 dict，跳过")
                    continue
                lora_state = payload.get("lora_state_dict")
                if isinstance(lora_state, dict):
                    subset = {
                        key: value
                        for key, value in lora_state.items()
                        if "lora_b" in str(key).lower()
                    }
                    result = _any_nonzero(subset)
                    if result is not None:
                        return WeightEvidence(
                            result,
                            WEIGHT_SOURCE_NATIVE_LORA_TORCH,
                            f"{pt.name} 的 lora_b 张量{'存在非零值' if result else '全部为零'}",
                        )
            # msswift / peft：adapter_model.safetensors 里的 lora_B
            for weights in sorted(checkpoint.glob("adapter_model.safetensors")):
                result = _safetensors_any_nonzero(weights, key_filter="lora_b")
                if result is not None:
                    return WeightEvidence(
                        result,
                        WEIGHT_SOURCE_ADAPTER_SAFETENSORS,
                        f"{weights.name} 的 lora_b 张量{'存在非零值' if result else '全部为零'}",
                    )
            continue
        # 全参：终态权重与基座权重做抽样比较
        result = _full_weights_differ(checkpoint, base_model_dir)
        if result is not None:
            return WeightEvidence(
                result,
                WEIGHT_SOURCE_BASE_COMPARE,
                f"{checkpoint.name} 与基座 {Path(base_model_dir).name} 的共享权重"
                f"{'不同' if result else '逐字节相同'}",
            )
    if tuner_type != "lora":
        if base_model_dir is None or not Path(base_model_dir).is_dir():
            return WeightEvidence(
                None,
                None,
                "全参判据需要 --base-model-root 做基座权重比对，本次未提供合法目录",
            )
    # ★ fail-closed **保持不变**：所有证据源都不可得 ⇒ 仍是取证缺口（``value is None``）。
    # 兜底路径的失败原因在此并入 detail（记账不吞掉，§2.2 显式即防呆）——台账从此能区分
    # "没有探测文件" / "文件坏了" / "torch 版本不兼容"，而不是一句笼统的"都没有证据"。
    detail = "checkpoint 与 run 逐步指标里都没有权重变化证据"
    if fallback_notes:
        detail = detail + "；兜底路径台账：" + "；".join(fallback_notes)
    return WeightEvidence(None, None, detail)


def _full_weights_differ(checkpoint_dir: Path, base_model_dir: Path | None) -> bool | None:
    """全参：checkpoint 与基座模型权重抽样比较（不同 → 已更新）。

    ``base_model_dir`` 是**唯一来源**（由调用方从 ``--base-model-root`` 解析后传入，
    见 :func:`_base_model_dir`）。此处**不再**读进程环境：历史上 ``None`` 会回退到
    环境变量 ``GRASPO_BASE_MODEL_FOR_COMPARE``，那是一条环境变量 fallback 链
    （§1.4/§7.1 明令禁止的"双真相源"）——同一个"权重是否变化"的结论会因该变量
    是否设置而不同，且变量不在 config 备份里、无法复现。

    ``None`` 的语义**保持原样**：给不出基座目录 ⇒ 返回 ``None``，A2 的权重证据按
    **取证缺口**处理（不可判定），不猜也不放松（§3 退路与防线之分）。

    **判据语义不变**：共享键里任何一个张量字节不同即"已更新"。
    """
    if base_model_dir is None or not Path(base_model_dir).is_dir():
        return None
    final_files = sorted(checkpoint_dir.glob("*.safetensors"))
    base_files = sorted(Path(base_model_dir).glob("*.safetensors"))
    if not final_files or not base_files:
        return None
    # 不可读（权限/IO）⇒ 取证缺口，不得当成"权重没变化"。
    if not (_file_readable(final_files[0]) and _file_readable(base_files[0])):
        return None
    final_parsed = _safetensors_index(final_files[0])
    base_parsed = _safetensors_index(base_files[0])
    if final_parsed is None or base_parsed is None:
        return None
    final_index, _ = final_parsed
    base_index, _ = base_parsed
    shared = [key for key in final_index if key in base_index][:8]
    if not shared:
        return None
    for key in shared:
        left = _read_range(final_files[0], *final_index[key])
        right = _read_range(base_files[0], *base_index[key])
        if left is None or right is None:
            return None
        if len(left) != len(right):
            return True
        if left != right:
            return True
    return False


#: 容器内 torch 探测证据的**文件名**与 **schema 标识**（唯一真相源）。
#:
#: 为什么是"证据文件"而不是"改 runner 来跑校验"：判据的**判定逻辑**必须集中在本
#: 收集器里（§1.4 单一真相源），runner 只负责"在**有 torch 的地方**把探测结果落盘"，
#: 不承担判定。契约全文（供 runner 侧工作包实现）见工位 ``report.md §⑦``。
TORCH_PROBE_FILENAME = "torch_probe.json"
TORCH_PROBE_SCHEMA = "graspo.torch_probe.v1"


def _sha256(path: Path) -> str | None:
    """文件内容的 sha256（十六进制小写）；读不到返回 ``None``。"""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _torch_zip_intact(path: Path) -> str | None:
    """**纯 stdlib** 校验 torch(>=1.6) 归档：能否当 zip 打开 + 成员 CRC 是否自洽。

    返回 ``None`` = 结构完整；返回字符串 = 损坏原因。

    ★ **这是弱证据**：torch 的 ``.pt`` 是 zip 容器（``<prefix>/data.pkl`` +
    ``<prefix>/data/N`` 原始张量）。能当 zip 打开且 CRC 自洽，只证明"文件完整、
    归档结构自洽"，**不证明 torch 能反序列化出张量**（pickle 层可能引用了不存在的
    类、storage 元数据可能不自洽）。因此调用方**不得**据此放行 A3——它的用途是
    把"取证缺口"里的两种情况分开：①文件真的坏了（⇒ **实质失败**，方向与
    ms-swift 的 safetensors 分支一致）；②文件好但没 torch 验不了（⇒ 取证缺口）。
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.endswith("data.pkl") for name in names):
                return "归档里没有 data.pkl（不是 torch>=1.6 归档）"
            broken = archive.testzip()
            if broken is not None:
                return f"成员 CRC 校验失败：{broken}"
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        return f"不是可解析的 zip 归档（{type(exc).__name__}）"
    return None


def read_torch_probe(run_dir: Path | None) -> ReloadEvidence | None:
    """读**容器内**的 torch 探测证据 ``<run_dir>/torch_probe.json``（§2.1 契约即防呆）。

    为什么需要它（本包要根治的环境性伪否，A3 侧）：native 的 ``rank_*.pt`` 是
    torch 私有归档，"能不能被重新加载"只有 torch 自己说了算。宿主 228 实测无 torch，
    但**运行这次训练的镜像里有**（``graspo-msswift:4.5.3``）⇒ 证据明明可得。

    契约（runner 侧接口，本包不改 runner；见 ``report.md §⑦``）::

        {"schema": "graspo.torch_probe.v1",
         "torch": "2.11.0+cu130",
         "all_ok": true,
         "checked": [{"relpath": "final/rank_00000_tp_00_pp_00.pt",
                      "ok": true, "sha256": "<64 hex>", "tensors": 882}]}

    校验规则（任一条不满足 ⇒ **不采信**，按"没有证据文件"处理，绝不放行）：

    1. ``schema`` 必须逐字等于 :data:`TORCH_PROBE_SCHEMA`（防呆：schema 变了说明
       语义变了，旧文件不得被新代码当成同一份证据）；
    2. ``all_ok`` 必须是 ``bool``；
    3. ``checked`` 非空，且每项的 ``relpath`` 相对 ``run_dir`` 解析后**必须真实存在**；
    4. ★ 每项的 ``sha256`` 必须等于该文件**当前**内容的 sha256。这条是**防陈旧证据**
       的防呆装置：同一档第二次跑会覆盖 ``final/`` 下的权重，若沿用上一次的探测结论，
       就等于用旧 run 的证据给新 run 作证（228 实测过跨 run ckpt 错配）。

    返回 ``None`` = **没有**探测证据文件（正常退路：宿主自带的 torch 路径）。返回
    ``value is None`` 的 :class:`ReloadEvidence` = 文件在但**不可采信**（原因写进
    ``detail``，台账可见，不静默丢弃）。
    """
    if run_dir is None:
        return None
    probe_path = Path(run_dir) / TORCH_PROBE_FILENAME
    if not probe_path.is_file():
        return None
    payload = _read_json(probe_path)
    if not isinstance(payload, dict):
        return ReloadEvidence(None, None, f"{TORCH_PROBE_FILENAME} 不是合法 JSON 对象 ⇒ 不采信")
    name = TORCH_PROBE_FILENAME
    schema = payload.get("schema")
    if schema != TORCH_PROBE_SCHEMA:
        return ReloadEvidence(
            None, None, f"{name} 的 schema={schema!r} ≠ {TORCH_PROBE_SCHEMA} ⇒ 不采信"
        )
    checked = payload.get("checked")
    if not isinstance(checked, list) or not checked:
        return ReloadEvidence(None, None, f"{name} 的 checked 为空 ⇒ 不采信")
    all_ok = payload.get("all_ok")
    if not isinstance(all_ok, bool):
        return ReloadEvidence(None, None, f"{name} 的 all_ok 不是布尔 ⇒ 不采信")
    verified = 0
    for item in checked:
        if not isinstance(item, dict):
            return ReloadEvidence(None, None, f"{name} 的 checked 项不是对象 ⇒ 不采信")
        relpath = item.get("relpath")
        digest = item.get("sha256")
        if not isinstance(relpath, str) or not isinstance(digest, str):
            return ReloadEvidence(None, None, f"{name} 的 checked 项缺 relpath/sha256 ⇒ 不采信")
        target = Path(run_dir) / relpath
        if not target.is_file():
            return ReloadEvidence(None, None, f"{name} 指向的文件不存在：{relpath} ⇒ 不采信")
        actual = _sha256(target)
        if actual is None:
            return ReloadEvidence(None, None, f"探测证据指向的文件读不到：{relpath} ⇒ 不采信")
        if actual != digest.lower():
            return ReloadEvidence(
                None,
                None,
                "探测证据与磁盘内容不一致（sha256 不符，疑似上一次 run 的证据）："
                f"{relpath} ⇒ 不采信",
            )
        verified += 1
    torch_version = payload.get("torch")
    if all_ok:
        return ReloadEvidence(
            True,
            RELOAD_SOURCE_CONTAINER_PROBE,
            f"容器内 torch.load 探测通过（{verified} 个文件，torch={torch_version}）",
        )
    return ReloadEvidence(
        False,
        RELOAD_SOURCE_CONTAINER_PROBE,
        f"容器内 torch.load 探测失败（{verified} 个文件已核对 sha256，torch={torch_version}）",
    )


def extract_checkpoint_reloadable(
    output_dirs: Sequence[Path], *, run_dir: Path | None = None
) -> ReloadEvidence:
    """A3 证据：checkpoint 能否被重新加载（结构化校验 + 反序列化）。

    两种布局走同一条判据：

      - native：``final/manifest.json`` + ``rank_*.pt``；
      - ms-swift / peft：``*.safetensors``（stdlib 解析头部 + 校验 data_offsets）。

    判据语义不变（结构完整 + 可反序列化）；**没有**因为后端不同而放松。native 侧的
    证据按**强度降序**取，每一档都在返回值里显式标注来源（§2.2）：

    1. **容器探测证据**（:func:`read_torch_probe`，强）：镜像内有 torch ⇒ 虽然宿主
       228 无 torch，证据仍然可得。这是本包要根治的"明明能取到却不取"。
    2. **宿主 torch 反序列化**（强）：宿主自带 torch 时直接 ``torch.load``。
    3. **stdlib 归档结构 / CRC**（**弱**）：宿主无 torch 时的降级检查。它**不放行**
       A3（结构完整 ≠ 可反序列化），但能把"文件真的坏了"判成**实质失败**，并把
       "文件好但验不了"的取证缺口在台账里说清楚。
    """
    probe = read_torch_probe(run_dir)
    if probe is not None and probe.value is not None:
        return probe
    probe_note = probe.detail if probe is not None else ""

    checkpoints = find_checkpoint_dirs(output_dirs)
    if not checkpoints:
        return ReloadEvidence(None, None, "没有发现任何 checkpoint 目录")
    checked = False
    structural_note: str | None = None
    for checkpoint in checkpoints:
        # native：manifest.json + rank_*.pt 能否被 torch 反序列化
        if (checkpoint / "manifest.json").exists():
            torch = _require_torch()
            rank_files = sorted(checkpoint.glob("rank_*.pt"))
            if not rank_files:
                continue
            if not _file_readable(rank_files[0]):
                # 权限/IO 不可读 ⇒ 取证缺口（继续找别的 checkpoint），不得判"重载失败"。
                continue
            if torch is None:
                # 宿主无 torch ⇒ 不直接放弃：先做**弱**的 stdlib 校验，把两种情况分开。
                broken = _torch_zip_intact(rank_files[0])
                if broken is not None:
                    return ReloadEvidence(
                        False,
                        RELOAD_SOURCE_STRUCTURAL_ONLY,
                        f"{rank_files[0].name} 归档结构已损坏：{broken}",
                    )
                # 探测证据被拒的原因**必须**跟着一起落台账（§2.2 显式即防呆）：旧写法
                # 只在"最终 return"处拼接 ``probe_note``，而这里一旦置上
                # ``structural_note`` 就会走那条提前 return ⇒ 拒绝原因被静默丢弃，
                # 台账上只剩"宿主无 torch"，读的人无法区分"没有探测文件"与"探测文件
                # 不可采信"（本包要修的正是这种不可区分）。
                structural_note = (
                    f"宿主无 torch，无法反序列化 native rank_*.pt；"
                    f"{rank_files[0].name} 归档结构/CRC 完整（弱证据 "
                    f"{RELOAD_SOURCE_STRUCTURAL_ONLY}，仅证明文件完整、**不**证明可反序列化）"
                )
                if probe_note:
                    structural_note = f"{probe_note}；{structural_note}"
                continue
            try:
                torch.load(rank_files[0], map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001 — 反序列化异常族无法穷举，见下
                # 为什么捕获**全部** Exception 而不再枚举具体类型（**真 bug 修复**）：
                # ``torch.load`` 对损坏的 checkpoint 抛的是 ``pickle.UnpicklingError``
                # （真机实测：``_pickle.UnpicklingError: could not find MARK``），
                # 旧实现只捕 ``RuntimeError / OSError / AttributeError`` ⇒ 异常一路冒到
                # 顶层，采集器 **rc≠0 崩掉**，而不是把这一档记成结论。
                # 反序列化失败在不同 torch 版本 / 不同损坏形态下会落到
                # ``UnpicklingError``、``EOFError``、``ValueError``、``KeyError``、
                # ``zipfile.BadZipFile``、``RuntimeError``、``MemoryError`` 等**互不派生**
                # 的类型上（``pickle.UnpicklingError`` 直接派生自 ``Exception``），枚举法
                # 必然漏；漏掉一个就是把"文件坏了"升级成"采集器崩溃"（§2.3 边界校验）。
                # 这里**不吞**异常、也**不**降级成 ``None``：返回值恒为 ``False``
                # ⇒ A3 判"checkpoint 无法重新加载"（**实质失败**，方向同 ms-swift 的
                # safetensors 分支），异常类型与消息原样写进 detail（§2.2 显式即防呆）。
                # ``KeyboardInterrupt`` / ``SystemExit`` 派生自 ``BaseException``，不受影响。
                return ReloadEvidence(
                    False,
                    RELOAD_SOURCE_HOST_TORCH,
                    f"宿主 torch.load({rank_files[0].name}) 反序列化失败（checkpoint 损坏）："
                    f"{type(exc).__name__}: {exc}",
                )
            return ReloadEvidence(
                True,
                RELOAD_SOURCE_HOST_TORCH,
                f"宿主 torch.load({rank_files[0].name}) 反序列化成功",
            )
        # peft / HF（含 ms-swift 的 checkpoint-<step>）：safetensors 头部能否解析
        for weights in sorted(checkpoint.glob("*.safetensors")):
            if not _file_readable(weights):
                continue  # 取证缺口：采集侧读不到，不是 checkpoint 坏
            checked = True
            parsed = _safetensors_index(weights)
            if parsed is None:
                # 能读但头部不可解析 ⇒ 这才是"checkpoint 无法重新加载"
                return ReloadEvidence(
                    False,
                    RELOAD_SOURCE_SAFETENSORS_HEADER,
                    f"{weights.name} 能被读取但头部不可解析",
                )
            index, _ = parsed
            if not index:
                return ReloadEvidence(
                    False, RELOAD_SOURCE_SAFETENSORS_HEADER, f"{weights.name} 头部里没有任何张量"
                )
            return ReloadEvidence(
                True,
                RELOAD_SOURCE_SAFETENSORS_HEADER,
                f"{weights.name} 头部与 data_offsets 校验通过（{len(index)} 个张量）",
            )
    if structural_note is not None:
        return ReloadEvidence(None, RELOAD_SOURCE_STRUCTURAL_ONLY, structural_note)
    if checked:
        return ReloadEvidence(
            False, RELOAD_SOURCE_SAFETENSORS_HEADER, "可读的 safetensors 里没有可解析的"
        )
    detail = "没有可读的 checkpoint（不可读或不存在）"
    if probe_note:
        detail = f"{detail}；{probe_note}"
    return ReloadEvidence(None, None, detail)


def extract_artifacts(run_dir: Path, output_dirs: Sequence[Path], log_text: str) -> dict[str, bool]:
    """A5 四件套产物：config_backup / training_log / checkpoint / metrics。

    **契约覆盖两种布局，判据语义一致**（同一件产物在两种后端下的等价落点）：

    | 件 | native | ms-swift | 语义（判据不放松） |
    |---|---|---|---|
    | config_backup | ``config.yaml``（graspo 配置备份）
    | ``args.json``（ms-swift **已解析生效**的训练参数） | 本次运行的配置被落盘、事后可复现 |
    | training_log | ``training.log`` / ``train.log``
    | ``logging.jsonl``（逐步日志） | 逐步训练日志落盘 |
    | checkpoint | ``final/`` | ``checkpoint-<step>/`` | 可恢复 checkpoint 落盘 |
    | metrics | ``events.jsonl`` / ``rank_metrics.*.jsonl`` / ``trainer_state.json``
    | ``logging.jsonl`` / ``events.out.tfevents.*``
    | 运行指标（逐步 loss/grad_norm 等）落盘 |

    容器 ``stdout.log`` 仍可作 training_log 的兜底（运行器一定产出它）。
    """
    artifacts = {name: False for name in _judge.REQUIRED_ARTIFACTS}
    for directory in output_dirs:
        if not artifacts["config_backup"]:
            artifacts["config_backup"] = (directory / "config.yaml").exists() or (
                directory / "args.json"
            ).exists()
        if not artifacts["training_log"]:
            artifacts["training_log"] = (
                any(directory.rglob("training.log"))
                or any(directory.rglob("train.log"))
                or (directory / "logging.jsonl").exists()
            )
        if not artifacts["metrics"]:
            artifacts["metrics"] = (
                any(directory.rglob("events.jsonl"))
                or any(directory.rglob("rank_metrics.rank_*.jsonl"))
                or any(directory.rglob("trainer_state.json"))
                or (directory / "logging.jsonl").exists()
                or any(directory.rglob("events.out.tfevents.*"))
            )
    # checkpoint 走统一的发现逻辑（唯一真相源），与 A2/A3 用的是同一份判据。
    artifacts["checkpoint"] = bool(find_checkpoint_dirs(output_dirs))
    # 运行器级日志也算训练日志证据（容器 stdout 一定存在）
    if not artifacts["training_log"] and (run_dir / "stdout.log").exists():
        artifacts["training_log"] = True
    if not artifacts["metrics"] and log_text:
        artifacts["metrics"] = bool(_LOSS_ANY.search(log_text))
    return artifacts


def extract_min_optimizer_steps(tier: dict[str, Any]) -> int | None:
    """从清单的 ``tiers[*].acceptance.formal_gate.min_optimizer_steps`` 取 A2 门槛。

    **单一真相源（§1.4）**：门槛的权威位置是清单，不是代码常量。本函数**只做取值**，
    不做合法性裁决——非法值（``<=0``/非整数/类型错）由判定层
    （``result_judge.resolve_min_optimizer_steps``）**fail-closed 判否**，
    采集层不得把它静默换成缺省值（那正是缺陷①的老病根：把"配置说的"换成"代码写死的"）。

    返回 ``None`` 只在两种情形：① 清单确实没给该键；② 清单结构缺失/形状不对。
    两者都 ⇒ 判定器回落到缺省门槛 5（**不放宽**）。
    """
    acceptance = tier.get("acceptance")
    if not isinstance(acceptance, dict):
        return None
    gate = acceptance.get("formal_gate")
    if not isinstance(gate, dict) or "min_optimizer_steps" not in gate:
        return None
    return gate["min_optimizer_steps"]


def extract_expected_optimizer_steps(tier: dict[str, Any]) -> tuple[int | None, int | None]:
    """从清单取该档的**步数上限**：``(每 epoch 上限, 整个 run 上限)``。

    **单一真相源（§1.4）**：两个上限的权威位置是清单
    ``tiers[*].expected_optimizer_steps_per_epoch`` / ``..._reachable``
    （由 ``tests/e2e/generate_matrix.py`` 的 ``expected_optimizer_steps_per_epoch`` /
    ``expected_optimizer_steps_reachable`` **一处**算出——见 AF1 §⑥ 方案 S1/S2）。

    与 :func:`extract_min_optimizer_steps` 同理，本函数**只做取值**，不做合法性裁决：
    非整数 / ``< 1`` 的值由判定层
    （``result_judge.resolve_step_gate_applicability``）**视为"没有结构性证据"**
    ⇒ 门槛照常硬判（fail-closed），采集层**不得**把它静默换成某个猜出来的上限。

    返回 ``None`` 表示"清单没给 / 形状不对" ⇒ 判定层退回原样硬判。老清单（本字段引入前
    生成）与新清单在**行为上等价**：都不开第三态 ⇒ **不放宽任何既有档位**。
    """
    per_epoch = tier.get("expected_optimizer_steps_per_epoch")
    reachable = tier.get("expected_optimizer_steps_reachable")
    return (
        per_epoch if isinstance(per_epoch, int) and not isinstance(per_epoch, bool) else None,
        reachable if isinstance(reachable, int) and not isinstance(reachable, bool) else None,
    )


def extract_run_first_timestamp(run_dir: Path) -> str | None:
    """该 run **首条带 ``timestamp`` 的记录**（ISO8601）——A4 判定"两次独立运行"的
    **内容级**身份证据（2026-09-22 裁定 2）。

    为什么必须有一条**内容级**证据：``run_root`` 与目录 inode 都只是"位置"，
    把同一份产物**拷到另一个路径**就能同时满足 ⇒ 会被误判成"两次独立运行"。
    而产物里记录的时间戳是训练当时写下的，拷贝不会改变它 ⇒ 拷贝场景下两端相同，
    判据层据此**拒绝**给独立性加分（见 ``_judge._a4_independence_signals``）。

    取不到 ⇒ ``None``（**不猜**；判据层会因此少一条证据，可能继续 fail-closed）。
    """
    for events in sorted(run_dir.rglob("rank_metrics.rank_*.jsonl")):
        try:
            text = events.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            stamp = payload.get("timestamp") if isinstance(payload, dict) else None
            if isinstance(stamp, str) and stamp:
                return stamp
    return None


def collect_run(
    run_dir: Path,
    tier_id: str,
    tuner_type: str,
    base_model_dir: Path | None = None,
    min_optimizer_steps: int | None = None,
    backend: str | None = None,
    cards: int | None = None,
    algorithm: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    expected_optimizer_steps_per_epoch: int | None = None,
    expected_optimizer_steps_reachable: int | None = None,
) -> tuple[Any, SeriesEvidence]:
    """把一个运行目录抽成 ``RunEvidence`` + 读数口径自证（``SeriesEvidence``）。

    ``min_optimizer_steps`` 由调用方从清单读出（见 :func:`extract_min_optimizer_steps`），
    缺省 ``None`` = "清单未提供" ⇒ 判定器用缺省门槛（**不放松**）。

    ``expected_optimizer_steps_per_epoch`` / ``expected_optimizer_steps_reachable``
    同样由调用方从清单读出（见 :func:`extract_expected_optimizer_steps`），缺省 ``None``
    = "清单未提供" ⇒ 判定器**不开第三态**、门槛照常硬判（fail-closed，**不放宽**）。

    ``backend`` / ``cards`` / ``algorithm`` / ``model`` / ``mode`` 是 **A4 分档标定**的
    档族坐标（§1.4；J2/P1 扩到五维）：容差由档族唯一决定，因此档族必须随证据一起走。
    缺省 ``None`` = 调用方没给 ⇒ 判定器按**只降维**的回落序取档族；连最泛化档族都没有
    ⇒ A4 fail-closed。**算法名不做归一**（透传原值，由判定器的 ``normalize_a4_algorithm``
    统一映射 ``GRASPO``→``GRPO``——单一真相源）。
    """
    exit_code: int | None = None
    exit_path = run_dir / "exit_code"
    if exit_path.exists():
        raw = _read_text(exit_path).strip()
        if raw.lstrip("-").isdigit():
            exit_code = int(raw)
    log_text = _read_text(run_dir / "stdout.log")
    output_dirs = find_output_dirs(run_dir, tier_id)
    series = extract_steps_and_series(output_dirs, log_text)
    timed_out = exit_code == 124 or bool(re.search(r"⏰|timeout: sending signal", log_text))
    weight_evidence = extract_weight_changed(output_dirs, tuner_type, base_model_dir)
    reload_evidence = extract_checkpoint_reloadable(output_dirs, run_dir=run_dir)
    # A4 独立通道（五要素第 5 条）：末步权重文件指纹。取不到就是 None（如实标注），
    # 由判定器决定"单通道"如何表述——采集层**不**替它判通过。
    ckpt_sha256, ckpt_sha256_detail, ckpt_sha256_source = extract_final_ckpt_sha256(
        output_dirs, series.steps
    )
    # ── A4 形态①的"独立性"输入（2026-09-22 裁定 2）────────────────────────
    # 位置级两条（运行根 / 产物根文件系统身份）+ 内容级一条（首条记录时间戳）。
    identity_target = output_dirs[0] if output_dirs else run_dir
    try:
        stat = identity_target.stat()
        output_identity: str | None = f"{stat.st_dev}:{stat.st_ino}"
    except OSError:
        output_identity = None
    # ── ★ P0-1（2026-09-22 复核，最高优先）：跳过计数**必须跨全部 rank**取 MAX ──────
    #   为什么放在这里而不是序列抽取里：`skipped_nonfinite` 是**逐 rank 局部**读数，
    #   而 `_rank_metric_steps` 只读 `rank_metrics.rank_00000.jsonl` ⇒ rank1 单独跳过、
    #   rank0 没跳时读到 0 ⇒ **A7 假通过**（一次权重/LR 已分叉的运行会被记成 ✅ 可用）。
    #   这里直接对**产物根**遍历全部 `rank_metrics.rank_*.jsonl` 取 MAX，与序列来源无关。
    _skip_count, _skip_per_rank, _skip_detail = extract_skipped_nonfinite_all_ranks(
        output_dirs[0] if output_dirs else run_dir
    )
    if _skip_count is not None:
        nonfinite_skips_value: int | None = _skip_count
        nonfinite_skips_source_value = "run_metrics:all_ranks_max"
        if len(set(_skip_per_rank.values())) > 1:
            nonfinite_skips_source_value = "run_metrics:all_ranks_max（ranks 不一致）"
    else:
        nonfinite_skips_value = series.nonfinite_skips
        nonfinite_skips_source_value = series.nonfinite_skips_source
    # ── ★ A7 三分类之 (A)（2026-09-22 指挥官裁定）：**逐档**核对该档是否走 fail-closed 路径 ──
    #   判据 = **该档自己的 stdout 里出现 native SFT 路径的实现痕迹**（`train_batch_sft` /
    #   `sft_trainer.py`）**且 rc=0** —— 不是按后端/档族整批推定。
    #   锚点（代码事实）：`sft_trainer.py:402-415 _assert_no_frozen_steps` 在落盘前**必然 raise**；
    #   `training_sft.py:313 _record_nonfinite_skip` 默认**硬失败**（非预授权时）。
    #   ⇒ 走这条路径还能 rc=0 跑完，就**正面证明**没有非有限跳过。
    fail_closed_guarantee = ""
    if exit_code == 0 and _NATIVE_SFT_PATH_MARKER.search(log_text or ""):
        evidence_c = (
            "src/graspo/flow/trainer/sft_trainer.py:402-415 (_assert_no_frozen_steps："
            "有跳过则落盘前必然 raise) + "
            "src/graspo/flow/adapters/models/qwen35_36/training_sft.py:313 "
            "(_record_nonfinite_skip：默认硬失败)"
        )
        fail_closed_guarantee = f"该档 stdout 实证走 native SFT 路径；锚点 {evidence_c}"
    evidence = _judge.RunEvidence(
        tier_id=tier_id,
        exit_code=exit_code,
        timed_out=timed_out,
        log_text=log_text,
        tuner_type=tuner_type,
        optimizer_steps=series.steps,
        epochs_completed=series.epochs,
        weight_changed=weight_evidence.value,
        checkpoint_reloadable=reload_evidence.value,
        artifacts_present=extract_artifacts(run_dir, output_dirs, log_text),
        losses=tuple(series.losses),
        grad_norms=tuple(series.grad_norms),
        optimizer_steps_per_step=(
            tuple(series.optimizer_steps_per_step) if series.optimizer_steps_per_step else None
        ),
        nonfinite_skips=nonfinite_skips_value,
        nonfinite_skips_source=nonfinite_skips_source_value,
        nonfinite_fail_closed_guarantee=fail_closed_guarantee,
        losses_nonfinite=series.loss_nonfinite,
        losses_unavailable=series.loss_unavailable,
        steps_declared_total=series.declared_total_steps,
        first_logged_step=series.first_logged_step,
        output_located=bool(output_dirs),
        min_optimizer_steps=min_optimizer_steps,
        expected_optimizer_steps_per_epoch=expected_optimizer_steps_per_epoch,
        expected_optimizer_steps_reachable=expected_optimizer_steps_reachable,
        run_root=str(run_dir.resolve()),
        output_identity=output_identity,
        first_metric_timestamp=extract_run_first_timestamp(run_dir),
        # 读数口径自证（§1.4）：台账必须能回答"这条权重/重载证据是哪种等级、来自哪里"。
        weight_evidence_source=weight_evidence.source,
        weight_evidence_detail=weight_evidence.detail,
        reload_evidence_source=reload_evidence.source,
        reload_evidence_detail=reload_evidence.detail,
        # A4 档族坐标 + 独立通道（§1.4：容差属于档族这个事实，事实的权威来源是清单）。
        backend=backend,
        cards=cards,
        algorithm=algorithm,
        model=model,
        mode=mode,
        final_ckpt_sha256=ckpt_sha256,
        final_ckpt_sha256_detail=ckpt_sha256_detail,
        final_ckpt_sha256_source=ckpt_sha256_source,
    )
    return evidence, series


# ── 主流程 ──────────────────────────────────────────────────────────────────


def _read_peak_memory(run_dir: Path, backend: str = "native") -> tuple[float | None, str]:
    """读 capability-matrix §7「实测每卡峰值显存(GiB)」列 —— **二分口径，按后端**。

    **返回 ``(值, 口径标签)``**：值与口径**成对返回**，调用方不可能只拿到数字而不知道
    它是什么口径（§2.2 显式即防呆）。

    口径的**权威定义**：`.local/本期工程跟踪.md` §9.1「★ §7 各列的读取口径」表的
    `实测每卡峰值显存(GiB)` 行（**可核定位**：该文件 `:420`，表头在 `:415`）——
    「**二分口径（按 A §7 的「后端」列即可推导，不必逐格加标签）：`native` 档 =
    容器内 PyTorch allocator 的 rank0 `max_allocated`；`ms-swift` 档 = 该进程的
    `max_memory_reserved`（同进程全部可见卡取 max）**」。

    后端 → 口径的**唯一映射**在
    ``graspo.core.result_judge.PEAK_MEMORY_CALIBER_BY_BACKEND``（本函数只按它分派，
    不自己写第二份判断）：

    - ``native`` ⇒ :data:`result_judge.PEAK_MEMORY_CALIBER` ⇒ 走
      `_read_allocator_peak_memory`（rank0 `rank_metrics` 的 `max_allocated_mib` 取最大）；
    - ``ms-swift`` ⇒ :data:`result_judge.MSSWIFT_RESERVED_CALIBER` ⇒ 走
      `_read_msswift_reserved_peak_memory`（自报 `memory(GiB)` = `max_memory_reserved`，
      **保守上界**：`reserved >= allocated`）；
    - **未登记的后端** ⇒ 口径标签 :data:`result_judge.PEAK_MEMORY_CALIBER_UNKNOWN`
      （显式"无口径"，**不猜、不默认取某一支**），值恒为 ``None``。

    **不可得一律返回 ``(None, 该档口径标签)``，绝不回退宿主口径、也绝不回退另一种口径**
    （§9.1 明文禁止混口径；宿主采样在共享机上会被同租户作业污染）。不产 `rank_metrics`
    的 ms-swift 后端曾因此结构性拿不到值——二分口径正是为消除该缺口而裁定（2026-09-21），
    但它**仍然不放宽**"不可得 ⇒ 显式标注"这条底线：台账由
    ``result_judge.peak_memory_unavailable_note`` 显式标注，而不是拿宿主采样补坑。

    ``backend`` 默认 ``"native"``：仅为兼容本函数的历史单参调用（历史语义即 allocator
    口径）；**生产路径 ``run()`` 一律显式传入该档 A §7 的「后端」列值**。
    """
    caliber = _judge.peak_memory_caliber_for_backend(backend)
    if caliber == _judge.PEAK_MEMORY_CALIBER:
        return _read_allocator_peak_memory(run_dir), caliber
    if caliber == _judge.MSSWIFT_RESERVED_CALIBER:
        return _read_msswift_reserved_peak_memory(run_dir), caliber
    # 未登记的后端：无口径可谈 ⇒ 显式返回"无口径"，值不可得。**不得**猜一个口径出来。
    return None, caliber


def _read_allocator_peak_memory(run_dir: Path) -> float | None:
    """**native 支路**：读 rank0 的 PyTorch allocator 峰值 ``max_allocated``（GiB）。

    （二分口径下本条只服务 `native` 档；分派见 :func:`_read_peak_memory`。）

    落点：rank0 的逐 rank 指标旁路 ``metrics/rank_metrics.rank_00000.jsonl``——训练侧
    ``transformer_adapter._emit_rank_memory_event`` 每写完一次显存快照就追加一行
    ``{"event": "rank_memory", "memory": {..., "max_allocated_mib": ...}}``，
    值由 ``tensor_utils._cuda_memory_snapshot`` 的
    ``torch.cuda.max_memory_allocated(device)`` 换算得到。

    取数：该文件内**所有** ``rank_memory`` 行的 ``memory.max_allocated_mib`` 取最大
    （allocator 的 max 是累计峰值，逐 phase 取最大值即该 run 峰值，无需按 phase 白名单
    筛选——白名单会随训练侧新增 phase 而漏值）。

    **不可得一律返回 ``None``，绝不回退宿主口径**（§9.1 明文禁止混口径；回退正是本函数
    旧实现的缺陷）。不产 ``rank_metrics`` 的后端（如 ms-swift）结构性拿不到该值 ⇒
    该档改走自己的二分口径支路（见 :func:`_read_peak_memory`）。
    """
    peaks: list[float] = []
    for events in sorted(run_dir.rglob(_PEAK_MEMORY_RANK_FILE)):
        for payload in _iter_jsonl_objects(events):
            memory = payload.get("memory")
            if not isinstance(memory, dict):
                continue
            value = memory.get("max_allocated_mib")
            if value is None:
                continue
            try:
                peaks.append(float(value))
            except (TypeError, ValueError):
                continue
    if not peaks:
        return None
    return max(peaks) / 1024.0


def _read_msswift_reserved_peak_memory(run_dir: Path) -> float | None:
    """**ms-swift 支路**：读该后端自报的 ``memory(GiB)``（GiB）——**reserved 口径**。

    **口径地位（2026-09-21 指挥官裁定「方案 A：二分口径」）**：它**就是** ms-swift 档在
    §7「实测每卡峰值显存(GiB)」列的合法取数来源（分派见 :func:`_read_peak_memory`）。
    ms-swift 会打印 ``logs['memory(GiB)']``，取的是 ``max_memory_reserved``：
      - ``.local/refs/ms-swift-4.5.3/git-v4.5.3/swift/trainers/patcher.py:27``
        ``state.max_memory = max(getattr(state, 'max_memory', 0), get_max_reserved_memory())``
        （`:29` 落 ``logs['memory(GiB)']``）
      - ``swift/utils/torch_utils.py:413-419``
        ``[get_torch_device().max_memory_reserved(device=device) ...]`` → ``max(...)/1024**3``
      - ``swift/megatron/callbacks/print.py:64``
        ``reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)``
    三处**均无** ``max_memory_allocated()`` ⇒ ms-swift 后端结构性产不出 rank0
    ``max_allocated``（也不产 `rank_metrics` 旁路）⇒ 旧的"单一口径"会让 36 个 ms-swift 档
    永久填不出值。二分口径（§9.1 `.local/本期工程跟踪.md:420`）据此把本值定为
    ms-swift 档该列的合法值。

    ★ **它是保守上界**：``reserved >= allocated`` 恒成立（reserved 含 caching allocator
    的空闲缓存块）⇒ 用户拿它判断"我的硬件够不够"**不会被低估**。这一点必须与数字一起
    出现在文档里（§9.1 与 §7 脚注均已写明）。

    除 §7 峰值列外，本值仍经 ``result_judge.MSSWIFT_RESERVED_PEAK_FIELD`` 另存为
    "上游自报字段的原值 + 来源标签"（同一防呆模式见
    :func:`_read_host_sample_peak_memory`；后者**不是**本列口径，仍严格另存）。

    **取数来源（按稳定性排序，前者命中即不再看后者）**：

    1. ``logging.jsonl``（ms-swift 自己用 ``append_to_jsonl`` 写的**机器可读**产物：
       ``swift/trainers/patcher.py:55``〔`ProgressCallbackNew`〕与 ``:98``
       〔`PrinterCallbackNew`〕）：逐行 ``json.loads``，取键 ``"memory(GiB)"``。
       主键名正则（若将来要在文本层匹配）：``memory\\(GiB\\)`` —— 括号需转义。
    2. ``stdout.log`` 回退：``swift/trainers/patcher.py:101`` 的
       ``print(logs, flush=True)``（tqdm 另有 ``write(str(logs))``）把 ``logs`` 的
       **dict repr** 打进 stdout，形如 ``{'loss': ..., 'memory(GiB)': '24.45', ...}``，
       因此用 ``memory\\(GiB\\)['\"]?\\s*:\\s*['\"]?(-?\\d+(?:\\.\\d+)?)`` 取值。
       为什么需要这条回退：实测档 ``T044`` **没有** ``logging.jsonl``，只留 stdout。

    **多卡怎么取"每卡峰值"**：ms-swift 每个日志事件**只打一行、一个数**——
    ``get_max_reserved_memory`` 已对**本进程可见的全部卡取 max**，即"最忙那张卡"。
    它不是逐卡多行输出 ⇒ 本函数的算法是：**把所有行的该值取最大**（``state.max_memory``
    本身也是单调累积 max，逐行取 max 与取末行等价），得到"该 run 的单卡峰值"。

    **不可得一律返回 ``None``**（不产 ``logging.jsonl`` / 无该键 / 解析失败）⇒ 台账按
    本档口径落 ``result_judge.MSSWIFT_RESERVED_UNAVAILABLE`` 显式标注，
    **绝不**回退 ``gpu_memory_summary.json``（宿主口径，#9.1 仍禁止混入本列），
    **也绝不**用 native 的 ``rank_metrics`` 值兜底（那是另一个 run 的读数、另一种口径）。
    """
    peaks: list[float] = []
    for path in sorted(run_dir.rglob(_MSSWIFT_LOGGING_JSONL)):
        for payload in _iter_jsonl_objects(path):
            raw = payload.get(_MSSWIFT_MEMORY_KEY)
            value = _parse_numeric(raw)
            if value is not None:
                peaks.append(value)
    if peaks:
        return max(peaks)

    for path in sorted(run_dir.rglob("stdout.log")):
        for match in _MSSWIFT_STDOUT_MEMORY_RE.finditer(_read_text(path)):
            value = _parse_numeric(match.group(1))
            if value is not None:
                peaks.append(value)
    if not peaks:
        return None
    return max(peaks)


def _read_host_sample_peak_memory(run_dir: Path) -> float | None:
    """读**宿主 `nvidia-smi` 采样**峰值（GiB）——**不是** §7 峰值列的口径。

    来源：``gpu/gpu_memory_summary.json`` 各卡 ``per_gpu[*].memory_used_mib_peak``
    的最大值 ÷ 1024。它是**取证链锚点**，回答"这张卡实际用了多少"（含非 PyTorch 的
    缓存/其它进程），因此与 allocator 口径**必然有可观测差异**。

    §9.1 明文：宿主采样**不得混入 §7 峰值列** ⇒ 本函数的结果只能经
    ``result_judge.HOST_SAMPLE_PEAK_FIELD``（``host_sample_peak_gib``）另存台账，
    下游不得用它填 ``peak_memory_gib``。
    """
    summary = _read_json(run_dir / "gpu" / "gpu_memory_summary.json")
    if not isinstance(summary, dict):
        return None
    peaks = [
        float(item["memory_used_mib_peak"])
        for item in (summary.get("per_gpu") or {}).values()
        if isinstance(item, dict) and item.get("memory_used_mib_peak") is not None
    ]
    if not peaks:
        return None
    return max(peaks) / 1024.0


def _read_host_sample_gap_mib(run_dir: Path) -> float | None:
    """读宿主采样的**采样缺口**（MiB）——"峰值被他人作业污染"的显式证据。

    来源：``gpu/gpu_memory_summary.json`` 的 ``max_peak_memory_gap_mib``。缺口 > 0
    说明采样窗口内该卡的占用含**非本档**成分（228 是共享机）：此时宿主峰值偏高，
    真值应取其"平段"而非峰值。**与宿主峰值同源、同一份摘要**，不由本函数另算口径。
    """
    summary = _read_json(run_dir / "gpu" / "gpu_memory_summary.json")
    if not isinstance(summary, dict):
        return None
    gap = summary.get("max_peak_memory_gap_mib")
    if gap is None:
        return None
    try:
        return float(gap)
    except (TypeError, ValueError):
        return None


def run(args: argparse.Namespace) -> int:
    manifest = _read_json(Path(args.manifest))
    if not isinstance(manifest, dict):
        print(f"FATAL: cannot read manifest {args.manifest}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 自比结构性防呆（AD1 §2.2 边界校验缺口 / 本包 ②）─────────────────────
    # 全库唯一能产生"自比"的路径就是调用方把**同一棵目录树**同时喂给 --run-root 与
    # --rerun-root（AD1 §2.3：判据层没有这条断言 ⇒ 台账里会出现一堆"差 0"的假 A4）。
    # ★ 必须 `resolve()` 后比较：软链/相对路径/尾斜杠都是"同一实体目录"的伪装。
    # 方向不变：这是**边界校验**（拒绝非法请求），不是退路（§2.3/§3.4）——
    # 相同 ⇒ 显式报错退出，绝不静默跑出一堆差 0 的 A4 结论。
    run_root_resolved = Path(args.run_root).resolve()
    rerun_root_resolved = Path(args.rerun_root).resolve() if args.rerun_root else None
    if rerun_root_resolved is not None and rerun_root_resolved == run_root_resolved:
        print(
            "FATAL: --run-root 与 --rerun-root 解析后指向同一实体目录"
            f"（{run_root_resolved}）⇒ 这是**自比**，A4 结论无意义。"
            "请给两个不同的运行根（批次驱动必须让两跑落两棵目录树，见 "
            "`tests/e2e/run_matrix54.sh` 的 RUN_ROOT 用法）。",
            file=sys.stderr,
        )
        return 2

    records: list[dict[str, Any]] = []
    for tier in manifest.get("tiers", []):
        tier_id = str(tier["tier_id"])
        tuner_type = "full" if tier.get("mode") == "全量" else "lora"
        # 本档 §7 峰值列的口径**由后端唯一决定**（§1.4 单一真相源）。
        # ★ 必须在"没跑过"分支也用它：否则下游会把该格读成"可以用宿主采样补上"。
        backend = str(tier.get("backend"))
        # A4 档族坐标（五要素第 1 条 + J2/P1 扩到五维）：容差由
        # backend × cards × algorithm × model × mode 唯一决定，因此这五个值必须从清单
        # （唯一真相源）透传到证据里，而不是让判定器自己猜。算法名原样透传，
        # 归一（GRASPO→GRPO）只在判定器的 `normalize_a4_algorithm` 一处做。
        algorithm = str(tier.get("algorithm")) if tier.get("algorithm") is not None else None
        model = str(tier.get("model")) if tier.get("model") is not None else None
        mode = str(tier.get("mode")) if tier.get("mode") is not None else None
        cards_raw = tier.get("cards")
        cards = int(cards_raw) if isinstance(cards_raw, int) else None
        tier_caliber = _judge.peak_memory_caliber_for_backend(backend)
        tier_caliber_note = _judge.peak_memory_unavailable_note(tier_caliber)
        first_dir = Path(args.run_root) / tier_id
        if not first_dir.is_dir():
            # ── ★ 六态口径（2026-09-22 指挥官裁定）────────────────────────────
            # 旧实现把三种"没跑"一律写成 `— 未测` ⇒ 丢掉用户明确关心过的区分
            # （"口径不可测是什么意思？无配方不适用呢？"）。三段必须分开：
            #   · `<T>.blocked.md`         ⇒ ⛔ 无配方（当前无配方、**将来可能做得了**）
            #   · `<T>.not_applicable.md`  ⇒ ⛔ 不适用（**逻辑上不适用**）
            #   · 其余（清单 ready 但未跑） ⇒ — 未测（尚未纳入跑批）
            # 出处写进 `status_provenance`（§2.2 显式即防呆：读者能查到依据文件）。
            _cfg_dir = (
                Path(getattr(args, "repo_root", None) or Path.cwd())
                / "samples" / "configs" / "matrix54"
            )
            _blocked_md = _cfg_dir / f"{tier_id}.blocked.md"
            _na_md = _cfg_dir / f"{tier_id}.not_applicable.md"
            if _blocked_md.is_file():
                _status = _judge.LEDGER_NO_RECIPE
                _provenance = str(_blocked_md.resolve())
                _why = "当前无配方（blocked，将来可能做得了）"
            elif _na_md.is_file():
                _status = _judge.LEDGER_NOT_APPLICABLE
                _provenance = str(_na_md.resolve())
                _why = "逻辑上不适用（not_applicable）"
            else:
                _status = _judge.LEDGER_UNTESTED
                _provenance = None
                _why = "尚未纳入跑批"
            records.append(
                {
                    "tier_id": tier_id,
                    # ★ 身份字段必须**根上补齐**（2026-09-22 覆盖度分析：`T016/T034/T052–T054`
                    #   这 5 行只有 22 个字段、缺身份键 ⇒ 下游只能做 fallback 回填，
                    #   容易与 manifest 漂移。这里从**清单**（唯一真相源）直接落盘。
                    "model": model,
                    "algorithm": algorithm,
                    "mode": mode,
                    "backend": backend,
                    "cards": cards,
                    # 未跑的档没有判据结论 ⇒ 显式给空表（而不是缺键），
                    # 让"缺字段"与"判据为空"在结构上可区分。
                    "criteria": {},
                    "status": _status,
                    "status_provenance": _provenance,
                    "note": f"运行目录不存在：{first_dir}（{_why}）",
                    "failure_class": None,
                    "max_context": None,
                    "max_context_kind": None,
                    # 峰值列同样要口径自证（§9.1 / §2.2）：没跑过 ⇒ 该档口径不可得，
                    # 但**不得**让下游把这格读成"可以用宿主采样补上"。
                    "peak_memory_gib": None,
                    "peak_memory_caliber": tier_caliber,
                    "peak_memory_note": tier_caliber_note,
                    _judge.HOST_SAMPLE_PEAK_FIELD: None,
                    _judge.HOST_SAMPLE_GAP_FIELD: None,
                    # 同上：ms-swift 自报 reserved 口径也要"没跑过 ⇒ 不可得 + 原因"，
                    # 不留一个缺失字段让下游分不清"没查"与"没值"。
                    _judge.MSSWIFT_RESERVED_PEAK_FIELD: None,
                    _judge.MSSWIFT_RESERVED_CALIBER_FIELD: _judge.MSSWIFT_RESERVED_CALIBER,
                    _judge.MSSWIFT_RESERVED_NOTE_FIELD: _judge.MSSWIFT_RESERVED_UNAVAILABLE,
                    "series_source": "none",
                    "series_notes": [],
                    "losses": [],
                    "grad_norms": [],
                    "optimizer_steps_per_step": [],
                    "nonfinite_skips": None,
                    "ignored_phase_records": {},
                }
            )
            continue
        first, first_series = collect_run(
            first_dir,
            tier_id,
            tuner_type,
            _base_model_dir(args, tier),
            min_optimizer_steps=extract_min_optimizer_steps(tier),
            backend=backend,
            cards=cards,
            algorithm=algorithm,
            model=model,
            mode=mode,
            # ★ 步数上限（AF1 §⑥ 方案 S1/S2）：清单读出的"结构可达性"证据，
            #   判定器据此把"口径不可测"与"训练失败"分开（**不放宽**门槛）。
            expected_optimizer_steps_per_epoch=extract_expected_optimizer_steps(tier)[0],
            expected_optimizer_steps_reachable=extract_expected_optimizer_steps(tier)[1],
        )
        second = None
        if args.rerun_root:
            second_dir = Path(args.rerun_root) / tier_id
            if second_dir.is_dir():
                second, _ = collect_run(
                    second_dir,
                    tier_id,
                    tuner_type,
                    _base_model_dir(args, tier),
                    min_optimizer_steps=extract_min_optimizer_steps(tier),
                    backend=backend,
                    cards=cards,
                    algorithm=algorithm,
                    model=model,
                    mode=mode,
                    expected_optimizer_steps_per_epoch=(
                        extract_expected_optimizer_steps(tier)[0]
                    ),
                    expected_optimizer_steps_reachable=(
                        extract_expected_optimizer_steps(tier)[1]
                    ),
                )
        # ★ 能力矩阵采集**必须**启用 A4 双通道强制口径（本包 ②/锁三）：否则
        # "两跑 loss 逐位相同（死通道）+ 独立通道取不到证据"会静默退化成 ✅ 通过，
        # AD1 §④ 的 7 条退化通过就漏过去了。
        judgement = _judge.judge_tier(
            first,
            second,
            context_length=args.context_length,
            require_dual_channel=True,
        )
        # §7 峰值列：**按后端选口径取数，值与口径成对拿到**（§9.1 二分口径 / §1.4）。
        peak_gib, peak_caliber = _read_peak_memory(first_dir, backend)
        row = _judge.ledger_row(
            judgement,
            model=str(tier.get("model")),
            algorithm=str(tier.get("algorithm")),
            mode=str(tier.get("mode")),
            backend=backend,
            cards=int(tier.get("cards", 0)),
            # 🔴-1 修正：**不要**在这里按 counts_toward_max_context 预筛。资格判定
            # 是 ledger_row 的唯一真相源（通过档 ⇒ 实测可行值；真 OOM ⇒ 边界候选），
            # 采集层再筛一遍会把"通过档"的上下文也丢掉（旧实现的实际后果：
            # max_context 恒为 None）。这里只把"这次实测的上下文长度"原样传下去。
            max_context=args.context_length,
            peak_memory_gib=peak_gib,
            # 口径标签随值一起落盘，且 `ledger_row` 会与 backend 的推导值核对（不一致即拒）。
            peak_memory_caliber=peak_caliber,
            date=args.date,
            # 宿主采样峰值**另存**（§9.1：它是取证链锚点，二分口径下**仍**不进 §7 峰值列）。
            host_sample_peak_gib=_read_host_sample_peak_memory(first_dir),
            host_sample_peak_gap_mib=_read_host_sample_gap_mib(first_dir),
            # ms-swift 自报的 reserved 峰值的**原始读数**：另存 + 带口径标签
            # （见 `_read_msswift_reserved_peak_memory` 的源码依据）；二分口径下它同时
            # 就是 ms-swift 档 §7 峰值列的取数来源，故此处复用同一个取数函数。
            msswift_reserved_peak_gib=_read_msswift_reserved_peak_memory(first_dir),
        )
        row["criteria"] = {item.criterion: item.passed for item in judgement.criteria}
        row["criteria_detail"] = {item.criterion: item.detail for item in judgement.criteria}
        # 读数口径自证（§1.4 单一真相源）：台账必须能回答"这个数字从哪来"。
        row["series_source"] = first_series.source
        row["series_notes"] = first_series.notes
        row["losses"] = list(first.losses)
        row["grad_norms"] = list(first.grad_norms)
        row["optimizer_steps_per_step"] = list(first.optimizer_steps_per_step or ())
        row["nonfinite_skips"] = first.nonfinite_skips
        row["nonfinite_skips_source"] = first.nonfinite_skips_source
        # 防呆（§2.2）：含 metrics 却被忽略的 phase 记录必须在台账里可见——
        # "0 条"也要显式落一个空 dict，便于下游区分"没丢"与"没查"。
        row["ignored_phase_records"] = dict(first_series.ignored_phase_records)
        row["steps_declared_total"] = first.steps_declared_total
        row["first_logged_step"] = first.first_logged_step
        # ── 自比防呆的两列（本包 ② / AD1 §2.3）──────────────────────────────
        # 台账必须**自证**"这两跑确实来自两棵目录树"。旧台账不含这两列 ⇒ 事后无法仅凭
        # 台账区分"真双跑"与"自比跑出一堆差 0"（AD1 §2.4 实测：键集合里没有它们）。
        row["run_root"] = str(run_root_resolved)
        row["rerun_root"] = str(rerun_root_resolved) if rerun_root_resolved else None
        # A4 档族坐标（五要素第 1 条 + J2/P1 五维）：让"这一档按哪个容差判的"在台账里可审计，
        # 与 min_optimizer_steps 同理（§1.4 单一真相源）。
        row["backend"] = backend
        row["cards"] = cards
        row["algorithm"] = algorithm
        row["model"] = model
        row["mode"] = mode
        # A4 独立通道（五要素第 5 条）：两跑末步权重指纹。经 `_judge.a4_ckpt_sha_fields`
        # 落盘 ⇒ **键名只有一处定义**，台账写一次、下游读同一名字（§1.4）。
        row.update(
            _judge.a4_ckpt_sha_fields(first.final_ckpt_sha256, second and second.final_ckpt_sha256)
        )
        row["final_ckpt_sha256_detail"] = first.final_ckpt_sha256_detail
        row["final_ckpt_sha256_source"] = first.final_ckpt_sha256_source
        # A4 形态①的独立性台账（裁定 2）：把"凭什么判两次独立运行"落盘，读者可复核。
        if second is not None:
            _signals, _independence_detail = _judge._a4_independence_signals(first, second)
            row["a4_independence_signals"] = _signals
            row["a4_independence_detail"] = _independence_detail
        records.append(row)

    # ★ 六态白名单断言（§2.2 显式即防呆）：产出里出现第 7 种状态词即 fail-closed。
    _allowed = set(_judge.LEDGER_STATUS_PHRASES)
    _bad = sorted({str(r.get("status")) for r in records} - _allowed)
    if _bad:
        print(f"FATAL: 台账出现白名单外的状态词：{_bad}（白名单见 LEDGER_STATUS_PHRASES）", file=sys.stderr)
        return 5

    jsonl = out_dir / "ledger.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    markdown = out_dir / "ledger.md"
    # `max_context` 的口径必须随数字一起显示（🔴-1）：同样是 "8192"，"实测通过"
    # 与"真 OOM 边界候选"的含义完全不同，只给数字会被下游读成"这个长度跑得通"。
    # 峰值列同理（§9.1 二分口径 / §2.2）：列头写死**按后端二分**的口径规则，空值写**原因**
    # 而不是一个光秃秃的 `—`，宿主副读数**单独一列**——三者分开才让"这一格是什么口径"
    # 可回答。逐格不再加标签：口径可由同表「后端」列**单值推导**（§9.1 的明文依据）。
    lines = [
        "| 条件档 | 模型 | 算法 | 模式 | 后端 | 卡数 | 最大可行上下文 | 口径 "
        f"| 每卡峰值(GiB)【native={_judge.PEAK_MEMORY_CALIBER} / "
        f"ms-swift={_judge.MSSWIFT_RESERVED_CALIBER}（保守上界）】 | 宿主采样峰值(GiB) "
        f"| 宿主采样缺口(MiB) | ms-swift自报(GiB)【{_judge.MSSWIFT_RESERVED_CALIBER}】 "
        "| 状态 | 失败类型 | 备注 |",
        "|---|---|---|---|:--:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        context_cell = record.get("max_context") or "—"
        if record.get("max_context_kind"):
            context_cell = f"{context_cell}（{record['max_context_kind']}）"
        peak_cell = record.get("peak_memory_gib")
        if peak_cell is None:
            # 显式标注"不可得"及其原因，**不**回退宿主口径（§9.1 禁止混口径）。
            peak_cell = record.get("peak_memory_note") or "—"
        host_cell = record.get(_judge.HOST_SAMPLE_PEAK_FIELD)
        if host_cell is None:
            host_cell = "—"
        gap_cell = record.get(_judge.HOST_SAMPLE_GAP_FIELD)
        if gap_cell is None:
            gap_cell = "—"
        # ms-swift 自报值**自带口径标签**地印在独立一列；无值印原因（不是光秃秃的 `—`）。
        msswift_cell = record.get(_judge.MSSWIFT_RESERVED_PEAK_FIELD)
        if msswift_cell is None:
            msswift_cell = record.get(_judge.MSSWIFT_RESERVED_NOTE_FIELD) or "—"
        lines.append(
            f"| {record['tier_id']} | {record.get('model', '')} | {record.get('algorithm', '')} "
            f"| {record.get('mode', '')} | {record.get('backend', '')} | {record.get('cards', '')} "
            f"| {context_cell} | {record.get('max_context_kind') or '—'} "
            f"| {peak_cell} | {host_cell} | {gap_cell} | {msswift_cell} "
            f"| {record['status']} | {record.get('failure_class') or '—'} "
            f"| {record.get('note', '')} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed = sum(1 for record in records if record["status"] == _judge.LEDGER_PASS)
    failed = sum(1 for record in records if record["status"] == _judge.LEDGER_FAIL)
    indeterminate = sum(1 for record in records if record["status"] == _judge.LEDGER_INDETERMINATE)
    untested = len(records) - passed - failed - indeterminate
    print(
        f"ledger: {jsonl} ({len(records)} tiers: pass={passed} fail={failed} "
        f"indeterminate={indeterminate} untested={untested})"
    )
    print(f"markdown: {markdown}")
    return 0


def _base_model_dir(args: argparse.Namespace, tier: dict[str, Any]) -> Path | None:
    """全参档 A2 的基座模型目录（用于"权重是否真的变了"）。

    口径：``--base-model-root``（**唯一来源**）+ 配置里 ``model_path`` 的目录名。
    给不出时返回 ``None`` ⇒ A2 的权重证据按**取证缺口**处理（不可判定），不猜。

    **双真相源已消除（§1.4/§7.1）**：旧写法是
    ``args.base_model_root or os.environ.get("GRASPO_MODELS_HOST_ROOT")`` —— 一条
    ``or`` 回退链：CLI 与进程环境**都能**决定"拿哪个目录对比权重"，两者同时给值时
    环境变量被静默忽略（"改了没生效"），只给环境变量时又不进 config 备份、无法复现。
    现在只有一个来源：CLI 的 ``--base-model-root``（宿主路径注入走 CLI 参数，§10.1；
    与 ``run.sh --model-dir`` 同口径）。环境变量对结果**再无任何影响**。

    **可核断言**（见 ``tests/e2e/test_collect_results.py``）：仅设置环境变量
    ``GRASPO_MODELS_HOST_ROOT`` 而**不**传 ``--base-model-root`` 时，本函数必须返回
    ``None``（证据判不可判定）——环境变量不再"恰好生效"。
    """
    root = args.base_model_root
    if not root:
        return None
    name = Path(str(tier.get("model_path") or "")).name
    if not name:
        return None
    candidate = Path(root) / name
    return candidate if candidate.is_dir() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect A1–A6 verdicts into ledger records.")
    parser.add_argument("--manifest", required=True, help="matrix54_manifest.json")
    parser.add_argument("--run-root", required=True, help="First-attempt run root.")
    parser.add_argument("--rerun-root", default=None, help="Second-attempt run root (for A4).")
    parser.add_argument("--repo-root", default=None,
                        help="仓库根（用于定位 samples/configs/matrix54/*.blocked.md / "
                             "*.not_applicable.md）；缺省 = 当前工作目录")
    parser.add_argument("--out", required=True, help="Output directory for ledger.jsonl/.md")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Tested context length (the length THIS run was launched with). "
            "Omitted ⇒ 台账不写 max_context（None，fail-closed：宁缺不猜；"
            "通过档写 max_context_kind=实测通过，真 OOM 档写 =真 OOM 边界候选）。"
        ),
    )
    parser.add_argument("--date", default="", help="Ledger date (YYYY-MM-DD).")
    parser.add_argument(
        "--base-model-root",
        default=None,
        help=(
            "全参档 A2 的基座模型宿主根目录（**唯一来源**，不再读环境变量）。"
            "collector 取 <root>/<model_path 的目录名> 与 checkpoint 权重抽样比较；"
            "给不出时 A2 的权重证据按取证缺口处理（不可判定），不猜也不放松。"
            "宿主路径注入一律走 CLI 参数（§10.1），与 run.sh --model-dir 同口径。"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
