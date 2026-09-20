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
import importlib.util
import json
import math
import re
import struct
import sys
from collections.abc import Sequence
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
    #: 累计因非有限梯度跳过优化器步的次数（``skipped_nonfinite``，跨步求和）。
    nonfinite_skips: int = 0
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
    for directory in output_dirs:
        metrics_rows, ignored_phases = _rank_metric_steps(directory)
        if metrics_rows:
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
            skipped = row.get("skipped_nonfinite")
            if isinstance(skipped, int):
                result.nonfinite_skips += skipped
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
    if _NONFINITE_GRAD_MARKER.search(log_text) and result.nonfinite_skips == 0:
        # 硬失败标记本身就是"这一步没推进"的证据（默认 1 次，仅用于让 A2 不通过；
        # 精确次数由 rank_metrics 的 skipped_nonfinite 给出）。
        result.nonfinite_skips = max(1, result.nonfinite_skips)
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


def extract_weight_changed(
    output_dirs: Sequence[Path], tuner_type: str, base_model_dir: Path | None = None
) -> bool | None:
    """抽取"权重是否真变化"证据（LoRA 看 lora_b 非零；全参看与基座是否不同）。

    两种后端的**判据语义完全相同**，只是 checkpoint 目录名不同（``final`` /
    ``checkpoint-<step>``），由 :func:`find_checkpoint_dirs` 统一发现。

    抽取不到返回 ``None`` → A2 按**取证缺口**处理（不可判定，而不是判训练失败）。
    """
    torch = _require_torch()
    for checkpoint in find_checkpoint_dirs(output_dirs):
        if tuner_type == "lora":
            # native：单个 rank pt 里的 lora_state_dict
            for pt in sorted(checkpoint.glob("rank_*.pt")):
                if torch is None:
                    break
                try:
                    payload = torch.load(pt, map_location="cpu", weights_only=False)
                except (RuntimeError, OSError, AttributeError):
                    continue
                if not isinstance(payload, dict):
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
                        return result
            # msswift / peft：adapter_model.safetensors 里的 lora_B
            for weights in sorted(checkpoint.glob("adapter_model.safetensors")):
                result = _safetensors_any_nonzero(weights, key_filter="lora_b")
                if result is not None:
                    return result
            continue
        # 全参：终态权重与基座权重做抽样比较
        result = _full_weights_differ(checkpoint, base_model_dir)
        if result is not None:
            return result
    return None


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


def extract_checkpoint_reloadable(output_dirs: Sequence[Path]) -> bool | None:
    """A3 证据：checkpoint 能否被重新加载（结构化校验 + 反序列化）。

    两种布局走同一条判据：
      - native：``final/manifest.json`` + ``rank_*.pt``（需 torch 反序列化）；
      - ms-swift / peft：``*.safetensors``（stdlib 解析头部 + 校验 data_offsets）。
    判据语义不变（结构完整 + 可反序列化）；**没有**因为后端不同而放松。
    """
    checkpoints = find_checkpoint_dirs(output_dirs)
    if not checkpoints:
        return None
    checked = False
    for checkpoint in checkpoints:
        # native：manifest.json + rank_*.pt 能否 torch.load
        if (checkpoint / "manifest.json").exists():
            torch = _require_torch()
            rank_files = sorted(checkpoint.glob("rank_*.pt"))
            if not rank_files:
                continue
            if torch is None:
                continue
            if not _file_readable(rank_files[0]):
                # 权限/IO 不可读 ⇒ 取证缺口（继续找别的 checkpoint），不得判"重载失败"。
                continue
            try:
                torch.load(rank_files[0], map_location="cpu", weights_only=False)
                return True
            except (RuntimeError, OSError, AttributeError):
                return False
        # peft / HF（含 ms-swift 的 checkpoint-<step>）：safetensors 头部能否解析
        for weights in sorted(checkpoint.glob("*.safetensors")):
            if not _file_readable(weights):
                continue  # 取证缺口：采集侧读不到，不是 checkpoint 坏
            checked = True
            parsed = _safetensors_index(weights)
            if parsed is None:
                return False  # 能读但头部不可解析 ⇒ 这才是"checkpoint 无法重新加载"
            index, _ = parsed
            if not index:
                return False
            return True
    return None if not checked else False


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


def collect_run(
    run_dir: Path,
    tier_id: str,
    tuner_type: str,
    base_model_dir: Path | None = None,
    min_optimizer_steps: int | None = None,
) -> tuple[Any, SeriesEvidence]:
    """把一个运行目录抽成 ``RunEvidence`` + 读数口径自证（``SeriesEvidence``）。

    ``min_optimizer_steps`` 由调用方从清单读出（见 :func:`extract_min_optimizer_steps`），
    缺省 ``None`` = "清单未提供" ⇒ 判定器用缺省门槛（**不放松**）。
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
    evidence = _judge.RunEvidence(
        tier_id=tier_id,
        exit_code=exit_code,
        timed_out=timed_out,
        log_text=log_text,
        tuner_type=tuner_type,
        optimizer_steps=series.steps,
        epochs_completed=series.epochs,
        weight_changed=extract_weight_changed(output_dirs, tuner_type, base_model_dir),
        checkpoint_reloadable=extract_checkpoint_reloadable(output_dirs),
        artifacts_present=extract_artifacts(run_dir, output_dirs, log_text),
        losses=tuple(series.losses),
        grad_norms=tuple(series.grad_norms),
        optimizer_steps_per_step=(
            tuple(series.optimizer_steps_per_step) if series.optimizer_steps_per_step else None
        ),
        nonfinite_skips=series.nonfinite_skips,
        losses_nonfinite=series.loss_nonfinite,
        losses_unavailable=series.loss_unavailable,
        steps_declared_total=series.declared_total_steps,
        first_logged_step=series.first_logged_step,
        output_located=bool(output_dirs),
        min_optimizer_steps=min_optimizer_steps,
    )
    return evidence, series


# ── 主流程 ──────────────────────────────────────────────────────────────────


def _read_peak_memory(run_dir: Path) -> float | None:
    """从可信采样摘要读每卡峰值显存（GiB）；缺失返回 None。"""
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


def run(args: argparse.Namespace) -> int:
    manifest = _read_json(Path(args.manifest))
    if not isinstance(manifest, dict):
        print(f"FATAL: cannot read manifest {args.manifest}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for tier in manifest.get("tiers", []):
        tier_id = str(tier["tier_id"])
        tuner_type = "full" if tier.get("mode") == "全量" else "lora"
        first_dir = Path(args.run_root) / tier_id
        if not first_dir.is_dir():
            records.append(
                {
                    "tier_id": tier_id,
                    "status": "— 未测",
                    "note": f"运行目录不存在：{first_dir}",
                    "failure_class": None,
                    "max_context": None,
                    "max_context_kind": None,
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
                )
        judgement = _judge.judge_tier(first, second, context_length=args.context_length)
        row = _judge.ledger_row(
            judgement,
            model=str(tier.get("model")),
            algorithm=str(tier.get("algorithm")),
            mode=str(tier.get("mode")),
            backend=str(tier.get("backend")),
            cards=int(tier.get("cards", 0)),
            # 🔴-1 修正：**不要**在这里按 counts_toward_max_context 预筛。资格判定
            # 是 ledger_row 的唯一真相源（通过档 ⇒ 实测可行值；真 OOM ⇒ 边界候选），
            # 采集层再筛一遍会把"通过档"的上下文也丢掉（旧实现的实际后果：
            # max_context 恒为 None）。这里只把"这次实测的上下文长度"原样传下去。
            max_context=args.context_length,
            peak_memory_gib=_read_peak_memory(first_dir),
            date=args.date,
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
        # 防呆（§2.2）：含 metrics 却被忽略的 phase 记录必须在台账里可见——
        # "0 条"也要显式落一个空 dict，便于下游区分"没丢"与"没查"。
        row["ignored_phase_records"] = dict(first_series.ignored_phase_records)
        row["steps_declared_total"] = first.steps_declared_total
        row["first_logged_step"] = first.first_logged_step
        records.append(row)

    jsonl = out_dir / "ledger.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    markdown = out_dir / "ledger.md"
    # `max_context` 的口径必须随数字一起显示（🔴-1）：同样是 "8192"，"实测通过"
    # 与"真 OOM 边界候选"的含义完全不同，只给数字会被下游读成"这个长度跑得通"。
    lines = [
        "| 条件档 | 模型 | 算法 | 模式 | 后端 | 卡数 | 最大可行上下文 | 口径 "
        "| 每卡峰值(GiB) | 状态 | 失败类型 | 备注 |",
        "|---|---|---|---|:--:|---|---|---|---|---|---|---|",
    ]
    for record in records:
        context_cell = record.get("max_context") or "—"
        if record.get("max_context_kind"):
            context_cell = f"{context_cell}（{record['max_context_kind']}）"
        lines.append(
            f"| {record['tier_id']} | {record.get('model', '')} | {record.get('algorithm', '')} "
            f"| {record.get('mode', '')} | {record.get('backend', '')} | {record.get('cards', '')} "
            f"| {context_cell} | {record.get('max_context_kind') or '—'} "
            f"| {record.get('peak_memory_gib') or '—'} "
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
