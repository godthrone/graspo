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
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

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
_NUMERIC_LITERAL = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^-?(?:nan|inf)$", re.IGNORECASE)

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


def find_output_dir(run_dir: Path, tier_id: str) -> Path | None:
    """定位训练输出目录：优先约定路径，其次任何含 config.yaml 的子目录。"""
    candidates = [
        run_dir / "outputs" / tier_id,
        run_dir / "out" / tier_id,
        run_dir / tier_id,
        run_dir / "outputs",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.yaml").exists():
            return candidate
    for child in sorted(run_dir.rglob("config.yaml")):
        return child.parent
    return None


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
    #: 累计因非有限梯度跳过优化器步的次数（``skipped_nonfinite``，跨步求和）。
    nonfinite_skips: int = 0
    source: str = "none"
    notes: list[str] = field(default_factory=list)
    #: 是否出现过 ``loss: null`` / 字段缺失（A6 必须对此 fail-closed）。
    loss_unavailable: bool = False
    #: 是否**真读到**非有限 loss（NaN/Inf 字面量）。与 ``loss_unavailable`` 严格
    #: 区分：前者是"数值异常"这个事实断言，后者只是"没有读数"。
    loss_nonfinite: bool = False


def _rank_metric_steps(output_dir: Path | None) -> list[dict[str, Any]]:
    """从 rank_metrics 旁路读「每个训练步一行」的全局指标（PP trainer 的权威读数）。

    ``_emit_rank_memory_event("pipeline_sft_train_batch_after", ...)`` 每步写一条
    ``{"phase": ..., "metrics": {...}}``；``metrics`` 里带全局聚合口径
    （``global_loss_mean`` / ``global_grad_norm_mean`` / ``global_optimizer_steps_sum``）
    与逐 rank 明细 ``rank_metrics``。这解决了 F-4 的核心观测缺陷：
    stdout 的 ``sft_step`` 打的是 **rank0 局部** loss（PP 下结构性恒 0.0）与
    rank0 局部 grad_norm（有限），全局 NaN 只在旁路可见。
    """
    if output_dir is None:
        return []
    rows: list[dict[str, Any]] = []
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
            if payload.get("phase") != "pipeline_sft_train_batch_after":
                continue
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                rows.append(metrics)
    return rows


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
            f"stdout 与 rank_metrics 的 loss 口径不一致（步 {', '.join(str(i + 1) for i in mismatched[:5])}）："
            "stdout 为 rank0 局部值，全局口径以 rank_metrics 为准"
        ]
    return []


def extract_steps_and_series(output_dir: Path | None, log_text: str) -> SeriesEvidence:
    """抽取 optimizer step / epoch / loss 序列 / grad_norm 序列 / 逐步推进证据。

    来源优先级：
      1. ``rank_metrics.rank_*.jsonl`` 旁路的**全局**逐步指标（PP 唯一可信口径）；
      2. ``trainer_state.json``（HF/ms-swift 结构化记录）；
      3. 日志中的 dict 行（stdout 兜底，口径为 rank0 局部，显式标注）。
    """
    result = SeriesEvidence()
    metrics_rows = _rank_metric_steps(output_dir)
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
        result.notes.extend(_cross_check_stdout(log_text, result.losses))
        # epoch 仍从 trainer_state 取（旁路不含 epoch）。
        if output_dir is not None:
            for state_path in output_dir.rglob("trainer_state.json"):
                state = _read_json(state_path)
                if isinstance(state, dict) and isinstance(state.get("epoch"), (int, float)):
                    result.epochs = float(state["epoch"])
                    break
        return result

    if output_dir is not None:
        for state_path in output_dir.rglob("trainer_state.json"):
            state = _read_json(state_path)
            if not isinstance(state, dict):
                continue
            if isinstance(state.get("global_step"), int):
                result.steps = state["global_step"]
            if isinstance(state.get("epoch"), (int, float)):
                result.epochs = float(state["epoch"])
            for entry in state.get("log_history", []) or []:
                if not isinstance(entry, dict):
                    continue
                if isinstance(entry.get("loss"), (int, float)):
                    result.losses.append(float(entry["loss"]))
                if isinstance(entry.get("grad_norm"), (int, float)):
                    result.grad_norms.append(float(entry["grad_norm"]))
            if result.losses:
                break
        if result.losses:
            result.source = "trainer_state"

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
                result.losses.append(
                    loss if loss is not None else _judge.MISSING_SENTINEL
                )
                result.grad_norms.append(
                    grad if grad is not None else _judge.MISSING_SENTINEL
                )
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
    ) or any(
        isinstance(value, float) and not math.isfinite(value) for value in result.grad_norms
    )
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


def extract_weight_changed(output_dir: Path | None, tuner_type: str) -> bool | None:
    """抽取"权重是否真变化"证据（LoRA 看 lora_b 非零；全参看与基座是否不同）。

    抽取不到返回 ``None`` → A2 fail-closed 判不通过。
    """
    if output_dir is None:
        return None
    torch = _require_torch()
    finals = [p for p in output_dir.rglob("final") if p.is_dir()]
    for final in finals:
        if tuner_type == "lora":
            # native：单个 rank pt 里的 lora_state_dict
            for pt in sorted(final.glob("rank_*.pt")):
                if torch is None:
                    return None
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
            # msswift/peft：adapter_model.safetensors 里的 lora_B
            for weights in sorted(final.glob("adapter_model.safetensors")):
                result = _safetensors_any_nonzero(weights, key_filter="lora_b")
                if result is not None:
                    return result
            continue
        # 全参：终态权重与基座权重做抽样比较
        result = _full_weights_differ(final)
        if result is not None:
            return result
    return None


def _safetensors_any_nonzero(path: Path, *, key_filter: str) -> bool | None:
    """用 safetensors 检查指定关键字张量是否有非零值（避免整包载入显存）。"""
    try:
        from safetensors import safe_open  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with safe_open(str(path), framework="pt") as handle:
            keys = [key for key in handle.keys() if key_filter in key.lower()]
            if not keys:
                return None
            for key in keys:
                tensor = handle.get_tensor(key)
                if hasattr(tensor, "any") and bool((tensor != 0).any()):
                    return True
            return False
    except (OSError, RuntimeError, TypeError):
        return None


def _full_weights_differ(final_dir: Path) -> bool | None:
    """全参：终态 checkpoint 与基座模型权重抽样比较（不同 → 已更新）。"""
    import os  # noqa: PLC0415

    base = os.environ.get("GRASPO_BASE_MODEL_FOR_COMPARE")
    if not base:
        return None
    base_path = Path(base)
    if not base_path.exists():
        return None
    try:
        from safetensors import safe_open  # noqa: PLC0415

        final_files = sorted(final_dir.glob("*.safetensors"))
        base_files = sorted(base_path.glob("*.safetensors"))
        if not final_files or not base_files:
            return None
        with safe_open(str(final_files[0]), framework="pt") as final_handle:
            with safe_open(str(base_files[0]), framework="pt") as base_handle:
                shared = [k for k in final_handle.keys() if k in set(base_handle.keys())][:8]
                if not shared:
                    return None
                for key in shared:
                    left = final_handle.get_tensor(key)
                    right = base_handle.get_tensor(key)
                    if left.shape != right.shape:
                        return True
                    if bool((left != right).any()):
                        return True
        return False
    except (OSError, RuntimeError, TypeError):
        return None


def extract_checkpoint_reloadable(output_dir: Path | None) -> bool | None:
    """A3 证据：checkpoint 能否被重新加载（结构化校验 + 反序列化）。"""
    if output_dir is None:
        return None
    finals = [p for p in output_dir.rglob("final") if p.is_dir()]
    if not finals:
        return None
    checked = False
    for final in finals:
        # native：manifest.json + rank_*.pt 能否 torch.load
        if (final / "manifest.json").exists():
            torch = _require_torch()
            rank_files = sorted(final.glob("rank_*.pt"))
            if not rank_files:
                continue
            if torch is None:
                continue
            try:
                torch.load(rank_files[0], map_location="cpu", weights_only=False)
                return True
            except (RuntimeError, OSError, AttributeError):
                return False
        # peft / HF：safetensors 能否打开
        for weights in sorted(final.glob("*.safetensors")):
            try:
                from safetensors import safe_open  # noqa: PLC0415

                with safe_open(str(weights), framework="pt") as handle:
                    list(handle.keys())
                checked = True
                return True
            except ImportError:
                return None
            except (OSError, RuntimeError, TypeError):
                return False
    return None if not checked else False


def extract_artifacts(run_dir: Path, output_dir: Path | None, log_text: str) -> dict[str, bool]:
    """A5 四件套产物：config_backup / training_log / checkpoint / metrics。"""
    artifacts = {name: False for name in _judge.REQUIRED_ARTIFACTS}
    if output_dir is not None:
        artifacts["config_backup"] = (output_dir / "config.yaml").exists() or any(
            output_dir.rglob("config.json")
        )
        artifacts["training_log"] = any(output_dir.rglob("training.log")) or any(
            output_dir.rglob("train.log")
        )
        artifacts["checkpoint"] = any(p.is_dir() for p in output_dir.rglob("final")) or any(
            output_dir.rglob("checkpoint-*")
        )
        artifacts["metrics"] = any(output_dir.rglob("events.jsonl")) or any(
            output_dir.rglob("rank_metrics.rank_*.jsonl")
        ) or any(output_dir.rglob("trainer_state.json"))
    # 运行器级日志也算训练日志证据（容器 stdout 一定存在）
    if not artifacts["training_log"] and (run_dir / "stdout.log").exists():
        artifacts["training_log"] = True
    if not artifacts["metrics"] and log_text:
        artifacts["metrics"] = bool(_LOSS_ANY.search(log_text))
    return artifacts


def collect_run(run_dir: Path, tier_id: str, tuner_type: str) -> tuple[Any, SeriesEvidence]:
    """把一个运行目录抽成 ``RunEvidence`` + 读数口径自证（``SeriesEvidence``）。"""
    exit_code: int | None = None
    exit_path = run_dir / "exit_code"
    if exit_path.exists():
        raw = _read_text(exit_path).strip()
        if raw.lstrip("-").isdigit():
            exit_code = int(raw)
    log_text = _read_text(run_dir / "stdout.log")
    output_dir = find_output_dir(run_dir, tier_id)
    series = extract_steps_and_series(output_dir, log_text)
    timed_out = exit_code == 124 or bool(re.search(r"⏰|timeout: sending signal", log_text))
    evidence = _judge.RunEvidence(
        tier_id=tier_id,
        exit_code=exit_code,
        timed_out=timed_out,
        log_text=log_text,
        tuner_type=tuner_type,
        optimizer_steps=series.steps,
        epochs_completed=series.epochs,
        weight_changed=extract_weight_changed(output_dir, tuner_type),
        checkpoint_reloadable=extract_checkpoint_reloadable(output_dir),
        artifacts_present=extract_artifacts(run_dir, output_dir, log_text),
        losses=tuple(series.losses),
        grad_norms=tuple(series.grad_norms),
        optimizer_steps_per_step=(
            tuple(series.optimizer_steps_per_step)
            if series.optimizer_steps_per_step
            else None
        ),
        nonfinite_skips=series.nonfinite_skips,
        losses_nonfinite=series.loss_nonfinite,
        losses_unavailable=series.loss_unavailable,
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
                }
            )
            continue
        first, first_series = collect_run(first_dir, tier_id, tuner_type)
        second = None
        if args.rerun_root:
            second_dir = Path(args.rerun_root) / tier_id
            if second_dir.is_dir():
                second, _ = collect_run(second_dir, tier_id, tuner_type)
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
        records.append(row)

    jsonl = out_dir / "ledger.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    markdown = out_dir / "ledger.md"
    # `max_context` 的口径必须随数字一起显示（🔴-1）：同样是 "8192"，"实测通过"
    # 与"真 OOM 边界候选"的含义完全不同，只给数字会被下游读成"这个长度跑得通"。
    lines = [
        "| 条件档 | 模型 | 算法 | 模式 | 后端 | 卡数 | 最大可行上下文 | 口径 | 每卡峰值(GiB) | 状态 | 失败类型 | 备注 |",
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
            f"| {record['status']} | {record.get('failure_class') or '—'} | {record.get('note', '')} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed = sum(1 for record in records if record["status"] == "✅ 通过")
    failed = sum(1 for record in records if record["status"] == "❌ 失败")
    untested = len(records) - passed - failed
    print(f"ledger: {jsonl} ({len(records)} tiers: pass={passed} fail={failed} untested={untested})")
    print(f"markdown: {markdown}")
    return 0


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
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
