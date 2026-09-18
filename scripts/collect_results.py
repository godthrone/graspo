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
import re
import sys
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


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")


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


def extract_steps_and_series(output_dir: Path | None, log_text: str) -> tuple[int | None, float | None, list[float], list[float]]:
    """抽取 optimizer step / epoch / loss 序列 / grad_norm 序列。

    来源优先级：trainer_state.json（HF/ms-swift 结构化记录）→ 日志中的 dict 行。
    """
    steps: int | None = None
    epochs: float | None = None
    losses: list[float] = []
    grad_norms: list[float] = []

    if output_dir is not None:
        for state_path in output_dir.rglob("trainer_state.json"):
            state = _read_json(state_path)
            if not isinstance(state, dict):
                continue
            if isinstance(state.get("global_step"), int):
                steps = state["global_step"]
            if isinstance(state.get("epoch"), (int, float)):
                epochs = float(state["epoch"])
            for entry in state.get("log_history", []) or []:
                if not isinstance(entry, dict):
                    continue
                if isinstance(entry.get("loss"), (int, float)):
                    losses.append(float(entry["loss"]))
                if isinstance(entry.get("grad_norm"), (int, float)):
                    grad_norms.append(float(entry["grad_norm"]))
            if losses:
                break

    for match in _LOSS_DICT.finditer(log_text):
        losses.append(_to_float(match.group(1)))
        grad_norms.append(_to_float(match.group(2)))
    if not losses:
        losses.extend(_to_float(m.group(1)) for m in _LOSS_ANY.finditer(log_text))
    if not grad_norms:
        grad_norms.extend(_to_float(m.group(1)) for m in _GRAD_ANY.finditer(log_text))
    if steps is None:
        matches = _GLOBAL_STEP.findall(log_text)
        if matches:
            steps = int(matches[-1])
    return steps, epochs, losses, grad_norms


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


def collect_run(run_dir: Path, tier_id: str, tuner_type: str) -> Any:
    """把一个运行目录抽成 ``RunEvidence``。"""
    exit_code: int | None = None
    exit_path = run_dir / "exit_code"
    if exit_path.exists():
        raw = _read_text(exit_path).strip()
        if raw.lstrip("-").isdigit():
            exit_code = int(raw)
    log_text = _read_text(run_dir / "stdout.log")
    output_dir = find_output_dir(run_dir, tier_id)
    steps, epochs, losses, grad_norms = extract_steps_and_series(output_dir, log_text)
    timed_out = exit_code == 124 or bool(re.search(r"⏰|timeout: sending signal", log_text))
    return _judge.RunEvidence(
        tier_id=tier_id,
        exit_code=exit_code,
        timed_out=timed_out,
        log_text=log_text,
        tuner_type=tuner_type,
        optimizer_steps=steps,
        epochs_completed=epochs,
        weight_changed=extract_weight_changed(output_dir, tuner_type),
        checkpoint_reloadable=extract_checkpoint_reloadable(output_dir),
        artifacts_present=extract_artifacts(run_dir, output_dir, log_text),
        losses=tuple(losses),
        grad_norms=tuple(grad_norms),
    )


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
                }
            )
            continue
        first = collect_run(first_dir, tier_id, tuner_type)
        second = None
        if args.rerun_root:
            second_dir = Path(args.rerun_root) / tier_id
            if second_dir.is_dir():
                second = collect_run(second_dir, tier_id, tuner_type)
        judgement = _judge.judge_tier(first, second, context_length=args.context_length)
        row = _judge.ledger_row(
            judgement,
            model=str(tier.get("model")),
            algorithm=str(tier.get("algorithm")),
            mode=str(tier.get("mode")),
            backend=str(tier.get("backend")),
            cards=int(tier.get("cards", 0)),
            max_context=args.context_length if judgement.counts_toward_max_context else None,
            peak_memory_gib=_read_peak_memory(first_dir),
            date=args.date,
        )
        row["criteria"] = {item.criterion: item.passed for item in judgement.criteria}
        row["criteria_detail"] = {item.criterion: item.detail for item in judgement.criteria}
        records.append(row)

    jsonl = out_dir / "ledger.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    markdown = out_dir / "ledger.md"
    lines = [
        "| 条件档 | 模型 | 算法 | 模式 | 后端 | 卡数 | 最大可行上下文 | 每卡峰值(GiB) | 状态 | 失败类型 | 备注 |",
        "|---|---|---|---|:--:|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['tier_id']} | {record.get('model', '')} | {record.get('algorithm', '')} "
            f"| {record.get('mode', '')} | {record.get('backend', '')} | {record.get('cards', '')} "
            f"| {record.get('max_context') or '—'} | {record.get('peak_memory_gib') or '—'} "
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
    parser.add_argument("--context-length", type=int, default=None, help="Tested context length.")
    parser.add_argument("--date", default="", help="Ledger date (YYYY-MM-DD).")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
