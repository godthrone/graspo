"""analyze-profile 的 rollout 归因分析（通用，与训练数据内容无关）。

读取 ``<run_dir>/logs/rollouts.readable.jsonl``，对每组 rollout 做
**结构层面**的归因：工具名、参数匹配——不涉及任何具体字段语义
（不假设字段名、不假设数值字段存在、不做数值误差统计）。
标签格式常量从 ``qwen_tool_parser`` 单一来源导入（方案 A）。

输出三类：
1. not_correct 组原因分类（tool_mismatch / content_all_wrong / format_shortfall）
2. 工具名/参数匹配正确率趋势（按 step）
3. 决策 × 工具/参数匹配交叉表
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from graspo.ripple.annotation.advantages import field_score
from graspo.ripple.parsing.qwen_tool_parser import (
    PARAMETER_CLOSE,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
)

# 宽容提取用的正则（只做存在性分析，不校验结构）
_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>")
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>")


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """从 completion 文本宽容提取 ``(name, {param: value_text})``。

    与训练判定的严格解析（``parse_qwen_tool_completion``）不同：本函数
    只做存在性提取，任何残损文本都能给出部分信息，用于归因画像。
    """
    calls: list[dict[str, Any]] = []
    for m in re.finditer(
        re.escape(TOOL_CALL_OPEN) + r"(.*?)(?:" + re.escape(TOOL_CALL_CLOSE) + r"|$)",
        text,
        re.DOTALL,
    ):
        body = m.group(1)
        fn = _FUNCTION_RE.search(body)
        if not fn:
            continue
        name = fn.group(1).strip()
        params: dict[str, str] = {}
        for p in _PARAMETER_RE.finditer(body):
            pname = p.group(1).strip()
            # 参数值取到对应 </parameter> 闭合（没有闭合则到下一个 <parameter= 或末尾）
            end = body.find(PARAMETER_CLOSE, p.end())
            if end == -1:
                end = len(body)
            params[pname] = body[p.end() : end].strip()
        if name and params:
            calls.append({"name": name, "arguments": params})
    return calls


def _has_comparable_target(record: dict[str, Any]) -> bool:
    """记录是否有可比对的 tool_call target（无 targets 的非工具任务跳过）。"""
    targets = record.get("targets") or []
    if not targets:
        return False
    out = targets[0].get("output") or {}
    calls = out.get("tool_calls") or []
    return bool(calls and calls[0].get("name"))


def _group_match(record: dict[str, Any]) -> tuple[int, int]:
    """计算一组 rollout 的工具名/参数匹配条数。

    :return: ``(fn_ok, param_ok)``
        fn_ok: 工具名与 target 一致的条数
        param_ok: 工具名一致且全部 target 参数精确匹配的条数
    """
    targets = record.get("targets") or []
    if not targets:
        return 0, 0
    out = targets[0].get("output") or {}
    calls = out.get("tool_calls") or []
    if not calls:
        return 0, 0
    t_fn = calls[0].get("name")
    t_args = calls[0].get("arguments") or {}
    if not t_fn:
        return 0, 0

    completions = record.get("completions") or []
    fn_ok = param_ok = 0
    for comp in completions:
        text = (comp.get("completion") if isinstance(comp, dict) else str(comp)) or ""
        extracted = _extract_tool_calls(text)
        if not extracted:
            continue
        first = extracted[0]
        if first["name"] != t_fn:
            continue
        fn_ok += 1
        if all(
            pname in first["arguments"] and field_score(first["arguments"][pname], gt, 0.0) >= 1.0
            for pname, gt in t_args.items()
        ):
            param_ok += 1
    return fn_ok, param_ok


def analyze_attribution(run_dir: Path) -> dict[str, Any]:
    """对一次运行目录做 rollout 归因（只读不写）。"""
    readable_path = run_dir / "logs" / "rollouts.readable.jsonl"
    if not readable_path.exists():
        return {"available": False, "reason": f"missing {readable_path}"}

    records: list[dict[str, Any]] = []
    for line in readable_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue

    # 按 (sample_index, step) 取终态（attempt_number 最大）
    terminal: dict[tuple[Any, Any], dict[str, Any]] = {}
    for rec in records:
        key = (rec.get("sample_index"), rec.get("step"))
        cur = terminal.get(key)
        if cur is None or rec.get("attempt_number", 0) >= cur.get("attempt_number", 0):
            terminal[key] = rec

    nc_causes: defaultdict[str, int] = defaultdict(int)  # not_correct 原因
    by_step: defaultdict[Any, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "fn_ok": 0, "param_ok": 0, "any_fn": 0, "any_param": 0}
    )
    cross: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "any_fn": 0, "any_param": 0, "all_fn": 0, "all_param": 0}
    )

    for key, rec in sorted(terminal.items()):
        decision = rec.get("decision")
        if not _has_comparable_target(rec):
            continue
        fn_ok, param_ok = _group_match(rec)
        step = rec.get("step")
        if step is not None and decision not in ("retry", "perfect_skip"):
            st = by_step[step]
            st["n"] += 1
            st["fn_ok"] += fn_ok
            st["param_ok"] += param_ok
            if fn_ok > 0:
                st["any_fn"] += 1
            if param_ok > 0:
                st["any_param"] += 1
        if decision:
            cx = cross[decision]
            cx["n"] += 1
            if fn_ok > 0:
                cx["any_fn"] += 1
            if param_ok > 0:
                cx["any_param"] += 1
            if fn_ok == 8:
                cx["all_fn"] += 1
            if param_ok == 8:
                cx["all_param"] += 1
        if decision == "trainable_not_correct":
            if fn_ok == 0:
                nc_causes["tool_mismatch"] += 1
            elif param_ok == 0:
                nc_causes["content_all_wrong"] += 1
            else:
                nc_causes["format_shortfall"] += 1

    step_rows = []
    for step in sorted(by_step):
        st = by_step[step]
        step_rows.append(
            {
                "step": step,
                "groups": st["n"],
                "fn_mean": round(st["fn_ok"] / st["n"], 3),
                "param_mean": round(st["param_ok"] / st["n"], 3),
                "any_fn_ratio": round(st["any_fn"] / st["n"], 3),
                "any_param_ratio": round(st["any_param"] / st["n"], 3),
            }
        )
    cross_rows = {}
    for decision in sorted(cross):
        cx = cross[decision]
        cross_rows[decision] = {
            "groups": cx["n"],
            "any_fn_ratio": round(cx["any_fn"] / cx["n"], 3) if cx["n"] else None,
            "any_param_ratio": round(cx["any_param"] / cx["n"], 3) if cx["n"] else None,
            "all_fn_ratio": round(cx["all_fn"] / cx["n"], 3) if cx["n"] else None,
            "all_param_ratio": round(cx["all_param"] / cx["n"], 3) if cx["n"] else None,
        }

    total_nc = sum(nc_causes.values())
    return {
        "available": True,
        "records": len(records),
        "terminal_groups": len(terminal),
        "not_correct_groups": total_nc,
        "not_correct_causes": dict(nc_causes),
        "step_trend": step_rows,
        "decision_cross": cross_rows,
    }


def print_attribution(attribution: dict[str, Any]) -> None:
    """以人类可读表格打印归因结果。"""
    if not attribution.get("available"):
        print(f"[attribution] unavailable: {attribution.get('reason')}")
        return
    print()
    print(f"=== rollout 归因（{attribution.get('terminal_groups')} 终态组）===")
    causes = attribution.get("not_correct_causes") or {}
    total = attribution.get("not_correct_groups") or 0
    print(f"not_correct 组: {total}")
    for name in ("tool_mismatch", "content_all_wrong", "format_shortfall"):
        n = causes.get(name, 0)
        pct = f"{n / total * 100:.0f}%" if total else "-"
        print(f"  {name:<22} {n:>5} ({pct})")

    trend = attribution.get("step_trend") or []
    if trend:
        print()
        print("工具/参数匹配趋势（按 step）:")
        print("step | 组数 | 工具对均值/8 | ≥1工具对 | ≥1参数对")
        for row in trend[-12:]:  # 只看最近 12 步，早期噪声大
            print(
                f"{row['step']:>4} | {row['groups']:>4} | "
                f"{row['fn_mean']:>9.2f} | {row['any_fn_ratio'] * 100:>6.0f}% | "
                f"{row['any_param_ratio'] * 100:>6.0f}%"
            )

    cross = attribution.get("decision_cross") or {}
    if cross:
        print()
        print("决策 × 工具/参数匹配:")
        print("decision | 组数 | ≥1工具对 | ≥1参数对 | 8条全对工具 | 8条全对参数")
        for decision, row in cross.items():
            any_fn = row["any_fn_ratio"] * 100 if row["any_fn_ratio"] is not None else 0
            any_param = row["any_param_ratio"] * 100 if row["any_param_ratio"] is not None else 0
            all_fn = row["all_fn_ratio"] * 100 if row["all_fn_ratio"] is not None else 0
            all_param = row["all_param_ratio"] * 100 if row["all_param_ratio"] is not None else 0
            print(
                f"{decision:<24} | {row['groups']:>3} | "
                f"{any_fn:>6.0f}% | {any_param:>6.0f}% | "
                f"{all_fn:>8.0f}% | {all_param:>8.0f}%"
            )


def analyze_epochs(run_dir: Path) -> dict[str, Any]:
    """epoch 级聚合统计（从 events.jsonl 的 epoch_summary 事件）。

    每个 epoch 结束时训练循环写一条 ``epoch_summary`` 事件，携带该 epoch 的
    **全量累计统计**（samples/completions/decisions/reward/content）——这是
    epoch 整体统计的权威来源（train_step 是单步口径，epoch_cumulative 只在
    progress=1.0 时等于 epoch 全量，epoch_summary 显式固化这一点）。

    输出 ``{available, epochs: [...], trends: {...}}``，epochs 按事件顺序排列。
    """
    events_path = run_dir / "logs" / "events.jsonl"
    if not events_path.exists():
        return {"available": False, "reason": f"missing {events_path}"}

    epochs: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("event") != "epoch_summary":
            continue
        ec = payload.get("epoch_cumulative") or {}
        d = ec.get("decisions") or {}
        epochs.append(
            {
                "epoch": payload.get("epoch"),
                "elapsed_sec": payload.get("elapsed_sec"),
                "samples_seen": ec.get("samples_seen"),
                "samples_total": ec.get("samples_total"),
                "progress": ec.get("progress"),
                "attempts": d.get("rollout_attempts") or {},
                "terminal": d.get("terminal") or {},
                "trainable": d.get("trainable") or {},
                "reward_mean": ec.get("reward_mean"),
                "content_mean": ec.get("content_mean"),
                "base_content_mean": ec.get("base_content_mean"),
                "best_reward": ec.get("best_reward"),
            }
        )

    if not epochs:
        return {"available": False, "reason": "no epoch_summary events in events.jsonl"}

    def _trend(key: str) -> list[float | int | None]:
        return [e.get(key) for e in epochs]

    return {
        "available": True,
        "epochs": epochs,
        "trends": {
            "mc_ratio": [round((e.get("trainable") or {}).get("ratio") or 0.0, 4) for e in epochs],
            "reward_mean": _trend("reward_mean"),
            "content_mean": _trend("content_mean"),
            "base_content_mean": _trend("base_content_mean"),
            "invalid": [(e.get("terminal") or {}).get("invalid") for e in epochs],
            "max_correct": [(e.get("trainable") or {}).get("max_correct") for e in epochs],
            "not_correct": [(e.get("trainable") or {}).get("not_correct") for e in epochs],
            "perfect_skip": [(e.get("terminal") or {}).get("perfect_skip") for e in epochs],
        },
    }


def print_epochs(epochs: dict[str, Any]) -> None:
    """以人类可读表格打印 epoch 聚合结果。"""
    if not epochs.get("available"):
        print(f"[epochs] unavailable: {epochs.get('reason')}")
        return
    print()
    print("=== epoch 聚合（epoch_summary 事件）===")
    hdr = (
        "epoch | 样本 | retry% | perfect | invalid | no_gap | mc | nc "
        "| mc_ratio | reward | content | 耗时"
    )
    print(hdr)
    for e in epochs.get("epochs", []):
        attempts = e.get("attempts") or {}
        total = attempts.get("total") or 0
        retry = attempts.get("retry") or 0
        retry_pct = f"{retry / total * 100:.0f}%" if total else "-"
        term = e.get("terminal") or {}
        tr = e.get("trainable") or {}
        elapsed = e.get("elapsed_sec")
        elapsed_str = f"{elapsed / 3600:.1f}h" if elapsed else "-"
        print(
            f"{e.get('epoch'):>5} | {e.get('samples_seen'):>4} | {retry_pct:>6} | "
            f"{term.get('perfect_skip', 0):>7} | {term.get('invalid', 0):>7} | "
            f"{term.get('invalid_no_preference_gap', 0):>6} | "
            f"{tr.get('max_correct', 0):>2} | {tr.get('not_correct', 0):>2} | "
            f"{tr.get('ratio', 0):>8.3f} | {e.get('reward_mean', 0):>7.3f} | "
            f"{e.get('content_mean', 0):>7.3f} | {elapsed_str:>5}"
        )
