"""analyze-profile 的 rollout 归因分析（通用，与训练数据内容无关）。

读取 ``<run_dir>/logs/rollouts.readable.jsonl``，对每组 rollout 做
**结构层面**的归因：工具名、参数匹配、格式错误类型——不涉及任何
具体字段语义（不假设字段名、不假设数值字段存在、不做数值误差统计）。
标签格式常量从 ``qwen_tool_parser`` 单一来源导入（方案 A）。

输出四类：
1. not_correct 组原因分类（tool_mismatch / content_all_wrong / format_shortfall）
2. 工具名/参数匹配正确率趋势（按 step）
3. 决策 × 工具/参数匹配交叉表
4. completion 格式错误类型分布（按 epoch；多调用检测在 reward 层，
   parser 重解析看不到，须数标签——v21 实测多调用 41→244→380 增长）
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
    parse_qwen_tool_completion,
)


def latest_log_dir(run_dir: Path) -> Path:
    """Most recent ``logs/<run_id>/`` subdir, falling back to ``logs`` (flat layout).

    Since ``logs`` moved under a per-launch ``<run_id>`` folder, readers that
    need "the current run's logs" should resolve the newest run folder.
    """
    logs = run_dir / "logs"
    if not logs.exists():
        return logs
    subdirs = [d for d in logs.iterdir() if d.is_dir()]
    if not subdirs:
        return logs
    return max(subdirs, key=lambda d: d.stat().st_mtime)

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


# ── 格式错误类型分类（与数据内容无关，仅格式层语义）─────────────────────────


def classify_completion_error(text: str, tools: list[dict[str, Any]] | None) -> str:
    """对单条 completion 做格式错误分类（'ok' 或错误类型名）。

    分类规则（不涉及字段名/数值语义）：
    - parser 严格解析失败 → 按错误消息细分（no_tool_call / malformed_xml /
      missing_param / other_parse）
    - 解析成功但 ``<tool_call>`` 标签数 > 1 → multi_call（reward 层的
      "too many tool calls" 在 parser 层不可见，必须数标签）
    """
    parsed = parse_qwen_tool_completion(text, expect_tool_calls=True, tools=tools)
    if parsed.parse_errors:
        for err in parsed.parse_errors:
            if "no tool call" in err:
                return "no_tool_call"
            if "malformed XML" in err:
                return "malformed_xml"
            if "missing required parameter" in err:
                return "missing_param"
        return "other_parse"
    n_calls = text.count(TOOL_CALL_OPEN)
    if n_calls > 1:
        return "multi_call"
    return "ok"


_ERROR_TYPES = ("ok", "no_tool_call", "malformed_xml", "missing_param", "multi_call", "other_parse")


def _classify_errors_by_epoch(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """按 epoch 聚合 completion 格式错误类型计数。"""
    by_epoch: defaultdict[Any, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        tools = rec.get("tools")
        epoch = rec.get("epoch")
        for comp in rec.get("completions") or []:
            text = (comp.get("completion") if isinstance(comp, dict) else str(comp)) or ""
            by_epoch[epoch][classify_completion_error(text, tools)] += 1
    return {str(epoch): dict(counts) for epoch, counts in sorted(by_epoch.items())}


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
    readable_path = latest_log_dir(run_dir) / "rollouts.readable.jsonl"
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
        "error_types_by_epoch": _classify_errors_by_epoch(records),
    }


def _read_events(run_dir: Path, event_name: str) -> list[dict[str, Any]]:
    """读取 events.jsonl 中指定事件类型的全部事件（分析端只读）。

    损坏行跳过（与 analyze_attribution 的容错一致）；缺失文件返回空列表，
    由调用方给出可用性原因。
    """
    events_path = latest_log_dir(run_dir) / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("event") == event_name:
            events.append(payload)
    return events


def analyze_steps(run_dir: Path) -> dict[str, Any]:
    """step 粒度进度表（从 train_step 事件聚合）。

    与 :func:`analyze_epochs` 共享**同一度量元组**（表头一致、粒度不同）。
    每行一个 optimize 步（train_step 事件）。``samples_start/end`` 为同 epoch
    内 samples_seen 差分（epoch 切换时从 0 重新累计）——train_step 是 optimize
    次数、每步消费的样本数动态（v22 实测 1-6 组块/步），样本区间是定位
    "现在跑到哪"的锚点，不能只看 step 号。

    输出 ``{available, steps: [...]}``，steps 按事件顺序排列。
    """
    events = _read_events(run_dir, "train_step")
    if not events:
        return {"available": False, "reason": "no train_step events in events.jsonl"}

    rows: list[dict[str, Any]] = []
    prev_epoch: Any = None
    prev_seen = 0
    for ev in events:
        epoch = ev.get("epoch")
        ec = ev.get("epoch_cumulative") or {}
        seen = int(ec.get("samples_seen") or 0)
        if epoch != prev_epoch:
            prev_seen = 0
        samples_start, samples_end = prev_seen, seen
        prev_seen, prev_epoch = seen, epoch

        batch = ev.get("batch") or {}
        decisions = batch.get("decisions") or {}
        terminal = decisions.get("terminal") or {}
        trainable = decisions.get("trainable") or {}
        attempts = decisions.get("rollout_attempts") or {}
        retry = int(attempts.get("retry") or 0)
        terminal_total = int(terminal.get("total") or 0)
        health = ev.get("health") or {}
        timing = ev.get("timing") or {}
        optimize = ev.get("optimize") or {}
        alarms: dict[str, int] = {}
        for reason in health.get("reasons") or []:
            alarms[reason] = alarms.get(reason, 0) + 1

        rows.append(
            {
                "step": ev.get("step")
                if ev.get("step") is not None
                else (ev.get("run_cumulative") or {}).get("step"),
                "epoch": epoch,
                "samples_start": samples_start,
                "samples_end": samples_end,
                "steps": 1,
                "perfect": terminal.get("perfect_skip", 0),
                "invalid": terminal.get("invalid", 0),
                "no_gap": terminal.get("invalid_no_preference_gap", 0),
                "mc": trainable.get("max_correct", 0),
                "nc": trainable.get("not_correct", 0),
                "mc_ratio": trainable.get("ratio"),
                "reward_mean": batch.get("reward", {}).get("mean"),
                "content_mean": batch.get("content", {}).get("mean"),
                "loss_mean": optimize.get("loss_mean"),
                "retry_rate": round(retry / (retry + terminal_total), 4)
                if (retry + terminal_total) > 0
                else 0.0,
                "alarms": alarms,
                "total_sec": timing.get("total_observed_sec"),
            }
        )
    return {"available": True, "steps": rows}


def analyze_epochs(run_dir: Path) -> dict[str, Any]:
    """epoch 级聚合统计（从 events.jsonl 的 epoch_summary 事件）。

    每个 epoch 结束时训练循环写一条 ``epoch_summary`` 事件，携带该 epoch 的
    **全量累计统计**（samples/completions/decisions/reward/content）——这是
    epoch 整体统计的权威来源（train_step 是单步口径，epoch_cumulative 只在
    progress=1.0 时等于 epoch 全量，epoch_summary 显式固化这一点）。

    输出 ``{available, epochs: [...], trends: {...}}``，epochs 按事件顺序排列。
    """
    events_path = latest_log_dir(run_dir) / "events.jsonl"
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

    # loss_mean 补齐：epoch_summary 无此字段，从同 epoch 的 train_step 聚合均值；
    # alarms 分类计数：health.reasons 按 epoch 聚合
    train_steps = _read_events(run_dir, "train_step")
    loss_by_epoch: defaultdict[Any, list[float]] = defaultdict(list)
    alarm_by_epoch: defaultdict[Any, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ev in train_steps:
        ep = ev.get("epoch")
        loss = (ev.get("optimize") or {}).get("loss_mean")
        if loss is not None:
            loss_by_epoch[ep].append(float(loss))
        for reason in (ev.get("health") or {}).get("reasons") or []:
            alarm_by_epoch[ep][reason] += 1
    for e in epochs:
        ep = e["epoch"]
        losses = loss_by_epoch.get(ep)
        e["loss_mean"] = round(sum(losses) / len(losses), 6) if losses else None
        e["alarms"] = dict(alarm_by_epoch.get(ep, {}))
        e["samples_start"] = 0
        e["samples_end"] = e.get("samples_total")

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


# ── 错误原因表（L1 格式层 + L2 匹配层 + other，completion 级互斥分类）─────────

# 分类全集（互斥：每条 completion 恰好归一类）
ERROR_CATEGORIES = (
    "ok",
    "no_tool_call",
    "malformed_xml",
    "missing_param",
    "multi_call",
    "other_parse",
    "tool_mismatch",
    "param_name_mismatch",
    "param_value_mismatch",
    "other",
)


def classify_completion(record: dict[str, Any], text: str) -> str:
    """对单条 completion 做互斥分类（错误原因表的基本单元）。

    - other（优先判定）：无 tool target 可比对（纯文本/JSON 任务——L1/L2
      均以工具调用为期望，对它们无意义），或宽容提取为空
    - L1 格式层：parse 严格失败按错误消息细分（no_tool_call / malformed_xml /
      missing_param / other_parse）；parse 成功但 ``<tool_call>`` 标签数 > 1
      归 multi_call
    - L2 匹配层（parse 成功且单调用，宽容提取后与 target[0] 比对，仅名字/
      字符串比较——通用性铁律，不做数值误差统计）：
        tool_mismatch        工具名 != target 工具名
        param_name_mismatch  工具对，参数名集合不等（缺/多余/拼错）
        param_value_mismatch 工具对、参数名全对，但值文本 != target 值
        ok                   工具对 + 参数名对 + 值全对
    """
    targets = record.get("targets") or []
    if not targets:
        return "other"
    out = targets[0].get("output") or {}
    calls = out.get("tool_calls") or []
    if not calls or not calls[0].get("name"):
        return "other"
    parsed = parse_qwen_tool_completion(text, expect_tool_calls=True, tools=record.get("tools"))
    if parsed.parse_errors:
        for err in parsed.parse_errors:
            if "no tool call" in err:
                return "no_tool_call"
            if "malformed XML" in err:
                return "malformed_xml"
            if "missing required parameter" in err:
                return "missing_param"
        return "other_parse"
    if text.count(TOOL_CALL_OPEN) > 1:
        return "multi_call"
    t_fn = calls[0]["name"]
    t_args = calls[0].get("arguments") or {}
    extracted = _extract_tool_calls(text)
    if not extracted:
        return "other"
    first = extracted[0]
    if first["name"] != t_fn:
        return "tool_mismatch"
    if set(first["arguments"]) != set(t_args):
        return "param_name_mismatch"
    if all(field_score(first["arguments"][pname], gt, 0.0) >= 1.0 for pname, gt in t_args.items()):
        return "ok"
    return "param_value_mismatch"


def _aggregate_errors(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """按 step 和 epoch 双粒度聚合 completion 错误分类。

    只统计终态组（(sample_index, step) 取 attempt_number 最大），与归因口径
    一致；retry 中间态不计。每类带去重 sample_index 列表（衔接 rollouts
    详表的引用键，AI 从表上即可定位样本追查）。
    """
    terminal: dict[tuple[Any, Any], dict[str, Any]] = {}
    for rec in records:
        key = (rec.get("sample_index"), rec.get("step"))
        cur = terminal.get(key)
        if cur is None or rec.get("attempt_number", 0) >= cur.get("attempt_number", 0):
            terminal[key] = rec

    by_step: defaultdict[str, defaultdict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "samples": []})
    )
    by_epoch: defaultdict[str, defaultdict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "samples": []})
    )
    for key, rec in terminal.items():
        step, epoch = rec.get("step"), rec.get("epoch")
        for comp in rec.get("completions") or []:
            text = (comp.get("completion") if isinstance(comp, dict) else str(comp)) or ""
            category = classify_completion(rec, text)
            if step is not None:
                cell = by_step[str(step)][category]
                cell["count"] += 1
                if rec.get("sample_index") not in cell["samples"]:
                    cell["samples"].append(rec.get("sample_index"))
            if epoch is not None:
                cell = by_epoch[str(epoch)][category]
                cell["count"] += 1
                if rec.get("sample_index") not in cell["samples"]:
                    cell["samples"].append(rec.get("sample_index"))

    def _to_rows(
        agg: defaultdict[str, defaultdict[str, dict[str, Any]]],
        granularity: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bucket in sorted(agg, key=int):
            for category in ERROR_CATEGORIES:
                cell = agg[bucket].get(category)
                if cell is None:
                    continue
                rows.append(
                    {
                        "granularity": granularity,
                        "bucket": int(bucket),
                        "category": category,
                        "count": cell["count"],
                        "samples": cell["samples"],
                    }
                )
        return rows

    return {
        "by_step": _to_rows(by_step, "step"),
        "by_epoch": _to_rows(by_epoch, "epoch"),
    }


def analyze_errors(run_dir: Path) -> dict[str, Any]:
    """错误原因统计（completion 级互斥分类，step/epoch 双粒度）。

    数据源 ``rollouts.readable.jsonl``；只统计终态组。L3（语义/数值层）
    诊断不在系统级归因内——由 AI/人工基于 rollouts 详表离线统计
    （用户裁定：真正的原因归类依赖训练数据语义，不可能系统级）。
    """
    readable_path = latest_log_dir(run_dir) / "rollouts.readable.jsonl"
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
    if not records:
        return {"available": True, "records": 0, "by_step": [], "by_epoch": []}
    agg = _aggregate_errors(records)
    return {"available": True, "records": len(records), **agg}


# ── 性能表（初级：只聚合 train_step timing 块，零训练侵入）───────────────────


def _perf_row_from_step(ev: dict[str, Any]) -> dict[str, Any]:
    """从单条 train_step 事件提取性能行（step 粒度）。"""
    batch = ev.get("batch") or {}
    decisions = batch.get("decisions") or {}
    attempts = decisions.get("rollout_attempts") or {}
    terminal = decisions.get("terminal") or {}
    timing = ev.get("timing") or {}
    retry = int(attempts.get("retry") or 0)
    terminal_total = int(terminal.get("total") or 0)
    rollout_sec = float(timing.get("rollout_total_sec") or 0.0)
    queue_sec = float(timing.get("rollout_queue_sec") or 0.0)
    decode_sec = float(timing.get("decode_sec") or 0.0)
    decode_tokens = int(timing.get("decode_tokens") or 0)
    return {
        "step": ev.get("step")
        if ev.get("step") is not None
        else (ev.get("run_cumulative") or {}).get("step"),
        "epoch": ev.get("epoch"),
        "groups": terminal_total,
        "total_sec": timing.get("total_observed_sec"),
        "rollout_sec": round(rollout_sec, 3),
        "queue_pct": round(queue_sec / rollout_sec * 100, 1) if rollout_sec > 0 else None,
        "prefill_sec": timing.get("prefill_sec"),
        "decode_sec": round(decode_sec, 3),
        "throughput_tok_s": round(decode_tokens / decode_sec, 1) if decode_sec > 0 else None,
        "optimize_sec": timing.get("optimize_sec"),
        "retry_rate": round(retry / (retry + terminal_total), 4)
        if (retry + terminal_total) > 0
        else 0.0,
    }


def analyze_perf(run_dir: Path) -> dict[str, Any]:
    """性能统计（step 粒度 + epoch 聚合，双粒度）。

    只聚合 train_step 事件 timing 块已有字段——零训练侵入、不影响训练速度。
    GPU 利用率/显存/温度不在本表（``graspo record-gpu-memory`` 的领域，
    高级性能分析走独立命令）。
    """
    events = _read_events(run_dir, "train_step")
    if not events:
        return {"available": False, "reason": "no train_step events in events.jsonl"}

    by_step = [_perf_row_from_step(ev) for ev in events]
    for row in by_step:
        row["granularity"] = "step"

    # epoch 聚合：耗时字段求和，比率字段重算（不用均值——口径一致）
    sums: defaultdict[Any, dict[str, float]] = defaultdict(
        lambda: {
            "groups": 0.0,
            "total_sec": 0.0,
            "rollout_sec": 0.0,
            "queue_sec": 0.0,
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "decode_tokens": 0.0,
            "optimize_sec": 0.0,
            "retry": 0.0,
            "terminal": 0.0,
        }
    )
    for ev in events:
        ep = ev.get("epoch")
        timing = ev.get("timing") or {}
        batch = ev.get("batch") or {}
        decisions = batch.get("decisions") or {}
        attempts = decisions.get("rollout_attempts") or {}
        terminal = decisions.get("terminal") or {}
        s = sums[ep]
        s["groups"] += float(terminal.get("total") or 0)
        s["total_sec"] += float(timing.get("total_observed_sec") or 0.0)
        s["rollout_sec"] += float(timing.get("rollout_total_sec") or 0.0)
        s["queue_sec"] += float(timing.get("rollout_queue_sec") or 0.0)
        s["prefill_sec"] += float(timing.get("prefill_sec") or 0.0)
        s["decode_sec"] += float(timing.get("decode_sec") or 0.0)
        s["decode_tokens"] += float(timing.get("decode_tokens") or 0)
        s["optimize_sec"] += float(timing.get("optimize_sec") or 0.0)
        s["retry"] += float(attempts.get("retry") or 0)
        s["terminal"] += float(terminal.get("total") or 0)

    by_epoch: list[dict[str, Any]] = []
    for ep in sorted(sums, key=int):
        s = sums[ep]
        by_epoch.append(
            {
                "granularity": "epoch",
                "bucket": ep,
                "groups": int(s["groups"]),
                "total_sec": round(s["total_sec"], 3),
                "rollout_sec": round(s["rollout_sec"], 3),
                "queue_pct": round(s["queue_sec"] / s["rollout_sec"] * 100, 1)
                if s["rollout_sec"] > 0
                else None,
                "prefill_sec": round(s["prefill_sec"], 3),
                "decode_sec": round(s["decode_sec"], 3),
                "throughput_tok_s": round(s["decode_tokens"] / s["decode_sec"], 1)
                if s["decode_sec"] > 0
                else None,
                "optimize_sec": round(s["optimize_sec"], 3),
                "retry_rate": round(s["retry"] / (s["retry"] + s["terminal"]), 4)
                if (s["retry"] + s["terminal"]) > 0
                else 0.0,
            }
        )
    return {"available": True, "by_step": by_step, "by_epoch": by_epoch}
