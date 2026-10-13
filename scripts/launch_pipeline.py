#!/usr/bin/env python3
"""启动流水线（生成 → 护栏 → 起跑 → 监控 → 评测 → 落 json）的可测骨架。

为什么有这件：228 战役里同一"护栏邻域"连踩 5 次坑（§8 断言位置错 → §7 谓词过窄 →
护栏件未同步部署树 → 评测脚本生成顺序错 → 生成链变量为空）。根因是**启动脚本每轮临时写**。
本模块把这条链**固化成一件可测的东西**：纯函数负责"生成前/起跑前"的全部判定与断言，
调用方（launcher / 评测脚本）只负责执行 docker 与 await。

设计要点（与 docs/run-discipline.md §7/§8/§10/§11 一一对应）：
  · §11  仓内件必须同批同步部署树：`require_tooling()` 探测依赖脚本存在性，缺失即 fail-closed；
  · 变量非空断言：`require_non_empty()` 对基座/checkpoint 等关键变量逐条断言；
  · §7   形态谓词：委托 `run_guards.detect_form_and_validate`（按形态分支）；
  · §8   新鲜度：委托 `run_guards.check_fresh`（pre/post）；
  · §10  Mamba：委托 `run_guards.check_mamba`。

用法（骨架示例）：
    plan = build_plan(tier="T008", checkpoint="…/checkpoint-100", base_model="/models/Qwen3.8-27B",
                      merge_dir="…/merged-hf-t008", tree="/data/…/graspo-b11fix-20261012",
                      required_tools=[…/scripts/run_guards.py], since_epoch=…)
    problems = validate_pre(plan, phase="pre")     # 起跑前与导出前的全部断言
    …执行 export…  problems = validate_post(plan)  # 导出后的新鲜度断言
    …执行 vLLM/eval/delta… reason = build_json_reason(...)

退出码约定：本模块只返回问题列表；由调用方决定 fail-closed（`raise` 或 `sys.exit(1)`）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_guards as guards  # noqa: E402


@dataclass
class Plan:
    tier: str
    checkpoint: str
    base_model: str
    merge_dir: str
    tree: str
    since_epoch: float
    required_tools: list[str] = field(default_factory=list)
    max_num_seqs: int | None = None
    mamba_blocks: int | None = None


def require_non_empty(name: str, value: str | None) -> list[str]:
    """变量非空断言（228 实测：空 `MP` 导致 export 2 s 即失败）。"""
    if value is None or not str(value).strip():
        return [f"变量 {name} 为空 ⇒ fail-closed（变量非空断言）"]
    return []


def require_tooling(tree: str, required_tools: list[str]) -> list[str]:
    """§11：调用方必须先探测依赖脚本存在性，缺失即 fail-closed。"""
    problems: list[str] = []
    for tool in required_tools:
        if not os.path.exists(tool):
            problems.append(f"§11 失败：依赖件不存在 {tool}（仓内件未同步到部署树？）⇒ fail-closed")
    return problems


def validate_pre(plan: Plan, phase: str = "pre") -> list[str]:
    """起跑前（phase=pre）的全部断言：变量非空 → §11 依赖 → §7 形态 → §8(pre) → §10。"""
    problems: list[str] = []
    problems += require_non_empty("checkpoint", plan.checkpoint)
    problems += require_non_empty("base_model", plan.base_model)
    problems += require_non_empty("merge_dir", plan.merge_dir)
    if problems:
        return problems
    problems += require_tooling(plan.tree, plan.required_tools)
    _, ok, msg = guards.detect_form_and_validate(plan.checkpoint)
    if not ok:
        problems.append(f"§7 失败：{msg}")
    ok_fresh, msg_fresh = guards.check_fresh(plan.merge_dir, "pre")
    if not ok_fresh:
        problems.append(msg_fresh)
    if plan.max_num_seqs is not None and plan.mamba_blocks is not None:
        ok_m, msg_m = guards.check_mamba(plan.max_num_seqs, plan.mamba_blocks)
        if not ok_m:
            problems.append(msg_m)
    elif plan.max_num_seqs is not None:
        problems.append("§10 失败：给了 max_num_seqs 但缺 mamba_blocks（无法断言 ⇒ fail-closed）")
    return problems


def validate_post(plan: Plan) -> list[str]:
    """导出后的断言：§8(post)（存在 ∧ 含权重 ∧ 创建时刻 ≥ 本批起跑）。"""
    ok, msg = guards.check_fresh(plan.merge_dir, "post", since_epoch=plan.since_epoch)
    return [] if ok else [msg]


def build_json_reason(after_correct: int, n_total: int, base_correct: int, status: str) -> str:
    """落 json 的备注（≤100 字）：判定 + 读数，与结构化字段同源。"""
    after = after_correct / n_total * 100 if n_total else 0.0
    base = base_correct / n_total * 100 if n_total else 0.0
    delta = after - base
    return f"{status}：after {after:.4f}%（{after_correct}/{n_total}）/ base {base:.4f}%（{base_correct}/{n_total}）/ Δ {delta:+.2f} pp"[:100]
