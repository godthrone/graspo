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


#: 容器内专有路径前缀 —— ⛔ 绝不能当宿主机挂载源（宿主上不存在 ⇒ docker 建空目录）
CONTAINER_ONLY_PREFIXES = ("/model", "/models/", "/evalout", "/workspace/graspo", "/runs/", "/work/elam-v5")


def check_mount_source(source: str) -> tuple[bool, str]:
    """挂载源**必须是宿主机路径**且**在宿主上存在**（T024 第 6 次"路径域"错误的收口）。

    ⛔ 拒绝容器内专有路径（`/models/...`、`/model`、`/evalout`、`/workspace/graspo`、`/runs/...`、
    `/work/elam-v5`）——它们只在容器里存在；写进 `-v <src>:<dst>` 会让 docker **自动建空目录**，
    随后 vLLM 报 `Invalid repository ID or local directory specified: '/model'`（引擎起不来）。
    """
    if not source or not str(source).strip():
        return (False, "挂载源为空 ⇒ fail-closed")
    src = str(source)
    for prefix in CONTAINER_ONLY_PREFIXES:
        if src == prefix or src.startswith(prefix.rstrip("/") + "/") or src == prefix.rstrip("/"):
            return (False, f"挂载源 {src} 是**容器内专有路径**（宿主不存在）⇒ fail-closed；请改用宿主路径")
    if not os.path.exists(src):
        return (False, f"挂载源 {src} 在宿主上不存在 ⇒ fail-closed（docker 会建空目录）")
    return (True, f"挂载源 {src} 为宿主路径且存在 ✓")


def check_mount_sources(sources: list[str]) -> list[str]:
    problems: list[str] = []
    for src in sources:
        ok, msg = check_mount_source(src)
        if not ok:
            problems.append(msg)
    return problems


#: 评测容器内可达的挂载点前缀（run-eval.sh 提供）—— §7 的"可达性"只认这些
EVAL_CONTAINER_PREFIXES = ("/evalout/", "/runs/")


def check_merged_artifact(host_merge_dir: str, eval_container_path: str) -> list[str]:
    """§7 扩到**合并产物**：merged 目录本身要过形态谓词，且**评测配置里写的路径必须评测容器内可达**。

    为什么单列：228 上 §7 的坑在"训练 checkpoint"与"合并产物"两个路径上**各踩过一次**
    （T004/T005 修了前者，T024 的 `eval-t024-after.yaml` 里又出现 `checkpoint_path: /model`）。
    规则：
      · **宿主侧** `host_merge_dir` 必须存在 ∧ 过 `run_guards.detect_form_and_validate`（至少含权重）；
      · **评测配置侧** `eval_container_path` 必须是**评测容器内**路径（`/evalout/...` 或 `/runs/...`），
        ⛔ 不得是 vLLM 容器里的 `/model`，⛔ 也不得是宿主路径（评测容器看不到宿主路径）。
    """
    problems: list[str] = []
    if not host_merge_dir or not eval_container_path:
        return ["§7 失败：merged 宿主目录或评测容器内路径为空 ⇒ fail-closed"]
    form, ok, msg = guards.detect_form_and_validate(host_merge_dir)
    if not ok:
        problems.append(f"§7 失败（merged 宿主侧）：{msg}")
    if eval_container_path == "/model" or eval_container_path.startswith("/model/"):
        problems.append(f"§7 失败：评测配置的 checkpoint_path={eval_container_path} 是 **vLLM 容器路径** ⇒ fail-closed")
    elif not any(eval_container_path.startswith(pfx) for pfx in EVAL_CONTAINER_PREFIXES):
        problems.append(
            f"§7 失败：评测配置的 checkpoint_path={eval_container_path} 不在评测容器可达前缀内"
            f"（{EVAL_CONTAINER_PREFIXES}）⇒ fail-closed"
        )
    return problems


#: 模型路径的"容器内 → 宿主"映射根（228）：容器里看到 /models，宿主上是 /home/user/models
MODEL_CONTAINER_ROOT = "/models/"
MODEL_HOST_ROOT = "/home/user/models/"


def resolve_base_mount_source(model_path: str, host_root: str = MODEL_HOST_ROOT) -> tuple[str | None, str]:
    """把"本档基座路径"解析成**可用于 `docker run -v <src>:<dst>` 的宿主机路径**。

    ★ 这是 228 上一次事故 + 一次险些事故的收口：`$MP`（本档 config 里的 `model_path`）是
    **容器内路径**（如 `/models/Qwen3.8-27B`），直接当挂载源写进 `-v` 会让 docker 建空目录，
    随后 vLLM 报 `Invalid repository ID or local directory specified: '/model'`（T024 实测）。
    规则：`/models/X` ⇒ `<host_root>X`（默认 `/home/user/models/X`）并断言其存在；
    若传入的已经是宿主路径则直接校验存在性。
    """
    if not model_path or not str(model_path).strip():
        return (None, "基座路径为空 ⇒ fail-closed（变量非空断言）")
    src = str(model_path).strip()
    if src.startswith(MODEL_CONTAINER_ROOT):
        src = os.path.join(host_root, src[len(MODEL_CONTAINER_ROOT):])
    ok, msg = check_mount_source(src)
    if not ok:
        return (None, msg)
    return (src, msg)


def render_vllm_base_mount(model_path: str, container_dst: str = "/model", host_root: str = MODEL_HOST_ROOT) -> tuple[str | None, str]:
    """**生成链唯一出口**：返回 base 侧 vLLM 的 `-v <host>:<dst>:ro` 片段。

    这是 §7/§8 那一族"路径域"错误的**根治**：生成评测脚本时**不要**手写 `-v $MP:/model:ro`
    （`$MP` 是容器内路径），而是调用本函数 —— 它内部走 `resolve_base_mount_source`，
    自动转宿主路径、断言存在，失败返回 `(None, 原因)` 由调用方 fail-closed。

    用法（生成脚本时）::

        mount, why = render_vllm_base_mount(cfg["model"]["model_path"])
        if mount is None:
            raise SystemExit(f"base 挂载源不可用：{why}")   # fail-closed
        cmd += f" {mount} "
    """
    src, msg = resolve_base_mount_source(model_path, host_root=host_root)
    if src is None:
        return (None, msg)
    if not container_dst.startswith("/"):
        return (None, f"容器内目标路径非法：{container_dst} ⇒ fail-closed")
    return (f"-v {src}:{container_dst}:ro", f"base 挂载源 {src}（宿主，存在）⇒ {container_dst}:ro")


def check_card_count(selected: list[str], required: int) -> list[str]:
    """★ 卡数必须等于该档 manifest 的 `cards`（第 8 个"启动前置"类坑的收口）。

    228 实测（2026-10-13 T009）：起跑脚本**写死了 `SEL=7`**（1 卡），而该档
    `nproc_per_node=4` ⇒ torchrun 起 4 个 rank、rank 1/2/3 全失败 ⇒ `exitcode=1`。
    护栏：起跑前**从 manifest 读 `cards`**，断言"选中卡数 == cards"，不匹配即 fail-closed。
    ⛔ 禁止在起跑脚本里写死卡集。
    """
    if required <= 0:
        return [f"manifest 的 cards={required} 非法 ⇒ fail-closed"]
    if len(selected) != required:
        return [f"卡数不匹配：该档 cards={required}，实选 {len(selected)} 张（{selected}）⇒ fail-closed"]
    if len(set(selected)) != len(selected):
        return [f"卡集有重复项：{selected} ⇒ fail-closed"]
    return []


def plan_cards(available: list[str], required: int) -> tuple[list[str] | None, str]:
    """从**实测空闲**卡里为该档取 `required` 张；不足则返回 `(None, 原因)`（fail-closed）。

    ⛔ 不写死卡集：卡集由"manifest 的 cards"＋"当前实测空闲"共同决定。
    """
    if required <= 0:
        return (None, f"required={required} 非法 ⇒ fail-closed")
    if len(available) < required:
        return (None, f"需 {required} 张，实测仅 {len(available)} 张空闲（{available}）⇒ 顺延，不硬上")
    return (available[:required], f"取自实测空闲：{available[:required]}（需 {required} 张）")


def check_script_self_contained(script_text: str, required_vars: list[str]) -> list[str]:
    """★ 生成的启动脚本必须**自己定义**它用到的每个变量（第 9 个"启动前置"类坑）。

    228 实测（2026-10-13 T024）：我用 sed 把容器名改成 `t024-export-${TS}`，**却把 `TS=` 的定义
    留在了外层脚本**⇒ 生成的脚本一跑到那行就 `TS: unbound variable` 而死（现象是"日志停住、
    容器已退、GPU 已空"，离真因很远）。护栏：生成后**断言每个必需变量在脚本内被赋值**。
    """
    import re as _re
    problems: list[str] = []
    if not script_text:
        return ["脚本内容为空 ⇒ fail-closed"]
    for var in required_vars:
        pat = _re.compile(rf"^\s*(?:export\s+)?{_re.escape(var)}=", _re.M)
        if not pat.search(script_text):
            problems.append(f"生成的脚本未定义 ${var}（形如 `{var}=…`）⇒ fail-closed（会 unbound variable）")
        # 若脚本里根本没用它，则属"多余要求"，也提示出来（避免护栏变成噪音）
        elif f"${var}" not in script_text and f"${{{var}}}" not in script_text:
            problems.append(f"脚本定义了 ${var} 但未使用（请从 required_vars 移除，避免护栏噪音）")
    return problems


def build_json_reason(after_correct: int, n_total: int, base_correct: int, status: str) -> str:
    """落 json 的备注（≤100 字）：判定 + 读数，与结构化字段同源。"""
    after = after_correct / n_total * 100 if n_total else 0.0
    base = base_correct / n_total * 100 if n_total else 0.0
    delta = after - base
    return f"{status}：after {after:.4f}%（{after_correct}/{n_total}）/ base {base:.4f}%（{base_correct}/{n_total}）/ Δ {delta:+.2f} pp"[:100]
