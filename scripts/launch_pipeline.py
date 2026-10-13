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


EVAL_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/launch_pipeline.render_eval_script() 生成 —— ⛔ 请勿手改（改了就会重演"每轮现写"那一族坑）
set -uo pipefail
T={tree}
EV={eval_dir}
IMG={image_run}
BASEIMG={image_base}
G={gpu}
PORT={port}
MD=$EV/{merge_name}
MP={base_mount_src}
TS=$(date -u +%s)
EVLOG=$EV/evidence
LOG=$EV/eval.log
say() {{ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }}
say "{tier} eval START GPU=$G port=$PORT ck={checkpoint} base=$MP"
python3 $T/scripts/run_guards.py fresh --dir "$MD" --phase pre | tee -a "$LOG" || {{ say "§8(pre) 失败 ⇒ fail-closed"; echo stale > "$EV/EVAL-ABORTED"; exit 6; }}
docker run --rm --name {tier}-export-$TS --network=none --runtime=nvidia --gpus "\"device=$G\"" -e NVIDIA_VISIBLE_DEVICES=$G \
  -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \
  -v $T:/workspace/graspo:ro -v /home/user/models:/models:ro -v $T/runs:/runs:ro -v $EV:/evalout \
  -v /home/user/elam-v5-balanced:/work/elam-v5:ro -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \
  graspo {export_cmd} > "$EVLOG/02-eval-export.txt" 2>&1
say "export rc=$?"
python3 $T/scripts/run_guards.py fresh --dir "$MD" --phase post --since $TS | tee -a "$LOG" || {{ say "§8(post) 失败 ⇒ fail-closed"; echo export-failed > "$EV/EVAL-ABORTED"; exit 6; }}
python3 $T/scripts/run_guards.py mamba --max-num-seqs {max_num_seqs} --blocks {mamba_blocks} | tee -a "$LOG" || {{ say "§10 失败 ⇒ fail-closed"; echo mamba > "$EV/EVAL-ABORTED"; exit 6; }}
wait_ready() {{ for i in $(seq 1 240); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q graspo-eval && return 0; sleep 15; done; return 1; }}
say "--- vLLM after（§10 max-num-seqs {max_num_seqs}）---"
nohup docker run --rm --name {tier}-vllm-after-$TS --runtime=nvidia --gpus "\"device=$G\"" -e NVIDIA_VISIBLE_DEVICES=$G \
  -v $MD:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g {vllm_image} --model /model --served-model-name graspo-eval \
  --port 8000 --max-model-len {max_model_len} --gpu-memory-utilization {util} --max-num-seqs {max_num_seqs} --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$EVLOG/vllm-after.log" 2>&1 &
wait_ready || {{ say "★ vLLM after 未就绪 ⇒ abort"; docker stop {tier}-vllm-after-$TS >/dev/null 2>&1; echo vllm-after-not-ready > "$EV/EVAL-ABORTED"; exit 8; }}
say "vLLM after READY"; say "--- eval after ---"
bash "$EV/run-eval.sh" "$T" "$EV" {tier_upper} after "$EV/eval-{tier_lower}-after.yaml" > "$EVLOG/03-run-eval-after.out" 2>&1
say "eval after rc=$?"; docker stop {tier}-vllm-after-$TS >/dev/null 2>&1
say "--- vLLM base（§10 同配，挂载源=宿主路径）---"
nohup docker run --rm --name {tier}-vllm-base-$TS --runtime=nvidia --gpus "\"device=$G\"" -e NVIDIA_VISIBLE_DEVICES=$G \
  -v $MP:/model:ro -p $PORT:8000 --ipc=host --shm-size=16g {vllm_image} --model /model --served-model-name graspo-eval \
  --port 8000 --max-model-len {max_model_len} --gpu-memory-utilization {util} --max-num-seqs {max_num_seqs} --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$EVLOG/vllm-base.log" 2>&1 &
wait_ready || {{ say "★ vLLM base 未就绪 ⇒ abort"; docker stop {tier}-vllm-base-$TS >/dev/null 2>&1; echo vllm-base-not-ready > "$EV/EVAL-ABORTED"; exit 9; }}
say "vLLM base READY"; say "--- eval base + delta ---"
bash "$EV/run-eval.sh" "$T" "$EV" {tier_upper} base "$EV/eval-{tier_lower}-base.yaml" > "$EVLOG/04-run-eval-base.out" 2>&1
say "eval base rc=$?"; docker stop {tier}-vllm-base-$TS >/dev/null 2>&1
docker run --rm --network=none -e GRASPO_IMG=$IMG -e GRASPO_BASEIMG=$BASEIMG \
  -v $T:/workspace/graspo:ro -v $EV:/evalout -v /home/user/elam-v5-balanced:/work/elam-v5:ro \
  -w /workspace/graspo -e PYTHONPATH=/workspace/graspo/src $IMG \
  graspo eval delta --before /evalout/.local/reports/base/eval_report.json --after /evalout/.local/reports/after/eval_report.json \
  --eval-config /evalout/eval-{tier_lower}-after.yaml > "$EVLOG/05-eval-delta.txt" 2>&1
say "delta rc=$?"; say "=== {tier_upper} EVAL DONE ==="; echo done > "$EV/EVAL-DONE"
"""


def render_eval_script(*, tier: str, tree: str, eval_dir: str, checkpoint: str, base_model_path: str,
                       merge_name: str, gpu: str, port: int, export_cmd: str,
                       image_run: str, image_base: str,
                       vllm_image: str = "vllm-openai:v0.22.0",
                       max_num_seqs: int = 192, mamba_blocks: int = 232,
                       max_model_len: int = 8192, util: float = 0.85,
                       host_root: str = MODEL_HOST_ROOT) -> tuple[str | None, str]:
    """**生成链唯一出口**：产出整份评测脚本（自含变量、唯一容器名、内嵌三护栏）。

    它把 campaign 里踩过的坑一次固化进模板：
      · `TS=$(date +%s)` **在脚本内定义**（§"生成脚本自含性"）；
      · 容器名带 `-$TS` 唯一后缀（§6.1）；
      · base 侧挂载源 = **宿主路径**（`resolve_base_mount_source`，§"挂载源"族）；
      · `run_guards.py fresh --phase pre/post` 与 `mamba` 断言内嵌（§8/§10）。
    失败（基座不可解析 / 变量为空）⇒ 返回 `(None, 原因)`，调用方 fail-closed。
    """
    src, why = resolve_base_mount_source(base_model_path, host_root=host_root)
    if src is None:
        return (None, why)
    for name, val in (("tier", tier), ("tree", tree), ("eval_dir", eval_dir), ("checkpoint", checkpoint),
                      ("merge_name", merge_name), ("gpu", gpu), ("export_cmd", export_cmd),
                      ("image_run", image_run), ("image_base", image_base)):
        if not val:
            return (None, f"参数 {name} 为空 ⇒ fail-closed")
    text = EVAL_SCRIPT_TEMPLATE.format(
        tier=tier, tier_upper=tier.upper(), tier_lower=tier.lower(), tree=tree, eval_dir=eval_dir,
        checkpoint=checkpoint, merge_name=merge_name, gpu=gpu, port=port, export_cmd=export_cmd,
        image_run=image_run, image_base=image_base, vllm_image=vllm_image,
        max_num_seqs=max_num_seqs, mamba_blocks=mamba_blocks, max_model_len=max_model_len,
        util=util, base_mount_src=src,
    )
    problems = check_script_self_contained(text, ["T", "EV", "G", "PORT", "MD", "MP", "TS"])
    if problems:
        return (None, "；".join(problems))
    return (text, f"已生成 {tier} 评测脚本（base 挂载源={src}，§8/§10 断言内嵌，容器名带 -$TS）")


#: 部署树里脚本运行所需的依赖根（228 实测：缺它会让 gpu_idle_assert 因 pydantic 导入失败而**非零退出**）
HOSTDEPS_SUFFIX = ".hostdeps"


def check_tooling_env(tree: str, pythonpath: str | None, require_hostdeps: bool = True) -> list[str]:
    """★ 断言"跑选卡门/护栏时 PYTHONPATH 里必须有部署树的 `.hostdeps`"。

    为什么单列：228 实测（2026-10-13）—— 我把 `PYTHONPATH` 只设成 `$TREE/scripts`，
    `gpu_idle_assert.py` 因 **`ModuleNotFoundError: No module named 'pydantic'`** 对**每一张卡**
    非零退出 ⇒ 选卡循环读出 **"0 张空闲"** ⇒ **工具自身的失败被误读成"资源不足"**，
    这会让人以为卡不够而白白顺延。规则：**必须显式区分"工具坏了"与"卡真的不够"**。

    返回问题列表（空 = 通过）。`require_hostdeps=False` 时只校验 PATH 非空。
    """
    problems: list[str] = []
    if require_hostdeps:
        want = tree.rstrip("/") + HOSTDEPS_SUFFIX
        pp = pythonpath or ""
        if want not in pp.split(os.pathsep):
            problems.append(
                f"工具环境不满足：PYTHONPATH 缺少 {want}（缺它会让 gpu_idle_assert 因 pydantic 导入失败"
                f"而对每张卡非零退出 ⇒ 会被误读成『0 张空闲』）⇒ fail-closed；"
                f"正确用法：PYTHONPATH={want}:{tree}/scripts"
            )
    if not (pythonpath or "").strip():
        problems.append("PYTHONPATH 为空 ⇒ 无法保证护栏脚本可运行 ⇒ fail-closed")
    return problems


def require_usable_gate(gate_rc: int, gate_stdout: str) -> list[str]:
    """★ 把"选卡门的返回码"与"卡的真实空闲情况"分开判读。

    · rc==0 ⇒ 该卡可用；
    · rc!=0 且输出里有 `ModuleNotFoundError`/`ImportError`/`Traceback` ⇒ **工具坏了**（⛔ 不能当成
      "卡被占用"）；调用方应**立即 fail-closed 并修工具**，而不是顺延等卡。
    """
    if gate_rc == 0:
        return []
    text = gate_stdout or ""
    for marker in ("ModuleNotFoundError", "ImportError", "Traceback (most recent call last)"):
        if marker in text:
            return [f"选卡门**工具自身失败**（{marker}）⇒ ⛔ 不得解读为『卡被占用/不够』；先修工具环境（见 check_tooling_env）"]
    return []


def plan_full_cards(cards: list[dict], required: int, allow_used_mib: int = 0) -> tuple[list[str] | None, str]:
    """★ 为**大模型训练档**取卡：只接受"整卡基本空闲"的卡（第 11 个坑的收口）。

    228 实测（2026-10-13 T009，27B 全参 4 卡）：卡集里混进了 **宿主 GPU2**（上面有第三方
    **13.01 GiB** 常驻）—— 旧判据"free ≥ 头寸(48 GiB) ∧ util == 0"**放行了它**，而该档实际要**接近整卡**
    ⇒ `torch.OutOfMemoryError: Tried to allocate 1.81 GiB. GPU 1 … of which 154.25 MiB is free`。

    规则（对训练档）：**该卡 `used_mib <= allow_used_mib`（默认 0 ⇒ 有第三方常驻就不选）**，
    且 `free_mib` 足够（由调用方按该档真实头寸传入 `cards[*]["need_mib"]`，缺省按整卡 95%）。
    与 §6.2（服务档 = `util × 总显存`）同族：**不同 workload 的每卡余量要求不同，⛔ 不用一把尺子量所有档**。
    """
    if required <= 0:
        return (None, f"required={required} 非法 ⇒ fail-closed")
    picked: list[str] = []
    rejected: list[str] = []
    for c in cards:
        idx = str(c.get("index"))
        used = int(c.get("used_mib", 0))
        free = int(c.get("free_mib", 0))
        need = int(c.get("need_mib", 0)) or 0
        if used > allow_used_mib:
            rejected.append(f"{idx}(常驻 {used} MiB > 允许 {allow_used_mib})")
            continue
        if need and free < need:
            rejected.append(f"{idx}(余量 {free} < 需 {need})")
            continue
        picked.append(idx)
        if len(picked) == required:
            break
    if len(picked) < required:
        return (None, f"需 {required} 张『整卡空闲』卡，实测仅 {len(picked)} 张 ({picked})；"
                      f"被排除：{rejected or '无'} ⇒ 顺延，⛔ 不硬上")
    return (picked, f"取自整卡空闲：{picked}（已排除：{rejected or '无'}）")


def check_script_mount_sources(script_text: str, host_model_prefix: str = MODEL_HOST_ROOT) -> list[str]:
    """★ 对**生成出来的脚本文本**做挂载源断言（⛔ 只校验函数输出不算数）。

    228 实测（2026-10-13，T009/T007/T008/T024 共四次）：我确实调用并核对了
    `render_vllm_base_mount()` 的返回值，**但生成脚本时仍写成容器内变量**（如 `-v $MPB:/model:ro`
    其中 `MPB=/models/...`）⇒ 起跑后 `/model` 是空目录 ⇒ `ValidationError … params.json`。
    规则：**必须对"即将执行的脚本文本"断言** —— 文本里任何 `-v <src>:/model...` 的 `<src>`
    **要么是宿主路径（以 `host_model_prefix` 开头且存在），要么是脚本内自身定义的宿主路径变量**；
    ⛔ **不得出现** `/model`、`/models/...`、`/evalout/...`、`/workspace/...`、`/runs/...` 这类容器内路径。
    """
    import re as _re
    problems: list[str] = []
    if not script_text:
        return ["脚本文本为空 ⇒ fail-closed"]
    # 只匹配挂到 **/model** 的挂载（`/models` 是另一回事，别误伤）
    for m in _re.finditer(r"-v\s+(\S+?):(/model(?::[a-z,]+)?)(?=\s|$)", script_text):
        src, dst = m.group(1), m.group(2)
        if src.startswith("$"):
            var = src.lstrip("${").rstrip("}")
            # 脚本内该变量的赋值必须是宿主路径
            m2 = _re.search(rf"(?:^|[;\s])(?:export\s+)?{_re.escape(var)}=([^;\n]+)", script_text, _re.M)
            if not m2:
                problems.append(f"挂载源 {src} 是未定义变量（{dst}）⇒ fail-closed")
                continue
            val = m2.group(1).strip().strip('"').strip("'")
            # 由其它脚本内变量拼出的宿主路径（如 MD=$EV/merged-hf-x）无法静态解析 ⇒ 放行
            if val.startswith("$"):
                continue
            if not val.startswith(host_model_prefix) and not val.startswith("/"):
                problems.append(f"挂载源 {src} 解析为 {val}，不是宿主路径 ⇒ fail-closed")
            elif any(val.startswith(pfx) for pfx in CONTAINER_ONLY_PREFIXES) and val == "/model":
                problems.append(f"挂载源 {src} 解析为容器内专有路径 {val} ⇒ fail-closed")
            elif val.startswith(MODEL_CONTAINER_ROOT):
                problems.append(f"挂载源 {src} 解析为容器内路径 {val}（应为 {host_model_prefix}…）⇒ fail-closed")
            continue
        if src.startswith(MODEL_CONTAINER_ROOT) or src in ("/model",):
            problems.append(f"挂载源 {src} 是容器内路径（{dst}）⇒ fail-closed；应为 {host_model_prefix}…")
        elif not src.startswith(host_model_prefix):
            problems.append(f"挂载源 {src} 不在宿主模型根 {host_model_prefix} 下（{dst}）⇒ fail-closed")
    return problems


def decide_eval_mode(form: str) -> tuple[str, str]:
    """★ 由 §7 形态探测结果决定评测怎么起：0=零 merge（全参直服务）/ 1=合并。

    228 实测（2026-10-13 T019）：我按"LoRA ⇒ 合并"的惯性写了 export 步骤，而该档实际是
    **全参/HF 形态**（`run_guards form` 已打印 `[full]`）⇒ `graspo eval export` 回执
    `checkpoint is already a servable base model — nothing to merge` ⇒ **合并产物根本不会产生**
    ⇒ `§8(post)` 正确 fail-closed（"没有产物的成功"）。规则：**先探测、再分支**，⛔ 不靠惯性。

    返回 `(mode, why)`：`mode` ∈ {"zero-merge", "merge", "unknown"}。
    """
    if form == "full":
        return ("zero-merge", "全参/HF 形态 ⇒ vLLM 直指 checkpoint 目录，零 merge（export 会回 nothing to merge）")
    if form in ("adapter", "native_rank"):
        return ("merge", f"{form} 形态 ⇒ 需先合并为 merged-hf（graspo eval export）再服务")
    return ("unknown", f"形态未知（{form}）⇒ ⛔ 不得凭惯性选路径，须人工确认 ⇒ fail-closed")


#: §10 的缺省安全值：228 实测 27B(Mamba) 的可用 cache blocks = 232 ⇒ 取 192
MAMBA_SAFE_MAX_NUM_SEQS = 192


def check_max_num_seqs_in_script(script_text: str, needs_limit: bool, safe: int = MAMBA_SAFE_MAX_NUM_SEQS) -> list[str]:
    """★ §10 的**产物级**落实（与 §15 同精神：纪律必须落在"被执行的那份产物"上）。

    §10 早已入纪律（"27B/Mamba 必须显式 `--max-num-seqs ≤ 可用 blocks`"），但 2026-10-13 T025
    评测仍炸：`max_num_seqs (1024) exceeds available Mamba cache blocks (232) … CUDA graph capture
    cannot proceed` —— 因为**我改生成器时丢了这条参数** ⇒ 纪律没有落在产物上。
    规则：需要限制的档（27B/Mamba），**生成的脚本文本里必须出现 `--max-num-seqs N`，且 N ≤ safe**。
    """
    import re as _re
    if not script_text:
        return ["脚本文本为空 ⇒ fail-closed"]
    if not needs_limit:
        return []
    hits = [int(m.group(1)) for m in _re.finditer(r"--max-num-seqs\s+(\d+)", script_text)]
    if not hits:
        return [f"该档属 27B/Mamba ⇒ 脚本必须显式 --max-num-seqs（≤ {safe}）；⛔ 缺参数会致 "
                f"`max_num_seqs (1024) exceeds available Mamba cache blocks (232)` ⇒ fail-closed"]
    bad = [n for n in hits if n > safe]
    if bad:
        return [f"--max-num-seqs 取值 {bad} > 安全上限 {safe} ⇒ fail-closed"]
    return []


def build_json_reason(after_correct: int, n_total: int, base_correct: int, status: str) -> str:
    """落 json 的备注（≤100 字）：判定 + 读数，与结构化字段同源。"""
    after = after_correct / n_total * 100 if n_total else 0.0
    base = base_correct / n_total * 100 if n_total else 0.0
    delta = after - base
    return f"{status}：after {after:.4f}%（{after_correct}/{n_total}）/ base {base:.4f}%（{base_correct}/{n_total}）/ Δ {delta:+.2f} pp"[:100]
