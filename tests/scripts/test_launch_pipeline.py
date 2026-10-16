"""`scripts/launch_pipeline.py` 的用例。

覆盖指挥部点名的 5 条：变量空 ⇒ 拒；脚本缺 ⇒ 拒；adapter 形态 ⇒ 过；
陈旧目录 ⇒ 拒；max_num_seqs > blocks ⇒ 拒。只建临时目录，不碰 GPU/容器。
"""

import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import launch_pipeline as lp  # noqa: E402


def _mk_checkpoint(tmp_path: Path, form: str = "adapter") -> Path:
    ck = tmp_path / "ckpt"
    ck.mkdir()
    if form == "adapter":
        (ck / "adapter_config.json").write_text("{}")
        (ck / "adapter_model.safetensors").write_text("x")
    else:
        (ck / "config.json").write_text("{}")
        (ck / "model.safetensors").write_text("x")
    return ck


def _plan(tmp_path: Path, **kw) -> lp.Plan:
    ck = kw.pop("checkpoint", None) or _mk_checkpoint(tmp_path)
    merge = kw.pop("merge_dir", tmp_path / "merged")
    tool = tmp_path / "run_guards.py"
    tool.write_text("# stub")
    defaults = dict(
        tier="T008",
        checkpoint=str(ck),
        base_model="/models/Qwen3.8-27B",
        merge_dir=str(merge),
        tree=str(tmp_path),
        since_epoch=time.time(),
        required_tools=[str(tool)],
        max_num_seqs=192,
        mamba_blocks=232,
    )
    defaults.update(kw)
    return lp.Plan(**defaults)


def test_happy_path_adapter_passes(tmp_path: Path):
    assert lp.validate_pre(_plan(tmp_path)) == []


def test_empty_variable_is_refused(tmp_path: Path):
    problems = lp.validate_pre(_plan(tmp_path, base_model=""))
    assert any("base_model" in p and "为空" in p for p in problems)


def test_missing_tooling_is_refused(tmp_path: Path):
    """§11：依赖件不在部署树 ⇒ fail-closed。"""
    problems = lp.validate_pre(_plan(tmp_path, required_tools=[str(tmp_path / "nope.py")]))
    assert any("§11" in p for p in problems)


def test_adapter_form_passes_without_config_json(tmp_path: Path):
    """T008 回归：adapter 目录没有 config.json 也必须过。"""
    ck = _mk_checkpoint(tmp_path, form="adapter")
    problems = lp.validate_pre(_plan(tmp_path, checkpoint=ck))
    assert problems == []


def test_stale_merge_dir_is_refused(tmp_path: Path):
    merge = tmp_path / "merged"
    merge.mkdir()
    (merge / "model-00001-of-00018.safetensors").write_text("stale")
    problems = lp.validate_pre(_plan(tmp_path, merge_dir=merge))
    assert any("§8(pre)" in p and "陈旧" in p for p in problems)


def test_mamba_over_blocks_is_refused(tmp_path: Path):
    problems = lp.validate_pre(_plan(tmp_path, max_num_seqs=1024, mamba_blocks=232))
    assert any("§10" in p and "Engine core initialization failed" in p for p in problems)


def test_mamba_without_blocks_is_refused(tmp_path: Path):
    problems = lp.validate_pre(_plan(tmp_path, max_num_seqs=192, mamba_blocks=None))
    assert any("§10" in p and "缺 mamba_blocks" in p for p in problems)


def test_post_freshness_rejects_older_dir(tmp_path: Path):
    merge = tmp_path / "merged"
    merge.mkdir()
    (merge / "model.safetensors").write_text("x")
    plan = _plan(tmp_path, merge_dir=merge, since_epoch=time.time() + 3600)
    problems = lp.validate_post(plan)
    assert any("§8(post)" in p for p in problems), problems


def test_post_freshness_accepts_new_dir(tmp_path: Path):
    merge = tmp_path / "merged"
    merge.mkdir()
    (merge / "model.safetensors").write_text("x")
    plan = _plan(tmp_path, merge_dir=merge, since_epoch=time.time() - 60)
    assert lp.validate_post(plan) == []


def test_json_reason_is_short_and_matches_fields():
    reason = lp.build_json_reason(after_correct=171, n_total=702, base_correct=118, status="通过（已验证可用）")
    assert len(reason) <= 100
    assert "171/702" in reason and "+7.55 pp" in reason


# ---------- 挂载源必须宿主路径（第 6 次"路径域"错误的收口） ----------
def test_mount_source_accepts_existing_host_path(tmp_path: Path):
    ok, msg = lp.check_mount_source(str(tmp_path))
    assert ok and "宿主路径且存在" in msg, msg


def test_mount_source_rejects_container_only_models_path():
    """★ 回归：base 侧曾写 `/models/Qwen3.8-27B`（容器内路径）⇒ docker 建空目录 ⇒ vLLM 起不来。"""
    ok, msg = lp.check_mount_source("/models/Qwen3.8-27B")
    assert ok is False and "容器内专有路径" in msg, msg


def test_mount_source_rejects_container_only_model_root():
    ok, msg = lp.check_mount_source("/model")
    assert ok is False and "容器内专有路径" in msg, msg


def test_mount_source_rejects_missing_host_path(tmp_path: Path):
    ok, msg = lp.check_mount_source(str(tmp_path / "nope"))
    assert ok is False and "不存在" in msg, msg


def test_mount_sources_reports_each_problem():
    problems = lp.check_mount_sources(["/models/Qwen3.8-27B", "/evalout/merged-hf-t024"])
    assert len(problems) == 2, problems


# ---------- §7 扩到"合并产物"（第 7 次路径域类问题的收口） ----------
def _mk_merged(tmp_path: Path) -> Path:
    d = tmp_path / "merged-hf-t024"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00018.safetensors").write_text("x")
    return d


def test_merged_artifact_accepts_evalout_path(tmp_path: Path):
    assert lp.check_merged_artifact(str(_mk_merged(tmp_path)), "/evalout/merged-hf-t024") == []


def test_merged_artifact_accepts_runs_path(tmp_path: Path):
    assert lp.check_merged_artifact(str(_mk_merged(tmp_path)), "/runs/T004/T004/v0-x/checkpoint-200") == []


def test_merged_artifact_rejects_vllm_model_path(tmp_path: Path):
    """★ 回归：T024 的 `eval-t024-after.yaml` 曾写 `checkpoint_path: /model` ⇒ after 报告 MISSING。"""
    problems = lp.check_merged_artifact(str(_mk_merged(tmp_path)), "/model")
    assert problems and "vLLM 容器路径" in problems[0], problems


def test_merged_artifact_rejects_host_path(tmp_path: Path):
    merged = _mk_merged(tmp_path)
    problems = lp.check_merged_artifact(str(merged), str(merged))
    assert problems and "不在评测容器可达前缀内" in problems[0], problems


def test_merged_artifact_rejects_empty_host_dir(tmp_path: Path):
    d = tmp_path / "merged-empty"
    d.mkdir()
    problems = lp.check_merged_artifact(str(d), "/evalout/x")
    assert problems and "§7 失败（merged 宿主侧）" in problems[0], problems


# ---------- 基座路径：容器内 → 宿主（T024 事故 + T008 险些事故的收口） ----------
def test_resolve_base_maps_container_path_to_host(tmp_path: Path):
    """`/models/Qwen3.8-27B` ⇒ `<host_root>/Qwen3.8-27B`（存在则通过）。"""
    host_root = tmp_path / "models"
    (host_root / "Qwen3.8-27B").mkdir(parents=True)
    src, msg = lp.resolve_base_mount_source("/models/Qwen3.8-27B", host_root=str(host_root) + "/")
    assert src == str(host_root / "Qwen3.8-27B") and "宿主路径且存在" in msg, (src, msg)


def test_resolve_base_refuses_when_host_counterpart_missing(tmp_path: Path):
    host_root = tmp_path / "models"
    host_root.mkdir()
    src, msg = lp.resolve_base_mount_source("/models/Qwen3.8-27B", host_root=str(host_root) + "/")
    assert src is None and "不存在" in msg, (src, msg)


def test_resolve_base_refuses_empty():
    src, msg = lp.resolve_base_mount_source("")
    assert src is None and "为空" in msg, (src, msg)


def test_resolve_base_accepts_already_host_path(tmp_path: Path):
    host_root = tmp_path / "models"
    (host_root / "Qwen3.8-27B").mkdir(parents=True)
    p = str(host_root / "Qwen3.8-27B")
    src, msg = lp.resolve_base_mount_source(p)
    assert src == p and "宿主路径且存在" in msg, (src, msg)


# ---------- 生成链唯一出口：base 侧 vLLM 挂载片段 ----------
def test_render_vllm_base_mount_from_container_path(tmp_path: Path):
    host_root = tmp_path / "models"
    (host_root / "Qwen3.8-27B").mkdir(parents=True)
    frag, why = lp.render_vllm_base_mount("/models/Qwen3.8-27B", host_root=str(host_root) + "/")
    assert frag == f"-v {host_root / 'Qwen3.8-27B'}:/model:ro", (frag, why)


def test_render_vllm_base_mount_refuses_unresolvable(tmp_path: Path):
    """★ 回归：`$MP` 原样当挂载源（宿主无对应件）⇒ 生成链必须 fail-closed，而不是产出坏命令。"""
    host_root = tmp_path / "models"
    host_root.mkdir()
    frag, why = lp.render_vllm_base_mount("/models/Qwen3.8-27B", host_root=str(host_root) + "/")
    assert frag is None and "不存在" in why, (frag, why)


def test_render_vllm_base_mount_refuses_bad_dst(tmp_path: Path):
    host_root = tmp_path / "models"
    (host_root / "Qwen3.8-27B").mkdir(parents=True)
    frag, why = lp.render_vllm_base_mount("/models/Qwen3.8-27B", container_dst="model", host_root=str(host_root) + "/")
    assert frag is None and "目标路径非法" in why, (frag, why)


# ---------- 卡数必须等于 manifest.cards（T009 事故：写死 1 卡而该档 nproc=4） ----------
def test_card_count_accepts_matching_set():
    assert lp.check_card_count(["3", "5", "6", "7"], 4) == []


def test_card_count_rejects_too_few_cards():
    """★ 回归：T009 给了 1 张卡但该档 cards=4 ⇒ 必须拒。"""
    problems = lp.check_card_count(["7"], 4)
    assert problems and "卡数不匹配" in problems[0], problems


def test_card_count_rejects_duplicates():
    assert lp.check_card_count(["7", "7"], 2)


def test_card_count_rejects_bad_manifest_value():
    assert lp.check_card_count([], 0)


def test_plan_cards_takes_exactly_required():
    cards, why = lp.plan_cards(["3", "5", "6", "7"], 2)
    assert cards == ["3", "5"], (cards, why)


def test_plan_cards_defers_when_not_enough():
    """★ 卡不足 ⇒ 返回 None（顺延），⛔ 不硬上。"""
    cards, why = lp.plan_cards(["7"], 4)
    assert cards is None and "顺延" in why, (cards, why)


# ---------- 生成的脚本必须自定义变量（T024 `TS: unbound variable` 事故） ----------
_OK_SCRIPT = """#!/usr/bin/env bash
set -uo pipefail
TS=$(date -u +%s)
nohup docker run --rm --name t024-export-$TS ...
"""

_BAD_SCRIPT = """#!/usr/bin/env bash
set -uo pipefail
nohup docker run --rm --name t024-export-$TS ...
"""


def test_script_self_contained_accepts_defined_var():
    assert lp.check_script_self_contained(_OK_SCRIPT, ["TS"]) == []


def test_script_self_contained_rejects_undefined_var():
    """★ 回归：T024 生成的脚本里 TS 未定义 ⇒ 一跑到那行就 unbound variable。"""
    problems = lp.check_script_self_contained(_BAD_SCRIPT, ["TS"])
    assert problems and "未定义 $TS" in problems[0], problems


def test_script_self_contained_flags_unused_var():
    problems = lp.check_script_self_contained("VAR=1\n", ["VAR"])
    assert problems and "未使用" in problems[0], problems


def test_script_self_contained_rejects_empty_script():
    assert lp.check_script_self_contained("", ["TS"])


# ---------- 生成链唯一出口：整份评测脚本 ----------
def _render_kwargs(tmp_path: Path, **kw):
    host_root = tmp_path / "models"
    (host_root / "Qwen3.8-27B").mkdir(parents=True, exist_ok=True)
    base = dict(host_root=str(host_root) + "/",
        tier="T024", tree="/data/tree", eval_dir="/data/eval", checkpoint="/runs/T024/T024/v0-x/checkpoint-100",
        base_model_path="/models/Qwen3.8-27B", merge_name="merged-hf-t024", gpu="3", port=18906,
        export_cmd="export --config /evalout/export-t024.yaml",
        image_run="graspo:run", image_base="graspo:base",
    )
    base.update(kw)
    return base, str(host_root) + "/"


def test_render_eval_script_is_self_contained(tmp_path: Path):
    kw, _ = _render_kwargs(tmp_path)
    text, why = lp.render_eval_script(**kw)
    assert text is not None, why
    # 生成的脚本必须自定义它用到的变量（T024 `TS: unbound variable` 回归）
    assert lp.check_script_self_contained(text, ["T", "EV", "G", "PORT", "MD", "MP", "TS"]) == []
    assert "TS=$(date -u +%s)" in text


def test_render_eval_script_embeds_guards_and_unique_names(tmp_path: Path):
    kw, _ = _render_kwargs(tmp_path)
    text, why = lp.render_eval_script(**kw)
    assert text is not None, why
    assert "run_guards.py fresh --dir" in text and "--phase pre" in text and "--phase post" in text
    assert "run_guards.py mamba --max-num-seqs 192 --blocks 232" in text
    assert "T024-vllm-after-$TS" in text and "T024-vllm-base-$TS" in text


def test_render_eval_script_refuses_unresolvable_base(tmp_path: Path):
    """★ 回归：基座用容器内路径且宿主无对应件 ⇒ 生成链 fail-closed（⛔ 不产出坏命令）。"""
    kw, _ = _render_kwargs(tmp_path)
    kw["base_model_path"] = "/models/DoesNotExist-27B"
    text, why = lp.render_eval_script(**kw)
    assert text is None and "不存在" in why, (text, why)


def test_render_eval_script_refuses_empty_param(tmp_path: Path):
    kw, _ = _render_kwargs(tmp_path)
    kw["checkpoint"] = ""
    text, why = lp.render_eval_script(**kw)
    assert text is None and "为空" in why, (text, why)


# ---------- 工具环境：hostdeps 必须在 PYTHONPATH（"假 0 张空闲"事故） ----------
def test_tooling_env_accepts_hostdeps_on_path():
    tree = "/data/tree"
    pp = f"/data/tree.hostdeps:{tree}/scripts"
    assert lp.check_tooling_env(tree, pp) == []


def test_tooling_env_rejects_missing_hostdeps():
    """★ 回归：只给 $TREE/scripts ⇒ gpu_idle_assert 因 pydantic 导入失败 ⇒ 假'0 张空闲'。"""
    problems = lp.check_tooling_env("/data/tree", "/data/tree/scripts")
    assert problems and "缺少 /data/tree.hostdeps" in problems[0], problems
    assert "0 张空闲" in problems[0]  # 给出可操作原因


def test_tooling_env_rejects_empty_pythonpath():
    assert lp.check_tooling_env("/data/tree", "")


def test_usable_gate_flags_broken_tool_not_busy_card():
    """★ 工具坏了（ModuleNotFoundError）⛔ 不得当成"卡被占用"。"""
    problems = lp.require_usable_gate(1, "ModuleNotFoundError: No module named 'pydantic'")
    assert problems and "工具自身失败" in problems[0], problems


def test_usable_gate_accepts_ok_and_plain_refusal():
    assert lp.require_usable_gate(0, "[gpu-idle] OK") == []
    # 普通拒绝（卡确实忙）不算工具故障
    assert lp.require_usable_gate(1, "目标卡 GPU0 起跑前不可用：利用率 71.0% != 0") == []


# ---------- 训练档取卡：只接受整卡空闲（T009 GPU2 混入 ⇒ OOM） ----------
def _cards(*specs):
    return [{"index": i, "used_mib": u, "free_mib": f, "need_mib": n} for (i, u, f, n) in specs]


def test_plan_full_cards_rejects_card_with_third_party_resident():
    """★ 回归：GPU2 有 13.01 GiB 常驻（free 67.8 GB ≥ 旧头寸 48 GiB）⇒ 必须拒。"""
    cards = _cards(("0", 0, 81153, 70000), ("2", 13335, 67818, 70000), ("5", 0, 81153, 70000))
    picked, why = lp.plan_full_cards(cards, 3)
    assert picked is None and "整卡空闲" in why and "2(" in why, (picked, why)


def test_plan_full_cards_accepts_clean_cards():
    cards = _cards(("0", 0, 81153, 70000), ("5", 0, 81153, 70000), ("6", 0, 81153, 70000))
    picked, why = lp.plan_full_cards(cards, 3)
    assert picked == ["0", "5", "6"], (picked, why)


def test_plan_full_cards_defers_when_not_enough_clean():
    cards = _cards(("0", 0, 81153, 70000), ("2", 13335, 67818, 70000))
    picked, why = lp.plan_full_cards(cards, 4)
    assert picked is None and "顺延" in why and "不硬上" in why, (picked, why)


def test_plan_full_cards_respects_need_mib():
    """余量不足（need 给足）⇒ 拒；同卡 need 放宽 ⇒ 过。"""
    tight = _cards(("0", 0, 60000, 70000))
    assert lp.plan_full_cards(tight, 1)[0] is None
    loose = _cards(("0", 0, 60000, 50000))
    assert lp.plan_full_cards(loose, 1)[0] == ["0"]


# ---------- 对"生成的脚本文本"断言挂载源（T009/T007/T008/T024 四次坑） ----------
#: ★ 本组"宿主路径"夹具**在运行期拼装**——源码内 ⛔ 不出现真实账号 / 宿主路径字面量
#: （§15.1 检查 1/3：宿主路径与真实账号属机密类别；本仓要进开源仓，红线**不因"是测试夹具"
#: 而豁免**，也不存在 A 类具名豁免通道）。先例：`tools/matrix/build_official.py` 的
#: `_FIXTURE_ACCOUNT`（同族做法：形状留真、字面量运行期拼）。
#: 形状仍必须是**真实形态**（`<宿主模型根>/<模型名>`），且前缀取**被测函数自己的**
#: `MODEL_HOST_ROOT` —— 否则"正例"会与实现脱钩，验不出"接受宿主路径"这条语义。
_HOST_MODEL_LITERAL = lp.MODEL_HOST_ROOT.rstrip("/") + "/Qwen3.8-27B"


def test_script_mount_sources_rejects_container_path_literal():
    """★ 回归：脚本里写死/解析为容器内模型路径 ⇒ 拒。"""
    text = "nohup docker run -v $MPB:/model:ro ...\nMPB=/models/Qwen3.8-27B\n"
    problems = lp.check_script_mount_sources(text)
    assert problems and "容器内路径" in problems[0], problems


def test_script_mount_sources_accepts_host_literal():
    text = f"nohup docker run -v {_HOST_MODEL_LITERAL}:/model:ro ...\n"
    assert lp.check_script_mount_sources(text) == []


def test_script_mount_sources_accepts_host_variable():
    text = f"HOSTMODEL={_HOST_MODEL_LITERAL}\nnohup docker run -v $HOSTMODEL:/model:ro\n"
    assert lp.check_script_mount_sources(text) == []


def test_script_mount_sources_flags_undefined_variable():
    text = "nohup docker run -v $MPB:/model:ro\n"
    problems = lp.check_script_mount_sources(text)
    assert problems and "未定义变量" in problems[0], problems


def test_rendered_eval_script_passes_mount_check(tmp_path: Path):
    """★ 生成链产物必须过这条断言（把'只校验函数输出'升级为'校验脚本产物'）。"""
    kw, _ = _render_kwargs(tmp_path)
    text, why = lp.render_eval_script(**kw)
    assert text is not None, why
    assert lp.check_script_mount_sources(text) == []


# ---------- 形态决定评测路径：零 merge vs 合并（T019 事故） ----------
def test_decide_eval_mode_full_is_zero_merge():
    mode, why = lp.decide_eval_mode("full")
    assert mode == "zero-merge" and "零 merge" in why, (mode, why)


def test_decide_eval_mode_adapter_and_native_merge():
    assert lp.decide_eval_mode("adapter")[0] == "merge"
    assert lp.decide_eval_mode("native_rank")[0] == "merge"


def test_decide_eval_mode_unknown_fails_closed():
    """★ 回归：形态未知时 ⛔ 不得凭惯性选路径。"""
    mode, why = lp.decide_eval_mode("unknown")
    assert mode == "unknown" and "fail-closed" in why, (mode, why)


# ---------- §10 的产物级落实（T025 实测 1024 > 232） ----------
def test_max_num_seqs_required_for_mamba_tier():
    """★ 回归：27B/Mamba 档脚本缺 `--max-num-seqs` ⇒ 拒（T025 事故）。"""
    text = "docker run ... --gpu-memory-utilization 0.85 --trust-remote-code ...\n"
    problems = lp.check_max_num_seqs_in_script(text, needs_limit=True)
    assert problems and "必须显式 --max-num-seqs" in problems[0], problems


def test_max_num_seqs_accepts_safe_value():
    assert lp.check_max_num_seqs_in_script("--max-num-seqs 192 --trust-remote-code\n", True) == []


def test_max_num_seqs_rejects_too_large():
    problems = lp.check_max_num_seqs_in_script("--max-num-seqs 1024\n", True)
    assert problems and "1024" in problems[0], problems


def test_max_num_seqs_not_required_for_small_tier():
    assert lp.check_max_num_seqs_in_script("--gpu-memory-utilization 0.85\n", False) == []


def test_plan_full_cards_hard_excludes_reserved():
    """★ §17.1 最后一块：占卡台账（含 loading 态）是**硬排除**，⛔ 不是建议。"""
    cards = _cards(("1", 0, 81153, 0), ("3", 0, 81153, 0), ("6", 0, 81153, 0))
    picked, why = lp.plan_full_cards(cards, 1, reserved={"1", "3"})   # 1/3 上有我方 loading 容器
    assert picked == ["6"], (picked, why)
    assert "§17" in why, why


# ---------- §17.3：禁用集不得复用旧快照 ----------
def test_require_card_registered_detects_stale_snapshot():
    """★ 回归（T022 撞卡）：起跑后该卡仍未入禁用集 ⇒ 拒（不得视为已占用）。"""
    problems = lp.require_card_registered("6", {"0", "1", "3", "5"})
    assert problems and "仍未登记" in problems[0], problems


def test_require_card_registered_passes_when_registered():
    assert lp.require_card_registered("7", {"0", "1", "3", "5", "7"}) == []
