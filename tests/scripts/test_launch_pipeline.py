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
