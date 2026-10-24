"""判据判定链对齐（2026-10-23）的机核 —— 四条新判定位必须**能红**。

为什么有这件（宪法 §15.1「能失败的断言才叫断言」）：
  基准报告第 2 章的四条判据此前与代码判定链有 5 处断点，全部是"**文档改了、判定位没改**"
  的静默缺口 —— 判据① 会悄悄退回落步数、CPT 的 loss 支会整条被跳过、7 字节的垃圾权重会
  被当成"可加载"、判据④ 会悄悄少判「合并后」「用 vLLM 启动」两条。⇒ 本文件的每一条都成对：
  **阳性用例（该过）＋ 阴性用例（必须红）**，缺任何一半都不算测试。

覆盖（与本轮 4 个断点一一对应）：
  ① `tools/matrix/backfill_matrix.py::judge_run_complete` 的 Epoch 三态（达/不达/缺键）；
  ② `judge_effect` 的 CPT loss 支（下降 ⇒ 过；未降 ⇒ 红；缺读数 ⇒ unknown）；
  ③ `scripts/make_result_json.py::inspect_checkpoint` 的**真加载**校验（垃圾权重 ⇒ False）；
  ④ `scripts/record_checks.py::judge_inference` 的「LoRA 合并后」「vLLM 起服务」两条。

纯标准库（不 import torch/pyyaml/项目包），按文件路径加载被测脚本原文。
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BF = _load("tools/matrix/backfill_matrix.py", "backfill_matrix_for_judge_align")
MRJ = _load("scripts/make_result_json.py", "make_result_json_for_judge_align")
RC = _load("scripts/record_checks.py", "record_checks_for_judge_align")

PASS, FAIL, UNKNOWN = BF.PASS, BF.FAIL, BF.UNKNOWN
#: 判据② 的冻结通过线（CPT = 不劣化线 0.0 pp）；判据测试只关心 CPT 的 loss 支。
FROZEN = {"sft": 10.0, "graspo": 5.0, "cpt": 0.0, "opd": 5.0}
ROW_CPT = {"train_method": "cpt"}
ROW_SFT = {"train_method": "sft", "derived_steps": "400"}


def _doc(**over) -> dict:
    """判据① 的一份最小合法跑次（M001：4 轮 / 400 步）。"""
    doc = {
        "schema": "matrix-run/v1",
        "tier": "M001",
        "params": {"algo": "SFT", "tuner_type": "lora", "max_epochs": 4},
        "run_result": {"exit_code": 0, "epochs_completed": 4, "epochs_planned": 4,
                       "final_ckpt_loadable": True, "nan_or_inf_detected": False},
        "steps_actual": 400,
    }
    doc.update(over)
    return doc


# ── ① 判据① 的 Epoch 三态 ────────────────────────────────────────────────────
def test_criterion1_epoch_pass():
    """阳性：跑满 4/4 epoch ⇒ 判据① 通过，且 reason 明确写"跑满 epoch"。"""
    verdict, why = BF.judge_run_complete(_doc(), ROW_SFT)
    assert verdict == PASS, why
    assert "4/4 epoch" in why


def test_criterion1_epoch_short_is_red_even_if_steps_match():
    """★ 阴性对照：epoch 3/4 而**步数与计划完全相等** ⇒ 必须判「不通过」（不许回落步数）。"""
    doc = _doc(run_result=dict(_doc()["run_result"], epochs_completed=3))
    verdict, why = BF.judge_run_complete(doc, ROW_SFT)
    assert verdict == FAIL, why
    assert "epoch" in why


def test_criterion1_missing_epoch_key_is_unknown_not_pass():
    """★ 阴性对照：缺 `epochs_completed`（步数仍相等）⇒ 必须 `unknown`（fail-closed）。"""
    rr = {k: v for k, v in _doc()["run_result"].items() if k != "epochs_completed"}
    verdict, why = BF.judge_run_complete(_doc(run_result=rr), ROW_SFT)
    assert verdict == UNKNOWN, why
    assert "epochs_completed" in why


def test_criterion1_failed_ckpt_or_nan_is_red():
    """阳性对照（能红）：检查点不可加载 / 有 NaN/Inf ⇒ 判据① 不通过。"""
    bad_ckpt = _doc(run_result=dict(_doc()["run_result"], final_ckpt_loadable=False))
    assert BF.judge_run_complete(bad_ckpt, ROW_SFT)[0] == FAIL
    bad_nan = _doc(run_result=dict(_doc()["run_result"], nan_or_inf_detected=True))
    assert BF.judge_run_complete(bad_nan, ROW_SFT)[0] == FAIL


def test_criterion1_null_evidence_is_unknown_not_fail():
    """缺证据（null）≠ 失败：检查点/NaN 的 null 判 `unknown`（整批拒收而非写"不通过"）。"""
    for key in ("final_ckpt_loadable", "nan_or_inf_detected"):
        doc = _doc(run_result=dict(_doc()["run_result"], **{key: None}))
        verdict, why = BF.judge_run_complete(doc, ROW_SFT)
        assert verdict == UNKNOWN, (key, why)
        assert key in why


# ── ② CPT 的 loss 支 ─────────────────────────────────────────────────────────
DECLINE = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1]
RISE = list(reversed(DECLINE))


def _cpt_doc(series) -> dict:
    ev = {"after_acc": 0.20, "base_acc": 0.20}
    if series is not None:
        ev["loss_curve"] = {"n": len(series), "first": series[0], "last": series[-1],
                            "min": min(series), "series": list(series)}
    return _doc(eval=ev)


def test_cpt_loss_declining_passes():
    """阳性：loss 首尾窗口均值下降 ∧ after == base ⇒ 判据② 通过。"""
    verdict, why = BF.judge_effect(_cpt_doc(DECLINE), ROW_CPT, FROZEN, None)
    assert verdict == PASS, why
    assert "CPT" in why


def test_cpt_loss_rising_is_red():
    """★ 阴性对照：loss **不降反升** ⇒ 判据② 必须判「不通过」。"""
    verdict, why = BF.judge_effect(_cpt_doc(RISE), ROW_CPT, FROZEN, None)
    assert verdict == FAIL, why
    assert "loss" in why


def test_cpt_missing_loss_is_unknown():
    """★ 阴性对照：缺 loss 读数（无 loss_curve）⇒ 必须 `unknown`，⛔ 不得静默跳过。"""
    verdict, why = BF.judge_effect(_cpt_doc(None), ROW_CPT, FROZEN, None)
    assert verdict == UNKNOWN, why
    assert "loss" in why


def test_cpt_loss_curve_without_series_is_unknown():
    """阴性对照：只给首/末、没有逐步序列 ⇒ 无法按窗口均值口径复算 ⇒ `unknown`。"""
    doc = _doc(eval={"after_acc": 0.20, "base_acc": 0.20,
                     "loss_curve": {"n": 10, "first": 2.0, "last": 1.1, "min": 1.1}})
    verdict, why = BF.judge_effect(doc, ROW_CPT, FROZEN, None)
    assert verdict == UNKNOWN, why
    assert "逐步序列" in why


# ── ③ 检查点"真可加载" ───────────────────────────────────────────────────────
def _write_safetensors(path: Path, tensor_name: str = "w") -> None:
    header = {"__metadata__": {"format": "pt"},
              tensor_name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")


def test_garbage_weights_are_not_loadable(tmp_path: Path):
    """★ 阴性对照：7 字节的"权重文件" ⇒ 最终检查点**不可加载**（旧的形态检查会判 True）。"""
    run = tmp_path / "run"
    ckpt = run / "checkpoint-400"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"garbage")
    loadable, why, _path = MRJ.ckpt_loadable(run)
    assert loadable is False, why
    assert "字节" in why


def test_valid_safetensors_is_loadable(tmp_path: Path):
    """阳性对照：头/张量偏移自洽的 safetensors ⇒ 可加载（证明上一条不是"永远 False"）。"""
    run = tmp_path / "run"
    ckpt = run / "checkpoint-400"
    ckpt.mkdir(parents=True)
    _write_safetensors(ckpt / "model.safetensors")
    loadable, why, _path = MRJ.ckpt_loadable(run)
    assert loadable is True, why


def test_index_with_missing_shard_is_not_loadable(tmp_path: Path):
    """阴性对照：权重索引说有两个分片、盘上只有一个 ⇒ 不可加载（缺件不许当完整）。"""
    ckpt = tmp_path / "checkpoint-400"
    ckpt.mkdir(parents=True)
    _write_safetensors(ckpt / "model-00001-of-00002.safetensors")
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {}, "weight_map": {"a": "model-00001-of-00002.safetensors",
                                       "b": "model-00002-of-00002.safetensors"}}))
    ok, why = MRJ.inspect_checkpoint(ckpt)
    assert ok is False, why
    assert "不存在" in why


def test_lora_adapter_requires_config_and_adapter_weights(tmp_path: Path):
    """阴性对照：有 adapter 权重却没有 adapter_config.json ⇒ 不可加载（LoRA 必须齐件）。"""
    ckpt = tmp_path / "final"
    ckpt.mkdir(parents=True)
    _write_safetensors(ckpt / "adapter_model.safetensors", "base_model.lora_A.weight")
    ok, why = MRJ.inspect_checkpoint(ckpt)
    assert ok is False, why
    assert "adapter_config" in why


# ── ④ 判据④ 的「合并后」「vLLM 起服务」 ──────────────────────────────────────
def _eval_report(**over) -> dict:
    """一份"该过的"评测报告（温度 0 / 10 条全可解析 / 合并产物 / vLLM 服务名）。"""
    report = {
        "schema_version": "graspo-eval-report-1",
        "model": {"served_name": "graspo-eval", "role": "after",
                  "path": "/evalout/merged-hf-m001",
                  "checkpoint_path": "/runs/M001/v0/checkpoint-400",
                  "export_format": "peft-adapter"},
        "dataset": {"sample_count_requested": 10},
        "decoding": {"temperature": 0.0},
        "summary": {"sample_count_total": 10, "sample_count_error": 0, "valid": 10},
        "samples": [{"pred_name": "rotate_arm", "error": None, "parse_error": None}
                    for _ in range(10)],
    }
    report.update(over)
    return report


def test_inference_full_five_elements_passes():
    """阳性：五要素齐（含合并产物 + vLLM 服务名）⇒ judge_inference 判 true。"""
    res = RC.judge_inference(_eval_report(), 10, Path("eval_report.json"), tuner_type="lora")
    assert res["ok"] is True, res["why"]
    assert res["evidence"]["lora_merged"] is True
    assert res["evidence"]["served_by_vllm"] is True


def test_inference_missing_lora_merged_is_unknown():
    """★ 阴性对照（gap 4a）：LoRA 档报告看不出"已合并" ⇒ 必须 `ok=None`（不写键）。"""
    report = _eval_report()
    report["model"] = {"served_name": "graspo-eval", "role": "after",
                       "path": "/runs/M001/checkpoint-400",
                       "checkpoint_path": "/runs/M001/checkpoint-400",
                       "export_format": "base-model"}
    res = RC.judge_inference(report, 10, Path("eval_report.json"), tuner_type="lora")
    assert res["ok"] is None, res["why"]
    assert res["evidence"]["lora_merged"] is None


def test_inference_missing_vllm_evidence_is_unknown():
    """★ 阴性对照（gap 4b）：报告没有 served_name / engine ⇒ 必须 `ok=None`。"""
    report = _eval_report()
    report["model"] = dict(report["model"], served_name="")
    res = RC.judge_inference(report, 10, Path("eval_report.json"), tuner_type="full")
    assert res["ok"] is None, res["why"]
    assert res["evidence"]["served_by_vllm"] is None


def test_inference_unknown_tuner_type_is_unknown():
    """阴性对照：清单缺/不认识 tuner_type ⇒ 不知道适不适用「合并 LoRA」⇒ 必须 `ok=None`。"""
    res = RC.judge_inference(_eval_report(), 10, Path("eval_report.json"), tuner_type="")
    assert res["ok"] is None, res["why"]
    assert "tuner_type" in res["evidence"]["lora_merged_why"]


def test_inference_non_vllm_engine_is_red():
    """阳性对照（能红）：报告 engine 明确不是 vLLM ⇒ 判据④ 判 false（有据的不通过）。"""
    res = RC.judge_inference(_eval_report(engine="tgi"), 10, Path("eval_report.json"),
                             tuner_type="lora")
    assert res["ok"] is False, res["why"]
    assert res["evidence"]["served_by_vllm"] is False


def test_inference_full_tuner_needs_no_merge():
    """全参档不适用「合并 LoRA」⇒ 记 true 并注明（⛔ 不因此要求 merged-hf 目录名）。"""
    report = _eval_report()
    report["model"] = {"served_name": "graspo-eval", "role": "after",
                       "path": "/runs/M035/checkpoint-100",
                       "checkpoint_path": "/runs/M035/checkpoint-100",
                       "export_format": "base-model"}
    res = RC.judge_inference(report, 10, Path("eval_report.json"), tuner_type="full")
    assert res["ok"] is True, res["why"]
    assert "不适用" in res["evidence"]["lora_merged_why"]


def test_inference_parse_failure_is_red():
    """阳性对照（能红）：可解析条数不足 ⇒ 判据④ 必须判 false（旧四条件仍然有效）。"""
    report = _eval_report()
    report["samples"] = report["samples"][:5] + [
        {"pred_name": "", "error": None, "parse_error": "bad json"} for _ in range(5)]
    res = RC.judge_inference(report, 10, Path("eval_report.json"), tuner_type="lora")
    assert res["ok"] is False, res["why"]


if __name__ == "__main__":                      # 允许无 pytest 直接跑
    raise SystemExit(pytest.main([__file__, "-q"]))
