"""D6 —— OPD **教师身份**取证链（`scripts/record_checks.py` × `scripts/gen_eval.py`）。

## 为什么有这套用例（实测驱动，不是凭空造的契约）

M036（OPD / LoRA / ms-swift / 单卡）学生读数早就在手：**after 19% / base 22%**，教师 **48/100**；
但 `eval.teacher_after_acc` **入不了证** ⇒ 判据② 永远落不了结论。根因是一对互斥的写法：

* 消费端 `scripts/record_checks.py`：`t_tier not in Path(ck_path).parts` ⇒ 报告里
  `model.checkpoint_path` 的**某一级目录**必须正好是教师档号（如 `M039`）；
* 产出端 `scripts/gen_eval.py`：教师 LoRA 合并落点写死 `/evalout/merged-hf-teacher`
  —— 路径里根本没有档号 ⇒ 上面那条**必然**不成立（不论评测跑得多干净）。

修法（二者取其一，选这一条）：**让落点带上教师档号**（`merged-hf-teacher-<教师档号>`，
与学生侧 `merged-hf-<档号>` 同一套命名惯例），并由**消费端一个函数**（`teacher_merge_dirname`）
同时供产出端 `import` 使用 —— 产出端与消费端同一份口径（宪法 §1.4），且**判据语义一字不动**；
不选"新增显式 `teacher_tier` 字段"的理由：那个字段是**跑批方自报**的，消费端只能选择相信它，
而路径是 runner 按清单现算出来的产物坐标 ⇒ 前者证明不了产物真在那个目录，后者能。

## 红线

* 缺登记（路径里没有档号，如旧写死的 `…/merged-hf-teacher`）⇒ **拒**（fail-closed）。
* ⛔ 不做子串匹配：`M03` 不得命中 `M039`。
* ⛔ 不改任何判据阈值/口径：本文件只核"这报告评的是不是那个教师档的产物"。
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import record_checks as rc  # noqa: E402

_TEACHER = "M039"


# ── ① 教师档号匹配 ⇒ 通过（两种合法形态：直服训练产物 / 合并落点） ────────────────
@pytest.mark.parametrize("ck_path", [
    "/runs/M039/M039/v0-20261025-120345/checkpoint-200",
    "/runs/M039/M039/final",
    "/evalout/merged-hf-teacher-M039",
])
def test_teacher_tier_match_passes(ck_path):
    assert rc.teacher_tier_in_checkpoint_path(ck_path, _TEACHER) is True, ck_path


# ── ② 不匹配 ⇒ 拒（别的档号的产物冒充不了这个教师） ──────────────────────────────
@pytest.mark.parametrize("ck_path", [
    "/runs/M038/M038/v0-20261025-120345/checkpoint-200",
    "/evalout/merged-hf-teacher-M038",
    "/evalout/merged-hf-m039",                       # 学生侧落点不等于教师档
])
def test_teacher_tier_mismatch_rejected(ck_path):
    assert rc.teacher_tier_in_checkpoint_path(ck_path, _TEACHER) is False, ck_path


# ── ③ 缺登记 ⇒ 拒（旧写死的落点里没有教师档号 ⇒ 入不了证，保持 fail-closed） ────────
@pytest.mark.parametrize("ck_path", [
    "/evalout/merged-hf-teacher",                    # ★ D6 的原始病灶
    "/evalout/merged-hf-teacher-",                   # 前缀有、档号空
    "",
])
def test_missing_registration_rejected(ck_path):
    assert rc.teacher_tier_in_checkpoint_path(ck_path, _TEACHER) is False, ck_path


# ── ④ 不做子串匹配：`M03` 不得命中 `M039`（防"档号前缀相同"的错位） ────────────────
def test_no_substring_matching():
    assert rc.teacher_tier_in_checkpoint_path("/runs/M039/final", "M03") is False
    assert rc.teacher_tier_in_checkpoint_path("/evalout/merged-hf-teacher-M039", "M03") is False


# ── ⑤ 产出端/消费端同一份口径：目录名只有一个来源，且 `gen_eval.py` 不再写死旧落点 ──
def test_merge_dirname_is_single_source():
    assert rc.teacher_merge_dirname(_TEACHER) == "merged-hf-teacher-M039"
    src = (_ROOT / "scripts" / "gen_eval.py").read_text(encoding="utf-8")
    assert '"merged-hf-teacher"' not in src, "产出端仍写死旧的 merged-hf-teacher 落点"
    assert "/evalout/merged-hf-teacher\"" not in src
    assert "teacher_merge_dirname(" in src, "产出端没有用消费端的唯一命名函数"
    # 教师侧 yaml 的 checkpoint_path 取的就是合并落点（merge 形态）⇒ 身份可机核
    assert "ckpt=(md if mode == \"merge\" else cp)" in src


# ── ⑥ 端到端：真实清单算出的教师档 + 旧落点 ⇒ 判据② 拒；新落点 ⇒ 通过 ─────────────
def _manifest():
    p = _ROOT / "tests" / "e2e" / "matrix68_manifest.json"
    if not p.is_file():                                  # pragma: no cover
        pytest.skip("仓内清单不在（只跑单元层）")
    import json
    return json.loads(p.read_text(encoding="utf-8"))


def _report(ck_path, base_path):
    return {
        "schema_version": "eval-report/v1",
        "model": {"role": "other", "checkpoint_path": ck_path, "base_model_path": base_path,
                  "path": ck_path, "export_format": "peft-adapter", "served_name": "graspo-eval",
                  "engine": "vllm"},
        "decoding": {"temperature": 0.0},
        "summary": {"sample_count_total": 100, "sample_count_error": 0, "correct": 48,
                    "sample_count_valid": 100, "accuracy": 0.48, "accuracy_percent": 48.0},
        "dataset": {"sha256": "5ad2d7cc588bab7b1b72750b3070d3c3f2de1690e105d46e541ca333566d3a6d"},
    }


def test_teacher_identity_gate_end_to_end():
    manifest = _manifest()
    teacher = rc.teacher_tier_of(manifest)
    assert teacher is not None, "清单算不出教师档"
    t_tier, t_model = teacher
    sha = "5ad2d7cc588bab7b1b72750b3070d3c3f2de1690e105d46e541ca333566d3a6d"

    def judge(ck_path):
        return rc.judge_teacher(report=_report(ck_path, t_model), expected=100, subset_sha=sha,
                                report_path=Path("/nonexistent-teacher-report.json"),
                                after_report=_report("/x/after", t_model), manifest=manifest)

    # 旧写死落点（没有教师档号）⇒ 教师身份必须核不过（fail-closed）
    bad = judge("/evalout/merged-hf-teacher")
    assert bad["ok"] is False, "旧写死落点（无教师档号）竟然通过了教师身份核验"
    assert any("教师身份不符" in p for p in bad["evidence"]["problems"])

    # 新落点（带教师档号）⇒ 身份这一关必须过，教师读数入证
    good = judge(f"/evalout/{rc.teacher_merge_dirname(t_tier)}")
    assert good["ok"] is True, good["evidence"]["problems"]
    assert good["acc"] == pytest.approx(0.48)
