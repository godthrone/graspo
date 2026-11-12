"""E1 单测：备注列「说人话」口径（`scripts/note_style.py` 是唯一真相源）。

病史（2026-10-27 主席裁定）：备注列曾被写成「工程日志」——M011 一格 1324 字，含整段 traceback、
正则原文、`/workspace/…/lora.py:170`、`stdout.log:20`、`GPU 3`。表格只该说结论。

本文件钉住 5 件事：
① 报错只留「异常类名 ＋ 正文前 ≤40 字…」；② 正常档保留全套结论；③ 缺证据档如实说缺口；
④ 超长必截断且收口「详见跑次证据」；⑤ 判据结论只写编号级（过/未过/未判定）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import note_style as ns  # noqa: E402

M011_ERROR = ("ValueError: lora.target_modules is required (a single real HF module-name regex; "
              "the regex form is the default because the cost is only for the modules that the "
              "weights come from)")


def _normal_facts(**over):
    facts = {"exit_code": 0, "steps": 400, "steps_src": "读自跑次读数文件",
             "epochs_done": 4, "epochs_planned": 4, "ckpt": "可加载", "naninf": "无",
             "peak": "30913.0 MiB", "after_acc": 0.46, "base_acc": 0.20, "delta_pp": 26.0,
             "fail_ids": [], "unk_ids": [], "judged": True, "gaps": 0}
    facts.update(over)
    return facts


# ① 崩溃档摘要 ────────────────────────────────────────────────────────────────
def test_crash_error_is_summarised_not_pasted():
    note = ns.render({"crash": ns.clip_error(M011_ERROR), "exit_code": 1, "steps": 0,
                      "epochs_done": 0, "epochs_planned": 4, "ckpt": "不可加载",
                      "naninf": "无", "peak": "425.0 MiB", "gaps": 0})
    assert note.startswith("训练崩溃，未产生有效训练：ValueError: lora.target_modules is required"), note
    # 英文帮助长句 / 正则原文 / 括号原文都不许进来
    assert "single real HF module-name" not in note and "(?:" not in note and "(" not in note
    assert note.endswith(ns.TAIL), note
    assert len(note) <= ns.NOTE_LIMIT


def test_clip_error_caps_body_at_40_chars():
    long_msg = "x" * 200
    out = ns.clip_error(f"RuntimeError: {long_msg}")
    assert out.startswith("RuntimeError: ")
    assert out.endswith("…") and len(out) <= len("RuntimeError: ") + ns.ERROR_BODY_LIMIT + 1


# ② 正常档：保留全套结论 ──────────────────────────────────────────────────────
def test_normal_tier_keeps_every_conclusion():
    note = ns.render(_normal_facts())
    for token in ("退出码 0", "实跑步数 400", "完成 4/4 个 epoch", "检查点：可加载",
                  "数值异常：无", "峰值显存：30913.0 MiB",
                  "效果读数：after 46.0% / base 20.0% / Δ+26.0pp", "判据结论：①②③④ 过"):
        assert token in note, (token, note)
    assert len(note) <= ns.NOTE_LIMIT, note
    assert not ns.has_noise(note), note


# ③ 缺证据档：如实说缺口，不编结论 ────────────────────────────────────────────
def test_missing_evidence_tier_states_gaps_honestly():
    note = ns.render({"exit_code": None, "steps": None, "ckpt": "未取证", "naninf": "未取证",
                      "after_acc": None, "base_acc": None, "gaps": 7})
    for token in ("退出码缺", "实跑步数未知", "检查点：未取证", "数值异常：未取证",
                  "效果读数：不可评", "证据缺口 7 项"):
        assert token in note, (token, note)
    assert note.endswith(ns.TAIL), note


# ④ 超长必截断 ────────────────────────────────────────────────────────────────
def test_overlong_note_is_truncated_at_clause_boundary():
    facts = _normal_facts(warns=[f"告警 {i}：这是一条很长的告警文本用来把备注顶过上限" for i in range(12)])
    note = ns.render(facts)
    assert len(note) <= ns.NOTE_LIMIT, len(note)
    assert note.endswith(ns.TAIL), note
    # 截断必须落在从句边界（⛔ 不许把一句话拦腰截断）
    assert note[: -len(ns.TAIL)].endswith(("MiB", "过", "pp", "epoch", "无", "不可评", "可加载",
                                           "不可加载", "未取证")) or "；" in note


# ⑤ 判据结论：只写编号级 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("fails,unks,expect", [
    ([], [], "判据结论：①②③④ 过"),
    (["判据① 未过：退出码 1 ≠ 0"], [], "判据结论：②③④ 过，① 未过"),
    (["判据① 未过：退出码 1 ≠ 0"],
     ["判据② 缺证据：eval.after_acc", "判据③ 缺证据：checks.algo_checks_passed"],
     "判据结论：④ 过，① 未过，②③ 未判定"),
])
def test_criteria_line_is_id_level_only(fails, unks, expect):
    f_ids, u_ids = ns.criteria_ids(fails, unks)
    note = ns.render(_normal_facts(fail_ids=f_ids, unk_ids=u_ids))
    assert expect in note, note
    assert "eval.after_acc" not in note, "判据原因串原文⛔ 不进人读件"


def test_normalize_legacy_is_idempotent_on_rendered_note():
    note = ns.render(_normal_facts(fail_ids=["②"]))
    assert ns.normalize_legacy(note) == note


# ⑥ 超限时丢谁（2026-11-10 缺陷：按输出顺序从尾部丢 ⇒ 尾部追加的判据结论第一个死）──────
def test_overlong_note_drops_the_annotation_not_the_verdict():
    """附注很长顶穿上限时：**判据结论必须保住**，长附注被丢，且报告点附注的名。

    病史（同日两次事故）：旧实现 `while kept: kept.pop()` 从**尾部**丢，而「判据结论」正是
    尾部追加的 ⇒ 最要紧的判据结论第一个被丢，最不要紧的长附注反倒留着（M049 的 McNemar
    从句被附注挤掉、B1 的整句附注被丢）。
    """
    long_extra = "实际训练 2500.3 s，台账 1580 s 为外层包装截断" + "，另有一段很长的补充说明" * 6
    facts = _normal_facts(extra=long_extra)

    text = ns.render(facts)
    assert len(text) <= ns.NOTE_LIMIT, len(text)
    assert "判据结论：①②③④ 过" in text, f"超限后判据结论被丢（旧缺陷复发）：{text}"
    assert "附注：" not in text, text

    text2, report = ns.render_with_report(facts)
    assert text2 == text, "render() 必须与 render_with_report()[0] 逐字相同（向后兼容）"
    assert report["truncated"] is True                 # 前提：真的超限并丢了东西
    assert report["dropped"], "丢了从句却给空报告 ＝ 静默丢"
    assert report["dropped"][0]["kind"] == "extra", report["dropped"]
    assert report["dropped"][0]["text"].startswith("附注："), report["dropped"]
    assert "criteria" not in [d["kind"] for d in report["dropped"]], report["dropped"]


def test_report_names_the_verdict_when_it_is_the_one_dropped(monkeypatch):
    """负例（判别力）：判据结论**真的**被丢时，报告必须点名它，⛔ 不许只报"有丢失"。

    在 200 字上限内「判据结论」（优先级 100）是最不容易倒的那句——那正是修法的目的所在；
    所以本负例把上限压到比单条判据结论还小（判据句 11 字 ＋ 收口 8 字），专测最后一层：
    判据结论倒下时，报告里能不能指名道姓。
    """
    monkeypatch.setattr(ns, "NOTE_LIMIT", 15)
    facts = _normal_facts()

    text, report = ns.render_with_report(facts)
    assert "判据结论" not in text, text
    assert report["truncated"] is True
    assert report["dropped"], "丢了从句却给空报告 ＝ 静默丢"
    assert [d["kind"] for d in report["dropped"]][-1] == "criteria", report["dropped"]
    assert report["dropped"][-1]["label"] == "判据结论", report["dropped"]
    assert report["dropped"][-1]["text"].startswith("判据结论："), report["dropped"]
