"""`build_official.py` 的口径机读字段守卫（修正轮 1 · 🟡W1 / 🟡W3）回归。

★ 为什么必须有这个文件：复核报告 🟡W1/🟡W3 的两条缺陷**都不是"值写错"那么轻**——
  它们是"**同一份正式件里两个 status 字段互相矛盾**"与"**拿已作废的读数当口径无关证据**"，
  而且当时的四闸门（verify_matrix / scan_matrix_leaks / regen_field_guard / run_self_proof）
  **一个都不响**。⇒ 只把 9 行值改对不够：必须把**判据本身**固化成能失败的断言。

本文件固化三件事：
  ① `caliber_field_guard()` 对 `status_after != status` 的**四态矛盾**必须报出（W1）；
  ② 对"引用已作废评测侧读数却不当场标记"的过度断言必须报出（W3）；
  ③ **真实在盘件**在这两条上都必须干净（否则这份件不能落）。

★ fresh-eyes 说明：断言只依赖 `build_official.caliber_field_guard` 与在盘 `docs/*.data.json`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_MATRIX_DIR = _HERE.parents[2] / "tools" / "matrix"
sys.path.insert(0, str(_MATRIX_DIR))
_SPEC = importlib.util.spec_from_file_location("build_official_uut", _MATRIX_DIR / "build_official.py")
assert _SPEC is not None and _SPEC.loader is not None
bo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bo)

_ON_DISK = _HERE.parents[2] / "docs" / "capability-matrix.data.json"


def _row(tier: str, status: str, *, status_after=None, cie=None) -> dict:
    r: dict = {"tier": tier, "status": status}
    if status_after is not None:
        r["eval_caliber_consistency"] = {"consistent": False, "status_after": status_after}
    if cie is not None:
        r["caliber_independent_evidence"] = cie
    return r


# ── ① W1：status_after 必须与最终 status 自洽（四态 vs 四态）──────────────────


def test_guard_passes_when_status_after_matches():
    assert bo.caliber_field_guard([_row("T052", "未纳入验收", status_after="未纳入验收")]) == []


def test_guard_flags_contradicting_status_after():
    """★ W1 的**阳性对照**：`status_after` 与最终 `status` 相反 ⇒ 必须报出（复核报告点名的形态）。"""
    # ★ 2026-10-21（状态词表四态 → 三态：通过 / 不通过 / 待测试）：原四处夹具里
    #   ("未纳入验收", "已验证不可用") 这一对在新词表下**归一到同一态**（都 ⇒ 不通过）
    #   ⇒ 按定义**不再构成矛盾**（⛔ 不是把断言放宽：仍要求"相反必报出"）。
    #   处置 = 把夹具换成三态下的**相反对**（三个方向齐备），并保留一对"旧字面 × 新状态"的相反组合
    #   （旧四态字面仍可读入，⛔ 不再要求它们在彼此之间可分）。
    for after, status in (("待测试", "通过"), ("待测试", "不通过"),
                          ("通过", "待测试"), ("不通过", "待测试"),
                          ("已验证不可用", "待测试")):
        probs = bo.caliber_field_guard([_row("T0XX", status, status_after=after)])
        assert probs, (after, status)
        assert "status_after" in probs[0]


def test_guard_normalises_raw_status_literals():
    """★ 归一化：`status_after` 可能仍是**原始字面**（`失败*`/`通过*`），⛔ 不得误判为矛盾。"""
    assert bo.caliber_field_guard([_row("T0XX", "已验证不可用", status_after="失败")]) == []
    assert bo.caliber_field_guard([_row("T0XX", "已验证可用", status_after="通过（判据①②③）")]) == []
    assert bo.caliber_field_guard([_row("T0XX", "尚未验证", status_after="—")]) == []
    # 反向：原始字面归一后仍矛盾 ⇒ 必须报出
    assert bo.caliber_field_guard([_row("T0XX", "已验证可用", status_after="失败")])


def test_state_literal_matches_build_official_derivation():
    """`_state_literal` 与 `main()` 的派生链同源（⛔ 不另立口径表）。"""
    assert bo._state_literal("通过") == bo.S_AVAILABLE
    assert bo._state_literal("失败（OOM）") == bo.S_UNAVAILABLE
    assert bo._state_literal("—") == bo.S_UNVERIFIED
    assert bo._state_literal("未纳入验收") == bo.S_EXCLUDED


# ── ② W3：已作废读数不得作为"口径无关"举证 ─────────────────────────────────


_OID = "0/702（旧 702 行评测集读数）"


def test_guard_flags_voided_reading_used_as_evidence():
    """★ W3 的**阳性对照**：CIE 引用 `0/702` 却未声明排除 ⇒ 必须报出（复核报告点名的旧文）。"""
    cie = {"evidence": f"训练侧退化 ⇒ {_OID}，而 base 为合法 JSON ⇒ 两侧自洽的真退化。",
           "voided_readings_excluded_from_evidence": False}
    probs = bo.caliber_field_guard([_row("T037", "已验证不可用", status_after="已验证不可用", cie=cie)])
    assert probs and "voided_readings_excluded_from_evidence" in probs[0]


def test_guard_flags_voided_reading_without_explicit_marking():
    """声明了排除标志、但文本没有逐字「已作废 … ⛔ 不作举证」⇒ 仍判过度断言。"""
    cie = {"evidence": f"评测期 after ⇒ {_OID}", "voided_readings_excluded_from_evidence": True}
    probs = bo.caliber_field_guard([_row("T037", "已验证不可用", status_after="已验证不可用", cie=cie)])
    assert probs and "过度断言" in probs[0]


def test_guard_passes_when_voided_reading_is_explicitly_excluded():
    """★ 正确形态：引用了已作废读数，但**显式**标「已作废」＋「⛔ 不作举证」＋机读标志 ⇒ 通过。"""
    cie = {
        "evidence": (f"★ **评测侧读数已作废、⛔ 不作举证**：旧读数的 {_OID} 属口径变更作废读数；"
                     "★ 本字段不以它作为口径无关的证明。"),
        "voided_readings_excluded_from_evidence": True,
    }
    assert bo.caliber_field_guard([_row("T037", "已验证不可用", status_after="已验证不可用", cie=cie)]) == []


def test_guard_ignores_evidence_without_voided_readings():
    """不含已作废读数的证据（如首步 OOM 的硬件证据）⇒ 本项不判否。"""
    cie = {"evidence": "P-25 首步实测 OOM（四 rank 各 79.08 GiB）—— 硬件不可行，与评测集大小无关。",
           "evidence_kind": "hardware_oom", "voided_readings_excluded_from_evidence": True}
    assert bo.caliber_field_guard([_row("T049", "已验证不可用", status_after="已验证不可用", cie=cie)]) == []


# ── ③ 真实在盘件在这两条上必须干净 ──────────────────────────────────────────


def test_on_disk_official_data_passes_the_guard():
    """★ 在盘 `docs/capability-matrix.data.json`：`caliber_field_guard` 必须 0 问题。"""
    if not _ON_DISK.is_file():
        pytest.skip("正式机读件不在盘")
    doc = json.loads(_ON_DISK.read_text(encoding="utf-8"))
    assert bo.caliber_field_guard(doc["rows"]) == []


def test_on_disk_status_after_matches_status_on_every_row():
    """★ W1 在盘复核（独立于上面的守卫实现）：逐行 `status_after` 与 `status` 归一后必须相等。"""
    if not _ON_DISK.is_file():
        pytest.skip("正式机读件不在盘")
    doc = json.loads(_ON_DISK.read_text(encoding="utf-8"))
    bad = []
    for r in doc["rows"]:
        ecc = r.get("eval_caliber_consistency")
        if not isinstance(ecc, dict):
            continue
        sa = ecc.get("status_after")
        if sa is None:
            continue
        if bo._state_literal(str(sa)) != bo._state_literal(str(r.get("status") or "")):
            bad.append((r.get("tier"), sa, r.get("status")))
    assert bad == [], f"status_after 与 status 相反：{bad}"


def test_on_disk_cie_never_presents_voided_reading_as_evidence():
    """★ W3 在盘复核：任何引用 `0/702` 的 CIE 都必须显式排除（且文本含「不作举证」）。"""
    if not _ON_DISK.is_file():
        pytest.skip("正式机读件不在盘")
    doc = json.loads(_ON_DISK.read_text(encoding="utf-8"))
    offenders = []
    for r in doc["rows"]:
        cie = r.get("caliber_independent_evidence")
        if not isinstance(cie, dict):
            continue
        blob = " ".join(str(cie.get(k) or "") for k in ("evidence", "note", "why"))
        if "0/702" not in blob:
            continue
        if cie.get("voided_readings_excluded_from_evidence") is not True \
                or "已作废" not in blob or "不作举证" not in blob:
            offenders.append(r.get("tier"))
    assert offenders == [], f"把已作废读数当举证的档：{offenders}"


def test_addendum_never_restates_a_concrete_four_state():
    """★ W1 的文本面：追补句结尾⛔ 不复述具体四态值（否则保留结论/未纳入验收的档会自相矛盾）。"""
    for kind in ("lr", "eval", "both"):
        text = bo.void_addendum_for("未纳入验收", kind)
        assert "（尚未验证）" not in text, text
        assert "以本行现行 `status` 字段为准" in text, text


# ── ④ W-8 机核化（修正轮 2）：prior_chain 规范形 ─────────────────────────────


def _row_with_chain(tier: str, prior, chain, corrected_to, status="已验证不可用") -> dict:
    return {
        "tier": tier,
        "status": status,
        "eval_caliber_consistency": {
            "consistent": False,
            "status_after": status,
            "status_after_correction_20261016": {
                "prior": prior, "prior_chain": chain, "corrected_to": corrected_to, "reason": "x",
            },
        },
    }


def test_guard_passes_canonical_prior_chain():
    assert bo.caliber_field_guard([_row_with_chain("T037", "尚未验证", ["尚未验证", "失败"], "已验证不可用")]) == []


def test_guard_flags_chain_not_starting_with_prior():
    """★ W-8 阳性对照：链首不是 `prior`（旧实现把中间值当首项）⇒ 必须报出。"""
    probs = bo.caliber_field_guard([_row_with_chain("T037", "尚未验证", ["失败", "已验证不可用"], "已验证不可用")])
    assert any("prior_chain" in p for p in probs), probs


def test_guard_flags_chain_containing_corrected_to():
    """★ 链里含现行态（= 非幂等增长的形态）⇒ 必须报出。"""
    probs = bo.caliber_field_guard([_row_with_chain("T049", "尚未验证", ["尚未验证", "已验证不可用"], "已验证不可用")])
    assert any("corrected_to" in p for p in probs), probs


def test_guard_flags_chain_with_duplicates():
    """★ 链有重复项（= 每跑一次 +1 的形态）⇒ 必须报出。"""
    probs = bo.caliber_field_guard([_row_with_chain("T050", "尚未验证", ["尚未验证", "尚未验证"], "已验证不可用")])
    assert any("重复" in p for p in probs), probs


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ A1（指挥裁定）：**发布投影不得含内部来源键**（架构级防回流）
#
# 背景（新事实）：矩阵整理包把 `docs/capability-matrix.data.json` 收成**面向使用者的发布投影**
#   （987,561 → 319,562 B），按 §17.7 剔除了 `rows[].eval_caliber_consistency`（含 `prior_chain`）
#   等非渲染字段 —— 该字段留在发布件会让 `scan_matrix_leaks` **命中 40 处**
#   （真实宿主路径 11 / 数据集名 11 / 内部称谓 18），与「矩阵不得写入宿主路径、内网路径、
#   内部节点与内部账号」**直接冲突**。
# ⇒ 原先那条「在盘发布件恰有 9 条 `prior_chain`」的断言**前提消失**（现为 0 条）。
#   ★ 指挥裁定：⛔ 不是"改期望值变绿"，而是换成**两条更强的断言** ——
#     本段（⑤）= 架构级不变量（防回流）；下段（⑥）= 单元级幂等（保住 W-8 守卫）。
#   ★ 两条都**不依赖在盘发布件的具体字段内容**，且**不依赖 `.local/`**（CI 全绿）。
# ══════════════════════════════════════════════════════════════════════════════

#: ★ **内部来源键（钉住，⛔ 与 `HUMAN_VIEW_ROW_KEYS` 解耦）** —— 即使有人把某个键**加进允许清单**
#: 让 ⑤-1 转绿，本节仍会红。
#: 来源可复算：「内部权威 `.local/…/matrix-data.json` 的行键 − `HUMAN_VIEW_ROW_KEYS`」
#:          ＋ 本轮新查出会由**重生成回流**的键（`voided_conclusion` / `conclusion_ref` /
#:          `list_stale` / `retest` / `status_before_derivation` / `matrix_conclusion` /
#:          `feasibility_model`）。
_INTERNAL_ROW_KEYS = frozenset({
    "eval_caliber_consistency", "caliber_independent_evidence", "eval_caliber", "caliber_verdict",
    "caliber_cause", "caliber_note", "reading_consistency", "reading_validity", "corrections",
    "evidence_paths", "provenance_command", "run_path", "run_provenance", "code_identity",
    "tree_fingerprint", "tree_fingerprint_role", "tree_fingerprint_drift",
    "tree_fingerprint_remeasured", "spec_status", "measurement_pending", "elapsed_evidence",
    "p5p6_readings", "after_acc", "base_acc", "n_correct", "n_total", "anomalies",
    "superseded_note", "step_count_reconciliation", "steps_alignment_note_20261016",
    "steps_generation_note", "not_evaluable_brief", "not_evaluable_reason",
    "not_executable_reason", "executability_note", "family_scope_note", "cards_note",
    "channel_gap_note", "delta_pp", "eval_dataset", "eval_dataset_anchor_correction",
    "gpus_note", "final_formats_note", "metric_absence_note",
    "note_extra_steps_formula_issue",
    # 重生成回流面（本轮新查出）
    "voided_conclusion", "conclusion_ref", "list_stale", "retest",
    "status_before_derivation", "matrix_conclusion", "feasibility_model",
})

#: ★ 内部来源 **meta 级**键（同上，钉住）。
_INTERNAL_META_KEYS = frozenset({
    "authoritative_source", "backfill", "common_params_policy", "data_revision",
    "lora_restoration", "lr_policy", "result_placeholder", "ruled_out", "spec_source",
    "subset_size_semantics", "doc_note", "stale_artifacts_20261016",
})


#: ★ **允许出现在嵌套位置的同名键**（⛔ 不是放宽：这些键名在**允许容器**里有正当语义）。
#: `after_acc`/`base_acc`/`delta_pp` = `rows[].caliber_change.reading_before/reading_after` 的子键；
#: `eval_caliber`/`lr_policy`/`ruled_out` = `meta.criteria_caliber` 的子键（∈ `HUMAN_VIEW_CALIBER_KEYS`）。
#: ★ 本表**自校验**（见 ⑤-5）：每个被排除的键必须**确实**以嵌套形式出现，且其容器在允许清单里
#:   ⇒ ⛔ 不许靠往这张表里塞键来把红变绿。
_NESTED_ALLOWED_KEYS = {
    "after_acc": "caliber_change", "base_acc": "caliber_change", "delta_pp": "caliber_change",
    "eval_caliber": "criteria_caliber", "lr_policy": "criteria_caliber", "ruled_out": "criteria_caliber",
}


def _all_keys(obj, *, _acc=None) -> set:
    """递归收集文档里出现过的**所有 dict 键名**（含嵌套，⛔ 只看结构不看值）。"""
    acc = set() if _acc is None else _acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(str(k))
            _all_keys(v, _acc=acc)
    elif isinstance(obj, list):
        for v in obj:
            _all_keys(v, _acc=acc)
    return acc


def _nested_keys(obj, *, _top=True, _acc=None) -> set:
    """收集**非顶层**出现的键名（用于自校验 `_NESTED_ALLOWED_KEYS`）。"""
    acc = set() if _acc is None else _acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not _top:
                acc.add(str(k))
            _nested_keys(v, _top=False, _acc=acc)
    elif isinstance(obj, list):
        for v in obj:
            _nested_keys(v, _top=_top, _acc=acc)
    return acc


def _release_doc() -> dict:
    return json.loads(_ON_DISK.read_text(encoding="utf-8"))


def test_release_row_keys_are_within_the_allow_list():
    """★ ⑤-1（机读不变量）：发布件**每一行**的键必须 ⊆ `build_official.HUMAN_VIEW_ROW_KEYS`。

    ★ 白名单**从代码里取**（⛔ 不在本文件手抄一份）⇒ 与投影实现同源；投影一放水就红。
    """
    if not _ON_DISK.is_file():
        pytest.skip("正式机读件不在盘")
    doc = _release_doc()
    allowed = set(bo.HUMAN_VIEW_ROW_KEYS)
    extra: dict = {}
    for r in doc["rows"]:
        bad = sorted(set(r) - allowed)
        if bad:
            extra[str(r.get("tier"))] = bad
    assert extra == {}, f"发布投影出现允许清单之外的行级键（内部字段回流）：{extra}"


def test_release_meta_keys_are_within_the_allow_list():
    """★ ⑤-2：发布件 meta 键必须 ⊆ `HUMAN_VIEW_META_KEYS`；`criteria_caliber` 子键同理。"""
    if not _ON_DISK.is_file():
        pytest.skip("正式机读件不在盘")
    doc = _release_doc()
    meta = doc.get("meta") or {}
    assert sorted(set(meta) - set(bo.HUMAN_VIEW_META_KEYS)) == []
    cc = meta.get("criteria_caliber")
    if isinstance(cc, dict):
        assert sorted(set(cc) - set(bo.HUMAN_VIEW_CALIBER_KEYS)) == []


def test_release_carries_no_internal_source_key_name():
    """★ ⑤-3（**具名钉住 ＋ 字节面**，⛔ 与允许清单解耦）：内部来源键不得出现在发布件。

    三层判据（都只在**投影真正管辖的位置**判，⛔ 不误伤允许容器里的同名子键）：
      ① 行**顶层** ∩ `_INTERNAL_ROW_KEYS` == ∅；
      ② meta **顶层** ∩ `_INTERNAL_META_KEYS` == ∅；
      ③ **字节面**：`(_INTERNAL_* − _NESTED_ALLOWED_KEYS)` 里任何键名都不得以 `"键名"` 出现。
    ★ 即使有人把某个内部键**加进 `HUMAN_VIEW_*` 允许清单**让 ⑤-1/⑤-2 转绿，本用例仍会红。
    """
    if not _ON_DISK.is_file():
        pytest.skip("正式机读件不在盘")
    doc = _release_doc()
    row_top = set().union(*[set(r) for r in doc["rows"]]) if doc["rows"] else set()
    meta_top = set(doc.get("meta") or {})
    assert sorted(row_top & _INTERNAL_ROW_KEYS) == [], "发布件行顶层出现内部来源键"
    assert sorted(meta_top & _INTERNAL_META_KEYS) == [], "发布件 meta 顶层出现内部来源键"
    sensitive = sorted((_INTERNAL_ROW_KEYS | _INTERNAL_META_KEYS) - set(_NESTED_ALLOWED_KEYS))
    raw = _ON_DISK.read_text(encoding="utf-8")
    leaked = [k for k in sensitive if f'"{k}"' in raw]
    assert leaked == [], f"发布件字节面出现内部键名：{leaked}"
    # ⑤-5 自校验：排除表不许被悄悄扩大 —— 每个被排除的键必须**确实**嵌套出现，且容器在允许清单里
    nested = _nested_keys(doc)
    for k, container in _NESTED_ALLOWED_KEYS.items():
        assert k in nested, f"排除表含未真正嵌套出现的键 {k!r} ⇒ ⛔ 不许靠排除表把红变绿"
        assert container in (set(bo.HUMAN_VIEW_ROW_KEYS) | set(bo.HUMAN_VIEW_META_KEYS)), container


def test_projection_function_really_drops_internal_keys():
    """★ ⑤-4（**机制级**）：`drop_non_view_fields()` 对含内部键的行／meta 必须**真剔除**，
    且剔除后 `forbidden_in_human_view()` 必须干净。

    ★ 阳性对照（同一函数内构造）：往会被渲染的字段注入一条内部术语 ⇒ 兜底扫描**必须报出**
    （证明"扫过了"≠"扫得到"）。
    """
    row = {"tier": "T031", "status": "尚未验证", "note": "★ 结论"}
    row.update({k: "…" for k in ("eval_caliber_consistency", "evidence_paths", "run_provenance")})
    doc = {"meta": {"title": "t", "doc_note": "internal"}, "rows": [row]}
    dropped_meta, counts, _dropped_cal = bo.drop_non_view_fields(doc)
    assert sorted(set(doc["rows"][0]) & _INTERNAL_ROW_KEYS) == []
    assert "doc_note" in dropped_meta and counts.get("eval_caliber_consistency") == 1
    assert bo.forbidden_in_human_view(doc) == []
    # 阳性对照：注入内部术语 ⇒ 必报
    doc["rows"][0]["note"] = "★ 本包修正轮的结论"
    assert bo.forbidden_in_human_view(doc) != []


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ A1（指挥裁定）：**并入/规范化函数幂等**（单元级；保住 W-8「并入器非幂等」守卫）
#
# ★ 为什么换到这里：`prior_chain` 的**规范化/并入**原先只由工位并入器承担
#   （`.local/…/merge_fix_r2.py::canonical_prior_chain`），**不在仓内公开面** ⇒
#   仓内可 import 的、且属同一失败类的，是 `build_official.py` 的**并入/规范化四件**
#   （剔除投影 / 逐档直通 / 作废结论包裹 / 相邻作废标记）+ 守卫 ③ 的**规范形判据**。
#   W-8 的失败形态 = 「每跑一次就追加一次 ⇒ 链长 +1」，本段用**夹具**把该形态钉死：
#   ① 并入管线连跑两次必须逐字节相同；② 规范形是守卫的**固定点**；③ 「第二遍并入」的增长
#   形态必须被守卫**拒**。★ 全部用夹具，⛔ 不读在盘发布件、⛔ 不依赖 `.local/`。
# ══════════════════════════════════════════════════════════════════════════════


def _merge_fixture() -> dict:
    """并入夹具：一行含内部键 ＋ 一行含规范形 `prior_chain` ＋ 一行含可被包裹的作废结论。"""
    return {
        "meta": {"title": "t", "doc_note": "internal", "rows_extra": 1},
        "rows": [
            {
                "tier": "T037", "status": "已验证不可用", "model": "m", "algo": "a", "gpus": [1],
                "note": "★ 结论：本档不可用",
                "note_full": "★ 结论全文（失败：首步 OOM，四 rank 各 79.08 GiB）",
                "eval_caliber_consistency": {
                    "consistent": False, "status_after": "已验证不可用",
                    "status_after_correction_20261016": {
                        "prior": "尚未验证", "prior_chain": ["尚未验证", "失败"],
                        "corrected_to": "已验证不可用", "reason": "x",
                    },
                },
                "caliber_independent_evidence": {"evidence": "x"},
                "evidence_paths": ["a/b.md"], "run_provenance": "…",
                "final_formats": ["jsonl"],
            },
        ],
    }


_MERGE_SRC = [{"tier": "T037", "final_formats": ["jsonl"]}]


def _run_merge_pipeline(doc: dict) -> dict:
    """把 `build_official` 的并入/规范化四件按 `main()` 的次序跑一遍（★ 经 `bo.` 取属性 ⇒ 可被 monkeypatch）。"""
    bo.drop_non_view_fields(doc)
    bo._carry_through_row_fields(doc, _MERGE_SRC)
    bo.wrap_voided_conclusions(doc["rows"])
    bo.ensure_adjacent_void_markers(doc["rows"])
    return doc


def _merge_twice() -> tuple:
    import copy

    first = _run_merge_pipeline(copy.deepcopy(_merge_fixture()))
    second = _run_merge_pipeline(copy.deepcopy(first))
    return (json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True))


def test_merge_pipeline_is_idempotent():
    """★★ ⑥-1：**同输入连跑两次产出逐字节相同**（夹具；⛔ 不依赖在盘件与 `.local/`）。"""
    once, twice = _merge_twice()
    assert once == twice, "并入/规范化管线**非幂等**（第二遍仍改字节 ⇒ 每次重跑都动已冻结内容）"


def test_prior_chain_canonical_form_is_a_fixed_point():
    """★★ ⑥-2：规范形必须被守卫**接受**，且"再并入一遍"不得改动它（固定点）。"""
    rec = _merge_fixture()["rows"][0]
    assert bo.caliber_field_guard([rec]) == []
    _run_merge_pipeline(_merge_fixture())          # 过一遍管线后仍应逐字节相同
    assert bo.caliber_field_guard([rec]) == []


def test_prior_chain_second_pass_growth_is_rejected():
    """★★ ⑥-3（**W-8 形态**）：把「第二遍并入」的增长形态喂给守卫 ⇒ **必须报出**。

    形态 = 旧并入器的 `prior_chain + [status_after]`（`status_after == corrected_to`）
    ⇒ 链里混入现行态 ＋ 每跑一次 +1。守卫 ③ 正是为此而设。
    """
    import copy

    rec = _merge_fixture()["rows"][0]
    grown = copy.deepcopy(rec)
    cur = grown["eval_caliber_consistency"]["status_after"]
    grown["eval_caliber_consistency"]["status_after_correction_20261016"]["prior_chain"] = \
        list(rec["eval_caliber_consistency"]["status_after_correction_20261016"]["prior_chain"]) + [cur]
    probs = bo.caliber_field_guard([grown])
    assert probs and any("corrected_to" in p or "重复" in p for p in probs), probs


def test_idempotency_and_growth_checks_can_fail(monkeypatch):
    """★★ **两条新断言的阳性对照**（能失败的断言才叫断言）：

    ① 把并入管线里的「逐档直通」换成**每次追加**（= 非幂等形态）⇒ ⑥-1 的判据**必须红**；
    ② 把守卫 ③（`prior_chain` 规范形判据）**去掉** ⇒ ⑥-3 的判据**必须红**（增长形态漏过）。
    """
    once, twice = _merge_twice()
    assert once == twice                     # 正常态先绿（同一判据）

    def _non_idempotent_carry(doc, src_rows):  # 注入重复追加
        for r in doc.get("rows", []):
            if str(r.get("tier")) == "T037":
                r["final_formats"] = list(r.get("final_formats") or []) + ["__second_pass__"]
        return 1

    monkeypatch.setattr(bo, "_carry_through_row_fields", _non_idempotent_carry)
    once2, twice2 = _merge_twice()
    assert once2 != twice2, "注入重复追加后判据竟然没有变红 ⇒ 该断言无检测力"

    # ② 去掉守卫 ③ 的规范形判据 ⇒ 增长形态不再被拦
    _orig = bo.caliber_field_guard
    monkeypatch.setattr(bo, "caliber_field_guard",
                        lambda rows: [p for p in _orig(rows) if "prior_chain" not in p])
    import copy

    rec = _merge_fixture()["rows"][0]
    grown = copy.deepcopy(rec)
    cur = grown["eval_caliber_consistency"]["status_after"]
    chain = grown["eval_caliber_consistency"]["status_after_correction_20261016"]["prior_chain"]
    grown["eval_caliber_consistency"]["status_after_correction_20261016"]["prior_chain"] = list(chain) + [cur]
    assert bo.caliber_field_guard([grown]) == [], "去掉判据后增长形态仍被拦 ⇒ 阳性对照构造无效"
    monkeypatch.setattr(bo, "caliber_field_guard", _orig)
    assert any("prior_chain" in p for p in bo.caliber_field_guard([grown])), "恢复判据后必须报出"


# ── ⑦ 可选（⛔ **不得成为通过的前提**）：内部权威在盘可达时，核那 9 条链 ──────

_AUTHORITY = (_HERE.parents[2] / ".local" / "hb-workspace" / "20261008-105451"
              / "task-matrix-html" / "matrix-data.json")


def test_internal_authority_prior_chains_are_canonical():
    """★ ⑦（可跳过）：内部权威仍保留 9 条修正记录 ⇒ 可达时逐条核规范形。

    ★ **CI 无 `.local/` ⇒ 本用例 skip，其余用例必须仍全绿**（本条⛔ 不作为任何断言的前提）。
    """
    if not _AUTHORITY.is_file():
        pytest.skip("内部权威不在盘（CI 常态）⇒ 仅在本地核这条")
    doc = json.loads(_AUTHORITY.read_text(encoding="utf-8"))
    n = 0
    for r in doc["rows"]:
        ecc = r.get("eval_caliber_consistency")
        rec = ecc.get("status_after_correction_20261016") if isinstance(ecc, dict) else None
        if not isinstance(rec, dict) or not isinstance(rec.get("prior_chain"), list):
            continue
        n += 1
        ch, prior, corr = rec["prior_chain"], rec.get("prior"), rec.get("corrected_to")
        assert str(ch[0]) == str(prior), (r.get("tier"), ch)
        assert len(ch) == len({str(x) for x in ch}), (r.get("tier"), ch)
        assert all(str(x) != str(corr) for x in ch), (r.get("tier"), ch, corr)
    assert n == 9, f"内部权威的修正记录数应为 9，实为 {n}"
