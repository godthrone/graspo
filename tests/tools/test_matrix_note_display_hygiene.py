"""备注列（人读件「实验数据」最后一格）的**展示卫生**：⛔ 不写「计划 N 步」、⛔ 不写内部记号。

## 主席裁定（D29，2026-10-27）

1. **删掉「计划 N 步」从句**（`scripts/make_result_json.py` 备注模板 + `tools/matrix/backfill_matrix.py`
   判据① 台账句）：矩阵是给人读的唯一真相源，一格内并排「实跑 400 / 计划 200」两个打架的数
   会被误读成"计划口径错了"。而清单侧**每 epoch 步数按总卡数折算**（真机只随 dp 变）本就是
   错的，正确口径要重生成数据件（须先呈报）⇒ 这一步**只做展示层**：⛔ 不动数据件、
   ⛔ 不动生成器公式、⛔ 不动判据（判据① 认的仍是 `epochs_completed ≥ epochs_planned`）。
2. **清掉备注正文里的内部记号**（`⛔` / `★` / `✅` / `❌` / `⚠`）：`verify_dated.sh` ⑥ 的泄漏扫描
   在**副本数据件/副本 HTML** 上命中 M011 备註里的 `⛔`（`⛔ 不作通过条件`）——那是内部写作记号，
   不该出现在给人看的矩阵里。

## 本文件钉住什么

* 备注模板产出**只有实跑步数 + 读数来源**，没有「计划 … 步」；本档**无计划数**时也不出现
  空括号 / 孤立分隔符 / 字面 `None`；
* 备注正文（含判据① 台账句、未判定项句）**过得了真扫描器** `tools/matrix/scan_matrix_leaks.py`；
* **状态列口径不含内部记号**（`meta.status_vocabulary` 只有纯中文三态）⇒ 状态列不受本裁定影响，
  ⛔ 本包没有动它；将来谁把记号塞进状态列，本文件的断言会红。
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tools" / "matrix"))
import make_result_json as m  # noqa: E402
import backfill_matrix as bf  # noqa: E402
import scan_matrix_leaks as leak  # noqa: E402

_MANIFEST = _ROOT / "tests" / "e2e" / "matrix68_manifest.json"
_DATA = _ROOT / "docs" / "capability-matrix.data.json"
#: 内部写作记号（泄漏扫描词表里那一族；本文件只断言这几枚不出现在**备注正文**）
_MARKERS = ("⛔", "★", "✅", "❌", "⚠")
#: 「计划 … 步」从句（D29 删除项）——只匹配"步"，⛔ 不误伤「计划 N 个 epoch」（那是判据① 官方口径）
_PLANNED_STEPS_RE = re.compile(r"计划[^；）]*步")


def _manifest_and_rows():
    if not _MANIFEST.is_file() or not _DATA.is_file():
        pytest.skip("仓内清单/数据件不在（只跑单元层）")
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    matrix = json.loads(_DATA.read_text(encoding="utf-8"))
    return manifest, {r["tier"]: r for r in matrix["rows"]}


def _run_dir(tmp: pathlib.Path, tier: str = "M018", *, steps: int | None = 400,
             exit_code: int = 0) -> pathlib.Path:
    """最小跑次目录：`readings.json`（含实跑步数与退出码）+ `gpu/peak_mem.json`。

    `steps=None` ⇒ readings 里不写 `steps_actual`，且日志里也没有步数读数（= 实跑未知）。
    """
    run = tmp / tier
    run.mkdir(parents=True)
    readings = {"tier": tier, "exit_code": exit_code, "elapsed_sec": 1234.0, "gpus": ["0"],
                "config": f"samples/configs/matrix68/{tier}.yaml", "image": "graspo:test",
                "timeout_sec": 1500, "command": f"run_tier_b09.sh {tier} 0 1500"}
    if steps is not None:
        readings["steps_actual"] = steps
    (run / "readings.json").write_text(json.dumps(readings, indent=1), encoding="utf-8")
    (run / "elapsed_sec").write_text("1234.0\n", encoding="utf-8")
    gpu = run / "gpu"
    gpu.mkdir()
    (gpu / "peak_mem.json").write_text(json.dumps({
        "caliber": "宿主侧 nvidia-smi memory.used 采样峰值（MiB）",
        "gpus": ["0"], "peak_used_mib_by_gpu": {"0": 1024.0},
        "peak_used_mib": 1024.0, "measured": True}, indent=1), encoding="utf-8")
    (run / "stdout.log").write_text("[train-worker] GPU lock OK\n", encoding="utf-8")
    return run


def _build(run: pathlib.Path, tier: str = "M018", *, manifest=None, rows=None):
    manifest, rows = (manifest, rows) if manifest is not None and rows is not None \
        else _manifest_and_rows()
    doc, diag = m.build_doc(tier=tier, run_dir=run, manifest=manifest, data_rows=rows,
                            eval_dir=None, tier_json=None, note_extra="")
    return doc, diag


def _compose_like_backfiller(doc, row) -> str:
    """按落盘时的**同一条路**拼备注：判据① 的 fails/unks 取自真判定函数，再交 `compose_note`。

    （不调 `judge_row`：四条判据里 ②③④ 需评测/签署证据，本 fixture 造不出来 ⇒ 它会 fail-closed
    返回 None；而备注列拼装逻辑本身与判据条数无关，故直接喂判据① 的真实结论串。）
    """
    status1, why1 = bf.judge_run_complete(doc, row)
    _s, why_unk = bf.judge_run_complete({"run_result": {"exit_code": 0}}, row)
    return bf.compose_note(doc["note"], "不通过",
                           [why1] if status1 == bf.FAIL else [],
                           [why_unk])


def _leak_hits(text: str, tmp: pathlib.Path) -> list:
    """用**真扫描器**（同一词表、同一实现）扫一段文本——不另写一份禁词表。"""
    p = tmp / "note_under_test.txt"
    p.write_text(text.replace("；", "；\n"), encoding="utf-8")
    return leak.scan([p], [])


# ── D29-① 备注里不再有「计划 N 步」（改前 = 「…；计划 200 步）」，改后 = 只有实跑 + 来源）──
def test_note_drops_planned_steps_clause(tmp_path):
    run = _run_dir(tmp_path)
    doc, _diag = _build(run)
    note = doc["note"]
    assert "实跑步数 400（读自跑次读数文件）" in note, note
    assert not _PLANNED_STEPS_RE.search(note), f"备注里还留着「计划 N 步」从句：{note}"
    # 判据① 的 epoch 口径文案**照旧保留**（D29 只动步数，⛔ 不动 judge 口径）
    assert "epoch" in note or "轮" in note, note
    # D29 补丁：实跑步数/轮数这两处短语，原因串自带括号时**不得**叠出「））」这种排版瑕疵
    # （证据缺口段里的嵌套括号是**合法嵌套**，不在此列）
    assert not re.search(r"实跑步数[^；]*[）)]）", note), note
    assert not re.search(r"实跑轮数未知[^；]*[）)]）", note), note


def test_judge_ledger_text_drops_planned_steps_and_marker():
    """判据① 的台账句会经 `compose_note` 进备注列 ⇒ 它自己也不得带「计划」或内部记号。"""
    row = {"derived_steps": "200"}
    ok = {"run_result": {"exit_code": 0, "epochs_completed": 4, "epochs_planned": 4,
                         "final_ckpt_loadable": True, "nan_or_inf_detected": False},
          "steps_actual": 400}
    status, why = bf.judge_run_complete(ok, row)
    assert status == bf.PASS, why
    assert "不作通过条件" in why, why
    assert "实跑 400 步" in why, why
    assert not _PLANNED_STEPS_RE.search(why), why
    assert not any(ch in why for ch in _MARKERS), why

    bad = json.loads(json.dumps(ok))
    bad["run_result"]["exit_code"] = 1
    status_b, why_b = bf.judge_run_complete(bad, row)
    assert status_b == bf.FAIL, why_b
    assert not any(ch in why_b for ch in _MARKERS), why_b


# ── D29 补丁：原因串自带括号 ⇒ 冒号引导，⛔ 不叠括号（两个分支各两例）──────────
@pytest.mark.parametrize("steps,why,expect", [
    (400, "读自跑次读数文件", "实跑步数 400（读自跑次读数文件）"),
    (0, "崩溃前未进入优化循环（stdout.log:20 的报错先于任何步数读数）",
     "实跑步数 0：崩溃前未进入优化循环（stdout.log:20 的报错先于任何步数读数）"),
    (None, "训练日志里没有步数读数（后端未打印/训练未进入优化循环）",
     "实跑步数未知：训练日志里没有步数读数（后端未打印/训练未进入优化循环）"),
    (None, "训练目录下读不到读数", "实跑步数未知（训练目录下读不到读数）"),
])
def test_steps_phrase_never_doubles_the_closing_paren(steps, why, expect):
    text = m.steps_phrase(steps, why)
    assert text == expect, text
    assert "））" not in text and "))" not in text, text
    assert text.count("（") == text.count("）"), f"括号不配对：{text}"


# ── D29-① 无计划数时：不得出现空括号 / 孤立分隔符 / 字面 None ────────────────
@pytest.mark.parametrize("how", ["null", "missing"])
def test_note_without_plan_numbers_has_no_empty_parens(tmp_path, how):
    manifest, rows = _manifest_and_rows()
    mf = json.loads(json.dumps(manifest))
    tier = next(t for t in mf["tiers"] if t["tier_id"] == "M018")
    if how == "null":
        tier["expected_optimizer_steps"] = None
    else:
        tier.pop("expected_optimizer_steps", None)
    run = _run_dir(tmp_path, steps=None)          # 实跑未知 ⇒ 走 else 分支（改前是「计划 N 步，实跑未知」）
    doc, _diag = _build(run, manifest=mf, rows=rows)
    note = doc["note"]
    assert re.search(r"实跑步数未知[（：]", note), note   # 括号或冒号引导（D29 补丁：原因自带括号用冒号）
    assert not _PLANNED_STEPS_RE.search(note), note
    for bad in ("None", "（）", "(）", "（；", "；）", "；；", "()"):
        assert bad not in note, f"备注里出现 {bad!r}：{note}"


# ── D29-② 备注正文（含判据① 台账句 / 未判定项句）过得了真扫描器 ──────────────
def test_composed_note_passes_the_leak_scanner(tmp_path):
    # 退出码 1 + 0 步 ⇒ 判据① 有据地不通过（与 M011 那次泄漏命中的备注同一形态）：
    # 「未过判据：判据① 未过：退出码 1 ≠ 0；…；台账：实跑 0 步（不作通过条件）」
    run = _run_dir(tmp_path, steps=0, exit_code=1)
    doc, _diag = _build(run)
    _manifest, rows = _manifest_and_rows()
    # 判据① 要的四个结构化事实（与 M011 真件同形：退出码 1、0/8 轮、检查点不可加载、无 NaN/Inf）
    doc = json.loads(json.dumps(doc))
    doc["run_result"].update({"exit_code": 1, "epochs_completed": 0, "epochs_planned": 8,
                              "final_ckpt_loadable": False, "nan_or_inf_detected": False})
    note = _compose_like_backfiller(doc, rows["M018"])
    assert "未过判据：判据① 未过：退出码 1 ≠ 0" in note, note   # 台账句确实拼进来了（不是空转）
    assert "台账：实跑 0 步（不作通过条件）" in note, note
    assert not any(ch in note for ch in _MARKERS), f"备注里有内部记号：{note}"
    assert _leak_hits(note, tmp_path) == [], _leak_hits(note, tmp_path)


def test_unknown_reason_text_passes_the_leak_scanner(tmp_path):
    """判据① 缺 `epochs_completed` 的那句（`…（Epoch 读数；不回落实跑步数比较）`）同样进备注。"""
    _status, why = bf.judge_run_complete({"run_result": {"exit_code": 0}, "steps_actual": 400},
                                        {"derived_steps": "200"})
    assert "epochs_completed" in why, why
    assert not any(ch in why for ch in _MARKERS), why
    assert _leak_hits(why, tmp_path) == [], _leak_hits(why, tmp_path)


# ── 状态列口径查清（机核）：三态纯中文，⛔ 不含内部记号 ⇒ 本裁定不动状态列 ────────
def test_status_column_carries_no_internal_marker():
    if not _DATA.is_file():
        pytest.skip("数据件不在")
    matrix = json.loads(_DATA.read_text(encoding="utf-8"))
    vocab = (matrix.get("meta") or {}).get("status_vocabulary") or []
    assert vocab, "数据件缺 meta.status_vocabulary（状态口径的唯一真相源）"
    assert not [v for v in vocab if any(ch in v for ch in _MARKERS)], vocab
    for row in matrix["rows"]:
        st = str(row.get("status") or "")
        assert st in vocab, (row["tier"], st)
        assert not any(ch in st for ch in _MARKERS), (row["tier"], st)
