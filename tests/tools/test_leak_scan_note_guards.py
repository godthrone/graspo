"""E3 单测：备注列的「工程日志形态」护栏（`scan_matrix_leaks.scan_notes` / 长度告警）。

病史：M011 备注曾 1324 字，正文含 `stdout.log:20`、`/workspace/…/lora.py:170`、正则原文
`^(?:model\\.)?…` 与英文帮助长句 ⇒ 用户看到的是日志而不是结论（2026-10-27 主席裁定改口径）。

护栏两层：
* **失败级**（`scan_notes`）：坐标 / 绝对路径 / `file:line` / 正则原文 一律 LEAK-SCAN FAIL；
* **告警级**（`note_length_warnings`）：备注 > 400 字提醒（⛔ 不判失败，人读上限 200 在 note_style）。

⚠️ 只扫 `rows[*].note`：`spec.lora_targets` 里的正则**是合法规格值**，⛔ 不许误伤。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "matrix"))
sys.path.insert(0, str(ROOT / "scripts"))
import note_style as ns  # noqa: E402
import scan_matrix_leaks as sm  # noqa: E402

#: 旧口径（E1 之前**真实落盘**的 M011 备注，取自 `docs/capability-matrix-20261025.data.json`）
LEGACY_M011_NOTE = (
    "训练崩溃，未产生有效训练：exit_code=1，ValueError: lora.target_modules is required "
    "(a single real HF module-name regex; the regex form is the default because the cost is "
    "only for the modules that the weights come from)（stdout.log:20）；退出码 1；实跑步数 0"
    "（崩溃前未进入优化循环（stdout.log:20 的报错先于任何步数读数；无 checkpoint / 无 "
    "trainer_state 的 global_step 读数）；计划 400 步）；完成轮数 0/4 个 epoch（训练崩溃于第 0 步）；"
    "检查点：无任何 checkpoint-*/ 或 final/ 目录（崩溃发生在产出检查点之前）⇒ 不存在可加载的最终检查点；"
    "同一条 traceback 的源码帧 = /workspace/graspo/src/graspo/core/lora.py:170"
)


def _write(tmp_path: Path, note: str, *, spec: str | None = None) -> Path:
    doc = {"rows": [{"tier": "M011", "note": note, "spec": {"lora_targets": spec}}]}
    p = tmp_path / "data.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


# ── 失败级：每类形态各一例 ────────────────────────────────────────────────────
@pytest.mark.parametrize("note,why", [
    ("退出码 1；报错见 stdout.log:20", "日志坐标（对人不表达的定位符）"),
    ("退出码 1；源码帧 = src/graspo/core/lora.py:170", "源码 file:line 坐标（对人不表达的定位符）"),
    ("退出码 1；权重在 /workspace/graspo/out", "容器内绝对路径"),
    ("退出码 1；数据在 /data/remote", "宿主/容器绝对路径"),
    ("退出码 1；目标匹配 ^(?:model\\.)?language_model\\..*$", "正则原文（非人读）"),
])
def test_each_engineering_log_shape_is_a_leak(tmp_path, note, why):
    hits = sm.scan_notes([_write(tmp_path, note)])
    assert hits, f"{why} 没被拦下：{note}"
    assert any(why in h[2] for h in hits), hits


# ── 告警级：长度超限只提醒 ────────────────────────────────────────────────────
def test_long_note_is_a_warning_not_a_failure(tmp_path):
    p = _write(tmp_path, "退出码 0；" + "很长的结论" * 100)
    assert sm.scan_notes([p]) == [], "长度⛔ 不该进失败级（那是人读质量线，不是泄漏）"
    warns = sm.note_length_warnings([p])
    assert warns and "备注过长" in warns[0][2], warns


def test_clean_note_has_neither_hits_nor_warnings(tmp_path):
    p = _write(tmp_path, "退出码 0；实跑步数 400；判据结论：①②③④ 过")
    assert sm.scan_notes([p]) == []
    assert sm.note_length_warnings([p]) == []


# ── 不误伤规格面里的合法正则 ──────────────────────────────────────────────────
def test_spec_regex_value_is_not_a_leak(tmp_path):
    p = _write(tmp_path, "退出码 0；判据结论：①②③④ 过",
               spec="^(?:model\\.)?language_model\\..*\\.(q_proj|k_proj)$")
    assert sm.scan_notes([p]) == [], "规格面的正则值是合法内容，⛔ 不许误伤"


# ── 红 → 绿：真实旧备注 FAIL，规范化后 PASS ───────────────────────────────────
def test_real_legacy_note_fails_and_normalised_note_passes(tmp_path):
    red = _write(tmp_path, LEGACY_M011_NOTE)
    assert sm.scan_notes([red]), "旧口径备注（坐标/路径/正则原文）必须 FAIL"
    assert sm.note_length_warnings([red]), "旧口径 1324 字必须触发长度告警"
    green = _write(tmp_path, ns.normalize_legacy(LEGACY_M011_NOTE))
    assert sm.scan_notes([green]) == [], sm.scan_notes([green])
    assert sm.note_length_warnings([green]) == []


def test_note_style_has_noise_matches_the_scanner_wordlist():
    """清洗器与扫描器必须同义（⛔ 别一边删、一边漏）。"""
    assert ns.has_noise(LEGACY_M011_NOTE)
    assert not ns.has_noise(ns.normalize_legacy(LEGACY_M011_NOTE))
