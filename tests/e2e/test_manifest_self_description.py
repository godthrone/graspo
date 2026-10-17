"""「自述与现盘事实一致」的能失败机检 —— 2026-10-17 由 `task-doc-sweep`（批 12）新增。

**为什么必须有这个文件**：文档/注释里的「例值」与「口径字样」是**文本**，没有任何判据在守
⇒ 一旦现盘事实前进，文本会静默变旧，后人据此误判（本批的由来就是一次这样的误判）。
本文件把两条固化成**能失败**的断言：

* **S1**：`tools/matrix/extract_spec.py` 的 **C5 ruling 例值**（`T028`/`T041`/`T042` 的 `gpus`）
  必须与**现盘** `tests/e2e/matrix54_manifest_source.json` 的逐档 `tiers[*].gpus` **逐值相等**。
  *阳性对照*：把例值改回旧值（如 `T041=[2,3]`）⇒ **S1 必红**。
* **S2**：`tests/e2e/generate_matrix_manifest_fsdp.py` 不得再出现**已作废**的「连续 3 次」口径字样
  （现行口径＝**一次有效训练**；作废来源 `.local/matrix-doc-decision-record-20261011.md:45`）。
  *阳性对照*：把旧口径字样塞回 ⇒ **S2 必红**。

★ 两条都是**纯读**：不写任何文件、不需要 GPU、不 import 项目代码。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_SOURCE = ROOT / "tests" / "e2e" / "matrix54_manifest_source.json"
EXTRACT_SPEC = ROOT / "tools" / "matrix" / "extract_spec.py"
FSDP_GENERATOR = ROOT / "tests" / "e2e" / "generate_matrix_manifest_fsdp.py"

#: C5 ruling 里以「例值」形式登记的档位。★ 将来若在 ruling 里增删例值，一并改这里。
C5_EXAMPLE_TIERS: tuple[str, ...] = ("T028", "T041", "T042")

#: 已作废口径字样（出现即红）。★ 只列**作废**的，不列现行口径。
RETIRED_CALIBER_PHRASES: tuple[str, ...] = ("连续 3 次", "连续3次", "三次全成功")


def _manifest_gpus() -> dict[str, list[int]]:
    doc = json.loads(MANIFEST_SOURCE.read_text(encoding="utf-8"))
    return {t["tier_id"]: sorted(int(g) for g in t["gpus"]) for t in doc["tiers"]}


def _c5_ruling_line() -> str:
    """取 C5 ruling 那一行（含「gpus 唯一定档」）。找不到 ⇒ 直接红（防「检查恒绿」）。"""
    for line in EXTRACT_SPEC.read_text(encoding="utf-8").splitlines():
        if "gpus 唯一定档" in line:
            return line
    raise AssertionError(
        "tools/matrix/extract_spec.py 中找不到 C5「gpus 唯一定档」ruling 行 "
        "⇒ 本检查已失效，必须随 extract_spec.py 同步更新（⛔ 不得让它静默变绿）"
    )


def _c5_example_values() -> dict[str, list[int]]:
    line = _c5_ruling_line()
    out: dict[str, list[int]] = {}
    for tier in C5_EXAMPLE_TIERS:
        m = re.search(rf"\b{tier}=\[([0-9,\s]+)\]", line)
        assert m is not None, (
            f"C5 ruling 行里找不到 {tier}=[...] 例值（该断言本身能失败）；"
            f"ruling 行 = {line!r}"
        )
        out[tier] = sorted(int(x) for x in m.group(1).split(",") if x.strip())
    return out


def test_c5_example_values_match_on_disk_manifest() -> None:
    """S1：`extract_spec.py` C5 例值 == 现盘清单 `gpus`（阳性对照：改回 `T041=[2,3]` ⇒ 红）。"""
    manifest = _manifest_gpus()
    example = _c5_example_values()
    assert example, "C5 例值集合为空 ⇒ 检查失效"
    for tier, got in example.items():
        want = manifest[tier]
        assert got == want, (
            f"{tier}：extract_spec.py C5 例值 {got} != 现盘清单 {want}。"
            f"⇒ 自述与事实漂移；请更新 tools/matrix/extract_spec.py 的 C5 ruling 例值"
            f"（源：{MANIFEST_SOURCE.relative_to(ROOT)}）"
        )


def test_fsdp_generator_has_no_retired_caliber_phrase() -> None:
    """S2：fsdp 生成器不得含已作废口径字样（阳性对照：塞回「连续 3 次」⇒ 红）。"""
    lines = FSDP_GENERATOR.read_text(encoding="utf-8").splitlines()
    hits = [
        (i, phrase, lines[i - 1].strip())
        for i, line in enumerate(lines, start=1)
        for phrase in RETIRED_CALIBER_PHRASES
        if phrase in line
    ]
    assert not hits, (
        "tests/e2e/generate_matrix_manifest_fsdp.py 出现已作废口径字样"
        f"（现行＝一次有效训练）：{hits}"
    )
