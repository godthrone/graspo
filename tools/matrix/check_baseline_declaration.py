#!/usr/bin/env python3
"""②b 机核：**冻结原件 ↔ 模板基线** 的差异必须**恰为**「边界声明」那一块。

判据（⛔ 全部为硬失败，rc≠0，不许降级为警告）
  1. 两个文件按行比对，**必须恰好 1 个 hunk**、且该 hunk **只有新增、删除 0 行**；
  2. 新增行数 == 模板里「边界声明」块的行数，且**逐行逐字符相同**；
  3. 新增块的位置必须在第 1 章 `</section>` 之前（即仍在导语之后、矩阵表之前）。
失败时打印**逐行差异**（unified diff），供人直接看清多/少了哪一行。

唯一真相源：期望块**从 `tools/matrix/template.html` 现抽**，⛔ 不在本脚本里复制一份文案
（改文案必须改模板；本脚本只负责"冻结件与模板基线之差 == 模板里那块"）。
"""
from __future__ import annotations
import argparse
import difflib
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
import paths as _paths  # noqa: E402

#: 模板里边界声明块的起始标记（注释里的固定片段；改标记等于改结构 ⇒ 本脚本会红，必须一起改）
MARK_START = "边界声明（宪法 §17.7"
MARK_HEADING = "<h3>边界声明</h3>"


def _lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def expected_block(template_lines: list[str]) -> list[str]:
    """从模板现抽「边界声明」块：从起始注释**前一空行**到块尾 `</ul>`。"""
    try:
        i = next(n for n, ln in enumerate(template_lines) if MARK_START in ln)
    except StopIteration:                                    # 模板里没有这块 ⇒ 直接失败
        raise SystemExit(f"✗ 模板里找不到边界声明块（缺标记 {MARK_START!r}）⇒ rc=1")
    start = i - 1 if i > 0 and template_lines[i - 1].strip() == "" else i
    j = next((n for n in range(i, len(template_lines)) if template_lines[n].strip() == "</ul>"), None)
    if j is None:
        raise SystemExit("✗ 模板里边界声明块没有收尾 `</ul>` ⇒ rc=1")
    return template_lines[start:j + 1]


def check(frozen: Path, baseline: Path, template: Path) -> tuple[bool, list[str], str]:
    """返回 (是否通过, 失败原因列表, 逐行差异文本)。"""
    fb, tb, tt = _lines(frozen), _lines(baseline), _lines(template)
    exp = expected_block(tt)
    why: list[str] = []

    # ★ 用「最长公共前缀 / 后缀」而不是 SequenceMatcher：两件里有重复行（`</ul>` 等），
    #   SequenceMatcher 可能把公共行对齐到错误位置，导致"新增块"错位。前缀+后缀法对
    #   "只在中间插入一段"这种形态是**精确**的：前缀、后缀之外若还有任何残留 ⇒ 那不是纯插入。
    n = min(len(fb), len(tb))
    pre = 0
    while pre < n and fb[pre] == tb[pre]:
        pre += 1
    suf = 0
    while suf < n - pre and fb[len(fb) - 1 - suf] == tb[len(tb) - 1 - suf]:
        suf += 1
    inserted = tb[pre:len(tb) - suf]
    leftover_base = fb[pre:len(fb) - suf]
    if leftover_base:
        why.append(f"冻结原件有 {len(leftover_base)} 行在模板基线里找不到（期望删除 0 行）")
    if len(inserted) != len(exp):
        why.append(f"新增行数 = {len(inserted)}（期望 {len(exp)}）")
    for k, (got, want) in enumerate(zip(inserted, exp), 1):
        if got != want:
            why.append(f"新增块第 {k} 行与模板不一致：在盘 {got!r} / 模板 {want!r}")
            break
    if not why:
        # 位置：插入点必须紧跟第 1 章导语列表的 `</ul>`，且到第 1 章 `</section>` 之间不得出现矩阵表
        if pre == 0 or fb[pre - 1].strip() != "</ul>":
            why.append(f"插入点不在第 1 章导语列表之后（其前一行 = {fb[pre - 1] if pre else '（文件头）'!r}）")
        else:
            close = next((k for k in range(pre, len(fb)) if "</section>" in fb[k]), None)
            if close is None:
                why.append("插入点之后找不到第 1 章 `</section>`（结构异常）")
            elif any('<table class="matrix"' in fb[k] for k in range(pre, close)):
                why.append("插入点在矩阵表与 `</section>` 之间（期望在导语之后、矩阵表之前）")
    if not why and not inserted:
        why.append("两件完全相同（期望差异恰为边界声明块）")

    diff_text = "".join(difflib.unified_diff(
        [f"{ln}\n" for ln in fb], [f"{ln}\n" for ln in tb],
        fromfile=str(frozen), tofile=str(baseline), n=2)) or "（无差异）\n"
    return (not why), why, diff_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="②b 机核：冻结原件 ↔ 模板基线 的差异必须恰为边界声明块")
    ap.add_argument("--frozen", default=None, help="冻结原件（缺省 = paths.baseline_html()）")
    ap.add_argument("--baseline", default=None, help="模板基线（缺省 = paths.template_baseline_html()）")
    ap.add_argument("--template", default=None, help="渲染模板（缺省 = tools/matrix/template.html）")
    a = ap.parse_args(argv)
    frozen = Path(a.frozen) if a.frozen else _paths.baseline_html()
    baseline = Path(a.baseline) if a.baseline else _paths.template_baseline_html()
    template = Path(a.template) if a.template else TOOL_DIR / _paths.TEMPLATE_NAME
    for p in (frozen, baseline, template):
        if not p.is_file():
            print(f"✗ 文件不存在：{p} ⇒ rc=1", file=sys.stderr)
            return 1

    ok, why, diff_text = check(frozen, baseline, template)
    print("# check_baseline_declaration.py — ②b：冻结原件 ↔ 模板基线 的差异必须恰为「边界声明」块")
    print(f"冻结原件    {frozen}")
    print(f"模板基线    {baseline}")
    print(f"模板        {template}（期望块从这里现抽，共 {len(expected_block(_lines(template)))} 行）")
    if ok:
        print("\n# 结论：PASS（rc=0）—— 两件之差恰为边界声明块（1 个 hunk / 删除 0 行 / 其余差异 0）")
        return 0
    print("\n✗ FAIL：冻结原件与模板基线的差异不是「恰为边界声明块」")
    for n, w in enumerate(why, 1):
        print(f"  {n}. {w}")
    print("\n逐行差异（unified diff，n=2）：")
    for ln in diff_text.rstrip("\n").splitlines():
        print("  " + ln)
    print("\n# 结论：FAIL（rc=1）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
