#!/usr/bin/env python3
"""★ 生成里程碑记录的「冻结件」节（★ 机读、值一律来自实物，⛔ 不得手工编辑）。

用法：
    python3 tools/matrix/freeze_table.py            # 打印该节文本
    python3 tools/matrix/freeze_table.py --inject <记录.md>   # 替换记录里的该节
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# ★ 2026-10-23（基准件唯一 + 按轮副本）：人读件路径的唯一真相源 = 同目录 `paths.py`（宪法 §1.4），
#   ⛔ 本文件不再自写「基准件文件名」字面量。病史：主席把基准件改名去空格后，本处仍指向带空格的旧名
#   ⇒ 无参运行 `build()` 现读实物时直接 FileNotFoundError（`--help` 冒烟抓不到，只有真跑才炸）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _paths  # noqa: E402

MARK = "## 冻结件"
RULE = ("★ **本节由 `tools/matrix/freeze_table.py` 生成，⛔ 不得手工编辑**"
        "（★ 值一律来自实物：`sha256sum` ＋ `wc -c`）。")
ITEMS = (
    ("数据", _paths.REL_BASELINE_DATA),
    ("矩阵 HTML", _paths.REL_BASELINE_HTML),
    ("里程碑快照", "docs/capability-matrix-milestone-20261014-v2.html"),
)


def build(root: Path) -> str:
    rows = []
    for name, rel in ITEMS:
        p = root / rel
        b = p.read_bytes()
        rows.append((name, rel, len(b), hashlib.sha256(b).hexdigest()))
    snap = rows[2][3]
    out = [MARK + "（★ **sha 唯一权威处**；★ 修订时只改这一节，且**由工具生成**）", "",
           RULE, "", "| 件 | 路径 | 字节 | sha256 |", "|---|---|---:|---|"]
    for name, rel, size, sha in rows:
        extra = "（★ 与矩阵 HTML 逐字节一致）" if name == "里程碑快照" else ""
        out.append(f"| {name} | `{rel}` | {size:,} | `{sha}`{extra} |")
    out += [f"| 本记录自身 | `.local/capability-baseline-milestone-20261014-v2.md` | ★ 见交付回执 | ★ **不写入文件**（避免自指） |",
            "", "★ **引用规则**：★ 本记录里 sha 的**唯一存值处 = 本节**；★ **其他任何节只引用它**（写\"见「冻结件」节\"，⛔ 不复制值）；★ 历史值必须标注\"历史值、已被取代\"。", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--inject", default="")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    text = build(root)
    if a.inject:
        p = Path(a.inject)
        s = p.read_text(encoding="utf-8")
        m = re.search(r"^## 冻结件.*?(?=^## |\Z)", s, re.S | re.M)
        if not m:
            print("FAIL：记录里找不到「冻结件」节")
            return 1
        p.write_text(s[:m.start()] + text + s[m.end():], encoding="utf-8")
        print(f"OK：已注入「冻结件」节 ⇒ {a.inject}")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
