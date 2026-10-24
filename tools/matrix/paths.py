#!/usr/bin/env python3
"""paths.py — 矩阵链「基准件 / 按轮副本」路径的**唯一真相源**（宪法 §1.4）。

★ 规则（2026-10-23 主席拍板起，取代「原版 = 带空格那份 + 一份带日期副本」的旧布局）：
  **基准件只有一份**（无日期 · 权限 444 · **结果永不写入**）：
      docs/GRASPO训练框架功能测试报告.html      ← 人读基准件（无空格）
      docs/capability-matrix.data.json          ← 机读基准数据件
  **按轮副本**（每轮开跑前由脚本**新建**，配额与填数只发生在副本）：
      docs/GRASPO训练框架功能测试报告-<YYYYMMDD>.html
      docs/capability-matrix-<YYYYMMDD>.data.json
  ⇒ 基准件只被读；副本由 `tools/matrix/new_round.sh`（或 `backfill_dated.sh` 的自动派生）产出。

★ 环境变量覆盖（**只有这两个**，⛔ 不增删）：
      GRASPO_BASELINE_HTML   基准人读件路径（缺省 = 仓内 docs/GRASPO训练框架功能测试报告.html）
      GRASPO_ROUND_DATE      轮次日期 YYYYMMDD（缺省 = 今天）

★ 命令行用法（bash 侧取同一份常量；⛔ 各脚本不再各写一份路径字面量）：
      eval "$(.venv/bin/python tools/matrix/paths.py --sh)"
      eval "$(.venv/bin/python tools/matrix/paths.py --sh --date 20261023)"
      .venv/bin/python tools/matrix/paths.py            # 人读一览
      .venv/bin/python tools/matrix/paths.py --set-human-view-path <data.json> --to <rel html>

★ 本模块**只**做路径与命名的解析，不读不写任何内容（唯一例外是显式调用
  `--set-human-view-path` 时改写一份**副本**数据件的 `meta.authority.human_view_path`）。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

#: 仓根（本文件在 `<repo>/tools/matrix/paths.py`）
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TOOLS = ROOT / "tools" / "matrix"

# ---- 命名（唯一字面量所在处）------------------------------------------------
#: 基准人读件文件名（**无空格**；444 只读；结果永不写入）
BASELINE_HTML_NAME = "GRASPO训练框架功能测试报告.html"
#: 基准机读数据件文件名
BASELINE_DATA_NAME = "capability-matrix.data.json"
#: 按轮副本命名模板（`{date}` = YYYYMMDD）
ROUND_HTML_TMPL = "GRASPO训练框架功能测试报告-{date}.html"
ROUND_DATA_TMPL = "capability-matrix-{date}.data.json"
#: 共用渲染模板
TEMPLATE_NAME = "template.html"

#: 仓根相对路径（供只写相对路径的模块复用，如 switch_corpus.py / 数据件 meta）
REL_BASELINE_HTML = f"docs/{BASELINE_HTML_NAME}"
REL_BASELINE_DATA = f"docs/{BASELINE_DATA_NAME}"
REL_ROUND_HTML_FMT = f"docs/{ROUND_HTML_TMPL}"
REL_ROUND_DATA_FMT = f"docs/{ROUND_DATA_TMPL}"
REL_TEMPLATE = f"tools/matrix/{TEMPLATE_NAME}"

#: 基准人读件的**冻结 sha256**（2026-10-24 第 3 章 OPD 行改「本地教师 / 27B 外置 vLLM 服务」口径后的定稿内容；2026-10-25 删「框架版本」行的提交号后更新）。
#: 机核：`verify_dated.sh` ①、`new_round.sh` ①。任何再次改写基准件 ⇒ 立刻对不上 ⇒ 红。
BASELINE_HTML_SHA256 = "b8ed080dd80edd04dee735af59788ebaa6e297107509f363c520e38e4b986f30"
#: 基准件权限（物理只读）
FROZEN_MODE = "444"
#: 副本权限（可写；由脚本回填）
ROUND_MODE = "644"

#: 轮次日期格式（外参一律按此校验，⛔ 不猜）
DATE_RE = re.compile(r"^\d{8}$")

ENV_BASELINE_HTML = "GRASPO_BASELINE_HTML"
ENV_ROUND_DATE = "GRASPO_ROUND_DATE"


def round_date(explicit: str | None = None) -> str:
    """轮次日期 YYYYMMDD：显式入参 > `$GRASPO_ROUND_DATE` > 今天。"""
    val = explicit if explicit is not None else os.environ.get(ENV_ROUND_DATE) or None
    if val is None:
        return datetime.now().strftime("%Y%m%d")
    val = val.strip()
    if not DATE_RE.match(val):
        raise SystemExit(f"paths.py：日期须为 YYYYMMDD，实际 {val!r}（来自 --date / ${ENV_ROUND_DATE}）")
    return val


def baseline_html() -> Path:
    """基准人读件路径（可被 `$GRASPO_BASELINE_HTML` 覆盖）。"""
    env = os.environ.get(ENV_BASELINE_HTML)
    return Path(env) if env else DOCS / BASELINE_HTML_NAME


def baseline_data() -> Path:
    """基准机读数据件路径（与基准人读件同目录；⛔ 不可覆盖）。"""
    return baseline_html().parent / BASELINE_DATA_NAME


def round_html(date: str | None = None) -> Path:
    return DOCS / ROUND_HTML_TMPL.format(date=round_date(date))


def round_data(date: str | None = None) -> Path:
    return DOCS / ROUND_DATA_TMPL.format(date=round_date(date))


def rel_of(path: Path) -> str:
    """仓根相对路径（不在仓内则原样返回绝对路径字符串）。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def rel_round_html(date: str | None = None) -> str:
    return rel_of(round_html(date))


def rel_round_data(date: str | None = None) -> str:
    return rel_of(round_data(date))


#: 模块级便捷常量（导入期求值；带 `--date` 的轮次路径请用 `round_html()` / `round_data()`）
BASELINE_HTML = baseline_html()
BASELINE_DATA = baseline_data()


# ---- 副本数据件的 human_view_path（副本 ⇒ 指向副本；基准 ⇒ 指向基准）------------
def set_human_view_path(data_path: Path | str, rel_human_view: str) -> tuple[str, str]:
    """把数据件 `meta.authority.human_view_path` 改成 `rel_human_view`（**行级**文本替换）。

    ★ 为什么行级替换而不是 `json.dump`：数据件是生成器的确定性产物，只要**那一行**变，
      其余字节（缩进/键序/中文/浮点写法）必须原样不动 —— 否则「副本与基准相异行数」
      这条机核（`verify_dated.sh` ④）就失去意义。
    返回 `(旧值, 新值)`；已是新值 ⇒ 直接返回且不写盘（幂等）。
    """
    import re as _re

    p = Path(data_path)
    raw = p.read_text(encoding="utf-8")
    doc = json.loads(raw)
    authority = (doc.get("meta") or {}).get("authority") or {}
    old = authority.get("human_view_path")
    if old == rel_human_view:
        return (old, rel_human_view)
    pat = _re.compile(r'("human_view_path"\s*:\s*)"(?:[^"\\]|\\.)*"')
    hits = len(pat.findall(raw))
    if hits != 1:
        raise SystemExit(
            f"paths.py：{p} 里 \"human_view_path\" 出现 {hits} 次（期望恰好 1 次，⛔ 不猜）")
    new_raw = pat.sub(lambda m: m.group(1) + json.dumps(rel_human_view, ensure_ascii=False), raw, count=1)
    p.write_text(new_raw, encoding="utf-8")
    return (old or "", rel_human_view)


# ---- 导出面 ------------------------------------------------------------------
def as_dict(date: str | None = None) -> dict[str, str]:
    d = round_date(date)
    return {
        "PATHS_REPO": str(ROOT),
        "PATHS_DOCS": str(DOCS),
        "PATHS_TOOLS": str(TOOLS),
        "PATHS_TEMPLATE": str(TOOLS / TEMPLATE_NAME),
        "PATHS_BASELINE_HTML": str(baseline_html()),
        "PATHS_BASELINE_HTML_NAME": BASELINE_HTML_NAME,
        "PATHS_BASELINE_DATA": str(baseline_data()),
        "PATHS_BASELINE_DATA_NAME": BASELINE_DATA_NAME,
        "PATHS_BASELINE_HTML_SHA256": BASELINE_HTML_SHA256,
        "PATHS_FROZEN_MODE": FROZEN_MODE,
        "PATHS_ROUND_MODE": ROUND_MODE,
        "PATHS_ROUND_DATE": d,
        "PATHS_ROUND_HTML": str(round_html(d)),
        "PATHS_ROUND_HTML_NAME": ROUND_HTML_TMPL.format(date=d),
        "PATHS_ROUND_DATA": str(round_data(d)),
        "PATHS_ROUND_DATA_NAME": ROUND_DATA_TMPL.format(date=d),
        "PATHS_REL_BASELINE_HTML": REL_BASELINE_HTML,
        "PATHS_REL_BASELINE_DATA": REL_BASELINE_DATA,
        "PATHS_REL_ROUND_HTML": rel_round_html(d),
        "PATHS_REL_ROUND_DATA": rel_round_data(d),
    }


USAGE = """paths.py — 矩阵链路径唯一真相源

用法（★ 各脚本 ⛔ 不再各写一份路径字面量，一律从这里取）：

  python3 tools/matrix/paths.py                         人读一览（当前轮日期）
  python3 tools/matrix/paths.py --date YYYYMMDD         指定轮次日期
  python3 tools/matrix/paths.py --sh [--date YYYYMMDD]  导出 shell 赋值（bash: eval "$(...)"）
  python3 tools/matrix/paths.py --set-human-view-path <data.json> --to <rel html>
                                                        改写**副本**数据件的 human_view_path

★ 环境变量覆盖：GRASPO_BASELINE_HTML（基准人读件路径）、GRASPO_ROUND_DATE（轮次日期）
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    date: str | None = None
    mode = "human"
    target: str | None = None
    rel_hv: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(USAGE, end="")
            return 0
        if a == "--sh":
            mode = "sh"; i += 1; continue
        if a == "--date":
            if i + 1 >= len(argv):
                raise SystemExit("paths.py：--date 缺参数值")
            date = argv[i + 1]; i += 2; continue
        if a == "--set-human-view-path":
            if i + 1 >= len(argv):
                raise SystemExit("paths.py：--set-human-view-path 缺参数值")
            mode = "hv"; target = argv[i + 1]; i += 2; continue
        if a == "--to":
            if i + 1 >= len(argv):
                raise SystemExit("paths.py：--to 缺参数值")
            rel_hv = argv[i + 1]; i += 2; continue
        raise SystemExit(f"paths.py：未识别参数 {a}（见 --help）")

    if mode == "sh":
        for k, v in as_dict(date).items():
            print(f"{k}={shlex.quote(v)}")
        return 0

    if mode == "hv":
        if not target or not rel_hv:
            raise SystemExit("paths.py：--set-human-view-path 需同时给 --to <rel html>")
        old, new = set_human_view_path(target, rel_hv)
        print(f"human_view_path: {old or '（缺失）'} → {new}")
        return 0

    d = round_date(date)
    print("# 矩阵链路径（唯一真相源 tools/matrix/paths.py）")
    print(f"仓根            {ROOT}")
    print(f"轮次日期        {d}")
    print(f"模板            {TOOLS / TEMPLATE_NAME}")
    print(f"基准人读件      {baseline_html()}   [{FROZEN_MODE} · 结果永不写入]")
    print(f"基准数据件      {baseline_data()}   [{FROZEN_MODE}]")
    print(f"基准人读件 sha  {BASELINE_HTML_SHA256}")
    print(f"本轮人读副本    {round_html(d)}   [{ROUND_MODE} · 不存在则由 new_round.sh 新建]")
    print(f"本轮数据副本    {round_data(d)}   [{ROUND_MODE}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
