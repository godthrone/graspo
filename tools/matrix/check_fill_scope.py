#!/usr/bin/env python3
"""check_fill_scope.py — **填表写范围硬闸门**（主席令②，2026-10-25；机制见 docs/run-discipline.md §22）。

★★ 主席原话（本文件的唯一判据来源）：
    「填写报告的脚本只能填写第四章表格里面的实验内容数据，不能修改其它任何地方。」

## 判据（逐条机核，任何一条不满足 ⇒ rc=2 并列违规清单）

输入 = **原版 HTML** 与 **填表后 HTML**（两份路径）。两份做**纯结构 diff**，每一处差异必须同时满足：

  ① 差异只出现在 `<table class="matrix">` 的 `<tbody>` 内（⛔ 第 1/2/3 章文字、`<colgroup>`、
     表头 `<thead>`、脚本、注释、样式一律零容忍）；
  ② 所在行必须是矩阵档位行（档号取自该行首个 `data-col="tier"` 单元格，须匹配 `^M\\d{3}$`）；
  ③ 所在单元格 `data-col` ∈ **实验结果列白名单** = 数据件 `data_keys`（⛔ 本文件不写死列名，
     一律从 `<data.json>` 现读；本轮 = `peak_mem` / `steps_actual` / `elapsed` / `throughput` /
     `effect` / `note`）+ **机算结论列** `status`（见下「status 例外」）；
  ④ 差异只是**单元格文本内容**变化：标签结构 / 属性 / 样式 / 脚本 / 注释 / 章节文字零变化
     —— 单元格内出现任何 `<…>` 标记即违规（"塞 `<script>`"当场红）。

### status 例外（唯一一处，且已收窄到最小面）

`status`（是否通过）是**由数据算出的结论列**，`render_matrix.py` 同时写它的**文本**与**配色类名**
（`st-pending` / `st-pass` / `st-fail` / `st-illegal` / `st-dash`）——两者不可分割（只改文本会让
绿/黄/红说假话）。故对 `status` 单元格只放开两件事：
  · 文本 ∈ 数据件 `meta.status_vocabulary`（⛔ 不得写词表外的散文）；
  · 该 `<td>` 的 **`st-*` 配色类名**可随之替换（其它 class / 属性 / 位置一字不许动）。
⇒ 若主席要求连 status 也不许动，需先改渲染器（让状态不落 HTML）——那是另一个工作包。

## 用法

    python3 tools/matrix/check_fill_scope.py --base <原版.html> --filled <填表后.html>
    python3 tools/matrix/check_fill_scope.py --base <原版> --filled <副本> --data <数据件>
    python3 tools/matrix/check_fill_scope.py --help

退出码：0 = 所有差异都在允许面内（含"零差异"）；2 = 有违规（逐条打印 `文件:行:列` + 文本片段）；
        1 = 参数/读文件/结构不可判等**用法级**错误（⛔ 与"违规"分开，便于调用方区分）。

★ 回填链已接入（`backfill_matrix.py` 落盘后自动调用本闸门，不过则**回滚本次写入**）；
  `verify_dated.sh` 亦在有副本态时调用（见 §22）。⛔ 本文件只读，不写任何东西。
"""
from __future__ import annotations

import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
import paths as _paths  # noqa: E402  —— 路径一律取自唯一真相源（§1.4）

#: 机算结论列（唯一例外；见模块 docstring「status 例外」）
MACHINE_STATUS_COL = "status"
#: `render_matrix.STATUS_CLASS` / `STATUS_CLASS_LEGACY` 的类名词表（配色随文本变，⛔ 不放开其它属性）
STATUS_CLASS_TOKENS = frozenset({"st-pass", "st-fail", "st-pending", "st-illegal", "st-dash"})
#: 违规清单最多打印这么多条（尾部结构错位会放大成上千条；rc 不受影响）
MAX_PRINT = 20

#: 矩阵表 / 档位 / 单元格的形态（⛔ 不用宽松正则猜整页结构）
TABLE_OPEN_RE = re.compile(r'<table class="matrix">')
CELL_RE = re.compile(r"<td\b([^>]*)>(.*?)</td>", re.S)
TR_OR_CELL_RE = re.compile(r"<tr\b[^>]*>|<td\b([^>]*)>(.*?)</td>", re.S)
ATTR_RE = r'(?:\b%s\s*=\s*"([^"]*)")'
TIER_RE = re.compile(r"^M\d{3}$")
TAG_RE = re.compile(r"<[^>]*>")
ST_TOKEN_RE = re.compile(r"\bst-(?:pass|fail|pending|illegal|dash)\b")

USAGE = """check_fill_scope.py — 填表写范围硬闸门（§22 · 主席令「只能填第四章实验数据」）

用法：
  python3 tools/matrix/check_fill_scope.py --base <原版.html> --filled <填表后.html> [--data <数据件>]
  python3 tools/matrix/check_fill_scope.py --help

  --base     原版（基准件）HTML
  --filled   填表后（按轮副本）HTML
  --data     数据件（实验结果列白名单 = 其 data_keys；缺省 = 基准数据件 docs/capability-matrix.data.json）

判定：两份纯结构 diff；任一处差异不在「第 4 章矩阵表 tbody 内 · M 档行 · 实验结果列 · 纯文本」面内
      ⇒ rc=2 并逐条打印 `文件:行:列` + 文本片段。允许列一律从数据件现读（⛔ 不写死列名）。

退出码：0 = 通过（含零差异）／2 = 有违规／1 = 用法级错误（参数、读文件、结构不可判）
★ 只读：本脚本不写任何文件。
"""


# ---- 数据件：白名单与词表（⛔ 不猜列名）---------------------------------------
def load_result_columns(data_path: Path | str) -> list[str]:
    """实验结果列白名单 = 数据件 `data_keys`（渲染链与回填链共用的同一份列清单）。"""
    doc = json.loads(Path(data_path).read_text(encoding="utf-8"))
    keys = doc.get("data_keys")
    if not isinstance(keys, list) or not keys:
        raise _UsageError(f"数据件 {data_path} 缺 data_keys 或为空 ⇒ 无法确定结果列白名单（fail-closed）")
    cols = [str(k) for k in keys]
    bad = [c for c in cols if not re.fullmatch(r"[a-z0-9_]+", c)]
    if bad:
        raise _UsageError(f"数据件 {data_path} 的 data_keys 含非法列名 {bad}（fail-closed）")
    return cols


def load_status_vocabulary(data_path: Path | str) -> frozenset[str]:
    """状态词表（`meta.status_vocabulary`）；缺失 ⇒ 空集（⇒ status 文本变化一律判违规）。"""
    doc = json.loads(Path(data_path).read_text(encoding="utf-8"))
    vocab = ((doc.get("meta") or {}).get("status_vocabulary")) or []
    return frozenset(str(v) for v in vocab if str(v).strip())


class _UsageError(Exception):
    """用法级错误（rc=1）：⛔ 与"违规"（rc=2）分开。"""


# ---- 结构解析 ----------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    attrs: str      # `<td` 与 `>` 之间的原始属性串（逐字节，含空白）
    inner: str      # 单元格内部 HTML（原样）
    start: int      # 文件内偏移（`<td` 的 `<` 位置）
    end: int        # 文件内偏移（`</td>` 之后）
    tier: str | None
    col: str | None


@dataclass(frozen=True)
class Doc:
    text: str
    cells: tuple[Cell, ...]
    #: 非单元格文本段：segs[i] = 第 i 个单元格**之前**的文本；segs[-1] = 最后一个单元格之后的文本
    #: （只覆盖 tbody 内、单元格之间的标记；章节正文与表头在 prefix 里整体比）
    segs: tuple[str, ...]
    seg_offsets: tuple[int, ...]
    #: 整个文档在「首个 tbody 单元格之前」/「末个 tbody 单元格之后」的原文
    #: —— 第 1/2/3 章正文、样式、脚本、注释、`<colgroup>`、`<thead>` 全在此面内，逐字节比
    prefix: str
    suffix: str
    header_cols: tuple[str, ...]


def _attr(attrs: str, name: str) -> str | None:
    m = re.search(ATTR_RE % re.escape(name), attrs)
    return m.group(1) if m else None


def _text_of(inner: str) -> str:
    return html.unescape(TAG_RE.sub("", inner)).strip()


def parse_doc(text: str, path: str) -> Doc:
    """解析「第 4 章矩阵表 tbody」：单元格列表 + 前后文本段 + 表头列名（结构不可判 ⇒ 用法级错误）。"""
    opens = list(TABLE_OPEN_RE.finditer(text))
    if len(opens) != 1:
        raise _UsageError(f"{path}：`<table class=\"matrix\">` 出现 {len(opens)} 次（期望恰好 1 次，fail-closed）")
    table_end = text.find("</table>", opens[0].end())
    if table_end < 0:
        raise _UsageError(f"{path}：矩阵表缺 `</table>`（fail-closed）")
    thead = text.find("<thead", opens[0].end())
    tbody = text.find("<tbody", opens[0].end())
    if tbody < 0 or tbody > table_end:
        raise _UsageError(f"{path}：矩阵表缺 `<tbody>`（fail-closed）")
    tbody_close = text.find("</tbody>", tbody)
    if tbody_close < 0 or tbody_close > table_end:
        raise _UsageError(f"{path}：矩阵表缺 `</tbody>`（fail-closed）")
    head_blob = text[opens[0].end():tbody] if thead < 0 else text[thead:tbody]
    header_cols = tuple(m.lower() for m in re.findall(r'<th\b[^>]*\bdata-col="([^"]+)"', head_blob))

    body = text[tbody:tbody_close]
    # ① 档号归属：按 `<tr>` 边界给每个单元格定 tier（行内首个 data-col="tier" 的单元格）
    tiers: list[str | None] = []
    cur: str | None = None
    for m in TR_OR_CELL_RE.finditer(body):
        if m.group(0).startswith("<tr"):
            cur = None
            continue
        col = _attr(m.group(1) or "", "data-col")
        if col == "tier":
            t = _text_of(m.group(2) or "")
            cur = t if TIER_RE.match(t) else None
        tiers.append(cur)

    cells: list[Cell] = []
    for m in CELL_RE.finditer(body):
        cells.append(Cell(
            attrs=m.group(1) or "",
            inner=m.group(2) or "",
            start=tbody + m.start(),
            end=tbody + m.end(),
            tier=None,
            col=(_attr(m.group(1) or "", "data-col") or None),
        ))
    if len(tiers) != len(cells):
        raise _UsageError(f"{path}：tbody 内 `<tr>`/`<td>` 交错无法对齐（{len(tiers)} vs {len(cells)}，fail-closed）")
    cells = [Cell(c.attrs, c.inner, c.start, c.end, t, c.col) for c, t in zip(cells, tiers)]

    segs, offsets, pos = [], [], tbody
    for c in cells:
        segs.append(text[pos:c.start]); offsets.append(pos)
        pos = c.end
    segs.append(text[pos:tbody_close]); offsets.append(pos)
    prefix = text[:cells[0].start] if cells else text[:tbody]
    suffix = text[cells[-1].end:] if cells else text[tbody_close:]
    return Doc(text=text, cells=tuple(cells), segs=tuple(segs), seg_offsets=tuple(offsets),
               prefix=prefix, suffix=suffix, header_cols=header_cols)


# ---- 定位（行/列 + 片段）-----------------------------------------------------
def _where(text: str, off: int) -> str:
    off = max(0, min(off, len(text)))
    line = text.count("\n", 0, off) + 1
    col = off - (text.rfind("\n", 0, off) + 1) + 1
    return f"{line}:{col}"


def _snippet(text: str, off: int, width: int = 60) -> str:
    lo = max(0, off - width // 2)
    hi = min(len(text), off + width)
    frag = text[lo:hi].replace("\r", "").replace("\n", "⏎")
    return f"…{frag}…" if (lo > 0 or hi < len(text)) else frag


# ---- 判定核心 ----------------------------------------------------------------
#: ★ 2026-10-28：**生成关系标记行**（渲染器按 `meta.authority.data_path` 派生，不是人写的正文）。
#:   副本 `-20261025` 现在自指自己的数据件（审查警告 5 的修法）⇒ 它与基准件的这 4 行必然不同，
#:   但那是**渲染器确定性产出**、不是越界手改 ⇒ 比对前把其中的「数据件路径」归一成占位符。
#:   ⛔ 只归一这四种形态里的路径；其余字节仍逐字比，⛔ 不放过任何真实改动。
_MARKER_PATH = r"[^\s`<>\"']*capability-matrix[^\s`<>\"']*\.data\.json"
_GENERATED_MARKERS = (
    re.compile(r"(<!--\s*generated-by:\s*tools/matrix/render_matrix\.py\s*;\s*source:\s*)"
               + _MARKER_PATH + r"(\s*;\s*do-not-edit\s*-->)"),
    re.compile(r'(<meta\s+name="matrix-source"\s+content=")' + _MARKER_PATH + r'(">)'),
    re.compile(r"(表内数字由脚本从数据文件\s*<code>)" + _MARKER_PATH + r"(</code>)"),
)


def canonicalize_generated_markers(text: str) -> str:
    """把「生成关系标记行」里的数据件路径归一成 `«DATA_PATH»`（只动这四种形态）。"""
    out = text
    for rx in _GENERATED_MARKERS:
        out = rx.sub(lambda m: m.group(1) + "«DATA_PATH»" + m.group(2), out)
    return out


def check_scope(base_text: str, filled_text: str, *, result_columns, status_vocab,
                base_path: str = "base", filled_path: str = "filled") -> list[str]:
    """返回违规清单（空 = 通过）。逐条含 `文件:行:列` + 片段。"""
    allowed = tuple(result_columns) + (MACHINE_STATUS_COL,)
    # ★ 比对前先归一「生成关系标记行」里的数据件路径（见 canonicalize_generated_markers 的说明）：
    #   规范化只作用在那四种由渲染器派生的形态上，其余字节仍逐字比。
    a = parse_doc(canonicalize_generated_markers(base_text), base_path)
    b = parse_doc(canonicalize_generated_markers(filled_text), filled_path)
    bad: list[str] = []

    def v(msg: str, text: str, off: int, *, where_file: str) -> None:
        bad.append(f"{where_file}:{_where(text, off)}  {msg}\n      片段：{_snippet(text, off)}")

    # ① 表外整片零变化（第 1/2/3 章正文、样式、脚本、注释、colgroup、thead —— 逐字节比）
    for label, sa, sb, base_off in (
        ("矩阵表之前（含第 1–3 章正文与表头）", a.prefix, b.prefix, 0),
        ("矩阵表之后（含页脚/脚本）", a.suffix, b.suffix, b.cells[-1].end if b.cells else 0),
    ):
        if sa == sb:
            continue
        k = 0
        while k < min(len(sa), len(sb)) and sa[k] == sb[k]:
            k += 1
        v(f"{label} 被改了（⛔ 只许改第四章实验数据单元格的文本）", b.text, base_off + k, where_file=filled_path)

    # ①b 表头列集合零变化（`<th data-col>` 增删改都红；prefix 已覆盖，此处给直白判据）
    if a.header_cols != b.header_cols:
        off = b.text.find("<thead")
        v(f"表头列集合变了：{list(a.header_cols)} → {list(b.header_cols)}", b.text, off, where_file=filled_path)

    # ② 单元格数量（增删行/单元格都红）
    if len(a.cells) != len(b.cells):
        n = min(len(a.cells), len(b.cells))
        off = b.cells[n].start if n < len(b.cells) else b.text.rfind("</tbody>")
        v(f"矩阵 tbody 单元格数量变了：{len(a.cells)} → {len(b.cells)}（⛔ 不许增删单元格/行）",
          b.text, off, where_file=filled_path)

    # ③ tbody 以外的文本段（第 1/2/3 章、样式、脚本、注释、表头、`<tr>` 标签本身…）
    for i in range(min(len(a.segs), len(b.segs))):
        if a.segs[i] == b.segs[i]:
            continue
        sa, sb = a.segs[i], b.segs[i]
        k = 0
        while k < min(len(sa), len(sb)) and sa[k] == sb[k]:
            k += 1
        v("tbody 内**单元格之外**的标记/文字变了（⛔ 只许改单元格文本）", b.text, b.seg_offsets[i] + k,
          where_file=filled_path)
    if len(a.segs) != len(b.segs):
        v(f"tbody 文本段数量变了：{len(a.segs)} → {len(b.segs)}", b.text, b.seg_offsets[-1], where_file=filled_path)

    # ④ 逐单元格
    for i in range(min(len(a.cells), len(b.cells))):
        ca, cb = a.cells[i], b.cells[i]
        if ca.col != cb.col:
            v(f"第 {i} 个单元格的 data-col 变了：{ca.col!r} → {cb.col!r}", b.text, cb.start, where_file=filled_path)
            continue
        if ca.tier != cb.tier:
            v(f"第 {i} 个单元格的档号归属变了：{ca.tier!r} → {cb.tier!r}", b.text, cb.start, where_file=filled_path)
            continue
        if cb.tier is None:
            v(f"单元格 {cb.col!r} 所在行没有合法档号（`data-col=\"tier\"` 须匹配 ^M\\d{{3}}$）"
              "⇒ 位置不可判，fail-closed", b.text, cb.start, where_file=filled_path)
            continue

        # ④a 属性串（含标签结构/样式内联/style/class）：status 列只放开 st-* 配色类名
        aa, ab = ca.attrs, cb.attrs
        if cb.col == MACHINE_STATUS_COL:
            aa, ab = ST_TOKEN_RE.sub("<ST>", aa), ST_TOKEN_RE.sub("<ST>", ab)
        if aa != ab:
            v(f"[{cb.tier}] `{cb.col}` 单元格的**标签/属性**变了（⛔ 只许改文本）\n"
              f"      原文：<td{ca.attrs}>…</td>\n      现值：<td{cb.attrs}>…</td>",
              b.text, cb.start, where_file=filled_path)
            continue

        if ca.inner == cb.inner:
            continue

        # ④b 内容变了 ⇒ 必须在白名单列 + 必须纯文本（无任何标记）
        if cb.col not in allowed:
            v(f"[{cb.tier}] `{cb.col}` **不是实验结果列**（允许：{', '.join(allowed)}）⇒ 不许有差异",
              b.text, cb.start, where_file=filled_path)
            continue
        for label, inner in (("原版", ca.inner), ("填表后", cb.inner)):
            if TAG_RE.search(inner):
                v(f"[{cb.tier}] `{cb.col}` {label}内容里出现**标记**（⛔ 只许纯文本，"
                  f"塞 `<script>`/`<span>` 等一律红）", b.text, cb.start, where_file=filled_path)
                break
        else:
            if cb.col == MACHINE_STATUS_COL:
                ta, tb = _text_of(ca.inner), _text_of(cb.inner)
                if status_vocab and (ta not in status_vocab or tb not in status_vocab):
                    v(f"[{cb.tier}] `status` 文本越出词表 {sorted(status_vocab)}：{ta!r} → {tb!r}",
                      b.text, cb.start, where_file=filled_path)
    return bad


def diff_summary(base_text: str, filled_text: str, **kw) -> tuple[int, list[str]]:
    """返回 (发生内容变化的单元格数, 违规清单) —— 供调用方打印一行摘要。"""
    allowed = tuple(kw.get("result_columns") or ()) + (MACHINE_STATUS_COL,)
    a, b = parse_doc(base_text, kw.get("base_path", "base")), parse_doc(filled_text, kw.get("filled_path", "filled"))
    changed = sum(1 for ca, cb in zip(a.cells, b.cells)
                  if ca.inner != cb.inner and cb.col in allowed and lb_ok(ca, cb))
    return changed, check_scope(base_text, filled_text, **kw)


def lb_ok(ca: Cell, cb: Cell) -> bool:
    """两侧都无标记（纯文本变化）⇒ True。"""
    return not TAG_RE.search(ca.inner) and not TAG_RE.search(cb.inner)


# ---- CLI --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base = filled = data = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(USAGE, end="")
            return 0
        if a in ("--base", "--filled", "--data"):
            if i + 1 >= len(argv):
                print(f"✗ {a} 缺参数值\n", file=sys.stderr)
                return 1
            if a == "--base":
                base = argv[i + 1]
            elif a == "--filled":
                filled = argv[i + 1]
            else:
                data = argv[i + 1]
            i += 2
            continue
        print(f"✗ 未识别参数 {a}（见 --help）\n", file=sys.stderr)
        return 1
    if not base or not filled:
        print("✗ --base 与 --filled 都必填（见 --help）\n", file=sys.stderr)
        return 1
    data_path = Path(data) if data else _paths.baseline_data()

    try:
        base_p, filled_p = Path(base), Path(filled)
        for p in (base_p, filled_p, data_path):
            if not p.is_file():
                raise _UsageError(f"文件不存在：{p}")
        result_columns = load_result_columns(data_path)
        status_vocab = load_status_vocabulary(data_path)
        base_text = base_p.read_text(encoding="utf-8")
        filled_text = filled_p.read_text(encoding="utf-8")
        kw = dict(result_columns=result_columns, status_vocab=status_vocab,
                  base_path=_paths.rel_of(base_p), filled_path=_paths.rel_of(filled_p))
        changed, violations = diff_summary(base_text, filled_text, **kw)
    except _UsageError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:   # noqa: BLE001 —— 读文件/解析失败一律 fail-closed
        print(f"✗ 读取/解析失败：{exc!r}", file=sys.stderr)
        return 1

    print("# check_fill_scope.py — 填表写范围闸门（§22 · 主席令「只许填第四章实验数据」）")
    print(f"原版        {kw['base_path']}  ({base_p.stat().st_size} B)")
    print(f"填表后      {kw['filled_path']}  ({filled_p.stat().st_size} B)")
    print(f"白名单列    {', '.join(result_columns)}  (+ {MACHINE_STATUS_COL}：文本须在词表内，"
          f"仅 st-* 配色类名可随文本变)")
    print(f"允许面      第 4 章 `<table class=\"matrix\">` 的 `<tbody>` 内 · M 档行 · 上述列 · 纯文本")
    print(f"内容变化     {changed} 个单元格")
    if violations:
        print(f"\n✗ SCOPE VIOLATION：{len(violations)} 处差异越出允许面（rc=2）")
        for n, msg in enumerate(violations[:MAX_PRINT], 1):
            print(f"  {n}. {msg}")
        if len(violations) > MAX_PRINT:
            print(f"  …（其余 {len(violations) - MAX_PRINT} 条同类违规略；⛔ rc 仍为 2）")
        print("\n# 结论：FAIL（rc=2）—— 主席令：⛔ 除第四章实验数据单元格的文本外，报告一字不许动")
        return 2
    print("\n# 结论：SCOPE PASS（rc=0）—— 所有差异都落在「第四章实验数据单元格的文本」面内")
    return 0


if __name__ == "__main__":
    sys.exit(main())
