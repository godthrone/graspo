#!/usr/bin/env python3
"""render_matrix.py — 由「模板 + 数据」渲染出功能矩阵 HTML（唯一允许的 HTML 产出通道）。

用法：
  python3 render_matrix.py --template template.html --data matrix-data.json \
                           --out capability-matrix-new.html
  python3 render_matrix.py --template template.html --data matrix-data.json --stdout

设计要点（对应主席硬要求 ★★）：
- HTML 的一切内容 = template.html（骨架 / 版式 / 第 1/2/3 章静态文字 / 第 4 章读表提示）+ 数据件
  （逐档参数 13 列 + 结果 6 列的**行数据**）。
- ★★ 使用者口径（2026-10-19 裁定）：「JSON 里面只有测试结果，脚本只负责把测试结果填写到最后一章
  测试表里面，其它所有的部分都不能动」。落地为三条：
    ① 表格结构（列 key / 表头 / 列组 / 顺序）= **本文件模块级常量**（`MATRIX_COLUMNS` / `MATRIX_GROUPS`），
       ⛔ 不再随数据件走（数据件已无 `columns` / `groups`）；
    ② 第 1/2/3 章 + 目录 + 页脚 + 首屏注释 = **模板静态文字**（模板里不再有这些章节的占位符）；
    ③ 渲染输出里**只有**最后一章表格的单元格随行数据变（自证见 `run_self_proof.py` ⑲）。
- 结果列禁止手工回填：本脚本是唯一写入 HTML 的通道；verify_matrix.py 会重新渲染并逐字节比对。
- 渲染确定性：不读时钟、不读环境、不读网络；同一 (template, data) ⇒ 同一字节串。
- 支持两类列：
    * 普通列：一个数据字段一格；
    * 合并列（列定义里带 "merge"）：把若干字段按 "sub_labels" 组装成「小节 + 值」的一格，
      用于满足主席「参数（模型/方法/模式/backend/并行维度/关键超参）」合并成一列的要求。
"""
import argparse, html, json, re, sys
from pathlib import Path

#: ★★ 模板占位符白名单（**只列模板里真实存在的**）：`render()` 逐个替换，替换后若仍见 `{{`
#:   ⇒ 直接 SystemExit（防"占位符漏配 ⇒ 静默把 `{{X}}` 发出去"）。
#:   ★ 2026-10-19（「脚本只填最后一章表格」）：第 1/2/3 章、目录、页脚、首屏注释已改为模板静态
#:     文字 ⇒ 原 `{{CRITERIA}}` / `{{EVAL_INFO}}` / `{{VENUE}}` / `{{CODE_VERSION}}` /
#:     `{{UNIFORM_PARAMS}}` / `{{VERDICTS}}` / `{{COVERAGE_WARNING}}` / `{{BOUNDARY}}` /
#:     `{{GEN_BANNER}}` 等占位符一律下架（⛔ 不留"静态区仍由数据现算"的通道）。
PLACEHOLDER_TOKENS = ("{{DOC_TITLE}}", "{{GENERATOR}}", "{{DATA_PATH}}", "{{MATRIX_TABLE}}")

#: ★★ 21 列的功能矩阵表结构（**单一真相源**，宪法 §1.4）：
#:   档号 → 状态 → 主要参数 13 列 → 实验数据 5 列 → 备注 1 列；两级表头 + 5 个列组。
#:   ★ 2026-10-19 对齐轮由旧版 `65ac5e7` 的 21 列逐字恢复；**本轮**这 21 列的结构（列 key /
#:     表头 / 列组归属 / 出现顺序 / sticky / group_start）从数据件 `columns` 上移到这里
#:     —— 数据件从此**只有行数据**（逐档参数 + 结果），结构不再随数据走。
#:   ★ 消费方（⛔ 不得各留一份）：
#:       `build_official.COLUMNS` / `build_official.GROUPS`（生成端）＝ 直接引用本常量；
#:       `verify_matrix.EXPECTED_COLUMNS` / `verify_matrix.EXPECTED_GROUPS`（校验端）＝ 由本常量现算。
#:   ★ 合并列（`merge`）保留同一套编码：`sub_labels` 给出"小节名"，取值一律取自行数据。
MATRIX_COLUMNS = (
    {"key": "tier", "label": "档号", "group": "base", "sticky": True},
    {"key": "status", "label": "是否通过", "group": "status", "group_start": True},
    {"key": "model", "label": "模型", "group": "param", "group_start": True},
    {"key": "algo", "label": "算法", "group": "param"},
    {"key": "mode", "label": "模式", "group": "param"},
    {"key": "backend", "label": "后端", "group": "param"},
    {"key": "tp", "label": "TP", "group": "param"},
    {"key": "pp", "label": "PP", "group": "param"},
    {"key": "dp", "label": "DP", "group": "param"},
    {"key": "cards_total", "label": "总卡数", "group": "param"},
    {"key": "lora_target", "label": "LoRA 目标", "group": "param"},
    {"key": "lr", "label": "lr", "group": "param"},
    {"key": "data_scale", "label": "数据规模", "group": "param"},
    {"key": "steps", "label": "步数/epoch", "group": "param"},
    {"key": "switches", "label": "关键开关", "group": "param"},
    {"key": "peak_mem", "label": "显存峰值(GiB)", "group": "data", "group_start": True},
    {"key": "steps_actual", "label": "实跑步数", "group": "data"},
    {"key": "elapsed", "label": "耗时（每步/总时长）", "group": "data"},
    {"key": "throughput", "label": "吞吐", "group": "data"},
    {"key": "effect", "label": "效果（评测 Δ / loss·reward 窗口）", "group": "data"},
    {"key": "note", "label": "备注（条件 / 限制 / 失败归因）", "group": "note", "group_start": True},
)
#: ★★ 列组的**逐字**定义（第一行表头 colspan 由它给出）：列组顺序 == 列出现顺序。
MATRIX_GROUPS = {"base": "基础", "status": "状态", "param": "主要参数",
                 "data": "实验数据", "note": "备注"}

# 结果列着色用的四态映射（与模板内联脚本同一口径；此处只用于生成带 class 的单元格）
STATUS_CLASS = {
    "已验证可用": "st-pass", "已验证不可用": "st-fail",
    "尚未验证": "st-pending", "未纳入验收": "st-illegal",
}


def cell_html(col, row, by_method=None):
    """渲染一个数据单元格的 HTML 片段。

    ★ `by_method` 只用于「效果」列的**展示层归一**（见 `human_view_effect`）：
      `rows[].effect` 是**冻结行**里的历史读数原文，⛔ 一字不动；旧口径措辞在渲染时改写。
    """
    key, group = col["key"], col["group"]
    classes = [f"g-{group}"]
    if col.get("sticky"):
        classes.append("col-tier")
    if col.get("group_start"):
        classes.append("col-start")

    if col.get("merge"):
        subs = col["merge"]
        labels = col.get("sub_labels", {})
        sep = col.get("merge_sep", " ／ ")
        parts = []
        for sub in subs:
            v = row.get(sub)
            v = "" if v is None else str(v)
            label = labels.get(sub)
            if label:
                cls = "hp" if sub in ("hparams",) else "f"
                parts.append(f'<span class="{cls}" data-f="{html.escape(sub)}">{html.escape(label)}</span> '
                             f'<span data-v="{html.escape(sub)}">{html.escape(v)}</span>')
            else:
                parts.append(f'<span data-v="{html.escape(sub)}">{html.escape(v)}</span>')
        inner = f'<span class="sep">{html.escape(sep.strip())}</span>'.join(parts)
        classes.append("param")
        return f'<td class="{" ".join(classes)}" data-col="{key}">{inner}</td>'

    val = "" if row.get(key) is None else str(row[key])
    if key == "effect":
        val = human_view_effect(val, row.get("train_method"), by_method)
    if key == "status":
        cls = STATUS_CLASS.get(val.strip(), "st-dash")
        classes.append(cls)
    return f'<td class="{" ".join(classes)}" data-col="{key}">{html.escape(val)}</td>'


#: ★★ 展示层归一（2026-10-17 修正轮 1，全局审查 B2）：
#:   `rows[].effect` 是**冻结行**里的历史读数原文，其中 T037/T038/T039 三格带已作废的
#:   旧口径措辞（「冻结阈值 +5.0 pp」/「目标 +10.0 pp 仅展示」），与同一页 §2「通过标准」
#:   自相矛盾。★ 修法**只能在展示层**（⛔ `rows[]` 一字不动），⇒ 与「rows sha 不变」不冲突。
#:   ★ 数值与方向**一律取自** `meta.criteria_caliber.threshold.by_method[<算法>]`
#:     （⛔ 本文件不写死任何阈值）；旧措辞的*模式*匹配不改变任何读数/结论。
_RE_STALE_CALIBER_PAREN = re.compile(r"（\s*冻结阈值\s*[^）]*?仅展示\s*）")
_RE_STALE_FROZEN_CLAUSE = re.compile(r"（\s*[^）]*?冻结阈值[^）]*?）")
_RE_STALE_TARGET_SENTENCE = re.compile(r"[；;]?\s*目标\s*[^。]*?仅展示[^。]*。?")
#: 兜底（防新变体漏网 ⇒ 机核 ④H 必红）：数字无关的旧措辞清除。
_RE_STALE_FROZEN_ANY = re.compile(r"冻结阈值\s*\+?\d+(?:\.\d+)?\s*pp")
_RE_STALE_TARGET_ANY = re.compile(r"目标\s*\+?\d+(?:\.\d+)?\s*pp\s*仅展示[^。]*。?")


def human_view_effect(text, method, by_method) -> str:
    """把「效果」列里的**旧口径措辞**归一为现行口径（展示层专用）。

    - 现行通过线文本逐字取自 `by_method[<算法>].pass_line_text`（唯一真相源）；
    - 数值/读数/结论⛔ 一字不改，只改**已作废的说法**；
    - `method` / `by_method` 缺失时仍然清除旧措辞（fail-safe，⛔ 不因读不到就把旧口径放行）。
    """
    line = str(((by_method or {}).get(str(method or "").lower()) or {}).get("pass_line_text") or "")
    name = str(method or "").upper() or "本算法"
    cur = f"（判据② {name} 通过线：{line}）" if line else f"（判据② {name} 通过线，见第 2 节）"
    cur_missed = (f"（未达判据② {name} 通过线：{line}）" if line
                  else f"（未达判据② {name} 通过线）")
    t = str(text)
    t = _RE_STALE_CALIBER_PAREN.sub(cur, t)
    t = _RE_STALE_FROZEN_CLAUSE.sub(cur_missed, t)
    t = _RE_STALE_TARGET_SENTENCE.sub("。", t)
    t = _RE_STALE_FROZEN_ANY.sub("判据② 通过线", t)
    t = _RE_STALE_TARGET_ANY.sub("", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def build_matrix_table(data):
    # ★★ 表结构来自**本模块常量**（⛔ 不再读 `data["columns"]` / `data["groups"]`）：
    #   使用者的口径是「JSON 里面只有测试结果，脚本只负责把测试结果填进最后一章测试表」
    #   ⇒ 列 key / 表头 / 列组 / 顺序属于**渲染器的静态结构**，不随数据走（§1.4 单一真相源）。
    cols = [dict(c) for c in MATRIX_COLUMNS]
    groups = dict(MATRIX_GROUPS)
    rows = data["rows"]
    # ★ 展示层归一用的现行通过线（唯一真相源；⛔ 渲染层不写死任何阈值数值）。
    by_method = (((data.get("meta") or {}).get("criteria_caliber") or {})
                 .get("threshold") or {}).get("by_method") or {}

    # 列组按出现顺序聚合（仅用于第一行表头的 colspan）
    order, spans = [], {}
    for c in cols:
        g = c["group"]
        if g not in spans:
            order.append(g)
            spans[g] = 0
        spans[g] += 1
    sticky_key = next((c["key"] for c in cols if c.get("sticky")), None)
    first = cols[0]
    assert first.get("sticky") is True and first["key"] == sticky_key, "首列必须是 sticky 档号列"

    out = []
    out.append('<table class="matrix">')
    out.append("<colgroup>")
    for c in cols:
        out.append(f'<col data-col="{c["key"]}" class="g-{c["group"]}">')
    out.append("</colgroup>")
    out.append("<thead>")
    head1 = [f'<th class="g-{first["group"]} col-tier col-start" data-colgroup="{groups[first["group"]]}" '
             f'rowspan="2">{html.escape(first["label"])}</th>']
    for g in order:
        span = spans[g] - 1 if g == first["group"] else spans[g]
        if span <= 0:
            continue
        head1.append(f'<th class="g-{g}" colspan="{span}">{html.escape(groups[g])}</th>')
    out.append("<tr>" + "".join(head1) + "</tr>")
    head2 = "".join(
        f'<th class="g-{c["group"]}{" col-start" if c.get("group_start") else ""}" '
        f'data-col="{c["key"]}">{html.escape(c["label"])}</th>'
        for c in cols if c["key"] != sticky_key)
    out.append("<tr>" + head2 + "</tr>")
    out.append("</thead>")
    out.append("<tbody>")
    for r in rows:
        out.append("<tr>" + "".join(cell_html(c, r, by_method) for c in cols) + "</tr>")
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)


# ★★ 2026-10-19（使用者口径「脚本只负责把测试结果填进最后一章测试表」）：原先在这里的
#   `build_criteria` / `build_eval_info` / `build_uniform_block` / `build_venue` /
#   `build_code_version` / `build_coverage_warning` / `build_boundary` / `build_gen_banner`
#   一律下架 —— 第 1/2/3 章（判据①–④ 与逐算法通过线、评测口径、实验条件统一口径块）现在是
#   **模板静态文字**（`template.html`），⛔ 不再由数据现算。
#   ⛔ 判据数值与语义一字未改：静态文字逐字取自本轮在盘渲染结果，只换了承载位置。
#   仍保留在渲染层的：`build_matrix_table`（最后一章表格，随行数据变）与
#   `human_view_effect`（「效果」列展示层归一，`verify_matrix` 同源引用）。


# ★★ 2026-10-19：`_lora_backends_html` / `_target_regex_html` 一并下架 —— 它们是**从未被
#   模板引用**的死函数（模板占位符里没有对应 token，渲染结果里也不出现其文本）。
#   本轮把「渲染器只负责填最后一章表格」写成可机核约束 ⇒ 顺带清掉这两个不再有出口的死码。
#   ⛔ 它们依赖的 `meta.lora_target_backends` 数据一字未动（仍在数据件里，另有消费方）。


def render(template_text, data):
    meta = data["meta"]
    table = build_matrix_table(data)
    # ★★ 只保留模板里**真实存在**的 4 个占位符（见 `PLACEHOLDER_TOKENS`）：
    #   第 1/2/3 章、目录、页脚、首屏注释已是模板静态文字 ⇒ ⛔ 不再从数据现算
    #   （使用者口径：脚本只负责把测试结果填进最后一章测试表）。
    repl = {
        "{{DOC_TITLE}}": meta.get("title", ""),
        "{{GENERATOR}}": (meta.get("authority") or {}).get("generator", ""),
        "{{DATA_PATH}}": (meta.get("authority") or {}).get("data_path", ""),
        "{{MATRIX_TABLE}}": table,
    }
    out = template_text
    for token in PLACEHOLDER_TOKENS:
        out = out.replace(token, repl[token])
    leftover = []
    i = 0
    while True:
        j = out.find("{{", i)
        if j < 0:
            break
        k = out.find("}}", j)
        leftover.append(out[j:k + 2] if k > 0 else out[j:j + 40])
        i = j + 2
    if leftover:
        raise SystemExit(f"ERROR: 模板中存在未替换的占位符: {leftover}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()

    template_text = Path(a.template).read_text(encoding="utf-8")
    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    out = render(template_text, data)

    if a.stdout:
        sys.stdout.write(out)
        return
    if not a.out:
        raise SystemExit("ERROR: 需要 --out 或 --stdout")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(out, encoding="utf-8")
    print(f"rendered {len(out.encode('utf-8'))} bytes -> {p}")


if __name__ == "__main__":
    main()
