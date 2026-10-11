#!/usr/bin/env python3
"""render_matrix.py — 由「模板 + 数据」渲染出功能矩阵 HTML（唯一允许的 HTML 产出通道）。

用法：
  python3 render_matrix.py --template template.html --data matrix-data.json \
                           --out capability-matrix-new.html
  python3 render_matrix.py --template template.html --data matrix-data.json --stdout

设计要点（对应主席硬要求 ★★）：
- HTML 的一切内容 = template.html（骨架 / 版式 / 规则文字）+ matrix-data.json（列定义 + 54 档数据）。
- 结果列禁止手工回填：本脚本是唯一写入 HTML 的通道；verify_matrix.py 会重新渲染并逐字节比对。
- 渲染确定性：不读时钟、不读环境、不读网络；同一 (template, data) ⇒ 同一字节串。
- 支持两类列：
    * 普通列：一个数据字段一格；
    * 合并列（列定义里带 "merge"）：把若干字段按 "sub_labels" 组装成「小节 + 值」的一格，
      用于满足主席「参数（模型/方法/模式/backend/并行维度/关键超参）」合并成一列的要求。
"""
import argparse, html, json, sys
from pathlib import Path

PLACEHOLDER_TOKENS = ("{{DATA_REVISION}}", "{{SOURCE_BASENAME}}", "{{SOURCE_SHA256_SHORT}}",
                      "{{SOURCE_PATH}}", "{{SOURCE_SHA256}}", "{{CRITERIA_VERSION}}",
                      "{{ROW_COUNT}}", "{{MATRIX_TABLE}}", "{{UNIFORM_PARAMS}}", "{{DOC_DATE}}",
                      "{{DOC_TITLE}}", "{{GEN_BANNER}}", "{{COVERAGE_WARNING}}",
                      "{{UNVERIFIED_LIST}}", "{{BOUNDARY}}", "{{CALIBER}}", "{{VERDICTS}}",
                      "{{GENERATOR}}", "{{DATA_PATH}}")

# 结果列着色用的四态映射（与模板内联脚本同一口径；此处只用于生成带 class 的单元格）
STATUS_CLASS = {
    "通过": "st-pass", "失败": "st-fail", "不通过": "st-fail",
    "条件不合逻辑": "st-illegal", "不适用": "st-illegal",
    "未完成测试": "st-pending", "待跑": "st-pending",
}


def cell_html(col, row):
    """渲染一个数据单元格的 HTML 片段。"""
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
    if key == "status":
        cls = STATUS_CLASS.get(val.strip(), "st-dash")
        classes.append(cls)
    return f'<td class="{" ".join(classes)}" data-col="{key}">{html.escape(val)}</td>'


def build_matrix_table(data):
    cols = data["columns"]
    rows = data["rows"]
    groups = data["groups"]

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
        out.append("<tr>" + "".join(cell_html(c, r) for c in cols) + "</tr>")
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)


def build_uniform_block(data):
    """渲染 §1「统一口径」块：对全部档位相同的参数（数据文件 uniform 字段）。

    覆盖列 coverage 形如 'config-51/54' / 'rule-54/54'，逐行如实标注；空 note 不显示。
    """
    uni = data.get("uniform") or {}
    items = uni.get("items") or []
    out = ['<table class="plain uniform">',
           '<colgroup><col class="k"><col><col class="u"></colgroup>',
           '<thead><tr><th>参数</th><th>统一取值</th><th>覆盖 / 说明</th></tr></thead>',
           '<tbody>']
    for it in items:
        note = it.get("note") or ""
        cov = it.get("coverage") or ""
        tail = "；".join(x for x in (cov, note) if x)
        out.append(
            f'<tr><td><b>{html.escape(it["label"])}</b><br>'
            f'<span class="u-key"><code>{html.escape(it["key"])}</code></span></td>'
            f'<td><code>{html.escape(it["value"])}</code></td>'
            f'<td>{html.escape(tail)}</td></tr>')
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)



def build_gen_banner(data):
    a = data["meta"].get("authority", {})
    gen, src = html.escape(a.get("generator", "")), html.escape(a.get("data_path", ""))
    return (f'<div class="warnbox"><b>自动生成文件。</b>本文件由 <code>{gen}</code> 从 <code>{src}</code> '
            f'自动生成，<b>禁止手工编辑</b>。任何修改都必须改数据或生成器后重新生成。</div>')


def build_coverage_warning(data):
    cov = data["meta"].get("coverage", {})
    if cov.get("gate_met"):
        return ('<div class="hint"><b>覆盖度：</b>全部档位均已有明确结论，覆盖度门槛已满足。</div>')
    return (f'<div class="warnbox"><b>★ 覆盖度警示（未达门槛）。</b>本矩阵共 <b>{cov.get("total")}</b> 档：'
            f'「已验证可用」<b>{cov.get("verified_available")}</b> 档、'
            f'「已验证不可用」<b>{cov.get("verified_unavailable")}</b> 档、'
            f'「未纳入验收」<b>{cov.get("not_in_acceptance")}</b> 档、'
            f'「<b>尚未验证</b>」<b>{cov.get("not_verified")}</b> 档。'
            f'⇒ <b>项目整体能力承诺尚未满足</b>；「尚未验证」档位不计入可用档位数量。</div>')


def build_unverified_list(data):
    cov = data["meta"].get("coverage", {})
    ts = cov.get("unverified_tiers", [])
    if not ts:
        return '<p class="note">（无「尚未验证」档位。）</p>'
    return (f'<p class="note"><b>尚未验证档位清单（{len(ts)} 档）</b>：' +
            "、".join(f"<code>{html.escape(t)}</code>" for t in ts) + "</p>")


def build_boundary(data):
    b = data["meta"].get("boundary_declaration", {})
    pts = "".join(f"<li>{html.escape(x)}</li>" for x in b.get("points", []))
    return (f'<div class="hint"><b>承诺边界声明。</b>{html.escape(b.get("text", ""))}'
            f'<ul>{pts}</ul><p class="note">{html.escape(b.get("coverage_caveat", ""))}</p></div>')


def build_caliber(data):
    c = data["meta"].get("criteria_caliber", {})
    d, th, bl = c.get("dataset", {}), c.get("threshold", {}), c.get("baseline", {})
    crit = "".join(f"<li>{html.escape(x)}</li>" for x in c.get("availability_criteria", []))
    return ('<table class="plain"><tbody>'
            f'<tr><th>可用性判据</th><td><ul>{crit}</ul></td></tr>'
            f'<tr><th>评测数据集</th><td>{html.escape(d.get("id", ""))}；'
            f'{html.escape(d.get("availability", ""))}；版本锚点 sha256 <code>{html.escape(d.get("version_anchor_sha256", ""))}</code>；'
            f'样本量 {d.get("sample_count")}；评测方法 {html.escape(d.get("method", ""))}</td></tr>'
            f'<tr><th>判据标识</th><td><code>{html.escape(c.get("criteria_id", ""))}</code></td></tr>'
            f'<tr><th>阈值与目标值</th><td>{html.escape(th.get("name", ""))} = {th.get("value")} {html.escape(th.get("unit", ""))}；'
            f'{html.escape(th.get("status", ""))}</td></tr>'
            f'<tr><th>基线口径</th><td>{html.escape(bl.get("role", ""))}；{html.escape(bl.get("caliber", ""))}</td></tr>'
            '</tbody></table>')


def build_verdicts(data):
    rows = [r for r in data.get("rows", []) if r.get("verdict")]
    excl = [r for r in data.get("rows", []) if r.get("excluded_reason")]
    if not rows and not excl:
        return ""
    out = ['<table class="plain"><thead><tr><th>档号</th><th>结论</th><th>未过判据</th><th>已通过项（事实保留）</th></tr></thead><tbody>']
    for r in rows:
        v = r["verdict"]
        passed = "；".join(html.escape(x) for x in v.get("passed_criteria", []))
        out.append(f'<tr><td><code>{html.escape(r["tier"])}</code></td><td>{html.escape(v.get("state", ""))}</td>'
                   f'<td>{html.escape(v.get("failed_criterion", ""))}<br><span class="note">'
                   f'{html.escape(v.get("failed_detail", ""))}</span></td><td>{passed}</td></tr>')
    out.append("</tbody></table>")
    if excl:
        out.append('<p class="note"><b>未纳入验收档位（逐档理由）</b>：</p><ul>')
        for r in excl:
            out.append(f'<li><code>{html.escape(r["tier"])}</code>：{html.escape(r["excluded_reason"])}</li>')
        out.append("</ul>")
    return "\n".join(out)


def render(template_text, data):
    meta = data["meta"]
    # 正式件不含 authoritative_source（脱敏）⇒ 用 .get 兜底；模板若未引用这些占位符则无需取值
    src = meta.get("authoritative_source") or {}
    table = build_matrix_table(data)
    repl = {
        "{{DATA_REVISION}}": meta.get("data_revision") or meta.get("version", ""),
        "{{SOURCE_BASENAME}}": src.get("basename", ""),
        "{{SOURCE_SHA256_SHORT}}": (src.get("sha256") or "")[:8],
        "{{SOURCE_PATH}}": src.get("path", ""),
        "{{SOURCE_SHA256}}": src.get("sha256", ""),
        "{{CRITERIA_VERSION}}": meta.get("criteria_version", ""),
        "{{ROW_COUNT}}": str(len(data["rows"])),
        "{{MATRIX_TABLE}}": table,
        "{{UNIFORM_PARAMS}}": build_uniform_block(data),
        "{{DOC_DATE}}": meta.get("doc_date", ""),
        "{{DOC_TITLE}}": meta.get("title", ""),
        "{{GEN_BANNER}}": build_gen_banner(data),
        "{{COVERAGE_WARNING}}": build_coverage_warning(data),
        "{{UNVERIFIED_LIST}}": build_unverified_list(data),
        "{{BOUNDARY}}": build_boundary(data),
        "{{CALIBER}}": build_caliber(data),
        "{{VERDICTS}}": build_verdicts(data),
        "{{GENERATOR}}": (meta.get("authority") or {}).get("generator", ""),
        "{{DATA_PATH}}": (meta.get("authority") or {}).get("data_path", ""),
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
