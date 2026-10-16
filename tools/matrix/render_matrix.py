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
                      "{{DOC_TITLE}}", "{{GEN_BANNER}}",
                      "{{BOUNDARY}}", "{{CALIBER}}", "{{VERDICTS}}",
                      "{{GENERATOR}}", "{{DATA_PATH}}",
                      "{{COVERAGE_WARNING}}", "{{DOC_SCOPE}}", "{{VENUE}}", "{{CODE_VERSION}}")

# 结果列着色用的四态映射（与模板内联脚本同一口径；此处只用于生成带 class 的单元格）
STATUS_CLASS = {
    "已验证可用": "st-pass", "已验证不可用": "st-fail",
    "尚未验证": "st-pending", "未纳入验收": "st-illegal",
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




def count_statuses(data):
    """按四态统计档位（★ 数据驱动：计数一律从 `rows` 现算，⛔ 不写死任何数字）。"""
    vocab = list((data.get("meta") or {}).get("status_vocabulary") or [])
    rows = data.get("rows") or []
    for r in rows:
        st = r.get("status")
        if st not in vocab:
            vocab.append(st)
    return {s: sum(1 for r in rows if r.get("status") == s) for s in vocab}


def build_coverage_warning(data):
    """★ 覆盖度门槛与首屏警示 + 「尚未验证」档位清单与数量（宪法 §17.7「覆盖度门槛与失败态」）。

    判据（§17.7）：只有当**全部档位都已有明确结论**（取值属于「已验证可用 / 已验证不可用 /
    未纳入验收」——即：不是「尚未验证」）时，才允许在文档任何位置宣称项目整体能力承诺；
    未达此条件时，**必须**在文档首屏写明显式警示，且**标题与摘要不得使用能力承诺式措辞**，
    并**同时列出「尚未验证」的档位清单与数量**。

    ★ 一切计数与清单从数据现算；★ 覆盖度达标时本警示块**自动消失**（⛔ 不留空壳、⛔ 不写死文案）。
    """
    counts = count_statuses(data)
    unverified = [str(r.get("tier")) for r in (data.get("rows") or [])
                  if r.get("status") == "尚未验证"]
    if not unverified:
        return ""
    vocab = list((data.get("meta") or {}).get("status_vocabulary") or [])
    order = [s for s in vocab if s in counts] or list(counts)
    cnt = "；".join(f"{s} <b>{counts[s]}</b> 档" for s in order)
    _total = len(data.get("rows") or [])
    return (
        f'<div class="warnbox" id="coverage-warning" data-row-count="{_total}">'
        '<b>覆盖度警示（结论尚未齐备）。</b>'
        f'本矩阵共 <b>{_total}</b> 个档位，当前结论分布为：{cnt}。'
        '<b>尚有档位没有明确结论</b>（状态为「尚未验证」）⇒ <b>本矩阵此时不得被读作'
        '「所列举的能力组合都可用」</b>，标题与摘要也不作能力承诺；'
        '「尚未验证」<b>不计入</b>可用档位数量。'
        f'<br><b>尚未验证的档位（{len(unverified)} 档）</b>：'
        + "、".join(f"<code>{html.escape(t)}</code>" for t in unverified) + "。</div>")


def build_venue(data):
    """实验场所（★ 数据驱动；⛔ 不在渲染层写死机器名 / 内网地址 / 主机编号）。"""
    return html.escape(str((data.get("meta") or {}).get("venue") or ""))


def build_code_version(data):
    """代码版本（实验条件之一；★ 数据驱动）。"""
    cv = ((data.get("meta") or {}).get("code_version") or {})
    if not cv:
        return ""
    parts = [str(cv.get(k)) for k in ("repo", "branch", "commit") if cv.get(k)]
    return html.escape(" · ".join(parts))


def build_boundary(data):
    """★ 承诺边界声明（§17.7）。措辞**逐字**取自宪法 §17.7 的引文（不得弱于），
    另加一行**覆盖度门槛注**：本矩阵的「整体能力承诺」以覆盖度门槛为前提，
    当前是否已达成由「覆盖度警示」块给出（⛔ 断言的目标不是覆盖度必须达标，
    而是**不得在未达标时读作已完成承诺**）。
    """
    b = data["meta"].get("boundary_declaration", {})
    pts = "".join(f"<li>{html.escape(x)}</li>" for x in b.get("points", []))
    n_pending = count_statuses(data).get("尚未验证", 0)
    gate = (f'<p class="note"><b>覆盖度门槛（本矩阵整体承诺的生效前提）：</b>'
            f'只有当全部档位都已有明确结论（不是「尚未验证」）时，才在文档任何位置宣称项目整体能力承诺。'
            f'当前尚有 {n_pending} 档为「尚未验证」⇒ <b>上述整体承诺尚未生效</b>；'
            f'本矩阵此时只呈现<b>逐档状态</b>，不把「未验证」读作「可用」。</p>') if n_pending else ""
    caveat = html.escape(b.get("coverage_caveat", ""))
    return (f'<div class="hint"><b>承诺边界声明。</b>{html.escape(b.get("text", ""))}'
            f'<ul>{pts}</ul>{gate}'
            + (f'<p class="note">{caveat}</p>' if caveat else "") + '</div>')


def threshold_html(th):
    """★ 判据②（效果）口径渲染：**逐算法分列**（SFT / GRASPO / CPT / OPD）。

    ★ 数据驱动：一律从 `threshold.by_method` 取值，⛔ 不在渲染层写死任何阈值。
    ★ 向后兼容：无 `by_method` 时回退旧的单一 `value` 形态。
    """
    bm = th.get("by_method") or {}
    if not bm:
        return (f'{html.escape(str(th.get("name", "")))} = {th.get("value")} {html.escape(str(th.get("unit", "")))}；'
                f'{html.escape(str(th.get("applies_to", "")))}；{html.escape(str(th.get("status", "")))}')
    names = {"sft": "SFT", "graspo": "GRASPO（RL）", "cpt": "CPT", "opd": "OPD"}
    rows = []
    for k in ("sft", "graspo", "cpt", "opd"):
        v = bm.get(k)
        if not v:
            continue
        extra = ""
        if v.get("also"):
            extra += f'；并须：{html.escape(str(v["also"]))}'
        if v.get("teacher"):
            extra += f'；教师 = {html.escape(str(v["teacher"]))}'
        ep = v.get("epoch")
        if ep is not None and v.get("epoch_previous") is not None and v.get("epoch_previous") != ep:
            # ★ 修正包 🔴-2：不得写「本轮/本次」这类**相对时间指代**（开发过程措辞）
            #   ⇒ 只陈述**现行取值**（数值逐字不变），变更史不进人读件。
            ep_txt = f'epoch={ep}（现行口径）'
        elif ep is not None:
            ep_txt = f'epoch={ep}（不变）'
        else:
            ep_txt = "—"
        rows.append(
            f'<tr><td><b>{names[k]}</b></td>'
            f'<td>{html.escape(str(v.get("pass_line_text", "")))}</td>'
            f'<td>主线 = {html.escape(str(v.get("criterion", "")))} {html.escape(str(v.get("op", "")))} '
            f'{v.get("value")} {html.escape(str(v.get("unit", "")))}'
            f'（{html.escape(str(v.get("metric", "")))}）{extra}</td>'
            f'<td>{html.escape(ep_txt)}</td></tr>')
    return ('<table class="plain"><thead><tr><th>算法</th><th>判据② 通过线（点估计 Δ）</th>'
            '<th>口径</th><th>epoch</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table>'
            + f'<p class="note">{html.escape(str(th.get("status", "")))}</p>'
            + f'<p class="note">旧口径留痕：{html.escape(str(th.get("historical_note", "")))}</p>')
            # ★ 「by_method 是跑次阈值判定的唯一真相源」= **内部回填机制说明** ⇒ 不进人读视图。


def caliber_change_html(c):
    ch = c.get("caliber_change") or {}
    if not ch:
        return "—"
    out = (f'<b>{html.escape(str(ch.get("effective", "")))}</b>｜来源：{html.escape(str(ch.get("source", "")))}<br>'
           f'<b>改动</b>：{html.escape(str(ch.get("changed", "")))}<br>'
           f'<b>未动</b>：{html.escape(str(ch.get("unchanged", "")))}<br>'
           f'<b>被取代</b>：{html.escape(str(ch.get("superseded", "")))}')
    # ★★ 「读数变化登记」= 内部跑次史（逐档旧读数作废 / 归因更正的经过）⇒ **不进人读视图**
    #    （宪法 §17.7：矩阵只承载判据 / 口径 / 实验条件 / 逐档结论；过程迁 `.local/`）。
    #    ★ 仍进人读视图的是上文的「改动 / 未动 / 被取代」三行——它们是**口径**，不是过程。
    return out


def _measured_authority_short(c):
    """★ 精简轮 2：把「步数对账」压成一句话——只留**实测权威 + 以实跑为准 + 结论**。

    ⛔ 不改数值：`measured_authority` 逐字保留；⛔ 不渲染内部概念对账过程。
    """
    sr = c.get("step_reconciliation") or {}
    ma = str(sr.get("measured_authority") or "—").strip()
    return (f"{ma}"
            "（**以实跑累计步数为准**：该组合的优化步数由运行期触发次数决定，"
            "无法从配置先验算出；表中「步数」列为按推导式的**上限**。）")


def build_caliber(data):
    c = data["meta"].get("criteria_caliber", {})
    d, th, bl = c.get("dataset", {}), c.get("threshold", {}), c.get("baseline", {})
    crit = "".join(f"<li>{html.escape(x)}</li>" for x in c.get("availability_criteria", []))
    return ('<table class="plain"><tbody>'
            f'<tr><th>判据（四条）</th><td><ul>{crit}</ul></td></tr>'
            # ★ 精简轮 2：内部口径迭代史（改动一/二/三、来源、未动项）已移出 ⇒ 归档件 §1.3 逐字保留；
            #   此处只留**访客需要的一句**：判据口径变更过、旧读数作废、只以现行口径结论为准。
            '<tr><th>判据口径（现行）</th><td>'
            '<b>本页结论一律以现行判据口径为准</b>（判据口径本身经过变更，变更明细不进本页面）；'
            '旧口径下的读数一律作废、<b>不得引用</b>（历史读数仅作为留痕保留在数据文件里）。变更明细不进本页面。</td></tr>'
            f'<tr><th>评测数据集</th><td>{html.escape(d.get("id", ""))}；'
            f'{html.escape(d.get("availability", ""))}；版本锚点 sha256 <code>{html.escape(d.get("version_anchor_sha256", ""))}</code>；'
            f'样本量 {d.get("sample_count")}；评测方法 {html.escape(d.get("method", ""))}'
            # ★ 「是否同分布」口径（诉求 5）：数据驱动 —— 数据里没写就渲染空串（⛔ 不写死文本）
            f'{"<br><b>训练/评测是否同分布</b>：" + html.escape(str(d.get("train_eval_relation"))) if d.get("train_eval_relation") else ""}'
            f'{"<br>" + html.escape(str(d.get("train_eval_relation_role"))) if d.get("train_eval_relation_role") else ""}</td></tr>'
            f'<tr><th>判据标识</th><td><code>{html.escape(c.get("criteria_id", ""))}</code></td></tr>'
            f'<tr><th>阈值与目标值（判据②）</th><td>{threshold_html(th)}</td></tr>'
            f'<tr><th>基线口径</th><td>{html.escape(bl.get("role", ""))}；{html.escape(bl.get("caliber", ""))}</td></tr>'
            f'<tr><th>结论与参数的对应</th><td>{html.escape((c.get("conclusion_rule") or {}).get("rule", ""))}</td></tr>'
            f'<tr><th>状态用词</th><td>{html.escape((c.get("conclusion_rule") or {}).get("status_wording", ""))}</td></tr>'
            f'<tr><th>epoch 口径</th><td>{html.escape((c.get("epoch_policy") or {}).get("by_method", ""))}</td></tr>'
            f'<tr><th>步数口径（取消上限）</th><td>{html.escape((c.get("epoch_policy") or {}).get("steps", ""))}</td></tr>'
            f'<tr><th>检查点间隔</th><td>{html.escape((c.get("epoch_policy") or {}).get("checkpoint", ""))}</td></tr>'
            # ★★ 精简轮 2：移出三段**给自己看的开发过程**（原文进《归档 · 轮 2》§1.3）：
            #   「口径证据（基准散布）」= 旧口径下的基准散布推算；「已知订正（待跑次侧）」= 空行；
            #   「已知问题（登记待修）」= 内部缺陷台账（含代码路径与内部推理）。
            #   ⛔ 不是丢事实：① 该缺陷的**面向使用者结论**已在实验条件「步数口径」里如实披露
            #      （「GRASPO × native 的推导式不成立 ⇒ 以实跑累计步数为准」）；
            #   ② 三段原文逐字保留在归档件里。
            # ★ 保留但重写措辞：「步数对账」只留**访客需要的那一句**（实测权威 + 以实跑为准），
            #   去掉内部概念对账（三者关系 / 名义 / 对账式 / 双验 = 内部推算过程）。
            f'<tr><th>步数口径 · 实测权威</th><td>'
            f'{html.escape(_measured_authority_short(c))}</td></tr>'
            
            f'<tr><th>LoRA 目标（按后端分流）</th><td>{_lora_backends_html(data)}</td></tr>'
            f'<tr><th>ms-swift 正则下发机制</th><td>{_target_regex_html(data)}</td></tr>'
            '</tbody></table>')


def build_verdicts(data):
    """★ 精简轮 2：结论说明**压到两小段**（逐档结论已在矩阵表的「状态 / 未过判据 / 备注」三列里）。

    只保留矩阵表**放不下**的两件事：
      ① 「已验证不可用」各档的**已通过项**（§17.7 要求"已通过项事实保留"）；
      ② 「未纳入验收」各档的**不实施理由**（§17.7 要求"逐档写明理由与适用范围"）。
    ⛔ 不再重复"状态 + 未过判据"（表里已有）；⛔ 不改任何取值。
    """
    rows = data.get("rows", [])
    passed_items, excl_items = [], []
    for r in rows:
        v = r.get("verdict") or {}
        if r.get("status") == "已验证不可用" and (v.get("passed_criteria") or []):
            txt = "；".join(str(x).strip() for x in v["passed_criteria"] if str(x).strip())
            if txt:
                passed_items.append((str(r.get("tier")), txt))
        if r.get("status") == "未纳入验收" and str(r.get("excluded_reason") or "").strip():
            excl_items.append((str(r.get("tier")), str(r["excluded_reason"]).strip()))
    out = []
    if passed_items:
        out.append('<p class="note" style="margin-top:12px;">'
                   '<b>已通过项（事实保留）</b>——「未过第 1 / 2 条」时，其余已通过的事实一律保留：'
                   '未过一条不等于整档没有可用的部分。</p>')
        out.append('<ul class="tight">')
        for t, txt in passed_items:
            out.append(f'<li><code>{html.escape(t)}</code>：{html.escape(txt)}</li>')
        out.append("</ul>")
    if excl_items:
        out.append('<p class="note" style="margin-top:12px;">'
                   '<b>未纳入验收档位（逐档理由）</b>——这些档位不实施、不产出可运行配置；'
                   '它们<b>不是</b>「试过了不行」，<b>不得</b>读作效果否定。</p>')
        out.append('<ul class="tight">')
        for t, txt in excl_items:
            out.append(f'<li><code>{html.escape(t)}</code>：{html.escape(txt)}</li>')
        out.append("</ul>")
    return "\n".join(out)


def _lora_backends_html(data):
    lb = (data.get("meta") or {}).get("lora_target_backends") or {}
    if not lb:
        return "—"
    n, m = lb.get("native", {}), lb.get("msswift", {})
    nc, mc = n.get("coverage") or {}, m.get("coverage") or {}
    ns = lb.get("namespaces") or {}
    reg = (m.get("expression") or [""])[0]
    lang = "、".join(f"<code>{html.escape(x)}</code>" for x in (n.get("language_targets") or []))
    return (
        f'<b>native</b>：preset 名 <code>{" + ".join(n.get("expression", []))}</code> ⇒ '
        f'实测覆盖 <b>{nc.get("total")}</b> 件（语言 <b>{nc.get("language")}</b> + 视觉 <b>{nc.get("visual")}</b>；'
        f'依据：{html.escape(nc.get("basis", ""))}）。<br>'
        f'native 语言 12 项逐字：{lang}。<br>'
        f'<b>ms-swift</b>：<b>不认 preset 名</b>（另一套语汇 ⇒ <code>ValueError: Target modules not found</code>）'
        f'⇒ 用<b>真实 HF 名正则</b>（仓内 13 份曾训练成功的 adapter 的既有形态）：<br>'
        f'<code>{html.escape(reg)}</code><br>'
        f'⇒ **实测 <b>{mc.get("total")}</b>** 件（language_model <b>{mc.get("language")}</b> '
        f'+ visual <b>{mc.get("visual")}</b>）。<br>'
        f'<span class="note"><b>两后端覆盖集不相等</b>（均为实测锚）：native <b>{nc.get("total")}</b> 件'
        f'（language {nc.get("language")} + visual {nc.get("visual")}） vs ms-swift <b>{mc.get("total")}</b> 件'
        f'（language {mc.get("language")} + visual {mc.get("visual")}）⇒ 不得宣称等价。'
        + ('' if not mc.get("comparability_note") else f'<br>可比性提示：{html.escape(str(mc.get("comparability_note")))}')
        + f'<br>三方命名空间各不相同：native 公开签名 <code>{html.escape(ns.get("native_public", ""))}</code>；'
        + f'native 内部存储 <code>{html.escape(ns.get("native_internal", ""))}</code>；'
        + f'HF/ms-swift <code>{html.escape(ns.get("hf_msswift", ""))}</code> ⇒ 不能跨后端共用同一串目标。'
        + '</span>')

def _target_regex_html(data):
    m = ((data.get("meta") or {}).get("lora_target_backends") or {}).get("msswift") or {}
    t, g = m.get("target_regex_mechanism") or {}, m.get("guard_fix") or {}
    if not t:
        return "—"
    return (f'<b>字段</b>：<code>{html.escape(t.get("field", ""))}</code>。<br>'
            f'<b>形状链</b>：{html.escape(t.get("shape_chain", ""))}<br>'
            f'<b>修法</b>：{html.escape(t.get("fix", ""))}（{html.escape(t.get("constraints", ""))}；提交 <code>{html.escape(t.get("commit", ""))}</code>）<br>'
            f'<b>守卫修正</b>：<code>{html.escape(g.get("function", ""))}</code> —— {html.escape(g.get("str_branch", ""))}；'
            f'{html.escape(g.get("list_branch", ""))}；{html.escape(g.get("fail_closed", ""))}；{html.escape(g.get("retained", ""))}'
            f'（提交 <code>{html.escape(g.get("commit", ""))}</code>）')


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
        "{{BOUNDARY}}": build_boundary(data),
        "{{CALIBER}}": build_caliber(data),
        "{{VERDICTS}}": build_verdicts(data),
        "{{GENERATOR}}": (meta.get("authority") or {}).get("generator", ""),
        "{{DATA_PATH}}": (meta.get("authority") or {}).get("data_path", ""),
        "{{COVERAGE_WARNING}}": build_coverage_warning(data),
        "{{DOC_SCOPE}}": meta.get("scope", ""),
        "{{VENUE}}": build_venue(data),
        "{{CODE_VERSION}}": build_code_version(data),
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
