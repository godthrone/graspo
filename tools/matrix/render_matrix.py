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
import argparse, html, json, re, sys
from pathlib import Path

PLACEHOLDER_TOKENS = ("{{DATA_REVISION}}", "{{SOURCE_BASENAME}}", "{{SOURCE_SHA256_SHORT}}",
                      "{{SOURCE_PATH}}", "{{SOURCE_SHA256}}", "{{CRITERIA_VERSION}}",
                      "{{ROW_COUNT}}", "{{MATRIX_TABLE}}", "{{UNIFORM_PARAMS}}", "{{DOC_DATE}}",
                      "{{DOC_TITLE}}", "{{GEN_BANNER}}",
                      "{{BOUNDARY}}", "{{CALIBER}}", "{{CRITERIA}}", "{{EVAL_INFO}}", "{{VERDICTS}}",
                      "{{GENERATOR}}", "{{DATA_PATH}}",
                      "{{COVERAGE_WARNING}}", "{{DOC_SCOPE}}", "{{VENUE}}", "{{CODE_VERSION}}")

#: ★★ 统一口径的**人读视图白名单**（精简轮 4，2026-10-17 主席令「只列复现需要的」）：
#:   机读件 `uniform.items` 一字不动（它是 yaml 生成器的输入，⛔ 不删）；此处只决定**渲染哪几项**。
#:   ★ 被排除项逐条理由：
#:     - `gpu_monitor.interval_sec`：巡检间隔不是复现参数（且含「GPU」字样，用户不关心测的哪张卡）；
#:     - `parallel_*` / `cp` / `sp` / `sequence_parallel_size`：**卡位 / 并行维度**，主席令不进人读件；
#:     - `criteria_id` / `criteria_version` / `eval_set_path`：判据与评测口径另在 §2 / §3 给出；
#:     - `train_source_form` / `use_vllm` / `save_checkpoint_every_epoch` / `lora_target_backend_note`：
#:       面向生成器的实现细节，不改变使用者复现所需的取值。
#:   ★ 校验端 `verify_matrix` **同源引用本常量**（§1.4 单一真相源，⛔ 不另抄一份）。
#:   ★ 呈现用短标签写在这里（版式层）；**取值一律取自数据**（⛔ 不写死任何数值）。
#:     数据里的 `label` 带内部字段名 / 内部说明（如「（恢复口径）」「（两后端同名字段，显式写出）」）
#:     ⇒ 按主席令「删括号补充说明 / 内部标识」换成使用者看得懂的名字。
UNIFORM_VIEW = (
    ("max_prompt_length", "上下文长度上限（token）"),
    ("torch_dtype", "权重精度"),
    ("gradient_checkpointing", "梯度检查点"),
    ("micro_batch_size", "单卡 micro batch size（native：micro_batch_size／ms-swift：per_device_train_batch_size）"),
    ("gradient_accumulation_micro_batches", "梯度累积 micro-batches"),
    ("lora_r", "LoRA rank r"),
    ("lora_alpha", "LoRA alpha"),
    ("lora_dropout", "LoRA dropout"),
    ("reward.kind", "GRASPO 奖励类型"),
)
UNIFORM_VIEW_KEYS = tuple(k for k, _l in UNIFORM_VIEW)

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
    cols = data["columns"]
    rows = data["rows"]
    groups = data["groups"]
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


def build_uniform_block(data):
    """渲染 §3「统一超参」块：只列**复现需要的**统一取值（白名单 `UNIFORM_VIEW`）。

    ★ 数据驱动：取值一律从 `uniform.items` 现取，⛔ 不在渲染层写死任何数值。
    ★ 输出 = 一个列表（⛔ 不再是三列表格，也⛔ 不再带 coverage / note 这些机读口径）。
    """
    uni = data.get("uniform") or {}
    by_key = {it.get("key"): it for it in (uni.get("items") or [])}
    out = ['<ul class="uniform">']
    for key, label in UNIFORM_VIEW:
        it = by_key.get(key)
        if not it:
            continue
        out.append(f'<li data-uniform-key="{html.escape(str(key))}">'
                   f'{html.escape(label)}：'
                   f'<code>{html.escape(str(it.get("value", "")))}</code></li>')
    out.append("</ul>")
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
        f'<b>覆盖度警示：</b>本矩阵共 <b>{_total}</b> 档，当前结论分布为 {cnt}。'
        f'尚有 <b>{len(unverified)}</b> 档没有结论（「尚未验证」）⇒ '
        '<b>不得被读作</b>「所列举的组合都可用」；「尚未验证」不计入可用档位数量。'
        f'<br><b>尚未验证的档位（{len(unverified)} 档）</b>：'
        + "、".join(f"<code>{html.escape(t)}</code>" for t in unverified) + "</div>")


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
    """★ 承诺边界声明（§17.7）：**两句话说完**（数据侧 `boundary_declaration.text` 已覆盖三个要件），
    外加一行**覆盖度门槛注**（未达门槛 ⇒ 整体承诺尚未生效）。⛔ 不再逐条展开 points / 旧注。
    """
    b = data["meta"].get("boundary_declaration", {})
    n_pending = count_statuses(data).get("尚未验证", 0)
    gate = (f'<li><b>覆盖度门槛：</b>当前尚有 {n_pending} 档为「尚未验证」⇒ 上述整体承诺尚未生效；'
            f'「尚未验证」不计入可用档位数量。</li>') if n_pending else ""
    return (f'<div class="hint"><b>承诺边界声明。</b>{html.escape(b.get("text", ""))}'
            f'<ul>{gate}</ul></div>')


_METHOD_NAMES = (("sft", "SFT（监督微调）"), ("graspo", "GRASPO（强化）"),
                 ("cpt", "CPT（继续预训练）"), ("opd", "OPD（蒸馏）"))


def _threshold_plain(key, v):
    """判据② 的逐算法通过线：**大白话 + 逐字数值/方向**（数值一律取自 `threshold.by_method`）。

    ★ epoch 不在本条重复（它属训练超参，第 3 节「实验条件」列一次）。
    """
    name = dict(_METHOD_NAMES)[key]
    line = html.escape(str(v.get("pass_line_text", "")))
    if v.get("also"):
        return f"{name}：{html.escape(str(v['also']))}、准确率不下降，判据 {line}"
    if key in ("sft", "graspo") and v.get("value") is not None:
        return (f"{name}：训练后工具调用准确率提升至少 {v.get('value')} 个百分点"
                f"，判据 {line}")
    return f"{name}：学生准确率不下降，判据 {line}"


def build_criteria(data):
    """★ §2「通过标准」= **一个列表**（有几条写几条，每条一句大白话）。

    ★ 判据文本的唯一真相源 = `meta.criteria_caliber.pass_criteria`（由生成端
      `build_official.criteria_caliber()` 产出）；判据② 的逐算法通过线由
      `threshold.by_method` **现算**（⛔ 不另写一份数值 ⇒ 数值与方向一字不错）。
    """
    c = data["meta"].get("criteria_caliber", {})
    th = c.get("threshold", {}) or {}
    bm = th.get("by_method") or {}
    items = []
    for it in (c.get("pass_criteria") or []):
        title = html.escape(str(it.get("title", "")))
        text = html.escape(str(it.get("text", "")))
        if it.get("from_threshold"):
            subs = "".join(f"<li>{_threshold_plain(k, bm[k])}</li>"
                           for k, _n in _METHOD_NAMES if k in bm)
            items.append(f'<li><b>{title}</b>：{text}<ul class="tight">{subs}</ul></li>')
        else:
            items.append(f'<li><b>{title}</b>：{text}</li>')
    rule = html.escape(str((c.get("conclusion_rule") or {}).get("pass_rule", "")))
    cid = html.escape(str(c.get("criteria_id", "")))
    return (f'<p class="rule">{rule}</p>'
            f'<ol class="criteria">{"".join(items)}</ol>'
            f'<p class="note">判据标识 <code>{cid}</code>。'
            f'{html.escape(str(th.get("status", "")))}</p>')


def build_eval_info(data):
    """★ §3 实验条件里的「评测集 / 评测方法」口径（§17.7：可复核到数据集标识 / 判据标识 /
    样本量 / 评测方法 / 版本锚点）。

    ★ 数据集标识**不照抄** `dataset.id`（该串内嵌 45 个档号，属实现细节）——改为
      「来源 + 版本锚点 + 样本量 + 评测方法 + 判据标识」，口径不缩水。
    """
    c = data["meta"].get("criteria_caliber", {})
    d = c.get("dataset", {}) or {}
    uni = {it.get("key"): it.get("value")
           for it in ((data.get("uniform") or {}).get("items") or [])}
    temp = html.escape(str(uni.get("temperature", "")))
    seed = html.escape(str(uni.get("seed", "")))
    parts = [
        (f"评测集：{html.escape(str(d.get('availability', '')))}。"
         "100 行档的评测集就是它自己训练用的那份 100 行文件"
         "（拿训练数据自测，只说明训练生效，不代表它在新数据上的表现）；"
         "其余档用平台方自有 702 行评测集。"),
        (f"样本量：{html.escape(str(d.get('sample_count', '')))}；"
         f"数据版本锚点：{html.escape(str(d.get('version_anchor_sha256', '')))}。"),
        (f"评测方法：{html.escape(str(d.get('method', '')))}；温度 <code>{temp}</code>；"
         f"随机种子 <code>{seed}</code>；一律用点估计 Δ 判定，不做假设检验。"),
        (f"判据标识 <code>{html.escape(str(c.get('criteria_id', '')))}</code>；"
         "达标阈值见第 2 节「通过标准」。"),
    ]
    return '<ul class="tight">' + "".join(f"<li>{x}</li>" for x in parts) + "</ul>"


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
        "{{CALIBER}}": build_criteria(data),
        "{{CRITERIA}}": build_criteria(data),
        "{{EVAL_INFO}}": build_eval_info(data),
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
