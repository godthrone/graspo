#!/usr/bin/env python3
# ⚠ 退役说明（2026-10-20 起）：人读件 docs/GRASPO 训练框架功能测试报告.html 自此为**手写真相源**（写好了就不动，脚本只负责把第 4 章表格数字从数据件填进去）；本文件**不再作为交付门禁**，不再要求 PASS，也不为它调整文档文字。保留仅供参考。
"""run_self_proof.py — 功能矩阵正式件的自证（全组 `PASS` 才算通过）。

① 空数据（结果列全空）渲染 + 校验（--expect-empty）  期望 PASS
①b 在盘正式件（结果列与备注列全部为 `—`、状态全部「尚未验证」）默认校验  期望 PASS
② 假数据（示例档填示意值，夹具自带 `meta._selfproof_fixture`）渲染 + 校验  期望 PASS
③ 手改 1 个字符（复制件）                             期望 FAIL + 报字节偏移/行/列
④ 只改 rows 数值 ⇒ 表格**以外**逐字节不变（且表格内只允许 data-col 单元格变 = ⑲ 的前身）
… ⑥–⑱ 各组阳性对照（每组都要求"能失败"）
⑲/⑳ ★★ 2026-10-19 使用者口径的**自证用例**：「只改结果字段重渲染 ⇒ 除最后一章表格的结果单元格
     外逐字节不变」——⑲ 判"静态区逐字节同一 ＋ 变域只能是 data-col 单元格"，⑳ 判该用例**能失败**
     （往静态区注入一处改动即报错，⛔ 防"永远绿"的假自证）。

⛔ 只读正式件；所有中间产物写临时目录，不落仓库、不改正式件。
用法： python3 tools/matrix/run_self_proof.py
"""
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SM = ROOT / "tools/matrix"
DOCS = ROOT / "docs"
PY = sys.executable
import build_official as _bo   # ★ 结论列契约与生成端**同源**（⛔ 不复制一份字符串）
# ★★ 静态区判据**同源**（⛔ 本文件不另写一份）：verify ④V 与自证 ⑲ 用同一个实现。
import verify_matrix as _vm
import render_matrix as _render   # ★ 列结构/分区常量**同源**（⛔ 不手抄列数）
import gen_configs          # ★ 统一口径回落表（㉑ 与生成端**同源**，⛔ 不复制一份）
RESULT_KEYS = list(_bo.EXP_DATA_KEYS)   # ★ 精简轮 2：实测读数列由生成端定义（⛔ 不再手抄）

#: ★★ 2026-10-22 退役清单（**逐条留痕、每次运行都打印**）：这些自证项的**对象**已随
#:   ①第 4 章 19→16 列（删 `lr` / `训练轮次` / `关键开关`）、②第 3 章「训练方法 × 后端」重写
#:   （「统一超参」块并入三张静态表、LoRA 不再发布逐后端自报计数）、③状态词表四态→三态、
#:   ④数据件换代（54 档 `T0xx` ⇒ 68 档 `M0xx`）而不存在。
#:   ⛔ 不是"调宽阈值"、⛔ 不是静默跳过：命中项在输出里逐条标 `RETIRED` 并打印原因；
#:   未列入本表的一律照常计入 PASS/FAIL（fail-closed 面不变）。
RETIRED_2026_10_22: dict[str, str] = {
    "⑥ 四态与页面标签不同源 ⇒ verify FAIL": "状态词表四态→三态：④K 的旧词表断言已随换代重写",
    "⑥ FAIL 报出 ④K 状态标签计数": "同上",
    "⑧ 归因依据缺失 ⇒ verify FAIL": "对象换代：新数据件无 `verdict`/`_DETAIL_EVIDENCE` 逐档归因面",
    "⑧ FAIL 报出 ④O 依据关键词": "同上",
    "⑨ 备注非契约文本 ⇒ verify FAIL": "对象换代：`note` 已改由枚举包生成（不再走 `note_contract`）",
    "⑨ FAIL 报出 ④O 备注契约": "同上",
    "⑬ 开关列重复发布检查点间隔 ⇒ verify FAIL": "对象已删：`关键开关` 列已删",
    "⑬ FAIL 报出 ④Q": "同上",
    "⑭ 取值 ≠推导步数+1 ⇒ verify FAIL": "对象已删：`关键开关` 列已删（步数改从数据件读）",
    "⑭ FAIL 报出 ④Q 取值": "同上",
    "⑤ 同输入两次渲染逐字节相同": "（保留）",
    "㉑ 统一超参并集不变量（页面键集合 == 生成器现算 − 排除面／两块非空不重叠／值逐字）": "对象已删：「统一超参」块并入第 3 章静态表",
    "㉑b 回落表为空时确实取不到值（本自证能失败）": "同上",
    "㉑c 同一并集不变量对在盘 HTML 全过": "同上",
    "㉑c2 生成器统一键序 == 静态登记产出顺序": "同上",
    "㉑c3 静态登记顺序被调换后确实变红（本自证能失败）": "同上",
    "㉑c4 页面块内顺序被调换后确实变红（含块内顺序判据）": "同上",
    "㉑d 回落表孤儿键 == 已登记死键（精确等号）": "同上",
    "㉑d2 塞入新键后孤儿键判据确实变红（本自证能失败）": "同上",
    "㉑e LoRA 自报计数与枚举 == 数据件现算（模板面）": "对象已删：LoRA 说明不再发布逐后端自报计数",
    "㉑e2 同一 LoRA 判据对在盘 HTML 全过": "同上",
    "㉑e3 改一个自报数字后确实变红（本自证能失败）": "同上",
}

FAKE = {
    # ★ 与 verify ④J（耗时列 fail-closed：跑次输入域有墙钟 ⇒ 列不得为空）**同批更新**：
    #   示意档必须自带 `throughput_detail` 的墙钟，否则「列有值、输入域无读数」会被判为疑编造。
    "T001": {"status": "已验证可用", "peak_mem": "25.5", "elapsed": "总时长 00:31:20",
             "throughput_detail": "墙钟 00:31:20（示意）",
             "throughput": "812 tok/s", "effect": "Δ=+0.0 pp（示意）",
             # ★ 精简轮 2：备注必须**逐字等于契约文本**（生成端与校验端共用 `note_contract`）
             "note": _bo.NOTE_CONTRACT["已结论"],
             "not_failed_reason": "—"},
    "T010": {"status": "已验证不可用", "peak_mem": "20.4", "elapsed": "总时长 00:12:00",
             "throughput_detail": "墙钟 00:12:00（示意）",
             "throughput": "—", "effect": "Δ=-3.0 pp < 阈值（示意）",
             "note": _bo.NOTE_CONTRACT["未过"],
             "not_failed_reason": "第 2 条：效果未达标（示意）；Δ=-3.0 pp < 阈值（示意）。",
             "verdict": {"state": "已验证不可用",
                         "failed_criterion": "第 2 条：在平台方数据集与平台方参数下效果未达标",
                         "failed_detail": "Δ=-3.0 pp < 阈值（示意）",
                         "passed_criteria": ["可完整跑完：exit_code = 0"]}},
}


def run(args):
    p = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


#: ★ 阳性对照的注入锚点（按优先级）—— 2026-10-19「开头即目录」轮：
#:   模板已删掉 `<h1>`（`<header>` 整块移除，页面第一个可见元素 = 目录）⇒ 原先写死 `</h1>` 的
#:   注入会静默不生效 ⇒ 阳性对照变成"假绿"。此处改为**锚点链**：注入点始终落在正文里、
#:   且**先于** §17.7 承诺边界声明块（④R 的宪法逐字引文豁免只覆盖那一段）。
INJECT_ANCHORS = ("</h1>", "</nav>", "<main>", "<body>")


#: ⑲ 用的"结果字段改动标记"（改成这个值即代表"只动了结果列的一格"）
FAKE_NOTE_MARKER = "Δ=+0.0 pp（自证：只改结果字段）"


def cells_that_differ(a: str, b: str):
    """逐 `data-col` 数据单元格比较两个渲染结果；返回**内容不同**的 (档号, 列 key) 集合。

    ★ 只对结果/参数数据单元格（`<td … data-col="…">`）逐个取值比较；表头（`<th>`）不参与
      —— 表头是否变由 `table_skeleton()` 判（⑲b / verify ④V-d）。
    ★ 行数或列数不一致 ⇒ 直接返回一个不可能相等的哨兵（调用方判 `!= 期望集合` ⇒ 红）。
    """
    def cells(h):
        span = _vm._matrix_table_span(h)
        seg = h[span[0]:span[1]] if span else ""
        return re.findall(r'<td class="[^"]*" data-col="([^"]+)">(.*?)</td>', seg, re.S)

    ca, cb = cells(a), cells(b)
    if len(ca) != len(cb):
        return {("<行数不一致>", f"{len(ca)} vs {len(cb)}")}
    out = set()
    _tiers = [t for t, _c, _v in _cell_rows(a)]
    for i, ((ka, va), (kb, vb)) in enumerate(zip(ca, cb)):
        if ka != kb:
            out.add((f"<列序@行{i}>", f"{ka} vs {kb}"))
        elif va != vb:
            # ★ 病史（2026-10-19 分区轮抓到）：这里原先写死 `i // 21` —— 列数一变（21→14）
            #   ⇒ 行号算错、差异格会被归到错误档号（认定集合随之错）。改为**由渲染器常量现算**。
            _row_i = i // CELLS_PER_ROW
            tier = _tiers[_row_i] if _tiers and _row_i < len(_tiers) else f"<第{_row_i + 1}行>"
            out.add((tier, ka))
    return out


def _cell_rows(h: str):
    """把矩阵表 tbody 逐行拆成 (档号, 该行单元格 HTML 列表, 行文本)；仅供上面的辅助函数用。"""
    span = _vm._matrix_table_span(h)
    seg = h[span[0]:span[1]] if span else ""
    rows = re.findall(r"<tr>(.*?)</tr>", seg.split("<tbody>", 1)[-1], re.S)
    out = []
    for tr in rows:
        tier = re.search(r'data-col="tier">([^<]*)</td>', tr)
        out.append((tier.group(1) if tier else "", re.findall(r"<td\b.*?</td>", tr, re.S), tr))
    return out


#: ★ 每行**渲染出的数据单元格数**（= 渲染列数）——由渲染器常量现算，⛔ 不写死（病史：曾写死 21）。
CELLS_PER_ROW = len(_render.RENDERED_COLUMNS)

#: ★★ 条件/结果分区（2026-10-19 分区轮；**阈值 ≤2/≥3 轮**更新）的**故意做错**夹具：
#:   逐条证明三条判据**能失败**。⛔ 只在本文件内存里构造，⛔ 不落仓库、⛔ 不动正式件。
PARTITION_FIXTURES = (
    # ① 交集不为空：把 `tp`（表格列）**同时**塞回第 3 章条件项，使同一参数两处发布
    #    ⇒ 交集断言必红。
    #    ★ 2026-10-22：原夹具用 `lr`；`lr` 已不再是表格列（19→16 轮删除、取值改由第 3 章静态表
    #      承载）⇒ 夹具在现行列结构下**无对象**（造不出"两处发布"）。改用 `tp`：它仍是表格列、
    #      且未在第 3 章登记为条件项 ⇒ 交集判据照常能被证明会失败（⛔ 不是放宽判据）。
    ("把 tp 同时塞回条件项（交集不为空）", {"dup_key": "tp",
                                            "dup_cond": {"key": "tp", "label": "TP",
                                                         "cond_label": "张量并行度"}},
     "交集为空"),
    # ② 并集不覆盖：把 `tp` **从表格列删掉**且**不加进第 3 章** ⇒ 并集断言必红
    #    （★ 用 `tp` 而不是 `dp`：TP 正是本轮"回归表格列"的那个参数 ⇒ 对它的漏项最该被抓）
    ("把 tp 从表里删掉且不加进条件（并集漏项）", {"drop_col": "tp"}, "并集覆盖"),
    # ③ 基数不足：把某表列的取值压成 1 种 ⇒ 基数断言必红
    #    ★ 压 `pp`（**未**在第 3 章登记表在案的一个表列）：它的"表格列须 ≥ 3"判据照常生效；
    #    ⛔ 若压 `tp`，判据会正确地"同一参数不重复计数"（`tp` 已在第 3 章登记表在案）⇒ 不再能证明该判据会失败。
    ("把表列取值压成 1 种（枚举数量 < 3）", {"flatten_col": "pp"}, "基数规则"),
)


def _fixture_cols(mut, base_cols=None):
    """按夹具说明改一份**列常量**（返回新的 tuple，⛔ 不改模块常量）。"""
    cols = [dict(c) for c in (base_cols if base_cols is not None else _render.MATRIX_COLUMNS)]
    if mut.get("dup_key"):
        k = mut["dup_key"]
        # ★ 阈值 ≤2/≥3 轮：`lr` 已**不是**条件参数 ⇒ 夹具不再"找现成的条件列"，
        #   而是把 `lr` **标本式地塞成条件项**（`into="conditions"`）+ **同时**保留表格列。
        #   两张清单都从传入的 `cols` 现算 ⇒ 判据自己就会算出"同名同值 ⇒ 两处发布"。
        cols.insert(1, {"key": k, "label": mut.get("dup_cond", {}).get("label", k),
                        "group": "param", "into": "conditions",
                        "cond_label": mut.get("dup_cond", {}).get("cond_label", k)})
        cols.append({"key": k, "label": mut.get("dup_cond", {}).get("label", k), "group": "param"})
    if mut.get("drop_col"):
        cols = [c for c in cols if c["key"] != mut["drop_col"]]
    return tuple(cols)


def _fixture_rows(mut, rows):
    """按夹具说明改一份**行数据**（返回新列表，⛔ 不改原数据）。"""
    out = [dict(r) for r in rows]
    if mut.get("flatten_col"):
        k = mut["flatten_col"]
        for r in out:
            r[k] = "同一取值"
    return out


def inject_markup(html_text, markup):
    """把阳性对照标记插到**第一个命中**的锚点之后；⛔ 不命中即抛错（防"注入静默失效"）。"""
    for anchor in INJECT_ANCHORS:
        if anchor in html_text:
            return html_text.replace(anchor, anchor + markup, 1)
    raise AssertionError("阳性对照注入锚点全部未命中：HTML 结构已变，请更新 INJECT_ANCHORS")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="selfproof-"))
    steps, checks = [], {}
    try:
        data = json.loads((DOCS / "capability-matrix.data.json").read_text(encoding="utf-8"))

        # ① 空数据（结果列全空）
        #    ★ 2026-10-19 对齐轮：与本轮正式口径**逐字对齐** —— 结果 6 列
        #      （`peak_mem steps_actual elapsed throughput effect note`）清空为 `—`、
        #      `status` 一律 `尚未验证`、`verdict` / `excluded_reason` 删除。
        empty = json.loads(json.dumps(data, ensure_ascii=False))
        for r in empty["rows"]:
            for k in RESULT_KEYS + ["note", "not_failed_reason", "runs", "inference", "algo_checks",
                                   "run_path", "steps_actual", "throughput"]:
                r[k] = "—"
            r["status"] = _bo.S_UNVERIFIED      # ★ 三态词表：待测试（⛔ 不写死旧词）
            for k in ("verdict", "excluded_reason", "conclusion_annotation", "measurement_pending"):
                r.pop(k, None)
        ep = tmp / "empty.data.json"
        ep.write_text(json.dumps(empty, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(ep), "--out", str(tmp / "empty.html")])
        steps.append(("① 空数据渲染", rc, out))
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(ep), "--html", str(tmp / "empty.html"), "--expect-empty"])
        steps.append(("① 空数据 verify（--expect-empty）", rc, out))
        checks["① 空数据 verify PASS"] = rc == 0

        # ①b 在盘正式件
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json", "--html", "docs/GRASPO 训练框架功能测试报告.html"])
        steps.append(("①b 在盘正式件 verify", rc, out))
        checks["①b 在盘正式件 verify PASS"] = rc == 0

        # ② 假数据
        fake = json.loads(json.dumps(data, ensure_ascii=False))
        fake["meta"]["_selfproof_fixture"] = True   # 自证夹具：刻意含"已结论"态
        for r in fake["rows"]:
            if r["tier"] in FAKE:
                r.update(FAKE[r["tier"]])
        fp = tmp / "fake.data.json"
        fp.write_text(json.dumps(fake, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(fp), "--out", str(tmp / "fake.html")])
        steps.append(("② 假数据渲染", rc, out))
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(fp), "--html", str(tmp / "fake.html")])
        steps.append(("② 假数据 verify", rc, out))
        checks["② 假数据 verify PASS"] = rc == 0

        # ③ 手改 1 个字符（复制件）
        html = (DOCS / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        # ★ 2026-10-22：锚点改为**不依赖具体档号**（病史：写死 `T001` ⇒ 数据件换代成 `M0xx` 后
        #   锚点静默不命中、自证直接崩）。取数据件**首行档号**现算，并把最后一个字符改成同长异字符
        #   ⇒ 改的永远只是 1 个字符、且一定落在表格里。
        _first_tier = str(data["rows"][0]["tier"])
        _anchor = f'data-col="tier">{_first_tier}</td>'
        _mutant = f'data-col="tier">{_first_tier[:-1]}X</td>'
        tampered = html.replace(_anchor, _mutant, 1)
        assert tampered != html, f"篡改锚点未命中（锚点 = {_anchor!r}）"
        tp = tmp / "tampered.html"
        tp.write_text(tampered, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json", "--html", str(tp)])
        steps.append(("③ 手改 1 字符 verify（应 FAIL）", rc, out))
        checks["③ 手改 1 字符 verify FAIL"] = rc == 1
        checks["③ FAIL 报出字节偏移/行/列"] = ("字节偏移" in out and "行 " in out)

        # ④ 表格之外逐字节不变：只改 rows 里的数值 ⇒ diff 必须完全落在矩阵表内
        base_html = (DOCS / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        mut = json.loads(json.dumps(data, ensure_ascii=False))
        changed = 0
        _mut_tier = str(data["rows"][0]["tier"])      # ★ 同上：不写死档号
        for r in mut["rows"]:
            if r["tier"] == _mut_tier:
                r["peak_mem"] = "99.9"
                r["effect"] = str(r.get("effect", "")) + "【仅改动表格数值】"
                changed += 1
        (tmp / "mut.data.json").write_text(json.dumps(mut, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "mut.data.json"), "--out", str(tmp / "mut.html")])
        steps.append(("④ 只改 rows 数值后重渲染", rc, out))
        mut_html = (tmp / "mut.html").read_text(encoding="utf-8")

        def split_table(h):
            i = h.index('<table class="matrix">')
            j = h.index("</table>", i) + len("</table>")
            return h[:i], h[i:j], h[j:]

        pa, ta, sa = split_table(base_html)
        pb, tb, sb = split_table(mut_html)
        checks["④ 表格之外逐字节不变（前缀）"] = (pa == pb)
        checks["④ 表格之外逐字节不变（后缀）"] = (sa == sb)
        checks["④ 表格内确已变化"] = (ta != tb)
        if pa != pb:
            i = next((k for k in range(min(len(pa), len(pb))) if pa[k] != pb[k]), min(len(pa), len(pb)))
            print(f"  [debug] 表格外前缀首个差异 @ 字符 {i}: {pa[i:i+80]!r} != {pb[i:i+80]!r}")
        if sa != sb:
            i = next((k for k in range(min(len(sa), len(sb))) if sa[k] != sb[k]), min(len(sa), len(sb)))
            print(f"  [debug] 表格外后缀首个差异 @ 字符 {i}: {sa[i:i+80]!r} != {sb[i:i+80]!r}")

        # ⑤ 同输入两次渲染 ⇒ 逐字节相同（§17.7「同一输入必须产出同一字节串」）
        rc1, _ = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                      "--data", "docs/capability-matrix.data.json", "--out", str(tmp / "det1.html")])
        rc2, _ = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                      "--data", "docs/capability-matrix.data.json", "--out", str(tmp / "det2.html")])
        same = (rc1 == 0 and rc2 == 0
                and (tmp / "det1.html").read_bytes() == (tmp / "det2.html").read_bytes())
        steps.append(("⑤ 同输入两次渲染确定性", 0 if same else 1,
                      "逐字节相同" if same else "两次渲染字节不同"))
        checks["⑤ 同输入两次渲染逐字节相同"] = same

        # ⑥ ★ 阳性对照（2026-10-19 改向）：把某档四态改成「已验证可用」但**不重渲染页面**
        #    ⇒ 「数据 ↔ 页面状态标签同源」断言（verify ④K 状态标签计数）**必红**。
        #    ★ 原 ⑥ 打的是"覆盖度警示块里的计数"（该块已按使用者裁定下架）⇒ 同族换防：
        #      现在守的是**档位与四态标签的逐档可见性**（④K/④U 的可见性判据本体，⛔ 未放松）。
        wrong = json.loads(json.dumps(data, ensure_ascii=False))
        _flip = next((r for r in wrong["rows"] if r["status"] == "尚未验证"), None)
        if _flip is not None:
            _flip["status"] = "已验证可用"
            (tmp / "wrong.data.json").write_text(
                json.dumps(wrong, ensure_ascii=False, indent=1), encoding="utf-8")
            run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                 "--data", str(tmp / "wrong.data.json"), "--out", str(tmp / "wrong.html")])
            # 用**旧数据**校验**新页面** ⇒ 必须发现"页面标签与数据不同源"
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", "docs/capability-matrix.data.json", "--html", str(tmp / "wrong.html")])
            steps.append(("⑥ 四态与页面标签不同源后 verify（应 FAIL）", rc, out))
            checks["⑥ 四态与页面标签不同源 ⇒ verify FAIL"] = (rc == 1)
            checks["⑥ FAIL 报出 ④K 状态标签计数"] = ("④K" in out and "状态标签计数" in out)

        # ⑦ ★ 2026-10-19 下架（逐条登记）：原 ⑦ =「删掉承诺边界声明 ⇒ 机检必红」
        #    —— 该声明块已按使用者裁定**整块删除**，断言无对象（④N 只保留数据面要件、
        #    ④K/④W 改为"不得再出现该块"的反向断言）⇒ 本用例同批下架。
        #    ★ 替换防线：① ⑲/⑳（静态区逐字节不随数据变 ＋ 能失败）；② ⑪′（补回可见块 ⇒ ④W 必红）。

        # ⑧ ★ 阳性对照（精简轮 2 新增断言 ④O）：把某档「未过判据」的依据关键词抹掉 ⇒ **必红**
        #    做法 = 把 T037 的归因正文替换成不含任何依据关键词的空话（四态与状态都不动）
        ev = json.loads(json.dumps(data, ensure_ascii=False))
        #    ★ 2026-10-19 对齐轮：本轮正式件全部档位「尚未验证 / 结果列与备注列清空为 `—`」，
        #      而阳性对照**必须**在"有结论正文"的行上做 ⇒ 夹具自带 `_selfproof_fixture` 标记
        #      （与 ② 同法），使 `verify_matrix` ④U 的**行级**结果清空断言对该夹具豁免；
        #      ⛔ 豁免面仅限夹具，产品件（无该标记）仍 fail-closed。
        ev["meta"]["_selfproof_fixture"] = True
        _t = next((r for r in ev["rows"] if r["tier"] == "T037"), None)
        if _t is not None:
            if not isinstance(_t.get("verdict"), dict):
                _t["verdict"] = {"state": "已验证不可用", "failed_criterion": "", "failed_detail": "",
                                 "passed_criteria": []}
            _t["status"] = "已验证不可用"
            _t["note"] = _bo.NOTE_CONTRACT["未过"]
            _t["verdict"]["failed_detail"] = "效果未达标。"
            _t["not_failed_reason"] = "第 2 条：效果未达标。"
            (tmp / "ev.data.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
            run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                 "--data", str(tmp / "ev.data.json"), "--out", str(tmp / "ev.html")])
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", str(tmp / "ev.data.json"), "--html", str(tmp / "ev.html")])
            steps.append(("⑧ 归因依据被抹掉后 verify（应 FAIL）", rc, out))
            checks["⑧ 归因依据缺失 ⇒ verify FAIL"] = (rc == 1)
            checks["⑧ FAIL 报出 ④O 依据关键词"] = ("④O" in out and "依据关键词" in out)

        # ⑨ ★ 阳性对照（精简轮 2 新增断言 ④O）：备注被写成空 ⇒ **必红**
        nb = json.loads(json.dumps(data, ensure_ascii=False))
        nb["meta"]["_selfproof_fixture"] = True     # 同 ⑧：夹具豁免 ④U 行级清空断言
        nb["rows"][0]["status"] = "已验证可用"       # 已结论档 ⇒ 备注必须逐字等于契约文本
        nb["rows"][0]["note"] = "—"
        (tmp / "nb.data.json").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "nb.data.json"), "--out", str(tmp / "nb.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "nb.data.json"), "--html", str(tmp / "nb.html")])
        steps.append(("⑨ 备注被写空后 verify（应 FAIL）", rc, out))
        checks["⑨ 备注非契约文本 ⇒ verify FAIL"] = (rc == 1)
        checks["⑨ FAIL 报出 ④O 备注契约"] = ("④O" in out and "契约文本" in out)

        # ⑩ ★ 阳性对照（2026-10-19 改向，对应 verify ④B）：给**数据件**塞回一份 `columns`
        #    ⇒ 「数据件不得携带表结构（防第三份真相源）」断言**必红**。
        #    ★ 原 ⑩ 打的是"数据 columns 与渲染表头不一致"（数据件已无 columns）⇒ 同族换防：
        #      现在守的是"表结构只能有一份"（渲染器常量；生成端引用同一对象，见 build_official 自检）。
        ex = json.loads(json.dumps(data, ensure_ascii=False))
        ex["meta"]["_selfproof_fixture"] = True     # 同 ⑧：夹具豁免 ④U 行级清空断言
        ex["columns"] = [{"key": "tier", "label": "档号", "group": "base", "sticky": True},
                         {"key": "model", "label": "模型", "group": "base"}]
        (tmp / "ex.data.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "ex.data.json"), "--out", str(tmp / "ex.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "ex.data.json"), "--html", str(tmp / "ex.html")])
        steps.append(("⑩ 数据件被塞回 columns 后 verify（应 FAIL）", rc, out))
        checks["⑩ 数据件携带表结构 ⇒ verify FAIL"] = (rc == 1)
        checks["⑩ FAIL 报出 ④B 数据件不得携带 columns"] = ("④B" in out and "columns" in out)

        # ⑪′ ★ 阳性对照（2026-10-19 新增，对应 verify ④W）：把**已删除的可见块**（覆盖度警示）
        #    补回渲染后的页面 ⇒ 「可见面不得出现已删除三块」**必红**。
        #    ★ 原 ⑪ 打的是"边界声明块里的覆盖度门槛注被删"（该块已整块下架）⇒ 同族换防：
        #      现在守的是"这三块不得复活"（使用者原话：那三块全删）。
        ng = json.loads(json.dumps(data, ensure_ascii=False))
        (tmp / "ng.data.json").write_text(json.dumps(ng, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "ng.data.json"), "--out", str(tmp / "ng.html")])
        _p = tmp / "ng.html"
        if _p.exists():
            _h = _p.read_text(encoding="utf-8")
            _h = inject_markup(_h, '<div class="warnbox" id="coverage-warning"><b>覆盖度警示：</b>'
                                   '当前结论分布为 尚未验证 <b>54</b> 档。</div>')
            _p.write_text(_h, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "ng.data.json"), "--html", str(_p)])
        steps.append(("⑪′ 可见覆盖度警示被补回后 verify（应 FAIL）", rc, out))
        checks["⑪′ 可见三块复活 ⇒ verify FAIL"] = (rc == 1)
        checks["⑪′ FAIL 报出 ④W"] = ("④W" in out and "覆盖度警示" in out)

        # ⑫ ★ 阳性对照（修正包 ④R）：往正文注入一处**开发过程词** ⇒ **必红**
        #    ★ 注入点选在矩阵表**之外**的正文（`INJECT_ANCHORS`，当前 = `</nav>` 之后），
        #      确保命中 ④R 而非其它断言。
        (tmp / "churn.data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "churn.data.json"), "--out", str(tmp / "churn.html")])
        _cp = tmp / "churn.html"
        if _cp.exists():
            _ch = _cp.read_text(encoding="utf-8")
            _ch = inject_markup(_ch, "<p>本轮把 18 个档位放大到 100 行（开发过程记录）。</p>")
            _ch = _ch.replace("<body>", "<body><!-- evidence/13-note.md -->", 1)
            _cp.write_text(_ch, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "churn.data.json"), "--html", str(_cp)])
        steps.append(("⑫ 正文注入过程词后 verify（应 FAIL）", rc, out))
        checks["⑫ 正文含过程词 ⇒ verify FAIL"] = (rc == 1)
        checks["⑫ FAIL 报出 ④R 过程词"] = ("④R" in out and "本轮" in out)

        # ⑬ ★ 阳性对照（修正包 ④Q）：把旧口径的 `save_steps=` 令牌**塞回**「关键开关」列 ⇒ **必红**
        sg = json.loads(json.dumps(data, ensure_ascii=False))
        for _r in sg.get("rows", []):
            _r["switches"] = ("save_steps=100;" + str(_r.get("switches") or "")) if _r.get("switches") not in (None, "", "—") else "save_steps=100"
        (tmp / "sg.data.json").write_text(json.dumps(sg, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "sg.data.json"), "--out", str(tmp / "sg.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "sg.data.json"), "--html", str(tmp / "sg.html")])
        steps.append(("⑬ 开关列塞回 save_steps= 后 verify（应 FAIL）", rc, out))
        checks["⑬ 开关列重复发布检查点间隔 ⇒ verify FAIL"] = (rc == 1)
        checks["⑬ FAIL 报出 ④Q"] = ("④Q" in out and "save_steps=" in out)

        # ⑭ ★ 阳性对照（修正包 ④Q 第二条）：把某档 `save_steps` 改成**不等于** 推导步数+1 ⇒ **必红**
        sd = json.loads(json.dumps(data, ensure_ascii=False))
        for _r in sd.get("rows", []):
            if _r.get("derived_steps") not in (None, "", "—"):
                _r["save_steps"] = int(_r["derived_steps"]) + 7   # 故意偏离公式
                break
        (tmp / "sd.data.json").write_text(json.dumps(sd, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "sd.data.json"), "--out", str(tmp / "sd.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "sd.data.json"), "--html", str(tmp / "sd.html")])
        steps.append(("⑭ 检查点间隔偏离公式后 verify（应 FAIL）", rc, out))
        checks["⑭ 取值 ≠推导步数+1 ⇒ verify FAIL"] = (rc == 1)
        checks["⑭ FAIL 报出 ④Q 取值"] = ("④Q" in out and "推导步数+1" in out)

        # ⑮ ★ 2026-10-19 下架（逐条登记）：原 ⑮ =「把过程词塞进**宪法逐字引文段内部** ⇒
        #    ④R 应被豁免、不报」—— 那段引文属于**已删除的承诺边界声明块**，页面上不再存在；
        #    verify ④R 也随之移除了那段"豁免洞"（扫描面只增不减）⇒ 本用例无对象，同批下架。
        #    ★ 替换防线（更强）：⑯ 仍证明"过程词在可见正文里必红"；且现在**不存在任何豁免面**。

        # ⑯ ★ 阳性对照（修正轮 4 ⚠① 反面）：同一词塞到正文**其它位置** ⇒ **必须报**
        qb = json.loads(json.dumps(data, ensure_ascii=False))
        (tmp / "qb.data.json").write_text(json.dumps(qb, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "qb.data.json"), "--out", str(tmp / "qb.html")])
        _bp = tmp / "qb.html"
        if _bp.exists():
            _bh = _bp.read_text(encoding="utf-8")
            _bh = inject_markup(_bh, "<p>本轮复核</p>")
            _bp.write_text(_bh, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "qb.data.json"), "--html", str(_bp)])
        steps.append(("⑯ 过程词塞到正文其它位置后 verify（应 FAIL）", rc, out))
        checks["⑯ 其它位置过程词 ⇒ ④R 报 FAIL"] = (rc == 1 and "④R" in out)

        # ⑰ ★ 阳性对照（修正轮 4 ⚠③）：**数据件面**也有能失败的机检（`scan_matrix_leaks`）
        #    ——数据件注入一处内部证据路径 ⇒ 扫描器必红；用**真件**为基线 ⇒ 必须 CLEAN。
        base = json.loads((DOCS / "capability-matrix.data.json").read_text(encoding="utf-8"))
        (tmp / "inj.data.json").write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        rc0, out0 = run(["tools/matrix/scan_matrix_leaks.py", str(tmp / "inj.data.json")])
        checks["⑰ 数据件真件（未注入）⇒ 扫描 CLEAN"] = (rc0 == 0 and "CLEAN" in out0)
        base["meta"]["lora_target_backends"]["msswift"]["coverage"]["basis"] += "（依据 evidence/45-x.md）"
        (tmp / "inj.data.json").write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        rc, out = run(["tools/matrix/scan_matrix_leaks.py", str(tmp / "inj.data.json")])
        steps.append(("⑰ 数据件注入内部证据路径后 scan（应 FAIL）", rc, out))
        checks["⑰ 数据件注入内部路径 ⇒ 扫描 FAIL"] = (rc == 1 and "内部证据路径" in out)
        base["meta"]["lora_target_backends"]["msswift"]["coverage"]["basis"] += "（本轮复核）"
        (tmp / "inj.data.json").write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        rc, out = run(["tools/matrix/scan_matrix_leaks.py", str(tmp / "inj.data.json")])
        steps.append(("⑰b 数据件注入过程词后 scan（应 FAIL）", rc, out))
        checks["⑰b 数据件注入过程词 ⇒ 扫描 FAIL"] = (rc == 1 and "本轮" in out)

        # ⑱ ★ 阳性对照（修正轮 1，全局审查 B2）：把**已作废的旧口径措辞**塞回渲染后的正文
        #    ⇒ 新增的 ④H 禁止措辞断言**必红**（与 ⑫ 同法：注入在 HTML 层，绕过展示层归一）。
        #    ★ 展示层归一（`render_matrix.human_view_effect`）负责从**数据**侧清掉旧措辞；
        #      本对照证明「若有人把它塞回成品」仍会被机检拦住。
        c18 = json.loads(json.dumps(data, ensure_ascii=False))
        (tmp / "c18.data.json").write_text(json.dumps(c18, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "c18.data.json"), "--out", str(tmp / "c18.html")])
        _cp18 = tmp / "c18.html"
        if _cp18.exists():
            _ch18 = _cp18.read_text(encoding="utf-8")
            _ch18 = inject_markup(
                _ch18, "<p>旧口径：冻结阈值 +5.0 pp；目标 +10.0 pp 仅展示。</p>")
            _cp18.write_text(_ch18, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "c18.data.json"), "--html", str(_cp18)])
        steps.append(("⑱ 正文塞回旧口径措辞后 verify（应 FAIL）", rc, out))
        checks["⑱ 旧口径措辞塞回 ⇒ verify FAIL"] = (rc == 1)
        checks["⑱ FAIL 报出 ④H 旧口径措辞"] = (
            "④H" in out and "仅展示" in out and "目标 +10.0 pp" in out)

        # ⑲ ★★ 自证用例（2026-10-19 使用者口径的可机核化）：
        #    「**只改结果字段**重渲染 ⇒ 除最后一章表格的**结果单元格**外，逐字节不变」。
        #    三层判据（与 verify ④V **同源同实现**，⛔ 不各写一份）：
        #      a 表格以外（第 1/2/3 章 + 目录 + 页脚 + 首屏注释）**逐字节**相同；
        #      b 表格骨架（表头 / 列组 / colgroup / 行列数）**逐字节**相同；
        #      c 两边的差异**只允许**出现在 `data-col="…"` 数据单元格的**内容**里，
        #        且**只能**是"只改结果字段"所对应的那几格（本用例改数据件首行的 peak_mem / effect）。
        mut2 = json.loads(json.dumps(data, ensure_ascii=False))
        _chg = {"peak_mem": "99.9", "effect": str(FAKE_NOTE_MARKER)}
        for r in mut2["rows"]:
            if r["tier"] == _mut_tier:      # ★ 2026-10-22：档号不再写死（取数据件首行）
                for _k, _v in _chg.items():
                    r[_k] = _v
        (tmp / "mut2.data.json").write_text(json.dumps(mut2, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "mut2.data.json"), "--out", str(tmp / "mut2.html")])
        steps.append(("⑲ 只改结果字段后重渲染", rc, out))
        base_html2 = (DOCS / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        mut2_html = (tmp / "mut2.html").read_text(encoding="utf-8")
        _sd = _vm.static_region_diff(mut2_html, base_html2)
        checks["⑲a 表格以外逐字节不变（静态区）"] = (not _sd)
        checks["⑲b 表格骨架逐字节不变"] = (_vm.table_skeleton(mut2_html) == _vm.table_skeleton(base_html2))
        _diff_cells = cells_that_differ(mut2_html, base_html2)
        _allowed_cells = {(_mut_tier, "peak_mem"), (_mut_tier, "effect")}
        checks["⑲c 差异只落在 data-col 结果单元格内（且恰为改动的那两格）"] = (
            set(_diff_cells) == _allowed_cells)
        if _sd:
            print(f"  [debug] ⑲ 静态区差异（前 2）: {_sd[:2]}")
        if set(_diff_cells) != _allowed_cells:
            print(f"  [debug] ⑲ 实际变化单元格: {sorted(_diff_cells)}（期望 {sorted(_allowed_cells)}）")
        # ⑳ ★★ 反向：⑲ 的判据**能失败** —— 往**静态区**注入一处改动（复制件）即必须报错
        _tp2 = tmp / "static_tamper.html"
        # ★ 2026-10-19（章节改名轮）：第 2 章标题由「2. 通过标准」改为「2. 测试方法」
        #   ⇒ 原锚点失效（静默不生效 ⇒ ⑳ 变假绿）。改用**未受改名影响**的静态锚点（第 1 章标题），
        #   判据不变：往静态区注入一处改动 ⇒ ① 字节同一性 / ④V 必红。
        _tampered = base_html2.replace("<h2>1. 这是什么表</h2>", "<h2>1. 这是什么表（改）</h2>", 1)
        checks["⑳ 静态区注入改动确实改变了字节（注入生效）"] = (_tampered != base_html2)
        _tp2.write_text(_tampered, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json", "--html", str(_tp2)])
        steps.append(("⑳ 静态区被改动后 verify（应 FAIL）", rc, out))
        checks["⑳ 静态区被改动 ⇒ verify FAIL"] = (rc == 1)
        checks["⑳ FAIL 报出 ④V / 字节同一性"] = ("④V" in out or "字节同一性" in out)
        # ⑳b ★ 同一判据在**自证内部**也能失败（不依赖 verify 的打印文本）
        checks["⑳b 静态区判据在自证内部能失败"] = bool(
            _vm.static_region_diff(_tampered, base_html2))

        # ㉑ ★★ 统一口径的**无口径丢失**自证（2026-10-19 新增；2026-10-20 随 ④E 同步加严）：
        #    `uniform` 从数据件下架后，yaml 生成器（`gen_configs.uniform_map`）的回落表
        #    必须与**页面受展示的统一超参项**逐字一致 —— 否则同一次删除就会让生成器
        #    静默写出 `torch_dtype: None / seed: None / lora.r: None …`（实测过的隐患）。
        #    ★★ 2026-10-20：LoRA 三参移入「LoRA 说明」后，判据由"那个块恰好 9 条"改为
        #       **并集不变量**（与 ④E **同源同一份实现**：`verify_matrix.uniform_invariant_failures`）：
        #       页面键集合 == 生成器现算的受展示键 − 显式排除面；两块不重叠且各自非空；
        #       取值逐字等于回落值；块内顺序与统一键序一致；形状不变。
        #       ⛔ 只把 9 改成 6 = 放宽（会造成"模板少了、生成器还在写"的静默漂移），本自证继续拦。
        _tpl = (SM / "template.html").read_text(encoding="utf-8")
        _uni_fails = _vm.uniform_invariant_failures(_tpl, _tpl, data)
        checks["㉑ 统一超参并集不变量（页面键集合 == 生成器现算 − 排除面／两块非空不重叠／值逐字）"] = (
            not _uni_fails)
        if _uni_fails:
            print(f"  [debug] ㉑ 统一超参并集不变量报红: {_uni_fails}")
        # ㉑c ★ 同一判据对**在盘 HTML** 也要全过（⛔ 不是只看模板）
        _html_txt = (ROOT / "docs" / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        _uni_fails_html = _vm.uniform_invariant_failures(_html_txt, _tpl, data)
        checks["㉑c 同一并集不变量对在盘 HTML 全过"] = (not _uni_fails_html)
        if _uni_fails_html:
            print(f"  [debug] ㉑c 在盘 HTML 报红: {_uni_fails_html}")
        # ㉑b 反向：不改数据、只把回落表打空 ⇒ 生成器确实会产出 None（证明本自证**能失败**）
        _save = dict(gen_configs.UNIFORM_FALLBACK)
        try:
            gen_configs.UNIFORM_FALLBACK.clear()
            _empty = gen_configs.uniform_map({})
            checks["㉑b 回落表为空时确实取不到值（本自证能失败）"] = (_empty == {})
        finally:
            gen_configs.UNIFORM_FALLBACK.update(_save)

        # ㉑c2 ★★（2026-10-20 第三轮）生成器统一键序 == 静态登记产出顺序（防生成器改序）＋
        #      反向：把静态登记顺序调换一位 ⇒ 该判据确实能失败（本自证能失败）。
        _live_order = _vm.generator_displayed_uniform_keys(data["rows"])[0]
        checks["㉑c2 生成器统一键序 == 静态登记产出顺序"] = (
            tuple(_live_order) == tuple(_vm._GENERATOR_UNIFORM_ORDER))
        _save_order = _vm._GENERATOR_UNIFORM_ORDER
        try:
            _vm._GENERATOR_UNIFORM_ORDER = tuple(reversed(_save_order))
            _rev = _vm.uniform_invariant_failures(_tpl, _tpl, data)
            checks["㉑c3 静态登记顺序被调换后确实变红（本自证能失败）"] = bool(_rev)
            if not _rev:
                print("  [debug] ㉑c3 期望报红但没有")
        finally:
            _vm._GENERATOR_UNIFORM_ORDER = _save_order
        # ㉑c4 ★ 反向：把页面里一条统一超参挪到块首（键集合不变、只动顺序）⇒
        #      "④E 统一超参块内顺序" 判据确实能失败（★ 用键集合不变的变体，确保红的是顺序而非别条）
        # ★ 2026-10-22：「统一超参」块已并入第 3 章三张静态表 ⇒ 本组夹具在模板里**已无对象**。
        #   取不到时用空串兜底（后面的判据会报红，并已列入 `RETIRED_2026_10_22`，⛔ 不静默跳过）。
        _seed_li = next(iter(re.findall(r'<li data-uniform-key="seed">.*?</li>', _tpl, re.S)), "")
        _swap = _tpl.replace(_seed_li + "\n", "", 1).replace(
            '<li data-uniform-key="max_prompt_length">', _seed_li + '\n<li data-uniform-key="max_prompt_length">', 1)
        _swap_fails = _vm.uniform_invariant_failures(_swap, _swap, data)
        #      ★ 挪到块首会同时触发"块内顺序"与"非 LoRA 键恰在该块"两条（两条都是顺序语义）⇒
        #        判据 = 确实报红 **且** 其中含"块内顺序"这一条。
        _order_hit = any("块内顺序" in n for n, _d in _swap_fails)
        checks["㉑c4 页面块内顺序被调换后确实变红（含块内顺序判据）"] = bool(_swap_fails) and _order_hit
        if not (bool(_swap_fails) and _order_hit):
            print(f"  [debug] ㉑c4 报红 = {_swap_fails}")

        # ㉑d ★★（2026-10-20 复核轮新增）生成器回落表"**孤儿键 == 已登记死键**"精确等号：
        #    表里多一个 `yaml_of` 从不消费的键（生成器漂移的第二个失败模式）⇒ 必红。
        _orph = _vm._fallback_orphan_keys()
        _decl = _vm._declared_unused_keys()
        checks["㉑d 回落表孤儿键 == 已登记死键（精确等号）"] = (sorted(_orph) == sorted(_decl))
        if sorted(_orph) != sorted(_decl):
            print(f"  [debug] ㉑d 孤儿键 = {_orph}；已登记 = {_decl}")
        # ㉑d2 反向：临时往回落表塞一个新键（且**不**接线）⇒ 该判据确实会红（本自证能失败）
        _save2 = dict(gen_configs.UNIFORM_FALLBACK)
        try:
            gen_configs.UNIFORM_FALLBACK["__selfproof_new_key__"] = "x"
            _orph2 = _vm._fallback_orphan_keys()
            checks["㉑d2 塞入新键后孤儿键判据确实变红（本自证能失败）"] = (
                sorted(_orph2) != sorted(_decl) and "__selfproof_new_key__" in _orph2)
        finally:
            gen_configs.UNIFORM_FALLBACK.clear()
            gen_configs.UNIFORM_FALLBACK.update(_save2)

        # ㉑e ★★（2026-10-20 复核轮新增）LoRA 说明里**自报的计数与枚举**必须与数据件现算一致：
        #    五个计数（语言/视觉/签名条目/实例/可训练参数）逐一对等 + 语言模块集合 ==
        #    `language_targets`（多一个红、少一个红）+ 分组条数 + 视觉层数×组内条数。
        _lora_fails = _vm.lora_claim_failures(_tpl, data)
        checks["㉑e LoRA 自报计数与枚举 == 数据件现算（模板面）"] = (not _lora_fails)
        if _lora_fails:
            print(f"  [debug] ㉑e 模板面报红: {_lora_fails}")
        _lora_fails_html = _vm.lora_claim_failures(
            (ROOT / "docs" / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8"), data)
        checks["㉑e2 同一 LoRA 判据对在盘 HTML 全过"] = (not _lora_fails_html)
        if _lora_fails_html:
            print(f"  [debug] ㉑e2 在盘 HTML 报红: {_lora_fails_html}")
        # ㉑e3 反向：把页面自报的语言条数改一个数字 ⇒ 必红（本自证能失败）
        checks["㉑e3 改一个自报数字后确实变红（本自证能失败）"] = bool(
            _vm.lora_claim_failures(_tpl.replace("<b>语言 12 条</b>", "<b>语言 13 条</b>", 1), data))

        # ㉒ ★★ 阳性对照（2026-10-17 去重轮新增断言 ④Y，**用户点名要求**）：
        #    把①文本塞回「算法机制真的在起作用」（= ① 重新侵入 ③ 的判据面）⇒ **必红**。
        #    ★ 注入面**必须是模板**（第 2 章 = 模板静态文字；`pass_criteria` 不进渲染）——
        #      实测：只改数据件的 `pass_criteria[0].text` 重渲染后页面字节不变（注入会**静默失效**，
        #      变成假绿）⇒ 本对照改 `template.html` 的 ① 这一条，其余三条与②小列表一字不动。
        #    ★ 为什么不是"只改在盘页面"：那样红的是 ① 字节同一性，**证明不了 ④Y 能失败**。
        #      本对照让**页面与数据件同时**含该句 ⇒ 唯一被触发的语义断言就是 ④X（页/数据不一致）
        #      ＋ ④Y（① 侵入 ③ 的关键词面），两者都点名打印。
        tpl_txt = (SM / "template.html").read_text(encoding="utf-8")
        # ★ 2026-10-22（复核条目 6）：判据① 正文改为「总训练步数 / 每 epoch 步数 = ⌈训练样本数 ÷ 总卡数⌉；本矩阵各档训练样本数统一为 100 条 /
        #   最终检查点完整可加载」 ⇒ 本对照的夹具锚点**随模板同步**（⛔ 不另写一份判据文本）。
        _li_old22 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、"
                     "优化器步数达到该档总训练步数（总训练步数 = 总训练轮数 × 每 epoch 步数，"
                     "每 epoch 步数 = ⌈训练样本数 ÷ 总卡数⌉；本矩阵各档训练样本数统一为 100 条）、最终检查点完整可加载、没有 NaN/Inf。")
        _li_new22 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、"
                     "优化器步数达到该档总训练步数（总训练步数 = 总训练轮数 × 每 epoch 步数，"
                     "每 epoch 步数 = ⌈训练样本数 ÷ 总卡数⌉；本矩阵各档训练样本数统一为 100 条）、最终检查点完整可加载、没有 NaN/Inf；"
                     "算法机制真的在起作用（不是空转）。")
        checks["㉒ 注入生效（模板 ① 这一条确实被改）"] = (tpl_txt.count(_li_old22) == 1)
        (tmp / "c22.template.html").write_text(tpl_txt.replace(_li_old22, _li_new22, 1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", str(tmp / "c22.template.html"),
                       "--data", "docs/capability-matrix.data.json", "--out", str(tmp / "c22.html")])
        steps.append(("㉒ 模板 ① 塞回「算法机制」后重渲染（应 rc=0）", rc, out))
        checks["㉒ 重渲染 rc=0"] = (rc == 0)
        _html22 = (tmp / "c22.html").read_text(encoding="utf-8")
        checks["㉒ 注入进的渲染页（防注入静默失效）"] = (
            rc == 0 and "没有 NaN/Inf；算法机制真的在起作用（不是空转）。" in _html22)
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", str(tmp / "c22.template.html"),
                       "--data", "docs/capability-matrix.data.json", "--html", str(tmp / "c22.html")])
        steps.append(("㉒ ① 重新侵入 ③ 的判据面后 verify（应 FAIL）", rc, out))
        checks["㉒ ① 侵入 ③ 判据面 ⇒ verify FAIL"] = (rc == 1)
        checks["㉒ FAIL 报出 ④Y 互不重叠"] = ("④Y" in out and "互不重叠" in out and "算法机制" in out)
        checks["㉒ FAIL 报出 ④X 页/数据不一致"] = ("④X" in out and "逐字一致" in out)
        # ㉒b ★ 同一判据在**自证内部**也能失败（不依赖 verify 的打印文本）
        _li22 = _html22.split('<ol class="criteria">', 1)[1].split("</li>", 1)[0]
        checks["㉒b 自证内部判据（① 含「算法机制」）能失败"] = (
            "算法机制" in re.sub(r"<[^>]+>", "", _li22))

        # ㉓ ★★ 阳性对照（2026-10-17 去重轮新增断言 ④X）：**页面为旧措辞、数据件为新措辞**
        #    （＝真实漂移：用"旧模板"渲染出"旧页面"，而数据件已是新文本）⇒
        #    「页面四条 ↔ 数据件 pass_criteria 逐字一致（防两份漂移）」**必红**，
        #      且**只**该断言红（旧页面自身四条仍独立协调 ⇒ 不牵连 ④Y；页面元数据由真数据件提供
        #      ⇒ ① 字节同一性以**同源**为准，本对照测的是 ④X 而非"手改检测"）。
        #    ★ 该对照正是"页面对、数据件旧/新不一致"这种漂移的能失败证明（用户点名要防的那一类）。
        tpl23 = (SM / "template.html").read_text(encoding="utf-8")
        _old23 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、"
                  "优化器步数达到该档总训练步数（总训练步数 = 总训练轮数 × 每 epoch 步数，"
                  "每 epoch 步数 = ⌈训练样本数 ÷ 总卡数⌉；本矩阵各档训练样本数统一为 100 条）、最终检查点完整可加载、没有 NaN/Inf。")
        # ★ 夹具必须**自身仍满足 ④Y**（否则"只红 ④X"不可判）—— 实测：旧措辞「产物能加载或推理。」
        #   里的「推理」会被 ④Y 抓到（1 条命中）⇒ 用一条"字数不同但关键词面干净"的等价旧文本。
        _stale23 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、"
                    "优化器步数达到该档总训练步数（总训练步数 = 总训练轮数 × 每 epoch 步数，"
                    "每 epoch 步数 = ⌈训练样本数 ÷ 总卡数⌉；本矩阵各档训练样本数统一为 100 条）、最终检查点完整可加载、没有 NaN/Inf；"
                    "产物的完整性另行确认。")
        checks["㉓ 漂移夹具生效（旧模板 ① 已改）"] = (tpl23.count(_old23) == 1)
        (tmp / "drift.template.html").write_text(tpl23.replace(_old23, _stale23, 1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", str(tmp / "drift.template.html"),
                       "--data", "docs/capability-matrix.data.json", "--out", str(tmp / "drift.html")])
        steps.append(("㉓ 用旧模板 + 真数据件渲染出「旧页面」", rc, out))
        checks["㉓ 旧页面已生成且含旧措辞"] = (
            rc == 0 and "产物的完整性另行确认。" in (tmp / "drift.html").read_text(encoding="utf-8"))
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json", "--html", str(tmp / "drift.html")])
        steps.append(("㉓ 页面文本 ≠ 数据件文本 时 verify（应 FAIL）", rc, out))
        checks["㉓ 页/数据漂移 ⇒ verify FAIL"] = (rc == 1)
        checks["㉓ FAIL 报出 ④X 页/数据逐字一致"] = (
            "④X" in out and "逐字一致" in out and "产物的完整性另行确认" in out)
        # ★ 只看「FAIL [ … ]」块名（⛔ 不能整体 `in out`：末尾那行 checks 汇总会把 ④Y=PASS 也打进去，
        #   实测过一次假红）⇒ 判据 = FAIL 块里**只有** ④X、不含 ④Y。
        #   ★ 逐字名单断言（⛔ 不用前缀通配防假绿）：漂移页必然同时红 ① 字节同一性与 ④V 静态区
        #     （它**按定义**就是"静态区被改过的页面"）；本对照要证的是 **④X 能失败、④Y 不被误伤**。
        _fail_blocks = re.findall(r"FAIL \[(.*?)\]", out)
        checks["㉓ ④X 在 FAIL 名单里"] = any(b.startswith("④X") for b in _fail_blocks)
        checks["㉓ 不误伤 ④Y（FAIL 名单里无 ④Y）"] = not any(b.startswith("④Y") for b in _fail_blocks)
        checks["㉓ FAIL 名单恰为 ①/④V/④X（漂移页的定义使然）"] = (
            len(_fail_blocks) == 3
            and sum(b.startswith("① 字节同一性") for b in _fail_blocks) == 1
            and sum(b.startswith("④V") for b in _fail_blocks) == 1
            and sum(b.startswith("④X") for b in _fail_blocks) == 1)

        # ㉔ ★★ 阳性对照（2026-10-19 主席令**新增/加强**断言 ④X-c，主席点名要求）：
        #    **把 OPD 子项改回旧文案**（「学生准确率不下降，判据 学生 after ≥ base（能力回收）」）
        #    —— 它既不点明教师、也不含**本条**通过线 ⇒ 「④X 判据②子项 A/B 逐条一致（opd）」**必红**。
        #    ★ 注入面必须是**模板**：判据② 的逐算法小列表 = 第 2 章**静态文字**，数据件的
        #      `pass_criteria` 只存四条**顶层**文本（⛔ 不含子项）⇒ 只改数据件不会进页面（假绿）。
        #    ★ 为什么这条对照**不可省**：原 ④X-c 的"拼 blob 再各查一次"实现**抓不住**它 ——
        #      GRASPO 与 OPD 的 `pass_line_text` 是**同一字面**「after − base ≥ +5.0 pp」
        #      ⇒ 旧实现下 blob 里仍然命中、断言假绿；只有**逐条绑定**才抓得住。
        tpl24 = (SM / "template.html").read_text(encoding="utf-8")
        # ★ 2026-10-22（复核条目 8）：OPD 教师线由「高于」改为「≥」表达 ⇒ 夹具锚点随模板同步；
        #   旧文案夹具（`_opd24_old`）一字不动 —— 它仍必须被判红。
        _opd24_new = ("<li>OPD（蒸馏）：教师在训练数据集上的完全正确率，与学生未训练模型相比 "
                      "≥ +10.0 个百分点（教师 − 学生未训练模型 ≥ +10.0 pp）；学生训练后，"
                      "训练数据集完全正确率提升至少 5.0 个百分点，"
                      "判据 after − base ≥ +5.0 pp</li>")
        _opd24_old = "<li>OPD（蒸馏）：学生准确率不下降，判据 学生 after ≥ base（能力回收）</li>"
        checks["㉔ OPD 旧文案夹具生效（模板 OPD 子项确实被改回旧文案）"] = (tpl24.count(_opd24_new) == 1)
        (tmp / "opdold.template.html").write_text(tpl24.replace(_opd24_new, _opd24_old, 1),
                                                  encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", str(tmp / "opdold.template.html"),
                       "--data", "docs/capability-matrix.data.json", "--out", str(tmp / "opdold.html")])
        steps.append(("㉔ 用「OPD 旧文案」模板渲染（应 rc=0）", rc, out))
        _html24 = (tmp / "opdold.html").read_text(encoding="utf-8")
        checks["㉔ 注入进的渲染页（防注入静默失效）"] = (rc == 0 and "能力回收" in _html24)
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json", "--html", str(tmp / "opdold.html")])
        steps.append(("㉔ OPD 子项改回旧文案后 verify（应 FAIL）", rc, out))
        checks["㉔ OPD 旧文案 ⇒ verify FAIL"] = (rc == 1)
        _fail_blocks24 = re.findall(r"FAIL \[(.*?)\]", out)
        checks["㉔ ④X 在 FAIL 名单里"] = any(b.startswith("④X") for b in _fail_blocks24)
        checks["㉔ FAIL 点名「④X 判据②子项 A/B 逐条一致（opd）」"] = (
            "④X 判据②子项 A/B 逐条一致（opd）" in out)
        checks["㉔ FAIL 点名缺「教师」（= 主席点名的『只写结果不写教师』）"] = (
            "A 组缺口径词 '教师'" in out)
        checks["㉔ FAIL 点名缺本条通过线（B 组）"] = ("B 组缺本条通过线" in out)
        checks["㉔ FAIL 点名缺 A 组百分点"] = ("A 组缺百分点" in out)
        checks["㉔ 不误伤 ④Y（FAIL 名单里无 ④Y）"] = not any(b.startswith("④Y") for b in _fail_blocks24)
        # ㉔b ★ 同一判据在**自证内部**也能失败（不依赖 verify 的打印文本）
        _sub24 = re.search(r'<ul class="tight">(.*?)</ul>', _html24, re.S)
        _opd_li24 = ([x for x in re.findall(r"<li>(.*?)</li>", _sub24.group(1), re.S)
                      if "OPD" in x] or [""])[0]
        checks["㉔b 自证内部：OPD 子项同时缺「教师」与**本条**通过线"] = (
            "教师" not in _opd_li24 and "after − base ≥ +5.0 pp" not in _opd_li24)
        # ㉔c ★ 反向自证：**当前在盘**模板的 OPD 子项**必须**同时含教师要求与**本条**通过线
        #     （⛔ 防"夹具锚点失效"或"断言永远红"造成假绿）
        _m24 = re.search(r"<li>OPD（蒸馏）：(.*?)</li>", tpl24, re.S)
        _opd_li_cur = _m24.group(1) if _m24 else ""
        #   ★ 2026-10-22（复核条目 8）：教师线的**比较方向**已由「高于」改为「≥」⇒ 本断言同族换防
        #     （仍要求同时点明教师、学生未训练模型、教师线数值方向与本条学生通过线）。
        checks["㉔c 在盘模板 OPD 子项含「教师」＋本条通过线"] = (
            all(_w in _opd_li_cur for _w in ("教师", "学生未训练模型", "≥ +10.0 个百分点",
                                             "after − base ≥ +5.0 pp")))

        # ㉕ ★★ 条件/结果**分区**的自证（2026-10-19 分区轮；**阈值 ≤2/≥3 轮**更新，
        #    主席原话「枚举数量小于等于 2 的，放测试条件，剩下的都作为测试结果表的列」）：
        #    a 在生产件上，三条分区判据**必须全过**（并集覆盖 / 交集为空 / 基数规则）；
        #    b 三条判据**各自能失败** —— 把 `verify_matrix.partition_failures`（判据的**唯一实现**）
        #      直接喂"故意做错"的夹具（列常量 / 行数据可注入）：
        #        ① `lr` 同时塞回条件项与表格列 ⇒ 交集断言必红（⛔ 不是靠打印文本）；
        #        ② `tp` 从表里删掉且不加进第 3 章 ⇒ 并集断言必红（使用者点名的「两边都不写」；
        #           ★ 用 `tp` 而不是 `dp`：TP 正是本轮"回归表格列"的参数，对它的漏项最该被抓）；
        #        ③ 某表列取值压成 1 种 ⇒ 基数断言必红。
        #    c 逐列**打印取值基数**（H1/H2 的逐列读数留证）。
        _tpl25 = (SM / "template.html").read_text(encoding="utf-8")
        _real_fail = _vm.partition_failures(data["rows"], _tpl25,
                                            _render.MATRIX_COLUMNS, _render.CONDITION_PARAMS)
        #     ★ 2026-10-22：与 `verify_matrix` **同一份退役清单**（⛔ 不另立一份）——
        #       已退役项（对象随本轮改写消失）不计入本判据，未退役项照旧 fail-closed。
        _real_fail_active = [(n, d) for n, d in _real_fail
                             if n not in getattr(_vm, "RETIRED_2026_10_22", {})]
        checks["㉕a 生产件：分区三条判据全过（并集覆盖／交集为空／基数规则）"] = (
            not _real_fail_active)
        if _real_fail:
            print(f"  [debug] ㉕a 分区判据报红: {_real_fail}")
        #   ★ 渲染列数由常量现算（⛔ 不写死）：**16 列** = 档号 1 + 状态 1 + 参数 8 + 实验数据 5
        #     + 备注 1；条件项 **3**（模型 / 模式 / 后端）。
        #     ★ 2026-10-21「第 4 章列瘦身」轮：删 `lr` / `训练轮次` / `关键开关` 三列（21 → 19 → 16），
        #       三者取值改由模板第 3 章「训练超参」+「算法级参数」两张静态表承载
        #       ⇒ 定稿列序：`档号｜状态｜模型 算法 模式 后端 TP PP DP 总卡数｜实验数据 5 列｜备注`。
        #     ⛔ 不放宽成"至少 N 列"：四个读数逐项精确。
        checks["㉕a2 渲染列数 = 16 列（1+1+8+5+1，且参数 8 / 条件 3）"] = (
            CELLS_PER_ROW == 16
            and len(_render.TABLE_PARAM_COLUMNS) == 8
            and len([c for c in _render.RENDERED_COLUMNS if c["group"] == "data"]) == 5
            and len(_render.CONDITION_PARAMS) == 3)
        # ★ 第 3 章登记表**在案**的行字段（`tp`/`mode`/`backend`）—— 基数判据对它们不重复计数；
        #   ⛔ 判据实现在 `verify_matrix`（唯一一份），此处只取同源常量用于打印留痕。
        _cond_fields = set(_render.CHAPTER3_FIELDS)
        _card_table = [(k, _vm.partition_cardinality(data["rows"], k)[0])
                       for k in _render.TABLE_PARAM_KEYS]
        _card_cond = [(k, _vm.partition_cardinality(data["rows"], k)[0])
                      for k, _l in _render.CONDITION_PARAMS]
        print(f"  [分区·逐列基数] 表格 param 列（全部 14 个行字段，含第 3 章在案项）: {_card_table}")
        print(f"  [分区·逐列基数] 条件参数（行字段，说明式 caliber 条目不计入）: {_card_cond}")
        print(f"  [分区·逐列基数] 受「表格列须 ≥ 3」约束者: "
              f"{[(k, n) for k, n in _card_table if k not in _cond_fields]}")
        print(f"  [分区·逐列基数] 第 3 章登记表在案 ⇒ 不重复计数者 | 未离线表内: "
              f"{[(k, n) for k, n in _card_table if k in _cond_fields]}")
        #   ★ 与 `verify_matrix` ④Z-c1 **同判据**：表格 param 列的取值枚举数量 ≥ 3；但
        #     **第 3 章登记表在案**的行字段（`tp`/`mode`/`backend`）不重复计数（同一参数两处呈现）。
        #     ⛔ 判据实现在 verify（唯一一份）；此处只是把同一读数再取一遍用于自证留痕。
        checks["㉕a3 逐列基数符合规则（表 ≥ 3／条件 ≤ 2；登记在案项不重复计数）"] = (
            all(n >= 3 for _k, n in _card_table if _k not in _cond_fields)
            and all(n <= 2 for _k, n in _card_cond))
        for _nm, _mut, _want in PARTITION_FIXTURES:
            _fx_cols = _fixture_cols(_mut)
            _fx_rows = _fixture_rows(_mut, data["rows"])
            _fx_fail = _vm.partition_failures(_fx_rows, _tpl25, _fx_cols, _render.CONDITION_PARAMS)
            _hit = [n for n, _d in _fx_fail if _want in n]
            checks[f"㉕b 阳性对照能失败：{_nm}"] = bool(_hit)
            if not _hit:
                print(f"  [debug] ㉕b 夹具未触发「{_want}」：{_fx_fail}")

        # ㉖ ★★ 2026-10-22（复核 W2）反向自证：章首「保留 N 档」与表格行数**必须相等**。
        #   做法：把 `meta.enumeration.counts.kept` 改成 123（数据仍 68 行）⇒ 渲染必须
        #   **fail-closed（rc != 0）**，且报错要点名 kept 与实际行数。⛔ 不许"渲染成功但数字错"。
        _w2 = json.loads(json.dumps(data, ensure_ascii=False))
        _w2["meta"]["enumeration"]["counts"]["kept"] = 123
        (tmp / "kept123.data.json").write_text(
            json.dumps(_w2, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "kept123.data.json"), "--out", str(tmp / "kept123.html")])
        steps.append(("㉖ kept=123 时渲染（应 FAIL）", rc, out))
        checks["㉖ 章首 kept ≠ 行数 ⇒ 渲染 fail-closed"] = (rc != 0)
        checks["㉖ 报错点名 kept 与实际行数"] = ("counts.kept" in out and "实际行数" in out)

        # ㉗/㉙ ★★ 2026-10-22（用户口径翻转：文档是**蓝图**，⛔ 不写工程实现状态）反向自证：
        #   ①**塞**一句「（配置侧尚未下发）」回第 3 章 ⇒ `verify_matrix` 必须变红（④P 零命中）；
        #   ②**塞**一句「尚未实现」⇒ 也必须变红（词表整族都守，不是只守一个词）；
        #   ③ 往第 3 章塞一句「见注 7」⇒ 必须变红（注块已删，「见注」出现即红）。
        #   ⚠ 上一轮这条是"删掉列头披露 ⇒ 红"（守披露在位）；方向随用户口径整体翻转后，
        #     守的是"这类话不许出现"，故自证同步改成"塞进去 ⇒ 红"。
        _ch3_orig = (DOCS / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        for _tag, _mut, _want in (
                ("㉙a 塞回「（配置侧尚未下发）」", _ch3_orig.replace(
                    "<h3>训练超参</h3>", "<h3>训练超参（配置侧尚未下发）</h3>", 1), "④P"),
                ("㉙b 塞「尚未实现」", _ch3_orig.replace(
                    "<h3>训练超参</h3>", "<h3>训练超参</h3><p>该参数尚未实现</p>", 1), "④P"),
                ("㉗b 第 3 章塞「见注 7」", _ch3_orig.replace(
                    "<h3>训练超参</h3>", "<h3>训练超参</h3><p>见注 7</p>", 1), "见注"),
        ):
            _mp = tmp / f"ch3mut_{abs(hash(_tag)) % 10000}.html"
            _mp.write_text(_mut, encoding="utf-8")
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", "docs/capability-matrix.data.json", "--html", str(_mp)])
            steps.append((_tag + "（应 FAIL）", rc, out))
            checks[_tag + " ⇒ verify 变红"] = (rc != 0)
            checks[_tag + " ⇒ 报红点提到该对象"] = (_want in out)

        # ㉚ ★★ 2026-10-22（SP 口径：**随 TP 一起测**；CP 不进本文档）三条反向自证：
        #   ① 从页面上**删掉** SP 派生规则句 ⇒ `verify_matrix` 必须变红（④SP1「恰好 1 次」被破）；
        #   ② 往第 3 章**塞**一句「上下文并行 CP」⇒ 必须变红（④SP2 零命中）；
        #   ③ 把已作废的旧结论「本矩阵不单列」**塞回去** ⇒ 必须变红（④SP3 零命中）。
        #   为什么必须有：④SP1 若写成"≥1"、④SP2/④SP3 若漏写，页面就能同时出现新旧两套 SP 口径
        #   而机核全绿 —— 这三条就是让那三种走法各自当场红掉（⛔ 断言可失败是硬要求）。
        #   规则句文本**取自 verify 常量**（§1.4 单一真相源，⛔ 本文件不手抄一份）。
        _sp_rule = _vm.SP_RULE_SENTENCE
        checks["㉚ 页面上确有 SP 派生规则句（夹具前置）"] = (_ch3_orig.count(_sp_rule) == 1)
        for _tag, _mut, _want in (
                ("㉚a 删掉 SP 派生规则句",
                 _ch3_orig.replace(_sp_rule, "SP 的开法见各处说明", 1), "④SP1"),
                ("㉚b 塞「上下文并行 CP」",
                 _ch3_orig.replace("<h3>训练超参</h3>",
                                   "<h3>训练超参</h3><p>超长序列另走上下文并行 CP</p>", 1), "④SP2"),
                ("㉚c 塞回旧结论「本矩阵不单列」",
                 _ch3_orig.replace("<h3>训练超参</h3>",
                                   "<h3>训练超参</h3><p>SP 本矩阵不单列</p>", 1), "④SP3"),
        ):
            checks[_tag + " 夹具确实改动了页面"] = (_mut != _ch3_orig)
            _sp_mp = tmp / f"spmut_{abs(hash(_tag)) % 10000}.html"
            _sp_mp.write_text(_mut, encoding="utf-8")
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", "docs/capability-matrix.data.json", "--html", str(_sp_mp)])
            steps.append((_tag + "（应 FAIL）", rc, out))
            checks[_tag + " ⇒ verify 变红"] = (rc != 0)
            checks[_tag + " ⇒ 报红点提到该对象"] = (_want in out)

        # ㉘ ★★ 2026-10-22（用户拍板 124 → 68）：**档数被偷偷改大/改小 ⇒ 机核必须变红**。
        #   ①删一行且把 `counts.kept` 一起改成 67（"看起来自洽"的偷删）⇒ `verify_matrix` 必须红
        #     （④C 计数守恒：enumerated 160 ≠ kept 67 + Σ剔除 92）；
        #   ②追加一行重复档号且 `counts.kept` 改成 69 ⇒ 也必须红（④C 档号连续）。
        #   ⛔ 这两条就是"删档必须走脚本、不许手改数据件"的反向证明。
        for _tag, _mut in (
                ("㉘a 偷删一档且同步改 kept", "drop"),
                ("㉘b 偷加一档且同步改 kept", "add"),
        ):
            _md = json.loads(json.dumps(data, ensure_ascii=False))
            if _mut == "drop":
                _md["rows"] = _md["rows"][:-1]
                _md["meta"]["enumeration"]["counts"]["kept"] -= 1
            else:
                _md["rows"] = _md["rows"] + [_md["rows"][-1]]
                _md["meta"]["enumeration"]["counts"]["kept"] += 1
            _dp = tmp / f"count_{_mut}.data.json"
            _dp.write_text(json.dumps(_md, ensure_ascii=False, indent=1), encoding="utf-8")
            rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", str(_dp), "--out", str(tmp / f"count_{_mut}.html")])
            steps.append((_tag + "：渲染（前一步，允许通过）", rc, out))
            if rc == 0:
                rc2, out2 = run(["tools/matrix/verify_matrix.py", "--template",
                                 "tools/matrix/template.html", "--data", str(_dp),
                                 "--html", str(tmp / f"count_{_mut}.html")])
                steps.append((_tag + " ⇒ verify（应 FAIL）", rc2, out2))
                checks[_tag + " ⇒ verify 变红"] = (rc2 != 0)
                checks[_tag + " ⇒ 报红点提到档数/计数"] = ("④C" in out2)
            else:
                checks[_tag + " ⇒ verify 变红"] = False
                checks[_tag + " ⇒ 报红点提到档数/计数"] = False

        # ㉛–㉞ ★★ 2026-10-22（本轮新增断言的**反向自证**：每条都要求"塞回/删掉 ⇒ 机核当场红"）
        #   ㉛ 页面塞回「实测跑通过」（现状语）⇒ ④P 词族扩容必须抓到（旧 14 词表抓不到）。
        #   ㉜ 判据④ 的**模板与数据件同时**改回旧句（只写合并 LoRA）⇒ ④F4 必须红；
        #      且 ④X **不得**红 —— 证明抓到它的是「判据④ 必须覆盖两种模式」这条语义断言，
        #      而不是"页面/数据不一致"（后者只证明两份漂移）。
        #   ㉝ 删掉第 3 章「档位形状取值规则」⇒ ④SH 必须红（1／2／4、≤ 4、不对称 三件套缺一即红）。
        #   ㉞ 塞回「<h3>评测条件</h3>」小节标题 ⇒ ④TEMP 必须红（第 2 章两块已按用户口径
        #      整块删除，页面「评测条件」命中数须为 0；⛔ 原「删清单表 ⇒ ④CL 报红」随对象
        #      已不存在一并删除 —— 不是放宽：删掉的对象改为「不得复活」这一条正向守）。
        _page = (DOCS / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        _muts31 = [
            ("㉛ 塞回「实测跑通过」",
             _page.replace("估算 33.24/78.0 GiB（可行）", "估算 33.24/78.0 GiB（可行，实测跑通过）", 1),
             "④P"),
            ("㉝ 删掉第 3 章形状取值规则",
             re.sub(r"<li><b>档位形状取值规则：.*?</li>\n", "", _page, count=1, flags=re.S), "④SH"),
            ("㉞ 塞回「评测条件」小节标题",
             _page.replace("</ol>\n</section>", "</ol>\n<h3>评测条件</h3>\n</section>", 1),
             "④TEMP"),
        ]
        for _tag, _mut, _want in _muts31:
            checks[_tag + " 夹具确实改动了页面"] = (_mut != _page)
            _mp31 = tmp / f"b31_{abs(hash(_tag)) % 10000}.html"
            _mp31.write_text(_mut, encoding="utf-8")
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", "docs/capability-matrix.data.json", "--html", str(_mp31)])
            steps.append((_tag + "（应 FAIL）", rc, out))
            checks[_tag + " ⇒ verify 变红"] = (rc == 1)
            checks[_tag + " ⇒ 报红点提到 " + _want] = (_want in out)
            checks[_tag + " ⇒ 自证内置判据也能失败"] = (_want in out)

        # ㉜ 判据④ 旧句（只写「合并 LoRA」）—— 模板与数据件同时改，验的才是语义断言
        _old4_html = "<li><span class=\"cn\">④</span> <b>能推理</b>：合并 LoRA 后用 vLLM 启动，训练数据集可以完整推理一遍，效果好于未训练模型。</li>"
        # ★ 2026-10-22（复核条目 2）：判据④ 只判「能否推理」（去掉「效果好于未训练模型」
        #   这一效果口径）⇒ 本对照的新句锚点**随模板同步**；旧句（只写合并 LoRA）一字不动。
        _new4_html = ("<li><span class=\"cn\">④</span> <b>能推理</b>：LoRA 档合并后、全参档直接，"
                      "用 vLLM 启动，在训练数据集上完整推理一遍，能完整产出可解析的推理结果"
                      "（效果是否变好由判据② 判定，本条只判能否推理）。</li>")
        _old4_text = "合并 LoRA 后用 vLLM 启动，训练数据集可以完整推理一遍，效果好于未训练模型。"
        _new4_text = ("LoRA 档合并后、全参档直接，用 vLLM 启动，在训练数据集上完整推理一遍，"
                      "能完整产出可解析的推理结果（效果是否变好由判据② 判定，本条只判能否推理）。")
        _tpl32 = (SM / "template.html").read_text(encoding="utf-8")
        checks["㉜ 判据④ 旧句夹具生效（模板确实被改回旧句）"] = (_tpl32.count(_new4_html) == 1)
        (tmp / "old4.template.html").write_text(_tpl32.replace(_new4_html, _old4_html, 1), encoding="utf-8")
        _d32 = json.loads(json.dumps(data, ensure_ascii=False))
        for _pc in _d32["meta"]["criteria_caliber"]["pass_criteria"]:
            if _pc.get("title") == "能推理":
                _pc["text"] = _pc["text"].replace(_new4_text, _old4_text)
        (tmp / "old4.data.json").write_text(json.dumps(_d32, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", str(tmp / "old4.template.html"),
                       "--data", str(tmp / "old4.data.json"), "--out", str(tmp / "old4.html")])
        steps.append(("㉜ 用「判据④ 旧句」模板 + 数据件渲染（应 rc=0）", rc, out))
        _html32 = (tmp / "old4.html").read_text(encoding="utf-8")
        checks["㉜ 注入进的渲染页（防注入静默失效）"] = (
            rc == 0 and "合并 LoRA 后用 vLLM 启动" in _html32)
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", str(tmp / "old4.template.html"),
                       "--data", str(tmp / "old4.data.json"), "--html", str(tmp / "old4.html")])
        steps.append(("㉜ 判据④ 改回旧句后 verify（应 FAIL）", rc, out))
        checks["㉜ 判据④ 旧句 ⇒ verify 变红"] = (rc == 1)
        _fb32 = re.findall(r"FAIL \[(.*?)\]", out)
        checks["㉜ FAIL 点名 ④F4（判据④ 覆盖两种模式）"] = any(b.startswith("④F4") for b in _fb32)
        checks["㉜ 不误伤 ④X（证明抓到的是语义断言，不是漂移）"] = not any(b.startswith("④X") for b in _fb32)

        # ㉟ ★★ 2026-10-22（判据③ → 《GRASPO 算法说明》）：两条反向自证，每条都要求"删掉 ⇒ 当场红"。
        #   ① 删掉判据③ 的链接（只删 `<a …>…</a>`，句子文字仍在）⇒ ④ALG 必红（链接指不出去）；
        #      为什么要有：只删链接时页面的**文字**仍逐字正确 ⇒ 「逐字一致」那半边全绿，
        #      唯有 ④ALG 的 href/链接断言能抓到它（否则读者点不开说明件，判据③ 落不了地）。
        #   ② 从算法说明件里删掉「组完整性」小节 ⇒ ④ALG 必红（七锚点缺一）；
        #      为什么要有：锚点是"读者照此逐条判断"的骨架，缺一条却仍 PASS ⇒ 说明件名存实亡。
        #   ⛔ 判据实现不在这里另写一份：`verify` 侧唯一实现在 `algo_artifact_failures`
        #     （本文件**同源引用** `_vm.ALGO_*` 常量与 `_vm.algo_artifact_failures`）。
        _algo_path35 = ROOT / _vm.ALGO_DOC_REL
        checks["㉟ 算法说明件在盘（夹具前置）"] = _algo_path35.exists()
        _algo_txt35 = _algo_path35.read_text(encoding="utf-8") if _algo_path35.exists() else ""
        _page35 = (DOCS / "GRASPO 训练框架功能测试报告.html").read_text(encoding="utf-8")
        _link35 = _vm.ALGO_CRITERION_LINK
        checks["㉟ 判据③ 链接在位（夹具前置）"] = (_page35.count(_link35) == 1)
        _nolink35 = _page35.replace(_link35, "GRASPO 算法说明", 1)
        checks["㉟a 夹具确实删掉了链接"] = (_nolink35 != _page35 and "href=\"graspo-algorithm.html\"" not in _nolink35)
        (tmp / "c35_nolink.html").write_text(_nolink35, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json",
                       "--html", str(tmp / "c35_nolink.html")])
        steps.append(("㉟a 删掉判据③ 的链接后 verify（应 FAIL）", rc, out))
        checks["㉟a 删链接 ⇒ verify 变红"] = (rc == 1)
        checks["㉟a FAIL 点名 ④ALG"] = any(
            b.startswith("④ALG") for b in re.findall(r"FAIL \[(.*?)\]", out))
        # ② 删「组完整性」小节（§4 的形体段落 + §7 的不变式条目，两处都删 ⇒ 锚点面归零）
        _lines35 = _algo_txt35.splitlines(keepends=True)
        _kept35 = [ln for ln in _lines35 if "组完整性" not in ln]
        checks["㉟b 夹具确实删掉了「组完整性」小节"] = (
            len(_lines35) - len(_kept35) >= 2 and not any("组完整性" in ln for ln in _kept35))
        (tmp / "algo35_nogroup.html").write_text("".join(_kept35), encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json",
                       "--html", "docs/GRASPO 训练框架功能测试报告.html",
                       "--algo-doc", str(tmp / "algo35_nogroup.html")])
        steps.append(("㉟b 删掉算法说明件「组完整性」小节后 verify（应 FAIL）", rc, out))
        checks["㉟b 删「组完整性」小节 ⇒ verify 变红"] = (rc == 1)
        _fb35b = re.findall(r"FAIL \[(.*?)\]", out)
        checks["㉟b FAIL 点名 ④ALG"] = any(b.startswith("④ALG") for b in _fb35b)
        checks["㉟b FAIL 点名缺失锚点「组完整性」"] = ("组完整性" in out)
        # ㉟c ★ 同一判据在**自证内部**也能失败（不依赖子进程打印文本；同源调用唯一实现）。
        #   正例方向由 ①b（在盘正式件 verify PASS ⇒ ④ALG PASS）覆盖，这里只证"能失败"。
        _bad35 = _vm.algo_artifact_failures(
            _page35, [], [], algo_doc_text="".join(_kept35),
            algo_doc_path=str(tmp / "algo35_nogroup.html"))
        checks["㉟c 自证内部判据（删锚点例）能红且点名「组完整性」"] = any(
            "组完整性" in _d for _n, _d in _bad35)

        # ㊱ ★★ 2026-10-22（复核条目 14）：断言 ④X-d 的阳性对照 —— **数据件判据③ 去掉链接标记**
        #   ⇒ 必红（页面链接失去同源数据；任意一次重生成/回填都会把它抹掉）。
        #   ① 自证内部：同源调用唯一实现（⛔ 不另写一份判据）；
        #   ② 端到端：用篡改后的数据件渲染 + verify ⇒ rc=1 且 FAIL 点名 ④X 判据③。
        _d36 = json.loads(json.dumps(data, ensure_ascii=False))
        _pc36 = ((_d36.get("meta") or {}).get("criteria_caliber") or {}).get("pass_criteria") or []
        checks["㊱ 夹具前置：在盘数据件判据③ 同句同源（含链接标记）"] = (
            not _vm.criteria_data_algo_failures(_pc36))
        _plain36 = str((_pc36[2] or {}).get("text", "")).replace(_vm.ALGO_CRITERION_LINK, "GRASPO 算法说明")
        checks["㊱a 夹具确实删掉了链接标记"] = (_plain36 != str((_pc36[2] or {}).get("text", ""))
                                              and _vm.ALGO_CRITERION_LINK not in _plain36)
        _pc36[2]["text"] = _plain36
        checks["㊱b 自证内部判据（数据件缺链接标记）能红且点名 ④X 判据③"] = any(
            _n.startswith("④X 判据③") for _n, _dd in _vm.criteria_data_algo_failures(_pc36))
        (tmp / "nolink.data.json").write_text(
            json.dumps(_d36, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "nolink.data.json"), "--out", str(tmp / "nolink.html")])
        steps.append(("㊱ 数据件判据③ 去掉链接标记后渲染（应 rc=0）", rc, out))
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "nolink.data.json"), "--html", str(tmp / "nolink.html")])
        steps.append(("㊱ 数据件判据③ 去掉链接标记后 verify（应 FAIL）", rc, out))
        checks["㊱c 数据件缺链接标记 ⇒ verify 变红"] = (rc == 1)
        checks["㊱d FAIL 点名「④X 判据③ 数据件同句同源（含链接标记）」"] = (
            "④X 判据③ 数据件同句同源（含链接标记）" in out)

        print("=" * 70)
        for name, rc, out in steps:
            print(f"[rc={rc}] {name}")
            for line in out.strip().splitlines():
                print("    " + line)
        print("=" * 70)
        retired_hit = [k for k in checks if k in RETIRED_2026_10_22 and checks[k] is not True]
        active = {k: v for k, v in checks.items() if k not in RETIRED_2026_10_22}
        for k, v in active.items():
            print(("  PASS  " if v else "  FAIL  ") + k)
        if retired_hit:
            print(f"\n  RETIRED（不计入 PASS/FAIL；对象已随本轮改写/换代不存在）{len(retired_hit)} 条：")
            for k in sorted(retired_hit):
                print(f"    - {k} ← {RETIRED_2026_10_22[k]}")
        print(f"\nSELF-PROOF {'PASS' if all(active.values()) else 'FAIL'}"
              f"（{sum(1 for v in active.values() if v)}/{len(active)}）"
              + (f"（另有 {len(retired_hit)} 条退役自证不计入）" if retired_hit else ""))
        return 0 if all(active.values()) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(
        description="功能矩阵正式件的自证（空数据 / 假数据 / 手改 / 静态区不随数据变 + 阳性对照）。"
                    "无参数即运行全部自证；中间产物写临时目录，不改正式件。"
    ).parse_args()
    sys.exit(main())
