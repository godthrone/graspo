#!/usr/bin/env python3
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
import gen_configs          # ★ 统一口径回落表（㉑ 与生成端**同源**，⛔ 不复制一份）
RESULT_KEYS = list(_bo.EXP_DATA_KEYS)   # ★ 精简轮 2：实测读数列由生成端定义（⛔ 不再手抄）
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
            tier = _tiers[i // 21] if _tiers and i // 21 < len(_tiers) else f"<第{i // 21 + 1}行>"
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
            r["status"] = "尚未验证"
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
                       "--data", "docs/capability-matrix.data.json", "--html", "docs/capability-matrix.html"])
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
        html = (DOCS / "capability-matrix.html").read_text(encoding="utf-8")
        tampered = html.replace('data-col="tier">T001</td>', 'data-col="tier">T00l</td>', 1)
        assert tampered != html, "篡改锚点未命中"
        tp = tmp / "tampered.html"
        tp.write_text(tampered, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", "docs/capability-matrix.data.json", "--html", str(tp)])
        steps.append(("③ 手改 1 字符 verify（应 FAIL）", rc, out))
        checks["③ 手改 1 字符 verify FAIL"] = rc == 1
        checks["③ FAIL 报出字节偏移/行/列"] = ("字节偏移" in out and "行 " in out)

        # ④ 表格之外逐字节不变：只改 rows 里的数值 ⇒ diff 必须完全落在矩阵表内
        base_html = (DOCS / "capability-matrix.html").read_text(encoding="utf-8")
        mut = json.loads(json.dumps(data, ensure_ascii=False))
        changed = 0
        for r in mut["rows"]:
            if r["tier"] == "T028":
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
        #        且**只能**是"只改结果字段"所对应的那几格（本用例改 T028 的 peak_mem / effect）。
        mut2 = json.loads(json.dumps(data, ensure_ascii=False))
        _chg = {"peak_mem": "99.9", "effect": str(FAKE_NOTE_MARKER)}
        for r in mut2["rows"]:
            if r["tier"] == "T028":
                for _k, _v in _chg.items():
                    r[_k] = _v
        (tmp / "mut2.data.json").write_text(json.dumps(mut2, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "mut2.data.json"), "--out", str(tmp / "mut2.html")])
        steps.append(("⑲ 只改结果字段后重渲染", rc, out))
        base_html2 = (DOCS / "capability-matrix.html").read_text(encoding="utf-8")
        mut2_html = (tmp / "mut2.html").read_text(encoding="utf-8")
        _sd = _vm.static_region_diff(mut2_html, base_html2)
        checks["⑲a 表格以外逐字节不变（静态区）"] = (not _sd)
        checks["⑲b 表格骨架逐字节不变"] = (_vm.table_skeleton(mut2_html) == _vm.table_skeleton(base_html2))
        _diff_cells = cells_that_differ(mut2_html, base_html2)
        _allowed_cells = {("T028", "peak_mem"), ("T028", "effect")}
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

        # ㉑ ★★ 统一口径的**无口径丢失**自证（2026-10-19 新增）：
        #    `uniform` 从数据件下架后，yaml 生成器（`gen_configs.uniform_map`）的回落表
        #    必须与**人读件静态块里的那 9 项**逐字一致 —— 否则同一次删除就会让生成器
        #    静默写出 `torch_dtype: None / seed: None / lora.r: None …`（实测过的隐患）。
        _tpl = (SM / "template.html").read_text(encoding="utf-8")
        _static_uni = dict(re.findall(
            r'<li data-uniform-key="([^"]+)">[^<]*<code>([^<]*)</code></li>', _tpl))
        _fb = gen_configs.UNIFORM_FALLBACK
        _mismatch = [(k, _static_uni.get(k), _fb.get(k)) for k in _static_uni
                     if _fb.get(k) != _static_uni[k]]
        checks["㉑ yaml 生成器回落表 == 静态统一口径块（9 项逐字一致）"] = (
            len(_static_uni) == 9 and not _mismatch)
        if _mismatch:
            print(f"  [debug] ㉑ 不一致（键, 静态块, 回落表）: {_mismatch}")
        # ㉑b 反向：不改数据、只把回落表打空 ⇒ 生成器确实会产出 None（证明本自证**能失败**）
        _save = dict(gen_configs.UNIFORM_FALLBACK)
        try:
            gen_configs.UNIFORM_FALLBACK.clear()
            _empty = gen_configs.uniform_map({})
            checks["㉑b 回落表为空时确实取不到值（本自证能失败）"] = (_empty == {})
        finally:
            gen_configs.UNIFORM_FALLBACK.update(_save)

        # ㉒ ★★ 阳性对照（2026-10-17 去重轮新增断言 ④Y，**用户点名要求**）：
        #    把①文本塞回「算法机制真的在起作用」（= ① 重新侵入 ③ 的判据面）⇒ **必红**。
        #    ★ 注入面**必须是模板**（第 2 章 = 模板静态文字；`pass_criteria` 不进渲染）——
        #      实测：只改数据件的 `pass_criteria[0].text` 重渲染后页面字节不变（注入会**静默失效**，
        #      变成假绿）⇒ 本对照改 `template.html` 的 ① 这一条，其余三条与②小列表一字不动。
        #    ★ 为什么不是"只改在盘页面"：那样红的是 ① 字节同一性，**证明不了 ④Y 能失败**。
        #      本对照让**页面与数据件同时**含该句 ⇒ 唯一被触发的语义断言就是 ④X（页/数据不一致）
        #      ＋ ④Y（① 侵入 ③ 的关键词面），两者都点名打印。
        tpl_txt = (SM / "template.html").read_text(encoding="utf-8")
        _li_old22 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、优化器步数达标、"
                     "检查点完整、没有 NaN/Inf。")
        _li_new22 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、优化器步数达标、"
                     "检查点完整、没有 NaN/Inf；算法机制真的在起作用（不是空转）。")
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
        _old23 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、优化器步数达标、"
                  "检查点完整、没有 NaN/Inf。")
        # ★ 夹具必须**自身仍满足 ④Y**（否则"只红 ④X"不可判）—— 实测：旧措辞「产物能加载或推理。」
        #   里的「推理」会被 ④Y 抓到（1 条命中）⇒ 用一条"字数不同但关键词面干净"的等价旧文本。
        _stale23 = ("<b>能完整跑完</b>：训练正常结束、无中断、无报错、退出码 0、优化器步数达标、"
                    "检查点完整、没有 NaN/Inf；产物的完整性另行确认。")
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
        _opd24_new = ("<li>OPD（蒸馏）：教师在训练数据集上的完全正确率，高于学生未训练模型 10 个百分点，"
                      "学生训练后，训练数据集完全正确率提升至少 5.0 个百分点，"
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
        checks["㉔c 在盘模板 OPD 子项含「教师」＋本条通过线"] = (
            all(_w in _opd_li_cur for _w in ("教师", "高于学生未训练模型", "after − base ≥ +5.0 pp")))

        print("=" * 70)
        for name, rc, out in steps:
            print(f"[rc={rc}] {name}")
            for line in out.strip().splitlines():
                print("    " + line)
        print("=" * 70)
        for k, v in checks.items():
            print(("  PASS  " if v else "  FAIL  ") + k)
        print(f"\nSELF-PROOF {'PASS' if all(checks.values()) else 'FAIL'}（{sum(checks.values())}/{len(checks)}）")
        return 0 if all(checks.values()) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(
        description="功能矩阵正式件的自证（空数据 / 假数据 / 手改 / 静态区不随数据变 + 阳性对照）。"
                    "无参数即运行全部自证；中间产物写临时目录，不改正式件。"
    ).parse_args()
    sys.exit(main())
