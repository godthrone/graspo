#!/usr/bin/env python3
"""run_self_proof.py — 功能矩阵正式件的三组自证。

① 空数据（结果列全空）渲染 + 校验（--expect-empty）  期望 PASS
①b 在盘正式件（含已回填档位）默认校验                  期望 PASS
② 假数据（示例档填示意值）渲染 + 校验                  期望 PASS
③ 手改 1 个字符（复制件）                             期望 FAIL + 报字节偏移/行/列

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


def main():
    tmp = Path(tempfile.mkdtemp(prefix="selfproof-"))
    steps, checks = [], {}
    try:
        data = json.loads((DOCS / "capability-matrix.data.json").read_text(encoding="utf-8"))

        # ① 空数据
        empty = json.loads(json.dumps(data, ensure_ascii=False))
        for r in empty["rows"]:
            for k in RESULT_KEYS + ["note", "not_failed_reason", "runs", "inference", "algo_checks",
                                   "run_path", "steps_actual", "throughput"]:
                r[k] = "—"
            r["status"] = "尚未验证"
            for k in ("verdict", "excluded_reason", "conclusion_annotation", "measurement_pending"):
                r.pop(k, None)
            # ★ 精简轮 2：备注必须逐字等于契约文本（与生成端 `note_contract` 同源）
            r["note"] = _bo.note_contract("尚未验证", raw_note="—")
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

        # ⑥ ★ 阳性对照：「尚未验证」档数写错 ⇒ 覆盖度断言（verify ④K）**必红**
        wrong = json.loads(json.dumps(data, ensure_ascii=False))
        _flip = next((r for r in wrong["rows"] if r["status"] == "尚未验证"), None)
        if _flip is not None:
            _flip["status"] = "已验证可用"          # ⇒ 警示里的计数与清单都会与 rows 不一致
            (tmp / "wrong.data.json").write_text(
                json.dumps(wrong, ensure_ascii=False, indent=1), encoding="utf-8")
            rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", str(tmp / "wrong.data.json"), "--out", str(tmp / "wrong.html")])
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", str(tmp / "wrong.data.json"), "--html", str(tmp / "wrong.html")])
            steps.append(("⑥ 覆盖度计数被改错后 verify（应 FAIL）", rc, out))
            checks["⑥ 覆盖度计数被改错 ⇒ verify FAIL"] = (rc == 1)
            checks["⑥ FAIL 报出覆盖度断言 ④K"] = ("④K" in out)

        # ⑦ ★ 阳性对照：删掉承诺边界声明 ⇒ 机检**必红**（④A 边界声明关键词 + ④I §17.7 要件）
        nob = json.loads(json.dumps(data, ensure_ascii=False))
        nob["meta"]["boundary_declaration"] = {"required": True, "text": "", "points": []}
        (tmp / "nob.data.json").write_text(json.dumps(nob, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "nob.data.json"), "--out", str(tmp / "nob.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "nob.data.json"), "--html", str(tmp / "nob.html")])
        steps.append(("⑦ 承诺边界声明被删后 verify（应 FAIL）", rc, out))
        checks["⑦ 承诺边界声明被删 ⇒ verify FAIL"] = (rc == 1)
        checks["⑦ FAIL 报出边界声明"] = ("承诺边界声明" in out or "边界" in out)

        # ⑧ ★ 阳性对照（精简轮 2 新增断言 ④O）：把某档「未过判据」的依据关键词抹掉 ⇒ **必红**
        #    做法 = 把 T037 的归因正文替换成不含任何依据关键词的空话（四态与状态都不动）
        ev = json.loads(json.dumps(data, ensure_ascii=False))
        _t = next((r for r in ev["rows"] if r["tier"] == "T037"), None)
        if _t is not None and isinstance(_t.get("verdict"), dict):
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
        nb["rows"][0]["note"] = "—"
        (tmp / "nb.data.json").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "nb.data.json"), "--out", str(tmp / "nb.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "nb.data.json"), "--html", str(tmp / "nb.html")])
        steps.append(("⑨ 备注被写空后 verify（应 FAIL）", rc, out))
        checks["⑨ 备注非契约文本 ⇒ verify FAIL"] = (rc == 1)
        checks["⑨ FAIL 报出 ④O 备注契约"] = ("④O" in out and "契约文本" in out)

        # ⑩ ★ 阳性对照（精简轮 2 新增结构断言）：表格被塞回一列（数据 columns 与渲染表头不一致）⇒ **必红**
        ex = json.loads(json.dumps(data, ensure_ascii=False))
        ex["columns"].insert(2, {"key": "model", "label": "模型", "group": "base"})
        for _r in ex["rows"]:
            _r["model"] = "9B"
        (tmp / "ex.data.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "ex.data.json"), "--out", str(tmp / "ex.html")])
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "ex.data.json"), "--html", str(tmp / "ex.html")])
        steps.append(("⑩ 表格列数被改后 verify（应 FAIL）", rc, out))
        checks["⑩ 列定义偏离已定 13 列 ⇒ verify FAIL"] = (rc == 1)
        checks["⑩ FAIL 报出 ④B 列定义"] = ("④B" in out)

        # ⑪ ★ 阳性对照（精简轮 2 新增断言 ④P）：把边界声明块里的覆盖度门槛注删掉 ⇒ **必红**
        ng = json.loads(json.dumps(data, ensure_ascii=False))
        _gt = (ng.get("meta") or {}).get("boundary_declaration")
        if isinstance(_gt, dict):
            _gt["coverage_gate_note"] = ""
            (tmp / "ng.data.json").write_text(json.dumps(ng, ensure_ascii=False, indent=1), encoding="utf-8")
            run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                 "--data", str(tmp / "ng.data.json"), "--out", str(tmp / "ng.html")])
            _p = tmp / "ng.html"
            if _p.exists():
                _h = _p.read_text(encoding="utf-8")
                _i = _h.find("承诺边界声明"); _j = _h.find("</div>", _i)
                _h = _h[:_i] + _h[_j:] if (_i >= 0 and _j > 0) else _h
                _p.write_text(_h, encoding="utf-8")
            rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                           "--data", str(tmp / "ng.data.json"), "--html", str(_p)])
            steps.append(("⑪ 覆盖度门槛注被删后 verify（应 FAIL）", rc, out))
            checks["⑪ 门槛注缺失 ⇒ verify FAIL"] = (rc == 1)
            checks["⑪ FAIL 报出 ④P 门槛注"] = ("④P" in out and "门槛" in out)

        # ⑫ ★ 阳性对照（修正包 ④R）：往正文注入一处**开发过程词** ⇒ **必红**
        #    ★ 注入点选在矩阵表**之外**的正文（`<h1>` 后），确保命中 ④R 而非其它断言。
        (tmp / "churn.data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "churn.data.json"), "--out", str(tmp / "churn.html")])
        _cp = tmp / "churn.html"
        if _cp.exists():
            _ch = _cp.read_text(encoding="utf-8")
            _ch = _ch.replace("</h1>", "</h1><p>本轮把 18 个档位放大到 100 行（开发过程记录）。</p>", 1)
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

        # ⑮ ★ 阳性对照（修正轮 4 ⚠①）：把过程词塞进**宪法逐字引文段内部** ⇒ **应被豁免、④R 不报**
        #    ★ 这是"豁免面真的生效"的直接证明（上一版豁免锚点在去标签后不存在 ⇒ 从不生效）。
        qg = json.loads(json.dumps(data, ensure_ascii=False))
        (tmp / "qa.data.json").write_text(json.dumps(qg, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "qa.data.json"), "--out", str(tmp / "qa.html")])
        _qp = tmp / "qa.html"
        if _qp.exists():
            _qh = _qp.read_text(encoding="utf-8")
            _qh = _qh.replace("都已在我们的数据集上", "都已在我们的数据集上本轮", 1)
            _qp.write_text(_qh, encoding="utf-8")
        rc, out = run(["tools/matrix/verify_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "qa.data.json"), "--html", str(_qp)])
        steps.append(("⑮ 过程词塞进宪法引文段内部后 verify（④R 应 PASS）", rc, out))
        checks["⑮ 引文段内过程词 ⇒ ④R 豁免（不报）"] = ("④R 正文无开发过程内容=PASS" in out)

        # ⑯ ★ 阳性对照（修正轮 4 ⚠① 反面）：同一词塞到正文**其它位置** ⇒ **必须报**
        qb = json.loads(json.dumps(data, ensure_ascii=False))
        (tmp / "qb.data.json").write_text(json.dumps(qb, ensure_ascii=False, indent=1), encoding="utf-8")
        run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
             "--data", str(tmp / "qb.data.json"), "--out", str(tmp / "qb.html")])
        _bp = tmp / "qb.html"
        if _bp.exists():
            _bh = _bp.read_text(encoding="utf-8")
            _bh = _bh.replace("</h1>", "</h1><p>本轮复核</p>", 1)
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
        description="功能矩阵正式件的三组自证（空数据 / 假数据 / 手改 1 字符）。"
                    "无参数即运行全部自证；中间产物写临时目录，不改正式件。"
    ).parse_args()
    sys.exit(main())
