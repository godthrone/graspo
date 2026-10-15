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
RESULT_KEYS = ["peak_mem", "steps_actual", "elapsed", "throughput", "effect"]
FAKE = {
    # ★ 与 verify ④J（耗时列 fail-closed：跑次输入域有墙钟 ⇒ 列不得为空）**同批更新**：
    #   示意档必须自带 `throughput_detail` 的墙钟，否则「列有值、输入域无读数」会被判为疑编造。
    "T001": {"status": "已验证可用", "peak_mem": "25.5", "elapsed": "总时长 00:31:20",
             "throughput_detail": "墙钟 00:31:20（示意）",
             "throughput": "812 tok/s", "effect": "Δ=+0.0 pp（示意）", "note": "示意数据"},
    "T010": {"status": "已验证不可用", "peak_mem": "20.4", "elapsed": "总时长 00:12:00",
             "throughput_detail": "墙钟 00:12:00（示意）",
             "throughput": "—", "effect": "Δ=-3.0 pp < 阈值（示意）", "note": "未过第 2 条：效果未达标（示意）",
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
            for k in RESULT_KEYS + ["note", "runs", "inference", "algo_checks", "run_path"]:
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
