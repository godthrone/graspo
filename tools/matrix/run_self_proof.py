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
RESULT_KEYS = ["status", "runs", "effect", "inference", "algo_checks",
               "throughput", "peak_mem", "run_path"]
FAKE = {
    "T001": {"status": "已验证可用", "runs": "1/1 真效果训练（两条可用性判据均满足）",
             "effect": "指标 Δ=+0.0 pp（示意）", "inference": "✅ 合并 LoRA → 服务化推理 → 评测集跑完（示意）",
             "algo_checks": "7/7", "throughput": "812 tok/s；墙钟 00:31:20",
             "peak_mem": "25.5", "run_path": "RUN-DEMO-T001-01（2026-10-11）"},
    "T010": {"status": "已验证不可用", "runs": "1/1 真效果训练（第 2 条判据未过）",
             "effect": "Δ=-3.0 pp < 阈值（示意）", "inference": "✅ 产物可推理", "algo_checks": "7/7",
             "throughput": "—", "peak_mem": "20.4", "run_path": "RUN-DEMO-T010-01（2026-10-11）",
             "verdict": {"state": "已验证不可用",
                         "failed_criterion": "第 2 条：在平台方数据集与平台方参数下效果未达标",
                         "failed_detail": "Δ=-3.0 pp < 阈值（示意）",
                         "passed_criteria": ["可完整跑完：exit_code = 0", "产物可推理"]}},
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
            for k in RESULT_KEYS:
                r[k] = "—"
            r["status"] = "尚未验证"
            for k in ("verdict", "excluded_reason"):
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

        # ④ 警示派生化：改数据里的覆盖度计数 ⇒ 首屏警示必须随之改变（非模板硬编码）
        derived = json.loads(json.dumps(data, ensure_ascii=False))
        cov = derived["meta"]["coverage"]
        cov["not_verified"] = 7
        cov["verified_available"] = 2
        cov["gate_met"] = False
        cov["unverified_tiers"] = [r["tier"] for r in derived["rows"]][:7]
        (tmp / "derived.data.json").write_text(json.dumps(derived, ensure_ascii=False, indent=1), encoding="utf-8")
        rc, out = run(["tools/matrix/render_matrix.py", "--template", "tools/matrix/template.html",
                       "--data", str(tmp / "derived.data.json"), "--out", str(tmp / "derived.html")])
        steps.append(("④ 警示派生化渲染（覆盖度改为 7 档尚未验证 / 2 档可用）", rc, out))
        checks["④ 警示派生化渲染 PASS"] = rc == 0
        html_d = (tmp / "derived.html").read_text(encoding="utf-8")
        checks["④ 警示随数据变化（含 7 / 2 计数）"] = ("覆盖度警示" in html_d
                                                 and "「<b>尚未验证</b>」<b>7</b> 档" in html_d
                                                 and "「已验证可用」<b>2</b> 档" in html_d)

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
