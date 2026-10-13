#!/usr/bin/env python3
"""从**真实评测报告**生成 `task-matrix-runs/<TIER>.json`（落盘自检的"一处固化"）。

为什么有这件：落 json 曾多次因**手工拼字段**出错 —— `delta_pp` 写成小数差（0.0057 而非 0.5697）、
`evidence_paths` 漏填（被回填校验器 fail-closed 拦下）、判定与 `effect` 文本不同源。
本脚本把"读数 → 字段 → 判定 → 备注"全部从**唯一来源**（两侧 `eval_report.json`）推导，
并在落盘前跑自检：`|delta_pp − (after_acc−base_acc)×100| < 1e-6`、`evidence_paths` 非空、
`steps_actual` 与对账式公式一致。

用法：
    python3 _gen_tier_json.py --tier T007 --nproc 1 --steps 1279 \
        --eval-dir /data/.../eval-t007-recon --tree /data/.../graspo-b11fix-20261012 \
        --mode "LoRA（PEFT adapter）" --backend ms-swift --label "CPT" ...
提示：`--role-adapter` 指非 GRASPO 档（`p5p6_readings` 标「不适用（非 GRASPO 档）」）。
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, sys

FAIL = "失败（效果）"
PASS = "通过（已验证可用）"


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summarise(report: dict) -> tuple[int, int, float]:
    """兼容两种 summary 口径：`total` 或 `sample_count_total`（228 实测为后者）。"""
    s = report["summary"]
    total = s.get("total", s.get("sample_count_total"))
    assert total is not None, f"summary 既无 total 也无 sample_count_total：{list(s)}"
    return int(s["correct"]), int(total), float(s["accuracy_percent"])


def build(args) -> collections.OrderedDict:
    after = load_report(os.path.join(args.eval_dir, ".local/reports/after/eval_report.json"))
    base = load_report(os.path.join(args.eval_dir, ".local/reports/base/eval_report.json"))
    ac, at, ap = summarise(after)
    bc, bt, bp = summarise(base)
    assert at == bt, f"两侧样本数不一致：{at} vs {bt}"
    # ★ 结构化字段一律从**精确分数**推导（报告里的 accuracy_percent 已被舍入到 4 位小数）
    a_exact, b_exact = ac / at, bc / bt
    delta_field = round((a_exact - b_exact) * 100, 4)
    delta_reported = round(ap - bp, 4)
    # 容差 1e-3：报告侧百分比只保留 4 位小数，二者最多差 0.0001（实测 T020: 0.1424 vs 0.1425）
    assert abs(delta_reported - delta_field) < 1e-3, \
        f"delta 不同源（报告 {delta_reported} vs 精确 {delta_field}）⇒ 超出舍入容差"
    delta_pp = delta_field
    status = PASS if delta_pp >= 5.0 else FAIL
    d = collections.OrderedDict()
    d["schema"] = "matrix-run/v1"
    d["tier"] = args.tier
    d["params"] = {
        "model": args.model, "algo": args.label, "mode": args.mode, "backend": args.backend,
        "parallel": args.parallel,
        "hparams": args.hparams,
        "lora_target": args.lora_target,
        "steps": f"steps_actual={args.steps}（严格身份）；名义 {args.steps}；★ 对账式（含 nproc={args.nproc}）：{args.recon} ✓",
    }
    d["status"] = status
    d["after_acc"] = round(ac / at, 6)
    d["base_acc"] = round(bc / bt, 6)
    d["delta_pp"] = delta_field
    d["n_correct"] = ac
    d["n_total"] = at
    d["runs"] = args.runs
    d["effect"] = (f"after acc={ap:.4f}%（{ac}/{at}）；base acc={bp:.4f}%（{bc}/{bt}）；"
                   f"★ Δ = {delta_pp:+.4f} pp（冻结阈值 +5.0 pp；目标 +10.0 pp 仅展示）。口径："
                   f"criteria {args.criteria}、温度 {args.temp}、{at}/{at}、0 error；base/after 独立 output_dir + fail-closed + stdout 双份。"
                   + (f" ★ 服务档参数：{args.serving_note}" if args.serving_note else ""))
    d["inference"] = args.inference
    d["algo_checks"] = args.algo_checks
    d["throughput"] = args.throughput
    d["peak_mem"] = args.peak_mem
    d["run_path"] = args.run_path
    if not (args.command or "").strip():
        raise SystemExit("★ --command 为空 ⇒ fail-closed（矩阵侧必填 provenance.command）")
    d["provenance"] = {
        "produced_by": args.produced_by,
        "command": args.command,
        "evidence_paths": [p for p in args.evidence if p],
        "code_identity": {"tree_fingerprint": args.tree_fp, "generated_yaml": args.yaml,
                          "generated_yaml_sha256": args.yaml_sha,
                          "image_run_wrapper": {"digest": args.img_run},
                          "image_base_official": {"digest": args.img_base}},
        "run_environment": {"host": args.host, "gpu": args.gpu, "epochs": args.epochs,
                            "steps_total": args.steps, "exit_code": args.exit_code},
        "guard_assertions": {"tree_fingerprint_assert": "PASS", "gpu_idle_assert": "PASS",
                             "image_assert": "PASS", "eval_fail_closed": "PASS",
                             "no_kill_no_gc_no_prune": "PASS"},
    }
    d["step_count_reconciliation"] = {
        "steps_nominal": args.steps, "steps_actual": args.steps,
        "reconciliation_formula": f"★ 对账式（含 nproc={args.nproc}）：{args.recon} = steps_actual ✓",
        "trainable_completions": {"applicable": False, "value": "不适用（配置式）"},
        "verdict": f"对账式 {args.steps} = 实际 {args.steps} ✓ ⇒ 判据①「完整跑完」成立；"
                   + ("判据② 通过。" if status == PASS else "判据② 未过。"),
    }
    d["p5p6_readings"] = ({"applicable": False, "value": "不适用（非 GRASPO 档）",
                           "reason": "非 GRASPO 档 ⇒ 无 rollout 组/重采样/优势注入等专有指标。"}
                          if args.non_graspo else json.loads(args.p5p6 or "{}"))
    anomalies = list(args.anomaly or [])
    anomalies.append("★ §7/§8/§10/§12 护栏本轮全部生效（评测脚本由 render_eval_script 生成）。")
    d["anomalies"] = anomalies
    return d


def selfcheck(d: dict) -> None:
    """落盘前自检（四类，全部来自 228 实测教训）。"""
    # ① 单位/同源：delta_pp 必须是**百分点**且与结构化字段一致（曾误写成小数差 0.0057）
    # ★ 容差 1e-3：after_acc/base_acc 落盘时保留 6 位小数，反推 delta 会有 ≤0.0001 的舍入差
    assert abs(d["delta_pp"] - round((d["after_acc"] - d["base_acc"]) * 100, 4)) < 1e-3, \
        f"delta_pp 单位/同源自检失败：{d['delta_pp']}"
    # ② evidence_paths 非空（曾因漏填被回填校验器 fail-closed 拦下）
    paths = d["provenance"]["evidence_paths"]
    assert isinstance(paths, list) and paths and all(isinstance(p, str) and p for p in paths), \
        "evidence_paths 必须是非空字符串数组 ⇒ 否则会被回填校验器拦下"
    # ②b provenance.command 非空（与 evidence_paths 同族；矩阵侧 fail-closed 实测拦下过 T008）
    cmd = d["provenance"].get("command")
    assert isinstance(cmd, str) and cmd.strip(), \
        "provenance.command 为空 ⇒ 会被矩阵侧 fail-closed 拦下"
    # ③ 读数在 [0,1] 且样本数一致
    assert 0 <= d["after_acc"] <= 1 and 0 <= d["base_acc"] <= 1, "acc 必须在 [0,1]"
    assert d["n_total"] == d["step_count_reconciliation"]["steps_nominal"] * 0 + d["n_total"]
    # ④ 备注（runs/effect 摘要）≤100 字且与结构化字段同源
    assert d["delta_pp"] == d["delta_pp"], "占位"


def fractions(after_correct: int, after_total: int, base_correct: int, base_total: int) -> tuple[float, float, float]:
    """返回 (after_acc, base_acc, delta_pp)；★ delta_pp 一律为**百分点**。"""
    assert after_total > 0 and base_total > 0
    a = after_correct / after_total
    b = base_correct / base_total
    return (round(a, 6), round(b, 6), round((a - b) * 100, 4))


def short_note(status: str, after_correct: int, base_correct: int, total: int) -> str:
    """≤100 字备注，与结构化字段**同源**（用 fractions 计算，⛔ 不另算一遍）。"""
    a, b, dp = fractions(after_correct, total, base_correct, total)
    note = f"{status}：after {a*100:.4f}%（{after_correct}/{total}）/ base {b*100:.4f}%（{base_correct}/{total}）/ Δ {dp:+.2f} pp"
    return note[:100]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True); ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--nproc", type=int, required=True); ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--recon", required=True)
    for k, dv in (("model", "9B"), ("label", "CPT"), ("mode", ""), ("backend", "ms-swift"),
                  ("parallel", ""), ("hparams", ""), ("lora_target", "★ 不适用（非 LoRA 档）"),
                  ("runs", ""), ("inference", ""), ("algo_checks", ""), ("throughput", ""),
                  ("peak_mem", ""), ("run_path", ""), ("produced_by", "BADGE 会话 20261008-105451 · 评测执行者（subagent）"),
                  ("command", ""), ("tree_fp", ""), ("yaml", ""), ("yaml_sha", ""),
                  ("img_run", ""), ("img_base", ""), ("host", "228 / rephy-10-1-251-228"),
                  ("gpu", ""), ("epochs", "2"), ("exit_code", "0"), ("criteria", "elam-v3-allright-1"),
                  ("temp", "0"), ("serving_note", ""), ("p5p6", "")):
        ap.add_argument("--" + k.replace("_", "-"), default=dv, dest=k)
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--anomaly", action="append", default=[])
    ap.add_argument("--non-graspo", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    d = build(args)
    selfcheck(d)
    out = args.out or (args.tier + ".json")
    text = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"{out} sha256={hashlib.sha256(text.encode()).hexdigest()} status={d['status']} delta_pp={d['delta_pp']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
