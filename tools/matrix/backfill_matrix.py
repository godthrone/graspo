#!/usr/bin/env python3
"""backfill_matrix.py — 跑次 json ⇒ matrix-data.json ⇒ 重渲染 ⇒ 逐字节校验（唯一回填通道）。

★ 唯一输入格式：`<runs-dir>/<档号>.json`（schema = `matrix-run/v1`）。
  字段与矩阵 10 列一一对应；另带 `provenance`（谁产出 / 哪条命令 / 原始证据路径）。
  格式与示例见 `matrix-schema.md` §3。

★ 流程（全部经脚本，⛔ 任何人不得手改 HTML / 不得手编辑结果字段）：
  ① 逐个读 `<档号>.json` 并**严格校验**（缺字段 / 空值 / 非法状态 / 档号与文件名不符 /
     自报参数与档位表不符 ⇒ 该批 **fail-closed**，一个字节都不写）；
  ② 合并进 `matrix-data.json`（**未跑档位保持 `—`**；同一 json sha 已回填则跳过，可反复运行）；
  ③ `render_matrix.py` 重渲染 `capability-matrix-new.html`（先写临时件）；
  ④ `verify_matrix.py --allow-filled` 逐字节校验临时件；PASS 才落盘，FAIL 即原样不动；
  ⑤ 追加一行「回填台账」（哪档何时回填、来自哪个 json、verify 结果）。

★ 增量语义：跑完一个档 ⇒ 回填一个档（不必等全量）。重复运行安全。
★ 旁路/演示：给 `--data-out` / `--html-out` / `--ledger-out` 指定别处路径，
  即可在不碰真交付件的前提下演练（evidence/backfill-demo-* 就是这么来的）。

用法：
  python3 backfill_matrix.py --runs-dir .local/hb-workspace/<会话>/task-matrix-runs
  python3 backfill_matrix.py --runs-dir <dir> --only T001,T010 --dry-run
  python3 backfill_matrix.py --runs-dir <demo-runs> \
      --data-in matrix-data.json --data-out evidence/backfill-demo-data.json \
      --html-out evidence/backfill-demo.html --ledger-out evidence/backfill-demo-ledger.md
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCHEMA = "matrix-run/v1"
REQUIRED_TOP = ["schema", "tier", "params", "status", "runs", "effect",
                "inference", "algo_checks", "throughput", "peak_mem",
                "run_path", "provenance"]
# ★ v3 同步（2026-10-11）：matrix-data/v3 把参数列拆成逐字段命名键，`hparams` 合并串已移除。
#   - IDENTITY_KEYS：跑次 json 必须自报且必须与档位表逐字一致（防「跑 A 填 B」错位）；
#   - 其余键：json 若提供，则与档位表比对；json 未提供不报错（档位表是参数真相源）；
#   - 表里已不存在的键（如 legacy `hparams`）：登记为 legacy，**忽略**而不 fail-closed（兼容旧 json）。
PARAM_IDENTITY_KEYS = ["model", "algo", "mode", "backend", "parallel"]
PARAM_KEYS_V3 = ["model", "algo", "mode", "backend", "parallel",
                 "lora_target", "lr", "data_subset", "steps", "switches"]
LEGACY_PARAM_KEYS = ["hparams"]
RESULT_KEYS = ["status", "runs", "effect", "inference", "algo_checks",
               "throughput", "peak_mem", "run_path"]
STATUSES = ["通过", "失败", "条件不合逻辑", "未完成测试"]
PLACEHOLDER = "—"

LEDGER_HEADER = """# 回填台账 — `backfill-ledger.md`

> **唯一回填通道**：`backfill_matrix.py` 消费 `task-matrix-runs/<档号>.json` ⇒ 更新 `matrix-data.json`
> ⇒ `render_matrix.py` 重渲染 ⇒ `verify_matrix.py --allow-filled` 逐字节校验。
> **增量**：跑完一个档 ⇒ 回填一个档；**未跑档位保持 `—`**；**缺字段 fail-closed**（整批一个字节都不写）。
> 每次回填写一行；`verify` 列 = 该次回填后端到端校验结果（render + 逐字节比对 + 结构/内容断言）。
> 演示（假数据）记录在 `evidence/backfill-demo-ledger.md`，⛔ 不进本台账。

| 回填时间 | 档号 | 来源 json | json sha256 | 状态 | verify | 备注 |
|:--:|---|---|---|---|:--:|---|
"""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _s(v):
    return v.strip() if isinstance(v, str) else v


def validate_run(doc, tier_from_file, tier_rows):
    """严格校验一份跑次 json。返回错误列表（空 = 合法）。"""
    if not isinstance(doc, dict):
        return ["顶层不是 JSON 对象"]
    errs = [f"缺字段 {k!r}" for k in REQUIRED_TOP if k not in doc]
    if errs:
        return errs

    if doc.get("schema") != SCHEMA:
        errs.append(f"schema 必须是 {SCHEMA!r}，实际 {doc.get('schema')!r}")

    tier = _s(doc["tier"])
    if not isinstance(tier, str) or not tier:
        errs.append("tier 为空或非字符串")
    else:
        if tier != tier_from_file:
            errs.append(f"档号 {tier!r} 与文件名 {tier_from_file!r} 不一致")
        if tier not in tier_rows:
            errs.append(f"档号 {tier!r} 不在矩阵档位表内")

    # --- params：身份键必须齐全且与 v3 档位表逐字一致（防「跑 A 填 B」错位） ---
    #     v3 同步：矩阵参数列已拆为逐字段键；json 里提供的键逐项比对，
    #     未提供不报错（档位表是参数真相源），legacy 键（hparams）忽略并登记。
    p = doc["params"]
    if not isinstance(p, dict):
        errs.append("params 不是 JSON 对象")
    else:
        for k in PARAM_IDENTITY_KEYS:
            v = _s(p.get(k))
            if k not in p:
                errs.append(f"params 缺身份键 {k!r}")
            elif not isinstance(v, str) or not v:
                errs.append(f"params.{k} 为空")
        if tier in tier_rows:
            ref = tier_rows[tier]
            for k in PARAM_KEYS_V3:
                if k in p and _s(p.get(k)) is not None and _s(ref.get(k, "")) != _s(p.get(k)):
                    errs.append(f"params.{k} 与档位表不一致（防错位）：json={p.get(k)!r} 表={ref.get(k)!r}")
            doc.setdefault("_legacy_params_ignored", [])
            for k in LEGACY_PARAM_KEYS:
                if k in p and k not in ref:
                    doc["_legacy_params_ignored"].append(k)

    # --- 结果列：8 项必须存在且非空；status=通过 时不得用占位符 '—' ---
    st = _s(doc["status"])
    if st not in STATUSES:
        errs.append(f"status {st!r} 非法（须为 {' / '.join(STATUSES)} 之一）")
    for k in RESULT_KEYS[1:]:
        v = _s(doc[k])
        if not isinstance(v, str) or not v:
            errs.append(f"结果字段 {k!r} 为空 / 非字符串")
        elif v == PLACEHOLDER and st == "通过":
            errs.append(f"status=通过 但 {k!r} 仍是占位符 '—'（不许半填）")

    # --- provenance：谁产出 / 哪条命令 / 原始证据路径 ---
    prov = doc["provenance"]
    if not isinstance(prov, dict):
        errs.append("provenance 不是 JSON 对象")
    else:
        for k in ["produced_by", "command", "evidence_paths"]:
            if k not in prov:
                errs.append(f"provenance 缺 {k!r}")
        for k in ["produced_by", "command"]:
            v = _s(prov.get(k))
            if not isinstance(v, str) or not v:
                errs.append(f"provenance.{k} 为空")
        ev = prov.get("evidence_paths")
        if not isinstance(ev, list) or not ev or not all(isinstance(x, str) and x.strip() for x in ev):
            errs.append("provenance.evidence_paths 必须是非空字符串数组（原始证据路径）")
    return errs


def run_cmd(args):
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def append_ledger(path: Path, rows):
    text = path.read_text(encoding="utf-8") if path.exists() else LEDGER_HEADER
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r for r in rows]
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="跑次 json 目录（唯一输入格式：<档号>.json）")
    ap.add_argument("--data-in", default="matrix-data.json")
    ap.add_argument("--data-out", default=None, help="默认 = --data-in（原地回填）")
    ap.add_argument("--template", default="template.html")
    ap.add_argument("--html-out", default="capability-matrix-new.html")
    ap.add_argument("--ledger-out", default="backfill-ledger.md")
    ap.add_argument("--only", default=None, help="只回填这些档号，逗号分隔（增量）")
    ap.add_argument("--dry-run", action="store_true", help="只校验与报告，不写任何文件")
    a = ap.parse_args()

    data_in = Path(a.data_in)
    data_out = Path(a.data_out) if a.data_out else data_in
    html_out = Path(a.html_out)
    ledger_out = Path(a.ledger_out)
    runs_dir = Path(a.runs_dir)

    if not data_in.exists():
        print(f"FAIL: 数据文件不存在: {data_in}")
        return 1
    if not Path(a.template).exists():
        print(f"FAIL: 模板不存在: {a.template}")
        return 1
    if not runs_dir.is_dir():
        print(f"FAIL: 跑次目录不存在: {runs_dir}")
        return 1

    matrix = json.loads(data_in.read_text(encoding="utf-8"))
    tier_rows = {r["tier"]: r for r in matrix["rows"]}

    only = set(x.strip() for x in a.only.split(",") if x.strip()) if a.only else None
    files = sorted(runs_dir.glob("*.json"))
    if only:
        files = [f for f in files if f.stem in only]
    if not files:
        print(f"无待回填的 json（目录 {runs_dir}" + (f"，--only {sorted(only)}" if only else "") + "）")
        return 0

    # ---- ① 严格校验（fail-closed：先全验，再统一写） ----
    valid, all_errs = [], []
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            all_errs.append((f, [f"JSON 解析失败: {e}"]))
            continue
        errs = validate_run(doc, f.stem, tier_rows)
        if errs:
            all_errs.append((f, errs))
        else:
            valid.append((f, doc))

    if all_errs:
        print("FAIL-CLOSED：以下跑次 json 不合法，本批**未写任何文件**：")
        for f, errs in all_errs:
            print(f"  ✗ {f}")
            for e in errs:
                print(f"      - {e}")
        return 1

    # ---- ② 合并（未跑档位保持 '—'；同源 sha 已回填则跳过） ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_matrix = json.loads(json.dumps(matrix, ensure_ascii=False))  # 深拷贝
    new_rows = {r["tier"]: r for r in new_matrix["rows"]}
    ledger_rows, skipped, filled = [], [], []
    for f, doc in valid:
        tier = doc["tier"]
        src_sha = sha256_file(f)
        prev = new_rows[tier].get("run_provenance") or {}
        changed = {k: new_rows[tier].get(k) != _s(doc[k]) for k in RESULT_KEYS}
        if prev.get("source_sha256") == src_sha and not any(changed.values()):
            skipped.append(tier)
            ledger_rows.append(f"| {now} | `{tier}` | `{f.name}` | `{src_sha[:12]}` | "
                               f"{doc['status']} | — | 跳过：同源 sha（`{src_sha[:12]}`）已回填，幂等 |")
            continue
        for k in RESULT_KEYS:
            new_rows[tier][k] = _s(doc[k])
        new_rows[tier]["run_provenance"] = {
            "produced_by": doc["provenance"]["produced_by"],
            "command": doc["provenance"]["command"],
            "evidence_paths": list(doc["provenance"]["evidence_paths"]),
            "source_json": str(f),
            "source_sha256": src_sha,
            "backfilled_at": now,
        }
        filled.append(tier)
        ledger_rows.append(f"| {now} | `{tier}` | `{f.name}` | `{src_sha[:12]}` | "
                           f"{doc['status']} | 见下 | 增量回填 1 档 |")

    unfilled = [t for t, r in new_rows.items()
                if all(r.get(k) in ("", PLACEHOLDER) for k in RESULT_KEYS)]
    new_matrix["meta"]["backfill"] = {
        "last_run": now,
        "filled_tiers": sorted(filled),
        "skipped_tiers": sorted(skipped),
        "unfilled_count": len(unfilled),
        "note": "仅脚本回填；未跑档位保持 '—'；缺字段 fail-closed",
    }

    print(f"待回填 {len(files)} 份；本次写入 {len(filled)} 档 {sorted(filled)}；"
          f"跳过（幂等）{len(skipped)} 档；仍未回填 {len(unfilled)} 档（保持 '—'）")

    if a.dry_run:
        print("--dry-run：未写任何文件。")
        return 0

    # ---- ③④ 先写临时件 ⇒ 渲染 ⇒ 逐字节校验 ⇒ PASS 才落盘 ----
    tmp = Path(tempfile.mkdtemp(prefix=".backfill-tmp-", dir=str(data_out.parent or Path("."))))
    try:
        tmp_data, tmp_html = tmp / "data.json", tmp / "out.html"
        tmp_data.write_text(json.dumps(new_matrix, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        rc, out = run_cmd(["render_matrix.py", "--template", str(a.template),
                           "--data", str(tmp_data), "--out", str(tmp_html)])
        if rc != 0:
            print(f"FAIL: 渲染失败\n{out}")
            return 1
        rc, out = run_cmd(["verify_matrix.py", "--template", str(a.template),
                           "--data", str(tmp_data), "--html", str(tmp_html), "--allow-filled"])
        print(out.strip())
        if rc != 0:
            print("FAIL: 校验未通过 ⇒ **不落盘**（交付件与数据保持原样）")
            return 1

        data_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_data, data_out)
        shutil.copyfile(tmp_html, html_out)
        ledger_rows = [r.replace("| 见下 |", "| ✅ PASS |") for r in ledger_rows]
        append_ledger(ledger_out, ledger_rows)
        print(f"OK: 回填 {len(filled)} 档 ⇒ {data_out} / {html_out}；台账 +{len(ledger_rows)} 行 ⇒ {ledger_out}")
        print(f"    data sha256 = {sha256_file(data_out)}")
        print(f"    html sha256 = {sha256_file(html_out)}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
