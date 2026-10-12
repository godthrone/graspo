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
import re
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


def norm_parallel(s):
    """把并行展示串归一化为维度元组（tp,pp,dp,nproc,总卡数），用于**语义**身份比对。

    容许措辞差异（如尾部多一句「总卡数 N」），但**维度必须逐项一致**——防「跑 A 填 B」错位的
    严格性不变；⛔ 不因展示串写法不同而绕过比对。
    """
    if not isinstance(s, str):
        return None

    def num(pat):
        m = re.search(pat, s)
        return m.group(1) if m else None

    return (num(r"TP\s*=\s*(\d+)"), num(r"PP\s*=\s*(\d+)"), num(r"DP\s*=\s*(\d+)"),
            num(r"nproc_per_node\s*=\s*(\d+)"), num(r"(\d+)\s*卡"))


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
                if k not in p or _s(p.get(k)) is None:
                    continue
                j, r_ = _s(p.get(k)), _s(ref.get(k, ""))
                if k == "parallel":
                    # ★ 语义比对：容许措辞差异，但维度必须逐项一致（防错位严格性不变）
                    nj, nr = norm_parallel(j), norm_parallel(r_)
                    if nj != nr:
                        errs.append(f"params.parallel 维度与档位表不一致（防错位）："
                                    f"json={j!r}→{nj} 表={r_!r}→{nr}")
                elif r_ != j:
                    errs.append(f"params.{k} 与档位表不一致（防错位）：json={j!r} 表={r_!r}")
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


TOOL_DIR = Path(__file__).resolve().parent


def run_cmd(args):
    """运行同目录工具脚本；脚本名按本文件所在目录解析（可从仓库根调用）。"""
    argv = [sys.executable]
    for i, x in enumerate(args):
        if i < 2 and x.endswith(".py") and not Path(x).is_absolute() and (TOOL_DIR / x).exists():
            argv.append(str(TOOL_DIR / x))
        else:
            argv.append(x)
    p = subprocess.run(argv, capture_output=True, text=True)
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
    ap.add_argument("--official-data", default="docs/capability-matrix.data.json",
                    help="正式机读件（由 build_official.py 从内部数据派生）")
    ap.add_argument("--official-html", default="docs/capability-matrix.html",
                    help="正式人读视图（由 render_matrix.py 派生）")
    ap.add_argument("--version", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--void", action="append", default=[],
                    help='显式声明"该跑次结论作废、仅作参考"，格式 档号:理由（可重复）。'
                         '用于"参数已更正 ⇒ 结论不对应当前行参数"（§结论与参数一一对应）')
    ap.add_argument("--dry-run", action="store_true", help="只校验与报告，不写任何文件")
    a = ap.parse_args()

    data_in = Path(a.data_in)
    data_out = Path(a.data_out) if a.data_out else data_in
    html_out = Path(a.html_out)
    # 相对路径的台账落在"矩阵数据所在目录"（工作区），⛔ 不再落进仓库根造成 stray 文件
    ledger_out = Path(a.ledger_out)
    if not ledger_out.is_absolute() and not ledger_out.exists():
        ledger_out = data_in.parent / ledger_out.name
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
    void_map = {}
    for spec in a.void:
        t, _, why = spec.partition(":")
        if not t.strip() or not why.strip():
            print(f"FAIL: --void 格式必须是 档号:理由，实际 {spec!r}")
            return 1
        void_map[t.strip()] = why.strip()
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
        # 幂等判据：只比"读数"；★ status 不由跑次 json 直接决定（阈值未冻结时保持「尚未验证」），故排除
        changed = {k: new_rows[tier].get(k) != _s(doc[k]) for k in RESULT_KEYS if k != "status"}
        if prev.get("source_sha256") == src_sha and not any(changed.values()):
            skipped.append(tier)
            ledger_rows.append(f"| {now} | `{tier}` | `{f.name}` | `{src_sha[:12]}` | "
                               f"{doc['status']} | — | 跳过：同源 sha（`{src_sha[:12]}`）已回填，幂等 |")
            continue
        for k in RESULT_KEYS:
            new_rows[tier][k] = _s(doc[k])
        # ★ 显式作废（参数已更正 ⇒ 结论不对应当前行参数）：作为"参考记录"留存，不作判定
        if tier in void_map:
            ref = "；".join(x for x in (_s(doc.get("runs")), _s(doc.get("effect"))) if x)
            new_rows[tier]["conclusion_annotation"] = {
                "conclusion_valid": False, "reason": void_map[tier], "reference_record": ref,
            }
            new_rows[tier]["status"] = "尚未验证"
            new_rows[tier].pop("measurement_pending", None)
            new_rows[tier]["note"] = ("参考记录（" + void_map[tier] + "）：" + (ref or "—"))
            new_rows[tier]["run_provenance"] = {
                "produced_by": doc["provenance"]["produced_by"],
                "command": doc["provenance"]["command"],
                "evidence_paths": list(doc["provenance"]["evidence_paths"]),
                "source_json": str(f), "source_sha256": src_sha, "backfilled_at": now,
                "voided": True,
            }
            filled.append(tier)
            ledger_rows.append(f"| {now} | `{tier}` | `{f.name}` | `{src_sha[:12]}` | "
                               f"作废（仅参考） | 见下 | 参数已更正 ⇒ 判定作废 |")
            continue
        # ★ 状态规则（§17.7：「已验证可用」须两条判据同时满足）：
        #   跑次 json 未声明「阈值已冻结」⇒ 判据②无法判定 ⇒ 状态保持「尚未验证」，只录读数。
        frozen = bool(doc.get("threshold_frozen"))
        if not frozen:
            new_rows[tier]["status"] = "尚未验证"
            new_rows[tier]["measurement_pending"] = True   # ★ 读数已录、阈值待定（不作判定）
            new_rows[tier].pop("conclusion_annotation", None)   # 新读数取代旧「作废参考记录」注记
            passed1 = _s(doc.get("status")) == "通过"
            new_rows[tier]["note"] = ("读数为实测记录，⛔ 暂不作判定：" +
                                      ("①「能完整跑完」已通过；" if passed1 else "") +
                                      "②「平台方数据集效果达标」读数已录，★ 阈值 δ 待定 ⇒ 状态暂保持「尚未验证」。")
        else:
            new_rows[tier]["status"] = "通过" if _s(doc.get("status")) == "通过" else "失败"
            new_rows[tier]["measurement_pending"] = False
            new_rows[tier]["note"] = _s(doc.get("note")) or "阈值已冻结，按两条可用性判据判定。"
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
        tmp_data, tmp_off, tmp_html = tmp / "internal.json", tmp / "official.json", tmp / "out.html"
        tmp_data.write_text(json.dumps(new_matrix, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        # 内部数据 ⇒ 正式机读件（四态 / epoch / 派生步数 等派生逻辑在 build_official 内）
        bo = ["build_official.py", "--in", str(tmp_data), "--out", str(tmp_off)]
        if a.version:
            bo += ["--version", a.version]
        date = a.date or (matrix.get("meta") or {}).get("backfill", {}).get("last_run", "")[:10] or None
        if date:
            bo += ["--date", date]
        rc, out = run_cmd(bo)
        if rc != 0:
            print(f"FAIL: 正式机读件生成失败\n{out}")
            return 1
        rc, out = run_cmd(["render_matrix.py", "--template", str(a.template),
                           "--data", str(tmp_off), "--out", str(tmp_html)])
        if rc != 0:
            print(f"FAIL: 渲染失败\n{out}")
            return 1
        rc, out = run_cmd(["verify_matrix.py", "--template", str(a.template),
                           "--data", str(tmp_off), "--html", str(tmp_html)])
        print(out.strip())
        if rc != 0:
            print("FAIL: 校验未通过 ⇒ **不落盘**（交付件与数据保持原样）")
            return 1

        for f in (data_out, Path(a.official_data), Path(a.official_html)):
            f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_data, data_out)
        shutil.copyfile(tmp_off, Path(a.official_data))
        shutil.copyfile(tmp_html, Path(a.official_html))
        ledger_rows = [r.replace("| 见下 |", "| ✅ PASS |") for r in ledger_rows]
        append_ledger(ledger_out, ledger_rows)
        print(f"OK: 回填 {len(filled)} 档 ⇒ {data_out}（内部）/ {a.official_data} / {a.official_html}；"
              f"台账 +{len(ledger_rows)} 行 ⇒ {ledger_out}")
        print(f"    official data sha256 = {sha256_file(Path(a.official_data))}")
        print(f"    official html sha256 = {sha256_file(Path(a.official_html))}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
