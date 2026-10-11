#!/usr/bin/env python3
"""build_official.py — 由内部 matrix-data.json 生成正式机读数据 docs/capability-matrix.data.json。

★ 定位：正式官方文档的**机读形式**，与 docs/capability-matrix.html 同为唯一真相源。
★ 脱敏（宪法 §15.1，A 类零容忍）：宿主路径 / 内网路径 / 机器名 / 人名一律替换为符号或删除；
   跑次只引用**不透明跑次 ID**，不含任何部署环境记录（尊重 §17.2 的立法目的）。
★ 只保留「定义 + 表格」层：parameters(columns/groups/uniform) + rows；过程件（provenance /
   conflicts / invariants / 取证叙述）不进正式件，迁至 .local/ 决策记录与工作日志。

用法：
  python3 scripts/matrix/build_official.py \
      --in  .local/hb-workspace/<会话>/task-matrix-html/matrix-data.json \
      --out docs/capability-matrix.data.json
"""
import argparse, datetime, json, re
from pathlib import Path

# 脱敏映射（顺序敏感：先长后短）
# 通用形状（⛔ 不写任何具体内网主机 / 账号 / 挂载点字面量）：
#   顺序敏感 —— 先处理带尾段的规则，再用通用二级/三级路径兜底。
PATH_MAP = [
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/subsets-v2/", "subsets-v2/"),
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/images", "images"),
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/data/", "data/"),
    (r"/models/", ""),
    (r"/home/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/", ""),
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/", ""),
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/", ""),
    (r"\.local/repo-moved-out/", ""),
    (r"\.local/hb-workspace/[A-Za-z0-9-]+/", ""),
    (r"\.local/", ""),
]
OPAQUE_RUN = {"T028": ("RUN-20261011-T028-01", "2026-10-11")}


def sanitize_str(s: str) -> str:
    for pat, rep in PATH_MAP:
        s = re.sub(pat, rep, s)
    s = re.sub(r"runs/[A-Za-z0-9._-]+/runs/(T\d{3})", r"\1", s)
    return s


def sanitize(obj):
    if isinstance(obj, str):
        return sanitize_str(obj)
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj


def official_run_path(row):
    """跑次列 = 不透明跑次 ID + 日期（不含任何文件系统路径）。"""
    t = row.get("tier")
    if row.get("status") in ("", "—", None):
        return "—"
    rid, date = OPAQUE_RUN.get(t, (f"RUN-{t}", ""))
    return f"{rid}（{date}）" if date else rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=None)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()

    src = json.loads(Path(a.inp).read_text(encoding="utf-8"))
    meta = src.get("meta", {})

    # 列：跑次列改名为「跑次 ID / 日期」（正式件不暴露路径）
    columns = []
    for c in src["columns"]:
        c = dict(c)
        if c["key"] == "run_path":
            c["label"] = "跑次 ID / 日期"
        columns.append(c)

    # 行：脱敏 + 跑次列替换为不透明 ID
    rows = []
    for r in src["rows"]:
        r = sanitize(dict(r))
        r.pop("run_provenance", None)          # 过程件：不进正式件
        if r.get("status") not in ("", "—", None):
            r["run_path"] = official_run_path(r)
        rows.append(r)

    doc = {
        "schema": "capability-matrix/v1-official",
        "meta": {
            "title": "GRASPO 能力矩阵",
            "version": a.version or meta.get("data_revision", ""),
            "doc_date": a.date,
            "scope": "GRASPO 训练框架的训练配置能力规格与实测结论",
            "authority": "本文件是所有训练实验的唯一真相源（Single Source of Truth）；训练配置以其为准派生。",
            "machine_interface": "docs/capability-matrix.html（人读）；本文件（机读）——二者同为真相源。",
            "criteria_version": meta.get("criteria_version", ""),
            "result_placeholder": meta.get("result_placeholder", "—"),
            "results_policy": "结果列由跑次记录回填；未跑档位保持占位符；不得手工填写。",
        },
        "groups": src["groups"],
        "columns": columns,
        "uniform": sanitize(src.get("uniform", {})),
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    filled = [r["tier"] for r in rows if any(r.get(k) not in ("", "—", None)
              for k in ("status", "runs", "effect", "inference", "algo_checks",
                        "throughput", "peak_mem", "run_path"))]
    print(f"official data: {len(rows)} 档 / {len(columns)} 列；已回填 {len(filled)} 档 {filled} -> {out}")


if __name__ == "__main__":
    main()
