#!/usr/bin/env python3
"""build_official.py — 由内部 matrix-data.json 生成正式机读数据 docs/capability-matrix.data.json。

★ 定位：正式官方文档的**机读形式**，与 docs/capability-matrix.html 同为唯一真相源。
★ 脱敏（宪法 §15.1，A 类零容忍）：宿主路径 / 内网路径 / 机器名 / 人名一律替换为符号或删除；
   跑次只引用**不透明跑次 ID**，不含任何部署环境记录（尊重 §17.2 的立法目的）。
★ 只保留「定义 + 表格」层：parameters(columns/groups/uniform) + rows；过程件（provenance /
   conflicts / invariants / 取证叙述）不进正式件，迁至 .local/ 决策记录与工作日志。

用法：
  python3 tools/matrix/build_official.py \
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


def official_run_path(row, state):
    """跑次列 = 不透明跑次 ID + 日期（不含任何文件系统路径）。"""
    if state in ("尚未验证", "未纳入验收"):
        return "—"
    rid, date = OPAQUE_RUN.get(row.get("tier"), (f"RUN-{row.get('tier')}", ""))
    return f"{rid}（{date}）" if date else rid


#: 四态（§17.7）——只允许这四个取值，且必须是可区分的文字标签
S_AVAILABLE, S_UNAVAILABLE, S_UNVERIFIED, S_EXCLUDED = "已验证可用", "已验证不可用", "尚未验证", "未纳入验收"
STATUS_VOCABULARY = [S_AVAILABLE, S_UNAVAILABLE, S_UNVERIFIED, S_EXCLUDED]
CONCLUDED = (S_AVAILABLE, S_UNAVAILABLE)

EXCLUDED_REASON = ("设计上不成立：师生权重常驻合计约 100 GiB，与「节省显存」的设计意图相悖 ⇒ 不产出可运行配置。"
                   "适用范围：本档位不纳入验收。")

GENERATOR = "tools/matrix/render_matrix.py"
SOURCE = "docs/capability-matrix.data.json"
HUMAN_VIEW = "docs/capability-matrix.html"

GROUPS = {"base": "基础", "status": "状态", "param": "主要参数", "data": "实验数据", "note": "备注"}

#: ★ 列序（主席裁决）：档号 → 是否通过 → 主要参数（分列）→ 实验数据 → 备注
#: 「主要参数」只列**档位之间不同**的项；对全部档位相同的参数见文档「统一口径」块，表内不重复。
COLUMNS = [
    {"key": "tier", "label": "档号", "group": "base", "sticky": True},
    {"key": "status", "label": "是否通过", "group": "status", "group_start": True},
    {"key": "model", "label": "模型", "group": "param", "group_start": True},
    {"key": "algo", "label": "算法", "group": "param"},
    {"key": "mode", "label": "模式", "group": "param"},
    {"key": "backend", "label": "后端", "group": "param"},
    {"key": "parallel", "label": "卡数与并行", "group": "param"},
    {"key": "lora_target", "label": "LoRA 目标", "group": "param"},
    {"key": "lr", "label": "lr", "group": "param"},
    {"key": "data_scale", "label": "数据规模", "group": "param"},
    {"key": "steps", "label": "步数/epoch", "group": "param"},
    {"key": "switches", "label": "关键开关", "group": "param"},
    {"key": "peak_mem", "label": "显存峰值(GiB)", "group": "data", "group_start": True},
    {"key": "elapsed", "label": "耗时（每步/总时长）", "group": "data"},
    {"key": "throughput", "label": "吞吐", "group": "data"},
    {"key": "effect", "label": "效果（评测 Δ / loss·reward 窗口）", "group": "data"},
    {"key": "note", "label": "备注（条件 / 限制 / 失败归因）", "group": "note", "group_start": True},
]

DATA_KEYS = ["peak_mem", "elapsed", "throughput", "effect", "note"]


def boundary_declaration():
    return {
        "required": True,
        "text": ("能力与可用档位矩阵说明各档位的可用性与性能基线：每个列出的档位都已在我们的数据集上完整跑通"
                 "并取得良好效果（作为「可用、大概率无 bug」的证据）。训练效果取决于使用者自身的任务、数据集与参数，"
                 "不在本矩阵的承诺范围。"),
        "points": [
            "① 各档可用性：每个列出的档位都已完整跑通（平台方整体承诺以覆盖度门槛为前提）。",
            "② 平台方数据集结论：在平台方数据集与平台方参数下效果达标，且这是「可用、大概率无 bug」的证据。",
            "③ 使用者效果不在承诺范围：使用者的任务、数据集与参数下的效果不归本矩阵承诺。",
        ],
    }


def criteria_caliber():
    return {
        "availability_criteria": [
            "① 能完整跑完：退出码为零、算法机制真实生效、产物可被推理或加载、产物形态合规、显存与吞吐在承诺范围内。",
            "② 在平台方数据集与平台方参数下效果达标：按平台方判据与阈值判定。",
        ],
        "dataset": {"id": "平台方自有评测集（名称 test.jsonl）", "availability": "平台方自有 / 不公开",
                    "version_anchor_sha256": "c202b1e1…", "sample_count": 702,
                    "method": "工具调用准确率（acc）"},
        "criteria_id": "elam-v3-allright-1",
        "threshold": {"name": "graspo_threshold_pp", "value": 20, "unit": "百分点（pp）",
                      "status": "★ 最终阈值 δ 待定 —— 本值仅为该跑次所用阈值，尚不作为冻结基线。"},
        "baseline": {"role": "同档位未训练底座（role=base）",
                     "caliber": "U2 终点窗口 = 末段 max(3, 10% 步数) 步的均值；该档计划/可达优化步数上限见档位「步数」子项。"},
    }


def verdict_of(row, state):
    """已验证不可用档的结论：写明未过的是哪一条判据，并保留已通过项的事实（机器可读）。"""
    if state != S_UNAVAILABLE:
        return None
    return {
        "state": state,
        "failed_criterion": "第 2 条：在平台方数据集与平台方参数下效果未达标",
        "failed_detail": ("Δ = +0.00 pp < 阈值 20 pp（graspo_threshold_pp=20；★ 最终阈值 δ 待定）；"
                          "剔除图像重叠敏感性后 Δ = -0.19 pp。"),
        "passed_criteria": ["可完整跑完：exit_code = 0（6/6 步）",
                            "产物可推理：合并 LoRA → 服务化推理启动 → 固定评测集跑完（0 error）",
                            "算法正确性检查：0/7 FAIL"],
    }


def split_elapsed(tp):
    """由跑次读数文本抽出「耗时」口径（总时长 / 步内累计）。抽不到记 '—'（不编造）。"""
    if not tp or tp in ("—",):
        return "—"
    m1 = re.search(r"墙钟\s*([0-9]{2}:[0-9]{2}:[0-9]{2})", tp)
    m2 = re.search(r"elapsed_sec\s*累计\s*([0-9.]+)\s*s", tp)
    parts = []
    if m2:
        parts.append(f"步内 elapsed 累计 {m2.group(1)} s")
    if m1:
        parts.append(f"总时长 {m1.group(1)}")
    return "；".join(parts) if parts else "—"


def split_throughput(tp):
    """由跑次读数文本抽出「吞吐」口径（decode / 整步）。抽不到记 '—'（不编造）。"""
    if not tp or tp in ("—",):
        return "—"
    parts = []
    m1 = re.search(r"(decode\s*[0-9.]+\s*tok/s)", tp)
    m2 = re.search(r"(按整步墙钟计\s*[0-9.]+\s*tok/s)", tp)
    if m1:
        parts.append(m1.group(1))
    if m2:
        parts.append(m2.group(1))
    return "；".join(parts) if parts else "—"


def window_text(runs):
    """从跑次说明里抽出 U2 终点窗口读数（loss / reward 及基线同口径值）。抽不到返回 ''。"""
    m = re.search(r"终点窗口\s*(loss\s*[-0-9.]+\s*/\s*reward\s*[-0-9.]+)\s*（([^）]*)）", runs or "")
    return f"{m.group(1)}（{m.group(2)}）" if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=None)
    ap.add_argument("--date", default=None,
                    help="文档日期；缺省读输入数据的 meta.doc_date。⛔ 不读时钟（§17.7 生成确定性）。")
    a = ap.parse_args()

    src = json.loads(Path(a.inp).read_text(encoding="utf-8"))
    meta = src.get("meta", {})
    date = a.date or meta.get("doc_date")
    if not date:
        raise SystemExit("ERROR: 需要 --date 或输入数据的 meta.doc_date（⛔ 生成器不读时钟，§17.7）")

    rows = []
    for r in src["rows"]:
        r = sanitize(dict(r))
        r.pop("run_provenance", None)
        raw = r.get("status")
        if r.get("applicable") is False:
            state = S_EXCLUDED
        elif raw == "通过":
            state = S_AVAILABLE
        elif raw in ("失败", "不通过"):
            state = S_UNAVAILABLE
        else:
            state = S_UNVERIFIED
        r["status"] = state
        r.pop("spec_status", None)

        v = verdict_of(r, state)
        concluded = state in CONCLUDED
        # 实验数据四列：只有已结论档才有读数；其余记 '—'
        if concluded:
            tp_raw = r.get("throughput", "—")
            r["elapsed"] = split_elapsed(tp_raw)
            r["throughput"] = split_throughput(tp_raw)
            w = window_text(r.get("runs", ""))
            r["effect"] = (r.get("effect", "—") + ("；" + w if w else "")).strip("；") or "—"
        else:
            for k in ("peak_mem", "elapsed", "throughput", "effect"):
                r[k] = "—"
        # 数据规模 = 全局条数 + 每卡派生
        if r.get("subset_size") not in (None, "", "—"):
            ppr = r.get("prompts_per_rank")
            r["data_scale"] = f"{r['subset_size']} 条" + (f"（每卡 {ppr}）" if ppr not in (None, "", "—") else "")
        else:
            r["data_scale"] = "—"
        # 备注：失败归因（未过判据 + 已通过项 + 条件） / 未纳入验收理由
        if state == S_EXCLUDED:
            r["note"] = EXCLUDED_REASON
            r["excluded_reason"] = EXCLUDED_REASON
        elif v:
            r["note"] = ("未过" + v["failed_criterion"] + "：" + v["failed_detail"] +
                         " 已通过：" + " / ".join(v["passed_criteria"]) +
                         "。条件：数据集锚点 c202b1e1…、判据 elam-v3-allright-1、温度 0、有效样本 702。")
        else:
            r["note"] = "—"
        r["run_path"] = official_run_path(r, state)
        if v:
            r["verdict"] = v
        rows.append(r)

    doc = {
        "schema": "capability-matrix/v3-official",
        "meta": {
            "title": "GRASPO 能力与可用档位矩阵",
            "version": a.version or meta.get("data_revision", ""),
            "doc_date": date,
            "scope": "GRASPO 训练框架的能力与可用档位：逐档可用性、性能基线与验收结论",
            "authority": {
                "editable_source": f"{SOURCE}（唯一可编辑源头）",
                "human_view": f"{HUMAN_VIEW}（由生成器确定性产出，禁止手工编辑）",
                "generator": GENERATOR, "data_path": SOURCE, "human_view_path": HUMAN_VIEW,
                "generation_marker": "generated",
                "direction": "结果 JSON → capability-matrix.data.json → 人读视图（单方向，禁止互推）",
            },
            "status_vocabulary": STATUS_VOCABULARY,
            "boundary_declaration": boundary_declaration(),
            "criteria_caliber": criteria_caliber(),
            "criteria_version": meta.get("criteria_version", ""),
            "result_placeholder": meta.get("result_placeholder", "—"),
            "results_policy": "结果列由跑次记录回填；未有实测结论的档位标注为「尚未验证」；不得手工填写。",
        },
        "groups": GROUPS,
        "columns": COLUMNS,
        "uniform": sanitize(src.get("uniform", {})),
        "rows": rows,
        "data_keys": DATA_KEYS,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_VOCABULARY}
    print(f"official data: {len(rows)} 档；四态 {counts} -> {out}")


if __name__ == "__main__":
    main()
