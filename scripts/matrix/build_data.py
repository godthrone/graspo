#!/usr/bin/env python3
"""build_data.py — 由 evidence/spec-54.json 生成 matrix-data.json（矩阵文档的机读数据层）。

★ 唯一真相源口径（2026-10-11 主席新增原则）：
  - 功能矩阵文档 + 其数据文件 = 所有实验的唯一真相源；测试 yaml 由本表派生（矩阵为源、yaml 为产物）。
  - 「主要训练参数」列（人读）= 该档与别档**不同**的参数；
  - 「主要参数（机读）」列 = 同一批字段的 **JSON 对象**，供 yaml 生成器直接消费（避免人读/机读两套）；
  - **对全部档位相同的参数**抽到 `uniform`（§1 前置「统一口径」块），矩阵表只列差异项。

- 参数来源：evidence/spec-54.json（由 extract_spec.py 抽 r3 权威件 + 54 档 v2 yaml + 归档生成器 + 8 个示例）；
  缺项写 '—'（不编造）。
- 结果列：默认占位符 '—'；`--preserve-from <旧 matrix-data.json>` 保留已回填档位的结果与 run_provenance（如 T028）。

用法：
  python3 extract_params.py --html <AUTH_HTML> --config-dir .../matrix54-v2 --out evidence/params-raw.json
  python3 extract_spec.py --params evidence/params-raw.json --config-dir .../matrix54-v2 \
      --out evidence/spec-54.json --conflicts-out evidence/spec-conflicts.md
  python3 build_data.py --spec evidence/spec-54.json --out matrix-data.json --preserve-from matrix-data.json
"""
import argparse, json
from pathlib import Path

PLACEHOLDER = "—"
RESULT_KEYS = ["status", "runs", "effect", "inference", "algo_checks",
               "throughput", "peak_mem", "run_path"]

ALGO_DISPLAY = {"cpt": "CPT", "sft": "SFT", "graspo": "GRASPO", "opd": "OPD"}
MODE_DISPLAY = {"lora": "LoRA", "full": "全参"}
BACKEND_DISPLAY = {"msswift": "ms-swift", "native": "native"}
ID = "（机读）"

GROUPS = {"base": "基础", "param": "主要训练参数", "result": "结果"}

# 11 列：档号 → 主要训练参数（人读）→ 主要参数（机读 JSON）→ 结果 8 列
PARAM_SUBKEYS = ["model", "algo", "mode", "backend", "parallel", "lora_target",
                 "lr", "data_subset", "steps", "switches"]
COLUMNS = [
    {"key": "tier", "label": "档号", "group": "base", "sticky": True},
    {"key": "params",
     "label": "主要训练参数（模型 / 方法 / 模式 / backend / 并行维度与卡数 / LoRA 目标 / lr / 数据子集与条数 / epoch或步数 / 关键开关）",
     "group": "param", "group_start": True,
     "merge": PARAM_SUBKEYS,
     "sub_labels": {"model": "模型", "algo": "方法", "mode": "模式", "backend": "backend",
                    "parallel": "并行", "lora_target": "LoRA 目标", "lr": "lr",
                    "data_subset": "数据子集", "steps": "步数", "switches": "开关"},
     "merge_sep": " ／ "},
    {"key": "spec_json", "label": "主要参数（机读 JSON：并行 / target_modules / lr / data / 步数 / lora / msswift·native / status）",
     "group": "param"},
    {"key": "status", "label": "是否通过", "group": "result", "group_start": True},
    {"key": "runs", "label": "跑次说明", "group": "result"},
    {"key": "effect", "label": "效果（评测 vs 基准）", "group": "result"},
    {"key": "inference", "label": "推理验证", "group": "result"},
    {"key": "algo_checks", "label": "算法检查", "group": "result"},
    {"key": "throughput", "label": "吞吐/墙钟", "group": "result"},
    {"key": "peak_mem", "label": "显存峰值(GiB)", "group": "result"},
    {"key": "run_path", "label": "跑次路径/日期", "group": "result"},
]

# 供 yaml 生成器机读的扁平字段（每行必须存在；'—' = 源缺失 / 不适用）
SPEC_KEYS = ["model_path", "train_method", "tuner_type", "cards", "parallel_dims",
             "parallel_scheme", "data_source", "subset_size", "prompts_per_rank",
             "subset_generation", "subset_r3_historical", "train_path",
             "planned_steps", "save_steps", "gpus", "target_modules", "target_modules_mode",
             "deepspeed", "allow_unverified_pp_rollout", "nnodes",
             "expected_optimizer_steps_per_epoch", "expected_optimizer_steps_reachable",
             "expected_optimizer_steps_historical", "steps_formula_v2", "min_optimizer_steps",
             "applicable", "spec_status", "spec_json"]

EXTRACT_CMD = ("python3 extract_params.py --html <AUTH_HTML> "
               "--config-dir <repo-root>/samples/configs/matrix54-v2 "
               "--out evidence/params-raw.json")
SPEC_CMD = ("python3 extract_spec.py --params evidence/params-raw.json "
            "--config-dir <repo-root>/samples/configs/matrix54-v2 "
            "--out evidence/spec-54.json --conflicts-out evidence/spec-conflicts.md")
BUILD_CMD = ("python3 build_data.py --spec evidence/spec-54.json "
             "--out matrix-data.json --preserve-from matrix-data.json")

PROVENANCE = {
    "tier": {"source": "r3 权威件 <td data-col=\"tier\">", "produced_by": "extract_params.py",
             "command": EXTRACT_CMD, "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "model": {"source": "r3 <td data-col=\"model\">", "produced_by": "extract_params.py",
              "command": EXTRACT_CMD, "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "algo": {"source": "档位 yaml 顶层 train_method（显示名）", "produced_by": "extract_spec.py",
             "command": SPEC_CMD, "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "mode": {"source": "档位 yaml 顶层 tuner_type", "produced_by": "extract_spec.py",
             "command": SPEC_CMD, "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "backend": {"source": "档位 yaml 顶层 backend", "produced_by": "extract_spec.py",
                "command": SPEC_CMD, "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "parallel": {"source": "r3 <td data-col=\"parallelism\"> + <td data-col=\"cards\">",
                 "produced_by": "extract_params.py + build_data.py:parallel_of()",
                 "command": EXTRACT_CMD + " ; " + BUILD_CMD,
                 "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "lora_target": {"source": ("★ 恢复既有正式口径 target_modules=[language_all_linear, vision_common]（122 件）；"
                               "出处 = 8 个 git 跟踪示例配置；全参档 '—（全参）'"),
                    "produced_by": "extract_spec.py:RESTORED_LORA + build_data.py:lora_target_of()",
                    "command": SPEC_CMD + " ; " + BUILD_CMD,
                    "owner": "主席裁定（2026-10-11）", "kind": "config"},
    "lr": {"source": "LoRA 档 = 恢复值 1.0e-05（8 个示例）；full 档 = 按方法既有值（SFT 5e-5 / GRASPO 5e-6）",
           "produced_by": "extract_spec.py + build_data.py", "command": SPEC_CMD + " ; " + BUILD_CMD,
           "owner": "主席裁定（2026-10-11）", "kind": "config"},
    "data_subset": {"source": "r3 <td data-col=\"data_source\"> + <td data-col=\"subset_size\">",
                    "produced_by": "build_data.py:data_subset_of()", "command": SPEC_CMD + " ; " + BUILD_CMD,
                    "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "steps": {"source": "r3 max_epochs + planned_steps + 生成器步数公式（上限）",
              "produced_by": "build_data.py:steps_of()", "command": SPEC_CMD + " ; " + BUILD_CMD,
              "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "switches": {"source": "档位 yaml save_steps / pretrain.* / distill.*（统一项 use_vllm/reward 已在 §1.0）",
                 "produced_by": "build_data.py:switches_of()", "command": SPEC_CMD + " ; " + BUILD_CMD,
                 "owner": "报表/收集脚本（矩阵侧）", "kind": "config"},
    "spec_json": {"source": "★ 逐档 spec 的机读 JSON（B 块：并行 / target_modules / lr / data / 步数 / lora / msswift·native / status）",
                  "produced_by": "extract_spec.py ⇒ build_data.py:spec_json_of()",
                  "command": SPEC_CMD + " ; " + BUILD_CMD,
                  "owner": "报表/收集脚本（矩阵侧）；yaml 生成器按此消费", "kind": "config"},
    "status": {"source": "跑次台账 / 判定层（§1 四条通用判据 + 算法检查）",
               "produced_by": "跑次 json（task-matrix-runs/<档号>.json）+ backfill_matrix.py",
               "command": "python3 backfill_matrix.py --runs-dir ../task-matrix-runs",
               "owner": "判定层 + 报表/收集脚本", "kind": "result"},
    "runs": {"source": "跑次台账（真效果训练 ×1 的口径与结果）",
             "produced_by": "跑次 json + backfill_matrix.py", "command": "同 status",
             "owner": "判定层 + 报表/收集脚本", "kind": "result"},
    "effect": {"source": "固定评测集上的评测报告 + 单卡首跑基准台账（δ 随基准冻结）",
               "produced_by": "评测脚本 → 跑次 json → backfill_matrix.py", "command": "同 status",
               "owner": "训练执行者（须主席批准基准）", "kind": "result"},
    "inference": {"source": "U4 推理验证：合并 LoRA → vLLM 启动 → 固定评测集跑完（且与训练末期一致/接近）",
                  "produced_by": "推理验证脚本 → 跑次 json → backfill_matrix.py", "command": "同 status",
                  "owner": "训练执行者 / 判定层", "kind": "result"},
    "algo_checks": {"source": "algorithm-correctness-checks.md 逐算法清单（每条含阳性对照）",
                    "produced_by": "逐条检查脚本 / pytest → 跑次 json → backfill_matrix.py",
                    "command": "同 status", "owner": "Ripple + flow·adapter + flow·backend", "kind": "result"},
    "throughput": {"source": "跑次指标（tokens/s）与墙钟", "produced_by": "跑次 json + backfill_matrix.py",
                   "command": "同 status", "owner": "判定层 + 报表/收集脚本", "kind": "result"},
    "peak_mem": {"source": "native：容器内 allocator rank0 max_allocated；ms-swift：该进程 max_memory_reserved",
                 "produced_by": "跑次 json + backfill_matrix.py", "command": "同 status",
                 "owner": "判定层 + 报表/收集脚本", "kind": "result"},
    "run_path": {"source": "跑次台账（runs/<...>/T0NN + 日期）", "produced_by": "跑次 json + backfill_matrix.py",
                 "command": "同 status", "owner": "判定层 + 报表/收集脚本", "kind": "result"},
}


def v(x):
    if x is None or str(x).strip() == "":
        return PLACEHOLDER
    return str(x).strip()


def jnum(x):
    """机读 JSON 里的数值：None -> None；否则尽量数值化。"""
    if x is None or str(x).strip() in ("", "—"):
        return None
    s = str(x).strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f
    except ValueError:
        return s


def parallel_of(t):
    """★ 并行维度按后端分列（C4 回执）：native 用 tp/dp/pp；ms-swift 用 nproc_per_node(+nnodes)。不可互填。"""
    p = t.get("parallel") or {}
    cards = p.get("cards")
    if t.get("backend") == "msswift":
        ms = t.get("msswift") or {}
        n = ms.get("nproc_per_node")
        if n is None:
            return f"—（{cards} 卡）" if cards else PLACEHOLDER
        extra = f"；nnodes={ms['nnodes']}" if ms.get("nnodes") not in (None, "") else ""
        return f"nproc_per_node={n}{extra}（{cards} 卡）"
    tp, dp, pp = p.get("tp_size"), p.get("dp_size"), p.get("pp_size")
    if tp is None and dp is None and pp is None:
        return f"—（{cards} 卡）" if cards else PLACEHOLDER
    return f"TP={v(tp)} PP={v(pp)} DP={v(dp)}（{cards} 卡）"


def lora_target_of(t):
    if t.get("tuner_type") == "full":
        return "—（全参）"
    tm = t.get("target_modules")
    if tm:
        return f"{' + '.join(tm)}（122 件：language 12 + visual 110）"
    return PLACEHOLDER


def data_subset_of(t):
    """★ 用【现行】subset_size 构造显示串（⛔ 不得沿用 r3 的历史条数：T040–42 的 r3 data_source 写作 72/96/192）。"""
    d = t.get("data") or {}
    n, form = d.get("subset_size"), d.get("source_form")
    name = {"mini": "ELAM V5 mini 冒烟集", "formal": "ELAM V5 正式集"}.get(form)
    if n is None:
        return PLACEHOLDER
    if name:
        return f"{name}（前 {n} 条）"
    return f"前 {n} 条"


def steps_of(t, planned):
    if not t.get("applicable"):
        return PLACEHOLDER
    ep = t.get("max_epochs")
    reach = t.get("expected_optimizer_steps_reachable")
    if ep is None and planned in (None, PLACEHOLDER):
        return PLACEHOLDER
    parts = [f"{v(ep)} epoch"]
    if planned not in (None, PLACEHOLDER):
        parts.append(f"计划 {planned} 步")
    if reach is not None:
        parts.append(f"上限 {reach} 步")
    return " / ".join(parts)


def switches_of(t):
    parts = []
    if t.get("save_steps") is not None:
        parts.append(f'save_steps={t["save_steps"]}')
    pre = t.get("pretrain") or {}
    if pre.get("loss_scale"):
        parts.append(f'loss_scale={pre["loss_scale"]}')
    if pre.get("use_chat_template") is not None:
        parts.append(f'chat_template={pre["use_chat_template"]}')
    dis = t.get("distill") or {}
    if dis.get("teacher_model_path"):
        parts.append(f'teacher={Path(dis["teacher_model_path"]).name}')
    if dis.get("offload_teacher_model") is not None:
        parts.append(f'offload_teacher={dis["offload_teacher_model"]}')
    ds = (t.get("msswift") or {}).get("deepspeed")
    if ds:
        parts.append(f"deepspeed={ds}")
    if (t.get("native") or {}).get("allow_unverified_pp_rollout"):
        parts.append("allow_unverified_pp_rollout=true")
    return ";".join(parts) if parts else PLACEHOLDER


def spec_json_of(t):
    """B 块机读 JSON（生成器直接消费）。紧凑、无空格。"""
    obj = {
        "tier_id": t["tier_id"], "applicable": t["applicable"], "status": t["status"],
        "model": t.get("model"), "model_path": t.get("model_path"),
        "train_method": t.get("train_method"), "tuner_type": t.get("tuner_type"),
        "backend": t.get("backend"),
        "parallel": {"scheme": t["parallel"].get("scheme"),
                     "tp_size": t["parallel"].get("tp_size"), "dp_size": t["parallel"].get("dp_size"),
                     "pp_size": t["parallel"].get("pp_size"), "cards": t["parallel"].get("cards"),
                     "gpus": t["parallel"].get("gpus"),
                     "semantics": t["parallel"].get("semantics")},
        "target_preset": t.get("target_preset"), "target_modules": t.get("target_modules"),
        "target_modules_mode": t.get("target_modules_mode"),
        "learning_rate": t.get("learning_rate"),
        "data": {"train_path": (t.get("data") or {}).get("train_path"),
                 "subset_size": (t.get("data") or {}).get("subset_size"),
                 "prompts_per_rank": (t.get("data") or {}).get("prompts_per_rank"),
                 "source_form": (t.get("data") or {}).get("source_form"),
                 "subset_generation": (t.get("data") or {}).get("subset_generation")},
        "max_epochs": t.get("max_epochs"),
        "expected_optimizer_steps_per_epoch": t.get("expected_optimizer_steps_per_epoch"),
        "expected_optimizer_steps_reachable": t.get("expected_optimizer_steps_reachable"),
        "save_steps": t.get("save_steps"), "lora": t.get("lora"),
        "steps_formula_v2": t.get("steps_formula_v2"),
        "msswift": t.get("msswift"), "native": t.get("native"),
        "reward_kind": t.get("reward_kind"), "pretrain": t.get("pretrain"), "distill": t.get("distill"),
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="evidence/spec-54.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preserve-from", default=None)
    ap.add_argument("--source-path", default=".local/repo-moved-out/docs/capability-matrix-20261006-r3.html")
    a = ap.parse_args()

    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    prev = {}
    if a.preserve_from and Path(a.preserve_from).exists():
        pd = json.loads(Path(a.preserve_from).read_text(encoding="utf-8"))
        prev = {r["tier"]: r for r in pd.get("rows", [])}

    rows, preserved = [], []
    for t in spec["tiers"]:
        d = t.get("data") or {}
        row = {
            "tier": t["tier_id"],
            # --- 人读：主要训练参数（10 子项） ---
            "model": v(t.get("model")),
            "algo": ALGO_DISPLAY.get(t.get("train_method"), v(t.get("train_method"))),
            "mode": MODE_DISPLAY.get(t.get("tuner_type")) or v(t.get("mode_raw")),
            "backend": BACKEND_DISPLAY.get(t.get("backend"), v(t.get("backend"))),
            "parallel": parallel_of(t),
            "lora_target": lora_target_of(t),
            "lr": v(t.get("learning_rate")),
            "data_subset": data_subset_of(t),
            "steps": steps_of(t, d.get("planned_steps") if "planned_steps" in d else None),
            "switches": switches_of(t),
            # --- 机读：JSON 列 ---
            "spec_json": spec_json_of(t),
            # --- 机读扁平字段（yaml 生成器口径） ---
            "model_path": v(t.get("model_path")),
            "train_method": v(t.get("train_method")),
            "tuner_type": v(t.get("tuner_type")),
            "cards": v((t.get("parallel") or {}).get("cards")),
            "parallel_dims": v((t.get("parallel") or {}).get("summary")),
            "gpus": json.dumps((t.get("parallel") or {}).get("gpus"), ensure_ascii=False),
            "parallel_scheme": v((t.get("parallel") or {}).get("scheme")),
            "target_modules": json.dumps(t.get("target_modules"), ensure_ascii=False),
            "target_modules_mode": v(t.get("target_modules_mode")),
            "deepspeed": v((t.get("msswift") or {}).get("deepspeed")),
            "nnodes": v((t.get("msswift") or {}).get("nnodes")),
            "allow_unverified_pp_rollout": v((t.get("native") or {}).get("allow_unverified_pp_rollout")),
            "subset_generation": v((t.get("data") or {}).get("subset_generation")),
            "subset_r3_historical": v((t.get("data") or {}).get("subset_r3_historical")),
            "expected_optimizer_steps_historical": v(t.get("expected_optimizer_steps_historical")),
            "steps_formula_v2": v(t.get("steps_formula_v2")),
            "min_optimizer_steps": v(t.get("min_optimizer_steps")),
            "data_source": data_subset_of(t),
            "subset_size": v(d.get("subset_size")),
            "prompts_per_rank": v(d.get("prompts_per_rank")),
            "train_path": v(d.get("train_path")),
            "planned_steps": v(d.get("planned_steps")),
            "save_steps": v(t.get("save_steps")),
            "expected_optimizer_steps_per_epoch": v(t.get("expected_optimizer_steps_per_epoch")),
            "expected_optimizer_steps_reachable": v(t.get("expected_optimizer_steps_reachable")),
            "applicable": t["applicable"],
            "spec_status": t["status"],
        }
        for k in RESULT_KEYS:
            row[k] = PLACEHOLDER
        p = prev.get(t["tier_id"])
        if p and any(p.get(k) not in ("", PLACEHOLDER) for k in RESULT_KEYS):
            for k in RESULT_KEYS:
                row[k] = p.get(k, PLACEHOLDER)
            if p.get("run_provenance"):
                row["run_provenance"] = p["run_provenance"]
            preserved.append(t["tier_id"])
        rows.append(row)

    doc = {
        "schema": "matrix-data/v4",
        "meta": {
            "title": "GRASPO 能力矩阵",
            "data_revision": "2026-10-11-main-params-v4",
            "criteria_version": "v3-2026-10-10（主席拍定）+ U2 起点/终点窗口比较口径",
            "results_policy": "结果列只允许经 backfill_matrix.py 由 task-matrix-runs/<档号>.json 回填；"
                              "未跑档位保持 '—'；禁止手工回填、禁止手改 HTML",
            "result_placeholder": PLACEHOLDER,
            "common_params_policy": "对全部档位相同的参数集中在 §1「统一口径」块（本文件 uniform 字段）一次写清；"
                                    "矩阵表（rows[]）只承载档位之间的差异项",
            "lora_restoration": "★ LoRA 目标恢复既有正式口径：target_modules=[language_all_linear, vision_common]"
                                "（122 件）、r=128/alpha=256/dropout=0.05、lr=1.0e-05；出处 = 8 个 git 跟踪示例配置",
            "subset_size_semantics": "权威含义 = 全局总条数；派生 prompts_per_rank = subset_size // cards",
            "authoritative_source": {
                "path": a.source_path,
                "sha256": spec["authoritative_source"]["sha256"],
                "basename": Path(a.source_path).name,
                "role": "权威工作件线末节点（r3）",
            },
            "spec_source": a.spec,
        },
        "groups": GROUPS,
        "columns": COLUMNS,
        "uniform": {"rule": "★ 矩阵表只列档位之间的差异项：下列参数对全部档位相同，集中在此写一次，不在表中逐档重复。",
                    "items": spec["uniform"]},
        "provenance": PROVENANCE,
        "invariants": spec.get("invariants", []),
        "conflicts": [{"id": c["id"], "level": c["level"], "title": c["title"]} for c in spec.get("conflicts", [])],
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    filled = [r["tier"] for r in rows if any(r[k] != PLACEHOLDER for k in RESULT_KEYS)]
    print(f"rows={len(rows)} 结果列非空档数={len(filled)} {filled}（--preserve-from 保留 {len(preserved)} 档）")
    print(f"列数={len(COLUMNS)} uniform={len(spec['uniform'])} 项；wrote {a.out}")


if __name__ == "__main__":
    main()
