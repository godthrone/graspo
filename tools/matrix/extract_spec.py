#!/usr/bin/env python3
"""extract_spec.py — 抽 54 档 spec（A 统一口径层 + B 逐档差异层），并登记缺项与冲突。

★ 唯一真相源口径：功能矩阵 HTML + matrix-data.json 共为唯一真相源；本脚本把
  「r3 权威件 + 归档生成器 + v1/v2 档位配置 + 8 个 git 跟踪示例 + not_applicable 档」抽成
  结构化 spec，交给 build_data.py 填进矩阵文档。⛔ 缺项/冲突**显式列出**，不猜填。

来源（全部只读）：
  ① r3 权威工作件        .local/repo-moved-out/docs/capability-matrix-20261006-r3.html（sha 见 params-raw）
  ② 档位配置 v2（冻结）   samples/configs/matrix54-v2/T0NN.yaml（51 档；T052–T054 为 .not_applicable.md）
  ③ v1 基线配置          samples/configs/matrix54/T0NN.yaml（对照，仅 train_path 前缀不同）
  ④ 归档生成器           tests/e2e/generate_matrix.py（GPU_SETS / SUBSET_SIZE_BY_ALGORITHM /
                         ROLLOUT_GROUP_SIZE / NATIVE_ROLLOUT_QUEUE_BATCH_SIZE / ELAM_* 常量；
                         lora_target_preset 规则 = 偏离根因）
  ⑤ 8 个 git 跟踪示例     samples/configs/a800x*_qwen35_*.yaml（既有正式 LoRA 口径出处）
  ⑥ rig/configs-v2       ⛔ 不存在（find 未命中）⇒ 来源不可用，已登记

用法：
  python3 extract_spec.py --params evidence/params-raw.json \
      --config-dir ../.../matrix54-v2 --out evidence/spec-54.json [--conflicts-out evidence/spec-conflicts.md]
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

import extract_params as ep

REPO = Path(__file__).resolve().parents[2]   # 仓库根（本脚本位于 tools/matrix/）
GEN = REPO / "tests/e2e/generate_matrix.py"

# ★ 主席 2026-10-11 裁定：LoRA 恢复既有正式口径（出处 = 8 个 git 跟踪示例配置）
RESTORED_LORA = {
    "target_modules": ["language_all_linear", "vision_common"],   # 122 件 = language 12 + visual 110
    "target_preset": None,
    "r": 128, "alpha": 256, "dropout": 0.05,
}

# ══ ★★ 主席 2026-10-15 裁定：lr =「一个方法一个值」，⛔ 不按卡数缩放 ══════════════
# 逐字裁定：
#   ·「针对不同的训练方式，给它不同的步长。比如：SFT 的步长要长一点，强化学习的步长要短一点。」
#   ·「这种东西，既然我们取消了，那就直接让这个机制消失就完了，干嘛后再除一次？」
# ⇒「按卡数缩放学习率」这个机制（其配置字段与计算方法）已**整体删除**；
#   lr 与 dp_size / tp_size / pp_size / 卡数**无关**，本表是 lr 的**唯一真相源**。
# 取值一律锚定实测，⛔ 不许发明数字（逐条依据见 LR_BASIS）。
LR_POLICY: dict[tuple[str, str], str] = {
    ("sft", "lora"): "2.0e-05",
    ("sft", "full"): "5.0e-06",
    ("graspo", "lora"): "5.0e-06",
    ("graspo", "full"): "5.0e-06",
    ("cpt", "lora"): "5.0e-06",
    ("cpt", "full"): "5.0e-06",
    ("opd", "lora"): "2.0e-05",
    ("opd", "full"): "5.0e-06",
}
LR_BASIS: dict[tuple[str, str], str] = {
    ("sft", "lora"): ("SFT-LoRA 2.0e-05 = 刚达标档 T011/T023 的**实际生效值**"
                      "（跑次 params.lr=1.0e-05 × DP=2；Δ=+11.2536pp / +10.6837pp ≥ +10.0pp）"),
    ("sft", "full"): "SFT 全参 5.0e-06 = 与 CPT 对齐（矩阵无 SFT 全参通过档，取同族保守值）",
    ("graspo", "lora"): ("GRASPO(RL)-LoRA 5.0e-06 = 跑次实测值；原在盘 1.0e-05 是"
                         "「LoRA 统一 1.0e-05」误抬值"),
    ("graspo", "full"): "GRASPO(RL) 全参 5.0e-06 = 跑次实测值",
    ("cpt", "lora"): "CPT 5.0e-06 = ★ 保持现状不动（CPT 9/9 档已通过，⛔ 不折腾能工作的东西）",
    ("cpt", "full"): "CPT 5.0e-06 = ★ 保持现状不动（CPT 9/9 档已通过，⛔ 不折腾能工作的东西）",
    ("opd", "lora"): "OPD-LoRA 2.0e-05 = 与 SFT 同族同量级",
    ("opd", "full"): "OPD 全参 5.0e-06 = 与 SFT 全参同族同量级",
}
#: 仓内原口径的**逐字**出处（报告须引用；本表与之同向，但 LoRA 档差 4×、全参档 1×）。
LEGACY_CALIBER = {
    "text": "RL 推荐比 SFT 低一个数量级",
    "sources": [
        "samples/configs/a800x1_qwen35_9b_rl.yaml:51",
        "samples/configs/a800x1_qwen35_27b_rl.yaml:54",
        "samples/configs/rl_example.yaml:50",
    ],
    "deviation": ("同向；SFT-LoRA(2e-05) vs RL-LoRA(5e-06) = **4×**（不是 10×，⛔ 不硬凑）；"
                  "全参档两者同为 5e-06（1×）。锚点：SFT-LoRA 取「刚达标的实际生效值 2e-05」，"
                  "而非示例里的 5e-05"),
}
#: ★ 主席已裁定「不做」的事项（登记进报告与口径节，避免后人再提）。
RULED_OUT = [
    "不补 epoch：多张卡不需要补 epoch（训练量不由卡数决定）。",
    "不做「按全局步数定长」。",
    "不做任何「按卡数补偿训练量」的机制，也不做任何「按卡数缩放/补偿 lr」的机制。",
]


def normalize_tuner(tuner_type: str | None) -> str:
    """``None`` ⇒ ``"lora"``（与 ``graspo.core.schema.resolve_tuner_type`` 同语义）。"""
    return "lora" if tuner_type is None else str(tuner_type)


def lr_for(train_method: str | None, tuner_type: str | None) -> str:
    """取该 (方法, 模式) 的学习率。

    ★ 显式 fail-closed：未知组合**当场报错**，⛔ 不静默回退默认值（宪法 §2.3 边界校验即防呆
    ——静默回退会让「一个方法一个值」的约束悄悄失效）。
    """
    key = (str(train_method), normalize_tuner(tuner_type))
    if key not in LR_POLICY:
        raise KeyError(
            f"lr policy: no learning rate defined for train_method={key[0]!r} / "
            f"tuner_type={key[1]!r}; known combos = {sorted(LR_POLICY)}"
        )
    return LR_POLICY[key]


def lr_basis(train_method: str | None, tuner_type: str | None) -> str:
    """取该组合的取值依据（供 spec / 报告引用）。"""
    return LR_BASIS[(str(train_method), normalize_tuner(tuner_type))]


# 统一口径块（spec A 清单）——config 项来自 51 份 yaml 全同；rule 项为规则口径
UNIFORM = [
    {"key": "reward.kind", "label": "奖励类型 reward.kind", "value": "graspo",
     "layer": "config", "coverage": "config-18/54", "note": "仅 GRASPO 的 18 档适用；其余档不适用"},
    {"key": "temperature", "label": "评测 / 采样温度（锁定）", "value": "0",
     "layer": "rule", "coverage": "rule-54/54", "note": "评测口径"},
    {"key": "criteria_id", "label": "评测 criteria 标识", "value": "elam-v3-allright-1",
     "layer": "rule", "coverage": "rule-54/54", "note": "与 T028 跑次 effect 字段所记一致"},
    {"key": "criteria_version", "label": "文档判据版本",
     "value": "v3-2026-10-10（主席拍定）+ U2 起点/终点窗口比较口径",
     "layer": "rule", "coverage": "rule-54/54", "note": "见实验条件（判据版本）"},
    {"key": "seed", "label": "随机种子 seed", "value": "42", "layer": "config",
     "coverage": "config-51/54", "note": "51 份 yaml 全为 42；与 8 个示例的 seed=43 属不同语境（仅登记差异）"},
    {"key": "max_prompt_length", "label": "上下文 / 序列上限（plen）", "value": "8192",
     "layer": "config", "coverage": "config-51/54", "note": "r3 seq_len_cap 亦为 8192"},
    {"key": "torch_dtype", "label": "权重精度", "value": "bfloat16", "layer": "config",
     "coverage": "config-51/54", "note": ""},
    {"key": "gradient_checkpointing", "label": "梯度检查点 gradient_checkpointing", "value": "true",
     "layer": "config", "coverage": "config-51/54", "note": ""},
    {"key": "micro_batch_size", "label": "单卡 micro batch size（★ 后端异名：native = micro_batch_size／ms-swift = per_device_train_batch_size）",
     "value": "1", "layer": "config", "coverage": "config-51/54",
     "note": "★ 后端异名必须显式写出，否则生成器需猜：native.micro_batch_size = ms-swift.msswift.per_device_train_batch_size = 1"},
    {"key": "gradient_accumulation_micro_batches", "label": "梯度累积 micro-batches（两后端同名字段，显式写出）", "value": "1",
     "layer": "config", "coverage": "config-51/54", "note": "r3 grad_accum 亦为 1"},
    {"key": "save_checkpoint_every_epoch", "label": "每 epoch 存 ckpt（两后端同名字段，显式写出）", "value": "false",
     "layer": "config", "coverage": "config-51/54", "note": "51 份 yaml 全为 false"},
    {"key": "gpu_monitor.interval_sec", "label": "GPU 巡检间隔（秒）", "value": "2.0",
     "layer": "config", "coverage": "config-51/54", "note": "51 份 yaml 全为 2.0"},
    {"key": "train_source_form", "label": "数据源形式 + 图像根",
     "value": "mini（默认，跑通用）| formal（正式集）；图像根 <data-root>/images，jsonl 内写作 ../images/…",
     "layer": "rule", "coverage": "rule-54/54", "note": "mini 结果不得当效果结论"},
    {"key": "eval_set_path", "label": "固定评测集路径",
     "value": "<data-root>/test.jsonl（702 行）",
     "layer": "rule", "coverage": "rule-54/54", "note": "见判定口径明细"},
]


def gen_constants():
    """从归档生成器读常量（正则抽取，不 import：避免拉起重依赖）。"""
    text = GEN.read_text(encoding="utf-8")
    def grab(pat, cast=str):
        m = re.search(pat, text, re.M)
        return cast(m.group(1)) if m else None
    gpu_sets_raw = grab(r'^GPU_SETS[^=]*=\s*(\{[^}]*\})')
    gpu_sets = {}
    if gpu_sets_raw:
        for k, v in re.findall(r'(\d+):\s*\(([^)]*)\)', gpu_sets_raw):
            gpu_sets[int(k)] = [int(x) for x in v.split(",") if x.strip()]
    subset_raw = grab(r'^SUBSET_SIZE_BY_ALGORITHM[^=]*=\s*(\{[^}]*\})')
    subset_by_algo = json.loads(subset_raw) if subset_raw else {}
    return {
        "GPU_SETS": gpu_sets,
        "SUBSET_SIZE_BY_ALGORITHM": subset_by_algo,
        "ROLLOUT_GROUP_SIZE": grab(r'^ROLLOUT_GROUP_SIZE\s*=\s*(\d+)', int),
        "NATIVE_ROLLOUT_QUEUE_BATCH_SIZE": grab(r'^NATIVE_ROLLOUT_QUEUE_BATCH_SIZE\s*=\s*(\d+)', int),
        "MATRIX_MAX_EPOCHS": grab(r'^MATRIX_MAX_EPOCHS\s*=\s*(\d+)', int),
        "ELAM_CONTAINER_ROOT": grab(r'^ELAM_CONTAINER_ROOT\s*=\s*"([^"]+)"'),
        "ELAM_TEST_JSONL": grab(r'^ELAM_TEST_JSONL\s*=\s*f"\{ELAM_CONTAINER_ROOT\}([^"]*)"'),
        "ELAM_TEST_COUNT": grab(r'^ELAM_TEST_COUNT\s*=\s*(\d+)', int),
        "lora_target_preset_rule": ("有视觉塔 且 数据含图 ⇒ vision_common；否则 language_safe"
                                    "（generate_matrix.py:lora_target_preset()）"),
    }


def read_example_lora():
    """读 8 个 git 跟踪示例配置，确认「既有正式口径」的取值与件数。"""
    files = sorted(REPO.glob("samples/configs/a800x*_qwen35_*.yaml"))
    recs = []
    for f in files:
        flat = ep.parse_yaml_lite(f)  # 只用它解析简单键
        text = f.read_text(encoding="utf-8")
        m = re.search(r'target_modules:\s*\[([^\]]*)\]', text)
        tm = [x.strip() for x in m.group(1).split(",")] if m else None
        recs.append({
            "file": str(f.relative_to(REPO)),
            "r": flat.get("lora_r"),
            "alpha": flat.get("lora_alpha"),
            "dropout": flat.get("lora_dropout"),
            "learning_rate": flat.get("learning_rate"),
            "target_modules": tm,
        })
    return recs


def sc(v):
    """None/空 -> None；否则字符串。"""
    if v is None or str(v).strip() in ("", "—", "null"):
        return None
    return str(v).strip()


def num(v):
    """把 '100' / '1.0e-05' 之类尽量转成数值，失败返回原字符串。"""
    v = sc(v)
    if v is None:
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v


def expected_steps(algo, backend, tuner, cards, subset_size, q, g, max_epochs):
    """移植 generate_matrix.py 的步数口径（三条路径显式分派）。

    · SFT/CPT/OPD           每 epoch = floor(subset_size / cards)
    · GRASPO × ms-swift     每 epoch = floor(subset_size × G / cards)   （数据集按 G 重复）
    · GRASPO × native       每 epoch = floor(floor(subset_size/cards) / Q) + 1（阈值触发 + epoch 末 flush）
    · _reachable            = 每 epoch × max_epochs
    """
    if not subset_size or not cards:
        return None, None
    cards = max(1, int(cards))
    if algo == "graspo":
        if backend == "native":
            per = max(1, (int(subset_size) // cards) // q + 1)
        else:
            per = max(1, (int(subset_size) * g) // cards)
    else:
        per = max(1, int(subset_size) // cards)
    return per, per * int(max_epochs or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conflicts-out", default=None)
    a = ap.parse_args()

    raw = json.loads(Path(a.params).read_text(encoding="utf-8"))
    g = gen_constants()
    examples = read_example_lora()
    v1_dir = REPO / "samples/configs/matrix54"
    v2_dir = Path(a.config_dir)

    conflicts, missing = [], []

    # ---- 冲突 1：LoRA 口径（v2 yaml 偏离 vs 恢复值） ----
    dev = Counter()
    for rec in raw["tiers"]:
        c = rec["from_config"]
        if c.get("tuner_type") == "lora":
            dev[(c.get("lora_r"), c.get("lora_alpha"), c.get("lora_dropout"),
                 c.get("lora_target"), c.get("learning_rate"))] += 1
    conflicts.append({
        "id": "C1", "level": "★★ 主席已裁",
        "title": "LoRA 目标口径：v2 档位 yaml 偏离既有正式口径",
        "detail": (f"v2 yaml（{len([r for r in raw['tiers'] if r['from_config'].get('tuner_type')=='lora'])} 个 LoRA 档）取值分布 = "
                   + "；".join(f"r={k[0]}/α={k[1]}/dropout={k[2]}/preset={k[3]}/lr={k[4]} × {n}"
                               for k, n in dev.most_common())),
        "evidence": "samples/configs/matrix54-v2/T0NN.yaml（lora 段）",
        "ruling": ("恢复既有正式口径：target_modules=[language_all_linear, vision_common]（122 件 = language 12 + visual 110）、"
                   "r=128 / alpha=256 / dropout=0.05；"
                   "★ lr **不再随 LoRA 口径统一**（原「统一 1.0e-05」已被主席 2026-10-15 裁定推翻）"
                   "——改按 (方法, 模式) 从 LR_POLICY 取值，⛔ 不按卡数缩放"),
        "basis": f"{len(examples)} 个 git 跟踪示例配置 " + "、".join(x["file"] for x in examples),
        "root_cause": "生成器 " + g["lora_target_preset_rule"] + " 把「训视觉塔」与「训语言层」隐含对立",
    })

    # ---- 冲突 2：v1 ↔ v2 仅 train_path 前缀 ----
    diffs = []
    for rec in raw["tiers"]:
        v1, v2 = v1_dir / f"{rec['tier']}.yaml", v2_dir / f"{rec['tier']}.yaml"
        if v1.exists() and v2.exists():
            t1 = v1.read_text(encoding="utf-8").replace("subsets/", "subsets-v2/")
            t2 = v2.read_text(encoding="utf-8")
            # 去掉头部注释行后比较
            b1 = "\n".join(l for l in t1.splitlines() if not l.startswith("#"))
            b2 = "\n".join(l for l in t2.splitlines() if not l.startswith("#"))
            if b1 != b2:
                diffs.append(rec["tier"])
    conflicts.append({
        "id": "C2", "level": "登记",
        "title": "v1（samples/configs/matrix54）↔ v2（matrix54-v2）差异",
        "detail": (f"51 档训练语义字段逐字节相同；仅 data.train_path 前缀 <data-root>/subsets/ → <data-root>/subsets-v2/。"
                   f"去前缀后仍不一致的档：{diffs or '无'}"),
        "evidence": "diff v1/v2（本脚本逐档复核）", "ruling": "照实登记；矩阵采用 v2 路径", "basis": "指挥官 2026-10-11 确认",
    })

    # ---- 冲突 3：8 个示例的 seed/max_epochs/grad_accum 与矩阵不同语境 ----
    ex_seed = sorted({str(x.get("seed")) for x in [ep.parse_yaml_lite(REPO / e["file"]) for e in examples]})
    conflicts.append({
        "id": "C3", "level": "登记（不改口径）",
        "title": "8 个示例配置的 seed / max_epochs / grad_accum 与矩阵统一口径不同语境",
        "detail": f"示例（a800x*）seed={ex_seed}、max_epochs=20、grad_accum=4；矩阵统一口径 seed=42、max_epochs=1、grad_accum=1",
        "evidence": "、".join(x["file"] for x in examples),
        "ruling": "只登记为显式差异，不改统一口径（示例是 4/8 卡长跑语境，矩阵是逐档跑通语境）",
        "basis": "指挥官 2026-10-11 确认",
    })

    # ---- 冲突 4：subset_size —— r3 逐档值 vs 生成器默认 ----
    ss_diff = []
    for rec in raw["tiers"]:
        h = rec["from_html"]
        algo_name = {"CPT": "CPT", "SFT": "SFT", "GRASPO": "GRASPO", "OPD": "OPD"}.get(h.get("algo"))
        gen_default = g["SUBSET_SIZE_BY_ALGORITHM"].get(algo_name)
        html_val = sc(h.get("subset_size"))
        if html_val and gen_default is not None and int(html_val) != gen_default:
            ss_diff.append((rec["tier"], h.get("algo"), html_val, gen_default))
    conflicts.append({
        "id": "C4", "level": "★ 已裁（单一权威含义）",
        "title": "subset_size 语义与取值：r3 逐档值 vs 生成器默认",
        "detail": (f"权威含义 = 全局总条数（⑤ 裁定）；另给派生 prompts_per_rank = subset_size // cards。"
                   f"r3 与生成器默认 SUBSET_SIZE_BY_ALGORITHM={g['SUBSET_SIZE_BY_ALGORITHM']} 不一致的档（{len(ss_diff)} 个）："
                   + "、".join(f"{t}/{al}({v} vs {d})" for t, al, v, d in ss_diff[:12])
                   + ("…" if len(ss_diff) > 12 else "")),
        "evidence": "r3 <td data-col='subset_size'> + generate_matrix.py:SUBSET_SIZE_BY_ALGORITHM",
        "ruling": "矩阵采用 r3 逐档值（权威件），并逐档同时给 prompts_per_rank；生成器默认仅作对照",
        "basis": "指挥官 2026-10-11 裁定⑤",
    })

    # ---- 缺项 / 不可用来源 ----
    missing.append({"id": "M1", "title": "rig/configs-v2 不存在",
                    "detail": "find 未命中；该来源不可用",
                    "substitute": "归档生成器 tests/e2e/generate_matrix.py + v1 基线配置 + 8 个 git 跟踪示例"})
    na = [r["tier"] for r in raw["tiers"] if not r.get("config_present")]
    missing.append({"id": "M2", "title": f"无 yaml 的档：{na}",
                    "detail": "T052/T053/T054 为 .not_applicable.md（法定终态，非缺项）",
                    "substitute": "status=not_applicable / applicable=false；理由见 .not_applicable.md"})
    missing.append({"id": "M3", "title": "gpus(list) 不在 r3 HTML",
                    "detail": "r3 无 gpus 列",
                    "substitute": f"generate_matrix.py:GPU_SETS={g['GPU_SETS']} + gpu_assignment.check_gpus()（按 cards 派生）"})

    # ---- 逐档 B 块 ----
    tiers = []
    for rec in raw["tiers"]:
        h, c = rec["from_html"], rec["from_config"]
        tier = rec["tier"]
        cards = num(h.get("cards"))
        algo = c.get("train_method") or {"CPT": "cpt", "SFT": "sft", "GRASPO": "graspo", "OPD": "opd"}.get(h.get("algo"))
        backend = {"ms-swift": "msswift", "native": "native"}.get(h.get("backend"), sc(h.get("backend")))
        tuner = sc(c.get("tuner_type"))
        applicable = bool(rec.get("config_present"))
        subset_size = num(h.get("subset_size"))
        max_epochs = (num(h.get("max_epochs")) or g["MATRIX_MAX_EPOCHS"]) if applicable else None
        per, reach = expected_steps(algo, backend, tuner, cards, subset_size,
                                    g["NATIVE_ROLLOUT_QUEUE_BATCH_SIZE"], g["ROLLOUT_GROUP_SIZE"], max_epochs)
        lora = None
        target_modules = target_preset = None
        # ★ lr 由 (方法, 模式) **唯一确定** —— 与卡数无关、与 v2 yaml 原值无关（⛔ 不做 dp 反除/
        #   正乘任何二次修正）。非适用档不写 lr。
        lr = lr_for(algo, tuner) if applicable else None
        lr_basis_text = lr_basis(algo, tuner) if applicable else None
        if tuner == "lora" and applicable:
            lora = {"r": RESTORED_LORA["r"], "alpha": RESTORED_LORA["alpha"], "dropout": RESTORED_LORA["dropout"]}
            target_modules = list(RESTORED_LORA["target_modules"])
        msswift = native = None
        if backend == "msswift" and applicable:
            # ★ ms-swift：无 dp_size（MsSwiftConfig 不含该字段）；卡数/进程数由 nproc_per_node(×nnodes) 表达
            msswift = {"nproc_per_node": num(c.get("msswift_nproc")),
                       "nnodes": num(c.get("msswift_nnodes")),
                       "per_device_train_batch_size": num(c.get("msswift_per_device_bs")),
                       "sequence_parallel_size": num(c.get("msswift_seq_parallel")),
                       "use_vllm": sc(c.get("msswift_use_vllm")),
                       "deepspeed": sc(c.get("msswift_deepspeed"))}
        if backend == "native" and applicable:
            # ★ native：tp/dp/pp + micro_batch_size；allow_unverified_pp_rollout 为逐档开关（仅 T035/T036）
            native = {"tp_size": num(c.get("native_tp")), "dp_size": num(c.get("native_dp")),
                      "pp_size": num(c.get("native_pp")), "micro_batch_size": num(c.get("native_micro_bs")),
                      "allow_unverified_pp_rollout": sc(c.get("native_allow_unverified_pp_rollout"))}
        gpus = g["GPU_SETS"].get(int(cards)) if (cards and not applicable is False) else None
        source_form = None
        if sc(h.get("data_source")):
            source_form = "mini" if "mini" in h["data_source"] else ("formal" if "正式集" in h["data_source"] else None)
        spec = {
            "tier_id": tier,
            "applicable": applicable,
            "status": "ready" if applicable else "not_applicable",
            "model": sc(h.get("model")),
            "mode_raw": sc(h.get("mode")),
            "model_path": sc(c.get("model_path")),
            "train_method": algo,
            "tuner_type": tuner,
            "backend": backend,
            "parallel": {
                "scheme": ("native: tp_size/dp_size/pp_size" if backend == "native"
                           else ("msswift: nproc_per_node(+nnodes)；★ ms-swift 无 dp_size" if backend == "msswift"
                                 else None)),
                "tp_size": (native or {}).get("tp_size"),
                "dp_size": (native or {}).get("dp_size"),
                "pp_size": (native or {}).get("pp_size"),
                "cards": cards,
                "gpus": gpus,
                "summary": sc(h.get("parallelism")),
                "semantics": ("native: dp_size = 数据并行副本数；ms-swift: nproc_per_node = torchrun 每节点进程数"
                              "（world_size = nproc_per_node × nnodes）。★ 单位都是「卡/进程数」，但字段名与语义不同 ⇒ 不可互填"),
                "gpus_source": "manifest tiers[].gpus（现行 override 后的落盘值）；★ 卡池口径 2026-10-11 已改，依赖 B10",
            },
            "target_preset": target_preset,
            "target_modules": target_modules,
            "target_modules_mode": ("multi_preset" if target_modules else None),
            "target_modules_semantics": ("list[str]：逐元素先按 preset 名展开、再 glob；★ 优先于 target_preset；"
                                         "与 target_preset(单值) 必须分开承载，否则生成器无法判定展开 1 个还是 N 个 preset"),
            "steps_basis": "manifest expected_optimizer_steps_per_epoch（v2 现行权威；native GRASPO 走 E-50 锚定覆盖）",
            "e54_note": ("历史键 expected_optimizer_steps 语义 = checkpoint_save_steps（ckpt 间隔），"
                         "⛔ 不是计划步数；计划步数看 expected_optimizer_steps_per_epoch（E-54）"),
            "learning_rate": lr,
            "learning_rate_basis": lr_basis_text,
            "data": {
                "train_path": sc(c.get("train_path")),
                "subset_size": subset_size,
                "prompts_per_rank": (int(subset_size) // int(cards)) if (subset_size and cards) else None,
                "source_form": source_form,
                "data_source": sc(h.get("data_source")),
                "planned_steps": num(h.get("planned_steps")),
            },
            "max_epochs": max_epochs,
            "expected_optimizer_steps_per_epoch": per,
            "expected_optimizer_steps_reachable": reach,
            "save_steps": num(c.get("save_steps")),
            "lora": lora,
            "msswift": msswift,
            "native": native,
            "reward_kind": sc(c.get("reward_kind")),
            "pretrain": ({"loss_scale": sc(c.get("pretrain_loss_scale")),
                          "use_chat_template": sc(c.get("pretrain_use_chat_template"))}
                         if sc(c.get("pretrain_loss_scale")) else None),
            "distill": ({"teacher_model_path": sc(c.get("distill_teacher")),
                         "offload_teacher_model": sc(c.get("distill_offload_teacher"))}
                        if sc(c.get("distill_teacher")) else None),
            "lora_deviation": ({"v2_yaml": {"target_preset": sc(c.get("lora_target")),
                                            "r": num(c.get("lora_r")), "alpha": num(c.get("lora_alpha")),
                                            "dropout": num(c.get("lora_dropout")),
                                            "learning_rate": sc(c.get("learning_rate"))},
                                "restored": dict(RESTORED_LORA, learning_rate=lr)}
                               if (tuner == "lora" and applicable) else None),
        }
        tiers.append(spec)

    # ---- ★★ 用 manifest v2（tests/e2e/matrix54_manifest_source.json）校准为「现行权威」 ----
    MANIFEST = REPO / "tests/e2e/matrix54_manifest_source.json"
    mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mf_by = {t["tier_id"]: t for t in mf["tiers"]}
    for t in tiers:
        m = mf_by.get(t["tier_id"])
        if not m:
            continue
        r3_sub = (t.get("data") or {}).get("subset_size")
        man_sub = (m.get("data") or {}).get("subset_size")
        cards = (t.get("parallel") or {}).get("cards")
        t["parallel"]["gpus"] = m.get("gpus")
        t["parallel"]["gpus_source"] = ("manifest tiers[].gpus（现行 override 后落盘值）；"
                                        "★ 卡池口径 2026-10-11 已改，依赖 B10")
        if man_sub is not None:
            t["data"]["subset_size"] = man_sub
            t["data"]["prompts_per_rank"] = (int(man_sub) // int(cards)) if cards else None
        t["data"]["subset_r3_historical"] = r3_sub
        t["data"]["subset_generation"] = (
            "E-49/E-55 现行世代（每卡 72；E-55 登记 2026-10-02）" if t["tier_id"] in ("T040", "T041", "T042")
            else ("现行（manifest v2 与 r3 同值）" if man_sub == r3_sub else "后台账世代（与 r3 不同）"))
        t["expected_optimizer_steps_per_epoch"] = m.get("expected_optimizer_steps_per_epoch")
        t["expected_optimizer_steps_reachable"] = m.get("expected_optimizer_steps_reachable")
        if m.get("checkpoint_save_steps") is not None:
            t["save_steps"] = m.get("checkpoint_save_steps")
        t["expected_optimizer_steps_historical"] = m.get("expected_optimizer_steps")
        t["steps_formula_v2"] = m.get("step_formula_v2")
        t["min_optimizer_steps"] = ((m.get("acceptance") or {}).get("formal_gate") or {}).get("min_optimizer_steps")
        t["applicable"] = bool(m.get("applicable", t["applicable"]))
        t["status"] = m.get("status") or t["status"]

    # E-55 世代切换登记（前像/后像 sha；★ T040 两世代都是 72 ⇒ 判别点看 T041/T042）
    E55 = {"T040": {"before_sha": "1c41f2ce…", "after_sha": "3345a0dc…", "r3": 72, "current": 72},
           "T041": {"before_sha": "818db067…", "after_sha": "f508185f…", "r3": 96, "current": 144},
           "T042": {"before_sha": "629bf997…", "after_sha": "79b8c814…", "r3": 192, "current": 288}}
    for t in tiers:
        if t["tier_id"] in E55:
            t["data"]["subset_e55"] = E55[t["tier_id"]]

    # ---- 冲突 5/6/7 + 不变量（并入跑次侧移交件 handover-to-matrix.md 的素材） ----
    conflicts.append({
        "id": "C5", "level": "★ 待 B10（不据旧值下结论）",
        "title": "gpus(list) 来源过期：退役生成器默认 vs 现行卡池口径",
        "detail": ("GPU_SETS={1:[0],2:[0,1],4:[0,1,2,3]} 是退役生成器默认；"
                   "现行 core/gpu_guard.py:ALLOWED_INDICES 默认 = 全卡 0–7（B10 2026-10-12 解禁卡池闸门），且主席已改口径为"
                   "「全卡先到先得、util 0% 且剩余显存够」—— 2026-10-17 主席令已定案（逐字：「228 先到先得，随便用。」＋「单个作业不限制卡数。」）"),
        "evidence": "tests/e2e/generate_matrix.py:GPU_SETS；跑次侧移交件 §2.1「卡池」",
        "ruling": "矩阵 gpus 列标为「仅参考 / 待 B10」；cards 仍为权威差异项",
        "basis": "跑次侧审计 + 指挥官 2026-10-11 口径变更",
    })
    e49_r3 = {t["tier_id"]: (t["data"]["subset_size"]) for t in tiers if t["tier_id"] in ("T040", "T041", "T042")}
    conflicts.append({
        "id": "C6", "level": "★★ 冲突待回执",
        "title": "E-49 放大量：r3 权威件 vs 跑次侧移交件",
        "detail": (f"r3 逐档值为 {e49_r3}；跑次侧移交件 §2.1 记 E-49 放大量 = 72/144/288（T040/T041/T042）"
                   "⇒ T042 上 r3=192 与移交件=288 不一致"),
        "evidence": "r3 <td data-col='subset_size'> + handover-to-matrix.md §2.1「E-49 例外」",
        "ruling": "矩阵暂采用 r3 逐档值（权威件），并把移交件值登记在 conflicts；★ 请主席/spec 回执以哪个为准",
        "basis": "⑤ 裁定「权威含义 = 全局总条数」但未裁定 E-49 三档的具体数值",
    })
    conflicts.append({
        "id": "C7", "level": "登记（已知偏离 4 条）",
        "title": "跑次侧移交件登记的 4 条已知偏离",
        "detail": ("① 矩阵 54 档全部只用单一 target_preset（无 target_modules 列表）——本条即 C1，按裁定恢复；"
                   "② T040/T041/T042 save_steps 48→72（已在 v2 yaml 头注登记）；"
                   "③ T035/T036 新增 native.allow_unverified_pp_rollout: true；"
                   "④ v2 生成器已退役（be20d9b, 09-29）⇒ 冻结件当前不可由仓库重算"),
        "evidence": "handover-to-matrix.md §2.2",
        "ruling": "① 已裁恢复；②③ 照实登记、矩阵按 v2 取值；④ 正是本单「文档为唯一真相源」要解的钥匙",
        "basis": "跑次侧移交件 §2.2 + 指挥官确认",
    })

    byid = {c["id"]: c for c in conflicts}
    byid["C4"].update({
        "level": "★★ 已回执（并行按后端分列，不可互填）",
        "ruling": ("native 用 tp_size/dp_size/pp_size；ms-swift 用 nproc_per_node(+nnodes)（world_size = nproc_per_node × nnodes）。"
                   "★ MsSwiftConfig 不含 dp_size ⇒ ms-swift 档 dp_size 保持 null，⛔ 不可用 nproc_per_node 同填/互换"),
        "basis": "跑次侧回执 spec-ack-to-matrix.md §C4 + core/schema.py::MsSwiftConfig + _config_mapping.py:566"})
    byid["C5"].update({
        "level": "★ 已回执（现状登记 + B10 依赖）",
        "ruling": ("gpus 唯一定档 = manifest tiers[].gpus（现行 override 后的落盘值，如 T028=[4]/T041=[3,5]/T042=[1,3,4,5]）；"
                   "B10 依赖：idle 阈值 + runner 硬编码 {4,5,6,7} + B7 的 27 条契约测试再推导；★ B10 必须排在批与批之间"),
        "basis": "跑次侧回执 §C5 + tests/e2e/matrix54_manifest_source.json"})
    byid["C6"].update({
        "level": "★★★ 已回执（288 为现行权威）",
        "ruling": ("现行权威 = T040 72 / T041 144 / T042 288（冻结 yaml + manifest v2 + 磁盘实文件三方一致）；"
                   "r3 的 192 是「E-49 前」历史值（同一档、不同世代）。文档 data_subset 定为 72/144/288 并登记 E-49/E-55 世代切换与前像/后像 sha"),
        "basis": ("CORRECTIONS.md E-49/E-55（2026-10-02；前像 sha T040 1c41f2ce…/T041 818db067…/T042 629bf997…，"
                  "后像 3345a0dc…/f508185f…/79b8c814…）+ 跑次侧回执 §C6")})

    conflicts.append({
        "id": "C8", "level": "★ 已按 manifest 校准（登记旧变体）",
        "title": "步数口径：manifest 现行权威 vs 退役生成器旧公式（+1）",
        "detail": ("退役生成器 native_graspo_steps_per_epoch = floor(P/Q)+1（T028 得 7）；"
                   "manifest v2 现行权威 expected_optimizer_steps_per_epoch：T028 = 6（= 实际跑次 6 步）、"
                   "T040/T041/T042 = 6（E-50 锚定覆盖；未锚定公式 ceil(72/8)=9，见 step_formula_v2 尾注）"),
        "evidence": "tests/e2e/matrix54_manifest_source.json（expected_optimizer_steps_per_epoch / step_formula_v2）+ T028 跑次 6/6 步",
        "ruling": "矩阵一律采用 manifest 值（T028=6 等），退役生成器的 +1 变体仅登记；★ E-55 失效条件：日后子集再放大必须重复 E-50 锚定，否则 ① 会被公式算回 9 而永不闭合",
        "basis": "跑次侧回执 §C6.3（E-54/E-55）+ manifest",
    })

    # 不变量：save_steps（= manifest checkpoint_save_steps）必须 > 可达步数上限
    inv_violations, inv_equal = [], []
    for t in tiers:
        ss, reach = t.get("save_steps"), t.get("expected_optimizer_steps_reachable")
        if ss is None or reach is None:
            continue
        if ss < reach:
            inv_violations.append({"tier": t["tier_id"], "save_steps": ss, "reachable": reach})
        elif ss == reach:
            inv_equal.append({"tier": t["tier_id"], "save_steps": ss, "reachable": reach})
    invariants = [{
        "id": "I1", "name": "save_steps > 可达步数上限（否则会落中间 ckpt）",
        "statement": ("training.save_steps（= manifest checkpoint_save_steps）必须 > expected_optimizer_steps_reachable，"
                      "否则会落下中间 step_<N> ckpt，破坏「每档只留一份 ckpt」"),
        "source": "manifest key_semantics + handover-to-matrix.md §2.1「不变量」",
        "violations": inv_violations,
        "equal_cases": inv_equal,
        "result": (f"★ {len(inv_violations)} 档不满足（须回执）" if inv_violations
                   else (f"无严格小于；{len(inv_equal)} 档相等（save_steps == 可达上限 ⇒ 保存点即最后一步，不产生中间 ckpt）"
                         if inv_equal else "全部严格满足")),
    }, {
        "id": "I2", "name": "步数公式 = 上限（不是保证）",
        "statement": ("SFT/CPT/OPD 每 epoch = floor(subset_size/cards)；GRASPO×ms-swift = floor(subset_size×G/cards)；"
                      "GRASPO×native = floor(floor(subset_size/cards)/Q)+1（阈值触发 + epoch 末 flush）。"
                      "★ 该值是**上限**：retry/invalid/perfect_skip 不进 buffer 时实际更低。"
                      "实测：T028 上限 7、实际 6 步（exit_code=0）"),
        "source": "generate_matrix.py:expected_optimizer_steps_per_epoch/_reachable + T028 跑次",
        "result": "已按此口径落字段，并在文档标明「上限」",
    }]

    doc = {
        "schema": "matrix-spec/v1",
        "built_on": "2026-10-11",
        "authoritative_source": {
            "html": raw["authoritative_html"], "sha256": raw["authoritative_html_sha256"],
            "config_dir_v2": str(v2_dir), "config_dir_v1": str(v1_dir),
            "generator": str(GEN.relative_to(REPO)),
        },
        "generator_constants": g,
        "restored_lora": RESTORED_LORA,
        # ★ lr 策略（唯一真相源）——随 spec 落盘，供 build_data / 报告引用
        "lr_policy": {
            "ruling": ("主席 2026-10-15：lr =「一个方法一个值」，⛔ 不按卡数缩放；"
                       "原「按卡数缩放学习率」机制的配置字段与计算方法已整体删除"),
            "table": {f"{m}|{t}": v for (m, t), v in LR_POLICY.items()},
            "basis": {f"{m}|{t}": b for (m, t), b in LR_BASIS.items()},
            "legacy_caliber": LEGACY_CALIBER,
            "ruled_out": RULED_OUT,
        },
        "example_configs": examples,
        "uniform": UNIFORM,
        "conflicts": conflicts,
        "missing": missing,
        "invariants": invariants,
        "handover_source": ".local/hb-workspace/<会话>/task-run-T028/handover-to-matrix.md（已并入）",
        "tiers": tiers,
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"spec: {len(tiers)} 档（applicable={sum(1 for t in tiers if t['applicable'])}）；"
          f"uniform {len(UNIFORM)} 项；冲突 {len(conflicts)} 条；缺项 {len(missing)} 条 -> {outp}")

    if a.conflicts_out:
        lines = ["# spec 抽取——缺项与冲突登记（54 档）", "",
                 f"来源：r3 权威件（sha256 `{raw['authoritative_html_sha256']}`）+ "
                 f"`samples/configs/matrix54-v2/`（v2 冻结）+ v1 基线 + 归档生成器 + 8 个 git 跟踪示例。",
                 "", "## 冲突（⛔ 不猜填；已裁的写明裁定与依据）", ""]
        for x in conflicts:
            lines += [f"### {x['id']} · {x['title']}（{x['level']}）", f"- 事实：{x['detail']}",
                      f"- 证据：{x['evidence']}", f"- 裁定/处置：{x['ruling']}", f"- 依据：{x['basis']}", ""]
        lines += ["## 缺项 / 不可用来源", ""]
        for x in missing:
            lines += [f"### {x['id']} · {x['title']}", f"- 事实：{x['detail']}", f"- 替代：{x['substitute']}", ""]
        lines += ["## 不变量核验", ""]
        for x in invariants:
            lines += [f"### {x['id']} · {x['name']}", f"- 口径：{x['statement']}", f"- 出处：{x['source']}",
                      f"- 结果：{x['result']}", ""]
        Path(a.conflicts_out).write_text("\n".join(lines), encoding="utf-8")
        print(f"conflicts written -> {a.conflicts_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
