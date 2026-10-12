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
    {"key": "tp", "label": "TP", "group": "param"},
    {"key": "pp", "label": "PP", "group": "param"},
    {"key": "dp", "label": "DP", "group": "param"},
    {"key": "cards_total", "label": "总卡数", "group": "param"},
    {"key": "lora_target", "label": "LoRA 目标", "group": "param"},
    {"key": "lr", "label": "lr", "group": "param"},
    {"key": "data_scale", "label": "数据规模", "group": "param"},
    {"key": "steps", "label": "步数/epoch", "group": "param"},
    {"key": "switches", "label": "关键开关", "group": "param"},
    {"key": "peak_mem", "label": "显存峰值(GiB)", "group": "data", "group_start": True},
    {"key": "steps_actual", "label": "实跑步数", "group": "data"},
    {"key": "elapsed", "label": "耗时（每步/总时长）", "group": "data"},
    {"key": "throughput", "label": "吞吐", "group": "data"},
    {"key": "effect", "label": "效果（评测 Δ / loss·reward 窗口）", "group": "data"},
    {"key": "note", "label": "备注（条件 / 限制 / 失败归因）", "group": "note", "group_start": True},
]

DATA_KEYS = ["peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note"]


#: ★ 并行维度的固定口径（写进表前「统一口径」块；⛔ 不进表格列）
PARALLEL_UNIFORM = [
    {"key": "parallel_relation", "label": "总卡数与并行维度的关系式",
     "value": "总卡数 = nproc_per_node × nnodes = TP × PP × CP × DP",
     "layer": "rule", "coverage": "rule-54/54",
     "note": "两个后端的配置面不同，见下两行；表中 TP / PP / DP / 总卡数 四列即由此式给出"},
    {"key": "native_parallel", "label": "native 后端的并行配置面",
     "value": "显式配置 tp_size / dp_size / pp_size",
     "layer": "rule", "coverage": "rule-54/54",
     "note": "表中 TP / PP / DP 三列对 native 档均为**配置值**"},
    {"key": "msswift_parallel", "label": "ms-swift 后端的并行配置面",
     "value": "不暴露 dp_size；world_size = nproc_per_node × nnodes；DP 由框架内部（DeepSpeed）决定",
     "layer": "rule", "coverage": "rule-54/54",
     "note": "★ 防呆：ms-swift 档**不得**设 dp_size（设了会直接报错）；因此表中 ms-swift 档的 DP 为**派生值**并显式标注，TP / PP 记「—（框架内部决定）」"},
    {"key": "cp", "label": "CP（上下文并行）",
     "value": "全部档 = 1", "layer": "rule", "coverage": "rule-54/54",
     "note": "全档同值 ⇒ ⛔ 不占列（只列差异项）"},
    {"key": "sp", "label": "SP（序列并行）",
     "value": "全部档 = 关闭（不切分）", "layer": "rule", "coverage": "rule-54/54",
     "note": "全档同值 ⇒ ⛔ 不占列；ms-swift 的 sequence_parallel_size = 1 即不切分，native 档未配置"},
]


def parallel_cols(r, state):
    """★ 并行维度列（TP / PP / DP / 总卡数）——由生成器拼装，⛔ 不手写。

    配置值 vs 派生值必须区分（§ 主席裁决）：
      · native：TP / PP / DP = **配置值**；总卡数 = TP×PP×CP×DP；
      · ms-swift：DP = **派生值**（nproc_per_node × nnodes）显式标注；TP / PP = 「—（框架内部决定）」；
        总卡数 = nproc_per_node × nnodes（标注来源）。
    """
    if state in (S_UNVERIFIED, S_EXCLUDED) and r.get("cards") in (None, "", "—"):
        return {"tp": "—", "pp": "—", "dp": "—", "cards_total": "—"}
    backend = r.get("backend")
    if backend == "native":
        tp, pp, dp = r.get("tp_size"), r.get("pp_size"), r.get("dp_size")
        cp = r.get("cp_size", "1")
        cards = r.get("cards")
        if tp in (None, "—") or pp in (None, "—") or dp in (None, "—"):
            return {"tp": "—", "pp": "—", "dp": "—", "cards_total": f"—（{cards} 卡）" if cards else "—"}
        total = f"{tp}×{pp}×{cp}×{dp} = {cards} 卡" if cards not in (None, "—") else f"{tp}×{pp}×{cp}×{dp}"
        return {"tp": str(tp), "pp": str(pp), "dp": str(dp), "cards_total": total,
                "dp_derived": False, "dp_source": "配置值（native.tp_size / pp_size / dp_size）"}
    if backend == "ms-swift":
        nproc = r.get("nproc_per_node")
        if nproc in (None, "", "—"):
            cards = r.get("cards")
            return {"tp": "—（框架内部决定）", "pp": "—（框架内部决定）",
                    "dp": "—（框架内部决定）", "cards_total": f"—（{cards} 卡）" if cards else "—"}
        nn = r.get("nnodes")
        nn = 1 if nn in (None, "", "—") else nn
        world = int(nproc) * int(nn)
        return {"tp": "—（框架内部决定）", "pp": "—（框架内部决定）",
                "dp": f"{world}（派生）",
                "cards_total": f"nproc_per_node×nnodes = {world} 卡",
                "dp_derived": True,
                "dp_source": "派生值（nproc_per_node × nnodes；ms-swift 不暴露 dp_size）"}
    return {"tp": "—", "pp": "—", "dp": "—", "cards_total": "—"}


#: ★ epoch 口径（主席裁定）：SFT / CPT / OPD = 1；GRASPO(RL) = 3（RL 需要更多 epoch 才能看到效果）
EPOCHS_BY_METHOD = {"sft": 2, "cpt": 2, "opd": 2, "graspo": 8}

#: ★ 取消"计划步数上限"：步数由 数据量 × epoch 推导，不设人为上限
STEPS_POLICY = "步数 = epoch 数 × 每 epoch 推导步数（由数据量与批大小推导）；⛔ 不设人为步数上限（无 max_steps 约束）。"


#: ★ 生成 yaml 所需的其余固定口径（LoRA 恢复值 / 推理与序列并行开关）
EXTRA_UNIFORM = [
    {"key": "lora_r", "label": "LoRA rank r（恢复口径）", "value": "128",
     "layer": "config", "coverage": "config-33/54", "note": "仅 LoRA 档；全参档不适用"},
    {"key": "lora_alpha", "label": "LoRA alpha（恢复口径）", "value": "256",
     "layer": "config", "coverage": "config-33/54", "note": "仅 LoRA 档"},
    {"key": "lora_dropout", "label": "LoRA dropout（恢复口径）", "value": "0.05",
     "layer": "config", "coverage": "config-33/54", "note": "仅 LoRA 档"},
    {"key": "sequence_parallel_size", "label": "序列并行尺寸（ms-swift）", "value": "1",
     "layer": "config", "coverage": "config-33/54", "note": "1 = 不切分；与 §统一口径 的 SP 一致"},
    {"key": "use_vllm", "label": "训练期启用 vLLM（ms-swift）", "value": "false",
     "layer": "config", "coverage": "config-33/54", "note": "推理验证阶段另起服务，见结果列"},
]


#: ★ LoRA 目标：native preset 名 → **解析后的具体模块名**（按本仓文档化的匹配语汇逐项构造）
#:   语言 12 项（attn 4 + 线性注意力 3+2 + MLP 3）+ 视觉 110 项（merger 2 + 27 层 × 4）= 122
LANGUAGE_TARGETS = [
    "language.full_attn.q_proj", "language.full_attn.k_proj",
    "language.full_attn.v_proj", "language.full_attn.o_proj",
    "language.linear_attn.q_proj", "language.linear_attn.k_proj", "language.linear_attn.v_proj",
    "language.linear_attn.in_proj_z", "language.linear_attn.out_proj",
    "language.mlp.gate_proj", "language.mlp.up_proj", "language.mlp.down_proj",
]
VISION_TARGETS = (["visual.merger.linear_fc1", "visual.merger.linear_fc2"]
                  + [f"visual.blocks.{i}.{m}" for i in range(27)
                     for m in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")])
RESOLVED_LORA_TARGETS = LANGUAGE_TARGETS + VISION_TARGETS        # 122（native canonical）

#: ★ ms-swift 的真实 HF 正则 = 仓内 13 份曾训练成功的 adapter 的 target_modules 形态 ∪ 视觉部分
#:   （证据见 lora_target_backends() 的 evidence 字段）
#:  ★ 修正记录（2026-10-12）：上一版有两处硬伤 ⇒ 探针实测**语言分支命中 0** ⇒ 用旧版生成的那批 yaml **已不可用**。
#:    ① 前缀：本机真实根前缀是 `language_model.layers`（旧写 `model.language_model.*` ⇒ 命中 0）；
#:    ② `(?=\.)` 是**零宽断言**、其后紧跟 `.*\.` ⇒ 语言分支**逻辑上不可能匹配**（对照实验证明这是主因）。
#:  修正后探针实测：语言 **248** + 视觉 **110** = **358**（证据 sha 1c3830b1…）。
#:  ★ 收紧版（2026-10-12，探针实测 358）：视觉侧**显式枚举** blocks/merger 的 Linear，
#:    ⛔ 不用过宽的 `visual\..*`（实测 528 命中，含 170 个非 Linear + Conv3d 1 ⇒ 已弃用）。
HF_LORA_TARGET_REGEX = (
    r"^(?:model\.)?language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_z|out_proj|in_proj_qkv|"
    r"in_proj_a|in_proj_b|gate_proj|up_proj|down_proj)$|"
    r"^(?:model\.)?visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc1|mlp\.linear_fc2)|"
    r"merger\.linear_fc1|merger\.linear_fc2)$"
)


def lora_target_backends():
    """两后端的 LoRA 目标表达与**真实覆盖集**（★ 矩阵为源；生成器据此分流，⛔ 不手写）。

    ★★ 结论：**两后端覆盖集不相等**，且**三方命名空间各不相同**（⇒ 不能跨后端共用同一串目标）。
      · native 公开签名 = `language.*`（**真存在**，由 T010/T011 checkpoint 的
        `lora_target_signature.resolved` 实测为 **122 = 语言 12 + 视觉 110**；
        字段名 `lora_target_signature.resolved` / `parameter_count`，⛔ 语义按字段名判、不猜）；
      · native 内部存储键名 = `…token_mixer.*`（LoRA 权重键名可见）；
      · HF / ms-swift = `language_model.layers.N.linear_attn.*`（探针实测 `language.*` 命中 **0**）。
    """
    return {
        "native": {
            "kind": "preset_names",
            "expression": ["language_all_linear", "vision_common"],
            "resolved": list(RESOLVED_LORA_TARGETS),
            "resolved_count": len(RESOLVED_LORA_TARGETS),
            "resolved_count_meaning": "签名/模式条目数（⛔ 不是实例数）；实例数见 coverage.total",
            "coverage": {"language": 248, "visual": 110, "total": 358,
                         "signature_entries": 122,
                         "basis": ("★ 口径订正（2026-10-12）：`lora_target_signature.resolved = 122` 是**签名/模式条目数**"
                                   "（12 条「无 layer 索引的语言模式」+ 110 条视觉实例），**不是件数**；"
                                   "12 条语言模式**逐层展开为 248 个实例** ⇒ 与 ms-swift **同为 358 个实例**"
                                   "（张量键数同为 716）。依据 `task-run-T012/evidence/13-lora-target-count.md`。")},
            "language_targets": list(LANGUAGE_TARGETS),
            "signature_evidence": {
                "field": "lora_target_signature.resolved / parameter_count",
                "runs": "T010（1 卡）与 T011（2 卡）签名完全相同",
                "parameter_count": 409923584,
                "note": "★ 该值即字段 `parameter_count` 的原值，语义按字段名判（⛔ 不推测）。",
            },
            "namespace": "language.* / visual.*（native canonical，公开签名）",
            "note": ("native 侧支持 preset 名（glob），由本仓 lora 模块解析。"
                     "★ 实测证明 `language_all_linear` **确实命中语言层 12 项、不是空转**"
                     "（与 T010 实测 Δ=+13.39 pp 互证：语言层空转不可能有该增益）。"),
        },
        "msswift": {
            "kind": "hf_regex",
            "expression": [HF_LORA_TARGET_REGEX],
            "resolved": [],
            "resolved_count": None,
            "coverage": {"status": "measured", "language": 248, "visual": 110, "total": 358,
                         "param_count": 410120192,
                         "basis": ("★ 探针实测（证据：evidence/45-lora-target-probe-newregex.md "
                                   "sha 1c3830b119946946fcfbbae0aa8e29ea6e1184c4c99b434e83ce8e4deeef34e5）。"
                                   "语言 248 分解：q/k/v/o_proj 各 8｜in_proj_qkv 24、in_proj_a/b 各 24｜"
                                   "in_proj_z 24｜out_proj 24｜gate/up/down_proj 各 32。"
                                   "★ 收紧版视觉侧显式枚举 ⇒ 命中 358 全为 nn.Linear（Conv3d 0）；"
                                   "旧过宽版 visual\\..* 实测 528 命中（含 170 非 Linear）⇒ 已弃用。")},
            "escape_note": ("转义：正则含反斜杠，YAML 出参用**单引号标量**（字面量）⇒ 加载值应为**单反斜杠**。"
                                   "★ 验收以**加载值**为准（GraspoConfig 加载后 repr），⛔ 不看文件文本。"),
            "comparability_note": ("★ 两后端 **LoRA 实例数同为 358**（language 248 + visual 110），但**命中的模块集合不同**"
                                   "（native `token_mixer.*` vs ms-swift `linear_attn.*`/`full_attn.*`）⇒ 两后端的效果/性能读数**不直接可比**，跨后端比较须注明。"),
            "target_regex_mechanism": {
                "field": "lora.target_regex（ms-swift 官方字段 swift/arguments/tuner_args.py:127；:201-202 覆盖 target_modules）",
                "shape_chain": ("仓内 lora.target_modules 是 list → graspo_to_ms_swift_argv() 出口是 str → "
                                "PEFT LoraConfig.__post_init__ 把 list 变 set → swift/tuners_utils.py:34 的 "
                                "isinstance(..., str) 判否 ⇒ 落「字面/点后缀」分支 ⇒ **正则永不命中**"),
                "fix": ("映射层识别「形如正则的单元素列表/单 str」并改发官方 --target_regex ⇒ PEFT 收 str ⇒ 走正则分支；"
                        "非正则的预设名/后缀名列表仍走 --target_modules（行为不变）"),
                "constraints": "⛔ 不改 ms-swift 源码、⛔ 不手改 yaml；native 路径不受影响",
                "commit": "30a8e15d588a8a763ffa40d21b2a40422b29b6f1",
                "evidence": "evidence/54-… sha 0e4f5998…",
            },
            "guard_fix": {
                "function": "merged_export.py::targets_use_language_model_namespace()",
                "str_branch": "正则分支改用 re.fullmatch 实测（原先按字面/后缀判断 ⇒ 误判）",
                "list_branch": "原行为不变",
                "fail_closed": "非法 / 不可判定一律拒绝",
                "retained": "★ 保留防呆：「仅视觉正则」仍被拒绝",
                "commit": "2381c54c…",
                "evidence": "evidence/56-… sha d28213b2…",
            },
            "equivalence_with_native": False,
            "evidence": ["checkpoint lora_target_signature（native 侧 122 实测）",
                         "evidence/44-lora-target-probe-report.md sha e20c3cf6…",
                         "收紧版探针 sha 1c3830b1… / 转义夹逼 sha e28314a5…",
                         "evidence/54-… sha 0e4f5998…（target_regex 机制）",
                         "evidence/56-… sha d28213b2…（守卫修正）",
                         "提交 30a8e15…（target_regex）/ 2381c54c…（守卫）",
                         "本地 13 份曾训练成功的 ms-swift adapter_config.json（本正则形态实证）",
                         "T010 实测 Δ=+13.39 pp（与 native 签名互证）"],
            "namespace": "language_model.layers.N.* / visual.*（HF 真实名）",
            "note": ("ms-swift 不认 native preset 名（另一套语汇 ⇒ `ValueError: Target modules not found`）"
                     "⇒ 用**真实 HF 名正则**（仓内 13 份曾训练成功的 adapter 的既有形态 ∪ 视觉）。"
                     "★★ **两后端不等价（结论不变，依据订正）**：★ 实例数**同为 358**（language 248 + visual 110；"
                     "张量键数同为 716），参数量 native 409,923,584 vs ms-swift 410,120,192（差 196,608 ≈0.048%）"
                     "—— ⛔ **不得再写「122 件 vs 358 件」**；★ 不等价的依据 = **命中的模块集合不同**："
                     "native `layers.N.token_mixer.{k,q,v,in_proj_z,out_proj}` + `mlp.*`；"
                     "ms-swift `language_model.layers.N.linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}` "
                     "+ `full_attn.{q,k,v,o}` + `mlp.*` ⇒ 跨后端读数不直接可比。"),
        },
        "namespaces": {
            "native_public": "language.* / visual.*（checkpoint 签名）",
            "native_internal": "…token_mixer.*（LoRA 权重键名）",
            "hf_msswift": "language_model.layers.N.*（HF 真实名；探针实测 language.* = 0）",
            "conclusion": "★ 三方命名空间各不相同 ⇒ **不能跨后端共用同一串 LoRA 目标**。",
        },
    }


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
        "threshold": {
            "name": "delta_pp", "unit": "百分点（pp）",
            "value": 5.0,
            "target": 10.0,
            "applies_to": ("训练类（SFT / CPT / OPD）与 GRASPO(RL)：判据② = after > base 且 Δ ≥ +5.0 pp"),
            "status": ("★ 已冻结（主席定案）：**训练类与 GRASPO 的通过线均为 δ = +5.0 pp**；"
                       "**目标值 = +10.0 pp（仅展示，⛔ 不用于判定）**——"
                       "Δ 落在 5.0–10.0 之间仍判**通过**，不判不可用。"),
            "historical_note": "该值取代此前跑次所用的临时阈值 20 pp（未冻结，已不再作为判据）。",
        },
        "baseline": {"role": "同档位未训练底座（role=base）",
                     "caliber": "U2 终点窗口 = 末段 max(3, 10% 步数) 步的均值；该档的 epoch 数与推导步数见档位「步数/epoch」子项。"},
        "epoch_policy": {
            "by_method": ("SFT / CPT / OPD = 2 epoch（⛔ 不动）；GRASPO(RL) = 8 epoch（主席令；若 8 仍不过线则升 10 为后备；理由：RL 耗时长、5 epoch 下效果仅差 0.44 pp 未过线）。理由：跨 epoch 容易出问题（循环结构决定）⇒ 2 epoch 正好覆盖一次 epoch 边界，这正是要压的地方；RL 需要更多 epoch 才能看到效果。"),
            "steps": STEPS_POLICY + " 推导式：steps = epoch 数 × 每 epoch 步数（每 epoch 步数由数据量与批大小推导）。",
            "checkpoint": "save_steps = 推导步数 + 1（大于总步数 ⇒ 只保留最终检查点）。",
            "known_issue": ("★★ 登记待修（未修）：推导步数公式对 native GRASPO 不成立，根因已读清：两个后端的 step 语义本就不同。\n"
                            "① ms-swift：步数 = epochs × (subset × num_generations) / (micro_batch × grad_accum)；锚：T031（5ep）推导 800 = 实际 800 ✓。\n"
                            "② native：步数 = replay 缓冲区触发次数（optimize.py::_maybe_optimize）；阈值 = rollout_queue_batch_size × rollout_group_size（core/schema.py:458 / ripple/group_decision.py:152），每次仅消耗整组（whole_group_count，组不拆），epoch 尾可有 force 刷尾；★ 与 micro_batch/grad_accum 解耦。\n"
                            "③ native 锚的正确账（口径订正）：**阈值 = Q×G = 8×8 = 64**"
            "（`runs/T028/config.yaml:159-160`；运行期 19/19 步均为 64 }交叉验证）；"
            "步数由**可训练补全数**决定：可训练组 147 × 8 = **1176** ⇒ 1176/64 = 18.4 "
            "⇒ **19 步**（末步可含 force_flush），与实测吻合。"
            "★ 此前推理中的 ≈29 是 **尝试组/步**（attempt_groups 547/19≈28.8，含 retry/perfect_skip/invalid）"
            "—— **不是阈值**；把它当阈值是旧推导 30≠19 的原因（把“尝试组”当成了“可训练组”）。"
            "★ 但**可训练补全数是运行期结果**（取决于 retry/perfect_skip），"
            "⇒ **无法从配置先验算出**；⛔ 公式仍不改，维持“native 以实跑 run_cumulative.step 为准”。"),
        },
            "step_reconciliation": {
                "nominal": ("**名义步数** = 矩阵按 epoch 口径推导的步数（写入 yaml 的 steps / save_steps）；"
                            "⛔ 对 native GRASPO 仅为名义值，不是承诺。"),
                "reconciliation_rule": ("**对账式** = 用运行期量校验实跑步数；两个后端各自复现本后端实际值："
                                        "① ms-swift（配置式）steps = epochs × (subset × num_generations) / "
                                        "(micro_batch × grad_accum)；② native（运行期式）steps = "
                                        "ceil(trainable_completions / (rollout_queue_batch_size × rollout_group_size))，"
                                        "本仓 Q×G = 8×8 = 64。"),
                "worked_example": ("T028（native, 5ep）：可训练组 147 × G=8 = **1176** ⇒ ceil(1176/64) = **19** "
                                   "= 实测 19 ✓；ms-swift 锚 T031（5ep）：5 × (20×8)/(1×1) = **800** = 实测 800 ✓。"),
                "input_note": ("★ native 对账式**必须**输入运行期量 `trainable_completions`（可训练补全数，"
                               "取决于 retry / perfect_skip / invalid）⇒ 不可先验计算；回填 json 未提供该字段时，"
                               "对账记录标为「缺输入 ⇒ 待补」，⛔ 不臆测、⛔ 不硬编码。"),
            },
        "conclusion_rule": {
            "rule": ("结论必须与同一行的参数一一对应。若一行的参数已被更正，而结论来自旧参数下的跑次 ⇒ "
                     "该结论作废，该档标「尚未验证」；仅在参数冻定后重跑，才可给出「已验证可用 / 已验证不可用」判定。"),
            "status_wording": ("「尚未验证」在本文档中等同「未测试」；宪法 §17.7 的固定取值为「尚未验证」，用词以它为准。"),
        },
            "caliber_evidence": [
                "★★ **噪声底 ≈ ±0.43 pp**（两次独立 base 评测：同底座 / 同 702 条 `c202b1e1…` / 同 temp 0·top_p 0.9·"
                "max_tokens 128·seed 42 / 同 criteria）：**116/702 = 16.5242%** vs **113/702 = 16.0969%**；"
                "逐样本 GT 不一致 **0**；预测 (name,action) 不同 **21/702 = 2.99%**；all_right 翻转 **5/702**。"
                "定性 = **vLLM 连续批处理浮点非确定性**。证据件 `13-base-rerun-variance.md` "
                "sha `449d5f7253860934558bca732f37d8fd71bf13c91e2f02dcd91827f961888620`。"
                "★ 用途：供「基准散布 / 阈值余量」判断；★ **δ 判定不变，⛔ 不因此改阈值**。",
            ],
            "caliber_corrections": [
                "★ **口径订正记录（2026-10-12）**：原写法「native 122 件 vs ms-swift 358 件」**有误**——"
                "`lora_target_signature.resolved = 122` 是**签名/模式条目数**（12 条语言模式 + 110 条视觉实例），"
                "其中 12 条语言模式实际展开为 **248** 个逐层实例 ⇒ **两后端实例数同为 358**、张量键数同为 716。"
                "依据：`task-run-T012/evidence/13-lora-target-count.md`（双路实测）。"
                "★ 结论「两后端不等价」**仍成立**，依据改为**命中的模块集合不同**（native `token_mixer.*` vs "
                "ms-swift `linear_attn.*`/`full_attn.*`）。",
            ],
            "known_corrections": [
                "★ 待订正：T028.json 的数据锚 `eval_dataset.sha256` 写作 `6da72199…`，实为 train.jsonl；正确的评测集锚 = `c202b1e1…`。"
                "评测执行者正在订正 ⇒ 订正后其 json sha 变化，我方将**重跑一次回填**（幂等按 sha，自动更新）。",
        ],
    }


def verdict_of(row, state):
    """已验证不可用档的结论：写明未过的是哪一条判据，并保留已通过项的事实（**数据驱动**）。"""
    if state != S_UNAVAILABLE:
        return None
    eff = str(row.get("effect", ""))
    m = re.search(r"Δ\s*=?\s*\+?([0-9.]+)\s*pp", eff)
    th = re.search(r"\+([0-9.]+)\s*pp\s*冻结阈值", eff)
    detail = (f"Δ = +{m.group(1)} pp < 冻结阈值 +{th.group(1)} pp" if (m and th)
              else (f"Δ = +{m.group(1)} pp（低于冻结阈值）" if m else "效果未达标（见 effect）"))
    passed = []
    algo = str(row.get("algo_checks", ""))
    if algo and algo not in ("—", ""):
        passed.append(algo)
    return {"state": state,
            "failed_criterion": "第 2 条：在平台方数据集与平台方参数下效果未达标",
            "failed_detail": detail,
            "passed_criteria": passed or ["判据① 通过（见 runs）"]}


def split_elapsed(tp):
    """由跑次读数文本抽出「耗时」口径（总时长 / 步内累计）。抽不到记 '—'（不编造）。"""
    if not tp or tp in ("—",):
        return "—"
    m1 = (re.search(r"墙钟\s*([0-9]{2}:[0-9]{2}:[0-9]{2})", tp)
          or re.search(r"墙钟\s*([0-9.]+\s*s)", tp))
    m2 = re.search(r"(?:elapsed_sec\s*累计|步内累计)\s*([0-9.]+)\s*s", tp)
    m3 = re.search(r"每步(?:约|平均)?\s*([0-9.]+)\s*s", tp)
    parts = []
    if m2:
        parts.append(f"步内累计 {m2.group(1)} s")
    if m1:
        parts.append(f"总时长 {m1.group(1)}")
    if m3:
        parts.append(f"每步约 {m3.group(1)} s")
    return "；".join(parts) if parts else "—"


def split_throughput(tp):
    """由跑次读数文本抽出「吞吐」口径（decode / 整步）。抽不到记 '—'（不编造）。"""
    if not tp or tp in ("—",):
        return "—"
    m0 = re.search(r"不适用（([^）]*)）", tp)
    if m0:
        return f"不适用（{m0.group(1)}）"      # ★ 如实在注明，⛔ 不填假数
    parts = []
    m1 = re.search(r"((?:decode|解码)\s*[0-9.]+\s*tok/s)", tp) or re.search(r"([0-9.]+\s*tok/s)", tp)
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
        elif isinstance(raw, str) and raw.startswith("失败"):
            state = S_UNAVAILABLE
        else:
            state = S_UNVERIFIED
        # ★ 结论有效性：结论必须与同一行的参数一一对应；参数已更正而结论来自旧参数 ⇒ 结论作废
        ann = r.get("conclusion_annotation")
        voided = bool(ann) and ann.get("conclusion_valid") is False
        if voided:
            state = S_UNVERIFIED
            r["voided_conclusion"] = ann
        r["status"] = state
        r.pop("spec_status", None)

        v = verdict_of(r, state)
        # ★ 读数已录但阈值未定（measurement_pending）⇒ 保留读数列，状态仍「尚未验证」（读数不作判定）
        pending = bool(r.get("measurement_pending"))
        concluded = (state in CONCLUDED) or pending
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
        # ★ epoch 口径（按方法）+ 步数推导（取消人为上限）
        method = (r.get("train_method") or "").strip().lower()
        epochs = EPOCHS_BY_METHOD.get(method)
        per_epoch = r.get("expected_optimizer_steps_per_epoch")
        if epochs and per_epoch not in (None, "", "—"):
            derived = int(per_epoch) * epochs
            r["max_epochs"] = epochs
            r["derived_steps"] = derived
            r["per_epoch_steps"] = int(per_epoch)
            r["steps"] = f"{epochs} epoch × {per_epoch} 步/epoch = 推导 {derived} 步"
            r["save_steps"] = derived + 1
            r["save_steps_basis"] = (f"= 推导步数 {derived} + 1（⛔ 不设人为步数上限；"
                                     f"save_steps 大于总步数 ⇒ 只留最终检查点）")
        # ★ 备注补充：LoRA 按后端的真实覆盖（对 LoRA 档；数据驱动的口径事实，⛔ 不手写）
        if r.get("tuner_type") == "lora" and r.get("note") not in (None, "", "—"):
            lb = lora_target_backends()
            r["note"] = r["note"] + (" ★ LoRA 目标按后端：ms-swift 实测 %d 实例（language %d + visual %d）｜native **同为 %d 实例**"
                                     "（122 为签名/模式条目数）⇒ ★ 两后端**实例数相同但命中集合不同（不等价）**。"
                                     % (lb["msswift"]["coverage"]["total"], lb["msswift"]["coverage"]["language"],
                                        lb["msswift"]["coverage"]["visual"], lb["native"]["coverage"]["total"]))
        # ★ 实跑步数（有跑次记录即填，含"作废参考记录"）
        scr = r.get("step_count_reconciliation") or {}
        if scr.get("actual_steps") is not None:
            r["steps_actual"] = f"{scr.get('actual_steps')}（推导 {scr.get('derived_steps')}）"
        else:
            m = re.search(r"(\d+)\s*/\s*(\d+)\s*步", r.get("runs", "") or "")
            r["steps_actual"] = f"{m.group(1)}/{m.group(2)}" if m else "—"
        # 数据规模 = 全局条数 + 每卡派生
        if r.get("subset_size") not in (None, "", "—"):
            ppr = r.get("prompts_per_rank")
            r["data_scale"] = f"{r['subset_size']} 条" + (f"（每卡 {ppr}）" if ppr not in (None, "", "—") else "")
        else:
            r["data_scale"] = "—"
        # 备注：失败归因（未过判据 + 已通过项 + 条件） / 未纳入验收理由
        if pending:
            r["note"] = r.get("note") or ("读数已录、★ 阈值 δ 待定 ⇒ 状态暂保持「尚未验证」（读数不作判定）。")
        elif voided:
            ref = (ann.get("reference_record") or "").strip()
            r["note"] = ("参考记录（" + (ann.get("reason") or "不对应当前行参数")
                         + " ⇒ 判定作废，不作为该档可用/不可用判定）：" + (ref or "—"))
            if r.get("corrections"):     # ★ 跑次侧的订正说明必须随备注保留（⛔ 不被重建覆盖）
                r["note"] += " ★ 订正说明（跑次侧）：" + "；".join(str(x) for x in r["corrections"][:3])
            r["conclusion_void_reason"] = ann.get("reason", "")
        elif state == S_EXCLUDED:
            r["note"] = EXCLUDED_REASON
            r["excluded_reason"] = EXCLUDED_REASON
        elif r.get("note") not in (None, "", "—"):
            pass                                   # ★ 回填写入的备注（含失败归因/步数对账）优先保留
        elif v:
            r["note"] = ("未过" + v["failed_criterion"] + "：" + v["failed_detail"] +
                         " 已通过：" + " / ".join(v["passed_criteria"]) +
                         "。条件：数据集锚点 c202b1e1…、判据 elam-v3-allright-1、温度 0、有效样本 702。")
        else:
            r["note"] = r.get("note") or "—"      # 已结论档保留回填写入的备注
        r.update(parallel_cols(r, state))
        # ★ LoRA 目标按 backend 呈现（记录两后端的表达差异，避免再混）
        if r.get("tuner_type") == "lora" and r.get("lora_target") not in (None, "", "—"):
            if r.get("backend") == "ms-swift":
                r["lora_target"] = ("ms-swift：真实 HF 名正则（修正版）⇒ 实测 **358** 实例"
                                    "（language_model 248 + visual 110）｜★ native **同为 358 实例**"
                                    "（122 为签名/模式条目数），但**命中集合不同** ⇒ 跨后端读数不直接可比")
            else:
                r["lora_target"] = (r["lora_target"]
                                    + "｜native：canonical preset 名（358 实例；122 为签名/模式条目数）"
                                    + "｜★ 两后端实例数相同但**命中集合不同**（不等价）")
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
            "lora_target_backends": lora_target_backends(),
            "criteria_version": meta.get("criteria_version", ""),
            "result_placeholder": meta.get("result_placeholder", "—"),
            "results_policy": "结果列由跑次记录回填；未有实测结论的档位标注为「尚未验证」；不得手工填写。",
        },
        "groups": GROUPS,
        "columns": COLUMNS,
        "uniform": {"rule": (src.get("uniform") or {}).get("rule", ""),
                    "items": sanitize((src.get("uniform") or {}).get("items", [])) + PARALLEL_UNIFORM + EXTRA_UNIFORM},
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
