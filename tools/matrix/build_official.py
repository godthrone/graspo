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

# ★ lr 口径的**唯一真相源**（`extract_spec.py` 与本脚本同目录）：正式机读件直接引用它，
#   ⛔ 不在本文件里再抄一份取值表（§1.4 单一真相源；否则改口径要改两处、必然漂移）。
import extract_spec as _spec

# 脱敏映射（顺序敏感：先长后短）
# 通用形状（⛔ 不写任何具体内网主机 / 账号 / 挂载点字面量）：
#   顺序敏感 —— 先处理带尾段的规则，再用通用二级/三级路径兜底。
PATH_MAP = [
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/subsets-v2/", "subsets-v2/"),
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/images", "images"),
    (r"/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/data/", "data/"),
    # ★ /home/ 规则必须先于 /models/ 等：否则先行替换会把 `/home/<u>/models/X` 打碎成 `/home/<u>X`（无法再匹配）
    (r"/home/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/", ""),
    (r"/home/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", ""),
    (r"/models/", ""),
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
    # ★★ 兜底（**顺序无关**）：无论规则先后如何，`/home/<seg>[/<seg>…]` 残形整体抹掉。
    #    动机（2026-10-13 实测泄漏）：`/models/` 先于 `/home/` 执行会把 `/home/<u>/models/X`
    #    打碎成 `/home/<u>X`（任何带斜杠的规则都匹配不到）⇒ 必须有一道与顺序无关的兜底。
    s = re.sub(r"/home/[A-Za-z0-9._-]+[A-Za-z0-9._-]*(?:/[A-Za-z0-9._-]+)*", "", s)
    s = re.sub(r"runs/[A-Za-z0-9._-]+/runs/(T\d{3})", r"\1", s)
    return s


#: ★ 脱敏回归夹具用的**宿主绝对路径在运行期拼装**——源码内⛔ 不出现宿主路径字面量
#: （§15.1 检查 3：宿主路径属机密类别；本文件要进开源仓，红线不因"是自测夹具"而豁免）。
#: 形状仍必须是**真实形态**（`/home/<账号>/<段>`），否则负例就验不出脱敏失效。
_FIXTURE_ACCOUNT = "someuser"
_FIXTURE_HOST_PATH = "/".join(["", "home", _FIXTURE_ACCOUNT, "models", "Qwen3.8-27B"])
_FIXTURE_HOST_PATH_NO_TRAILING_SLASH = "/".join(["", "home", _FIXTURE_ACCOUNT, "elam-v5-balanced"])


def _sanitizer_self_test():
    """★ 脱敏回归（含**对抗性输入**与**顺序无关**验证）。"""
    cases = [
        ("① 负例·本次真实案例：含宿主绝对路径（`/home/<u>/models/X`）⇒ 必须 0 命中",
         _FIXTURE_HOST_PATH, _FIXTURE_ACCOUNT),
        ("② 负例·**无尾斜杠**同形 ⇒ 必须被抹掉",
         _FIXTURE_HOST_PATH_NO_TRAILING_SLASH, _FIXTURE_ACCOUNT),
    ]
    ok = True
    for name, raw, needle in cases:
        out = sanitize_str(raw)
        good = needle not in out
        print(f"  [{'PASS' if good else 'FAIL'}] {name} ⇒ {out!r}")
        ok = ok and good
    # ★ 顺序无关：把 PATH_MAP 反转（最坏顺序）再跑，仍须 0 命中
    global PATH_MAP
    orig = PATH_MAP
    try:
        PATH_MAP = list(reversed(orig))
        out = sanitize_str(_FIXTURE_HOST_PATH)
        good = _FIXTURE_ACCOUNT not in out
        print(f"  [{'PASS' if good else 'FAIL'}] ③ ★ 顺序无关：反转 PATH_MAP 后仍须 0 命中 ⇒ {out!r}")
        ok = ok and good
    finally:
        PATH_MAP = orig
    extra = [
        ("★ 按档型：GRASPO 档套用 `2×100/4=50` ⇒ **必须判外来**",
         recon_formula_foreign({"train_method": "graspo"}, "对账式 2×100/4 = 50 ✓") is True),
        ("★ 按档型：GRASPO 档用 `8×20×8/4=320` ⇒ 放行",
         recon_formula_foreign({"train_method": "graspo"}, "对账式 8×20×8/4 = 320 ✓") is False),
        ("★ 按档型：ms-swift SFT 档套用 `8×20×8` ⇒ **必须判外来**",
         recon_formula_foreign({"train_method": "sft"}, "对账式 8×20×8/2 = 640") is True),
    ]
    for name, good in extra:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok = ok and good
    print(f"SANITIZER-SELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def sanitize(obj):
    if isinstance(obj, str):
        return sanitize_str(obj)
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj


def _final_sanitize(obj):
    """★ 全量脱敏：递归抹除矩阵三件中的数据集存放路径与机器号（宪法 §15／§10.1 同族）。"""
    if isinstance(obj, str):
        return matrix_text_sanitize(obj)
    if isinstance(obj, list):
        return [_final_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _final_sanitize(v) for k, v in obj.items()}
    return obj


NOT_EVALUABLE_FULL_NOTE = ("★★ 不可评（工程缺口）——★ 依主席 2026-10-14 口径「跑了没成功即失败」记入失败；"
                           "★ 真因见 `not_evaluable_reason`（逐字）。⛔ 本档**未经效果评测** ⇒ "
                           "⛔ 不得表述为「效果未达标」、⛔ 不得表述为「未过第 2 条」。")

#: ★ 常设脱敏：矩阵三件**不得出现**数据集存放路径与机器号
BANNED_IN_MATRIX = ("/homedata/", "test.jsonl")


#: ★ 无意义残片（★ 生成期清理：⛔ 不留"已通过：。"、"判据①「完整跑完」成立；判据② 未过。"等悬空片段）
DEAD_NOTE_FRAGMENTS = ("判据①「完整跑完」成立；判据② 未过。", "已通过：。", "已通过：", "未过。 ")


def strip_dead_fragments(t: str) -> str:
    t = str(t or "")
    for fr in DEAD_NOTE_FRAGMENTS:
        t = t.replace(fr, "")
    import re as _re
    return _re.sub(r"\s{2,}", " ", t).strip()


def matrix_text_sanitize(t: str) -> str:
    """★ 矩阵文本脱敏：抹除数据集存放路径与机器号（宪法 §15／§10.1 同族）。"""
    import re as _re
    t = str(t or "")
    t = t.replace("/homedata/test.jsonl", "平台方自有数据集（不公开）")
    t = _re.sub(r"/homedata/[^\s，。；]*", "平台方自有数据集（不公开）", t)
    t = t.replace("test.jsonl", "平台方自有数据集（不公开）")   # ★ 裸文件名（无路径前缀）亦抹
    t = _re.sub(r"(?<![0-9A-Za-z])228(?![0-9A-Za-z])", "<HOST>", t)
    return t


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
                   "适用范围：依 2026-10-14 裁定，本档位不实施（不产出可运行配置）。")

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
EPOCHS_BY_METHOD = {"sft": 4, "cpt": 2, "opd": 2, "graspo": 8}

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


def distill_teacher():
    """★ OPD 教师 pin：**唯一来源 = `tests/e2e/matrix54_manifest.json::feasibility.opd_teacher`**。

    ★ 二者别混：**配置值是容器内路径**（`/models/Qwen3.8-27B`）；**挂载源用宿主路径**（由起跑侧决定）。
    """
    import json as _json
    local = Path("tests/e2e/matrix54_manifest.json")
    try:
        m = _json.loads(local.read_text(encoding="utf-8"))
        t = ((m.get("feasibility_model") or {}).get("opd_teacher")) or \
            ((m.get("feasibility") or {}).get("opd_teacher")) or {}
        if t.get("teacher_model_path"):
            _cp = str(t["teacher_model_path"]).strip("/")          # 如 models/Qwen3.8-27B
            _root, _dir = _cp.split("/", 1) if "/" in _cp else (_cp, "")
            return {"model": t.get("teacher_model"), "path_parts": [_root, _dir],   # ★ 分片存，避免整串进文档
                    "measured": t.get("measured"),          # ★ 原样透传（可为 "partial"，⛔ 不 bool 化）
                    "pending_measurement": t.get("pending_measurement"),
                    "min_cards": t.get("min_cards"),          # ★ 若编排侧按 P-25 实测回填，则成为硬约束
                    "teacher_weights_gib_bf16": t.get("teacher_weights_gib_bf16"),
                    "verdict_rule": ("★ **部分实测（partial）**：1 卡 OOM@训练第 1 步；**2 卡生成可通过但训练第 1 步双 rank OOM ⇒ 2 卡不足已实测**；"
                                     "★ **min_cards 未定（4 卡量测待做）⇒ OPD 全族 ⛔ 不得起跑**"
                                     if t.get("measured") == "partial" else
                                     ("★ 已实测：min_cards=%s（低于即 fail-closed）" % t.get("min_cards")
                                      if t.get("measured") is True else
                                      "unmeasured（教师侧未测算）⇒ 不得声称可行；待 P-25 上机测量后回填")),
                    "offload": True, "source": f"{local}::feasibility_model.opd_teacher",
                    "note": "★ 配置值 = **容器内路径**（由 path_parts 拼成）；**挂载源用宿主路径**（起跑侧决定）——两者别混。"}
    except Exception:
        pass
    # 兜底 = 已 pin 的常量（与 manifest 同值；manifest 缺失时不至于产出空 teacher）
    return {"model": "Qwen3.8-27B", "path_parts": ["models", "Qwen3.8-27B"], "offload": True,
            "source": "内置常量（manifest 不可读时的兜底）"}


def recon_formula_foreign(row, text):
    """★ 对账式"按档型选公式"断言：该档文本是否**套用了别族的公式**。

    · GRASPO 档 ⇒ 正确为 `8×20×8/cards`；若文本出现 ms-swift SFT 口径（如 `2×100/`）⇒ 外来公式
    · ms-swift SFT/CPT 档 ⇒ 正确为 `epochs×subset/nproc`；若文本出现 `8×20×8` ⇒ 外来公式
    返回 True = **外来公式**（应 fail-closed）。
    """
    import re as _re
    t = str(text or "")
    tm = str(row.get("train_method") or "").lower()
    if tm == "graspo":
        return bool(_re.search(r"2\s*[×x]\s*100\s*/", t))
    if tm in ("sft", "cpt"):
        return bool(_re.search(r"8\s*[×x]\s*20\s*[×x]\s*8", t))
    return False


def _lr_policy_block():
    """★★ lr 口径登记（机读）：新口径 =「按方法一个值」；两条旧口径作历史；三条「已裁定不做」。

    ★ 取值表**直接来自** `extract_spec.LR_POLICY`（唯一真相源，⛔ 不抄写、不发明数字）；
      主席 2026-10-15 裁定「按卡数缩放 lr」机制**整体删除**（不是保留开关）⇒
      现行生效 lr = 声明 lr，与 dp_size / tp_size / pp_size / 卡数**无关**。
    """
    return {
        "effective": "2026-10-15",
        "source": "主席 2026-10-15 裁定（新学习率口径 / lr 按方法一个值）",
        "single_source": "tools/matrix/extract_spec.py:LR_POLICY",
        "rule": ("lr = 按 (训练方法, 微调模式) **一个值**；⛔ 不随 dp_size / tp_size / pp_size / "
                 "总卡数变化；⛔ 不按卡数补偿训练量。"),
        "by_method_and_tuner": {f"{m}|{t}": v for (m, t), v in sorted(_spec.LR_POLICY.items())},
        "history": [
            ("① 旧口径（已被取代，留痕、⛔ 不再作为判定依据）：**LoRA 档统一 1.0e-05、不分算法**"
             "（由该口径产生的旧跑次读数在切换后与现行声明不再一一对应）。"),
            ("② 旧口径（已被取代，留痕、⛔ 不再作为判定依据）：**按卡数缩放 lr** —— "
             "`lr_scaling=linear` ⇒ 生效 lr = 配置 lr × dp_size（默认 linear；仅 native 后端消费）。"
             "★ 该机制已**整体删除**（配置字段与方法都不再存在）；带该键的旧配置被 "
             "pydantic `extra=\"forbid\"` **fail-closed 拒绝**（⛔ 不静默忽略）。"),
        ],
        "mechanism_removed": ("「按卡数缩放 lr」机制（配置字段 `lr_scaling` ＋ 方法 "
                              "`effective_learning_rate` ＋ 各 native 消费点）已**整体删除** ⇒ "
                              "现行生效 lr 一律 = 声明 lr（⛔ 不再有 ×dp_size 这一步）。"),
        "ruled_out": list(_spec.RULED_OUT),
        "consequence": ("★ 口径变更后**读数与配置的一一对应性**须逐档复核：凡读数来自旧口径跑次"
                        "（旧 lr 或旧 epoch）而本行参数已更正者 ⇒ 结论作废、该档标「尚未验证」"
                        "（行级机读字段 `reading_consistency`；旧读数保留为历史、⛔ 不抹）。"),
    }


def criteria_caliber():
    return {
        "availability_criteria": [
            "① 跑通/工程合规：退出码 0、无未捕获异常；优化器步数达标；checkpoint 完整；NaN/Inf 为 0；算法机制真实生效；产物形态合规；显存与吞吐在承诺范围内。",
            "② 效果（★ 本轮唯一改动）：SFT ⇒ 工具调用准确率 after − base ≥ +10.0 pp（epoch=4）；"
            "GRASPO ⇒ after − base ≥ +5.0 pp（epoch=8）；"
            "CPT ⇒ 训练 loss 正常下降（U2 口径）且准确率 after ≥ base（不劣化；epoch=2）；"
            "OPD ⇒ 学生 after ≥ base（能力回收；epoch=2；教师 = SFT 之后的 27B 产物）。"
            "★ **评测集口径（2026-10-16 主席裁定）**：**100 行档（T001–T027）= fit 口径（训练集自评）** —— "
            "评测集就是**该档自身训练用的同一份文件**（100 行、`dataset_split: train`）；"
            "其余档（20 行 mini 档 / 9 个 formal 档）仍用平台方自有 702 行评测集。"
            "★ **通过线数值逐字未动**（⛔ 未放宽、⛔ 未收紧）。"
            "★ 一律用**点估计 Δ** 判定：⛔ 不做假设检验，不引入置信区间 / 显著性。",
            "③ 算法正确性检查（逐字未动）：各算法按各自清单（见第 3.2 节）；GRASPO 算法逻辑严禁改动。",
            "④ 推理验证（逐字未动）：产物可正常推理（合并 LoRA + vLLM 启动、实际前向一次拿到可解析输出）。",
        ],
        "caliber_change": {
            "effective": "2026-10-15",
            "source": "主席最新裁定（新四条）",
            "changed": ("★ **改动一 = 判据②（效果）**：由「训练类与 GRASPO 统一 δ = +5.0 pp」改为**逐算法分列**；"
                        "SFT epoch 由 2 改为 4；OPD 教师改为 SFT 之后的 27B 产物；★ 明确「⛔ 不做假设检验」。"
                        "★★ **改动二（2026-10-16，无冲突、两条并列）= 评测集口径**：**100 行档（T001–T027）**"
                        "的评测集由平台方自有 702 行评测集改为**该档自身训练用的同一份文件**（100 行、"
                        "`dataset_split: train`）⇒ 读数的含义是 **fit 口径（训练集自评）**，"
                        "⛔ 不得读作训练分布之外的表现；★ 该改动**不改通过线数值**，旧读数按「结论与口径"
                        "一一对应」纪律**作废**并保留于各行 `voided_conclusion.historical_effect`；"
                        "★ 20 行档**不适用**（n=20 ⇒ 每样本 5.00 pp ＝ GRASPO 通过线 ⇒ 判据退化为 1 个样本，"
                        "主席裁定未回、待定）、9 个 formal 档**不动**。"),
            "unchanged": "★ 判据 ①③④ **逐字未动**（见 availability_criteria 第 1/3/4 项）。",
            "superseded": ("★ 旧口径（留痕，⛔ 不再作为判定依据）：训练类（SFT / CPT / OPD）与 GRASPO **统一 δ = +5.0 pp**；"
                           "目标 +10.0 pp 仅展示、不用于判定；SFT epoch = 2。"),
            "reading_change_registry": [],      # ★ 由 main() 从**行数据**派生（⛔ 不在此处写死档号/数值）
            "reading_change_registry_note": (
                "★★ **「读数变化」≠「判据变化」**：本清单逐条登记「因缺陷修复/重跑产生了新读数"
                "（旧读数作废、保留为历史）、而判据口径逐字未变」的档，并**逐条注明四态是否翻盘**"
                "（四态翻盘 / 四态未变——仅归因与读数更正）。"
                "★ 每条的 `kind` = `reading_change（★ 不是判据变更）`，可在档位行的 `caliber_change` 里机读。"),
        },
        "dataset": {"id": ("平台方自有评测集（名称 test.jsonl）＋ ★ **100 行档（T001–T027）的评测集 = "
                           "该档自身训练用的同一份文件**（fit 口径，见下）"),
                    "availability": "平台方自有 / 不公开（100 行档用该档自身训练文件，同为非公开数据）",
                    "version_anchor_sha256": ("两种口径 —— ★ 100 行档 = 该档自身训练文件（100 行，sha256 594d41b0…）；"
                                              "其余档 = 平台方自有评测集（702 行，c202b1e1…）"),
                    "sample_count": "★ 100 行档 = 100（fit 口径）；其余档 = 702",
                    "method": "工具调用准确率（acc）",
                    # ★★ 「是否同分布」口径（主席诉求 5；审查项 W9/W10）＋ 2026-10-16 评测集口径变更（fit）：
                    #    100 行档的评测集**就是它自己的训练文件**（不是"同分布的另一份"）；其余档仍为同源不同样本。
                    #    ⛔ 一律不评测「训练分布之外的表现」；⛔ 「泛化」一词不进本件（主席口径）。
                    "train_eval_relation": (
                        "★★ **评测集口径（2026-10-16 主席裁定）逐档分列**："
                        "① **100 行档（T001–T027）= fit 口径（训练集自评）** —— 评测集 = **该档自身训练用的"
                        "同一份文件**（100 行、`dataset_split: train`）⇒ 读数回答的是「该档训练是否生效」；"
                        "⛔ **不得读作训练分布之外的表现**（⛔ 本矩阵不评测该问题）。"
                        "旧 702 行口径下的读数**已作废**，保留于各行 `voided_conclusion.historical_effect`。"
                        "② **其余档**（20 行 mini 档 / 9 个 formal 档）= 平台方自有 702 行评测集"
                        "（与各档训练数据同源、样本 id 不重叠）。"
                        "★ **判别力（100 行档）**：该 100 行的 (工具, 动作) 类别**最大类占 19%**"
                        "（19/100）⇒ **常数预测器上界 19%**；而通过线 = `after − base ≥ +10.0 pp`（点估计）"
                        "⇒ **跨过它必须真的读图**，任何常数输出都拿不到。"
                        "★ **同时作废的旧读数（⛔ 不得引用）**：「剔除 169 条图像重叠后 32.27%」与"
                        "「训练/评测图像去重交集 752 张（占评测图像 38.5%）」出自 **702 行评测集 × 全量 6378 行"
                        "训练集**口径；mini 档实测 **mini100 × 702 图像交集 = 0** ⇒ 该口径对 mini 档"
                        "**自始不成立**（生成的评测 yaml 原先写 `train_limit: null`，已随本次口径变更订正）。"
                        "★ **V1 档的重叠剔除次级读数按定义退化**：评测集 ≡ 训练集 ⇒ 重叠 100/100、剔除后 0 条 "
                        "⇒ `accuracy_percent_excluding_overlap` / `overlap_excluded_count` 在 100 行档上"
                        "**不构成次级读数**、⛔ 不得引用。"),
                    "train_eval_relation_role": (
                        "★ 本条是**评测集口径**（不是能力承诺）：矩阵只声明「已在本口径下完整跑通并取数」，"
                        "使用者自身任务/数据集/参数下的效果不在承诺范围（见边界声明）。"
                        "★ **100 行档为 fit 口径（训练集自评）**：其读数只证明该档在其自身训练数据上的"
                        "拟合与训练有效性，⛔ 不得据此外推到其他数据上的表现。")},
        # ★★ 评测集口径登记（机读，2026-10-16 主席裁定）：与 `dataset` 的散文同源，供机读消费。
        "eval_caliber": {
            "effective": "2026-10-16",
            "source": "主席 2026-10-16 裁定（「用同一个数据集」：训练用什么、评测就用什么）",
            "rule": ("★ **100 行档（判据 = 该档 `subset_size == 100`，即 T001–T027）**：评测集 = **该档自身"
                     "训练用的同一份文件**（100 行、`dataset_split: train`）⇒ **fit 口径（训练集自评）**；"
                     "★ 其余档（20 行 mini 档 / 9 个 formal 档）：保持平台方自有 702 行评测集 + `dataset_split: test`。"),
            "pass_lines_unchanged": ("★ **通过线数值逐字未动**：SFT ⇒ `after − base ≥ +10.0 pp`；"
                                     "GRASPO ⇒ `after − base ≥ +5.0 pp`；CPT ⇒ `after ≥ base（不劣化）`；"
                                     "OPD ⇒ `学生 after ≥ base（能力回收）`；一律点估计、⛔ 不做假设检验。"),
            "fit_caliber_label": "fit 口径（训练集自评）",
            "discriminative_power": ("★ 100 行档：训练集 (工具, 动作) 类别最大类 **19/100 = 19%** ⇒ "
                                     "**常数预测器上界 19%** ⇒ 跨 +10.0 pp 必须真读图；"
                                     "每样本粒度 **1.00 pp**（达线需净跳 ≥10 条）。"),
            "not_applicable": ("★ **20 行档明确不适用**：n=20 ⇒ 每样本 **5.00 pp**，而 GRASPO 通过线正好 "
                               "+5.0 pp ⇒ 判据退化为「1 个样本决定成败」；★ **主席对该批的裁定尚未回来**"
                               " ⇒ 本件**不改其口径**，其状态待定。"),
            "untouched": "★ 9 个 formal 档（T028–T030 / T034–T036 / T040–T042）不动：训练源 = 全量训练集前 N，与 702 评测集本就同分布。",
            "overlap_analysis_fix": ("★ 顺带订正：生成的评测 yaml 原先写 `train_limit: null` ⇒ 重叠分析用的是"
                                     "**全量 6378 行训练集**（实证 `train_subset.sample_count=6378`、"
                                     "`overlap_excluded_count=169`），而 mini 档实际只训 100 行 ⇒ 现改为"
                                     "**该档自身训练集 + 该档子集条数（100）**。"),
            "consequence": ("★ 后果判据：**旧读数取自 702 行评测集、与本行现行评测集口径不再一一对应** ⇒ 结论作废、"
                            "该档标「尚未验证」（行级机读字段 `eval_caliber_consistency`；旧读数保留于 "
                            "`voided_conclusion.historical_effect`，⛔ 不抹）。"
                            "★ 与 lr 口径的叠加：两批档**有交叠** ⇒ ⛔ 不覆盖既有 `caliber_cause`，两条原因**并列写清**。"),
        },
        # ★★ lr 口径登记（审查项 🔴B1/W3）：机读字段 `lr_policy` / `ruled_out` 必须在正式件可查。
        #    取值表**直接引用唯一真相源** `extract_spec.LR_POLICY`（⛔ 不在此抄写）。
        "lr_policy": _lr_policy_block(),
        "ruled_out": list(_spec.RULED_OUT),
        "criteria_id": "elam-v3-allright-1",
        "threshold": {
            "name": "delta_pp（SFT / GRASPO）＋ non_inferiority（CPT / OPD）", "unit": "百分点（pp）",
            "value": None,
            "target": None,
            "applies_to": "★ 按算法分列（见 by_method）：SFT / GRASPO 走 Δ 通过线；CPT / OPD 走不劣化线（after ≥ base）。",
            "status": ("★ 已冻结（主席最新裁定，2026-10-15）：判据② 逐算法分列；**⛔ 不放宽判据**；"
                       "**★ 判定一律用点估计 Δ —— ⛔ 不做假设检验、不引入置信区间 / 显著性**。"),
            "historical_note": ("★ 旧口径（已被取代、留痕，⛔ 不再作为判定依据）：训练类与 GRASPO **统一 δ = +5.0 pp**；"
                                "目标值 +10.0 pp（仅展示、不用于判定）；SFT epoch = 2。"),
            "by_method": {
                "sft": {"criterion": "delta_pp", "op": ">=", "value": 10.0, "unit": "pp",
                        "metric": ("工具调用准确率（after − base）；★ **100 行档 = fit 口径**"
                                   "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                        "pass_line_text": "after − base ≥ +10.0 pp",
                        "epoch": 4, "epoch_previous": 2,
                        "epoch_note": ("★ epoch 由 2 改为 4（判据② 的配置改动）；"
                                       "★ 现有 SFT 跑次均为 epoch=2 旧配置 ⇒ 其结论一律标注「待 epoch=4 重跑复核」。")},
                "graspo": {"criterion": "delta_pp", "op": ">=", "value": 5.0, "unit": "pp",
                           "metric": ("工具调用准确率（after − base）；★ **100 行档 = fit 口径**"
                                   "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                           "pass_line_text": "after − base ≥ +5.0 pp",
                           "epoch": 8, "epoch_previous": 8, "epoch_note": "epoch=8 不变。"},
                "cpt": {"criterion": "non_inferiority", "op": ">=", "value": 0.0, "unit": "pp",
                        "metric": ("工具调用准确率（after − base）；★ **100 行档 = fit 口径**"
                                   "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                        "pass_line_text": "after ≥ base（不劣化）",
                        "also": "训练 loss 正常下降（U2 口径）",
                        "epoch": 2, "epoch_previous": 2, "epoch_note": "epoch=2 不变。"},
                "opd": {"criterion": "non_inferiority", "op": ">=", "value": 0.0, "unit": "pp",
                        "metric": ("工具调用准确率（学生 after − base）；★ **100 行档 = fit 口径**"
                                 "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                        "pass_line_text": "学生 after ≥ base（能力回收）",
                        "teacher": "SFT 之后的 27B 产物",
                        "epoch": 2, "epoch_previous": 2,
                        "epoch_note": "epoch=2 不变；教师改为 SFT 之后的 27B 产物。"},
            },
            "by_method_single_source": ("★ 本 `by_method` 是**跑次阈值判定的唯一真相源**：跑次 json 自述阈值必须与之逐算法一致，"
                                        "否则回填 fail-closed。"),
        },
        "baseline": {"role": "同档位未训练底座（role=base）",
                     "caliber": "U2 终点窗口 = 末段 max(3, 10% 步数) 步的均值；该档的 epoch 数与推导步数见档位「步数/epoch」子项。"},
        "epoch_policy": {
            "by_method": ("SFT = **4 epoch**（★ 本轮由 2 改为 4）；CPT = 2 epoch（⛔ 不动）；OPD = 2 epoch（⛔ 不动）；"
                          "GRASPO(RL) = 8 epoch（⛔ 不动；若 8 仍不过线则升 10 为后备；理由：RL 耗时长、5 epoch 下效果仅差 0.44 pp 未过线）。"
                          "理由：跨 epoch 容易出问题（循环结构决定）；RL 需要更多 epoch 才能看到效果。"),
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
                "three_concepts": ("★ **三者关系（务必分清）**：① **名义步数 `steps_derived`（可宽松）** = 矩阵按 epoch 口径推导、"
                                   "写入 yaml 的 steps / save_steps —— 是**派生值**且正被公式纠正；"
                                   "② **实测步数 `steps_actual`（严格身份）** = 实跑 `run_cumulative.step` / epoch 实测 —— "
                                   "★ **属训练身份，留在防「跑 A 填 B」的严格比对里**；"
                                   "③ **对账式（核算）** = 用运行期量校验实跑步数，归 `step_count_reconciliation` 负责。"),
                "nominal": ("**名义步数** = 矩阵按 epoch 口径推导的步数（写入 yaml 的 steps / save_steps）；"
                            "⛔ 对 native GRASPO 仅为名义值，不是承诺。"),
                "reconciliation_rule": ("**对账式** = 用运行期量校验实跑步数；两个后端各自复现本后端实际值："
                                        "① ms-swift（配置式）steps = epochs × (subset × num_generations) / "
                                        "(micro_batch × grad_accum × nproc)；★ 实测校正：T032（nproc=2, 8ep, subset 20, G 8）"
                                        "= 8×(20×8)/(1×1)/2 = **640 = 实际 640 ✓**（未除 nproc 的「1280」已被实测否证）；"
                                        "T031（nproc=1）= 800 ✓；T033（nproc=4）= 8×20×8/4 = 320 ✓ ⇒ ★ **/nproc 形式已在 nproc=1/2/4 三锚验证**（T031 800 / T032 640 / T033 320）；② native（运行期式）steps = "
                                        "floor(trainable_completions / (rollout_queue_batch_size × rollout_group_size)) "
                                        "+ (force_flush 步数)，本仓 Q×G = 8×8 = 64。"
                                        "★ **带 flush 修正项**：实测 T029 的 22 步里有多次「不足一组的 flush」"
                                        "（per-step = 64,56,56,56,56,48,56,32,48,64×7,48,32,40,56…，Σpartial=1224）"
                                        "⇒ `ceil(Σ/(Q×G)) = 20 < 实际 22` ⇒ **一次性 ceil 会低估**，必须写成"
                                        "`floor(Σ/(Q×G)) + flush 步数`。"),
                "worked_example": ("① T028（native, dp=1, 5ep）：可训练组 147 × G=8 = 1176 ⇒ floor(1176/64) = 18 + 1（末步 flush）= **19 = 实测 19 ✓**；"
                                   "② T029（native, **dp=2**, 5ep）：Σ `trainable_completions` = **1240** ⇒ ceil(1240/64) = 20 "
                                   "⇒ ★ **≠ 实际 23，差 −3 ⇒ 不吻合，如实记差**（⛔ 不为凑吻合改公式、⛔ 不写死 23）；"
                                   "★ **假设（明确标为假设，未证实）**：`dp>1` 时阈值可能**按 rank 生效**（Q×G per rank）"
                                   "⇒ 需更多 dp>1 实测样本才能定论。③ ms-swift 锚与 SFT 四档核对见上。"),                "measured_authority": ("★★ **实测总步数 = 1184（权威）**（T047，native 2 卡 OPD，P-25 上机实测）；"
                                       "★ 推导 `8×20×8/2 = 640` **已证伪**（原因待定：`num_generations` / subset 语义或运行期项）"
                                       "⇒ ★ **一律以实测为准**；该档 `steps_actual` = **1184**。"
                                       "★ **族纪律**：步数口径必须以**该档自身配置 ＋ 运行期产物**为准，"
                                       "⛔ **不用通用公式推断**（与「没按本档自身实体取值」同族）。"),
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
                "★ 用途：供「基准散布 / 阈值余量」判断；★ **δ 判定不变，⛔ 不因此改阈值**。"
                "★★ **2026-10-16 限定**：上列噪声底**绑定 702 行评测集** ⇒ **对 100 行档（fit 口径）"
                "不再适用、⛔ 不得引用**（100 行档的配对噪声未实测；按翻转率等比外推约 ±0.84 pp，"
                "**属推断、未实测**，⛔ 不作判定依据）。"
                "★ **100 行档的分辨力（如实）**：每样本 **1.00 pp**、达线需净跳 **≥10 条**；"
                "通过线仍 `after − base ≥ +10.0 pp`（点估计、⛔ 不做假设检验）。",
                "★ 用途：供「基准散布 / 阈值余量」判断；★ **δ 判定不变，⛔ 不因此改阈值**。",
            ],
            "caliber_corrections": [
                "★★ **runner 日志跨版本交错 ⇒ 不可用作判据**：`status-T033/train.T033.stdout.log` 里**新版 runner 的行在前**、"
                "**旧版 runner 的 `ckpt-retention FATAL` / `torch-probe SKIP` 行在后**（来自第一次尝试的旧树 `3df0a7de…`）"
                "⇒ ★ **判成败一律用三件**：`exit_code` + `ckpt_retention.state` + `torch_probe.json`（配锚点时间序）。",
                "★★ **`batch_high_retry_rate` 代理与真值矛盾（已证）**：代理（classify_group retry **33/36 = 0.917 ≥ 0.5**）"
                "**说触发**，而**指标实测 0 次触发** ⇒ **代理分母 ≠ 指标分母**；★ 真值到手前该列必须标「**缺输入 ⇒ 待补**」，"
                "**⛔ 不得用代理冒充真值**。",
                "★ **T033 树指纹无漂移**：起跑断言 **`820fb5a5…` = 评测期间现测值**；第一次尝试的旧树 **`3df0a7de…`** "
                "已登记为 `first_attempt_fingerprint`。",
                "★★ **锚点纠正（按「起跑断言值」，两处）**：`T029` ⇒ 起跑断言 **`db209997…`**（现测 `1335546a…`，post-launch 漂移）；"
                "`T032` ⇒ 起跑断言 **`ea930330…`**（现测 `820fb5a5…`）—— ★ 指挥部此前转来的 `8020010f…` 在该树 "
                "`launch.log` / `.env.sh` / `status-T032/` 及**全树 grep 中均查无**，**无法佐证**，如实登记；"
                "★ 一律**以起跑断言值为锚**。",
                "★ **native 档自动评测流程统一用 `graspo export --config`**（T029 首轮在 `eval export --eval-config` 处 fail-fast，"
                "已改续跑脚本后成功）。",
                "★★ **坑（务必防错位）：228 上存在两个 T010 跑次** —— `graspo-batch2-20261012/runs/T010` = **1ep/100 步**"
                "（对应已作废的 1-epoch 口径）；`graspo-batch3-20261012/runs/T010` = **2ep/200 步/Δ=+13.39 pp/指纹 `e57a5701…`**"
                "⇒ ★ **本矩阵的 T010 一律以 batch3 为准**；原 json 的 `run_path` 曾错指 batch2、"
                "`guard_assertions.tree_fingerprint_assert` 由 `f2b0f481…` 订正为 `e57a5701…`（均已登记）。",
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


def short_note(row, full):
    """★ 备注瘦身（主席令）：备注列只留「判定 + 关键读数 + 必要口径例外」，≤100 字。

    长文（证据链/自检/订正全文/锚点/P5P6 细则）一律进 `note_full`（→ 工作日志历史，⛔ 不进表格）。
    """
    st = str(row.get("status") or "—")
    _rule0 = _caliber_rule(row.get("train_method"))
    parts = [f"判定：{st}。"]
    # ★★ 口径一致性作废：**放在最前**（截断也不丢这条硬要求）。
    #    该档结论因「读数取自旧参数/旧口径、本行现行口径已变」而作废（**两条独立原因**：
    #    ① lr 口径（`reading_consistency`）；② 评测集口径 = fit 口径（`eval_caliber_consistency`））
    #    ⇒ ⛔ 不得在短备注里把旧读数当**有效读数**呈现（旧读数在 `voided_conclusion` 里保留为历史）。
    _rcc0 = row.get("reading_consistency") if isinstance(row.get("reading_consistency"), dict) else {}
    _ecc0 = row.get("eval_caliber_consistency") if isinstance(row.get("eval_caliber_consistency"), dict) else {}
    _voided = st == S_UNVERIFIED and row.get("voided_conclusion")
    if _voided:
        _lr_mm = "；".join(str(x) for x in (_rcc0.get("mismatch") or []))
        _ev_mm = "；".join(str(x) for x in (_ecc0.get("mismatch") or []))
        _lr_bad = bool(_rcc0.get("consistent") is False)
        _ev_bad = bool(_ecc0.get("consistent") is False)
        if _lr_bad and _ev_bad:
            _why = "两条口径变更叠加（lr ＋ 评测集）"
            _mm = "／".join(x for x in (_lr_mm, _ev_mm) if x) or "参数已更正"
        elif _ev_bad:
            _why = "评测集口径变更"
            _mm = _ev_mm or "评测集已由 702 行改为本档自身训练文件（100 行）"
        else:
            _why = "读数与当前配置不对应"
            _mm = _lr_mm or "参数已更正"
        parts.append(f"★ 结论作废（{_why}）：{_mm} ⇒ 待重跑复核。")
    # ★★ 「epoch 已变 ⇒ 结论待重跑复核」标记（**数据驱动**：由 `threshold.by_method.<方法>` 的
    #    epoch / epoch_previous 推出，⛔ 不写死方法名与数字）。放在最前 ⇒ 截断也不会丢这条硬要求。
    #    ★ 若该档已**按新 epoch 复核过**（行级 `caliber_change.epoch_recheck` 非空）⇒ 改标「已按 epoch=N 复核」。
    _ep, _epp = _rule0.get("epoch"), _rule0.get("epoch_previous")
    _cc0 = row.get("caliber_change") if isinstance(row.get("caliber_change"), dict) else {}
    _rechecked = _cc0.get("epoch_recheck")
    if _rechecked and st in (S_AVAILABLE, S_UNAVAILABLE):
        parts.append(f"★ 已按 epoch={_rechecked} 复核。")
    elif _ep and _epp and _ep != _epp and st in (S_AVAILABLE, S_UNAVAILABLE):
        parts.append(f"★ 待 epoch={_ep} 重跑复核（现跑次为 epoch={_epp} 旧配置）。")
    eff = str(row.get("effect") or "").replace("*", "")     # 去 markdown 强调符
    # ★ 优先用**结构化字段**（after_acc/base_acc/delta_pp）；无则回退 effect（固定格式结果字段）
    aa, ba, dp = row.get("after_acc"), row.get("base_acc"), row.get("delta_pp")
    # ★★ 读数有效性（数据驱动）：被判**无效**的判据② 读数（如"多卡 DP 副本分歧、无唯一正解"
    #    ⇒ 判据① 归因）⛔ 不得在短备注里当**有效读数**呈现 ⇒ 只留无效标记（不打数字），
    #    完整历史读数与无效理由在 `effect` / `verdict` / `note_full`（⛔ 不在此处丢证据）。
    # ★★ 数据驱动：仅当**当前**读数被判无效时才打无效标记。行级
    #    `caliber_change.reading_after.valid = true` ⇒ 旧读数的无效性**只作历史**
    #    （如 T024：判据① 产物唯一性缺陷已修、新读数有效）⇒ ⛔ 不得把**新**读数标成无效、⛔ 不得吞掉读数。
    _cur_reading_valid = (_cc0.get("reading_after") or {}).get("valid")
    if _voided:
        pass                                             # ★ 作废档：旧读数不打进短备注（历史在 note_full）
    elif str(row.get("reading_validity") or "").strip() and st == S_UNAVAILABLE and _cur_reading_valid is not True:
        parts.append("★ 判据② 读数无效、⛔ 不得引用。")
    elif isinstance(aa, (int, float)) and isinstance(ba, (int, float)) and isinstance(dp, (int, float)):
        def _pct(x):
            v = float(x) * 100
            return "0.0" if abs(v) < 1e-9 else f"{v:.4f}"
        parts.append(f"after {_pct(aa)}%/base {_pct(ba)}%/Δ={dp:+.4f} pp。")
    else:
        a = re.search(r"after[^0-9]{0,12}([0-9]+\.[0-9]+)\s*%", eff)
        b = re.search(r"base[^0-9]{0,12}([0-9]+\.[0-9]+)\s*%", eff)
        d = re.search("Δ[^0-9+\\-\u2212]{0,6}([+\\-\u2212]?[0-9]+\\.[0-9]+)\\s*pp", eff)
        if a and b and d:
            parts.append(f"after {a.group(1)}%/base {b.group(1)}%/Δ={d.group(1)} pp。")
    _ner = str(row.get("not_evaluable_reason") or "").strip()
    _cause = str(row.get("caliber_cause") or "").strip()
    _rule = _caliber_rule(row.get("train_method"))
    _passline = str(_rule.get("pass_line_text") or "见判据②")
    if st == S_UNAVAILABLE and _ner:
        # ★★ "不可评（工程缺口）"：★ **未经效果评测** ⇒ ⛔ 不得写"未过第 2 条"（那是效果层判据）
        #    只写"不可评：<真因摘要>"（★ 真因全文在 `not_evaluable_reason`；★ 且已由校验器保证不含"效果"）
        # ★ 优先用**短摘要字段**（`not_evaluable_brief`）⇒ 保证备注 ≤100 字；无则由真因截断。
        #   ★ 给了短摘要就**原样用**（⛔ 不再按"；/⇒"切分 —— 否则会误切摘要里的箭头）。
        _given = str(row.get("not_evaluable_brief") or "").strip()
        if _given:
            _brief = _given[:60]
        else:
            _brief = _ner.replace("不可评：", "").strip()
            _brief = _brief.split("；")[0].split("⇒")[0].strip()[:48]
        parts.append(f"不可评：{_brief}。")
    elif st == S_UNAVAILABLE and _cause:
        # ★★ §17.7 第①项：未过的是**判据①（跑通/工程合规）** ⇒ ⛔ 不得写成"未过第 2 条 / 效果未达标"
        #    （真因逐字在 `caliber_cause` / `caliber_note` / `blocked_reason`，本处只做短摘要）
        _c = _cause.replace("*", "").replace("`", "").strip().lstrip("★ ").strip()
        parts.append(f"未过第 1 条（跑通/工程合规，★ 非效果结论）：{_c[:40]}。")
    elif st == S_UNAVAILABLE:
        parts.append(f"Δ<通过线，未过第 2 条（{_passline}）。")
    _p5 = row.get("p5p6_readings")
    # "待补"判定：**整块未取数**才算（有 1..6 项即算已补；子项里的"待补"字样不改变整块状态）
    _p5_na = bool(_p5) and "不适用" in str(_p5) and not any(
        str(k).startswith(tuple("123456")) for k in (_p5 if isinstance(_p5, dict) else {}))
    _p5_done = bool(_p5) and (isinstance(_p5, dict) and _p5.get("applicable") is True
                              or any(str(k).startswith(tuple("123456")) for k in (_p5 or {})))
    # ★ P5/P6 是 **GRASPO 专有**指标：非 GRASPO 档（SFT/CPT/OPD）一律"不适用" ⇒ ⛔ 不加任何标记
    _not_graspo = str(row.get("train_method") or "").lower() != "graspo"
    if _not_graspo or _p5_na or (isinstance(_p5, dict) and _p5.get("applicable") is False):
        pass                              # ★ 明确"不适用"（如非 GRASPO 档）⇒ 不加任何 P5/P6 标记
    elif not _p5_done:
        parts.append("P5/P6 待补。")
    elif st == S_UNAVAILABLE:
        parts.append("P5/P6 已补。")      # ★ 六项已填实（细则在 note_full / json）
    elif st == S_UNVERIFIED and row.get("voided_conclusion"):
        parts.append("5ep 作废参考。")
    return "".join(parts)[:100]


def _caliber_rule(method):
    """★ 判据② 逐算法通过线：**唯一真相源 = `criteria_caliber.threshold.by_method`**（⛔ 不写死）。"""
    try:
        return (((criteria_caliber().get("threshold") or {}).get("by_method") or {})
                .get(str(method or "").lower()) or {})
    except Exception:
        return {}


def verdict_of(row, state):
    """已验证不可用档的结论：写明未过的是哪一条判据，并保留已通过项的事实（**数据驱动**）。"""
    if state != S_UNAVAILABLE:
        return None
    # ★★ 不可评档（`not_evaluable_reason` 非空）⇒ ⛔ 不得套用"效果未达标/未过第 2 条"模板
    #    （该档**未经效果评测**）⇒ 返回"不可评"结构，真因指向 `not_evaluable_reason`。
    if str(row.get("not_evaluable_reason") or "").strip():
        return {"state": state,
                "failed_criterion": "不可评（工程缺口）—— 依主席 2026-10-14 口径记入失败",
                "failed_detail": str(row.get("not_evaluable_reason") or "").strip(),
                "passed_criteria": []}
    # ★★ §17.7 第①项：有 `caliber_cause` 的档 ⇒ 未过的是**判据①（跑通/工程合规）**，
    #    ⛔ 不得兜底成"第 2 条：效果未达标"（那是效果层否定，与真因冲突）。
    #    真因逐字保留在 `caliber_cause`（含 OOM／看门狗超时／未实现／硬件不可行／未做）。
    cause = str(row.get("caliber_cause") or "").strip()
    if cause:
        return {"state": state,
                "failed_criterion": "第 1 条：跑通/工程合规未通过（★ 不是效果结论）",
                "failed_detail": cause,
                "passed_criteria": []}
    eff = str(row.get("effect", ""))
    m = re.search(r"Δ\s*[=≈]?\s*([+\-−＋]?)\s*([0-9.]+)\s*pp", eff)
    rule = _caliber_rule(row.get("train_method"))
    passline = str(rule.get("pass_line_text") or "见判据②")
    method_txt = str(row.get("train_method") or "").upper()
    if m:
        sign = "-" if m.group(1) in ("-", "−") else "+"
        detail = f"Δ = {sign}{m.group(2)} pp < 通过线（{method_txt} 判据②：{passline}）"
    else:
        detail = f"效果未达标（{method_txt} 判据② 通过线：{passline}；读数见 effect）"
    passed = []
    algo = str(row.get("algo_checks", ""))
    if algo and algo not in ("—", ""):
        passed.append(algo)
    return {"state": state,
            "failed_criterion": f"第 2 条：效果未达标（{method_txt} 判据② 通过线 = {passline}）",
            "failed_detail": detail,
            "passed_criteria": passed or ["判据① 通过（见 runs）"]}


#: ★ 耗时解析：**训练墙钟**的取数形态（★ 单一真源；`split_elapsed()` 与 verify 断言共用）。
#    ⛔ 只认「训练墙钟」语境（墙钟 / *complete elapsed / train_runtime / trainer 末步 elapsed_sec /
#    batch_sec mean / s/it / s/step / 步/s）——**评测侧**读数（如"评测侧单样本 elapsed_sec"）**不取**。
ELAPSED_TOTAL_PATTERNS = (
    re.compile(r"墙钟\s*\**\s*([0-9]{2}:[0-9]{2}:[0-9]{2})"),                       # 墙钟 HH:MM:SS
    re.compile(r"墙钟\s*\**\s*([0-9]+(?:\.[0-9]+)?\s*s)\b"),                        # 墙钟 N s
    re.compile(r"(?:complete|完成)[^0-9\n]{0,30}elapsed\s*[=:]\s*\**\s*"            # *complete elapsed=979.4/979.3 s
               r"([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?\s*s", re.I),
    re.compile(r"train_runtime\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b"),             # train_runtime=2655 s
    re.compile(r"elapsed_time\s*[=:]\s*[\"']?([0-9]+)m\s*([0-9]+)s"),               # elapsed_time=44m 14s
    re.compile(r"训练\s*[0-9]+\s*步\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*s"),              # 训练 200 步 / 1981.8 s
    re.compile(r"elapsed\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*s\b"),                    # elapsed=536.9s
    # ↑ ⛔ 不取「评测侧单样本 elapsed_sec ≈ 3.99 s」这类：该形态是 `elapsed_sec ≈`（下划线/≈）⇒ 本式不匹配。
)
ELAPSED_CUM_PATTERNS = (
    re.compile(r"(?:trainer\s*末步|末步|末|train_step|sft_step)[^0-9\n]{0,20}"      # trainer 末步 `elapsed_sec=977.2 s`
               r"elapsed_sec\s*[=:]\s*≈?\s*([0-9]+(?:\.[0-9]+)?)\s*s"),
    re.compile(r"(?:elapsed_sec\s*累计|步内累计)\s*([0-9]+(?:\.[0-9]+)?)\s*s"),
)
ELAPSED_PER_PATTERNS = (
    re.compile(r"batch_sec[^0-9\n]{0,20}mean\s*[=＝]?\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*s"),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s/step"),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s/it"),
    re.compile(r"每步(?:约|平均)?\s*([0-9]+(?:\.[0-9]+)?)\s*s"),
)

#: ★ 供 verify 断言用：耗时解析的**输入域**（这些字段里出现墙钟形态 ⇒ 表格列不得为空）
ELAPSED_SOURCE_KEYS = ("throughput_detail", "throughput", "runs", "provenance_command")


def elapsed_source_blob(row):
    """★ 耗时解析输入域（**单一真源**：生成器与校验器共用）。"""
    return " ".join(str(row.get(k) or "") for k in ELAPSED_SOURCE_KEYS)


def split_elapsed(tp):
    """由跑次读数文本抽出「耗时」口径（总时长 / 训练累计 / 每步约）。抽不到记 '—'（⛔ 不编造）。

    ★ 2026-10-15 扩解析：原先只认「墙钟 HH:MM:SS / 墙钟 N s / 步内累计 / 每步约」四种，
      ⇒ 全表 54/54 `elapsed` 为空（实证：T011 的 977.2/979.4 s 明明在 `throughput_detail` 里）。
      ★ 本次按**现行实际文本形态**补足（见 `ELAPSED_*_PATTERNS`），并**⛔ 不取评测侧读数**。
    """
    if not tp or tp in ("—",):
        return "—"
    t = str(tp)
    tot = cum = per = None
    for p in ELAPSED_TOTAL_PATTERNS:
        m = p.search(t)
        if not m:
            continue
        if m.re.groups == 2:                      # elapsed_time=44m 14s
            tot = f"{m.group(1)}m{m.group(2)}s"
        else:
            tot = str(m.group(1))
            if not tot.endswith(" s") and not re.search(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", tot):
                tot = f"{tot} s"
        break
    for p in ELAPSED_CUM_PATTERNS:
        m = p.search(t)
        if m:
            cum = f"{m.group(1)} s"
            break
    for p in ELAPSED_PER_PATTERNS:
        m = p.search(t)
        if m:
            per = f"{m.group(1)} s"
            break
    parts = []
    if tot:
        parts.append(f"总时长 {tot}")
    if cum:
        parts.append(f"训练累计 {cum}")
    if per:
        parts.append(f"每步约 {per}")
    return "；".join(parts) if parts else "—"


#: ★ 耗时列的标签与固定展示顺序（单一真源）
ELAPSED_LABELS = ("总时长", "训练累计", "每步约")


def _elapsed_parts(text):
    """把「总时长 X；训练累计 Y；每步约 Z」拆成 {标签: 值}。"""
    out = {}
    for seg in str(text or "").split("；"):
        m = re.match(r"^\s*(总时长|训练累计|每步约)\s*(.+?)\s*$", seg)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _elapsed_num(s):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(s or ""))
    return float(m.group(1)) if m else None


def resolve_elapsed(row, tp_raw):
    """★ 「耗时」列取值的**唯一入口**（三层，⛔ 不编造）：
        ① 矩阵内**跑次文本**解析（`split_elapsed`）；
        ② 228 **只读侦察**所得 `elapsed_evidence`（`status=found`）：① 为 '—' 时用它；① 已有则只**补缺标签**
           （同一标签两来源数值差 >2% ⇒ **fail-closed**，⛔ 不猜、⛔ 不覆盖）；
        ③ `elapsed_evidence.status=not_found` ⇒ 写「**未取到（<具名原因>）**」（⛔ 不留空白、⛔ 不写估计值）。
    """
    parsed = _elapsed_parts(split_elapsed(elapsed_source_blob(row) or tp_raw))
    ev = row.get("elapsed_evidence") if isinstance(row.get("elapsed_evidence"), dict) else {}
    st = str(ev.get("status") or "")
    if st == "not_found":
        reason = str(ev.get("reason") or "").strip()
        if not reason:
            raise SystemExit(f"FAIL-CLOSED：{row.get('tier')} elapsed_evidence=not_found 但缺具名原因")
        if parsed:
            raise SystemExit(f"FAIL-CLOSED：{row.get('tier')} 跑次文本已有读数却又标 not_found ⇒ 口径矛盾")
        return f"未取到（{reason}）"
    if st == "found":
        evp = _elapsed_parts(ev.get("value_text"))
        for k, v in evp.items():
            if k in parsed:
                a, b = _elapsed_num(v), _elapsed_num(parsed[k])
                if a is not None and b is not None and abs(a - b) > 0.02 * max(abs(a), 1.0):
                    raise SystemExit(
                        f"FAIL-CLOSED：{row.get('tier')} 「{k}」两来源冲突：跑次文本 {parsed[k]} vs 228 侦察 {v}")
        merged = {k: v for k, v in parsed.items()}
        merged.update({k: v for k, v in evp.items() if k not in merged})
    else:
        merged = dict(parsed)
    if not merged:
        return "—"
    return "；".join(f"{k} {merged[k]}" for k in ELAPSED_LABELS if k in merged)


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


# ★★ 2026-10-16 追补（审查项 🔴R1 整改）——「发布件里不得出现与现行行 status 矛盾的结论断言」★★
#
# 病史：lr 口径变更（按卡数缩放机制整体删除 + lr 按方法一个值）后，37 档的旧跑次读数与现行声明参数
#   不再一一对应 ⇒ 行级 `status` 回落为「尚未验证」、`voided_conclusion` 记录作废；但**历史叙述字段**
#   （`runs` / `note_full` / `caliber_change.rule` / `caliber_verdict` / `conclusion_annotation` …）
#   仍逐字保留旧四态结论句（「⇒ 判「已验证不可用」」「⇒ 翻为「已验证可用」」…）⇒ **同一份发布件内自相矛盾**
#   （复查者 🟡R1 点名 T012/T024；本包系统性扫查另命中 32 处，见工位 report.md §③）。
#
# 处置（只追加、⛔ 不删改原句 ⇒ 历史完整保留；幂等：已有作废标记者跳过）：
#   凡本行结论因该审计作废者（`status == 尚未验证` 且 `reading_consistency.consistent is False`），
#   行内**含四态结论措辞**的历史字段逐条追加日期化追补句 ⇒ 历史句一律被显式「已作废/追补」包裹。
VOID_ADDENDUM_20261016 = (
    "★★ 2026-10-16 追补：本条所记结论已因 lr 口径变更作废，"
    "以本行现行 `status` 为准（尚未验证）。"
)
#: 「四态结论措辞」的字面（出现即视为该字段在断言四态之一）
_FOUR_STATE_LITERALS = ("已验证可用", "已验证不可用")
#: 已有的作废/留痕标记（命中即认定该历史句**已被包裹** ⇒ ⛔ 不重复追加，保证幂等与不啰嗦）
_VOID_WRAP_MARKERS = ("已作废", "追补", "作废", "留痕", "不得引用", "保留为历史",
                      "不得视为", "已被取代", "已被推翻")
#: 机读「状态快照」键：键名本身即声明历史角色（作废前 / 后状态快照），⛔ 不追加散文追补
_AUDIT_SNAPSHOT_KEYS = ("status_before", "status_after")

# ============================================================================================
# ★★ 2026-10-16 **评测集口径变更（V1 = fit 口径）** —— 与 lr 口径**并列的第二条作废原因** ★★
#
# 主席 2026-10-16 裁定（原话「我们只要同一个数据集，先训练一遍，然后测试的时候还是这个数据集」）：
#   **100 行档**（判据 = 矩阵该档 `subset_size == 100`，即 T001–T027）的**评测集改为该档自身
#   训练用的同一份文件**（100 行、`dataset_split: train`）⇒ 读数含义 = **fit 口径（训练集自评）**，
#   ⛔ 不得读作训练分布之外的表现。
# ⇒ 后果判据：**旧读数取自 702 行评测集、与本行现行评测集口径不再一一对应** ⇒ 结论作废、
#   该档标「尚未验证」；旧读数保留于 `voided_conclusion.historical_effect`（⛔ 不抹）。
# ★ 只对 100 行档生效（行级机读判据 = `eval_caliber_consistency.consistent is False`，
#   ⛔ 本文件不写死档号）：20 行档（n=20 ⇒ 每样本 5.00 pp ＝ GRASPO 通过线 ⇒ 判据退化为
#   1 个样本）**不适用**、9 个 formal 档**不动**（训练源与 702 评测集同分布）。
# ★ 与 lr 口径的关系：两批档**有交叠**（22 档同时命中）⇒ ⛔ **不覆盖**既有 `voided_conclusion`
#   与既有 `caliber_cause`，第二条原因**叠加**写入（`voided_conclusion.superimposed` ＋ 数据侧散文）。
EVAL_CALIBER_AUDIT = "eval-caliber-audit/v1"
EVAL_CALIBER_ADDENDUM_20261016 = (
    "★★ 2026-10-16 追补：本条所记结论已因**评测集口径变更**作废 —— "
    "100 行档的评测集改为**该档自身训练文件**（100 行、`dataset_split: train` ⇒ fit 口径），"
    "旧读数取自 702 行评测集 ⇒ 以本行现行 `status` 为准（尚未验证）。"
)
VOID_ADDENDUM_BOTH_20261016 = (
    "★★ 2026-10-16 追补：本条所记结论已作废（**lr 口径变更** ＋ **评测集口径变更** 两条原因叠加），"
    "以本行现行 `status` 为准（尚未验证）。"
)


def _void_kind(row):
    """该行结论作废的原因类别（★ 数据驱动，⛔ 不写死档号）：`lr` / `eval` / `both` / `None`。"""
    _rcc = row.get("reading_consistency")
    lr = isinstance(_rcc, dict) and _rcc.get("consistent") is False
    _ecc = row.get("eval_caliber_consistency")
    ev = isinstance(_ecc, dict) and _ecc.get("consistent") is False
    if lr and ev:
        return "both"
    if ev:
        return "eval"
    if lr:
        return "lr"
    return None


def _void_addendum(row):
    """该行适用的「作废追补句」；**无口径作废** ⇒ 返回 ``None``。

    ★ 逐行按原因类别取（lr / 评测集 / 两条叠加），⛔ 不一把梭用 lr 措辞；
    ★ 返回 ``None`` 是**必须**的：调用方 `_fix_adjacent` 对「非口径作废」的档
      （如 `已验证不可用` 的 OOM 档）要回退到**按现行四态**选追补句，⛔ 不能被这里的默认值顶掉。
    """
    return {"eval": EVAL_CALIBER_ADDENDUM_20261016,
            "both": VOID_ADDENDUM_BOTH_20261016,
            "lr": VOID_ADDENDUM_20261016}.get(_void_kind(row))


def _naked_four_state(text):
    """该文本是否**裸**断言四态之一（含四态字面、且无任何作废/留痕标记）。"""
    if not any(lit in str(text) for lit in _FOUR_STATE_LITERALS):
        return False
    return not any(m in str(text) for m in _VOID_WRAP_MARKERS)


def wrap_voided_conclusions(rows):
    """★★ 把「因口径一致性审计作废」的档位行内**裸四态结论句**逐条追加日期化追补。

    ★ 生效面（**数据驱动**）：`status == 尚未验证` 且（`reading_consistency.consistent is False`【lr 口径】
      或 `eval_caliber_consistency.consistent is False`【评测集口径】）的行；
    ★ 追补句**逐行按原因类别取**（lr / 评测集 / 两条叠加），⛔ 不把评测集口径的档套上 lr 措辞；
    ⛔ 不改原句（追加在句尾）；⛔ 不碰 `status_before/status_after` 等机读快照枚举。
    """
    for r in rows:
        if r.get("status") != S_UNVERIFIED:
            continue
        if _void_kind(r) is None:
            continue
        _add = _void_addendum(r)
        for key, val in list(r.items()):
            if isinstance(val, str):
                if _naked_four_state(val):
                    r[key] = val + " " + _add
            elif isinstance(val, dict):
                for k2, v2 in list(val.items()):
                    if k2 in _AUDIT_SNAPSHOT_KEYS:
                        continue
                    if isinstance(v2, str) and _naked_four_state(v2):
                        val[k2] = v2 + " " + _add
    return rows


# ============================================================================================
# ★★ 2026-10-16 追补（🟡W1 + 扫查规则补齐，task-matrix-r3-fix）：
#    前一包的「系统性扫查 33→0」规则偏窄，漏了 `superseded_note` / `blocked_reason` 等字段
#    （详见工位 report.md §④）。本段把规则**补全为**：
#      · 扫查面 = 行内**全部字段**（递归每一层）＋ meta 派生文本；
#      · 断言面 = 四态字面（已验证可用/已验证不可用/尚未验证/未纳入验收）＋「当前有效读数」；
#      · 矛盾面 = 断言与本行现行 `status` 不一致（「由「X」翻为「Y」」按**单个断言 Y**计，
#        X 是过渡叙述的旧状态，⛔ 不算独立断言）；
#      · 包裹面 = 同字段内**紧邻（标记起点 ↔ 断言起点 ≤ 40 字符）**的作废/追补标记。
#    ⚠️ 前一包把光杆「留痕」当标记 ⇒ T032 的 `superseded_note`（「旧测量留痕」）
#       被误判为已包裹而漏网；本段的标记集**不含**「留痕」「旧读数」这类只修饰旧测量的词。
#    处置 = 在原断言**紧邻处前缀**作废标记 ＋ 字段末段补日期化追补；⛔ 不改原句。
# ============================================================================================
VOID_MARK_ADJ_20261016 = "**【已作废，见末段追补】**"
#: 2026-10-14 主席口径（跑了没成功即记失败）追补 —— 供 `status = 已验证不可用` 的档用
OOM_ADDENDUM_20261016 = (
    "★★ 2026-10-16 追补：本条旧表述「保持尚未验证」**已失效**；"
    "按 2026-10-14 主席口径（OOM／超时／未实现／跑过未成功一律记失败），"
    "本档现状为 `已验证不可用`，以本行现行 `status` 为准。"
)
#: 结论类断言：四态字面 ＋ 措辞
_ASSERT_PHRASES = ("当前有效读数",)
#: 判定「紧邻」用的标记集（★ ⛔ 不含「留痕」——它只修饰旧测量，不构成「本条结论已作废」声明）
_ADJ_MARKERS = ("作废", "追补", "不得引用", "不得视为", "已被取代", "已被推翻",
                "保留为历史", "旧读数无效", "不作为判定")
#: 标记就近插入点（在这些分隔符之后插入）
_ADJ_DELIMS = set("。；\n★⇒—")
_MAX_ADJ_GAP = 40
_TRANS_RE = re.compile(r"由[「（](已验证[^」）]*)[」）]\s*(?:翻为|改为|转为)[「（](已验证[^」）]*)[」）]")


def _stale_asserts(text, cur_status):
    """返回 [(start, end, 断言原文)]：与本行现行 `status` 矛盾的结论类断言。"""
    old = [m.span(1) for m in _TRANS_RE.finditer(text)]
    out = []
    for lit in STATUS_VOCABULARY:
        for m in re.finditer(re.escape(lit), text):
            if any(m.start() == s and m.end() == e for s, e in old):
                continue                        # 过渡叙述的**旧状态**，⛔ 非独立断言
            if lit != cur_status:
                out.append((m.start(), m.end(), m.group(0)))
    for ph in _ASSERT_PHRASES:
        if cur_status in (S_UNVERIFIED, S_EXCLUDED):
            for m in re.finditer(re.escape(ph), text):
                out.append((m.start(), m.end(), m.group(0)))
    out.sort()
    return out


def _adj_gap(text, pos):
    """该位置到同字段内最近作废/追补标记的字符距离（标记起点 ↔ 断言起点）；无标记 ⇒ None。"""
    best = None
    for mk in _ADJ_MARKERS:
        for m in re.finditer(re.escape(mk), text):
            g = abs(m.start() - pos)
            if best is None or g < best:
                best = g
    return best


def _insert_adj_marker(text, a_start):
    """在断言 a_start 之前**就近**插入标记（优先落在最近的分隔符之后；过远则直接贴断言前）。"""
    ins = None
    for j in range(a_start - 1, max(-1, a_start - 121), -1):
        if text[j] in _ADJ_DELIMS:
            ins = j + 1
            while ins < len(text) and text[ins] in " \u3000":
                ins += 1
            break
    if ins is None:
        ins = a_start
    if (a_start + len(VOID_MARK_ADJ_20261016) + 1) - ins > _MAX_ADJ_GAP:
        ins = a_start
    return text[:ins] + VOID_MARK_ADJ_20261016 + " " + text[ins:]


def _fix_adjacent(text, cur_status, addendum=None):
    """反复加标记直到本字段无「无紧邻标记的矛盾断言」；再在字段末段补日期化追补。

    ★ `addendum` 由调用方**逐行**给出（lr 口径 / 评测集口径 / 两条叠加）——⛔ 不再写死 lr 措辞，
      否则评测集口径作废的档会被挂上"因 lr 口径变更作废"的**错误**追补。
    """
    for _ in range(12):
        hits = [s for s, _e, _r in _stale_asserts(text, cur_status)
                if _adj_gap(text, s) is None or _adj_gap(text, s) > _MAX_ADJ_GAP]
        if not hits:
            break
        for s in sorted(set(hits), reverse=True):
            text = _insert_adj_marker(text, s)
    if VOID_MARK_ADJ_20261016 in text and "2026-10-16 追补" not in text:
        add = addendum or (OOM_ADDENDUM_20261016 if cur_status == S_UNAVAILABLE
                           else VOID_ADDENDUM_20261016)
        text = text.rstrip() + " " + add
    return text


def ensure_adjacent_void_markers(rows):
    """★★ W1：把行内**全部字段**（递归每一层）中「与本行现行 status 矛盾的结论断言」
    逐条加**紧邻**作废标记（≤40 字符）；⛔ 不改原句；幂等（已紧邻者跳过）。

    ★ 必须在本函数**之后**再派生 `reading_change_registry`（registry 取行内 `caliber_change.rule`
      ⇒ 否则 registry 条目会漏掉标记）。★ 本函数在 `note_full` 组装**之后**调用，
      故派生键 `note_full` 亦被覆盖（⛔ 只改上游数据做不到这一点）。
    """
    for r in rows:
        st = r.get("status")
        if st not in STATUS_VOCABULARY:
            continue
        _add = _void_addendum(r)
        for key, val in list(r.items()):
            if key in _AUDIT_SNAPSHOT_KEYS:
                continue
            if isinstance(val, str):
                if _stale_asserts(val, st) or VOID_MARK_ADJ_20261016 in val:
                    nv = _fix_adjacent(val, st, _add)
                    if nv != val:
                        r[key] = nv
            elif isinstance(val, dict):
                for k2, v2 in list(val.items()):
                    if k2 in _AUDIT_SNAPSHOT_KEYS or not isinstance(v2, str):
                        continue
                    if _stale_asserts(v2, st) or VOID_MARK_ADJ_20261016 in v2:
                        nv = _fix_adjacent(v2, st, _add)
                        if nv != v2:
                            val[k2] = nv
            elif isinstance(val, list):
                for i, v2 in enumerate(val):
                    if not isinstance(v2, str):
                        continue
                    if _stale_asserts(v2, st) or VOID_MARK_ADJ_20261016 in v2:
                        nv = _fix_adjacent(v2, st, _add)
                        if nv != v2:
                            val[i] = nv
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=None)
    ap.add_argument("--self-test", action="store_true", help="脱敏回归自检（对抗性输入 + 顺序无关）")
    ap.add_argument("--date", default=None,
                    help="文档日期；缺省读输入数据的 meta.doc_date。⛔ 不读时钟（§17.7 生成确定性）。")
    a = ap.parse_args()
    if getattr(a, "self_test", False):
        return _sanitizer_self_test()

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
        elif isinstance(raw, str) and raw.startswith("通过"):
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
        # ★★ 口径变更（主席 2026-10-14）：**"跑了没成功＝失败"** ⇒ 命中 `caliber_verdict` 的档
        #    按该值记四态（★ 真因仍逐字保留在 `blocked_reason`/备注/`anomalies`，⛔ 不抹证据）。
        #    ★ 但若该档同时命中「读数-配置一致性」作废（下表）⇒ 以作废为**最终**四态（⛔ 不被覆盖）。
        _rcc = r.get("reading_consistency")
        _voided_reading = bool(_rcc) and isinstance(_rcc, dict) and _rcc.get("consistent") is False
        _cv = r.get("caliber_verdict")
        if _cv in STATUS_VOCABULARY and not _voided_reading:
            state = _cv
            r["caliber_change"] = {
                "effective": "2026-10-14",
                "rule": "跑了没成功＝失败（OOM／超时／未实现／硬件不可行 均记失败）",
                "source": "主席口径变更（2026-10-14）",
            }
        # ★★ 读数-配置一致性（审查项 🔴B1/B2）：**结论必须与同一行的参数一一对应** ——
        #    凡跑次读数取自旧参数（旧 lr 口径 / 旧 epoch）而本行参数已更正者 ⇒ 结论**作废**，
        #    该档标「尚未验证」；旧读数与旧结论**保留为历史**（机器可读字段仍在行内，⛔ 不抹）。
        if _voided_reading:
            state = S_UNVERIFIED
            r["voided_conclusion"] = {
                "kind": "reading_config_mismatch",
                "audit": _rcc.get("audit"),
                "status_before": _rcc.get("status_before"),
                "current": _rcc.get("current"),
                "reading": _rcc.get("reading"),
                "mismatch": _rcc.get("mismatch"),
                "reason": _rcc.get("cause"),
            }
        # ★★ 评测集口径一致性（V1 = fit 口径，2026-10-16 主席裁定）：100 行档的**评测集**
        #    由 702 行改为**该档自身训练文件**（100 行、`dataset_split: train`）⇒ 旧读数与
        #    本行现行**评测集口径**不再一一对应 ⇒ 结论**作废**、该档标「尚未验证」。
        #    ★ ★ 与 lr 口径的关系（**强制**）：两批档有交叠（22 档）⇒ 若该档**已**因 lr 口径作废
        #      （或已由 `conclusion_annotation` 作废）⇒ ⛔ **不覆盖**既有 `voided_conclusion`，
        #      只把第二条原因**叠加**为 `superimposed`（⛔ 既有 `kind` / `reading` / `mismatch` 逐字保留）。
        _ecc = r.get("eval_caliber_consistency")
        if isinstance(_ecc, dict) and _ecc.get("consistent") is False:
            state = S_UNVERIFIED
            _prev = r.get("voided_conclusion")
            if isinstance(_prev, dict):
                _prev["superimposed"] = {
                    "kind": "eval_caliber_change",
                    "audit": _ecc.get("audit"),
                    "current": _ecc.get("current"),
                    "reading": _ecc.get("reading"),
                    "mismatch": _ecc.get("mismatch"),
                    "reason": _ecc.get("cause"),
                }
            else:
                r["voided_conclusion"] = {
                    "kind": "eval_caliber_change",
                    "audit": _ecc.get("audit"),
                    "status_before": _ecc.get("status_before"),
                    "current": _ecc.get("current"),
                    "reading": _ecc.get("reading"),
                    "mismatch": _ecc.get("mismatch"),
                    "reason": _ecc.get("cause"),
                }
        r["status"] = state
        r.pop("spec_status", None)

        v = verdict_of(r, state)
        # ★ 读数已录但阈值未定（measurement_pending）⇒ 保留读数列，状态仍「尚未验证」（读数不作判定）
        pending = bool(r.get("measurement_pending"))
        concluded = (state in CONCLUDED) or pending
        # 实验数据四列：只有已结论档才有读数；其余记 '—'
        if concluded:
            tp_raw = r.get("throughput", "—")
            r["throughput_detail"] = tp_raw        # ★ 归一化**之前**的原文（含 ≈3.9 s/step 等可核数）
            # ★ 耗时 = 「跑次文本解析 ⊕ 228 只读侦察证据」；取不到 ⇒ 「未取到（具名原因）」（单一入口，fail-closed）
            r["elapsed"] = resolve_elapsed(r, tp_raw)
            r["throughput"] = split_throughput(tp_raw)
            w = window_text(r.get("runs", ""))
            r["effect"] = (r.get("effect", "—") + ("；" + w if w else "")).strip("；") or "—"
        else:
            # ★★ 结论作废（读数-配置不对应）档：⛔ 旧读数不得**抹掉** ⇒ 原文迁入
            #    `voided_conclusion.historical_effect` 作为历史（结果列本身仍按规则置 '—'）。
            if r.get("voided_conclusion") and str(r.get("effect") or "").strip() not in ("", "—"):
                r["voided_conclusion"]["historical_effect"] = r["effect"]
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
        # ★ 双锚（树 + 官方基底/包装）与吞吐口径 ⇒ 随件进备注（数值全取自跑次件，⛔ 不臆造）
        _ci = r.get("code_identity") or {}
        if _ci:
            _anchors = []
            if _ci.get("tree_fingerprint"):
                _anchors.append("树 " + str(_ci["tree_fingerprint"])[:8] + "…")
            for _k in ("image_base_official", "image_run_wrapper"):
                _v = _ci.get(_k)
                if isinstance(_v, dict) and _v.get("digest"):
                    _anchors.append(str(_v.get("role") or _k) + " " + str(_v["digest"])[:8] + "…")
            if _anchors:
                r["note"] = (r.get("note") or "") + " ★ 锚点：" + " + ".join(_anchors)
        _tp = str(r.get("throughput_detail") or r.get("throughput") or "")
        if "s/step" in _tp:
            import re as _re
            _m = _re.search(r"[≈~]?\s*([0-9.]+)\s*s/step", _tp)
            if _m:
                r["note"] = (r.get("note") or "") + f" ★ 吞吐（如实）：SFT 无生成式 tok/s；均值 ≈{_m.group(1)} s/step。"

        # ★ P5/P6 六项（标准交付项）⇒ 逐档结构化备注段（逐项可读；"待补"如实标）
        _p = r.get("p5p6_readings")
        if _p:
            _lab = {"1_group_integrity": "① 组完整性恒等式", "2_retry_count": "② retry_count 真值",
                    "3_group_decision_distribution": "③ 组决策分布", "4_reward_alignment": "④ 奖励对齐",
                    "5_steps_mapping": "⑤ steps 映射", "6_batch_high_retry_rate": "⑥ batch_high_retry_rate"}
            _seg = " ｜ ".join(f"{v}：{_p[k]}" for k, v in _lab.items() if _p.get(k))
            r["p5p6"] = _seg
            if _seg:
                r["note"] = ((r.get("note") or "") + " ★★ P5/P6 六项对账：" + _seg)

        # ★ 实跑步数（有跑次记录即填，含"作废参考记录"）
        scr = r.get("step_count_reconciliation") or {}
        if scr.get("actual_steps") is not None:
            r["steps_actual"] = f"{scr.get('actual_steps')}（推导 {scr.get('derived_steps')}）"
        else:
            _t = r.get("runs", "") or ""
            m = re.search(r"(\d+)\s*/\s*(\d+)\s*步", _t)
            if m:
                r["steps_actual"] = f"{m.group(1)}/{m.group(2)}"
            else:
                # ★ 回退：接受"2 epoch / 100 步"这类形态（取最后出现的 "N 步"）
                _all = re.findall(r"(\d+)\s*步", _t)
                r["steps_actual"] = _all[-1] if _all else "—"
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
            _pc = " / ".join(v["passed_criteria"])
            r["note"] = ("未过" + v["failed_criterion"] + "：" + v["failed_detail"] +
                         ((" 已通过：" + _pc) if _pc else "") +
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
        else:
            # ★★ 翻为「已验证可用」（或本就不需要结论块）⇒ ⛔ **不得残留旧的失败结论块**
            #    （实证：T011 由判据① 失败翻为可用时，旧的「第 1 条未过」会留在可用行上）
            r.pop("verdict", None)
        # ★★ 短/长备注分离（**行循环最末端**：所有备注追加/覆盖之后；⛔ 不在中途插入）
        _full = str(r.get("note") or "")
        _sup = str(r.get("superseded_note") or "").strip()
        if _sup:
            _full = (_full + "\n" + _sup).strip()
        if _full and _full != "—":           # ★ 仅对"确有结论/读数"的档做短/长分离；⛔ 不碰未跑档的占位符
            r["note_full"] = _full
            r["note"] = short_note(r, _full)
        rows.append(r)

    # ★★ 2026-10-16 追补（🔴R1）：先把「因读数-配置一致性审计作废」的行内**裸四态结论句**
    #    逐条包裹（只追加、⛔ 不改原句）⇒ 其后派生的 registry 亦自动携带该追补。
    wrap_voided_conclusions(rows)
    # ★★ 2026-10-16（🟡W1 + 扫查规则补齐）：把「无紧邻作废标记的矛盾断言」逐条加**紧邻**标记。
    #    ★ 必须在 registry 派生**之前**调用（registry 从行内 `caliber_change.rule` 取文本）。
    ensure_adjacent_void_markers(rows)

    # ★★ 「读数变化登记表」：**数据驱动**地从档位行派生（⛔ 不写死档号/数值）——
    #    凡行级 `caliber_change.kind` 以 `reading_change` 开头者 ⇒ 逐条进 `criteria_caliber.caliber_change`。
    _cal = criteria_caliber()
    _reg = []
    for _r in rows:
        _cc = _r.get("caliber_change")
        if not (isinstance(_cc, dict) and str(_cc.get("kind", "")).startswith("reading_change")):
            continue
        _b, _a = _cc.get("reading_before") or {}, _cc.get("reading_after") or {}
        def _rd(x):
            if x.get("delta_pp") is not None:
                return f"Δ={x.get('delta_pp')} pp（epoch={x.get('epoch')}；{x.get('why', '')}）"
            return f"无实测读数（epoch={x.get('epoch')}；{x.get('why', '')}）"
        # ★★ 四态是否翻盘**逐条数据驱动**（⛔ 不写死）：`caliber_change.four_state_flip`
        #    缺省 True（兼容既有 T011 等翻盘档的既有渲染文本）；T024 为「读数由无效转有效 + 归因更正，
        #    但四态未变（仍为不可用）」⇒ 显式 False。
        _flip = "四态翻盘" if _cc.get("four_state_flip", True) else "四态未变（★ 仅归因与读数更正）"
        # ★★ 2026-10-16 追补（🔴R1）：该档若已被口径一致性审计作废（**原因类别逐行取**：
        #    lr 口径 / 评测集口径 / 两条叠加）⇒ 条目**末尾再加一句**追补，
        #    与条目内 `rule` 上携带的追补一起构成「显式包裹」（只追加、⛔ 不改原句）。
        _voided_entry = _r.get("status") == S_UNVERIFIED and _void_kind(_r) is not None
        _entry = (
            f"★ {_cc.get('effective', '')} **{_r.get('tier')}**：{_flip} —— {_cc.get('rule', '')} "
            f"★ 旧读数 {_rd(_b)}**无效、⛔ 不得引用**（保留为历史）；"
            f"★ 新读数 {_rd(_a)}**有效**。"
            f"★ 与判据口径的关系：{_cc.get('verdict_unchanged_by_caliber', '')} "
            f"来源：{_cc.get('source', '')}。{_cc.get('epoch_marker', '')}")
        if _voided_entry:
            # ★ 逐行按原因类别取追补句；★ 条目内（`rule` 上）已逐字带同一句 ⇒ ⛔ 不重复追加。
            _add = _void_addendum(_r)
            if _add and _add not in _entry:
                _entry += " " + _add
        _reg.append(_entry)
    # ★★ 2026-10-16 追补（🔴R1 第③条）：把「**本次 lr 口径变更导致读数作废**」正式登记进本清单。
    #    ★ 数据驱动：作废范围由行内 `reading_consistency.consistent is False` 现算（⛔ 不写死档号/数字）；
    #    ★ 口径依据一律取 `criteria_caliber.lr_policy`（唯一真相源在 `extract_spec.LR_POLICY`）。
    _lrvoid_tiers = [_r.get("tier") for _r in rows
                     if isinstance(_r.get("reading_consistency"), dict)
                     and _r["reading_consistency"].get("consistent") is False]
    if _lrvoid_tiers:
        _lp = _cal.get("lr_policy") or {}
        _reg.append(
            "★ 2026-10-16 **本次 lr 口径变更 ⇒ 读数作废（正式登记）**：★ 新口径 = "
            f"{_lp.get('rule', '')} ★ 机制面 = {_lp.get('mechanism_removed', '')} "
            f"★ **作废范围 = {len(_lrvoid_tiers)} 档**"
            f"（{', '.join(str(x) for x in _lrvoid_tiers)}；"
            "逐档机读判据 `reading_consistency.consistent == false`，即现行 `status = 尚未验证` 的档）"
            "—— 这些档的读数取自**旧口径跑次**（native：生效 lr = 配置 lr × dp_size；"
            "或旧 epoch）⇒ **结论作废、⛔ 不得引用**；"
            "★ 旧读数与旧结论一律保留于行内 `voided_conclusion` / `reading_consistency`（⛔ 不抹）。"
            f"★ 依据 = `meta.criteria_caliber.lr_policy`（source = {_lp.get('source', '')}；"
            f"effective = {_lp.get('effective', '')}；single_source = {_lp.get('single_source', '')}）；"
            f"★ 后果判据 = {_lp.get('consequence', '')}")
    # ★★ 2026-10-16 追补（本次评测集口径变更 · 第二条原因）：把「**评测集改为该档自身训练文件
    #    ⇒ 100 行档旧读数作废**」正式登记进同一清单（★ 与上面 lr 条目**并列**，⛔ 不合并、不覆盖）。
    #    ★ 数据驱动：作废范围由行内 `eval_caliber_consistency.consistent is False` 现算（⛔ 不写死档号）；
    #    ★ 口径依据一律取 `criteria_caliber.eval_caliber`（本函数上方生成，单一入口）。
    _ecvoid_tiers = [_r.get("tier") for _r in rows
                     if isinstance(_r.get("eval_caliber_consistency"), dict)
                     and _r["eval_caliber_consistency"].get("consistent") is False]
    if _ecvoid_tiers:
        _ec = _cal.get("eval_caliber") or {}
        _lrset = set(_lrvoid_tiers)
        _both = [t for t in _ecvoid_tiers if t in _lrset]
        _reg.append(
            "★ 2026-10-16 **本次评测集口径变更 ⇒ 读数作废（正式登记）**：★ 新口径 = "
            f"{_ec.get('rule', '')} ★ 标定 = **{_ec.get('fit_caliber_label', '')}** —— "
            "读数回答「该档训练是否生效」，⛔ **不得读作训练分布之外的表现**。"
            f"★ **口径证据（判别力）** = {_ec.get('discriminative_power', '')} "
            f"{_ec.get('pass_lines_unchanged', '')} "
            f"★ **作废范围 = {len(_ecvoid_tiers)} 档**"
            f"（{', '.join(str(x) for x in _ecvoid_tiers)}；"
            "逐档机读判据 `eval_caliber_consistency.consistent == false`）"
            "—— 这些档的读数取自**旧 702 行评测集**、与本行现行评测集口径不再一一对应 ⇒ "
            "**结论作废、⛔ 不得引用**；★ 旧读数与旧结论一律保留于行内 "
            "`voided_conclusion.historical_effect`（⛔ 不抹）。"
            f"★ **与 lr 口径的叠加（⛔ 不重复标注、⛔ 不覆盖既有 `caliber_cause`）**："
            f"两批**有交叠 {len(_both)} 档**（{', '.join(str(x) for x in _both)}）"
            "⇒ 这些档的作废原因是**两条并列**（lr 口径变更 ＋ 评测集口径变更），"
            "机读字段 `voided_conclusion.superimposed` 记第二条；"
            f"★ 仅因本次评测集口径变更作废的档 = {len(_ecvoid_tiers) - len(_both)} 档"
            f"（{', '.join(str(x) for x in _ecvoid_tiers if x not in _lrset)}）。"
            f"★ **顺带订正** = {_ec.get('overlap_analysis_fix', '')} "
            f"★ **20 行档**：{_ec.get('not_applicable', '')} "
            f"★ **不动**：{_ec.get('untouched', '')} "
            f"★ 依据 = `meta.criteria_caliber.eval_caliber`（source = {_ec.get('source', '')}；"
            f"effective = {_ec.get('effective', '')}）；{_ec.get('consequence', '')}")
    _cal["caliber_change"]["reading_change_registry"] = _reg

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
        "distill_teacher": distill_teacher(),
            "criteria_caliber": _cal,
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
    # ★★ 落盘前**全量脱敏**（数据集存放路径／机器号 ⇒ 占位）—— 与"脱敏顺序无关兜底"同族
    doc = _final_sanitize(doc)
    for _r in doc.get("rows", []):
        for _k in ("note", "note_full"):
            if _r.get(_k):
                _r[_k] = strip_dead_fragments(_r[_k])
    _banned = [b for b in BANNED_IN_MATRIX if b in json.dumps(doc, ensure_ascii=False)]
    if _banned:
        print(f"FAIL-CLOSED：矩阵件仍含禁用串 {_banned} ⇒ 拒绝落盘")
        return 1
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_VOCABULARY}
    print(f"official data: {len(rows)} 档；四态 {counts} -> {out}")


if __name__ == "__main__":
    main()
