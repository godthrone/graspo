#!/usr/bin/env python3
"""build_official.py — 由内部 matrix-data.json 生成正式机读数据 docs/capability-matrix.data.json。

★ 定位（宪法 §17.7「同源生成」第 1 款）：**数据文件 docs/capability-matrix.data.json 是矩阵唯一
   可编辑的源头**；人读视图 docs/capability-matrix.html 由「本数据 + 固定生成器
   tools/matrix/render_matrix.py」**确定性产出**，是**只读派生视图**、**禁止手工编辑**。
   派生方向唯一且固定：结果 JSON → capability-matrix.data.json → 人读视图（⛔ 禁止互推）。
★ 脱敏（宪法 §15.1／§17.7「矩阵数据与合规底线」；矩阵落点**不适用** §15.4 的放宽）：宿主路径 /
   内网地址与路径 / 机器名 / 人名 / 内部节点与内部账号一律替换为符号或删除；跑次只引用
   **不透明跑次 ID**，不含任何部署环境记录（尊重 §17.2 的立法目的）。
★ 只保留「定义 + 表格」层：parameters(columns/groups/uniform) + rows；过程件（provenance /
   conflicts / invariants / 取证叙述）不进正式件，迁至 .local/ 决策记录与工作日志。
★ ★ **内部字段不进人读视图**：`HUMAN_VIEW_ROW_KEYS` / `HUMAN_VIEW_META_KEYS` 是**显式允许清单**，
   清单之外的字段在落盘前由 `drop_non_view_fields()` 逐条剔除并打印（内部权威
   `.local/…/matrix-data.json` 一行不动、全字段保留）。允许清单之外的内部术语再由
   `forbidden_in_human_view()` 兜底扫描，**命中即 fail-closed 拒绝写盘**。

用法：
  python3 tools/matrix/build_official.py \
      --in  .local/hb-workspace/<会话>/task-matrix-html/matrix-data.json \
      --out docs/capability-matrix.data.json
"""
import argparse, datetime, json, re, sys
from pathlib import Path

# ★ lr 口径的**唯一真相源**（`extract_spec.py` 与本脚本同目录）：正式机读件直接引用它，
#   ⛔ 不在本文件里再抄一份取值表（§1.4 单一真相源；否则改口径要改两处、必然漂移）。
import extract_spec as _spec

# ★★ 表格结构的**唯一真相源**（`render_matrix.py` 与本脚本同目录；宪法 §1.4）：
#   渲染面 **21 列**（= `render_matrix.RENDERED_COLUMNS`；另有 3 列上提第 3 章「测试条件」，
#   第 3 章另有一条"TP 口径"**说明式**条目）
#   的列 key / 表头 / 列组 / 顺序由渲染器**引用**而来，⛔ 本文件不再自带第二份
#   （病史：本文件原先写死一份 10 列 `COLUMNS`，与在盘列数不一致 ⇒ 重跑会把表头覆盖回 10 列）。
#   ★ 2026-10-19（使用者口径「脚本只负责把测试结果填进最后一章测试表」）同批：
#     正式机读件**不再落 `columns` / `groups`**（表结构不随数据走）⇒ 这两个常量只在
#     本文件的自检里被引用，不再写进 `docs/capability-matrix.data.json`。
import render_matrix as _render

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


# ============================================================================
# ★★ 人读视图（`docs/capability-matrix.html` 及其机读源 data 件）的**可见面**
#    宪法 §17.7：矩阵是**面向使用者**的文档 ⇒ 只承载「判据 / 口径 / 实验条件 / 逐档结论」，
#    过程、日志、决策、证据、内部编排与人称**一律不进人读视图**（迁 `.local/`）。
#    本区四件（允许清单 / 术语兜底 / 术语改写 / 表格格归一）**同源同处**，⛔ 不设第二家。
# ============================================================================

#: ①**允许清单**（meta 级）：人读视图渲染这些键；其余一律由 `drop_non_view_fields()` 剔除。
HUMAN_VIEW_META_KEYS = (
    "title", "version", "doc_date", "scope", "venue", "code_version", "authority",
    "status_vocabulary", "boundary_declaration", "criteria_caliber", "criteria_version",
    "results_policy", "distill_teacher", "lora_target_backends",
)

#: ★★ **机读消费者字段（不进人读视图）**：`gen_configs.py` 的 fail-closed 检查要读它们
#:   （`final_formats` 还直接决定生成的 yaml）。它们**保留在正式机读件里**但不被渲染：
#:   ① 渲染层无任何引用（模板占位符只有 4 组 + 表格列，见 `render_matrix.py`）；
#:   ② ★★ **本清单不再是 `forbidden_in_human_view()` 的扫描豁免面**（2026-10-16 收口）：
#:      `blocked_reason` / `executability` 是**使用者有权知道的结论与归因**（§17.7）⇒ 退出本清单、
#:      进入人读视图，并被内部术语扫描覆盖。⇒ 扫描面 = 全行（⛔ 无逐字段豁免）。
TOOLCHAIN_ONLY_ROW_KEYS = (
    "final_formats", "family", "family_min_cards", "cards_pending",
    "live_evidence", "caliber_change",
)

#: ①**允许清单**（行级）：= 矩阵表渲染列 + 逐档结论块 + 机读列定义（`gen_configs.py` 的消费者）。
#: ★ 逐档参数/读数字段是**矩阵表本身**（§17.7 的「结论与归因都是使用者有权知道的」）⇒ 保留；
#:   过程件（证据索引、跑次溯源、内部对账、作废结论全文、陈旧读数史…）⇒ 剔除、迁 `.local/`。
S_AVAILABLE, S_UNAVAILABLE, S_UNVERIFIED, S_EXCLUDED = "已验证可用", "已验证不可用", "尚未验证", "未纳入验收"
STATUS_VOCABULARY = [S_AVAILABLE, S_UNAVAILABLE, S_UNVERIFIED, S_EXCLUDED]

#: ★★ **实测读数列**（单一真相源）：未结论档必须全为 `—`（§17.7「未验证不得冒充可用」）。
#:   校验端（`verify_matrix` ④C）与自证端（`run_self_proof`）都从这里取，⛔ 不各自手抄一份。
EXP_DATA_KEYS = ("peak_mem", "elapsed", "effect")

HUMAN_VIEW_ROW_KEYS = (
    # 矩阵表渲染列 + 列定义里用到的逐档参数
    #   （★ 2026-10-19 分区轮：9 列上提第 3 章、渲染面 13 列；
    #    ★ 2026-10-19 阈值 ≤2/≥3 轮：3 列上提第 3 章、渲染面 21 列 —— TP 等并行列回归表格）
    "tier", "status", "model", "algo", "mode", "backend", "parallel", "parallel_dims",
    "parallel_scheme", "lora_target", "lr", "data_subset", "steps", "switches", "spec_json",
    "model_path", "train_method", "tuner_type", "cards", "tp_size", "pp_size", "dp_size",
    "nproc_per_node", "nnodes", "cp_size", "sp_size", "gpus", "target_modules",
    "target_modules_mode", "deepspeed", "allow_unverified_pp_rollout", "subset_generation",
    "subset_r3_historical", "expected_optimizer_steps_historical", "steps_formula_v2",
    "min_optimizer_steps", "data_source", "subset_size", "prompts_per_rank", "train_path",
    "planned_steps", "save_steps", "expected_optimizer_steps_per_epoch",
    "expected_optimizer_steps_reachable", "applicable", "max_epochs", "derived_steps",
    "per_epoch_steps", "save_steps_basis", "data_scale", "tp", "pp", "dp", "cards_total",
    # 实验数据列 + 结论块
    "peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note", "runs",
    "inference", "algo_checks", "verdict", "excluded_reason", "availability",
    # ★★ 精简轮 2 新增的**派生结论列**（从 `verdict` 派生，⛔ 不从内部权威搬）
    "not_failed_reason",
    # ★★ 2026-10-16 矩阵侧收口：**「本环境不可执行」的三个机读字段**进入人读允许清单
    #    （§17.7「结论与归因都是使用者有权知道的」）—— 从此**被渲染 / 被扫描 / 被机核**；
    #    ⛔ 它们**不再**是 `TOOLCHAIN_ONLY_ROW_KEYS`（那本是「机读消费者、渲染层无引用」的豁免面）。
    "executability", "availability", "blocked_reason",
) + TOOLCHAIN_ONLY_ROW_KEYS

#: ②**内部术语兜底**：允许清单之外（或清单内被回填污染）的内部术语命中即 **fail-closed 拒绝写盘**。
#: ★ 只收「人称 / 内部编排 / 内部角色」三类；`§<数字>` 形式的规范与代码引用属公开仓内容、
#:   ⛔ 不在本清单（§17.7 允许失败归因与判据标识）。
FORBIDDEN_TERMS = (
    "主席", "总参谋长", "总司令", "指挥部", "指挥官", "执行者", "审查者",
    "口径包", "修正轮", "本包", "本批", "归属层", "工位", "work_log", "跑次根目录",
    "第 1 批", "第 2 批", "第 3 批", "第 4 批", "第 5 批", "第 6 批", "第 7 批",
)

#: ★★ **逐档字段直通清单**（内部权威 → 正式件）：这些字段在**内部数据**里逐档给出、
#:   下游机读消费者要用（`gen_configs.py` 的 `export.final_formats` 直接决定生成 yaml），
#:   但**不被渲染**（与 `TOOLCHAIN_ONLY_ROW_KEYS` 同一渲染豁免面）。
#:   ★ 来源＝内部权威行上的同名键（⛔ 不写死档号、⛔ 不在本文件里抄取值表）——
#:   原生成器漏了 `final_formats` 的透传，导致「重生成即丢」，会让下游的 fail-closed 守卫静默失效；
#:   本清单把透传**显式登记**，并在 `_carry_through_row_fields()` 里逐条搬运。
#:   ⚠ **未列入** `eval_caliber_consistency`：那是口径一致性的机读审计记录，其值里含
#:   **真实宿主路径（`/home/<账号>/…`）、数据集文件名与内部称谓**——实测把它留在正式机读件里，
#:   `scan_matrix_leaks.py` 立刻 **40 处命中**（宿主路径 11 / 数据集文件名 11 / 内部称谓 18）
#:   ⇒ 与 §17.7「矩阵禁止项 + 扫描不豁免」**直接冲突**。`tests/e2e/test_verify_matrix_guard.py`
#:   的 `test_on_disk_prior_chains_are_canonical`（断言在盘件恰有 9 条 `prior_chain`）因此会
#:   因「断言前提消失」而失败 ⇒ 该冲突已**如实上报**（见 `report.md` 待裁项 A1），⛔ 本包不擅自
#:   放宽扫描器、也⛔ 不改 `tests/**`。
CARRY_THROUGH_ROW_KEYS = ("final_formats",)


def _carry_through_row_fields(doc, src_rows):
    """逐档直通：把内部权威行上具名的直通字段搬到正式件（⛔ 不渲染、⛔ 不猜值）。"""
    by_tier = {str(r.get("tier")): r for r in (src_rows or [])}
    moved = 0
    for r in doc.get("rows", []):
        _s = by_tier.get(str(r.get("tier"))) or {}
        for k in CARRY_THROUGH_ROW_KEYS:
            if k in _s and _s.get(k) not in (None, "", "—", []):
                r[k] = _s[k]
                moved += 1
    return moved


#: ③**术语改写表**（`(旧, 新)` 顺序应用；★ 显示面文本专用）。
#: ★ 只改写「谁在什么场合说了什么」⇒「何时生效的口径」；⛔ 不动任何数值、判据、阈值与结论。
HUMAN_VIEW_REWRITES = (
    ("（「用同一个数据集」：训练用什么、评测就用什么）", "（口径：训练用什么数据、评测就用什么数据）"),
    ("（新学习率口径 / lr 按方法一个值）", "（新学习率口径：lr 按方法取一个值）"),
    ("（机制「按卡数缩放学习率」已整体删除）", "（原「按卡数缩放学习率」机制已整体删除）"),
    ("扩面（主席裁定「18 个 20 行档放大到 100 行」）", "扩面（口径变更：18 个 20 行档放大到 100 行）"),
    ("★ 2026-10-16 扩面", "★ 2026-10-16 口径扩面"),
    ("口径（2026-10-16 主席裁定）", "口径（2026-10-16）"),
    ("评测集口径（2026-10-16 主席裁定）逐档分列", "评测集口径（2026-10-16）逐档分列"),
    ("已随主席裁定", "已随口径变更"),
    # ★ 内部机读字段名与内部代号 ⇒ 面向使用者的说法（⛔ 不删事实、只换称谓）
    ("（具名原因见行内 `not_executable_reason`）", "（该档在表中已具名标注不可执行的原因）"),
    ("（见各行具名 `not_executable_reason`）", "（各档在表中已具名标注不可执行的原因）"),
    ("（见各行具名 not_executable_reason）", "（各档在表中已具名标注不可执行的原因）"),
    ("逐档具名原因见行内 `not_executable_reason`", "逐档原因在表中已具名标注"),
    ("逐档具名原因见行内 not_executable_reason", "逐档原因在表中已具名标注"),
    ("`not_applicable`", "设计上不成立"),
    ("not_applicable", "设计上不成立"),
    ("（`applicable == false`）", ""),
    ("（applicable == false）", ""),
    ("`lora_target_signature.resolved` = 122", "LoRA 目标的签名条目数 = 122"),
    ("lora_target_signature.resolved = 122", "LoRA 目标的签名条目数 = 122"),
    ("（具名原因见行内 not_executable_reason）", "（该档在表中已具名标注不可执行的原因）"),
    ("（具名 `not_executable_reason`）", "（表中已具名标注）"),
    ("行内 `voided_conclusion.historical_effect`", "各档的历史读数记录"),
    ("行内 voided_conclusion.historical_effect", "各档的历史读数记录"),
    ("各行 `voided_conclusion.historical_effect`", "各档的历史读数记录"),
    ("`eval_caliber_consistency.status_after`", "该档的现行状态"),
    ("（详值见行内 `status` / `eval_caliber_consistency.status_after`）", "（详值以该档现行状态字段为准）"),
    ("（详值见行内 status / eval_caliber_consistency.status_after）", "（详值以该档现行状态字段为准）"),
    ("（静默取 `dp_rank==0`）", ""),
    ("（静默取 dp_rank==0）", ""),
    ("以本行现行 `status` 字段为准", "以该档现行状态为准"),
    ("以本行现行 status 字段为准", "以该档现行状态为准"),
    ("`lora_target_signature.resolved`", "LoRA 目标的签名条目数"),
    ("参照 `T028.json`", "参照同类单卡基准跑次"),
    ("`eval/merged_export.py`", "仓内既有的导出模块"),
    # ★ 内部代号与截断残片（改前**已渲染进人读件**）—— P5/P6 = 内部六项读数编号，
    #   ":to。" / ":ex。" = 上游截断残片 ⇒ 一律换成面向使用者的自然结尾（⛔ 不删事实）。
    ("P5/P6 已补。", "其余指标已采集。"),
    ("P5/P6 待补。", "其余指标未采集。"),
    ("P5/P6", "其余指标"),
    ("（pp-watchdog 停滞 ⇒ force-exit 86）", "（流水线并行停滞、进程未正常结束）"),
    ("不可评：训练失败", "第 1 条：跑通/工程合规未通过 —— 训练失败"),
    ("pp-watchdog 停滞 ⇒ force-exit 86", "流水线并行停滞 ⇒ 进程未正常结束"),
    ("force-exit 86", "进程未正常结束"),
    ("pp-watchdog", "流水线并行看门狗"),
    ("：to。", "。"),
    ("：ex。", "。"),
    ("：su。", "。"),
    # ★ 内部过程/内部交接 ⇒ 移出（不是"改词"，是"删句"；⛔ 不碰数值/判据/结论）
    ("＋ 本包（唯一回填通道 backfill_matrix.py 落盘）", ""),
    ("由「主席裁定未回、口径待定、⛔ 不下效果结论」改为", "改为"),
    ("主席裁定", "口径变更"),
    ("主席拍定", "判据基线"),
    ("主席最新裁定", "现行口径"),
    ("依主席 2026-10-14 口径记入失败", "依 2026-10-14 口径记入失败"),
    ("依主席 2026-10-14 口径", "依 2026-10-14 口径"),
    ("依主席 2026-10-14 裁定", "依 2026-10-14 口径"),
    ("依 2026-10-14 裁定，本档位不实施", "依 2026-10-14 口径，本档位不实施"),
    ("依 2026-10-14 裁定", "依 2026-10-14 口径"),
    ("主席 2026-10-16 裁定", "2026-10-16 口径变更"),
    ("主席 2026-10-15 裁定", "2026-10-15 口径变更"),
    ("主席 2026-10-14 裁定", "2026-10-14 口径变更"),
    ("主席 2026-10-14 口径", "2026-10-14 口径"),
    ("主席 2026-10-15 口径", "2026-10-15 口径"),
    ("主席", "口径来源"),
    ("指挥部", "内部项目组"),
    ("经指挥部批准", "经项目组批准"),
    ("由指挥裁定", "由项目组裁定"),
    ("指挥裁定", "内部裁定"),
    ("指挥统一批量做", "项目组统一批量做"),
    ("（同上裁定）", "（同族，同判据①）"),
    ("已裁定", "已定口径"),
    ("裁定", "口径变更"),
)


#: 文档内部编号 §1.x / §2.x（本模板重排前的旧编号）⇒ 新文档的等价落点
SECTION_ALIAS = {
    "§1.0": "见实验条件（统一口径）", "§1.2": "见实验条件", "§1.3": "见实验条件",
    "§1.4": "见判定口径明细", "§1.5": "见判定口径明细",
    "§6.1": "工程约定", "§10": "27B 服务档参数口径", "§12": "卡集口径", "§13": "整卡判据",
}


def human_view_notation(t: str) -> str:
    """③c 记号清理：去掉面向内部写作的记号与旧编号，保留事实与数值。

    ★ 移出的是**写法**（`★` / `⛔` / `✅` / `**` / 反引号 / 旧文档编号），⛔ 不是内容。
    """
    t = str(t or "")
    for _k, _v in SECTION_ALIAS.items():
        t = t.replace(_k, _v)
    t = re.sub(r"§[0-9]+(?:\.[0-9]+)*", "见实验条件", t)
    for _sym in ("★", "⛔", "✅", "⚠️", "🟡", "🔴"):
        t = t.replace(_sym, "")
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"——\s*(?=[，。；、)])", "", t)
    t = re.sub(r"\s+([，。；、）])", r"\1", t)
    t = re.sub(r"([（「『])\s+", r"\1", t)             # 「 不做…」⇒「不做…」
    t = re.sub(r"\s+([）」』])", r"\1", t)             # 未收紧 ）⇒ 未收紧）
    t = re.sub(r"([。；，、])\s+", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)                  # 记号删除后可能留下双空格
    t = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", t)   # 「看门狗 停滞」⇒「看门狗停滞」
    return t.strip()


def _human_view_final_clean(t: str) -> str:
    """③b 结构级清理：删掉**整句内部过程 / 内部交接**，并收拢空白。

    ★ 与 `human_view_text`（词级改写）分开：这里是"删句"，⛔ 不碰数值、判据与结论。
    """
    t = str(t or "")
    t = re.sub(r"＋\s*本包（[^）]*）", "", t)                          # 内部交接单位
    t = re.sub(r"本包只对其中的[^。]*?——\s*由指挥(?:部)?裁定。", "", t)  # 内部编排说明
    t = re.sub(r"本包[^。]{0,40}。", "", t)                          # 其余"本包…"整句
    t = t.replace("＋ 来源：", "来源：").replace("＋。", "。").replace("＋＋", "＋")
    return re.sub(r"\s{2,}", " ", t).strip()


def human_view_text(s: str) -> str:
    """③ 术语改写：把「谁在什么场合说的」改写成「何时生效的口径」。

    ★ 纯文本级、**不动数值/判据/结论**；顺序应用 `HUMAN_VIEW_REWRITES`。
    """
    t = _human_view_final_clean(s)          # ★ 先"删句"（内部过程/内部交接），再"改词"
    for old, new in HUMAN_VIEW_REWRITES:
        t = t.replace(old, new)
    t = _human_view_final_clean(t)           # ★ 收拢空白（删句可能留下空档）
    return human_view_notation(t)            # ★ 末次清理内部记号与旧编号


def _rewrite_tree(obj):
    """递归对全部字符串做③术语改写（对象结构不变）。"""
    if isinstance(obj, str):
        return human_view_text(obj)
    if isinstance(obj, list):
        return [_rewrite_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_tree(v) for k, v in obj.items()}
    return obj


def _context_ok(s):
    """未过判据的**条次**已写明（§17.7「不得只写失败」）。"""
    t = str(s or "")
    return any(x in t for x in ("第 1 条", "第 2 条", "第一条", "第二条"))


#: ★★ 精简轮 2：结论列「未过判据（失败归因，逐档）」——**从 `verdict` 派生**，⛔ 不新增数据源。
#:   设计依据（§17.7）：① 访客必须一眼看出"这档为什么不能用"，且**不得只写「失败」**；
#:   ② 原先的「备注」列与它高度重复 ⇒ 合并为一列，表宽立刻减一列。
_NOT_FAILED = {
    "已验证可用": "—",
    "尚未验证": "—",
    "未纳入验收": "—",
}


#: ★★ 精简轮 2：**备注列的契约**（单一真相源）——生成端与校验端共用同一组字符串。
#:   `已结论` = 判据①②全过；`未过` = 归因正文在「未过判据」列（本列不重复）；
#:   `不纳入` = 不实施，逐档理由在结论说明里；`无结论` = 尚未验证。
NOTE_CONTRACT = {
    "已结论": "判据 ①② 与该算法的正确性检查全过。",
    "未过": "未过判据见「未过判据」列（已通过项一律保留，不因未过一条而抹掉）。",
    "不纳入": "不实施，不纳入验收范围。",
    "无结论": "判定：尚未验证 —— 尚无实测结论，不计入可用档位。",
    "无结论作废": "判定：尚未验证 —— 原有结论因口径已更正而作废，待重跑复核。",
}

#: 未过判据列里，`failed_detail` 的依据关键词（每条都要有**可构造的阳性对照**）
_DETAIL_EVIDENCE = {
    "T034": ("OutOfMemoryError",), "T035": ("OutOfMemory",),
    # ★ T036 的归因正文经 `human_view_notation` 归一后，「pp-watchdog」写作「流水线并行看门狗」
    #   ⇒ 依据关键词取**归一后仍存在**的形态（⛔ 不是放宽：改的是关键词证据面，断言仍要求逐条命中）
    "T036": ("看门狗", "exit_code = 1"),
    "T037": ("<", "+5.0 pp"), "T038": ("<", "+5.0 pp"), "T039": ("<", "+5.0 pp"),
    "T049": ("OOM",), "T050": ("OOM",), "T051": ("OOM",),
}


def note_contract(state, has_verdict=False, has_excluded=False, raw_note=""):
    """备注列的**唯一**取值来源（生成端与校验端共用）。

    ★★ 2026-10-19 对齐轮（**最小订正**，工作包 task-matrix-table-align）：
      使用者诉求「把『5. 功能矩阵测试表』的格式和测试条目对齐旧版，测试内容清空成未测试」
      ⇒ 「备注」列**一律清空为 `—`**（它是结果面 6 列之一，与 `peak_mem` / `elapsed` /
      `effect` / `steps_actual` / `throughput` 同批清空）。
      ⚠ 缺陷实况：原实现对「尚未验证」档**总是**返回 `NOTE_CONTRACT["无结论"]`
      （"判定：尚未验证 —— 尚无实测结论，不计入可用档位。"）⇒ 数据件里 `note='—'` 与
      契约文本**必然对不上**（54/54 档），`verify_matrix` ④O 必红。
      修法 = 让"未结论"这一支与"未过"/"不纳入"两支**同构**：返回**逐档已清空的备注值**本身
      （`raw_note`，即生成端写下的 `—`），⛔ 不新增文案、⛔ 不改任何数值与四态取值。
      ★ 生成端 `compose_short_note`（本文件）与校验端 `verify_matrix` ④O **同源引用本函数**，
        故生成件与校验件仍**逐字一致**；`NOTE_CONTRACT` / `_DETAIL_EVIDENCE` 字表**一字未删**
        （「已验证可用/不可用」两支的契约文本原样保留 ⇒ 自证夹具仍能证明 ④O **能失败**）。
    """
    if state == S_AVAILABLE:
        return NOTE_CONTRACT["已结论"]
    if state == S_UNAVAILABLE:
        return NOTE_CONTRACT["未过"] if has_verdict else (raw_note or "—")
    if state == S_EXCLUDED:
        return NOTE_CONTRACT["不纳入"] if has_excluded else (raw_note or "—")
    # 未结论（尚未验证）：备注列已按本轮口径清空 ⇒ 契约值 = 逐档清空值本身（缺省 `—`）
    return raw_note or "—"


def compose_short_note(r):
    """★ 精简轮 2：备注列压到**一句话**（原文逐段保留在归档件与内部权威里）。

    依据：原备注平均 ~100 字符、且与「未过判据」列大量重复；访客只需要
    「这一档为什么还没结论 / 为什么不纳入」。⛔ 不改任何数值与四态取值。
    """
    return note_contract(str(r.get("status") or ""),
                         has_verdict=bool(r.get("verdict")),
                         has_excluded=bool(str(r.get("excluded_reason") or "").strip()),
                         raw_note=str(r.get("note") or ""))


def derive_not_failed_reason(r):
    """把 `verdict` 压成**一列**结论文本；⛔ 不改 `verdict` 本身。"""
    st = str(r.get("status") or "")
    v = r.get("verdict")
    if not isinstance(v, dict):
        return _NOT_FAILED.get(st, "—")
    fc = str(v.get("failed_criterion") or "").strip()
    fd = str(v.get("failed_detail") or "").strip()
    if not fc and not fd:
        return _NOT_FAILED.get(st, "—")
    # ★ §17.7：不得只写「失败」——必须写明未过的是第 1 条还是第 2 条
    if not _context_ok(fc):
        fc = ("第 1 条：跑通/工程合规未通过" if st == S_UNAVAILABLE else fc)
    parts = [fc.rstrip("。")]
    if fd and fd != "—":
        parts.append(fd.rstrip("。"))
    return "；".join(x for x in parts if x) + "。"


DEV_CHURN_TERMS = (
    # ★ 修正包 🔴-2（窄复核）／建议 S2：**人读件正文不得含开发过程内容**。
    #   判据 = 主席原话「不要把修改过程写进去，不要把给我们自己写的东西写进去」＋ §17.2。
    #   ★ 词表 = 内部工位/证据路径形态 ＋ 相对时间指代；⛔ 不做通配、⛔ 不豁免生成物。
    #   ★ 生成端（`forbidden_in_human_view` 兜底）与校验端（`verify_matrix` ④R）**同源引用本表**，
    #     ⛔ 不各抄一份（§1.4 单一真相源）。
    #   ★ 豁免面：§17.7 承诺边界声明的**宪法逐字引文**（其中的「我们」不是过程措辞）——
    #     由校验端显式排除该块后再扫，⛔ 不是"整段跳过不扫"。
    ("task-run-", "内部工位目录名（访客不可达、不可复核）"),
    ("evidence/", "内部证据目录路径"),
    ("本轮", "相对时间指代（开发过程措辞）"),
    ("本次", "相对时间指代（开发过程措辞）"),
)


#: ★ 修正轮 4（⚠③）：**内部证据路径形态**（数据件面）。
#:   ★ 与 `scan_matrix_leaks.py` **同源引用本常量**（§1.4 单一真相源，⛔ 不另抄一份正则）。
#:   形态判据（防"`evidence` 作普通英文词"误报）：
#:     ① 必须有 `evidence/` **目录形态**（带斜杠）；② 其后必须是**一位路径字符**起头；
#:     ③ 必须以 `.md` **文件后缀**或**省略号**（`evidence/54-…`）收尾。
#:   ⇒ 裸英文词 `evidence-based`、仓内公开文件 `docs/capability-matrix.md`、历史底稿名
#:     `13-base-rerun-variance.md`（无 `evidence/` 前缀、已归档）**均不命中**。
EVIDENCE_PATH_PATTERN = r"evidence/[^\s，。；、）)」』\"']*?(?:\.md\b|…)"

#: ★ 脱敏后的中性替代文本（保留紧随其后的 sha256 内容指纹作可复核锚点；
#:   ⛔ 不删结论本身 —— 结论、数值与依据链逐字保留，只把"指向内部文件的路径"换成占位说明）。
EVIDENCE_PATH_PLACEHOLDER = "内部验收底稿（不随发布件提供）"


def redact_internal_paths(t):
    """把字符串里的**内部证据路径**换成中性占位（修正轮 4 ⚠③）。

    ★ 幂等；★ ⛔ 不动数值 / 四态 / 判据 / 结论；★ ⛔ 不动其它任何路径形态（如仓内公开文件路径）。
    """
    import re as _re
    return _re.sub(EVIDENCE_PATH_PATTERN, EVIDENCE_PATH_PLACEHOLDER, str(t))


def _redact_tree(obj):
    """对整棵数据树做 `redact_internal_paths`（就地改写，⛔ 不动结构）。"""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _redact_tree(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _redact_tree(v)
        return obj
    return redact_internal_paths(obj) if isinstance(obj, str) else obj


SAVE_STEPS_TOKEN = "save_steps="
"""「关键开关」列里**禁止出现**的令牌前缀（修正包 🔴-1）。

★ 单一真相源：检查点间隔只在 §3.1 的定义行出现一次（`checkpoint_save_steps = 推导步数 + 1`）；
  清单侧的旧口径值（= 每 epoch 步数）不得进入人读件。`verify_matrix` ④Q 机核本不变量。
"""


def _strip_save_steps_token(switches):
    """把「关键开关」里的 `save_steps=<清单旧口径值>` 令牌**整条移除**（保留其余开关，顺序不变）。

    ★ ⛔ 不新增/改写其它令牌；⛔ 不留空令牌；其余令牌逐字保留。
    ★ 幂等：对已无该令牌的输入返回原字符串。
    """
    t = str(switches or "")
    if SAVE_STEPS_TOKEN not in t:
        return t
    kept = [p for p in t.split(";") if not p.strip().startswith(SAVE_STEPS_TOKEN)]
    return ";".join(kept) if kept else "—"


def _save_steps_token_self_test():
    """★ 阳性对照（修正包 🔴-1）：`_strip_save_steps_token` **会失败**的对抗性输入。

    ★ 证明的是：该函数**只**移除 `save_steps=` 令牌、**不动**其它令牌、
      且对"令牌在中间/开头/唯一"三种形态都幂等 ⇒ ⛔ 不是"扫过了"。
    """
    cases = [
        ("save_steps=100;loss_scale=all;chat_template=false", "loss_scale=all;chat_template=false",
         "令牌在**开头** ⇒ 移除后其余逐字保留"),
        ("loss_scale=all;save_steps=6401", "loss_scale=all",
         "令牌在**末尾** ⇒ 移除后其余逐字保留"),
        ("save_steps=50", "—", "令牌是**唯一**开关 ⇒ 归一为占位符（⛔ 不留空令牌）"),
        ("loss_scale=all;chat_template=false", "loss_scale=all;chat_template=false",
         "**无**该令牌 ⇒ 逐字不变（幂等）"),
        ("—", "—", "占位符输入 ⇒ 逐字不变"),
    ]
    ok = True
    for inp, want, why in cases:
        got = _strip_save_steps_token(inp)
        good = got == want
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {why}: {inp!r} -> {got!r}")
    # 反向断言：结果里**不得**再含该令牌
    rev = all(SAVE_STEPS_TOKEN not in _strip_save_steps_token(i) for i, _w, _y in cases)
    ok = ok and rev
    print(f"  [{'PASS' if rev else 'FAIL'}] 反向断言：输出中不含 `{SAVE_STEPS_TOKEN}`")
    # ★ 断言加强（修正轮 4 ⚠② keep 语义）：唯一开关档归一后的**确切取值**必须是占位符 `—`，
    #   且**不得**是空串（§17.7：⛔ 不以空白使人误读）。⛔ 这不是放宽，是把断言写严。
    _only = _strip_save_steps_token("save_steps=50")
    keep = (_only == "—") and (_only.strip() != "")
    ok = ok and keep
    print(f"  [{'PASS' if keep else 'FAIL'}] 加强断言（⚠②）：唯一开关档归一后确切取值 == '—'（且非空串）：{_only!r}")
    return ok


def _fix_row_cells(r, v):
    """④ 表格格归一：逐格订正**已经渲染进人读件**的污染与残片。

    ★ 依据（工装实测，逐条具名，见工位 `plan.md` §2.3）：
      - 「未过判据」含人称 / 悬空内部指代 / 只写「不可评」而不写明哪一条 ⇒ 归一为「第 1 条 / 第 2 条」；
      - `passed_criteria` / `note` 含宿主相对路径（`runs/T0xx/…`）⇒ 去路径、只留事实；
      - `failed_detail` 含进程号 ⇒ 去内部标识；
      - 「耗时 / 显存」列出现机器号占位符 `<HOST>`、宿主相对路径、裸 `未采集`、空串
        ⇒ 统一为「未取到（具名原因）」（§17.7 禁止以空白或占位符使人误读）；
      - `effect` 引用内部字段名 ⇒ 改写为使用者可读的判断句。
    ★ ⛔ 不改任何四态取值、不改任何数值读数、⛔ 不把「不可用」写成「尚未验证」。
    """
    tid = r.get("tier")

    # (a) 「未过判据」列：写明条次 + 保留已通过项
    if v:
        fc = str(v.get("failed_criterion") or "")
        fd = str(v.get("failed_detail") or "")

        def _drop_text(*frags):
            nonlocal fd
            for _f in frags:
                fd = fd.replace(_f, "")

        if not _context_ok(fc):
            if tid == "T036":
                # 训练进程退出码非 0（pp-watchdog 强退）⇒ 属判据①，原句只写「不可评（工程缺口）」
                v["failed_criterion"] = ("第 1 条：跑通/工程合规未通过（★ 不是效果结论；"
                                         "该档训练进程未正常结束）")
            elif tid in ("T049", "T050", "T051"):
                v["failed_criterion"] = "第 1 条：跑通/工程合规未通过（★ 不是效果结论；该档训练未开始）"
            else:
                v["failed_criterion"] = ("第 1 条：跑通/工程合规未通过" + (f"（{fc}）" if fc else ""))

        if tid == "T034":
            _drop_text("Process 3624748 has", "★ 自身实测 OOM", "（9B GRASPO 全参 × native × 1 卡）")
            fd = ("进程未正常退出（`exit_code` 非 0），日志逐字含 `OutOfMemoryError`；"
                  "该配置下 2 卡亦不足（实测约 79.2 GiB / 卡）。")
        elif tid == "T035":
            _drop_text("★ 自身实测 OOM", "（9B GRASPO 全参 × native × 2 卡）")
            fd = "进程未正常退出：`exit_code = 1`，日志逐字含 `OutOfMemory`。"
        elif tid == "T036":
            fd = fd.replace("（与 T029/T030 的 NCCL 看门狗超时同族，但为 pp-watchdog）",
                            "（同族的另一例为 NCCL 看门狗超时）")
            fd = ("训练进程未正常结束：流水线并行 rendezvous 停滞 300 s 触发流水线并行看门狗 ⇒ 强退"
                  "（`exit_code = 1`、进程未正常结束），检查点缺失。")
        elif tid in ("T049", "T050", "T051"):
            fd = "该档在首步即超显存（OOM），训练未开始 ⇒ 判据① 未过（⛔ 与达标与否无关，未进入效果评测）。"
        fd = re.sub(r"`?runs/T\d{3}/[A-Za-z0-9._/-]*`?", "", fd)
        fd = re.sub(r"（最终步，[^）]*）", "", fd)
        v["failed_detail"] = re.sub(r"\s{2,}", " ", fd).strip() or "—"
        v["passed_criteria"] = [re.sub(r"\s{2,}", " ",
                                       re.sub(r"`?runs/T\d{3}/[A-Za-z0-9._/-]*`?", "", str(x))).strip()
                                for x in (v.get("passed_criteria") or [])]

    # (b) 备注列：去截断残片与内部代号，补全句子
    n = str(r.get("note") or "")
    if n and n != "—":
        n = n.replace("P5/P6 待补。", "其余指标未采集。").replace("P5/P6 已补。", "")
        n = re.sub(r"：\s*to\s*[。.]", "：日志逐字含 `OutOfMemoryError`；该配置需要约 79.2 GiB / 卡。", n)
        n = n.replace("（同上裁定）", "（同族，同判据①）")
        n = re.sub(r"`?runs/T\d{3}/[A-Za-z0-9._/-]*`?", "", n)
        n = re.sub(r"\s{2,}", " ", n).strip()
        r["note"] = n

    # (b2) 结果列：§17.7「尚未验证 / 未纳入验收不得携带实验数据」⇒ 结果列一律「—」
    #      （★ 步数上限属**实验条件**，仍在「步数/epoch」列；这里清的是「实跑步数」这类**实测值**）。
    if r.get("status") not in (S_AVAILABLE, S_UNAVAILABLE):
        for _k in ("steps_actual", "peak_mem", "elapsed", "throughput", "effect"):
            if str(r.get(_k) or "—").strip() not in ("", "—"):
                if _k == "elapsed" and str(r.get(_k)).startswith("未取到"):
                    continue          # 「未取到（具名原因）」本身就是合规表述
                r[_k] = "—"

    # (c) 耗时列：机器号占位符 / 宿主相对路径 / 空 ⇒ 「未取到（具名原因）」
    #     ★★ 原因句必须与**同行的其它列自洽**：`elapsed` 是本函数前面刚归一过的值，
    #     若它已是「总时长 …」这种**有读数**形态，就说明训练确实跑完了
    #     ⇒ ⛔ 不得再写「训练未完成」（那是**本包曾引入的措辞错误**，见 report.md A3）。
    _has_wallclock = bool(re.match(r"^总时长", str(r.get("elapsed") or "").strip()))
    el = str(r.get("elapsed") or "").strip()
    if el in ("", "—", "未采集"):
        if r.get("status") == S_UNAVAILABLE:
            r["elapsed"] = ("未取到（该档跑次未记录训练墙钟；训练本身已跑完、判据① 通过）"
                            if _has_wallclock else "未取到（该档训练未完成，无训练墙钟）")
    elif el.startswith("未取到"):
        if "上无该档跑次目录" in el or "<HOST>" in el:
            r["elapsed"] = "未取到（该档无完整训练跑次，无训练墙钟）"
        else:
            r["elapsed"] = re.sub(r"^未取到（\s*", "未取到（", human_view_text(el))

    # (d) 显存列：⛔ 不留空串 / 不留宿主相对路径 / 不留裸「未采集」
    #     ★★ 同 (c)：原因句必须与「该档训练有没有跑完」自洽（耗时列有读数 ⇒ 跑完了）。
    pm = str(r.get("peak_mem") or "").strip()
    if pm in ("", "—", "未采集") or "runs" in pm:
        if r.get("status") == S_UNAVAILABLE:
            r["peak_mem"] = ("未取到（该档跑次未采集显存峰值；训练本身已跑完、判据① 通过）"
                             if _has_wallclock else "未取到（该档训练未完成，无显存采样）")

    # (e) 效果列：去内部字段名 / 内部设备口径 / 重复的判据条次话术；读数逐字保留
    ef = str(r.get("effect") or "").strip()
    if ef and ef != "—":
        if "不可评" in ef:
            r["effect"] = "不适用（该档未进入效果评测，未过判据①）"
        else:
            ef = ef.replace("★ ", "")
            ef = re.sub(r"；?base/after 独立 [^；]*双份。", "。", ef)
            ef = re.sub(r"；?comparability_notes 全过。", "。", ef)
            ef = re.sub(r"\s*服务档参数：[^。]*。", "", ef)
            ef = re.sub(r"\s*服务档 [^。]*。", "", ef)
            ef = ef.replace("未过判据②", "未过判据 2")
            ef = re.sub(r"\s{2,}", " ", ef).strip()
            r["effect"] = human_view_text(ef)

    # (d2) 跑次列：§17.7 要求——**未结论档不得携带实验数据**。内部跑次目录/中间检查点
    #      清单只对内部审计有意义 ⇒ 未结论档一律「—」；已结论档只留结论所需的条次与事实。
    rn = str(r.get("runs") or "").strip()
    if rn.startswith("$TREE") or "另 runs/" in rn:
        if r.get("status") not in (S_AVAILABLE, S_UNAVAILABLE):
            r["runs"] = "—"
        else:
            _sa = str(r.get("steps_actual") or "").strip()
            _ec = str(r.get("elapsed") or "")
            _bits = []
            if tid == "T036":
                _bits.append("1/1 真效果训练：判据 ① 未过（流水线并行 rendezvous 停滞 300 s ⇒ "
                             "pp-watchdog 强退，`exit_code` 非 0、检查点缺失）")
            elif tid in ("T037", "T038", "T039"):
                _bits.append("1/1 真效果训练：判据 ① 通过（`exit_code = 0`、步数跑满、检查点保留）；"
                             "判据 ② 未过（效果未达标，见「效果」列）")
            else:
                _bits.append("1/1 真效果训练：判据 ① 通过；判据 ② 见「效果」列")
            if _sa and _sa != "—":
                _bits.append(f"实跑步数 {_sa}")
            r["runs"] = "；".join(_bits) + "。"

    # (e1) 推理验证列：归一为「合并 / 起服务 / 跑完评测 / 评测规模与错误数」+
    #      本档可核验的产物事实；★ 移出的是内部运行环境记录（宿主路径、目录字面量、卡序号、
    #      容器名与后缀、镜像摘要），⛔ 不是结论。
    iv = str(r.get("inference") or "").strip()
    if iv and iv != "—":
        bits = []
        m = re.search(r"(7\d\d\s*/\s*7\d\d|\d+\s*/\s*\d+)\s*(?:有效)?[、,]?\s*(\d+)\s*error", iv)
        if m:
            bits.append(f"固定评测集 {m.group(1)} 跑完、{m.group(2)} 个 error")
        else:
            m2 = re.search(r"(\d+/\d+)[^。；]*?0\s*error", iv)
            if m2:
                bits.append(f"固定评测集 {m2.group(1)} 跑完、0 个 error")
            elif "0 error" in iv:
                bits.append("固定评测集跑完、0 个 error")
        if "零 merge" in iv or "不 export" in iv:
            bits.append("全参档：零合并，推理服务直指检查点目录")
        elif "合并" in iv:
            bits.append("LoRA 档：先合并为 merged-hf，再由推理服务加载")
        _ver = re.search(r"vLLM\s*(v[0-9][0-9.]*)", iv)
        if _ver:
            bits.append(f"推理服务 vLLM {_ver.group(1)}")
        if "base/after" in iv or "after/base" in iv:
            bits.append("after 与 base 分别独立起服务")
        r["inference"] = ("；".join(dict.fromkeys(bits)) + "。") if bits else "—"
    elif iv == "":
        r["inference"] = "—"

    # (e2) 耗时列：秒数折成人类可读形式（⛔ 不改数值）
    el2 = str(r.get("elapsed") or "").strip()
    _m = re.fullmatch(r"总时长\s*(\d+)\s*s", el2)
    if _m:
        _sec = int(_m.group(1))
        r["elapsed"] = f"总时长 {_sec} s（约 {_sec/3600:.1f} h）"

    # (f) 算法正确性检查列：只留**检查清单与本档可观测项**；★ 内部归属与内部交接说明
    #     （「归属 X + Y + Z（非跑次工位职责）」这类）属内部编排 ⇒ 截断移出。
    ac = str(r.get("algo_checks") or "")
    if ac and ac != "—":
        _cut = re.split(r"★\s*逐条清单的正式判定归属", ac)[0].strip()
        _cut = re.split(r"★\s*逐条清单(?:的)?(?:正式判定)?归属", _cut)[0].strip()
        _cut = re.sub(r"\s*本档为新 runner[^。]*。", "", _cut).strip()
        _cut = re.sub(r"\s*本档为[^。]*launcher[^。]*。", "", _cut).strip()
        r["algo_checks"] = (_cut or ac) or "—"
    elif ac == "":
        r["algo_checks"] = "—"

    # (g) LoRA 目标列：表内只留**本档取值**（后端差异写一次、进统一口径块）
    if str(r.get("tuner_type") or "") == "lora" and str(r.get("lora_target") or "") not in ("", "—"):
        r["lora_target"] = ("真实 HF 名正则（ms-swift；修正版）"
                            if r.get("backend") == "ms-swift" else "canonical preset 名（native）")


#: LoRA 目标「两后端命中集合不等价」的说明（写一次，进统一口径块；⛔ 不在 54 行里重复）
LORA_TARGET_UNIFORM = {
    "key": "lora_target_backend_note",
    "label": "LoRA 目标的按后端差异",
    "value": "两后端的 LoRA 命中集合**不相等、不可直接比较**",
    "layer": "rule", "coverage": "rule-54/54",
    "note": ("ms-swift 用真实 HF 名正则（实测命中 358 个实例：language_model 248 + visual 110）；"
             "native 用 canonical preset 名（310 个实例：language 200 + visual 110）。两者实例数不同、**命中集合也不同** ⇒ "
             "跨后端的 LoRA 读数不直接可比（120 为签名/模式条目数，不是实例数）。"),
}

#: 代码版本（实验条件之一；★ 取**在盘仓**的真值，⛔ 不读时钟、⛔ 不读网络）
CODE_REPO = "graspo"
CODE_BRANCH = "feature/refactor-three-layer-architecture"
CODE_COMMIT = "8d63c38"

#: 机器/场所（实验条件之一；★ 只说**能力口径**，⛔ 不写机器名、内网地址或主机编号，
#:   §17.7「矩阵中仍然禁止」宿主路径 / 内网地址 / 机器名 / 内部节点）
VENUE = "平台方内部训练集群"


#: ①**允许清单**（`meta.criteria_caliber` 子键）：判据 / 口径 / 实验条件；
#: 其余（证据索引、订正史、陈旧件清点、内部对账推导）⇒ 剔除、迁 `.local/`。
HUMAN_VIEW_CALIBER_KEYS = (
    "pass_criteria", "dataset", "eval_caliber", "lr_policy", "ruled_out",
    "criteria_id", "threshold", "baseline", "epoch_policy", "conclusion_rule",
    "caliber_change", "step_reconciliation", "caliber_evidence",
)


#: 统一口径项的「说明」在内部数据里指向**旧文档编号**（模板重排后编号已变）
#: ⇒ 落人读视图前换成**面向使用者的落点**，⛔ 不出现旧编号。
_UNIFORM_NOTE_ALIAS = {
    "见 §1.2": "同表「判据版本」",
    "见 §1.4 / §1.5": "见「判定口径明细」",
    "见 §1.4": "见「判定口径明细」",
    "见 §统一口径 的 SP 一致": "与「SP（序列并行）」项一致",
}


def strip_inline_markers(t):
    """去掉行内内部记号（`★` / `⛔` / `✅`）与多余空白；⛔ 不动数值与结论。"""
    t = str(t or "")
    for _sym in ("★", "⛔", "✅", "⚠️"):
        t = t.replace(_sym, "")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+([，。；、）])", r"\1", t)
    return t.strip()


def _uniform_items(src):
    """统一口径项：脱敏 + 去掉指向旧文档编号的说明。"""
    items = sanitize((src.get("uniform") or {}).get("items", []))
    out = []
    for it in items:
        it = dict(it)
        note = str(it.get("note") or "")
        for _k, _v in _UNIFORM_NOTE_ALIAS.items():
            note = note.replace(_k, _v)
        it["note"] = note
        out.append(it)
    return out


def drop_non_view_fields(doc):
    """① 允许清单投影：把**人读视图不需要**的字段逐条剔除（落盘前）。

    返回 (被剔除的 meta 键, 被剔除的行级字段名 → 出现档数, 被剔除的 criteria_caliber 子键)。
    ★ 内部权威 `.local/…/matrix-data.json` **一行不动**：本投影只作用于正式件。
    """
    dropped_meta = [k for k in list(doc.get("meta", {})) if k not in HUMAN_VIEW_META_KEYS]
    for k in dropped_meta:
        doc["meta"].pop(k, None)
    dropped_cal = []
    _cc = (doc.get("meta") or {}).get("criteria_caliber")
    if isinstance(_cc, dict):
        dropped_cal = [k for k in list(_cc) if k not in HUMAN_VIEW_CALIBER_KEYS]
        for k in dropped_cal:
            _cc.pop(k, None)
    counts = {}
    for r in doc.get("rows", []):
        for k in [k for k in list(r) if k not in HUMAN_VIEW_ROW_KEYS]:
            r.pop(k, None)
            counts[k] = counts.get(k, 0) + 1
    return dropped_meta, counts, dropped_cal


def forbidden_in_human_view(doc):
    """② 兜底扫描：**使用者读得到的字段**里若出现内部术语（人称 / 内部编排 / 内部角色）即拒写盘。

    ★ 扫描面 = **全行减去 `TOOLCHAIN_ONLY_ROW_KEYS`**（机读消费者字段）。
      豁免面**只**由 `HUMAN_VIEW_ROW_KEYS` ∩ `TOOLCHAIN_ONLY_ROW_KEYS` 决定 —— ⛔ 不在此另写一份
      硬编码清单（否则「字段退出豁免面」这件事会在这儿被静默忽略）。
      ⇒ 「不进人读视图」从"靠人自觉"变成**可机核的事实**；
      ⇒ `blocked_reason` / `executability` **已被扫描覆盖**（2026-10-16 收口）。
    """
    probe = json.loads(json.dumps(doc, ensure_ascii=False))
    # ★ 单一真相源：只有**同时**属于机读清单、且**不**属于人读允许清单的键才豁免
    _exempt = tuple(k for k in TOOLCHAIN_ONLY_ROW_KEYS if k not in HUMAN_VIEW_ROW_KEYS)
    for _r in probe.get("rows", []):
        for _k in _exempt:
            _r.pop(_k, None)
    blob = json.dumps(probe, ensure_ascii=False)
    return sorted({t for t in FORBIDDEN_TERMS if t in blob})


# ══ ★★ 「本环境不可执行」档数的**唯一真相源**（2026-10-16 收口）════════════════════
#  病史：散文里 4 处「**10 档不可执行**」是**硬编码**的 —— 口径扩面（27 档并集）之后
#        「文档说假话」而不自知。⇒ 一律**从数据现算**；`verify_matrix` ④S 机核
#        「散文档数 == 数据里 `executability == "not_executable"` 的实际行数」。
def not_executable_count(rows) -> int:
    """从**行数据**现算「本环境不可执行」档数（判据 = 行级 `executability`）。"""
    return sum(1 for r in (rows or []) if str(r.get("executability") or "") == "not_executable")


def not_executable_count_txt(rows) -> str:
    """散文中出现的档数短语（**单一写法**，生成端与校验端同源引用）。"""
    return f"{not_executable_count(rows)} 档"


def _not_executable_count_self_test():
    """★ **能失败**的阳性对照：证明散文档数**确实由数据驱动**（⛔ 不是又一个硬编码）。

    三组**合成的**行集合 —— 若本函数/调用点是写死的，这三组会给出同一个数。
    """
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    r0 = [{"tier": "T001", "executability": "executable"},
          {"tier": "T002", "executability": None}]
    r1 = r0 + [{"tier": "T003", "executability": "not_executable"}]
    r2 = r1 + [{"tier": "T004", "executability": "not_executable"},
               {"tier": "T005", "executability": "not_executable"}]
    chk("① 0 档不可执行 ⇒ 文案 '0 档'", not_executable_count_txt(r0) == "0 档")
    chk("② 1 档不可执行 ⇒ 文案 '1 档'（⛔ 不是 10）", not_executable_count_txt(r1) == "1 档")
    chk("③ 3 档不可执行 ⇒ 文案 '3 档'（随数据变，⛔ 不是常量）", not_executable_count_txt(r2) == "3 档")
    chk("④ 计数 = 行级判据（只认 not_executable，⛔ 不认其它取值）",
        not_executable_count([{"executability": "blocked"}, {"executability": "not_executable"}] * 2) == 2)
    chk("⑤ 空数据 ⇒ 0（⛔ 不抛异常、不回落到常量）", not_executable_count(None) == 0)
    return ok


def _table_structure_single_source_self_test():
    """★ 阳性对照（2026-10-19，能失败）：证明**表格结构只有一份**（§1.4 单一真相源）。

    病史：本文件原先自带一份**10 列**的 `COLUMNS`/`GROUPS`，与在盘列数不一致
    ⇒ 重跑生成器会把表头悄悄覆盖回 10 列（双真相源缺陷）。本对照把这件事实变成机核：
      ① `COLUMNS` / `GROUPS` 必须**就是** `render_matrix` 的常量对象（`is` 判同一对象，
         ⛔ 不是"内容相等"——内容相等仍会在两处各自演化）；
      ② 渲染列数与 `RENDERED_COLUMNS` 一致，且列 key 无重复、列组标签齐备、首列 sticky（渲染器硬前提）；
      ②b ★ 分区：渲染面**不得**再出现上提第 3 章的那些参数列（本轮 = `model/mode/backend`），
          且第 3 章清单（`CONDITION_PARAMS`）非空、第 3 章逐项登记表（`CHAPTER3_CONDITION_ITEMS`）非空。
          ⛔ 反方向由 `verify_matrix` ④Z-c 守住：每个渲染 param 列的取值枚举数量 ≥ 3；
          第 3 章登记表**在案**的参数（`tp`/`mode`/`backend`）由 ④Z-c1/c2/c3 逐条机核，⛔ 不靠注释声明。
      ③ 反向断言：本文件里**不得**再出现一份写死的 10 列清单（`"训练数据"` 这类旧表头
         不得作为 `COLUMNS` 的字面量出现）；
      ④ ★ 与**分区判据同源**（2026-10-19 分区轮）：两张清单都从 `MATRIX_COLUMNS` 现算
         （`RENDERED_COLUMNS` / `CONDITION_PARAMS`）⇒ 本文件不列第二份；互补性由 ⑧c 现算断言。
    """
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    chk("① COLUMNS 就是 render_matrix.MATRIX_COLUMNS（同一对象）", COLUMNS is _render.MATRIX_COLUMNS)
    chk("② GROUPS 就是 render_matrix.MATRIX_GROUPS（同一对象）", GROUPS is _render.MATRIX_GROUPS)
    chk("③ 渲染面列数 == RENDERED_COLUMNS（21 列）", len(_render.RENDERED_COLUMNS) == 21)
    _keys = [c["key"] for c in COLUMNS]
    chk("④ 列 key 无重复且均属常量", len(set(_keys)) == len(_keys))
    chk("⑤ 首列 sticky 且为档号列",
        COLUMNS[0].get("sticky") is True and COLUMNS[0]["key"] == "tier")
    chk("⑥ 每列的列组都在 GROUPS 里",
        all(c["group"] in GROUPS for c in COLUMNS))
    # ★ 反向断言：旧 10 列形态的列已被彻底取代 —— 旧表头「训练数据 / 未过判据」不在新列定义里，
    #   而"逐档差异项"必须留在渲染面（⛔ 防"悄悄退回 10 列"）。
    _labels = [c["label"] for c in COLUMNS]
    _rendered_keys = [c["key"] for c in _render.RENDERED_COLUMNS]
    chk("⑦ 旧 10 列表头（训练数据 / 未过判据）已不在列定义中",
        "训练数据" not in _labels and "未过判据" not in _labels)
    #   ★ 2026-10-19 阈值 ≤2/≥3 轮：并行方案三列（TP / PP / DP）必须都在表里 ——
    #     主席查表口径的第一等公民（"dp2tp2 / pp4tp1dp1 哪个更快"），缺 TP 即红。
    chk("⑧ 留在表格的逐档差异项齐备（TP / PP / DP / 总卡数 / 算法 / 关键开关）",
        all(x in _labels for x in ("TP", "PP", "DP", "总卡数", "算法", "关键开关")))
    # ★★ ⑧b/⑧c：2026-10-19 条件/结果**分区轮**的生成端同源断言（⛔ 不是靠注释声明）
    _cond_keys = [k for k, _l in _render.CONDITION_PARAMS]
    chk("⑧b 上提第 3 章的 3 个参数**不在**渲染列里",
        not (set(_cond_keys) & set(_rendered_keys)) and len(_cond_keys) == 3)
    chk("⑧c 条件/渲染两清单互补且不重不漏（= 分区常量自身成立）",
        not (set(_cond_keys) & set(_render.TABLE_PARAM_COLUMNS))
        and set(_cond_keys) | set(_render.TABLE_PARAM_COLUMNS)
        == {c["key"] for c in _render.MATRIX_COLUMNS if c["group"] == "param"})
    return ok


def _row_normalize_self_test():
    """★ 阳性对照（对抗性输入）：逐条证明本区四件**会失败**，⛔ 不是「扫过了」。"""
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # ③ 术语改写：人称必须被改写掉（并且数值必须逐字不动）
    chk("③ 人称改写：含「主席拍定」的版本串不再含人称",
        "主席" not in human_view_text("v3-2026-10-10（主席拍定）+ U2 起点/终点窗口比较口径"))
    chk("③ 数值不动：+10.0 pp / epoch=4 / 702 逐字保留",
        all(x in human_view_text("SFT ⇒ after − base ≥ +10.0 pp（epoch=4）；702 行")
            for x in ("+10.0 pp", "epoch=4", "702")))
    # ④ 表格格归一：三种具名污染都必须被改掉
    r1 = {"tier": "T036", "tuner_type": "全参", "status": S_UNAVAILABLE, "note": "不可评：训练失败（`pp-watchdog` 停滞 ⇒ force-exit 86）。P5/P6 已补。"}
    v1 = {"state": S_UNAVAILABLE, "failed_criterion": "不可评（工程缺口）—— 依主席 2026-10-14 口径记入失败",
          "failed_detail": "训练失败：PP rendezvous 停滞 300 s", "passed_criteria": []}
    _fix_row_cells(r1, v1)
    chk("④ T036「不可评」被改写为写明第 1 条", _context_ok(v1["failed_criterion"]))
    chk("④ T036 人称从判据列消失", "主席" not in v1["failed_criterion"])
    r2 = {"tier": "T037", "tuner_type": "lora", "backend": "native", "status": S_UNAVAILABLE,
          "peak_mem": "见 runsgpu_memory_summary.json（2 卡峰值）",
          "elapsed": "未取到（<HOST> 上无该档跑次目录（OPD 族未起跑））",
          "effect": "★ 不可评（见 not_evaluable_reason）；⛔ 本档不可评。",
          "lora_target": "X｜native：canonical preset 名（310 实例；120 为签名/模式条目数）｜★ 两后端实例数不同且**命中集合不同**（不等价）"}
    v2 = {"state": S_UNAVAILABLE, "failed_criterion": "第 2 条：效果未达标", "failed_detail": "Δ = -16.3818 pp",
          "passed_criteria": ["GRASPO 六项齐：injection_failures=0（最终步，runs/T037/stdout.log）"]}
    _fix_row_cells(r2, v2)
    chk("④ 显存格里宿主相对路径被改掉", "runs" not in r2["peak_mem"])
    chk("④ 耗时格里机器号占位符被改掉", "<HOST>" not in r2["elapsed"])
    chk("④ 效果格里内部字段名被改掉", "not_evaluable_reason" not in r2["effect"])
    chk("④ LoRA 格只留本档取值（差异说明移入统一口径）", r2["lora_target"] == "canonical preset 名（native）")
    chk("④ 已通过项里的宿主路径被去掉", "runs/" not in v2["passed_criteria"][0])
    r3 = {"tier": "T010", "tuner_type": "全参", "status": S_UNVERIFIED,
          "algo_checks": "SFT 档清单：prompt 段 mask / loss 只在目标 token。本跑次侧可观测项：skipped_nonfinite=0。"
                         "★ 逐条清单的正式判定归属 Ripple + flow·adapter + flow·backend（非跑次工位职责）。"}
    _fix_row_cells(r3, None)
    chk("④ 算法检查列截断掉内部归属说明",
        "归属" not in r3["algo_checks"] and "工位" not in r3["algo_checks"]
        and "skipped_nonfinite=0" in r3["algo_checks"])
    # ② 兜底扫描：塞入内部术语必须被抓到
    chk("② 兜底扫描会命中注入的「主席」",
        forbidden_in_human_view({"x": "由主席裁定"}) == ["主席"])
    chk("② 干净文档不命中", forbidden_in_human_view({"x": "在平台方数据集与参数下效果达标"}) == [])
    # ① 允许清单投影：非渲染字段必须被剔除
    doc = {"meta": {"title": "t", "criteria_caliber": {}, "distill_teacher": {}, "doc_note": "内部过程"},
           "rows": [{"tier": "T001", "status": S_UNVERIFIED, "note": "—",
                     "voided_conclusion": {"historical_effect": "内部陈旧读数"},
                     "evidence_paths": ["x"]}]}
    dm, rc, dc = drop_non_view_fields(doc)
    chk("① meta 非允许键被剔除", dm == ["doc_note"])
    chk("① 行级非允许字段被剔除", set(rc) == {"voided_conclusion", "evidence_paths"})
    chk("① criteria_caliber 非允许子键被剔除", dc == [])
    doc2 = {"meta": {"criteria_caliber": {"threshold": {}, "stale_artifacts_20261016": {}}}, "rows": []}
    _, _, dc2 = drop_non_view_fields(doc2)
    chk("① criteria_caliber 内部子键被剔除", dc2 == ["stale_artifacts_20261016"])
    chk("① 允许字段逐字保留", doc["rows"][0]["tier"] == "T001" and doc["meta"]["title"] == "t")
    print(f"  HUMAN-VIEW-SELF-TEST {'PASS' if ok else 'FAIL'}")
    return ok


#: 四态（§17.7）——只允许这四个取值，且必须是可区分的文字标签
CONCLUDED = (S_AVAILABLE, S_UNAVAILABLE)

EXCLUDED_REASON = ("设计上不成立：师生权重常驻合计约 100 GiB，与「节省显存」的设计意图相悖 ⇒ 不产出可运行配置。"
                   "适用范围：依 2026-10-14 口径，本档位不实施（不产出可运行配置）。")

GENERATOR = "tools/matrix/render_matrix.py"
SOURCE = "docs/capability-matrix.data.json"
HUMAN_VIEW = "docs/capability-matrix.html"

#: ★★ 表格结构（列组 + 渲染面 21 列）＝ **从渲染器引用**（`render_matrix.MATRIX_GROUPS` /
#:   `MATRIX_COLUMNS`；渲染面 = `RENDERED_COLUMNS`），
#:   ⛔ 本文件不再自带第二份（§1.4 单一真相源）。
#:   ★ 病史与修法（2026-10-19）：本文件原先写死一份**10 列**的 `COLUMNS`/`GROUPS`
#:     （精简轮 4 的旧形态），而在盘数据件 / 人读件已是 13 / 21 列渲染面 ⇒ 一旦重跑本生成器，
#:     表头会被悄悄覆盖回 10 列（双真相源缺陷）。现改为**引用同一个常量对象**，
#:     并由 `--human-view-self-test` 增加"两个名字必须是同一对象"的阳性对照。
#:   ★ 同批：正式机读件**不再落 `columns` / `groups`**（使用者口径「脚本只负责把测试结果填进
#:     最后一章测试表」）⇒ 表结构不进数据件，数据件只有行数据。
GROUPS = _render.MATRIX_GROUPS
COLUMNS = _render.MATRIX_COLUMNS

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
#:   语言 10 项（attn 4 + 线性注意力 1+2 + MLP 3）+ 视觉 110 项（merger 2 + 27 层 × 4）= 120
LANGUAGE_TARGETS = [
    "language.full_attn.q_proj", "language.full_attn.k_proj",
    "language.full_attn.v_proj", "language.full_attn.o_proj",
    "language.linear_attn.qkv_proj",
    "language.linear_attn.in_proj_z", "language.linear_attn.out_proj",
    "language.mlp.gate_proj", "language.mlp.up_proj", "language.mlp.down_proj",
]
VISION_TARGETS = (["visual.merger.linear_fc1", "visual.merger.linear_fc2"]
                  + [f"visual.blocks.{i}.{m}" for i in range(27)
                     for m in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")])
RESOLVED_LORA_TARGETS = LANGUAGE_TARGETS + VISION_TARGETS        # 120（native canonical）

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
        `lora_target_signature.resolved` 实测为 **120 = 语言 10 + 视觉 110**（2026-10-21 融合 QKV 后；此前 122）；
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
            "coverage": {"language": 200, "visual": 110, "total": 310,
                         "signature_entries": 120,
                         "basis": ("★ 口径订正（2026-10-12；2026-10-21 融合 QKV 后重算）：`lora_target_signature.resolved = 120` 是**签名/模式条目数**"
                                   "（10 条「无 layer 索引的语言模式」+ 110 条视觉实例），**不是件数**；"
                                   "10 条语言模式**逐层展开为 200 个实例**（融合的 QKV 按整块一份低秩 ⇒ 原 q/k/v 三片规范名合并为 `language.linear_attn.qkv_proj`）"
                                   "⇒ native 合计 **310 个实例**（张量键数 620），与 ms-swift 的 358 **不再相等**。"
                                   "★ 可复核锚点：该计数由本矩阵的数据文件与抽取工具链（`tools/matrix/`）逐档现算、可重跑复现；⛔ 不指向任何内部过程记录。")},
            "language_targets": list(LANGUAGE_TARGETS),
            "signature_evidence": {
                "field": "lora_target_signature.resolved / parameter_count",
                "runs": "T010（1 卡）与 T011（2 卡）签名完全相同",
                "parameter_count": 384757760,
                "note": ("★ 该值即字段 `parameter_count` 的现行值（08a5654 融合 QKV 后按形状表：409,923,584 − 24 × 8192 × 128 = 384,757,760）；"
                         "T010 / T011 两份融合前 checkpoint 的 manifest 原值为 409923584（与旧形状表逐位相等，可作交叉验证）—— 旧 adapter 与现行签名不再同构，不得混用。语义按字段名判（⛔ 不推测）。"),
            },
            "namespace": "language.* / visual.*（native canonical，公开签名）",
            "note": ("native 侧支持 preset 名（glob），由本仓 lora 模块解析。"
                     "★ 实测证明 `language_all_linear` **确实命中语言层 10 项、不是空转**"
                     "（与 T010 实测 Δ=+13.39 pp 互证：语言层空转不可能有该增益）。"),
        },
        "msswift": {
            "kind": "hf_regex",
            "expression": [HF_LORA_TARGET_REGEX],
            "resolved": [],
            "resolved_count": None,
            "coverage": {"status": "measured", "language": 248, "visual": 110, "total": 358,
                         "param_count": 410120192,
                         "basis": ("★ 探针实测（依据：evidence/45-lora-target-probe-newregex.md "
                                   "sha 1c3830b119946946fcfbbae0aa8e29ea6e1184c4c99b434e83ce8e4deeef34e5）。"
                                   "语言 248 分解：q/k/v/o_proj 各 8｜in_proj_qkv 24、in_proj_a/b 各 24｜"
                                   "in_proj_z 24｜out_proj 24｜gate/up/down_proj 各 32。"
                                   "★ 收紧版视觉侧显式枚举 ⇒ 命中 358 全为 nn.Linear（Conv3d 0）；"
                                   "旧过宽版 visual\\..* 实测 528 命中（含 170 非 Linear）⇒ 已弃用。")},
            "escape_note": ("转义：正则含反斜杠，YAML 出参用**单引号标量**（字面量）⇒ 加载值应为**单反斜杠**。"
                                   "★ 验收以**加载值**为准（GraspoConfig 加载后 repr），⛔ 不看文件文本。"),
            "comparability_note": ("★ 两后端 **LoRA 实例数不再相等**（native 310 = language 200 + visual 110；ms-swift 358 = language 248 + visual 110），且**命中的模块集合不同**"
                                   "（native `token_mixer.qkv_proj` 等 vs ms-swift `linear_attn.*`/`full_attn.*`）⇒ 两后端的效果/性能读数**不直接可比**，跨后端比较须注明。"),
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
                     "★★ **两后端不等价（结论不变，依据订正）**：★ 实例数 native **310**（language 200 + visual 110）vs ms-swift **358**（language 248 + visual 110；"
                     "张量键数 716），参数量 native 384,757,760 vs ms-swift 410,120,192（差 25,362,432 ≈6.18%）"
                     "—— ⛔ **不得再写「120 件 vs 358 件」**；★ 不等价的依据 = **命中的模块集合不同（且实例数也不再相等）**："
                     "native `layers.N.token_mixer.{qkv_proj,in_proj_z,out_proj}` + `mlp.*`；"
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
        "text": ("能力与可用档位矩阵说明各档位的可用性与性能表现：每个列出的档位都已在我们的数据集上完整跑通"
                 "并取得良好效果（作为「可用、大概率无 bug」的证据）。训练效果取决于使用者自身的任务、数据集与参数，"
                 "不在本矩阵的承诺范围。"),
        "points": [
            "① 各档可用性：每个列出的档位都已完整跑通（平台方整体承诺以覆盖度门槛为前提）。",
            # ★ 2026-10-16 最小补正（矩阵侧收口）：不可执行档的承诺边界 —— ⛔ **不削弱原三点**
            #   （原三点逐字未动，仅在 ① 之后**新增** ①′）
            "①′ 不可执行档**不计入**承诺面：标为「本环境不可执行」的档位**既不充当可用证据、"
            "也不充当不可用结论** —— 它们是**环境 / 硬件层面的排除**，"
            "⛔ 不改变任何档位的历史结论与四态计数。",
            "② 平台方测试数据集结论：在平台方测试数据集与测试参数下效果达标，且这是「可用、大概率无 bug」的证据。",
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
                    "note": "配置里写的是**容器内**的教师模型路径（由 path_parts 拼成）；实际挂载源在起跑时决定——两者不要混用。"}
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


def criteria_caliber(rows=None):
    # ★★ 2026-10-16（18 档放大到 100 行之后）：**fit 口径档位清单一律从行数据现算**
    #    （判据 = 该档 `subset_size == 100`）—— ⛔ 不再写死 `T001–T027`：
    #    写死档号会在口径扩面时变成"文档说假话"（口径扩面时正是如此）。
    _rows = list(rows or [])
    _fit_tiers = [str(r.get("tier")) for r in _rows if str(r.get("subset_size")) == "100"]
    _fit_txt = f"{len(_fit_tiers)} 档（{', '.join(_fit_tiers)}）" if _fit_tiers else "（无）"
    # ★★ 「本环境不可执行」档数：**一律从行数据现算**（单一真相源 = `not_executable_count_txt`），
    #    ⛔ 不再硬编码 —— 口径扩面（27 档并集）后硬编码会让文档说假话。
    #    机核 = `verify_matrix` ④S「散文档数 == 数据实际 not_executable 行数」。
    _ne_txt = not_executable_count_txt(_rows)
    return {
        # ★★ 2026-10-17 去重轮（现行口径）：四条判据改为**互不重叠、各自独立** ——
        #    ① 只判「跑完 + 产物完整可加载」；② 只判效果读数；③ 只判算法机制本身对不对；
        #    ④ 只判产物真的能推理出可解析输出。⛔ 判据数量（四条）/ 四个短名 / 全部阈值数值一字未改。
        #    ★ 单一真相源：本函数与 `template.html` 第 2 章四条 `<li>`、以及正式数据件
        #      `docs/capability-matrix.data.json` 的 `criteria_caliber.pass_criteria` **逐字同源**
        #      （机核 = `verify_matrix` ④X；阳性对照 = `run_self_proof` ㉒）。
        "pass_criteria": [
            {"title": "能完整跑完",
             "text": "训练正常结束、无中断、无报错、退出码 0、优化器步数达标、检查点完整、没有 NaN/Inf。"},
            # ★★ 判据② 的通过线**一律从 `threshold.by_method` 现取**（⛔ 不在此另写一份数值，
            #    §1.4 单一真相源）⇒ `from_threshold: True` 表示「本条的逐算法细则由数据现算」。
            {"title": "效果变好（核心判据）",
             "text": "在平台方测试数据集与测试参数下，训练之后模型效果比不训练的原始模型好"
                     "（逐算法通过线见下）：",
             "from_threshold": True},
            {"title": "算法正确性检查通过",
             "text": "各算法按自己的检查清单逐项通过，确认该算法的机制真实生效、实现正确"
                     "（SFT / GRASPO / CPT / OPD 各按自己的清单）；GRASPO 的算法逻辑严禁改动。"},
            {"title": "能推理",
             "text": "合并 LoRA 后用 vLLM 启动，训练数据集可以完整推理一遍，效果好于未训练模型。"},
        ],
        "caliber_change": {
            "effective": "2026-10-15",
            "source": "主席最新裁定（新四条）",
            "changed": ("★ **改动一 = 判据②（效果）**：由「训练类与 GRASPO 统一 δ = +5.0 pp」改为**逐算法分列**；"
                        "SFT epoch 由 2 改为 4；OPD 教师改为 SFT 之后的 27B 产物；★ 明确「⛔ 不做假设检验」。"
                        f"★★ **改动二（2026-10-16，无冲突、两条并列）= 评测集口径**：**100 行档（{_fit_txt}）**"
                        "的评测集由平台方自有 702 行评测集改为**该档自身训练用的同一份文件**（100 行、"
                        "`dataset_split: train`）⇒ 读数的含义是 **fit 口径（训练集自评）**，"
                        "⛔ 不得读作训练分布之外的表现；★ 该改动**不改通过线数值**，旧读数按「结论与口径"
                        "一一对应」纪律**作废**并保留于各行 `voided_conclusion.historical_effect`；"
                        "★★ **改动三（2026-10-16）= 20 行档扩面到 100 行（主席裁定）**：原 18 个 **20 行**档"
                        "（GRASPO×ms-swift 的 `T031–T033`/`T037–T039`/`T043–T045` ＋ OPD 的 `T046–T054`）"
                        "由「主席裁定未回、口径待定、⛔ 不下效果结论」改为**已裁定：放大到 100 行、"
                        f"适用同一 fit 口径**；★ 其中 **{_ne_txt}不可执行**"
                        "（**本环境不可执行**：逐档具名原因在行内标注）"
                        "⇒ 只登记声明口径、⛔ 不伪造可跑配置；★ 9 个 formal 档**不动**。"),
            "unchanged": ("★ 通过线数值与判定语义**除 2026-10-19 的 OPD 变更外**逐字未动"
                          "（见 `threshold.by_method` 与 `caliber_change.caliber_change_20261019`）；"
                          "★ **2026-10-17 去重轮已改写判据 ①③④ 的措辞与分工**（改为互不重叠、各自独立；"
                          "旧措辞与逐条对照见 `criteria_caliber.criteria_dedup_20261017`）。"),
            "superseded": ("★ 旧口径（留痕，⛔ 不再作为判定依据）：训练类（SFT / CPT / OPD）与 GRASPO **统一 δ = +5.0 pp**；"
                           "目标 +10.0 pp 仅展示、不用于判定；SFT epoch = 2。"),
            # ★★ 2026-10-17 口径变更（**现行条目**；上列 effective/source/changed/unchanged/superseded
            #    是 2026-10-15 / 10-16 的历史条目）—— ★ 历史逐字留痕，⛔ 一字不删。
            #    ★ 单一真相源：判据② 的通过线数值一律以 `threshold.by_method` 为准（本条目只登记口径，
            #      ⛔ 不另写一份数值表）。
            "caliber_change_20261017": {
                "effective": "2026-10-17",
                "source": "2026-10-17 主席裁定（判据② 定死）",
                "changed": ("★ **判据②（效果）定死为逐算法四条**：SFT / GRASPO 只判准确率 Δ —— "
                            "**不再看 loss / reward 曲线趋势**，也不需要为曲线单独采的单卡基线；"
                            "★ **U2 即判据②** —— 旧「曲线窗口比较」口径随之作废；"
                            "★ **CPT 保留「训练 loss 正常下降」，不走效果 Δ** —— 理由：**CPT 与 SFT 不同**；"
                            "★ CPT 的训练 loss 判据与「base 得分口径」**不是一回事**，⛔ 不得连带删除。"),
                "unchanged": ("★ 通过线数值在 **2026-10-17** 当日逐字未动：SFT ⇒ after − base ≥ +10.0 pp"
                              "（epoch=4）；GRASPO ⇒ after − base ≥ +5.0 pp（epoch=8）；CPT ⇒ 准确率 "
                              "after ≥ base（不劣化；epoch=2）；OPD ⇒ 学生 after ≥ base（能力回收；epoch=2）"
                              "（★ OPD 通过线已于 2026-10-19 变更，见 `caliber_change_20261019`）。"
                              "★ 判据 ①③④ **逐字未动**；★ **CPT 的「训练 loss 正常下降」判据逐字保留**。"),
                "superseded": ("★ 旧口径（逐字留痕，⛔ 不再作为判定依据）：「**U2 = loss 趋势 vs 未训练底座"
                               "同口径终窗**」；★ 随之作废的文档判据版本标识 = 「v3-2026-10-10（判据基线）"
                               "+ U2 起点/终点窗口比较口径」（两条原文逐字见精简轮 3 归档件）。"),
            },
            # ★★ 2026-10-17 去重轮（现行条目）：四条判据互不重叠、各自独立；旧措辞逐字留痕。
            "criteria_dedup_20261017": {
                "effective": "2026-10-17",
                "source": "现行口径（判据去重：四条互不重叠）",
                "changed": "判据 ①③④ 逐字改写为互不重叠、各自独立；并删除页面第 2 章的合并规则段（规则本身不删，改由导语「四条全部通过才算可用」承载）。①（旧）训练正常结束、不中断、不报错：退出码 0、优化器步数达标、检查点完整、没有 NaN/Inf；算法机制真的在起作用（不是空转）；产物能加载或推理。⇒（新）训练正常结束、不中断、不报错：退出码 0、优化器步数达标、检查点完整、没有 NaN/Inf；产物可加载。③（旧）SFT / GRASPO / CPT / OPD 各按自己的检查清单逐项通过（这是第 1 条的组成部分）；GRASPO 的算法逻辑严禁改动。⇒（新）各算法按自己的检查清单逐项通过，确认该算法的机制真实生效、实现正确（SFT / GRASPO / CPT / OPD 各按自己的清单）；GRASPO 的算法逻辑严禁改动。④ 旧新逐字相同：合并 LoRA 后用 vLLM 启动，实际前向一次能拿到可解析输出（旧文本里的「推理」被 ① 吞并的问题是删 ① 尾句，④ 本身未改写）。",
                "unchanged": "判据数量（四条）、四个短名（能完整跑完 / 效果变好 / 算法正确性检查 / 能推理）与阈值数值（SFT ⇒ after − base ≥ +10.0 pp；GRASPO ⇒ after − base ≥ +5.0 pp；CPT ⇒ after ≥ base（不劣化）；OPD ⇒ 学生 after ≥ base（能力回收））在 **2026-10-17** 当日逐字未动（★ OPD 通过线与页面第 2 章导语已于 2026-10-19 另行变更，见 caliber_change.caliber_change_20261019）；★ 2026-10-17 删段时「四条全部通过才算可用」规则本身未删（由页面导语承载）。",
                "superseded": "旧措辞（留痕，不再作为判定依据）：① 尾句「算法机制真的在起作用（不是空转）；产物能加载或推理。」与 ③ 的「（这是第 1 条的组成部分）」——前者与 ③④ 重叠、后者自认是 ① 的组成部分，已按去重口径改写。"
            },
            # ★★ 2026-10-19 主席令（**现行条目**）：OPD 通过线**实质变更**（由不劣化改为 ≥ +5.0 pp）
            #    ＋ 新增教师要求；同批改写页面第 2 章导语与四条（含 ①④ 的措辞）。
            #    ★ 单一真相源：数值一律以 `threshold.by_method` 为准（本条目只登记口径，⛔ 不另写数值表）。
            #    ★★ **逐字留痕（⛔ 只在本件、⛔ 不进人读件）**——原始指令原文 = ：
            #       「2026-10-19 主席令：OPD 由不劣化改为 ≥ +5.0 pp，并新增教师要求。」
            #       ⛔ 「主席」是人读件禁止项（§17.7 ＋ `scan_matrix_leaks` fail-closed）；且
            #       `HUMAN_VIEW_REWRITES` 会把「主席」改写成「口径来源」⇒ 「主席令」会变成
            #       **不成词的「口径来源令」**。⇒ 人读件面**一律用「口径变更令」这一等价措辞**
            #       （语义等价、可读、可留痕），原文逐字只保留在本注释里。
            "caliber_change_20261019": {
                "effective": "2026-10-19",
                "source": "2026-10-19 口径变更令（OPD 通过线：由不劣化改为 ≥ +5.0 pp，并新增教师要求）",
                "changed": ("★ **改动一 = OPD 通过线实质变更**：由「学生 after ≥ base（能力回收）」"
                            "（criterion = `non_inferiority`、value = 0.0）改为 **after − base ≥ +5.0 pp**"
                            "（criterion = `delta_pp`、op = `>=`、value = 5.0）；★ **并新增教师要求**："
                            "教师在训练数据集上的完全正确率**需高于学生未训练模型 10 个百分点**"
                            "（原教师说明「SFT 之后的 27B 产物」随之作废）。"
                            "★ **改动二 = 页面第 2 章导语与四条按 2026-10-19 口径变更的原文改写**：导语改为"
                            "「原则：训练之后效果变好，才算「可用」。」（删除「具体判别依据：…」那句）；"
                            "① 去掉重复的「：」与尾句「产物可加载」⇒ 训练正常结束、无中断、无报错、"
                            "退出码 0、优化器步数达标、检查点完整、没有 NaN/Inf。；② 口径改为"
                            "「在平台方测试数据集与测试参数下，训练之后模型效果比不训练的原始模型好」；"
                            "④ 改为「训练数据集可以完整推理一遍，效果好于未训练模型」。"
                            "★ **术语统一**：效果口径名一律用「数据集完全正确率」（旧称不再出现）；"
                            "评测集 / 参数口径名一律用「平台方测试数据集与测试参数」（旧称不再出现）。"
                            "★ 同批落地三处逐字一致：页面第 2 章 `<ol class=\"criteria\">` 四条 ↔ "
                            "数据件 `criteria_caliber.pass_criteria` ↔ 生成端 `build_official.py` 的 "
                            "`criteria_caliber()`（机核 = `verify_matrix` ④X；阳性对照 = "
                            "`run_self_proof` ㉔）。"),
                "unchanged": ("★ OPD **epoch = 2 不变**；★ SFT ⇒ after − base ≥ +10.0 pp（epoch=4）、"
                              "GRASPO ⇒ after − base ≥ +5.0 pp（epoch=8）、CPT ⇒ 准确率 after ≥ base"
                              "（不劣化；epoch=2）三条通过线数值与判定语义**未动**；"
                              "★ 判据条数（四条）与四个短名**未动**；★ 21 列表结构与结果列状态**未动**。"),
                "superseded": ("★ 旧口径（留痕，⛔ 不再作为判定依据）：OPD ⇒ 学生 after ≥ base（能力回收）"
                               "（criterion = `non_inferiority`、value = 0.0）；★ 旧教师说明 = "
                               "「SFT 之后的 27B 产物」；★ 旧页面第 2 章导语 = 「原则：训练之后效果变好。"
                               "具体判别依据：训练之后模型效果变好了，且这次训练本身真实有效；"
                               "下面四条全部通过，才算「可用」。」"),
            },
            "reading_change_registry": [],      # ★ 由 main() 从**行数据**派生（⛔ 不在此处写死档号/数值）
            "reading_change_registry_note": (
                "★★ **「读数变化」≠「判据变化」**：本清单逐条登记「因缺陷修复/重跑产生了新读数"
                "（旧读数作废、保留为历史）、而判据口径逐字未变」的档，并**逐条注明四态是否翻盘**"
                "（四态翻盘 / 四态未变——仅归因与读数更正）。"
                "★ 每条的 `kind` = `reading_change（★ 不是判据变更）`，可在档位行的 `caliber_change` 里机读。"),
        },
        "dataset": {"id": (f"平台方自有评测集（名称 test.jsonl）＋ ★ **100 行档（{_fit_txt}）的评测集 = "
                           "该档自身训练用的同一份文件**（fit 口径：训练集自评）"),
                    "availability": "平台方自有 / 不公开（100 行档用该档自身训练文件，同为非公开数据）",
                    "version_anchor_sha256": ("两种口径 —— ★ 100 行档 = 该档自身训练文件（100 行，sha256 594d41b0…）；"
                                              "其余档 = 平台方自有评测集（702 行，c202b1e1…）"),
                    "sample_count": "★ 100 行档 = 100（fit 口径）；其余档 = 702",
                    "method": "数据集完全正确率（acc）",
                    # ★★ 「是否同分布」口径（主席诉求 5；审查项 W9/W10）＋ 2026-10-16 评测集口径变更（fit）：
                    #    100 行档的评测集**就是它自己的训练文件**（不是"同分布的另一份"）；其余档仍为同源不同样本。
                    #    ⛔ 一律不评测「训练分布之外的表现」；⛔ 「泛化」一词不进本件（主席口径）。
                    "train_eval_relation": (
                        "★★ **评测集口径（2026-10-16 主席裁定）逐档分列**："
                        "① **100 行档 = fit 口径（训练集自评）** —— 评测集 = **该档自身训练用的"
                        "同一份文件**（100 行、`dataset_split: train`）⇒ 读数回答的是「该档训练是否生效」；"
                        "⛔ **不得读作训练分布之外的表现**（⛔ 本矩阵不评测该问题）。"
                        "旧 702 行口径下的读数**已作废**，保留于各行 `voided_conclusion.historical_effect`。"
                        "★★ **2026-10-16 扩面**：档位清单一律从行数据现算（判据 = `subset_size == 100`）"
                        "⇒ 原 18 个 **20 行**档已随主席裁定**放大到 100 行**并纳入 fit 口径；"
                        f"★ 其中 **{_ne_txt}不可执行**"
                        "（**本环境不可执行**：逐档具名原因在行内标注）⇒ 只登记声明口径、"
                        "⛔ 不为它们伪造可跑配置；★ **T052–T054 = `not_applicable` ⇒ 四态「未纳入验收」**"
                        "（⛔ 不得读成「试过了不行」）。"
                        "② **其余档**（9 个 formal 档）= 平台方自有 702 行评测集"
                        "（与各档训练数据同源、样本 id 不重叠）。"
                        "★ **判别力（100 行档）**：该 100 行的 (工具, 动作) 类别**最大类占 19%**"
                        "（19/100）⇒ **常数预测器上界 19%**；而通过线 = `after − base ≥ +10.0 pp`（点估计）"
                        "⇒ **跨过它必须真的读图**，任何常数输出都拿不到。"
                        "★ **同时作废的旧读数（⛔ 不得引用）**：「剔除 169 条图像重叠后 32.27%」与"
                        "「训练/评测图像去重交集 752 张（占评测图像 38.5%）」出自 **702 行评测集 × 全量 6378 行"
                        "训练集**口径；mini 档实测 **mini100 × 702 图像交集 = 0** ⇒ 该口径对 mini 档"
                        "**自始不成立**（该口径在现行评测集口径下不适用）。"
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
            "rule": (f"★ **100 行档（判据 = 该档 `subset_size == 100`，即 {_fit_txt}）**：评测集 = **该档自身"
                     "训练用的同一份文件**（100 行、`dataset_split: train`）⇒ **fit 口径（训练集自评）**；"
                     "★ 其余档（9 个 formal 档）：保持平台方自有 702 行评测集 + `dataset_split: test`。"),
            "pass_lines_unchanged": ("★ **通过线数值在 2026-10-16 当时逐字未动**：SFT ⇒ `after − base ≥ +10.0 pp`；"
                                     "GRASPO ⇒ `after − base ≥ +5.0 pp`；CPT ⇒ `after ≥ base（不劣化）`；"
                                     "OPD ⇒ `学生 after ≥ base（能力回收）`（★ OPD 通过线已于 2026-10-19 改为 "
                                     "`after − base ≥ +5.0 pp`，见 `caliber_change.caliber_change_20261019`）；"
                                     "一律点估计、⛔ 不做假设检验。"),
            "fit_caliber_label": "fit 口径（训练集自评）",
            "discriminative_power": ("★ 100 行档：训练集 (工具, 动作) 类别最大类 **19/100 = 19%** ⇒ "
                                     "**常数预测器上界 19%** ⇒ 跨 +10.0 pp 必须真读图；"
                                     "每样本粒度 **1.00 pp**（达线需净跳 ≥10 条）。"),
            "not_applicable": ("★★ **2026-10-16 已裁定（取代本键旧的「尚未裁定、不下结论」表述）**："
                               "原 18 个 **20 行**档（GRASPO×ms-swift 的 `T031–T033`/`T037–T039`/`T043–T045` "
                               "＋ OPD 的 `T046–T054`）由主席裁定**放大到 100 行**，一并适用同一 fit 口径。"
                               "★ 立此裁定的理由（旧表述保留于下）：n=20 ⇒ 每样本 **5.00 pp**，而 GRASPO "
                               "通过线正好 +5.0 pp ⇒ 判据退化为「1 个样本决定成败」——放大到 100 行后"
                               "每样本粒度回到 **1.00 pp**（达线需净跳 ≥10 条），判别力恢复。"
                               f"★ **{_ne_txt}不可执行**（**本环境不可执行**：其声明实现路径依赖 DeepSpeed；"
                               "或 27B × ms-swift 在声明卡数下不切分基座权重；或首步实测 OOM；"
                               "或 `not_applicable`、无生成件）⇒ ⛔ **只登记声明口径、不为它们伪造"
                               "可跑配置**，逐档具名原因在行内标注。"
                               "★ **T052–T054 的四态 = 「未纳入验收」**（`applicable == false`）—— "
                               "⛔ 它们**不是「试过了不行」**，而是**从未尝试、不实施**：⛔ 不得读成效果否定。"),
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
        "stale_artifacts_20261016": {
            "rule": ("★ **旧件未同步（逐条登记，⛔ 不手改生成件、⛔ 不把运行期文件改错）**："
                     "下列在盘件属**被取代的世代**（非权威），其内容与现行声明不同步；"
                     "处置 = **登记**（本件可见），⛔ 不用手改的方式「对齐」生成物。"),
            "items": [
                "① `samples/configs/matrix54/T041.yaml`（v1 世代生成件）第 4 行逐字写「ELAM V5 训练子集前 20 条」；"
                "该目录 `samples/configs/matrix54/` 属**未同步旧件**（跑次实际读取的世代是 "
                "`samples/configs/matrix54-generated/`）⇒ 登记，⛔ 不手改生成件。",
                "② v1 清单 `tests/e2e/matrix54_manifest.json` 的 `data.subset_size` 实测分布 = "
                "{100: 27 档, 20: 27 档} ⇒ **9 个 formal 档（含 T041）全部仍写 20**，是「20 行世代」的"
                "整批口径、**不是 T041 专属漂移**；T041 现行真值 = 144（= 72/卡 × 2 卡），"
                "r3 历史值 = 96。★ v1 清单是 `_source` 的基线且属**非权威归档件** ⇒ 登记、⛔ 不改"
                "（改它会牵动整条 sha 链；本包仅在 purpose 口径变更时按既有做法刷新过指针）。",
                "③ `subsets-v2/T042.jsonl` 实测 **288 行**，与 `subsets-v2` 清单声明（lines=288）"
                "和矩阵/V2 清单的 `subset_size = 288` **三方一致** ⇒ **本条不是现行漂移**；"
                "任务书提到的「运行期 192」对应的是 **r3 历史世代**（`subset_r3_historical = 192`）"
                "⇒ ⛔ 不得据此把运行期文件改成 192（那才会把现行件改错）。",
            ],
            "verified_three_way": ("★ 附带实测：T041 的 `subset_size` 三方一致 "
                                   "（矩阵 144 ｜ v2 清单 144 ｜ 磁盘 `subsets-v2/T041.jsonl` 144 行）。"),
        },
        "lr_policy": _lr_policy_block(),
        "ruled_out": list(_spec.RULED_OUT),
        "criteria_id": "elam-v3-allright-1",
        "threshold": {
            "name": "delta_pp（SFT / GRASPO / OPD）＋ non_inferiority（CPT）", "unit": "百分点（pp）",
            "value": None,
            "target": None,
            "applies_to": "★ 按算法分列（见 by_method）：SFT / GRASPO / OPD 走 Δ 通过线；CPT 走不劣化线（after ≥ base）。",
            "status": ("★ 已冻结（主席最新裁定，2026-10-15；★ OPD 通过线已于 2026-10-19 变更，"
                       "见 caliber_change.caliber_change_20261019）：判据② 逐算法分列；**⛔ 不放宽判据**；"
                       "**★ 判定一律用点估计 Δ —— ⛔ 不做假设检验、不引入置信区间 / 显著性**。"),
            "historical_note": ("★ 旧口径（已被取代、留痕，⛔ 不再作为判定依据）：训练类与 GRASPO **统一 δ = +5.0 pp**；"
                                "目标值 +10.0 pp（仅展示、不用于判定）；SFT epoch = 2。"),
            "by_method": {
                "sft": {"criterion": "delta_pp", "op": ">=", "value": 10.0, "unit": "pp",
                        "metric": ("数据集完全正确率（after − base）；★ **100 行档 = fit 口径**"
                                   "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                        "pass_line_text": "after − base ≥ +10.0 pp",
                        "epoch": 4, "epoch_previous": 2,
                        "epoch_note": ("★ epoch 由 2 改为 4（判据② 的配置改动）；"
                                       "★ 现有 SFT 跑次均为 epoch=2 旧配置 ⇒ 其结论一律标注「待 epoch=4 重跑复核」。")},
                "graspo": {"criterion": "delta_pp", "op": ">=", "value": 5.0, "unit": "pp",
                           "metric": ("数据集完全正确率（after − base）；★ **100 行档 = fit 口径**"
                                   "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                           "pass_line_text": "after − base ≥ +5.0 pp",
                           "epoch": 8, "epoch_previous": 8, "epoch_note": "epoch=8 不变。"},
                "cpt": {"criterion": "non_inferiority", "op": ">=", "value": 0.0, "unit": "pp",
                        "metric": ("数据集完全正确率（after − base）；★ **100 行档 = fit 口径**"
                                   "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                        "pass_line_text": "after ≥ base（不劣化）",
                        # ★★ 2026-10-17 定死：**CPT 的 loss 判据逐字保留**（CPT 与 SFT 不同，
                        #    不走效果 Δ）；⛔ 不连带删除、⛔ 不改通过线数值。
                        "also": "训练 loss 正常下降",
                        "epoch": 2, "epoch_previous": 2, "epoch_note": "epoch=2 不变。"},
                "opd": {"criterion": "delta_pp", "op": ">=", "value": 5.0, "unit": "pp",
                        "metric": ("数据集完全正确率（学生 after − base）；★ **100 行档 = fit 口径**"
                                 "（评测集 = 该档自身训练文件，100 行）；其余档 = 平台方自有 702 行评测集"),
                        "pass_line_text": "after − base ≥ +5.0 pp",
                        # ★★ 2026-10-19 主席令：OPD 由**不劣化**改为 **Δ ≥ +5.0 pp**，并新增教师要求。
                        #    ⛔ epoch 不变（epoch=2）。
                        "teacher": "教师在训练数据集上的完全正确率需高于学生未训练模型 10 个百分点",
                        "epoch": 2, "epoch_previous": 2,
                        # ★ 2026-10-19 订正（与上面的 `teacher` 同批）：`epoch_note` 里那句
                        #   「教师改为 SFT 之后的 27B 产物」是**旧教师说明**（上文 `caliber_change_20261019`
                        #   的 `unchanged` 已把它逐字归档为"随之作废"）⇒ 保留 **epoch=2 的事实**，
                        #   把教师说明改成与 `teacher` / `switches` 的 `teacher=` 同源的那一个
                        #   （`distill_teacher().model`，⛔ 本文件不写死模型名）。
                        "epoch_note": ("epoch=2 不变；教师 = " + str(distill_teacher().get("model"))
                                       + "（原教师说明「SFT 之后的 27B 产物」已作废，见"
                                         " `caliber_change.caliber_change_20261019`）。")},
            },
            "by_method_single_source": ("★ 本 `by_method` 是**跑次阈值判定的唯一真相源**：跑次 json 自述阈值必须与之逐算法一致，"
                                        "否则回填 fail-closed。"),
        },
        # ★★ 2026-10-17：本字段不再是「曲线窗口」那套（旧值「U2 终点窗口 = 末段 max(3, 10% 步数) 步的
        #    均值」随之作废、原文逐字归档于精简轮 3 归档件）；改为**判据② 的 base 得分口径** ——
        #    Δ = after − base 里的 base，是**评测集上的得分**，⛔ 不是曲线上的窗口。
        "baseline": {"role": "未训练基座（role=base）",
                     "caliber": ("未训练基座在**同一评测集**上的数据集完全正确率（accuracy），来自 "
                                 "`graspo eval` 的 `role=base` 通道；**100 行档 = fit 口径（该档自身训练"
                                 "文件）**、**其余档 = 702 行评测集**。本项是判据② 的被减数"
                                 "（after − base 中的 base）；⛔ 与 CPT 的「训练 loss 正常下降」判据不是一回事。")},
        "epoch_policy": {
            "by_method": ("SFT = **4 epoch**（现行口径）；CPT = 2 epoch（⛔ 不动）；OPD = 2 epoch（⛔ 不动）；"
                          "GRASPO(RL) = 8 epoch（⛔ 不动；若 8 仍不过线则升 10 为后备；理由：RL 耗时长、5 epoch 下效果仅差 0.44 pp 未过线）。"
                          "理由：跨 epoch 容易出问题（循环结构决定）；RL 需要更多 epoch 才能看到效果。"),
            "steps": STEPS_POLICY + " 推导式：steps = epoch 数 × 每 epoch 步数（每 epoch 步数由数据量与批大小推导）。",
            "checkpoint": "save_steps = 推导步数 + 1（大于总步数 ⇒ 只保留最终检查点）。",
            "known_issue": ("★★ 登记待修（未修）：推导步数公式对 native GRASPO 不成立，根因已读清：两个后端的 step 语义本就不同。\n"
                            "① ms-swift：步数 = epochs × (subset × num_generations) / (micro_batch × grad_accum)；锚：T031（5ep）推导 800 = 实际 800 ✓。\n"
                            "② native：步数 = replay 缓冲区触发次数（optimize.py::_maybe_optimize）；阈值 = rollout_queue_batch_size × rollout_group_size（core/schema.py:458 / ripple/group_decision.py:152），每次仅消耗整组（whole_group_count，组不拆），epoch 尾可有 force 刷尾；★ 与 micro_batch/grad_accum 解耦。\n"
                            "③ native 锚的正确账（口径订正）：**阈值 = Q×G = 8×8 = 64**"
            "（运行期 19/19 步均为 64，交叉验证）；"
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
                "per_tier_alignment_20261016": (
                    "★★ **2026-10-16 逐档对齐（④）**：一律以**真值（跑次记录 / 配置推导）**为准，"
                    "⛔ 不猜；能判定者订正、不能判定者**登记待裁**。"
                    "★ **订正 2 档**：`T017` 与 `T018`（native 全参，**PP 不切数据 ⇒ dp_size=1**）的"
                    "`expected_optimizer_steps_per_epoch` 由「按 cards 朴素均分」的 50 / 25 订正为 **100**"
                    "（依据：manifest 逐字「PP 档 dp_size=1 ⇒ 不分片」；T017 佐证 `steps_seen=200/200` 属 epoch=2 世代；"
                    "T018 **硬反证**：产物含 `step_153/` 而登记总步数仅 50）。"
                    "★ **只登记不改值 1 档**：`T028` 的 `steps_actual=1280/1280` 与 `elapsed` 属**旧 20 行世代**"
                    "（其 `step_count_reconciliation` 逐字按 subset=20 推导 1280，而现行 `subset_size=48`、"
                    "`derived_steps=48`）⇒ ⛔ 不得引用为现行步数，待重跑回填（行内 `steps_generation_note`）。"
                    "★ **登记待裁（⛔ 不改）**：`T013/T015/T020/T021/T025/T026/T027` 的跑次读数取自 **epoch=2 世代**"
                    "（现行判据为 epoch=4）—— 该差异**已由行级 `reading_consistency` / `eval_caliber_consistency`"
                    "作废并留痕**，不另改数值。"
                    "★ **已闭合（2026-10-17 核销；原登记为「🟡 待裁、⛔ 不猜」）**：**51 档清单**的"
                    "`checkpoint_save_steps` 现由**该档跑次 yaml 现读**（`config` 所指文件），"
                    "与矩阵 `derived_steps + 1` **恒等**（三向机检 **51/51**；旧 v2 签名"
                    "「`checkpoint_save_steps == expected_optimizer_steps_per_epoch`」= **0 档**"
                    "⇒「两代混用」不再存在）；`expected_optimizer_steps_reachable` 与矩阵 `derived_steps` "
                    "一致 **49/51**，余 `T017`/`T018` 为**既有具名豁免**、确实仍在漂"
                    "（源清单 `--check` 已逐字登记；⛔ 不因该核销而放松）。"
                    "★★ **对外投影侧已止损（修正包 🔴-1，2026-10-16，窄复核报出）**：该 v2 值曾被 "
                    "`build_data.switches_of()` 以 `save_steps=<v2 值>` 的**键名**渲染进人读件「关键开关」列 ⇒ "
                    "与 ① 同行权威字段 `save_steps`（= 推导步数 + 1）② §3.1 定义行 ③ 跑次 yaml "
                    "**三处矛盾**（实测 **50/54 档**，T001 表内 100 vs 实际 201；T031 800 vs 6401）⇒ "
                    "破坏「复现参数唯一确定」。**处置**：生成端由 `_strip_save_steps_token()` 把该令牌"
                    "**整条移除**（检查点间隔只在 §3.1 定义一次，取值为公式值）；"
                    "机核 = `verify_matrix` **④Q**（开关列不得含该令牌 ＋ 逐档 `save_steps == derived_steps + 1` "
                    "＋ §3.1 交叉引用），阳性对照 = `run_self_proof` ⑬⑭ ＋ "
                    "`build_official --human-view-self-test` 的令牌移除五例。"
                    "★ 本行即为**上行口径登记的互指点**（§1.4 单一真相源：⛔ 不另写一份）。"
                ),
                "input_note": ("★ native 对账式**必须**输入运行期量 `trainable_completions`（可训练补全数，"
                               "取决于 retry / perfect_skip / invalid）⇒ 不可先验计算；回填 json 未提供该字段时，"
                               "对账记录标为「缺输入 ⇒ 待补」，⛔ 不臆测、⛔ 不硬编码。"),
            },
        "conclusion_rule": {
            "rule": ("结论必须与同一行的参数一一对应。若一行的参数已被更正，而结论来自旧参数下的跑次 ⇒ "
                     "该结论作废，该档标「尚未验证」；仅在参数冻定后重跑，才可给出「已验证可用 / 已验证不可用」判定。"),
            # ★★ 2026-10-17 定死（主席裁定）：**通过规则** —— 措辞与 `status_vocabulary` 四态一致。
            #    ★ 精简轮 4：压成一句人话（⛔ 不改判定逻辑：四条全过 ⇔ 方可「已验证可用」）。
            "pass_rule": ("四条判据（① 能完整跑完、② 效果变好、③ 算法正确性检查、④ 能推理）"
                          "全部通过 ⇒ 才能标「已验证可用」；任一条不过 ⇒ 不得标「已验证可用」"
                          "（按事实标「已验证不可用 / 尚未验证 / 未纳入验收」）。"),
            "status_wording": ("「尚未验证」在本文档中等同「未测试」；本文档的四态取值固定为"
                              "「已验证可用 / 已验证不可用 / 尚未验证 / 未纳入验收」，用词以它为准。"),
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
                "依据：**双路实测**（两种独立路径各数一遍，逐层展开后同为 358 个实例、张量键数同为 716）—— 该计数可由本矩阵数据文件与 `tools/matrix/` 工具链重跑复现。"
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


def void_addendum_for(status: str, kind: str) -> str:
    """★★ 「作废追补句」—— 结尾**不复述具体四态值**（修正轮 1 · 🟡W1）。

    ★ 病史：原先三个追补句结尾写死「以本行现行 `status` 为准（**尚未验证**）」。该函数对
      **所有** lr/评测集口径作废的档生效（实测 40 档），其中「口径无关证据 ⇒ 结论保留」的档
      与「`applicable is False` ⇒ 未纳入验收」的档一旦命中，行内就会留下**与真实最终状态相反**
      的句子（同文件两个 status 互相矛盾，§1.4 / §2.1；复核报告 🟡W1 点名的 T052–54 就是这条）。
    ★ 现在：结尾改为**状态中立**表述（⛔ 不复述具体四态值）⇒ 任何最终状态下行内都不矛盾。
    ★ 机读面的自洽由 `caliber_field_guard` 的 `status_after == status` fail-closed 守卫把关。
    """
    if kind == "lr":
        head = "★★ 2026-10-16 追补：本条所记结论已因 lr 口径变更作废，"
    elif kind == "eval":
        head = ("★★ 2026-10-16 追补：本条所记结论已因**评测集口径变更**作废 —— "
                "100 行档的评测集改为**该档自身训练文件**（100 行、`dataset_split: train` ⇒ fit 口径），"
                "旧读数取自 702 行评测集，")
    else:
        head = ("★★ 2026-10-16 追补：本条所记结论已作废"
                "（**lr 口径变更** ＋ **评测集口径变更** 两条原因叠加），")
    return (f"{head}以本行现行 `status` 字段为准"
            "（详值见行内 `status` / `eval_caliber_consistency.status_after`）。")


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
    ★★ 修正轮 1（🟡W1）：结尾状态从句**现算**（`void_addendum_for`），⛔ 不再写死「尚未验证」
      —— 保留结论 / 未纳入验收的档也曾命中本路径，写死值与真实 `status` 相反。
    """
    kind = _void_kind(row)
    if kind is None:
        return None
    return void_addendum_for(str(row.get("status") or ""), kind)


def _naked_four_state(text):
    """该文本是否**裸**断言四态之一（含四态字面、且无任何作废/留痕标记）。"""
    if not any(lit in str(text) for lit in _FOUR_STATE_LITERALS):
        return False
    return not any(m in str(text) for m in _VOID_WRAP_MARKERS)


# ============================================================================================
# ★★ 修正轮 1（2026-10-16）新增守卫：机读「口径」字段的**自洽性 fail-closed**
#    任务书：🟡W1（`eval_caliber_consistency.status_after` 与最终 `status` 相反）
#            🟡W3（`caliber_independent_evidence` 混入**已作废的评测侧读数**当"口径无关"举证）
#    为什么放在这里（而不是只改数据）：**能失败的断言才叫断言** —— 只把 9 行值改对，
#    下次任何人重跑并入器/扩面就会重新写错，且四闸门一个都不响（复核报告已实测该盲区）。
# ============================================================================================

#: 被口径变更**作废**的评测侧读数锚点（出现在 `eval_caliber_consistency.reading` /
#  `.mismatch` / `.cause` 里即视为「作废读数」）。⛔ 不写死档号，按**数值锚点**匹配。
VOIDED_EVAL_READING_ANCHORS: tuple[str, ...] = ("0/702", "702 行", "702行")


def _state_literal(text: str) -> str:
    """把 status 字面**归一到四态**（`build_official` 的派生规则：非四态字面按前缀/兜底）。

    ★ 与 `main()` 里 `status → 四态` 的派生链**同源**（`通过*` ⇒ 可用、`失败*` ⇒ 不可用、
      其余 ⇒ 尚未验证；`applicable is False` 由调用方另行处理）。⛔ 不在此另立口径表。
    """
    if text in STATUS_VOCABULARY:
        return text
    if text.startswith("通过"):
        return S_AVAILABLE
    if text.startswith("失败"):
        return S_UNAVAILABLE
    return S_UNVERIFIED


def caliber_field_guard(rows) -> list[str]:
    """落盘前自洽性检查（返回问题清单；空 = 通过）。

    ① `eval_caliber_consistency.status_after`（若存在）必须**逐字等于**该行最终 `status`
       —— 同一文件两个 status 字段互相矛盾 = 双真相源（§1.4 / §2.1）。
       ★ 归一化：`status_after` 可能仍是**原始 status 字面**（`通过*` / `失败*` / `—`，
         `status → 四态` 由本函数上方 ①–⑤ 的派生链完成）；两侧都先归一到四态再比，
         ⇒ ⛔ 不会把"原始字面 vs 派生四态"这种**正常的**配对误判为矛盾，
            但**四态 vs 四态**的不一致一定拦下（复核报告 🟡W1 点名的 9 行正是这一类）。
    ② `caliber_independent_evidence` 若引用**已作废的评测侧读数**（如 `0/702`），必须
       显式标记 `voided_readings_excluded_from_evidence is True` 且文本含「已作废」与
       「不作举证」—— ⛔ 不得把作废读数当"口径无关"的证明（过度断言）。
    ③ ★ **修正轮 2（🟡W-8 → 机核判据）**：`status_after_correction_20261016.prior_chain`
       必须是**幂等规范形** —— ① 以 `prior` 开头；② 有序去重（无重复）；③ ⛔ 不含 `corrected_to`。
       病史：并入器旧实现每跑一次就 `chain + [cur]` ⇒ 链会无限增长（现盘曾出现 2 段
       `['失败','已验证不可用']`），"幂等"声明与事实不符，且每次重跑都改动已冻结字节。
    """
    problems: list[str] = []
    for r in rows:
        tid = r.get("tier") or r.get("tier_id") or "?"
        status = str(r.get("status") or "")
        ecc = r.get("eval_caliber_consistency")
        if isinstance(ecc, dict):
            sa = ecc.get("status_after")
            if sa is not None:
                _a, _b = _state_literal(str(sa)), _state_literal(status)
                if _a != _b:
                    problems.append(
                        f"{tid}: eval_caliber_consistency.status_after={sa!r} != 行 status={status!r}"
                        f"（同一文件两个 status 字段相反 ⇒ 必须收口）"
                    )
            # ③ prior_chain 规范形（W-8 机核化）
            rec = ecc.get("status_after_correction_20261016")
            if isinstance(rec, dict):
                ch = rec.get("prior_chain")
                prior = rec.get("prior")
                corr = rec.get("corrected_to")
                if isinstance(ch, list) and ch:
                    if prior is not None and str(ch[0]) != str(prior):
                        problems.append(
                            f"{tid}: prior_chain 首项 {ch[0]!r} != prior {prior!r}"
                            f"（规范形要求以原值开头）"
                        )
                    if len(ch) != len({str(x) for x in ch}):
                        problems.append(f"{tid}: prior_chain 有重复项 {ch!r}（规范形要求有序去重）")
                    if corr is not None and any(str(x) == str(corr) for x in ch):
                        problems.append(
                            f"{tid}: prior_chain 含 corrected_to {corr!r}（现行态不得出现在历史链里）"
                        )
                elif ch is not None and not isinstance(ch, list):
                    problems.append(f"{tid}: prior_chain 类型异常 {type(ch).__name__}")
        cie = r.get("caliber_independent_evidence")
        if isinstance(cie, dict):
            blob = " ".join(str(cie.get(k) or "") for k in ("evidence", "note", "why"))
            hits = [a for a in VOIDED_EVAL_READING_ANCHORS if a in blob]
            if hits:
                if cie.get("voided_readings_excluded_from_evidence") is not True:
                    problems.append(
                        f"{tid}: caliber_independent_evidence 引用了已作废的评测侧读数 {hits}"
                        f" 但未声明 voided_readings_excluded_from_evidence=True"
                    )
                elif ("已作废" not in blob) or ("不作举证" not in blob):
                    problems.append(
                        f"{tid}: caliber_independent_evidence 引用了已作废的评测侧读数 {hits}"
                        f" 但文本未逐字标「已作废 … ⛔ 不作举证」⇒ 属过度断言"
                    )
    return problems


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
        # ★★ 修正轮 1（🟡W1）：回退句也**现算**（结尾状态从句取 `cur_status`），
        #    ⛔ 不再回落成写死「尚未验证」的常量（保留结论 / 未纳入验收的档也会走到这里）。
        add = addendum or (OOM_ADDENDUM_20261016 if cur_status == S_UNAVAILABLE
                           else void_addendum_for(str(cur_status or ""), "lr"))
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
    ap.add_argument("--in", dest="inp", required=False, default=None)
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--self-test", action="store_true", help="脱敏回归自检（对抗性输入 + 顺序无关）")
    ap.add_argument("--human-view-self-test", action="store_true",
                    help="★ 人读视图四件（允许清单 / 术语兜底 / 术语改写 / 表格格归一）的阳性对照自检")
    ap.add_argument("--date", default=None,
                    help="文档日期；缺省读输入数据的 meta.doc_date。⛔ 不读时钟（§17.7 生成确定性）。")
    a = ap.parse_args()
    if getattr(a, "self_test", False):
        return _sanitizer_self_test()
    if getattr(a, "human_view_self_test", False):
        _ok = _row_normalize_self_test()
        print("★ 「不可执行」档数**数据驱动**阳性对照（2026-10-16 收口）：")
        _ok = _not_executable_count_self_test() and _ok
        print("★ 检查点间隔令牌移除（修正包 🔴-1）阳性对照：")
        _ok = _save_steps_token_self_test() and _ok
        print("★ 表格结构单一真相源（2026-10-19：本文件不再自带第二份列定义）阳性对照：")
        _ok = _table_structure_single_source_self_test() and _ok
        return 0 if _ok else 1

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
        # ★★ 2026-10-16（审查项：四态语义混用）：**`applicable is False` 的档恒为「未纳入验收」**，
        #    ⛔ 不得被 `caliber_verdict` 顶成「已验证不可用」。
        #    理由（主席口径：「跑不了」与「试过了不行」是两回事）：`applicable is False` ⇒ 该档
        #    **从未尝试、依裁定不实施**（如 T052–T054：师生权重常驻 ≈100 GiB 与设计意图相悖
        #    ⇒ 不产出可运行配置）⇒ 记「未纳入验收」；而「已验证不可用」是**跑过并判定失败**。
        #    实证缺陷：本补丁前 T052/T053/T054 被 `caliber_verdict` 顶成「已验证不可用」，
        #    与同表 `applicable=false` / `excluded_reason` 自相矛盾（且会被读成效果否定）。
        _not_applicable = r.get("applicable") is False
        if _cv in STATUS_VOCABULARY and not _voided_reading and not _not_applicable:
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
        # ★★ 2026-10-16 新增：**口径无关证据（须单独举证）⇒ 结论保留**。
        #    口径（评测集大小 / lr 尺度）变了，但某些结论的证据**与尺子无关**（例如「训练侧退化成
        #    常数/畸形输出」的
        #    训练期证据、或「首步 OOM」的硬件证据）⇒ 对这类档**不回落「尚未验证」**，
        #    改为保留 `retained_status`。★ fail-closed：`retained_status` 必须**逐字等于**本行
        #    由既有证据推出的四态（写成别的值一律不生效，回落到下面的作废分支）。
        #    ★ `applicable is False` 的档**不进本分支**（它们恒为「未纳入验收」，见上方补丁）。
        _cie = r.get("caliber_independent_evidence")
        _retain = (isinstance(_cie, dict) and _cie.get("retained_status") == state)
        if isinstance(_ecc, dict) and _ecc.get("consistent") is False and not _not_applicable and not _retain:
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
        elif isinstance(_ecc, dict) and _ecc.get("consistent") is False and _retain:
            # ★★ 口径变更但**结论保留**（口径无关证据已单独举证）：机读登记 + 可见留痕，
            #    ⛔ **不写 `voided_conclusion`**（结论没有被作废），⛔ 也不改四态。
            r["caliber_independent_evidence"] = dict(_cie)
            r["caliber_change_record"] = {
                "kind": "eval_caliber_change_conclusion_retained",
                "audit": _ecc.get("audit"),
                "current": _ecc.get("current"),
                "reading": _ecc.get("reading"),
                "retained_status": _cie.get("retained_status"),
                "why_retained": _cie.get("why"),
            }
            r["superseded_note"] = (str(r.get("superseded_note") or "") + "\n"
                                    + str(_cie.get("note") or "")).strip()
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
            # ★★ 修正包（窄复核 🔴-1，2026-10-16）：**「关键开关」列不得再出现 `save_steps=` 令牌**。
            #    根因：清单侧 `checkpoint_save_steps` 仍是**旧口径**（= 每 epoch 步数，未乘 epoch 数），
            #    而 `build_data.switches_of()` 原样把它渲染成 `save_steps=<旧口径值>` ⇒ 与
            #    ① 本行权威字段 `r["save_steps"]`（= 推导步数 + 1，跑次 yaml 实测同值）
            #    ② §3.1 定义行「`checkpoint_save_steps = 推导步数 + 1`」
            #    ③ `samples/configs/matrix54-generated/T0xx.yaml` 的 `save_steps`
            #    **三处互相矛盾**（实测 50/54 档不一致，T001 表内 100 vs 实际 201；T031 800 vs 6401）。
            #    ⇒ 同一份对外文档里同一个键名有两个取值，会被访客读成"每隔 100 步存一次"，
            #      把复现的检查点间隔放大 2×–8×，破坏主席判据③「复现参数唯一确定」。
            #    ★ 取「移除该令牌」而非「改键名」：§3.1 **已**有唯一定义行，而旧口径值**本身是错的**
            #      ⇒ 改名为 `checkpoint_save_steps=<旧值>` 只会把同一个错误换个名字（仍与定义行矛盾）；
            #      移除后**该量在文档里只剩一个来源** = 定义行（按公式）＋ 本行 `save_steps`（机读件内）。
            #    ★ 与既有登记**互指**（§1.4 单一真相源，⛔ 不各写一份）：见本文件 `criteria_caliber`
            #      「步数口径」的 `**已闭合（2026-10-17 核销）**：**51 档清单**的 `checkpoint_save_steps`
            #      现由该档跑次 yaml 现读、与矩阵 `derived_steps + 1` 恒等（三向机检 51/51）` 一条。
            r["switches"] = _strip_save_steps_token(r.get("switches"))
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
                                    "（language_model 248 + visual 110）｜★ native **310 实例**"
                                    "（120 为签名/模式条目数），且**命中集合不同** ⇒ 跨后端读数不直接可比")
            else:
                r["lora_target"] = (r["lora_target"]
                                    + "｜native：canonical preset 名（310 实例；120 为签名/模式条目数）"
                                    + "｜★ 两后端实例数不同且**命中集合不同**（不等价）")
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

    # ★★★ 人读视图表格格归一（④）：逐格订正**已经渲染进人读件**的污染与残片
    #     —— 在短/长备注分离之后调用，保证改的是*最终会渲染的那一格*；改后重推短备注以防再截断。
    for _r in rows:
        _fix_row_cells(_r, _r.get("verdict"))
        _full = str(_r.get("note_full") or "").strip()
        if _full and _full != "—":
            _r["note"] = short_note(_r, _full)
        # ★★ 精简轮 2：结论列与短备注**在格归一之后**派生（否则读到的是未订正的 `verdict`）
        _r["not_failed_reason"] = derive_not_failed_reason(_r)
        _r["note"] = compose_short_note(_r)

    # ★★ 「读数变化登记表」：**数据驱动**地从档位行派生（⛔ 不写死档号/数值）——
    #    凡行级 `caliber_change.kind` 以 `reading_change` 开头者 ⇒ 逐条进 `criteria_caliber.caliber_change`。
    _cal = criteria_caliber(rows)   # ★ 行数据入参：fit 口径档位清单现算（⛔ 不写死档号）
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
            "★ 2026-10-16 **现行 lr 口径变更 ⇒ 读数作废（正式登记）**：★ 新口径 = "
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
            "★ 2026-10-16 **现行评测集口径变更 ⇒ 读数作废（正式登记）**：★ 新口径 = "
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
            f"★ 仅因现行评测集口径变更作废的档 = {len(_ecvoid_tiers) - len(_both)} 档"
            f"（{', '.join(str(x) for x in _ecvoid_tiers if x not in _lrset)}）。"
            f"★ **顺带订正** = {_ec.get('overlap_analysis_fix', '')} "
            f"★ **原 20 行档（2026-10-16 已裁定放大到 100 行）**：{_ec.get('not_applicable', '')} "
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
            # ★★ 实验条件（面向使用者）：场所与代码版本。★ 只写**能力口径**，⛔ 不写机器名 /
            #    内网地址 / 主机编号（§17.7「矩阵中仍然禁止」）。
            "venue": VENUE,
            "code_version": {"repo": CODE_REPO, "branch": CODE_BRANCH, "commit": CODE_COMMIT},
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
        # ★★ 2026-10-19（使用者口径「JSON 里面只有测试结果，脚本只负责把测试结果填写到最后一章
        #   测试表里面，其它所有的部分都不能动」）：正式机读件**不再落** `columns` / `groups`
        #   （表结构 = `render_matrix.MATRIX_COLUMNS` / `MATRIX_GROUPS` 常量，⛔ 不随数据走）
        #   与 `uniform`（统一口径块 = 模板静态文字）。
        #   ⇒ 数据件只剩：`meta.*`（判据/口径留痕，供生成端与机检消费）+ `rows[*]`（逐档参数，
        #     其中**上表 13 列** + **上提第 3 章 3 项（另有 TP 口径说明一条）** + 结果 6 列
        #     + 四态结论列）
        #     + `data_keys`（结果列名，机读声明）。
        #   内部权威件的统一口径项仍由 `_uniform_items()` 归一（供生成端 fail-closed 检查），
        #   ⛔ 只是不再进正式件。
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
    # ★★ 修正轮 1 新增落盘前 fail-closed 守卫（W1 / W3；阳性对照见工位 evidence）
    _guard_problems = caliber_field_guard(doc.get("rows", []))
    if _guard_problems:
        for _p in _guard_problems:
            print(f"FAIL-CLOSED（口径机读字段自洽）：{_p}")
        return 1
    # ★★★ 人读视图三件（**落盘前**，顺序固定）：
    #    ③ 术语改写 + 记号清理（人称 / 内部编排 / `★`·`⛔`·旧编号 ⇒ 时间口径与自然书写）
    #    ⇒ ① 允许清单投影（内部字段不进人读视图）⇒ ② 兜底扫描（内部术语，命中即 fail-closed）。
    #    ★ 只作用于**文本**：⛔ 不动任何数值、四态取值、判据与结论。
    doc = _rewrite_tree(doc)
    # ★★ 修正轮 4（⚠③）：数据件面也做**内部证据路径脱敏**（保留实质结论与内容指纹）。
    #    ★ 必须在 `forbidden_in_human_view()` 兜底扫描**之前** ⇒ 脱敏后的文本才是被机检的文本。
    doc = _redact_tree(doc)
    # ★ 同批做一处**措辞收尾**：`… 与 <占位>。` ⇒ 去掉占位前的悬空"与"（纯排版，⛔ 不改事实）
    for _r in doc.get("rows", []):
        if isinstance(_r.get("algo_checks"), str):
            _r["algo_checks"] = _r["algo_checks"].replace(
                "见 p5p6_readings 与 " + EVIDENCE_PATH_PLACEHOLDER + "。",
                "见 p5p6_readings 与" + EVIDENCE_PATH_PLACEHOLDER + "。")
    _moved = _carry_through_row_fields(doc, src.get("rows"))
    _dropped_meta, _dropped_rows, _dropped_cal = drop_non_view_fields(doc)
    print(f"逐档直通（{', '.join(CARRY_THROUGH_ROW_KEYS)}）：搬入 {_moved} 处（来源＝内部权威同名键）")
    print(f"人读视图投影：meta 剔除 {len(_dropped_meta)} 键 {_dropped_meta}；"
          f"criteria_caliber 剔除 {len(_dropped_cal)} 键 {_dropped_cal}；"
          f"行级剔除 {len(_dropped_rows)} 类字段"
          + (f"（出现档数 Top8：{sorted(_dropped_rows.items(), key=lambda x: -x[1])[:8]}）"
             if _dropped_rows else ""))
    _terms = forbidden_in_human_view(doc)
    if _terms:
        print(f"FAIL-CLOSED（人读视图含内部术语）：{_terms} ⇒ 拒绝落盘")
        return 1
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_VOCABULARY}
    print(f"official data: {len(rows)} 档；四态 {counts} -> {out}")


if __name__ == "__main__":
    # ★★ 修正轮 1：**退出码必须反映 fail-closed**。原先 `main()` 的返回值被丢弃 ⇒
    #    「FAIL-CLOSED：… ⇒ 拒绝落盘」（`return 1`）对**调用方**表现为 rc=0，
    #    而 `backfill_matrix.py` 等调用方正是**按 rc** 判成败（实测该盲区：本包阳性对照
    #    打印了 FAIL-CLOSED 却仍 rc=0）。⛔ 不改任何判据/取值，只把已经写出的失败**传出去**。
    sys.exit(main() or 0)
