#!/usr/bin/env python3
"""gen_configs.py — 由矩阵数据（唯一可编辑源头）确定性生成各档测试 yaml。

★ 方向固定：**矩阵数据 → yaml（产物）**。⛔ yaml 不得手工编辑。
★ 生成器**不读时钟 / 环境 / 网络 / 人机交互**；同一 (数据, 参数) ⇒ 同一 yaml 字节。
★ 矩阵数据的路径为符号路径（已脱敏）：真实前缀由命令行参数 `--model-root` / `--data-root` 注入，
   因此生成结果只依赖 (数据, 参数)，可复现。

用法：
  python3 tools/matrix/gen_configs.py --data docs/capability-matrix.data.json \
      --out-dir samples/configs/matrix54-generated --only T010,T028
  确定性/幂等自证：
  python3 tools/matrix/gen_configs.py --data <data.json> --out-dir <dir> --check-determinism
  与既有冻结 yaml 的差异归因：
  python3 tools/matrix/gen_configs.py --data <data.json> --out-dir <dir> --check-dir samples/configs/matrix54-v2
"""
import argparse
import json
import re
import sys
from pathlib import Path

HEADER = ("# 本文件由 tools/matrix/gen_configs.py 从 docs/capability-matrix.data.json 自动生成，"
          "⛔ 禁止手工编辑；请改数据后重新生成。\n")


def opd_teacher_missing(text):
    """★ 断言辅助（可单测）：OPD 档是否缺教师（`teacher_model_path` 非空 或 显式 `teacher_model_server`）。

    返回 True = **缺**（应 fail-closed）。
    """
    if "train_method: opd" not in text:
        return False
    import re as _re
    m = _re.search(r"^\s+teacher_model_path:\s*(\S+)\s*$", text, _re.M)
    srv = _re.search(r"^\s+teacher_model_server:\s*(\S+)\s*$", text, _re.M)
    ok_path = bool(m) and m.group(1) not in ('""', "''", "~", "null")
    ok_srv = bool(srv) and srv.group(1) not in ('""', "''", "~", "null")
    return not (ok_path or ok_srv)


def opd_cards_shortfall(teacher, cards):
    """★ OPD 卡数断言（**数据驱动**）：manifest 若声明了 `min_cards`（P-25 实测回填后），则 `cards` 不得低于它。

    返回 None = 通过；否则返回拒绝理由。★ manifest **未声明**时不做限制（其规则明载教师侧"未测算"、
    判定恒为 unmeasured，禁止凭空编造卡数）。
    """
    if not teacher or not teacher.get("min_cards"):
        return None
    try:
        c = int(cards or 0)
    except Exception:
        return f"cards 不可解析（{cards!r}）"
    mc = int(teacher["min_cards"])
    if c < mc:
        return f"OPD 卡数不足：cards={c} < feasibility_model.opd_teacher.min_cards={mc}"
    return None


def opd_pending_unmarked(teacher, row):
    """★ 未定态断言：`measured` 非 True 且 `min_cards` 未声明 ⇒ 该档**必须带未定标记**（`cards_pending`），
    ⛔ 不允许"看起来已定"地生成（否则会被误当可跑）。返回 True = 缺标记（应 fail-closed）。
    """
    if not teacher:
        return False
    if row.get("availability") == "blocked":
        return False          # ★ blocked 档由 blocked 断言负责（它们本就不得起跑）
    if teacher.get("measured") is True and teacher.get("min_cards"):
        why = opd_cards_shortfall(teacher, row.get("cards"))
        return bool(why) is False and False        # 已定态：由卡数断言负责
    return not bool(row.get("cards_pending"))


def cards_procs_mismatch(row):
    """★ 同源性断言：**卡数与训练进程数必须一致**（"两个必须同源的参数只改一个"是事故根因）。

    · native：`cards == tp_size × pp_size × dp_size`（并行维度落在 native 段）
    · ms-swift：`cards == nproc_per_node`（并行维度落在 parallel 展示串）
    · 另：`gpus` 的**个数**必须等于 `cards`（★ set_cards 类改动必须三处同改）
    返回 None = 一致；否则返回拒绝理由。
    """
    def _i(v):
        try:
            return int(v)
        except Exception:
            return None
    cards = _i(row.get("cards"))
    if cards is None:
        return None
    tp, pp, dp = (_i(row.get("native_tp")), _i(row.get("native_pp")), _i(row.get("native_dp")))
    if any(v is not None for v in (tp, pp, dp)):
        if all(v is not None for v in (tp, pp, dp)):
            world = tp * pp * dp
            if world != cards:
                return f"native 并行维度与卡数不同源：tp×pp×dp={world} ≠ cards={cards}"
    parallel = str(row.get("parallel") or "")
    m = re.search(r"nproc_per_node\s*=\s*(\d+)", parallel)
    if m:
        nproc = int(m.group(1))
        if nproc != cards:
            return f"ms-swift 进程数与卡数不同源：nproc_per_node={nproc} ≠ cards={cards}（⛔ 不得只改一半）"
    gpus = row.get("gpus")
    if isinstance(gpus, list) and gpus and len(gpus) != cards:
        return f"gpus 个数与卡数不同源：len(gpus)={len(gpus)} ≠ cards={cards}"
    return None


def config_pins_gpus(text):
    """★ 断言辅助：配置文本里**不得**出现 `gpus:`（★ 卡号一律不进配置；由起跑侧动态择卡，§12 一族）。"""
    import re as _re
    return bool(_re.search(r"^\s*gpus\s*:", text, _re.M))


def blocked_vs_unavailable_conflict(row):
    """★ 轴冲突断言：**同一档不得同时标 `blocked` 与四态的「已验证不可用」**。

    理由（2026-10-13 裁定）：**「已验证不可用」是效果层否定**（方法/输出退化），
    **`blocked` 是可用性层**（硬件/依赖不可行）⇒ 混用会污染结论语义（后人会读成"该方法失败"）。
    """
    if row.get("availability") != "blocked" or str(row.get("status") or "") != "已验证不可用":
        return False
    # ★★ 例外（主席口径变更 2026-10-14）：**"跑了没成功＝失败"** ⇒ blocked 档**可以**记四态「已验证不可用」，
    #    但**必须**带 `caliber_change` 标记（★ 且真因仍在 `blocked_reason`/`caliber_cause` 里逐字保留）
    #    ⇒ ★ 否则仍判轴混用（防"悄悄把硬件不可行写成效果失败"）。
    return not bool(row.get("caliber_change"))


def family_under_min_unblocked(row):
    """★ 合成谬误拦截：**档级"可起"必须与族级实测相容**。

    若该档属某族（`family`）且该族有实测 `family_min_cards`，而本档 `cards < family_min_cards`，
    则本档**必须**已标 `availability == "blocked"`（★ 单档看"ready"、与族级事实矛盾 ⇒ 拒绝产出）。
    返回 True = **不相容**（应 fail-closed）。
    """
    fmc = row.get("family_min_cards")
    if not fmc or not row.get("family"):
        return False
    try:
        under = int(row.get("cards") or 0) < int(fmc)
    except Exception:
        return False
    return under and row.get("availability") != "blocked"


def blocked_contradicted_by_live(row):
    """★ 反证拦截：标了 `blocked`，但**存在与之矛盾的活证据**（如已在跑的训练进度 / `ec=0`）⇒ 必须拒。

    与 `family_under_min_unblocked` 同族、**方向相反**：那条防"该标没标"，这条防"**不该标乱标**"。
    """
    if row.get("availability") != "blocked":
        return False
    return bool(str(row.get("live_evidence") or "").strip())


def combo_infeasible_still_runnable(row, falsified):
    """★ 同配拦截：**同族同后端内已被实测证伪的 `(cards)` 组合**，⛔ 不得再有标为可起的同组合档。

    `falsified`: dict[(family, backend)] = set(cards)（实测 OOM 的卡数）
    返回 True = **违规**（应 fail-closed）。
    """
    key = (row.get("family"), row.get("backend"))
    try:
        cards = int(row.get("cards") or 0)
    except Exception:
        return False
    if key in falsified and cards in falsified[key] and row.get("availability") != "blocked":
        return True
    return False


def citation_missing_config(text, cited_tier, cited_row):
    """★ 引用纪律：结论若**引用了其他档的实测（外推）**，则**必须同时引用被引档的配置字段**才能成立。

    `text`: 结论文本；`cited_tier`: 被引档号；`cited_row`: 被引档的**配置**（须含 backend/cards 等）。
    返回 True = **缺配置依据**（应 fail-closed）。
    """
    if not text or cited_tier not in str(text):
        return False
    if not cited_row:
        return True
    need = ("backend", "cards")
    return any(not cited_row.get(k) for k in need)


def derive_tier_status(matrix_status, availability=None, running=False):
    """★★ **`status` 由矩阵推导**（单一真相源，⛔ 不得手工维护）。

    词表：已判定 ⇒ `judged`｜在跑 ⇒ `running`｜blocked ⇒ `blocked`｜未纳入验收 ⇒ `not_applicable`｜其余 ⇒ `ready`
    """
    if availability == "blocked":
        return "blocked"
    if matrix_status == "未纳入验收":
        return "not_applicable"
    if matrix_status in ("已验证可用", "已验证不可用"):
        return "running" if running else "judged"
    return "running" if running else "ready"


def status_ready_but_judged(matrix_status, status):
    """★ 一致性断言：四态 ∈ {已验证可用, 已验证不可用} ⇒ ⛔ `status` 不得为 `ready`（否则会被重跑）。"""
    return matrix_status in ("已验证可用", "已验证不可用") and status == "ready"


def running_but_ready(row):
    """★ 有活证据（在跑）却标 `ready` ⇒ 违规（与"已判定却 ready"同族、方向相反）。"""
    return bool(str(row.get("live_evidence") or "").strip()) and row.get("_status") == "ready"


#: ★ 生效集合：**由指挥部给、与铺开进度同步扩大**（⛔ 不得一次要求全部档）。
#: 2026-10-13 起 = {"T023"}（单档小步验证：该档已知无评测通道）。
FINAL_FORMATS_REQUIRED: set[str] = {"T018", "T023", "T030", "T036", "T042"}
#: ★ 逐档取值（★ 与生效集合同步维护；2026-10-13 起 = T023 单档小步）
FINAL_FORMATS_BY_TIER: dict[str, list[str]] = {
    # ★ 只对"未跑/未判定/非 blocked"的档生效（已判定档保持可比性、⛔ 不动）
    "T018": ["merged-hf"], "T023": ["merged-hf"], "T030": ["merged-hf"],
    "T036": ["merged-hf"], "T042": ["merged-hf"],
}
#: ★ 取值域（与 `lora_io.export_from_checkpoint` 对齐，⛔ 不得自造）
FINAL_FORMATS_ALLOWED: set[str] = {"peft-adapter", "merged-hf"}


def final_formats_explicit(row):
    """★ 断言：**生效集合内**的档必须显式给出 `export.final_formats`（缺 / 空 ⇒ 拒）。

    ★ 作用范围 = `FINAL_FORMATS_REQUIRED`（★ 与铺开同步扩大；⛔ 作用范围外的档不受约束，防误伤）。
    """
    if row.get("tier") not in FINAL_FORMATS_REQUIRED:
        return False
    ff = row.get("final_formats")
    return not (isinstance(ff, list) and len(ff) > 0)


def final_formats_valid(vals):
    """★ 断言：取值必须 ⊆ `{peft-adapter, merged-hf}`（★ 与上游 ValueError 对齐，提前到生成期）。"""
    if vals is None or not isinstance(vals, list) or not vals:
        return False          # ★ 缺/空 由 final_formats_explicit 负责；这里只管"给了但非法"
    return any(str(v) not in FINAL_FORMATS_ALLOWED for v in vals)


#: ★★ 镜像 `LaunchConfig` 的**允许键集合**（逐字来自镜像报错信息；`extra="forbid"` ⇒ 越界即 SystemExit）。
LAUNCH_ALLOWED_KEYS: set[str] = {
    "determinism", "master_addr", "master_port", "nnodes", "node_rank", "nproc_per_node", "python",
}


def launch_keys_invalid(text: str) -> bool:
    """★ 断言：产物 `launch:` 段的键集合必须 ⊆ `LAUNCH_ALLOWED_KEYS`（否则起跑当场 SystemExit）。"""
    import re as _re
    m = _re.search(r"^launch:\s*$([\s\S]*?)(?=^\S|\Z)", text, _re.M)
    if not m:
        return False                      # ★ 无 launch 段 ⇒ 放行（非必需段）
    keys = set(_re.findall(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", m.group(1), _re.M))
    return not keys <= LAUNCH_ALLOWED_KEYS


def blocked_unmarked(row):
    """★ blocked 档必须有 `blocked_reason`（起跑侧据此拒跑）；缺理由 ⇒ fail-closed。"""
    if row.get("availability") != "blocked":
        return False
    return not bool(str(row.get("blocked_reason") or "").strip())


def _gen_self_test():
    cases = [
        ("① OPD 有 teacher_model_path ⇒ 放行", not opd_teacher_missing("train_method: opd\n  teacher_model_path: /models/Qwen3.8-27B\n")),
        ("② ★ 负例：OPD 缺 teacher ⇒ **必须判缺**（fail-closed）", opd_teacher_missing("train_method: opd\nlora:\n  r: 8\n")),
        ("③ ★ 负例：OPD 空值 teacher ⇒ **必须判缺**", opd_teacher_missing("train_method: opd\n  teacher_model_path:\n")),
        ("④ 非 OPD 档 ⇒ 不判缺", not opd_teacher_missing("train_method: sft\n")),
        ("⑤ 卡数断言：min_cards=2 且 cards=1 ⇒ **必须拒**",
         opd_cards_shortfall({"min_cards": 2}, 1) is not None),
        ("⑥ 卡数断言：min_cards=2 且 cards=2 ⇒ 放行", opd_cards_shortfall({"min_cards": 2}, 2) is None),
        ("⑦ ★ 卡数断言：manifest 未声明 min_cards ⇒ **不放宽也不臆造**（不限制）",
         opd_cards_shortfall({"measured": False}, 1) is None),
        ("⑧ ★ 未定态：实测未完成且未声明 min_cards ⇒ 缺 `cards_pending` **必须判缺**（fail-closed）",
         opd_pending_unmarked({"measured": "partial", "min_cards": None}, {"cards": 4}) is True),
        ("⑫ ★ 同源性：cards=2 而 gpus 只给 1 张 ⇒ **必须拒**（本次事故形态）",
         cards_procs_mismatch({"cards": 2, "gpus": [1, 7]}) is None
         or cards_procs_mismatch({"cards": 2, "gpus": [7]}) is not None),
        ("⑬ ★ 同源性：cards=2 但 nproc_per_node=1 ⇒ **必须拒**",
         cards_procs_mismatch({"cards": 2, "parallel": "nproc_per_node=1（1 卡）"}) is not None),
        ("⑭ 同源性：cards=2 + gpus 两张 + nproc=2 ⇒ 放行",
         cards_procs_mismatch({"cards": 2, "gpus": [1, 7], "parallel": "nproc_per_node=2（2 卡）"}) is None),
        ("⑮ ★ 卡号纪律：配置含 `gpus:` ⇒ **必须拒**（§12 一族）",
         config_pins_gpus("launch:\n  nproc_per_node: 2\n  gpus: [1, 7]\n") is True),
        ("⑯ 卡号纪律：只含 cards/nproc ⇒ 放行",
         config_pins_gpus("launch:\n  nproc_per_node: 2\n") is False),
        ("⑯b ★ 轴冲突：同档既 `blocked` 又「已验证不可用」⇒ **必须拒**",
         blocked_vs_unavailable_conflict({"availability": "blocked", "status": "已验证不可用"}) is True),
        ("⑯c 轴分离：`blocked` + 「尚未验证」⇒ 放行",
         blocked_vs_unavailable_conflict({"availability": "blocked", "status": "尚未验证"}) is False),
        ("⑰b ★ 族级相容：族 min_cards=4 而本档 cards=2 却未标 blocked ⇒ **必须拒**",
         family_under_min_unblocked({"family": "graspo_lora_27b", "family_min_cards": 4, "cards": 2}) is True),
        ("⑰c 族级相容：同档已标 blocked ⇒ 放行",
         family_under_min_unblocked({"family": "graspo_lora_27b", "family_min_cards": 4, "cards": 2, "availability": "blocked"}) is False),
        ("⑰d 族级相容：cards=4 ≥ min_cards ⇒ 放行",
         family_under_min_unblocked({"family": "graspo_lora_27b", "family_min_cards": 4, "cards": 4}) is False),
        ("⑯b2 ★ 轴冲突例外：blocked + 已验证不可用 + **带口径变更标记** ⇒ 放行（主席 2026-10-14）",
         blocked_vs_unavailable_conflict({"availability": "blocked", "status": "已验证不可用",
                                          "caliber_change": {"effective": "2026-10-14"}}) is False),
        ("⑯b3 ★ 轴冲突：blocked + 已验证不可用 + **无口径标记** ⇒ **必须拒**（防悄悄改写）",
         blocked_vs_unavailable_conflict({"availability": "blocked", "status": "已验证不可用"}) is True),
        ("⑰e ★ 反证拦截：标 blocked 却已有活证据（768/1280 进度）⇒ **必须拒**",
         blocked_contradicted_by_live({"availability": "blocked", "live_evidence": "★ 已到 768/1280"}) is True),
        ("⑰f 反证拦截：标 blocked 且无活证据 ⇒ 放行",
         blocked_contradicted_by_live({"availability": "blocked"}) is False),
        ("⑰g ★ 同配拦截：同族同后端 2 卡已实测 OOM，另一同配档未标 blocked ⇒ **必须拒**",
         combo_infeasible_still_runnable({"family": "g", "backend": "native", "cards": 2}, {("g", "native"): {2}}) is True),
        ("⑰h 同配拦截：同配档已标 blocked ⇒ 放行",
         combo_infeasible_still_runnable({"family": "g", "backend": "native", "cards": 2, "availability": "blocked"}, {("g", "native"): {2}}) is False),
        ("⑰i 同配拦截：**不同后端**同卡数 ⇒ 不套用（⛔ 不跨后端外推）",
         combo_infeasible_still_runnable({"family": "g", "backend": "ms-swift", "cards": 2}, {("g", "native"): {2}}) is False),
        ("⑰j ★ 引用纪律：结论引用了 T044 却**未同时引用其配置** ⇒ **必须拒**",
         citation_missing_config("★ 与 T044 同族 ⇒ blocked", "T044", None) is True),
        ("⑰k 引用纪律：引用时**同时给出配置字段** ⇒ 放行",
         citation_missing_config("★ 与 T044 同族 ⇒ blocked", "T044", {"backend": "ms-swift", "cards": 2}) is False),
        ("⑱ ②★ 清单一致性：已判定却标 ready ⇒ **必须判违规**（T028 事故）",
         status_ready_but_judged("已验证不可用", "ready") is True),
        ("⑱ ③ 清单一致性：已判定 ⇒ `judged` ⇒ 放行",
         status_ready_but_judged("已验证不可用", "judged") is False),
        ("⑱ ④ 清单一致性：未判定 + ready ⇒ 放行",
         status_ready_but_judged("尚未验证", "ready") is False),
        ("⑱ ⑥ ★ 在跑却标 ready ⇒ **必须判违规**（T017/T037 事故族）",
         running_but_ready({"live_evidence": "★ 在跑 504/1280", "_status": "ready"}) is True),
        ("⑱ ⑦ 在跑 ⇒ `running` ⇒ 放行",
         running_but_ready({"live_evidence": "★ 在跑", "_status": "running"}) is False),
        ("⑱ ⑧ ★ 有活证据 ⇒ 推导必得 `running`（不得 ready）",
         derive_tier_status("尚未验证", None, running=bool("x")) == "running"),
        ("⑱ ⑤ status 推导：blocked ⇒ `blocked`（优先于四态）",
         derive_tier_status("尚未验证", "blocked") == "blocked"),
        ("⑲ ① ★ final_formats：生效集合内缺该字段 ⇒ **必须拒**",
         final_formats_explicit({"tier": "T023"}) is True),
        ("⑲ ② ★ final_formats：生效集合内空数组 ⇒ **必须拒**",
         final_formats_explicit({"tier": "T023", "final_formats": []}) is True),
        ("⑲ ③ final_formats：生效集合内显式给出 ⇒ 放行",
         final_formats_explicit({"tier": "T023", "final_formats": ["merged-hf"]}) is False),
        ("⑲ ④ ★ final_formats：**作用范围外**的档不受约束（防误伤）",
         final_formats_explicit({"tier": "T099"}) is False),
        ("⑲ ⑤ ★ final_formats：非法取值（`hf`）⇒ **必须拒**",
         final_formats_valid(["hf"]) is True),
        ("⑲ ⑥ final_formats：合法取值 ⇒ 放行",
         final_formats_valid(["merged-hf"]) is False),
        ("⑳① ★ launch 键集合：含 `cards` ⇒ **必须拒**（本次 P0 事故形态）",
         launch_keys_invalid("launch:\n  cards: 2\n") is True),
        ("⑳② launch 键集合：`nproc_per_node` ⇒ 放行",
         launch_keys_invalid("launch:\n  nproc_per_node: 2\n") is False),
        ("⑳③ launch 键集合：`nproc_per_node` + `python` ⇒ 放行",
         launch_keys_invalid("launch:\n  nproc_per_node: 2\n  python: python3\n") is False),
        ("⑳④ ★ launch 键集合：`cards` + `nproc_per_node` ⇒ **必须拒**",
         launch_keys_invalid("launch:\n  cards: 1\n  nproc_per_node: 1\n") is True),
        ("⑳⑤ launch 键集合：无 `launch` 段 ⇒ 放行",
         launch_keys_invalid("backend: native\n") is False),
        ("⑩ ★ blocked 档缺 `blocked_reason` ⇒ **必须判缺**（fail-closed）",
         blocked_unmarked({"availability": "blocked"}) is True),
        ("⑪ blocked 档带理由 ⇒ 放行（但仍不得起跑）",
         blocked_unmarked({"availability": "blocked", "blocked_reason": "P-25 实测不可行"}) is False),
        ("⑨ 未定态：带 `cards_pending` ⇒ 放行",
         opd_pending_unmarked({"measured": "partial", "min_cards": None}, {"cards": 4, "cards_pending": True}) is False),
    ]
    ok = True
    for name, good in cases:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok = ok and good
    print(f"GEN-SELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _load_scalars(path):
    """极简 YAML 标量加载（覆盖本生成器输出的形态）：单引号=字面量（'' -> '），双引号=转义，plain=字面量。

    ★ 验收以**加载值**为准（⛔ 不看文件文本写了几个反斜杠）。
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+(.*\S)\s*$", line)
        if not m:
            continue
        v = m.group(1)
        if v.startswith("'") and v.endswith("'"):
            out.append(v[1:-1].replace("''", "'"))
        elif v.startswith('"') and v.endswith('"'):
            out.append(v[1:-1].encode().decode("unicode_escape"))
        else:
            out.append(v)
    return out


def uniform_map(data):
    """统一口径取值表：`key → value`。

    ★★ 2026-10-19（「脚本只负责把测试结果填进最后一章测试表」轮）：
      正式机读件 `docs/capability-matrix.data.json` 已**不再携带 `uniform`**
      （统一口径块改由人读件模板静态文字承载；数据件只剩逐档参数 + 结果）。
      ⇒ 本函数改为**先读数据、读不到则回落到 `UNIFORM_FALLBACK`**：
        · 内部权威件仍带 `uniform` 时，行为与原先**逐字相同**（⛔ 不放宽）；
        · 正式件（无 `uniform`）时**不再静默产出 `None`** —— 病史实况：本轮删除后实测
          `torch_dtype: None / seed: None / lora.r: None …`（生成器会把 20+ 个字段写成 None
          而不报错 ⇒ 隐患是"静默产出错配置"，比崩溃更坏）。
      ★ 回落表 = 本轮在盘人读件静态块里的**同类 9 项**（逐字取自删除前的数据件 `uniform.items`），
        仅覆盖生成 yaml 直接消费的键；其余键（`sequence_parallel_size` / `use_vllm` /
        `save_checkpoint_every_epoch` / `gpu_monitor.interval_sec` 等）仍**只**来自数据件
        ⇒ 用不带 `uniform` 的正式件生成时会在 `--verify-loadable` / 断言处暴露，⛔ 不伪造。
    """
    got = {it["key"]: it["value"] for it in (data.get("uniform") or {}).get("items", [])}
    return {**UNIFORM_FALLBACK, **got}


#: ★ 统一口径的**回落取值**（仅 `uniform_map` 用；见其 docstring 的处置说明）。
#:   ⛔ 不是新的真相源：数据件/内部权威件里有 `uniform` 时一律以数据为准（`got` 覆盖本表）。
#:   ★ 覆盖面 = `yaml_of()` **实际消费**的全部键（逐键实测，⛔ 不多抄）：
#:     上下文/精度/梯度检查点/单卡 batch/梯度累积/每 epoch 存 ckpt/LoRA 三参/奖励类型/
#:     序列并行尺寸/训练期 vLLM/GPU 巡检间隔 ⇒ 14 键。
#:   ★ 取值逐字取自删除前的在盘数据件 `uniform.items`（⛔ 未改任何数值）。
UNIFORM_FALLBACK = {
    "max_prompt_length": "8192",
    "torch_dtype": "bfloat16",
    "gradient_checkpointing": "true",
    "micro_batch_size": "1",
    "gradient_accumulation_micro_batches": "1",
    "save_checkpoint_every_epoch": "false",
    "lora_r": "128",
    "lora_alpha": "256",
    "lora_dropout": "0.05",
    "reward.kind": "graspo",
    "sequence_parallel_size": "1",
    "use_vllm": "false",
    "gpu_monitor.interval_sec": "2.0",
    "temperature": "0",
    "seed": "42",
}


def declared_switch_gaps(cfg_dir, declared_by_tier, tiers) -> list:
    """★★ 机检：「**声明所依赖的开关 ⊆ 该档实际配置**」（第 4 批）。
    逐档取清单 `feasibility.recipe.<backend>.*` 的**非几何**开关，核其是否真的写进了该档 yaml。
    ★ 与 `--check-dir` 走**同一条通路**（同一 `iter_declared`／同一 yaml 目录）；⛔ 不另立第二家。
    """
    base = Path(cfg_dir)
    gaps = []
    for tier in tiers:
        declared = declared_by_tier.get(tier) or {}
        p = base / f"{tier}.yaml"
        body = p.read_text(encoding="utf-8") if p.is_file() else ""
        for backend in sorted(declared):
            for k, v in iter_declared(declared, backend):
                if not re.search(rf"^\s*{re.escape(k)}\s*:", body, re.M):
                    gaps.append((tier, backend, k, v))
    return gaps


def report_declared_switches(cfg_dir, declared_by_tier, tiers, label="declared") -> int:
    gaps = declared_switch_gaps(cfg_dir, declared_by_tier, tiers)
    if gaps:
        print(f"DECLARED-SWITCH FAIL（{label}）：{len(gaps)} 处「已声明的前提未落进配置」")
        for tier, backend, k, v in gaps:
            print(f"    {tier}.{backend}.{k} = {v!r} ⇒ 该档 yaml 中缺失")
        return 1
    n = sum(len(iter_declared(declared_by_tier.get(t) or {}, be))
            for t in tiers for be in sorted(declared_by_tier.get(t) or {}))
    print(f"DECLARED-SWITCH OK（{label}）：{len(tiers)} 档、{n} 条已声明开关全部落进配置")
    return 0


#: 开关类声明的**几何键**（已由 native 段固定写出，不进通用规则）
DECLARED_GEOMETRY_KEYS = frozenset({"tp_size", "dp_size", "pp_size", "micro_batch_size", "cards"})

#: 跑次侧口径权威清单（提供 `feasibility.recipe.<backend>.*` ⇒ **已声明的前提必须落进配置**）
DEFAULT_DECLARED_MANIFEST = "tests/e2e/matrix54_manifest_source.json"


def _yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return "null" if v is None else str(v)


def iter_declared(declared, backend):
    """从清单 ``feasibility.recipe.<backend>`` 取**已声明的前提**（几何键除外），确定性排序。"""
    recipe = ((declared or {}).get(backend) or {}) if isinstance(declared, dict) else {}
    return [(k, recipe[k]) for k in sorted(recipe) if k not in DECLARED_GEOMETRY_KEYS]


def load_declared(path) -> dict:
    """读清单 ⇒ ``{tier: feasibility.recipe}``。★ fail-closed：清单不在 ⇒ 直接拒绝生成。"""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"FATAL: 声明清单不存在：{path}\n"
                         "  ⇒ 无法保证「清单里已声明的前提落进该档配置」；"
                         "⛔ 不静默降级（曾因丢键导致 T016/T034 真机 OOM）。")
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {(t.get("tier_id") or t.get("tier")): ((t.get("feasibility") or {}).get("recipe") or {})
            for t in (doc.get("tiers") or [])}


def premises_by_tier(doc) -> dict:
    """★ 已声明的**可行性前提**：``{tier: [(backend, key, value), ...]}``。

    判据（**机算**、⛔ 不手写档号）：
      ① 取自清单 ``feasibility.recipe.<backend>.*``，键**减去几何键**
         （`DECLARED_GEOMETRY_KEYS` —— 几何键已由「卡数 / 并列」列唯一决定）；
      ② 且该键**在该档 `feasibility.estimate` 正文里被点名**（``k`` 或 ``<backend>.k`` 形态）
         ⇒ 即「该档可行性所依赖的开关」，而非「该档恰好也写下的全部开关」。

    ★ 单一真相源：本函数是「前提」这一定义的**唯一实现**，并入端（矩阵侧并入器）
      与校验端（`verify_matrix` ④T）共同 import，⛔ 不各抄一份。
    """
    out = {}
    for t in ((doc or {}).get("tiers") or []):
        tier = t.get("tier_id") or t.get("tier")
        fe = t.get("feasibility") or {}
        estimate = str(fe.get("estimate") or "")
        recipe = fe.get("recipe") or {}
        items = []
        for backend in sorted(recipe):
            for key, val in iter_declared(recipe, backend):
                if key in estimate or f"{backend}.{key}" in estimate:
                    items.append((backend, key, val))
        if items:
            out[tier] = items
    return out


def load_premises(path) -> dict:
    """读清单 ⇒ `premises_by_tier()`。★ fail-closed：清单不在 ⇒ 直接拒绝（⛔ 不静默跳过检查）。"""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"FATAL: 声明清单不存在：{path}\n"
                         "  ⇒ 无法核对「清单里已声明的前提是否出现在对外人读面」；"
                         "⛔ 不静默降级（跳过即等于把透明性缺口藏起来）。")
    return premises_by_tier(json.loads(p.read_text(encoding="utf-8")))


def yaml_of(row, uni, model_root, data_root, lora_backends=None, teacher=None,
            declared=None):
    """渲染一个档位的 yaml 文本（确定性：固定键序、无时钟/环境）。"""
    tier = row["tier"]
    tuner = row.get("tuner_type")
    backend = row.get("backend")
    L = [HEADER]
    L.append(f"train_method: {row.get('train_method')}\n")
    L.append(f"backend: {'msswift' if backend == 'ms-swift' else backend}\n")
    L.append(f"tuner_type: {tuner}\n")

    L.append("model:\n")
    L.append(f"  model_path: {model_root.rstrip('/')}/{row.get('model_path')}\n")
    L.append(f"  torch_dtype: {uni.get('torch_dtype')}\n")
    L.append(f"  gradient_checkpointing: {uni.get('gradient_checkpointing')}\n")

    L.append("data:\n")
    L.append(f"  train_path: {data_root.rstrip('/')}/{row.get('train_path')}\n")
    L.append(f"  max_prompt_length: {uni.get('max_prompt_length')}\n")

    L.append("training:\n")
    L.append(f"  output_dir: /out/{tier}\n")
    L.append(f"  run_name: {tier}\n")
    L.append("  overwrite_output_dir: true\n")
    L.append(f"  seed: {uni.get('seed')}\n")
    L.append(f"  max_epochs: {row.get('max_epochs')}\n")
    L.append(f"  learning_rate: {row.get('lr')}\n")
    L.append(f"  gradient_accumulation_micro_batches: {uni.get('gradient_accumulation_micro_batches')}\n")
    L.append(f"  save_checkpoint_every_epoch: {uni.get('save_checkpoint_every_epoch')}\n")
    L.append(f"  save_steps: {row.get('save_steps')}\n")

    # ★★ 卡数/进程数**同源**：`nproc_per_node` 一律由 `cards` 推导（⛔ 不得各改一半；
    #    事故背景：set_cards.py 曾只改 gpus、未同步 nproc ⇒ dry-run 出 `gpus=1,7 nproc=1`）。
    _cards = row.get("cards") or row.get("cards_total")
    try:
        _cards = int(_cards)
    except Exception:
        _cards = None
    _ff = FINAL_FORMATS_BY_TIER.get(str(row.get("tier")))
    if _ff:
        L.append("export:\n")
        L.append("  final_formats: [" + ", ".join(f'"{x}"' for x in _ff) + "]\n")

    if _cards:
        L.append("launch:\n")
        # ★★ 只写镜像认识的键（LaunchConfig 允许集：determinism/master_addr/master_port/
        #    nnodes/node_rank/nproc_per_node/python）；⛔ 不写 `cards`（extra=forbid ⇒ 直接 SystemExit）。
        L.append(f"  nproc_per_node: {_cards}   # ★ 由 cards={_cards} 推导（§12 同源：卡数与进程数不得各改一半）\n")

    # ★ OPD：教师身份是**已 pin 的事实**（manifest `feasibility_model.opd_teacher`）⇒ 必须落进 yaml。
    #   ★ 配置值 = **容器内路径**（`/models/Qwen3.8-27B`）；**挂载源用宿主路径**（起跑侧决定）——两者别混。
    if str(row.get("train_method") or "").lower() == "opd" and teacher:
        L.append("distill:\n")
        _parts = teacher.get("path_parts") or []
        _cp = "/" + "/".join(str(x) for x in _parts if x) if _parts else ""
        L.append(f"  teacher_model_path: {_cp}\n")
        if teacher.get("offload"):
            L.append("  offload_teacher_model: true\n")
    if row.get("train_method") == "graspo" and uni.get("reward.kind"):
        L.append("reward:\n")
        L.append(f"  kind: {uni.get('reward.kind')}\n")

    if tuner == "lora":
        # ★ 按 backend 分流：native 用 preset 名；ms-swift 不认 preset 名（另一套语汇）⇒ 用同覆盖集的显式模块名列表
        lb = (lora_backends or {})
        if backend == "ms-swift":
            expr = (lb.get("msswift") or {}).get("expression") or []
        else:
            expr = (lb.get("native") or {}).get("expression") or []
        if not expr:
            expr = json.loads(row.get("target_modules") or "null") or []
        L.append("lora:\n")
        L.append(f"  r: {uni.get('lora_r')}\n")
        L.append(f"  alpha: {uni.get('lora_alpha')}\n")
        L.append(f"  dropout: {uni.get('lora_dropout')}\n")
        L.append("  target_modules:\n")
        for t in expr:
            ent = str(t)
            # 正则/含特殊字符的条目加单引号，避免 YAML 误读（内部无单引号）
            ent = f"'{ent}'" if any(c in ent for c in "^|*\\:") else ent
            L.append(f"    - {ent}\n")

    if backend == "native":
        L.append("native:\n")
        _emitted = set()
        for _k, _line in (("tp_size", f"  tp_size: {row.get('tp_size')}\n"),
                          ("dp_size", f"  dp_size: {row.get('dp_size')}\n"),
                          ("pp_size", f"  pp_size: {row.get('pp_size')}\n"),
                          ("micro_batch_size", f"  micro_batch_size: {uni.get('micro_batch_size')}\n")):
            L.append(_line); _emitted.add(_k)
        if row.get("allow_unverified_pp_rollout") not in (None, "", "—"):
            L.append(f"  allow_unverified_pp_rollout: {row.get('allow_unverified_pp_rollout')}\n")
            _emitted.add("allow_unverified_pp_rollout")
        # ★★ 通用规则（第 4 批）：**清单里已声明**、且本段尚未写出的开关 ⇒ **一律写出**。
        #    ⛔ 无按档手写表；⛔ 不改任何对外宣称参数（卡数/步数/epoch/方法/超参）；
        #    只把**已声明的前提**落进配置。由来：现行生成器重写时**丢了**旧世代
        #    （`samples/configs/matrix54-v2/`）已有的 `native.offload_optimizer_state`
        #    ⇒ T016/T034 真机 48.5 s／0 步／OOM。
        for _k, _v in iter_declared(declared, backend):
            if _k not in _emitted:
                L.append(f"  {_k}: {_yaml_scalar(_v)}\n"); _emitted.add(_k)
    elif backend == "ms-swift":
        nn = row.get("nnodes")
        nn = 1 if nn in (None, "", "—") else nn
        L.append("msswift:\n")
        L.append(f"  nproc_per_node: {row.get('nproc_per_node')}\n")
        L.append(f"  nnodes: {nn}\n")
        L.append(f"  per_device_train_batch_size: {uni.get('micro_batch_size')}\n")
        L.append(f"  sequence_parallel_size: {uni.get('sequence_parallel_size')}\n")
        L.append(f"  use_vllm: {uni.get('use_vllm')}\n")
        if row.get("deepspeed") not in (None, "", "—"):
            L.append(f"  deepspeed: {row.get('deepspeed')}\n")
    if uni.get("gpu_monitor.interval_sec"):
        L.append("gpu_monitor:\n")
        L.append("  output_dir: /out/gpu\n")
        L.append(f"  tag: {tier}\n")
        L.append(f"  interval_sec: {uni.get('gpu_monitor.interval_sec')}\n")
    return "".join(L)


def main():
    ap = argparse.ArgumentParser(description="矩阵数据 → 测试 yaml（确定性生成）")
    ap.add_argument("--data", default="docs/capability-matrix.data.json")
    ap.add_argument("--manifest", default=DEFAULT_DECLARED_MANIFEST,
                    help="跑次侧口径权威清单；其 feasibility.recipe.<backend>.* 即**已声明前提**，必须落进配置")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-root", default="/models")
    ap.add_argument("--data-root", default="/work/elam-v5")
    ap.add_argument("--only", default=None, help="逗号分隔档号；缺省=全部 applicable 档")
    ap.add_argument("--check-determinism", action="store_true", help="生成两次并逐字节比对（幂等自证）")
    ap.add_argument("--check-dir", default=None, help="与既有冻结 yaml 目录做差异归因")
    ap.add_argument("--check-declared", default=None, metavar="DIR",
                    help="★ 机检：清单里**已声明的前提**是否落进 DIR 下各档配置（缺失 ⇒ rc=1）")
    ap.add_argument("--self-test", action="store_true", help="OPD 教师断言自检（含缺值/空值负例）")
    ap.add_argument("--verify-loadable", action="store_true",
                    help="按【加载值】验收：正则项加载后须为单反斜杠、可编译、结构符合收紧版")
    a = ap.parse_args()

    if getattr(a, "self_test", False):
        return _gen_self_test()

    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    uni = uniform_map(data)
    teacher = (data.get("meta") or {}).get("distill_teacher")
    lora_backends = (data.get("meta") or {}).get("lora_target_backends") or {}
    only = set(x.strip() for x in a.only.split(",")) if a.only else None
    rows = [r for r in data["rows"] if r.get("applicable") and (only is None or r["tier"] in only)]
    declared_by_tier = load_declared(a.manifest)
    if a.check_declared:
        rc = report_declared_switches(a.check_declared, declared_by_tier,
                                      [r["tier"] for r in rows], label=a.check_declared)
        return rc

    # ★★ 第 6 批（①）：**被比对目录**（`--check-dir`）的声明开关校验必须**能失败**。
    #    病史：原实现把这道校验写在写盘后的 `--check-dir` 分支里、且**校验的是 `a.out_dir`**
    #    —— 即**刚刚生成的新目录**；而 `yaml_of()` 的通用规则保证新目录恒含全部已声明开关
    #    ⇒ 该分支**永不能失败**（「不能失败的检查」），真正被比对的 `a.check_dir` **从未被校验**。
    #    ★ 位置：**早于生成、且早于一切写盘侧守卫** ——「被比对目录是否缺声明开关」是**纯只读判定**，
    #    与本次能否生成无关。否则它会像写盘守卫一样被**既有红**挡在后面而永不可达
    #    （实测：全量 `--check-dir` 在 `blocked_contradicted_by_live` 的 T037/T043 处 rc=1，
    #      `--check-dir` 分支根本走不到）。
    #    ★ 两个目录都校验：本处查 `a.check_dir`（被比对者）；分支内仍查 `a.out_dir`（产出者）。
    if a.check_dir:
        rc_cmp = report_declared_switches(a.check_dir, declared_by_tier,
                                          [r["tier"] for r in rows], label=str(a.check_dir))
        if rc_cmp:
            return rc_cmp

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    for r in rows:
        text = yaml_of(r, uni, a.model_root, a.data_root, lora_backends, teacher,
                       declared_by_tier.get(r["tier"]))
        p = out / f"{r['tier']}.yaml"
        p.write_text(text, encoding="utf-8")
        written[r["tier"]] = p
        print(f"wrote {p}  max_epochs={r.get('max_epochs')}  derived_steps={r.get('derived_steps')}  save_steps={r.get('save_steps')}")

    if a.verify_loadable:
        bad = []
        for tier, pth in sorted(written.items()):
            for v in _load_scalars(pth):
                if "language_model" not in v and "visual" not in v:
                    continue
                if "\\\\" in v:                      # 加载值出现双反斜杠 ⇒ 过度转义
                    bad.append((tier, "加载值含双反斜杠（过度转义）"))
                try:
                    re.compile(v)
                except re.error as e:
                    bad.append((tier, f"正则不可编译: {e}"))
                if "visual\\.(blocks\\.\\d+" not in v and "language_model" in v:
                    bad.append((tier, "未采用收紧版视觉枚举（仍为过宽写法）"))
        print(f"LOADABLE {'PASS' if not bad else 'FAIL'}（{len(written)} 档；问题 {bad}）")
        if bad:
            return 1

    _pin = [t for t, pth in sorted(written.items()) if config_pins_gpus(Path(pth).read_text(encoding="utf-8"))]
    if _pin:
        print(f"FAIL-CLOSED：以下配置 pin 了物理卡号 `gpus`（⛔ 卡号不进配置，§12 一族）⇒ 拒绝产出：{_pin}")
        return 1

    _mm = [(r["tier"], why) for r in rows if (why := cards_procs_mismatch(r))]
    if _mm:
        print(f"FAIL-CLOSED：卡数与进程数/卡列表不同源（⛔ 不得只改一半）⇒ 拒绝产出：{_mm}")
        return 1

    _fals = {}
    for r in rows:
        if r.get("availability") == "blocked" and r.get("family"):
            try:
                _fals.setdefault((r["family"], r.get("backend")), set()).add(int(r.get("cards") or 0))
            except Exception:
                pass
    _combo = [r["tier"] for r in rows if combo_infeasible_still_runnable(r, _fals)]
    if _combo:
        print(f"FAIL-CLOSED：以下档与**同族同后端的实测证伪组合**冲突（同 cards 却未标 blocked）⇒ 拒绝产出：{_combo}")
        return 1

    _live = [(r["tier"], r.get("live_evidence")) for r in rows if blocked_contradicted_by_live(r)]
    if _live:
        print(f"FAIL-CLOSED：以下档标 blocked 却存在**矛盾的活证据** ⇒ 拒绝产出：{_live}")
        return 1

    _lk = [t for t, pth in sorted(written.items()) if launch_keys_invalid(Path(pth).read_text(encoding="utf-8"))]
    if _lk:
        print(f"FAIL-CLOSED：以下产物的 `launch` 段含镜像不认识的键（⛔ 起跑会 SystemExit）⇒ 拒绝产出：{_lk}")
        return 1

    _ff_bad = [(r["tier"], r.get("final_formats")) for r in rows
               if final_formats_explicit(r) or final_formats_valid(r.get("final_formats"))]
    if _ff_bad:
        print(f"FAIL-CLOSED：export.final_formats 缺失/为空或取值非法（生效集合 {sorted(FINAL_FORMATS_REQUIRED)}）⇒ 拒绝产出：{_ff_bad}")
        return 1

    _fam = [r["tier"] for r in rows if family_under_min_unblocked(r)]
    if _fam:
        print(f"FAIL-CLOSED：以下档与**族级实测**矛盾（cards < 族 min_cards 却未标 blocked）⇒ 拒绝产出：{_fam}")
        return 1

    _conf = [r["tier"] for r in rows if blocked_vs_unavailable_conflict(r)]
    if _conf:
        print(f"FAIL-CLOSED：以下档同时标 blocked 与「已验证不可用」（轴混用）⇒ 拒绝产出：{_conf}")
        return 1

    _bl = [r["tier"] for r in rows if blocked_unmarked(r)]
    if _bl:
        print(f"FAIL-CLOSED：以下档标 blocked 却无理由 ⇒ 拒绝产出：{_bl}")
        return 1

    _unmarked = [r["tier"] for r in rows
                 if str(r.get("train_method") or "").lower() == "opd" and opd_pending_unmarked(teacher, r)]
    if _unmarked:
        print(f"FAIL-CLOSED：以下 OPD 档处于「教师卡数未定」却缺 cards_pending 标记 ⇒ 拒绝产出：{_unmarked}")
        return 1

    _short = []
    for r in rows:
        if str(r.get("train_method") or "").lower() == "opd":
            why = opd_cards_shortfall(teacher, r.get("cards") or r.get("cards_total"))
            if why:
                _short.append((r["tier"], why))
    if _short:
        print(f"FAIL-CLOSED：OPD 卡数低于 manifest 声明 ⇒ 拒绝产出：{_short}")
        return 1

    _no_teacher = [t for t, pth in sorted(written.items()) if opd_teacher_missing(Path(pth).read_text(encoding="utf-8"))]
    if _no_teacher:
        print(f"FAIL-CLOSED：以下 OPD 档缺教师（teacher_model_path 空 且无 teacher_model_server）⇒ 拒绝产出：{_no_teacher}")
        return 1

    if a.check_determinism:
        bad = []
        for r in rows:
            # ★★ 第 6 批（⑤）：自证重渲染**必须与写盘循环（见上 :665）同参**。
            #    病史：本行原漏传 `declared_by_tier.get(r["tier"])` ⇒ 自证版本不含
            #    「已声明的前提」开关，而写盘版本含 ⇒ T016/T034 的两份重渲染**必不相等**，
            #    该「幂等自证」自第 4 批接线起**永远 FAIL**（「不能通过的检查」；
            #    与 `--check-dir` 那条「不能失败的检查」同族）。
            again = yaml_of(r, uni, a.model_root, a.data_root, lora_backends, teacher,
                            declared_by_tier.get(r["tier"]))
            if again != written[r["tier"]].read_text(encoding="utf-8"):
                bad.append(r["tier"])
        print(f"DETERMINISM {'PASS' if not bad else 'FAIL'}（{len(rows)} 档；不一致 {bad}）")
        if bad:
            return 1

    if a.check_dir:
        rc_declared = report_declared_switches(a.out_dir, declared_by_tier,
                                               [r["tier"] for r in rows], label=str(a.out_dir))
        base = Path(a.check_dir)
        print(f"\n=== 与既有冻结 yaml 的差异归因（{base}）===")
        for tier, p in sorted(written.items()):
            old = base / f"{tier}.yaml"
            if not old.exists():
                print(f"  {tier}: 旧目录无此档")
                continue
            d_old = flatten(old.read_text(encoding="utf-8"))
            d_new = flatten(p.read_text(encoding="utf-8"))
            keys = sorted(set(d_old) | set(d_new))
            rows_diff = [(k, d_old.get(k, "(缺)"), d_new.get(k, "(缺)")) for k in keys if d_old.get(k) != d_new.get(k)]
            print(f"  {tier}: 字段级差异 {len(rows_diff)} 处")
            for k, o, n in rows_diff:
                print(f"      {k}: {o!r} -> {n!r}   【{attribute(k)}】")
        return rc_declared
    return 0


def flatten(text):
    """把 yaml 文本摊平成 {key.path: value}（列表项合并为逗号串）；忽略注释与空行。"""
    d, stack = {}, []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if body.startswith("- "):
            key = ".".join(x[1] for x in stack)
            d[key] = (d.get(key, "") + "," + body[2:].strip()).strip(",")
            continue
        k, _, v = body.partition(":")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([x[1] for x in stack] + [k])
        if v.strip() == "":
            stack.append((indent, k))
        else:
            d[path] = v.strip()
    return d


#: 差异归因表（逐条对应已登记的口径变更）
ATTRIBUTION = [
    ("lora.target_modules", "LoRA 恢复口径：单 preset → 列表 [language_all_linear, vision_common]（358 实例；122 为签名/模式条目数）"),
    ("lora.target_preset", "LoRA 恢复口径：旧单 preset 已被 target_modules 列表取代"),
    ("lora.r", "LoRA 恢复口径：r 8 → 128"),
    ("lora.alpha", "LoRA 恢复口径：alpha 16 → 256"),
    ("lora.dropout", "LoRA 恢复口径：dropout 0.0 → 0.05"),
    ("training.learning_rate", "lr 口径（主席 2026-10-15）：一个方法一个值（SFT 2e-05(LoRA)/5e-06(全参)、"
                               "GRASPO(RL) 5e-06、CPT 5e-06、OPD 2e-05(LoRA)/5e-06(全参)）；⛔ 不按卡数缩放"),
    ("training.max_epochs", "epoch 口径（主席最新裁定，2026-10-15）：SFT=4（★ 由 2 改为 4）；CPT/OPD=2；GRASPO(RL)=8"),
    ("training.save_steps", "save_steps 重算：= 推导步数 + 1（取消人为步数上限）"),
    ("gpu_monitor", "生成器未覆盖的既有字段（gpu_monitor.*）：矩阵数据未收录 ⇒ 本次生成不含"),
    ("reward", "内容不变（kind=graspo），仅块序前移（生成器固定键序）"),
]


def attribute(key):
    for pref, why in ATTRIBUTION:
        if key == pref or key.startswith(pref + "."):
            return why
    return "未归因（需人工确认）"


if __name__ == "__main__":
    sys.exit(main())
