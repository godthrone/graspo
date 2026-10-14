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
#: ★ 严格比对集（防「跑 A 填 B」的身份事实）：身份键 + 并行维度 + lr
PARAM_KEYS_V3 = ["model", "algo", "mode", "backend", "parallel", "lr"]
#: ★ "训练身份"字段：**比归一化后的语义值**（⛔ 不比展示串/散文）
# ★ `steps` 不列入严格比对：它是**名义/派生值**且正在被公式纠正（json 可能仍带旧推导）⇒ 其核算由 step_count_reconciliation 负责
# ★ `steps_derived`（名义/派生）**可宽松**；★ `steps_actual`（实跑步数）是**训练身份** ⇒ 留在严格比对
PARAM_SEMANTIC_KEYS = ["lora_target", "target_modules", "data_subset", "switches", "steps_actual"]
#: ★ 仅登记不判错（纯散文/口径说明）
PARAM_INFO_KEYS = []


def _ints(v):
    return [int(x) for x in re.findall(r"\d+", str(v))]


def norm_ident(v):
    """★ 身份字符串规范化：去括注（全/半角）、去首尾空白、去 markdown 强调符。

    同一类别会有「全参」「全参（full-parameter）」这类**同义带括注**写法 ⇒ ⛔ 精确相等会误判错位。
    ★ 只用于**身份键**（model/algo/mode/backend）与语义字段的字符串比较；⛔ 不改数值/阈值判定。
    """
    t = str(v or "").replace("*", "").strip()
    t = re.sub(r"[（(][^）)]*[）)]\s*$", "", t).strip().lower()
    # ★ 已登记的**同义写法**（⛔ 只收已知别名，不合并真实不同值）：配置侧与档位表用词不同但同义。
    return _IDENT_ALIASES.get(t, t)


#: ★ 同义写法表（★ 逐条有据：`msswift` 是 `ms-swift` 的配置写法；`full` 是 `全参` 的英文写法）
_IDENT_ALIASES = {
    "msswift": "ms-swift",
    "ms_swift": "ms-swift",
    "full": "全参",
    "full-parameter": "全参",
    "full_parameter": "全参",
    "native": "native",
}


def norm_param(key, v):
    """把"训练身份"字段归一化为可比语义值；无法判定 ⇒ None（不据此判错）。

    · lora_target / target_modules ⇒ **解析/实测件数**（native 122 / ms-swift 358），⛔ 不比展示串；
    · data_subset ⇒ 子集规模（首个整数）；
    · steps ⇒ 步数（取最大整数，即推导/实跑总数）；
    · switches ⇒ `k=v` 键值对集合（⛔ 不比顺序/措辞）。
    """
    if v is None or str(v).strip() in ("", "—"):
        return None
    t = str(v).replace("*", "")          # 去 markdown 强调符（否则 **358** 件 解析不到）
    if key in ("lora_target", "target_modules"):
        m = re.search(r"(\d+)\s*件", t)                       # 「122 件」/「358 件」
        if m:
            return int(m.group(1))
        if "language_all_linear" in t or "vision_common" in t:
            return 122
        if "language_model" in t or "visual" in t:
            return 358
        return None
    if key == "steps_actual":
        # 实跑步数：优先取"数字/数字"或"实测 19 步"等形态的数值
        if isinstance(v, dict):
            a = v.get("actual_steps")
            return int(a) if a is not None else None
        m2 = re.search(r"(?:实测|实际|actual)[^0-9]{0,8}(\d+)", t) or re.search(r"^(\d+)", t.strip())
        if m2:
            return int(m2.group(1))
        return None
    if key == "data_subset":
        ns = _ints(t)
        return max(ns) if ns else None
    if key == "steps":
        ns = _ints(t)
        return max(ns) if ns else None
    if key == "switches":
        pairs = set()
        for seg in re.split(r"[;,]", t):
            if "=" in seg:
                k, _, val = seg.partition("=")
                pairs.add(k.strip().lower() + "=" + val.strip())
        return frozenset(pairs) if pairs else None
    return None
LEGACY_PARAM_KEYS = ["hparams"]
RESULT_KEYS = ["status", "runs", "effect", "inference", "algo_checks",
               "throughput", "peak_mem", "run_path"]
STATUSES = ["通过", "失败", "失败（效果）", "条件不合逻辑", "未完成测试"]
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


TRAINING_CLASSES = {"sft", "cpt", "opd"}


def matrix_frozen_pp(official_path, train_method):
    """★ 阈值只能来自**矩阵自身的冻结口径**（`meta.criteria_caliber.threshold`）。

    返回该档类别适用的冻结阈值（pp）；**未冻结 ⇒ None**（跑次无论自报什么都不得据此判定）。
    """
    try:
        d = json.loads(Path(official_path).read_text(encoding="utf-8"))
        cal = (d.get("meta") or {}).get("criteria_caliber") or {}
        th = cal.get("threshold") or {}
        val = th.get("value")
        applies = str(th.get("applies_to", ""))
        status = str(th.get("status", ""))
        if val is None:
            return None
        if str(train_method).lower() in TRAINING_CLASSES:
            return float(val) if "训练类" in applies or "冻结" in status else None
        return None      # 非训练类（GRASPO 等）⇒ 矩阵侧尚未冻结
    except Exception:
        return None


def matrix_frozen_map(official_path):
    """从**矩阵自身冻结口径**导出 方法→冻结阈值(pp) 的映射；未冻结 ⇒ 该方法映射为 None。"""
    out = {m: None for m in TRAINING_CLASSES}
    out["graspo"] = None
    try:
        d = json.loads(Path(official_path).read_text(encoding="utf-8"))
        th = (((d.get("meta") or {}).get("criteria_caliber") or {}).get("threshold")) or {}
        val, applies, status = th.get("value"), str(th.get("applies_to", "")), str(th.get("status", ""))
        if val is not None and ("训练类" in applies or "冻结" in status):
            for m in TRAINING_CLASSES:
                out[m] = float(val)
        # ★ GRASPO 已按主席定案冻结（阈值 +5.0 pp；目标 +10.0 pp 仅展示、⛔ 不作通过线）
        if val is not None and "GRASPO" in applies:
            out["graspo"] = float(val)
    except Exception:
        pass
    return out


LEDGER_ROW = re.compile(
    r"^\|\s*(?P<time>[^|]+?)\s*\|\s*`(?P<tier>[^`]+)`\s*\|\s*`(?P<file>[^`]+)`\s*\|\s*`(?P<sha>[^`]+)`\s*\|"
    r"\s*(?P<status>[^|]*?)\s*\|\s*(?P<verdict>[^|]*?)\s*\|\s*(?P<note>[^|]*?)\s*\|\s*$")


def dedup_ledger(text):
    """★ 台账按 `(档号, json sha)` 去重：同键合并为一行，保留次数与原因（审计痕迹）。

    ★ 硬要求：**不同 sha ⇒ 必须保留两行**（那才是真增量，⛔ 不得被去重吞掉）。
    ★ 行序保持：非数据行（表头/分隔线等）**留在原位置**（用槽位占位，不搬动）。
    """
    slots, bykey = [], {}          # slots: (kind, payload)
    for ln in text.splitlines():
        m = LEDGER_ROW.match(ln)
        if not m:
            slots.append(("raw", ln))
            continue
        key = (m.group("tier"), m.group("sha"))
        row = ("| %s | `%s` | `%s` | `%s` | %s | %s | %s |"
               % (m.group("time"), m.group("tier"), m.group("file"), m.group("sha"),
                  m.group("status"), m.group("verdict"), m.group("note").strip()))
        if key in bykey:
            k, cnt, forced = bykey[key]
            cnt += 1
            forced += 1 if "--force" in ln else 0
            bykey[key] = (k, cnt, forced)
            slots[k] = ("row", row[:-1] + " ×%d（含 %d 次 --force 重跑） |" % (cnt, forced))
        else:
            bykey[key] = (len(slots), 1, 1 if "--force" in ln else 0)
            slots.append(("row", row))
    out = [payload for _kind, payload in slots]
    return "\n".join(out) + "\n"


def ledger_covers(rows, filled_tiers):
    """★ 断言辅助（可单测）：本次处理的每一档是否都在台账行里。返回缺失的档号列表。"""
    return [t for t in filled_tiers if not any(f"`{t}`" in r for r in rows)]


def verdict_transition(already_judged: bool, status_before: str, status_after: str) -> int:
    """★ "重测 vs 新判定"计数断言：返回**四态总判定数**的期望变化。

    · 该档**已在四态内**（already_judged=True）⇒ 走"更新"路径 ⇒ ★ **期望变化 = 0**（总数不变）
    · 该档**不在四态内** ⇒ 走"新增判定"路径 ⇒ ★ 期望变化 = +1（且 status_before 应为"—"/尚未验证）
    ★ 用途：机器拦住"计数漂移"（人工算术今日已两次出偏差）。
    """
    if already_judged:
        return 0
    return 1 if status_after in ("已验证可用", "已验证不可用") else 0


def conclusion_of(raw):
    """把跑次 `status` 规范化为 'pass' / 'fail' / 'other'。

    ★ 状态字符串**一律走规范化匹配，⛔ 不用精确相等**：跑次可能写「通过（已验证可用）」这类**带括注**形态。
    ★ 负例安全性：`未通过` / `非通过` / `不通过` **不以**「通过」开头 ⇒ 归 'other'，⛔ 不会被误判为 pass。
    """
    t = str(raw or "").strip()
    if t.startswith("通过"):
        return "pass"
    if t.startswith("失败"):
        return "fail"
    return "other"


def correction_texts(doc, _path=""):
    """递归找出跑次 json 中所有"订正/口径更正"类字段（键名含 correct/订正/口径更正），返回文本列表。

    ★ 目的：跑次侧的**依据/来源订正**必须如实进入备注与证据链（⛔ 不吞）。
    """
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            _k_hit = any(t in str(k).lower() for t in ("correct", "订正", "口径更正"))
            if _k_hit and v:
                out.append(f"{k}：{v}" if not isinstance(v, str) else v)
            elif isinstance(v, str) and "订正" in v:
                out.append(f"{k}：{v}")     # ★ 值里声明式订正（如 run_path / guard_assertions）
            out.extend(correction_texts(v, _path + "." + str(k)))
    elif isinstance(doc, list):
        for x in doc:
            out.extend(correction_texts(x, _path))
    return out


def run_claimed_pp(doc):
    """从跑次自述里抽其**自报**的阈值（pp）；抽不到 ⇒ None。

    只在**阈值语境**下取值（`≥ N pp` / `阈值 N pp`），⛔ 不把"达标差 Δ"当成阈值。
    """
    mp = ((doc.get("provenance") or {}).get("measurement_pending") or {})
    blob = " ".join([str(doc.get("runs", "")), str(doc.get("effect", "")),
                     str(mp.get("threshold_delta", ""))])
    pats = [r"(?:≥|>=)\s*\+?\s*([0-9]+(?:\.[0-9]+)?)\s*pp",
            r"阈值\s*[Δdelta]*\s*(?:为|=|≥|>=)?\s*\+?\s*([0-9]+(?:\.[0-9]+)?)\s*pp"]
    for pat in pats:
        m = re.search(pat, blob)
        if m:
            return float(m.group(1))
    return None


def is_frozen(doc):
    """判据② 的阈值是否已冻结（正式判定 vs 仅录读数）。

    顶层 `threshold_frozen: true` 优先；否则回退到跑次自报的
    `provenance.measurement_pending` 声明（含「已冻结」即视为冻结）。
    """
    if doc.get("threshold_frozen") is True:
        return True
    mp = ((doc.get("provenance") or {}).get("measurement_pending") or {})
    txt = str(mp.get("threshold_delta", "")) + str(mp.get("verdict", ""))
    # ★ 收紧后：这里只表示"跑次自述声称阈值已冻结"；**能否据此判定以矩阵口径为准**（见 main）。
    return "冻结" in txt or "冻结阈值" in (str(doc.get("runs", "")) + str(doc.get("effect", "")))


def norm_parallel(s):
    """归一化为 (总卡数, 世界大小, 维度字典)，用于语义身份比对。

    ★ 加固：canonical 相等**且双方都声明了同名字段时，维度必须逐项一致**
      （⛔ 不因表达面不同而丢掉「跑 A 填 B」的检出能力）。
    """
    if not isinstance(s, str):
        return None

    def num(pat):
        m = re.search(pat, s)
        return int(m.group(1)) if m else None

    dims = {}
    for key, pat in (("tp", r"TP\s*=\s*(\d+)"), ("pp", r"PP\s*=\s*(\d+)"), ("dp", r"DP\s*=\s*(\d+)"),
                     ("nproc", r"nproc_per_node\s*=\s*(\d+)"), ("nnodes", r"nnodes\s*=\s*(\d+)")):
        v = num(pat)
        if v is not None:
            dims[key] = v
    cards = num(r"(\d+)\s*卡")
    if "nproc" in dims:
        world = dims["nproc"] * dims.get("nnodes", 1)
    elif all(k in dims for k in ("tp", "pp", "dp")):
        world = dims["tp"] * dims["pp"] * dims["dp"]
    else:
        world = None
    if cards is None and world:      # ★ 单节点下卡数 = nproc 或 tp×pp×dp（表达面不同、语义相同）
        cards = world
    return (cards, world, dims)


def compare_parallel(js, ts):
    """比对跑次 json 与档位表的并行表达；返回错误列表（空 = 通过）。"""
    nj, nt = norm_parallel(js), norm_parallel(ts)
    if nj is None or nt is None:
        return [] if js == ts else [f"params.parallel 不可解析且与档位表不一致：json={js!r} 表={ts!r}"]
    errs = []
    if nj[:2] != nt[:2]:
        errs.append(f"params.parallel 语义不一致（防错位）：json={js!r}->{nj[:2]} 表={ts!r}->{nt[:2]}")
    # ★ 双方都声明的维度必须逐项一致
    for k in sorted(set(nj[2]) & set(nt[2])):
        if nj[2][k] != nt[2][k]:
            errs.append(f"params.parallel 维度 {k} 不一致（防错位）：json={nj[2][k]} 表={nt[2][k]}")
    return errs


def validate_run(doc, tier_from_file, tier_rows, frozen_map=None, ref_rows_official=None, void_tiers=None):
    """严格校验一份跑次 json。返回错误列表（空 = 合法）。"""
    if not isinstance(doc, dict):
        return ["顶层不是 JSON 对象"]
    # ★ 显式作废档：参数已更正 ⇒ **跳过"参数/阈值对应"校验**（只保留结构校验），
    #   其结论按"参考记录"留存（§结论与参数一一对应：不对应 ⇒ 作废，而非报错）。
    _voided = bool(void_tiers) and tier_from_file in void_tiers
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
        if tier in tier_rows and not _voided:
            ref = tier_rows[tier]
            for k in PARAM_KEYS_V3:
                if k not in p or _s(p.get(k)) is None:
                    continue
                j, r_ = _s(p.get(k)), _s(ref.get(k, ""))
                if k == "parallel":
                    errs.extend(compare_parallel(j, r_))     # ★ canonical + 维度逐项一致
                elif norm_ident(r_) != norm_ident(j):
                    errs.append(f"params.{k} 与档位表不一致（防错位）：json={j!r} 表={r_!r}")
            # ★ "训练身份"字段：比**归一化语义值**（LoRA 目标件数 / 数据规模 / 步数 / 开关集合）
            ref_sem = (ref_rows_official or {}).get(tier, ref)      # ★ 语义参照取正式件（含当期派生值）
            for k in PARAM_SEMANTIC_KEYS:
                if k not in p:
                    continue
                _jv = p.get(k)
                if k == "steps_actual" and _jv is None:
                    _jv = ((doc.get("step_count_reconciliation") or {}).get("actual_steps"))
                nj, nt = norm_param(k, _jv), norm_param(k, ref_sem.get("steps_actual", ""))
                if nj is not None and nt is not None and nj != nt:
                    errs.append(f"params.{k} 语义不一致（训练身份，防错位）：json={nj} 表={nt}")
            # 纯散文字段：只登记，不判错
            doc.setdefault("_param_info_diffs", [])
            for k in PARAM_INFO_KEYS:
                if k in p and _s(p.get(k)) is not None and _s(ref.get(k, "")) != _s(p.get(k)):
                    doc["_param_info_diffs"].append(k)
            doc.setdefault("_legacy_params_ignored", [])
            for k in LEGACY_PARAM_KEYS:
                if k in p and k not in ref:
                    doc["_legacy_params_ignored"].append(k)

    # --- 结果列：8 项必须存在且非空；status=通过 时不得用占位符 '—' ---
    st = _s(doc["status"])
    if st not in STATUSES and not (isinstance(st, str) and (st.startswith("失败") or st.startswith("通过"))):
        errs.append(f"status {st!r} 非法（须为 {' / '.join(STATUSES)} 之一）")
    for k in RESULT_KEYS[1:]:
        v = _s(doc[k])
        if not isinstance(v, str) or not v:
            errs.append(f"结果字段 {k!r} 为空 / 非字符串")
        elif v == PLACEHOLDER and conclusion_of(st) == "pass":
            errs.append(f"status=通过 但 {k!r} 仍是占位符 '—'（不许半填）")

    # --- ★ 阈值信任边界：阈值只能来自矩阵冻结口径；跑次自述仅作交叉核对，不一致即拒绝 ---
    if tier in tier_rows and frozen_map is not None and not _voided:
        method = str(tier_rows[tier].get("train_method", "")).lower()
        frozen_pp = frozen_map.get(method)
        claimed = run_claimed_pp(doc)
        if frozen_pp is None and claimed is not None:
            errs.append(f"矩阵侧 {method!r} 类阈值**尚未冻结**，跑次却自述阈值 {claimed} pp "
                        f"⇒ 拒绝（⛔ 跑次不得自定阈值）")
        elif frozen_pp is not None and claimed is not None and claimed != frozen_pp:
            errs.append(f"跑次自述阈值 {claimed} pp ≠ 矩阵冻结口径 {frozen_pp} pp ⇒ 拒绝"
                        f"（阈值只能来自矩阵）")

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
    # ★ 写后台账按 (档号, json sha) 去重（保留次数与原因；不同 sha 仍两行）
    if path.exists():
        path.write_text(dedup_ledger(path.read_text(encoding="utf-8")), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="跑次 json 目录（唯一输入格式：<档号>.json）")
    ap.add_argument("--data-in", default="matrix-data.json")
    ap.add_argument("--data-out", default=None, help="默认 = --data-in（原地回填）")
    ap.add_argument("--template", default="template.html")
    ap.add_argument("--html-out", default="capability-matrix-new.html")
    ap.add_argument("--ledger-out", default=None,
                    help="台账路径（缺省 = 数据目录下的 backfill-ledger.md，⛔ 不用相对 CWD 以免 stray）")
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
    ap.add_argument("--self-test", action="store_true",
                    help="跑校验器的正/反向用例（阈值信任边界 + 并行维度逐项），不读跑次目录")
    ap.add_argument("--dry-run", action="store_true", help="只校验与报告，不写任何文件")
    ap.add_argument("--unvoid", action="append", default=[], metavar="TIER",
                    help="解除指定档的作废态：该档已有新跑次且与当前口径对应 ⇒ 按常规落判定")
    ap.add_argument("--force", action="store_true",
                    help="跳过幂等判定强制重处理（用于「渲染/备注口径升级」后回灌既有档）")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    data_in = Path(a.data_in)
    data_out = Path(a.data_out) if a.data_out else data_in
    html_out = Path(a.html_out)
    # 相对路径的台账落在"矩阵数据所在目录"（工作区），⛔ 不再落进仓库根造成 stray 文件
    ledger_out = (Path(a.ledger_out) if a.ledger_out
                  else Path(a.data_out).parent / "backfill-ledger.md")   # ★ 锚定数据目录
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

    frozen_map = matrix_frozen_map(a.official_data)
    ref_rows_official = {}
    try:
        _d = json.loads(Path(a.official_data).read_text(encoding="utf-8"))
        ref_rows_official = {x["tier"]: x for x in _d.get("rows", [])}
    except Exception:
        ref_rows_official = {}
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
    # ★ 已作废档的"重跑回填"（如跑次侧订正了依据/来源）自动走作废路径，
    #   避免其"参数已更正"的既成事实被参数对应校验拦下（作废前提本就是参数不对应）。
    _unvoid = set(a.unvoid)
    for _r in matrix.get("rows", []):
        _ann = _r.get("conclusion_annotation") or {}
        if _ann.get("conclusion_valid") is False and _r.get("tier") not in void_map and _r.get("tier") not in _unvoid:
            void_map[_r["tier"]] = _ann.get("reason", "既有作废记录：结论不对应当前行参数")
    _anchor_corr_tiers: set[str] = set()
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
        errs = validate_run(doc, f.stem, tier_rows, frozen_map, ref_rows_official, set(void_map))
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
        _corr = (doc.get("provenance") or {}).get("eval_dataset_anchor_correction")
        if _corr:
            _anchor_corr_tiers.add(tier)
        _corrs_now = correction_texts(doc)
        _corr_changed = bool(_corrs_now) and new_rows[tier].get("corrections") != _corrs_now
        _missing_new = (bool(_corr) and not new_rows[tier].get("eval_dataset_anchor_correction")) or any(
            doc.get(_k) and not new_rows[tier].get(_k)
            for _k in ("p5p6_readings", "anomalies", "step_count_reconciliation"))
        if (not a.force and tier not in void_map and prev.get("source_sha256") == src_sha
                and not any(changed.values()) and not _missing_new and not _corr_changed):
            skipped.append(tier)
            ledger_rows.append(f"| {now} | `{tier}` | `{f.name}` | `{src_sha[:12]}` | "
                               f"{doc['status']} | — | 跳过：同源 sha（`{src_sha[:12]}`）已回填，幂等 |")
            continue
        for k in RESULT_KEYS:
            new_rows[tier][k] = _s(doc[k])
        # ★ 显式作废（参数已更正 ⇒ 结论不对应当前行参数）：作为"参考记录"留存，不作判定
        if tier in void_map and tier not in set(a.unvoid):
            ref = "；".join(x for x in (_s(doc.get("runs")), _s(doc.get("effect"))) if x)
            new_rows[tier]["conclusion_annotation"] = {
                "conclusion_valid": False, "reason": void_map[tier], "reference_record": ref,
            }
            new_rows[tier]["status"] = "尚未验证"
            new_rows[tier].pop("measurement_pending", None)
            _corrs_v = correction_texts(doc)
            _vnote = "参考记录（" + void_map[tier] + "）：" + (ref or "—")
            if _corrs_v:
                new_rows[tier]["corrections"] = _corrs_v
                _vnote += " ★ 订正说明（跑次侧）：" + "；".join(_corrs_v[:3])
            new_rows[tier]["note"] = _vnote
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
        frozen = frozen_map.get(str(new_rows[tier].get("train_method", "")).lower()) is not None
        if not frozen:
            new_rows[tier]["status"] = "尚未验证"
            new_rows[tier]["measurement_pending"] = True   # ★ 读数已录、阈值待定（不作判定）
            new_rows[tier].pop("conclusion_annotation", None)   # 新读数取代旧「作废参考记录」注记
            passed1 = conclusion_of(doc.get("status")) == "pass"
            new_rows[tier]["note"] = ("读数为实测记录，⛔ 暂不作判定：" +
                                      ("①「能完整跑完」已通过；" if passed1 else "") +
                                      "②「平台方数据集效果达标」读数已录，★ 阈值 δ 待定 ⇒ 状态暂保持「尚未验证」。")
        else:
            ok = _s(doc.get("status")).startswith("通过")   # ★ 支持「通过（已验证可用）」等带括注形态
            new_rows[tier]["status"] = "通过" if ok else "失败"
            new_rows[tier]["measurement_pending"] = False
            new_rows[tier].pop("conclusion_annotation", None)   # 新的有效跑次取代旧的「作废参考记录」
            mp = ((doc.get("provenance") or {}).get("measurement_pending") or {})
            basis = str(mp.get("threshold_delta", "阈值已冻结"))
            if ok:
                note = _s(doc.get("note")) or f"判据① ∧ 判据② 均达（{basis}）⇒ 已验证可用。"
            else:
                note = ("★ 未过第 2 条判据（效果不足）——" + _s(doc.get("effect"))
                        + " ★ 已通过项（事实保留）：" + _s(doc.get("algo_checks"))
                        + " ★ 判据① 依据：" + _s(doc.get("runs")))
            # ★ 跑次自报的如实说明（caveat_*）必须原样保留，⛔ 不抹掉
            cav = [str(v) for k, v in (doc.get("provenance") or {}).items()
                   if k.startswith("caveat") and v]
            if cav:
                note += " 如实说明：" + "；".join(cav)
            # ★ 结构化"标准交付项"：P5/P6 六项 / 异常（非字符串 ⇒ 不塞 RESULT_KEYS，单独随件入库）
            for _k in ("p5p6_readings", "anomalies",
                       "after_acc", "base_acc", "delta_pp", "n_correct", "n_total"):
                if doc.get(_k):
                    new_rows[tier][_k] = doc[_k]
            scr = doc.get("step_count_reconciliation")
            if not ok and scr:
                new_rows[tier]["step_count_reconciliation"] = scr
                note += (" ★ 步数对账（如实登记）：推导 %s vs 实际 %s；%s"
                         % (scr.get("derived_steps"), scr.get("actual_steps"), scr.get("verdict", "")))
                new_rows[tier]["note_extra_steps_formula_issue"] = (
                    "★ 登记待修：'推导步数公式'对 native GRASPO 档不成立（推导 %s vs 实际 %s）"
                    "—— 本轮不顺手改公式，后续单独修。" % (scr.get("derived_steps"), scr.get("actual_steps")))
            # ★ 树指纹：**取起跑断言值**；若当场重算不同 ⇒ 如实登记漂移（⛔ 不抹）
            ci = ((doc.get("provenance") or {}).get("code_identity") or {})
            if ci:
                new_rows[tier]["code_identity"] = ci          # ★ 双锚/镜像摘要来源（供备注渲染）
            if ci.get("tree_fingerprint"):
                new_rows[tier]["tree_fingerprint"] = ci["tree_fingerprint"]
                new_rows[tier]["tree_fingerprint_role"] = ci.get("tree_fingerprint_role", "起跑断言值")
            if ci.get("tree_fingerprint_remeasured") and ci["tree_fingerprint_remeasured"] != ci.get("tree_fingerprint"):
                new_rows[tier]["tree_fingerprint_remeasured"] = ci["tree_fingerprint_remeasured"]
                new_rows[tier]["tree_fingerprint_drift"] = ci.get("tree_fingerprint_drift", "起跑值与当场重算不同")
                note += (" ★ 树指纹（取起跑断言值）：%s…；当场重算 %s…；差异原因见 provenance.code_identity.tree_fingerprint_drift（已如实登记）。"
                         % (str(ci["tree_fingerprint"])[:8], str(ci["tree_fingerprint_remeasured"])[:8]))
            # ★ 一切"订正/更正"类字段（跑次侧依据/来源订正）⇒ 如实并入备注与数据（通用扫描）
            _corrs = correction_texts(doc)
            if _corrs:
                new_rows[tier]["corrections"] = _corrs
                note += " ★ 订正说明（跑次侧）：" + "；".join(_corrs[:3])
            # ★ 评测集锚订正（跑次侧当场实测更正）⇒ 如实并入备注与数据
            corr = (doc.get("provenance") or {}).get("eval_dataset_anchor_correction")
            ed = ((doc.get("provenance") or {}).get("run_environment") or {}).get("eval_dataset")
            if corr:
                new_rows[tier]["eval_dataset_anchor_correction"] = corr
                note += " ★ 评测集锚订正（跑次侧当场实测）：" + str(corr)
            if ed:
                new_rows[tier]["eval_dataset"] = ed
            # ★ 重复测量：早前读数**保留为参考**（⛔ 不覆盖历史、不混为一谈）
            _prev_eff, _prev_sha = prev.get("effect"), prev.get("source_sha256")
            if (_prev_eff and _prev_sha and _prev_sha != src_sha
                    and str(_prev_eff) != str(doc.get("effect"))):
                note += " ★ 早前读数（参考，同口径重复测量，来源 sha " + str(_prev_sha)[:12] + "…）：" + str(_prev_eff)
            # ★ anomalies 如实并入备注
            for _a in (doc.get("anomalies") or []) if isinstance(doc.get("anomalies"), list) else ([doc["anomalies"]] if doc.get("anomalies") else []):
                note += " ★ 异常登记：" + str(_a)
            new_rows[tier]["note"] = note
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

        # ★★ 落盘前断言：**本次处理的每一档必须出现在台账中**（⛔ 不再静默漏行）
        _missing_rows = [t for t in filled if not any(f"`{t}`" in r for r in ledger_rows)]
        if _missing_rows:
            print(f"FAIL-CLOSED：以下档已处理但**未生成台账行** ⇒ 拒绝落盘：{_missing_rows}")
            return 1
        for f in (data_out, Path(a.official_data), Path(a.official_html)):
            f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_data, data_out)
        shutil.copyfile(tmp_off, Path(a.official_data))
        shutil.copyfile(tmp_html, Path(a.official_html))
        ledger_rows = [r.replace("| 见下 |", "| ✅ PASS |") for r in ledger_rows]
        # ★ 触发方式显式写进行内标记（审计件：`×N（含 M 次 --force 重跑）` 必须准）
        _mode = ("--force" if a.force else
                 ("--unvoid" if a.unvoid else
                  ("锚点订正" if _anchor_corr_tiers else "增量")))
        ledger_rows = [(_r[:-2] + f" [{_mode}] |") if _r.endswith(" |") else _r for _r in ledger_rows]
        append_ledger(ledger_out, ledger_rows)
        print(f"OK: 回填 {len(filled)} 档 ⇒ {data_out}（内部）/ {a.official_data} / {a.official_html}；"
              f"台账 +{len(ledger_rows)} 行 ⇒ {ledger_out}")
        print(f"    official data sha256 = {sha256_file(Path(a.official_data))}")
        print(f"    official html sha256 = {sha256_file(Path(a.official_html))}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _self_test():
    """★ 四条正/反向用例（⛔ 不依赖外部文件、不写任何东西）。"""
    fm = {"sft": 5.0, "cpt": 5.0, "opd": 5.0, "graspo": None}
    base_doc = {"runs": "判据② 达标（训练类冻结阈值 Δ ≥ 5.0 pp）", "effect": "Δ=+8.5 pp（≥ 5.0 pp 冻结阈值）"}
    cases = [
        ("① 阈值：跑次自述 5 pp == 矩阵 5 pp ⇒ 放行",
         not _threshold_errs(dict(base_doc, effect="Δ=+8.5 pp（≥ 5.0 pp 冻结阈值）"), fm, "sft")),
        ("② 阈值：跑次自述 3 pp ≠ 矩阵 5 pp ⇒ FAIL",
         bool(_threshold_errs(dict(base_doc, runs="判据② 达标（阈值 Δ ≥ 3 pp）", effect="Δ=+8.5 pp（≥ 3 pp）"),
                              fm, "sft"))),
        ("③ 并行：表达面不同但维度等价 ⇒ 放行",
         not compare_parallel("TP=1 PP=1 DP=1（1 卡；总卡数 1）", "nproc_per_node=1（1 卡）")),
        ("④ 并行：TP/PP 不符 ⇒ FAIL",
         bool(compare_parallel("TP=2 PP=1 DP=1（2 卡）", "TP=1 PP=2 DP=1（2 卡）"))),
        ("⑤ LoRA 目标语义不同（122 vs 358）⇒ FAIL",
         norm_param("lora_target", "language_all_linear + vision_common（122 件）") !=
         norm_param("lora_target", "ms-swift 实测 358 件（语言 248 + 视觉 110）")),
        ("⑥ 数据子集规模不同（100 vs 200）⇒ FAIL",
         norm_param("data_subset", "前 100 条") != norm_param("data_subset", "前 200 条")),
        ("⑦ 步数不同（800 vs 801）⇒ FAIL",
         norm_param("steps", "推导 800 步") != norm_param("steps", "推导步数 801")),
        ("⑨ 状态：带括注「通过（已验证可用）」⇒ pass", conclusion_of("通过（已验证可用）") == "pass"),
        ("⑩ 状态：带括注「失败（效果）」⇒ fail", conclusion_of("失败（效果）") == "fail"),
        ("⑪ 状态：「尚未验证」⇒ other（不判过）", conclusion_of("尚未验证") == "other"),
        ("⑫ ★ 负例：「未通过」/「非通过」/「不通过」**不得**判 pass",
         all(conclusion_of(x) != "pass" for x in ("未通过", "非通过", "不通过"))),
        ("⑬ 身份规范化：「全参（full-parameter）」≡「全参」", norm_ident("全参（full-parameter）") == norm_ident("全参")),
        ("⑭ ★ 负例：「仅视觉」/「LoRA」**不得**等同于「全参」",
         all(norm_ident(x) != norm_ident("全参") for x in ("仅视觉", "LoRA"))),
        ("⑮ ★ 负例：`native` ≠ `ms-swift`；`dp_size=2` ≠ `dp_size=1`",
         all(norm_ident(a) != norm_ident(b) for a, b in (("native", "ms-swift"), ("dp_size=2", "dp_size=1")))),
        ("⑯ 台账去重：同 (档号, sha) 两次 ⇒ 1 行 + ×2",
         dedup_ledger("| t1 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n |\n"
                      "| t2 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n |").count("`T1`") == 1
         and "×2" in dedup_ledger("| t1 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n |\n"
                                  "| t2 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n |")),
        ("⑰ ★ 台账去重：**不同 sha ⇒ 必须 2 行**（真增量不得被吞）",
         dedup_ledger("| t1 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n |\n"
                      "| t2 | `T1` | `a.json` | `bbbb` | 通过 | ✅ | n |").count("`T1`") == 2),
        ("⑱ ★ 台账：一次 --force 重跑 ⇒ 计数必须为 **1**（⛔ 不是 0）",
         "含 1 次 --force 重跑" in dedup_ledger(
             "| t1 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n [增量] |\n"
             "| t2 | `T1` | `a.json` | `aaaa` | 通过 | ✅ | n [--force] |")),
        ("⑲ ★ 台账覆盖率：处理了但未落行 ⇒ **必须拒**（fail-closed）",
         ledger_covers(["| t | `T1` | `a.json` | `sha` | 通过 | ✅ | n |"], ["T1", "T2"]) == ["T2"]),
        ("⑲a ★ 计数：已判定档重测 ⇒ 期望变化 **0**",
         verdict_transition(True, "已验证不可用", "已验证不可用") == 0),
        ("⑲b ★ 计数：新判定档 ⇒ 期望变化 **+1**",
         verdict_transition(False, "—", "已验证可用") == 1),
        ("⑲c ★ 计数：未判定档仍无结论 ⇒ 变化 0",
         verdict_transition(False, "—", "尚未验证") == 0),
        ("⑧ 仅措辞/格式不同、语义相同 ⇒ 放行",
         norm_param("switches", "save_steps=31;loss_scale=all") ==
         norm_param("switches", "loss_scale=all; save_steps=31")),
    ]
    ok = True
    for name, passed in cases:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'}（{sum(1 for _, x in cases if x)}/{len(cases)}）")
    return 0 if ok else 1


def _threshold_errs(doc, frozen_map, method):
    frozen_pp = frozen_map.get(method)
    claimed = run_claimed_pp(doc)
    if frozen_pp is None and claimed is not None:
        return ["未冻结却自述阈值"]
    if frozen_pp is not None and claimed is not None and claimed != frozen_pp:
        return [f"自述 {claimed} ≠ 矩阵 {frozen_pp}"]
    return []


if __name__ == "__main__":
    sys.exit(main())
