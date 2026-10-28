#!/usr/bin/env python3
"""backfill_matrix.py — 跑次 json ⇒ 数据件（原地回填）⇒ 重渲染 ⇒ 逐字节校验 ⇒ 落盘（唯一回填通道）。

★★ 口径（2026-10-22 对齐；本文件此前"回填即崩 / 写了不该写的列"的病史见工位 work_log）：

  结果 JSON **只承载「实验数据 + 检查结果」**。⛔ 不再接受人工指定「是否通过」：
  顶层出现 `status` 字段 ⇒ **报错拒绝**（不静默忽略、不悄悄采纳）——通过与否不能由填写者说了，
  必须由回填器按第 2 章四条判据**从数据算出**。

  ① 回填写满第 4 章「实验数据」组**6 列**（与数据件 `data_keys` 同源）：
       显存峰值 `peak_mem`／实跑步数 `steps_actual`／耗时 `elapsed`／吞吐 `throughput`／
       效果（评测 Δ）`effect`／备注 `note`。
       ★ `steps_actual`（整数步）与 `elapsed`（秒）由本脚本**直接写**（⛔ 不再依赖下游派生），
         并按固定形态格式化（`N 步` / `总时长 N s（约 X h）`）。
  ② 状态列按四条判据**自动计算**（阈值一律从数据件 `meta.criteria_caliber.threshold` **现读**，
     ⛔ 本文件不写任何阈值常量）：
       判据① 跑完（★ 2026-10-23 对齐第 2 章「跑完总训练轮数（Epoch）」）：退出码 0 ∧
              **`epochs_completed ≥ epochs_planned`** ∧ 最终检查点**真可加载** ∧ 无 NaN/Inf；
              实跑步数（数据件 `derived_steps`）**只作台账**（`steps_actual` 仍必填写进表格），
              ⛔ 不作通过条件；缺 `epochs_completed` ⇒ `unknown`（fail-closed，⛔ 不回落步数）
       判据② 效果：`threshold.by_method[<方法>]` 的通过线（Δ ≥ 通过线）；OPD 另需
              「教师 − 学生未训练 ≥ 教师线」（教师线从 `by_method.opd.teacher` 现解析）；
              **CPT 另有 loss 支**：「训练 loss 正常下降」⇒ 复用
              `src/graspo/core/result_judge.judge_cpt_loss`（首尾窗口均值严格下降）；
              缺 `eval.loss_curve.series` ⇒ `unknown`（fail-closed，⛔ 不静默跳过）
       判据③ 算法正确性检查
       判据④ 能完整产出可解析的推理结果（只判能否推理）
  ③ **fail-closed 铁律**：
       · 任一条判据「有据且不过」⇒ 判「不通过」（合取已假），未过项与未判定项写入备注；
       · 四条全「有据且过」⇒ 判「通过」；
       · 否则（存在缺证且无任一判据失败）⇒ **整批拒收、一个字节都不写**，档位**保持「待测试」**，
         报错逐条点名缺哪条证据。
       ★ 之所以"保持"而不是"写待测试 + 读数"：数据件词表把「待测试」定义为**未结论（数据必须为空）**，
         回填成「待测试 + 带数据」会让机核 ④C 必红（详见 verify_matrix.py ④C）。
  ④ 渲染**只走 `render_matrix.py`**（与现役生成链一致）；⛔ 不走 `build_official.py` 的重渲染路径
     （其产物的 meta 不含 `enumeration`，而渲染器对该键 fail-closed）。

★ 唯一输入格式：`<runs-dir>/<档号>.json`（schema = `matrix-run/v1`）；字段说明见下方 `REQUIRED_*`。

★ 用法（默认 = 原地回填正式数据件 + 渲染正式人读件）：
    python3 tools/matrix/backfill_matrix.py --runs-dir <dir>
    python3 tools/matrix/backfill_matrix.py --runs-dir <dir> --only M001,M002 --dry-run
  旁路演练（⛔ 不碰正式件）：
    python3 tools/matrix/backfill_matrix.py --runs-dir <demo-runs> --data-in <副本>.json [--html-out <副本>.html]
  ★ `--html-out` 缺省 = 数据件同目录下 `meta.authority.human_view_path` 的 basename
    ⇒ 数据件副本在别处时，产物也落在别处（不会写回 docs/）。
"""
import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parents[1]

# ★ 2026-10-23（基准件唯一 + 按轮副本）：人读件文件名/路径的唯一真相源 = 同目录 `paths.py`（宪法 §1.4）。
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parents[1] / "scripts"))   # ← `<repo>/scripts`（note_style 所在）
import paths as _paths  # noqa: E402
import note_style  # noqa: E402  —— 备注列「说人话」口径（唯一真相源，⛔ 不另抄一套规则）


#: ★★ 判据类型名的**唯一真相源** = `src/graspo/eval/thresholds.KNOWN_CRITERIA`
#:   —— 本文件⛔ 不再抄第三份名字表（病史：`delta_pp → gain_pp` 改名时本文件漏改，
#:   导致 `frozen_map_from_caliber()` 把 sft/opd/graspo 判成 `None`，**合法跑次被判「未冻结却自述阈值」而拒收**）。
#:   取法：直接 import 该模块（同仓、同解释器；`src/` 不在 sys.path 时按路径载入）。
def _load_known_criteria() -> tuple[str, ...]:
    """读 `graspo.eval.thresholds.KNOWN_CRITERIA`；取不到 ⇒ **fail-closed**（⛔ 不回落本地常量）。"""
    import importlib.util as _ilu

    _path = ROOT / "src" / "graspo" / "eval" / "thresholds.py"
    if not _path.is_file():
        raise SystemExit(f"FATAL: 找不到判据真相源 {_path} ⇒ backfill 的判据白名单无法确定（fail-closed）")
    _spec = _ilu.spec_from_file_location("_graspo_thresholds_for_backfill", _path)
    if _spec is None or _spec.loader is None:
        raise SystemExit(f"FATAL: 载入 {_path} 失败 ⇒ 判据白名单无法确定（fail-closed）")
    _mod = _ilu.module_from_spec(_spec)
    # ★ dataclasses 需要 `sys.modules[cls.__module__]` 存在 ⇒ 先登记再 exec
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    _known = tuple(getattr(_mod, "KNOWN_CRITERIA", ()) or ())
    if not _known:
        raise SystemExit(f"FATAL: {_path} 的 KNOWN_CRITERIA 为空 ⇒ fail-closed")
    return _known


#: 本文件认得的判据名 = 真相源**原样**（⛔ 不增不减；自检见 `_known_criteria_selfcheck()`）。
KNOWN_CRITERIA: tuple[str, ...] = _load_known_criteria()


def _known_criteria_selfcheck() -> None:
    """★ 防第四份副本：本文件用到的判据名集合必须与 `KNOWN_CRITERIA` **完全相等**。

    `frozen_map_from_caliber` 只从本常量取名字；一旦有人在别处再写死一份字面表，
    本函数在 `--self-test` 与正式跑里都会 fail-closed。
    """
    _lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    # `--self-test` 的夹具**必须**构造判据名（那是它的被测输入）⇒ 该函数体豁免于本扫描
    _st_lo = next((i for i, _l in enumerate(_lines, 1) if _l.startswith("def _self_test(")), 0)
    _st_hi = len(_lines) + 1
    for _i in range(_st_lo + 1, len(_lines) + 1):
        _l = _lines[_i - 1]
        if _l.startswith("def ") or _l.startswith("class "):
            _st_hi = _i
            break
    _hits = []
    for _i, _line in enumerate(_lines, 1):
        if _st_lo <= _i < _st_hi:
            continue                      # 自证夹具（见上）
        _t = _line.lstrip()
        if _t.startswith("#") or _t[:1] in ('"', "'"):
            continue                      # 注释 / 文档行不算（允许记录改名病史）
        if "criterion" not in _line:
            continue
        for _c in KNOWN_CRITERIA:
            if ('"' + _c + '"') in _line or ("'" + _c + "'") in _line:
                _hits.append((_i, _c))
    if _hits:
        raise SystemExit(f"FATAL: backfill 源码出现判据名字面量 {_hits[:5]} "
                         "⇒ 又抄了第四份副本（fail-closed）")


SCHEMA = "matrix-run/v1"

#: 第 4 章「实验数据」组 6 列（= 数据件 `data_keys`；回填器必须**写满**）
DATA_COLUMNS = ("peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note")

#: 顶层必填键。★ `steps_actual` / `elapsed` 在必填之列（缺了报错，⛔ 不静默补空）。
#: ⛔ `status` **不在**其中，且出现即拒收（见 `validate_run`）。
REQUIRED_TOP = ["schema", "tier", "params", "run_result", "steps_actual", "elapsed",
                "peak_mem", "throughput", "effect", "note", "run_path", "provenance"]

#: 判据① 的结构化检查结果（可机判；缺键 ⇒ 校验失败，⛔ 不猜）。
#: ★ 2026-10-23：判据① 改为「**跑满 Epoch**」（第 2 章原文），⇒ `epochs_completed` 进必填；
#:   `epochs_planned`（配置 `max_epochs`）与其余键允许 `null` —— `null` = 缺证据 ⇒ judge 判
#:   `unknown`（fail-closed），⛔ 不得判成 pass，也⛔ 不得回落成步数比较。
REQUIRED_RUN_RESULT = ["exit_code", "epochs_completed", "final_ckpt_loadable",
                       "nan_or_inf_detected"]

# ★ 参数身份（防「跑 A 填 B」错位）：身份键必须自报且与档位表逐项一致。
PARAM_IDENTITY_KEYS = ["model", "algo", "mode", "backend", "parallel"]
#: ★ 严格比对集：身份键 + 并行维度 + lr
PARAM_KEYS_V3 = ["model", "algo", "mode", "backend", "parallel", "lr"]
#: ★ "训练身份"语义键：按归一化后的语义值比对（⛔ 不比展示串/散文）
#:   ★★ 2026-10-22：`steps_actual` 已从本表移除 —— 它现在是**结果列**（回填器写），
#:      不再属于参数身份；留在表里会让"同档重测、步数变了"被误判成错位。
PARAM_SEMANTIC_KEYS = ["lora_target", "target_modules", "data_subset", "switches"]
LEGACY_PARAM_KEYS = ["hparams"]

#: 占位符 / 空值集（「写满 6 列」的反面：这些值一律视为"没填"）
EMPTY_VALUES = ("", "—", None)

LEDGER_HEADER = """# 回填台账 — `backfill-ledger.md`

> **唯一回填通道**：`backfill_matrix.py` 消费 `<runs-dir>/<档号>.json` ⇒ 原地更新
> `capability-matrix.data.json` ⇒ `render_matrix.py` 重渲染 ⇒ `verify_matrix.py` 逐字节校验。
> **增量**：跑完一个档 ⇒ 回填一个档；**未跑/未判定的档位保持「待测试」**；**缺字段/缺判据证据 ⇒ 整批 fail-closed**。
> 「是否通过」由脚本按四条判据从数据算出，⛔ 结果 json 不得自带 `status`。
> 每次回填写一行；`verify` 列 = 该次回填后端到端校验结果（render + 逐字节比对 + 结构/内容断言）。

| 回填时间 | 档号 | 来源 json | json sha256 | 状态 | verify | 备注 |
|:--:|---|---|---|---|:--:|---|
"""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _s(v):
    return v.strip() if isinstance(v, str) else v


def _num(v):
    """数值化（bool 是 int 子类 ⇒ 必须排除，§2.4 防呆）；非数 ⇒ None。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _ints(v):
    return [int(x) for x in re.findall(r"\d+", str(v))]


# ---- 身份归一化（⛔ 不做任何可能"合并真实不同值"的归一） ----
def norm_ident(v):
    """★ 身份字符串规范化：去括注（全/半角）、去首尾空白、去 markdown 强调符。

    同一类别会有「全参」「全参（full-parameter）」这类**同义带括注**写法 ⇒ ⛔ 精确相等会误判错位。
    ★ 只用于**身份键**（model/algo/mode/backend）与语义字段的字符串比较；⛔ 不改数值/阈值判定。
    """
    t = str(v or "").replace("*", "").strip()
    t = re.sub(r"[（(][^）)]*[）)]\s*$", "", t).strip().lower()
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
    · switches ⇒ `k=v` 键值对集合（⛔ 不比顺序/措辞）。
    """
    if v is None or str(v).strip() in ("", "—"):
        return None
    t = str(v).replace("*", "")          # 去 markdown 强调符（否则 **358** 件 解析不到）
    if key in ("lora_target", "target_modules"):
        # ★ 优先取「实例数」：档位表文本形如「…（**122 件**）｜native：canonical preset 名（**358 实例**…）」
        #   —— 358 = 模块实例数（跑次 json 写的也是实例数）；按第一个"N 件"取值会把同一事实的两种计数判成错位。
        m = re.search(r"(\d+)\s*(?:个)?\s*(?:LoRA\s*)?(?:模块\s*)?实例", t)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)\s*件", t)
        if m:
            return int(m.group(1))
        if "language_all_linear" in t or "vision_common" in t:
            return 122
        if "language_model" in t or "visual" in t:
            return 358
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


def norm_parallel(s):
    """归一化为 (总卡数, 世界大小, 维度字典)，用于语义身份比对。"""
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
    for k in sorted(set(nj[2]) & set(nt[2])):
        if nj[2][k] != nt[2][k]:
            errs.append(f"params.parallel 维度 {k} 不一致（防错位）：json={nj[2][k]} 表={nt[2][k]}")
    return errs


# ---- 判据② 的阈值：一律从数据件现读（⛔ 本文件无阈值常量） ----
#: ★ 判据②（效果）逐算法通过线的方法键（顺序即文档展示顺序）
CALIBER_METHODS = ("sft", "graspo", "cpt", "opd")
TRAINING_CLASSES = {"sft", "cpt", "opd"}


def _frozen_text(s):
    """★ 「已冻结」判定：含「冻结」且**不含「未冻结」**（⛔ 子串误判会解除 fail-closed）。"""
    t = str(s or "")
    return "冻结" in t and "未冻结" not in t


def frozen_map_from_caliber(th):
    """★ 纯函数：数据件 `meta.criteria_caliber.threshold` ⇒ 方法→冻结阈值(pp) 映射。

    ★ 唯一真相源 = `threshold.by_method`（逐算法）；未冻结 ⇒ None。
    ★ 回退（**仅当** `by_method` 缺失，如旧代际正式件）：用旧的单一 `value` —— 该值已被取代，
      只为兼容旧数据而保留，⛔ 不得据此声称"当前口径仍是统一阈值"。
    """
    out = {m: None for m in CALIBER_METHODS}
    th = th if isinstance(th, dict) else {}
    bm = th.get("by_method") or {}
    if bm:
        for m in CALIBER_METHODS:
            e = bm.get(m) or {}
            v = e.get("value")
            if v is None:
                continue
            # ★ 只认判据② 的两类通过线；其余判据类型 ⇒ 不据此判定（fail-closed）
            # ★ 判据名一律取真相源（⛔ 不写字面元组）
            if str(e.get("criterion", "")) not in KNOWN_CRITERIA:
                continue
            if _frozen_text(th.get("status")) or _frozen_text(e.get("status")):
                out[m] = float(v)
        return out
    val, applies = th.get("value"), str(th.get("applies_to", ""))
    status = str(th.get("status", ""))
    if val is not None and ("训练类" in applies or "冻结" in status):
        for m in TRAINING_CLASSES:
            out[m] = float(val)
    if val is not None and "GRASPO" in applies:
        out["graspo"] = float(val)
    return out


#: OPD 教师线的可解析形态（散文里的「≥ +10.0 个百分点」；⛔ 不在本文件写死数值）
RE_TEACHER_LINE = re.compile(r"≥\s*\+?\s*([0-9]+(?:\.[0-9]+)?)\s*个?百分点")


def teacher_line_pp(th):
    """从数据件现解析 OPD 教师线（教师 − 学生未训练 的 pp 下限）；缺 ⇒ None（判据② 不可判定）。"""
    bm = (th if isinstance(th, dict) else {}).get("by_method") or {}
    m = RE_TEACHER_LINE.search(str((bm.get("opd") or {}).get("teacher") or ""))
    return float(m.group(1)) if m else None


# ---- 台账 ----
LEDGER_ROW = re.compile(
    r"^\|\s*(?P<time>[^|]+?)\s*\|\s*`(?P<tier>[^`]+)`\s*\|\s*`(?P<file>[^`]+)`\s*\|\s*`(?P<sha>[^`]+)`\s*\|"
    r"\s*(?P<status>[^|]*?)\s*\|\s*(?P<verdict>[^|]*?)\s*\|\s*(?P<note>[^|]*?)\s*\|\s*$")


def dedup_ledger(text):
    """★ 台账按 `(档号, json sha)` 去重：同键合并为一行，保留次数（审计痕迹）。

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
            k, cnt = bykey[key]
            cnt += 1
            bykey[key] = (k, cnt)
            slots[k] = ("row", row[:-1] + " ×%d（同源重复回填） |" % cnt)
        else:
            bykey[key] = (len(slots), 1)
            slots.append(("row", row))
    out = [payload for _kind, payload in slots]
    return "\n".join(out) + "\n"


def ledger_covers(rows, filled_tiers):
    """★ 断言辅助（可单测）：本次处理的每一档是否都在台账行里。返回缺失的档号列表。"""
    return [t for t in filled_tiers if not any(f"`{t}`" in r for r in rows)]


def append_ledger(path: Path, rows):
    text = path.read_text(encoding="utf-8") if path.exists() else LEDGER_HEADER
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    path.write_text(dedup_ledger(path.read_text(encoding="utf-8")), encoding="utf-8")


# ---- 防呆辅助（可单测） ----
def unknown_tier_stems(stems, tier_rows) -> list[str]:
    """结果目录里**不属于本矩阵**的档号（排序返回；非空 ⇒ 报错退出）。"""
    return sorted({str(s) for s in stems} - set(tier_rows))


def resolve_only(only_csv, tier_rows, stems):
    """解析 `--only`：返回 (档号集合 或 None, 错误列表)。

    ★ 防呆：档号必须存在于本矩阵（`M777` ⇒ 报错）；且结果目录里必须有对应 json
      （否则"什么都没回填却 rc=0"会把「漏跑 / 打错档号」藏起来）。
    """
    if not only_csv:
        return None, []
    only = {x.strip() for x in only_csv.split(",") if x.strip()}
    if not only:
        return None, ["--only 为空（逗号分隔的档号列表）"]
    errs = []
    bad = sorted(only - set(tier_rows))
    if bad:
        errs.append(f"--only 指定了不存在的档号 {bad}（本矩阵档号范围 {min(tier_rows)}…{max(tier_rows)}）")
    no_file = sorted(only - {str(s) for s in stems})
    if no_file:
        errs.append(f"--only 指定的档号在结果目录内无对应 json：{no_file}")
    return only, errs


def untouched_drift(orig_rows, new_rows, processed) -> list[str]:
    """★ 部分回填的保真断言：**未处理的档必须与原件逐字段相等**。返回漂移档号列表。"""
    return [t for t, r in orig_rows.items() if t not in processed and new_rows.get(t) != r]


# ---- 校验：跑次 json 结构 + 身份 + 「不许人工指定 status」 ----
def validate_run(doc, tier_from_file, tier_rows):
    """严格校验一份跑次 json。返回错误列表（空 = 合法）。"""
    if not isinstance(doc, dict):
        return ["顶层不是 JSON 对象"]
    errs = []
    # ★★ 口径：状态由回填器按判据算出 ⇒ 结果 json 不得自带 `status`（不静默忽略、不悄悄采纳）
    if "status" in doc:
        errs.append("结果 json 不得含 `status` 字段：是否通过由回填器按第 2 章四条判据从数据算出，"
                    "⛔ 不接受人工指定（删掉该字段后重跑）")
    errs += [f"缺字段 {k!r}" for k in REQUIRED_TOP if k not in doc]
    if errs:
        return errs

    if doc.get("schema") != SCHEMA:
        errs.append(f"schema 必须是 {SCHEMA!r}，实际 {doc.get('schema')!r}")

    # --- tier：必须与文件名一致、且在本矩阵档位表内 ---
    tier = _s(doc["tier"])
    if not isinstance(tier, str) or not tier:
        errs.append("tier 为空或非字符串")
    else:
        if tier != tier_from_file:
            errs.append(f"档号 {tier!r} 与文件名 {tier_from_file!r} 不一致")
        if tier not in tier_rows:
            errs.append(f"档号 {tier!r} 不在矩阵档位表内")

    # --- params：身份键必须齐全且与档位表逐项一致（防「跑 A 填 B」错位） ---
    p = doc["params"]
    if not isinstance(p, dict):
        errs.append("params 不是 JSON 对象")
    elif tier in tier_rows:
        ref = tier_rows[tier]
        for k in PARAM_IDENTITY_KEYS:
            v = _s(p.get(k))
            if k not in p:
                errs.append(f"params 缺身份键 {k!r}")
            elif not isinstance(v, str) or not v:
                errs.append(f"params.{k} 为空")
        for k in PARAM_KEYS_V3:
            if k not in p or _s(p.get(k)) is None:
                continue
            j, r_ = _s(p.get(k)), _s(ref.get(k, ""))
            if k == "parallel":
                errs.extend(compare_parallel(j, r_))
            elif norm_ident(r_) != norm_ident(j):
                errs.append(f"params.{k} 与档位表不一致（防错位）：json={j!r} 表={r_!r}")
        for k in PARAM_SEMANTIC_KEYS:
            if k not in p:
                continue
            nj, nt = norm_param(k, p.get(k)), norm_param(k, ref.get(k, ""))
            if nj is not None and nt is not None and nj != nt:
                errs.append(f"params.{k} 语义不一致（训练身份，防错位）：json={nj} 表={nt}")
        for k in LEGACY_PARAM_KEYS:
            if k in p and k not in ref:
                doc.setdefault("_legacy_params_ignored", []).append(k)

    # --- 判据① 的结构化检查结果（可机判；缺键/类型不对 ⇒ 报错，⛔ 不猜） ---
    rr = doc["run_result"]
    if not isinstance(rr, dict):
        errs.append("run_result 不是 JSON 对象")
    else:
        for k in REQUIRED_RUN_RESULT:
            if k not in rr:
                errs.append(f"run_result 缺 {k!r}")
        ex = rr.get("exit_code")
        if "exit_code" in rr and (isinstance(ex, bool) or not isinstance(ex, int)):
            errs.append(f"run_result.exit_code 必须是整数，实际 {ex!r}")
        # ★ 判据① 的 epoch 键：整数或 null（null = 缺读数 ⇒ judge 判 unknown，fail-closed）
        for k in ("epochs_completed", "epochs_planned"):
            if k not in rr:
                continue
            v = rr.get(k)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, int):
                errs.append(f"run_result.{k} 必须是整数或 null（null = 缺读数），实际 {v!r}")
            elif v < 0:
                errs.append(f"run_result.{k} 不得为负，实际 {v}")
        # ★ 两个布尔项：null = 缺证据（判 unknown，⛔ 不是"不通过"）；其它类型一律拒收
        for k in ("final_ckpt_loadable", "nan_or_inf_detected"):
            if k not in rr:
                continue
            v = rr.get(k)
            if v is None or isinstance(v, bool):
                continue
            errs.append(f"run_result.{k} 必须是布尔值或 null（null = 缺证据），实际 {v!r}")

    # --- 必填数值列：实跑步数 / 耗时（⛔ 不许缺、不许用占位符糊过去） ---
    st = doc["steps_actual"]
    if isinstance(st, bool) or not isinstance(st, int):
        errs.append(f"steps_actual 必须是整数（实跑步数），实际 {st!r}")
    elif st < 0:
        errs.append(f"steps_actual 不得为负，实际 {st}")
    el = _num(doc["elapsed"])
    if el is None:
        errs.append(f"elapsed 必须是数值（训练墙钟秒数），实际 {doc['elapsed']!r}")
    elif el < 0:
        errs.append(f"elapsed 不得为负，实际 {el}")

    # --- 其余 4 列：必须是非空文本（⛔ 不用占位符冒充"写满"） ---
    for k in ("peak_mem", "throughput", "effect", "note"):
        v = doc[k]
        if not isinstance(v, str) or v.strip() in EMPTY_VALUES:
            errs.append(f"结果列 {k!r} 必须是非空文本（实际 {v!r}）；缺数据请写清原因，⛔ 不许留空/占位符")

    # --- 可选：判据② 的评测读数（缺 ⇒ 判据② 不可判定，由 judge 层 fail-closed） ---
    ev = doc.get("eval")
    if ev is not None:
        if not isinstance(ev, dict):
            errs.append("eval 不是 JSON 对象")
        else:
            for k in ("after_acc", "base_acc", "teacher_after_acc"):
                if k in ev:
                    v = ev[k]
                    n = _num(v)
                    if n is None or not (0.0 <= n <= 1.0):
                        errs.append(f"eval.{k} 必须是 0–1 的比例（数据集完全正确率），实际 {v!r}")
            if "delta_pp" in ev and _num(ev["delta_pp"]) is None:
                errs.append(f"eval.delta_pp 必须是数值（pp），实际 {ev['delta_pp']!r}")
            a, b = _num(ev.get("after_acc")), _num(ev.get("base_acc"))
            d = _num(ev.get("delta_pp"))
            if a is not None and b is not None and d is not None:
                if abs((a - b) * 100.0 - d) > 1e-6:
                    errs.append(f"eval.delta_pp={d} 与 (after_acc - base_acc)×100 = {(a - b) * 100.0:.6f} 不符"
                                "（前后矛盾 ⇒ 拒收）")

    # --- 可选：判据③④ 的检查结果（缺 ⇒ 对应判据不可判定，由 judge 层 fail-closed） ---
    ck = doc.get("checks")
    if ck is not None:
        if not isinstance(ck, dict):
            errs.append("checks 不是 JSON 对象")
        else:
            for k in ("algo_checks_passed", "inference_parseable"):
                if k in ck and not isinstance(ck[k], bool):
                    errs.append(f"checks.{k} 必须是布尔值，实际 {ck[k]!r}")

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
        evp = prov.get("evidence_paths")
        if not isinstance(evp, list) or not evp or not all(isinstance(x, str) and x.strip() for x in evp):
            errs.append("provenance.evidence_paths 必须是非空字符串数组（原始证据路径）")
    return errs


# ---- 四条判据：逐条给出 pass / fail / unknown（⛔ unknown 不得被当成 pass） ----
PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


def crash_marker(doc):
    """崩溃档标记（`scripts/make_result_json.py --recover-crashed-tier` 写进 provenance）。

    ★ D30（2026-10-28 主席口径：崩溃一律记「不通过」，⛔ 不得留「待测试」）：
      本函数只做**读取**，不判任何东西；`applies is True` 的前提是四条证据门槛全成立
      （退出码≠0 ∧ stdout 报错原文与帧 ∧ 无任何训练产物 ∧ 身份件 + readings.tier 一致）。
    ⛔ 绝不因为"缺证据"就置位：置位只能由产出端按证据做，回填器这边不许自我授权。
    """
    prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    crash = prov.get("crashed_tier_evidence")
    return crash if isinstance(crash, dict) and crash.get("applies") is True else None


def crash_unavailable_text(doc, cid: str) -> str | None:
    """崩溃档上该条判据"为什么没有读数"的**唯一文案**（⛔ 不是"没通过"，也不是留空/填 0）。"""
    crash = crash_marker(doc)
    if crash is None:
        return None
    detail = (crash.get("criteria_unavailable") or {}).get(cid)
    return f"判据{cid} 缺证据：未取得（训练崩溃{('；' + detail) if detail else ''}）"


def judge_run_complete(doc, row):
    """判据① 跑完：**跑满 Epoch** ∧ 退出码 0 ∧ 最终检查点真可加载 ∧ 无 NaN/Inf。

    ★ 2026-10-23 口径对齐（第 2 章原文 =「跑完总训练轮数（Epoch）」）：
      · **通过条件 = `epochs_completed >= epochs_planned`**（`epochs_planned` = 清单 `max_epochs`，
        缺则回落同一真相源 `params.max_epochs`）；
      · `steps_actual` 仍必填并回填进「实跑步数」列，但**不再参与判定**（它只是台账：
        判分/取数粒度变了就会与 `derived_steps` 漂移，拿它当门槛会把"跑满 8 轮"判成失败）；
      · 缺 `epochs_completed` ⇒ **UNKNOWN**（fail-closed；⛔ 不回落步数比较）；
      · `final_ckpt_loadable` / `nan_or_inf_detected` 为 `null` ⇒ **UNKNOWN**（缺证据 ≠ 失败）。
    """
    rr = doc.get("run_result") if isinstance(doc.get("run_result"), dict) else {}
    ex = rr.get("exit_code")
    if isinstance(ex, bool) or not isinstance(ex, int):
        return UNKNOWN, "判据① 缺证据：run_result.exit_code"
    # ★ D30（2026-10-28 主席口径：崩溃一律记「不通过」）：崩溃档上"退出码≠0"是**有据且不过**
    #   —— 合取已假，⛔ 不得再因为"崩溃天然没有 epoch / 没有可加载 checkpoints / 没有 NaN 读数"
    #   掉进 UNKNOWN（那正是 M039/M040 卡在「待测试」的最后一道口子）。
    #   ★ 这不是放宽判据：判据① 的结论仍是"未过"，依据仍是退出码（原值，未改写）；
    #   且这条只在 `crash_marker` 成立（四条证据门槛全过）时生效，⛔ 不是给所有档开后门。
    if ex != 0 and crash_marker(doc) is not None:
        steps = doc.get("steps_actual")
        return FAIL, (f"判据① 未过：退出码 {ex} ≠ 0（起跑即崩，未进入优化循环；"
                      f"台账：实跑 {steps} 步（不作通过条件））")
    epochs_done = rr.get("epochs_completed")
    epochs_planned = rr.get("epochs_planned")
    if epochs_planned is None:
        planned = (doc.get("params") or {}).get("max_epochs")
        epochs_planned = (planned if isinstance(planned, int)
                          and not isinstance(planned, bool) else None)
    if epochs_done is None or isinstance(epochs_done, bool) or not isinstance(epochs_done, int):
        # D29：这句会经 `compose_note` 进**备注列**（人读件）⇒ ⛔ 不写内部记号（`⛔`）。
        return UNKNOWN, ("判据① 缺证据：run_result.epochs_completed（Epoch 读数；"
                         "不回落实跑步数比较）")
    if (epochs_planned is None or isinstance(epochs_planned, bool)
            or not isinstance(epochs_planned, int)):
        return UNKNOWN, ("判据① 缺证据：计划轮数（run_result.epochs_planned / params.max_epochs）")
    fails = []
    if ex != 0:
        fails.append(f"退出码 {ex} ≠ 0")
    if epochs_done < epochs_planned:
        fails.append(f"跑满 {epochs_done}/{epochs_planned} epoch（未跑满总训练轮数）")
    loadable, nan_inf = rr.get("final_ckpt_loadable"), rr.get("nan_or_inf_detected")
    if loadable is False:
        fails.append("最终检查点不可加载")
    elif loadable is None:
        return UNKNOWN, "判据① 缺证据：run_result.final_ckpt_loadable（检查点真加载校验未做）"
    if nan_inf is True:
        fails.append("检测到 NaN/Inf（含被跳过的非有限梯度步）")
    elif nan_inf is None:
        return UNKNOWN, "判据① 缺证据：run_result.nan_or_inf_detected（三处证据源都读不到）"
    # ★ D29（2026-10-27 主席裁定）：台账句会经 `compose_note` 进**备注列**（人读件）⇒
    #   ① ⛔ 不写「计划 N 步」：清单侧每 epoch 步数按**总卡数**折算，与真机（只随 dp 变）差 2~8 倍，
    #      两个数并排会让人误读成"计划口径错了"；正确口径须重生成数据件（另案呈报）⇒ 只动文案。
    #   ② ⛔ 不写内部记号（`⛔` 属内部写作记号；矩阵是给人看的唯一真相源）。
    #   ⛔ 判据不变：判据① 认的仍是 `epochs_completed ≥ epochs_planned`（见本函数 docstring）。
    steps = doc.get("steps_actual")
    ledger = f"；台账：实跑 {steps} 步（不作通过条件）"
    if fails:
        return FAIL, "判据① 未过：" + "；".join(fails) + ledger
    return PASS, (f"判据① 跑完（{epochs_done}/{epochs_planned} epoch，检查点可加载，"
                  f"无 NaN/Inf{ledger}）")


#: ★ 判据② 的 **CPT 支**（"训练 loss 正常下降"）—— 复用 `src/graspo/core/result_judge.py`
#: 的**同一个判定函数** `judge_cpt_loss`（⛔ 不另写阈值：窗口 = 首尾各
#: `min(max(3, round(n × 0.10)), n)` 的均值，判定式 = **终窗均值 < 起窗均值**，严格小于；
#: 口径与该文件 `CPT_LOSS_WINDOW_MIN/CPT_LOSS_WINDOW_FRACTION/judge_cpt_loss` 逐字一致）。
_RESULT_JUDGE = None


def result_judge_module():
    """按路径载入判据真相源 `src/graspo/core/result_judge.py`（同仓、同解释器）。

    为什么不在本文件重抄阈值：CPT 的 loss 口径只能有一份（§1.4）。载入失败 ⇒ fail-closed
    （判据② 判 unknown），⛔ 不回落任何本地常量。
    """
    global _RESULT_JUDGE
    if _RESULT_JUDGE is not None:
        return _RESULT_JUDGE
    import importlib.util as _ilu

    path = ROOT / "src" / "graspo" / "core" / "result_judge.py"
    if not path.is_file():
        raise SystemExit(f"FATAL: 找不到判据真相源 {path} ⇒ CPT loss 口径无法确定（fail-closed）")
    spec = _ilu.spec_from_file_location("_graspo_result_judge_for_backfill", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: 载入 {path} 失败 ⇒ CPT loss 口径无法确定（fail-closed）")
    mod = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod          # ★ dataclasses 需要模块已登记
    spec.loader.exec_module(mod)
    _RESULT_JUDGE = mod
    return mod


def loss_series_from_doc(doc):
    """取 CPT 的训练 loss **逐步序列**（唯一载体：`eval.loss_curve.series`）。

    返回 `(series|None, why)`。★ 只给首/末两点**不可**判定：`judge_cpt_loss` 的口径是
    首尾**窗口**均值，⛔ 本层不自造"首末两点"的替代阈值。
    """
    ev = doc.get("eval") if isinstance(doc.get("eval"), dict) else {}
    curve = ev.get("loss_curve")
    where = "eval.loss_curve"
    if not isinstance(curve, dict):
        detail = doc.get("effect_detail") if isinstance(doc.get("effect_detail"), dict) else {}
        curve = detail.get("loss_curve")
        where = "effect_detail.loss_curve"
    if isinstance(doc.get("effect"), dict):        # 兼容旧/未来形态（effect 若为结构化对象）
        curve = doc["effect"].get("loss_curve")
        where = "effect.loss_curve"
    if not isinstance(curve, dict):
        return None, "缺 eval.loss_curve（CPT 的训练 loss 读数未产出）"
    series = curve.get("series")
    if not isinstance(series, list) or not series:
        return None, f"{where} 只有首/末读数、没有逐步序列（无法按首尾窗口均值口径复算）"
    out = []
    for v in series:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, f"{where}.series 含非数值项 {v!r}（读数不干净）"
        out.append(float(v))
    return out, f"{where}（{len(out)} 个逐步 loss 点）"


def judge_cpt_loss_evidence(doc):
    """判据② 的 **CPT 支**：训练 loss 正常下降（缺读数 ⇒ UNKNOWN，未下降 ⇒ FAIL）。"""
    series, why = loss_series_from_doc(doc)
    if series is None:
        return UNKNOWN, f"判据② 缺证据（CPT）：{why}"
    rj = result_judge_module()
    evidence = rj.RunEvidence(
        tier_id=str(doc.get("tier") or ""),
        exit_code=(doc.get("run_result") or {}).get("exit_code"),
        timed_out=False,
        log_text="",
        tuner_type=str((doc.get("params") or {}).get("tuner_type") or ""),
        optimizer_steps=(doc.get("steps_actual")
                         if isinstance(doc.get("steps_actual"), int) else None),
        epochs_completed=(doc.get("run_result") or {}).get("epochs_completed"),
        weight_changed=None,
        checkpoint_reloadable=None,
        artifacts_present={},
        losses=tuple(series),
        grad_norms=(),
        algorithm="CPT",
    )
    res = rj.judge_cpt_loss(evidence)
    if res.evidence_missing:
        return UNKNOWN, f"判据② 缺证据（CPT）：{res.detail}"
    return (PASS if res.passed else FAIL), res.detail


def judge_effect(doc, row, frozen_map, teacher_pp):
    """判据② 效果：按该算法冻结通过线判 Δ；OPD 另判「教师 − 学生未训练 ≥ 教师线」。"""
    _cu = crash_unavailable_text(doc, "②")
    if _cu:
        return UNKNOWN, _cu
    method = str(row.get("train_method") or "").lower()
    line = frozen_map.get(method)
    if line is None:
        return UNKNOWN, (f"判据② 缺证据：{method or '未知算法'} 的通过线未冻结/取不到"
                         "（数据件 meta.criteria_caliber.threshold.by_method）")
    # ★ CPT：判据② 的定义是「**训练 loss 正常下降**（且 after ≥ base）」⇒ 先判 loss 支。
    #   ⛔ 缺 loss 读数不得被静默跳过（此前"无 loss 读数 + after=base"会返回通过）。
    cpt_verdict, cpt_detail = (PASS, "")
    if method == "cpt":
        cpt_verdict, cpt_detail = judge_cpt_loss_evidence(doc)
    ev = doc.get("eval") if isinstance(doc.get("eval"), dict) else {}
    a, b = _num(ev.get("after_acc")), _num(ev.get("base_acc"))
    if a is None or b is None:
        return UNKNOWN, "判据② 缺证据：eval.after_acc / eval.base_acc（评测读数）"
    delta = (a - b) * 100.0
    fails = []
    if cpt_verdict == FAIL:
        fails.append(cpt_detail)
    if delta < line - 1e-9:
        fails.append(f"{method} Δ = {delta:+.2f} pp 未达通过线 {line:+.1f} pp")
    if method == "opd":
        if teacher_pp is None:
            return UNKNOWN, ("判据② 缺证据：数据件未给出 OPD 教师线"
                             "（threshold.by_method.opd.teacher 无可解析数值）")
        t = _num(ev.get("teacher_after_acc"))
        if t is None:
            return UNKNOWN, "判据② 缺证据：OPD 需 eval.teacher_after_acc（教师在训练数据集上的完全正确率）"
        td = (t - b) * 100.0
        if td < teacher_pp - 1e-9:
            fails.append(f"OPD 教师 − 学生未训练 = {td:+.2f} pp 未达教师线 {teacher_pp:+.1f} pp")
    if fails:
        return FAIL, "判据② 未过：" + "；".join(fails)
    # ★ 合取已真 ⇒ 缺证据才判 unknown（缺证据 ⛔ 不得越过"有据且不过"）
    if cpt_verdict == UNKNOWN:
        return UNKNOWN, cpt_detail
    extra = ""
    if method == "opd":
        extra = (f"，教师 − 学生未训练 = {(_num(ev.get('teacher_after_acc')) - b) * 100.0:+.2f} pp"
                 f" ≥ {teacher_pp:+.1f} pp")
    cpt_extra = f"；{cpt_detail}" if method == "cpt" else ""
    return PASS, f"判据② {method} Δ = {delta:+.2f} pp ≥ {line:+.1f} pp{extra}{cpt_extra}"


def judge_algo_checks(doc):
    """判据③ 算法正确性检查（只判机制本身对不对）。"""
    ck = doc.get("checks") if isinstance(doc.get("checks"), dict) else {}
    v = ck.get("algo_checks_passed")
    if not isinstance(v, bool):
        _cu = crash_unavailable_text(doc, "③")
        if _cu:
            return UNKNOWN, _cu
        return UNKNOWN, "判据③ 缺证据：checks.algo_checks_passed（算法正确性检查结果）"
    return (PASS, "判据③ 算法正确性检查通过") if v else (FAIL, "判据③ 未过：算法正确性检查未通过")


def judge_inference(doc):
    """判据④ 能推理（只判能否完整产出可解析的推理结果；效果已由判据② 判）。"""
    ck = doc.get("checks") if isinstance(doc.get("checks"), dict) else {}
    v = ck.get("inference_parseable")
    if not isinstance(v, bool):
        _cu = crash_unavailable_text(doc, "④")
        if _cu:
            return UNKNOWN, _cu
        return UNKNOWN, "判据④ 缺证据：checks.inference_parseable（推理结果是否可解析）"
    return (PASS, "判据④ 可完整推理") if v else (FAIL, "判据④ 未过：推理结果不可解析")


def judge_row(doc, row, frozen_map, teacher_pp):
    """四条判据的合取。返回 (status|None, fails, unknowns)。

    status = `通过`（四条全过）／`不通过`（任一有据且不过）／None（缺证且无失败 ⇒ ⛔ 不得判定）。
    """
    verdicts = [judge_run_complete(doc, row),
                judge_effect(doc, row, frozen_map, teacher_pp),
                judge_algo_checks(doc),
                judge_inference(doc)]
    fails = [detail for verdict, detail in verdicts if verdict == FAIL]
    unks = [detail for verdict, detail in verdicts if verdict == UNKNOWN]
    if fails:
        return "不通过", fails, unks
    if unks:
        return None, [], unks
    return "通过", [], []


def compose_note(base, status, fails, unks):
    """备注列 = 跑次自报的结论 ＋ **一句**判据结论（`判据结论：①③④ 过，② 未过`）。

    ★ E1（2026-10-27 主席裁定）：此前把判据原因串**原样**拼进备注（含「台账：实跑 N 步」、
    退出码比较等内部措辞，M011 曾达 1324 字）⇒ 人读件被写成工程日志。现在：
      · 只留编号级的结论（过 / 未过 / 未判定），明细仍在 stdout 日志、台账文件与结果 json 的
        `fails`/`unknowns` 里（⛔ 没有删证据）；
      · 拼装走**唯一**渲染器 `scripts/note_style.py::render`（与存量规范化同一个），并受
        200 字上限约束。
    """
    note = str(base or "").strip()
    if status not in ("通过", "不通过"):
        return note
    f_ids, u_ids = note_style.criteria_ids(fails, unks)
    facts = note_style.parse_legacy(note)
    facts["fail_ids"], facts["unk_ids"], facts["judged"] = f_ids, u_ids, True
    return note_style.render(facts)


def fmt_steps(n: int) -> str:
    return f"{n} 步"


def fmt_elapsed(seconds) -> str:
    """「耗时」列展示形态（与生成链同一形态：总时长 N s（约 X h））。"""
    sec = float(seconds)
    return f"总时长 {int(round(sec))} s（约 {sec / 3600.0:.2f} h）"


# ---- 工具脚本调用（脚本名按本文件所在目录解析，可从仓库根调用） ----
def run_cmd(args):
    argv = [sys.executable]
    for i, x in enumerate(args):
        if i < 2 and x.endswith(".py") and not Path(x).is_absolute() and (TOOL_DIR / x).exists():
            argv.append(str(TOOL_DIR / x))
        else:
            argv.append(x)
    p = subprocess.run(argv, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def default_html_out(data_path: Path, matrix) -> Path:
    """`--html-out` 缺省：数据件同目录下 `meta.authority.human_view_path` 的 basename。

    ⇒ 正式数据件 → `docs/GRASPO训练框架功能测试报告.html`；数据件副本在 /tmp → 产物也在 /tmp
    （⛔ 不会把演练产物写回 docs/）。缺 `human_view_path` 时的兜底名取 `paths.py` 的基准件常量
    （宪法 §1.4：⛔ 本文件不再自写一份基准件文件名）。
    """
    hv = (((matrix.get("meta") or {}).get("authority") or {}).get("human_view_path") or "")
    name = Path(str(hv)).name or _paths.BASELINE_HTML_NAME
    return Path(data_path).parent / name


def main():
    # ★ allow_abbrev=False（R1-2，对应审查 🔴-2 的根治）：关掉 argparse 缩写 ⇒ `--html`/`--data-i`/
    #   `--le` 之类前缀不再被当成 `--html-out`/`--data-in`/`--ledger-out`，绕过 wrapper 直调本脚本也
    #   绕不过落点闸门（⛔ 不加任何"跳过校验"的豁免开关；本处只改这一行/一处，其余逻辑不动）。
    ap = argparse.ArgumentParser(description="跑次 json ⇒ 数据件（原地回填）⇒ 重渲染 ⇒ 机核 ⇒ 落盘",
                                 allow_abbrev=False)
    ap.add_argument("--runs-dir", default=None, help="跑次 json 目录（唯一输入格式：<档号>.json）")
    ap.add_argument("--data-in", default=str(ROOT / "docs" / "capability-matrix.data.json"),
                    help="矩阵数据件（默认 = 正式数据件）")
    ap.add_argument("--data-out", default=None, help="默认 = --data-in（原地回填）")
    ap.add_argument("--template", default=str(TOOL_DIR / "template.html"))
    ap.add_argument("--html-out", default=None,
                    help="人读件输出路径（默认 = 数据件同目录下的正式人读件 basename）")
    ap.add_argument("--ledger-out", default=None,
                    help="台账路径（缺省 = 数据件同目录下的 backfill-ledger.md；⛔ 不用相对 CWD 以免 stray）")
    ap.add_argument("--only", default=None, help="只回填这些档号，逗号分隔（增量）")
    ap.add_argument("--self-test", action="store_true", help="跑校验器的正/反向用例，不读跑次目录")
    ap.add_argument("--dry-run", action="store_true", help="只校验与报告，不写任何文件")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()
    if not a.runs_dir:
        ap.error("--runs-dir 必填（或改用 --self-test）")

    _known_criteria_selfcheck()

    data_in = Path(a.data_in)
    if not data_in.is_file():
        print(f"FAIL: 数据文件不存在: {data_in}")
        return 1
    template = Path(a.template)
    if not template.is_file():
        print(f"FAIL: 模板不存在: {template}")
        return 1
    runs_dir = Path(a.runs_dir)
    if not runs_dir.is_dir():
        print(f"FAIL: 跑次目录不存在: {runs_dir}")
        return 1

    # ★ 缺陷 1 的修法：默认值**先解析**成 data_out / html_out / ledger_out，下游一律只用解析后的值
    data_out = Path(a.data_out) if a.data_out else data_in
    ledger_out = Path(a.ledger_out) if a.ledger_out else data_out.parent / "backfill-ledger.md"

    matrix = json.loads(data_in.read_text(encoding="utf-8"))
    html_out = Path(a.html_out) if a.html_out else default_html_out(data_out, matrix)

    # ★★ §22 写前闸门（2026-10-25 主席令「这个报告就定稿冻结了，没有我的允许，不能修改报告」）：
    #   回填器**只**写按轮副本；落点若规范化后（含软链/硬链接/相对路径）指向**基准两件**
    #   ⇒ 拒起、**零写入**。基准件只经 `render_matrix.py --allow-baseline-write` 由主席批准改。
    for _tag, _dest in (("--data-out", data_out), ("--html-out", html_out)):
        try:
            _paths.assert_not_baseline_write(_dest, who=f"backfill_matrix.py {_tag}")
        except SystemExit as _exc:
            print(f"FAIL-CLOSED：{_exc}\n"
                  f"   ⇒ 正确用法：bash tools/matrix/backfill_dated.sh --date YYYYMMDD --runs-dir <dir>"
                  f"（副本由 new_round.sh 派生；结果只落副本）")
            return 1
    rows = matrix.get("rows") or []
    if not isinstance(rows, list) or not rows:
        print(f"FAIL: 数据件没有 rows: {data_in}")
        return 1
    tier_rows = {r["tier"]: r for r in rows}
    orig_rows = copy.deepcopy(tier_rows)

    threshold = (((matrix.get("meta") or {}).get("criteria_caliber") or {}).get("threshold")) or {}
    frozen_map = frozen_map_from_caliber(threshold)
    teacher_pp = teacher_line_pp(threshold)

    files = sorted(runs_dir.glob("*.json"))
    stems = [f.stem for f in files]
    # ★ 防呆：结果目录里不得混入不属于本矩阵的档号
    stray = unknown_tier_stems(stems, tier_rows)
    if stray:
        print(f"FAIL: 结果目录 {runs_dir} 含不属于本矩阵的档号 {stray} ⇒ 拒收（未写任何文件）")
        return 1

    only, only_errs = resolve_only(a.only, tier_rows, stems)
    if only_errs:
        print("FAIL: " + "；".join(only_errs) + " ⇒ 未写任何文件")
        return 1
    if only:
        files = [f for f in files if f.stem in only]
    if not files:
        print(f"无待回填的 json（目录 {runs_dir}" + (f"，--only {sorted(only)}" if only else "") + "）")
        return 0

    # ---- ① 严格校验 + 四条判据（fail-closed：先全验，再统一写） ----
    valid, all_errs, refusals = [], [], []
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            all_errs.append((f, [f"JSON 解析失败: {e}"]))
            continue
        errs = validate_run(doc, f.stem, tier_rows)
        if errs:
            all_errs.append((f, errs))
            continue
        status, fails, unks = judge_row(doc, tier_rows[doc["tier"]], frozen_map, teacher_pp)
        if status is None:
            refusals.append((f, doc["tier"], unks))
            continue
        valid.append((f, doc, status, fails, unks))

    if all_errs:
        print("FAIL-CLOSED：以下跑次 json 不合法，本批**未写任何文件**：")
        for f, errs in all_errs:
            print(f"  ✗ {f}")
            for e in errs:
                print(f"      - {e}")
        return 1
    if refusals:
        print("FAIL-CLOSED：以下跑次**缺判据所需证据**，无法判定 ⇒ 本批**未写任何文件**，"
              "相关档位保持「待测试」：")
        for f, tier, unks in refusals:
            print(f"  ✗ {f}（{tier}）")
            for u in unks:
                print(f"      - {u}")
        print("  （补齐上述证据后重跑；⛔ 不得以缺数据判「通过」）")
        return 1

    # ---- ② 合并（整表深拷贝；只覆盖处理集；未处理档原样保留） ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_matrix = copy.deepcopy(matrix)
    new_rows = {r["tier"]: r for r in new_matrix["rows"]}
    filled, ledger_rows = [], []
    for f, doc, status, fails, unks in valid:
        tier = doc["tier"]
        src_sha = sha256_file(f)
        row = new_rows[tier]
        row["status"] = status
        row["peak_mem"] = _s(doc["peak_mem"])
        row["steps_actual"] = fmt_steps(int(doc["steps_actual"]))
        row["elapsed"] = fmt_elapsed(doc["elapsed"])
        row["throughput"] = _s(doc["throughput"])
        row["effect"] = _s(doc["effect"])
        row["note"] = compose_note(doc["note"], status, fails, unks)
        filled.append(tier)
        ledger_rows.append(f"| {now} | `{tier}` | `{f.name}` | `{src_sha[:12]}` | {status} | 见下 | "
                           f"回填 6 列 + 状态（按四条判据现算） |")

    # ★ 保真断言：未处理的档必须与原材料逐字段相等（部分回填不得污染其它档）
    drift = untouched_drift(orig_rows, new_rows, set(filled))
    if drift:
        print(f"FAIL-CLOSED：未回填档出现漂移 {drift[:10]} ⇒ 拒绝落盘（本次处理集 {sorted(filled)}）")
        return 1

    unfilled = [t for t, r in new_rows.items()
                if all(r.get(k) in EMPTY_VALUES for k in DATA_COLUMNS)]
    new_matrix["meta"]["backfill"] = {
        "last_run": now,
        "filled_tiers": sorted(filled),
        "unfilled_count": len(unfilled),
        "note": "仅脚本回填；未判定档位保持「待测试」；结果 json 不得自带 status；缺字段/缺判据证据 fail-closed",
    }

    print(f"待回填 {len(files)} 份；本次写入 {len(filled)} 档 {sorted(filled)}；"
          f"状态 {sorted({s for _f, _d, s, _a, _b in valid})}；仍未回填 {len(unfilled)} 档（保持「待测试」）")

    if a.dry_run:
        print("--dry-run：未写任何文件。")
        return 0

    # ---- ③④ 先写临时件 ⇒ render_matrix 渲染 ⇒ verify 逐字节校验 ⇒ PASS 才落盘 ----
    tmp = Path(tempfile.mkdtemp(prefix=".backfill-tmp-", dir=str(data_out.parent)))
    try:
        tmp_data, tmp_html = tmp / "data.json", tmp / "out.html"
        tmp_data.write_text(json.dumps(new_matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # ★ 缺陷 2 的修法：回填后**直接用 render_matrix.py** 渲染（与现役生成链一致），
        #   ⛔ 不走 build_official 的重渲染路径（其产物 meta 无 enumeration ⇒ 渲染 fail-closed）。
        rc, out = run_cmd(["render_matrix.py", "--template", str(template),
                           "--data", str(tmp_data), "--out", str(tmp_html)])
        if rc != 0:
            print(f"FAIL: 渲染失败\n{out}")
            return 1
        rc, out = run_cmd(["verify_matrix.py", "--template", str(template),
                           "--data", str(tmp_data), "--html", str(tmp_html)])
        print(out.strip())
        if rc != 0:
            print("FAIL: 校验未通过 ⇒ **不落盘**（交付件与数据保持原样）")
            return 1

        # ★★ 落盘前断言：**本次处理的每一档必须出现在台账中**（⛔ 不再静默漏行）
        _missing_rows = ledger_covers(ledger_rows, filled)
        if _missing_rows:
            print(f"FAIL-CLOSED：以下档已处理但**未生成台账行** ⇒ 拒绝落盘：{_missing_rows}")
            return 1

        for p in (data_out, html_out, ledger_out):
            p.parent.mkdir(parents=True, exist_ok=True)
        # ★★ §22 落盘 + 写范围硬闸门：写完后**立刻**比「本轮副本 vs 基准件」，
        #    任何差异越出「第 4 章实验数据单元格文本」面 ⇒ **回滚本次写入**并 rc≠0。
        gate_rc, gate_out = apply_with_scope_gate(
            tmp_data=tmp_data, tmp_html=tmp_html, data_out=data_out, html_out=html_out)
        print(gate_out.strip())
        if gate_rc != 0:
            return gate_rc
        ledger_rows = [r.replace("| 见下 |", "| PASS |") for r in ledger_rows]
        try:
            append_ledger(ledger_out, ledger_rows)
        except Exception as exc:                       # noqa: BLE001 —— 审计件写失败不得静默
            print(f"FAIL: 数据/人读件已落盘，但台账写入失败 {ledger_out}：{exc!r} ⇒ 请修复后重跑"
                  "（重跑幂等，会补齐台账并去重）")
            return 1
        print(f"OK: 回填 {len(filled)} 档 ⇒ {data_out} / {html_out}；台账 +{len(ledger_rows)} 行 ⇒ {ledger_out}")
        print(f"    data sha256 = {sha256_file(data_out)}")
        print(f"    html sha256 = {sha256_file(html_out)}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- §22 落盘 + 写范围闸门 + 回滚（唯一实现）----------------------------------
def apply_with_scope_gate(*, tmp_data: Path, tmp_html: Path, data_out: Path, html_out: Path) -> tuple[int, str]:
    """把临时产物落盘，并用 `check_fill_scope.py` 机核「**只许改第 4 章实验数据单元格文本**」。

    ★ 事务语义（⛔ 不留半写状态）：先记两件在盘原状（字节 + 权限；不存在记 None）⇒ 落盘
      ⇒ 跑闸门 ⇒ 闸门 rc≠0 ⇒ **原样回滚**（恢复字节与权限；本次新建的则删掉）⇒ 返回 rc=1。
    ★ 闸门的基准永远是**基准件**（`paths.baseline_html()`），⛔ 不是"上一版副本"：
      否则连续两轮填表会把上一轮的越界当成新基线悄悄放行。
    返回 `(rc, 输出文本)`；rc=0 = 已落盘且闸门通过。
    """
    prev: dict[Path, tuple[bytes | None, int | None]] = {}
    for p in (data_out, html_out):
        _paths.assert_not_baseline_write(p, who="backfill_matrix.py")   # 防御纵深（main 已先拒）
        prev[p] = (p.read_bytes(), p.stat().st_mode & 0o777) if p.exists() else (None, None)

    def _restore() -> None:
        for p, (blob, mode) in prev.items():
            if blob is None:
                p.unlink(missing_ok=True)
                continue
            p.write_bytes(blob)
            if mode is not None:
                p.chmod(mode)

    for _src, _dst in ((tmp_data, data_out), (tmp_html, html_out)):
        if Path(_src).resolve() != Path(_dst).resolve():        # 同路径 = 已在位（演练/幂等）
            shutil.copyfile(_src, _dst)
    rc, out = run_cmd(["check_fill_scope.py", "--base", str(_paths.baseline_html()),
                       "--filled", str(html_out), "--data", str(data_out)])
    out = out.strip()
    if rc == 0:
        return 0, out + f"\nOK: §22 写范围闸门通过 ⇒ {data_out} / {html_out}"
    _restore()
    return 1, (out + f"\nFAIL-CLOSED：§22 写范围闸门 rc={rc} ⇒ **已回滚本次写入**"
                    f"（{data_out} / {html_out} 恢复为写前状态，未留半写；台账未追加）")


def _self_test():
    """★ 正/反向用例（⛔ 不依赖外部文件、不写任何东西）。"""
    # ★ 真实路径探针：用仓内在盘数据件的 `criteria_caliber.threshold` 走一遍
    #   `frozen_map_from_caliber()` / `teacher_line_pp()` —— 判据改名/散文改写时，合成字典看不见，
    #   而真实数据下四算法会全被读成 None ⇒ 合法跑次被判「不可判定」而拒收。此探针就是那类漏改的机核。
    probe_errs = []
    names_ok = False
    try:
        official = ROOT / "docs" / "capability-matrix.data.json"
        real_th = (json.loads(official.read_text(encoding="utf-8"))
                   .get("meta", {}).get("criteria_caliber", {}).get("threshold", {}))
        real_bm = real_th.get("by_method") or {}
        names_ok = all(str(e.get("criterion")) in KNOWN_CRITERIA
                       for e in real_bm.values() if isinstance(e, dict))
        for m, pp in frozen_map_from_caliber(real_th).items():
            if pp is None:
                probe_errs.append((m, "读成 None（判据名不认识/未冻结）"))
        if teacher_line_pp(real_th) is None:
            probe_errs.append(("opd.teacher", "教师线解析不到"))
    except Exception as exc:                      # noqa: BLE001 —— 取不到真实件也算不通过
        probe_errs.append(("<读取>", repr(exc)))

    fm = {"sft": 10.0, "cpt": 0.0, "opd": 5.0, "graspo": 5.0}
    th_fixture = {"status": "已冻结", "by_method": {
        "sft": {"criterion": "gain_pp", "value": 10.0},
        "graspo": {"criterion": "gain_pp", "value": 5.0},
        "cpt": {"criterion": "non_inferiority", "value": 0.0},
        "opd": {"criterion": "gain_pp", "value": 5.0,
                "teacher": "教师 − 学生未训练模型 ≥ +10.0 个百分点"}}}
    rows = {"M001": {"tier": "M001", "model": "9B", "algo": "SFT", "mode": "LoRA",
                     "backend": "ms-swift", "parallel": "nproc_per_node=1（1 卡）",
                     "lr": "2.0e-05", "derived_steps": "400", "train_method": "sft"},
            "M068": {"tier": "M068", "model": "27B", "algo": "OPD", "mode": "LoRA",
                     "backend": "ms-swift", "parallel": "nproc_per_node=4（4 卡）",
                     "lr": "2.0e-05", "derived_steps": "10", "train_method": "opd"}}

    def doc_fixture(**over):
        d = {
            "schema": "matrix-run/v1", "tier": "M001",
            "params": {"model": "9B", "algo": "SFT", "mode": "LoRA", "backend": "ms-swift",
                       "parallel": "nproc_per_node=1（1 卡）", "lr": "2.0e-05"},
            "run_result": {"exit_code": 0, "epochs_completed": 4, "epochs_planned": 4,
                           "final_ckpt_loadable": True, "nan_or_inf_detected": False},
            "steps_actual": 400, "elapsed": 3720,
            "peak_mem": "42.1 GiB（rank0 max_allocated）",
            "throughput": "3.2 samples/s", "effect": "after 27.00% / base 15.00%；Δ = +12.00 pp",
            "note": "单卡复现；数据版本 v1",
            "run_path": "runs/M001/v0-20261022-101010/checkpoint-400",
            "provenance": {"produced_by": "self-test", "command": "self-test",
                           "evidence_paths": ["runs/M001/stdout.log"]},
            "eval": {"after_acc": 0.27, "base_acc": 0.15},
            "checks": {"algo_checks_passed": True, "inference_parseable": True},
        }
        d.update(over)
        return d

    def j(doc, tier="M001"):
        return judge_row(doc, rows[tier], fm, teacher_line_pp(th_fixture))

    # ★ CPT 的 loss 夹具：DECLINE = 正常下降（首尾窗口均值严格下降）／RISE = 反向（阳性对照）
    decline = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1]
    rise = list(reversed(decline))

    def cpt_ev(series):
        """CPT 档的评测读数（`eval.loss_curve` 的唯一形态：首/末/最小 + **逐步序列**）。"""
        return {"after_acc": 0.20, "base_acc": 0.20,
                "loss_curve": {"n": len(series), "first": series[0], "last": series[-1],
                               "min": min(series), "series": list(series)}}

    base = doc_fixture()
    cases = [
        ("★ 真实数据件：四算法通过线 + OPD 教师线都能现读（⛔ 不是假 FAIL）",
         not probe_errs and all(v is not None for v in frozen_map_from_caliber(real_th_raw()).values())),
        ("★ 真实数据件的判据名 ⊆ 真相源 KNOWN_CRITERIA（第四份副本会在此报红）",
         names_ok and _known_criteria_selfcheck() is None),
        ("① 结果 json 自带 status ⇒ **必须拒收**（不静默忽略、不悄悄采纳）",
         any("不得含 `status`" in e for e in validate_run(dict(doc_fixture(), status="通过"), "M001", rows))),
        ("② 缺 steps_actual ⇒ **必须拒收**（该列必填）",
         any("steps_actual" in e for e in validate_run(
             {k: v for k, v in base.items() if k != "steps_actual"}, "M001", rows))),
        ("③ 缺 elapsed ⇒ **必须拒收**（该列必填）",
         any("elapsed" in e for e in validate_run(
             {k: v for k, v in base.items() if k != "elapsed"}, "M001", rows))),
        ("④ 某列写成占位符 `—` ⇒ **必须拒收**（⛔ 不许静默补空 / 半填）",
         any("peak_mem" in e for e in validate_run(dict(base, peak_mem="—"), "M001", rows))),
        ("⑤ 身份错位（model 不符）⇒ **必须拒收**",
         any("params.model" in e for e in validate_run(
             dict(base, params=dict(base["params"], model="27B")), "M001", rows))),
        ("⑥ 四条全过 ⇒ 判「通过」", j(base)[0] == "通过"),
        ("⑦ 效果未达线 ⇒ 判「不通过」且点名判据②",
         (lambda r: r[0] == "不通过" and any("判据②" in x for x in r[1]))(
             j(dict(base, eval={"after_acc": 0.18, "base_acc": 0.15})))),
        ("⑧ 缺 inference 证据 ⇒ **不得判通过**（判据④ 不可判定 ⇒ judge 返回 None）",
         (lambda r: r[0] is None and any("判据④" in x for x in r[2]))(j(
             dict(base, checks={"algo_checks_passed": True})))),
        ("⑨ 缺 eval 读数（且无其它失败）⇒ **不得判通过**",
         (lambda r: r[0] is None and any("判据②" in x for x in r[2]))(
             j({k: v for k, v in base.items() if k != "eval"}))),
        ("⑩ 退出码 ≠ 0 ⇒ 判「不通过」（即使评测读数缺失，合取已假）",
         (lambda r: r[0] == "不通过" and any("退出码" in x for x in r[1]))(
             j(dict({k: v for k, v in base.items() if k != "eval"},
                    run_result={"exit_code": 1, "epochs_completed": 4, "epochs_planned": 4,
                                "final_ckpt_loadable": False,
                                "nan_or_inf_detected": False})))),
        ("⑪ ★ 实跑步数 ≠ 计划步数**但 epoch 跑满** ⇒ 仍判「通过」（步数只作台账，⛔ 不作门槛）",
         (lambda d: judge_run_complete(d, rows["M001"])[0] == "pass"
          and "台账" in judge_run_complete(d, rows["M001"])[1])(dict(base, steps_actual=399))),
        ("⑪b ★ 阳性对照：epoch 未跑满（3/4）⇒ 判「不通过」且点名 epoch",
         (lambda r: r[0] == "不通过" and any("epoch" in x for x in r[1]))(
             j(dict(base, run_result=dict(base["run_result"], epochs_completed=3))))),
        ("⑪c 缺 epochs_completed ⇒ **不得判定**（unknown；⛔ 不回落步数——步数相等也不行）",
         (lambda r: r[0] is None and any("epochs_completed" in x for x in r[2]))(
             j(dict(base, run_result={k: v for k, v in base["run_result"].items()
                                      if k != "epochs_completed"})))),
        ("⑪d final_ckpt_loadable = null ⇒ **不得判定**（unknown；缺证据 ≠ 失败）",
         (lambda r: r[0] is None and any("final_ckpt_loadable" in x for x in r[2]))(
             j(dict(base, run_result=dict(base["run_result"], final_ckpt_loadable=None))))),
        ("⑪e nan_or_inf_detected = null ⇒ **不得判定**（unknown，fail-closed）",
         (lambda r: r[0] is None and any("nan_or_inf_detected" in x for x in r[2]))(
             j(dict(base, run_result=dict(base["run_result"], nan_or_inf_detected=None))))),
        ("⑪f 校验器：run_result 缺 epochs_completed ⇒ **必须拒收**",
         any("epochs_completed" in e for e in validate_run(
             dict(base, run_result={k: v for k, v in base["run_result"].items()
                                    if k != "epochs_completed"}), "M001", rows))),
        ("⑫ CPT：loss 正常下降 ∧ after == base ⇒ 通过（Δ = +0.00 pp ≥ +0.0 pp）",
         judge_effect(dict(base, eval=cpt_ev(decline)),
                      {"train_method": "cpt"}, fm, None)[0] == "pass"),
        ("⑬ ★ 阳性对照：CPT 的 loss **未下降** ⇒ 不通过",
         judge_effect(dict(base, eval=cpt_ev(rise)),
                      {"train_method": "cpt"}, fm, None)[0] == "fail"),
        ("⑬b CPT：缺 loss 读数（无 loss_curve）⇒ **不得判定**（unknown，⛔ 不静默跳过）",
         judge_effect(dict(base, eval={"after_acc": 0.20, "base_acc": 0.20}),
                      {"train_method": "cpt"}, fm, None)[0] == "unknown"),
        ("⑬c CPT：loss 下降但 Δ 低于通过线 ⇒ 不通过（loss 支不得覆盖 Δ 支）",
         judge_effect(dict(base, eval=dict(cpt_ev(decline), after_acc=0.19, base_acc=0.20)),
                      {"train_method": "cpt"}, fm, None)[0] == "fail"),
        ("⑭ OPD 学生线达标但教师 − 学生未训练不足 ⇒ 不通过且点名教师线",
         (lambda d: judge_effect(d, rows["M068"], fm, teacher_line_pp(th_fixture))[0] == "fail"
          and "教师线" in judge_effect(d, rows["M068"], fm, teacher_line_pp(th_fixture))[1])(
             dict(base, tier="M068", steps_actual=10,
                  eval={"after_acc": 0.30, "base_acc": 0.20, "teacher_after_acc": 0.25}))),
        ("⑮ OPD 学生线与教师线双达标 ⇒ 通过",
         judge_effect(dict(base, tier="M068", steps_actual=10,
                           eval={"after_acc": 0.30, "base_acc": 0.20, "teacher_after_acc": 0.35}),
                      rows["M068"], fm, teacher_line_pp(th_fixture))[0] == "pass"),
        ("⑯ OPD 缺 teacher_after_acc ⇒ 不得判定（unknown）",
         judge_effect(dict(base, eval={"after_acc": 0.30, "base_acc": 0.20}),
                      rows["M068"], fm, teacher_line_pp(th_fixture))[0] == "unknown"),
        ("⑰ 教师线数值**从数据件现解析**（散文 ≥ +10.0 个百分点 ⇒ 10.0）",
         teacher_line_pp(th_fixture) == 10.0),
        ("⑱ 教师线散文缺失 ⇒ None（⛔ 不回落常量）", teacher_line_pp({"by_method": {"opd": {}}}) is None),
        ("⑲ 未冻结的通过线 ⇒ 判据② 不可判定（⛔ 不据此判过）",
         judge_effect(base, rows["M001"], frozen_map_from_caliber({"status": "未冻结", "by_method": {
             "sft": {"criterion": "gain_pp", "value": 10.0}}}), None)[0] == "unknown"),
        ("⑳ 备注：不通过时未过项与未判定项必须写进备注",
         "判据① 未过" in compose_note("条件 X", "不通过", ["判据① 未过：退出码 1 ≠ 0"], ["判据② 缺证据：x"])),
        ("㉑ 备注：通过时只保留跑次自报条件（不追加判据结论）",
         compose_note("条件 X", "通过", [], []) == "条件 X"),
        ("㉒ 耗时列形态：`总时长 N s（约 X h）`",
         fmt_elapsed(3720) == "总时长 3720 s（约 1.03 h）"),
        ("㉓ 实跑步数列形态：`N 步`", fmt_steps(400) == "400 步"),
        ("㉔ `--only M777` ⇒ **必须报错**（非本矩阵档号）",
         bool(resolve_only("M777", rows, ["M001"])[1])),
        ("㉕ `--only M001` 但结果目录无该档 json ⇒ **必须报错**（⛔ 静默 rc=0）",
         bool(resolve_only("M001", rows, ["M068"])[1])),
        ("㉖ 结果目录含非本矩阵档号 ⇒ **必须报错**",
         unknown_tier_stems(["M001", "M777"], rows) == ["M777"]),
        ("㉗ 未回填档漂移检测：**未处理**的档被改动即被点名", untouched_drift(
            {"M001": {"a": 1}, "M002": {"b": 2}},
            {"M001": {"a": 9}, "M002": {"b": 9}}, {"M001"}) == ["M002"]),
        ("㉘ 未回填档保真：只覆盖处理集 ⇒ 0 漂移", untouched_drift(
            {"M001": {"a": 1}, "M002": {"b": 2}},
            {"M001": {"a": 9}, "M002": {"b": 2}}, {"M001"}) == []),
        ("㉙ 幂等：同一份 doc 判两次，结论一致（状态列不受时钟/顺序影响）",
         j(base)[0] == j(base)[0] == "通过" and compose_note("x", "通过", [], []) == compose_note("x", "通过", [], [])),
        ("㉚ 身份规范化：「全参（full-parameter）」≡「全参」", norm_ident("全参（full-parameter）") == norm_ident("全参")),
        ("㉛ ★ 负例：「仅视觉」/「LoRA」**不得**等同于「全参」",
         all(norm_ident(x) != norm_ident("全参") for x in ("仅视觉", "LoRA"))),
        ("㉜ ★ 负例：`native` ≠ `ms-swift`；`dp_size=2` ≠ `dp_size=1`",
         all(norm_ident(a) != norm_ident(b) for a, b in (("native", "ms-swift"), ("dp_size=2", "dp_size=1")))),
        ("㉝ 并行：表达面不同但维度等价 ⇒ 放行",
         not compare_parallel("TP=1 PP=1 DP=1（1 卡；总卡数 1）", "nproc_per_node=1（1 卡）")),
        ("㉞ 并行：TP/PP 不符 ⇒ FAIL",
         bool(compare_parallel("TP=2 PP=1 DP=1（2 卡）", "TP=1 PP=2 DP=1（2 卡）"))),
        ("㉟ LoRA 目标语义不同（122 vs 358）⇒ FAIL",
         norm_param("lora_target", "language_all_linear + vision_common（122 件）") !=
         norm_param("lora_target", "ms-swift 实测 358 件（语言 248 + 视觉 110）")),
        ("㊱ 实例数优先：表「（122 件：…）｜（358 实例；122 为签名/模式条目数）」⇒ 取 358",
         norm_param("lora_target",
                    "language_all_linear + vision_common（122 件：language 12 + visual 110）｜native："
                    "canonical preset 名（358 实例；122 为签名/模式条目数）") == 358),
        ("㊲ 数据子集规模不同（100 vs 200）⇒ FAIL",
         norm_param("data_subset", "前 100 条") != norm_param("data_subset", "前 200 条")),
        ("㊳ 台账去重：同 (档号, sha) 两次 ⇒ 1 行 + ×2",
         "×2" in dedup_ledger("| t1 | `M001` | `a.json` | `aaaa` | 通过 | PASS | n |\n"
                              "| t2 | `M001` | `a.json` | `aaaa` | 通过 | PASS | n |")),
        ("㊴ ★ 台账去重：**不同 sha ⇒ 必须 2 行**（真增量不得被吞）",
         dedup_ledger("| t1 | `M001` | `a.json` | `aaaa` | 通过 | PASS | n |\n"
                      "| t2 | `M001` | `a.json` | `bbbb` | 通过 | PASS | n |").count("`M001`") == 2),
        ("㊵ ★ 台账覆盖率：处理了但未落行 ⇒ **必须拒**（fail-closed）",
         ledger_covers(["| t | `M001` | `a.json` | `sha` | 通过 | PASS | n |"], ["M001", "M002"]) == ["M002"]),
        ("㊶ by_method 未冻结 ⇒ 全 None（⛔ 不据此判定）",
         frozen_map_from_caliber({"status": "未冻结", "by_method": {
             "sft": {"criterion": "gain_pp", "value": 10.0}}})
         == {"sft": None, "graspo": None, "cpt": None, "opd": None}),
        ("㊷ by_method 解析：逐算法取值（sft10 / graspo5 / cpt0 / opd5）",
         frozen_map_from_caliber(th_fixture) == {"sft": 10.0, "graspo": 5.0, "cpt": 0.0, "opd": 5.0}),
        # ★ 2026-10-25 §22 写范围闸门（主席令「填写报告的脚本只能填写第四章表格里面的实验内容数据」）
        ("㊸ 填表白名单与回填器 6 列**同源**（数据件 data_keys ⇒ ⛔ 不写死第二份列名）",
         _fill_scope_columns() == DATA_COLUMNS),
        ("㊹ ★ 写范围闸门：改第 3 章文字 ⇒ **必须判违规**（同一切面内，只改结果单元格文本 ⇒ 放行）",
         _fill_scope_probe() == (True, True)),
    ]
    ok = True
    for name, passed in cases:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'}（{sum(1 for _, x in cases if x)}/{len(cases)}）")
    return 0 if ok else 1


def _fill_scope_columns() -> tuple[str, ...]:
    """自证辅助：`check_fill_scope.py` 从数据件 `data_keys` 现读的列白名单（⛔ 不写死）。"""
    import importlib.util as _ilu

    _path = TOOL_DIR / "check_fill_scope.py"
    _spec = _ilu.spec_from_file_location("_fill_scope_for_selftest", _path)
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    return tuple(_mod.load_result_columns(ROOT / "docs" / "capability-matrix.data.json"))


def _fill_scope_probe() -> tuple[bool, bool]:
    """自证辅助（**纯内存**，零文件写入）：返回 (改章节文字被判违规, 只改结果单元格文本被放行)。

    夹具是一个最小矩阵表骨架 —— 闸门的判定逻辑与真实报告同一条代码路径。
    """
    head = ('<table class="matrix"><thead>'
            '<tr><th class="g-base col-tier col-start" data-col="tier">档号</th>'
            '<th class="g-data" data-col="peak_mem">显存峰值(GiB)</th></tr></thead><tbody>\n')
    row = ('<tr><td class="g-base col-tier" data-col="tier">M001</td>'
           '<td class="g-data col-start" data-col="peak_mem">—</td></tr>\n')
    tail = "</tbody></table>\n</body>\n"
    base_text = "<h3>第 3 章</h3>\n" + head + row + tail
    ok_text = "<h3>第 3 章</h3>\n" + head + row.replace(">—<", ">63.4<") + tail
    bad_text = "<h3>第 3 章（改）</h3>\n" + head + row + tail
    kw = dict(result_columns=DATA_COLUMNS, status_vocab=frozenset({"待测试", "通过", "不通过"}))
    bad_hit = bool(check_scope_fn(base_text, bad_text, **kw))
    ok_clean = check_scope_fn(base_text, ok_text, **kw) == []
    return bad_hit, ok_clean


def check_scope_fn(*args, **kwargs):
    """自证辅助：`check_fill_scope.check_scope` 的惰性取用（⛔ 不在模块顶层 import 造环）。"""
    import importlib.util as _ilu

    _path = TOOL_DIR / "check_fill_scope.py"
    if "_fill_scope_for_selftest" not in sys.modules:
        _spec = _ilu.spec_from_file_location("_fill_scope_for_selftest", _path)
        _mod = _ilu.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
    return sys.modules["_fill_scope_for_selftest"].check_scope(*args, **kwargs)


def real_th_raw():
    """测试辅助：真实数据件的 threshold 段（取不到 ⇒ 空字典，让自证用例报红）。"""
    try:
        return (json.loads((ROOT / "docs" / "capability-matrix.data.json").read_text(encoding="utf-8"))
                .get("meta", {}).get("criteria_caliber", {}).get("threshold", {}))
    except Exception:          # noqa: BLE001
        return {}


if __name__ == "__main__":
    sys.exit(main())
