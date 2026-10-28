"""矩阵备注列（人读件「实验数据」最后一格）的**说人话**口径 —— 唯一真相源。

## 为什么有这件（任务 E，2026-10-27 主席裁定）

备注列是给**不读日志的人**看的：一格只说结论。此前它被写成了「工程日志」——
1324 字里塞进了整段 traceback、正则原文、容器内绝对路径 ＋ `file:line`、`stdout.log:20`
这类日志坐标、`GPU 3` 这类硬件细节。诚实要靠**证据件**（`provenance` / `<runs>/<档号>.json` /
台账），不靠把证据原文抄进表格。

## 三条硬规则（本模块即机核口径）

1. **删**：日志坐标（`stdout.log:N`）、宿主机/容器绝对路径（`/workspace/…`、`/data/…`）、
   `file:line`、正则原文、英文帮助长句；
2. **留**：退出码、实跑步数、完成 X/Y 个 epoch、峰值显存、效果读数（after/base/Δ）、
   检查点有无、判据①②③④ 的过/未过结论；必要时可加**一句**「附注：…」（数字诚实类补充，
   如墙钟口径另有出处；由 `GRASPO_FILL_NOTE_EXTRA` 注入，仍受同一套去噪与字数口径）；
3. **报错只留摘要**：异常类名 ＋ 正文前 ≤40 字，后接「详见跑次证据」；单档备注 **≤ 200 字**。

模块同时提供两条路的**同一个渲染器**：
* 新回填：`scripts/make_result_json.py` ＋ `tools/matrix/backfill_matrix.compose_note()` 调查表式
  `render(facts)`；
* 存量规范化：`tools/matrix/normalize_notes.py` 用 `parse_legacy(note) → render(facts)`
  —— **纯文本规范化，⛔ 不重新推导任何读数**（这也是两者能收敛到同一句话的原因）。
"""

from __future__ import annotations

import re

#: 单档备注上限（字）。超过 ⇒ 丢掉最低优先级的尾巴 ＋ 追加 :data:`TAIL`。
NOTE_LIMIT = 200
#: 被省略时的收口话术（诚实：证据在跑次目录里）
TAIL = "（详见跑次证据）"
#: 报错摘要正文上限（字）
ERROR_BODY_LIMIT = 40

#: 判据编号（唯一真相源的次序；⛔ 不在别处另抄一份）
CRITERIA = ("①", "②", "③", "④")

#: 判据编号提取：`判据① 未过` / `判据③ 缺证据`
_RE_FAIL_ID = re.compile(r"判据\s*([①②③④])\s*未过")
_RE_UNK_ID = re.compile(r"判据\s*([①②③④])\s*缺证据")

#: 「不该进人读件」的形态（与 `tools/matrix/scan_matrix_leaks.py` 的词表同源同义；
#: ⛔ 这里只用于**清洗**，扫描仍以扫描器为准）
NOISE_PATTERNS = (
    r"stdout\.log:\d+",
    r"[A-Za-z0-9_./-]+\.py:\d+",
    r"/workspace/[^\s）；]*",
    r"/data/[^\s）；]*",
    r"\(\?[<:=!]?[^\s）；]{0,80}",
)
_NOISE_RE = [re.compile(p) for p in NOISE_PATTERNS]


def has_noise(text: str) -> bool:
    """文本里是否含「工程日志」形态（供单测与清洗前后对照用）。"""
    return any(r.search(str(text or "")) for r in _NOISE_RE)


def strip_noise(text: str) -> str:
    """剥掉坐标 / 路径 / 正则原文（就地替换，⛔ 不删整句）。"""
    out = str(text or "")
    for r in _NOISE_RE:
        out = r.sub("", out)
    return out


def clip_error(text, limit: int = ERROR_BODY_LIMIT) -> str:
    """报错 ⇒ 「异常类名: 正文前 ≤limit 字…」。

    例（M011 实测逐字）：`ValueError: lora.target_modules is required (a single real HF module-name
    regex; …)` ⇒ `ValueError: lora.target_modules is required`（括号里的英文帮助整段丢掉）。
    """
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return ""
    m = re.match(r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit))\s*:?\s*(.*)", t)
    name, body = (m.group(1), m.group(2)) if m else ("", t)
    # 括号内的英文帮助 / 正则 / 日志坐标：整段丢弃（半角 ＋ 全角括号都算，反复剥到稳定）
    prev = None
    while prev != body:
        prev = body
        body = re.sub(r"\([^()]*\)", "", body)
        body = re.sub(r"（[^（）]*）", "", body)
    body = strip_noise(body)
    body = re.sub(r"\s+", " ", body).strip(" ：:，,。.；;")
    if len(body) > limit:
        cut = body[:limit]
        if " " in cut:
            cut = cut[:cut.rfind(" ")]
        body = cut.rstrip(" ，,。.；;") + "…"
    text = f"{name}: {body}" if name else body
    return text.strip()


def _num(x, digits: int = 1) -> str:
    """0.46 ⇒ '46.0'（⛔ 不做四舍五入口径变更，只统一展示位数）。"""
    try:
        return f"{float(x) * 100:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def effect_phrase(facts: dict) -> str:
    """效果读数短语：`after 46.0% / base 20.0% / Δ+26.0pp`（无评测 ⇒ 不可评）。"""
    a, b, d = facts.get("after_acc"), facts.get("base_acc"), facts.get("delta_pp")
    if a is None or b is None:
        return "不可评"
    delta = d if d is not None else (float(a) - float(b)) * 100.0
    sign = "+" if float(delta) >= 0 else ""
    return f"after {_num(a)}% / base {_num(b)}% / Δ{sign}{float(delta):.1f}pp"


def _ids(ids) -> str:
    """['①','③'] ⇒ '①③'（⛔ 不加书名号/括号）。"""
    seen, out = set(), []
    for i in ids or ():
        i = str(i)
        if i in CRITERIA and i not in seen:
            seen.add(i)
            out.append(i)
    return "".join(out)


def criteria_ids(fails, unks) -> tuple[list[str], list[str]]:
    """从判据原因串里取编号：`判据① 未过：…` ⇒ ['①']；`判据③ 缺证据：…` ⇒ ['③']。"""
    f_ids, u_ids = [], []
    for f in fails or ():
        m = _RE_FAIL_ID.search(str(f))
        if m:
            f_ids.append(m.group(1))
    for u in unks or ():
        m = _RE_UNK_ID.search(str(u))
        if m:
            u_ids.append(m.group(1))
    return f_ids, u_ids


def render(facts: dict) -> str:
    """按**优先级**拼备注，超 :data:`NOTE_LIMIT` 时从尾部丢最低优先级的从句。

    `facts` 允许缺键（缺 = 这一项不写）。键见模块 docstring 与 `parse_legacy()`。
    """
    clauses: list[str] = []
    if facts.get("extra"):
        # 附注带显式前缀：本列是人读件，读者要能一眼看出「这句是补充说明」而不是读数；
        # 同时让 `parse_legacy` 能把它**往返解析**回来（否则 `compose_note` 重渲染会把它丢掉）。
        clauses.append(f"附注：{facts['extra']}")
    if facts.get("crash"):
        clauses.append(f"训练崩溃，未产生有效训练：{facts['crash']}")
    if facts.get("exit_derived"):
        raw, new = facts["exit_derived"]
        clauses.append(f"退出码 {new}（按证据重派生，原始 {raw}）")
    elif facts.get("exit_code") is not None:
        clauses.append(f"退出码 {facts['exit_code']}")
    else:
        clauses.append("退出码缺")
    if isinstance(facts.get("steps"), int):
        src = str(facts.get("steps_src") or "").strip()
        clauses.append(f"实跑步数 {facts['steps']}（{src}）" if src else f"实跑步数 {facts['steps']}")
    elif "steps" in facts:
        clauses.append("实跑步数未知")
    done, planned = facts.get("epochs_done"), facts.get("epochs_planned")
    if isinstance(done, int) and isinstance(planned, int):
        clauses.append(f"完成 {done}/{planned} 个 epoch")
    elif isinstance(planned, int):
        clauses.append(f"完成轮数未知（计划 {planned} 个 epoch）")
    ckpt = facts.get("ckpt")
    if ckpt:
        clauses.append(f"检查点：{ckpt}")
    if facts.get("naninf"):
        clauses.append(f"数值异常：{facts['naninf']}")
    if facts.get("peak"):
        clauses.append(f"峰值显存：{facts['peak']}")
    clauses.append(f"效果读数：{effect_phrase(facts)}")
    if facts.get("loss"):
        clauses.append(facts["loss"])
    fail_s, unk_s = _ids(facts.get("fail_ids")), _ids(facts.get("unk_ids"))
    if facts.get("judged") or fail_s or unk_s:
        parts = []
        passed = "".join(i for i in CRITERIA if i not in set(fail_s + unk_s))
        if passed:
            parts.append(f"{passed} 过")
        if fail_s:
            parts.append(f"{fail_s} 未过")
        if unk_s:
            parts.append(f"{unk_s} 未判定")
        clauses.append("判据结论：" + "，".join(parts))
    for w in facts.get("warns") or ():
        clauses.append(str(w))
    gaps = facts.get("gaps")
    if isinstance(gaps, int) and gaps > 0:
        clauses.append(f"证据缺口 {gaps} 项")

    # ── 上限控制：超了就从**尾部**丢（尾部 = 优先级最低），并追加收口话术 ─────
    kept, dropped = list(clauses), False
    while kept and len("；".join(kept) + TAIL) > NOTE_LIMIT:
        kept.pop()
        dropped = True
    text = strip_noise("；".join(kept))
    if dropped or facts.get("crash") or facts.get("gaps"):
        text += TAIL
    return text


# ── 存量规范化：老备注（工程日志体）⇒ facts ⇒ render（⛔ 不重新推导读数）───────────
_CLAUSE_KEYS = ("退出码", "实跑步数", "完成轮数", "计划", "检查点", "数值异常", "峰值显存",
                "效果读数", "训练 loss 序列", "未过判据", "未判定项", "证据缺口", "训练崩溃")


def split_clauses(note: str) -> list[str]:
    """按**括号深度 0** 的「；」切分（原因串里的「；」不切）。"""
    out, buf, depth = [], "", 0
    for ch in str(note or ""):
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
        if ch == "；" and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return [c.strip() for c in out if c.strip()]


def parse_legacy(note: str) -> dict:
    """把**已落盘**的老备注解析成 facts（只读文本，⛔ 不碰任何证据件/读数）。"""
    text = str(note or "")
    facts: dict = {}
    fail_from_terse: list[str] = []
    unk_from_terse: list[str] = []
    clauses = split_clauses(text)
    for c in clauses:
        if c.startswith("附注"):
            facts["extra"] = c.split("：", 1)[-1].strip()
        elif c.startswith("训练崩溃，未产生有效训练"):
            body = c.split("：", 1)[1] if "：" in c else ""
            body = body.split("exit_code=", 1)[-1]
            body = re.sub(r"^[0-9]+，", "", body)          # 去掉 "exit_code=1，" 的后半
            facts["crash"] = clip_error(body)
        elif c.startswith("退出码按证据重派生"):
            m = re.search(r"原始\s*(\d+)\s*→\s*(\d+)", c)
            if m:
                facts["exit_derived"] = (int(m.group(1)), int(m.group(2)))
        elif c.startswith("退出码"):
            m2 = re.search(r"退出码\s*(\d+)（按证据重派生，原始\s*(\d+)）", c)
            if m2:
                facts["exit_derived"] = (int(m2.group(2)), int(m2.group(1)))
                facts["exit_code"] = int(m2.group(1))
                continue
            m = re.search(r"退出码\s*(?:缺)?\s*(\d+)", c)
            facts["exit_code"] = int(m.group(1)) if m else None
        elif c.startswith("实跑步数"):
            m = re.search(r"实跑步数\s*(\d+)", c)
            if m:
                facts["steps"] = int(m.group(1))
                src = re.search(r"（([^（）]*)）", c)
                if src and not has_noise(src.group(1)):
                    why = re.sub(r"[；，]?\s*计划\s*[0-9]+\s*步", "", src.group(1)).strip()
                    why = re.sub(r"\s*[；，]\s*$", "", why).strip()
                    if why:
                        facts["steps_src"] = why
        elif c.startswith(("完成轮数", "完成 ")):
            m = re.search(r"(?:完成轮数|完成)\s*(\d+)\s*/\s*(\d+)", c)
            if m:
                facts["epochs_done"], facts["epochs_planned"] = int(m.group(1)), int(m.group(2))
        elif c.startswith("计划"):
            m = re.search(r"计划\s*(\d+)\s*个\s*epoch", c)
            if m and "epochs_planned" not in facts:
                facts["epochs_planned"] = int(m.group(1))
        elif c.startswith("检查点"):
            if "不可加载" in c or "无任何 checkpoint" in c or "不存在可加载" in c:
                facts["ckpt"] = "不可加载"
            elif "可加载" in c or "可解析" in c:
                facts["ckpt"] = "可加载"
            else:
                facts["ckpt"] = "未取证"
        elif c.startswith("数值异常"):
            tail = c.split("：", 1)[-1].strip()
            if tail in ("无", "无 NaN/Inf") or "未出现 NaN" in c or "无 NaN" in c:
                facts["naninf"] = "无"
            elif "检测到" in c or "出现 NaN" in c:
                facts["naninf"] = "有 NaN/Inf"
            else:
                facts["naninf"] = "未取证"
        elif c.startswith("峰值显存"):
            m = re.search(r"([0-9][0-9.]*)\s*(MiB|GiB)", c)
            if m:
                facts["peak"] = f"{m.group(1)} {m.group(2)}"
        elif c.startswith("训练 loss 序列"):
            m = re.search(r"(\d+)\s*个点", c)
            facts["loss"] = f"训练 loss 序列：{m.group(1)} 个点" if m else "训练 loss 序列：有"
        elif c.startswith("判据结论"):
            body = c.split("：", 1)[-1]
            for seg in re.split(r"[，,]", body):
                ids = [ch for ch in seg if ch in CRITERIA]
                if "未过" in seg:
                    fail_from_terse.extend(ids)
                elif "未判定" in seg:
                    unk_from_terse.extend(ids)
                elif "过" in seg:
                    facts["judged"] = True
        elif c.startswith("效果读数") and "after" in c:
            facts.update(parse_effect_column(c))
        elif c.startswith("证据缺口") and re.fullmatch(r"证据缺口\s*\d+\s*项", c.strip()):
            facts["gaps"] = int(re.search(r"(\d+)", c).group(1))
        elif c.startswith("证据缺口"):
            m = re.search(r"证据缺口[^：]*：?(.*)", c)
            n = len([x for x in str(m.group(1) if m else "").split("；") if x.strip()])
            facts["gaps"] = max(1, n)
    # 判据编号：整段扫（老形态「判据① 未过 / 缺证据」＋ 新形态「未过判据：①③」）
    fail_ids = [m.group(1) for m in _RE_FAIL_ID.finditer(text)]
    unk_ids = [m.group(1) for m in _RE_UNK_ID.finditer(text)]
    for c in clauses:
        if c.startswith(("未过判据", "未判定")):
            terse = [ch for ch in c.split("：", 1)[-1] if ch in CRITERIA]
            (fail_ids if c.startswith("未过判据") else unk_ids).extend(terse)
    facts["fail_ids"] = fail_ids + fail_from_terse
    facts["unk_ids"] = unk_ids + unk_from_terse
    return facts


def normalize_legacy(note: str) -> str:
    """老备注 ⇒ 新口径备注（**幂等**：对新口径再跑一次逐字不变）。"""
    return render(parse_legacy(note))


def parse_effect_column(text) -> dict:
    """从**效果列**（数据件 `effect`）取 after/base/Δ 的展示读数（⛔ 不重算，只读现成数字）。

    形态（实测逐字）：`after acc=46.0000%；base acc=20.0000%；Δ = +26.0000 pp`。
    取不到（`—` / 不可评）⇒ 返回空 dict（渲染成「不可评」）。
    """
    t = str(text or "")
    m_a = re.search(r"after\s*(?:acc\s*=\s*)?([0-9.]+)\s*%", t)
    m_b = re.search(r"base\s*(?:acc\s*=\s*)?([0-9.]+)\s*%", t)
    m_d = re.search(r"Δ\s*=?\s*([+-]?[0-9.]+)\s*pp", t)
    if not (m_a and m_b):
        return {}
    out = {"after_acc": float(m_a.group(1)) / 100.0, "base_acc": float(m_b.group(1)) / 100.0}
    if m_d:
        out["delta_pp"] = float(m_d.group(1))
    return out
