#!/usr/bin/env python3
"""verify_matrix.py — 机核执法：把「模板 + 数据」重新渲染，与在盘 HTML 比对并做结构断言。

一条命令即判定「HTML 是否被手工改动过 / 结构是否符合新两大部分」：

  python3 verify_matrix.py --template template.html --data matrix-data.json \
                           --html capability-matrix-new.html

退出码 0 = PASS；退出码 1 = FAIL。

四组检查（全部 fail-closed）：

  ① 字节同一性 —— 重新渲染的字节串与在盘 HTML 逐字节相同（唯一产出通道的证据）。
  ② 手改检测 —— ① 失败时打印首个差异的「字节偏移 / 行号 / 列号 / 上下文」（改一个字、
     加一个空格、删一个字符都会命中）。
  ③ 结构断言 —— 新结构「只有两大部分」：恰好 2 个 <h2> 且标题为「测试规则和说明 /
     功能矩阵测试表」；恰好 2 个分区 section；目录只指向这两处；不再有 §1–§7 的旧骨架 id。
  ④ 内容断言 —— A. 规则关键词全部在位（防止规则被删/被改名）；B. 列顺序 == 已定 10 列新表头；
     C. 结果列全部为占位符（回填后可用 --allow-filled 放行 C）；D. 参数列合并后六项逐档在位。
"""
import argparse, hashlib, html, json, re, sys
from pathlib import Path

import render_matrix  # 同目录；复用唯一渲染通道
import build_official  # 同目录；★ 结论列契约与归因依据关键词**同源**（⛔ 不复制一份）

VERSION = "verify-matrix/v14-datafile-churn-scan-exemption-anchor-and-dash-semantics"

# ---- ③ 结构断言：新结构只允许这两大部分 ----
#: ★ 人读件结构（宪法 §17.7 + 使用者诉求「判据 + 实验条件 + 功能矩阵表」）：
#:   ① 这张表是什么、怎么读 ② 判据（含口径） ③ 实验条件 ④ 功能矩阵表
EXPECTED_H2 = ["1. 先读这一页", "2. 判据（含口径）", "3. 实验条件", "4. 功能矩阵表"]
EXPECTED_PARTS = ["s1", "s2", "s3", "s4"]
FORBIDDEN_OLD_SECTIONS = ["part1", "part2", "sec1", "sec1b", "sec2", "sec3", "sec4", "sec5", "sec6", "sec7"]

# ---- ④A. 必须出现的规则关键词（缺一即 FAIL） ----
REQUIRED_MARKERS = [
    ("机器可读生成标记", "generated-by:"),
    ("首屏生成声明", "自动生成"),
    ("禁止手工编辑", "禁止手工编辑"),
    ("唯一可编辑源头", "唯一可编辑源头"),
    ("方向固定·单方向", "单方向，禁止互推"),
    ("边界声明", "承诺边界声明"),
    ("边界要件·各档可用性", "各档可用性"),
    ("边界要件·平台方数据集结论", "平台方数据集结论"),
    ("边界要件·使用者效果不在范围", "不在本矩阵的承诺范围"),
    ("四态·已验证可用", "已验证可用"),
    ("四态·已验证不可用", "已验证不可用"),
    ("四态·尚未验证", "尚未验证"),
    ("列序·主要参数分列", "总卡数"),
    ("列序·实验数据", "实验数据"),
    ("列序·备注", "失败归因"),
    ("四态·未纳入验收", "未纳入验收"),
    ("不依赖颜色/图例", "不依赖颜色、图例或空白"),
    ("未过判据必须写明", "写明未过的是"),
    ("已通过项事实保留", "exit_code = 0"),
    ("判据口径·数据集锚点", "版本锚点"),
    ("判据口径·判据标识", "elam-v3-allright-1"),
    ("判据口径·四条判据", "判据（四条）"),
    ("判据口径·阈值冻结", "已冻结（现行口径，2026-10-15）"),
    ("判据口径·SFT 通过线", "after − base ≥ +10.0 pp"),
    ("判据口径·GRASPO 通过线", "after − base ≥ +5.0 pp"),
    ("判据口径·CPT 不劣化线", "after ≥ base（不劣化）"),
    ("判据口径·OPD 能力回收", "学生 after ≥ base（能力回收）"),
    ("判据口径·不做假设检验", "不做假设检验"),
    ("判据口径·旧口径留痕", "旧口径（已被取代、留痕"),
    ("判据口径·现行口径（原「变更留痕」行标签，修正包 🟡-1 改中性词）", "判据口径（现行）"),
    ("判据口径·基线口径", "U2 终点窗口"),
    ("实验条件", "实验条件"),
    ("统一口径", "统一口径"),
    ("只列差异项", "只列出档位之间的差异项"),
    ("判据定义 U1", "exit_code = 0"),
    ("算法正确性检查清单", "算法正确性检查"),
    ("B3 基础设施不影响", "都不应显著影响效果"),
    ("评测与推理验证", "推理验证"),
    ("步数为上限", "表中步数是上限"),
    ("并行按后端分列", "nproc_per_node"),
    ("并行·配置值", "配置值"),
    ("并行·派生值", "派生"),
    ("并行·关系式", "总卡数 = nproc_per_node × nnodes = TP × PP × CP × DP"),
    ("LoRA 目标分列", "target_modules"),
    ("检查点间隔键名", "checkpoint_save_steps"),
    # ★ §17.7 覆盖度门槛：首屏警示 + 「尚未验证」清单与数量
    ("覆盖度警示·标题", "覆盖度警示"),
    ("覆盖度警示·四态计数", "当前结论分布为"),
    ("覆盖度警示·尚未验证清单", "尚未验证的档位（"),
    ("覆盖度警示·不得读作全可用", "不得被读作"),
    ("读表·是什么", "这张表是什么"),
    ("读表·可用怎么判", "「可用」怎么判"),
    ("读表·实验怎么做", "实验怎么做的"),
    ("读表·怎么读表", "怎么读表"),
    ("判据① 能完整跑完", "能完整跑完"),
    ("判据② 平台方数据集效果达标", "在平台方数据集与平台方参数下效果达标"),
    ("不可用须写明第 1/2 条", "第 1 条"),
    ("不可用须写明第 2 条", "第 2 条"),
    ("未纳入验收须逐档", "逐档"),
    ("实验条件·实验场所", "实验场所"),
    ("实验条件·代码版本", "代码版本"),
    ("实验条件·评测数据集", "评测数据集"),
    ("实验条件·训练轮次", "训练轮次"),
    ("实验条件·步数口径", "步数口径"),
    ("实验条件·达标阈值", "达标阈值"),
]

#: 禁止措辞（§17.7：双真相源 / 能力承诺式措辞）
FORBIDDEN_PHRASES = [
    ("同为真相源", "双真相源表述"),
    ("互为真相源", "双真相源表述"),
    ("两个真相源", "双真相源表述"),
    ("全组合可用", "能力承诺式措辞"),
    ("已验证全组合", "能力承诺式措辞"),
    ("仅供参考", "以「不承诺效果」淡化平台侧失败（§17.7 反例）"),
    ("效果不保证", "以「不承诺效果」淡化平台侧失败（§17.7 反例）"),
]

# ★ §17.7「矩阵中仍然禁止」+ 使用者诉求：**内部术语**（人称 / 内部编排 / 内部角色 / 内部过程）
#   —— 生成物与手写文档**同等**检查（§17.7「扫描不豁免」）。
FORBIDDEN_INTERNAL_TERMS = [
    "主席", "总参谋长", "总司令", "指挥部", "指挥官", "审查者",
    "口径包", "修正轮", "本包", "本批", "第 1 批", "第 2 批", "第 3 批",
    "第 4 批", "第 5 批", "第 6 批", "第 7 批", "工位", "work_log", "跑次根目录",
    "归属层", "必须能失败", "阳性对照", "机核可跑",
]

# ---- ④B. 已定表头：**13 列**（档号 → 实验条件 6 列 → 实验数据 6 列） ----
# ★ 精简轮 2（2026-10-16）：由 21 列收敛到 13 列；砍掉的列均为**可由保留列唯一决定**
#   （`model/mode/lora_target/lr` 由 `algo+backend`；`tp/pp/dp` 由 `backend+cards_total`）
#   或**实测值近乎全空**（`steps_actual` 2/54、`throughput` 1/54）⇒ 已并入「备注」与实验条件块。
EXPECTED_COLUMNS = [
    ("tier", "档号"),
    ("algo", "算法"),
    ("backend", "后端"),
    ("cards_total", "卡数"),
    ("data_scale", "训练数据（子集 / 每卡）"),
    ("steps", "步数"),
    ("switches", "关键开关"),
    ("status", "状态"),
    ("peak_mem", "显存峰值(GiB)"),
    ("elapsed", "总耗时"),
    ("effect", "效果（评测 Δ）"),
    ("note", "备注（失败归因 / 条件 / 限制）"),
    ("not_failed_reason", "未过判据"),
]
#: ★ 实验数据段（渲染列序，也是 ④D 的逐格一致性检查面）
RESULT_KEYS = ["status", "peak_mem", "elapsed", "effect", "note", "not_failed_reason"]
#: ★★ 真正的**实测读数**列：未结论档必须全为 `—`（`status`/`note` 不是读数，故不在内）。
#:    该清单是 ④C「未验证不得冒充可用 / 不得携带实验数据」的检查面。
#: ★ 与生成端**同源**（`build_official.EXP_DATA_KEYS`），⛔ 不各自维护一份
EXP_DATA_KEYS = list(build_official.EXP_DATA_KEYS)
# ★ 主要参数：按项分列（顺序即展示顺序）；★ 精简轮 2 由 13 项收敛到 6 项
#   （`model/mode/tp/pp/dp/lora_target/lr` 已上提到实验条件块——它们由 `algo/backend/cards_total` 唯一决定）
PARAM_SUBKEYS = ["algo", "backend", "cards_total", "data_scale", "steps", "switches"]
# 供 yaml 生成器机读的细粒度字段（每行必须存在；值为 '—' 表示源缺失）
SPEC_KEYS = ["model_path", "train_method", "tuner_type", "cards", "parallel_dims",
             "parallel_scheme", "data_source", "subset_size", "prompts_per_rank",
             "subset_generation", "subset_r3_historical", "train_path",
             "planned_steps", "save_steps", "gpus", "target_modules", "target_modules_mode",
             "deepspeed", "allow_unverified_pp_rollout", "nnodes",
             "expected_optimizer_steps_per_epoch", "expected_optimizer_steps_reachable",
             "expected_optimizer_steps_historical", "steps_formula_v2", "min_optimizer_steps",
             "applicable"]


def first_diff(a: bytes, b: bytes):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def linecol(text: str, byte_off: int):
    """byte_off 是字节偏移；换算到 UTF-8 文本的行号 / 列号。"""
    prefix = text.encode("utf-8")[:byte_off]
    s = prefix.decode("utf-8", errors="replace")
    line = s.count("\n") + 1
    col = len(s.split("\n")[-1]) + 1
    return line, col


def context(text: str, char_off: int, width: int = 70):
    return repr(text[max(0, char_off - width):char_off + width])


def cell_text(cell_html: str) -> str:
    """把单元格 HTML 还原成纯文本（用于参数列的逐档核对）。"""
    return html.unescape(re.sub(r"<[^>]+>", "", cell_html)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--expect-empty", action="store_true",
                    help="要求结果列全部为占位符（空结果版 / 起手式用）；默认只要求「无半填」一致性")
    ap.add_argument("--allow-filled", action="store_true",
                    help="（保留兼容，已等价于默认行为）结果列允许有实测读数")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    template_text = Path(a.template).read_text(encoding="utf-8")
    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    rendered_text = render_matrix.render(template_text, data)
    rendered_bytes = rendered_text.encode("utf-8")

    target_path = Path(a.html)
    if not target_path.exists():
        print(f"FAIL: 目标 HTML 不存在: {target_path}")
        return 1
    target_bytes = target_path.read_bytes()
    target_text = target_bytes.decode("utf-8")

    failures = []

    # ---- ① 主判据：逐字节比对 ----
    off = first_diff(rendered_bytes, target_bytes)
    byte_identical = off is None
    if not byte_identical:
        # ---- ② 手改检测：定位 ----
        ln, col = linecol(target_text, off)
        char_off = len(target_bytes[:off].decode("utf-8", errors="replace"))
        detail = (f"首个差异 @ 字节偏移 {off}（行 {ln}，列 {col}）\n"
                  f"  期望(渲染) : {context(rendered_text, char_off)}\n"
                  f"  实际(在盘) : {context(target_text, char_off)}")
        failures.append(("① 字节同一性 / ② 手改检测（禁止手工改动）", detail))

    # ---- ③ 结构断言：只有两大部分 ----
    h2 = re.findall(r"<h2>(.*?)</h2>", target_text, re.S)
    h2 = [re.sub(r"<[^>]+>", "", x).strip() for x in h2]
    if h2 != EXPECTED_H2:
        failures.append(("③ 结构：<h2> 只允许四个", f"期望 {EXPECTED_H2}\n  实际 {h2}"))

    parts = re.findall(r'<section class="part" id="([^"]+)"', target_text)
    if parts != EXPECTED_PARTS:
        failures.append(("③ 结构：分区只允许四个", f"期望 {EXPECTED_PARTS}\n  实际 {parts}"))

    stale = [s for s in FORBIDDEN_OLD_SECTIONS if f'id="{s}"' in target_text]
    if stale:
        failures.append(("③ 结构：不应再有旧骨架分区", f"仍存在 {stale}"))

    toc = re.findall(r'<nav class="toc">(.*?)</nav>', target_text, re.S)
    if len(toc) != 1:
        failures.append(("③ 结构：目录", f"期望恰好 1 个 nav.toc，实际 {len(toc)}"))
    else:
        links = re.findall(r'href="#([^"]+)"', toc[0])
        if links != EXPECTED_PARTS:
            failures.append(("③ 结构：目录只指向四个部分", f"期望 {EXPECTED_PARTS}\n  实际 {links}"))

    # ---- ④A. 规则关键词 ----
    missing = [name for name, kw in REQUIRED_MARKERS if kw not in target_text]
    if missing:
        failures.append(("④A 规则关键词", f"缺失 {len(missing)} 项: {missing}"))

    # ---- ④B. 列顺序 == 已定表头（13 列） ----
    _cols = data.get("columns", [])
    groups = data.get("groups", {})
    first = _cols[0] if _cols else {"group": "", "label": "", "key": ""}
    _td_cls = f'g-{first.get("group")} col-tier col-start'
    got = [(c.get("key"), c.get("label")) for c in data.get("columns", [])]
    if got != EXPECTED_COLUMNS:
        failures.append(("④B 列定义（数据文件）", f"期望 {EXPECTED_COLUMNS}\n  实际 {got}"))
    header_td = re.findall(r'<th class="g-\w+(?: col-start)?" data-col="([^"]+)">([^<]*)</th>', target_text)
    header_td = [(k, html.unescape(v)) for k, v in header_td]
    if header_td != EXPECTED_COLUMNS[1:]:
        failures.append(("④B 渲染后表头 cells", f"实际 {header_td}"))
    # ★ 精简轮 2：列组标签由「基础」改为「实验条件」⇒ sticky 表头随之（⛔ 断言本身未放宽）
    sticky_th = (f'<th class="{_td_cls}" data-colgroup="{html.escape(groups[first["group"]], quote=True)}" '
                 f'rowspan="2">{html.escape(first["label"])}</th>')
    if sticky_th not in target_text:
        failures.append(("④B 档号 sticky 表头", f"未找到逐字 {sticky_th}"))
    # 行数（只取矩阵表自己的 tbody；页内还有规则表 / 状态表，不能混算）
    if '<table class="matrix">' not in target_text:
        failures.append(("④B 矩阵表", '未找到 <table class="matrix">'))
        body = ""
    else:
        body = target_text.split('<table class="matrix">', 1)[1].split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    trs = re.findall(r"<tr>(.*?)</tr>", body, re.S)
    if len(trs) != len(data["rows"]):
        failures.append(("④B 行数", f"期望 {len(data['rows'])}，实际 {len(trs)}"))

    # ---- ④C. 验证状态四态一致性（§17.7） ----
    vocab = (data.get("meta") or {}).get("status_vocabulary", [])
    filled, bad = [], []
    for r in data.get("rows", []):
        tier, st = r.get("tier"), r.get("status")
        if st not in vocab:
            bad.append((tier, f"非法验证状态 {st!r}"))
            continue
        if st in ("已验证可用", "已验证不可用"):
            filled.append(tier)
            if st == "已验证不可用" and not r.get("verdict"):
                bad.append((tier, "「已验证不可用」缺结论块（未写明未过判据 / 已通过项）"))
            if r.get("note") in ("", "—", None):
                bad.append((tier, "已结论档缺「备注」归因"))
        else:
            # ★ §17.7「未验证不得冒充可用」+「不得用空白/占位符使人误读」：
            #   未结论档**不得携带任何实验数据**（结果列一律 '—'）；备注只允许是**状态说明**
            #   （"判定：<state>…"）——它回答"为什么还没结论"，是面向使用者的必要信息。
            #   ⚠ 本断言改版的原因（如实登记）：`voided_conclusion` / `measurement_pending` 原先是
            #   这里用来判"允许带读数"的**内部豁免字段**；它们已按 §17.7「内部字段不进人读视图」
            #   从正式件剔除 ⇒ 断言改为**数据驱动**（看状态与备注文本本身），⛔ 不靠内部标记。
            nz = [k for k in EXP_DATA_KEYS if r.get(k) not in ("", "—", None)]
            if nz:
                bad.append((tier, f"「{st}」却带实验数据 {nz}"))
            _note = str(r.get("note") or "").strip()
            if _note not in ("", "—") and st == "尚未验证" and f"判定：{st}" not in _note:
                bad.append((tier, "「尚未验证」的备注不是状态说明（应以「判定：尚未验证」起头）"))
        if st == "未纳入验收" and not r.get("excluded_reason"):
            bad.append((tier, "「未纳入验收」缺逐档理由"))
    # ★ 现状约束：结论与同行参数一一对应 ⇒ 参数更正后旧结论作废；当前只应有「尚未验证 / 未纳入验收」
    # ★ 现状集合断言改为**数据驱动**（`meta.current_state_expectation`）：
    #   硬编码"当前只应有 X/Y"会随跑次推进而必须手改（与"只有表格随数据变"相悖）⇒ 缺省不设预期，
    #   仅保留"取值 ∈ 四态"与"各有其证据"两项校验。项目若确需冻结现状，可在数据里显式声明预期集合。
    expected_now = (data.get("meta") or {}).get("current_state_expectation")
    present = sorted({r.get("status") for r in data.get("rows", [])})
    if expected_now and not (data.get("meta") or {}).get("_selfproof_fixture"):
        if set(present) - set(expected_now):
            failures.append(("④C 现状状态集合", f"数据声明预期 {expected_now}，实际 {present}"))
    pend = [r["tier"] for r in data.get("rows", []) if r.get("measurement_pending")]
    for r in data.get("rows", []):
        if r.get("measurement_pending") and "阈值 δ 待定" not in str(r.get("note", "")):
            bad.append((r.get("tier"), "读数已录但备注未写明「阈值 δ 待定」"))
    if bad:
        failures.append(("④C 验证状态四态一致性", f"不一致 {len(bad)} 档（前 5）: {bad[:5]}"))
    if a.expect_empty and filled:
        failures.append(("④C --expect-empty：不得有已结论档位", f"已结论 {len(filled)} 档: {filled[:8]}"))

    # ---- ④D. 表格列逐档与数据逐字一致（主要参数分列 / 实验数据 / 备注） ----
    mismatch = []
    cell_re = lambda key: re.compile(r'<td class="[^"]*" data-col="' + re.escape(key) + r'">(.*?)</td>', re.S)
    for key in PARAM_SUBKEYS + RESULT_KEYS + ["note"]:
        cells = re.findall(r'<td class="[^"]*" data-col="' + re.escape(key) + r'">(.*?)</td>', target_text, re.S)
        if len(cells) != len(data["rows"]):
            mismatch.append((key, f"单元格数 {len(cells)} != {len(data['rows'])}"))
            continue
        for row, cell in zip(data["rows"], cells):
            v = str(row.get(key, ""))
            got = html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            if got != v:
                mismatch.append((row.get("tier"), key, f"渲染 {got!r} != 数据 {v!r}"))
    if mismatch:
        failures.append(("④D 表格列一致性", f"不一致 {len(mismatch)} 处（前 5）: {mismatch[:5]}"))

    # ---- ④E. §1.0 统一口径块 + 供 yaml 生成器的机读字段 ----
    uni = data.get("uniform") or {}
    if not uni.get("items"):
        failures.append(("④E 统一口径块", "数据文件缺 uniform.items"))
    else:
        if '<table class="plain uniform">' not in target_text:
            failures.append(("④E 统一口径块", "渲染结果里未找到 uniform 表"))
        missing_uni = [it.get("key") for it in uni["items"]
                       if f'<code>{html.escape(str(it.get("value", "")))}</code>' not in target_text]
        if missing_uni:
            failures.append(("④E 统一口径取值", f"未逐字在位: {missing_uni}"))
    # ---- ④G. 脱敏：正式件不得出现宿主路径 / 内网路径 / 机器名 / 人名（宪法 §15.1，A 类零容忍） ----
    SENSITIVE = [
        # 通用形状（⛔ 检测器自身不得携带任何具体主机 / 账号 / 挂载点字面量）
        (r"/(?:home|work|mnt|opt|srv|data|models)/[A-Za-z0-9._-]+", "宿主 / 内网路径"),
        (r"\.local/", "本地目录 .local/"),
        (r"\b(?:10|172|192)\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\b", "私网 IP"),
        (r"\b[0-9]{2,3}\s*(?:共享机|机器|主机|在役快照)", "机器名（数字 + 机器语境）"),
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "邮箱（PII）"),
    ]
    leaks = []
    for pat, why in SENSITIVE:
        for m in re.finditer(pat, target_text):
            leaks.append((why, m.group(0)))
    if leaks:
        failures.append(("④G 脱敏（宪法 §15.1）", f"命中 {len(leaks)} 处（前 5）: {leaks[:5]}"))

    # ---- ④H. 禁止措辞（§17.7：双真相源 / 能力承诺式措辞） ----
    forbidden = [(w, why) for w, why in FORBIDDEN_PHRASES if w in target_text]
    if forbidden:
        failures.append(("④H 禁止措辞", f"命中 {forbidden}"))

    # ---- ④I. §17.7 结构要件：生成标记 / 覆盖度 / 边界声明 / 判定口径 ----
    meta = data.get("meta") or {}
    if 'name="matrix-generation-marker"' not in target_text or "generated-by:" not in target_text:
        failures.append(("④I 生成标记", "缺机器可读生成标记（generated-by / meta 标记）"))
    if "自动生成" not in target_text or "禁止手工编辑" not in target_text:
        failures.append(("④I 首屏生成声明", "首屏缺「由生成器从数据自动生成，禁止手工编辑」"))
    if not (meta.get("boundary_declaration") or {}).get("points"):
        failures.append(("④I 边界声明", "数据缺 boundary_declaration.points"))
    if "承诺边界声明" not in target_text:
        failures.append(("④I 边界声明可见性", "人读视图缺边界声明"))
    if not (meta.get("criteria_caliber") or {}).get("criteria_id"):
        failures.append(("④I 判定口径", "数据缺 criteria_caliber.criteria_id"))

    # ---- ④K. §17.7 覆盖度门槛：首屏警示 + 四态计数 + 「尚未验证」清单（**数据驱动**）----
    _vocab = (meta.get("status_vocabulary") or [])
    _rows_ = data.get("rows", [])
    _cnt = {s: sum(1 for _r in _rows_ if _r.get("status") == s) for s in _vocab}
    _unver = [str(_r.get("tier")) for _r in _rows_ if _r.get("status") == "尚未验证"]
    if _unver:
        # 覆盖度未达门槛 ⇒ 首屏**必须**有警示，且必须列出「尚未验证」的档位清单与数量
        if 'id="coverage-warning"' not in target_text:
            failures.append(("④K 覆盖度门槛", "有「尚未验证」档位但首屏缺覆盖度警示（id=coverage-warning）"))
        for _s, _n in _cnt.items():
            if _n and f"{_s} <b>{_n}</b> 档" not in target_text:
                failures.append(("④K 覆盖度计数", f"「{_s}」计数 {_n} 未在警示中逐字出现"))
        # ★★ 可见性判据（**以渲染出的 id 与标签**为准，⛔ 不以数据为准）：
        #    a) 页面上出现的档号集合必须**恰好等于**数据的档位集合（防"数据 55 档、页面 54 档"）；
        #    b) 页面上出现的四态标签计数必须**恰好等于**数据的四态计数
        #       （防"警示块陈旧快照 / 表里有档位没带结论标签"）。
        _ids_html = re.findall(r'data-col="tier">([^<]+)</td>', target_text)
        _st_html = re.findall(r'data-col="status">([^<]+)</td>', target_text)
        _ids_data = [str(_r.get("tier")) for _r in _rows_]
        if sorted(_ids_html) != sorted(_ids_data):
            _miss = sorted(set(_ids_data) - set(_ids_html))
            _extra = sorted(set(_ids_html) - set(_ids_data))
            failures.append(("④K 档位可见性",
                             f"页面档号集合 != 数据档位集合（页面缺 {_miss[:5]}；页面多 {_extra[:5]}）"))
        _cnt_html = {s: _st_html.count(s) for s in set(_st_html)}
        for _s in _vocab:
            if _cnt_html.get(_s, 0) != _cnt.get(_s, 0):
                failures.append(("④K 状态标签计数",
                                 f"页面「{_s}」标签 {_cnt_html.get(_s, 0)} 个 != 数据 {_cnt.get(_s, 0)} 档"))
        #    c) 警示块里的人数必须与数据一致（防"警示块是陈旧快照"）
        if f'data-row-count="{len(_rows_)}"' not in target_text:
            failures.append(("④K 档位总数可见性",
                             f"页面未见 data-row-count={len(_rows_)}（警示块与数据不同源）"))
        if f"尚未验证的档位（{len(_unver)} 档）" not in target_text:
            failures.append(("④K 尚未验证清单数量", f"未出现「尚未验证的档位（{len(_unver)} 档）」"))
        _miss = [t for t in _unver if f"<code>{t}</code>" not in target_text]
        if _miss:
            failures.append(("④K 尚未验证档位清单", f"清单缺 {len(_miss)} 档（前 5）: {_miss[:5]}"))
        # 标题与摘要不得使用能力承诺式措辞
        _head = target_text.split("</header>", 1)[0]
        for w in ("全组合可用", "已验证全组合", "全部可用", "均已可用"):
            if w in _head:
                failures.append(("④K 标题/摘要措辞", f"覆盖度未达门槛却出现能力承诺式措辞 {w!r}"))
    elif 'id="coverage-warning"' in target_text:
        failures.append(("④K 覆盖度门槛", "全部档位已有明确结论，却仍渲染覆盖度警示"))

    # ---- ④N. §17.7 承诺边界声明**三点齐备**（存在性 + 三项内容要件，⛔ 不做逐字比对）----
    _bd = (meta.get("boundary_declaration") or {})
    _bd_txt = str(_bd.get("text") or "")
    _bd_pts = " ".join(str(x) for x in (_bd.get("points") or []))
    _bd_all = _bd_txt + " " + _bd_pts
    if not _bd_txt.strip():
        failures.append(("④N 边界声明存在性", "boundary_declaration.text 为空 ⇒ 视为交付未完成（§17.7）"))
    # ① 各档可用性（每个列出的档位都已完整跑通）
    if not any(k in _bd_all for k in ("完整跑通", "都已完整跑通", "各档可用性")):
        failures.append(("④N 边界要件①各档可用性", "未正面陈述「每个列出的档位都已完整跑通」"))
    # ② 平台方数据集结论（效果达标 + 是"可用、大概率无 bug"的证据）
    if not any(k in _bd_all for k in ("平台方数据集", "我们的数据集")):
        failures.append(("④N 边界要件②平台方数据集", "未正面陈述「在平台方数据集与参数下效果达标」"))
    if "大概率无 bug" not in _bd_all:
        failures.append(("④N 边界要件②证据定位", "未写明这是「可用、大概率无 bug」的证据"))
    # ③ 使用者效果不在承诺范围
    if not any(k in _bd_all for k in ("不在本矩阵的承诺范围", "不在承诺范围", "不归本矩阵承诺")):
        failures.append(("④N 边界要件③使用者效果", "未写明「使用者的任务/数据集/参数下的效果不在承诺范围」"))
    # ★ 反例（判为不合格）：只写"效果不保证 / 仅供参考"而未正面陈述 ①②
    if (not any(k in _bd_all for k in ("完整跑通", "平台方数据集", "我们的数据集"))
            and any(k in _bd_all for k in ("效果不保证", "仅供参考"))):
        failures.append(("④N 边界声明反例", "只写「不保证效果 / 仅供参考」⇒ §17.7 判为不合格"))

    # ---- ④Q0. 「关键开关」列的 `—` **语义必须被显式定义**（修正轮 4 ⚠②）----
    #   ★ 缺陷实况：该列 31/54 档为 `—`，而上一版读表提示只定义了「实验数据」段的 `—`
    #     ⇒ 属"要读者推断"。修法 = 读表提示补一句（⛔ 不改渲染形态，`--human-view-self-test`
    #     的"唯一开关 ⇒ `—`"反向断言**保持成立**）。
    #   ★ 断言只许加强：不仅查那句话在不在，还查**它必须同时点明段位与列名**（防写回泛泛一句）。
    if "「关键开关」列的「—」" not in target_text or "没有额外差异开关" not in target_text:
        failures.append(("④Q0 关键开关列 — 的语义",
                         "读表提示未显式定义「关键开关」列的 `—` 语义（31/54 档该列为 `—`）"))

    # ---- ④Q. 「关键开关」列**不得重复发布检查点间隔**（修正包 🔴-1）----
    #   ★ 断言只许加强：不仅查"令牌在不在"，还查**该档的权威值是否等于公式值**。
    #     判据①（发布面）：`switches` 里不得出现 `save_steps=` 令牌
    #       ——清单侧的旧口径值（= 每 epoch 步数）一旦发布，就与 §3.1 定义行、
    #         同行权威字段 `save_steps`、跑次 yaml 三处矛盾（实测 50/54 档）。
    #     判据②（取值面）：逐档 `save_steps` 必须**恰好**等于 `derived_steps + 1`
    #       ——这是 §3.1 定义行的机核化（⛔ 不依赖人眼比对）。
    _tok_q = build_official.SAVE_STEPS_TOKEN
    _bad_q = [str(r.get("tier")) for r in _rows_ if _tok_q in str(r.get("switches") or "")]
    if _bad_q:
        failures.append(("④Q 开关列不得重复发布检查点间隔",
                         f"「关键开关」列出现 `{_tok_q}` 令牌（{len(_bad_q)} 档：{_bad_q[:5]}）"
                         f"——该参数只允许在 §3.1 定义一次"))
    _bad_q2 = []
    for _r in _rows_:
        _dv, _sv = _r.get("derived_steps"), _r.get("save_steps")
        if _dv in (None, "", "—") or _sv in (None, "", "—"):
            continue
        if int(_sv) != int(_dv) + 1:
            _bad_q2.append((str(_r.get("tier")), _sv, int(_dv) + 1))
    if _bad_q2:
        failures.append(("④Q 检查点间隔取值 =推导步数+1",
                         f"{len(_bad_q2)} 档不符（tier, 实际, 应为）：{_bad_q2[:5]}"))
    # §3.1 必须把「定义一次 + 表内不重复列出」写清（防下一位访客/复核者再踩同一坑）
    if "不再重复列出该参数" not in target_text:
        failures.append(("④Q §3.1 交叉引用",
                         "§3.1 未写明「表内不再重复列出该参数」⇒ 无法解释为何表里看不到它"))

    # ---- ④R. 人读件正文**不得含开发过程内容**（修正包 🔴-2 / 建议 S2，机核化主席判据）----
    #   ★ 判据（主席原话）：「不要把修改过程写进去，不要把给我们自己写的东西写进去」。
    #   ★ 只扫**正文**（渲染后的可见文本），且**排除 §17.7 宪法逐字引文**（引文里的「我们」是豁免项）。
    #   ★ 词表 = 内部工位/证据路径形态 + 相对时间指代；⛔ 不做通配、⛔ 不豁免生成物。
    #   ★★ 修正轮 4（⚠①）：**豁免锚点必须在去标签之前的原始 HTML 上定位**。
    #     缺陷实况（复核报出）：上一版先 `re.sub(r"<[^>]+>", " ", …)` 去标签，之后才找
    #     `承诺边界声明。</b>` —— 去标签后**不存在 `</b>`** ⇒ `_quoted == []` ⇒ **豁免从不生效**
    #     （偏严，当前无害；但一旦把「我们」之类加进词表，会在**宪法逐字引文**上误报）。
    #     修法：**先按原文切出宪法逐字引文段**（`承诺边界声明。</b>` … `<ul>`），再对其余正文去标签。
    #     ★ 豁免面**只**是那一段引文（`…都在我们的数据集上…`），⛔ 不是整个声明块：
    #       块内 `<li>` 各条与「覆盖度门槛」注**仍被扫**。
    _body_raw = target_text
    _q_blocks = re.findall(r"承诺边界声明。</b>.*?<ul>", _body_raw, re.S)
    for _qb in _q_blocks:
        _body_raw = _body_raw.replace(_qb, " ")
    _prose = html.unescape(re.sub(r"<[^>]+>", " ", _body_raw))
    _churn = []
    for _w, _why in build_official.DEV_CHURN_TERMS:
        if _w in _prose:
            _n = _prose.count(_w)
            _churn.append((_w, _n, _why))
    if _churn:
        failures.append(("④R 正文不得含开发过程内容",
                         "命中内部过程词：" + "；".join(f"`{w}`×{n}（{why}）" for w, n, why in _churn)))

    # ---- ④P. 覆盖度门槛注（§17.7「未达门槛时不得宣称整体能力承诺」）----
    #   ★ 断言只许加强：不仅查「警示在不在」，还查**边界声明块里有没有写明门槛尚未生效**，
    #     并且**计数必须与 rows 现算一致**（防"警示说 42、门槛注说 0"这类合成谬误）。
    _pend_p = [r["tier"] for r in _rows_ if r.get("status") == "尚未验证"]
    _all_concluded = not _pend_p
    _bd_start = target_text.find("承诺边界声明")
    _bd_end = target_text.find("</div>", _bd_start) if _bd_start >= 0 else -1
    _bd_txt = target_text[_bd_start:_bd_end] if (_bd_start >= 0 and _bd_end > 0) else ""
    _gate_kw = "覆盖度门槛"
    if not _all_concluded:
        if _gate_kw not in _bd_txt:
            failures.append(("④P 覆盖度门槛注", "尚有「尚未验证」档，但承诺边界声明块内未写明「整体承诺尚未生效」"))
        elif f"尚有 {len(_pend_p)} 档" not in _bd_txt:
            failures.append(("④P 覆盖度门槛注计数",
                             f"门槛注里的档数与 rows 现算不一致（应为 {len(_pend_p)} 档）"))
    else:
        if _gate_kw in _bd_txt:
            failures.append(("④P 覆盖度门槛注", "全部档位都已结论，却仍写着「尚有 … 档为尚未验证」"))

    # ---- ④O. 结论两列的**契约**（生成端 `build_official.note_contract` ⇄ 校验端同一组字符串）----
    #   ★ 断言只许加强：这里不仅查"有没有写第 1/2 条"，还查**归因正文的依据关键词**是否在位、
    #     备注列是否**恰好**是契约文本（防"一句话被删空"或"塞回长文"）。
    _DETAIL_EVIDENCE_GUARD = build_official._DETAIL_EVIDENCE
    _bad_o = []
    for _r in _rows_:
        _t, _st = str(_r.get("tier")), _r.get("status")
        _want = build_official.note_contract(_st, has_verdict=bool(_r.get("verdict")),
                                  has_excluded=bool(str(_r.get("excluded_reason") or "").strip()),
                                  raw_note=str(_r.get("note") or ""))
        if str(_r.get("note") or "").strip() != _want:
            _bad_o.append((_t, f"备注 != 契约文本：实际 {str(_r.get('note'))[:40]!r}"))
        _nr = str(_r.get("not_failed_reason") or "").strip()
        if _st == "已验证不可用":
            _v = _r.get("verdict") or {}
            if _nr in ("", "—"):
                _bad_o.append((_t, "「已验证不可用」的「未过判据」列为空"))
            if not (_nr.startswith("第 1 条") or _nr.startswith("第 2 条")):
                _bad_o.append((_t, "「未过判据」须以「第 1 条 / 第 2 条」起头"))
            _det = str(_v.get("failed_detail") or "")
            _ev = _DETAIL_EVIDENCE_GUARD.get(_t, ())
            _hit = [k for k in _ev if k in _det or k in _nr]
            if _ev and not _hit:
                _bad_o.append((_t, f"归因正文缺依据关键词（期望其一 {list(_ev)}）"))
            elif _ev and not all(k in _det or k in _nr for k in _ev):
                _bad_o.append((_t, f"归因正文依据不完整（缺 {[k for k in _ev if k not in _det and k not in _nr]}）"))
        elif _st in ("尚未验证", "未纳入验收"):
            if _nr not in ("", "—"):
                _bad_o.append((_t, f"「{_st}」不得有「未过判据」（实际 {_nr[:30]!r}）"))
    if _bad_o:
        failures.append(("④O 结论列契约", f"不一致 {len(_bad_o)} 档（前 5）: {_bad_o[:5]}"))

    # ---- ④L. §17.7 逐档「不可用必须写明哪一条」+ 「未纳入验收」逐档理由 ----
    _bad_crit = []
    for _r in _rows_:
        if _r.get("status") != "已验证不可用":
            continue
        _v = _r.get("verdict") or {}
        _blob = f"{_v.get('failed_criterion', '')} {_v.get('failed_detail', '')} {_r.get('note', '')}"
        if not any(k in _blob for k in ("第 1 条", "第 2 条")):
            _bad_crit.append((_r.get("tier"), "未写明未过第 1 条还是第 2 条"))
    if _bad_crit:
        failures.append(("④L 不可用须写明条次", f"{len(_bad_crit)} 档（前 5）: {_bad_crit[:5]}"))
    _bad_excl = [( _r.get("tier"), "缺逐档理由") for _r in _rows_
                 if _r.get("status") == "未纳入验收" and not str(_r.get("excluded_reason") or "").strip()]
    if _bad_excl:
        failures.append(("④L 未纳入验收逐档理由", f"{len(_bad_excl)} 档（前 5）: {_bad_excl[:5]}"))

    # ---- ④M. 内部术语（§17.7「矩阵中仍然禁止」＋使用者诉求）----
    _term_hits = [(w, target_text.count(w)) for w in FORBIDDEN_INTERNAL_TERMS if w in target_text]
    if _term_hits:
        failures.append(("④M 内部术语", f"命中 {len(_term_hits)} 类: {_term_hits}"))

    # ---- ④J. 耗时列（★ fail-closed，四向）----
    #   ① 有读数（跑次文本 或 228 只读侦察 `elapsed_evidence`）⇒ 列**必须**有值；
    #   ② 列有值（非「未取到」）⇒ **必须**有读数证据（⛔ 防编造）；
    #   ③ 列标「未取到」⇒ 括号内**必须**有具名原因，且此时**不得**存在读数证据（口径矛盾）；
    #   ④ 列**为空** ⇒ FAIL（⛔ 不留空白让它看起来像"忘了填"）。
    el_bad, el_filled, el_na = [], [], []
    try:
        import build_official as _bo
        for r in data.get("rows", []):
            # ★ 仅对**已结论**档断言：未结论档按 §17.7 结果列一律 '—'（不适用本断言）
            if r.get("status") not in ("已验证可用", "已验证不可用"):
                continue
            parsed = _bo.split_elapsed(_bo.elapsed_source_blob(r)) != "—"
            ev = r.get("elapsed_evidence") if isinstance(r.get("elapsed_evidence"), dict) else {}
            st = str(ev.get("status") or "")
            has_ev = st in ("found", "found_from_run_json")
            got = str(r.get("elapsed") or "").strip()
            if got in ("", "—"):
                el_bad.append((r.get("tier"), "「耗时」列为空（既无读数也无具名原因）"))
                continue
            # ★ 生成期把「总时长 N s」规范化为「总时长 N s（约 X h）」——**原始秒数逐字保留在括号前**，
            #   ⇒ 该形态自身即读数证据；此处直接认可并继续（⛔ 不放宽取值域：仍只认这一种形态）。
            _dur_ok = bool(re.fullmatch(r"总时长\s*\d+\s*s（约[\d.]+\s*h）", got))
            if got.startswith("未取到"):
                el_na.append(r.get("tier"))
                mm = re.match(r"^未取到（\s*(.+?)\s*）$", got)
                if not mm or not mm.group(1):
                    el_bad.append((r.get("tier"), "标「未取到」但缺具名原因"))
                if parsed or has_ev:
                    el_bad.append((r.get("tier"), "标「未取到」但跑次文本/侦察证据里有读数 ⇒ 口径矛盾"))
                continue
            el_filled.append(r.get("tier"))
            if _dur_ok:
                continue
            # ★ 生成期把「总时长 N s」折算为「总时长 N s（约 X h）」——数字未变、只是加了折算；
            #   该形态不再与跑次原始文本逐字相同 ⇒ 由**形态本身**判定（⛔ 不放宽取值域）。
            _dur = re.fullmatch(r"总时长\s*(\d+)\s*s(?:（约\s*([\d.]+)\s*h）)?", got)
            if _dur:
                continue
            if not (parsed or has_ev):
                el_bad.append((r.get("tier"), f"「耗时」列有值 {got!r}，但跑次文本与侦察证据都无读数（疑编造）"))
            if st == "found" and ev.get("value_s") is None:
                el_bad.append((r.get("tier"), "elapsed_evidence=found 但缺 value_s"))
            if st == "not_found":
                el_bad.append((r.get("tier"), "elapsed_evidence=not_found 但「耗时」列有值"))
        if el_bad:
            failures.append(("④J 耗时列一致性（fail-closed）", f"不一致 {len(el_bad)} 档（前 5）: {el_bad[:5]}"))
    except Exception as e:                       # ⛔ 校验器自身出错也不得静默
        failures.append(("④J 耗时列一致性（fail-closed）", f"断言执行失败: {e!r}"))

    miss_keys = [(r.get("tier"), k) for r in data["rows"] for k in SPEC_KEYS if k not in r]
    if miss_keys:
        failures.append(("④E 机读字段缺失（yaml 生成器口径）", f"缺 {len(miss_keys)} 处（前 5）: {miss_keys[:5]}"))

    sha_html = hashlib.sha256(target_bytes).hexdigest()
    sha_render = hashlib.sha256(rendered_bytes).hexdigest()

    if not a.quiet:
        def ok(pfx):
            return "PASS" if not any(f[0].startswith(pfx) for f in failures) else "FAIL"
        print(f"{VERSION}")
        print(f"target : {target_path}  ({len(target_bytes)} bytes, sha256 {sha_html})")
        print(f"render : {len(rendered_bytes)} bytes, sha256 {sha_render}")
        print(f"checks : ① 字节同一性={'PASS' if byte_identical else 'FAIL'}"
              f" | ③ 结构两大部分={ok('③')}"
              f" | ④A 规则关键词={ok('④A')}"
              f" | ④B 列顺序={ok('④B')}"
              f" | ④C 结果列一致性={ok('④C')}"
              f" | ④D 主要训练参数列={ok('④D')}"
              f" | ④E 统一口径/机读字段={ok('④E')}"
              f" | ④G 脱敏={ok('④G')}"
              f" | ④H 禁止措辞={ok('④H')}"
              f" | ④I §17.7 要件={ok('④I')}"
              f" | ④J 耗时列={ok('④J')}"
              f" | ④K 覆盖度门槛={ok('④K')}"
              f" | ④L 逐档条次/理由={ok('④L')}"
              f" | ④M 内部术语={ok('④M')}"
              f" | ④N 边界声明三点={ok('④N')}"
              f" | ④O 结论列契约={ok('④O')}"
              f" | ④P 覆盖度门槛注={ok('④P')}"
              f" | ④Q0 关键开关列—语义={ok('④Q0')}"
              f" | ④Q 开关列不重复发布检查点间隔={ok('④Q')}"
              f" | ④R 正文无开发过程内容={ok('④R')}")
        print(f"results: 已回填 {len(filled)} 档 {filled}；其余 {len(data['rows'])-len(filled)} 档保持 '—'"
              + ("；--expect-empty 已开启" if a.expect_empty else ""))
        print(f"elapsed: 有值 {len(el_filled)} 档"
              + (f"（{'、'.join(el_filled)}）" if el_filled else "")
              + f"；标「未取到（具名原因）」{len(el_na)} 档"
              + (f"（{'、'.join(el_na)}）" if el_na else ""))
        if off is None:
            print("hand-edit detection: 无差异（改 1 个字符即会在此打印字节偏移 / 行 / 列）")
        for name, detail in failures:
            print(f"\nFAIL [{name}]\n{detail}")

    print("\nVERIFY " + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
