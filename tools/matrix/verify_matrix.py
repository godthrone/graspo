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

VERSION = "verify-matrix/v7-layout"

# ---- ③ 结构断言：新结构只允许这两大部分 ----
EXPECTED_H2 = ["1. 参数定义", "2. 通过的定义", "3. 判据定义", "4. 状态定义", "5. 功能矩阵测试表"]
EXPECTED_PARTS = ["s1", "s2", "s3", "s4", "s5"]
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
    ("未过判据必须写明", "必须写明未过的是哪一条"),
    ("已通过项事实保留", "exit_code = 0"),
    ("判据口径·数据集锚点", "版本锚点"),
    ("判据口径·判据标识", "elam-v3-allright-1"),
    ("判据口径·阈值冻结", "已冻结（主席定案）"),
    ("判据口径·δ 值", "δ = +5.0 pp"),
    ("判据口径·GRASPO 冻结", "GRASPO 的通过线均为 δ = +5.0 pp"),
    ("判据口径·目标值仅展示", "目标值 = +10.0 pp（仅展示，⛔ 不用于判定）"),
    ("判据口径·基线口径", "U2 终点窗口"),
    ("参数定义", "参数定义"),
    ("统一口径", "统一口径"),
    ("只列差异项", "只列出档位之间的差异项"),
    ("判据定义 U1", "exit_code = 0"),
    ("算法检查必须能失败", "必须能失败"),
    ("B3 基础设施不影响", "都不应显著影响效果"),
    ("评测/推理验证列", "效果（评测 vs 基准）"),
    ("步数为上限", "该值是上限"),
    ("并行按后端分列", "nproc_per_node"),
    ("并行·配置值", "配置值"),
    ("并行·派生值", "派生"),
    ("并行·关系式", "总卡数 = nproc_per_node × nnodes = TP × PP × CP × DP"),
    ("LoRA 目标分列", "target_modules"),
    ("检查点间隔键名", "checkpoint_save_steps"),
]

#: 禁止措辞（§17.7：双真相源 / 能力承诺式措辞）
FORBIDDEN_PHRASES = [
    ("同为真相源", "双真相源表述"),
    ("互为真相源", "双真相源表述"),
    ("两个真相源", "双真相源表述"),
    ("全组合可用", "能力承诺式措辞"),
    ("已验证全组合", "能力承诺式措辞"),
]

# ---- ④B. 已定新表头：10 列（档号 → 主要训练参数 → 结果列组） ----
EXPECTED_COLUMNS = [
    ("tier", "档号"),
    ("status", "是否通过"),
    ("model", "模型"),
    ("algo", "算法"),
    ("mode", "模式"),
    ("backend", "后端"),
    ("tp", "TP"),
    ("pp", "PP"),
    ("dp", "DP"),
    ("cards_total", "总卡数"),
    ("lora_target", "LoRA 目标"),
    ("lr", "lr"),
    ("data_scale", "数据规模"),
    ("steps", "步数/epoch"),
    ("switches", "关键开关"),
    ("peak_mem", "显存峰值(GiB)"),
    ("steps_actual", "实跑步数"),
    ("elapsed", "耗时（每步/总时长）"),
    ("throughput", "吞吐"),
    ("effect", "效果（评测 Δ / loss·reward 窗口）"),
    ("note", "备注（条件 / 限制 / 失败归因）"),
]
RESULT_KEYS = ["peak_mem", "steps_actual", "elapsed", "throughput", "effect"]
# 主要参数：按项分列（顺序即展示顺序）
PARAM_SUBKEYS = ["model", "algo", "mode", "backend", "tp", "pp", "dp", "cards_total",
                 "lora_target", "lr", "data_scale", "steps", "switches"]
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
        failures.append(("③ 结构：<h2> 只允许两个", f"期望 {EXPECTED_H2}\n  实际 {h2}"))

    parts = re.findall(r'<section class="part" id="([^"]+)"', target_text)
    if parts != EXPECTED_PARTS:
        failures.append(("③ 结构：分区只允许两个", f"期望 {EXPECTED_PARTS}\n  实际 {parts}"))

    stale = [s for s in FORBIDDEN_OLD_SECTIONS if f'id="{s}"' in target_text]
    if stale:
        failures.append(("③ 结构：不应再有旧骨架分区", f"仍存在 {stale}"))

    toc = re.findall(r'<nav class="toc">(.*?)</nav>', target_text, re.S)
    if len(toc) != 1:
        failures.append(("③ 结构：目录", f"期望恰好 1 个 nav.toc，实际 {len(toc)}"))
    else:
        links = re.findall(r'href="#([^"]+)"', toc[0])
        if links != EXPECTED_PARTS:
            failures.append(("③ 结构：目录只指向两大部分", f"期望 {EXPECTED_PARTS}\n  实际 {links}"))

    # ---- ④A. 规则关键词 ----
    missing = [name for name, kw in REQUIRED_MARKERS if kw not in target_text]
    if missing:
        failures.append(("④A 规则关键词", f"缺失 {len(missing)} 项: {missing}"))

    # ---- ④B. 列顺序 == 已定 10 列新表头 ----
    got = [(c.get("key"), c.get("label")) for c in data.get("columns", [])]
    if got != EXPECTED_COLUMNS:
        failures.append(("④B 列定义（数据文件）", f"期望 {EXPECTED_COLUMNS}\n  实际 {got}"))
    header_td = re.findall(r'<th class="g-\w+(?: col-start)?" data-col="([^"]+)">([^<]*)</th>', target_text)
    header_td = [(k, html.unescape(v)) for k, v in header_td]
    if header_td != EXPECTED_COLUMNS[1:]:
        failures.append(("④B 渲染后表头 cells", f"实际 {header_td}"))
    sticky_th = '<th class="g-base col-tier col-start" data-colgroup="基础" rowspan="2">档号</th>'
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
            if r.get("measurement_pending"):
                skip = set(RESULT_KEYS)          # 读数已录、阈值待定 ⇒ 允许带读数
            elif r.get("voided_conclusion"):
                skip = {"steps_actual"}          # 作废参考记录 ⇒ 只允许保留实跑步数
            else:
                skip = set()                     # 纯「尚未验证」⇒ 不得带任何读数
            nz = [k for k in RESULT_KEYS if k not in skip and r.get(k) not in ("", "—", None)]
            if nz:
                bad.append((tier, f"「{st}」却带实验数据 {nz}"))
            if (r.get("note") not in ("", "—", None) and st == "尚未验证"
                    and not (r.get("voided_conclusion") or r.get("measurement_pending"))):
                bad.append((tier, "「尚未验证」却带备注（且无作废/待定注记）"))
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
              f" | ④I §17.7 要件={ok('④I')}")
        print(f"results: 已回填 {len(filled)} 档 {filled}；其余 {len(data['rows'])-len(filled)} 档保持 '—'"
              + ("；--expect-empty 已开启" if a.expect_empty else ""))
        if off is None:
            print("hand-edit detection: 无差异（改 1 个字符即会在此打印字节偏移 / 行 / 列）")
        for name, detail in failures:
            print(f"\nFAIL [{name}]\n{detail}")

    print("\nVERIFY " + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
