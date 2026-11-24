#!/usr/bin/env python3
"""scan_matrix_leaks.py — 能力与可用档位矩阵的脱敏扫描（宪法 §17.7「矩阵的脱敏扫描」）。

职责：对矩阵产物（机读数据 + 人读视图）做**禁止项**扫描，命中即失败（rc=1）。

禁止项（§17.7「矩阵中仍然禁止」）：
  宿主路径 / 内网地址、域名、路径 / 机器名 / 人名 / 内部节点与内部账号。
  ＋★ 人脸与内部编排（内部称谓、内部批次/工位/口径包…）——矩阵的读者是**使用者**，
  过程、日志、决策与内部编排不进人读视图（§17.7「矩阵数据与合规底线」＋ §16.2 存放位置）。

设计约束（§15.1 首段分工：脚本承载具体判定规则）：
  - 模式一律为**通用形状**，⛔ 不在本脚本内写任何具体主机名 / 账号 / 内网路径字面量
    （检测器自身不得成为泄漏源）；
  - 需要按项目补充的**具体禁词**（内部账号、人名等）由可选的外部清单提供，
    默认路径 `<repo>/.local/matrix-leak-denylist.txt`（一行一词，`.local/` 不入库）；
  - ★ **主机短名**同样不写字面量：词表来自本地 git config
    `gate.content.host-terms`（见 `build_official.HOST_TERMS`）。
    ⚠️ **未配置词表 ⇒ 本条规则不产生检测**（失败方向是「少扫」，与机构词表规则同款，
    此处显式声明、不静默）；
  - 生成物与手写文档同等扫描（§17.7「扫描不豁免」）。

用法：
  python3 tools/matrix/scan_matrix_leaks.py            # 扫默认三件
  python3 tools/matrix/scan_matrix_leaks.py <path>...  # 扫指定文件
退出码：0 = 无命中；1 = 有命中。
"""
import argparse
import json
import re
import sys
from pathlib import Path

# ★★ 修正轮 4（⚠③）：**生成端与扫描端同源引用**内部过程词表与内部证据路径形态
#    （§1.4 单一真相源：⛔ 不在本脚本另抄一份词表/正则）。
#    ★ 现在**数据件面**也有能失败的机检 —— 上一版本脚本只覆盖宿主路径/数据集位置/denylist，
#      而 ④R 只扫渲染后正文 ⇒ `docs/capability-matrix.data.json`（**发布件**）这一面无机检。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_official as _bo    # noqa: E402  （同目录，纯常量与纯函数，⛔ 无副作用）
import paths as _paths          # noqa: E402  （同目录；★ 基准件/副本路径的唯一真相源，§1.4）

REPO = Path(__file__).resolve().parents[2]
# ★ 2026-10-23（基准件唯一 + 按轮副本）：默认扫描面里的「人读件」取 `paths.py` 的基准件常量
#   （⛔ 不再在本脚本写一份路径字面量；基准件改名只需改 paths.py 一处）。
DEFAULT_TARGETS = [_paths.REL_BASELINE_DATA, _paths.REL_BASELINE_HTML]
#: ★ 2026-10-22（判据③ 指向算法说明）：`docs/graspo-algorithm.html` 是**手写人读件**，
#:   判据③ 的链接目标 ⇒ 与另外两件**同等扫描**（§17.7「扫描不豁免」）。
#:   机核 = 本脚本默认三件全扫；`verify_matrix` ④ALG 另有同源一道（算法说明件全文过同一套规则）。
DEFAULT_TARGETS.append("docs/graspo-algorithm.html")
DENYLIST = REPO / ".local" / "matrix-leak-denylist.txt"

# 通用形状（⛔ 无具体内部字面量）
PATTERNS = [
    # ★★ 2026-10-22（第四轮复核）：内部**写作记号** ★ / ⛔ 不得出现在发布件与页面。
    #   病史：发布件由 `enumerate_matrix.py` 落盘，它当时**没有**调用
    #   `build_official.human_view_notation()`（只走本扫描器），而本词表不含 ★/⛔
    #   ⇒ M093 备注正文漏出一个 `⛔`、模板 CSS 注释漏出 5 个。现两处都补：
    #   ① 生成侧复用**唯一**清理器；② 本词表增这两项（唯一真相源，`verify_matrix` 同源引用）。
    (r"★", "内部写作记号（对人不表达的强调符）"),
    (r"⛔", "内部写作记号（对人不表达的禁止符）"),
    # ★ 内部称谓 / 内部编排 / 内部过程（⛔ 通用形状，不带任何具体人名或档号字面量）
    (r"主席|总参谋长|总司令|指挥部|指挥官|审查者|执行者|口径包|修正轮|归属层",
     "内部称谓 / 内部角色"),
    (r"(?<![0-9])第\s*[0-9一二三四五六七八九十]+\s*批(?![号次])", "内部批次编号"),
    (r"工位|work_log|跑次根目录|机核可跑|必须能失败", "内部编排 / 内部过程用语"),
    (r"/homedata/", "★ 数据集存放路径（⛔ 矩阵不得出现）"),
    (r"test\.jsonl", "★ 数据集文件名（⛔ 矩阵不得出现）"),
    (r"(?:^|[^A-Za-z0-9._/-])/(?:home|root|Users|mnt|work|data|opt|srv|var|models)/[A-Za-z0-9._-]+",
     "宿主路径 / 内网路径"),
    (r"\b(?:10|172|192)\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\b", "内网 IP"),
    (r"\b[A-Za-z0-9-]+\.(?:internal|corp|lan|intra)\b", "内网域名"),
    (r"\b[A-Za-z0-9-]*(?:gpu|node|host|server|worker)[-_]?[0-9]{1,3}\b", "机器名（主机命名式）"),
    (r"\b[0-9]{2,3}\s*(?:共享机|机器|主机|在役快照)", "机器名（数字 + 机器语境）"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "邮箱（PII）"),
    (r"\bssh[-_][A-Za-z0-9._-]+\b", "内部账号（ssh 账号式）"),
] + (
    # ★★ 主机短名：词表来自**本地 git config**（同源常量 `build_official.HOST_TERMS`），
    #   ⛔ 不在本文件写具体主机名 —— 检测器自身不得成为泄漏源（§15.2）。
    #   ⚠️ 代价（显式声明、不静默）：**词表未配置 ⇒ 本条不产生检测**（失败方向是
    #   "少扫"，与 `tools/gate_content.py` 的机构词表规则同款）。
    [(_bo.HOST_TERM_PATTERN, "★ 主机短名（⛔ 矩阵不得出现）")]
    if _bo.HOST_TERM_PATTERN
    else []
) + [
    # ★ 内部证据路径形态（同源常量 `build_official.EVIDENCE_PATH_PATTERN`）
    #   形态判据：必须 `evidence/` 目录形态 ＋ 一位路径字符起头 ＋ `.md` 后缀或省略号收尾
    #   ⇒ ⛔ 不误伤普通英文词 `evidence-based`、仓内公开文件 `docs/capability-matrix.md`。
    (_bo.EVIDENCE_PATH_PATTERN, "内部证据路径（发布件不得出现）"),
] + [
    # ★ 内部开发过程词（同源常量 `build_official.DEV_CHURN_TERMS`，⛔ 不另抄）
    (re.escape(_w), f"内部过程用语：{_why}")
    for _w, _why in _bo.DEV_CHURN_TERMS
]


#: ★★ 2026-10-27（任务 E）备注列**专用**禁入形态 —— 日志坐标 / 绝对路径 / `file:line` /
#: 正则原文。为什么单列一套而⛔ 不塞进 :data:`PATTERNS`：一是长度/正则这类**单元格级**口径
#: 逐行正则量不出来；二是 `(?:` 在规格面（如 `spec.lora_targets` 的正则值）里是**合法内容**，
#: 文件级匹配会误伤 ⇒ 只扫 `rows[*].note`。病史：M011 备注曾 1324 字，含整段 traceback、
#: 正则原文、`/workspace/…/lora.py:170`、`stdout.log:20` ⇒ 用户看到的是日志而不是结论。
NOTE_NOISE_PATTERNS = (
    (r"stdout\.log:\d+", "日志坐标（对人不表达的定位符）"),
    (r"[A-Za-z0-9_./-]+\.py:\d+", "源码 file:line 坐标（对人不表达的定位符）"),
    (r"/workspace/", "容器内绝对路径"),
    (r"/data/", "宿主/容器绝对路径"),
    (r"\(\?:", "正则原文（非人读）"),
)


def scan_notes(paths) -> list[tuple[str, int, str]]:
    """扫**备注列**（`rows[*].note`）里的工程日志形态（失败级）。

    只对 `.json` 数据件生效；解析不了就当没扫到（结构不合法由 verify 那边管）。
    """
    hits = []
    for p in paths:
        f = Path(p)
        if not f.is_file() or f.suffix != ".json":
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for r in doc.get("rows") or ():
            note = _bo.strip_sanctioned(str(r.get("note") or ""))
            for pat, why in NOTE_NOISE_PATTERNS:
                for m in re.finditer(pat, note):
                    hits.append((str(p), 0, f"{why} ⇒ {m.group(0)!r}（{r.get('tier')}.note）"))
    return hits


#: 备注列长度上限的**告警**阈值（字）。超了只提醒、不判 FAIL（人读质量红线在 200 字，
#: 这里留一档余量，避免把「略长但清楚」的备注卡成失败）。
NOTE_LEN_WARN = 400


def note_length_warnings(paths, limit: int = NOTE_LEN_WARN) -> list[tuple[str, int, str]]:
    """备注列过长的**告警**（⛔ 不判 FAIL）：只对 JSON 数据件的 `rows[*].note` 计长度。

    ★ 为什么单列一条而不是塞进 PATTERNS：`PATTERNS` 是正则逐行匹配，量不出「一格多长」；
    长度是**单元格级**的口径，必须按 JSON 结构量。
    """
    out = []
    for p in paths:
        f = Path(p)
        if not f.is_file() or f.suffix != ".json":
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for r in doc.get("rows") or ():
            note = str(r.get("note") or "")
            if len(note) > limit:
                out.append((str(p), 0, f"备注过长 {len(note)} 字（>{limit}）：{r.get('tier')}"))
    return out


def load_denylist() -> list[tuple[str, str]]:
    """可选外部禁词清单（`.local/`，不入库）；缺失或含 # 注释行均安全跳过。"""
    if not DENYLIST.exists():
        return []
    out = []
    for line in DENYLIST.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            out.append((re.escape(w), f"外部禁词清单命中（{DENYLIST.name}）"))
    return out


def scan(paths, denylist):
    hits = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            hits.append((str(p), 0, "目标文件不存在"))
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            # ★ 2026-10-24：授权摘句（主席逐字钉死的判据② 正文「同口径读数」×2）先挖掉，
            #   词表与其余扫描面一字不动（同源常量，见 `build_official.SANCTIONED_PHRASES`）。
            line = _bo.strip_sanctioned(line)
            for pat, why in PATTERNS + denylist:
                m = re.search(pat, line)
                if m:
                    hits.append((str(p), i, f"{why} ⇒ {m.group(0)!r}"))
    return hits


def main():
    ap = argparse.ArgumentParser(description="能力与可用档位矩阵脱敏扫描（宪法 §17.7）")
    ap.add_argument("targets", nargs="*", default=None, help=f"默认：{' '.join(DEFAULT_TARGETS)}")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    targets = a.targets or DEFAULT_TARGETS
    hits = scan(targets, load_denylist()) + scan_notes(targets)
    if not a.quiet:
        print(f"scan_matrix_leaks: 目标 {len(targets)} 件；禁止项命中 {len(hits)} 处"
              f"（含备注列专用词表）")
        for p, i, why in hits:
            print(f"  ✗ {p}:{i}: {why}")
    warns = note_length_warnings(targets)
    if not a.quiet:
        for p, i, why in warns:
            print(f"  ⚠ {p}: {why}（告警，不计入 FAIL）")
        if warns:
            print(f"scan_matrix_leaks: 备注长度告警 {len(warns)} 处（阈值 {NOTE_LEN_WARN} 字；"
                  f"人读上限见 scripts/note_style.py::NOTE_LIMIT）")
    print("LEAK-SCAN " + ("CLEAN" if not hits else "FAIL"))
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())
