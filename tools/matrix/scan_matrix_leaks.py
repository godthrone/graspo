#!/usr/bin/env python3
"""scan_matrix_leaks.py — 能力与可用档位矩阵的脱敏扫描（宪法 §17.7「矩阵的脱敏扫描」）。

职责：对矩阵产物（机读数据 + 人读视图）做**禁止项**扫描，命中即失败（rc=1）。

禁止项（§17.7「矩阵中仍然禁止」）：
  宿主路径 / 内网地址、域名、路径 / 机器名 / 人名 / 内部节点与内部账号。

设计约束（§15.1 首段分工：脚本承载具体判定规则）：
  - 模式一律为**通用形状**，⛔ 不在本脚本内写任何具体主机名 / 账号 / 内网路径字面量
    （检测器自身不得成为泄漏源）；
  - 需要按项目补充的**具体禁词**（内部账号、人名等）由可选的外部清单提供，
    默认路径 `<repo>/.local/matrix-leak-denylist.txt`（一行一词，`.local/` 不入库）；
  - 生成物与手写文档同等扫描（§17.7「扫描不豁免」）。

用法：
  python3 tools/matrix/scan_matrix_leaks.py            # 扫默认两件
  python3 tools/matrix/scan_matrix_leaks.py <path>...  # 扫指定文件
退出码：0 = 无命中；1 = 有命中。
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = ["docs/capability-matrix.data.json", "docs/capability-matrix.html"]
DENYLIST = REPO / ".local" / "matrix-leak-denylist.txt"

# 通用形状（⛔ 无具体内部字面量）
PATTERNS = [
    (r"/homedata/", "★ 数据集存放路径（⛔ 矩阵不得出现）"),
    (r"test\.jsonl", "★ 数据集文件名（⛔ 矩阵不得出现）"),
    (r"(?<![0-9A-Za-z])228(?![0-9A-Za-z])", "★ 机器号（⛔ 矩阵不得出现）"),
    (r"(?:^|[^A-Za-z0-9._/-])/(?:home|root|Users|mnt|work|data|opt|srv|var|models)/[A-Za-z0-9._-]+",
     "宿主路径 / 内网路径"),
    (r"\b(?:10|172|192)\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\.(?:[0-9]{1,3})\b", "内网 IP"),
    (r"\b[A-Za-z0-9-]+\.(?:internal|corp|lan|intra)\b", "内网域名"),
    (r"\b[A-Za-z0-9-]*(?:gpu|node|host|server|worker)[-_]?[0-9]{1,3}\b", "机器名（主机命名式）"),
    (r"\b[0-9]{2,3}\s*(?:共享机|机器|主机|在役快照)", "机器名（数字 + 机器语境）"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "邮箱（PII）"),
    (r"\bssh[-_][A-Za-z0-9._-]+\b", "内部账号（ssh 账号式）"),
]


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
    hits = scan(targets, load_denylist())
    if not a.quiet:
        print(f"scan_matrix_leaks: 目标 {len(targets)} 件；禁止项命中 {len(hits)} 处")
        for p, i, why in hits:
            print(f"  ✗ {p}:{i}: {why}")
    print("LEAK-SCAN " + ("CLEAN" if not hits else "FAIL"))
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())
