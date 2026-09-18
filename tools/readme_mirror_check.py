#!/usr/bin/env python3
"""§17.1 双语 README 镜像核验：内联代码 token 完整性 + 逐节人工核对清单。

职责边界（§1.1）：
- 只回答两个可控问题：
  ① 两侧章节结构是否一一对应（标题顺序/数量）；
  ② 两侧**内联代码 token**（配置键、命令、路径、字段名）集合是否有差集——
     这是"内容镜像"里唯一能机器判定的部分。
- 不做翻译、不改文件，也不假装能自动判定自然语言段落是否等价。
  自然语言段落必须人工逐节核对；本工具输出"需人工核对的节清单"与差集，
  供 `report.md` 登记"仅见一侧"条目。

为什么不用行数或段落自动对齐（§17.1 要求的是内容镜像，不是行数镜像）：
- 中文行宽更短、英文换行更多 ⇒ 行数差与内容差无关。
- 自动段落对齐会在"同义不同词"和"换行方式不同"上大量误报，误报会掩盖真差异。
  因此把可判定的（token 集合）做死，把不可判定的（自然语言）显式交给人工。

用法：
    python3 tools/readme_mirror_check.py                 # 人读报告
    python3 tools/readme_mirror_check.py -o json         # 机器可读
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"`([^`\n]+)`")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

SECTION_PAIRS = [
    ("Quick Start", "快速开始"),
    ("CLI Reference", "CLI 参考"),
    ("Data Format", "数据格式"),
    ("Reward Scoring", "Reward 计分方式"),
    ("Token-Level Annotation", "Token 级标注"),
    ("Configuration", "配置说明"),
    ("LoRA Targets", "LoRA Targets"),
    ("Native Model Implementation Boundary", "Native 模型实现边界"),
    ("Export", "导出"),
    ("Outputs And Monitoring", "输出和监控"),
    ("Development", "开发检查"),
    ("FAQ", "常见问题"),
    ("License", "License"),
]

# 语言特定内容白名单（§17.6 精神：确实只应存在于一侧的，显式登记理由）。
LANGUAGE_SPECIFIC: list[dict] = [
    {
        "token": "tp=T,dp=D,pp=P",
        "side": "ZH",
        "reason": "中文侧用 `tp=T,dp=D,pp=P` 这种紧凑写法表达并行度组合，"
                  "英文侧在同一位置写作展开式 `tp_size × dp_size × pp_size`。"
                  "两侧语义完全相同，只是记号不同——不是内容缺失。",
    },
    {
        "token": "final/",
        "side": "EN",
        "reason": "英文侧一条列表项以 `final/` 结尾（写成 `final/`），"
                  "中文侧同一项写作 `final`（不带尾斜杠）。指向同一个产物，"
                  "为路径写法差异，不是内容缺失。",
    },
]


def split_sections(path: Path) -> tuple[list[tuple[str, int, list[str]]], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    preamble: list[str] = []
    current: list[str] | None = None
    infence = False
    for lineno, line in enumerate(lines, 1):
        if line.startswith("```"):
            infence = not infence
        if not infence and line.startswith("## "):
            current = []
            sections.append((line[3:].strip(), lineno, current))
            continue
        (current if current is not None else preamble).append(line)
    return sections, preamble


def tokens_of(lines: list[str], drop_fences: bool = False) -> set[str]:
    out: list[str] = []
    infence = False
    for line in lines:
        if line.startswith("```"):
            infence = not infence
            continue
        if drop_fences and infence:
            continue
        out.extend(TOKEN_RE.findall(line))
    return {t for t in out if t and not CJK_RE.search(t)}


def code_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    infence = False
    for line in lines:
        if line.startswith("```"):
            if infence:
                out.append("\n".join(buf))
                buf = []
            infence = not infence
            continue
        if infence:
            buf.append(line)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools/readme_mirror_check.py",
        description="§17.1 双语 README 镜像核验（token 完整性 + 人工核对清单）",
    )
    ap.add_argument("--en", default="README.md")
    ap.add_argument("--zh", default="README.zh-CN.md")
    ap.add_argument("-o", "--output", choices=("text", "json"), default="text")
    args = ap.parse_args(argv)

    en_secs, _ = split_sections(Path(args.en))
    zh_secs, _ = split_sections(Path(args.zh))

    problems: list[str] = []
    if len(en_secs) != len(zh_secs):
        problems.append(f"章节数不一致：EN {len(en_secs)} / ZH {len(zh_secs)}")

    report: list[dict] = []
    for idx, (en_title, en_line, en_body) in enumerate(en_secs):
        zh_title, zh_line, zh_body = (
            zh_secs[idx] if idx < len(zh_secs) else ("<缺失>", 0, [])
        )
        tok_en = tokens_of(en_body, drop_fences=True)
        tok_zh = tokens_of(zh_body, drop_fences=True)
        code_en = code_blocks(en_body)
        code_zh = code_blocks(zh_body)
        only_en = sorted(tok_en - tok_zh)
        only_zh = sorted(tok_zh - tok_en)
        # 语言特定例外扣除
        only_en = [t for t in only_en if not _exempt(t, "EN")]
        only_zh = [t for t in only_zh if not _exempt(t, "ZH")]
        lines_en = len([l for l in en_body if l.strip()])
        lines_zh = len([l for l in zh_body if l.strip()])
        report.append(
            {
                "index": idx,
                "en_title": en_title,
                "zh_title": zh_title,
                "en_line": en_line,
                "zh_line": zh_line,
                "en_nonblank_lines": lines_en,
                "zh_nonblank_lines": lines_zh,
                "en_code_blocks": len(code_en),
                "zh_code_blocks": len(code_zh),
                "only_en_tokens": only_en,
                "only_zh_tokens": only_zh,
                "needs_manual_review": bool(only_en or only_zh),
            }
        )

    if args.output == "json":
        print(
            json.dumps(
                {
                    "structure_problems": problems,
                    "sections": report,
                    "language_specific_exceptions": LANGUAGE_SPECIFIC,
                    "only_en_token_total": sum(len(r["only_en_tokens"]) for r in report),
                    "only_zh_token_total": sum(len(r["only_zh_tokens"]) for r in report),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if problems else 0

    print(f"§17.1 双语镜像核验：{args.en}  vs  {args.zh}")
    print(f"章节数：EN {len(en_secs)} / ZH {len(zh_secs)}" + ("  ✅" if not problems else "  ❌"))
    for p in problems:
        print(f"  ❌ {p}")
    print()
    only_en_total = sum(len(r["only_en_tokens"]) for r in report)
    only_zh_total = sum(len(r["only_zh_tokens"]) for r in report)
    print(f"内联代码 token 差集：仅 EN {only_en_total} 个 / 仅 ZH {only_zh_total} 个")
    print()
    for r in report:
        mark = "⚠️" if r["needs_manual_review"] else "✅"
        print(
            f"{mark} {r['en_title']} (EN:{r['en_line']}) | {r['zh_title']} (ZH:{r['zh_line']})"
            f"  非空行 {r['en_nonblank_lines']}/{r['zh_nonblank_lines']}"
            f"  代码块 {r['en_code_blocks']}/{r['zh_code_blocks']}"
        )
        if r["only_en_tokens"]:
            print(f"    仅 EN token: {r['only_en_tokens']}")
        if r["only_zh_tokens"]:
            print(f"    仅 ZH token: {r['only_zh_tokens']}")
    print()
    print("已登记的语言特定例外（不算差异）：")
    for e in LANGUAGE_SPECIFIC:
        print(f"  - [{e['side']}] `{e['token']}`：{e['reason']}")
    return 1 if problems else 0


def _exempt(token: str, side: str) -> bool:
    return any(e["token"] == token and e["side"] == side for e in LANGUAGE_SPECIFIC)


if __name__ == "__main__":
    raise SystemExit(main())
