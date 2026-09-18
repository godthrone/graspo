#!/usr/bin/env python3
"""§15.1 检查 1 — 机密扫描门禁：密钥、密码、token、私钥。

职责边界（§1.1）：
- 只做「凭据材料」匹配——不是内网地址，也不是 PII。内网 IP/域名/路径属于
  §15.1 检查 3「内容」（由 `gate_content.py` 负责），个人邮箱/手机号属于
  §15.1 检查 4（由 `gate_pii.py` 负责），提交元数据属于检查 5
  （由 `gate_commit_identity.py` 负责）。五条门禁各自可独立跑、判定面互不重叠。
- 只看文本；二进制由 `gate_filetypes.py` 负责。

判据来源：BADGE 宪法 §15.1 检查 1 + §15.1 开篇「推送会公开三样东西」。
扫描目标（两者都要，缺一即不合规）：
  ① **将要进入本次提交的内容** = 索引暂存内容（`git diff --cached` ∩
     `git ls-files`，用 `git cat-file :path` 读暂存 blob，**包含本次新增但尚未
     提交的文件**）+ 工作区已跟踪文件的当前内容；
  ② 全部 git 历史（`git log --all -p`，含已删除文件的旧内容）。

退出码：0 通过 / 1 发现阻断级违规 / 2 用法或环境错误 / 3 内部错误。
用法见 `--help`；`-o` 支持 text（默认）/ json / quiet。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_common as gc  # noqa: E402

GATE = "gate_secrets"
DESCRIPTION = "§15.1 检查 1：机密扫描（密钥/密码/token/私钥）"

RULES: list[gc.LineRule] = [
    gc.LineRule(
        "private_key_block",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        transform=gc.mask_secret,
    ),
    gc.LineRule(
        "aws_access_key_id",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        transform=gc.mask_secret,
    ),
    gc.LineRule(
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
        transform=gc.mask_secret,
    ),
    gc.LineRule(
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        transform=gc.mask_secret,
    ),
    gc.LineRule(
        "anthropic_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
        transform=gc.mask_secret,
    ),
    gc.LineRule(
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        transform=gc.mask_secret,
    ),
    gc.LineRule(
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        transform=gc.mask_secret,
    ),
    # 硬编码凭据赋值：值 ≥16 字符且同时含字母与数字。排除占位符
    # （`<...>`、`REPLACE_ME`、`your-...`）与纯文档句子——§15.2 的豁免面。
    gc.LineRule(
        "hardcoded_credential",
        re.compile(
            r"""(?i)\b(?:password|passwd|secret|token|api[_-]?key|apikey|access[_-]?key"""
            r"""|private[_-]?key|auth[_-]?token|client[_-]?secret)\b\s*[:=]\s*["']?"""
            r"""(?=[^\s"']{16,})(?=[^\s"']*[A-Za-z])(?=[^\s"']*[0-9])(?![<{])(?!REPLACE_ME)"""
            r"""(?!your[-_])([A-Za-z0-9!@#$%^&*_\-+=/.]{16,})"""
        ),
        transform=gc.mask_secret,
    ),
]

# 历史命中降级为 warning：§19.1「已推送至公共仓库的历史提交不得回溯修改」，
# 历史泄露的处置走 §15.3（BFG / filter-repo + force-push），需维护者与用户拍板。
HISTORY_IS_WARNING = True


def _impl(args) -> gc.ScanResult:
    root = gc.repo_root(args.repo)
    whitelist = gc.load_whitelist(args)

    result = gc.ScanResult(gate=GATE)
    if not args.history_only:
        tracked = gc.scan_worktree(root, RULES, whitelist)
        tracked.gate = GATE
        gc.merge(result, tracked)
    if not args.tracked_only:
        history = gc.scan_history(root, RULES, whitelist)
        if HISTORY_IS_WARNING:
            history.warnings.extend(history.violations)
            history.violations.clear()
        gc.merge(result, history)
        result.notes.append(
            "历史命中一律记 warning：§19.1 禁止回溯改写已推送历史，"
            "维护者按 §15.3 决定是否重写（本脚本不阻断历史）。"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    return gc.run_gate(GATE, DESCRIPTION, _impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
