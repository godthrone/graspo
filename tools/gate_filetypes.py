#!/usr/bin/env python3
"""§15.1 检查 2 — 文件类型门禁：不得被跟踪的本地/临时/产物/凭据文件。

职责边界（§1.1）：
- 只看「路径与文件名」，不读内容 ⇒ 与 `gate_secrets.py`（内容）/ `gate_content.py`
  （内部引用）不重叠。
- 判定对象是「会被提交/已被提交的路径」，不是工作区磁盘上的文件——磁盘上存在但
  未被跟踪的文件（如 `src/graspo.egg-info/`）不构成本门禁的违规。

判据来源：BADGE 宪法 §15.1 检查 2（`.env`、构建产物、缓存、虚拟环境、IDE 配置、
AI 助手本地文件）+ §16.1（本地与临时文件唯一归宿是 `.local/`）+ §19.2（`.gitignore`
必须覆盖清单）。
扫描目标：① `git ls-files`（当前索引路径）∪ `git diff --cached`（已暂存路径，
含本次新增但尚未提交的文件；违规=阻断）；② `git rev-list --objects --all`
的全历史路径集（含已删除文件，违规=警告，理由同 §19.1 历史不可改写）。

退出码：0 通过 / 1 发现阻断级违规 / 2 用法或环境错误 / 3 内部错误。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_common as gc  # noqa: E402

GATE = "gate_filetypes"
DESCRIPTION = "§15.1 检查 2：文件类型（.env / 构建产物 / 缓存 / IDE / AI 本地文件 / 凭据文件）"

# 每条：(规则名, 路径正则, 说明)。说明用于报告中解释"为什么这条路径不允许提交"。
PATH_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "dotenv_file",
        re.compile(r"(^|/)\.env(\.[A-Za-z0-9_-]+)?$"),
        "`.env` 永不入库（§15.1 检查 2 / §15.2）；只允许 `.env-example` 模板",
    ),
    (
        "dotenv_variant",
        re.compile(r"(^|/)[^/]*\.env$"),
        "`*.env` 变体同样永不入库（§19.2）",
    ),
    (
        "zone_identifier",
        re.compile(r":Zone\.Identifier$"),
        "Windows 下载标记（备用数据流）是本地垃圾文件，不得提交",
    ),
    (
        "ide_or_cache_dir",
        re.compile(
            r"(^|/)(\.idea|\.vscode|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.tox|node_modules|site-packages)(/|$)"
        ),
        "IDE 配置 / 缓存 / 虚拟环境目录不得入库（§15.1 检查 2、§19.2）",
    ),
    (
        "build_output_dir",
        re.compile(r"(^|/)(dist|build|outputs|venv|\.venv|venvs|\.eggs)(/|$)"),
        "构建产物与虚拟环境不得入库（§15.1 检查 2、§19.2）",
    ),
    (
        "build_artifact",
        re.compile(r"\.(pyc|pyo|pyd|so|o|a|dll|dylib|whl|egg-info|egg|jar|class)$"),
        "编译/打包产物不得入库（§15.1 检查 2）",
    ),
    (
        "archive_artifact",
        re.compile(r"\.(zip|tar|tgz|tar\.gz|gz|bz2|xz|7z|rar)$"),
        "归档产物不得入库（§15.1 检查 2）",
    ),
    (
        "ai_assistant_file",
        re.compile(
            r"(^|/)(CLAUDE|AGENTS|GEMINI|QWEN|COPILOT|CURSOR)(\.[A-Za-z0-9-]+)*\.(md|mdc|rules)$"
        ),
        "AI 助手本地指令/草稿文件必须 gitignore（§15.1 检查 2、§17.3、§19.2）",
    ),
    (
        "os_junk",
        re.compile(r"(^|/)(Thumbs\.db|\.DS_Store|desktop\.ini)$"),
        "系统垃圾文件不得入库（§19.2）",
    ),
    (
        "key_material",
        re.compile(
            r"(^|/)(id_rsa|id_dsa|id_ecdsa|id_ed25519)$|\.(pem|key|p12|pfx|jks|keystore|ppk)$"
        ),
        "私钥/证书材料不得入库（§15.1 检查 1+2）",
    ),
    (
        "credential_file",
        re.compile(r"(^|/)\.(netrc|pgpass|htpasswd|npmrc|pypirc|git-credentials)$"),
        "凭据文件不得入库（§15.1 检查 2）",
    ),
    (
        "local_dump",
        re.compile(r"\.(log|sqlite|sqlite3|db|dump|bak|orig|rej|swp|swo)$"),
        "日志/数据库/备份/编辑器残留不得入库（§15.1 检查 2、§19.2）",
    ),
]

# 允许的显式例外（路径精确匹配）。理由写在这里，不写进白名单 JSON——
# 这两条是“宪法明文的允许项”，不是误报。
ALLOWED_EXACT = {
    ".env-example": "§15.1 检查 2 明文例外：只有 `.env-example` 可提交",
}


def classify(path: str) -> tuple[str, str] | None:
    base = path.rsplit("/", 1)[-1]
    if base in ALLOWED_EXACT:
        return None
    for rule, pattern, why in PATH_RULES:
        if pattern.search(path):
            return rule, why
    return None


def _scan_paths(paths: list[str], origin: str) -> gc.ScanResult:
    result = gc.ScanResult(gate=GATE)
    for path in paths:
        hit = classify(path)
        if hit is None:
            continue
        rule, why = hit
        result.scanned_files += 1
        finding = gc.Finding(
            rule=rule,
            path=path,
            line=None,
            snippet=why,
            origin=origin,
        )
        if origin == "history":
            result.warnings.append(finding)
        else:
            result.violations.append(finding)
    return result


def _impl(args) -> gc.ScanResult:
    root = gc.repo_root(args.repo)
    result = gc.ScanResult(gate=GATE)
    if not args.history_only:
        # 扫描面 = 索引现存路径 ∪ 已暂存路径。后者覆盖"本次新增但尚未提交"的
        # 文件（如新落的 `.env`：`git ls-files` 看不到，但它一旦提交就进入历史）。
        tracked = gc.tracked_paths(root)
        staged = gc.staged_paths(root)
        covered = sorted(set(tracked) | set(staged))
        result.scanned_files += len(covered)
        gc.merge(result, _scan_paths(covered, "tracked"))
        result.notes.append(
            f"扫描面口径：索引 {len(tracked)} 个 ∪ 已暂存 {len(staged)} 个 "
            f"⇒ 唯一路径 {len(covered)} 个（含本次新增未提交文件）。"
        )
    if not args.tracked_only:
        history_paths = sorted(gc.collect_history_paths(root))
        result.scanned_files += len(history_paths)
        gc.merge(result, _scan_paths(history_paths, "history"))
        result.notes.append(
            "历史路径命中一律记 warning：§19.1 禁止回溯改写已推送历史；"
            "已删除文件同样进入扫描（§15.1「全部 git 历史」）。"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    return gc.run_gate(GATE, DESCRIPTION, _impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
