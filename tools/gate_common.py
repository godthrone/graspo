"""提交前门禁共用设施：扫描目标枚举、报告输出、白名单匹配。

职责边界（§1.1）：
- 本模块只提供「扫描目标从哪来」「命中怎么报」「误报怎么白名单」三件事。
- 具体匹配模式由各门禁脚本（`gate_secrets.py` 等）自己定义，本模块不内置任何规则。
- 不依赖第三方包（纯标准库），保证在受限环境（无 torch/pyyaml、Python 版本不受
  项目 `requires-python` 约束）下也能复跑。

对应宪法条款：
- §15.1「推送会公开三样东西」⇒ 扫描目标必须同时覆盖**已跟踪文件**与**全部 git 历史**。
- §2.3「边界校验即防呆」⇒ 违规必须报出文件路径 + 行号 + 命中原文位置，
  不得只说"发现违规"。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# ── 退出码语义（§10.1 / CLI 契约）────────────────────────────────────────────
EXIT_CLEAN = 0  # 扫描通过
EXIT_VIOLATION = 1  # 发现阻断级违规
EXIT_USAGE = 2  # 用法/环境错误（不是门禁结论）
EXIT_INTERNAL = 3  # 内部错误


# ── 命中记录 ────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    """一条违规命中。`origin` 区分"当前已跟踪内容"与"git 历史"。"""

    rule: str
    path: str
    line: int | None
    snippet: str
    origin: str  # "tracked" | "history"
    commit: str | None = None
    source_line: str = ""  # 命中所在行的完整原文（白名单上下文匹配用，不输出）

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "path": self.path,
            "line": self.line,
            "snippet": self.snippet,
            "origin": self.origin,
            "commit": self.commit,
        }

    def format_line(self) -> str:
        where = f"{self.path}:{self.line}" if self.line is not None else self.path
        trail = f" [commit {self.commit[:12]}]" if self.commit else ""
        return f"  - [{self.rule}] {where}{trail}: {self.snippet}"


@dataclass
class ScanResult:
    """一次扫描的完整结果，供文本/JSON 两种输出共用。

    `scanned_*` 记的是**扫描对象**计数：同一路径在"索引段"与"工作区段"各扫一次
    会被计两次；`skipped_binary` 是被文本门禁按二进制跳过的对象数（这些对象
    改由 `gate_filetypes.py` 按路径判定）。报告里两者都打印，避免"402 个对象"
    被读成"仓库只有 402 个文件"。
    """

    gate: str
    violations: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scanned_files: int = 0
    scanned_bytes: int = 0
    skipped_binary: int = 0
    unreadable: int = 0

    @property
    def exit_code(self) -> int:
        return EXIT_VIOLATION if self.violations else EXIT_CLEAN


# ── 白名单 ──────────────────────────────────────────────────────────────────
@dataclass
class WhitelistEntry:
    """一条误报白名单。

    `path_pattern` 用 `re.search` 匹配文件路径；`content_pattern` 匹配命中行原文
    （含命中片段本身）。`reason` 是宪法 §15.1 精神要求的"写明理由"，不可为空——
    为过检而删除误报逻辑是禁止的，但误报必须留下可审计的理由。
    """

    rule: str
    path_pattern: str
    content_pattern: str | None
    reason: str

    def matches(self, rule: str, path: str, snippet: str, line_text: str = "") -> bool:
        if self.rule not in ("*", rule):
            return False
        if not re.search(self.path_pattern, path):
            return False
        if self.content_pattern is None:
            return True
        target = line_text or snippet
        return re.search(self.content_pattern, target) is not None


class Whitelist:
    """误报白名单集合，从 JSON 文件加载。"""

    def __init__(self, entries: Sequence[WhitelistEntry] = ()) -> None:
        self.entries = list(entries)

    @classmethod
    def load(cls, path: Path | None) -> "Whitelist":
        if path is None or not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            WhitelistEntry(
                rule=item["rule"],
                path_pattern=item["path"],
                content_pattern=item.get("content"),
                reason=item["reason"],
            )
            for item in raw.get("whitelist", [])
        ]
        return cls(entries)

    def suppressed_reason(self, rule: str, path: str, snippet: str, line_text: str = "") -> str | None:
        for entry in self.entries:
            if entry.matches(rule, path, snippet, line_text):
                return entry.reason
        return None


DEFAULT_WHITELIST = Path(__file__).resolve().parent / "gate_whitelist.json"


# ── git 访问 ────────────────────────────────────────────────────────────────
def repo_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    out = _git(["rev-parse", "--show-toplevel"], cwd=Path.cwd(), check=False)
    if out is None:
        raise SystemExit("错误：当前目录不是 git 仓库（且未提供 --repo）")
    return Path(out.strip()).resolve()


def _git(args: Sequence[str], cwd: Path, check: bool = True) -> str | None:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        if check:
            raise RuntimeError(
                f"git {' '.join(args)} 失败（exit {proc.returncode}）："
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        return None
    return proc.stdout.decode("utf-8", "replace")


def tracked_paths(root: Path) -> list[str]:
    """当前索引中的全部已跟踪文件（相对仓库根的 posix 路径）。

    注意：这是**索引**口径。未 `git add` 的新文件不在其中，故"当前树"扫描
    不能只用本函数——见 `staged_paths()` 与 `scan_worktree()` 的口径说明。
    """
    out = _git(["ls-files", "-z"], cwd=root, check=False)
    if not out:
        # 兼容无 -z 的极老 git
        out = _git(["ls-files"], cwd=root) or ""
        return [p for p in out.splitlines() if p]
    return [p for p in out.split("\0") if p]


def staged_paths(root: Path) -> list[str]:
    """★ 已暂存（即将进入本次提交）的路径——扫描面的"将要提交的内容"口径。

    为什么必须有这个函数（§2.3 边界校验即防呆 / §15.1「已跟踪文件的内容」）：

    只扫 `git ls-files`（索引）有一个**结构性盲区**：一次**新增**文件在被
    `git add`、但尚未 commit 之前，它既不在 HEAD 里、也不在索引里——
    换句话说，**门禁看不到自己正在放行的内容**。实测后果有二：
      ① 交付门禁工具链时 `tools/` 还是 untracked ⇒ 五个门禁"全绿"是**假绿**
         （门禁从未审过自己）；
      ② 按该口径，"把第一版门禁提交进去"这个动作**必然被门禁自己拦下**。

    `git diff --cached --name-only` 报出的正是"索引里有、HEAD 里没有"的路径，
    其中**已暂存的删除**同样会被列出——删除的内容不在索引里，故必须用
    `git ls-files -z`（索引现存路径）求交，只保留**索引中确实存在**的路径。

    返回：去重排序后的 posix 路径列表。
    """
    out = _git(["diff", "--cached", "--name-only", "-z"], cwd=root, check=False) or ""
    if not out:
        out = _git(["diff", "--cached", "--name-only"], cwd=root, check=False) or ""
    candidates = {p for p in out.replace("\0", "\n").splitlines() if p}
    if not candidates:
        return []
    present = set(tracked_paths(root))
    return sorted(candidates & present)


def read_index_text(root: Path, rel: str, limit: int | None = None) -> str | None:
    """★ 读**索引**里该路径的内容（= 即将被提交的字节），二进制/不存在返回 None。

    为什么读索引而不是工作区：`git add` 之后、`commit` 之前，**被提交的内容
    是索引里的 blob**，工作区可能与它不一致（`git add` 后又手改、或
    `git checkout <commit> -- <path>` 只落暂存不动工作区）。只读工作区会
    出现"看得见却没审、审了又不是要提交的版本"两种错配。读索引即精确等于
    "这次提交会写进历史的字节"，因此暂存后跑门禁是**真实覆盖**，不是近似。

    白名单与规则匹配都用这份内容，命中行号即提交后的真实行号。
    """
    # `git cat-file` 对每个路径起一次进程，但暂存路径通常只有个位数到数十个；
    # 与全历史扫描（4~10 s）相比可忽略。
    out = _git(["cat-file", "-p", f":{rel}"], cwd=root, check=False)
    if out is None:
        return None
    data = out.encode("utf-8", "replace")
    if limit is not None and len(data) > limit:
        data = data[:limit]
    if not is_text_bytes(data):
        return None
    return data.decode("utf-8", "replace")


def read_worktree_text(root: Path, rel: str, limit: int | None = None) -> str | None:
    """读取工作区文件文本；二进制/不可读返回 None。"""
    path = root / rel
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if limit is not None and len(data) > limit:
        data = data[:limit]
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", "replace")


def is_text_bytes(data: bytes) -> bool:
    return b"\0" not in data[:8192]


def unstaged_diff_paths(root: Path) -> set[str]:
    """工作区与索引**不一致**的已跟踪路径集合（一次 `git diff` 取全集）。

    用于在报告里列出"未暂存改动"——它们本次不会被提交，故不在阻断面内，但要
    让操作者看见。用一次 `git diff --name-only` 取全集，避免逐路径起 git 进程
    （逐进程会让 500 个路径的扫描慢一个数量级）。
    """
    out = _git(["diff", "--name-only", "-z"], cwd=root, check=False) or ""
    if not out:
        out = _git(["diff", "--name-only"], cwd=root, check=False) or ""
    return {p for p in out.replace("\0", "\n").splitlines() if p}


def versions_range(root: Path, rev_range: str) -> Iterator[tuple[str, str]]:
    """在指定版本区间内遍历「改动过的文件内容」，产出 (path, content)。

    典型用途：pre-commit hook 只需扫 `--versions-range HEAD`（即本次提交相对
    上一次提交的改动），避免每次提交都全历史重扫。默认（不传该参数）仍按
    §15.1 原文扫全历史。
    """
    out = _git(["diff", "--name-only", rev_range], cwd=root, check=False)
    if out is None:
        out = _git(["diff-tree", "-r", "--root", "--name-only", rev_range], cwd=root, check=False) or ""
    for path in [p for p in out.splitlines() if p]:
        text = read_worktree_text(root, path)
        if text is not None:
            yield path, text.encode("utf-8", "replace")


def iter_history_blobs(root: Path, max_bytes: int = 4 * 1024 * 1024) -> Iterator[tuple[str, bytes]]:
    """遍历全部 git 历史中「带路径」的文本 blob（去重后）。产出 (path, content)。

    §15.1 原文「全部 git 历史（包括已删除的文件）」：本函数通过
    `git rev-list --objects --all` 枚举所有可达对象及其路径，因此**已删除的
    文件同样进入扫描**。同一内容（blob sha）仅扫描一次；单个 blob 超过
    `max_bytes` 时跳过——超大文件多为二进制，由 `gate_filetypes.py` 负责。

    实现用 `git cat-file --batch` 一次进程读全部对象，仓库放大到数万 blob
    时仍是秒级；逐对象起进程会慢一个数量级。
    """
    listing = _git(["rev-list", "--objects", "--all"], cwd=root) or ""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in listing.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, path = parts
        if sha in seen:
            continue
        seen.add(sha)
        pairs.append((sha, path))
    if not pairs:
        return
    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=str(root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        for sha, path in pairs:
            proc.stdin.write((sha + "\n").encode("ascii"))
            proc.stdin.flush()
            header = proc.stdout.readline().decode("utf-8", "replace").strip()
            if not header or header.endswith("missing"):
                continue
            fields = header.split()
            if len(fields) < 3:
                continue
            try:
                size = int(fields[2])
            except ValueError:
                continue
            if size > max_bytes:
                # 仍需把内容读掉，否则后续 header 错位
                _drain(proc.stdout, size)
                continue
            data = proc.stdout.read(size)
            proc.stdout.read(1)  # 尾随换行
            yield path, data
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.stdout.close()
        proc.wait()


def _drain(stream, size: int) -> None:
    remaining = size + 1
    while remaining > 0:
        chunk = stream.read(min(65536, remaining))
        if not chunk:
            break
        remaining -= len(chunk)


def iter_history_patches(root: Path) -> Iterator[tuple[str, str]]:
    """按 §15.1 原文口径流式遍历 `git log --all -p`，产出 (commit, 补丁文本)。

    用于内容/机密/PII 的历史扫描——补丁文本里 `+` 行即"曾经进入历史的内容"，
    已删除文件的旧内容也在其中。逐 commit 流式读取，避免把整份历史读进内存。
    """
    proc = subprocess.Popen(
        ["git", "log", "--all", "-p", "--pretty=format:%x01%H%x02%an <%ae>%x02%s"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    commit = "?"
    buf: list[str] = []
    try:
        for raw in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
            if raw.startswith("\x01"):
                if buf:
                    yield commit, "".join(buf)
                    buf = []
                meta = raw[1:].split("\x02")
                commit = meta[0] if meta else "?"
                buf.append(raw)
            else:
                buf.append(raw)
        if buf:
            yield commit, "".join(buf)
    finally:
        proc.stdout.close()
        proc.wait()


def strip_patch_line(line: str) -> str | None:
    """从补丁行还原"曾进入历史的原文"；非内容行返回 None。"""
    if line.startswith("+++") or line.startswith("---"):
        return None
    if line.startswith("@@"):
        return None
    if line.startswith("+"):
        return line[1:]
    # 上下文行也要扫：它同样是仓库历史里的内容
    if line.startswith(" "):
        return line[1:]
    return None


def current_branch_paths(root: Path, spec: str = "HEAD") -> list[str]:
    out = _git(["ls-tree", "-r", "--name-only", spec], cwd=root, check=False) or ""
    return [p for p in out.splitlines() if p]


# ── 行级扫描驱动 ────────────────────────────────────────────────────────────
class LineRule:
    """一条按行匹配的规则。

    `severity`：`violation`（阻断，退出码 1）或 `warning`（告警，不阻断）。
    `transform`：对命中文本的可选脱敏函数（机密类规则用，避免报告二次抄录机密）。
    """

    def __init__(
        self,
        name: str,
        pattern: re.Pattern[str],
        severity: str = "violation",
        transform=None,
        line_based: bool = True,
    ) -> None:
        self.name = name
        self.pattern = pattern
        self.severity = severity
        self.transform = transform
        self.line_based = line_based


def scan_text_lines(
    text: str,
    rules: Sequence[LineRule],
    path: str,
    origin: str,
    whitelist: Whitelist,
    commit: str | None = None,
    prefilter: Sequence[re.Pattern[str]] = (),
) -> tuple[list[Finding], list[Finding], list[str], int]:
    """按行跑规则，返回 (violations, warnings, notes, 命中所属行内容映射数)。

    `prefilter` 命中的整行会被跳过：用于排除"结构性不可能含目标信息"的行
    （例如 `uv.lock` 里 sha256 十六进制串会被手机号/银行卡正则误命中）。
    跳过理由必须由调用方在 docstring 与报告 note 里写明，不做静默丢弃。
    """
    violations: list[Finding] = []
    warnings: list[Finding] = []
    notes: list[str] = []
    line_hits = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(p.search(line) for p in prefilter):
            continue
        for rule in rules:
            m = rule.pattern.search(line)
            if not m:
                continue
            snippet = m.group(0) if m.group(0) else line.strip()
            if rule.transform is not None:
                snippet = rule.transform(snippet)
            reason = whitelist.suppressed_reason(rule.name, path, m.group(0) or line, line)
            if reason is not None:
                notes.append(f"{path}:{lineno} [{rule.name}] 白名单豁免：{reason}")
                continue
            line_hits += 1
            finding = Finding(
                rule=rule.name,
                path=path,
                line=lineno,
                snippet=_clip(snippet),
                origin=origin,
                commit=commit,
                source_line=line,
            )
            (violations if rule.severity == "violation" else warnings).append(finding)
    return violations, warnings, notes, line_hits


def _clip(text: str, width: int = 200) -> str:
    text = text.strip()
    if len(text) > width:
        return text[:width] + "…"
    return text


def mask_secret(text: str) -> str:
    """机密命中片段脱敏：保留前后各 6 字符，中间打码。

    门禁报告本身可能被贴进 issue，脱敏避免"扫描脚本把机密又抄了一遍"。
    """
    text = text.strip()
    if len(text) <= 14:
        return text[:4] + "*" * max(len(text) - 4, 0)
    return f"{text[:6]}{'*' * 8}{text[-6:]}"


# ── 参数与输出 ──────────────────────────────────────────────────────────────
def build_arg_parser(gate: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"tools/{gate}",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "退出码：0 = 通过；1 = 发现违规；2 = 用法错误；3 = 内部错误。\n"
            "示例：\n"
            f"  python3 tools/{gate}                 # 扫已跟踪文件 + 全部 git 历史\n"
            f"  python3 tools/{gate} --tracked-only  # 只扫已跟踪文件（快）\n"
            f"  python3 tools/{gate} -o json         # 机器可读输出\n"
            f"  python3 tools/{gate} -o quiet        # 只报计数，不列明细\n"
        ),
    )
    parser.add_argument("--repo", default=None, help="仓库根目录（默认自动探测）")
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="只扫已跟踪文件当前内容，跳过 git 历史（§15.1 要求默认含历史）",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="只扫 git 历史（含已删除文件）",
    )
    parser.add_argument(
        "--whitelist",
        default=None,
        help=f"误报白名单 JSON 路径（默认 {DEFAULT_WHITELIST.name}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=("text", "json", "quiet"),
        default="text",
        help="输出模式：text（默认，列明细）/ json / quiet（只报计数）",
    )
    parser.add_argument(
        "--max-note-lines",
        type=int,
        default=200,
        help="text 模式下最多打印多少条白名单豁免说明（默认 200）",
    )
    return parser


def merge(result: ScanResult, extra: ScanResult) -> ScanResult:
    """把 `extra` 合并进 `result`（扫描驱动分阶段跑时用）。"""
    result.violations.extend(extra.violations)
    result.warnings.extend(extra.warnings)
    result.notes.extend(extra.notes)
    result.scanned_files += extra.scanned_files
    result.scanned_bytes += extra.scanned_bytes
    result.skipped_binary += extra.skipped_binary
    result.unreadable += extra.unreadable
    return result


def report(result: ScanResult, output: str, max_note_lines: int = 200) -> int:
    """统一输出 + 返回退出码。"""
    scale = f"扫描 {result.scanned_files} 个文本对象"
    if result.scanned_bytes:
        scale += f" / {result.scanned_bytes} 字节"
    if result.skipped_binary or result.unreadable:
        parts = []
        if result.skipped_binary:
            parts.append(f"跳过二进制 {result.skipped_binary} 个（由 gate_filetypes 按路径判定）")
        if result.unreadable:
            parts.append(f"不可读 {result.unreadable} 个")
        scale += "；" + "、".join(parts)
    if output == "json":
        payload = {
            "gate": result.gate,
            "exit_code": result.exit_code,
            "scanned_files": result.scanned_files,
            "scanned_bytes": result.scanned_bytes,
            "skipped_binary": result.skipped_binary,
            "unreadable": result.unreadable,
            "violation_count": len(result.violations),
            "warning_count": len(result.warnings),
            "note_count": len(result.notes),
            "violations": [f.as_dict() for f in result.violations],
            "warnings": [f.as_dict() for f in result.warnings],
            "notes": result.notes,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return result.exit_code

    if output == "quiet":
        print(
            f"[{result.gate}] {scale}；违规 {len(result.violations)}、"
            f"警告 {len(result.warnings)}、说明 {len(result.notes)}"
        )
        if result.violations:
            print(f"[{result.gate}] ❌ 不通过")
            return EXIT_VIOLATION
        print(f"[{result.gate}] ✅ 通过")
        return EXIT_CLEAN

    # text
    print(f"[{result.gate}] {scale}")
    if result.violations:
        print(f"\n❌ 阻断级违规 {len(result.violations)} 条：")
        for finding in result.violations:
            print(finding.format_line())
    if result.warnings:
        print(f"\n⚠️  警告 {len(result.warnings)} 条（不阻断）：")
        for finding in result.warnings:
            print(finding.format_line())
    if result.notes:
        shown = result.notes[:max_note_lines]
        print(f"\nℹ️  说明 {len(result.notes)} 条（白名单理由 / 判定依据 / 覆盖范围）：")
        for note in shown:
            print(f"  - {note}")
        if len(result.notes) > len(shown):
            print(f"  … 其余 {len(result.notes) - len(shown)} 条见 -o json")
    if result.violations:
        print(f"\n[{result.gate}] ❌ 不通过（退出码 {EXIT_VIOLATION}）")
        return EXIT_VIOLATION
    print(f"\n[{result.gate}] ✅ 通过（退出码 {EXIT_CLEAN}）")
    return EXIT_CLEAN


def run_gate(gate: str, description: str, impl, argv: Sequence[str] | None = None) -> int:
    """门禁脚本统一入口：解析参数 → 调 impl(args) → 输出 → 返回退出码。"""
    parser = build_arg_parser(gate, description)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)
    try:
        result = impl(args)
    except RuntimeError as exc:
        print(f"[{gate}] 环境错误：{exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - 门禁脚本必须给出明确退出码
        import traceback

        print(f"[{gate}] 内部错误：{exc}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_INTERNAL
    return report(result, args.output, args.max_note_lines)


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


def load_whitelist(args: argparse.Namespace) -> Whitelist:
    path = Path(args.whitelist).resolve() if args.whitelist else DEFAULT_WHITELIST
    if not path.exists():
        return Whitelist()
    return Whitelist.load(path)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0", "false", "False")


def scan_worktree(
    root: Path,
    rules: Sequence[LineRule],
    whitelist: Whitelist,
    prefilter: Sequence[re.Pattern[str]] = (),
) -> ScanResult:
    """扫「将要进入本次提交的内容」（§15.1 公开物之一）——**索引 + 工作区**。

    两段口径，互不重叠、合起来覆盖"提交会公开什么"：

    1. **索引内容**（`git diff --cached` ∩ `git ls-files`，用 `git cat-file :path`
       读暂存 blob）：即**这次 commit 真正要写进历史的字节**，包含**本次新增
       但尚未提交的文件**。这是原来的缺口所在——缺了这一段，新增文件在
       `git add` 之前完全不在扫描面内。
    2. **工作区已跟踪文件**：索引中已跟踪、但工作区有新改动（未 `git add`）的
       路径也读一遍，使"工作区当前内容"同样可见（§15.1 口径是已跟踪文件的
       内容）。闸门在提交前跑，`pre-commit` 的语义本就是"检查工作区"。

    因此 `scanned_files` 大于"文件数"是正常的（同一路径可能两段各计一次），
    报告里的计数是**扫描对象数**而非文件数——此语义在 `-o` 输出与 `note` 中
    明示，避免读报告的人把 512 当成"仓库有 512 个文件"。
    """
    result = ScanResult(gate="worktree")

    def _consume(content: str | None, rel: str, origin: str, exists: bool) -> None:
        """把一份内容交给规则；content=None = 二进制或缺失（记入 skipped/unreadable）。

        `exists` 由调用方给出（索引路径必然存在；工作区路径可能已被删除），
        避免在判定分支里再起一次 git 进程。
        """
        if content is None:
            if exists:
                result.skipped_binary += 1
            else:
                result.unreadable += 1
            return
        result.scanned_files += 1
        result.scanned_bytes += len(content.encode("utf-8", "replace"))
        violations, warnings, notes, _ = scan_text_lines(
            content, rules, rel, origin, whitelist, prefilter=prefilter
        )
        result.violations.extend(violations)
        result.warnings.extend(warnings)
        result.notes.extend(notes)

    staged_set = set(staged_paths(root))
    tracked = tracked_paths(root)
    dirty = unstaged_diff_paths(root)
    unstaged_diffs: list[str] = []
    seen: set[str] = set()

    def _consume_path(rel: str, origin: str) -> None:
        """按索引读取并扫描一个路径（单位路径只扫一次，不重复计数）。"""
        if rel in seen:
            return
        seen.add(rel)
        content = read_index_text(root, rel)
        _consume(content, rel, origin, exists=True)
        if content is not None and rel not in staged_set and rel in dirty:
            unstaged_diffs.append(rel)

    # ── 索引内容：覆盖全部已跟踪路径 + 本次新增的已暂存路径 ────────────────
    # 单位路径只扫一次（不因"已暂存"而重复计数）。
    for rel in tracked:
        _consume_path(rel, "staged" if rel in staged_set else "tracked")
    # 兜底：已暂存但不在 `git ls-files` 输出中的路径（正常不会出现）
    for rel in sorted(staged_set):
        _consume_path(rel, "staged")

    coverage = sorted(set(tracked) | staged_set)
    result.notes.append(
        f"扫描面口径：索引（已跟踪 {len(tracked)} 个 ∪ 已暂存 {len(staged_set)} 个）"
        f"⇒ 覆盖 {len(coverage)} 个唯一路径。**扫的是索引里的内容**——即这次提交"
        "真正会写进历史的字节（含本次新增但尚未提交的文件）。"
    )
    if staged_set:
        shown = sorted(staged_set)[:20]
        result.notes.append(
            "本次已暂存（即将提交）的路径："
            + "、".join(shown)
            + (f" …（共 {len(staged_set)} 个）" if len(staged_set) > len(shown) else "")
        )
    if unstaged_diffs:
        shown = sorted(unstaged_diffs)[:20]
        result.notes.append(
            f"⚠️ 有 {len(unstaged_diffs)} 个已跟踪路径的工作区内容与索引不一致"
            "（**未暂存**，本次不会被提交，故不在本门禁的阻断面内）："
            + "、".join(shown)
            + (f" …（共 {len(unstaged_diffs)} 个）" if len(unstaged_diffs) > len(shown) else "")
            + "。如需让它们进入扫描，先 `git add`。"
        )
    return result


def scan_history(
    root: Path,
    rules: Sequence[LineRule],
    whitelist: Whitelist,
    prefilter: Sequence[re.Pattern[str]] = (),
) -> ScanResult:
    """扫「全部 git 历史」的文件内容（§15.1 公开物之二，含已删除文件）。

    产出 `path` = 真实历史路径、`line` = 真实行号、`commit` = 该 blob 的 sha
    （内容地址）。路径为 `<history:...>` 前缀以避免与当前树文件混淆，但保留
    原始路径末段，便于人直接 `git show` 复核。

    白名单匹配用**真实路径**（去掉 `<history:...>` 前缀），这样 `uv.lock` 这类
    路径级白名单对当前树与历史同时生效，不需要重复登记一次历史条目。
    """
    result = ScanResult(gate="history")
    for path, data in iter_history_blobs(root):
        if not is_text_bytes(data):
            continue
        text = data.decode("utf-8", "replace")
        result.scanned_files += 1
        result.scanned_bytes += len(data)
        label = f"<history:{path}>"
        violations, warnings, notes, _ = scan_text_lines(
            text, rules, label, "history", whitelist, prefilter=prefilter
        )
        # 白名单在 scan_text_lines 内用 label 匹配；这里补一次用真实路径的判定，
        # 使路径级白名单（如 `^uv\.lock$`）对历史同样生效。
        filtered_v: list[Finding] = []
        for finding in violations:
            reason = whitelist.suppressed_reason(
                finding.rule, path, finding.snippet, finding.source_line
            )
            if reason is not None:
                notes.append(f"history:{path} [{finding.rule}] 白名单豁免：{reason}")
            else:
                filtered_v.append(finding)
        filtered_w: list[Finding] = []
        for finding in warnings:
            reason = whitelist.suppressed_reason(
                finding.rule, path, finding.snippet, finding.source_line
            )
            if reason is not None:
                notes.append(f"history:{path} [{finding.rule}] 白名单豁免：{reason}")
            else:
                filtered_w.append(finding)
        result.violations.extend(filtered_v)
        result.warnings.extend(filtered_w)
        result.notes.extend(notes)
    return result


def _commit_label(commit: str) -> str:
    return f"<history:{commit[:12]}>"


def collect_history_paths(root: Path) -> set[str]:
    """全部历史中出现过的路径集合（含已删除文件）。"""
    out = _git(["rev-list", "--objects", "--all"], cwd=root) or ""
    paths: set[str] = set()
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            paths.add(parts[1])
    return paths
