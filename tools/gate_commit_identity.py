#!/usr/bin/env python3
"""§15.1 检查 5 — 提交元数据门禁：作者/提交者身份与提交信息隐私。

职责边界（§1.1）：
- 只管 git 元数据层：`user.name` / `user.email`（当前生效身份与被提交历史）、
  提交信息正文（§15.1 原文「提交信息同样纳入扫描」）。
- 不管文件内容——那是 `gate_secrets.py` / `gate_pii.py` / `gate_content.py`。

★ 判定语义：**合规性 allowlist，而不是"命中某个真实身份的黑名单"**
  （设计理由见下「为什么不是黑名单」）：
  - **允许**（`ALLOWED_PATTERNS`，对完整地址做 `fullmatch`）：
    托管平台的 `noreply` 地址、项目/机器人公开邮箱、保留域（`example.com` 等）、
    以及**由配置声明**的项目自有邮箱/域名（`gate.identity.allowed-emails` /
    `gate.identity.allowed-domains`，见下）。
  - **拒绝**：其余一切"不属于本项目公开身份"的地址——包括个人邮箱与机构
    （公司/单位）邮箱。判定是"它**不是**项目允许的身份"，**不是**"它等于某个
    写死在代码里的真实邮箱"。
  - 拒绝时区分两类便于处置：命中 `CONSUMER_MAIL_DOMAINS`（个人 ISP 邮箱域名）
    记 `personal_isp_domain`，其余记 `unexpected_identity`（非项目允许身份，
    通常是机构邮箱）。两条都是阻断级。

为什么不是黑名单（本次修复的设计缺陷）：
  原实现把"本机已知的真实个人+公司邮箱"当作 `PERSONAL_IDENTITY` 黑名单字面量
  写在源码里。那等于**把要防的机密抄进门禁源码**——门禁自己成了泄露源，且换个
  开发者/换台机器就立刻失效（黑名单里没有他的邮箱）。§15.1 检查 5 要的是
  "允许 noreply/项目邮箱，拒绝个人邮箱"这条**合规判据**；合规判据不依赖任何
  真实身份字面量，因此本文件（及本仓库任何已跟踪文件）都**不得出现真实
  个人/机构身份串**。真实域名只允许出现在 **gitignored 的 `.local/` 配置或
  `git config`** 里（§15.2：机密走本地覆写，不入 git）。

配置来源（§1.4 单一真相源；不引入新配置文件）：
  `git config --local gate.identity.allowed-domains`  逗号/空白分隔；空串元素 = 任意域名
  `git config --local gate.identity.allowed-emails`   逗号/空白分隔（完整地址）
  例（**在 .local/ 或本地 shell 里执行，不得写进仓库**）：
      git config --local gate.identity.allowed-domains "corp.example,example.com"
  未配置时默认集合就是"noreply + 保留域 + 平台机器人"——**失败方向是收紧**
  （配置缺失拒绝更多，不会放行）。

历史中的个人邮箱：记 **warning 而非阻断**。理由：§19.1「历史连贯优先于
回溯修正」+「已推送至公共仓库的历史提交不得回溯修改」；历史泄露的正确处置
是 §15.3（`filter-repo --email-callback` + force-push + 平台清缓存），
需要用户拍板，门禁不替用户做破坏性决定。

输出脱敏：本脚本会在报告中回显"本机全局身份"等**外部取得**的地址。回显一律
经 `mask_email` 脱敏——否则本脚本的输出本身会把真实邮箱二次抄进 issue/终端
日志/门禁报告（与 `gate_secrets.py` 的 `mask_secret` 同一纪律）。

退出码：0 通过 / 1 发现阻断级违规 / 2 用法或环境错误 / 3 内部错误。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_common as gc  # noqa: E402

GATE = "gate_commit_identity"
DESCRIPTION = "§15.1 检查 5：提交元数据隐私（作者/提交者邮箱 + 提交信息）"

# ── 允许的身份形态（合规判据，全部是**形态**而非真实值）─────────────────────
# 用 fullmatch 逐条匹配完整地址。**不要**把这些模式合并成一个带 `|` 与 `^...$`
# 的大正则再用 search——`^`/`$` 在多分支里容易只锚定到子串，历史上就出过
# `^(?:...|gitlab|...)@` 让「本地部分以 bot 结尾」的非平台地址因为含 `bot@` 而被误放行的问题。
ALLOWED_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 托管平台 noreply：70987748+godthrone@users.noreply.github.com
    re.compile(r"(?i).*\+.*@users\.noreply\.github\.com"),
    re.compile(r"(?i).*\+.*@noreply\.(?:github|gitlab)\.com"),
    # 通用 noreply 前缀：noreply@ / no-reply@ / donotreply@（任意域名）
    re.compile(r"(?i)(?:noreply|no-reply|donotreply)@[^@]+"),
    re.compile(r"(?i)[^@]+@(?:users\.)?noreply\.github\.com"),
    # §15.1 检查 4/5 明文豁免的保留域与无效域
    re.compile(r"(?i)[^@]+@(?:example\.(?:com|org|net)|invalid|localhost|example\.invalid)"),
    # 平台机器人/CI 公开邮箱：本地部分必须恰好是这些公开角色名（fullmatch 锚定
    # 整个地址），且域名必须是托管平台的 noreply——本地部分"恰好以 bot 开头"
    # 的非平台地址（例如本地部分以 bot 结尾的地址）不得被放行。
    re.compile(r"(?i)(?:github|gitlab|actions|ci|bot|noreply)@noreply\.github\.com"),
    re.compile(r"(?i)(?:github|gitlab|actions|ci|bot)@users\.noreply\.github\.com"),
)

# ── 个人 ISP 邮箱域名（消费者邮件服务）────────────────────────────────────
# 这是**公开的邮件服务商域名清单**，不是任何人的身份信息；它只用于把"个人邮箱"
# 与"机构邮箱"在报告里分开，便于处置。注意：允许集合之外的地址一律阻断，
# 所以本清单**即使为空也不会放宽判定**（它只影响报告分类，不影响通过/拒绝）。
CONSUMER_MAIL_DOMAINS = re.compile(
    r"(?i)@(?:qq\.com|163\.com|126\.com|sina\.(?:com|cn)|sohu\.com|foxmail\.com|"
    r"outlook\.com|hotmail\.com|gmail\.com|icloud\.com|139\.com|189\.cn|"
    r"21cn\.com|tom\.com|yeah\.net)$"
)

# 提交信息里的 PII（§15.1 原文「提交信息同样纳入扫描」）。
# 豁免：AI 助手/机器人的 vendor noreply 署名（`Co-Authored-By: Claude
# <noreply@anthropic.com>`）不是自然人邮箱，属 §15.1 检查 4 的 `noreply@` 豁免面。
MESSAGE_EMAIL_EXEMPT = re.compile(
    r"(?i)<(?:noreply|no-reply|donotreply)@"
    r"(?:anthropic\.com|openai\.com|github\.com|google\.com|microsoft\.com|"
    r"cursor\.com|jetbrains\.com)>"
    r"|@(?:example\.(?:com|org|net)|invalid)$"
)

MESSAGE_RULES: list[gc.LineRule] = [
    gc.LineRule(
        "personal_email_in_message",
        re.compile(
            r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+"
            r"(?:com|cn|net|org|io|edu|gov|xyz|top|vip|me|co|cc|info|biz)\b"
        ),
    ),
    gc.LineRule("cn_mobile_in_message", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    # 私网 IPv4：与 gate_content 同口径（四段完整地址；两处必须同步修改——
    # 原实现在 `172.16-31` 分支漏了一个 `\.`、在 `10` 分支只补两段，导致
    # `10.x.x.x` 与 `172.16-31.x.x` 整段漏检）。
    gc.LineRule(
        "internal_ip_in_message",
        re.compile(
            r"(?<![\dA-Za-z_.\-/])"
            r"((?:(?:192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}"
            r"|10(?:\.\d{1,3}){3}))"
            r"(?!\d)(?!\.\d)"
        ),
    ),
]


def mask_email(value: str) -> str:
    """把地址脱敏为 `ab***@ex***.com` 形态（保留可核对的前后缀，抹掉身份部分）。

    门禁报告可能被贴进 issue / CI 日志，回显真实地址等于二次泄露；与
    `gate_common.mask_secret` 同一纪律，故本脚本一切"外部取得的地址"都先脱敏。
    """
    value = value.strip()
    if "@" not in value:
        return value[:2] + "***" if len(value) > 2 else "***"
    local, _, domain = value.partition("@")
    keep_local = local[:2]
    head, _, tail = domain.rpartition(".")
    keep_domain = (head[:2] if head else domain[:2])
    return f"{keep_local}***@{keep_domain}***" + (f".{tail}" if tail else "")


def is_allowed_identity(email: str, extra_emails: frozenset[str], extra_domains: tuple[str, ...]) -> bool:
    """判定完整地址是否属于"项目允许的公开身份"。

    `extra_emails` / `extra_domains` 来自本地 git config（真实域名不入代码）。
    """
    email = email.strip()
    if not email or "@" not in email:
        return False
    lowered = email.lower()
    if lowered in {e.lower() for e in extra_emails}:
        return True
    _, _, domain = lowered.rpartition("@")
    for entry in extra_domains:
        entry = entry.lower()
        if entry == "" or domain == entry or domain.endswith(f".{entry}"):
            return True
    return any(pattern.fullmatch(email) for pattern in ALLOWED_PATTERNS)


def _split_config_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,\s]+", value) if item.strip())


def _identity_config(root: Path) -> tuple[frozenset[str], tuple[str, ...]]:
    """从仓库级 git config 读项目自有身份（§1.4 单一真相源，无 fallback 链）。"""
    emails = gc._git(
        ["config", "--local", "--get", "gate.identity.allowed-emails"], cwd=root, check=False
    )
    domains = gc._git(
        ["config", "--local", "--get", "gate.identity.allowed-domains"], cwd=root, check=False
    )
    email_set = frozenset(_split_config_list(emails or ""))
    domain_set = frozenset(_split_config_list(domains or ""))
    return email_set, tuple(sorted(domain_set))


def _git_lines(root: Path, args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return []
    return proc.stdout.decode("utf-8", "replace").split("\x1e\n")


def _config_map(root: Path, scope: str) -> dict[str, str]:
    out = gc._git(["config", scope, "--list"], cwd=root, check=False) or ""
    values: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def classify_identity(
    email: str, extra_emails: frozenset[str], extra_domains: tuple[str, ...]
) -> str | None:
    """返回 None（允许）或规则名（阻断）。规则名只反映"为什么不允许"。"""
    if is_allowed_identity(email, extra_emails, extra_domains):
        return None
    if CONSUMER_MAIL_DOMAINS.search(email):
        return "personal_isp_domain"
    return "unexpected_identity"


def _impl(args) -> gc.ScanResult:
    root = gc.repo_root(args.repo)
    allowed_emails, allowed_domains = _identity_config(root)
    result = gc.ScanResult(gate=GATE)

    # ── ① 仓库级生效身份（阻断对象）────────────────────────────────────────
    local = _config_map(root, "--local")
    effective_name = local.get("user.name", "")
    effective_email = local.get("user.email", "")
    result.scanned_files += 1
    result.notes.append(
        "仓库级生效身份："
        f"user.name={effective_name or '(未设置)'} / "
        f"user.email={effective_email or '(未设置)'}"
    )
    result.notes.append(
        "判定语义：合规 allowlist（noreply + 保留域 + 平台机器人 + 本地 git config "
        "`gate.identity.allowed-emails`/`allowed-domains` 声明的项目自有身份）；"
        "不在允许集合内的地址一律阻断——**不依赖任何写死在代码里的真实身份字面量**。"
    )

    global_cfg = _config_map(root, "--global")
    global_email = global_cfg.get("user.email", "")
    if global_email and global_email != effective_email:
        # ★ 脱敏：全局身份是"外部取得的数据"，原样回显 = 门禁输出二次抄录个人邮箱。
        result.notes.append(
            f"全局 user.email={mask_email(global_email)}（与本仓库不同，已脱敏）。"
            "§2 防呆提示：仓库级覆盖是**约定**而非装置——换 clone/未继承 local config 时"
            "全局邮箱会进入提交（本门禁的生效身份检查正是拦这一条）。"
        )

    if not effective_email:
        result.violations.append(
            gc.Finding(
                rule="missing_repo_identity",
                path="<git config --local>",
                line=None,
                snippet="仓库级 user.email 未设置 ⇒ 提交可能继承全局邮箱（可能是个人邮箱）",
                origin="tracked",
            )
        )
    else:
        rule = classify_identity(effective_email, allowed_emails, allowed_domains)
        if rule is not None:
            result.violations.append(
                gc.Finding(
                    rule=rule,
                    path="<git config --local>",
                    line=None,
                    snippet=(
                        f"仓库级生效身份 `{effective_name} <{mask_email(effective_email)}>`"
                        "（已脱敏）不属于本项目允许的公开身份。§15.1 检查 5 要求公开"
                        "仓库不得使用个人邮箱或机构邮箱；如需纳入项目自有地址，用"
                        "`git config --local gate.identity.allowed-domains <域名>` 声明"
                        "（本地配置，不入 git）。"
                    ),
                    origin="tracked",
                )
            )

    # ── ② 全部提交的元数据（历史，警告级）────────────────────────────────────
    if args.tracked_only:
        result.notes.append("--tracked-only：跳过提交历史元数据扫描。")
        return result

    raw = gc._git(
        ["log", "--all", "--format=%H%x1f%an <%ae>%x1f%cn <%ce>%x1e"], cwd=root, check=False
    ) or ""
    seen_identities: dict[str, int] = {}
    history_bad: dict[str, str] = {}
    for record in raw.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x1f")
        if len(parts) < 3:
            continue
        commit, author, committer = parts[0].strip(), parts[1], parts[2]
        result.scanned_files += 1
        for role, ident in (("author", author), ("committer", committer)):
            seen_identities[ident] = seen_identities.get(ident, 0) + 1
            email = ident.rsplit("<", 1)[-1].rstrip(">").strip()
            if classify_identity(email, allowed_emails, allowed_domains) is None:
                continue
            history_bad.setdefault(f"{role}:{ident}", commit)

    for ident, commit in history_bad.items():
        role, _, who = ident.partition(":")
        result.warnings.append(
            gc.Finding(
                rule="personal_identity_in_history",
                path=f"<git log --all: {role}>",
                line=None,
                snippet=f"{mask_email(who)}（示例提交 {commit[:12]}）——不属于本项目允许的公开身份",
                origin="history",
                commit=commit,
            )
        )
    result.notes.append(
        "历史身份汇总（唯一值，已脱敏）："
        + "；".join(f"{mask_email(k)} ×{v}" for k, v in sorted(seen_identities.items()))
    )
    result.notes.append(
        "历史个人邮箱记 warning：§19.1 禁止回溯改写已推送历史；"
        "处置走 §15.3（filter-repo --email-callback + force-push），需用户拍板。"
    )

    # ── ③ 提交信息正文（§15.1「提交信息同样纳入扫描」）───────────────────────
    # 同一句式会在几百个提交里重复出现，报告按 (规则, 行文) 聚合，每类只列
    # 前 3 条实例并给出总数——否则 text 输出会被大量同义告警淹没。
    messages = _git_lines(
        root, ["log", "--all", "--format=%H%n%B%x1e", "--no-merges"]
    )
    aggregated: dict[tuple[str, str], list[str]] = {}
    commit_sha = ""
    for chunk in messages:
        commit_sha = ""
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.fullmatch(r"[0-9a-f]{40}", line):
                commit_sha = line
                continue
            if MESSAGE_EMAIL_EXEMPT.search(line):
                continue
            for rule in MESSAGE_RULES:
                if not rule.pattern.search(line):
                    continue
                key = (f"{rule.name}@commit_message", gc._clip(line, 120))
                aggregated.setdefault(key, []).append(commit_sha[:12] or "?")
    for (rule_name, text), commits in sorted(aggregated.items()):
        result.warnings.append(
            gc.Finding(
                rule=rule_name,
                path=f"<commit message, 合计 {len(commits)} 处>",
                line=None,
                snippet=text,
                origin="history",
                commit=commits[0] if commits else None,
            )
        )
    result.notes.append(
        "提交信息正文命中记 warning（改动已推送的提交信息等同重写历史，§19.1）；"
        f"已按 (规则, 行文) 聚合，共 {len(aggregated)} 类。"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    return gc.run_gate(GATE, DESCRIPTION, _impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
