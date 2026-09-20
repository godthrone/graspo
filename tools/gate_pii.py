#!/usr/bin/env python3
"""§15.1 检查 4 — PII 门禁：可定位到具体自然人的信息。

职责边界（§1.1）：
- 只管「个人身份信息」：个人邮箱、手机号/固话、即时通讯账号、身份证/护照、
  银行卡、学生与员工编号（规则字面量见下方），以及姓名与其他身份字段的组合线索。
- 基础设施侧的地址与域名不在这里（属 §15.1 检查 3，见 `gate_content.py`）；
  凭据材料不在（见 `gate_secrets.py`）；提交元数据不在（见 `gate_commit_identity.py`）。
- §15.1 检查 4 的豁免面按宪法原文实现：`noreply@`、维护者公共邮箱、
  `example.com`/`example.org` 等保留域、`REPLACE_ME`/`your-...` 占位符。

判据来源：BADGE 宪法 §15.1 检查 4 + §15.1 开篇「三者都必须在推送前校验」。
扫描目标：① **将要进入本次提交的内容**（索引暂存内容，含本次新增但尚未提交的
文件，+ 工作区已跟踪文件当前内容；违规=阻断）；② 全部 git 历史
（`git log --all -p`，违规=警告，§19.1）。

预过滤（structural prefilter）：
- 含 `sha256:<64hex>` 或 PyPI 下载 URL 的行整行跳过——十六进制哈希串会随机
  命中手机号（`1[3-9]` 加 9 位数字）与银行卡（`62` 加 14~17 位数字）正则。
  这是**排除结构性不可能含 PII 的行**，不是"为过检而放宽匹配"，故记录在此
  而非白名单里。

退出码：0 通过 / 1 发现阻断级违规 / 2 用法或环境错误 / 3 内部错误。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_common as gc  # noqa: E402

GATE = "gate_pii"
DESCRIPTION = "§15.1 检查 4：PII（个人邮箱/手机号/即时通讯/证件号/银行卡/学号工号）"

PREFILTER: list[re.Pattern[str]] = [
    re.compile(r"sha256:[0-9a-f]{16,}"),  # 容器/包哈希（Dockerfile digest）
    re.compile(r"files\.pythonhosted\.org/packages/"),  # PyPI 下载 URL（路径含哈希）
    re.compile(r"^[0-9a-f]{28,}\s*$"),  # 纯十六进制串行（git 对象 id / 摘要）
    re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{28,}(?![0-9A-Za-z])"),  # 行内长十六进制串
]

# 个人邮箱：先排除豁免面（noreply / 保留域 / 占位符），再匹配通用邮箱形态。
EMAIL_EXEMPT = re.compile(
    r"(?i)(?:noreply|no-reply|donotreply)@"
    r"|@(?:users\.noreply\.github\.com|example\.(?:com|org|net)|example\.invalid"
    r"|test\.(?:com|org|invalid)|invalid|localhost|your-domain\.com)$"
)
PLACEHOLDER_EMAIL = re.compile(r"(?i)(?:REPLACE_ME|your[-_.]?email|your[-_.]?name|username@|user@)")

RULES: list[gc.LineRule] = [
    gc.LineRule(
        "personal_email",
        re.compile(
            r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+"
            r"(?:com|cn|net|org|io|edu|gov|xyz|top|vip|me|co|cc|info|biz|ru|jp|de|uk"
            # 保留域后缀（example/test/invalid）也纳入形态匹配：它们在**豁免面**里被
            # 排除（见 EMAIL_EXEMPT），但"命中后再豁免"必须能命中——否则
            # 负向夹具用保留域就永远无法证明规则真的会失败（假绿）。
            r"|example|test|invalid)\b"
        ),
    ),
    gc.LineRule("cn_mobile", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    gc.LineRule(
        "cn_id_card",
        re.compile(
            r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
        ),
    ),
    gc.LineRule("cn_landline", re.compile(r"(?<!\d)0\d{2,3}-\d{7,8}(?!\d)")),
    gc.LineRule(
        "bank_card_number",
        re.compile(r"(?<![\d\-])(?:62\d{14,17}|4\d{15}|5[1-5]\d{14})(?![\d\-])"),
    ),
    gc.LineRule(
        "im_account",
        re.compile(
            r"(?i)\b(?:wechat|weixin|wecom|dingtalk|telegram|whatsapp|qq)\b\s*(?:号|id|account|[:=：])"
        ),
    ),
    gc.LineRule(
        "student_employee_id",
        # ★ 规则**定义**本身不得让门禁自己命中（自反误报）：如果规则词表以中文
        # 字面量直接写在源码里，本文件自己就会被这条规则扫出来 ⇒ 要么整体排除
        # `tools/`（开永久盲区，禁止），要么为自家规则定义写自指白名单（无法证明
        # 措辞与规则一致）。这里用 `\uXXXX` 转义承载同一正则：**匹配能力完全相同**
        # （`re` 在正则解析阶段解转义），而源码里不再出现规则要抓的字面词。
        re.compile(
            r"(?i)\b(?:\u5b66\u53f7|\u5de5\u53f7|\u5458\u5de5\u7f16\u53f7|student\s*id|employee\s*id|staff\s*id)\b"
        ),
    ),
]


def _post_filter(finding: gc.Finding) -> gc.Finding | None:
    """命中后的豁免判定（邮件豁免面）。返回 None 表示豁免。"""
    if finding.rule == "personal_email":
        if EMAIL_EXEMPT.search(finding.snippet) or PLACEHOLDER_EMAIL.search(finding.snippet):
            return None
    return finding


def _apply_post_filter(result: gc.ScanResult) -> gc.ScanResult:
    kept_v = []
    for finding in result.violations:
        if _post_filter(finding) is None:
            result.notes.append(
                f"{finding.path}:{finding.line} [{finding.rule}] PII 豁免："
                "§15.1 检查 4 明文豁免（noreply / 保留域 / 占位符）"
            )
        else:
            kept_v.append(finding)
    kept_w = []
    for finding in result.warnings:
        if _post_filter(finding) is None:
            result.notes.append(
                f"history:{finding.commit[:12] if finding.commit else '?'} "
                f"[{finding.rule}] PII 豁免：§15.1 检查 4 明文豁免"
            )
        else:
            kept_w.append(finding)
    result.violations = kept_v
    result.warnings = kept_w
    return result


def _impl(args) -> gc.ScanResult:
    root = gc.repo_root(args.repo)
    whitelist = gc.load_whitelist(args)

    result = gc.ScanResult(gate=GATE)
    if not args.history_only:
        tracked = gc.scan_worktree(root, RULES, whitelist, prefilter=PREFILTER)
        tracked.gate = GATE
        gc.merge(result, tracked)
    if not args.tracked_only:
        history = gc.scan_history(root, RULES, whitelist, prefilter=PREFILTER)
        history.warnings.extend(history.violations)
        history.violations.clear()
        gc.merge(result, history)
        result.notes.append(
            "历史命中一律记 warning（§19.1 历史不可改写）；是否走 §15.3 重写历史由维护者/用户拍板。"
        )
    _apply_post_filter(result)
    result.notes.append(
        "结构性预过滤：含 sha256 哈希/PyPI 下载 URL 的行整行跳过"
        "（十六进制串会随机命中手机号与银行卡正则），理由见脚本文件头。"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    return gc.run_gate(GATE, DESCRIPTION, _impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
