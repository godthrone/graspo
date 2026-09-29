#!/usr/bin/env python3
"""§15.1 检查 3 — 内容门禁：内网资源引用、非公开数据集与内部表述。

职责边界（§1.1）：
- 管「内部资产地址与内容表述」：内网 IP、内网域名、开发机绝对路径、
  公司/项目内部标识、非公开数据集线索，以及禁止外传类保密标注措辞
  （词表见下方 confidential_marker 规则字面量）。
- **机构名/机构域名词表来自本地 git config**（`gate.content.org-terms` /
  `gate.content.org-domains`）——真实机构标识不得写进本文件，否则门禁自身成为
  泄露源。未配置时该两条规则不产生检测，由 §15.1 检查 3 的人工复核补位。
- 不管凭据材料（`gate_secrets.py`）、不管 PII（`gate_pii.py`）、不管路径形态
  （`gate_filetypes.py`）。
- §15.1 检查 3 明文包含「人工 + AI 辅助」：脚本能自动化的只是可模式化的部分，
  数据来源授权等判断仍需人工——脚本不假装覆盖了人工那半。

判据来源：BADGE 宪法 §15.1 检查 3 + §7.1（代理/内网地址不入镜像）。
扫描目标：① **将要进入本次提交的内容**（索引暂存内容，含本次新增但尚未提交的
文件，+ 工作区已跟踪文件当前内容；违规=阻断）；② 全部 git 历史
（`git log --all -p`，违规=警告，§19.1）。

退出码：0 通过 / 1 发现阻断级违规 / 2 用法或环境错误 / 3 内部错误。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_common as gc  # noqa: E402

GATE = "gate_content"
DESCRIPTION = "§15.1 检查 3：内容（内网 IP/域名/路径、内部标识、非公开数据线索）"

# ── 机构内部标识：配置驱动，代码内不得出现真实机构名/域名 ────────────────────
# ★ 设计纪律（§15.2 + §15.1「如果你不想它出现在 GitHub 上，它就不该出现在任何
# 已跟踪文件中」）：把**真实的公司名/内网域名**写进门禁源码当检测字面量，等于
# 门禁自己成为泄露源——与"PII 黑名单靠抄真实 PII"是同一类设计缺陷。检测这些
# 标识的能力必须保留，但词表来自**本地 git config**（不入 git）：
#     git config --local gate.content.org-terms   "example-corp"
#     git config --local gate.content.org-domains "<unit>.internal"   # 单位自有域
# 未配置 ⇒ 这两条规则不产生检测（失败方向是"少扫"），故 §15.1 检查 3 明文要求的
# 「人工 + AI 辅助」复核必须补上；本脚本在 notes 里显式声明，不静默。
ORG_TERMS_CONFIG_KEY = "gate.content.org-terms"
ORG_DOMAINS_CONFIG_KEY = "gate.content.org-domains"


def _config_list(root: Path, key: str) -> tuple[str, ...]:
    value = gc._git(["config", "--local", "--get", key], cwd=root, check=False) or ""
    return tuple(item.strip() for item in re.split(r"[,\s]+", value) if item.strip())


def build_org_rules(root: Path) -> list[gc.LineRule]:
    """由本地配置构造「机构标识」规则；未配置时返回空列表（notes 里显式声明）。"""
    rules: list[gc.LineRule] = []
    terms = _config_list(root, ORG_TERMS_CONFIG_KEY)
    if terms:
        rules.append(
            gc.LineRule(
                "internal_org_term",
                re.compile(r"(?i)\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"),
            )
        )
    domains = _config_list(root, ORG_DOMAINS_CONFIG_KEY)
    if domains:
        rules.append(
            gc.LineRule(
                "internal_org_domain",
                re.compile(
                    r"(?i)\b[a-z0-9][a-z0-9.\-]{0,60}(?:"
                    + "|".join(re.escape(d) for d in domains)
                    + r")\b"
                ),
            )
        )
    return rules


# 静态规则（与真实机构无关的形态判据）；机构词表由 build_org_rules() 运行时注入。
RULES: list[gc.LineRule] = [
    # 完整私网 IPv4（RFC 1918 三段网段：`10/8`、`192.168/16`、`172.16-31/12`）。
    # 三段前缀分支（`192.168`、`172.16-31`）后面补两段，`10` 分支补三段——两分支
    # 都得到**四段完整地址**，后置边界 `(?!\d)(?!\.\d)` 再拒绝"更长数字串被截断
    # 匹配"的形态；前置边界排除前置字符是数字/字母/点/横线的情形
    # （形如 `x10.a.b.c` 或 `nvidia_curand-10.a.b.c` 的包版本号不误报）。
    #
    # ★ 本轮修复的覆盖缺口（实测发现的**门禁自身缺陷**，非夹具问题）：
    # 原写法 `172\.(?:1[6-9]|2\d|3[01])\.` 漏了一个 `\.`，第三分支实际要求
    # `17216.5.5` 这种形态 ⇒ 整个 `172.16/12` 段**从未被检测**。同理，原写法
    # 在 `10` 分支只补两段，命中前三段后会被 `(?!\.\d)` 判为"截断"而回退，
    # 导致 `10.x.x.x` 四段地址**也从未被检测**（上游注释把文档式示例地址记作
    # "被拒绝的部分匹配"，实测语义是"整段漏检"）。修复后：RFC 1918 三段网段内的
    # 四段地址全部命中；段外地址与 RFC 5737 TEST-NET 保留段仍不命中。
    # 代价（如实记录）：形如 `10.a.b.c` 的"看起来像四段私网地址的包版本号"
    # 以及 `10/8` 这类网段写法现在会命中；前者已在 `gate_whitelist.json` 里按
    # `(规则, 路径, 内容)` 逐条白名单化（`pyproject.toml` 等的 pillow 版本约束），
    # 后者是文档里描述私网段本身的写法，命中后按内容白名单或改写文本处理。
    gc.LineRule(
        "private_ipv4",
        re.compile(
            r"(?<![\dA-Za-z_.\-/])"
            r"((?:(?:192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}"
            r"|10(?:\.\d{1,3}){3}))"
            r"(?!\d)(?!\.\d)"
        ),
    ),
    # 内网主机短名 + 服务端口（如 `NNN:22`、`NNN:8080`）——这是内网资产指认的
    # 常见写法；纯数字文件名（`T001.yaml`）与前缀不匹配的版本号不会命中。
    gc.LineRule(
        "internal_host_port",
        re.compile(
            r"(?<![\w.\-/])(?:1[0-9]{2}|2[0-4][0-9])\s*:\s*(?:22|80|443|8000|8080|8888|9000|9999)\b"
        ),
    ),
    gc.LineRule(
        "internal_domain",
        re.compile(
            r"(?i)\b[a-z0-9][a-z0-9.\-]{1,60}\.(?:intra|intranet|internal|corp|lan|localdomain)\b"
        ),
    ),
    # 机构名/机构域名规则不再在此硬编码真实标识——见文件上部 build_org_rules()。
    # 开发机/部署机绝对路径。占位符形式 `/home/<user>/` 与裸段白名单
    # （`user/`、`username/`、`runner/` 等，见下面的负向前瞻）、相对路径
    # `samples/data/...`、示例路径 `/data/<vol>` 都不命中——只有**真实用户名**的
    # 绝对路径与 `/mnt/<盘>/Users/` 才命中。
    gc.LineRule(
        "internal_absolute_path",
        re.compile(
            r"(?<![\w/])(?:/home/(?!user/|USER/|you/|example/|runner/|username/)[a-z_][a-z0-9_-]{2,}/"
            r"|/mnt/[a-z]/Users/)"
        ),
    ),
    # 内部资源引用措辞（§15.1 检查 3 明定「README 中无内部资源引用」）。
    gc.LineRule(
        "internal_resource_reference",
        re.compile(
            r"(?i)\b(?:internal (?:wiki|docs?|repo|registry|mirror|jump ?host|bastion)"
            r"|jump ?host|bastion host|company intranet)\b"
        ),
    ),
    # 机密性措辞（§15.1 检查 3）。「内部使用」一词有歧义——中文里它常作动词短语
    # （"训练内部使用 native TP/PP"），字面命中会在历史里造成 4 条假阳性，故只保留
    # 语义无歧义的强标记词。
    gc.LineRule(
        "confidential_marker",
        re.compile(
            r"(?i)(?:内部资料|仅限内部|禁止外传|请勿外传|禁止传播|内部机密"
            r"|internal use only|do not distribute|not for distribution|company confidential)"
        ),
    ),
    # 非公开数据集线索：见下方 private_dataset_reference 的规则字面量（含机密限定词
    # 的绝对数据路径、明确私有数据表述）；`/data/user/...` 是样例配置里的通用占位
    # 路径，不算。词表不进注释，避免规则词表在源码里出现两次。
    gc.LineRule(
        "private_dataset_reference",
        re.compile(
            r"(?i)(?<![\w/])/data/(?!(?:user|users|username|youruser|example)\b)"
            r"(?:user|customer|private|prod|internal)/"
            r"|\bproprietary (?:dataset|data)\b|\bnon-?public (?:dataset|data)\b"
            r"|用户原始数据|真实用户数据|生产数据导出"
        ),
    ),
]

HISTORY_IS_WARNING = True


def _impl(args) -> gc.ScanResult:
    root = gc.repo_root(args.repo)
    whitelist = gc.load_whitelist(args)
    rules = list(RULES) + build_org_rules(root)
    org_configured = len(rules) > len(RULES)

    result = gc.ScanResult(gate=GATE)
    if not args.history_only:
        tracked = gc.scan_worktree(root, rules, whitelist)
        tracked.gate = GATE
        gc.merge(result, tracked)
    if not args.tracked_only:
        history = gc.scan_history(root, rules, whitelist)
        if HISTORY_IS_WARNING:
            history.warnings.extend(history.violations)
            history.violations.clear()
        gc.merge(result, history)
        result.notes.append(
            "历史命中一律记 warning（§19.1 历史不可改写）；"
            "§15.1 检查 3 的「数据来源与授权」部分无法自动化，需人工确认。"
        )
    if not org_configured:
        result.notes.append(
            "机构名/机构域名规则**未启用**（本地未配置 "
            f"`{ORG_TERMS_CONFIG_KEY}` / `{ORG_DOMAINS_CONFIG_KEY}`）：真实机构标识"
            "不得写进门禁源码（否则门禁自己成为泄露源），故词表只能来自本地 "
            "git config。§15.1 检查 3 明文要求的「人工 + AI 辅助」复核必须覆盖"
            "这部分——本脚本不假装已自动覆盖。"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    return gc.run_gate(GATE, DESCRIPTION, _impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
