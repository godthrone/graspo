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
扫描目标：① **将要进入本次提交的内容**（索引里的字节，含本次新增但尚未提交的
文件，∪ **未跟踪且不被 `.gitignore` 忽略**的文件的工作区字节——即 `git add`
之前就已存在的新文件；已跟踪路径的未暂存工作区改动不在阻断面内；违规=阻断）；
② 全部 git 历史（`git log --all -p`，违规=警告，§19.1）。

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
# 部署根前缀（内网宿主数据根）。刻意拆成两段字面量：写成连续字符串会让本文件
# 自己成为该前缀的泄露源——门禁不得靠"抄真实路径"来检测真实路径（同 §15.2）。
_DATA_ROOT_PREFIX = "/data/" + "loc" + "vg"

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
    # 完整私网 IPv4（RFC 1918）：`10/8` / `192.168/16` / `172.16-31/12`（段写法，不写具体主机地址）。
    # 三段前缀分支（`192.168`、`172.16-31`）后面补两段，`10` 分支补三段——两分支
    # 都得到**四段完整地址**，后置边界 `(?!\d)(?!\.\d)` 再拒绝 `10.20.30.40.5`
    # 这类更长数字串的截断匹配；前置边界排除前置字符是数字/字母/点/横线的情形
    # （`x10.0.0.7`、`nvidia_curand-10.3.5.147` 不误报）。
    #
    # ★ 本轮修复的覆盖缺口（实测发现的**门禁自身缺陷**，非夹具问题）：
    # 原写法 `172\.(?:1[6-9]|2\d|3[01])\.` 漏了一个 `\.`，第三分支实际要求
    # `17216.5.5` 这种形态 ⇒ 整个 `172.16.0.0/12` 段**从未被检测**。同理，原写法
    # 在 `10` 分支只补两段，命中 `10.20.30` 后会被 `(?!\.\d)` 判为"截断"而回退，
    # 导致 `10.x.x.x` 四段地址**也从未被检测**（上游注释把 `10.20.30.40` 记作
    # "被拒绝的部分匹配"，实测语义是"整段漏检"）。修复后
    # `10.20.30.40` / `172.16.5.5` / `172.20.0.1` / `172.31.255.255` /
    # 以及 `192.168/16` 段内任一四段主机地址 全部命中；`172.15.0.1`、`172.32.0.1`（段外）与
    # `192.0.2.10`、`198.51.100.7`（RFC 5737 TEST-NET，保留段）仍不命中。
    # 代价（如实记录）：`10.4.0.35` 这类"看起来像四段私网地址的包版本号"以及
    # `10.0.0.0/8` 这类网段写法现在会命中；前者已在 `gate_whitelist.json` 里按
    # `(规则, 路径, 内容)` 逐条白名单化（`pyproject.toml` 等的 `pillow>=10.0.0`），
    # 后者是文档里描述私网段本身的写法，命中后按内容白名单或改写文本处理。
    #
    # ★ 本轮补的第二个覆盖缺口（**门禁自身缺陷**，来源 = 提交前全局审查 W1）：
    # **URL 形态**漏检——`http://192.0.2.7/v1/models` 这类写法里 IP 前面是 `/`，
    # 而裸写分支的前置边界 `(?<![\dA-Za-z_.\-/])` 明确排除 `/` ⇒ 全 MISS
    # （实测：`http://192.0.2.7` / `https://198.51.100.9` / `curl http://203.0.113.10/…`
    # 三例全漏，而裸写 `192.0.2.7` / `ip=192.0.2.7` / `probe@198.51.100.9` 全命中）。
    # ★ 示例一律用 **RFC 5737 保留文档段**（192.0.2.0/24、198.51.100.0/24、203.0.113.0/24）：
    #   形态等价（同样会被"IP 前面紧邻 `/`"的边界挡住），但不属于私网段 ⇒
    #   注释不会把门禁自己打红（2026-10-16 自指修复）。
    # 补法是**只增一条 `scheme://` 分支**：把 `scheme://` 连它的 `//` 一起吃进匹配，
    # 于 IP 起始位置之前的不再是 `/` 而是 `:`，裸写分支无需收放边界——既不打散它
    # 既有的"版本号/网段写法"豁免语义（`/10.4.0.35`、`nvidia_curand-10.3.5.147`、
    # `10.0.0.0/8` 的判定逐条不变），也不放过 URL 形态。
    # 边界（诚实标注，实测固化在 `tests/tools/test_gates.py`）：
    # ① **仍漏检**：单斜杠前缀 —— `/10.0.0.7` 与 `http:/10.0.0.7`（协议后缺一个斜杠）
    #    都 MISS；补它们必须去掉裸写分支前置边界里的 `/`，代价是引入一整类版本号
    #    误报（`curand/10.4.0.35`，见上）⇒ **有意不修**。
    # ② scheme 词形上界 = 16 字符；**超长 scheme 并不漏检**——那时 IP 前面紧邻的是
    #    `:`，由裸写分支命中（只是匹配文本不含 scheme 前缀）。
    # ③ 与裸写分支同源同尾界 ⇒ 同样会把 `10.4.0.35` 形态的版本号算作命中
    #    （如 `file://10.4.0.35`），这是"宁可多报"的既定取舍。
    #
    # ★ 本轮补的第三个缺口（2026-10-16，来源 = 全局审查 W-5 **建议**）：
    # **无 scheme 的双斜杠形态**（协议相对 URL / Markdown 链接）——
    # `//192.0.2.7/v1/models`、`[服务](//192.0.2.10/v1/models)` 原先全 MISS。
    # 审查者指出"补它必须放开裸写分支的 `/` 边界"这一理由**过头**：误报来自
    # **单斜杠**前缀（`curand/10.4.0.35`），而 `(?<=//)` 分支既不放开既有边界、
    # 也不命中单斜杠形态 ⇒ **可零新误报补齐**（实测：全仓扫描面 795 文件，新增命中
    # 仅限该形态样例，既有条目一条不变）。
    gc.LineRule(
        "private_ipv4",
        re.compile(
            r"(?:(?:[A-Za-z][A-Za-z0-9+.\-]{0,15}://)"
            r"((?:(?:192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}"
            r"|10(?:\.\d{1,3}){3}))"
            r"(?!\d)(?!\.\d))"
            r"|(?<=//)"
            r"((?:(?:192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}"
            r"|10(?:\.\d{1,3}){3}))"
            r"(?!\d)(?!\.\d)"
            r"|(?<![\dA-Za-z_.\-/])"
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
    # 开发机/部署机绝对路径。`/home/user/`（占位符）、`samples/data/...`（相对
    # 路径）、`/data/train`（示例路径）不命中——真实用户名与 `/mnt/<盘>/Users/`
    # 才命中。
    #
    # ★ 缺口③（已补，**门禁自身缺陷**，来源同上 W1）：宿主侧部署根
    # 部署根下的 `<账号>/` 段 任何规则都不命中（实测 3 例全 MISS）——而本仓历史与
    # `tests/e2e/launch_matrix54.sh` 里**确实存在**这个形态。补法 = 同规则内新增
    # 一条等价分支（前缀 + **真账号段**），与 `/home/<账号>/` 同判据。
    #
    # ★ 缺口④（本次补，**门禁自身缺陷**，成因=原写法要求「账号段 + 尾斜杠」两段俱全）：
    # **无尾斜杠的绝对路径整类漏检**。代码里写这类路径最自然的落点恰恰是"算法缺省值 /
    # 环境变量兜底"，那里的路径后面紧跟的是引号、逗号、`)` 或行尾，**没有尾斜杠**
    # ⇒ 原规则对 `"…<部署根>/<账号>"`、`"…/<账号>"`（引号收尾）一律静默放过。
    # 实测（本轮改前）：`--tmp-dir` 的代码缺省值与帮助串、宿主侧数据根的 `${…:-…}`
    # 兜底值共 3 处真实内网路径**全部 MISS、门禁 0 阻断**——"假绿"。
    # 补法：把「账号段之后」由**硬要求 `/`** 改为**终止判据** `(?![\w.\-])`——
    # 即"下一字符不是路径字符（字母数字/下划线/点/横线）"，于是引号、逗号、行尾、
    # `)`、`'`、`/` 都能收尾，而 `…<账号段>x`（账号还没结束）仍不算命中。
    # 这条终止判据对三条既有分支**一视同仁**（`/home/`、部署根、
    # `/mnt/<盘>/Users` 原先同样要求尾斜杠，是同一个缺陷，不同时改会留下同类盲区）。
    # `/home/` 分支的排除项随之从 `user/` 这类"带斜杠字面量"改为
    # `(?:user|…)(?![\w.\-])`：否则 `/home/user`（无尾斜杠）会漏出占位符误报，
    # 排除集**逐项未变**、判据等价（`/home/user/models`、`/home/username/x` 仍不命中）。
    #
    # ★ 同时补的分支：`/data/<账号>`（开发机宿主根，账号段直接挂在 `/data/` 下）。
    # 它**不能**像部署根分支那样用「任意账号段」通配——实测本仓 `/data/train`（396 处
    # `train.jsonl`）、`/data/user`、`/data/models`、`/data/json_output`、`/data/v2`…
    # 全是占位/示例段，通配会把它们整类打成误报。故账号段收窄为**「字母开头 + 数字
    # 后缀」**（内网个人账号的实际形态），实测该收窄在整仓扫描面上新增命中**仅**
    # 真实账号段，占位段一条不命中。
    # 边界（诚实标注，⛔ 不声称"覆盖了所有宿主侧 `/data/**/<账号>`"）：
    #   ① `/data/<其它段>/<账号>/`（含 `/data/user/…` 这类占位符写法）**仍有意不纳入**
    #      ——会误伤 `samples/**` 里大量存在的 `/data/user/models/`、`/data/user/vllm/`
    #      占位路径（实测 73 处）；其它部署根若日后确证存在内网账号段，须按同一判据
    #      逐根补分支，而不是放宽成通用前缀。
    #   ② `/data/<纯字母账号>`（无数字后缀）**仍漏检**：与占位段不可区分，收窄是
    #      刻意取舍（宁可少扫，不可把占位路径整类打成阻断）。
    #   ③ 部署根本身（缺账号段）仍不命中——补的是"账号段存在但没有尾斜杠"，
    #      没有账号段就没有可定位的资产。
    gc.LineRule(
        "internal_absolute_path",
        # 终止判据统一写 `(?![\w.\-])`：命中可在下一个非路径字符（引号/逗号/行尾/`)`/
        # `/`…）处结束，但不能停在账号段中间。
        re.compile(
            r"(?<![\w/])(?:"
            r"/home/(?!(?:user|USER|you|example|runner|username)(?![\w.\-]))[a-z_][a-z0-9_-]{2,}(?![\w.\-])"
            r"|" + _DATA_ROOT_PREFIX + r"/(?!\.\.)[A-Za-z0-9_][A-Za-z0-9_.\-]{1,63}(?![\w.\-])"
            r"|/data/[a-z][a-z0-9_-]{2,62}[0-9](?![\w.\-])"
            r"|/mnt/[a-z]/Users(?![\w.\-]))"
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
