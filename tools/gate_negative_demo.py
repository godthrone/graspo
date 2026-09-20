#!/usr/bin/env python3
"""§15.1 门禁负向验证 —— 可复跑的实测证据脚本（W1 交付物配套）。

职责边界（§1.1）：
- 与 `tests/tools/test_gates.py` 的分工：测试文件用断言验证门禁行为；
  本脚本用**可粘贴输出**记录"注入了什么、脚本报了什么、退出码是多少"，
  作为 report.md 里的实测证据（§2.3 要求报出违规文件与命中位置）。
- 纯标准库；在临时目录里造仓库，不动真实仓库树。

用法：
    python3 tools/gate_negative_demo.py            # 打印证据表
    python3 tools/gate_negative_demo.py -o json    # 机器可读
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PY = sys.executable

CASES: list[dict] = [
    {
        "gate": "gate_secrets.py",
        "inject": "config_app.py",
        "content": (
            'API_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
            'password = "Sup3rS3cretP4ssw0rd!"\n'
            "-----BEGIN RSA PRIVATE KEY-----\n"
        ),
        "expect_rule": "aws_access_key_id",
        "what": "假 AWS 密钥 + 假硬编码口令 + 假私钥块",
    },
    {
        "gate": "gate_filetypes.py",
        "inject": ".env",
        "content": "API_KEY=REPLACE_ME\n",
        "expect_rule": "dotenv_file",
        "what": "假 .env 文件",
    },
    {
        "gate": "gate_filetypes.py",
        "inject": "notes.txt:Zone.Identifier",
        "content": "[ZoneTransfer]\nZoneId=3\n",
        "expect_rule": "zone_identifier",
        "what": "假 Windows 下载标记（备用数据流）",
    },
    {
        "gate": "gate_content.py",
        "inject": "deploy.md",
        # 合成值：RFC 1918 私网段里选定的文档式示例地址（10.20.30.40 / 10.20.30.99），
        # 不是任何真实内网资产——负向夹具不得复用真实内网标识。
        "content": "生产服务器：10.20.30.40\n其余：10.20.30.99\n",
        "expect_rule": "private_ipv4",
        "what": "合成内网 IPv4（RFC 1918 文档式示例地址，注明为合成值）",
    },
    {
        "gate": "gate_pii.py",
        "inject": "contacts.md",
        # 合成值：保留域（RFC 2606/6761，永不解析）+ 明显的合成号段。
        "content": "维护者：dev@personal.example\n手机：13900000000\n",
        "expect_rule": "personal_email",
        "what": "合成个人邮箱（保留域）+ 合成手机号",
    },
    {
        "gate": "gate_commit_identity.py",
        "inject": "(git config --local user.email)",
        # 合成值：保留域地址。判定语义是 allowlist（不在"noreply/保留域/平台机器人/
        # 本地 config 声明的项目身份"之内即阻断），**不依赖任何真实邮箱字面量**。
        "content": "dev@personal.example",
        "expect_rule": "unexpected_identity",
        "what": "仓库级生效身份改成非项目允许身份（保留域合成值）",
    },
]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_repo(base: Path) -> Path:
    repo = base / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Gate Demo")
    _git(repo, "config", "user.email", "gate-demo@users.noreply.github.com")
    (repo / "README.md").write_text("# baseline\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "chore: baseline")
    return repo


def run_case(case: dict) -> dict:
    base = Path(tempfile.mkdtemp(prefix="gate-demo-"))
    try:
        repo = make_repo(base)
        if case["gate"] == "gate_commit_identity.py":
            _git(repo, "config", "user.email", case["content"].strip())
        else:
            target = repo / case["inject"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(case["content"], encoding="utf-8")
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(repo),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Gate Demo",
                    "-c",
                    "user.email=gate-demo@users.noreply.github.com",
                    "commit",
                    "-q",
                    "-m",
                    "test: inject",
                ],
                cwd=str(repo),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        proc = subprocess.run(
            [PY, str(TOOLS / case["gate"]), "--repo", str(repo), "--tracked-only"],
            capture_output=True,
            text=True,
        )
        return {
            "gate": case["gate"],
            "what": case["what"],
            "injected_path": case["inject"],
            "injected_content": case["content"].strip().splitlines(),
            "exit_code": proc.returncode,
            "expected_nonzero": proc.returncode != 0,
            "reported_rule": case["expect_rule"] in proc.stdout,
            "reported_location": (
                case["inject"].replace(".env", ".env") in proc.stdout
                or case["inject"] in proc.stdout
            ),
            "stdout": proc.stdout,
        }
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main(argv: list[str]) -> int:
    results = [run_case(case) for case in CASES]
    ok = all(r["expected_nonzero"] and r["reported_rule"] for r in results)
    if "-o" in argv and argv[argv.index("-o") + 1] == "json":
        print(
            json.dumps(
                {"all_failed_as_expected": ok, "results": results}, ensure_ascii=False, indent=2
            )
        )
        return 0 if ok else 1
    print("W1 门禁负向验证实测（每个用例：注入 → 期望非 0 退出 + 报出违规位置）\n")
    for r in results:
        verdict = "✅ 真失败" if r["expected_nonzero"] and r["reported_rule"] else "❌ 未被拦截"
        print(f"{verdict}  {r['gate']}")
        print(f"   注入：{r['what']}  →  {r['injected_path']}")
        for line in r["stdout"].splitlines():
            if "违规" in line or line.strip().startswith("- ["):
                print(f"   报告：{line.strip()}")
        print(
            f"   退出码：{r['exit_code']}（期望非 0）  命中规则：{'是' if r['reported_rule'] else '否'}"
        )
        print()
    print("汇总：" + ("全部用例都产生真实失败 ✅" if ok else "存在未被拦截的用例 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
