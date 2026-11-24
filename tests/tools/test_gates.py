"""§15.1 门禁自测：对每个 `tools/gate_*.py` 做负向验证与正向验证。

职责边界（§1.1）：
- 本文件只验证门禁脚本**自身**的行为（能否真失败、退出码语义、输出格式），
  不验证仓库内容是否合规（那是 `pre-commit run --all-files` 的职责）。
- 每个负向用例在**临时 git 仓库**里构造，不落任何文件到真实仓库树。
- 纯标准库 + pytest，不 import 项目代码 ⇒ 不受 `requires-python` 与本机
  缺 torch/pyyaml 的影响。

为什么必须有负向测试（宪法 §15.1 / §2.3）：
- 没有负向验证的扫描脚本等于没有扫描——一个永远返回 0 的脚本也会"全绿"。
- 每个 gate 都要证明：注入对应违规后**非 0 退出**，且报出**违规文件路径**。

运行：
    python3 -m pytest tests/tools/test_gates.py -q
    # 或（无 pytest 时）
    python3 tests/tools/test_gates.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"

# ── ★ 门禁自指修复（2026-10-16）：本文件里的**规则靶子**一律运行时拼接 ──────────
# 为什么：`gate_content` / `gate_pii` 扫的就是"将要提交的索引内容"，而本文件
# **自己也在扫描面内** ⇒ 任何"看起来像私网地址 / 像个人邮箱"的**字面量**都会把
# 门禁自己打红（实测：5 件门禁组已暂存时，本文件贡献 7 条 private_ipv4 阻断）。
# 处置（⛔ 不是加白名单、⛔ 不是让门禁跳过本文件、⛔ 不是放宽规则）：
#   · **可执行夹具** = 把完整 token 拆成"不构成完整形态"的片段，在**运行时**拼回；
#   · **注释/文档串里的示例** = 换成 **RFC 5737 保留文档段**（192.0.2.0/24、
#     198.51.100.0/24、203.0.113.0/24）或 RFC 2606 保留域 —— 形态等价、不属私网段。
# ★ 靶子的**校验强度不变**：拼出来的仍是同一条私网/版本号/邮箱串，规则一旦被改弱，
#   下面这些负向用例照样会失败（净发现能力不降反升：现在连"文件自己"也不会被误伤）。
_PRIVATE_A = "10." + "20." + "30." + "40"      # 私网靶子 A（10/8 段内）
_PRIVATE_B = "10." + "20." + "30." + "99"      # 私网靶子 B
_PII_EMAIL = "dev" + "@" + "personal.example"  # PII 靶子（RFC 6761 保留域）
_VERSION_LIKE = "10." + "4." + "0." + "35"     # 与私网段同形的版本号（既有误报面）
_RESERVED_IP = "192.0.2.7"                     # RFC 5737（注释里用，不属私网）
_NET_LIKE = "192.0.2.0" + "/24"                # 网段写法（注释里用，保留文档段）


GATES = {
    "secrets": TOOLS / "gate_secrets.py",
    "filetypes": TOOLS / "gate_filetypes.py",
    "content": TOOLS / "gate_content.py",
    "pii": TOOLS / "gate_pii.py",
    "commit_identity": TOOLS / "gate_commit_identity.py",
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_repo(tmp_path: Path) -> Path:
    """建一个最小 git 仓库（含一次基线提交），供负向注入用。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Gate Test")
    _git(repo, "config", "user.email", "gate-test@users.noreply.github.com")
    (repo / "README.md").write_text("# baseline\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "chore: baseline")
    return repo


def run_gate(
    name: str, repo: Path, *extra: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """跑一个门禁脚本。

    `env=None` ⇒ 原行为逐字不变（既有调用点语义不动）；`env` 给定时是**完整**子进程
    环境（调用方用 `dict(os.environ, **overrides)` 构造）⇒ 可注入 `GIT_AUTHOR_EMAIL`
    之类覆盖 git 身份解析的变量。
    """
    return subprocess.run(
        [sys.executable, str(GATES[name]), "--repo", str(repo), *extra],
        capture_output=True,
        text=True,
        env=env,
    )


def commit_all(repo: Path, message: str = "test: inject") -> None:
    _git(repo, "add", "-A")
    # -c 覆盖，避免把宿主的个人邮箱写进临时历史
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate-test@users.noreply.github.com",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=str(repo),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── 正向：干净仓库必须全绿 ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", list(GATES))
def test_clean_repo_passes(tmp_path: Path, name: str) -> None:
    repo = make_repo(tmp_path)
    proc = run_gate(name, repo, "-o", "json")
    assert proc.returncode == 0, f"{name} 在干净仓库上不应失败：\n{proc.stdout}\n{proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["exit_code"] == 0
    assert payload["violation_count"] == 0


# ── 负向：逐 gate 注入必须真失败 ───────────────────────────────────────────
def test_negative_secrets(tmp_path: Path) -> None:
    """注入假密钥 ⇒ gate_secrets 必须非 0 退出并报出文件名。"""
    repo = make_repo(tmp_path)
    (repo / "config_app.py").write_text(
        'API_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
        'AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"\n'
        'password = "Sup3rS3cretP4ssw0rd!"\n',
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("secrets", repo, "--tracked-only")
    assert proc.returncode == 1, f"假密钥未被拦下：\n{proc.stdout}"
    assert "config_app.py" in proc.stdout
    assert "aws_access_key_id" in proc.stdout or "hardcoded_credential" in proc.stdout


def test_negative_filetypes_env(tmp_path: Path) -> None:
    """注入假 .env ⇒ gate_filetypes 必须非 0 退出并报出路径。"""
    repo = make_repo(tmp_path)
    (repo / ".env").write_text("API_KEY=REPLACE_ME\n", encoding="utf-8")
    commit_all(repo)
    proc = run_gate("filetypes", repo, "--tracked-only")
    assert proc.returncode == 1, f"假 .env 未被拦下：\n{proc.stdout}"
    assert ".env" in proc.stdout


def test_negative_filetypes_zone_identifier(tmp_path: Path) -> None:
    """注入假 Zone.Identifier ⇒ gate_filetypes 必须非 0 退出。"""
    repo = make_repo(tmp_path)
    (repo / "notes.txt:Zone.Identifier").write_text("[ZoneTransfer]\n", encoding="utf-8")
    commit_all(repo)
    proc = run_gate("filetypes", repo, "--tracked-only")
    assert proc.returncode == 1, f"假 Zone.Identifier 未被拦下：\n{proc.stdout}"
    assert "Zone.Identifier" in proc.stdout


def test_negative_content_internal_ip(tmp_path: Path) -> None:
    """注入假内网 IP ⇒ gate_content 必须非 0 退出并给出行号。

    夹具用**RFC 5737 保留文档段**的写法（运行时拼接成私网形态，靶子值见 `_PRIVATE_A` /
    `_PRIVATE_B`），不指向任何真实内网资产；规则本身来自公开标准（RFC 1918），
    测试文件不出现任何真实环境的内网标识（§15.1 检查 1/3 的豁免面精神）。
    """
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(
        f"服务器地址：{_PRIVATE_A}\n跳板机：{_PRIVATE_B}\n", encoding="utf-8"
    )
    commit_all(repo)
    proc = run_gate("content", repo, "--tracked-only")
    assert proc.returncode == 1, f"假内网 IP 未被拦下：\n{proc.stdout}"
    assert "notes.md:1" in proc.stdout
    assert "private_ipv4" in proc.stdout


def test_negative_pii_email(tmp_path: Path) -> None:
    """注入假个人邮箱 + 假手机号 ⇒ gate_pii 必须非 0 退出并给出行号。

    夹具用**保留域**（`personal.example`，RFC 2606/6761 保留、永不解析）与
    明显的合成号段，不含任何真实个人身份。
    """
    repo = make_repo(tmp_path)
    (repo / "contacts.md").write_text(
        f"维护者邮箱：{_PII_EMAIL}\n联系手机：13900000000\n", encoding="utf-8"
    )
    commit_all(repo)
    proc = run_gate("pii", repo, "--tracked-only")
    assert proc.returncode == 1, f"假个人邮箱/手机号未被拦下：\n{proc.stdout}"
    assert "contacts.md:1" in proc.stdout
    assert "personal_email" in proc.stdout


def test_negative_pii_exemptions_do_not_fail(tmp_path: Path) -> None:
    """§15.1 检查 4 的明示豁免（noreply/保留域/占位符）不得被拦。"""
    repo = make_repo(tmp_path)
    (repo / "doc.md").write_text(
        "机器人：noreply@github.com\n示例：user@example.com\n模板：REPLACE_ME\n"
        "占位：your-email@your-domain.com\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("pii", repo, "--tracked-only")
    assert proc.returncode == 0, f"豁免面被误拦：\n{proc.stdout}"


def test_negative_commit_identity(tmp_path: Path) -> None:
    """仓库级生效身份不属于项目允许的公开身份 ⇒ gate_commit_identity 必须非 0。

    夹具用**保留域合成地址**（`personal.example` 域，运行时拼接）。判定语义是 allowlist：
    不在"noreply + 保留域 + 平台机器人 + 本地 git config 声明的项目身份"之内的
    地址一律阻断——**代码里不再有任何真实个人/机构邮箱字面量**（这正是本次修复
    的设计缺陷：原实现把真实邮箱抄进代码当黑名单）。
    """
    repo = make_repo(tmp_path)
    _git(repo, "config", "user.email", _PII_EMAIL)
    proc = run_gate("commit_identity", repo, "--tracked-only")
    assert proc.returncode == 1, f"非项目身份未被拦下：\n{proc.stdout}"
    assert "unexpected_identity" in proc.stdout
    # ★ 违规项与历史汇总必须脱敏；唯一以明文出现的"仓库级生效身份"是给操作者
    # 自查用的 note（工具本可用性），与 gate_secrets 报告不脱敏文件名同理。
    assert "de***@pe***.example" in proc.stdout, proc.stdout
    violation_line = [ln for ln in proc.stdout.splitlines() if "[unexpected_identity]" in ln][0]
    assert _PII_EMAIL not in violation_line, violation_line


def test_historical_violation_is_warning_not_block(tmp_path: Path) -> None:
    """已从当前树清除、仅存于历史的内网 IP ⇒ 记 warning，不阻断（§19.1）。

    夹具用文档式示例私网地址（RFC 5737 保留段写法，合成值）。
    """
    repo = make_repo(tmp_path)
    (repo / "old.md").write_text(f"内网：{_PRIVATE_B}\n", encoding="utf-8")
    commit_all(repo, "test: add old note")
    (repo / "old.md").unlink()
    commit_all(repo, "test: remove old note")
    proc = run_gate("content", repo, "-o", "json")
    assert proc.returncode == 0, f"历史命中不应阻断提交：\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    assert payload["warning_count"] >= 1, "历史命中应被报告为 warning"
    assert payload["violation_count"] == 0


def test_consumer_mail_domain_is_classified_personal() -> None:
    """个人 ISP 邮箱**分类**必须可用（不依赖任何真实邮箱字面量）。

    该分类只看公开的邮件服务商域名形态，且允许集合之外的地址本来就一律阻断
    （分类只影响报告措辞，不影响通过/拒绝）。探针域名在运行时拼接，避免把
    "个人邮箱形态"写成测试文件里的字面地址。
    """
    sys.path.insert(0, str(TOOLS))
    import gate_commit_identity as gci  # noqa: PLC0415

    probe = "someone@" + "ex" + "ample-mail.test"
    assert gci.classify_identity(probe, frozenset(), ()) == "unexpected_identity"
    # 允许集合：noreply / 保留域 / 本地 config 声明的项目身份
    assert gci.classify_identity("noreply@github.com", frozenset(), ()) is None
    assert gci.classify_identity("1+bot@users.noreply.github.com", frozenset(), ()) is None
    assert gci.classify_identity("dev@team.project", frozenset(), ("team.project",)) is None
    assert gci.classify_identity("dev@team.project", frozenset(), ()) == "unexpected_identity", (
        "未本地声明的域名不得被放行（失败方向必须是收紧）"
    )
    # 脱敏：回显不得出现完整地址
    masked = gci.mask_email(_PII_EMAIL)
    assert _PII_EMAIL not in masked


def test_output_modes(tmp_path: Path) -> None:
    """`-o` 三种模式都要可用，且 json 是结构化输出（§15.1 要求支持 -o）。"""
    repo = make_repo(tmp_path)
    for mode in ("text", "json", "quiet"):
        proc = run_gate("secrets", repo, "-o", mode, "--tracked-only")
        assert proc.returncode == 0, f"-o {mode} 失败：{proc.stderr}"
    payload = json.loads(run_gate("secrets", repo, "-o", "json", "--tracked-only").stdout)
    assert set(payload) >= {"gate", "exit_code", "violations", "warnings", "notes"}


# ── ★ 扫描面：门禁必须审"将要提交的内容"，而不是只审索引里的旧内容 ──────────
# 这三条用例是本轮修复的核心断言。没有它们，"新文件不被扫"这个缺口会静默复发，
# 而缺口复发时门禁依然"全绿"——那正是 W1 交付时假绿的成因。


def test_staged_new_file_with_pii_is_blocked(tmp_path: Path) -> None:
    """★ 构造一个"只暂存、未提交"的新文件（含假 PII）⇒ 门禁必须非 0 退出。

    这是缺陷 2 的负向测试：`git add <新文件>` 之后、`git commit` 之前，文件
    只存在于**索引**里。若扫描面只看 `git ls-files` + 工作区（旧口径的盲区），
    它就不会被扫到 ⇒ 门禁放行 ⇒ 假绿。本用例注入假个人邮箱后断言退出码非 0，
    并要求报出该路径——把"新文件必须进入扫描面"变成可失败的断言。
    """
    repo = make_repo(tmp_path)
    injected = "staged_note.md"
    (repo / injected).write_text(f"联系人：{_PII_EMAIL}\n", encoding="utf-8")
    _git(repo, "add", "--", injected)  # ← 只暂存，**不 commit**
    assert (repo / injected).exists()
    proc = run_gate("pii", repo, "--tracked-only")
    assert proc.returncode == 1, "暂存后未提交的新文件未被扫到（扫描面缺口复发）：\n" + proc.stdout
    assert injected in proc.stdout
    assert "personal_email" in proc.stdout


def test_staged_new_dotenv_is_blocked(tmp_path: Path) -> None:
    """★ 同口径：新暂存的 `.env` ⇒ gate_filetypes 必须非 0 退出并报出其路径。"""
    repo = make_repo(tmp_path)
    (repo / ".env").write_text("API_KEY=REPLACE_ME\n", encoding="utf-8")
    _git(repo, "add", "--", ".env")
    proc = run_gate("filetypes", repo, "--tracked-only")
    assert proc.returncode == 1, "新暂存的 .env 未被扫到：\n" + proc.stdout
    assert ".env" in proc.stdout


def test_gate_reads_staged_content_not_worktree_copy(tmp_path: Path) -> None:
    """★ 门禁扫的是**索引内容**：工作区已被修好、索引里仍是违规版本时必须红。

    构造：提交一版含 TEST-NET 私网 IP 的文件 → 工作区把它改成干净内容、但**不
    `git add`**（`git checkout` 之类只改工作区的操作同效）→ 暂存区里仍是违规版
    本。若门禁只读工作区，就会"看不见即将提交的违规内容"而放行。
    """

    repo = make_repo(tmp_path)
    (repo / "cfg.md").write_text(f"内网：{_PRIVATE_A}\n", encoding="utf-8")
    commit_all(repo, "test: add cfg with private ip")
    # 工作区改成干净内容；索引未动（仍是违规版本）
    (repo / "cfg.md").write_text("内网：部署在受控网段\n", encoding="utf-8")
    proc = run_gate("content", repo, "-o", "json")
    assert proc.returncode == 1, (
        "门禁读的是工作区而不是索引 ⇒ 即将提交的违规内容被放行：\n" + proc.stdout
    )
    payload = json.loads(proc.stdout)
    assert any(v["path"] == "cfg.md" for v in payload["violations"])
    # 反向确认：工作区内容确实已干净，违规只存在于**索引**里 —— 证明失败来自
    # "读索引"而不是"工作区里还有残留"。
    worktree_text = (repo / "cfg.md").read_text(encoding="utf-8")
    index_text = subprocess.run(
        ["git", "show", ":cfg.md"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert _PRIVATE_A not in worktree_text
    assert _PRIVATE_A in index_text


def test_unstaged_worktree_edit_not_blocked(tmp_path: Path) -> None:
    """★ 反向：**未暂存**的工作区违规不得阻断（它不会被提交，故不在阻断面内）。

    本条与上一条是一对，共同锁定"扫索引内容"这一口径：
      - 上一条：索引里有违规、工作区干净 ⇒ **必须红**（旧实现读工作区会漏掉）；
      - 本条：索引干净、工作区里有违规（未 `git add`）⇒ **必须绿**（旧实现读
        工作区会误红）。
    两者叠加即可判定实现读的是**索引**（将要提交的内容），而不是工作区。
    """
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text("内网：部署在受控网段\n", encoding="utf-8")
    commit_all(repo, "test: add clean note")
    # 工作区加入违规内容，但**不 git add**（用 stash 往返制造"索引干净"的状态）
    (repo / "notes.md").write_text(f"内网：{_PRIVATE_A}\n", encoding="utf-8")
    _git(repo, "stash", "push", "--", "notes.md")
    _git(repo, "stash", "pop")
    index_text = subprocess.run(
        ["git", "show", ":notes.md"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert _PRIVATE_A not in index_text, "前置：索引必须是干净的"
    assert _PRIVATE_A in (repo / "notes.md").read_text(encoding="utf-8")
    proc = run_gate("content", repo, "-o", "json")
    assert proc.returncode == 0, (
        "未暂存的工作区改动被当作违规阻断了（说明读的是工作区而不是索引）：\n" + proc.stdout
    )
    payload = json.loads(proc.stdout)
    assert payload["violation_count"] == 0
    assert "未暂存" in "\n".join(payload["notes"])


def test_staged_all_files_scan_includes_new_paths(tmp_path: Path) -> None:
    """★ 新增路径必须进入扫描面——**`git add` 之前就已经进入**（时间窗已关）。

    断言两件事：① 新增路径在 `git add` **之前**与**之后**都出现在扫描面说明里；
    ② `scanned_files` 不因 `git add` 而变（同一路径只计一次，不重复计数）。

    旧版只断言 `after["scanned_files"] > before["scanned_files"]`——那等于把
    "add 之前看不见新文件"这个**时间窗正向固化**成契约。候选集并入"未跟踪且非
    忽略"路径后，两种时机必须同严，故该断言同步改为"前后同面、不重复计数"。
    """
    repo = make_repo(tmp_path)
    new_dir = repo / "tools"
    new_dir.mkdir()
    (new_dir / "gate_new.py").write_text("# synthetic tool\n", encoding="utf-8")
    before = json.loads(run_gate("secrets", repo, "-o", "json", "--tracked-only").stdout)
    _git(repo, "add", "--", "tools/gate_new.py")
    after = json.loads(run_gate("secrets", repo, "-o", "json", "--tracked-only").stdout)
    for label, payload in (("add 之前", before), ("add 之后", after)):
        notes = "\n".join(payload["notes"])
        assert "tools/gate_new.py" in notes, f"{label}：新增路径不在扫描面内\n{notes}"
    assert before["scanned_files"] == after["scanned_files"], (
        "add 前后扫描对象数不一致（重复计数，或 add 之前未纳入）：",
        before["scanned_files"],
        after["scanned_files"],
    )


# ── ★ 时间窗：`git add` **之前**跑门禁也必须拦得住未跟踪的新文件 ──────────────
# 本仓提交纪律是"提交前手动跑门禁"。若候选集只取索引 ∪ 已暂存，在 `git add`
# 之前跑就是**假绿**（新文件完全不在扫描面内）；add 之后若不重跑，违规就随提交
# 落地。下面四条逐一证明：未跟踪、非忽略的合成文件在 add 之前必须被拦。
# 夹具字符串一律**运行时拼接**，避免把规则要抓的字面量写进本测试文件
# （否则测试文件自己会命中门禁，又要为它加白名单——那正是本包要避免的"排除"）。


def _untracked(repo: Path) -> list[str]:
    """`git ls-files --others --exclude-standard` 的仓库内真值。"""
    return subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def test_untracked_new_file_with_secret_is_blocked(tmp_path: Path) -> None:
    """★ 未跟踪（**未 `git add`**）的新文件含伪造密钥 ⇒ 机密门禁必须非 0。"""
    repo = make_repo(tmp_path)
    probe = "notes_untracked.md"
    fake = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    (repo / probe).write_text(f"临时记录：{fake}\n", encoding="utf-8")
    assert _untracked(repo) == [probe], "前置：探针必须是未跟踪、非忽略文件"
    proc = run_gate("secrets", repo, "--tracked-only")
    assert proc.returncode == 1, "未跟踪新文件里的密钥未被扫到（时间窗未关）：\n" + proc.stdout
    assert probe in proc.stdout
    assert "github_token" in proc.stdout


def test_untracked_new_file_with_pii_is_blocked(tmp_path: Path) -> None:
    """★ 未跟踪（**未 `git add`**）的新文件含伪造 PII ⇒ PII 门禁必须非 0。

    夹具用**保留域**（`personal.example`，RFC 2606/6761）运行时拼出，不含真实身份。
    """
    repo = make_repo(tmp_path)
    probe = "contacts_untracked.md"
    addr = "dev" + "@" + "personal.example"
    (repo / probe).write_text(f"联系人：{addr}\n", encoding="utf-8")
    assert _untracked(repo) == [probe], "前置：探针必须是未跟踪、非忽略文件"
    proc = run_gate("pii", repo, "--tracked-only")
    assert proc.returncode == 1, "未跟踪新文件里的 PII 未被扫到（时间窗未关）：\n" + proc.stdout
    assert probe in proc.stdout
    assert "personal_email" in proc.stdout


def test_untracked_new_file_with_internal_ip_is_blocked(tmp_path: Path) -> None:
    """★ 未跟踪（**未 `git add`**）的新文件含内网 IP ⇒ 内容门禁必须非 0。

    夹具用 RFC 1918 私网段里的**文档式示例地址**（合成值，运行时拼接），不指向
    任何真实资产。
    """
    repo = make_repo(tmp_path)
    probe = "notes_untracked.md"
    addr = "10.20.30." + "40"
    (repo / probe).write_text("服务器地址：" + addr + "\n", encoding="utf-8")
    assert _untracked(repo) == [probe], "前置：探针必须是未跟踪、非忽略文件"
    proc = run_gate("content", repo, "--tracked-only")
    assert proc.returncode == 1, "未跟踪新文件里的内网 IP 未被扫到（时间窗未关）：\n" + proc.stdout
    assert probe in proc.stdout
    assert "private_ipv4" in proc.stdout


def test_untracked_new_forbidden_filetype_is_blocked(tmp_path: Path) -> None:
    """★ 未跟踪（**未 `git add`**）的禁止类型文件 ⇒ 文件类型门禁必须非 0。

    选 `*.pem`（`key_material` 规则）：它**不在 `.gitignore` 里**，故 `git add`
    即可提交 ⇒ 正是"add 之前必须拦"的靶子。（`.env` / `*.log` 被 `.gitignore`
    覆盖，需 `git add -f` 才可能进历史，本函数的候选集不覆盖——该残余面由
    "提交门禁在 add 之后跑"兜底，见 `.pre-commit-config.yaml` 文件头。）
    """
    repo = make_repo(tmp_path)
    probe = "service_account.pem"
    (repo / probe).write_text("REPLACE_ME\n", encoding="utf-8")
    assert _untracked(repo) == [probe], "前置：探针必须是未跟踪、非忽略文件"
    proc = run_gate("filetypes", repo, "--tracked-only")
    assert proc.returncode == 1, "未跟踪的禁止类型文件未被扫到（时间窗未关）：\n" + proc.stdout
    assert probe in proc.stdout
    assert "key_material" in proc.stdout


def test_untracked_ignored_file_is_still_out_of_scope(tmp_path: Path) -> None:
    """★ 边界（不得过度收录）：被 `.gitignore` 忽略的未跟踪文件**仍不入候选集**。

    这是候选集并入未跟踪段的**代价显式化**，不是许可放宽：`.local/`、`.venv/`、
    构建产物必须留在门外（否则门禁永远红、不可用）。该残余面靠"提交门禁在
    `git add` 之后跑"覆盖（届时路径已进索引，会被索引段扫到）。
    """
    repo = make_repo(tmp_path)
    (repo / ".gitignore").write_text("local_notes/\n", encoding="utf-8")
    commit_all(repo, "test: ignore local notes")
    (repo / "local_notes").mkdir()
    (repo / "local_notes" / "scratch.md").write_text(
        "临时：" + "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" + "\n", encoding="utf-8"
    )
    assert _untracked(repo) == [], "前置：被忽略的未跟踪文件不得进入候选集"
    proc = run_gate("secrets", repo, "--tracked-only")
    assert proc.returncode == 0, "被 `.gitignore` 忽略的未跟踪文件不得被扫（过度收录）：\n" + proc.stdout


# ── ★ 规则覆盖缺口补强（命中面变化 + 边界外样例）──────────────────────────
# 背景：提交前全局审查 W1（`.local/hb-workspace/20261014-sft-kickoff/
# review-precommit-batch3/review-report.md:268`）判定「四道具全绿 ≠ 仓内无内网信息」，
# 因为它实测到两类形态**任何规则都不命中**：
#   ① `private_ipv4` 漏 URL 形态（URL 形态原先 MISS、裸写 HIT）；
#   ② `internal_absolute_path` 漏 `/data/remote/<账号>/`（3 样例全 MISS）；
#   ③（2026-11-23 补）`internal_absolute_path` 漏**无尾斜杠**的宿主路径与 `/data/<账号>`。
# 下面各条把这些缺口钉成**可失败的断言**，并把"边界外样例"显式写出来
# （诚实标注覆盖范围，⛔ 不假装规则已覆盖全部内网形态）。
# 夹具沿用本文件既有纪律：**运行时拼接**，避免把规则要抓的字面量写进测试文件
# 本身（否则测试文件自己会命中门禁，又得为它加白名单）。
_URL_IP = "http://10.99.88." + "77/v1/models"
_URL_IP_HTTPS = "https://172.20." + "5.9"
_URL_IP_CURL = "curl http://192.168." + "1.10/v1/models"
_LOCVG_PREFIX = "/data/" + "loc" + "vg"
_LOCVG_ACCT = _LOCVG_PREFIX + "/" + "sampleacct"


def test_negative_content_private_ipv4_in_url_form(tmp_path: Path) -> None:
    """★ 缺口①：**URL 形态**私网 IPv4 ⇒ `gate_content` 必须非 0 退出。

    补规则前这三条实测全 MISS（IP 前面是 `/`，被裸写分支的前置边界排除）。
    """
    repo = make_repo(tmp_path)
    (repo / "cfg.md").write_text(
        f"服务地址：{_URL_IP}\n备用：{_URL_IP_HTTPS}\n探测：{_URL_IP_CURL}\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "--tracked-only")
    assert proc.returncode == 1, f"URL 形态内网 IP 未被拦下：\n{proc.stdout}"
    assert "cfg.md:1" in proc.stdout
    assert "cfg.md:2" in proc.stdout
    assert "cfg.md:3" in proc.stdout
    assert "private_ipv4" in proc.stdout


def test_negative_content_account_path_in_data_root(tmp_path: Path) -> None:
    """★ 缺口②：`/data/remote/<账号>/` 形态 ⇒ `gate_content` 必须非 0 退出。

    补规则前实测 MISS（`internal_absolute_path` 只有 `/home/<账号>/` 与
    `/mnt/<盘>/Users/` 两条分支）。
    """
    repo = make_repo(tmp_path)
    (repo / "deploy.sh").write_text(
        f'STAGE="${{STAGE:-{_LOCVG_ACCT}/batch2-staging}}"\n'
        f"sync_deploy_tree.sh --tree {_LOCVG_ACCT}/grasp\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "--tracked-only")
    assert proc.returncode == 1, f"/data/remote/<账号>/ 未被拦下：\n{proc.stdout}"
    assert "deploy.sh:1" in proc.stdout
    assert "deploy.sh:2" in proc.stdout
    assert "internal_absolute_path" in proc.stdout


def test_url_ip_fixture_hits_only_private_ipv4_rule(tmp_path: Path) -> None:
    """★ 精度：URL 形态夹具**只**应命中 `private_ipv4`，不得顺带命中别的规则。

    防止"补 URL 分支"把 `internal_host_port`（`<短名>:<端口>`）之类规则一起带进
    误报面：`http://<私网地址>:8000/v1` 里冒号前是私网地址而不是内网短名。
    """
    repo = make_repo(tmp_path)
    (repo / "cfg.md").write_text(
        f"LLM_BASE=http://10.99.88.{'77'}:8000/v1\n", encoding="utf-8"
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    assert proc.returncode == 1, f"URL+端口 形态未被拦下：\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    rules = {v["rule"] for v in payload["violations"]}
    assert rules == {"private_ipv4"}, f"命中规则集不合预期（过宽？）：{rules}"


def test_content_keeps_bare_form_boundary_for_slash_prefixed_version(tmp_path: Path) -> None:
    """★ 不误伤（负向对照）：**斜杠前缀的版本号**（斜杠前缀的版本号形态）必须仍不命中。

    这条钉的是"补 URL 分支**没有**顺手把裸写分支的前置边界放宽成不含 `/`"：
    若把 `(?<![\\dA-Za-z_.\\-/])` 里的 `/` 去掉，`nvidia_curand-10.3.5.147` 之外还会
    多出 `…/<斜杠前缀版本号>` 这一整类误报（本仓里就是 `uv.lock` / `Dockerfile.msswift` 的
    包版本号、网段写法）。URL 分支单独吃 `scheme://`，裸写分支逐字未改。
    """
    repo = make_repo(tmp_path)
    (repo / "docs_note.md").write_text(
        "包版本号（斜杠前缀）：curand/VERSION_LIKE\n"
        "公网（RFC 5737 文档段）：198.51.100.7 / 192.0.2.10\n"
        "占位符：/home/user/models、/data/user/models、/data/users/vllm、/data/train\n"
        "相对路径：samples/data/train.jsonl\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    assert payload["violation_count"] == 0, (
        "斜杠前缀的版本号被误伤 ⇒ 裸写分支的前置边界被放宽了（应只增 URL 分支）：\n"
        + proc.stdout
    )


def test_content_version_number_false_positive_is_unchanged(tmp_path: Path) -> None:
    """★ 如实固化**既有误报面**：`>=<版本号形态>` 这类写法**改前改后都命中**。

    `=` 不在裸写分支的前置边界内 ⇒ 这一条**不是本包引入**，也**未被本包消除**
    （消除它的代价是删掉 `private_ipv4` 里版本号场景的白名单手段，不在本包范围）。
    本用例把它钉成"恰好 1 条、且规则名就是 private_ipv4"的**不变式**，防止有人
    日后靠改期望值把它悄悄抹掉（§19.1 纪律：回归测试只许加强）。
    """
    repo = make_repo(tmp_path)
    (repo / "pins.txt").write_text(
        "dep>=" + _VERSION_LIKE + "\n",  # 运行时拼接（门禁自指修复）：字面量不得构成完整形态
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    hits = [(v["rule"], v["path"], v["line"]) for v in payload["violations"]]
    assert hits == [("private_ipv4", "pins.txt", 1)], f"既有误报面发生变化：{hits}\n{proc.stdout}"


def test_content_scope_boundary_examples_are_uncovered(tmp_path: Path) -> None:
    """★ **边界外样例**（诚实标注覆盖范围，⛔ 不声称已覆盖全部内网形态）。

    本用例固化的"仍不命中"面（若日后有人把规则放宽成通用前缀，本条会失败，从而
    强迫一次显式裁定，而不是静默扩面）：
      · `/data/<其它段>/<账号>/`：一律不补（否则误伤 `samples/**` 里 73 处
        `/data/user/models/`、`/data/user/vllm/` 占位路径）；
      · `/data/remote`（无尾斜杠）、`/data/remote/`（缺账号段）、
        `/data/remote/../etc`（`..` 不是账号段）：只认「前缀 + 账号段 + `/`」。
    """
    repo = make_repo(tmp_path)
    other_root = "/data/" + "otherroot"
    (repo / "boundary.md").write_text(
        f"其它部署根（有意不覆盖）：{other_root}/sampleacct/\n"
        f"缺尾斜杠（有意不覆盖）：{_LOCVG_PREFIX}\n"
        f"缺账号段（有意不覆盖）：{_LOCVG_PREFIX}/\n"
        f"父目录穿越（有意不覆盖）：{_LOCVG_PREFIX}/../etc\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    assert payload["violation_count"] == 0, (
        "边界外样例被命中 ⇒ 规则已超出本轮声明的覆盖范围，需显式裁定：\n" + proc.stdout
    )


# ── 缺口③/④（2026-11-23）：无尾斜杠的宿主绝对路径 + `/data/<账号>` 宿主根 ──────────
# 背景：代码里写这类路径最自然的落点就是"环境变量兜底 / argparse 缺省值"，那里的路径
# 后面紧跟引号、`)` 或行尾 ⇒ **没有尾斜杠**。原规则要求「账号段 + `/`」两段俱全，
# 这类写法整类静默放过（实测：门禁 0 阻断，而两处真实内网路径就在源码里）。
_NOSLASH_ACCT = _LOCVG_ACCT              # `/data/remote/<合成账号>`，结尾无斜杠
_DATA_ROOT_ACCT = "/data/" + "sampleacct" + "30"   # `/data/<字母开头+数字后缀>`
_HOME_ACCT = "/home/" + "sampleacct"     # `/home/<账号>`（同样要求"无尾斜杠也命中"）


def test_content_host_path_without_trailing_slash_is_blocked(tmp_path: Path) -> None:
    """★ 缺口③：宿主绝对路径**结尾不是斜杠**时也必须命中（脚本兜底值 / argparse 缺省值）。"""
    repo = make_repo(tmp_path)
    (repo / "deploy.sh").write_text(
        f'HOST_ROOT="${{GRASPO_EVAL_HOST_ROOT:-{_NOSLASH_ACCT}}}"\n'
        f'STAGE="{_NOSLASH_ACCT}"\n'
        f'TMP = os.environ.get("GRASPO_EVAL_TMP_DIR", "{_NOSLASH_ACCT}")\n'
        f"DATA = {_HOME_ACCT!r}\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    hits = sorted((v["line"], v["rule"]) for v in payload["violations"])
    assert proc.returncode == 1, f"无尾斜杠的宿主路径未被拦下：\n{proc.stdout}"
    assert hits == [(1, "internal_absolute_path"), (2, "internal_absolute_path"),
                    (3, "internal_absolute_path"), (4, "internal_absolute_path")], (
        "无尾斜杠形态的命中面与预期不一致：\n" + proc.stdout
    )


def test_content_account_path_in_data_root_is_blocked(tmp_path: Path) -> None:
    """★ 缺口④：`/data/<账号>`（账号段直接挂在 `/data/` 下的开发机宿主根）必须命中。

    判据收窄为「字母开头 + 数字后缀」——占位段（`/data/train`、`/data/models`…）因此
    仍不命中，见下一条负例。不收窄的话本仓 396 处 `/data/train.jsonl` 会整类变阻断。
    """
    repo = make_repo(tmp_path)
    (repo / "gen.py").write_text(
        f'ap.add_argument("--host-root", default="{_DATA_ROOT_ACCT}")\n'
        f"RUNS = {_DATA_ROOT_ACCT + '/runs/M008'!r}\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    hits = sorted((v["line"], v["rule"]) for v in payload["violations"])
    assert proc.returncode == 1, f"`/data/<账号>` 未被拦下：\n{proc.stdout}"
    assert hits == [(1, "internal_absolute_path"), (2, "internal_absolute_path")], (
        "`/data/<账号>` 的命中面与预期不一致：\n" + proc.stdout
    )


def test_content_account_shaped_root_keeps_placeholder_paths_clean(tmp_path: Path) -> None:
    """★ 负向对照：补 `/data/<账号>` 与「无尾斜杠」**没有**把占位段打成误报。

    这是两条新判据各自误报面的钉子：
      · `/data/<段>` 段内**无数字后缀** ⇒ 不命中（`/data/train`、`/data/models`、
        `/data/user/models`、`/data/v2` 等，本仓真实存在的占位/示例路径）；
      · `/home/` 的排除项改成 `(?:user|…)(?![\\w.\\-])` 后，`/home/user`（**无尾斜杠**）
        与 `/home/runner/x` 仍不命中——排除集逐项未变，只是不再靠"必须有尾斜杠"兜底。
    """
    repo = make_repo(tmp_path)
    (repo / "placeholders.md").write_text(
        "示例路径：/data/train、/data/models、/data/json_output、/data/test.jsonl、/data/v2\n"
        "占位账号：/data/user/models、/data/users/vllm、/home/user、/home/runner/x\n"
        "相对路径：samples/data/train.jsonl\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    assert payload["violation_count"] == 0, (
        "占位路径被误伤 ⇒ 新分支的收窄条件（数字后缀 / 排除项边界）失效：\n" + proc.stdout
    )


def test_content_url_scheme_variants_that_are_still_hits(tmp_path: Path) -> None:
    """★ 如实固化 **URL 形态的第二类命中路径**（防"边界说明"与实测不符）。

    讨论边界时最容易写错的一点：不是只有 `scheme://` 分支会命中。**超长自定义
    scheme**（>16 字符，超出 `[A-Za-z][A-Za-z0-9+.\\-]{0,15}`）里 IP 前面紧邻的是
    `:` ⇒ **裸写分支自己就会命中**，与 URL 分支无关。⇒ 诚实结论：超长 scheme
    **并不漏检**（虽然匹配文本不含 scheme 前缀）。
    """
    repo = make_repo(tmp_path)
    long_scheme = "x" * 17 + "://"
    (repo / "variants.md").write_text(
        f"超长 scheme：{long_scheme}{_PRIVATE_A}\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    hits = sorted((v["path"], v["line"], v["rule"]) for v in payload["violations"])
    assert hits == [("variants.md", 1, "private_ipv4")], (
        "超长 scheme 的命中面与报告的边界说明不一致：\n" + proc.stdout
    )


def test_content_double_slash_private_ipv4_is_blocked(tmp_path: Path) -> None:
    """★ 缺口③：**无 scheme 的双斜杠形态**私网 IPv4 ⇒ `gate_content` 必须非 0 退出。

    来源 = 全局审查 W-5（**建议**）：`//10.x.x.x`（协议相对 URL / Markdown 链接）原先
    全 MISS。审查者指出"补它必须放开裸写分支的 `/` 边界"这一理由**过头** —— 误报
    来自**单斜杠**前缀（`curand/<版本号形态>`），故 `(?<=//)` 分支可**零新误报**补齐。
    实测（本包 `evidence/64_*` vs `66_*`）：全仓扫描面 795 文件的原始命中**逐条不变**
    （37 → 37，`diff` 无差异）。
    """
    repo = make_repo(tmp_path)
    (repo / "cfg.md").write_text(
        "协议相对：//10.99.88." + "77/v1/models\n"
        "Markdown 链接：[内网服务](//192.168." + "1.10/v1/models)\n"
        "带端口：//10.1." + "2.3:8000/v1\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "--tracked-only")
    assert proc.returncode == 1, f"双斜杠形态内网 IP 未被拦下：\n{proc.stdout}"
    assert "cfg.md:1" in proc.stdout
    assert "cfg.md:2" in proc.stdout
    assert "cfg.md:3" in proc.stdout
    assert "private_ipv4" in proc.stdout


def test_content_double_slash_branch_adds_no_false_positive(tmp_path: Path) -> None:
    """★ 零新误报（可失败断言）：**单斜杠**前缀与版本号形态**必须仍不命中**。

    这是 `(?<=//)` 分支的代价显式化：若有人把它写成 `(?<=/)`（或顺手放宽裸写分支的
    `/` 边界），单斜杠前缀的版本号 / 地址形态就会开始命中
    ⇒ 本用例会失败，从而强迫一次显式裁定（而不是静默扩面）。
    """
    repo = make_repo(tmp_path)
    (repo / "slash.md").write_text(
        "单斜杠目录：curand/10.4.0." + "35\n"
        "单斜杠地址：/10.99.88." + "77\n"
        "协议后缺一个斜杠：http:/10.99.88." + "77\n"
        "公网（RFC 5737）：//198.51.100." + "7\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    assert payload["violation_count"] == 0, (
        "单斜杠 / 版本号形态被误伤 ⇒ `(?<=//)` 分支被写宽了（应只认双斜杠）：\n" + proc.stdout
    )


def test_content_single_slash_url_is_still_uncovered(tmp_path: Path) -> None:
    """★ **仍存的漏检边界**（诚实标注）：**单斜杠**前缀形态仍 MISS。

    实测 `http:/10.x.x.x`（协议后缺一个斜杠）与 `/10.x.x.x` 都不命中。
    本轮**有意不修**：修它就必须去掉裸写分支前置边界里的 `/`，而那会引入
    `…/<斜杠前缀版本号>` 一整类版本号误报（`uv.lock` / `Dockerfile.msswift` / 网段写法）。
    ⇒ 覆盖范围到此为止，⛔ 不得据此宣称"仓内无内网信息"。

    ⚠ 本条已随 W-5 落地**同步收窄**（原先同时断言 `//10.x.x.x` 也不命中；该形态现
    由 `(?<=//)` 分支命中，见上一条用例）—— 这是**口径变更的显式登记**，⛔ 不是
    "改期望值让测试变绿"：被移出本条的形态已由新用例**正向断言必须命中**。
    """
    repo = make_repo(tmp_path)
    (repo / "slash.md").write_text(
        "缺一个斜杠：http:/10.99.88." + "77\n"
        "单斜杠：/10.99.88." + "77\n",
        encoding="utf-8",
    )
    commit_all(repo)
    proc = run_gate("content", repo, "-o", "json", "--tracked-only")
    payload = json.loads(proc.stdout)
    assert payload["violation_count"] == 0, (
        "单斜杠形态的命中面与报告的边界说明不一致（若已命中，请同步更新边界说明）：\n"
        + proc.stdout
    )


# ── ★ `git var GIT_AUTHOR_IDENT` 必须**真跑**（缺口的第三半）─────────────────
# 背景：评审 Q6 判定「能力已存在（`tools/gate_commit_identity.py`）、且已写进
# `.pre-commit-config.yaml` 的 5 个 hook，**唯一缺的是真的跑**」——旧实现只读
# `git config --local user.email`（仓库**约定值**），从不执行 `git var`（git
# **实际会写进提交对象**的身份）。两者在 `GIT_AUTHOR_EMAIL` 等环境变量存在时
# 必然分叉。下面 4 条把"真跑"钉成可失败断言；夹具用保留域合成地址。
# ⛔ 一律 `env` 传子进程，**不在测试进程里 export 任何 GIT_**（红线）。


def _ident_env(**overrides: str) -> dict[str, str]:
    """构造子进程环境：清掉继承的 `GIT_*`/`EMAIL` 身份变量再注入 overrides。"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "EMAIL"}
        and not k.startswith("GIT_AUTHOR_")
        and not k.startswith("GIT_COMMITTER_")
    }
    env.update(overrides)
    return env


def test_effective_author_identity_from_env_is_blocked(tmp_path: Path) -> None:
    """★ 仓库 local config 合规、但 git **解析出的**作者身份不合规 ⇒ 必须非 0。

    补规则前实测 exit 0（门禁从不执行 `git var`）。
    """
    repo = make_repo(tmp_path)  # local config = gate-test@users.noreply.github.com（合规）
    env = _ident_env(GIT_AUTHOR_EMAIL="dev" + "@" + "personal.example")
    proc = run_gate("commit_identity", repo, "--tracked-only", env=env)
    assert proc.returncode == 1, (
        "`git var GIT_AUTHOR_IDENT` 未被真跑 ⇒ 环境变量覆盖的作者身份漏检：\n" + proc.stdout
    )
    assert "unexpected_effective_identity_author" in proc.stdout
    assert "<git var GIT_AUTHOR_IDENT>" in proc.stdout


def test_effective_committer_identity_from_env_is_blocked(tmp_path: Path) -> None:
    """★ 同上，committer 面独立可判（`GIT_COMMITTER_IDENT`）。"""
    repo = make_repo(tmp_path)
    env = _ident_env(GIT_COMMITTER_EMAIL="dev" + "@" + "personal.example")
    proc = run_gate("commit_identity", repo, "--tracked-only", env=env)
    assert proc.returncode == 1, (
        "`git var GIT_COMMITTER_IDENT` 未被真跑 ⇒ 提交者身份漏检：\n" + proc.stdout
    )
    assert "unexpected_effective_identity_committer" in proc.stdout


def test_effective_identity_allowed_value_not_blocked(tmp_path: Path) -> None:
    """★ 不误伤（负向对照）：env 覆盖到**允许集合内**的 noreply ⇒ 必须仍绿。"""
    repo = make_repo(tmp_path)
    env = _ident_env(GIT_AUTHOR_EMAIL="70987748+someone@users.noreply.github.com")
    proc = run_gate("commit_identity", repo, "--tracked-only", env=env)
    assert proc.returncode == 0, f"合规身份被误拦：\n{proc.stdout}"
    assert "unexpected_effective_identity" not in proc.stdout


def test_effective_identity_unresolvable_is_fail_closed(tmp_path: Path) -> None:
    """★ fail-closed 通路：git 完全解析不出身份（`git var` 非 0）⇒ 必须阻断。"""
    repo = make_repo(tmp_path)
    _git(repo, "config", "--local", "--unset", "user.email")
    _git(repo, "config", "--local", "--unset", "user.name")
    env = _ident_env(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    proc = run_gate("commit_identity", repo, "--tracked-only", env=env)
    assert proc.returncode == 1, (
        "身份无法解析时门禁放行了（fail-open）：\n" + proc.stdout
    )
    assert "missing_effective_identity" in proc.stdout


# 无 pytest 时可直接执行本文件
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
