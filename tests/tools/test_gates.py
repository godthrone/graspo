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
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"

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


def run_gate(name: str, repo: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATES[name]), "--repo", str(repo), *extra],
        capture_output=True,
        text=True,
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

    夹具用 RFC 1918 私网段里选定的**文档式示例地址**（10.20.30.40 / 10.20.30.99，
    合成值），不指向任何真实内网资产；规则本身来自公开标准（RFC 1918），
    测试文件不出现任何真实环境的内网标识（§15.1 检查 1/3 的豁免面精神）。
    """
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text(
        "服务器地址：10.20.30.40\n跳板机：10.20.30.99\n", encoding="utf-8"
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
        "维护者邮箱：dev@personal.example\n联系手机：13900000000\n", encoding="utf-8"
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

    夹具用**保留域合成地址**（`dev@personal.example`）。判定语义是 allowlist：
    不在"noreply + 保留域 + 平台机器人 + 本地 git config 声明的项目身份"之内的
    地址一律阻断——**代码里不再有任何真实个人/机构邮箱字面量**（这正是本次修复
    的设计缺陷：原实现把真实邮箱抄进代码当黑名单）。
    """
    repo = make_repo(tmp_path)
    _git(repo, "config", "user.email", "dev@personal.example")
    proc = run_gate("commit_identity", repo, "--tracked-only")
    assert proc.returncode == 1, f"非项目身份未被拦下：\n{proc.stdout}"
    assert "unexpected_identity" in proc.stdout
    # ★ 违规项与历史汇总必须脱敏；唯一以明文出现的"仓库级生效身份"是给操作者
    # 自查用的 note（工具本可用性），与 gate_secrets 报告不脱敏文件名同理。
    assert "de***@pe***.example" in proc.stdout, proc.stdout
    violation_line = [ln for ln in proc.stdout.splitlines() if "[unexpected_identity]" in ln][0]
    assert "dev@personal.example" not in violation_line, violation_line


def test_historical_violation_is_warning_not_block(tmp_path: Path) -> None:
    """已从当前树清除、仅存于历史的内网 IP ⇒ 记 warning，不阻断（§19.1）。

    夹具用文档式示例私网地址（10.20.30.40，合成值）。
    """
    repo = make_repo(tmp_path)
    (repo / "old.md").write_text("内网：10.20.30.99\n", encoding="utf-8")
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
    masked = gci.mask_email("dev@personal.example")
    assert "dev@personal.example" not in masked


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
    (repo / injected).write_text("联系人：dev@personal.example\n", encoding="utf-8")
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
    (repo / "cfg.md").write_text("内网：10.20.30.40\n", encoding="utf-8")
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
    assert "10.20.30.40" not in worktree_text
    assert "10.20.30.40" in index_text


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
    (repo / "notes.md").write_text("内网：10.20.30.40\n", encoding="utf-8")
    _git(repo, "stash", "push", "--", "notes.md")
    _git(repo, "stash", "pop")
    index_text = subprocess.run(
        ["git", "show", ":notes.md"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "10.20.30.40" not in index_text, "前置：索引必须是干净的"
    assert "10.20.30.40" in (repo / "notes.md").read_text(encoding="utf-8")
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


# 无 pytest 时可直接执行本文件
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
