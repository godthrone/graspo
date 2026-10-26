"""`scripts/eval_runner.py`（`$EVAL_DIR/run-eval.sh` 的唯一产出者）的回归用例。

覆盖本包七条判据（每条都能失败，⛔ 不是"看着像会失败"）：

1. **幂等安装**：同内容重装 ⇒ 字节不变、状态 `unchanged`、sha 记录一致；
2. **不覆盖异内容**：目标已有异内容 ⇒ 拒写（rc=2）、文件保持原样；`--force` 才覆盖且**留备份**；
3. **缺模型目录 fail-closed**：`checkpoint_path` 在盘不存在 ⇒ rc=4 且消息含「待训练产物」，
   并证明**没有触到 docker**（stub docker 的调用标记文件不存在）；
4. **路径不写死**：渲染产物与生成器源码都不得出现宿主路径/内网地址/账号名（§15.1）；
5. **落点/口径**：`output_dir` 不是本 role 独立目录 ⇒ rc=5；报告目录非空 ⇒ rc=5；报告缺失 ⇒ rc=7；
6. **接进编排链**：`scripts/gen_eval.py --dry-run` 会在生成前把入口落到 `$EVAL_DIR`，
   且生成的 `run-<tier>-eval.sh` 里两处调用（after/base）指向 `$EV/run-eval.sh`；
7. **评测数据集 fail-closed（2026-10-26 修）**：`$GRASPO_ELAM_V5_DIR/subsets-v2/<TIER>.jsonl`
   缺失或 sha256 ≠ `tier.json.subset_sha256` ⇒ rc=3 且**不触 docker**（⛔ 不许等评测容器起来才
   `DatasetError`）；齐备时 `--check-only` 打印实测 sha 并 rc=0。

运行：`.venv/bin/python -m pytest tests/scripts/test_eval_runner_install.py -q`
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import eval_runner as er  # noqa: E402
import gen_eval as ge  # noqa: E402
import launch_pipeline as lp  # noqa: E402

BASH = "/bin/bash"
PY = sys.executable


# ── 造件 ────────────────────────────────────────────────────────────────────
def _install(ev: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(_ROOT / "scripts" / "eval_runner.py"), "install",
                           "--eval-dir", str(ev), *extra], capture_output=True, text=True)


def _run_script(ev: Path, tree: Path, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([BASH, str(ev / er.RUNNER_NAME), *args], capture_output=True, text=True, env=e)


def _write_cfg(ev: Path, *, role: str = "after", out_dir: str | None = None,
               ckpt: str = "/models/MISSING/checkpoint-100", basem: str | None = "/models/BASE",
               endpoint: str = "http://127.0.0.1:18899", name: str | None = None) -> Path:
    cfg = ev / (name or f"eval-m001-{role}.yaml")
    cfg.write_text(
        "dataset_path: /work/elam-v5/subsets-v2/M001.jsonl\n"
        f"base_model_path: {basem if basem else 'null'}\n"
        f"checkpoint_path: {ckpt}\n"
        f"output_dir: {out_dir or f'/evalout/.local/reports/{role}'}\n"
        f"endpoint: {endpoint}\n"
        "served_model_name: graspo-eval\n"
        'gpus: "6"\n'
        f"role: {role if role != 'teacher' else 'other'}\n", encoding="utf-8")
    return cfg


def _stub_docker(tmp_path: Path, *, report: Path | None = None, rc: int = 0) -> tuple[Path, Path]:
    """造一个 docker 替身：把调用记进 marker，并按需写出报告（证明脚本没真起容器）。"""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "docker-called.log"
    docker = bin_dir / "docker"
    body = "#!/usr/bin/env bash\n"
    body += f'printf "%s\\n" "$*" >> "{marker}"\n'
    if report is not None:
        body += f'mkdir -p "{report.parent}" && printf \'{{"schema":"stub"}}\\n\' > "{report}"\n'
    body += f"exit {rc}\n"
    docker.write_text(body, encoding="utf-8")
    docker.chmod(0o755)
    return bin_dir, marker


@pytest.fixture()
def ev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """已落位 run-eval.sh 的 EVAL_DIR（+ 一个可跑的 TREE + 真口径 ELAM 数据根）。

    ★ 2026-10-26：`run-eval.sh` 起跑前要核 `subsets-v2/<TIER>.jsonl` 的存在与 sha，故夹具
      必须同时给出 ELAM 数据根与其 `tier.json`（缺 ⇒ 会以 rc=3 fail-closed，测试就不该绕开它）。
    """
    d = tmp_path / "EV"
    d.mkdir()
    assert _install(d, "--apply").returncode == 0
    monkeypatch.setenv("GRASPO_ELAM_V5_DIR", str(_seed_elam(tmp_path)))
    return d


def _seed_elam(tmp_path: Path, *, tier: str = "M001", sha: str | None = None) -> Path:
    """造"真口径"ELAM 数据根：`<root>/subsets-v2/<tier>.jsonl` + `<tree>/runs/<tier>/tier.json`。

    `sha` 缺省 = 子集文件的真实 sha256（阳性用例）；给了别的值即"sha 不符"的阴性用例。
    """
    root = tmp_path / "elam"
    (root / "subsets-v2").mkdir(parents=True, exist_ok=True)
    subset = root / "subsets-v2" / f"{tier}.jsonl"
    subset.write_text('{"messages": []}\n', encoding="utf-8")
    real = hashlib.sha256(subset.read_bytes()).hexdigest()
    tdir = tmp_path / "tree" / "runs" / tier
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "tier.json").write_text(json.dumps({"subset_sha256": sha or real}), encoding="utf-8")
    return root


# ── 1. 幂等 + sha 记录 ──────────────────────────────────────────────────────
def test_install_plan_mode_writes_nothing(tmp_path: Path) -> None:
    d = tmp_path / "EV"
    d.mkdir()
    p = _install(d)
    assert p.returncode == 0, p.stderr
    assert "would-install" in p.stdout
    assert not (d / er.RUNNER_NAME).exists(), "计划模式⛔不得落盘"


def test_install_is_idempotent_and_records_sha(ev: Path) -> None:
    dst = ev / er.RUNNER_NAME
    meta = ev / er.META_NAME
    first = dst.read_bytes()
    sha = er.sha256_of(first)
    doc = json.loads(meta.read_text(encoding="utf-8"))
    assert doc["sha256"] == sha, "sha 记录必须等于在盘件 sha"
    assert doc["template_version"] == er.TEMPLATE_VERSION
    assert doc["contract"].startswith("bash run-eval.sh <TREE> <EVALDIR> <TIER> <ROLE> <EVALCFG>")
    assert oct(dst.stat().st_mode & 0o777) == "0o755"

    p = _install(ev, "--apply")
    assert p.returncode == 0, p.stderr
    assert "[unchanged]" in p.stdout
    assert dst.read_bytes() == first, "同内容重装⛔不得改变字节"
    assert json.loads(meta.read_text(encoding="utf-8"))["sha256"] == sha


def test_render_is_deterministic_and_uses_gen_eval_images() -> None:
    a, b = er.render(), er.render()
    assert a == b, "渲染必须确定性（同输入同字节）"
    assert ge.IMG in a and ge.BASEIMG in b, "镜像名只从 scripts/gen_eval.py 取（单一真相源）"
    assert "graspo eval run --eval-config" in a, "评测客户端仍是既有 CLI，⛔ 不复制评测逻辑"
    assert "eval_serve_vllm.sh" in a, "vLLM 起服只指向既有脚本，⛔ 不自己起容器"


# ── 2. 不覆盖异内容 ─────────────────────────────────────────────────────────
def test_install_refuses_foreign_content(tmp_path: Path) -> None:
    d = tmp_path / "EV"
    d.mkdir()
    foreign = "#!/usr/bin/env bash\n# 另一套实现（⛔ 不得被静默顶替）\necho hi\n"
    (d / er.RUNNER_NAME).write_text(foreign, encoding="utf-8")

    p = _install(d, "--apply")
    assert p.returncode == 2, p.stdout
    assert "[blocked]" in p.stdout and "--force" in p.stdout
    assert (d / er.RUNNER_NAME).read_text(encoding="utf-8") == foreign, "拒写必须保持原样"

    p2 = _install(d, "--apply", "--force")
    assert p2.returncode == 0, p2.stderr
    assert "[forced]" in p2.stdout
    assert (d / er.RUNNER_NAME).read_text(encoding="utf-8") == er.render(), "force 后即模板产物"
    old_sha8 = er.sha256_of(foreign.encode())[:8]
    assert (d / f"{er.RUNNER_NAME}.bak.{old_sha8}").read_text(encoding="utf-8") == foreign, "覆盖须留备份（可回退）"


def test_check_detects_tampering(ev: Path) -> None:
    ok = subprocess.run([PY, str(_ROOT / "scripts" / "eval_runner.py"), "check", "--eval-dir", str(ev)],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "CHECK PASS" in ok.stdout, ok.stdout
    dst = ev / er.RUNNER_NAME
    dst.write_text(dst.read_text(encoding="utf-8") + "\n# hand-edit\n", encoding="utf-8")
    bad = subprocess.run([PY, str(_ROOT / "scripts" / "eval_runner.py"), "check", "--eval-dir", str(ev)],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "CHECK FAIL" in bad.stdout, bad.stdout


# ── 3. 缺模型目录 ⇒ fail-closed（且不触 docker）──────────────────────────────
def test_missing_model_dir_fails_closed_without_docker(tmp_path: Path, ev: Path) -> None:
    cfg = _write_cfg(ev)                                    # checkpoint_path=/models/MISSING/...
    models = tmp_path / "models"
    (models / "BASE").mkdir(parents=True)                   # base 在盘，checkpoint 不在 ⇒ 精确指向训练缺件
    bin_dir, marker = _stub_docker(tmp_path)

    p = _run_script(ev, tmp_path / "tree", [str(tmp_path / "tree"), str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(models), "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert p.returncode == er.RC["no_artifact"], p.stdout + p.stderr
    assert "待训练产物" in p.stdout + p.stderr
    assert "MISSING" in p.stdout + p.stderr, "错误信息必须点名缺的那个目录"
    assert not marker.exists(), "缺产物时⛔不得调用 docker（不许起容器/占卡）"


def test_check_only_self_passes_and_help_works(ev: Path) -> None:
    h = _run_script(ev, Path("/"), ["-h"])
    assert h.returncode == 0 and "用法" in h.stdout
    s = _run_script(ev, Path("/"), ["--check-only", "--self"])
    assert s.returncode == 0 and "SELF-CHECK PASS" in s.stdout


# ── 3.5 评测数据集（ELAM subsets-v2）fail-closed ────────────────────────────
def test_elam_subset_missing_fails_closed_without_docker(tmp_path: Path, ev: Path) -> None:
    """★ 阴性：`subsets-v2/M001.jsonl` 不在盘 ⇒ rc=3、点名路径、**不触 docker**。

    这是实测缺陷的复现形态：sidecar 的 `GRASPO_ELAM_V5_DIR` 指向一个没有 `subsets-v2/`
    的目录时，旧脚本会把评测容器拉起来，跑到加载数据才
    `DatasetError: dataset not found: /work/elam-v5/subsets-v2/M001.jsonl`（白等一次 vLLM 加载）。
    """
    models = tmp_path / "models"
    (models / "OK").mkdir(parents=True)
    cfg = _write_cfg(ev, ckpt="/models/OK", basem="/models/OK")
    (tmp_path / "elam" / "subsets-v2" / "M001.jsonl").unlink()
    bin_dir, marker = _stub_docker(tmp_path)

    p = _run_script(ev, tmp_path / "tree", [str(tmp_path / "tree"), str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(models), "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    out = p.stdout + p.stderr
    assert p.returncode == er.RC["elam"], out
    assert "评测数据集缺失" in out, out
    assert "subsets-v2/M001.jsonl" in out, "错误信息必须点名缺的那个数据集文件"
    assert not marker.exists(), "缺数据集时⛔不得调用 docker（不许起评测容器白等加载）"


def test_elam_subset_sha_mismatch_fails_closed(tmp_path: Path, ev: Path) -> None:
    """★ 阴性：子集在盘但 sha256 ≠ `tier.json.subset_sha256` ⇒ rc=3（⛔ 不拿错数据充本档）。"""
    models = tmp_path / "models"
    (models / "OK").mkdir(parents=True)
    cfg = _write_cfg(ev, ckpt="/models/OK", basem="/models/OK")
    (tmp_path / "tree" / "runs" / "M001" / "tier.json").write_text(
        json.dumps({"subset_sha256": "0" * 64}), encoding="utf-8")
    bin_dir, marker = _stub_docker(tmp_path)

    p = _run_script(ev, tmp_path / "tree", [str(tmp_path / "tree"), str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(models), "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    out = p.stdout + p.stderr
    assert p.returncode == er.RC["elam"], out
    assert "sha256 不符" in out, out
    assert not marker.exists(), "sha 不符时⛔不得调用 docker"


def test_elam_subset_ok_passes_and_prints_sha(tmp_path: Path, ev: Path) -> None:
    """★ 阳性：数据根真口径 ⇒ `--check-only --no-endpoint` rc=0，并打印实测 sha 与来源。"""
    models = tmp_path / "models"
    (models / "OK").mkdir(parents=True)
    cfg = _write_cfg(ev, ckpt="/models/OK", basem="/models/OK")

    p = _run_script(ev, tmp_path / "tree", ["--check-only", "--no-endpoint", str(tmp_path / "tree"),
                                            str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(models)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert "sha256" in p.stdout and "== 期望" in p.stdout, p.stdout
    assert "tier.json" in p.stdout, "期望 sha 的来源必须写进日志（可审计）"



def test_output_dir_mismatch_rejected(tmp_path: Path, ev: Path) -> None:
    cfg = _write_cfg(ev, role="after", out_dir="/evalout/.local/reports/base")
    p = _run_script(ev, tmp_path / "tree", [str(tmp_path / "tree"), str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(tmp_path / "models")})
    assert p.returncode == er.RC["cfg"], p.stdout
    assert "output_dir" in p.stdout


def test_cfg_outside_eval_dir_rejected(tmp_path: Path, ev: Path) -> None:
    cfg = _write_cfg(tmp_path, role="after", name="eval-m001-after.yaml")
    p = _run_script(ev, tmp_path / "tree", [str(tmp_path / "tree"), str(ev), "M001", "after", str(cfg)])
    assert p.returncode == er.RC["cfg"]
    assert "EVALDIR 之内" in p.stdout


def test_nonempty_report_dir_rejected(tmp_path: Path, ev: Path) -> None:
    models = tmp_path / "models"
    (models / "OK").mkdir(parents=True)
    cfg = _write_cfg(ev, ckpt="/models/OK", basem="/models/OK")
    (ev / ".local" / "reports" / "after").mkdir(parents=True)
    (ev / ".local" / "reports" / "after" / "eval_report.json").write_text("{}", encoding="utf-8")
    p = _run_script(ev, tmp_path / "tree", [str(tmp_path / "tree"), str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(models)})
    assert p.returncode == er.RC["cfg"], p.stdout
    assert "非空" in p.stdout


def test_success_path_tees_stdout_and_gates_on_report(tmp_path: Path, ev: Path) -> None:
    models = tmp_path / "models"
    (models / "BASE").mkdir(parents=True)
    (models / "OK").mkdir()
    cfg = _write_cfg(ev, ckpt="/models/OK")
    report = ev / ".local" / "reports" / "after" / "eval_report.json"
    bin_dir, marker = _stub_docker(tmp_path, report=report)
    env = {"GRASPO_MODEL_HOST_ROOT": str(models), "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    ok = _run_script(ev, tmp_path / "tree", ["--no-endpoint", str(tmp_path / "tree"), str(ev), "M001",
                                             "after", str(cfg)], env=env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert marker.exists(), "齐备时必须真调用 docker（stub 记录到调用）"
    assert "--network=host" in marker.read_text(encoding="utf-8")
    assert (ev / "evidence" / "eval-after-stdout.log").exists(), "stdout 必须双份留证"
    assert "graspo eval run --eval-config /evalout/eval-m001-after.yaml" in marker.read_text(encoding="utf-8")

    # 命令 rc=0 但报告没产出 ⇒ rc=7（无证据的成功不算成功）
    report.unlink()
    bin_dir2, _ = _stub_docker(tmp_path, report=None)
    bad = _run_script(ev, tmp_path / "tree", ["--no-endpoint", str(tmp_path / "tree"), str(ev), "M001",
                                              "after", str(cfg)],
                      env={"GRASPO_MODEL_HOST_ROOT": str(models), "PATH": f"{bin_dir2}:{os.environ['PATH']}"})
    assert bad.returncode == er.RC["no_report"], bad.stdout


# ── 5. §15.1：路径/账号不写死 ───────────────────────────────────────────────
def test_no_host_or_internal_literals() -> None:
    rendered = er.render()
    src = (_ROOT / "scripts" / "eval_runner.py").read_text(encoding="utf-8")
    home = str(Path(os.path.expanduser("~")))
    account = Path(home).name                       # 具体账号名**运行期取**，⛔ 不写成本文件字面量
    pats = list(er.DENY_PATTERNS) + [re.escape(home + "/"), re.escape(account)]
    for label, blob in (("渲染产物", rendered), ("生成器源码", src)):
        for pat in pats:
            assert not re.search(pat, blob), f"{label} 命中禁止项 {pat}（§15.1：宿主/内网/账号不入源码）"
    # 宿主根一律运行期取值（与 launch_pipeline 的 MODEL_HOST_ROOT/ELAM_HOST_DIR 同一规则）
    assert "${GRASPO_MODEL_HOST_ROOT:-$HOME/models}" in rendered
    assert "${GRASPO_ELAM_V5_DIR:-$HOME/elam-v5-balanced}" in rendered
    # 缺省值规则必须与 launch_pipeline 的 MODEL_HOST_ROOT/ELAM_HOST_DIR 一致（**干净 env 下实测**，
    # ⛔ 不靠 import 时的环境快照；否则本机已导出的变量会让断言失真）
    clean = {k: v for k, v in os.environ.items()
             if k not in ("GRASPO_MODEL_HOST_ROOT", "GRASPO_ELAM_V5_DIR")}
    code = ("import sys;sys.path.insert(0,'scripts');import launch_pipeline as lp;"
            "print(lp.MODEL_HOST_ROOT.rstrip('/'));print(lp.ELAM_HOST_DIR)")
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=clean, cwd=str(_ROOT))
    assert out.returncode == 0, out.stderr
    model_root, elam_root = out.stdout.strip().splitlines()
    assert model_root == str(Path(os.path.expanduser("~")) / "models")
    assert elam_root == str(Path(os.path.expanduser("~")) / "elam-v5-balanced")
    # 挂载点：$TREE/runs → /runs（docs/run-discipline.md §7 的前置由本脚本提供）
    assert '-v "$T/runs:/runs:ro"' in rendered
    assert '-v "$EV_ABS:/evalout"' in rendered


def test_host_map_matches_launch_pipeline(tmp_path: Path) -> None:
    """`/models/X` → 宿主路径的映射必须与 `launch_pipeline.resolve_base_mount_source` 同结果。"""
    (tmp_path / "models" / "Qwen3.5-9B").mkdir(parents=True)
    src, _why = lp.resolve_base_mount_source("/models/Qwen3.5-9B", host_root=str(tmp_path / "models"))
    assert src == str(tmp_path / "models" / "Qwen3.5-9B")
    ev = tmp_path / "EV"
    ev.mkdir()
    assert _install(ev, "--apply").returncode == 0
    elam = _seed_elam(tmp_path)                    # 起跑前要核 subsets-v2/<TIER>.jsonl 的存在与 sha
    cfg = _write_cfg(ev, ckpt="/models/Qwen3.5-9B", basem="/models/Qwen3.5-9B")
    p = _run_script(ev, tmp_path / "tree", ["--check-only", "--no-endpoint", str(tmp_path / "tree"),
                                            str(ev), "M001", "after", str(cfg)],
                    env={"GRASPO_MODEL_HOST_ROOT": str(tmp_path / "models"),
                         "GRASPO_ELAM_V5_DIR": str(elam)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"宿主 {src} [在盘]" in p.stdout


# ── 6. 接进编排链（评测前置）─────────────────────────────────────────────────
def test_gen_eval_installs_runner_before_generating(tmp_path: Path) -> None:
    ev = tmp_path / "EV"
    ev.mkdir()
    gen = tmp_path / "gen"
    env = dict(os.environ)
    env.update({"GRASPO_EVAL_DIR": str(ev), "GRASPO_EVAL_TMP_DIR": str(gen),
                "GRASPO_EVAL_TREE": str(_ROOT), "GRASPO_MODEL_HOST_ROOT": str(tmp_path / "models")})
    p = subprocess.run([PY, str(_ROOT / "scripts" / "gen_eval.py"), "--tier", "M001", "--gpu", "6",
                        "--port", "18899", "--dry-run", "--no-ssh"],
                       capture_output=True, text=True, env=env, cwd=str(_ROOT))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "[eval-prep]" in p.stdout and "run-eval.sh" in p.stdout
    assert (ev / er.RUNNER_NAME).is_file(), "生成评测脚本之前必须先把入口落位"
    assert (ev / er.META_NAME).is_file(), "落位必须留 sha 记录"

    tier_script = gen / "run-m001-eval.sh"
    text = tier_script.read_text(encoding="utf-8")
    calls = re.findall(r'bash "\$EV/run-eval\.sh" "\$T" "\$EV" M001 (after|base) ', text)
    assert calls == ["after", "base"], f"两处调用（after/base）必须指向 $EV/run-eval.sh：{calls}"
