"""`tests/e2e/run_matrix68.sh` 的判据③④ 渠道（2026-10-26 修"每档都要手工重跑"）。

缺陷形态：runner 旧代码**恒**传 `--algo-checks-dir <run-root>/algo-checks`（该目录从来不存在），
而 `scripts/record_checks.py` 对**显式**目录"只认该目录、不回退仓内签署件" ⇒ 判据③ 每档都缺证据，
只能事后手工不带该参数重跑（M016 / M052 都是这么救回来的）。

修法（(a) 语义更清楚、改一处）：**缺省不传**该参数（record_checks 走既有回退：
`<run-dir>/algo-checks` 缺件 ⇒ 仓内 `samples/configs/matrix68-checks/`）；只有显式设了
`GRASPO_ALGO_CHECKS_DIR` 才传（显式即不回退，⛔ 防止偷偷用仓内旧签署件）。

本用例跑**真 runner 的 dry-run**（零副作用、不调 docker/nvidia-smi），断言它打印的
`record_checks.py` 命令里 `--algo-checks-dir` 的有无与取值 —— 断言的是行为，不是源码文本。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tests" / "e2e" / "run_matrix68.sh"
CHANNEL_MARK = "判据③④ 渠道"


def _dry_run_line(tmp_path: Path, extra_env: dict | None = None) -> str:
    env = dict(os.environ)
    env.pop("GRASPO_ALGO_CHECKS_DIR", None)          # 缺省态：⛔ 不继承本机已导出的值
    env.update(extra_env or {})
    p = subprocess.run(["bash", str(RUNNER), "--only", "M001", "--dry-run",
                        "--run-root", str(tmp_path / "runs")],
                       capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert p.returncode == 0, p.stdout + p.stderr
    hits = [ln for ln in p.stdout.splitlines() if CHANNEL_MARK in ln]
    assert len(hits) == 1, f"dry-run 必须逐档打印判据③④ 渠道：{p.stdout}"
    return hits[0]


def test_runner_omits_arg_by_default_so_record_checks_can_fall_back(tmp_path: Path) -> None:
    line = _dry_run_line(tmp_path)
    assert "record_checks.py" in line
    assert "--algo-checks-dir" not in line, (
        "缺省⛔不得传 --algo-checks-dir（它会把 record_checks 的仓内回退关掉 ⇒ 判据③ 恒待测试）："
        + line)


def test_runner_passes_arg_only_when_explicitly_configured(tmp_path: Path) -> None:
    explicit = str(tmp_path / "explicit-checks")
    line = _dry_run_line(tmp_path, {"GRASPO_ALGO_CHECKS_DIR": explicit})
    assert f"--algo-checks-dir {explicit}" in line, line
    # 显式目录未被改写成 <run-root>/algo-checks（改了就等于又回到"传一个不存在的目录"）
    assert "algo-checks" in line and explicit in line, line
