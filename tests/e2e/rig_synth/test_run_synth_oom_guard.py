#!/usr/bin/env python3
"""`run_synth.sh` 真 OOM 判定的回归测试（纯 shell + 纯标准库，不触 GPU、不起容器）。

**为什么有这份测试**（宪法 §2 防呆 / §1.4 单一真相源）

旧实现把 OOM 判定 grep 在 `$LOG/<case>.out`（rig 宿主侧 stdout）与
`<LENRAMP_RIG>/cases/<case>/runner.log`（runner 的 bash 输出）上，而**真 OOM 报文只在
容器的 `stdout.log`** 里（容器 `/out` 绑定到宿主 `<RIGROOT>/runs/<case>/<TIER>/`）。
实测后果（task-r3-bulk §⑦-5）：32k/64k/96k/128k 真 OOM，`<case>.oom` **全部误报 0**
⇒ driver 继续递增长度 ⇒ **会测出假边界**。

本测试直接 source **生产实现**（`GRASPO_SYNTH_LIB_ONLY=1` 库模式）调用 `real_oom_case`，
**不另抄一份判定逻辑**（抄一份 = 两套口径，测试通过也证明不了生产代码对）。

两个方向都必须真：

1. :func:`test_real_oom_report_is_detected` —— 构造含**真 OOM 报文**的容器 stdout
   ⇒ **必须判 OOM**（旧实现判否 ⇒ 反向用例）；
2. :func:`test_non_oom_log_is_not_detected` —— 构造不含 OOM 报文的日志
   ⇒ **必须判否**（防止"把任何失败都当 OOM"的反向错误）；
3. :func:`test_fallback_paths_still_work` —— 旧实现能抓到的文件仍被抓到（不放松）；
4. :func:`test_missing_logs_warn_and_judge_false` —— 候选全缺失 ⇒ 判否 **且** 打 WARN
   （判定依据不可达必须暴露，不得静默）。

运行：``python3 -m pytest tests/e2e/rig_synth/test_run_synth_oom_guard.py -q``
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_RUN_SYNTH = _HERE / "run_synth.sh"

#: 真 OOM 报文的**逐字样例**（取自 ms-swift/torch 实测形态）。
_REAL_OOM_TEXT = """\
[INFO] Train:   0%|          | 0/1 [00:12<?, ?it/s]
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.20 GiB
  (GPU 0; 47.53 GiB total capacity; 46.10 GiB already allocated)
"""


def _fake_rig(tmp_path: Path, case: str, tier: str, stdout_text: str) -> Path:
    """造一个 `RIGROOT`：`runs/<case>/<TIER>/stdout.log` = 容器 stdout 的宿主落点。"""
    rigroot = tmp_path / "rigroot"
    run_dir = rigroot / "runs" / case / tier
    run_dir.mkdir(parents=True)
    (run_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
    return rigroot


def _manifest(tmp_path: Path, tier_id: str) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tiers": [{"tier_id": tier_id}]}), encoding="utf-8")
    return path


def _call_real_oom_case(
    tmp_path: Path, case: str, tier: str, rigroot: Path, manifest: Path, driver_out: str = ""
) -> subprocess.CompletedProcess[str]:
    """source 生产脚本（库模式）后调用 `real_oom_case`，返回进程结果（stdout/stderr/rc）。"""
    driver_dir = tmp_path / "synth" / "driver"  # = $SYNTH_RIG_REMOTE/driver（脚本里的 $LOG）
    driver_dir.mkdir(parents=True, exist_ok=True)
    if driver_out:
        (driver_dir / f"{case}.out").write_text(driver_out, encoding="utf-8")
    script = f"""
set -uo pipefail
export GRASPO_REPO_HOST_ROOT="{tmp_path}"
export GRASPO_SYNTH_RIG_ROOT="{tmp_path}/synth"
export GRASPO_LENRAMP_RIG_ROOT="{tmp_path}/synth"
export GRASPO_SYNTH_RIGROOT="{rigroot}"
export GRASPO_SYNTH_LIB_ONLY=1
source "{_RUN_SYNTH}" --variant v1-image-copy --real-root "{tmp_path}"
if real_oom_case "{case}" "{manifest}"; then echo "VERDICT=oom"; else echo "VERDICT=not_oom"; fi
"""
    # `source file --args...` 在本 shell 里不会把参数交给脚本；改为先 set -- 再 source。
    script = script.replace(
        f'source "{_RUN_SYNTH}" --variant v1-image-copy --real-root "{tmp_path}"',
        f'set -- --variant v1-image-copy --real-root "{tmp_path}"\nsource "{_RUN_SYNTH}"',
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)


def test_real_oom_report_is_detected(tmp_path: Path) -> None:
    """★ 反向用例：真 OOM 报文在**容器 stdout** 里 ⇒ 必须判 OOM。"""
    case, tier = "v1-image-copy-32k", "T013"
    rigroot = _fake_rig(tmp_path, case, tier, _REAL_OOM_TEXT)
    manifest = _manifest(tmp_path, tier)

    result = _call_real_oom_case(tmp_path, case, tier, rigroot, manifest)

    assert "VERDICT=oom" in result.stdout, (
        f"真 OOM 未被判出（旧实现的缺陷）；stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "命中真 OOM 报文" in result.stderr


def test_non_oom_log_is_not_detected(tmp_path: Path) -> None:
    """★ 反向用例（另一方向）：日志不含 OOM ⇒ 必须判否，不得把任何失败都当 OOM。"""
    case, tier = "v1-image-copy-4k", "T013"
    benign = (
        "[INFO] Train: 100%|##########| 2/2 [00:30<00:00, 15.2s/it]\n"
        "ValueError: Failed to retrieve the dataset\n"  # 数据问题，**不是** OOM
        "exitcode: 1\n"
    )
    rigroot = _fake_rig(tmp_path, case, tier, benign)
    manifest = _manifest(tmp_path, tier)

    result = _call_real_oom_case(tmp_path, case, tier, rigroot, manifest)

    assert "VERDICT=not_oom" in result.stdout, result.stdout
    assert "命中真 OOM 报文" not in result.stderr


def test_oom_in_torch_error_line_only(tmp_path: Path) -> None:
    """`CUDA out of memory` 的简写形态也要抓到（模式与判定链同源）。"""
    case, tier = "v1-image-copy-64k", "T013"
    rigroot = _fake_rig(
        tmp_path, case, tier, "RuntimeError: CUDA out of memory. Tried to allocate 5.00 GiB\n"
    )
    manifest = _manifest(tmp_path, tier)

    result = _call_real_oom_case(tmp_path, case, tier, rigroot, manifest)

    assert "VERDICT=oom" in result.stdout


def test_fallback_paths_still_work(tmp_path: Path) -> None:
    """不放松：旧实现能抓到的 `<case>.out` 命中，新实现仍要抓到。"""
    case, tier = "v1-image-copy-16k", "T013"
    rigroot = _fake_rig(tmp_path, case, tier, "[INFO] no error here\n")
    manifest = _manifest(tmp_path, tier)

    result = _call_real_oom_case(
        tmp_path,
        case,
        tier,
        rigroot,
        manifest,
        driver_out="runner: torch.OutOfMemoryError: CUDA out of memory\n",
    )

    assert "VERDICT=oom" in result.stdout, result.stdout


def test_missing_logs_warn_and_judge_false(tmp_path: Path) -> None:
    """候选日志全缺失 ⇒ 判否 **且** 打 WARN（判定依据不可达必须暴露，不得静默）。"""
    case, tier = "v1-image-copy-128k", "T013"
    rigroot = tmp_path / "rigroot"  # 目录都不建
    manifest = _manifest(tmp_path, tier)

    result = _call_real_oom_case(tmp_path, case, tier, rigroot, manifest)

    assert "VERDICT=not_oom" in result.stdout
    assert "WARN(oom-check)" in result.stderr


@pytest.mark.parametrize("case,tier", [("c1", "T013")])
def test_library_mode_does_not_run_synth(tmp_path: Path, case: str, tier: str) -> None:
    """库模式不得触发任何 stage / 上机流程（否则测试会真的动 rig）。"""
    rigroot = tmp_path / "rigroot"
    manifest = _manifest(tmp_path, tier)

    result = _call_real_oom_case(tmp_path, case, tier, rigroot, manifest)

    assert "阶段 B" not in result.stdout
    assert "driver.log" not in result.stdout
