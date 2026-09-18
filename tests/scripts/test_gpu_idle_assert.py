"""`scripts/gpu_idle_assert.py`（F-10 宿主侧空闲断言 CLI）的负向测试。

**为什么单独测这个脚本**：它是"防抢别人卡"的前置装置，入口层的 fail-closed 语义
（读不到数 / 卡被占 ⇒ rc≠0）必须在 CLI 边界上被钉住——底层函数正确但 CLI 吞掉
异常返回 0，等于防线不存在。

红线：只用**假 PATH 里的 nvidia-smi**，不调用真 nvidia-smi、不碰任何 GPU。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_idle_assert.py"

#: 假 nvidia-smi：由环境变量控制行为，测试不依赖真卡。
_FAKE_SMI = """#!/bin/bash
args="$*"
case "$args" in
  "-L") printf 'GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-fake0)\\n'
         printf 'GPU 1: NVIDIA A800-SXM4-80GB (UUID: GPU-fake1)\\n'; exit 0;;
esac
if [ -n "${FAKE_SMI_EMPTY:-}" ]; then exit 0; fi
sel="none"
case "$args" in *" -i "*) sel="${args##* -i }";; esac
IFS=',' read -ra ids <<< "$sel"
for id in "${ids[@]}"; do
  case "$id" in
    0) printf '%s\\n' "${FAKE_SMI_ROW0:-0, 4, 0}";;
    1) printf '%s\\n' "${FAKE_SMI_ROW1:-1, 0, 0}";;
    *) printf '%s\\n' "${FAKE_SMI_ROWX:-$id, 4, 0}";;
  esac
done
exit 0
"""


def _run(
    tmp_path: Path, argv: list[str], env_extra: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text(_FAKE_SMI, encoding="utf-8")
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        **env_extra,
    }
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_idle_assert_passes_when_target_card_is_idle(tmp_path: Path):
    """正例：目标卡实测空闲（4 MiB / 0%）⇒ rc=0。"""
    completed = _run(tmp_path, ["--visible", "0,1"], {})

    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


@pytest.mark.parametrize(
    "row0,expect_in_stderr",
    [
        ("0, 6029, 0", "显存占用"),  # 被他人占用（目标 GPU 服务器实测过的写法）
        ("0, 4, 90", "利用率"),  # util 高但显存没涨
    ],
)
def test_idle_assert_rejects_occupied_target_card(tmp_path: Path, row0, expect_in_stderr):
    """★负向：目标卡被占 ⇒ rc=1 且消息点明是哪种占用。"""
    completed = _run(tmp_path, ["--visible", "0,1"], {"FAKE_SMI_ROW0": row0})

    assert completed.returncode == 1, completed.stdout
    assert expect_in_stderr in completed.stderr


def test_idle_assert_is_fail_closed_when_readings_are_empty(tmp_path: Path):
    """★负向（最关键）：**读不到任何读数 ⇒ rc≠0**。

    若这里 fail-open，"读不到"就会被当成"空闲"，断言反过来变成抢卡许可证。
    """
    completed = _run(tmp_path, ["--visible", "0"], {"FAKE_SMI_EMPTY": "1"})

    assert completed.returncode != 0
    assert "fail-closed" in completed.stderr or "拒绝" in completed.stderr


def test_idle_assert_rejects_single_field_reading(tmp_path: Path):
    """★负向：解析不了的行（如只有 2 字段的 ``0, 0``）⇒ rc≠0，绝不静默通过。"""
    completed = _run(tmp_path, ["--visible", "0"], {"FAKE_SMI_ROW0": "0, 0"})

    assert completed.returncode != 0
    assert "解析" in completed.stderr


def test_idle_assert_rejects_unset_visible_devices(tmp_path: Path):
    """未锁卡（env 未设置）⇒ 拒绝（不放松既有防呆）。"""
    completed = _run(tmp_path, [], {})

    assert completed.returncode != 0
    assert "未设置" in completed.stderr
