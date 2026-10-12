"""`scripts/gpu_idle_assert.py`（宿主侧**起跑前可用**断言 CLI）的负向测试。

**为什么单独测这个脚本**：它是"防抢别人卡"的前置装置，入口层的 fail-closed 语义
（读不到数 / 卡不够头寸 / util≠0 ⇒ rc≠0）必须在 CLI 边界上被钉住——底层函数正确但
CLI 吞掉异常返回 0，等于防线不存在。

★ B10 再推导（2026-10-12）：判据由"``memory.used ≤ 64 MiB`` 且 ``util ≤ 5%``"
改为"**剩余显存 ≥ 本档头寸** 且 **util == 0**"，且默认**连续采样 3 帧**。
因此假 nvidia-smi 的读数由 3 字段（index, used, util）改为 **4 字段**
（index, used, total, util）；测试用 ``--samples 2 --interval-s 0`` 免真等待。

红线：只用**假 PATH 里的 nvidia-smi**，不调用真 nvidia-smi、不碰任何 GPU。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_idle_assert.py"

#: 假 nvidia-smi：由环境变量控制行为，测试不依赖真卡。
#:
#: ★ B10：默认允许集合 = **全卡 0–7**，故夹具取卡 4/5 只是沿用历史（0/1 同样合法）。
#: 读数形状 = ``index, memory.used, memory.total, utilization.gpu``，total = 81920 MiB。
_FAKE_SMI = """#!/bin/bash
args="$*"
case "$args" in
  "-L") printf 'GPU 4: NVIDIA A800-SXM4-80GB (UUID: GPU-fake4)\\n'
         printf 'GPU 5: NVIDIA A800-SXM4-80GB (UUID: GPU-fake5)\\n'; exit 0;;
esac
if [ -n "${FAKE_SMI_EMPTY:-}" ]; then exit 0; fi
sel="none"
case "$args" in *" -i "*) sel="${args##* -i }";; esac
IFS=',' read -ra ids <<< "$sel"
for id in "${ids[@]}"; do
  case "$id" in
    4) printf '%s\\n' "${FAKE_SMI_ROW0:-4, 4, 81920, 0}";;
    5) printf '%s\\n' "${FAKE_SMI_ROW1:-5, 0, 81920, 0}";;
    *) printf '%s\\n' "${FAKE_SMI_ROWX:-$id, 4, 81920, 0}";;
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
    # 连续采样是真语义，但测试不该真等 12 s ⇒ 采 2 帧、间隔 0。
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv, "--samples", "2", "--interval-s", "0"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_idle_assert_passes_when_target_card_is_idle(tmp_path: Path):
    """正例：目标卡剩余显存 81916 MiB ≥ 头寸 48 GiB 且 util=0 ⇒ rc=0。"""
    completed = _run(tmp_path, ["--visible", "4,5"], {})

    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


@pytest.mark.parametrize(
    "row0,expect_in_stderr",
    [
        # total 81920、头寸默认 48 GiB：used 40000 ⇒ 剩余 41920 < 49152 ⇒ 拒绝
        ("4, 40000, 81920, 0", "头寸"),
        # util 高但显存没涨（别人在做推理）
        ("4, 4, 81920, 90", "利用率"),
    ],
)
def test_idle_assert_rejects_occupied_target_card(tmp_path: Path, row0, expect_in_stderr):
    """★负向：目标卡不满足双判据 ⇒ rc=1 且消息点明是哪一条不满足。"""
    completed = _run(tmp_path, ["--visible", "4,5"], {"FAKE_SMI_ROW0": row0})

    assert completed.returncode == 1, completed.stdout
    assert expect_in_stderr in completed.stderr


def test_idle_assert_recovers_when_neighbour_is_resident_but_not_computing(tmp_path: Path):
    """★B10 核心回归：邻居**常驻但不在算**（used 20 GiB、util=0）⇒ 仍可用（rc=0）。

    旧判据（used > 64 MiB 即拒）会在这里误拒；新判据只看"剩余 ≥ 头寸 ∧ util == 0"。
    """
    completed = _run(tmp_path, ["--visible", "4"], {"FAKE_SMI_ROW0": "4, 20480, 81920, 0"})

    assert completed.returncode == 0, completed.stderr


def test_idle_assert_rejects_when_tier_position_env_raises_the_bar(tmp_path: Path):
    """★env 覆盖生效：剩余 81916 MiB 在头寸 70 GiB 下够用、90 GiB 下拒绝。"""
    ok = _run(
        tmp_path,
        ["--visible", "4"],
        {"GRASPO_GPU_POSITION_MIB": str(70 * 1024)},
    )
    assert ok.returncode == 0, ok.stderr

    bad = _run(
        tmp_path,
        ["--visible", "4"],
        {"GRASPO_GPU_POSITION_MIB": str(90 * 1024)},
    )
    assert bad.returncode == 1, bad.stdout
    assert "头寸" in bad.stderr


def test_idle_assert_is_fail_closed_when_readings_are_empty(tmp_path: Path):
    """★负向（最关键）：**读不到任何读数 ⇒ rc≠0**。

    若这里 fail-open，"读不到"就会被当成"空闲"，断言反过来变成抢卡许可证。
    """
    completed = _run(tmp_path, ["--visible", "4"], {"FAKE_SMI_EMPTY": "1"})

    assert completed.returncode != 0
    assert "fail-closed" in completed.stderr or "拒绝" in completed.stderr


def test_idle_assert_rejects_single_field_reading(tmp_path: Path):
    """★负向：解析不了的行（字段数不足）⇒ rc≠0，绝不静默通过。"""
    completed = _run(tmp_path, ["--visible", "4"], {"FAKE_SMI_ROW0": "4, 0"})

    assert completed.returncode != 0
    assert "解析" in completed.stderr


def test_idle_assert_rejects_unset_visible_devices(tmp_path: Path):
    """未锁卡（env 未设置）⇒ 拒绝（不放松既有防呆）。"""
    completed = _run(tmp_path, [], {})

    assert completed.returncode != 0
    assert "未设置" in completed.stderr


def test_idle_assert_accepts_cards_that_used_to_be_outside_the_allowlist(tmp_path: Path):
    """★B10：旧的集合外卡（0–3）现在**默认允许**——只要满足双判据就通过。"""
    completed = _run(tmp_path, ["--visible", "0,3"], {})

    assert completed.returncode == 0, completed.stderr
