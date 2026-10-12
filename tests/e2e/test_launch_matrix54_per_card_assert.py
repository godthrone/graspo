"""Launcher `tests/e2e/launch_matrix54.sh` — per-card pre-flight idle assertion.

Regression for the 228 defect (2026-10-12): the launcher used to hand-roll a
**single-card** parse of ``nvidia-smi -i <comma list>``.  For a multi-card tier
the command returns one row per card, so the parse produced a two-line value and
bash aborted with ``[: 81153\\n81153: integer expression expected`` -> FATAL.
The workaround (passing one card to the pre-flight while the manifest keeps N)
silently downgraded the guard: **only one card of a multi-card tier was verified**.

Fix: delegate to the already-fixed, already-unit-tested
``scripts/gpu_idle_assert.py``, which checks **every** target card independently
(remaining memory >= tier position **and** ``utilization.gpu == 0`` on **3
consecutive frames**) and fails closed when any single card does not qualify.

Red line: fake ``nvidia-smi`` only; no real GPU, no real docker, no real container.
"""

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LAUNCHER = _ROOT / "tests" / "e2e" / "launch_matrix54.sh"
_IDLE_ASSERT = _ROOT / "scripts" / "gpu_idle_assert.py"

#: Fake nvidia-smi: behaviour is driven by env vars, no real GPU is touched.
#: Reading shape (B10) = ``index, memory.used, memory.total, utilization.gpu``.
_FAKE_SMI = """#!/bin/bash
args="$*"
case "$args" in
  "-L") printf 'GPU 3: NVIDIA A800-SXM4-80GB (UUID: GPU-fake3)\\n'
         printf 'GPU 4: NVIDIA A800-SXM4-80GB (UUID: GPU-fake4)\\n'; exit 0;;
esac
sel="none"
case "$args" in *" -i "*) sel="${args##* -i }";; esac
IFS=',' read -ra ids <<< "$sel"
for id in "${ids[@]}"; do
  case "$id" in
    3) printf '%s\\n' "${FAKE_SMI_ROW3:-3, 1, 81920, 0}";;
    4) printf '%s\\n' "${FAKE_SMI_ROW4:-4, 1, 81920, 0}";;
    *) printf '%s\\n' "$id, 1, 81920, 0";;
  esac
done
exit 0
"""


def _assert_idle(tmp_path: Path, visible: str, env_extra: dict[str, str] | None = None):
    """Run the real per-card assertion with a fake nvidia-smi."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "nvidia-smi"
    fake.write_text(_FAKE_SMI, encoding="utf-8")
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        **(env_extra or {}),
    }
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    return subprocess.run(
        [
            sys.executable, str(_IDLE_ASSERT),
            "--visible", visible,
            "--samples", "2", "--interval-s", "0",
        ],
        capture_output=True, text=True, check=False, env=env,
    )


def test_launcher_delegates_the_idle_assertion_to_the_tested_script():
    """Source invariant: assert 1 delegates; it must not hand-roll a GPU parse."""
    text = _LAUNCHER.read_text(encoding="utf-8")

    assert "scripts/gpu_idle_assert.py" in text
    assert "ASSERT_OK_GPU_IDLE_PER_CARD" in text
    # The old single-card defect: a hand-rolled `nvidia-smi ... -i "$GPU"` parse.
    assert '-i "$GPU"' not in text


def test_single_card_all_idle_passes(tmp_path: Path):
    """One card, idle on every sampled frame -> accepted."""
    completed = _assert_idle(tmp_path, "3")

    assert completed.returncode == 0, completed.stderr


def test_multi_card_all_idle_passes(tmp_path: Path):
    """Two cards, both idle -> accepted (the case the old parser broke)."""
    completed = _assert_idle(tmp_path, "3,4")

    assert completed.returncode == 0, completed.stderr


def test_multi_card_with_one_occupied_card_is_refused(tmp_path: Path):
    """Negative: one card of two is short of the position -> refused, naming it."""
    # total 81920, default position 48 GiB: used 40000 -> 41920 remaining < 49152.
    completed = _assert_idle(tmp_path, "3,4", {"FAKE_SMI_ROW4": "4, 40000, 81920, 0"})

    assert completed.returncode == 1, completed.stdout
    assert "4" in completed.stderr
    assert "头寸" in completed.stderr


def test_multi_card_with_one_busy_card_is_refused(tmp_path: Path):
    """Negative: one card of two is computing -> refused even with memory free."""
    completed = _assert_idle(tmp_path, "3,4", {"FAKE_SMI_ROW3": "3, 1, 81920, 90"})

    assert completed.returncode == 1, completed.stdout
    assert "3" in completed.stderr
