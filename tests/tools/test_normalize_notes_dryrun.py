"""E2 单测：存量备注规范化脚本（`tools/matrix/normalize_notes.py`）。

铁律：**dry-run 必须一个字节都不写**；语义守卫必须拦下「把不通过洗成通过」这类篡改。
（真正的落盘要等主席看 dry-run 差异后批准，故本文件只钉 dry-run 与守卫。）
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "matrix"))
import note_style as ns  # noqa: E402
import normalize_notes as nn  # noqa: E402

DATA = ROOT / "docs" / "capability-matrix-20261025.data.json"
HTML = ROOT / "docs" / "GRASPO训练框架功能测试报告-20261025.html"
TEMPLATE = ROOT / "tools" / "matrix" / "template.html"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_dry_run_touches_no_file():
    before = (_sha(DATA), _sha(HTML))
    out = subprocess.run([sys.executable, str(ROOT / "tools/matrix/normalize_notes.py"),
                          "--data", str(DATA), "--html-out", str(HTML),
                          "--template", str(TEMPLATE), "--quiet-diff"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    assert "未写任何文件" in out.stdout, out.stdout[-500:]
    assert (_sha(DATA), _sha(HTML)) == before, "dry-run 竟然改了文件"


def test_guard_rejects_washing_a_failure_into_pass():
    old = "退出码 1；判据结论：① 未过，②③④ 未判定（详见跑次证据）"
    bad = "退出码 1；判据结论：①②③④ 过"
    problems = nn.guard("M011", old, bad)
    assert any("fail_ids" in p for p in problems), problems


def test_guard_rejects_dropped_load_bearing_numbers():
    old = "退出码 1；实跑步数 6400；峰值显存：26717.0 MiB；判据结论：①②③④ 过"
    bad = "退出码 1；判据结论：①②③④ 过"
    problems = nn.guard("M016", old, bad)
    assert any("steps" in p or "峰值显存" in p for p in problems), problems


def test_guard_accepts_the_normalised_form():
    old = ("训练崩溃，未产生有效训练：exit_code=1，ValueError: x（stdout.log:20）；退出码 1；"
           "实跑步数 0；判据结论：① 未过，②③④ 未判定")
    new = ns.normalize_legacy(old)
    assert nn.guard("M011", old, new) == [], (new, nn.guard("M011", old, new))
