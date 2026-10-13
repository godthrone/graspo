"""★ 端到端：**生成后的产物**必须含 `export.final_formats`（★ 教训：断言过了 ≠ 产物对了）。"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "capability-matrix.data.json"
OUT = REPO / "samples" / "configs" / "matrix54-generated"


def test_t023_artifact_carries_final_formats(tmp_path):
    """★ 真跑生成器（输出到临时目录）⇒ 断言 T023 **产物文件**里含该字段。"""
    r = subprocess.run(
        [sys.executable, str(REPO / "tools/matrix/gen_configs.py"),
         "--data", str(DATA), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    text = (tmp_path / "T023.yaml").read_text(encoding="utf-8")
    assert "final_formats" in text, "★ 产物缺失该字段（只加门没造钥匙）"
    assert '"merged-hf"' in text


def test_other_ready_tiers_stay_without_the_field(tmp_path):
    """★ 与"生效集合只含 T023"一致：其余 ready 档**产物不得**含该字段。"""
    r = subprocess.run(
        [sys.executable, str(REPO / "tools/matrix/gen_configs.py"),
         "--data", str(DATA), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    for tier in ("T018", "T022", "T029", "T030", "T034", "T036", "T041", "T042"):
        assert "final_formats" not in (tmp_path / f"{tier}.yaml").read_text(encoding="utf-8"), tier
