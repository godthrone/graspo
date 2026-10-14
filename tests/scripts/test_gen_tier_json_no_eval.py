"""★ 回归（T018，2026-10-14）：`gen_tier_json.py` 的"无评测分支"。

不可评档（如产物为 PP 分片、拼接未实现）既不能评出读数，也不能被当成
"达标性失败"。故：三件为空时必须给非空原因，且原因**不得含"效果"字样**
（与矩阵侧 `backfill_matrix.py::not_evaluable_ok` 同判据）。
"""
import json
import subprocess
import sys
from pathlib import Path

GEN = "scripts/gen_tier_json.py"
BASE = [sys.executable, GEN, "--tier", "T999", "--mode", "full", "--parallel",
        "TP=1 PP=1 DP=1", "--model", "9B", "--label", "SFT", "--nproc", "1",
        "--steps", "1", "--recon", "1", "--backend", "native", "--host", "228",
        "--gpu", "0", "--exit-code", "1", "--non-graspo",
        "--evidence", "e.txt", "--command", "x", "--out", "/tmp/_ne_test.json"]


def _run(extra):
    return subprocess.run(BASE + extra, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2])


def test_no_eval_reason_accepted():
    """★ 正例：三件为空 + 原因非空 ⇒ 放行，且三件落 None。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：PP 分片拼接未实现"])
    assert r.returncode == 0, r.stderr[-400:]
    d = json.load(open("/tmp/_ne_test.json", encoding="utf-8"))
    assert d["after_acc"] is None and d["delta_pp"] is None
    assert d["not_evaluable_reason"]


def test_no_eval_empty_reason_rejected():
    """★ 负例：三件为空但原因为空 ⇒ FAIL-CLOSED。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "  "])
    assert r.returncode != 0
    assert "not-evaluable-reason" in (r.stderr + r.stdout)


def test_no_eval_reason_with_effect_word_rejected():
    """★ 负例：原因含"效果"字样 ⇒ FAIL-CLOSED（不可评 ≠ 达标性失败）。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "效果未达标"])
    assert r.returncode != 0
    assert "效果" in (r.stderr + r.stdout)


def test_parallel_must_match_nproc():
    """★ 回归（T018 真错位）：`params.parallel` 抄了别档模板 ⇒ TP×PP×DP 与卡数不符 ⇒ FAIL-CLOSED。"""
    r = _run(["--no-eval", "--not-evaluable-reason", "不可评：测试用",
              "--parallel", "TP=1 PP=1 DP=1", "--nproc", "4"])
    assert r.returncode != 0, "★ 应 fail-closed 却放行"
    assert "parallel" in (r.stderr + r.stdout)
