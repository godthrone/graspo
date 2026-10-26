"""`scripts/record_checks.py` 的 `--algo-checks-dir` 语义（2026-10-26 修判据③ 断链）。

## 为什么有这件（实测缺陷）

`tests/e2e/run_matrix68.sh` 旧代码**恒**传 `--algo-checks-dir <run-root>/algo-checks`，而该目录
**从来不存在**；`record_checks.py` 的语义是"**显式**给目录 ⇒ 只认该目录、**不回退**仓内
`samples/configs/matrix68-checks/`" ⇒ 每一档的判据③ 都恒为"缺证据 ⇒ 待测试"，只能事后手工
不带该参数重跑一次才救得回来（M016 / M052 都是这么救的）。

修法定为 **(a) runner 缺省不传该参数**（改一处、语义更清楚；`run_matrix68.sh` 现只在显式设了
`GRASPO_ALGO_CHECKS_DIR` 时才传）。本文件把 `record_checks.py` 这一侧的**三种语义**钉死，
每种都成对（该回退的必须回退、该拒绝的必须拒绝）：

  ① **未给**（`--algo-checks-dir` 缺省、`$GRASPO_ALGO_CHECKS_DIR` 未设）⇒
     查 `<run-dir>/algo-checks`，缺件 ⇒ **回退仓内** `samples/configs/matrix68-checks/`
     ⇒ 判据③ 落 `true`；
  ② **给了且存在** ⇒ **只认显式**目录：即使仓内有一份 `pass` 签署件，显式目录写 `false`
     也必须落 `false`（⛔ 不回退、⛔ 不偷偷用旧签署件）；
  ③ **给了但不存在 / 给了但为空** ⇒ **fail-closed**：不写 `algo_checks_passed` 键（rc=4），
     ⛔ 不静默回退（这正是缺陷的形态）。

纯标准库 + importlib 按文件路径加载被测脚本（不拉 torch / 项目包）。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RC = _load("scripts/record_checks.py", "record_checks_for_algo_dir")

TIER = "M001"


def _signature(*, tier: str = TIER, verdict: bool = True) -> dict:
    """一份合契约的 GRASPO 逐档签署件（第 9 节 8 条不变式逐条 + 检查人/时间）。"""
    return {
        "tier": tier,
        "algorithm": "GRASPO",
        "checker": "unit-test",
        "checked_at": "2026-10-26T00:00:00+00:00",
        "items": [{"id": iid, "pass": verdict, "evidence": f"unit-test：{iid} 已核"}
                  for iid in RC.GRASPO_ITEM_IDS],
    }


def _run_root(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run" / TIER
    run_dir.mkdir(parents=True)
    (run_dir / "readings.json").write_text(
        json.dumps({"schema": "matrix-run/v1", "tier": TIER}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return run_dir


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tiers": [{
        "tier_id": TIER,
        "algorithm": "GRASPO",
        "tuner_type": "lora",
        "data": {"subset_size": 100, "subset_sha256": "a" * 64},
    }]}, ensure_ascii=False), encoding="utf-8")
    return path


def _write_sig(directory: Path, doc: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{doc['tier']}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _checks(run_dir: Path) -> dict:
    doc = json.loads((run_dir / "readings.json").read_text(encoding="utf-8"))
    return doc.get("checks") or {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """把两个"隐式指定目录"的环境变量清掉——否则本机已导出的值会让三态断言失真。"""
    monkeypatch.delenv("GRASPO_ALGO_CHECKS_DIR", raising=False)
    monkeypatch.delenv("GRASPO_EVAL_DIR", raising=False)


def _main(tmp_path: Path, run_dir: Path, *extra: str) -> int:
    return RC.main(["--tier", TIER, "--run-dir", str(run_dir),
                    "--manifest", str(_manifest(tmp_path)), *extra])


# ── ① 未给 ⇒ 回退仓内（判据③ 必须落 true）─────────────────────────────────────
def test_absent_arg_falls_back_to_in_repo_signed_dir(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture) -> None:
    run_dir = _run_root(tmp_path)
    repo_dir = tmp_path / "repo-checks"
    _write_sig(repo_dir, _signature())
    monkeypatch.setattr(RC, "GRASPO_SIG_DIR", repo_dir)      # 仓内默认目录（回退源）

    rc = _main(tmp_path, run_dir)
    out = capsys.readouterr().out

    assert _checks(run_dir).get("algo_checks_passed") is True, out
    assert "缺件时回退仓内默认" in out, out
    # 判据④ 无评测报告 ⇒ 缺证据 ⇒ rc=4（本用例只关心③；④ 的 fail-closed 是既有行为）
    assert rc == 4, out


# ── ② 给了且存在 ⇒ 只认显式（⛔ 不回退到仓内）────────────────────────────────
def test_explicit_dir_with_file_is_the_only_source(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture) -> None:
    run_dir = _run_root(tmp_path)
    explicit = tmp_path / "explicit-checks"
    _write_sig(explicit, _signature())
    monkeypatch.setattr(RC, "GRASPO_SIG_DIR", tmp_path / "repo-checks")   # 仓内**无件**

    rc = _main(tmp_path, run_dir, "--algo-checks-dir", str(explicit))
    out = capsys.readouterr().out

    assert _checks(run_dir).get("algo_checks_passed") is True, out
    assert "显式指定 ⇒ 不回退" in out, out
    assert rc == 4, out


def test_explicit_false_is_not_overridden_by_in_repo_pass(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture) -> None:
    """★ 防"偷偷用旧签署件"：显式目录写 `false`，仓内那份是 `pass` ⇒ 必须落 `false`。"""
    run_dir = _run_root(tmp_path)
    explicit = tmp_path / "explicit-checks"
    _write_sig(explicit, _signature(verdict=False))
    repo_dir = tmp_path / "repo-checks"
    _write_sig(repo_dir, _signature(verdict=True))
    monkeypatch.setattr(RC, "GRASPO_SIG_DIR", repo_dir)

    rc = _main(tmp_path, run_dir, "--algo-checks-dir", str(explicit))
    out = capsys.readouterr().out

    assert _checks(run_dir).get("algo_checks_passed") is False, out
    assert rc == 4, out


# ── ③ 给了但不存在/为空 ⇒ fail-closed（不写键，⛔ 不静默回退）─────────────────
@pytest.mark.parametrize("make_dir", [False, True], ids=["缺目录", "空目录"])
def test_explicit_missing_or_empty_dir_fails_closed(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture,
                                                    make_dir: bool) -> None:
    run_dir = _run_root(tmp_path)
    explicit = tmp_path / "explicit-checks"
    if make_dir:
        explicit.mkdir()                                     # 存在但空
    repo_dir = tmp_path / "repo-checks"
    _write_sig(repo_dir, _signature())                       # 仓内**有**合法签署件
    monkeypatch.setattr(RC, "GRASPO_SIG_DIR", repo_dir)

    rc = _main(tmp_path, run_dir, "--algo-checks-dir", str(explicit))
    out = capsys.readouterr().out

    assert "algo_checks_passed" not in _checks(run_dir), f"显式目录缺件⛔不得回退/不得写键：{out}"
    assert "显式指定 ⇒ 不回退" in out, out
    assert "缺 GRASPO 算法检查证据" in out, out
    assert rc == 4, out
