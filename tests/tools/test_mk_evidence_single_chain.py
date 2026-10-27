"""取证链**唯一入口** `tools/matrix/mk_evidence.sh` —— 收编 A800 侧在版控之外的
`work/b09/mk_evidence.sh`，并把「读数件合成」这一步**接回链里**（D23 缺陷的根因）。

## 病史（实证，两次人工补 + 一份空壳）

A800 侧那份 `mk_evidence.sh` 不在版控里，它自己直连
`scripts/record_checks.py` + `scripts/make_result_json.py`，**绕过 `fill_one.py`**
⇒ `fill_one.py` 链首的「读数件合成（`scripts/mk_readings.py`）」这一步在 A800 上
根本没被调用。后果：
· M013 / M018：`mk_evidence` 报 `rc=2`、`FAIL: readings.json 缺失或不可解析`，各人工补一次；
· M008（`/data/remote/runs/M008`）：`readings.json` 是**空壳**（`exit_code=99`、其余全 null），
  而同一目录 `readings.raw.json` 写着 `exit_code=0 / elapsed=1394 / gpus=[4,5,6,7]`
  ⇒ 判据① 会把**跑成功**的档按"退出码 99"判失败。

## 本文件钉住什么（⛔ 只钉这几条，不测别的）

1. **单一真相源**：⓪ 读数件合成走 `fill_one.py::ensure_readings()`**同一个函数**（⛔ 不另写
   skip 判定/argv）；② 派生+判据+闸门+落盘走 `tools/matrix/fill_one.sh`**同一条命令**
   （⛔ 源码里不得再出现 `make_result_json.py` / `scripts/mk_readings.py` 的第二份调用
   —— 那正是本轮缺陷的成因）；
2. **顺序**：⓪ 在 ① 之前（`record_checks.py` 读 `readings.json`，缺它直接 FAIL）；
3. **幂等**：同一份输入重复跑 ⇒ `readings.json` 逐字节不变（第二次 `unchanged`、不写盘）；
4. **失败可见**：`readings.raw.json` 不可解析 ⇒ 非零退出 + 原样贴出 `FATAL`，且 ①② 都**不得**被调用；
5. **`--dry-run` 全程不落盘**：⓪ 只算不写、①（写盘步）跳过并明示、② 透传 `--dry-run`。
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MK = _ROOT / "tools" / "matrix" / "mk_evidence.sh"
_FILL_ONE = _ROOT / "tools" / "matrix" / "fill_one.sh"
_RECORD_CHECKS = _ROOT / "scripts" / "record_checks.py"


# ── 夹具：跑次目录 + 两个可注入的桩（① 与 ② 是外部进程，桩只为"看得见调用没调用"）──
def _run_dir(tmp: pathlib.Path, name: str = "M018", *, raw: bool = True,
             raw_text: str | None = None) -> pathlib.Path:
    run = tmp / "runs" / name
    run.mkdir(parents=True)
    if raw_text is not None:
        (run / "readings.raw.json").write_text(raw_text, encoding="utf-8")
    elif raw:
        (run / "readings.raw.json").write_text(json.dumps({
            "tier": name, "exit_code": 0, "elapsed_sec": 1394, "gpus": ["4", "5", "6", "7"],
            "config": f"samples/configs/matrix68/{name}.yaml", "image": "graspo:test",
            "timeout_sec": 1500, "repo": "/data/remote/work/graspo-baseline-r1-6189bd8",
            "command": f"run_tier_b09.sh {name} 4,5,6,7 1500", "note": "elapsed_sec = 外层墙钟",
        }), encoding="utf-8")
    (run / "exit_code").write_text("0\n", encoding="utf-8")
    (run / "elapsed_sec").write_text("1394\n", encoding="utf-8")
    return run


def _stub(tmp: pathlib.Path, name: str, *, rc: int = 0) -> pathlib.Path:
    """造一个"被调用就留痕"的桩：把 **完整 argv** 追加进 <tmp>/<name>.calls，退出码 = rc。"""
    log = tmp / f"{name}.calls"
    body = tmp / f"{name}.stub"
    if name == "fill_one":                       # ② 是 bash 入口 ⇒ bash 桩
        body.write_text("#!/usr/bin/env bash\n"
                        f'printf "%s\\n" "$*" >> "{log}"\n'
                        f'exit {rc}\n', encoding="utf-8")
        body.chmod(0o755)
    else:                                        # ① 是 python 入口 ⇒ python 桩
        body.write_text("import sys\n"
                        f'open(r"{log}", "a", encoding="utf-8").write(" ".join(sys.argv[1:]) + "\\n")\n'
                        f"sys.exit({rc})\n", encoding="utf-8")
    return body


def _calls(tmp: pathlib.Path, name: str) -> list[str]:
    p = tmp / f"{name}.calls"
    return p.read_text(encoding="utf-8").splitlines() if p.is_file() else []


def _mk(tmp: pathlib.Path, args: list[str], *, fill_rc: int = 0, rec_rc: int = 0,
        tier_root: pathlib.Path | None = None) -> tuple[int, str]:
    """真跑 shell 入口（**不 import**）：① ② 换成桩，其余（⓪ 读数件合成）走真实现。"""
    root = tier_root or (tmp / "runs")
    env = dict(os.environ)
    env.update({
        "GRASPO_RUN_ROOT": str(root),
        "GRASPO_EVIDENCE_STEP_FILL_ONE": str(_stub(tmp, "fill_one", rc=fill_rc)),
        "GRASPO_EVIDENCE_STEP_RECORD_CHECKS": str(_stub(tmp, "record_checks", rc=rec_rc)),
    })
    for k in ("GRASPO_EVAL_ROOT", "GRASPO_EVIDENCE_EVAL_DIR"):
        env.pop(k, None)
    p = subprocess.run(["bash", str(_MK), *args], capture_output=True, text=True,
                       cwd=str(_ROOT), env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _code_lines() -> list[str]:
    """取 shell 源里**非注释、非文档头**的行（注释里提到旧文件名不算"第二份实现"）。"""
    out = []
    for ln in _MK.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


# ── ① 单一真相源（结构钉：⛔ 不许出现第二份口径/第二份派生调用） ──────────────
def test_chain_uses_the_one_and_only_implementations():
    code = "\n".join(_code_lines())
    assert "ensure_readings(" in code, \
        "⓪ 读数件合成必须走 fill_one.py::ensure_readings（单一真相源），⛔ 不许在 shell 里自写一份"
    assert "scripts/mk_readings.py" not in code, \
        "⛔ shell 里出现了第二份 mk_readings 调用（argv/口径会漂移）—— 这正是本缺陷的成因"
    assert "make_result_json.py" not in code, \
        "⛔ shell 里出现了第二份 make_result_json 调用（绕过 fill_one 的判据+闸门）"
    assert "tools/matrix/fill_one.sh" in code, "② 必须委托仓内唯一入口 tools/matrix/fill_one.sh（作为缺省值）"


def test_readings_step_runs_before_record_checks():
    """⓪ 必须排在 ① 之前：`record_checks.py` 读 `readings.json`，缺它直接 FAIL（M013/M018 实证）。"""
    src = _MK.read_text(encoding="utf-8")
    assert src.index("readings_step() {") < src.index("record_checks $TIER"), \
        "⓪ 读数件合成必须排在 ① record_checks 之前"


# ── ② 顺序 + argv（行为钉） ─────────────────────────────────────────────────
def test_calls_readings_then_record_checks_then_fill_one(tmp_path):
    run = _run_dir(tmp_path)
    rc, out = _mk(tmp_path, ["M018"])
    assert rc == 0, out
    i0, i1, i2 = out.index("⓪"), out.index("① record_checks M018"), out.index("② 委托")
    assert i0 < i1 < i2, "链序必须是 ⓪ → ① → ②：" + out

    rec = _calls(tmp_path, "record_checks")
    assert rec and rec[0].split()[:2] == ["--tier", "M018"], rec
    assert "--run-dir" in rec[0] and str(run) in rec[0], rec

    fill = _calls(tmp_path, "fill_one")
    assert fill and fill[0].strip() == "M018", fill     # 缺省不带 --dry-run
    # readings.json 已由 ⓪ 合成（空壳修成真件）：真退出码 0、gpus 搬自 raw
    doc = json.loads((run / "readings.json").read_text(encoding="utf-8"))
    assert doc["exit_code"] == 0 and doc["gpus"] == ["4", "5", "6", "7"], doc


# ── ③ 幂等：重复跑逐字节不变，第二次不写盘 ──────────────────────────────────
def test_duplicate_run_is_byte_identical(tmp_path):
    run = _run_dir(tmp_path)
    # 盘上先摆一份"M008 型空壳"（全 null + exit_code=99）
    (run / "readings.json").write_bytes(json.dumps(
        {"tier": "M018", "exit_code": 99, "exit_code_outer": None, "elapsed_sec": None,
         "gpus": None, "note": None}, indent=1, ensure_ascii=False).encode("utf-8"), )

    rc1, out1 = _mk(tmp_path, ["M018"])
    assert rc1 == 0 and "wrote" in out1, out1
    first = (run / "readings.json").read_bytes()
    assert json.loads(first)["exit_code"] == 0, json.loads(first)

    rc2, out2 = _mk(tmp_path, ["M018"])
    assert rc2 == 0 and "unchanged" in out2, out2
    assert (run / "readings.json").read_bytes() == first, "重复跑改了字节 ⇒ 不幂等"


# ── ④ 失败可见：读数件合成失败 ⇒ 非零 + 原样贴出，且 ①② 一次都不许被调用 ─────
def test_readings_failure_is_loud_and_stops_the_chain(tmp_path):
    run = _run_dir(tmp_path, raw_text="{not json")
    rc, out = _mk(tmp_path, ["M018"])
    assert rc != 0, "读数件合成失败必须非零退出（⛔ 不许静默吞）"
    assert "FATAL" in out and "readings.raw.json" in out, "错误必须**原样可见**：" + out
    assert "拒绝取证" in out, out
    assert _calls(tmp_path, "record_checks") == [], "读数件都没合成好，① 不许被调用"
    assert _calls(tmp_path, "fill_one") == [], "读数件都没合成好，② 不许被调用"
    assert not (run / "readings.json").is_file(), "失败时不许落一份半成品"


# ── ⑤ --dry-run 全程不落盘（⓪ 只算不写、① 跳过、② 透传） ────────────────────
def test_dry_run_writes_nothing_and_skips_the_write_step(tmp_path):
    run = _run_dir(tmp_path)
    rc, out = _mk(tmp_path, ["M018", "--dry-run"])
    assert rc == 0, out
    assert not (run / "readings.json").is_file(), "--dry-run 居然落盘了"
    assert "would-write" in out, out
    assert "① record_checks.py **跳过**" in out, out
    assert _calls(tmp_path, "record_checks") == [], "record_checks 是写盘步，--dry-run 必须跳过"
    assert _calls(tmp_path, "fill_one") == ["M018 --dry-run"], _calls(tmp_path, "fill_one")


def test_dry_run_does_not_write_even_when_readings_is_stale(tmp_path):
    run = _run_dir(tmp_path)
    sentinel = b'{"tier": "M018", "exit_code": 99}'
    (run / "readings.json").write_bytes(sentinel)
    rc, out = _mk(tmp_path, ["M018", "--dry-run"])
    assert rc == 0 and "would-write" in out, out
    assert (run / "readings.json").read_bytes() == sentinel, "--dry-run 改了盘上的读数件"


# ── ⑥ 用法/边界：非法档号、缺跑次目录都不得动别人的件 ──────────────────────
@pytest.mark.parametrize("args,want", [(["M18"], "非法档号"), (["X001"], "非法档号"),
                                       ([], "用法")])
def test_bad_usage_is_refused(tmp_path, args, want):
    rc, out = _mk(tmp_path, args)
    assert rc == 2, out
    assert _calls(tmp_path, "fill_one") == [], out


def test_missing_run_dir_never_calls_the_chain(tmp_path):
    rc, out = _mk(tmp_path, ["M018"], tier_root=tmp_path / "empty-runs")
    assert rc == 3 and "缺跑次目录" in out, out
    assert _calls(tmp_path, "record_checks") == [] and _calls(tmp_path, "fill_one") == [], out
