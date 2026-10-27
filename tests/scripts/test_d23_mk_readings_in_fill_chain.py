"""D23：`mk_readings.py` 是**手工步**、批处理链不调用 —— 接进填表/取证链，幂等、失败可见。

## 病史（实证）

`readings.json` 是 `make_result_json.py` / `record_checks.py` 的**唯一读数入口**。
编排层只落 `readings.raw.json`（runner 写）+ `exit_code` / `exit_code_outer` / `elapsed_sec`
（容器落盘），把这堆合成 `readings.json` 的那一步**没人调用** ⇒ 漏补一次就卡死取证链：
`mk_evidence` 报 `FAIL: readings.json 缺失或不可解析`（M013、M018 两次人工补过）。

更糟的一份实测（M008，A800-33 `/data/remote/runs/M008`）：`readings.json` 在盘上但**是空壳**
——`exit_code=99`、`elapsed_sec=null`、`gpus=null`、`note=null`，而同一目录的
`readings.raw.json` 写着 `exit_code=0 / elapsed_sec=1394 / gpus=[4,5,6,7]`
⇒ 判据① 会按"退出码 99"把一档**跑成功**的档判成失败。

## 本文件钉住什么

1. **链中自动调用**：`tools/matrix/fill_one.py` 在调用 `make_result_json.py` **之前**
   无条件先跑 `scripts/mk_readings.py`（结构 + 行为双向钉，⛔ 不靠"看着像调用了"）；
2. **幂等**：同一份输入重复跑 ⇒ 逐字节相同，且第二次**不写盘**（打印 `unchanged`）；
   合成时**保留** `checks` / `checks_evidence` / `teacher_after_acc`（唯一产出者是
   `record_checks.py`，⛔ 不许被洗掉）；
3. **失败可见**：`readings.raw.json` 不可解析 / 档号错配 ⇒ 非零退出 + 原样输出，
   ⛔ 不静默吞；链里遇到非零 ⇒ 拒绝回填（rc=3，**未写任何文件**）；
4. **不越界**：没有 `readings.raw.json` 的跑次（228 那类由 runner 自写 `readings.json`）
   ⇒ 明确跳过，⛔ 不重写别人的件。
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import pathlib
import sys
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MK_READINGS = _ROOT / "scripts" / "mk_readings.py"
sys.path.insert(0, str(_ROOT / "tools" / "matrix"))


def _load_fill_one():
    """按路径加载 `tools/matrix/fill_one.py`（模块级只做 import/常量，不落盘）。"""
    spec = importlib.util.spec_from_file_location("fill_one_for_d23", _ROOT / "tools" / "matrix" / "fill_one.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


FILL = _load_fill_one()


def _run_dir(tmp_path: pathlib.Path, name: str = "M018", *, raw=True, exit_code="0",
             elapsed="1234.5") -> pathlib.Path:
    run = tmp_path / name
    run.mkdir()
    if raw:
        (run / "readings.raw.json").write_text(json.dumps({
            "tier": name, "exit_code": 0, "elapsed_sec": 1234, "gpus": ["0", "1"],
            "config": f"samples/configs/matrix68/{name}.yaml", "image": "graspo:test",
            "timeout_sec": 1500, "repo": "/data/remote/work/graspo-baseline-r1-6189bd8",
            "command": f"run_tier_b09.sh {name} 0,1 1500",
            "note": "elapsed_sec = 外层墙钟",
        }), encoding="utf-8")
    if exit_code is not None:
        (run / "exit_code").write_text(exit_code + "\n", encoding="utf-8")
    if elapsed is not None:
        (run / "elapsed_sec").write_text(elapsed + "\n", encoding="utf-8")
    return run


def _mk(argv: list[str]) -> tuple[int, str]:
    """用**当前解释器**真跑 `scripts/mk_readings.py`（不是 import 后调 main）。"""
    import subprocess

    p = subprocess.run([sys.executable, str(_MK_READINGS), *argv],
                       capture_output=True, text=True, cwd=str(_ROOT))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ── ① 链中自动调用（结构：调用点在 make_result_json 之前 + 失败分支返回 3） ──
def test_chain_calls_mk_readings_before_make_result_json():
    src = inspect.getsource(FILL.main)
    assert "ensure_readings(" in src, "fill_one.main 没有调用前置合成步 ⇒ D23 未接进链"
    assert src.index("ensure_readings(") < src.index("make_result_json.py"), \
        "前置合成步必须在读数派生**之前**（readings.json 是 make_result_json 的输入）"

    tree = ast.parse(textwrap.dedent(src))
    guard = [n for n in ast.walk(tree)
             if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
             and isinstance(n.test.left, ast.Name) and n.test.left.id == "mk_rc"]
    assert guard, "mk_rc 判定缺失 ⇒ 该步失败会被静默吞掉"
    body = guard[0].body
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
               and n.value.value == 3 for n in body), "失败分支必须 rc=3（拒绝回填，未写任何文件）"
    assert any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, "id", "") == "print" for n in body), \
        "失败分支必须**打印**原因（失败可见，⛔ 不静默吞）"


def test_chain_hands_the_run_dir_and_honours_dry_run(tmp_path):
    run = _run_dir(tmp_path)
    calls: list[list[str]] = []

    def fake(cmd):
        calls.append(list(cmd))
        return 0, "wrote …"

    rc, out = FILL.ensure_readings(run, runner=fake)
    assert (rc, out) == (0, "wrote …")
    assert calls[0][1].endswith("scripts/mk_readings.py"), calls[0]
    assert calls[0][-2:] == ["--run-dir", str(run)], calls[0]
    assert "--check" not in calls[0]

    FILL.ensure_readings(run, dry_run=True, runner=fake)
    assert calls[1][-1] == "--check", "dry-run 必须只算不写（--dry-run 全程不落盘）"


def test_chain_skips_runner_written_readings(tmp_path):
    """没有 `readings.raw.json` ⇒ **不调用**（228 那类跑次的 readings.json 由 runner 自写）。"""
    run = _run_dir(tmp_path, raw=False)
    calls: list[list[str]] = []
    rc, out = FILL.ensure_readings(run, runner=lambda cmd: calls.append(list(cmd)) or (0, ""))
    assert rc == 0 and calls == [], calls
    assert "跳过" in out and "readings.raw.json" in out, out


# ── ② 幂等：重复跑逐字节相同，且保留 checks ────────────────────────────────
def test_mk_readings_is_idempotent_byte_identical(tmp_path):
    run = _run_dir(tmp_path)
    # 盘上已有一份"空壳"件（逐字照 M008 实盘：全 null + exit_code=99），并带一条 checks 证据
    (run / "readings.json").write_bytes(json.dumps(
        {"tier": "M018", "exit_code": 99, "exit_code_outer": None, "elapsed_sec": None,
         "comparable": None, "env_diff_rc": None, "gpus": None, "config": None, "image": None,
         "timeout_sec": None, "repo": None, "command": None, "note": None},
        indent=1, ensure_ascii=False).encode("utf-8"))

    rc1, out1 = _mk(["--run-dir", str(run)])
    assert rc1 == 0 and "wrote" in out1, out1
    first = (run / "readings.json").read_bytes()
    doc = json.loads(first)
    assert doc["exit_code"] == 0, doc                       # 取容器内落盘的真退出码（不是 99）
    assert doc["elapsed_sec"] == 1234.5 and doc["gpus"] == ["0", "1"], doc

    rc2, out2 = _mk(["--run-dir", str(run)])
    assert rc2 == 0 and "unchanged" in out2, out2
    assert (run / "readings.json").read_bytes() == first, "重复跑改了字节 ⇒ 不幂等"

    # checks 是 record_checks.py 的唯一产出物 ⇒ 合成时**保留**（⛔ 不许被洗掉）
    doc["checks"] = {"algo_checks_passed": True, "inference_parseable": True}
    (run / "readings.json").write_bytes(json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8"))
    assert _mk(["--run-dir", str(run)])[0] == 0
    again = json.loads((run / "readings.json").read_bytes())
    assert again["checks"] == doc["checks"], again


def test_check_mode_writes_nothing(tmp_path):
    run = _run_dir(tmp_path)
    before = (run / "readings.json").read_bytes() if (run / "readings.json").is_file() else None
    rc, out = _mk(["--run-dir", str(run), "--check"])
    assert rc == 0 and "would-write" in out, out
    assert before is None and not (run / "readings.json").is_file(), "--check 居然落盘了"


# ── ③ 失败可见 ──────────────────────────────────────────────────────────────
def test_unparsable_raw_fails_loudly(tmp_path):
    run = _run_dir(tmp_path, raw=False)
    (run / "readings.raw.json").write_text("{not json", encoding="utf-8")
    rc, out = _mk(["--run-dir", str(run)])
    assert rc == 1, out
    assert "FATAL" in out and "readings.raw.json" in out, out
    assert not (run / "readings.json").is_file(), "失败时不许落一份半成品"

    # 链里同一条失败必须**冒出来**（非零 + 输出非空），⛔ 不静默吞
    chain_rc, chain_out = FILL.ensure_readings(run)
    assert chain_rc == 1 and "FATAL" in chain_out, (chain_rc, chain_out)


def test_tier_mismatch_is_refused(tmp_path):
    run = _run_dir(tmp_path, name="M018")
    raw = json.loads((run / "readings.raw.json").read_text(encoding="utf-8"))
    raw["tier"] = "M013"                                     # 跑 A 填 B ⇒ 拒绝
    (run / "readings.raw.json").write_text(json.dumps(raw), encoding="utf-8")
    rc, out = _mk(["--run-dir", str(run)])
    assert rc == 1 and "档号错配" in out, out


def test_usage_error_on_missing_dir(tmp_path):
    rc, out = _mk(["--run-dir", str(tmp_path / "nope")])
    assert rc == 2 and "不存在" in out, out


# ── ④ 不越界：没有 raw ⇒ 盘上那份 readings.json 一个字节都不动 ───────────────
def test_no_raw_never_touches_existing_readings(tmp_path):
    run = _run_dir(tmp_path, raw=False)
    sentinel = b'{"tier": "M018", "exit_code": 0, "note": "runner \xe8\x87\xaa\xe5\x86\x99"}'
    (run / "readings.json").write_bytes(sentinel)
    rc, out = _mk(["--run-dir", str(run)])
    assert rc == 0 and "跳过" in out, out
    assert (run / "readings.json").read_bytes() == sentinel, "覆盖了 runner 自写的 readings.json"
