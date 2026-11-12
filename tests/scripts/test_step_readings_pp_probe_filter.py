"""`scripts/step_readings.py`：**PP 调试探针行不得被读成训练步数**（M019 实证缺陷）。

## 缺陷（⛔ 不是口味差异，是实测的错读数）

native 的 pp-debug 会在 rollout 解码阶段逐字打：

    [pp-debug rank=0] probe=decode_step_enter step=1441 stage=0

这里的 `step=` 是**解码/生成步**，不是训练优化步。`NATIVE_STEP_RE` 只看
`(?<![A-Za-z0-9_])step['"]?\\s*[:=]\\s*(\\d+)`，不区分两者 ⇒ 探针行被当成 native 裸步数读数。

**M019 实测（A800-33 `runs/M019/stdout.log`，3 731 999 B，2026-11-10）**：
该档在 rollout 阶段即崩（`rank1 exitcode 86`、`error_file: <N/A>`），**一步优化都没跑**，
但 `grep -c global_step` = **0**、无任何 `"step": N` 训练行 ⇒ 日志里**唯一**像步数的就是探针行
⇒ `steps_actual` 被读成 **1441**。这会把「起跑即崩」的档伪装成「跑了 1441 步」。

## 实证来源（⛔ 不是凭空造的 fixture）

`_M019_PROBE_LINES` 是 `runs/M019/stdout.log` 的**逐字节原文行**（物理行号按 `\\n` 计，
与 `grep -n` 同口径）：第 `18` / `19` / `32` / `35` / `50` / `35595` / `35597` 行。
复现方式：

    sed -n '18p;19p;32p;35p;50p;35595p;35597p' runs/M019/stdout.log

训练步那几行取自仓内既有的真件摘录 `fixtures/m007_native_stdout_excerpt.txt`
（M007 真件，`"step": 1 / 3 / 400`，行号已在 `test_n3b_native_step_single_source.py` 登记）。

## 红 → 绿

* 红：起点 HEAD `312e095` 的 `scan_text` 在 `_M019_PROBE_LINES` 上取到 **1441**
  （本文件真去 `git show` 取那份实现来跑，⛔ 不手抄"期望值"）；
* 绿：新实现取到 **None**（探针行整行跳过，日志里没有别的步数读数）；
* 不误伤：**没有探针行**的常规日志（M007 / M008 / M016 三份真件摘录），
  新实现与 HEAD 实现的候选列表 / `best_of` / `origin_of` **逐字节相同**。
"""

import importlib.util
import pathlib
import subprocess
import sys
import tempfile

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import step_readings as sr  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
#: 修复**前**的起点 HEAD（本包在此之上加前置过滤）。
_HEAD_REV = "312e095"

#: `runs/M019/stdout.log` 逐字节原文行（物理行号 → 原文）。
#: ⛔ 不改写、不去 `\r`、不补全省略字段：这正是"探针行长什么样"的现场。
_M019_PROBE_LINES = """\
[pp-debug rank=0] gen stage=0 tag=0 pp_size=2 input_seq=4197 recv_alloc_seq=4197 batch=64 hidden=4096 use_cache=True
[pp-debug rank=0] probe=fwd_enter stage=0 tag=0 input_seq=4197 recv_alloc_seq=4197 batch=64 use_cache=True phase=gen
[pp-debug rank=0] probe=decode_step_enter step=1 stage=0
[pp-debug rank=1] probe=decode_step_enter step=1 stage=1
[pp-debug rank=0] probe=decode_step_enter step=2 stage=0
[pp-debug rank=0] probe=decode_step_enter step=1441 stage=0
[pp-debug rank=1] probe=decode_step_enter step=1441 stage=1"""

#: 无探针行的常规日志（真件摘录，见模块 docstring）。
_NORMAL_FIXTURES = [
    FIXTURES / "m007_native_stdout_excerpt.txt",
    FIXTURES / "m008_native_stdout_excerpt.txt",
    FIXTURES / "m016_stdout_excerpt.txt",
]


def _load_head_module():
    """取出起点 HEAD 的 `scripts/step_readings.py` 原样加载 ⇒ 当"改动前"基准。

    ★ 为什么真去取 HEAD 那份：红必须是**旧实现**的红。手写一个"应该红"的断言
      只能证明我自己的预期，不能证明缺陷存在。
    """
    src = subprocess.run(["git", "show", f"{_HEAD_REV}:scripts/step_readings.py"],
                         cwd=str(_ROOT), capture_output=True, text=True, check=True).stdout
    path = pathlib.Path(tempfile.mkdtemp()) / "head_step_readings.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("head_step_readings_for_pp_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_HEAD = _load_head_module()


def _probe_plus_training_text() -> str:
    """M019 探针行 + M007 真件的训练步行（`"step": 1 / 3 / 400`）。"""
    training = (FIXTURES / "m007_native_stdout_excerpt.txt").read_text(encoding="utf-8")
    return training.rstrip("\n") + "\n" + _M019_PROBE_LINES + "\n"


# ── ① 红：改动前的实现把探针的 `step=1441` 当成训练步读数 ─────────────────────
def test_red_head_parser_reads_pp_probe_step_as_training_step():
    """★红：HEAD 实现在探针行上取到 1441 —— 缺陷的**逐字**复现（M019 实测值）。"""
    best = sr.best_of(_HEAD.scan_text(_M019_PROBE_LINES, "stdout.log"))
    assert best is not None, "HEAD 居然没取到读数？⇒ 缺陷复现失败，先查探针行是否被改写"
    assert best["value"] == 1441, best
    # 而且它的形态被记成 native 训练步（`native_step`）——错的就是这个身份
    assert best["kind"] == "native_step", best


# ── ② 绿：新实现整行跳过探针，只剩训练步 ────────────────────────────────────
def test_probe_lines_are_skipped_so_only_training_steps_remain():
    text = _probe_plus_training_text()
    got = sr.scan_text(text, "stdout.log")
    values = [c["value"] for c in got]
    # 1441 完全消失：探针行整行跳过，⛔ 不是"取了最大值之后才发现它太大"
    assert 1441 not in values, f"探针的 1441 仍在候选里：{values}"
    # 训练步三行照旧被认出来，且最大值是 400（M007 训练末态）
    assert sorted(values) == [1, 3, 400], values
    assert all(c["kind"] == "native_step" for c in got), got
    assert sr.best_of(got)["value"] == 400


# ── ③ 只喂探针行 ⇒ fail-closed（没有训练步就是没有，⛔ 不拿探针兜底） ──────────
def test_probe_only_log_yields_no_reading_at_all():
    """M019 的现场：日志里唯一的"像步数"的行是探针 ⇒ 新实现判"取不到"（⇒ 回填器 fail-closed）。"""
    assert sr.scan_text(_M019_PROBE_LINES, "stdout.log") == []
    assert sr.best_of([]) is None


# ── ④ 不误伤：无探针行的常规日志，与改动前逐字节相同 ────────────────────────
@pytest.mark.parametrize("fixture", _NORMAL_FIXTURES, ids=lambda p: p.stem.split("_")[0])
def test_normal_logs_unchanged_byte_for_byte(fixture: pathlib.Path):
    """★ 反向断言：没有探针行时，新实现与 HEAD 实现的输出**逐条相同**。"""
    assert fixture.is_file(), fixture
    text = fixture.read_bytes().decode("utf-8", "replace")
    assert "pp-debug" not in text and "probe=" not in text, (
        f"{fixture.name} 含探针行 ⇒ 这份『常规日志』的前提没了，请换一份真件")
    old = _HEAD.scan_text(text, fixture.name)
    new = sr.scan_text(text, fixture.name)
    assert new == old, f"{fixture.name}：新口径改了常规日志的读数"
    # `best_of` / provenance 也必须一致（消费者看到的是这两个）
    assert sr.best_of(new) == _HEAD.best_of(old)
    if old:
        assert sr.origin_of(sr.best_of(new)) == _HEAD.origin_of(_HEAD.best_of(old))
