"""`scripts/make_result_json.py`：**信号硬退出的等价分支**（`error_located.kind="signal_exit"`）。

## 缺陷（M019 实证）

老逻辑只认「**消息非空**的异常行 + 其上方有 `File "…", line N` 帧」才算定位到错误。
torchrun 在子进程被信号杀死 / 异常退出且**不产出 traceback 正文**时，只在 Root Cause 段报
`exitcode: 86` + `error_file: <N/A>`，末条异常行是**空消息**的
`torch.distributed.elastic.multiprocessing.errors.ChildFailedError: `（尾随空格、无正文）
⇒ 老分支必然 `error_located.ok=False` ⇒ **已经发生的崩溃无法入表**。

## 实证来源（⛔ 不是凭空造的 fixture）

`_M019_TAIL` 是远端机 `runs/M019/stdout.log`（3 731 999 B，2026-11-10）第 `50041–50080` 行的
**逐字节原文**（40 行；★ 只把主机名脱敏了，其余逐字节未改；
复现：`sed -n '50041,50080p' runs/M019/stdout.log`）。四条条件全部落在这一段里：

| # | 条件 | 原文位置 |
|---|------|----------|
| ① | `ChildFailedError` 全名 | `stdout.log:50058` `torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ` |
| ② | 非超时非零 `exitcode` | `stdout.log:50041` `failed (exitcode: 86) …`（`echo 86 > exit_code` 的现场） |
| ③ | 「无 traceback」标记 | `stdout.log:50068/50077` `error_file: <N/A>`；`50078` `traceback : To enable traceback see: …` |
| ④ | 根因文档 | `runs/M019/why-invalid.md`（4795 B，人工撰写，含 `停止位置`/`终止阶段`、`进入优化步`、`终止方式`） |

★ ④ 依赖**人工产物**，仓内无任何脚本产出它 —— 契约已写进 `crashed_tier_evidence()` 的紧邻注释。

## 本文件钉住什么（判别力是重点）

1. 四条同时成立 ⇒ `error_located.ok=True` 且 `kind="signal_exit"`（并给出可复核的日志行号）；
2. **逐条拆掉一个条件**（四个负例，一个都不能少）⇒ 仍为 `False`。
   少了任何一个负例，这条断言就退化成"只要日志长得像 torchrun 失败就放行"；
3. 老分支优先：**有**可定位的「异常行 + 帧」时，即使信号分支四条也成立，也仍记老分支的结果
   （`kind` 不出现 `signal_exit`）；
4. 超时码（124/137）⛔ 不得借这条分支入表（那是 D1 两条既有通道的地盘）。
"""

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import make_result_json as m  # noqa: E402

#: `runs/M019/stdout.log:50041–50080` 逐字节原文（含 `ChildFailedError: ` 的**尾随空格**）。
_M019_TAIL = """\
E1110 06:56:06.471000 10 torch/distributed/elastic/multiprocessing/api.py:986] failed (exitcode: 86) local_rank: 1 (pid: 146) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/usr/local/bin/torchrun", line 6, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 990, in main
    run(args)
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/run.py", line 981, in run
    elastic_launch(
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
graspo.cli.train_worker FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-11-10_06:56:06
  host      : remote-node
  rank      : 0 (local_rank: 0)
  exitcode  : -15 (pid: 145)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 145
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-11-10_06:56:05
  host      : remote-node
  rank      : 1 (local_rank: 1)
  exitcode  : 86 (pid: 146)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================"""

#: M019 的 `runs/M019/why-invalid.md` 里**逐字**出现的根因三要素（见该文件第 5 节表格）：
#: `停止位置` + `进入优化步` + `终止方式`。此处只留要素，不抄整篇人工文档。
_ROOT_DOC = """\
## 5. 首跑 vs 重跑（不合并归因）
| | 首跑 | 重跑 |
|---|---|---|
| 停止位置 | GRASPO rollout 的 phase=logprob | phase=gen |
| 终止方式 | rank1 exitcode 86，无 traceback | 主动 abort |
| 进入优化步 | 否 | 否 |
"""


def _run_dir(tmp_path: pathlib.Path, *, stdout: str = _M019_TAIL,
             root_doc: str | None = _ROOT_DOC,
             doc_name: str = "why-invalid.md") -> pathlib.Path:
    """造一个跑次目录（只读输入：`stdout.log` + 可选人工根因文档）。"""
    run = tmp_path / "M019"
    run.mkdir()
    (run / "stdout.log").write_bytes(stdout.encode("utf-8"))
    if root_doc is not None:
        (run / doc_name).write_bytes(root_doc.encode("utf-8"))
    return run


def _evidence(run: pathlib.Path, *, exit_code: int | None = 1) -> dict:
    """只跑证据函数本身（⛔ 不经 `build_doc`：本文件钉的是 `error_located` 这一条判据）。"""
    return m.crashed_tier_evidence(
        run_dir=run, tier="M019", exit_code=exit_code, steps_actual=0,
        ckpt_dirs=[], readings={"tier": "M019"})


_SIG_KEYS = ("child_failed_reported", "non_timeout_exitcode",
             "no_traceback_marker", "root_cause_doc")


def _assert_blocked_by_only(el: dict, key: str) -> None:
    """断言：信号分支被**且只被** `key` 这一条挡住（其余三条仍成立）。

    ★ 为什么负例必须写到这一层（而不是只断言 `ok is False`）：
      「ok=False」在**改动前**也成立（老分支从来不 ok）⇒ 一句 `assert ok is False`
      在旧码上照样绿，等于没有判别力。只有钉住"挡它的是我拆掉的那一条、别的三条还站着"，
      这个负例才真的在测**新分支**的四条门槛，而不是在测"老代码本来就不行"。
    """
    detail = el["detail"]
    assert "信号退出等价分支也未成立" in detail, (
        f"四条门槛的读数没进 detail ⇒ 看不出是哪一条挡的：{detail}")
    assert f"{key}=False" in detail, f"挡住它的不是 {key}：{detail}"
    for other in _SIG_KEYS:
        if other != key:
            assert f"{other}=True" in detail, (
                f"拆 {key} 时把 {other} 也弄坏了 ⇒ 这个负例测的不是单一条门槛：{detail}")


# ── ① 四条同时成立 ⇒ ok=True 且 kind="signal_exit" ──────────────────────────
def test_signal_exit_branch_locates_error_when_all_four_hold(tmp_path):
    ev = _evidence(_run_dir(tmp_path))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is True, el["detail"]
    assert el["kind"] == "signal_exit", el
    # 可复核：指向 Root Cause 段那条非超时退出码所在的**物理行**（50041，摘录内为第 1 行）
    assert ev["error"]["log_line"] == 1, ev["error"]
    assert ev["error"]["exception"] == "ChildFailedError", ev["error"]
    assert "86" in ev["error"]["text"], ev["error"]
    # 四条子判据都留了痕（人读 detail 就能知道是哪四条放行的）
    assert all(k in el["detail"] for k in _SIG_KEYS), el["detail"]


# ── ② 逐条拆掉一个条件 ⇒ 仍为 False（四个负例，一个都不能少） ────────────────
def test_negative_missing_child_failed_error(tmp_path):
    """拆 ①：日志里没有 `…ChildFailedError` ⇒ 拒。"""
    stdout = _M019_TAIL.replace("ChildFailedError", "SomeOtherFailure").replace(
        "raise SomeOtherFailure(", "raise SomeOtherFailure(")
    ev = _evidence(_run_dir(tmp_path, stdout=stdout))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is False, el
    assert el.get("kind") != "signal_exit", el
    _assert_blocked_by_only(el, "child_failed_reported")


def test_negative_timeout_exitcode_is_not_a_crash(tmp_path):
    """拆 ②：非零退出码是超时码（124/137）⇒ 拒（属 D1 两条既有通道，⛔ 不许抢活）。"""
    stdout = _M019_TAIL.replace("failed (exitcode: 86)", "failed (exitcode: 137)")
    ev = _evidence(_run_dir(tmp_path, stdout=stdout))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is False, el
    assert el.get("kind") != "signal_exit", el
    _assert_blocked_by_only(el, "non_timeout_exitcode")


def test_negative_zero_exitcode_is_not_a_crash(tmp_path):
    """拆 ②（另一半）：退出码 0 不算崩溃 ⇒ 拒。"""
    stdout = _M019_TAIL.replace("failed (exitcode: 86)", "failed (exitcode: 0)")
    ev = _evidence(_run_dir(tmp_path, stdout=stdout))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is False, el
    _assert_blocked_by_only(el, "non_timeout_exitcode")


def test_negative_no_traceback_marker(tmp_path):
    """拆 ③：既无 `error_file: <N/A>`，也无 `traceback : To enable traceback see:` ⇒ 拒。"""
    stdout = _M019_TAIL.replace("  error_file: <N/A>", "  error_file: /tmp/some/error.json").replace(
        "  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
        "  traceback : Signal 15 (SIGTERM) received by PID 146")
    ev = _evidence(_run_dir(tmp_path, stdout=stdout))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is False, el
    assert el.get("kind") != "signal_exit", el
    _assert_blocked_by_only(el, "no_traceback_marker")


def test_negative_missing_root_cause_doc(tmp_path):
    """拆 ④：跑次目录里没有 `why-invalid.md` / `crash-evidence.md` ⇒ 拒。"""
    ev = _evidence(_run_dir(tmp_path, root_doc=None))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is False, el
    assert el.get("kind") != "signal_exit", el
    _assert_blocked_by_only(el, "root_cause_doc")


def test_negative_root_cause_doc_without_required_elements(tmp_path):
    """拆 ④（另一半）：文档在，但缺根因三要素之一（这里缺「是否进入优化步」）⇒ 拒。"""
    ev = _evidence(_run_dir(tmp_path, root_doc="## 复盘\n停止位置：phase=gen\n终止方式：abort\n"))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is False, el
    assert el.get("kind") != "signal_exit", el
    _assert_blocked_by_only(el, "root_cause_doc")


# ── ③ 第二个可接受的文档名 ─────────────────────────────────────────────────
def test_crash_evidence_md_is_an_accepted_doc_name(tmp_path):
    """契约里两个名字都认：`crash-evidence.md` 与 `why-invalid.md` 等价。"""
    ev = _evidence(_run_dir(tmp_path, doc_name="crash-evidence.md"))
    assert ev["conditions"]["error_located"]["ok"] is True


# ── ④ 老分支优先：有「异常行 + 帧」时不吃信号分支 ────────────────────────────
def test_old_branch_wins_when_a_real_traceback_is_locatable(tmp_path):
    """四条也成立，但日志里有**消息非空**的异常行 + 其上 `File "…", line N` 帧 ⇒ 走老分支。"""
    stdout = ("Traceback (most recent call last):\n"
              '  File "/workspace/graspo/src/graspo/core/lora.py", line 170, in resolve\n'
              "ValueError: lora.target_modules is required (boom)\n"
              + _M019_TAIL)
    ev = _evidence(_run_dir(tmp_path, stdout=stdout))
    el = ev["conditions"]["error_located"]
    assert el["ok"] is True, el["detail"]
    assert el.get("kind") != "signal_exit", f"老分支可用时不该走信号分支：{el}"
    assert ev["error"]["exception"] == "ValueError", ev["error"]
