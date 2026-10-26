"""`scripts/make_result_json.py` 的**实跑步数**解析（D2）与 D1 超时伪影取证。

## D2 的实证来源（⛔ 不是凭空造的 fixture）

`fixtures/m016_stdout_excerpt.txt` = **M016 真实 stdout.log 的逐字节摘录**：

* 原件：`228:/data/remote/graspo-baseline-r1-ebd5968/runs/M016/stdout.log`
  （19 458 449 B，sha256 `b79ed573b4f3603624bbd2763f7ee5f1fcd81ac6981e2ca88ce58440c9a572ec`）；
* 摘录的**物理行号**（1 起，按 `\\n` 计数，与 `grep -n` / `sed -n` 同口径）：
  `2542, 2547, 190375, 190379, 190381, 190382, 190383, 190384, 190385`；
* 复现方式：`sed -n '2542p;2547p;190375p;190379p;190381p;190382p;190383p;190384p;190385p' stdout.log`；
* ★ 后缀是 `.txt` 不是 `.log`：仓里 `.gitignore` 忽略 `*.log`（防日志污染），
  而这是**受控 fixture**（3.5 KB），不是跑批日志——⛔ 不用 `git add -f` 绕过忽略规则。
* ★ 摘录**保留原始的 `\\r`**：ms-swift 的 tqdm 进度帧与指标 dict 落在同一物理行里（`\\r` 刷新），
  这正是"旧正则认不出"的现场之一，去掉 `\\r` 就是把现场改干净了。

## 红 → 绿（本模块的主线）

旧实现（HEAD `73de917`）的两条正则：
`['\\"]?global_step['\\"]?\\s*[:=]\\s*(\\d+)` / `global_step/max_steps['\\"]?\\s*[:=]\\s*(\\d+)`
—— 值前面那个 `'` 让 `(\\d+)` 永远匹配不上（ms-swift 打的是
`'global_step/max_steps': '6400/6400'`）⇒ M016 的 1282 行读数**一条都认不出** ⇒
`steps_actual=None` ⇒ `throughput=None` ⇒ 回填器 fail-closed。本模块把"旧口径必红"钉死
（`_OLD_*` 正则逐字取自 HEAD），并钉住新口径的四种实盘形态与 provenance。
"""

import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import make_result_json as m  # noqa: E402

_FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "m016_stdout_excerpt.txt"

#: 旧实现（HEAD `73de917`）的两条正则——逐字保留，用来证明"红"是**旧口径**的红，
#: 不是 fixture 不真实。⛔ 不得用来解析（只用它做负向断言）。
_OLD_GLOBAL_STEP_RE = r"['\"]?global_step['\"]?\s*[:=]\s*(\d+)"
_OLD_GLOBAL_STEP_SLASH_RE = r"global_step/max_steps['\"]?\s*[:=]\s*(\d+)"


# ── ① 红 → 绿：M016 真实摘录 ────────────────────────────────────────────────
def test_fixture_is_the_real_m016_excerpt():
    """fixture 必须真的含 M016 的实盘形态（防"fixture 被人改干净了还叫真实日志"）。"""
    raw = _FIXTURE.read_bytes().decode("utf-8", "replace")
    assert "'global_step/max_steps': '6400/6400'" in raw, "摘录里没有 M016 的实盘 dict 形态"
    assert "| 6400/6400 [" in raw, "摘录里没有 tqdm 完成帧"
    assert "\r" in raw, "tqdm 帧的 \\r 被洗掉了 ⇒ 已不是原样现场"
    assert "End time of running main: 2026-10-26 00:10:10" in raw, "摘录里没有训练结束标记"


def test_old_regexes_are_blind_to_the_real_form():
    """★红：旧口径在**同一份**摘录上**一条都匹配不到**（这就是 D2 的实证根因）。"""
    import re

    raw = _FIXTURE.read_bytes().decode("utf-8", "replace")
    assert re.search(_OLD_GLOBAL_STEP_RE, raw) is None, "旧 global_step 口径居然认出了？"
    assert re.search(_OLD_GLOBAL_STEP_SLASH_RE, raw) is None, "旧 global_step/max_steps 口径居然认出了？"


def test_green_parses_the_real_excerpt_with_provenance():
    """★绿：同一份摘录 ⇒ 6400，且 provenance 指到**真实行号**（能被 sed/grep 复核）。"""
    steps, why = m.parse_steps(_FIXTURE)
    assert steps == 6400, why
    assert "m016_stdout_excerpt.txt:3" in why, why      # 第 3 行 = 原件第 190375 行
    assert "global_step/max_steps" in why, why


# ── ② 四种实盘形态（spec 逐条点名） ─────────────────────────────────────────
@pytest.mark.parametrize(
    "line, expect",
    [
        # ① 单引号 + 值是字符串（M016 实盘形态，数字换成 200 == spec 举的例子）
        ("{'loss': '0', 'global_step/max_steps': '200/200', 'epoch': '1'}", 200),
        # ① 双引号 + 值是字符串 + 冒号两侧空格
        ('{"global_step/max_steps" : "200/200"}', 200),
        # ① 值是数字（无引号）
        ('{"global_step/max_steps": 200/200}', 200),
        # ② 同一 dict 家族的裸 global_step：值是字符串
        ("{'global_step': '200'}", 200),
        # ② 值是数字（JSON / trainer_state 同形）
        ('{"global_step": 200}', 200),
        # ③ tqdm 进度行（M016 的完成帧逐字）
        ("\rTrain: 100%|██████████| 200/200 [00:10<00:00,  4.95s/it]", 200),
        # ③ tqdm 中途帧（分子 = 当前步）
        ("\rTrain:  50%|█████     | 100/200 [00:05<00:05, 4.95s/it]", 100),
    ],
)
def test_recognizes_required_forms(tmp_path, line, expect):
    log = tmp_path / "stdout.log"
    log.write_bytes(line.encode("utf-8"))
    steps, why = m.parse_steps(log)
    assert steps == expect, (steps, why)


def test_unrelated_progress_bar_is_not_a_step_count(tmp_path):
    """★防呆：日志里还有下载/存盘等**无关进度条**——它们不是步数。

    口径是"多源取最大" ⇒ 一个偏大的假读数会静默污染 `steps_actual`（这里构造
    `51200/51200` 的无关条：若被当成步数，就会盖掉真实的 6400）。
    """
    log = tmp_path / "stdout.log"
    log.write_bytes(
        "\rDownloading: 100%|██████████| 51200/51200 [00:01<00:00, 1.0MB/s]\n"
        "\rTrain: 100%|██████████| 6400/6400 [14:27:53<00:00,  4.95s/it]\n"
        "{'global_step/max_steps': '6400/6400'}\n".encode("utf-8"))
    steps, why = m.parse_steps(log)
    assert steps == 6400, why


def test_missing_log_is_none_not_zero(tmp_path):
    """fail-closed：读不到就是 None（⛔ 不写 0、不估算）。"""
    steps, why = m.parse_steps(tmp_path / "nope.log")
    assert steps is None and "缺失" in why, why
    empty = tmp_path / "stdout.log"
    empty.write_text("\n".join(["[INFO:swift] begin", "nothing here"]), encoding="utf-8")
    steps, why = m.parse_steps(empty)
    assert steps is None, why


# ── ③ 多源取最大 + provenance（stdout.log ∪ trainer_state.json） ─────────────
def test_multi_source_takes_max_and_records_provenance(tmp_path):
    run = tmp_path / "M999"
    (run / "out" / "checkpoint-200").mkdir(parents=True)
    (run / "stdout.log").write_bytes(
        "{'global_step/max_steps': '200/200'}\n"
        "\rTrain: 100%|██████████| 200/200 [00:10<00:00, 4.95s/it]\n".encode("utf-8"))
    (run / "out" / "checkpoint-200" / "trainer_state.json").write_text(
        json.dumps({"global_step": 200, "epoch": 1.0}), encoding="utf-8")
    (run / "out" / "checkpoint-300").mkdir()
    (run / "out" / "checkpoint-300" / "trainer_state.json").write_text(
        json.dumps({"global_step": 300, "epoch": 1.5}), encoding="utf-8")

    steps, why, prov = m.parse_steps_evidence(run)
    assert steps == 300, why
    # provenance：取到最大值的那一处必须可定位（哪个文件、哪个字段）
    assert prov and prov[-1]["value"] == 300, prov
    assert prov[-1]["source"].endswith("checkpoint-300/trainer_state.json"), prov
    assert "checkpoint-300/trainer_state.json" in why, why


# ── ④ D1 超时伪影取证（`--recover-timeout-artifact` 的唯一判据实现） ─────────
def _fake_run(tmp_path, *, finish=True, steps=6400, ckpt_ok=True, nan=False, epochs=(8, 8)):
    """造一个"训练已跑完、却被外层 timeout 杀掉"的跑次目录（形态照 M016）。"""
    run = tmp_path / "M016"
    run.mkdir()
    lines = ["[INFO:swift] begin"]
    if finish:
        lines.append("[INFO:swift] End time of running main: 2026-10-26 00:10:10.903826")
    (run / "stdout.log").write_bytes(
        ("\n".join([f"\rTrain: 100%|██████████| {steps}/{steps} [00:10<00:00, 4.95s/it]",
                     f"{{'global_step/max_steps': '{steps}/{steps}'}}"]) + "\n"
         + "\n".join(lines) + "\n").encode("utf-8"))
    (run / "readings.json").write_text(json.dumps(
        {"tier": "M016", "exit_code": 137, "elapsed_sec": 57720.0, "timeout_sec": 57600}),
        encoding="utf-8")
    (run / "exit_code").write_text("137\n", encoding="utf-8")
    (run / "elapsed_sec").write_text("57720.0\n", encoding="utf-8")
    (tmp_path / "M016.console.log").write_text(
        "2026-10-25T09:41:51Z [start] M016 GPUS=2 image=x timeout=57600s\n"
        "2026-10-26T01:43:52Z [end] M016 exit_code=137 elapsed=57720s\n", encoding="utf-8")
    return run


def _artifact(run, **over):
    kw = dict(exit_code=137, elapsed_raw=57720.0, steps_actual=6400, steps_planned=800,
              epochs_done=8, epochs_planned=8, loadable=True, nan_inf=False)
    kw.update(over)
    return m.timeout_artifact_evidence(run_dir=run, **kw)


def test_timeout_artifact_derives_success_only_with_full_evidence(tmp_path):
    ev = _artifact(_fake_run(tmp_path))
    assert ev["applies"] is True, ev
    assert ev["exit_code_raw"] == 137 and ev["exit_code_derived"] == 0, ev
    # elapsed = runner 起跑(09:41:51Z) → 主流程返回(00:10:10) = 52099 s（⛔ 不含收尾空等）
    assert ev["elapsed_derived"] == pytest.approx(52099.0, abs=1.0), ev["elapsed_cross_check"]
    assert ev["elapsed_derived"] < ev["elapsed_raw"]
    assert all(c["ok"] for c in ev["conditions"].values()), ev["conditions"]
    assert "stdout.log" in ev["finish_marker"]["source"]


def test_timeout_artifact_on_the_real_m016_excerpt(tmp_path):
    """用**M016 真实摘录**当 stdout.log：判据全部成立，且交叉校验取到 trainer 自报墙钟。

    ⚠ 这条钉的是"同一物理行里既有 `train_runtime` 又有 `global_step/max_steps`"这个真实现场
      （M016 stdout.log:190379）——三个判据必须各自独立地看同一行，否则自报墙钟会被吞掉。
    """
    run = _fake_run(tmp_path)
    (run / "stdout.log").write_bytes(_FIXTURE.read_bytes())
    ev = _artifact(run)
    assert ev["applies"] is True, ev
    assert ev["train_runtime_sec"] == pytest.approx(52070.0, abs=1.0), ev["train_runtime_sec"]
    assert ev["elapsed_derived"] == pytest.approx(52099.0, abs=1.0)
    assert ev["elapsed_cross_check"]["train_runtime_sec"] <= ev["elapsed_derived"]
    assert ev["last_step_reading"]["pair"] == [6400, 6400], ev["last_step_reading"]


@pytest.mark.parametrize(
    "over, broken",
    [
        ({}, "trainer_main_returned"),          # 没有 End time 标记 ⇒ 真的没跑完（超时失败）
        ({"loadable": None}, "final_ckpt_loadable"),
        ({"nan_inf": True}, "nan_or_inf_free"),
        ({"epochs_done": 3}, "epochs_completed"),
        ({"elapsed_raw": 100.0}, "elapsed_reached_timeout"),   # 没到超时上限 ⇒ 不是超时产物
        ({"exit_code": 1}, "raw_is_timeout_code"),
    ],
)
def test_timeout_artifact_never_fires_without_evidence(tmp_path, over, broken):
    """★fail-closed：任一条件不成立 ⇒ `applies=False`，**一个字段都不改**。"""
    ev = _artifact(_fake_run(tmp_path, finish=("trainer_main_returned" not in broken)), **over)
    assert ev["applies"] is False, ev
    assert ev["conditions"][broken]["ok"] is False, ev["conditions"]
    assert ev["exit_code_derived"] is None, "不成立时仍给了派生退出码 ⇒ 会掩盖真失败"
    assert ev["elapsed_derived"] is None, "不成立时仍改了墙钟 ⇒ 会掩盖真失败"
