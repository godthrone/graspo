"""N3b：native 档**步数取不到**（日志只有 `"step": N`、没有 `global_step`）——修法与固化。

## 实证来源（⛔ 不是凭空造的 fixture）

两份**真件**日志（只读取得，⛔ 未在远端写任何字节）：

| 档 | 机器 / 路径 | 字节 | sha256 | `grep -c global_step` | `grep -c '"step":'` |
|----|-------------|------|--------|----------------------|---------------------|
| M007 | 远端机 `…/graspo-baseline-r1-6189bd8/runs/M007/stdout.log` | 13 563 677 | `b3ae64af…` | **0** | 400 |
| M008 | 远端机 `<主机侧数据根>/runs/M008/stdout.log` | 6 784 734 | `0c58e67b…` | **0** | 200 |

fixture 是这两份日志的**逐字节摘录**（`.txt` 而非 `.log`：仓里 `.gitignore` 忽略 `*.log`，
这是受控 fixture，⛔ 不用 `git add -f` 绕过忽略规则）：

* `fixtures/m007_native_stdout_excerpt.txt` —— 原件物理行号 `416 / 1212 / 159224`
  （`"step": 1 / 3 / 400`；末行 = 训练末态 400 步）；
* `fixtures/m008_native_stdout_excerpt.txt` —— 原件物理行号 `79246 / 79644`
  （`"step": 199 / 200`；dp2 档 `"optimizer_steps": 4`）。

## 红 → 绿

老口径只认三族：`global_step/max_steps`、tqdm `Train:` 条、`trainer_state.json` 的
`global_step`。native 后端**逐字**打的是 `{"event": "sft_step", "step": N, …}`
（`flow/backend/native/trainer/{optimize,sft_trainer,trainer,rollout}.py`）⇒ 上面两份日志里
**一条读数都认不出** ⇒ `steps_actual=None` ⇒ 回填器 fail-closed、训练已成功（rc=0、跑满 4 epoch）
的档进不了表。`scripts/collect_results.py` 的 `_GLOBAL_STEP` 是**同源缺陷的第二份副本**
（18 条红），本包把它删掉、两边共用 `scripts/step_readings.py`（§1.4 单一真相源）。

## 本文件钉住什么

1. 真件摘录确实只有 `"step"`、老口径**必然取不到**（红是"旧口径"的红，不是 fixture 不真实）；
2. 新口径在真件上取到**正确总值**（400 / 200）并给出**可 sed 复核**的行号；
3. **ms-swift 四种实盘形态逐字节不变**（golden = 直接跑 `HEAD 924f01a` 那份实现，逐字比对
   `(steps, why)` 与候选列表 —— ⛔ 不是我手抄一份"应该长这样"）；
4. 多源并存取**最大** + provenance；一个读数都没有 ⇒ fail-closed（⛔ 不猜、不 0 兜底、
   ⛔ 不把"末条"当总步数）；像步数但不是步数的键（`max_steps` / `save_step` / `first_step`）
   ⛔ 不得被读成已跑步数；
5. `collect_results.py` 与主解析器**同值**（同一份日志两个消费者不许给出两个答案）；
6. 分片检查点目录（`rank_*_tp_*_pp_*.pt`）**不影响**步数解析（N3c 是另一件事）。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
import collect_results as collector  # noqa: E402
import make_result_json as m  # noqa: E402
import step_readings as sr  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
_M007 = FIXTURES / "m007_native_stdout_excerpt.txt"
_M008 = FIXTURES / "m008_native_stdout_excerpt.txt"

#: 修复前（起点 HEAD `924f01a`）的步数口径 —— 逐字取自 HEAD 的两处实现，**只用于负向断言**：
#: ① `make_result_json.py::GLOBAL_STEP_RE`；② `collect_results.py::_GLOBAL_STEP`（同源第二副本）。
_OLD_GLOBAL_STEP_RE = r"['\"]?global_step['\"]?\s*[:=]\s*['\"]?(\d+)"
_OLD_TQDM_RE = r"(?:^|[\r\n])\s*(?:Train|Training):[^\r\n]*?\|\s*(\d+)\s*/\s*(\d+)\s*\["
_HEAD_REV = "924f01a"


def _load_head_module(tmp_path: pathlib.Path):
    """把**起点 HEAD** 的 `scripts/make_result_json.py` 原样取出加载 ⇒ 当 golden 基准。

    ★ 为什么真去取 HEAD 那份（而不是在测试里手写"期望值"）：本包的要求是
      "ms-swift 四种形态**逐字节不变**" —— 只有拿旧实现跑同一份输入再逐字比对
      `(steps, why)` 与候选列表，才叫"不变"；手抄期望值就只是我自己给自己作证。
    """
    src = subprocess.run(["git", "show", f"{_HEAD_REV}:scripts/make_result_json.py"],
                         cwd=str(_ROOT), capture_output=True, text=True, check=True).stdout
    path = tmp_path / "head_make_result_json.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("head_make_result_json_for_n3b", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_dir(tmp_path: pathlib.Path, name: str, log_bytes: bytes) -> pathlib.Path:
    run = tmp_path / name
    run.mkdir()
    (run / "stdout.log").write_bytes(log_bytes)
    return run


# ── ① 红：真件里 old 口径一条都认不出 ────────────────────────────────────────
@pytest.mark.parametrize("fixture", [_M007, _M008], ids=["M007", "M008"])
def test_real_native_log_has_no_global_step_at_all(fixture: pathlib.Path):
    """★红：真件里 **0 处** `global_step`，老口径（正则 + tqdm）**必然取不到**步数。"""
    raw = fixture.read_bytes().decode("utf-8", "replace")
    assert "global_step" not in raw, "摘录里出现了 global_step ⇒ 已不是 native 实盘形态"
    assert '"step"' in raw, "摘录里没有 native 的 \"step\" 键"
    assert re.search(_OLD_GLOBAL_STEP_RE, raw) is None, "旧 global_step 口径居然认出了？"
    assert re.search(_OLD_TQDM_RE, raw) is None, "旧 tqdm 口径居然认出了？"
    # 旧 collect_results 的取值方式（findall + 末条）在这份文本上就是空集 ⇒ steps 取不到
    assert re.findall(_OLD_GLOBAL_STEP_RE, raw) == []


# ── ② 绿：真件摘录取到正确总值 + 可复核的行号 ──────────────────────────────
@pytest.mark.parametrize("fixture, expect, line", [
    (_M007, 400, 3),    # 摘录第 3 行 = 原件 stdout.log:159224（训练末态 400 步）
    (_M008, 200, 2),    # 摘录第 2 行 = 原件 stdout.log:79644（训练末态 200 步）
], ids=["M007", "M008"])
def test_real_native_log_parses_total_steps_with_provenance(fixture, expect, line):
    steps, why = m.parse_steps(fixture)
    assert steps == expect, why
    assert f"{fixture.name}:{line}" in why, why
    assert "native_step" in why, why


def test_native_step_excerpt_keeps_the_real_keys(tmp_path):
    """摘录确实是 native `sft_step` 事件（不是被人改干净的合成件）。"""
    raw = _M007.read_bytes().decode("utf-8", "replace")
    first = json.loads(raw.splitlines()[0])
    assert first["event"] == "sft_step"
    assert first["step"] == 1 and first["batch"] == 0 and first["epoch"] == 0
    assert "optimizer_steps" in first and "skipped_nonfinite" in first


# ── ③ ms-swift 四种实盘形态：逐字节不变（golden = HEAD 那份实现） ────────────
_MS_SWIFT_FORMS = [
    "{\"loss\": \"0\", \"global_step/max_steps\": \"200/200\", \"epoch\": \"1\"}",
    "{\"global_step/max_steps\" : \"200/200\"}",
    "{\"global_step/max_steps\": 200/200}",
    "{\"global_step\": \"200\"}",
    "{\"global_step\": 200}",
    "\rTrain: 100%|██████████| 200/200 [00:10<00:00,  4.95s/it]",
    "\rTrain:  50%|█████     | 100/200 [00:05<00:05, 4.95s/it]",
]


@pytest.mark.parametrize("line", _MS_SWIFT_FORMS)
def test_ms_swift_forms_are_byte_identical_to_head(tmp_path, line):
    """★golden：ms-swift 七种既有写法在**新旧实现**上得到逐字节相同的 `(steps, why)`。"""
    head = _load_head_module(tmp_path)
    log = tmp_path / "stdout.log"
    log.write_bytes(line.encode("utf-8"))
    assert m.parse_steps(log) == head.parse_steps(log)
    assert sr.scan_text(line, "stdout.log") == head._scan_steps_text(line, "stdout.log")


def test_m016_real_excerpt_is_unchanged_golden(tmp_path):
    """★golden：M016 真件摘录（D2 的 fixture）读数与 provenance **逐字不变**。"""
    head = _load_head_module(tmp_path)
    fixture = FIXTURES / "m016_stdout_excerpt.txt"
    assert m.parse_steps(fixture) == head.parse_steps(fixture)
    steps, why = m.parse_steps(fixture)
    assert steps == 6400 and fixture.name + ":3" in why, why


def test_public_regex_names_still_exported():
    """审计脚本/文档会按名引用这三条正则 ⇒ 收敛真相源后**名字仍在**（行为不变）。"""
    assert m.GLOBAL_STEP_RE is sr.GLOBAL_STEP_RE
    assert m.GLOBAL_STEP_SLASH_RE is sr.GLOBAL_STEP_SLASH_RE
    assert m.TQDM_STEP_RE is sr.TQDM_STEP_RE
    assert m._STEP_KIND_RANK is sr.STEP_KIND_RANK


# ── ④ 多源取最大 + provenance ───────────────────────────────────────────────
def test_multi_source_takes_max_and_records_every_candidate(tmp_path):
    """stdout（native 200 步）∪ 两份 trainer_state（300 / 400）⇒ 取 **400** 并留坐标。"""
    run = _run_dir(tmp_path, "M008", _M008.read_bytes())
    for step in (300, 400):
        d = run / f"M008/step_{step}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "epoch": step / 100}), encoding="utf-8")

    steps, why, prov = m.parse_steps_evidence(run)
    assert steps == 400, why
    assert len(prov) == 3, prov                     # 2×trainer_state + 1×native stdout
    assert prov[-1]["value"] == 400 and prov[-1]["kind"] == "trainer_state.global_step"
    assert prov[-1]["source"].endswith("step_400/trainer_state.json"), prov[-1]
    assert "step_400/trainer_state.json" in why, why


def test_same_value_prefers_the_more_authoritative_form(tmp_path):
    """数值相同时 provenance 记**更权威**的形态（显式 dict > tqdm），⛔ 不随机挑一条。"""
    run = _run_dir(tmp_path, "M016", (
        "{'global_step/max_steps': '6400/6400'}\n"
        "\rTrain: 100%|██████████| 6400/6400 [00:10<00:00, 4.95s/it]\n"
    ).encode("utf-8"))
    steps, why, prov = m.parse_steps_evidence(run)
    assert steps == 6400
    assert prov[-1]["kind"] == "global_step/max_steps", prov
    assert "global_step/max_steps" in why


# ── ⑤ fail-closed：一个读数都没有 ⇒ None（⛔ 不猜 / 不 0 兜底 / 不取末条） ────
def test_no_step_reading_is_fail_closed(tmp_path):
    """只有 batch / epoch / optimizer_steps、没有 step ⇒ **不可回填**（⛔ 用 batch+1 猜步数）。"""
    log = tmp_path / "stdout.log"
    log.write_bytes(
        b'{"event": "sft_step", "epoch": 3, "batch": 99, "loss": 0.05,'
        b' "grad_norm": 0.2, "optimizer_steps": 2, "skipped_nonfinite": 0}\n')
    steps, why = m.parse_steps(log)
    assert steps is None, why
    assert "没有步数读数" in why, why


@pytest.mark.parametrize("line", [
    '{"max_steps": 800, "save_step": 50}',       # 计划值/存盘周期 —— 不是已跑步数
    '{"first_step": 1, "optimizer_steps": 4}',   # 首步/累计优化步 —— 不是步号
    '{"global_optimizer_steps_sum": 400}',       # 旁路累计量 —— 不是步号
    '{"event": "sft_step", "epoch_progress": 0.5}',
])
def test_step_like_keys_are_not_read_as_step_count(tmp_path, line):
    """★防呆：带 `step` 字样但语义不同的键 ⛔ 不得被读成已跑步数（会静默污染 steps_actual）。"""
    log = tmp_path / "stdout.log"
    log.write_bytes((line + "\n").encode("utf-8"))
    assert m.parse_steps(log)[0] is None, line


def test_missing_log_and_empty_log_are_none(tmp_path):
    assert m.parse_steps(tmp_path / "nope.log")[0] is None
    empty = tmp_path / "stdout.log"
    empty.write_text("[INFO] begin\nnothing here\n", encoding="utf-8")
    assert m.parse_steps(empty)[0] is None


# ── ⑥ 两个消费者同值（§1.4：同一份日志不许给出两个答案） ────────────────────
@pytest.mark.parametrize("fixture, expect", [
    (_M007, 400), (_M008, 200),
], ids=["M007", "M008"])
def test_collect_results_agrees_with_main_parser_on_real_logs(fixture, expect):
    text = fixture.read_bytes().decode("utf-8", "replace")
    series = collector.extract_steps_and_series([], text)
    assert series.steps == expect, series.notes
    assert m.parse_steps(fixture)[0] == series.steps
    assert any("native_step" in n for n in series.notes), series.notes


def test_collect_results_agrees_on_ms_swift_forms(tmp_path):
    text = ("{'global_step/max_steps': '6400/6400'}\n"
            "\rTrain: 100%|██████████| 6400/6400 [00:10<00:00, 4.95s/it]\n")
    log = tmp_path / "stdout.log"
    log.write_bytes(text.encode("utf-8"))
    assert collector.extract_steps_and_series([], text).steps == m.parse_steps(log)[0] == 6400


def test_collect_results_has_no_second_copy_of_the_regex():
    """★单一真相源：`collect_results.py` 里**不再有**自己那份 `global_step` 正则。"""
    assert not hasattr(collector, "_GLOBAL_STEP")
    assert collector.step_readings is sr


# ── ⑦ 分片检查点目录不影响步数解析（N3c 是另一件事） ──────────────────────
def test_sharded_ckpt_dirs_do_not_disturb_step_parsing(tmp_path):
    """native tp/pp 档的 `rank_*_tp_*_pp_*.pt` 分片目录存在时，步数照旧取 200。"""
    run = _run_dir(tmp_path, "M008", _M008.read_bytes())
    # 文件名逐字照 M007/M008 实盘的 native 分片布局（tp=2 pp=1）
    shards = ("rank_00000_tp_00_pp_00.pt", "rank_00001_tp_01_pp_00.pt")
    for sub in ("M008/step_201", "M008/final"):
        d = run / sub
        d.mkdir(parents=True)
        for name in shards:
            (d / name).write_bytes(b"")
        (d / "manifest.json").write_text(json.dumps({"format": "native-sharded"}), encoding="utf-8")

    steps, why, prov = m.parse_steps_evidence(run)
    assert steps == 200, why
    assert len(prov) == 1 and prov[0]["kind"] == "native_step", prov
    assert all(not str(c["source"]).endswith(".pt") for c in prov), prov
