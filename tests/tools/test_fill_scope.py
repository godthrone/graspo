"""填表写范围硬闸门（§22 · 主席令「填写报告的脚本只能填写第四章表格里面的实验内容数据」）。

为什么有这件（宪法 §15.1「能失败的断言才叫断言」）：
  主席 2026-10-25 的定稿冻结令是**两条边界**：① 报告不许改；② 填表脚本只许动第 4 章
  「实验数据」列的**单元格文本**。此前报告链只有 `verify_matrix`（比对"当前模板+当前数据"
  的渲染一致性）——它**看不见**「模板/章节文字被悄悄改过」这类越界（模板改了，重渲染自然一致）。
  `check_fill_scope.py` 把判据换成「**填表后副本 vs 冻结基准件**」，本节即它的能红证明。

覆盖（每一条都成对：该过的过、该红的红）：
  ① 阳性对照 1：改一个结果单元格文本 ⇒ rc=0；
  ② 阳性对照 2：改第 3 章正文 ⇒ rc=2；
  ③ 阳性对照 3：改表格表头（文字 / 属性各一版）⇒ rc=2；
  ④ 阳性对照 4：在结果单元格里塞 `<script>` ⇒ rc=2；
  ⑤ 阳性对照 5：改「备注列之外」的 `<td class="col-start">` 结构（属性 / 标签各一版）⇒ rc=2；
  ⑥ 正常回填一次（6 结果列 + status 文本/配色）⇒ rc=0（证明闸门不是"一律红"）；
  ⑦ 回填链事务：`apply_with_scope_gate` 越界 ⇒ **回滚**、rc=1、两件与写前逐字节一致；
     合法件 ⇒ rc=0、两件落盘（⑧ 与 ⑦ 同一函数，即 `backfill_matrix.py` 落盘走的那条路径）；
  ⑨ 写前闸门：基准两件（人读件/数据件，含软链/相对写法）⇒ 拒绝；按轮副本 ⇒ 放行。

纯标准库（不 import torch/pyyaml/项目包），按文件路径加载被测脚本原文。
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_HTML = REPO_ROOT / "docs" / "GRASPO训练框架功能测试报告.html"
BASE_DATA = REPO_ROOT / "docs" / "capability-matrix.data.json"
GATE = REPO_ROOT / "tools" / "matrix" / "check_fill_scope.py"


def _load(rel: str, name: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PATHS = _load("tools/matrix/paths.py", "paths_for_fill_scope")
GATE_MOD = _load("tools/matrix/check_fill_scope.py", "check_fill_scope_under_test")


# ---- 夹具（全部在 tmp_path 内，⛔ 不写仓内任何文件）---------------------------
@pytest.fixture()
def base_text() -> str:
    return BASE_HTML.read_text(encoding="utf-8")


def _edit(text: str, old: str, new: str, n: int = 1) -> str:
    assert text.count(old) >= 1, f"夹具锚点未命中（报告已改 ⇒ 本测试需同步）：{old[:60]!r}"
    return text.replace(old, new, n)


PEAK = '<td class="g-data col-start" data-col="peak_mem">—</td>'
PEAK_FILLED = '<td class="g-data col-start" data-col="peak_mem">63.4</td>'


def _run_gate(base: Path, filled: Path, data: Path = BASE_DATA) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), "--base", str(base), "--filled", str(filled), "--data", str(data)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), check=False,
    )


def _materialise(tmp_path: Path, name: str, text: str, base: Path) -> Path:
    base.write_text(BASE_HTML.read_text(encoding="utf-8"), encoding="utf-8")
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---- ①–⑥ 五条阳性对照 + 正常回填一次 -----------------------------------------
def test_result_cell_text_change_is_allowed(tmp_path, base_text):
    """① 改一个**结果单元格**的文本 ⇒ rc=0（这是唯一被允许的写面）。"""
    base = tmp_path / "base.html"
    filled = _materialise(tmp_path, "filled.html", _edit(base_text, PEAK, PEAK_FILLED), base)
    r = _run_gate(base, filled)
    assert r.returncode == 0, f"改结果单元格文本竟被判违规：\n{r.stdout}\n{r.stderr}"
    assert "SCOPE PASS" in r.stdout


def test_chapter_text_change_is_rejected(tmp_path, base_text):
    """② 改第 3 章正文 ⇒ rc=2（⛔ 章节文字一字不许动）。"""
    base = tmp_path / "base.html"
    anchor = "下面这些参数<b>在每个后端的所有档位上都取同一个值</b>"
    filled = _materialise(tmp_path, "filled.html", _edit(base_text, anchor, anchor + "（并另加一句）"), base)
    r = _run_gate(base, filled)
    assert r.returncode == 2, f"改章节文字竟被放行：\n{r.stdout}"
    assert "SCOPE VIOLATION" in r.stdout and "矩阵表之前" in r.stdout


def test_thead_text_change_is_rejected(tmp_path, base_text):
    """③a 改表头**文字** ⇒ rc=2。"""
    base = tmp_path / "base.html"
    th = '<th class="g-data" data-col="steps_actual">实跑步数</th>'
    filled = _materialise(tmp_path, "filled.html",
                          _edit(base_text, th, th.replace("实跑步数", "实跑步数（改）")), base)
    r = _run_gate(base, filled)
    assert r.returncode == 2, f"改表头文字竟被放行：\n{r.stdout}"


def test_thead_attr_change_is_rejected(tmp_path, base_text):
    """③b 改表头**属性**（colspan 6→7）⇒ rc=2。"""
    base = tmp_path / "base.html"
    filled = _materialise(
        tmp_path, "filled.html",
        _edit(base_text, '<th class="g-data" colspan="6">实验数据</th>',
              '<th class="g-data" colspan="7">实验数据</th>'), base)
    r = _run_gate(base, filled)
    assert r.returncode == 2, f"改表头属性竟被放行：\n{r.stdout}"


def test_script_inside_result_cell_is_rejected(tmp_path, base_text):
    """④ 在结果单元格里塞 `<script>` ⇒ rc=2（只许纯文本）。"""
    base = tmp_path / "base.html"
    filled = _materialise(
        tmp_path, "filled.html",
        _edit(base_text, PEAK,
              '<td class="g-data col-start" data-col="peak_mem"><script>alert(1)</script></td>'), base)
    r = _run_gate(base, filled)
    assert r.returncode == 2, f"单元格内嵌脚本竟被放行：\n{r.stdout}"
    assert "标记" in r.stdout


@pytest.mark.parametrize("mutated", [
    # ⑤a 属性结构：给「备注列之外」的 `<td class="col-start">` 加 style
    '<td class="g-data col-start" style="color:red" data-col="peak_mem">—</td>',
    # ⑤b 标签结构：在该单元格内部再包一层 <span>
    '<td class="g-data col-start" data-col="peak_mem"><span>—</span></td>',
    # ⑤c 拆掉 col-start 类名本身
    '<td class="g-data" data-col="peak_mem">—</td>',
])
def test_col_start_structure_change_is_rejected(tmp_path, base_text, mutated):
    """⑤ 改「备注列之外」的 `<td class="col-start">` 结构 ⇒ rc=2。"""
    base = tmp_path / "base.html"
    filled = _materialise(tmp_path, "filled.html", _edit(base_text, PEAK, mutated), base)
    r = _run_gate(base, filled)
    assert r.returncode == 2, f"单元格结构被改竟放行：{mutated}\n{r.stdout}"


def test_normal_fill_once_is_allowed(tmp_path, base_text):
    """⑥ 正常回填一次（6 个结果列 + status 文本/配色）⇒ rc=0 —— 闸门不是"一律红"。"""
    text = _edit(base_text, PEAK, PEAK_FILLED)
    for col, val in (("steps_actual", "400 步"), ("elapsed", "总时长 3600 s（约 1.00 h）"),
                     ("throughput", "0.11 步/s/卡"), ("effect", "SFT Δ +12.3 pp"), ("note", "正常回填")):
        text = _edit(text, f'<td class="g-data" data-col="{col}">—</td>',
                     f'<td class="g-data" data-col="{col}">{val}</td>')
    text = _edit(text, '<td class="g-status col-start st-pending" data-col="status">待测试</td>',
                 '<td class="g-status col-start st-pass" data-col="status">通过</td>')
    base = tmp_path / "base.html"
    filled = _materialise(tmp_path, "filled.html", text, base)
    r = _run_gate(base, filled)
    assert r.returncode == 0, f"一次正常回填被判违规：\n{r.stdout}\n{r.stderr}"
    assert "内容变化     7 个单元格" in r.stdout


def test_status_prose_outside_vocabulary_is_rejected(tmp_path, base_text):
    """⑥b 负例：把 status 写成词表外的散文 ⇒ rc=2（status 只认 `meta.status_vocabulary`）。"""
    base = tmp_path / "base.html"
    filled = _materialise(
        tmp_path, "filled.html",
        _edit(base_text, 'data-col="status">待测试</td>', 'data-col="status">基本可用，问题不大</td>'), base)
    r = _run_gate(base, filled)
    assert r.returncode == 2, f"status 写散文竟放行：\n{r.stdout}"


# ---- ⑦ 回填链事务：越界 ⇒ 回滚（不留半写） -----------------------------------
@pytest.fixture()
def gate_chain(tmp_path):
    """`backfill_matrix.apply_with_scope_gate` + 一个沙箱副本对（⛔ 不碰 docs/）。"""
    bf = _load("tools/matrix/backfill_matrix.py", "backfill_matrix_for_fill_scope")
    sandbox = tmp_path / "sandbox"
    (sandbox / "tmp").mkdir(parents=True)
    data_out = sandbox / "data.json"
    html_out = sandbox / "out.html"
    shutil.copyfile(BASE_DATA, data_out)
    html_out.write_text("写前状态（上一版副本）\n", encoding="utf-8")
    return bf, sandbox, data_out, html_out


def test_backfill_gate_rolls_back_on_out_of_scope(gate_chain, base_text, tmp_path):
    """⑦a 落盘后闸门不过 ⇒ **回滚**本次写入、rc≠0、两件与写前逐字节一致。"""
    bf, sandbox, data_out, html_out = gate_chain
    prev_data, prev_html = data_out.read_bytes(), html_out.read_bytes()
    bad = sandbox / "tmp" / "bad.html"
    bad.write_text(_edit(base_text, "（绿色）：判据全部满足。", "（绿）：判据全部满足。"), encoding="utf-8")

    rc, out = bf.apply_with_scope_gate(tmp_data=data_out, tmp_html=bad,
                                       data_out=data_out, html_out=html_out)

    assert rc != 0, f"越界渲染件竟落盘成功：\n{out}"
    assert "回滚" in out and "SCOPE VIOLATION" in out
    assert html_out.read_bytes() == prev_html, "回滚后 html 未恢复写前状态（半写！）"
    assert data_out.read_bytes() == prev_data, "回滚后数据件未恢复写前状态（半写！）"


def test_backfill_gate_accepts_legal_fill(gate_chain, base_text):
    """⑦b 合法填表件 ⇒ rc=0 且两件落盘（同一函数，即 `backfill_matrix.py` 的落盘路径）。"""
    bf, sandbox, data_out, html_out = gate_chain
    good = sandbox / "tmp" / "good.html"
    text = _edit(base_text, PEAK, PEAK_FILLED)
    text = _edit(text, '<td class="g-status col-start st-pending" data-col="status">待测试</td>',
                 '<td class="g-status col-start st-pass" data-col="status">通过</td>')
    good.write_text(text, encoding="utf-8")

    rc, out = bf.apply_with_scope_gate(tmp_data=data_out, tmp_html=good,
                                       data_out=data_out, html_out=html_out)

    assert rc == 0, f"合法填表件竟被回滚：\n{out}"
    assert html_out.read_bytes() == good.read_bytes(), "合法件未被落盘"


# ---- ⑧ 冻结断言 + 写前闸门 ---------------------------------------------------
def test_baseline_frozen_three_assertions():
    """⑧ 基准件三断言（sha + 字节 + 权限 444）必须全绿 —— 任一不符即红（§22）。"""
    bad = PATHS.check_baseline_frozen()
    assert bad == [], f"报告基准件冻结被破坏：{bad}"


def test_write_guard_refuses_baseline_allows_round_copy(tmp_path):
    """⑧b 写前闸门：基准两件（含相对/软链写法）拒写；按轮副本放行；`--allow-baseline-write` 才允许。"""
    for p in (BASE_HTML, BASE_DATA, Path("docs/GRASPO训练框架功能测试报告.html"),
              Path("docs/../docs/capability-matrix.data.json")):
        with pytest.raises(SystemExit):
            PATHS.assert_not_baseline_write(p)

    link = tmp_path / "baseline-link.html"
    os.symlink(BASE_HTML, link)
    with pytest.raises(SystemExit):
        PATHS.assert_not_baseline_write(link)

    # 按轮副本 / 任意其它路径 ⇒ 放行
    assert PATHS.assert_not_baseline_write(PATHS.round_html("20261025")) == PATHS.round_html("20261025")
    # 显式授权（唯一通道 = render_matrix.py --allow-baseline-write，需主席批准）
    assert PATHS.assert_not_baseline_write(BASE_HTML, allow_baseline_write=True) == BASE_HTML


def test_gate_whitelist_is_single_source_of_truth():
    """⑨ 白名单来自数据件 `data_keys`（⛔ 不写死第二份列名）+ 机算结论列 status。"""
    cols = GATE_MOD.load_result_columns(BASE_DATA)
    data_keys = json.loads(BASE_DATA.read_text(encoding="utf-8"))["data_keys"]
    assert cols == data_keys
    assert GATE_MOD.MACHINE_STATUS_COL == "status"
    vocab = GATE_MOD.load_status_vocabulary(BASE_DATA)
    assert vocab == frozenset(json.loads(BASE_DATA.read_text(encoding="utf-8"))
                              ["meta"]["status_vocabulary"])
    assert vocab, "状态词表为空 ⇒ status 文本变化会被一律判违规（fail-closed）"
