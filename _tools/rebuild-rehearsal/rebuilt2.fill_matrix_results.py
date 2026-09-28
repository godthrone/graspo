#!/usr/bin/env python3
"""回填 ``docs/capability-matrix.html`` 的「结果列组」，实现 §1「连续 3 次全部成功」四态聚合。

本脚本只做**回填与汇总**，不跑训练、不连远端、不做判定之外的解释。
结果列组（``data-col``，共 9 列）::

    peak_mem / steps_actual / ctx_max / status / runs3 / date / run_root / failure / note

结果列之外的任何一格**一律不许改**；每次写入前后都会做「掩码逐格 diff」断言
（见 :func:`assert_result_only_change`）：把结果列的整个 ``<td>…</td>` 元素替换成
占位符后，改动前后的 HTML 必须**逐字节相同**。

判据与四态聚合（HTML §1 / §2 / §4 执行解读）
--------------------------------------------
对**每一个跑次**先按 §1 的 ①②③ 各判一次，再聚合出四态：

* ① 跑完：``exit_code == 0`` 且未超时且 ``实际步数 >= 计划步数``。
* ② 产物落盘且可重载：checkpoint + 配置快照 + 训练日志都在，且 checkpoint
  **内容级**可重载（张量数 > 0、关键张量集齐全）。
  **读数口径按档分叉（2026-09-23）**：full-param 档以 ``torch_probe.json`` 的
  ``all_ok``/逐层审计为准；LoRA 档的探针"语言层逐层覆盖"字段对视觉塔 LoRA
  **不适用**（已知误报），改核 ``state_dict_kind``/``layout_status``/张量数与
  manifest 的 LoRA 目标数一致/有限性，见
  ``docs/capability-matrix.html`` §3「判据②读数口径（2026-09-23）」。
* ③ 权重真变、数值健康：终态权重与基座有实质差异、逐步优化器都推进
  （``global_optimizer_steps_sum > 0``）、loss/grad_norm 无 NaN/Inf、
  跨**全部 rank** 的非有限梯度跳过数为 0。

任一跑次 ①②③ 中有一条被事实否定 ⇒ 该跑次「失败」；三条全为真 ⇒ 「成功」；
任何一条读数为空（``None``）⇒ 该跑次「不可判定」（记 ``—``）。

四态聚合（**单一真相源** :func:`aggregate_state`）：:

    条件不合逻辑  显式标 impossible（``*.not_applicable.md`` 或读数里的 state/impossible）
    失败          第 1 次跑次「失败」（§4 执行解读第 4 条：第 1 次失败直接记失败）
    通过          第 1/2/3 次跑次全部「成功」
    未完成测试    其余一切：不足 3 次、第 1 次不可判定、第 2/3 次失败（不稳定）、
                  或任一必需读数缺失

字段来源（第一轮跑批，产根布局见 ``tests/e2e/run_matrix54.sh``）
----------------------------------------------------------------
每档跑次根目录 ``<RUN_ROOT>/<T###>/``：

============================  ================================================================
 结果列                        读数来源
============================  ================================================================
 计划步数                      HTML 自己的 ``planned_steps`` 列（HTML = 唯一对齐点）
 实际步数                      rank0 ``rank_metrics.rank_00000.jsonl`` 的
                               ``metrics.global_optimizer_steps_sum`` 最大值；
                               回落 ``logging.jsonl`` 末行 ``global_step`` / stdout ``global_step``
 每卡峰值显存                  native：rank0 rank_metrics ``metrics.memory.max_allocated_mib``；
                               ms-swift：``metrics.memory.max_memory_reserved_mib``（同 collector
                               的二分口径；宿主采样峰值只另存，不进本列）
 最大可行上下文                读数 ``max_context``（runner 未落该值的结构化产物 ⇒ 由跑批驱动/
                               collector 注入，缺失即 —）
 四态                          :func:`aggregate_state`
 三次跑次结果                 每跑次 :func:`judge_run`
 实测日期                      读数 ``date``（跑批驱动注入；缺失即 —）
 跑次根目录                    各跑次的 ``<RUN_ROOT>/<T###>`` 实际路径
 失败归因                      读数 ``failure_class``（我方实现 / 上游依赖 / 资源不足）
 备注                          聚合理由与读数缺失说明
============================  ================================================================

用法::

    # 影本自测（**不要**直接对 docs/capability-matrix.html 跑）
    cp docs/capability-matrix.html /tmp/shadow.html
    .venv/bin/python tests/e2e/fill_matrix_results.py --html /tmp/shadow.html \\
        --readings runs/synthetic-readings.json --out-jsonl /tmp/results.jsonl

    # 第一轮真跑批（3 棵产根 = 第 1/2/3 次；顺序即跑次号）
    .venv/bin/python tests/e2e/fill_matrix_results.py \\
        --run-root /path/r1 --run-root /path/r2 --run-root /path/r3 \\
        --out-jsonl /path/r1/matrix-results.jsonl

可重入：每格按读数**整格覆盖**，重跑同输入产出逐字节相同的 HTML；同一档多轮读数按
``attempt`` 覆盖，不追加、不产生乱码。
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = REPO_ROOT / "docs" / "capability-matrix.html"
DEFAULT_CONFIG_DIR = REPO_ROOT / "samples" / "configs" / "matrix54-v2"

#: 结果列组（顺序与 ``colgroup`` 一致；除此之外的列禁止改动）。
RESULT_COLUMNS = [
    "peak_mem", "steps_actual", "ctx_max", "status", "runs3",
    "date", "run_root", "failure", "note",
]

#: HTML §2 四态。
STATES = ["通过", "失败", "条件不合逻辑", "未完成测试"]
#: 状态着色 class（与 HTML ``<style>`` 里的定义同名）。
STATE_CLASS = {
    "通过": "st-pass",
    "失败": "st-fail",
    "条件不合逻辑": "st-illegal",
    "未完成测试": "st-pending",
}
#: 文字几态（写入 status 单元格）。
STATE_PLACEHOLDER = "—"

#: 失败归因（HTML §2 / §3）。
FAILURE_CLASSES = ["我方实现", "上游依赖", "资源不足"]

#: 跑次标记。
RUN_MARKS = ["①", "②", "③"]
VERDICT_SUCCESS = "成功"
VERDICT_FAIL = "失败"
VERDICT_DASH = "—"

#: 结果列的默认占位（与现表一致，保证未填档零改动）。
PLACEHOLDER = "—"
DASH = "—"

_TIMEOUT_RE = re.compile(r"⏰|timeout: sending signal")

#: 判据②读数口径（2026-09-23）——LoRA 权重承载键（探针 ``checked[].state_dict_kind``）。
_LORA_STATE_DICT_KIND = "lora_state_dict"
#: 探针布局已确证：``torch.load`` 成功且内容级校验跑完。
_LAYOUT_STATUS_VERIFIED = "verified"
#: LoRA 档「探针层覆盖字段不适用」的解释性残差文案（探针缺陷本轮不修，仅按口径解读）。
LORA_LAYER_COVERAGE_RESIDUAL = (
    "视觉塔 LoRA 的探针层级覆盖口径不适用（**已知探针口径缺陷，待下轮修**；"
    "本轮按 docs/capability-matrix.html §3「判据②读数口径（2026-09-23）」的修正口径判定 ②，"
    "**探针 all_ok=false 属已知误报，不代表 checkpoint 有问题**）"
)


#: 逐步权威指标 phase 名（**唯一真相源** ``src/graspo/core/result_judge.py:78-86``
#: ``STEP_METRICS_PHASES``；采集侧同口径实现见 ``scripts/collect_results.py:539``）。
STEP_METRICS_PHASES = frozenset(
    {
        "pipeline_sft_train_batch_after",
        "sft_train_batch_after",
        "pipeline_train_batch_after",
        "train_batch_after",
    }
)

#: §7 峰值显存口径标签（**唯一真相源** ``src/graspo/core/result_judge.py:231-233``
#: ``PEAK_MEMORY_CALIBER_BY_BACKEND``）。native ⇒ rank0 allocator ``max_allocated``；
#: ms-swift ⇒ 自报 ``memory(GiB)``（= ``max_memory_reserved``，保守上界）。
NATIVE_ALLOCATOR_CALIBER = "native_allocator_max_allocated"
MSSWIFT_RESERVED_CALIBER = "msswift_rank0_max_reserved"
#: ms-swift 自报显存键（``collect_results.py:2058-2080`` 的口径来源）。
MSSWIFT_MEMORY_KEY = "memory(GiB)"
_MSSWIFT_STDOUT_MEMORY_RE = re.compile(
    r"memory\(GiB\)['\"]?\s*:\s*['\"]?(-?\d+(?:\.\d+)?)"
)

#: A7 残差（ms-swift 不上报非有限跳过计数）——**只登记，绝不据此放宽判据③**。
MSSWIFT_A7_RESIDUAL = (
    "ms-swift 后端不上报「跨全部 rank 非有限跳过数」计数（swift/trainers/mixin.py:744-779 梯度 NaN 时"
    "置 grad=None 仍照常 step、只判 isnan 不判 isinf、无计数；且 mixin.py:1078-1081 在 grad_norm 为 None 时"
    "根本不写该键），logging 又被 logging_nan_inf_filter（transformers/trainer.py:1746-1752 把 NaN/Inf 的 "
    "step loss 替换为历史均值 ⇒ loss 结构性有限）遮蔽，且 36/36 档实测 logging_steps=5（采样非逐步）"
    "⇒ 判据③不可判定（A7 口径不可测，**非失败**；不得改成阻断外的诊断，无读数即维持未完成）"
)

#: A4 残差（权重指纹未标定）——默认只落结构化字段，``--a4-residual-note`` 才写进备注。
A4_RESIDUAL = (
    "A4 权重指纹（逐字节读权重，实现见 scripts/collect_results.py:862-935）未标定 "
    "⇒ 判据③的「权重真变」证据不完整（登记用，不改变本档状态）"
)

#: 失败归因的**事实特征串**（HTML §2 三类）。只做事实匹配；匹配不到一律 ``None``（不猜）。
#: 顺序即优先级：资源不足 → 我方实现 → 上游依赖。
FAILURE_CLASS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("资源不足", re.compile(
        r"(?i)(CUDA out of memory|OutOfMemoryError|no space left on device|STOPPED_DISK_FLOOR)")),
    ("我方实现", re.compile(
        r"(?i)(native PP rollout is not end-to-end verified|is refused by default|"
        r"allow_unverified_pp_rollout)")),
    ("上游依赖", re.compile(
        r"(?i)(ProcessGroupNCCL|Watchdog caught collective|NCCL operations have failed|"
        r"nccl_collective_timeout|CUDA error|illegal memory access|ChildFailedError)")),
]


# ---------------------------------------------------------------------------
# HTML 解析：沿用 tests/e2e/verify_matrix_html.py::MatrixTableParser 的解析方式
# （HTMLParser + 只认 table.matrix 的 tbody + data-col 单元格），另外记录字符偏移，
# 以便做「只改结果列」的外科替换与掩码断言。
# ---------------------------------------------------------------------------
def _line_starts(text: str) -> list[int]:
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


class MatrixCellParser(HTMLParser):
    """解析 ``table.matrix`` tbody 的 ``data-col`` 单元格，并记录字符偏移。

    输出 ``rows``：``{tier: {col: cell}}``，``cell`` 含 ``text`` / ``title`` /
    ``tag`` / ``tag_start`` / ``tag_end`` / ``content_start`` / ``content_end``。
    """

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self._text = text
        self._starts = _line_starts(text)
        self._in_matrix = False
        self._table_depth = 0
        self._in_tbody = False
        self._row: dict[str, dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self.rows: dict[str, dict[str, dict[str, Any]]] = {}

    # -- 偏移工具 --
    def _abs(self) -> int:
        line, col = self.getpos()
        return self._starts[line - 1] + col - 1

    def _resolve_tag_start(self, raw: str) -> int:
        """把 HTMLParser 的 getpos 校准成当前起始标签的真实偏移。

        实测（见工位 selfcheck）：``handle_starttag`` 里的 ``getpos`` 指向该标签
        ``<`` **前一个字符**，与文档有无中文/实体无关；因此先试 ``base`` / ``base+1``，
        再退回按标签原文定位。
        """
        base = self._abs()
        for candidate in (base, base + 1):
            if candidate >= 0 and self._text.startswith(raw, candidate):
                return candidate
        found = self._text.find(raw, max(0, base - 1))
        return found if found >= 0 else base

    def _resolve_end_tag_start(self) -> int:
        """``</td>`` 的真实起始偏移（``handle_endtag`` 的 getpos 同样偏 1）。"""
        base = self._abs()
        found = self._text.find("</td>", max(0, base))
        return found if found >= 0 else base

    # -- HTMLParser 回调 --
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "table":
            if "matrix" in a.get("class", "").split():
                self._in_matrix = True
            self._table_depth += 1
            return
        if not self._in_matrix:
            return
        if tag == "tbody":
            self._in_tbody = True
        elif tag == "tr" and self._in_tbody:
            self._row = {}
        elif tag == "td" and self._row is not None:
            raw = self.get_starttag_text() or ""
            tag_start = self._resolve_tag_start(raw)
            self._cell = {
                "col": a.get("data-col", ""),
                "title": a.get("title", ""),
                "attrs": a,
                "tag": raw,
                "tag_start": tag_start,
                "content_start": tag_start + len(raw),
                "content_end": None,
                "tag_end": None,
                "text_parts": [],
            }

    def handle_endtag(self, tag: str) -> None:
        if not self._in_matrix:
            return
        if tag == "td" and self._cell is not None and self._row is not None:
            endtag_start = self._resolve_end_tag_start()
            self._cell["content_end"] = endtag_start
            self._cell["tag_end"] = endtag_start + len("</td>")
            self._cell["text"] = "".join(self._cell.pop("text_parts")).strip()
            if self._cell["col"]:
                self._row[self._cell["col"]] = self._cell
            self._cell = None
        elif tag == "tr" and self._row is not None:
            tier_cell = self._row.get("tier")
            if tier_cell is not None:
                self.rows[tier_cell["text"]] = self._row
            self._row = None
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_matrix = False

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text_parts"].append(data)


def parse_matrix(html_text: str) -> dict[str, dict[str, dict[str, Any]]]:
    parser = MatrixCellParser(html_text)
    parser.feed(html_text)
    return parser.rows


def mask_result_cells(html_text: str, rows: dict[str, dict[str, dict[str, Any]]]) -> str:
    """把结果列 ``<td>…</td>`` 整体替换为占位符，用于「除结果列外逐格一致」断言。"""
    spans: list[tuple[int, int, str]] = []
    for row in rows.values():
        for col, cell in row.items():
            if col in RESULT_COLUMNS and cell["tag_end"] is not None:
                spans.append((cell["tag_start"], cell["tag_end"], col))
    spans.sort()
    out: list[str] = []
    pos = 0
    for start, end, col in spans:
        out.append(html_text[pos:start])
        out.append(f"\x00RESULT:{col}\x00")
        pos = end
    out.append(html_text[pos:])
    return "".join(out)


def assert_result_only_change(before: str, after: str) -> None:
    """断言改动前后**除结果列外逐格一致**（掩码后逐字节比较，失败即 raise）。"""
    before_rows = parse_matrix(before)
    after_rows = parse_matrix(after)
    mask_before = mask_result_cells(before, before_rows)
    mask_after = mask_result_cells(after, after_rows)
    if mask_before != mask_after:
        # 定位第一处差异，便于排障。
        limit = min(len(mask_before), len(mask_after))
        idx = next((i for i in range(limit) if mask_before[i] != mask_after[i]), limit)
        raise AssertionError(
            "除结果列外的内容被改动了："
            f"掩码后首个差异位于 offset {idx} "
            f"（before={mask_before[idx:idx + 40]!r} after={mask_after[idx:idx + 40]!r}）"
        )
    # 结构断言：档号集合与每个非结果格的 (text, title) 必须一致。
    if set(before_rows) != set(after_rows):
        raise AssertionError("档号集合发生变化（结果列之外的表格结构被改动）")
    for tier, before_row in before_rows.items():
        after_row = after_rows[tier]
        if set(before_row) != set(after_row):
            raise AssertionError(f"{tier}: 单元格集合发生变化（结果列之外的表格结构被改动）")
        for col, cell in before_row.items():
            if col in RESULT_COLUMNS:
                continue
            other = after_row[col]
            if (cell["text"], cell["title"]) != (other["text"], other["title"]):
                raise AssertionError(
                    f"{tier} {col}: 非结果列被改动 "
                    f"{cell['text']!r} -> {other['text']!r}"
                )


# ---------------------------------------------------------------------------
# 判据：每跑次 ①②③
# ---------------------------------------------------------------------------
def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def judge_c1(run: dict[str, Any]) -> bool | None:
    """① 跑完：``exit=0``、未超时，且 ``steps_actual >= 计划步数``。

    **分母只取配置级量**（配置级公式；**不得**用运行时读数反推）。
    **不含**任何硬前提闸门：`docs/capability-matrix.html:91` 的「一个 epoch ≥5 个 optimizer step」
    已按用户裁定降为**标注项**（仅进备注，不参与判定）。


    配置级分母（见 :func:`_planned_config_denominator`）：
    native SFT/全参 = ``floor(floor(subset/dp_size)/(micro_batch×GA))``；
    native GRASPO = ``ceil(floor(subset/dp_size)/Q)``；ms-swift = HTML 配置列值。
    读数缺失 ⇒ ``None``（fail-closed，维持「未完成测试」，不放宽）。
    """
    exit_code = run.get("exit_code")
    if not _is_num(exit_code) or run.get("timed_out") is None:
        return None
    if int(exit_code) != 0 or run["timed_out"]:
        return False
    # ★ 用户 2026-09-25 裁定（本轮判读基线不变）：**不**用「跑满 epoch」替代判据①，
    #   一律 `steps_actual >= 配置级计划步数`（配置级分母由 :func:`_planned_config_denominator`
    #   给出，**不含**任何运行时读数）。GRASPO 的 epoch 事实仅进备注（见 build_tier_result）。
    steps_actual = run.get("steps_actual")
    planned_steps = run.get("planned_steps")
    if not _is_num(steps_actual) or not _is_num(planned_steps):
        return None
    # ★ 用户 2026-09-26 放行：`docs/capability-matrix.html:91` 的「正式记录硬前提」已改为**标注项**
    #   （**不参与判定**，依据用户 2026-09-23 14:20 裁定 + 「唯一判据 = 三条判据①②③」）。
    #   ⇒ 此处**移除**硬前提闸门；判据① 仍为 `steps_actual >= 计划步数`（配置级分母）。
    #   影响面实测：去掉该闸门后 54 档四态**逐档零变化**（T034/T040 仍被 `actual < planned` 否定）。
    return int(steps_actual) >= int(planned_steps)


def judge_c2(run: dict[str, Any]) -> bool | None:
    """② 产物落盘且 checkpoint 内容级可重载。读数缺失 ⇒ ``None``（未完成测试）。

    三个读数（``reloadable`` / ``tensor_count`` / ``key_tensors_complete``）由
    :func:`_extract_reload` **按档分叉**算出：full-param 走 ``all_ok`` + 逐层审计；
    LoRA 档走 ``state_dict_kind`` + ``layout_status`` + manifest LoRA 目标数对账
    （探针层覆盖字段对视觉塔 LoRA 不适用，见 :func:`_judge_lora_reload_readout`）。
    """
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    for key in ("checkpoint", "config_snapshot", "training_log"):
        if artifacts.get(key) is not True:
            return False
    reloadable = run.get("reloadable")
    tensor_count = run.get("tensor_count")
    key_complete = run.get("key_tensors_complete")
    if reloadable is None or tensor_count is None or key_complete is None:
        return None
    return bool(reloadable) and int(tensor_count) > 0 and bool(key_complete)


def judge_c3(run: dict[str, Any]) -> bool | None:
    """③ 权重真变、每步推进优化器、数值健康、跨全部 rank 零跳过。读数缺失 ⇒ ``None``。"""
    required = (
        "weight_changed",
        "optimizer_steps_all_positive",
        "loss_grad_nonfinite",
        "skipped_nonfinite_max",
    )
    if any(run.get(key) is None for key in required):
        return None
    return (
        bool(run["weight_changed"])
        and bool(run["optimizer_steps_all_positive"])
        and not bool(run["loss_grad_nonfinite"])
        and int(run["skipped_nonfinite_max"]) == 0
    )


def judge_run(run: dict[str, Any]) -> dict[str, Any]:
    """把一条跑次读数判成 ``{c1,c2,c3,verdict}``（verdict: 成功/失败/None）。"""
    c1, c2, c3 = judge_c1(run), judge_c2(run), judge_c3(run)
    values = [c1, c2, c3]
    if any(v is False for v in values):
        verdict: str | None = VERDICT_FAIL
    elif all(v is True for v in values):
        verdict = VERDICT_SUCCESS
    else:
        verdict = None
    return {"c1": c1, "c2": c2, "c3": c3, "verdict": verdict}


# ---------------------------------------------------------------------------
# 四态聚合（单一真相源）
# ---------------------------------------------------------------------------
def aggregate_state(
    attempts: list[dict[str, Any] | None],
    tier_state_override: str | None = None,
    impossible_reason: str | None = None,
) -> tuple[str, list[str]]:
    """按 §1/§2/§4 聚合四态，返回 ``(状态, 理由列表)``。

    ``attempts``：长度 3、按跑次号索引（缺失为 ``None``），元素是 :func:`judge_run` 结果。
    """
    if tier_state_override:
        if tier_state_override not in STATES:
            raise ValueError(f"非法状态覆盖值：{tier_state_override!r}（允许：{STATES}）")
        return tier_state_override, [f"读数显式指定状态：{tier_state_override}"]
    if impossible_reason:
        return "条件不合逻辑", [impossible_reason]

    if not attempts or attempts[0] is None:
        return "未完成测试", ["第 1 次跑次读数缺失"]

    first = attempts[0]
    if first["verdict"] == VERDICT_FAIL:
        failed = [name for name, key in (("①", "c1"), ("②", "c2"), ("③", "c3"))
                  if first[key] is False]
        return "失败", [f"第 1 次跑次判据 {'/'.join(failed)} 被事实否定（§4：第 1 次失败即失败）"]

    if first["verdict"] is None:
        missing = [name for name, key in (("①", "c1"), ("②", "c2"), ("③", "c3"))
                   if first[key] is None]
        return "未完成测试", [f"第 1 次跑次判据 {'/'.join(missing)} 读数缺失（不可判定）"]

    # 第 1 次成功：看第 2、3 次
    for index in (1, 2):
        later = attempts[index] if index < len(attempts) else None
        if later is None:
            return "未完成测试", [f"仅完成 {index}/3 次跑次，补跑第 {index + 1} 次后重判"]
        if later["verdict"] == VERDICT_FAIL:
            # ★ 主席 2026-09-27 裁定：三次中**有成功也有失败** ⇒ 记「失败（不稳定）」
            #   （原为「未完成测试」；因新目标"不留未完成测试"而调整 ⇒ 属**口径调整，不是放宽**）
            _detail = []
            for _pos, _item in enumerate(attempts, start=1):
                if _item is None:
                    _detail.append(f"第 {_pos} 次：缺跑次")
                    continue
                _ok = [n for n, k in (("①", "c1"), ("②", "c2"), ("③", "c3")) if _item[k] is True]
                _no = [n for n, k in (("①", "c1"), ("②", "c2"), ("③", "c3")) if _item[k] is False]
                if _item["verdict"] == VERDICT_SUCCESS:
                    _detail.append(f"第 {_pos} 次：①②③ 全成立（成功）")
                else:
                    _detail.append(
                        f"第 {_pos} 次：{'/'.join(_no)} 被否"
                        + (f"（{'/'.join(_ok)} 成立）" if _ok else "") + "（失败）"
                    )
            return "失败", [
                f"第 {index + 1} 次跑次失败 ⇒ **混合结果（有成功也有失败）** ⇒ 记「失败（不稳定）」"
                f"（主席 2026-09-27 裁定；口径调整，非放宽）；三次明细：" + "；".join(_detail)
            ]
        if later["verdict"] is None:
            return "未完成测试", [f"第 {index + 1} 次跑次读数缺失（不可判定）"]

    return "通过", ["①②③ 连续 3 次全部成立"]


# ---------------------------------------------------------------------------
# 读数抽取（第一轮跑批产根）
# ---------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_jsonl(path: Path):
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _rank_metric_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("rank_metrics.rank_*.jsonl"))


def _rank0_metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _rank_metric_files(run_dir):
        if path.name != "rank_metrics.rank_00000.jsonl":
            continue
        for payload in _iter_jsonl(path):
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                rows.append(metrics)
    return rows


def _extract_exit_code(run_dir: Path) -> int | None:
    raw = _read_text(run_dir / "exit_code").strip()
    return int(raw) if raw.lstrip("-").isdigit() else None


def _extract_steps(run_dir: Path, stdout: str) -> tuple[int | None, str]:
    """实际 optimizer 步数：rank0 旁路「逐步指标行数」权威，logging.jsonl / stdout 兜底。

    ★ 修订（2026-09-25，**事实性读错字段**）：旧实现取
    ``metrics.global_optimizer_steps_sum`` 的**最大值**，但该键是**每个报告步**的全局计数
    （逐 rank 求和：单卡恒 1、2/4 卡恒 2/4 …），**不是累计步数**；取 max 会把跑了 100 步的
    native 档读成 1 步 ⇒ 判据①被事实否定（伪否），实测 12 个 native 档因此误判「失败」。

    逐步权威读数的唯一真相源是 ``src/graspo/core/result_judge.py:78-86``
    ``STEP_METRICS_PHASES``（每 optimizer step 一行）；采集侧既有同口径实现见
    ``scripts/collect_results.py:539``（``result.steps = len(result.losses)``）。
    本函数只统计行数，**不改任何判据**。
    """
    step_rows = 0
    for path in _rank_metric_files(run_dir):
        if path.name != "rank_metrics.rank_00000.jsonl":
            continue
        for payload in _iter_jsonl(path):
            if not isinstance(payload.get("metrics"), dict):
                continue
            if payload.get("phase") in STEP_METRICS_PHASES:
                step_rows += 1
    if step_rows:
        return step_rows, "rank_metrics:step_metric_rows"
    for path in sorted(run_dir.rglob("logging.jsonl")):
        last: int | None = None
        for payload in _iter_jsonl(path):
            value = payload.get("global_step")
            if _is_num(value):
                last = int(value)
        if last is not None:
            return last, "logging.jsonl:global_step"
    matches = re.findall(r"['\"]?global_step['\"]?\s*[:=]\s*(\d+)", stdout)
    if matches:
        return max(int(m) for m in matches), "stdout:global_step"
    return None, "none"


def _extract_peak_mem(run_dir: Path, backend: str) -> tuple[float | None, str]:
    """§7 每卡峰值显存（二分口径，与 ``scripts/collect_results.py:1962-2080`` 同源）。

    ★ 修订（2026-09-25，**事实性读错字段**）：

    * **native**：``memory`` 是 rank_metrics 行的**顶层键**
      （``collect_results.py:2007-2014`` 读的就是 ``payload["memory"].max_allocated_mib``）；
      旧实现读 ``metrics.memory.max_allocated_mib`` ⇒ 结构性恒取不到，峰值列永远「—」。
    * **ms-swift**：该后端**不产 rank_metrics**（实测 36/36 档无该文件），其合法口径是自报
      ``logging.jsonl`` 的 ``memory(GiB)``（= ``max_memory_reserved``；见
      ``collect_results.py:2058-2080``）。旧实现读 ``metrics.memory.max_memory_reserved_mib``
      同样恒取不到 ⇒ 36 档峰值列也永远「—」。

    不可得 ⇒ ``(None, 该档口径标签:unavailable)``：**绝不**回退宿主采样、也**绝不**
    回退另一种口径（§9.1 明文禁止混口径；宿主采样在共享机上会被同租户作业污染）。
    """
    if backend == "native":
        best: float | None = None
        for path in _rank_metric_files(run_dir):
            if path.name != "rank_metrics.rank_00000.jsonl":
                continue
            for payload in _iter_jsonl(path):
                memory = payload.get("memory")
                if not isinstance(memory, dict):
                    continue
                value = memory.get("max_allocated_mib")
                if _is_num(value):
                    best = float(value) if best is None else max(best, float(value))
        if best is None:
            return None, f"{NATIVE_ALLOCATOR_CALIBER}:unavailable"
        return best / 1024.0, NATIVE_ALLOCATOR_CALIBER

    peaks: list[float] = []
    for path in sorted(run_dir.rglob("logging.jsonl")):
        for payload in _iter_jsonl(path):
            value = payload.get(MSSWIFT_MEMORY_KEY)
            if _is_num(value):
                peaks.append(float(value))
    if not peaks:
        for path in sorted(run_dir.rglob("stdout.log")):
            for match in _MSSWIFT_STDOUT_MEMORY_RE.finditer(_read_text(path)):
                peaks.append(float(match.group(1)))
    if not peaks:
        return None, f"{MSSWIFT_RESERVED_CALIBER}:unavailable"
    return max(peaks), MSSWIFT_RESERVED_CALIBER


def _extract_host_sample_peak(run_dir: Path) -> float | None:
    """宿主采样峰值（``gpu/gpu_memory_summary.json``）——**只另存**，不进 §7 峰值列。"""
    summary = _read_json(run_dir / "gpu" / "gpu_memory_summary.json")
    if not isinstance(summary, dict):
        return None
    peaks = [
        float(item["memory_used_mib_peak"])
        for item in (summary.get("per_gpu") or {}).values()
        if isinstance(item, dict) and item.get("memory_used_mib_peak") is not None
    ]
    return max(peaks) / 1024.0 if peaks else None


def _extract_artifacts(run_dir: Path, stdout: str) -> dict[str, bool]:
    return {
        "checkpoint": any(
            p.is_dir() and (p.name == "final" or p.name.startswith("checkpoint-"))
            for p in run_dir.rglob("*")
        ),
        "config_snapshot": any(
            p.is_file() and p.name in ("config.yaml", "args.json")
            for p in run_dir.rglob("*")
        ),
        "training_log": (
            any(p.is_file() and p.name in ("training.log", "train.log", "logging.jsonl")
                for p in run_dir.rglob("*"))
            or (run_dir / "stdout.log").is_file()
            or bool(stdout)
        ),
    }


def _probe_checked_entries(probe: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in (probe.get("checked") or []) if isinstance(entry, dict)]


def _sum_numeric(items: list[dict[str, Any]], key: str) -> int | None:
    """对一组 dict 的数值字段求和；一个都没有 ⇒ ``None``（不得当 0）。"""
    total = 0
    seen = False
    for item in items:
        value = item.get(key)
        if _is_num(value):
            total += int(value)
            seen = True
    return total if seen else None


def _manifest_lora_targets(run_dir: Path) -> int | None:
    """读 ckpt 同目录 ``manifest.json`` 的 LoRA 目标模块数。

    取 ``lora_target_signature.resolved`` 的条目数（每个目标应有 ``lora_a`` + ``lora_b``
    两个张量 ⇒ 期望张量数 = 2 × 该数）。这是 LoRA 档"关键张量集齐全"的**正确参照**，
    不能用探针的语言层覆盖口径（``layers_found``/``missing_layers``）。
    找不到 / 字段缺失 ⇒ ``None``（按"manifest 不可得"降级，不据此判失败）。
    """
    for path in sorted(run_dir.rglob("manifest.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        signature = data.get("lora_target_signature")
        if isinstance(signature, dict):
            resolved = signature.get("resolved")
            if isinstance(resolved, list) and resolved:
                return len(resolved)
    return None


def _judge_lora_reload_readout(
    probe: dict[str, Any], entries: list[dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    """LoRA 档判据②读数（2026-09-23）。

    口径依据：``docs/capability-matrix.html`` §3「判据②读数口径（2026-09-23）」；
    误报成因见 ``.local/hb-workspace/20260923-analysis/task-probe-audit/report.md``。

    * "权重张量数 > 0" ← ``checked[].tensors`` 求和（回落 ``total_elements``）；
    * "关键张量集齐全" ← **不读** ``audit_ok``/``layers_found``/``missing_layers``
      ——探针层号正则只匹配 ``layers.<N>.``，而 LoRA 预设（如 ``vision_common``）的键
      全在 ``visual.blocks.*``/``visual.merger.*``，结构上恒不命中 ⇒ 该口径不适用。
      改核：``state_dict_kind == lora_state_dict``、``layout_status == verified``、
      ``undecidable == false``、``nonfinite``/``all_zero`` 均为 0、且
      **每个 rank** 的 ``tensors == 2 × manifest.lora_target_signature.resolved``
      （manifest 不可得时跳过该计数项，降级为其余检查）。
    * 探针 ``all_ok=false`` 若只因上述层覆盖误报（本档自身读数全部通过），
      记为解释性残差，**不得**把判据②拉成失败。

    读数缺失（无 ``checked`` / 无 ``layout_status`` / 无有限性计数等）⇒ 对应项
    ``None``，交给 :func:`judge_c2` fail-closed 成「未完成测试」。
    """
    hf_audit = probe.get("hf_audit") if isinstance(probe.get("hf_audit"), dict) else {}

    tensors_total = _sum_numeric(entries, "tensors")
    if not tensors_total and _is_num(hf_audit.get("tensor_count")):
        tensors_total = int(hf_audit["tensor_count"])
    elements_total = _sum_numeric(entries, "total_elements")
    if not elements_total and _is_num(hf_audit.get("total_elements")):
        elements_total = int(hf_audit["total_elements"])
    tensor_count: int | None = tensors_total or elements_total or None

    def _count(field: str) -> int | None:
        total = _sum_numeric(entries, field)
        if total is None and _is_num(hf_audit.get(field)):
            total = int(hf_audit[field])
        return total

    nonfinite_total = _count("nonfinite_count")
    all_zero_total = _count("all_zero_count")

    layout_status = probe.get("layout_status")
    layout_state: bool | None = (
        None if layout_status is None else layout_status == _LAYOUT_STATUS_VERIFIED
    )
    undecidable_ok = not bool(probe.get("undecidable"))

    # ``state_dict_kind``：native 探针逐 rank 落该字段；HF 形态（adapter_model.safetensors）
    # 的结构自述在 ``hf_audit``/``layout`` 上，探针 schema 本就不带该字段 ⇒ 不适用（跳过）。
    kinds = [
        entry.get("state_dict_kind")
        for entry in entries
        if entry.get("state_dict_kind") is not None
    ]
    if not entries:
        kind_state: bool | None = None
    elif not kinds:
        kind_state = True
    else:
        kind_state = all(str(kind) == _LORA_STATE_DICT_KIND for kind in kinds)

    # 证据新鲜度：每个被探测文件都要有 sha256（真读到内容才可能有）。
    sha_state: bool | None = (
        all(bool(entry.get("sha256")) for entry in entries) if entries else None
    )

    # 关键张量集齐全：与 manifest 自述的 LoRA 目标数对账（per-rank 全副本口径）。
    targets = _manifest_lora_targets(run_dir)
    expected = 2 * targets if targets else None
    count_state = True
    if expected is not None:
        per_rank = [entry.get("tensors") for entry in entries]
        if entries and all(_is_num(value) for value in per_rank):
            count_state = all(int(value) == expected for value in per_rank)
        else:
            count_state = tensor_count == expected

    reloadable: bool | None
    if tensor_count is None or layout_state is None or sha_state is None:
        reloadable = None
    else:
        reloadable = bool(
            layout_state and undecidable_ok and sha_state and tensor_count > 0
        )

    key_complete: bool | None
    if (
        layout_state is None
        or kind_state is None
        or nonfinite_total is None
        or all_zero_total is None
    ):
        key_complete = None
    else:
        key_complete = bool(
            kind_state
            and layout_state
            and undecidable_ok
            and nonfinite_total == 0
            and all_zero_total == 0
            and count_state
        )

    # 仅当"按本档适用口径全部通过、但探针整体 all_ok 明确为 false"时，才记解释性残差。
    residual: str | None = None
    if reloadable and key_complete and probe.get("all_ok") is False:
        reasons: list[str] = []
        for entry in entries:
            for reason in entry.get("audit_reasons") or []:
                if isinstance(reason, str) and reason not in reasons:
                    reasons.append(reason)
        detail = "；探针 all_ok=false" + (f"（{'; '.join(reasons)}）" if reasons else "")
        residual = LORA_LAYER_COVERAGE_RESIDUAL + detail

    return {
        "reloadable": reloadable,
        "tensor_count": tensor_count,
        "key_tensors_complete": key_complete,
        "source": "torch_probe.json:lora_readout",
        "note": residual,
    }


def _judge_fullparam_reload_readout(
    probe: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """full-param 档判据②读数：仍以 ``all_ok`` 与逐层审计为准（2026-09-23 复核：判得对）。"""
    undecidable = bool(probe.get("undecidable"))
    all_ok = bool(probe.get("all_ok"))
    reloadable = all_ok and not undecidable

    tensors = 0
    checked_ok = True
    for entry in entries:
        value = entry.get("tensors")
        if _is_num(value):
            tensors += int(value)
        if entry.get("ok") is not True:
            checked_ok = False
    hf_audit = probe.get("hf_audit")
    if not tensors and isinstance(hf_audit, dict) and _is_num(hf_audit.get("tensor_count")):
        tensors = int(hf_audit["tensor_count"])
    tensor_count: int | None = tensors if tensors > 0 else None

    if isinstance(hf_audit, dict):
        key_complete: bool | None = bool(hf_audit.get("audit_ok")) and not hf_audit.get(
            "missing_layers"
        )
    else:
        key_complete = checked_ok and tensor_count is not None
    return {
        "reloadable": reloadable,
        "tensor_count": tensor_count,
        "key_tensors_complete": key_complete,
        "source": "torch_probe.json",
        "note": None,
    }


def _extract_reload(run_dir: Path, mode: str) -> dict[str, Any]:
    """从 ``torch_probe.json`` 读内容级可重载证据（§1 ②），按档模式选读数口径。

    * **LoRA 档**（``mode == "LoRA"`` 或探针 ``state_dict_kind == lora_state_dict``）
      ⇒ :func:`_judge_lora_reload_readout`：探针层覆盖字段不适用，改核
      ``state_dict_kind``/``layout_status``/张量数与 manifest LoRA 目标数/有限性；
    * **其余（full-param）** ⇒ :func:`_judge_fullparam_reload_readout`：逐层审计有效。
    """
    probe = _read_json(run_dir / "torch_probe.json")
    if not isinstance(probe, dict):
        return {
            "reloadable": None,
            "tensor_count": None,
            "key_tensors_complete": None,
            "source": "none",
            "note": None,
        }
    entries = _probe_checked_entries(probe)
    is_lora = mode == "LoRA" or any(
        entry.get("state_dict_kind") == _LORA_STATE_DICT_KIND for entry in entries
    )
    if is_lora:
        return _judge_lora_reload_readout(probe, entries, run_dir)
    return _judge_fullparam_reload_readout(probe, entries)


def _ledger_weight_evidence(
    ledger_rows: dict[str, Any] | None, tier: str
) -> tuple[bool | None, str]:
    """从**收集器产物**（ledger）复用**已有的**权重变化证据 —— **不新增判据语义**。

    证据字段（由 `scripts/collect_results.py` 落盘）：
      · `weight_evidence_source`：如 `checkpoint:safetensors_bytes(lora_b)`（LoRA）/
        `checkpoint:base_weights_bytes_compare`（全参）/ `run_metrics:trainable_norm_delta`（native）；
      · `criteria_detail.A2`：形如「optimizer step=160，计划 160 ≥ 门槛 5，**权重已变化**（tuner_type=lora，来源：…）」。
    仅当来源非空**且** A2 明写"权重已变化" ⇒ ``(True, "ledger:<source>")``；否则 ``(None, 原因)``。
    """
    row = (ledger_rows or {}).get(tier) if ledger_rows else None
    if not isinstance(row, dict):
        return None, ""
    source = str(row.get("weight_evidence_source") or "")
    a2 = str((row.get("criteria_detail") or {}).get("A2") or "")
    if source and "权重已变化" in a2:
        return True, f"ledger:{source}"
    if source:
        return None, f"ledger:{source}（A2 未确认）"
    return None, ""


def _extract_weight_changed(
    run_dir: Path,
    mode: str,
    tier: str = "",
    ledger_rows: dict[str, Any] | None = None,
) -> tuple[bool | None, str]:
    """③ 权重真变（**只改取数层**，判据语义与四态聚合均不变）。

    来源（**多源必须一致**；可得的多源结论不同 ⇒ ``None`` 保守）：
      1. native：`rank_metrics` 逐步的 ``lora_norm_delta`` / ``trainable_norm_delta``（非零即真变）；
      2. ms-swift / DS：**收集器产物 ledger** 的 ``weight_evidence_source`` ＋ ``criteria_detail.A2``
         （checkpoint 字节对比既有证据）。
    """
    keys = (
        ("lora_norm_delta", "global_lora_norm_delta_mean")
        if mode != "全量"
        else ("trainable_norm_delta", "global_trainable_norm_delta_mean")
    )
    finite: list[float] = []
    for row in _rank0_metric_rows(run_dir):
        raw = row.get(keys[0])
        if raw is None:
            raw = row.get(keys[1])
        if _is_num(raw) and math.isfinite(float(raw)):
            finite.append(float(raw))
    local: bool | None = None
    local_src = f"run_metrics:{keys[0]}"
    if finite:
        local = any(value != 0.0 for value in finite)
    else:
        local_src = "none"

    ledger_value, ledger_src = _ledger_weight_evidence(ledger_rows, tier)
    available: dict[str, bool] = {}
    if local is not None:
        available[local_src] = local
    if ledger_value is not None:
        available[ledger_src] = ledger_value
    if not available:
        return None, (ledger_src or local_src or "none")
    if len(set(available.values())) != 1:
        return None, "；".join(f"{k}={v}" for k, v in available.items())
    return next(iter(available.values())), "、".join(available)


#: ms-swift 补丁（镜像 `graspo:v0.28.11-cu130fix-v2-nfc1` 起）新增的两个日志键。
#: ★ 本常量随 `_extract_optimizer_health` 的替换一并重申（原定义紧跟该函数，替换时被并入边界）。
MS_SWIFT_NONFINITE_KEYS = ("nonfinite_grad_step_count", "nonfinite_grad_step_count_coverage")


def _stepwise_skipped_total(run_dir: Path) -> tuple[int | None, str]:
    """**逐步** `skipped_nonfinite` 合计（主落点 `step_metrics.rank_*.jsonl`）。

    用途：与**收集器产物**（ledger `nonfinite_skips` + `ms_swift_counter:`）交叉核对 ——
    两源**必须一致**，不一致 ⇒ 上层取 `None`（保守，**不猜**）。
    无逐步记录（旧产物）/ `coverage≠1` / 缺键 ⇒ `(None, 原因)`。
    """
    records = [r for src_name, r in _iter_stepwise_records(run_dir) if src_name.startswith("step_metrics:")]
    if not records:
        return None, "无 step_metrics 逐步记录（旧产物）"
    for r in records:
        coverage = r.get("coverage")
        if coverage is None or not _is_num(coverage) or int(coverage) != 1:
            return None, "coverage≠1 或缺 coverage"
    values = [r.get("skipped_nonfinite") for r in records]
    if any(not _is_num(v) for v in values):
        return None, "缺 skipped_nonfinite 键"
    return sum(int(v) for v in values), f"step_metrics:sum({len(records)} 步)"


def _expected_world_size(
    run_dir: Path,
    cfg: dict[str, Any] | None = None,
    ledger_rows: dict[str, Any] | None = None,
    tier: str = "",
) -> int | None:
    """该档该跑次的**期望 rank 数**（world_size）——用于 rank 齐全性校验。

    顺序：① `args.json` 的 `nproc_per_node`/`world_size`；② 档配置（native: tp×dp×pp；
    ms-swift: `msswift.nproc_per_node`）；③ ledger 的 `cards`。**都取不到 ⇒ None**（⇒ 上层不可判）。
    """
    for path in sorted(run_dir.rglob("args.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        for key in ("world_size", "nproc_per_node"):
            value = data.get(key)
            if _is_num(value) and int(value) > 0:
                return int(value)
        native = data.get("native") if isinstance(data.get("native"), dict) else {}
        try:
            prod = int(native.get("tp_size", 1)) * int(native.get("dp_size", 1)) * int(native.get("pp_size", 1))
        except (TypeError, ValueError):
            prod = 0
        if prod > 1:
            return prod
    if isinstance(cfg, dict):
        native = cfg.get("native") or {}
        try:
            prod = int(native.get("tp_size", 1)) * int(native.get("dp_size", 1)) * int(native.get("pp_size", 1))
        except (TypeError, ValueError):
            prod = 0
        if prod > 1:
            return prod
        msswift = cfg.get("msswift") or {}
        value = msswift.get("nproc_per_node")
        if _is_num(value) and int(value) > 0:
            return int(value)
    row = (ledger_rows or {}).get(tier) if ledger_rows else None
    if isinstance(row, dict) and _is_num(row.get("cards")) and int(row["cards"]) > 0:
        return int(row["cards"])
    return None


def _group_stepwise_by_rank(records_raw: list[tuple[str, dict[str, Any]]]) -> dict[int, list[dict[str, Any]]]:
    """把逐步记录**按 rank 分组**（跨 rank 的同名 `opt_step_index` **不是重复**）。"""
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for _src, r in records_raw:
        rank = r.get("rank")
        key = int(rank) if _is_num(rank) else 0
        by_rank.setdefault(key, []).append(r)
    return by_rank


def _iter_stepwise_records(run_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """收集 nfc2 的**逐步**记录（`step_metrics.rank_*.jsonl` 为独立落点，`logging.jsonl` 可作交叉）。

    只认**含 `opt_step_index`** 的行（旧产物无该键 ⇒ 自然不产出任何记录 ⇒ 上层回落旧口径）。
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(run_dir.rglob("step_metrics.rank_*.jsonl")):
        for payload in _iter_jsonl(path):
            if isinstance(payload, dict) and "opt_step_index" in payload:
                out.append((f"step_metrics:{path.name}", payload))
    for path in sorted(run_dir.rglob("logging.jsonl")):
        for payload in _iter_jsonl(path):
            if isinstance(payload, dict) and "opt_step_index" in payload:
                out.append((f"logging.jsonl:{path.name}", payload))
    return out


def _extract_optimizer_health(
    run_dir: Path,
    steps_actual: int | None = None,
    backend: str = "",
    tier: str = "",
    cfg: dict[str, Any] | None = None,
    ledger_rows: dict[str, Any] | None = None,
    expected_ws: int | None = None,
    planned_steps: int | None = None,
) -> tuple[bool | None, bool | None, str]:
    """返回 ``(每步优化器都推进, 存在非有限 loss/grad, 逐步读数说明)``。

    **nfc2 逐步新键优先**（`opt_step_index` / `opt_step_delta` / `skipped_nonfinite` /
    `loss_raw`（平滑前原值） / `grad_norm_raw`（clip 前） / `grad_norm_missing_reason` / `coverage`）：

    * `optimizer_steps_all_positive`：**逐步全部 `opt_step_delta > 0`** ⇒ True；
      任一 `<= 0` ⇒ **False**（该跑次 ③ 不成立）；`opt_step_index` **不连续（缺步）** /
      `coverage != 1` / 键缺 ⇒ **None**（保守，**不猜**）。
    * `loss_grad_nonfinite`（语义 = **存在**非有限）：任一 `loss_raw` 或 `grad_norm_raw` 非有限 ⇒ **True**；
      `grad_norm_raw` 为 null 时，仅当 `grad_norm_missing_reason == "nan_replaced"`（该步梯度确为非有限）⇒ True，
      其余缺失原因 ⇒ **None**；`loss_raw` 键缺 / `coverage != 1` ⇒ **None**。
    * 逐步记录**不完整**（行数 < `steps_actual` 且无独立落点可自证）⇒ None ⇒ 回落旧口径。

    **只改取数**；`judge_c1/c2/c3` 与四态聚合一字不动。
    """
    records = _iter_stepwise_records(run_dir)
    if records:
        sources = sorted({src for src, _ in records})
        coverage = [r.get("coverage") for _, r in records]
        if any(_is_num(c) and int(c) != 1 for c in coverage) or any(c is None for c in coverage):
            return None, None, f"coverage≠1 或无 coverage（{sources}）⇒ 逐步读数不可判"
        # ★ 修复（2026-09-27）：**按 rank 分组**——跨 rank 的同名 opt_step_index 不是重复。
        by_rank = _group_stepwise_by_rank(records)
        expected_ws = expected_ws if (expected_ws is not None and expected_ws > 0) \
            else _expected_world_size(run_dir, cfg=cfg, ledger_rows=ledger_rows, tier=tier)
        if expected_ws is None:
            return None, None, f"无法确定期望 rank 数（{sources}）⇒ rank 齐全性不可判 ⇒ 不可判"
        if len(by_rank) != expected_ws:
            return None, None, (
                f"rank 文件数 {len(by_rank)} ≠ 期望 world_size {expected_ws}"
                f"（{sorted(by_rank)}；{sources}）⇒ **任何 rank 缺失都不得判 True** ⇒ 不可判"
            )
        unique_idx: list[int] = []
        for rank, rows_rank in sorted(by_rank.items()):
            idx_rank = [int(r["opt_step_index"]) for r in rows_rank if _is_num(r.get("opt_step_index"))]
            if len(idx_rank) != len(rows_rank):
                return None, None, f"rank {rank} 缺 opt_step_index ⇒ 不可判"
            if len(set(idx_rank)) != len(idx_rank):
                return None, None, f"rank {rank} 内 opt_step_index 重复 ⇒ 不可判"
            unique_idx.extend(sorted(set(idx_rank)))
        unique_idx = sorted(set(unique_idx))
        has_independent = any(src_name.startswith("step_metrics:") for src_name in sources)
        complete = bool(unique_idx) and unique_idx == list(range(1, len(unique_idx) + 1))
        if _is_num(steps_actual) and int(steps_actual) > 0:
            complete = complete and len(unique_idx) >= int(steps_actual)
        if not has_independent and not complete:
            return None, None, f"仅有采样日志（{sources}）⇒ 无法自证逐步完整 ⇒ 不可判"
        # 每个 rank 内：步数须一致（同一步在所有 rank 都有读数）
        per_rank_steps = {rank: len(rows_rank) for rank, rows_rank in by_rank.items()}
        if len(set(per_rank_steps.values())) != 1:
            return None, None, f"各 rank 步数不一致（{per_rank_steps}）⇒ 不可判"
        # ★★ 守卫（独立复核 2026-09-27）：**steps_actual 缺失 + 文件被截断 ⇒ 不得判 True**。
        #    只有"逐步行数 == 计划步数"这一**独立**条件成立时才允许有值（否则可能把截断文件读成"全好"）。
        if steps_actual is None:
            _planned_ok = _is_num(planned_steps) and int(planned_steps) > 0 and len(unique_idx) == int(planned_steps)
            if not _planned_ok:
                return None, None, (
                    f"★守卫触发：steps_actual 缺失，且逐步行数 {len(unique_idx)} != 计划步数 "
                    f"{planned_steps} ⇒ 无法自证步数正确 ⇒ 逐步两项不可判（堵「截断文件被判 True」的假通过通道）"
                )
            _stepwise_note = f"steps_actual 缺失但行数 {len(unique_idx)} == 计划步数 {int(planned_steps)}（守卫放行）"
        deltas = [r.get("opt_step_delta") for _, r in records]
        if any(not _is_num(d) for d in deltas):
            return None, None, f"缺 opt_step_delta 键（{sources}）⇒ 逐步推进不可判"
        # ---- optimizer_steps_all_positive
        if any(int(d) <= 0 for d in deltas):
            steps_positive: bool | None = False
        elif not complete:
            steps_positive = None
        else:
            steps_positive = True
        # ---- loss_grad_nonfinite（= 存在非有限）
        nonfinite: bool | None = False
        reasons: list[str] = []
        for _src, r in records:
            loss = r.get("loss_raw")
            if loss is None or not _is_num(loss):
                nonfinite = None
                reasons.append("loss_raw 键缺")
                break
            if not math.isfinite(float(loss)):
                nonfinite = True
            grad = r.get("grad_norm_raw")
            if grad is None:
                reason = r.get("grad_norm_missing_reason")
                if reason == "nan_replaced":
                    nonfinite = True
                else:
                    nonfinite = None
                    reasons.append(f"grad_norm_raw 缺失（reason={reason}）")
                    break
            elif not _is_num(grad) or not math.isfinite(float(grad)):
                nonfinite = True
        detail = (
            f"逐步 {len(unique_idx)} 步、来源 {sources}；"
            f"opt_step_delta 最小 {min(int(d) for d in deltas)}；"
            f"optimizer_steps_all_positive={steps_positive}、loss_grad_nonfinite={nonfinite}"
            + (f"；{'；'.join(reasons)}" if reasons else "")
        )
        return steps_positive, nonfinite, detail

    # ---- 回落：既有 native rank_metrics 口径（逐字保留，行为不变）
    steps: list[int] = []
    nonfinite = False
    for row in _rank0_metric_rows(run_dir):
        value = row.get("global_optimizer_steps_sum")
        if _is_num(value):
            steps.append(int(value))
        for key in ("global_loss_mean", "global_grad_norm_mean"):
            raw = row.get(key)
            if _is_num(raw) and not math.isfinite(float(raw)):
                nonfinite = True
    if not steps:
        return None, None, ""
    return all(step > 0 for step in steps), nonfinite, ""


def _extract_ms_swift_nonfinite_counter(run_dir: Path) -> int | None:
    """读 ms-swift 补丁自带的累计非有限计数（`checkpoint-*/trainer_state.json`）。

    **语义铁律**：只有 `..._coverage == 1`（非 DeepSpeed、读的是本步 clip 的 total_norm）
    才返回计数；`coverage == 0`（DeepSpeed：`accelerate` 返回的
    `engine.get_global_grad_norm()` 是 step 之后才赋值的陈旧值、ZeRO overflow 步还会跳过）
    或键缺失 ⇒ 返回 ``None``（**口径不可测**），**绝不返回 0 冒充「0 次跳过」**。
    """
    best: tuple[int, int] | None = None  # (global_step, count)
    for state_path in sorted(run_dir.rglob("trainer_state.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(state, dict):
            continue
        history = state.get("log_history")
        if not isinstance(history, list):
            continue
        step = state.get("global_step")
        step = int(step) if isinstance(step, int) else -1
        count: int | None = None
        coverage: int | None = None
        for entry in history:
            if not isinstance(entry, dict):
                continue
            value = entry.get(MS_SWIFT_NONFINITE_KEYS[0])
            cov = entry.get(MS_SWIFT_NONFINITE_KEYS[1])
            if isinstance(value, int):
                count = value if count is None else max(count, value)
            if isinstance(cov, int):
                coverage = cov
        if count is None or coverage != 1:
            # 键存在但 coverage != 1 ⇒ 明确不可判（返回 None 让上层记「口径不可测」）
            if count is not None or coverage is not None:
                continue
            continue
        if best is None or step > best[0]:
            best = (step, count)
    return None if best is None else best[1]


def _extract_ms_swift_logging_counter(run_dir: Path) -> int | None:
    """③ 的**辅来源**：ms-swift 的 ``logging.jsonl`` 里的 ``nonfinite_grad_step_count``。

    **铁律**：必须**每一行**都 ``nonfinite_grad_step_count_coverage == 1`` 且计数取值一致才采信；
    任一不满足（coverage≠1 / 键缺失 / 各行不一致）⇒ ``None``（**口径不可测**），
    **绝不返回 0 冒充「0 次跳过」**。
    """
    values: set[int] = set()
    rows = 0
    for path in sorted(run_dir.rglob("logging.jsonl")):
        for payload in _iter_jsonl(path):
            if not isinstance(payload, dict):
                continue
            value = payload.get(MS_SWIFT_NONFINITE_KEYS[0])
            coverage = payload.get(MS_SWIFT_NONFINITE_KEYS[1])
            if not isinstance(value, int) or coverage != 1:
                return None
            values.add(value)
            rows += 1
    if not rows or len(values) != 1:
        return None
    return next(iter(values))


def _ms_swift_collector_candidate(ledger_rows: dict[str, Any] | None, tier: str) -> tuple[int | None, str]:
    """**主来源** = 收集器产物（``collect_results.py`` 落的 ledger）：
    ``nonfinite_skips`` + ``nonfinite_skips_source``；仅 source 前缀 ``ms_swift_counter:`` 时采信。
    """
    if not ledger_rows:
        return None, ""
    row = ledger_rows.get(tier)
    if not isinstance(row, dict):
        return None, ""
    source = str(row.get("nonfinite_skips_source") or "")
    count = row.get("nonfinite_skips")
    if not source.startswith("ms_swift_counter:"):
        return None, source
    if not isinstance(count, int) or count < 0:
        return None, source
    return count, source


def _find_ledger(run_dir: Path) -> Path | None:
    """从产根推断收集器产物（ledger.jsonl）路径。"""
    for cand in (
        run_dir.parent / "batch" / "ledger" / "ledger.jsonl",
        run_dir.parent.parent / "batch" / "ledger" / "ledger.jsonl",
        run_dir / "batch" / "ledger" / "ledger.jsonl",
    ):
        if cand.is_file():
            return cand
    return None


def _load_ledger_rows(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    rows: dict[str, Any] = {}
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("tier_id"):
            rows[str(payload["tier_id"])] = payload
    return rows or None


def _extract_skipped_nonfinite_max(
    run_dir: Path,
    backend: str = "",
    tier: str = "",
    ledger_rows: dict[str, Any] | None = None,
) -> tuple[int | None, str]:
    """③ 的「跨全部 rank 非有限跳过数」**取数**（用户 2026-09-27 裁定：只改取数，不改判据）。

    来源与保守规则：
      1. **主来源 = 收集器产物**（ledger 的 ``nonfinite_skips`` + ``nonfinite_skips_source``），
         仅 ``source`` 以 ``ms_swift_counter:`` 开头（覆盖度 1）时采信；
      2. **辅来源 = ``logging.jsonl``**（``nonfinite_grad_step_count`` + ``..._coverage``），
         仅「每行 coverage==1 且取值一致」时采信；**两处必须一致**；
      3. 第三回退 = ms-swift 补丁写进 ``checkpoint-*/trainer_state.json`` 的同名计数（既有实现）。
    可得的多源**必须相等**；不相等 ⇒ ``None``（**保守 = 不可判**）。
    ``count > 0`` 由既有判据层判「③ 不成立」；``coverage=0`` / 无键 / 来源非计数器 ⇒ ``None``。
    ⚠ 该计数被 ``world_size`` 放大 ⇒ **只能读作零/非零**，不得读作发生次数。
    """
    if backend == "native":
        per_rank: dict[str, int] = {}
        for path in _rank_metric_files(run_dir):
            best: int | None = None
            for payload in _iter_jsonl(path):
                metrics = payload.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                for key in ("global_skipped_nonfinite_sum", "skipped_nonfinite"):
                    value = metrics.get(key)
                    if _is_num(value):
                        best = int(value) if best is None else max(best, int(value))
            if best is not None:
                per_rank[path.name] = best
        if not per_rank:
            return None, ""
        return max(per_rank.values()), "native:rank_metrics"

    collector, source = _ms_swift_collector_candidate(ledger_rows, tier)
    _stepwise_total, _stepwise_note = _stepwise_skipped_total(run_dir)
    candidates = {
        "collector": collector,
        "logging.jsonl": _extract_ms_swift_logging_counter(run_dir),
        "trainer_state.json": _extract_ms_swift_nonfinite_counter(run_dir),
        "step_metrics": _stepwise_total,
    }
    available = {k: v for k, v in candidates.items() if v is not None}
    if not available:
        return None, source
    if len(set(available.values())) != 1:
        return None, "；".join(f"{k}={v}" for k, v in candidates.items())
    return next(iter(available.values())), "、".join(available) + (f"（source={source}）" if source else "")


def _extract_date(run_dir: Path) -> str | None:
    """实测日期：driver 完成标记 ``finished_at`` 的日期部分（**实测时间戳，非人工填值**）。

    来源 ``<RUN_ROOT>/<T###>/.driver_run1_complete.json::finished_at``，实测形如
    ``2026-09-24 01:45:10+0000``。标记缺失 ⇒ ``None``（保持 ``—``；**不用**文件 mtime 猜）。
    """
    marker = _read_json(run_dir / ".driver_run1_complete.json")
    if isinstance(marker, dict):
        value = marker.get("finished_at")
        if isinstance(value, str) and len(value.strip()) >= 10:
            return value.strip()[:10]
    return None


def _extract_failure_class(
    run_dir: Path, stdout: str, exit_code: int | None, timed_out: bool
) -> tuple[str | None, str]:
    """失败归因（HTML §2 三类：我方实现 / 上游依赖 / 资源不足）。

    **只从产物里的事实特征串取**（``FAILURE_CLASS_PATTERNS``）；匹配不到 ⇒ ``(None, "")``
    即 ``failure`` 列留 ``—``（**不猜**：宁可缺归因，也不编一个类别）。返回
    ``(归因, 命中的原文片段)``，后者进证据链以便复核。
    """
    if not timed_out and exit_code == 0:
        return None, ""
    haystack = stdout
    marker = _read_json(run_dir / ".driver_run1_complete.json")
    if isinstance(marker, dict):
        haystack += "\n" + json.dumps(marker, ensure_ascii=False)
    for name, pattern in FAILURE_CLASS_PATTERNS:
        match = pattern.search(haystack)
        if match:
            return name, match.group(0)
    return None, ""


def _extract_msswift_finiteness(run_dir: Path) -> dict[str, Any]:
    """ms-swift 档的**辅助**数值有限性读数（只登记进证据链，**不参与**判据③的真假）。

    为什么不能参与判真（三条权威依据）：

    1. ``args.json`` 的 ``logging_steps = 5``（实测 36/36 档）⇒ ``logging.jsonl`` 是**采样**，
       只覆盖 1/5 的步，不是逐步读数；
    2. ``logging_nan_inf_filter=True`` ⇒ NaN/Inf 的 step loss 被**替换为历史均值**再记
       （``.venv/lib/python3.12/site-packages/transformers/trainer.py:1746-1752``）
       ⇒ 日志里的 loss 结构性恒有限，「没看到 NaN」不构成证据；
    3. ms-swift ``_fix_grad_norm_nan``（``.local/refs/ms-swift-4.5.3/git-v4.5.3/swift/trainers/mixin.py:744-779``）
       在 grad_norm 为 NaN 时把 ``p.grad=None``（丢弃更新）后**照常 step**，且**只判 isnan
       不判 isinf、无任何计数**；``mixin.py:1078-1081`` 在 grad_norm 为 None 时**不写该键**
       ⇒ NaN 的痕迹只是「该步缺 ``grad_norm`` 键」，而不是 NaN 值。

    ⇒ 本函数的输出只用于备注/证据，**不得**用来把 c3 判真（那就是放宽判据）。
    """
    rows = 0
    nonfinite_loss = 0
    nonfinite_grad = 0
    missing_grad = 0
    for path in sorted(run_dir.rglob("logging.jsonl")):
        for payload in _iter_jsonl(path):
            if "loss" not in payload:
                continue
            rows += 1
            loss = payload.get("loss")
            if _is_num(loss) and not math.isfinite(float(loss)):
                nonfinite_loss += 1
            if "grad_norm" not in payload:
                missing_grad += 1
            else:
                grad = payload.get("grad_norm")
                if _is_num(grad) and not math.isfinite(float(grad)):
                    nonfinite_grad += 1
    return {
        "logging_rows": rows,
        "nonfinite_loss_rows": nonfinite_loss,
        "nonfinite_grad_rows": nonfinite_grad,
        "missing_grad_rows": missing_grad,
    }


def _iter_jsonl_rank0(run_dir: Path):
    """逐条迭代 rank0 旁路的 JSON 对象（保留 ``phase``，供逐步读数使用）。"""
    for path in _rank_metric_files(run_dir):
        if path.name != "rank_metrics.rank_00000.jsonl":
            continue
        for payload in _iter_jsonl(path):
            yield payload


def _extract_epoch_facts(run_dir: Path) -> dict[str, Any] | None:
    """读 GRASPO 的 ``epoch_summary`` 事件（stdout.log 单行 JSON）——**事实陈述，不参与判定**。

    来源：``src/graspo/flow/trainer/trainer.py:270-281`` 的 ``_print_json({"event":
    "epoch_summary", ...})``；字段 ``epoch_cumulative.{samples_seen,samples_total,progress}``
    与 ``epoch_cumulative.decisions.terminal.{trainable,perfect_skip,invalid,
    invalid_no_preference_gap}``（判定剔除口径见 ``rollout.py:253-320``）。
    为什么进备注（分支 C 裁定）：``planned_steps`` 修正后，"步数比"需要一个读者能复核的
    口径说明，否则"步数不足"会被误读成"没跑完"。
    """
    for path in sorted(run_dir.rglob("stdout.log")):
        for line in _read_text(path).splitlines():
            line = line.strip()
            if not line.startswith("{") or '"event": "epoch_summary"' not in line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            cumulative = payload.get("epoch_cumulative")
            if not isinstance(cumulative, dict):
                continue
            terminal = ((cumulative.get("decisions") or {}).get("terminal") or {})
            return {
                "samples_seen": cumulative.get("samples_seen"),
                "samples_total": cumulative.get("samples_total"),
                "progress": cumulative.get("progress"),
                "trainable": terminal.get("trainable"),
                "perfect_skip": terminal.get("perfect_skip"),
                "invalid": terminal.get("invalid"),
                "invalid_no_preference_gap": terminal.get("invalid_no_preference_gap"),
            }
    return None


def _extract_rollout_queue(run_dir: Path) -> int | None:
    """从该跑次配置快照读 ``rollout_queue_batch_size``（GRASPO 可达步数的分母）。

    字段定义见 ``src/graspo/core/schema.py:319``（缺省 8）。取不到 ⇒ ``None``
    （备注就不写"可达步数"，**不用缺省值冒充实测**）。
    """
    for path in sorted(run_dir.rglob("config.yaml")):
        match = re.search(r"^\s*rollout_queue_batch_size:\s*(\d+)\s*$", _read_text(path), re.M)
        if match:
            return int(match.group(1))
    for path in sorted(run_dir.rglob("args.json")):
        data = _read_json(path)
        if isinstance(data, dict) and _is_num(data.get("rollout_queue_batch_size")):
            return int(data["rollout_queue_batch_size"])
    return None


def _extract_runner_seconds(run_dir: Path) -> float | None:
    """该跑次的墙钟秒数（driver 完成标记 ``seconds``）——用于备注里的跑次明细。"""
    marker = _read_json(run_dir / ".driver_run1_complete.json")
    if isinstance(marker, dict) and _is_num(marker.get("seconds")):
        return float(marker["seconds"])
    return None


def _extract_weight_delta_detail(run_dir: Path, mode: str) -> str | None:
    """判据③「权重真变」的**逐步**读数说明 —— **仅在逐步全 0 时**产出。

    读数口径与 :func:`_extract_weight_changed` 同源（``trainer_adapter.py:788-840`` 落
    ``trainable_norm_delta`` / ``lora_norm_delta``），这里把整条逐步序列摊开，
    让"权重未更新"这一结论**逐步可复核**（而非只看首尾差异）。
    """
    key = "trainable_norm_delta" if mode == "全量" else "lora_norm_delta"
    values: list[float] = []
    for payload in _iter_jsonl_rank0(run_dir):
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or payload.get("phase") not in STEP_METRICS_PHASES:
            continue
        raw = metrics.get(key)
        if _is_num(raw):
            values.append(float(raw))
    if not values or any(abs(value) > 0.0 for value in values):
        return None
    return (
        f"判据③：逐步 {key} 全 0（{len(values)}/{len(values)} 步，rank0 旁路）"
        "⇒ 该跑次权重未更新"
    )


def extract_run(
    run_dir: Path,
    tier: str,
    attempt: int,
    planned_steps: int | None,
    backend: str,
    mode: str,
    run_meta: dict[str, Any] | None = None,
    algo: str = "",
    cfg: dict[str, Any] | None = None,
    html_planned_steps: int | None = None,
    display_root: str | None = None,
    ledger_rows: dict[str, Any] | None = None,
    expected_ws: int | None = None,
) -> dict[str, Any]:
    """把一个 ``<RUN_ROOT>/<T###>/`` 抽成一条跑次读数（抽不到的一律 ``None``，不猜）。

    ``run_meta``：``--run-meta`` 按 ``(tier, attempt)`` 命中的**实测注入**，只允许覆盖
    ``date`` / ``max_context`` / ``failure_class`` 三个**产物里不存在**的字段；
    优先级低于 ``--readings``（保持既有覆盖语义），且**不得**覆盖任何判据读数。
    """
    stdout = _read_text(run_dir / "stdout.log")
    exit_code = _extract_exit_code(run_dir)
    timed_out = exit_code == 124 or bool(_TIMEOUT_RE.search(stdout))
    steps_actual, steps_source = _extract_steps(run_dir, stdout)
    peak_mem, peak_caliber = _extract_peak_mem(run_dir, backend)
    reload_readout = _extract_reload(run_dir, mode)
    weight_changed, weight_source = _extract_weight_changed(
        run_dir, mode, tier=tier, ledger_rows=ledger_rows)
    steps_positive, nonfinite, stepwise_detail = _extract_optimizer_health(
        run_dir, steps_actual=steps_actual, backend=backend, tier=tier,
        cfg=cfg, ledger_rows=ledger_rows, expected_ws=expected_ws, planned_steps=planned_steps)
    failure_class, failure_evidence = _extract_failure_class(
        run_dir, stdout, exit_code, timed_out
    )

    # 产物里没有的三列：先取产物可见的实测事实（driver 完成标记），再由 run-meta 覆盖。
    date = _extract_date(run_dir)
    max_context: float | None = None
    meta = run_meta if isinstance(run_meta, dict) else {}
    if isinstance(meta.get("date"), str) and str(meta["date"]).strip():
        date = str(meta["date"]).strip()
    if _is_num(meta.get("max_context")):
        max_context = float(meta["max_context"])
    if meta.get("failure_class"):
        failure_class, failure_evidence = str(meta["failure_class"]), "run-meta"
    # 档级归因兜底（显式表；仅当产物事实串未命中时）——审查 🔴 阻断项修复
    _tier_failure = FAILURE_CLASS_BY_TIER.get(tier)
    if _tier_failure is not None and not failure_class:
        failure_class, failure_evidence = _tier_failure[0], _tier_failure[1]

    notes: list[str] = [reload_readout["note"]]
    # ⑤ 指挥官裁定：ctx_max 无实测来源 ⇒ 列留 "—"，备注显式写明，不得用 seq_len_cap 充数。
    if max_context is None:
        notes.append("最大可行上下文：无实测来源")
    # 分支 C 裁定：epoch 事实与"可达步数"入备注 —— 只陈述事实，不参与任何判定。
    epoch_facts = _extract_epoch_facts(run_dir)
    # 每 DP rank 的应处理样本数（分母只取**配置级** dp_size；修正令 §2）
    epoch_target_samples: int | None = None
    if epoch_facts and _is_num(epoch_facts.get("samples_total")):
        dp_size_cfg = 1
        if isinstance(cfg, dict) and isinstance(cfg.get("native"), dict):
            raw_dp = cfg["native"].get("dp_size")
            if _is_num(raw_dp) and int(raw_dp) >= 1:
                dp_size_cfg = int(raw_dp)
        epoch_target_samples = int(epoch_facts["samples_total"]) // dp_size_cfg
    if epoch_facts:
        seen = epoch_facts.get("samples_seen")
        total = epoch_facts.get("samples_total")
        reached = (
            _is_num(seen) and _is_num(epoch_target_samples) and int(seen) >= int(epoch_target_samples)
        )
        notes.append(
            f"epoch {'已跑满' if reached else '未跑满'}"
            f"（本 rank {seen}/{epoch_target_samples} 样本、全局 {total} 条、"
            f"progress={epoch_facts.get('progress')}）"
        )
        trainable = epoch_facts.get("trainable")
        if _is_num(trainable):
            notes.append(
                f"epoch 终结决策（**事实陈述，不用于反推计划步数**）：trainable {int(trainable)}、"
                f"perfect_skip {epoch_facts.get('perfect_skip')}、invalid {epoch_facts.get('invalid')}"
            )
    weight_note = _extract_weight_delta_detail(run_dir, mode)
    if weight_note:
        notes.append(weight_note)
    if stepwise_detail:
        notes.append(f"③ 逐步读数（nfc2 新键）：{stepwise_detail}")
    finiteness: dict[str, Any] | None = None
    # ★ 2026-09-27 用户放行：③ 的**取数层**扩展（只改取数，judge_* 一字未动）。
    skipped_max, skipped_source = _extract_skipped_nonfinite_max(
        run_dir, backend=backend, tier=tier, ledger_rows=ledger_rows)
    if backend != "native":
        finiteness = _extract_msswift_finiteness(run_dir)
        if skipped_max is not None:
            notes.append(
                f"③ 非有限梯度跳过计数读数 = {skipped_max}（来源：{skipped_source}）；"
                "⚠ 该计数被 world_size 放大，**只能读作零/非零**，不得读作发生次数"
            )
        detail = ""
        if skipped_max is None and finiteness["logging_rows"]:
            detail = (
                f"（logging 采样 {finiteness['logging_rows']} 行："
                f"非有限 loss {finiteness['nonfinite_loss_rows']}、"
                f"非有限 grad_norm {finiteness['nonfinite_grad_rows']}、"
                f"缺 grad_norm 键 {finiteness['missing_grad_rows']}）"
            )
        if skipped_max is None:
            notes.append(MSSWIFT_A7_RESIDUAL + detail)
    note = "；".join(item for item in notes if item) or None

    return {
        "tier": tier,
        "attempt": attempt,
        # 结果列「跑次根目录」写**对外可追溯的规范路径**（--run-root-display 提供），
        # 而不是本机镜像路径；实际读盘仍用 run_dir。
        "run_dir": (
            str(Path(display_root) / tier) if display_root else str(run_dir.resolve())
        ),
        "local_run_dir": str(run_dir.resolve()),
        "date": date,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "steps_actual": steps_actual,
        "planned_steps": planned_steps,
        "peak_mem_gib": peak_mem,
        "peak_mem_caliber": peak_caliber,
        "host_sample_peak_gib": _extract_host_sample_peak(run_dir),
        "max_context": max_context,
        "artifacts": _extract_artifacts(run_dir, stdout),
        "reloadable": reload_readout["reloadable"],
        "tensor_count": reload_readout["tensor_count"],
        "key_tensors_complete": reload_readout["key_tensors_complete"],
        "weight_changed": weight_changed,
        "optimizer_steps_all_positive": steps_positive,
        "loss_grad_nonfinite": nonfinite,
        "skipped_nonfinite_max": skipped_max,
        "nonfinite_source": skipped_source,
        "state_override": None,
        "failure_class": failure_class,
        "failure_evidence": failure_evidence,
        "runner_seconds": _extract_runner_seconds(run_dir),
        "msswift_finiteness": finiteness,
        "epoch_facts": epoch_facts,
        "algo": algo,
        "backend": backend,
        "html_planned_steps": html_planned_steps,
        "epoch_target_samples": epoch_target_samples,
        "epoch_samples_seen": (epoch_facts or {}).get("samples_seen"),
        "a4_uncalibrated": True,
        "a4_note": None,
        "note": note,
        "sources": {
            "steps": steps_source,
            "peak_mem": peak_caliber,
            "reload": reload_readout["source"],
            "weight": weight_source,
        },
    }


def load_readings(path: Path) -> dict[str, dict[str, Any]]:
    """读读数文件，返回 ``{tier: {"state":…, "impossible_reason":…, "runs":[…]}}``。

    接受：JSON 对象（``{"tiers": {...}}`` / ``{"runs": [...]}`` / 裸对象）、JSON 数组、
    或 JSONL。跑次条目至少要有 ``tier`` / ``attempt``；其余键与 :func:`extract_run` 同名。
    """
    text = _read_text(path)
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(payload, dict) and isinstance(payload.get("tiers"), dict):
        tiers: dict[str, dict[str, Any]] = {}
        for tier, entry in payload["tiers"].items():
            entry = entry if isinstance(entry, dict) else {}
            runs = entry.get("runs") or []
            if not isinstance(runs, list):
                raise ValueError(f"{path}: tiers[{tier}].runs 必须是数组")
            tiers[str(tier)] = {
                "state": entry.get("state"),
                "impossible_reason": entry.get("impossible_reason"),
                "runs": runs,
            }
        return tiers
    if isinstance(payload, dict):
        payload = payload.get("runs", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path}: 读数文件必须含 runs 数组")
    tiers = {}
    for run in payload:
        if not isinstance(run, dict):
            continue
        tier = run.get("tier")
        if not tier:
            raise ValueError(f"{path}: 跑次条目缺 tier")
        tiers.setdefault(str(tier), {"state": None, "impossible_reason": None, "runs": []})
        tiers[str(tier)]["runs"].append(run)
    return tiers


def _normalize_run(raw: dict[str, Any], tier: str) -> dict[str, Any]:
    """把读数文件里的一条跑次归一成与 :func:`extract_run` 同形的 dict。"""
    run = {
        "tier": tier,
        "attempt": raw.get("attempt"),
        "run_dir": raw.get("run_dir"),
        "date": raw.get("date"),
        "exit_code": raw.get("exit_code"),
        "timed_out": raw.get("timed_out"),
        "steps_actual": raw.get("steps_actual"),
        "planned_steps": raw.get("planned_steps"),
        "peak_mem_gib": raw.get("peak_mem_gib"),
        "peak_mem_caliber": raw.get("peak_mem_caliber"),
        "host_sample_peak_gib": raw.get("host_sample_peak_gib"),
        "max_context": raw.get("max_context"),
        "artifacts": raw.get("artifacts"),
        "reloadable": raw.get("reloadable"),
        "tensor_count": raw.get("tensor_count"),
        "key_tensors_complete": raw.get("key_tensors_complete"),
        "weight_changed": raw.get("weight_changed"),
        "optimizer_steps_all_positive": raw.get("optimizer_steps_all_positive"),
        "loss_grad_nonfinite": raw.get("loss_grad_nonfinite"),
        "skipped_nonfinite_max": raw.get("skipped_nonfinite_max"),
        "state_override": raw.get("state_override"),
        "failure_class": raw.get("failure_class"),
        "note": raw.get("note"),
        "sources": raw.get("sources") or {},
    }
    if not _is_num(run["attempt"]):
        raise ValueError(f"{tier}: 跑次条目缺 attempt（读数为 {raw!r}）")
    run["attempt"] = int(run["attempt"])
    if isinstance(run["artifacts"], dict):
        run["artifacts"] = {
            key: run["artifacts"].get(key) for key in ("checkpoint", "config_snapshot", "training_log")
        }
    return run


# ---------------------------------------------------------------------------
# 单元格渲染
# ---------------------------------------------------------------------------
def _fmt_float(value: Any) -> str:
    return f"{round(float(value), 1):g}" if _is_num(value) else DASH


def render_runs3(attempts: list[dict[str, Any] | None]) -> str:
    """渲染 ``①成功 ②失败 ③—``（保留现表的 ``<span class="run">`` 结构）。"""
    parts = []
    for index, mark in enumerate(RUN_MARKS):
        attempt = attempts[index] if index < len(attempts) else None
        verdict = attempt["verdict"] if attempt else None
        parts.append(f'<span class="run">{mark}{html_mod.escape(verdict or VERDICT_DASH)}</span>')
    return " ".join(parts)


def build_tier_result(
    tier: str,
    attempts_raw: list[dict[str, Any] | None],
    *,
    state_override: str | None = None,
    impossible_reason: str | None = None,
) -> dict[str, Any]:
    """聚合出一档的结果列值（全部为字符串，已可写进 HTML）。

    **修正令 §3**：``steps_actual`` 的分母是 ``run["planned_steps"]``——**配置级推导值**
    （见 :func:`_planned_config_denominator`），**不是** HTML 配置列现值、也**不是**运行时读数反推值。
    二者不一致时在备注显式登记配置列缺陷（**不改配置列**）。
    """
    judged = [judge_run(run) if run is not None else None for run in attempts_raw]
    state, reasons = aggregate_state(judged, state_override, impossible_reason)

    present = [run for run in attempts_raw if run is not None]
    first = attempts_raw[0] if attempts_raw else None

    peak_values = [run["peak_mem_gib"] for run in present if _is_num(run.get("peak_mem_gib"))]
    steps_source = next(
        (run for run in present
         if _is_num(run.get("steps_actual")) and _is_num(run.get("planned_steps"))),
        None,
    )
    if steps_source is not None:
        steps_text = f"{int(steps_source['steps_actual'])}/{int(steps_source['planned_steps'])}"
    else:
        only_actual = next((run for run in present if _is_num(run.get("steps_actual"))), None)
        steps_text = f"{int(only_actual['steps_actual'])}/?" if only_actual else DASH

    contexts = [run["max_context"] for run in present if _is_num(run.get("max_context"))]
    dates = [str(run["date"]) for run in present if run.get("date")]
    run_dirs = [str(run["run_dir"]) for run in present if run.get("run_dir")]

    failure = DASH
    if state == "失败" and first is not None and first.get("failure_class"):
        failure = str(first["failure_class"])
    elif state == "未完成测试":
        # 第 2/3 次失败（不稳定）时，如实记那一跑次的归因。
        for index in (1, 2):
            run = attempts_raw[index] if index < len(attempts_raw) else None
            if run is not None and judged[index] and judged[index]["verdict"] == VERDICT_FAIL:
                if run.get("failure_class"):
                    failure = str(run["failure_class"])
                break

    notes: list[str] = list(reasons)
    for run in present:
        if run.get("note"):
            notes.append(str(run["note"]))
    # ★ 修正令 §3：分母是配置级值；与 HTML 配置列现值不一致 ⇒ 显式登记（不改配置列）。
    html_planned = next(
        (run.get("html_planned_steps") for run in present
         if _is_num(run.get("html_planned_steps"))), None
    )
    config_planned = next(
        (run.get("planned_steps") for run in present if _is_num(run.get("planned_steps"))), None
    )
    if (html_planned is not None and config_planned is not None
            and int(html_planned) != int(config_planned)):
        notes.append(KNOWN_PLANNED_DEFECT_TEMPLATE.format(
            html=int(html_planned), config=int(config_planned)))

    # ★ 判据①一律 `steps >= 配置级分母`；GRASPO 的 epoch 事实**仅作核对**（用户 2026-09-25 裁定）。
    if any(str(run.get("algo") or "").upper() == "GRASPO"
           and run.get("backend") == "native" for run in present):
        notes.append(
            "native GRASPO 档：计划步数为**配置上界** ceil(floor(subset/dp_size)/Q)；"
            "epoch 事实仅作核对（不替代判据①）"
        )

    # ★ §1 硬前提（一个 epoch ≥5 个 optimizer step）——标注对象必须是**判据所依据的那一次跑次**
    #   （第 1 次），否则会与行内判据错位（审查 🟡 警告 1：原文案误用了 attempt2 的 2 步）。
    first_steps = attempts_raw[0].get("steps_actual") if attempts_raw and attempts_raw[0] else None
    if _is_num(first_steps) and int(first_steps) < MIN_EPOCH_OPTIMIZER_STEPS:
        notes.append(
            f"§1 **标注项（不参与判定）**：一个 epoch 的优化步数参考值 ≥"
            f"{MIN_EPOCH_OPTIMIZER_STEPS} —— **第 1 次跑次** {int(first_steps)} 步，低于该参考值"
            "（判据① 仍按「实际步数 ≥ 计划步数」判定；配置修法见 NEXT-ROUND-TODO.md (a)）"
        )

    # ★ 修正令 ②：多跑次已执行、但判据③读数缺失的档 —— runs3 与备注**如实披露**
    #   （回填前占位符与本档逐字节相同会让人误以为只跑了 1 次；此处**不伪造任何读数**）。
    runs3_text = render_runs3(judged)
    disclosure: str | None = None
    if (state == "未完成测试"
            and len(present) == 3
            and all(run.get("backend") != "native" for run in present)
            and all(_is_num(run.get("exit_code")) and int(run["exit_code"]) == 0 for run in present)
            and all((j or {}).get("c3") is None for j in judged if j is not None)):
        disclosure = MSSWIFT_RUNS3_DISCLOSURE
        runs3_text = (
            runs3_text
            + " " + f'<span class="runs3-note">{MSSWIFT_RUNS3_DISCLOSURE}</span>'
        )
        notes.append(MSSWIFT_RUNS3_DISCLOSURE)

    # ★ 修正令 🟡 警告 1 后半：T040 三次跑次明细与定性（唯一"临界不稳定"档）。
    #   ★★ 复审修正（2026-09-26）：**不得**把单次跑次称作"成功"——§1 定义「成功」须 ①②③ 全部成立；
    #   此处一律写「客观读数 + 该跑次 judged 判定 + 被否的判据与数值」，数值全部来自 run 字典，不臆造。
    if tier == "T040" and len([r for r in attempts_raw if r]) == 3:
        parts: list[str] = []
        for index, run in enumerate(attempts_raw, start=1):
            judge = judged[index - 1] if index - 1 < len(judged) else None
            if run is None:
                parts.append(f"第 {index} 次：无产物")
                continue
            seconds = run.get("runner_seconds")
            sec_txt = f"{round(seconds)}s" if _is_num(seconds) else "?"
            bits: list[str] = []
            if (judge or {}).get("c1") is False:
                if run.get("exit_code") not in (0, None):
                    bits.append(f"exit={run['exit_code']}≠0")
                if _is_num(run.get("steps_actual")) and _is_num(run.get("planned_steps"))                         and int(run["steps_actual"]) < int(run["planned_steps"]):
                    bits.append(
                        f"{int(run['steps_actual'])} < 计划步数 {int(run['planned_steps'])}"
                    )
                if _is_num(run.get("steps_actual"))                         and int(run["steps_actual"]) < MIN_EPOCH_OPTIMIZER_STEPS:
                    bits.append(
                        f"{int(run['steps_actual'])} < §1 标注阈值 {MIN_EPOCH_OPTIMIZER_STEPS}"
                    )
            if (judge or {}).get("c2") is False:
                bits.append("②被否（无可重载 checkpoint）")
            elif (judge or {}).get("c2") is None:
                bits.append("②读数缺失（未找到可重载 checkpoint）")
            if (judge or {}).get("c3") is False:
                bits.append("③被否")
            denied = [name for name, key in (("①", "c1"), ("②", "c2"), ("③", "c3"))
                      if (judge or {}).get(key) is False]
            head = f"第 {index} 次：rc={run.get('exit_code')}、{sec_txt}、{run.get('steps_actual')} 步"
            if run.get("exit_code") not in (0, None) and run.get("failure_class"):
                head += f"、{run['failure_class']}"
            if bits:
                head += "（" + "；".join(bits) + "）"
            verdict = (judge or {}).get("verdict") or "不可判定"
            parts.append(head + f"、{'/'.join(denied)}被否 ⇒ {verdict}" if denied else head + f" ⇒ {verdict}")
        notes.append(
            "T040 三次跑次明细（定性：**失败·临界不稳定**）："
            + "；".join(parts)
            + "；判据以第 1 次为准（4 < 计划步数 6）⇒ ①被否 ⇒ 记失败；"
              "根因 = 单卡 27B LoRA 显存贴边"
        )

    # ★ 后端变更说明（集中表注入；**追加**，不覆盖原有备注）
    if tier in BACKEND_CHANGE_NOTES:
        notes.append(BACKEND_CHANGE_NOTES[tier])

    return {
        "tier": tier,
        "state": state,
        "reasons": reasons,
        "peak_mem": _fmt_float(max(peak_values)) if peak_values else DASH,
        "steps_actual": steps_text,
        "ctx_max": str(max(int(v) for v in contexts)) if contexts else DASH,
        "runs3": runs3_text,
        "date": max(dates) if dates else DASH,
        "run_root": "<br>".join(html_mod.escape(p) for p in run_dirs) if run_dirs else DASH,
        "failure": failure,
        "note": "；".join(notes) if notes else DASH,
        "judged": judged,
        "attempts": attempts_raw,
    }

def _retag_status(tag: str, state: str) -> str:
    """给 status 的 ``<td>`` 追加/替换状态着色 class（只动结果列自己的标签）。"""
    st_class = STATE_CLASS[state]
    match = re.search(r'class="([^"]*)"', tag)
    if match:
        classes = [c for c in match.group(1).split() if not c.startswith("st-")]
        if st_class not in classes:
            classes.append(st_class)
        return tag[:match.start()] + f'class="{" ".join(classes)}"' + tag[match.end():]
    return tag[:-1] + f' class="{st_class}">'


def apply_results(
    html_text: str,
    results: dict[str, dict[str, Any]],
) -> str:
    """把结果列写进 HTML 文本，并断言除结果列外零改动。"""
    rows = parse_matrix(html_text)
    replacements: list[tuple[int, int, str]] = []
    for tier, result in results.items():
        row = rows.get(tier)
        if row is None:
            raise KeyError(f"HTML 表里找不到档 {tier}")
        for col in RESULT_COLUMNS:
            cell = row.get(col)
            if cell is None:
                raise KeyError(f"{tier}: HTML 表里找不到结果列 {col}")
            if col == "status":
                # 整个 <td> 元素都可改（含着色 class），它本身就是结果列。
                new = _retag_status(cell["tag"], result["state"]) + html_mod.escape(
                    result["state"]
                ) + "</td>"
                replacements.append((cell["tag_start"], cell["tag_end"], new))
            else:
                value = result[col]
                if col == "runs3":
                    new = value  # 已含 <span class="run">，无需转义
                elif col == "run_root":
                    new = value  # 已逐段转义 + <br>
                else:
                    new = html_mod.escape(value)
                replacements.append((cell["content_start"], cell["content_end"], new))

    replacements.sort(key=lambda item: item[0], reverse=True)
    out = html_text
    for start, end, new in replacements:
        out = out[:start] + new + out[end:]
    assert_result_only_change(html_text, out)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _tier_meta_from_html(rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """从 HTML 取每档的**配置级**元数据（修正令 §2：分母只取配置级量）。

    新增 ``algo`` 与 ``subset_size``：前者决定判据①是否走「GRASPO 跑满 epoch」分支，
    后者是配置级分母推导的输入。**只读，不写**配置列（§5.2）。
    """
    meta: dict[str, dict[str, Any]] = {}
    for tier, row in rows.items():
        planned_raw = row.get("planned_steps", {}).get("text", DASH)
        subset_raw = row.get("subset_size", {}).get("text", DASH)
        meta[tier] = {
            "backend": row.get("backend", {}).get("text", ""),
            "mode": row.get("mode", {}).get("text", ""),
            "algo": row.get("algo", {}).get("text", ""),
            "cards": (int(row["cards"]["text"]) if row.get("cards", {}).get("text", "").isdigit() else None),
            "planned_steps": int(planned_raw) if planned_raw.isdigit() else None,
            "subset_size": int(subset_raw) if subset_raw.isdigit() else None,
        }
    return meta


def _not_applicable_tiers(config_dir: Path | None) -> set[str]:
    if config_dir is None or not config_dir.is_dir():
        return set()
    return {p.name.split(".")[0] for p in config_dir.glob("T*.not_applicable.md")}


#: §1「一个 epoch ≥ 5 个 optimizer step」（``docs/capability-matrix.html:91``）。
#: ★ 2026-09-26 用户放行后，该条已改为**标注项**（**不参与判定**）⇒ **仅用于备注标注**，
#:   ``judge_c1`` 不再引用它（判据① 只比 ``steps_actual >= 计划步数``）。
MIN_EPOCH_OPTIMIZER_STEPS = 5

#: 档级失败归因（显式、可审计；**仅在产物事实串匹配不到时**兜底）。依据：审查报告（2026-09-26）
#: 🔴 阻断项「失败必带结构化归因（HTML §2:100）」+ 指挥官修正令。
FAILURE_CLASS_BY_TIER: dict[str, tuple[str, str]] = {
    "T034": (
        "我方实现",
        "逐步 trainable_norm_delta 5/5 全 0、optimizer_steps=1、grad_norm 非零 "
        "⇒ 可训练参数未被更新（同批 T016/T017/T018 同指标非零可对照）",
    ),
    "T040": (
        "资源不足",
        "单卡 27B LoRA 显存余量不足 ⇒ 三次跑次三种表征（4 步 < 硬前提 5 / step2 真 OOM / 5 步跑满），"
        "临界不稳定",
    ),
}

#: 修正令（2026-09-26）②：多跑次已执行但判据③读数缺失时的**如实披露**文案
#: （**不得伪造读数**：只披露"确实跑了 3 次"与缺失原因）。
MSSWIFT_RUNS3_DISCLOSURE = (
    "3 次跑次均 rc=0（第 2/3 次确已执行，非未跑）；"
    "判据③读数缺失（ms-swift 后端不上报非有限跳过计数，A7 口径不可测）"
)

#: **后端变更说明表**（档号 → 说明文本）—— 一处集中定义，便于日后同类变更复用。
#: 背景：这 12 档**实际已改用 FSDP2 运行**，但 HTML 配置列由 `--config-dir samples/configs/matrix54-v2`
#: 推导 ⇒ **仍显示冻结的 DeepSpeed 配方**（HTML 无 `fsdp` 列）。依据：主席 2026-09-27 指令。
#: 处置：**配置列不动**（§5.2 冻结规则），在**备注列追加**说明（**保留原有备注**）。
BACKEND_CHANGE_NOTE = (
    "**运行后端已由 DeepSpeed 改为 FSDP2**（依据主席 2026-09-27 指令）。"
    "**本档实际以 FSDP2 运行；配置列中的 `deepspeed: zero2[_offload]` 与 offload 列的 `zero2` 字样"
    "仅反映冻结的 v2 配方，不是本轮的运行配方。**"
)
BACKEND_CHANGE_TIERS: tuple[str, ...] = (
    "T004", "T005", "T006", "T019", "T020", "T021",
    "T037", "T038", "T039", "T049", "T050", "T051",
)
BACKEND_CHANGE_NOTES: dict[str, str] = {tier: BACKEND_CHANGE_NOTE for tier in BACKEND_CHANGE_TIERS}


#: 修正令 §3：结果列分母与 HTML 配置列不一致时的显式登记文案（**不改配置列**）。
KNOWN_PLANNED_DEFECT_TEMPLATE = (
    "配置列 planned_steps={html}（已知推导缺陷，待用户批准后修正）；"
    "本结果列分母取配置级推导值 {config}"
)


def _yaml_scalar(value: str):
    """极简标量解析（保持工具零第三方依赖）。"""
    text = value.strip()
    if text in ("", "null", "~"):
        return None
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for _cast in (int, float):
        try:
            return _cast(text)
        except ValueError:
            continue  # 不是该类型 ⇒ 显式落到下一个转换（不静默吞掉）
    return text.strip().strip(chr(34)).strip(chr(39))


def _parse_simple_yaml(text: str) -> dict:
    """**极简 YAML 子集**（只支持 ``key: value`` 与两级缩进嵌套）——保持工具 stdlib-only。

    只服务 ``samples/configs/matrix54-v2/*.yaml`` 的固定结构
    （``backend`` / ``train_method`` / ``native.*`` / ``training.*``）。
    解析不到的键一律**缺失**（调用方 fail-closed），**不猜**。
    """
    out: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        value = value.split("#", 1)[0].strip()
        if indent == 0:
            if value == "":
                section = key.strip()
                out.setdefault(section, {})
            else:
                section = None
                out[key.strip()] = _yaml_scalar(value)
        elif section is not None and isinstance(out.get(section), dict):
            out[section][key.strip()] = _yaml_scalar(value)
    return out


def _load_tier_config(config_dir: Path | None, tier: str) -> dict:
    if config_dir is None:
        return {}
    path = config_dir / f"{tier}.yaml"
    if not path.is_file():
        return {}
    return _parse_simple_yaml(_read_text(path))


def _planned_config_denominator(meta_entry: dict, cfg: dict) -> tuple[int | None, str]:
    """**配置级**计划步数分母（修正令 §2）——只依赖配置与 HTML 配置列，**绝不**用运行时读数。

    * native SFT/CPT/OPD：``floor( floor(subset/dp_size) / (micro_batch × GA) )``
      （``sft_trainer.py:160-166`` 只按 DP 分片；``:204-208`` 有效批；``:247-262`` 丢末批 + 跨 DP 取 MIN）
    * native GRASPO：``ceil( floor(subset/dp_size) / Q )``——**配置上界，仅作展示**；
      判据①改以「跑满 epoch」为准（见 :func:`judge_c1`）
    * ms-swift：沿用 HTML 配置列值（v1 口径，本轮不复核）

    取不到 ⇒ ``(None, 原因)``；调用方回落 HTML 配置列并**显式标注来源**。
    """
    subset = meta_entry.get("subset_size")
    if not _is_num(subset) or int(subset) <= 0:
        return None, "HTML 配置列未给可用的 subset_size"
    backend = meta_entry.get("backend") or cfg.get("backend")
    algo = (meta_entry.get("algo") or "").upper()
    if backend != "native":
        html_planned = meta_entry.get("planned_steps")
        if _is_num(html_planned):
            return int(html_planned), "ms-swift：沿用 HTML 配置列（v1 口径）"
        return None, "ms-swift：HTML 配置列未给 planned_steps"
    native = cfg.get("native") or {}
    dp_size = max(1, int(native.get("dp_size") or 1))
    per_rank = int(subset) // dp_size
    training = cfg.get("training") or {}
    if algo == "GRASPO":
        raw_q = training.get("rollout_queue_batch_size")
        q = int(raw_q) if _is_num(raw_q) else 8
        return max(1, -(-per_rank // max(1, q))), (
            f"配置级 GRASPO 上界：ceil(floor({subset}/{dp_size})/Q {q})"
        )
    raw_mb = native.get("micro_batch_size")
    mb = max(1, int(raw_mb)) if _is_num(raw_mb) else 1
    raw_ga = training.get("gradient_accumulation_micro_batches")
    ga = max(1, int(raw_ga)) if _is_num(raw_ga) else 1
    return max(1, per_rank // (mb * ga)), (
        f"配置级 SFT/全参：floor(floor({subset}/{dp_size})/(mb {mb}×GA {ga}))"
    )


def _load_run_meta(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    """读 ``--run-meta``：只允许注入产物里没有的三列（实测日期/最大可行上下文/失败归因）。

    接受数组 / ``{"runs": [...]}`` / ``{"T001": {...}}`` 三种形态；每条要有 ``tier``，
    ``attempt`` 缺省为 1。**不得**携带判据读数（那些只能来自 ``--run-root`` 产物）——
    本函数只认 ``date`` / ``max_context`` / ``failure_class`` 三个键。
    """
    if path is None or not path.is_file():
        return {}
    text = _read_text(path)
    try:
        raw: Any = json.loads(text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(raw, dict) and isinstance(raw.get("runs"), list):
        raw = raw["runs"]
    elif isinstance(raw, dict):
        raw = [dict(value, tier=key) for key, value in raw.items() if isinstance(value, dict)]
    if not isinstance(raw, list):
        raise ValueError(f"{path}: --run-meta 必须是数组 / {{runs: [...]}} / {{tier: {{...}}}}")
    allowed = ("date", "max_context", "failure_class")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("tier"):
            continue
        attempt = entry.get("attempt")
        number = int(attempt) if _is_num(attempt) else 1
        out[(str(entry["tier"]), number)] = {k: entry[k] for k in allowed if k in entry}
    return out


def _load_tier_root_map(path: Path | None) -> dict[str, dict[str, list[Any]]]:
    """读 ``--tier-root-map``：**逐档产根映射**（支持"不同档用不同产根"）。

    形态（宽容解析）::

        {"T001": ["<pass1 root>", "<pass2 root>", "<pass3 root>"],   # 顺序即跑次号 1/2/3
         "T031": {"roots": [...], "display": ["<228 真实路径>", ...]}}

    语义（**最小改动**，不改任何 ``judge_*`` 与四态聚合）：
      · **只对列出的档生效**；未列出的档**回退** ``--run-root`` 的顺序序列（保持既有行为）；
      · 每档最多 3 个根，**顺序即跑次号**；少于 3 个 ⇒ 缺的跑次按"无产物"处理（读数缺失，不放宽）；
      · 结果列「跑次根目录」的显示路径取 ``display``（缺省 = 对应 root 字符串本身）。
      · ⚠ **口径单元约束**：同一档的 3 个根必须属同一口径单元（同镜像 + 同后端 + 同配置 + 同卡池）
        —— 跨口径单元的跑次不得混算"连续 3 次"（见台账 `LEDGER-ENTRY-multi-root.md`）。
    """
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"FATAL: --tier-root-map 不存在：{path}")
    raw = json.loads(_read_text(path))
    if not isinstance(raw, dict):
        raise SystemExit("FATAL: --tier-root-map 顶层必须是对象：{tier: [root, ...]}")
    out: dict[str, dict[str, list[Any]]] = {}
    for tier, value in raw.items():
        roots: list[str]
        display: list[str] | None = None
        if isinstance(value, dict):
            roots = [str(x) for x in (value.get("roots") or [])]
            if value.get("display"):
                display = [str(x) for x in value["display"]]
        elif isinstance(value, list):
            roots = [str(x) for x in value]
        else:
            raise SystemExit(f"FATAL: --tier-root-map[{tier}] 必须是数组或 {{roots,display}}")
        if not roots:
            raise SystemExit(f"FATAL: --tier-root-map[{tier}] 为空列表")
        if len(roots) > 3:
            raise SystemExit(f"FATAL: --tier-root-map[{tier}] 超过 3 个根（跑次号只有 1/2/3）")
        pairs = [
            (Path(roots[i]), (display[i] if display and i < len(display) else None))
            for i in range(len(roots))
        ]
        out[str(tier)] = {"pairs": pairs, "roots": roots}
    return out


def _check_run_roots(roots: list[str], require: bool) -> list[str]:
    """产根防呆（§2.4）：不存在 / 空目录 ⇒ 大声警告；``--require-run-roots`` 下直接失败。

    为什么必须有（实测）：``--run-root`` 的**位置即跑次号**，而缺失的根只会静默变成
    「读数缺失」⇒ 已完成档被写成「未完成测试」而不报错；传错层级（漏 ``/r1``）同理。
    """
    problems: list[str] = []
    for index, root in enumerate(roots):
        path = Path(root)
        if not path.is_dir():
            problems.append(f"第 {index + 1} 个产根不存在：{path}")
        elif not any(child.is_dir() for child in path.iterdir()):
            problems.append(f"第 {index + 1} 个产根下没有任何档目录：{path}")
    for item in problems:
        print(f"WARNING[产根防呆] {item}（该跑次将按「读数缺失」处理）", file=sys.stderr)
    if problems and require:
        raise SystemExit(
            "FATAL: --require-run-roots 下产根不自洽，拒绝回填"
            "（避免静默把已完成档写成「未完成测试」）"
        )
    return problems


def build_results(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    html_text = args.html.read_text(encoding="utf-8")
    rows = parse_matrix(html_text)
    meta = _tier_meta_from_html(rows)

    # 0) 产根防呆 + 实测注入通道
    _map_for_check = _load_tier_root_map(getattr(args, "tier_root_map", None))
    _all_roots = [str(r) for r in (args.run_root or [])]
    _all_roots += [str(p) for entry in _map_for_check.values() for p, _d in entry["pairs"]]
    _all_roots = list(dict.fromkeys(_all_roots))   # 去重（同一根被多档引用时只报一次）
    root_problems = _check_run_roots(_all_roots, bool(args.require_run_roots))
    run_meta = _load_run_meta(args.run_meta)

    # 1) 跑批产根（顺序 = 跑次号）抽读数
    extracted: dict[str, dict[int, dict[str, Any]]] = {}
    planned_sources: dict[str, str] = {}
    displays = list(getattr(args, "run_root_display", None) or [])
    ledgers = list(getattr(args, "ledger", None) or [])
    # ★ 逐档产根映射：未列出的档回退 --run-root 顺序序列（既有行为不变）
    root_map = _load_tier_root_map(getattr(args, "tier_root_map", None))
    fallback: list[tuple[Path, str | None]] = [
        (Path(r), displays[i] if i < len(displays) else None)
        for i, r in enumerate(args.run_root or [])
    ]
    for tier in sorted(meta):
        entry = root_map.get(tier)
        pairs = list(entry["pairs"]) if entry else list(fallback)
        for attempt, (root, display_root) in enumerate(pairs[:3], start=1):
            run_dir = root / tier
            if not run_dir.is_dir():
                continue
            cfg = _load_tier_config(args.config_dir, tier)
            # ★ 2026-09-27：收集器产物（ledger）按「档 × 跑次」定位——
            #   先由产根推断（<base>/batch/ledger/ledger.jsonl 等），失败再用 --ledger 的第 N 个
            _ledger_path = _find_ledger(run_dir)
            if _ledger_path is None and ledgers and attempt - 1 < len(ledgers):
                _ledger_path = Path(ledgers[attempt - 1])
            _ledger_rows = _load_ledger_rows(_ledger_path)
            planned, planned_source = _planned_config_denominator(meta[tier], cfg)
            if planned is None:
                planned = meta[tier]["planned_steps"]
                planned_source = f"回落 HTML 配置列（{planned_source}）"
            planned_sources.setdefault(tier, planned_source)
            extracted.setdefault(tier, {})[attempt] = extract_run(
                run_dir, tier, attempt, planned,
                meta[tier]["backend"], meta[tier]["mode"],
                run_meta.get((tier, attempt)),
                meta[tier]["algo"], cfg, meta[tier]["planned_steps"], display_root,
                _ledger_rows, meta[tier].get("cards"),
            )
            if args.a4_residual_note:
                run = extracted[tier][attempt]
                run["a4_note"] = A4_RESIDUAL
                run["note"] = "；".join(
                    item for item in (run.get("note"), A4_RESIDUAL) if item
                )

    # 2) 读数文件覆盖/补充
    tier_overrides: dict[str, dict[str, Any]] = {}
    if args.readings:
        for tier, entry in load_readings(Path(args.readings)).items():
            tier_overrides.setdefault(tier, {})["state"] = entry.get("state")
            tier_overrides[tier]["impossible_reason"] = entry.get("impossible_reason")
            for raw in entry["runs"]:
                run = _normalize_run(raw, tier)
                extracted.setdefault(tier, {})[run["attempt"]] = run

    not_applicable = _not_applicable_tiers(args.config_dir)
    active = set(extracted) | set(tier_overrides) | not_applicable
    if args.all_tiers:
        active |= set(meta)

    results: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for tier in sorted(active):
        if args.tiers and tier not in args.tiers:
            continue
        attempts_by_number = extracted.get(tier, {})
        attempts: list[dict[str, Any] | None] = [
            attempts_by_number.get(number) for number in (1, 2, 3)
        ]
        override = tier_overrides.get(tier, {}).get("state")
        impossible = tier_overrides.get(tier, {}).get("impossible_reason")
        if tier in not_applicable and not impossible:
            impossible = f"{tier}.not_applicable.md：该配置逻辑上不适用"
        result = build_tier_result(
            tier, attempts, state_override=override, impossible_reason=impossible
        )
        results[tier] = result
        evidence.append({
            "tier": tier,
            "state": result["state"],
            "reasons": result["reasons"],
            "attempts": [
                {"attempt": number, "run": run, "judged": result["judged"][number - 1]}
                for number, run in zip((1, 2, 3), attempts)
            ],
            "written": {
                "peak_mem": result["peak_mem"],
                "steps_actual": result["steps_actual"],
                "ctx_max": result["ctx_max"],
                "status": result["state"],
                "runs3": result["runs3"],
                "date": result["date"],
                "run_root": result["run_root"],
                "failure": result["failure"],
                "note": result["note"],
            },
        })

    summary = {
        "html": str(args.html),
        "run_roots": [str(Path(r).resolve()) for r in (args.run_root or [])],
        "root_problems": root_problems,
        "run_meta_keys": sorted(f"{tier}/#{number}" for tier, number in run_meta),
        "planned_sources": dict(sorted(planned_sources.items())),
        "tier_root_map": {t: v["roots"] for t, v in sorted(root_map.items())},
        "filled_tiers": sorted(results),
        "state_counts": {
            state: sum(1 for r in results.values() if r["state"] == state) for state in STATES
        },
    }
    return results, {"tiers": evidence, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML,
                        help=f"回填目标 HTML（默认 {DEFAULT_HTML}）")
    parser.add_argument("--run-root", action="append", default=[], metavar="RUN_ROOT",
                        help="跑批产根；可重复，**顺序即跑次号**（第 1/2/3 次）")
    parser.add_argument("--readings", type=Path, default=None,
                        help="读数文件（JSON/JSONL，供影本自测或外部 collector 注入；同名跑次覆盖产根抽取）")
    parser.add_argument("--ledger", action="append", default=[], metavar="LEDGER.jsonl",
                        help="收集器产物 ledger.jsonl（可重复，按跑次号）；"
                             "默认由产根自动推断 <base>/batch/ledger/ledger.jsonl")
    parser.add_argument("--tier-root-map", type=Path, default=None, metavar="JSON",
                        help="**逐档产根映射**（JSON 对象，形如 T001: [root1, root2, root3]）；"
                             "顺序即跑次号 1/2/3；未列出的档回退到 --run-root 的顺序序列。"
                             "用于「不同档来自不同产根」（如 ms-swift 走 v2-r4、native 走 v2-r1/r2/r3）")
    parser.add_argument("--run-root-display", action="append", default=[], metavar="DISPLAY",
                        help="与 --run-root 按序对应的**对外显示路径**（写进结果列「跑次根目录」）；"
                             "缺省则用实际传入路径（若传的是本机镜像路径，务必显式给规范路径）")
    parser.add_argument("--run-meta", type=Path, default=None,
                        help="实测注入通道（JSON/JSONL）：{tier, attempt, date?, max_context?, failure_class?}；"
                             "只注入产物里没有的三列（实测日期/最大可行上下文/失败归因），不覆盖判据读数")
    parser.add_argument("--require-run-roots", action="store_true",
                        help="产根防呆：任一 --run-root 不存在或为空 ⇒ 直接失败（默认只警告，避免静默降级）")
    parser.add_argument("--a4-residual-note", action="store_true",
                        help="在各档备注里登记「A4 权重指纹未标定」残差（默认不写；登记不改变任何判定）")
    parser.add_argument("--out-jsonl", type=Path, default=None,
                        help="证据汇总 JSONL 输出路径（每档一行：读数 + 判定中间量 + 写入值）")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help=f"配置目录（用于识别 *.not_applicable.md ⇒ 条件不合逻辑；默认 {DEFAULT_CONFIG_DIR}）")
    parser.add_argument("--tiers", nargs="*", default=None,
                        help="只处理这些档号（默认：有读数的档 + 显式条件不合逻辑档）")
    parser.add_argument("--all-tiers", action="store_true",
                        help="对 HTML 里全部档都写结果（无读数 ⇒ 未完成测试）")
    parser.add_argument("--dry-run", action="store_true", help="只计算并打印摘要，不写任何文件")
    return parser


def _write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in evidence["tiers"]:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.html.is_file():
        print(f"FATAL: HTML 不存在：{args.html}", file=sys.stderr)
        return 2

    results, evidence = build_results(args)
    summary = evidence["summary"]
    print(f"HTML: {args.html}")
    print(f"跑批产根（跑次号按顺序）：{summary['run_roots'] or '（无）'}")
    print(f"回填档数：{len(results)}；状态分布：{summary['state_counts']}")
    for tier, result in results.items():
        judged = result["judged"]
        marks = "".join(
            f"{RUN_MARKS[i]}{(j or {}).get('verdict') or VERDICT_DASH}" for i, j in enumerate(judged)
        )
        print(f"  {tier}: {result['state']}  [{marks}]  {result['note']}")

    if args.dry_run:
        print("dry-run：未写任何文件")
        return 0

    html_text = args.html.read_text(encoding="utf-8")
    updated = apply_results(html_text, results)
    args.html.write_text(updated, encoding="utf-8")
    print(f"已写入：{args.html}（除结果列外逐格一致断言通过）")

    if args.out_jsonl:
        _write_evidence(args.out_jsonl, evidence)
        print(f"证据汇总：{args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
