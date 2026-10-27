#!/usr/bin/env python3
"""「实跑步数」读数的**唯一真相源**（消费方：`scripts/make_result_json.py` · `scripts/collect_results.py`）。

为什么单独成件（2026-10-28 · N3b）：步数正则此前是**两份副本** ——
`make_result_json.py::GLOBAL_STEP_RE/GLOBAL_STEP_SLASH_RE/TQDM_STEP_RE` 与
`collect_results.py::_GLOBAL_STEP`。两份的**口径必然漂移**（§1.4 单一真相源）：
D2 只把前者扩到 ms-swift 的带引号形态，后者仍是旧口径 ⇒ 同一份日志两个消费者给出两个答案。
本件把「认哪些形态 / 怎么挑 / provenance 怎么记」收在一处，两处只调不抄。

## 认可的形态族（每一种都有实盘出处，⛔ 不凭空造）

| # | 形态 | 实盘出处 | `kind` |
|---|------|----------|--------|
| ① | `'global_step/max_steps': '6400/6400'`（值是**带引号字符串**，单/双引号皆可） | ms-swift 指标 dict（M016 `stdout.log` 1282 行） | `global_step/max_steps` |
| ② | tqdm 完成/中途帧 `Train: 100% <竖线> ███ <竖线> 6400/6400 [` | ms-swift `stdout.log` | `tqdm_progress` |
| ③ | `'global_step': '200'` / `"global_step": 200` | 同一 dict 家族 / `logging.jsonl` 末行 | `global_step` |
| ④ | `trainer_state.json` 的 `"global_step": 6400` | HF/ms-swift 检查点（与 ③ 共用 `GLOBAL_STEP_RE`） | `trainer_state.global_step` |
| ⑤ | `"step": N` | **native 后端**逐字（`flow/backend/native/trainer/{optimize,sft_trainer,trainer,rollout}.py`） | `native_step` |

★ ⑤ 的**语义登记（⛔ 不许当 ④ 用）**：native 的 `"step"` 是**单进程（rank0）训练步计数**，
不是 HF 的 `global_step`。之所以仍可作 `steps_actual`：native 每个训练步在**每个 rank 上各推进
一次**（tp/pp 只切权重、dp 只切数据，步数在所有 rank 上一致），故日志里该读数的**最大值**
= 本次实跑的训练步数（M007 tp2pp1dp1 400 步 / M008 tp2pp1dp2 200 步 实测两次复现）。
关键词里带 `native_` 前缀就是为了让人读 provenance 时**看得见**这不是 global 读数。

## 挑候选的口径（两个消费者共用的唯一实现）

* 多源并存（stdout.log ∪ 各 `trainer_state.json`）⇒ 取**数值最大**者（不是"第一条"，也**不是**
  "末条"——日志被截断/交错时末条未必最大）；
* 数值相同 ⇒ 取**更权威的形态**（`STEP_KIND_RANK`：显式 `global_step/max_steps` >
  `trainer_state.json` / `global_step` / native `step` > tqdm 进度条）；
* provenance 逐条可复核：`来源:行号（形态 …，原文 …）`，行号按 `\\n` 计（与 `grep -n` 同口径）。

## fail-closed（⛔ 三条禁令）

一个步数读数都没有 ⇒ :func:`best_of` 返回 `None`（调用方判"不可回填"）。
⛔ 不猜（不从 batch/epoch 换算）、⛔ 不用 `0` 兜底、⛔ 不把「末条读数」当总步数。
"""
from __future__ import annotations

import re
from pathlib import Path

#: ③④ `global_step`（值可为带引号字符串或裸数字）。D2 之前的老口径卡在值前面那个 `'` 上。
GLOBAL_STEP_RE = re.compile(r"['\"]?global_step['\"]?\s*[:=]\s*['\"]?(\d+)")
#: ① `global_step/max_steps`（分子 = 当前步；分母 = 计划步数，可缺）。
GLOBAL_STEP_SLASH_RE = re.compile(
    r"global_step\s*/\s*max_steps['\"]?\s*[:=]\s*['\"]?(\d+)(?:\s*/\s*(\d+))?")
#: ② tqdm 进度条里的「已跑/总数」。**必须锚住训练条**（`Train:` 标签 + `|计数 [` 框线）：
#:   日志里还有下载/存盘等**无关进度条**（M016 实测 stdout.log 里有 7724 处 `|N/M [`，
#:   其中绝大多数来自 `Train:` 之外）——裸扫 `|\s*\d+/\d+\s*\[` 会把无关计数当步数，
#:   而口径是"多源取最大" ⇒ 一个偏大的假读数会**静默污染** steps_actual。
#:   ⛔ 无标签的 HF 风格条（`  5%|▌ | 320/6400 [`）不取：那种情形有 ①③④ 兜底，宁缺勿错。
TQDM_STEP_RE = re.compile(
    r"(?:^|[\r\n])\s*(?:Train|Training):[^\r\n]*?\|\s*(\d+)\s*/\s*(\d+)\s*\[")
#: ⑤ native 的裸 `"step": N`。**必须用"键边界"锚住**（`(?<![A-Za-z0-9_])`）：
#:   不加锚点的话 `global_step` / `first_step` / `sft_step` 里的 `step` 会被当成第五形态，
#:   同一行出两个候选 —— provenance 记错来源还算小事，真正危险的是把
#:   `"max_steps": 800`（计划值）这类**不是步数**的键读成已跑步数（`steps` 后面的 `s` 挡住了
#:   `[:=]`，但 `"save_step": 50` 这种写法挡不住）⇒ 静默污染。
#:   值同样允许带引号（native 目前打裸数字，留出字符串形态不额外增加假阳性面）。
NATIVE_STEP_RE = re.compile(
    r"(?<![A-Za-z0-9_])['\"]?step['\"]?\s*[:=]\s*['\"]?(\d+)")

#: 同一读数出现在多处（且数值相等）时，**取更权威的形态**记 provenance：
#: 显式 `global_step/max_steps` dict > `trainer_state.json` / `global_step` / native `step` > tqdm。
STEP_KIND_RANK = {
    "global_step/max_steps": 3,
    "trainer_state.global_step": 2,
    "global_step": 2,
    "native_step": 2,
    "tqdm_progress": 1,
}

def scan_text(text: str, source: str) -> list[dict]:
    """在一段文本里逐行找步数读数 ⇒ 候选列表（`value` / `line` / `kind` / `raw` / `source`）。

    逐行（而不是整篇 `finditer`）的唯一原因：provenance 要报**哪一行**（审计可复现）。
    ⚠ 调用方必须按 `\\n` 切行（⛔ `splitlines()`）：tqdm 用 `\\r` 刷新进度帧，`splitlines()`
      会把同一**物理行**切成几十份 ⇒ provenance 行号与 grep/sed 口径不符、肉眼复核不到。
    """
    out: list[dict] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        if "global_step" in line:
            for m in GLOBAL_STEP_SLASH_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "global_step/max_steps", "raw": m.group(0)[:60],
                            "max_steps": int(m.group(2)) if m.group(2) else None,
                            "source": source})
            for m in GLOBAL_STEP_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "global_step", "raw": m.group(0)[:60],
                            "source": source})
        elif "Train:" in line or "Training:" in line:
            for m in TQDM_STEP_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "tqdm_progress", "raw": m.group(0)[:60],
                            "source": source})
        if "step" in line:
            # ★ 与上面的 `elif` **并列**（不是嵌套）：native 的 `"step": N` 与 ms-swift 的
            #   `global_step` 家族是两条独立的形态轴，一行里可能同时出现（回填器自己写的
            #   混合日志就是这样）——用 elif 会吞掉后者。
            for m in NATIVE_STEP_RE.finditer(line):
                out.append({"value": int(m.group(1)), "line": lineno,
                            "kind": "native_step", "raw": m.group(0)[:60],
                            "source": source})
    return out


def best_of(candidates: list[dict]) -> dict | None:
    """挑"最有代表性"的那条读数：**数值最大**，同值取更权威形态。空列表 ⇒ `None`。"""
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c["value"], STEP_KIND_RANK.get(c["kind"], 0)))


def origin_of(best: dict) -> str:
    """provenance 串：`来源:行号（形态 …，原文 …）`——审计不必再翻日志。"""
    where = (f"{best['source']}:{best['line']}" if best.get("line") is not None
             else f"{best['source']}")
    return f"{where}（形态 {best['kind']}，原文 {str(best['raw']).strip()!r}）"


def read_text(path: Path) -> str:
    """按**字节**读再解码（⛔ `read_text()`）：文本模式会把 `\\r` 规范化成 `\\n`，
    行号就与 grep/sed 口径不符（provenance 里的行号必须能被人直接复核）。"""
    return path.read_bytes().decode("utf-8", "replace")


def scan_file(path: Path, source: str | None = None) -> list[dict]:
    """扫一份日志文件 ⇒ 候选列表（文件缺失 ⇒ 空列表，⛔ 不报 0）。"""
    if not path.is_file():
        return []
    return scan_text(read_text(path), source if source is not None else path.name)
