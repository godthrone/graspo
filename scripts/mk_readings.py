#!/usr/bin/env python3
"""由 `readings.raw.json` + `exit_code` / `elapsed_sec` **合成** `<run-dir>/readings.json`。

## 为什么有这件（D23，2026-10-28）

`readings.json` 是 `make_result_json.py` / `record_checks.py` 的**唯一读数入口**
（缺它 ⇒ `FAIL: readings.json 缺失或不可解析`）。但它的合成此前是**手工步**：
跑批编排只落 `readings.raw.json`（runner 写）+ `exit_code` / `exit_code_outer` / `elapsed_sec`
（容器落盘），把两者合成 `readings.json` 的那一步**没人调用** ⇒ 漏补一次就卡住取证链
（M013 / M018 两次人工补过，`mk_evidence` rc=2）。本件把这一步变成**可被链自动调用**的一步：
`tools/matrix/fill_one.py` 在调用 `make_result_json.py` **之前**无条件先跑它（见该文件 ②′）。

## 口径（⛔ 一个数值都不改，只做搬运）

* `exit_code` = 容器内落盘的 `exit_code`（训练真实结果）；缺 ⇒ `exit_code_outer`；两者都缺 ⇒ `99`
  （沿用既有哨兵，⛔ 不用 0 冒充"成功"）；
* `elapsed_sec` / `gpus` / `config` / `image` / `timeout_sec` / `repo` / `command` / `note`
  = 从 `readings.raw.json` 原样搬运；
* `checks` / `checks_evidence` / `teacher_after_acc` = **保留**盘上现值（它们的唯一产出者是
  `scripts/record_checks.py`；本件 ⛔ 不写、⛔ 不清空，否则会把判据③④ 的证据洗掉）。

## 两条硬约束

1. **幂等**：同一份输入重复跑 ⇒ 逐字节相同，且第二次**不写盘**（打印 `unchanged`）。
   实现方式是"先算字节、再比对、只在不同时写"，⛔ 不是"写完再看运气"。
2. **有输入证据才动**：只有 `readings.raw.json` 在盘上才合成/覆盖。**没有 raw 就明确跳过**——
   228 那类跑次（`readings.json` 由 runner 自己写、没有 raw）⛔ 不许被本件重写成一个全 null 的件。

## 用法

    python3 scripts/mk_readings.py <run-dir>            # 合成（幂等）
    python3 scripts/mk_readings.py --run-dir <run-dir>  # 同上（链里显式写法）
    python3 scripts/mk_readings.py <run-dir> --check    # 只算不写：报 would-write / unchanged

退出码：0 = 已合成 / 已是最新 / 明确跳过；1 = 输入不可解析或身份不符（**失败必须可见**）；
        2 = 用法错误（缺参数 / 目录不存在）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: 无 `exit_code` / `exit_code_outer` 时的既有哨兵（沿用编排层原口径，⛔ 不用 0 冒充成功）。
EXIT_UNKNOWN = 99
#: 合成时**保留**盘上现值的键（唯一产出者是 `scripts/record_checks.py`，本件只搬运不覆盖）。
PRESERVED_KEYS = ("checks", "checks_evidence", "teacher_after_acc")


def _read_json(path: Path) -> dict | None:
    """读 JSON；缺失或不可解析 ⇒ `None`（调用方**显式**决定怎么办，⛔ 不静默当空对象）。"""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_float(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def compose(run_dir: Path) -> tuple[dict | None, str]:
    """算出应当落盘的 `readings.json` 内容 ⇒ (doc, 说明)；不可合成 ⇒ (None, 原因)。"""
    raw_path = run_dir / "readings.raw.json"
    if not raw_path.is_file():
        return None, "无 readings.raw.json（本步的输入证据不在盘上 ⇒ 明确跳过，⛔ 不覆盖 runner 自写的件）"
    raw = _read_json(raw_path)
    if raw is None:
        return None, f"FATAL: {raw_path} 不可解析为 JSON 对象 ⇒ 拒绝合成（⛔ 不拿空值兜底）"
    tier = run_dir.name
    raw_tier = raw.get("tier")
    if isinstance(raw_tier, str) and raw_tier != tier:
        return None, (f"FATAL: 档号错配 raw.tier={raw_tier!r} ≠ 跑次目录名 {tier!r}"
                      " ⇒ 拒绝合成（防「跑 A 填 B」）")
    inner = _read_int(run_dir / "exit_code")
    outer = _read_int(run_dir / "exit_code_outer")
    out = {
        "tier": tier,
        "exit_code": inner if inner is not None else (outer if outer is not None else EXIT_UNKNOWN),
        "exit_code_outer": outer,
        "elapsed_sec": _read_float(run_dir / "elapsed_sec"),
        "comparable": None,
        "env_diff_rc": None,
        "gpus": raw.get("gpus"),
        "config": raw.get("config"),
        "image": raw.get("image"),
        "timeout_sec": raw.get("timeout_sec"),
        "repo": raw.get("repo"),
        "command": raw.get("command"),
        "note": raw.get("note"),
    }
    prev = _read_json(run_dir / "readings.json")
    if prev is not None:
        for k in PRESERVED_KEYS:
            if k in prev:
                out[k] = prev[k]
    return out, "已从 readings.raw.json + exit_code/elapsed_sec 合成"


def dumps(doc: dict) -> bytes:
    """落盘字节的**唯一**写法（幂等的前提：同一 doc ⇒ 同一串字节）。"""
    return json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="由 readings.raw.json + exit_code/elapsed_sec 合成 readings.json（幂等 · D23）",
        allow_abbrev=False)
    ap.add_argument("run_dir", nargs="?", help="跑次目录（<run-root>/<档号>/）")
    ap.add_argument("--run-dir", dest="run_dir_opt", help="同上（链里显式写法）")
    ap.add_argument("--check", action="store_true",
                    help="只算不写：报 would-write / unchanged（供 fill_one --dry-run 用）")
    a = ap.parse_args(argv)
    arg = a.run_dir_opt or a.run_dir
    if not arg:
        print("用法: python3 scripts/mk_readings.py <run-dir> [--check]", file=sys.stderr)
        return 2
    run_dir = Path(arg)
    if not run_dir.is_dir():
        print(f"FATAL: 跑次目录不存在：{run_dir}", file=sys.stderr)
        return 2

    doc, why = compose(run_dir)
    if doc is None:
        print(f"跳过 {run_dir.name}：{why}")
        return 0 if "FATAL" not in why else 1

    payload = dumps(doc)
    target = run_dir / "readings.json"
    unchanged = target.is_file() and target.read_bytes() == payload
    state = "unchanged" if unchanged else ("would-write" if a.check else "wrote")
    if not unchanged and not a.check:
        target.write_bytes(payload)
    print(f"{state} {target}")
    print(json.dumps({k: doc[k] for k in ("tier", "exit_code", "exit_code_outer", "elapsed_sec")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    sys.exit(main())
