#!/usr/bin/env python3
"""存量备注列**纯文本规范化**（任务 E2）—— 只洗文案，⛔ 不重新回填、⛔ 不读证据、⛔ 不改其它字段。

## 为什么单独一件

任务 E 把备注列的口径改成「说人话」（`scripts/note_style.py`：删坐标/路径/`file:line`/正则原文/
英文长句，报错只留摘要，单档 ≤200 字）。**已落盘的 21 档**是旧口径写进去的（551~1324 字），
要跟新口径对齐。⚠️ 但它们**不能靠"重新回填"来洗**：回填要读跑次证据、会顺带重算判据，
风险与成本都不该由一次文案整理承担。

⇒ 本脚本只做**文本级**规范化：`note_style.parse_legacy(旧备注) ⇒ note_style.render(事实)`，
其余字段一个字节都不碰；效果读数的 after/base/Δ **从效果列现读**（⛔ 不重算，只搬运展示位数）。

## 幂等 & 可复跑

* 规范化后的备注再跑一遍**逐字不变**（`note_style.normalize_legacy` 幂等，单测钉住）；
* 落盘前先自证：`json.dumps(原数据件)` 能逐字节复原原文件（格式口径一致）⇒ 全量重写✚
  「只改 note」可被机核证明；
* `--write` 后还会跑 `verify_matrix.py` 复核「数据件 ⇄ 人读件」一致。

## 用法

```bash
GRASPO_ROUND_DATE=20261025 python tools/matrix/normalize_notes.py            # dry-run（默认）
GRASPO_ROUND_DATE=20261025 python tools/matrix/normalize_notes.py --write    # 落盘
```

退出码：0 = 成功（含「本来就规范」）；2 = 用法/路径错；3 = 语义守卫拦下（**不写任何文件**）；
4 = 落盘后复核失败（已回滚到原字节）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import note_style  # noqa: E402  —— 备注口径的唯一真相源（⛔ 不在本脚本里另写一套规则）


def _paths() -> dict:
    """路径一律取自 `tools/matrix/paths.py`（唯一真相源，⛔ 不在这里拼日期）。"""
    out = subprocess.run([sys.executable, str(HERE / "paths.py"), "--sh"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"paths.py 取路径失败：{out.stderr.strip()}")
    env = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'")
    return env


def _dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def _numbers(note: str) -> list[str]:
    import re
    return re.findall(r"[0-9]+(?:\.[0-9]+)?", note)


def guard(tier: str, old: str, new: str) -> list[str]:
    """语义守卫：清洗后**不许**丢/改任何结论或承重数字（⛔ 一旦不符就停下报人）。

    ⚠️ 只查**承重**的那几项 —— `r=32`、「1020 个张量名」、签署时间戳、`stdout.log:4248`
    这类**证据细节本来就该被删**（E1 的「删」清单），不能当丢失来拦。
    """
    problems = []
    f_old, f_new = note_style.parse_legacy(old), note_style.parse_legacy(new)
    for key in ("ckpt", "naninf", "exit_code", "exit_derived",
                "epochs_done", "epochs_planned", "peak", "steps"):
        if f_old.get(key) != f_new.get(key):
            problems.append(f"{key} 变了：{f_old.get(key)!r} → {f_new.get(key)!r}")
    # 承重数字必须逐字还在（退出码 / 步数 / 轮数 / 峰值显存）
    for key in ("exit_code", "steps", "epochs_done", "epochs_planned"):
        v = f_old.get(key)
        if v is not None and str(v) not in new:
            problems.append(f"承重数字丢了：{key}={v}")
    peak = f_old.get("peak")
    if peak and peak not in new:
        problems.append(f"峰值显存丢了：{peak}")
    # 判据集合守恒（去重比较：老备注里同一编号可能出现多次，⛔ 关键是不许洗成"通过"）
    for key in ("fail_ids", "unk_ids"):
        if sorted(set(f_old.get(key) or [])) != sorted(set(f_new.get(key) or [])):
            problems.append(f"{key} 集合变了：{sorted(set(f_old.get(key) or []))} → "
                            f"{sorted(set(f_new.get(key) or []))}")
    if len(new) > note_style.NOTE_LIMIT:
        problems.append(f"超长：{len(new)} > {note_style.NOTE_LIMIT} 字")
    if note_style.has_noise(new):
        problems.append("清洗后仍含工程日志形态")
    return problems


def differences_outside_note(old, new, _path: str = "") -> list[str]:
    """机核：两份数据件**除 note 外**逐字段相同 ⇒ 返回差异路径列表（空 = 相同）。

    ★ 「能失败」是这个函数存在的意义：落盘后用它复核，任何越出 note 的字节漂移都会被点名
    （`rows[3].status` / `meta.verdict` / `rows[7].effect` …），并触发**回滚原字节**。
    """
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for k in sorted(set(old) | set(new)):
            if k == "note":
                continue
            if k not in old or k not in new:
                out.append(f"{_path}.{k}（缺失/新增）")
                continue
            out += differences_outside_note(old[k], new[k], f"{_path}.{k}")
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{_path}（长度 {len(old)} → {len(new)}）"]
        out = []
        for i, (a, b) in enumerate(zip(old, new)):
            out += differences_outside_note(a, b, f"{_path}[{i}]")
        return out
    if old != new:
        return [f"{_path or '<root>'}: {old!r} → {new!r}"]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="存量备注列纯文本规范化（⛔ 不重新回填）")
    ap.add_argument("--data", default=None, help="数据件路径（缺省 = paths.py 的当轮副本）")
    ap.add_argument("--html-out", default=None, help="人读件路径（缺省 = paths.py 的当轮副本）")
    ap.add_argument("--template", default=None, help="渲染模板（缺省 = paths.py 的 PATHS_TEMPLATE）")
    ap.add_argument("--write", action="store_true", help="落盘（缺省只做 dry-run，⛔ 不写任何文件）")
    ap.add_argument("--no-criteria-line", dest="criteria_line", action="store_false",
                    default=True, help="不给「通过」档补「判据结论：①②③④ 过」那句")
    ap.add_argument("--quiet-diff", action="store_true", help="只打汇总，不打逐档全文")
    a = ap.parse_args(argv)

    env = _paths()
    data = Path(a.data or env["PATHS_ROUND_DATA"])
    html_out = Path(a.html_out or env["PATHS_ROUND_HTML"])
    template = Path(a.template or env["PATHS_TEMPLATE"])
    if not data.is_file():
        print(f"FAIL：数据件不存在：{data}", file=sys.stderr)
        return 2

    original = data.read_text(encoding="utf-8")
    doc = json.loads(original)
    if _dump(doc) != original:
        print("FAIL：数据件不是 `indent=2` 规范格式 ⇒ 全量重写无法证明「只改 note」，拒绝继续。",
              file=sys.stderr)
        return 2

    filled = [r for r in doc["rows"] if str(r.get("note") or "").strip()]
    changes, problems = [], []
    for row in filled:
        old = str(row["note"])
        facts = note_style.parse_legacy(old)
        facts.update(note_style.parse_effect_column(row.get("effect")))
        # ★ 老备注只有「不通过」档才带判据句（`compose_note` 当年只在 不通过 时追加）。
        #   新口径下 `compose_note` 对 通过/不通过 **都**写一句编号级结论 ⇒ 这里用
        #   **状态列**投影同一句（⛔ 不读证据、⛔ 不重算判据；`--no-criteria-line` 可关）。
        if (a.criteria_line and row.get("status") == "通过"
                and not facts.get("fail_ids") and not facts.get("unk_ids")):
            facts["judged"] = True
        new = note_style.render(facts)
        if new == old:
            continue
        problems += [f"{row['tier']}：{p}" for p in guard(row["tier"], old, new)]
        changes.append((row["tier"], old, new))

    print(f"▶ 已回填 {len(filled)} 档；需规范化 {len(changes)} 档；"
          f"口径 = scripts/note_style.py（上限 {note_style.NOTE_LIMIT} 字）")
    total_before = sum(len(o.encode()) for _t, o, _n in changes)
    total_after = sum(len(n.encode()) for _t, _o, n in changes)
    for tier, old, new in changes:
        if not a.quiet_diff:
            print(f"\n── {tier}  {len(old)} → {len(new)} 字（{len(old.encode())} → "
                  f"{len(new.encode())} 字节）")
            print(f"   改前：{old}")
            print(f"   改后：{new}")
    print(f"\n▶ 合计：{len(changes)} 档；{total_before} → {total_after} 字节"
          f"（{total_before - total_after:+d}）")

    if problems:
        print("\nFAIL-CLOSED：语义守卫拦下（⛔ 未写任何文件）：", file=sys.stderr)
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        return 3

    # ── 预演：全量重写后的数据件 + 重新渲染的人读件（都只落到临时目录）────────────
    new_doc = json.loads(json.dumps(doc, ensure_ascii=False))
    by_tier = {t: n for t, _o, n in changes}
    for row in new_doc["rows"]:
        if row["tier"] in by_tier:
            row["note"] = by_tier[row["tier"]]
    new_text = _dump(new_doc)

    tmp = Path(tempfile.mkdtemp(prefix=".normalize-notes-", dir=str(data.parent)))
    try:
        tmp_data, tmp_html = tmp / data.name, tmp / "out.html"
        tmp_data.write_text(new_text, encoding="utf-8")
        rc = subprocess.run([sys.executable, str(HERE / "render_matrix.py"), "--template",
                             str(template), "--data", str(tmp_data), "--out", str(tmp_html)])
        if rc.returncode != 0:
            print("FAIL：渲染失败（⛔ 未写任何文件）", file=sys.stderr)
            return 2
        rc = subprocess.run([sys.executable, str(HERE / "verify_matrix.py"), "--template",
                             str(template), "--data", str(tmp_data), "--html", str(tmp_html)],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"FAIL：verify_matrix 未通过（⛔ 未写任何文件）：\n{rc.stdout[-1500:]}", file=sys.stderr)
            return 2
        html_before = html_out.read_bytes() if html_out.is_file() else b""
        html_after = tmp_html.read_bytes()
        print(f"▶ 人读件：{len(html_before)} → {len(html_after)} 字节"
              f"（{len(html_after) - len(html_before):+d}）")

        if not a.write:
            print("\n--dry-run：**未写任何文件**（只打印差异；⛔ 数据件/人读件一字未改）")
            return 0

        # ── 落盘：数据件 + 人读件（⛔ 基准件永不涉及：path 来自 PATHS_ROUND_*）─────────
        data.write_text(new_text, encoding="utf-8")
        html_out.write_bytes(html_after)
        # 自证：重读后「除 note 外」与原数据件逐字段相同（机核；不符 ⇒ 回滚原字节）
        back = json.loads(data.read_text(encoding="utf-8"))
        drift = differences_outside_note(json.loads(original), back)
        if drift:
            data.write_text(original, encoding="utf-8")
            print("FAIL：落盘后「除 note 外」出现差异 ⇒ 已回滚原字节", file=sys.stderr)
            for d in drift[:20]:
                print(f"  ✗ {d}", file=sys.stderr)
            return 4
        print(f"\n✅ 已落盘：{data}（{len(changes)} 档备注规范化）与 {html_out}")
        print(f"   机核：除 note 外 **0 处**差异；已回填 {len(filled)} 档不变")
        return 0
    finally:
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()


if __name__ == "__main__":
    sys.exit(main())
