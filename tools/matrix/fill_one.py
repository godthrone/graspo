#!/usr/bin/env python3
"""fill_one.py — **单档回填**：把某一档的实测读数单独写进「当轮日期副本」（一条命令的实现体）。

主席要求（工作包原文）：
  「单条更新测试结果数据，每完成一条就更新一条，以便随时看见进度。」
  ⇒ 本文件只做一件事：**一条命令把 <档号> 那一行写进当轮日期副本，其它档/其它章节一字不动。**

★★ 不新造回填通道，只把现役两件**封装**成一条命令（宪法 §1.4 单一真相源）：
   ① 读数派生 = `scripts/make_result_json.py`（从 `runs/<档号>/` 的机器可读证据实测派生；
      缺证据 ⇒ rc≠0，⛔ 不用 false/null 冒充"判过了"）
   ② 落盘 = `tools/matrix/backfill_matrix.py`（四条判据现算 status ⇒ 渲染 ⇒
      `check_fill_scope.py` 写范围硬闸门 ⇒ 不过则**逐字节回滚**）
  本文件**不手填任何数值**：所有数字都从 ① 的产物来；本文件只做「档号/证据定位 + 排练比较 +
  范围断言 + before/after 打印 + 进度行 + 三锁回执」。

★ 证据定位（228 实测定下的口径，⛔ 不猜）：
  · 跑次目录 = `${GRASPO_RUN_ROOT:-<树>/runs}/<档号>/`（`readings.json` / `derived_steps.json` /
    `exit_code` / `elapsed_sec` / `gpu/peak_mem.json` / `<档号>/v0-*/checkpoint-*/trainer_state.json` …）
  · 评测证据 = `${GRASPO_EVAL_ROOT}`（缺省 = 树根的兄弟目录 `eval-*` 里**恰好一个**含
    `.local/reports-archive/<档号>/{after,base}/eval_report.json` 的）的**按档归档**那一路。
    ⛔ 不用 `<EVAL_ROOT>/.local/reports/{after,base}`：那是滚动覆盖的，跨档会串号。
    报告内的 `dataset.path=…/<档号>.jsonl` / `model.path=…/merged-hf-<档号小写>` 必须命中本档 ⇒ 否则拒收。

★ 幂等（工作包行为要求 4）：先在**临时沙盘**里把整条链排练一遍，若
  「该档行现值 == 排练产物」且「当轮副本 HTML 与排练渲染逐字节相等」⇒ **不写盘**、直接 rc=0，
  于是同一档重复跑 sha256 必不变（现有通道每次会重写 `meta.backfill.last_run`，故必须短路）。

★ 可选开关（缺省关，未设时行为逐字节不变）：`GRASPO_FILL_RECOVER_TIMEOUT_ARTIFACT=1`
  ⇒ 读数派生加 `--recover-timeout-artifact`：对「训练已按主流程返回、却被外层 timeout 记成
  exit_code∈{124,137}」的跑次，按 `make_result_json.py` 的七条带坐标判据**证据化重派生**
  exit_code/elapsed（provenance 同时留 `exit_code_raw`）。⛔ 不放宽判据、不填任何数值：
  任一条不成立 ⇒ 一个字段都不改。用于 D1（训练跑完却不收尾）污染过的老跑次。

★ 退出码：0 = 成功（写入 / dry-run / 幂等短路）；2 = 用法或档号非法；
  3 = **证据不足**（该档保持「待测试」，⛔ 未写任何字节）；4 = 机核/范围闸门失败（已回滚或未落盘）；
  5 = 基准件三锁不成立（⛔ 拒起）；9 = 解释器环境不足（缺 pydantic/graspo ⇒ 需在容器内跑）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))
import check_fill_scope as _scope  # noqa: E402  —— 单元格级 diff 判据的唯一实现（⛔ 不另写一份）
import paths as _paths  # noqa: E402  —— 基准件/副本 路径与三锁常量的唯一真相源

TIER_RE = re.compile(r"^M\d{3}$")
#: 该档行的**结果面** = 第六章表格的 7 个单元格（status + 6 个实验数据列）
STATUS_COL = "status"
#: 数据件的实验数据列（顺序与 render 链一致；列清单本身从数据件 `data_keys` 现读，⛔ 本文件不写死）
ROW_COLS = (STATUS_COL, "peak_mem", "steps_actual", "elapsed", "throughput", "effect", "note")
#: 「未回填」的占位形态（与渲染链的 `—` 同义；用于**实算**进度行，⛔ 不读 meta 里的计数）
EMPTY = {"", "—", "-", "null", "None"}

LEDGER_DEFAULT = ".local/matrix-fill-ledger.md"
EXIT_ENV = "FILL_ONE_EXIT"
PY = sys.executable


# ── 小工具 ───────────────────────────────────────────────────────────────────
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_of(p: Path) -> tuple[str, int, str]:
    st = p.stat()
    return sha256(p), st.st_size, format(st.st_mode & 0o777, "o")


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def rows_of(doc: dict) -> dict[str, dict]:
    return {r["tier"]: r for r in (doc.get("rows") or []) if isinstance(r, dict) and "tier" in r}


def run(cmd: list[str]) -> tuple[int, str]:
    """跑子命令（cwd = 仓根）；stdout+stderr 合并，供失败时原样贴出。"""
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def pick(out: str, *keys: str) -> list[str]:
    """从子命令输出里挑出关键行（闸门结论、变化面摘要）用于本命令的回执。"""
    return [ln.strip() for ln in out.splitlines() if any(k in ln for k in keys)]


def print_row(tag: str, row: dict, note_limit: int = 260) -> None:
    print(f"── {tag} ──")
    for k in ROW_COLS:
        v = row.get(k)
        s = "null" if v is None else str(v)
        if k == "note" and len(s) > note_limit:
            s = s[:note_limit] + f"…（共 {len(s)} 字，完整值见数据件该行）"
        print(f"  {k:<13} {s}")


def filled_count(rows: dict[str, dict]) -> int:
    """**实算**已回填档数（6 个实验数据列全非占位）——⛔ 不读 `meta.backfill.unfilled_count`。"""
    return sum(1 for r in rows.values()
               if all(str(r.get(k, "")).strip() not in EMPTY for k in ROW_COLS[1:]))


def print_progress(rows: dict[str, dict]) -> None:
    print(f"已回填 {filled_count(rows)} / {len(rows)} 档")


def print_locks(baseline: Path, *, before: dict[str, str] | None = None) -> int:
    """贴三锁（基准件 sha/字节/权限）+ 当轮副本 sha；返回 0 = 三锁成立，5 = 不成立。"""
    sha, size, mode = lock_of(baseline)
    ok_sha = sha == _paths.BASELINE_HTML_SHA256
    ok_size = size == _paths.BASELINE_HTML_BYTES
    ok_mode = mode == _paths.FROZEN_MODE
    mark = lambda b: "✓" if b else "✗"  # noqa: E731
    print("基准件三锁："
          f"sha256={sha[:8]}…{mark(ok_sha)}（期望 {_paths.BASELINE_HTML_SHA256[:8]}…）  "
          f"字节={size}{mark(ok_size)}（期望 {_paths.BASELINE_HTML_BYTES}）  "
          f"权限={mode}{mark(ok_mode)}（期望 {_paths.FROZEN_MODE}）  [{baseline.name}]")
    if before is not None:
        print(f"当轮副本 sha256：写前 {before.get('data','?')[:16]}… / {before.get('html','?')[:16]}…")
    return 0 if (ok_sha and ok_size and ok_mode) else 5


# ── 证据定位 ─────────────────────────────────────────────────────────────────
def run_dir_of(tier: str) -> Path:
    root = Path(os.environ.get("GRASPO_RUN_ROOT") or (ROOT / "runs"))
    return root / tier


def eval_archive_of(tier: str) -> tuple[Path | None, str]:
    """按档评测归档 `<EVAL_ROOT>/.local/reports-archive/<档号>/`（含 after+base 两报告）。"""
    env = os.environ.get("GRASPO_EVAL_ROOT")
    if env:
        cands = [Path(env)]
    else:
        cands = sorted(p for p in ROOT.parent.glob("eval-*") if p.is_dir())
    hits = [c / ".local" / "reports-archive" / tier for c in cands
            if (c / ".local" / "reports-archive" / tier / "after" / "eval_report.json").is_file()
            and (c / ".local" / "reports-archive" / tier / "base" / "eval_report.json").is_file()]
    if len(hits) == 1:
        return hits[0], ""
    if len(hits) > 1:
        return None, (f"评测归档不唯一（{len(hits)} 处）⇒ fail-closed："
                      + "、".join(str(h) for h in hits) + "（用 GRASPO_EVAL_ROOT 指定）")
    return None, ("找不到按档评测归档 `.local/reports-archive/<档号>/{after,base}/eval_report.json`"
                  "（判据② 无实测读数 ⇒ 该档不可判定）")


def stage_eval_dir(tmp: Path, tier: str) -> tuple[Path | None, list[str]]:
    """把按档归档摆成 `make_result_json.py --eval-dir` 认的 `.local/reports/{after,base}` 形。

    ★ 只建**符号链接**（⛔ 不复制、⛔ 不改原报告）；并核对报告内的档号身份，防"跑 A 填 B"。
    """
    arch, why = eval_archive_of(tier)
    if arch is None:
        return None, [why]
    problems: list[str] = []
    stage = tmp / "ev"
    for role in ("after", "base"):
        src = arch / role / "eval_report.json"
        doc = read_json(src)
        ds = str(((doc.get("dataset") or {}).get("path")) or "")
        mp = str(((doc.get("model") or {}).get("path")) or "")
        if not (ds.endswith(f"/{tier}.jsonl") or tier.lower() in mp.lower()):
            problems.append(f"{role} 报告不属本档（dataset.path={ds!r} / model.path={mp!r} 未见 {tier}）")
        dst = stage / ".local" / "reports" / role / "eval_report.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())
    if problems:
        return None, problems
    return stage, []


# ── 范围断言（比闸门更严：只许**本档**的 7 个单元格变）────────────────────────
def html_cell_changes(before: str, after: str, cols: list[str], vocab) -> tuple[list[tuple[str, str]], list[str]]:
    """返回 (变的单元格 [(档号, 列名)], 违规清单)。base = **写前当轮副本**（⛔ 不是基准件）。"""
    bad = _scope.check_scope(before, after, result_columns=cols, status_vocab=vocab,
                             base_path="当轮副本(写前)", filled_path="当轮副本(写后)")
    allowed = set(cols) | {STATUS_COL}
    a = _scope.parse_doc(before, "before")
    b = _scope.parse_doc(after, "after")
    changed = [(cb.tier or "?", cb.col or "?") for ca, cb in zip(a.cells, b.cells)
               if ca.inner != cb.inner and (cb.col in allowed)]
    return changed, bad


def data_scope_violations(before: dict, after: dict, tier: str) -> list[str]:
    """数据件层面：除 `rows[<档号>]` 的 7 个结果字段与 `meta.backfill` 外，其它一字不许动。"""
    bad: list[str] = []

    def walk(a, b, path: str) -> None:
        if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            bad.append(f"{path}: 类型变了 {type(a).__name__} → {type(b).__name__}")
            return
        if isinstance(a, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    bad.append(f"{path}.{k}: 键{'新增' if k in b else '丢失'}")
                    continue
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if len(a) != len(b):
                bad.append(f"{path}: 长度变了 {len(a)} → {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif a != b:
            bad.append(f"{path}: {a!r} → {b!r}")

    for k in sorted(set(before) | set(after)):
        if k in ("rows", "meta"):
            continue
        walk(before.get(k), after.get(k), k)
    # `meta.backfill` 是回填器的**记账段**（last_run/filled_tiers/未回填数），由通道自己维护 ⇒ 不计入范围；
    # 其余 meta 段（口径/词表/枚举…）一字不许动。
    for k in sorted((set(before.get("meta", {})) | set(after.get("meta", {}))) - {"backfill"}):
        walk(before.get("meta", {}).get(k), after.get("meta", {}).get(k), f"meta.{k}")
    br, ar = rows_of(before), rows_of(after)
    if sorted(br) != sorted(ar):
        bad.append(f"rows 档号集合变了：{len(br)} → {len(ar)} 档")
    for t in sorted(set(br) & set(ar)):
        if t != tier:
            if br[t] != ar[t]:
                bad.append(f"rows[{t}]（非本档）被改了")
            continue
        for k in sorted(set(br[t]) | set(ar[t])):
            if k in ROW_COLS:
                continue
            if br[t].get(k) != ar[t].get(k):
                bad.append(f"rows[{t}].{k}（非结果列）被改了")
    return bad


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="单档回填：把某一档的实测读数单独写进当轮日期副本（其它档/章节一字不动）",
        allow_abbrev=False, add_help=False)
    ap.add_argument("tier", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    a = ap.parse_args(argv)

    if a.help or not a.tier:
        print("fill_one.sh — 单档回填：把某一档的实测读数单独写进当轮日期副本（其它档/章节一字不动）")
        print("用法: bash tools/matrix/fill_one.sh <M档号> [--dry-run]")
        print("例:   bash tools/matrix/fill_one.sh M041 --dry-run      # 只打印，不落盘")
        print("      bash tools/matrix/fill_one.sh M041               # 真写（幂等：重复跑 sha 不变）")
        return 0 if a.help else 2

    tier = a.tier
    # ① 档号形态 + 必须在矩阵 68 档内（⛔ 不猜、不建新档）
    if not TIER_RE.match(tier):
        print(f"⛔ 拒绝：非法档号 {tier!r}（须形如 M001…M068）⇒ **未写任何文件**（rc=2）")
        return 2

    baseline = _paths.baseline_html()
    round_data = _paths.round_data()
    round_html = _paths.round_html()
    template = TOOL_DIR / _paths.TEMPLATE_NAME
    #: 写前逐字节快照（落盘后复核发现越界时用于**原样回滚**）
    snapshot = {"data": round_data.read_bytes() if round_data.is_file() else b"",
                "html": round_html.read_bytes() if round_html.is_file() else b""}
    before_sha = {k: (hashlib.sha256(v).hexdigest() if v else "") for k, v in snapshot.items()}
    before_html = snapshot["html"].decode("utf-8") if snapshot["html"] else ""

    def rollback() -> bool:
        """把当轮副本两件恢复到写前字节（含权限）；返回 True = 逐字节已复原。"""
        for key, path in (("data", round_data), ("html", round_html)):
            if snapshot[key]:
                path.write_bytes(snapshot[key])
                path.chmod(int(_paths.ROUND_MODE, 8))
        return sha256(round_data) == before_sha["data"] and sha256(round_html) == before_sha["html"]

    print(f"→ 档号 {tier}｜当轮副本 {round_data.name} / {round_html.name}"
          f"（GRASPO_ROUND_DATE={_paths.round_date()}）｜dry-run={'是' if a.dry_run else '否'}")
    rc_locks = print_locks(baseline)
    if rc_locks != 0:
        print("⛔ 基准件三锁不成立 ⇒ 拒绝执行（先修基准件，⛔ 不许拿坏基线做对比）")
        return rc_locks
    if not round_data.is_file() or not round_html.is_file():
        print(f"⛔ 当轮日期副本缺失 ⇒ 未写任何文件（rc=2）。先派生：\n"
              f"   bash tools/matrix/new_round.sh --date {_paths.round_date()}")
        return 2

    matrix = read_json(round_data)
    rows = rows_of(matrix)
    if tier not in rows:
        print(f"⛔ 拒绝：档号 {tier} 不在矩阵档位表内（本件共 {len(rows)} 档："
              f"{min(rows)}…{max(rows)}）⇒ **未写任何文件**（rc=2）")
        return 2
    if not template.is_file():
        print(f"⛔ 渲染模板缺失：{template} ⇒ 未写任何文件（rc=2）")
        return 2

    rdir = run_dir_of(tier)
    print(f"→ 跑次证据 {rdir}")
    if not rdir.is_dir():
        print(f"⛔ 缺跑次目录 {rdir} ⇒ 该档保持「待测试」，**未写任何文件**（rc=3）")
        print_locks(baseline)
        print_progress(rows)
        return 3

    manifest = ROOT / "tests" / "e2e" / "matrix68_manifest.json"
    if not manifest.is_file():
        print(f"⛔ 缺清单 {manifest}（读数派生的身份真相源）⇒ 未写任何文件（rc=2）")
        return 2
    data_keys = matrix.get("data_keys") or []
    vocab = ((matrix.get("meta") or {}).get("status_vocabulary")) or []

    tmp = Path(tempfile.mkdtemp(prefix="fill-one-"))
    try:
        # ② 读数派生（唯一产出者；缺证据 ⇒ rc≠0，⛔ 不填数）
        ev_dir, ev_problems = stage_eval_dir(tmp, tier)
        collect_dir = tmp / "collect"
        collect_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "scripts" / "make_result_json.py"), "--tier", tier,
               "--run-dir", str(rdir), "--out", str(collect_dir / f"{tier}.json"),
               "--collect-dir", str(collect_dir), "--data", str(round_data),
               "--manifest", str(manifest)]
        if ev_dir is not None:
            cmd += ["--eval-dir", str(ev_dir)]
        # ★ D1 的超时伪影重派生：显式 opt-in（缺省关 ⇒ 未设该变量时行为逐字节不变）。
        #   为什么需要这条：D1（训练跑完却不收尾）会把**已经跑完**的档记成 exit_code=137 /
        #   elapsed 含收尾空等；`make_result_json.py --recover-timeout-artifact` 按七条带坐标的
        #   判据（超时码 ∧ elapsed≥timeout ∧ End time 标记 ∧ 步数跑满 ∧ 轮数跑满 ∧ ckpt 可加载
        #   ∧ 无 NaN/Inf）**证据化**地重派生，并在 provenance 里同时留 exit_code_raw。
        #   ⛔ 本开关不填任何数值、不放宽判据：任一条不成立 ⇒ 一个字段都不改（fail-closed）。
        recover = os.environ.get("GRASPO_FILL_RECOVER_TIMEOUT_ARTIFACT", "").strip() == "1"
        if recover:
            cmd += ["--recover-timeout-artifact"]
        rc_c, out_c = run(cmd)
        print(f"→ 读数派生 make_result_json.py rc={rc_c}"
              + ("（含 D1 超时伪影重派生：GRASPO_FILL_RECOVER_TIMEOUT_ARTIFACT=1）" if recover else ""))
        if ev_dir is not None:
            print(f"  评测证据（按档归档）{ev_dir}")
        for why in ev_problems:
            print(f"  ⚠ {why}")
        if rc_c != 0:
            print(out_c.rstrip())
            print(f"⛔ 拒绝写入：{tier} 的实测证据不足/口径不符（make_result_json rc={rc_c}）"
                  f"⇒ 该档保持「待测试」，**未写任何文件**（rc=3）")
            print_locks(baseline)
            print_progress(rows)
            return 3

        # ③ 排练（临时沙盘里跑完整链：判据现算 ⇒ 渲染 ⇒ 机核 ⇒ 写范围闸门）
        reh_dir = tmp / "rehearsal"
        reh_dir.mkdir(parents=True, exist_ok=True)
        reh_data, reh_html = reh_dir / "data.json", reh_dir / "out.html"
        shutil.copyfile(round_data, reh_data)
        shutil.copyfile(round_html, reh_html)
        rc_r, out_r = run([PY, str(TOOL_DIR / "backfill_matrix.py"),
                           "--runs-dir", str(collect_dir), "--data-in", str(reh_data),
                           "--html-out", str(reh_html), "--template", str(template),
                           "--ledger-out", str(reh_dir / "ledger.md"), "--only", tier])
        if rc_r != 0:
            print(out_r.rstrip())
            print(f"⛔ 排练未通过（rc={rc_r}）⇒ **未写任何文件**（rc=4）")
            print_locks(baseline)
            print_progress(rows)
            return 4
        pred = rows_of(read_json(reh_data))
        pred_row = pred.get(tier) or {}
        reh_bad = data_scope_violations(matrix, read_json(reh_data), tier)
        changed, html_bad = html_cell_changes(round_html.read_text(encoding="utf-8"),
                                              reh_html.read_text(encoding="utf-8"),
                                              list(data_keys), set(vocab))
        cross = sorted({t for t, _c in changed} - {tier})
        if reh_bad or html_bad or cross:
            print("⛔ 排练产物**超出本档范围** ⇒ 拒绝写入（rc=4）")
            for b in (reh_bad + html_bad)[:10]:
                print(f"   - {b}")
            if cross:
                print(f"   - 除 {tier} 外还改了这些档的行：{cross}")
            print_locks(baseline)
            print_progress(rows)
            return 4

        print_row(f"{tier} 行 · 写前（当轮副本现值）", rows[tier])
        print_row(f"{tier} 行 · 写后（{'dry-run 预测' if a.dry_run else '预测'}）", pred_row)
        print("本次变化面（排练实测）："
              f"该档 {len(changed)} 个单元格 {[c for _t, c in changed]}；其它档 0 个单元格"
              + ("（该档现值已等于排练产物 ⇒ 不写盘）" if not changed else ""))
        for ln in pick(out_r, "SCOPE PASS", "内容变化", "写范围闸门通过"):
            print(f"  排练闸门：{ln}")

        html_same = sha256(reh_html) == sha256(round_html)
        if a.dry_run:
            print("--dry-run：**未写任何文件**（只打印；排练沙盘在临时目录，跑完即删）")
            print(f"当轮副本 sha256 未变：data {before_sha['data'][:16]}… / html {before_sha['html'][:16]}…")
            print_progress(rows)
            print_locks(baseline)
            return 0

        # ④ 幂等短路：该档行与当轮 HTML 都已是排练产物 ⇒ 一个字节都不写（sha 必不变）
        if pred_row == rows[tier] and html_same:
            print(f"✓ 幂等：{tier} 已是最新（该档行与当轮副本 HTML 逐字节等于排练产物）⇒ **未写盘**")
            print(f"当轮副本 sha256 未变：data {before_sha['data'][:16]}… / html {before_sha['html'][:16]}…")
            print_progress(rows)
            print_locks(baseline)
            return 0

        # ⑤ 真写：走唯一回填通道（它自己带 §22 闸门 + 逐字节回滚）
        ledger = Path(os.environ.get("GRASPO_FILL_LEDGER") or (ROOT / LEDGER_DEFAULT))
        rc_w, out_w = run([PY, str(TOOL_DIR / "backfill_matrix.py"),
                           "--runs-dir", str(collect_dir), "--data-in", str(round_data),
                           "--html-out", str(round_html), "--template", str(template),
                           "--ledger-out", str(ledger), "--only", tier])
        if rc_w != 0:
            print(out_w.rstrip())
            now_data = sha256(round_data) if round_data.is_file() else ""
            now_html = sha256(round_html) if round_html.is_file() else ""
            rolled = (now_data == before_sha["data"] and now_html == before_sha["html"])
            print(f"⛔ 落盘失败（rc={rc_w}）⇒ {'已逐字节回滚，当轮副本 sha 未变' if rolled else '副本可能未回滚，请人核'}（rc=4）")
            print_progress(rows)
            print_locks(baseline)
            return 4

        # ⑥ 落盘后复核：数据件只动该档 7 字段 + meta.backfill；HTML 只动该档单元格
        after_matrix = read_json(round_data)
        after_rows = rows_of(after_matrix)
        post_bad = data_scope_violations(matrix, after_matrix, tier)
        changed_post, html_bad_post = html_cell_changes(before_html,
                                                       round_html.read_text(encoding="utf-8"),
                                                       list(data_keys), set(vocab))
        cross_post = sorted({t for t, _c in changed_post} - {tier})
        if post_bad or html_bad_post or cross_post:
            ok_rb = rollback()
            print(f"⛔ 落盘后复核发现越界 ⇒ {'已逐字节回滚（sha 复原）' if ok_rb else '回滚后 sha 不符，请人核'}（rc=4）")
            for b in (post_bad + html_bad_post)[:10]:
                print(f"   - {b}")
            if cross_post:
                print(f"   - 除 {tier} 外还改了这些档：{cross_post}")
            print_progress(after_rows)
            print_locks(baseline)
            return 4

        print_row(f"{tier} 行 · 写后（落盘实测）", after_rows[tier])
        print("本次变化面（落盘实测）："
              f"该档 {len(changed_post)} 个单元格 {[c for _t, c in changed_post]}；其它档 0 个单元格")
        for ln in pick(out_w, "SCOPE PASS", "内容变化", "写范围闸门通过"):
            print(f"  机核闸门：{ln}")
        print(f"✓ 已写入当轮副本：{round_data.name} / {round_html.name}"
              f"（台账 +1 行 ⇒ {ledger.relative_to(ROOT) if str(ledger).startswith(str(ROOT)) else ledger}）")
        print(f"当轮副本 sha256：写前 data {before_sha['data'][:16]}… → 写后 {sha256(round_data)[:16]}…"
              f"；写前 html {before_sha['html'][:16]}… → 写后 {sha256(round_html)[:16]}…")
        print_progress(after_rows)
        print_locks(baseline)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
