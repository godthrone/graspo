#!/usr/bin/env python3
"""switch_corpus.py — 把 68 档（M001–M068）的训练语料从**旧 100 条**切到**新 100 条**，并逐件改写下游锚点。

它解决的问题（一名一物）
------------------------
矩阵 v3 的 68 个 M 档共用**同一份 100 行冻结语料**，这个 sha256 被**七类件**重复登记：

  ① 68 份训练子集 ``subsets-v2/M0nn.jsonl``（在盘实体）
  ② 子集登记清单 ``subsets-v2/subsets-v2.manifest.json``（逐档 sha/来源 + source_files.m68_frozen
     + m68.source_sha256）
  ③ 生成器常量 ``tests/e2e/make_subsets.py::M68_SOURCE_SHA256``（+ 源实体候选 + 文案）
  ④ 跑次清单 ``tests/e2e/matrix68_manifest{_source}.json``（**由 ③ 的姊妹生成器现算**，本工具重跑它）
  ⑤ 数据件 ``docs/capability-matrix.data.json``（判据锚点文案 1 处 + 逐档 ``data_source`` 68 处）
  ⑥ 数据件日期镜像 ``docs/capability-matrix-20261022.data.json``（同 ⑤）
  ⑦ 生成端文案 ``tools/matrix/{build_official,enumerate_matrix}.py``、``scripts/measure_elam_seq_len.py``

手工改这七类件是「同一个事实抄了 7 份」的典型事故面：漏一处 ⇒ 闸门红或更糟（静默不一致）。
本文件把「旧 sha → 新 sha」的**全部落点**写成一张可审的计划表：``--dry-run`` 只出计划、**一个字节都不写**。

设计纪律（对齐宪法 §2.3 边界校验 / §2.4 操作防呆 / §3 退路）
-----------------------------------------------------------
  · **默认 dry-run**：必须显式 ``--apply`` 才落盘；dry-run 只写 ``--plan-out``（一次性的计划件）。
  · **源三重校验**：``--expect-sha256`` 相等 ∧ 行数 == 100 ∧ 每行 JSON 可解析；任一不满足 ⇒ rc=2、零写入。
  · **冻结件不自行解锁**：``chmod 444`` 的目标件（数据件 / 人读件）未显式 ``--unlock-frozen`` 时
    **停在写入之前**并打印人工解锁命令（⛔ 本工具默认绝不 chmod）。给了 ``--unlock-frozen`` 才临时
    644 → 写 → 复原 444，并逐件打印警告。
  · **先备份再写**：所有将被改/将被新建的路径在第一次写入前登记进
    ``<repo>/.local/hb-workspace/20261022-baseline-r1/task-corpus-switch/backup-<ts>/``（含原 mode）。
  · **落盘后自检四道**：``check_68_readiness`` / ``generate_matrix68_manifest_source --verify`` /
    ``render_matrix``（渲染到**带日期副本**）/ ``verify_matrix``（**原版对 + 副本对**）。
    任何一步 rc≠0 ⇒ **整体回滚到备份**并 rc≠0（回滚后再次核对每件 sha，回滚失败即如实报出）。
  · **幂等**：期望内容 = 由**当前在盘内容**推出来的目标内容；第二次 ``--apply`` 时目标 == 现状
    ⇒ 打印「已切换（sha 匹配），无变更」rc=0。
  · **⛔ 不做的事**：不 ``git``、不连远端、不跑训练、不改 ``samples/configs/**``、不动
    ``mini-dataset/**``、不改写 ``subsets-v2.overwrite-log.jsonl`` 的**历史**条目（只追加）。

为什么 113 处只改 70 处（★ 别用 sed 全局替换）
----------------------------------------------
``subsets-v2.manifest.json`` 里旧 sha 出现 **113** 次，但**只有 70 次该变**：
  · 68 次 = M 档逐档 ``tiers[].sha256``
  · 1 次  = ``source_files.m68_frozen.sha256``
  · 1 次  = ``m68.source_sha256``
**另有 43 次必须原样保留**：42 次是 T 档（``source: mini``）逐档 sha —— 它们登记的是**没变的**
``subsets-v2/T0nn.jsonl``；1 次是 ``source_files.mini.sha256`` —— 同一个道理（mini 源文件不在本次切换范围，
被覆盖会把清单写成假话）。⇒ 本工具按**字段路径**定点改写，绝不按字符串全局替换。

用法::

    # 出计划（默认 dry-run，零写入）
    .venv/bin/python tools/matrix/switch_corpus.py \
        --source-jsonl .local/hb-workspace/20261022-baseline-r1/task-corpus/balanced-100-seed20261022-sha5ad2d7cc588b.jsonl \
        --expect-sha256 5ad2d7cc588bab7b1b72750b3070d3c3f2de1690e105d46e541ca333566d3a6d \
        --plan-out .local/hb-workspace/20261022-baseline-r1/task-corpus-switch/plan.json

    # 真落盘（冻结件需先人工 chmod 644；或显式 --unlock-frozen 由本工具临时解锁并复原）
    ... --apply --unlock-frozen

退出码：0 = 成功（含幂等无变更）；1 = 落盘后自检失败（已回滚）；2 = 用法/源文件不合格（零写入）；
        3 = 冻结件未解锁（零写入）；4 = 在盘状态与计划前提不符（文案/字段漂移，零写入）。
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── 事实常量（唯一来源：本包输入）─────────────────────────────────────────────
#: 切换前的 100 条冻结语料 sha256（mini 冒烟集 100 条；68 个 M 档在盘实体的现值）。
OLD_SOURCE_SHA256 = "594d41b0b8af3dc77cfad45768eca116c3eb1e2314ce33b19ac3e12052669cad"
#: 切换前单个子集字节数（68 档同值）。
OLD_SOURCE_BYTES = 208822
#: 新语料行数（口径：68 档一律 100 行；切换前后**步数不变**）。
EXPECTED_LINES = 100
#: 新语料在仓内的冻结实体落点（仓根相对；与 candidate 文件同名，便于双向追溯）。
NEW_ENTITY_REL = "subsets-v2/balanced-100-seed20261022-sha5ad2d7cc588b.jsonl"
#: 数据件逐档 ``data_source`` 措辞：旧 → 新（切换后 68 档的来源描述）。
OLD_DATA_SOURCE = "ELAM V5 mini 冒烟集（按算法子集条数）"
NEW_DATA_SOURCE = "ELAM V5 均衡子集（12 格分层 · seed 20261022 · 100 条）"

#: 68 个 M 档号（逐档具名：少一个档号即被下面的集合断言当场抓住）。
M_TIERS: tuple[str, ...] = tuple(f"M{i:03d}" for i in range(1, 69))
assert len(M_TIERS) == 68

#: 顶层事实件（只读判定用；人读件本工具**不改**）。
DATA_JSON = "docs/capability-matrix.data.json"
DATA_JSON_DATED = "docs/capability-matrix-20261022.data.json"
HTML_FROZEN = "docs/GRASPO 训练框架功能测试报告.html"
HTML_DATED_TMPL = "docs/GRASPO 训练框架功能测试报告-{date}.html"
MANIFEST = "subsets-v2/subsets-v2.manifest.json"
OVERWRITE_LOG = "subsets-v2/subsets-v2.overwrite-log.jsonl"
MAKE_SUBSETS = "tests/e2e/make_subsets.py"
MEASURE_ELAM = "scripts/measure_elam_seq_len.py"
M68_SRC_JSON = "tests/e2e/matrix68_manifest_source.json"
M68_JSON = "tests/e2e/matrix68_manifest.json"
GEN68 = "tests/e2e/generate_matrix68_manifest_source.py"
TEMPLATE = "tools/matrix/template.html"
CHECK68 = "tools/matrix/check_68_readiness.py"
RENDER = "tools/matrix/render_matrix.py"
VERIFY = "tools/matrix/verify_matrix.py"

#: 备份根（仓根下；=\ `.local/\` 已整体不入 git）。会话级目录名写死，便于指挥官复核。
WORKSPACE_REL = ".local/hb-workspace/20261022-baseline-r1/task-corpus-switch"


class PlanError(RuntimeError):
    """在盘状态与计划前提不符（零写入退出，rc=4）。"""


# ── 小工具 ───────────────────────────────────────────────────────────────────
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dump_json(doc) -> str:
    """与三份在盘 JSON 件**逐字节**同形的序列化（round-trip 已在包内实测一致）。"""
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def mode_of(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def is_frozen(path: Path) -> bool:
    """冻结件判定：存在且**没有**任何写位（chmod 444）。"""
    return path.exists() and (stat.S_IMODE(path.stat().st_mode) & 0o222) == 0


def line_changes(old: str, new: str, limit: int = 40) -> list[list[int]]:
    """老文本坐标下的「被改/被删」连续行区间（供计划表给出行号；超 limit 段只留前 limit 段）。"""
    a, b = old.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    spans: list[list[int]] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            spans.append([i1 + 1, i2])
        elif tag == "insert" and i1 > 0:
            spans.append([i1, i1])          # 纯插入：记在上一行之后
    return spans[:limit]


# ── 变更项模型 ────────────────────────────────────────────────────────────────
def make_change(root: Path, relpath: str, *, group: str, action: str,
                old_bytes: bytes | None, new_bytes: bytes | None,
                field_changes: list[dict], note: str = "") -> dict:
    """一条「文件级」变更：老/新字节 + 字段级明细 + 行号区间（老坐标）。"""
    path = root / relpath
    old_text = old_bytes.decode("utf-8", "replace") if old_bytes is not None else ""
    new_text = new_bytes.decode("utf-8", "replace") if new_bytes is not None else ""
    return {
        "path": relpath,
        "group": group,
        "action": action,
        "note": note,
        "old": {
            "exists": old_bytes is not None,
            "sha256": sha256_bytes(old_bytes) if old_bytes is not None else None,
            "bytes": len(old_bytes) if old_bytes is not None else None,
            "mode": mode_of(path) if path.exists() else None,
        },
        "new": {
            "sha256": sha256_bytes(new_bytes) if new_bytes is not None else None,
            "bytes": len(new_bytes) if new_bytes is not None else None,
        },
        "_old_bytes": old_bytes,
        "_new_bytes": new_bytes,
        "field_changes": field_changes,
        "line_changes": (line_changes(old_text, new_text) if old_bytes is not None else []),
    }


def public(change: dict) -> dict:
    """去掉内部字节（计划件不塞 68 × 200 KB 的正文）。"""
    return {k: v for k, v in change.items() if not k.startswith("_")}


# ── 源文件校验 ────────────────────────────────────────────────────────────────
def load_source(path: Path, expect_sha: str) -> tuple[bytes, dict]:
    if not path.is_file():
        raise SystemExit(f"FATAL: 源文件不存在：{path}")
    data = path.read_bytes()
    got = sha256_bytes(data)
    problems = []
    if got != expect_sha:
        problems.append(f"sha256 不符：在盘 {got} ≠ --expect-sha256 {expect_sha}")
    lines = data.splitlines()
    if data.count(b"\n") != EXPECTED_LINES or len(lines) != EXPECTED_LINES:
        problems.append(f"行数不符：换行 {data.count(chr(10).encode())} / splitlines {len(lines)} "
                        f"（需恰 {EXPECTED_LINES}）")
    bad = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            bad.append((i, "空行"))
            continue
        try:
            json.loads(line)
        except Exception as exc:  # noqa: BLE001 - 逐行如实报出
            bad.append((i, str(exc)[:60]))
    if bad:
        problems.append(f"不可解析行：{bad[:3]}（共 {len(bad)} 行）")
    if problems:
        for p in problems:
            print(f"FATAL: {p}", file=sys.stderr)
        raise SystemExit(2)
    return data, {"path": str(path), "sha256": got, "bytes": len(data),
                  "lines": len(lines), "unique_ids": len({json.loads(x)["id"] for x in lines})}


# ── 计划：逐件算出「目标内容」 ────────────────────────────────────────────────
def build_plan(root: Path, src_bytes: bytes, src_sha: str) -> dict:
    changes: list[dict] = []

    def add(change: dict) -> None:
        # 目标内容与现状逐字节相同 ⇒ 不是变更（幂等的机制就在这里）
        if change["_new_bytes"] is not None and change["_old_bytes"] == change["_new_bytes"]:
            return
        changes.append(change)

    # ① 新语料在仓内的冻结实体（旧内容 = 不存在）
    ent_path = root / NEW_ENTITY_REL
    ent_old = ent_path.read_bytes() if ent_path.is_file() else None
    add(make_change(root, NEW_ENTITY_REL, group="new_entity",
                    action="write" if ent_old is None else "rewrite",
                    old_bytes=ent_old, new_bytes=src_bytes,
                    field_changes=[{"locator": "(整文件)", "old": None if ent_old is None
                                    else sha256_bytes(ent_old),
                                    "new": src_sha,
                                    "note": "100 行候选语料原样落仓（与 candidate 逐字节同）"}],
                    note="新语料唯一实体；M68_REPO_SOURCE_CANDIDATES 指向它"))

    # ② 68 份训练子集（M0nn.jsonl）：整文件替换
    for tier in M_TIERS:
        p = root / "subsets-v2" / f"{tier}.jsonl"
        old = p.read_bytes() if p.is_file() else None
        add(make_change(root, f"subsets-v2/{tier}.jsonl", group="subsets", action="write",
                        old_bytes=old, new_bytes=src_bytes,
                        field_changes=[{
                            "locator": "(整文件)",
                            "old": None if old is None else
                                   f"sha256={sha256_bytes(old)} bytes={len(old)} lines={old.count(b'\n')}",
                            "new": f"sha256={src_sha} bytes={len(src_bytes)} lines={EXPECTED_LINES}",
                            "note": "100 行逐字节副本（68 档共用同一份）"}]))

    # ③ 子集登记清单：定点改字段（⛔ 不做字符串全局替换）
    mpath = root / MANIFEST
    m_old_bytes = mpath.read_bytes()
    m_old = json.loads(m_old_bytes)
    m_new = json.loads(m_old_bytes)
    fields: list[dict] = []

    def set_field(container: dict, key: str, old_ok: tuple, new_val, locator: str, note: str = ""):
        cur = container.get(key)
        if cur == new_val:
            return
        if cur not in old_ok:
            raise PlanError(f"{MANIFEST}::{locator} 现值 {cur!r} 不在预期集合 {old_ok} 里"
                            f"（在盘状态已漂移，零写入退出）")
        container[key] = new_val
        fields.append({"locator": locator, "old": cur, "new": new_val, "note": note})

    set_field(m_new["source_files"]["m68_frozen"], "relative_path",
              ("subsets-v2/T001.jsonl", NEW_ENTITY_REL), NEW_ENTITY_REL,
              "source_files.m68_frozen.relative_path", "M68 语料实体换名")
    set_field(m_new["source_files"]["m68_frozen"], "sha256",
              (OLD_SOURCE_SHA256, src_sha), src_sha, "source_files.m68_frozen.sha256")
    set_field(m_new["m68"], "source_sha256", (OLD_SOURCE_SHA256, src_sha), src_sha,
              "m68.source_sha256")
    m_by_id = {str(t.get("tier_id")): t for t in m_new.get("tiers") or []}
    missing = [t for t in M_TIERS if t not in m_by_id]
    if missing:
        raise PlanError(f"{MANIFEST} 缺 M 档条目：{missing[:5]}（共 {len(missing)} 档）")
    for tier in M_TIERS:
        ent = m_by_id[tier]
        set_field(ent, "sha256", (OLD_SOURCE_SHA256, src_sha), src_sha, f"tiers[{tier}].sha256")
        set_field(ent, "source_jsonl", ("subsets-v2/T001.jsonl", NEW_ENTITY_REL), NEW_ENTITY_REL,
                  f"tiers[{tier}].source_jsonl", "逐档来源实体换名")
        set_field(ent, "status", ("unchanged", "written"), "written", f"tiers[{tier}].status",
                  "本次切换确实逐档重写（≠ unchanged）")
    # 113 次出现里必须原样保留的那些（算清楚，写进计划供人复核）
    kept = 0
    kept_where = []
    for t in m_new.get("tiers") or []:
        if t.get("sha256") == OLD_SOURCE_SHA256:
            kept += 1
            kept_where.append(str(t.get("tier_id")))
    if m_new["source_files"]["mini"].get("sha256") == OLD_SOURCE_SHA256:
        kept += 1
        kept_where.append("source_files.mini")
    add(make_change(root, MANIFEST, group="manifest", action="edit_json",
                    old_bytes=m_old_bytes, new_bytes=dump_json(m_new).encode("utf-8"),
                    field_changes=fields,
                    note=(f"旧 sha 出现 {m_old_bytes.decode('utf-8').count(OLD_SOURCE_SHA256)} 次；"
                          f"本工具改 {len([f for f in fields if OLD_SOURCE_SHA256 in str(f.get('old'))])} 处 sha、"
                          f"原样保留 {kept} 处（{', '.join(kept_where[:3])}…，T 档与 mini 源不动）")))

    # ④ 数据件 + 日期镜像：判据锚点 1 处 + 逐档 data_source 68 处
    for relpath, group in ((DATA_JSON, "data"), (DATA_JSON_DATED, "dated_mirror")):
        p = root / relpath
        if not p.is_file():
            raise PlanError(f"{relpath} 不存在（数据件是唯一真相源，⛔ 不跳过）")
        raw = p.read_bytes()
        doc = json.loads(raw)
        dfs: list[dict] = []
        ds = (doc.get("meta") or {}).get("criteria_caliber", {}).get("dataset") or {}
        cur = ds.get("version_anchor_sha256")
        if isinstance(cur, str) and "594d41b0" in cur:
            ds["version_anchor_sha256"] = cur.replace("594d41b0", src_sha[:8])
            dfs.append({"locator": "meta.criteria_caliber.dataset.version_anchor_sha256",
                        "old": cur, "new": ds["version_anchor_sha256"]})
        elif not isinstance(cur, str) or src_sha[:8] not in cur:
            raise PlanError(f"{relpath}::version_anchor_sha256 现值 {cur!r} 既非旧锚点也非新锚点")
        rows = doc.get("rows") or []
        hit = 0
        for i, row in enumerate(rows):
            if row.get("data_source") == OLD_DATA_SOURCE:
                row["data_source"] = NEW_DATA_SOURCE
                hit += 1
                dfs.append({"locator": f"rows[{i}].data_source", "old": OLD_DATA_SOURCE,
                            "new": NEW_DATA_SOURCE})
            elif row.get("data_source") == NEW_DATA_SOURCE:
                hit += 1
        if hit == 0:
            raise PlanError(f"{relpath} 里没有一处 data_source 命中旧/新措辞（在盘漂移？）")
        add(make_change(root, relpath, group=group, action="edit_json",
                        old_bytes=raw, new_bytes=dump_json(doc).encode("utf-8"),
                        field_changes=dfs,
                        note="锚点文案 1 处 + 逐档 data_source 68 处（本件不在渲染字段里 ⇒ HTML 不变）"))

    # ⑤ 文案/常量：逐条精确替换（旧、新都不命中 ⇒ 漂移，零写入退出）
    for relpath, group, edits in text_edits(src_sha):
        p = root / relpath
        if not p.is_file():
            raise PlanError(f"{relpath} 不存在")
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        applied: list[dict] = []
        for old, new, note in edits:
            if old in text:
                if text.count(old) != 1:
                    raise PlanError(f"{relpath}: 待替换片段命中 {text.count(old)} 次（须唯一）："
                                    f"{old[:50]!r}")
                line = text[: text.index(old)].count("\n") + 1
                text = text.replace(old, new)
                applied.append({"locator": f"line {line}", "old": old, "new": new, "note": note})
            elif new in text:
                continue                      # 已切换（幂等）
            else:
                raise PlanError(f"{relpath}: 旧片段与新片段都不在盘（漂移）：{old[:50]!r}")
        add(make_change(root, relpath, group=group, action="edit_text",
                        old_bytes=raw, new_bytes=text.encode("utf-8"),
                        field_changes=applied))

    # ⑥ 覆盖审计日志：**只追加**（历史条目一个字节都不动）
    p = root / OVERWRITE_LOG
    raw = p.read_bytes()
    text = raw.decode("utf-8")
    already = f'"postimage_sha256": "{src_sha}"' in text
    recs = []
    if not already:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        for tier in M_TIERS:
            recs.append(json.dumps({
                "tier_id": tier,
                "preimage_sha256": OLD_SOURCE_SHA256,
                "postimage_sha256": src_sha,
                "authorized_by": "tools/matrix/switch_corpus.py --apply",
                "before": {"subset_size": 100, "lines": 100, "bytes": OLD_SOURCE_BYTES},
                "after": {"subset_size": 100, "lines": 100, "bytes": len(src_bytes)},
                "reason": ("★ 语料切换（2026-10-22 第一轮基线）：68 档共用的 100 行训练语料由 mini 冒烟集"
                           "换成 V5 均衡 100 条（12 格分层 · seed 20261022）；条数不变 ⇒ 步数不变"),
                "at": stamp,
            }, ensure_ascii=False))
        text = text + "".join(r + "\n" for r in recs)
    add(make_change(root, OVERWRITE_LOG, group="overwrite_log", action="append_jsonl",
                    old_bytes=raw, new_bytes=text.encode("utf-8"),
                    field_changes=([{"locator": f"append {len(recs)} 行（逐档一条）",
                                     "old": "(无)", "new": recs[0]}] if recs else []),
                    note="只追加；15 条历史条目的旧 sha 原样保留" if not already else "已追加过（幂等）"))

    # ⑦ 跑次清单：**由生成器现算**（不由本工具手写），但仍纳入备份/回滚面
    for relpath in (M68_SRC_JSON, M68_JSON):
        p = root / relpath
        if not p.is_file():
            raise PlanError(f"{relpath} 不存在（跑次清单必须先在盘，否则无法备份/回滚）")
        changes.append({
            "path": relpath, "group": "runner_manifest", "action": "regenerate",
            "note": "由 tests/e2e/generate_matrix68_manifest_source.py 现算（data_sha256 / 逐档 "
                    "subset_sha256 / subset_dir_fingerprint 随之变）",
            "old": {"exists": True, "sha256": sha256_file(p), "bytes": p.stat().st_size,
                    "mode": mode_of(p)},
            "new": {"sha256": None, "bytes": None},
            "_old_bytes": p.read_bytes(), "_new_bytes": None,
            "field_changes": [], "line_changes": [],
        })

    return {
        "changes": changes,
        "old_sha_occurrences_in_manifest": m_old_bytes.decode("utf-8").count(OLD_SOURCE_SHA256),
        "kept_old_sha_in_manifest": kept,
        "warnings": semantic_followups(),
    }


def semantic_followups() -> list[dict]:
    """切换会「让措辞变旧」但**不该由工具自动改**的地方（数值断言/口径解释需人裁定）。"""
    return [
        {"path": DATA_JSON,
         "locator": "meta.criteria_caliber.dataset.train_eval_relation / eval_caliber.discriminative_power",
         "issue": ("文中「该 100 行 (工具,动作) 最大类占 17%（17/100）⇒ 常数预测器上界 17%」已按均衡语料"
                   "（12 格 · 68 档通用）实测**订正**（2026-10-22：6 类 16–17 条 ⇒ 最大类 17/100 = 17%）；"
                   "该断言随语料变化 ⇒ 再次换语料后会重新变旧"),
         "action": "无（2026-10-22 已订正为 17%）；换语料后需重新实测并同步；本工具⛔ 不自动改数值断言"},
        {"path": "subsets-v2/mini 源（仓外：${GRASPO_ELAM_DATA_ROOT}/mini-dataset/mini-short-mm-train.jsonl）",
         "locator": "(整文件)",
         "issue": "旧 100 条仍是该文件的实体；本次**不覆盖、不删除**（历史证据 + T 档 42 件仍引用它）",
         "action": "无（保留）；新语料另起名，不与 mini 同名"},
        {"path": "tests/e2e/make_subsets.py",
         "locator": "main() 的 --repo-root",
         "issue": ("切换后若用**绝对** ``--repo-root`` 重跑 ``--tier-set m68``，它会把绝对路径写进 "
                   "``source_files.m68_frozen.relative_path`` / ``tiers[].source_jsonl``（沿用其既有行为）"),
         "action": "重跑时留在仓根并用默认 ``--repo-root .``；或另派小修把该路径固定为仓根相对"},
        {"path": "tests/e2e/matrix54_manifest{,_source,_fsdp}.json、legacy_matrix54.data.json",
         "locator": "各 1 处旧 sha",
         "issue": "旧 54 档世代的历史留痕，与 68 档无关",
         "action": "⛔ 不改（改了会伪造历史）"},
        {"path": "docs/GRASPO 训练框架功能测试报告.html",
         "locator": "(整文件)",
         "issue": ("实测 data_source / version_anchor_sha256 **都不在渲染字段里** ⇒ 用新数据件重渲染，"
                   "副本与原版**逐字节相同**（本包演练已实证）"),
         "action": "无需改人读件；带日期副本可由 render_matrix 重渲染"},
        {"path": "docs/GRASPO 训练框架功能测试报告.html",
         "locator": "(未决)",
         "issue": "chain_dec / chain_err 的真实语义仍未证（corpus-report ⑤.1）⇒ D1 分层的解释口径需另行裁定",
         "action": "与语料切换解耦；本工具不触碰"},
    ]


def text_edits(src_sha: str) -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """文案/常量落点（逐条：旧片段 → 新片段，命中必须唯一）。"""
    new_short = src_sha[:8]
    return [
        (MAKE_SUBSETS, "constants", [
            ('#: 100 条共用语料的**冻结 sha256**（= mini 源 sha = 数据件判据锚点）。',
             '#: 100 条共用语料的**冻结 sha256**（= 均衡子集 sha = 数据件判据锚点）。',
             "常量注释：不再等价于 mini 源 sha"),
            (f'M68_SOURCE_SHA256 = "{OLD_SOURCE_SHA256}"',
             f'M68_SOURCE_SHA256 = "{src_sha}"',
             "冻结 sha 常量本身"),
            ('#     · 该 sha 同时是数据件判据锚点（100 行 / `594d41b0…`）与',
             f'#     · 该 sha 同时是数据件判据锚点（100 行 / `{new_short}…`）与',
             "docstring 锚点文案"),
            ('#       `subsets-v2.manifest.json::source_files.mini.sha256`。',
             '#       `subsets-v2.manifest.json::source_files.m68_frozen.sha256`。',
             "docstring 指向的清单字段（mini 已不再等价）"),
            ('#   ⇒ 因此**不需要重新造数据**：M 档产物 = 仓内既有 100 条冻结实体的逐档副本\n'
             '#     （`subsets-v2/T001.jsonl` 就是该实体；`mini-dataset/mini-short-mm-train.jsonl` 内容相同，\n'
             '#      但它**不在仓库里**、住在 `${GRASPO_ELAM_DATA_ROOT}` 下 ⇒ 只作次选）。',
             f'#   ⇒ 因此**不需要重新造数据**：M 档产物 = 仓内 100 条冻结实体的逐档副本\n'
             f'#     （`{NEW_ENTITY_REL}` 就是该实体 —— 2026-10-22 语料切换后取代\n'
             f'#      旧 `subsets-v2/T001.jsonl`（= mini 100 条）；旧件保留作历史，不再被 M 档引用）。',
             "docstring 实体说明"),
            ('#: 仓内既有实体（按优先级）：先 `subsets-v2/T001.jsonl`（在仓里），再 mini 源（宿主数据根下）。',
             f'#: 仓内实体（唯一；2026-10-22 语料切换后）：`{NEW_ENTITY_REL}`。\n'
             '#: 历史：切换前为 `subsets-v2/T001.jsonl`（在仓里）→ mini 源（宿主数据根下）。',
             "候选注释"),
            ('M68_REPO_SOURCE_CANDIDATES: tuple[str, ...] = ("subsets-v2/T001.jsonl", MINI_TRAIN_REL)',
             f'M68_REPO_SOURCE_CANDIDATES: tuple[str, ...] = ("{NEW_ENTITY_REL}",)',
             "源实体候选（收窄到唯一实体 ⇒ 不再可能静默取回旧语料）"),
        ]),
        (MEASURE_ELAM, "text", [
            (f'    ... --expect-sha256 {OLD_SOURCE_SHA256}',
             f'    ... --expect-sha256 {src_sha}',
             "用法示例里的 --expect-sha256"),
            ('**已知的 hash 争议（务必按实测值，不要照抄历史数字）**：mini 数据集\n'
             '（`.local/mini-dataset/mini/mini-short-mm-train.jsonl`，100 行 / 208822 B）的\n'
             f'**权威 sha256 = `{OLD_SOURCE_SHA256}`**\n'
             '（两个独立来源一致：① 磁盘文件 `sha256sum`；② `task-d2-token-measure` 的机器产物\n'
             '`artifacts/measure_result.json` 的 `mini_sha256_actual`）。',
             '**已知的 hash 争议（务必按实测值，不要照抄历史数字）**：现行 100 条训练语料\n'
             f'（`{NEW_ENTITY_REL}`，100 行 / 237565 B；2026-10-22 语料切换后取代旧 mini 100 条）的\n'
             f'**权威 sha256 = `{src_sha}`**\n'
             '（旧 mini 数据集 `.local/mini-dataset/mini/mini-short-mm-train.jsonl`，100 行 / 208822 B，\n'
             f'其 sha256 = `{OLD_SOURCE_SHA256}`；旧 sha 的两个独立来源一致：\n'
             '① 磁盘文件 `sha256sum`；② `task-d2-token-measure` 的机器产物 `artifacts/measure_result.json`\n'
             '的 `mini_sha256_actual`）。',
             "权威 sha 段落（旧 mini 事实保留为历史，不抹）"),
        ]),
        ("tools/matrix/enumerate_matrix.py", "text", [
            (f'        "data_source": "{OLD_DATA_SOURCE}",',
             f'        "data_source": "{NEW_DATA_SOURCE}",',
             "逐档 data_source 生成端措辞（★ corpus-report ④.6 #10 漏了这一处）"),
            ('"value": "评测集 = 该档自身训练文件（100 行，sha256 594d41b0…）",',
             f'"value": "评测集 = 该档自身训练文件（100 行，sha256 {new_short}…）",',
             "criteria_caliber 锚点文案"),
        ]),
        ("tools/matrix/build_official.py", "text", [
            ('sha256 594d41b0…', f'sha256 {new_short}…', "判据锚点文案"),
        ]),
    ]


# ── 备份 / 回滚 ───────────────────────────────────────────────────────────────
def backup_paths(root: Path, paths: list[str], extra: list[str]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    bdir = root / WORKSPACE_REL / f"backup-{stamp}"
    (bdir / "files").mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": stamp, "entries": []}
    for relpath in list(paths) + list(extra):
        src = root / relpath
        entry = {"path": relpath, "existed": src.is_file(), "sha256": None,
                 "bytes": None, "mode": None, "backup": None}
        if src.is_file():
            dst = bdir / "files" / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            entry.update({"sha256": sha256_file(src), "bytes": src.stat().st_size,
                          "mode": mode_of(src), "backup": str(dst.relative_to(bdir))})
        manifest["entries"].append(entry)
    (bdir / "backup-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return bdir


def restore(root: Path, bdir: Path) -> list[dict]:
    manifest = json.loads((bdir / "backup-manifest.json").read_text(encoding="utf-8"))
    report = []
    for entry in manifest["entries"]:
        dst = root / entry["path"]
        ok = True
        if entry["existed"]:
            src = bdir / entry["backup"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            os.chmod(dst, int(entry["mode"], 8))
            ok = sha256_file(dst) == entry["sha256"]
        else:
            if dst.exists():
                dst.unlink()
            ok = not dst.exists()
        report.append({"path": entry["path"], "restored": ok,
                       "sha256": entry["sha256"] if entry["existed"] else None})
    return report


# ── 自检四道 ─────────────────────────────────────────────────────────────────
def run(cmd: list[str], cwd: Path, timeout: int = 900) -> dict:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    tail = [ln for ln in (proc.stdout + proc.stderr).strip().splitlines() if ln.strip()]
    return {"cmd": " ".join(cmd), "rc": proc.returncode,
            "tail": tail[-14:], "stdout_bytes": len(proc.stdout)}


def self_check(root: Path, dated_html: Path) -> tuple[list[dict], dict]:
    """落盘后四道自检；返回 (读数列表, HTML 对比结论)。"""
    py = sys.executable
    results: list[dict] = []
    results.append(run([py, CHECK68], root))
    results.append(run([py, GEN68, "--verify", M68_JSON], root))
    results.append(run([py, RENDER, "--template", TEMPLATE, "--data", DATA_JSON,
                        "--out", rel(dated_html, root)], root))
    results.append(run([py, VERIFY, "--template", TEMPLATE, "--data", DATA_JSON,
                        "--html", HTML_FROZEN, "--quiet"], root))
    results.append(run([py, VERIFY, "--template", TEMPLATE, "--data", DATA_JSON,
                        "--html", rel(dated_html, root), "--quiet"], root))

    frozen = (root / HTML_FROZEN).read_bytes()
    dated = dated_html.read_bytes() if dated_html.is_file() else b""
    html = {"frozen_sha256": sha256_bytes(frozen), "dated_sha256": sha256_bytes(dated),
            "dated_path": rel(dated_html, root), "identical": frozen == dated}
    if not html["identical"]:
        a = frozen.decode("utf-8", "replace").splitlines()
        b = dated.decode("utf-8", "replace").splitlines()
        spans = line_changes("\n".join(a), "\n".join(b), limit=10)
        html["diff_spans"] = spans
    return results, html


# ── 计划表打印 ────────────────────────────────────────────────────────────────
def print_plan(plan: dict, src: dict, root: Path) -> None:
    ch = plan["changes"]
    print(f"\n=== 语料切换计划（共 {len(ch)} 件）===")
    print(f"源：{src['path']}\n    sha256={src['sha256']}  {src['bytes']:,} B  {src['lines']} 行  "
          f"{src['unique_ids']} 个唯一 id")
    print(f"仓根：{root}")
    group_titles = {
        "subsets": "① 68 份训练子集", "new_entity": "①′ 新语料冻结实体",
        "manifest": "② 子集登记清单", "data": "③ 数据件（chmod 444）",
        "dated_mirror": "③′ 数据件日期镜像", "runner_manifest": "④ 跑次清单（生成器现算）",
        "constants": "⑤ 生成器常量/文案", "text": "⑤′ 其余文案",
        "overwrite_log": "⑥ 覆盖审计日志（只追加）",
    }
    for key, title in group_titles.items():
        items = [c for c in ch if c["group"] == key]
        if not items:
            continue
        print(f"\n{title}（{len(items)} 件）")
        for c in items:
            old = c["old"]
            old_s = (f"{str(old['sha256'])[:12]}…/{old['bytes']:,}B" if old["exists"]
                     else "(新文件)")
            new_s = (f"{str(c['new']['sha256'])[:12]}…/{c['new']['bytes']:,}B"
                     if c["new"]["sha256"] else "(生成器现算)")
            extra = ""
            if key == "subsets":
                extra = "   ← 100 行整文件替换"
            elif c["field_changes"]:
                extra = f"   ← {len(c['field_changes'])} 处字段/片段"
            print(f"  {c['path']:<58} {c['action']:<12} {old_s:>22} → {new_s:>22}{extra}")
            if key in ("manifest", "data", "dated_mirror") and c["field_changes"]:
                n_sha = sum(1 for f in c["field_changes"] if "594d41b0" in str(f.get("old")))
                n_ds = sum(1 for f in c["field_changes"] if f.get("old") == OLD_DATA_SOURCE)
                print(f"      字段明细：sha 类 {n_sha} 处 / data_source {n_ds} 处 / "
                      f"其它 {len(c['field_changes']) - n_sha - n_ds} 处；"
                      f"行区间 {c['line_changes'][:6]}{'…' if len(c['line_changes']) > 6 else ''}")
            elif c["field_changes"] and key in ("constants", "text"):
                for f in c["field_changes"]:
                    print(f"      {f['locator']}: {f['note']}")
    print(f"\n清单旧 sha 出现 {plan['old_sha_occurrences_in_manifest']} 次："
          f"改 {plan['old_sha_occurrences_in_manifest'] - plan['kept_old_sha_in_manifest']} 处 / "
          f"原样保留 {plan['kept_old_sha_in_manifest']} 处（T 档 + mini 源，⛔ 不可全局替换）")
    print(f"\n语义跟随项（本工具⛔ 不改，需人裁定）：{len(plan.get('warnings') or [])} 条")
    for w in plan.get("warnings") or []:
        print(f"  · {w['path']}::{w['locator']}\n    {w['issue']}")

    locked = sorted({c["path"] for c in ch if (root / c["path"]).exists() and is_frozen(root / c["path"])})
    if locked:
        print(f"\n⚠ 冻结件（chmod 444）将被改写：{locked}")
        print("  --apply 前需人工解锁：  chmod 644 " + " ".join(locked))
        print("  或显式授权本工具临时解锁并复原：  --unlock-frozen")


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把 68 档 M001–M068 的训练语料从旧 100 条切到新 100 条（默认 dry-run，零写入）")
    ap.add_argument("--source-jsonl", required=True, type=Path, help="新 100 条语料（JSONL）")
    ap.add_argument("--expect-sha256", required=True, help="新语料的期望 sha256（钉死字节）")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", action="store_true", default=True, help="只出计划（默认）")
    grp.add_argument("--apply", action="store_true", help="真正落盘（含备份 + 四道自检 + 失败回滚）")
    ap.add_argument("--repo-root", default=".", type=Path, help="仓根（默认当前目录）")
    ap.add_argument("--plan-out", default=None, type=Path, help="计划件落点（JSON；dry-run 也写）")
    ap.add_argument("--unlock-frozen", action="store_true",
                    help="★ 显式授权本工具对 chmod 444 的冻结件临时 644→写→复原 444（默认绝不 chmod）")
    args = ap.parse_args(argv)
    root = args.repo_root.resolve()
    if not (root / MANIFEST).is_file():
        print(f"FATAL: {root} 不像仓根（缺 {MANIFEST}）", file=sys.stderr)
        return 2

    src_bytes, src = load_source(args.source_jsonl, args.expect_sha256)
    try:
        plan = build_plan(root, src_bytes, src["sha256"])
    except PlanError as exc:
        print(f"FATAL: 计划构建失败（零写入）：{exc}", file=sys.stderr)
        return 4

    changes = plan["changes"]
    mode = "apply" if args.apply else "dry-run"
    doc = {
        "schema": "corpus-switch-plan/v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "repo_root": str(root),
        "source": src,
        "anchor": {"old_sha256": OLD_SOURCE_SHA256, "old_bytes": OLD_SOURCE_BYTES,
                   "new_sha256": src["sha256"], "new_bytes": src["bytes"],
                   "old_data_source": OLD_DATA_SOURCE, "new_data_source": NEW_DATA_SOURCE},
        "summary": {
            "files_to_change": len(changes),
            "by_group": {g: sum(1 for c in changes if c["group"] == g)
                         for g in {c["group"] for c in changes}},
            "manifest_sha_occurrences": plan["old_sha_occurrences_in_manifest"],
            "manifest_sha_changed": (plan["old_sha_occurrences_in_manifest"]
                                     - plan["kept_old_sha_in_manifest"]),
            "manifest_sha_kept": plan["kept_old_sha_in_manifest"],
            "noop": len(changes) == 0,
        },
        "files": [public(c) for c in changes],
        "warnings": plan["warnings"],
        "html_expectation": ("data_source / version_anchor_sha256 均不在渲染字段里 ⇒ "
                             "渲染副本应与原版 HTML 逐字节相同"),
    }

    print_plan(plan, src, root)

    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
        print(f"\n计划件：{args.plan_out}")

    if not args.apply:
        print(f"\nDRY-RUN：未写仓库任何文件（{len(changes)} 件待改）。加 --apply 才落盘。")
        return 0

    # ── 幂等出口 ─────────────────────────────────────────────────────────────
    # 两份跑次清单由生成器现算（本工具不手写其字节）⇒ 其「是否已一致」只能问生成器的 --verify；
    # 其余文件的目标内容都是现算的 ⇒ 目标 == 现状即已切换。
    nonrunner = [c for c in changes if c["group"] != "runner_manifest"]
    if not nonrunner:
        same = all(sha256_file(root / "subsets-v2" / f"{t}.jsonl") == src["sha256"] for t in M_TIERS)
        chk = run([sys.executable, GEN68, "--verify", M68_JSON], root)
        if same and chk["rc"] == 0:
            print(f"\n已切换（sha 匹配），无变更。rc=0（68 档在盘 sha 全 == {src['sha256'][:12]}…：{same}；"
                  f"跑次清单 --verify rc=0）")
            return 0
        print("\n子集/数据件已切换，但跑次清单滞后 ⇒ 仅重算清单（不重写其它件）")

    # ── 冻结件闸门（⛔ 默认绝不 chmod）───────────────────────────────────────
    locked = [c["path"] for c in nonrunner
              if (root / c["path"]).exists() and is_frozen(root / c["path"])]
    if locked and not args.unlock_frozen:
        print("\nFATAL: 以下冻结件（chmod 444）在计划里，但未授权自动解锁 —— 已停在写入之前（零写入）：",
              file=sys.stderr)
        for p in locked:
            print(f"  - {p}", file=sys.stderr)
        print("  处置（二选一）：① 人工解锁后重跑：chmod 644 " + " ".join(locked)
              + "；② 显式授权：追加 --unlock-frozen（本工具临时 644 → 写 → 复原 444 并告警）",
              file=sys.stderr)
        return 3

    # ── 备份（含生成器产物与带日期副本：回滚面必须覆盖它们）────────────────
    dated_html = root / HTML_DATED_TMPL.format(date=datetime.now().strftime("%Y%m%d"))
    extra = [M68_SRC_JSON, M68_JSON, rel(dated_html, root)]
    bdir = backup_paths(root, [c["path"] for c in changes], extra)
    print(f"\n备份：{bdir}（{len(changes) + len(extra)} 条登记，含 mode）")

    unlocked: list[tuple[Path, int]] = []
    try:
        # ── 写：先实体/子集，再清单/数据，再文案，最后日志 ──────────────────
        order = {"new_entity": 0, "subsets": 1, "manifest": 2, "data": 3,
                 "dated_mirror": 4, "constants": 5, "text": 6, "overwrite_log": 7}
        for c in sorted(changes, key=lambda x: order.get(x["group"], 9)):
            if c["action"] == "regenerate":
                continue
            path = root / c["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and is_frozen(path):
                unlocked.append((path, stat.S_IMODE(path.stat().st_mode)))
                os.chmod(path, 0o644)
                print(f"⚠ 已临时解锁冻结件：{c['path']}（写入后复原 444）")
            path.write_bytes(c["_new_bytes"])
            got = sha256_file(path)
            if got != c["new"]["sha256"]:
                raise RuntimeError(f"落盘校验失败：{c['path']} sha256={got} "
                                   f"≠ 计划 {c['new']['sha256']}")
            print(f"  [wrote] {c['path']}  {c['new']['bytes']:,} B  sha256={got[:12]}…")

        # 生成器现算两份跑次清单（本工具不手写它们的字节）
        print("\n[gen] tests/e2e/generate_matrix68_manifest_source.py")
        gen = run([sys.executable, GEN68], root)
        for ln in gen["tail"]:
            print(f"  {ln}")
        if gen["rc"] != 0:
            raise RuntimeError(f"生成器 rc={gen['rc']}（跑次清单未生成）")

        # 四道自检
        results, html = self_check(root, dated_html)
        print("\n=== 落盘后自检读数 ===")
        for r in results:
            print(f"  rc={r['rc']}  {r['cmd']}")
            for ln in r["tail"][-6:]:
                print(f"        {ln}")
        print(f"  HTML 对比：原版 sha256={html['frozen_sha256'][:16]}…  "
              f"副本 sha256={html['dated_sha256'][:16]}…  "
              f"{'逐字节相同（数据件改动不在渲染字段里）' if html['identical'] else '有差异'}")
        if not html["identical"]:
            print(f"        差异行区间（老坐标）：{html.get('diff_spans')}")

        bad = [r for r in results if r["rc"] != 0]
        if bad or not html["identical"]:
            raise RuntimeError("自检未全绿：" + "；".join(f"rc={r['rc']} {r['cmd']}" for r in bad)
                               + ("" if html["identical"] else "；HTML 副本与原版有差异"))

        # 复原冻结位
        for path, old_mode in unlocked:
            os.chmod(path, old_mode)
            print(f"  冻结位已复原：{rel(path, root)} → {old_mode:04o}")
        print(f"\nAPPLY OK：{len(changes)} 件已切换；四道自检全绿；备份 {bdir}")
        (bdir.parent / f"apply-result-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps({"plan": doc, "checks": results, "html": html,
                        "backup": str(bdir)}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        return 0

    except Exception as exc:  # noqa: BLE001 - 任何异常都要走回滚（退路优先）
        print(f"\nFATAL: {exc}\n→ 回滚到备份 {bdir}", file=sys.stderr)
        for path, old_mode in unlocked:
            try:
                os.chmod(path, old_mode)
            except OSError:
                pass
        report = restore(root, bdir)
        bad = [r for r in report if not r["restored"]]
        print(f"回滚：{len(report) - len(bad)}/{len(report)} 件已还原"
              + ("" if not bad else f"；⚠ 未还原：{[b['path'] for b in bad]}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
