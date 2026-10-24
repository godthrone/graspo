#!/usr/bin/env python3
"""sign_graspo_invariants.py — 判据③ GRASPO 逐档签署件的**唯一产出者**。

## 为什么需要本文件（基线审查 🟡/🔴-5）

判据③ 对 GRASPO 的口径是「按《GRASPO 算法说明》（`docs/graspo-algorithm.html`）**逐条判断**，
算法逻辑严禁改动」。消费端 `scripts/record_checks.py::judge_graspo_algo()` 只认**签署件**
`<algo-checks-dir>/<tier>.json`（第 9 节 8 条不变式逐条 + `checker`/`checked_at`）：
缺件 ⇒ 不写 `algo_checks_passed` 键 ⇒ 回填器 fail-closed ⇒ **整批不写**。
而 2026-10-24 前仓内 GRASPO 签署件 **0/26 份**、**无生成器、无模板、无校验前例** ⇒ 26 个
GRASPO 档当天不可能判出 ③。本文件补产出渠道：**清单从算法说明现解析、档集合从清单现算、
证据必须外部给出**，⛔ 不硬编码条文、⛔ 不预填 `true`。

## 单一真相源（§1.4：本文件不另立口径）

| 量 | 唯一来源（本文件只**读**） |
|---|---|
| 8 条不变的**条文与顺序** | `--algo-html`（缺省 `docs/graspo-algorithm.html`）**第 9 节** `<ol><li>` 逐条现解析 |
| 8 条不变的**id / 名称** | `scripts/record_checks.py::GRASPO_INVARIANTS`（消费端的契约；⛔ 本文件不抄一份） |
| **哪些档是 GRASPO 档** | `--manifest`（缺省 `tests/e2e/matrix68_manifest.json`）里 `algorithm=="GRASPO"` 现算 |
| 算法关键参数的**默认值** | `graspo.core.schema.TrainingConfig / RewardConfig` 的字段默认值（现读） |
| 组切片是否合法 | `native.assignment.validate_rollout_slicing` / `msswift.group_sharding`（现算，见 `tier_check`） |
| 每条不变的**结论与证据** | `--evidence` 给的外部证据件（本文件**只校验、不用默认值、不生成结论**） |

## 契约（⛔ 少一条就 rc=2 且**零写入**）

1. 第 9 节必须恰好解析出 8 条 `<li>`，且条数 == 消费端 `GRASPO_INVARIANTS` 的长度；
2. `--evidence` 必须逐条覆盖 8 个 id（支持 `S1…S8` 别名），**无缺、无重、无未知**；
3. 每条 `verdict` 必须是 `true`（本条核过）——出现 `false` ⇒ **拒绝签署**（有 false 就不签）；
4. 每条 `evidence` 必须是非空数组，元素形如 `path/to/file.py:123` 或
   `tests/…::test_name`（可机械定位）；`checker` / `checked_at` 必须非空（签署人/时间）；
5. 档号必须属于清单现算的 GRASPO 档集合（⛔ 不接受手填/不存在的档号）；
6. 该档配置 **不得覆写**算法关键参数（`training.temperature` / `policy_ratio_clip_eps` /
   `rollout_group_size` / `rollout_max_retries` / `reject_unparseable_groups` / `reward.*`）；
   覆写了就不是"共享实现"，须逐档专有证据 ⇒ 本文件**拒签该档**（rc=2）；
7. 该档组切片合法（native: `G % dp == 0`；ms-swift: `per_device×steps ≡ 0 (mod G)`）——
   现算不过 ⇒ 拒签该档（组完整性 ① 在本档不成立，不能签）。

## 落点与幂等

- 签署件落 `--out-dir`（缺省 `samples/configs/matrix68-checks/`），文件名 `graspo-<tier>.json`；
- `--algo-checks-dir DIR` 额外写一份**字节相同**的 `<tier>.json` 到 `DIR`
  （= 消费端 `record_checks.py --algo-checks-dir` 的找件命名；跑次侧填 `<run-root>/algo-checks/`）；
- 内容不变 ⇒ **不重写**（`git status` 干净）；内容变了 ⇒ 必须给 `--overwrite-log`（`.local/` 下）
  才落盘，否则 rc=2 —— 防止"悄悄改签署时间/证据"。

用法::

    # ① 先看计划（默认不写盘）
    python3 tools/matrix/sign_graspo_invariants.py --all \
        --evidence samples/configs/matrix68-checks/graspo-invariants-evidence.json

    # ② 落盘（26 档签署件；同时铺一份给 record_checks 的 <tier>.json）
    python3 tools/matrix/sign_graspo_invariants.py --all --apply \
        --evidence samples/configs/matrix68-checks/graspo-invariants-evidence.json \
        --algo-checks-dir <run-root>/algo-checks

    # ③ 只签某几档
    python3 tools/matrix/sign_graspo_invariants.py --tiers M019,M020 --apply --evidence …

退出码：0 = 计划/落盘成功；2 = 拒签或用法错误（**零写入**）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = ROOT / "docs" / "graspo-algorithm.html"
DEFAULT_MANIFEST = ROOT / "tests" / "e2e" / "matrix68_manifest.json"
DEFAULT_OUT_DIR = ROOT / "samples" / "configs" / "matrix68-checks"
SIG_PREFIX = "graspo-"
SCHEMA_VERSION = "graspo-graspo-tier-signature/1"

#: 算法关键参数（该档**不得**覆写；否则"共享实现"这一前提不成立 ⇒ 拒签该档）。
#: 键集合从 schema 的字段名现取，⛔ 不在此另写默认值。
ALGO_KEYS: tuple[tuple[str, str], ...] = (
    ("training", "temperature"),
    ("training", "policy_ratio_clip_eps"),
    ("training", "rollout_group_size"),
    ("training", "rollout_max_retries"),
    ("training", "reject_unparseable_groups"),
    ("training", "top_p"),
    ("reward", "check_think"),
    ("reward", "check_json_markdown"),
    ("reward", "marker_reward_weight"),
    ("reward", "content_reward_weight"),
    ("reward", "anti_useless_str_half_reward_len"),
    ("reward", "numeric_tolerance"),
)

#: 证据元素的可机械定位形态：`path/file.py:123` 或 `tests/...::test_name`。
RE_EVIDENCE = re.compile(r"^(?:[\w./\\-]+\.(?:py|sh|json|yaml|yml|html|md):\d+(?:-\d+)?"
                         r"|[\w./\\-]+::[\w:\[\]{}\-., =]+)$")

_TAG_RE = re.compile(r"<[^>]+>")


class Refusal(Exception):
    """拒签（rc=2）。消息必须点名**缺什么证据 / 违反了哪条**。"""


# ── 读：算法说明第 9 节 ─────────────────────────────────────────────────────
def section9_items(html_path: Path) -> list[str]:
    """从《算法说明》现解析第 9 节「不变式清单」的 8 条条文（⛔ 不硬编码条文）。"""
    try:
        html = html_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"算法说明不可读：{html_path}（{exc}）") from exc
    m = re.search(r"<h2>\s*9\.[^<]*不变式清单\s*</h2>(.*?)(?:<h2>|<footer)", html, re.S)
    if not m:
        raise Refusal(f"{html_path} 找不到「9. 不变式清单」小节 ⇒ 无法确定逐条清单")
    body = m.group(1)
    block = re.search(r"<ol>(.*?)</ol>", body, re.S)
    if not block:
        raise Refusal(f"{html_path} 第 9 节里没有 <ol> 清单")
    items = [_TAG_RE.sub("", x).strip() for x in re.findall(r"<li>(.*?)</li>", block.group(1), re.S)]
    items = [re.sub(r"\s+", " ", x) for x in items]
    if len(items) != 8:
        raise Refusal(f"第 9 节解析出不 {len(items)} 条（期望 8 条）：{items} ⇒ 算法说明与"
                      "消费端 record_checks.GRASPO_INVARIANTS 已分叉，先对齐文档再签")
    return items


def consumer_invariants() -> tuple[tuple[str, str], ...]:
    """消费端契约 = `scripts/record_checks.py::GRASPO_INVARIANTS`（⛔ 不抄一份）。"""
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import record_checks  # noqa: PLC0415

    got = tuple(record_checks.GRASPO_INVARIANTS)
    if len(got) != 8:
        raise Refusal(f"消费端 GRASPO_INVARIANTS 是 {len(got)} 条（期望 8 条）⇒ 契约漂移")
    return got


# ── 读：清单现算 GRASPO 档集合 ──────────────────────────────────────────────
def graspo_tiers(manifest_path: Path) -> dict[str, dict]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"清单不可读：{manifest_path}（{exc}）") from exc
    tiers = manifest.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise Refusal(f"清单没有非空 tiers[]：{manifest_path}")
    out = {str(t.get("tier_id")): t for t in tiers if str(t.get("algorithm") or "") == "GRASPO"}
    if not out:
        raise Refusal(f"清单里算不出任何 GRASPO 档：{manifest_path}")
    return out


# ── 读：外部证据（本文件只校验，不生成结论）──────────────────────────────────
def _assert_coordinate(iid: str, item: str) -> None:
    """证据坐标必须**当下可核**：文件在盘 ∧ 行号在文件长度内；或测试函数名真存在。

    ⛔ 不做"像文件路径就算"的宽松判定：指向不存在的文件或已改名的单测，
    等于一条不可复核的结论（判据③ 要求的正是"可核"）。
    """
    if "::" in item:
        rel, _, test_name = item.partition("::")
        path = ROOT / rel
        if not path.is_file():
            raise Refusal(f"{iid}: 证据指向不存在的测试文件 {rel}")
        src = path.read_text(encoding="utf-8", errors="replace")
        short = test_name.split("[", 1)[0]
        if f"def {short}(" not in src:
            raise Refusal(f"{iid}: 证据引用的单测 {rel}::{short} 在文件里不存在（改名/删除了？）")
        return
    rel, _, lineno = item.rpartition(":")
    path = ROOT / rel
    if not path.is_file():
        raise Refusal(f"{iid}: 证据指向不存在的文件 {rel}")
    try:
        first = int(lineno.split("-", 1)[0])
        last = int(lineno.split("-", 1)[1]) if "-" in lineno else first
    except ValueError as exc:
        raise Refusal(f"{iid}: 证据行号不是整数：{item}") from exc
    total = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    if not (1 <= first <= last <= total):
        raise Refusal(f"{iid}: 证据行号越界 {item}（该文件共 {total} 行）")


def load_evidence(path: Path, ids: tuple[str, ...]) -> dict[str, dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"证据件不可读：{path}（{exc}）") from exc
    entries = doc.get("invariants") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not entries:
        raise Refusal(f"证据件 {path} 缺非空 invariants[] 数组")

    alias = {f"S{i}": iid for i, iid in enumerate(ids, start=1)}
    got: dict[str, dict] = {}
    for idx, ent in enumerate(entries, start=1):
        if not isinstance(ent, dict):
            raise Refusal(f"证据件第 {idx} 项不是对象：{ent!r}")
        raw_id = str(ent.get("invariant") or ent.get("id") or "").strip()
        iid = raw_id if raw_id in ids else alias.get(raw_id.upper(), "")
        if not iid:
            raise Refusal(f"证据件出现未知不变式 id：{raw_id!r}（只认 {list(ids)} 或 S1…S8）")
        if iid in got:
            raise Refusal(f"证据件不变式 id 重复：{iid!r}（每条只能签一次）")
        verdict = ent.get("verdict")
        if not isinstance(verdict, bool):
            raise Refusal(f"{iid}: verdict 必须是布尔值，实际 {verdict!r}")
        if verdict is not True:
            raise Refusal(f"{iid}: verdict=false ⇒ 该条**未核过**，8 条不全 true 就不签"
                          f"（缺什么证据：{ent.get('evidence')!r}）")
        evidence = ent.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise Refusal(f"{iid}: evidence 为空 ⇒ 无据可核，拒签")
        for item in evidence:
            if not isinstance(item, str) or not RE_EVIDENCE.match(item.strip()):
                raise Refusal(f"{iid}: evidence 项须为 `path/to/file.py:123` 或 "
                              f"`tests/…::test_name`，实际 {item!r}")
        checker = str(ent.get("checker") or "").strip()
        checked_at = str(ent.get("checked_at") or "").strip()
        if not checker:
            raise Refusal(f"{iid}: 缺签名 checker（检查人）")
        if not checked_at:
            raise Refusal(f"{iid}: 缺签名 checked_at（检查时间）")
        try:
            datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise Refusal(f"{iid}: checked_at 不是 ISO 时间：{checked_at!r}") from exc
        for item in evidence:
            _assert_coordinate(iid, item.strip())
        got[iid] = {"verdict": True, "evidence": [x.strip() for x in evidence],
                    "checker": checker, "checked_at": checked_at,
                    "note": str(ent.get("note") or "").strip()}
    missing = [i for i in ids if i not in got]
    if missing:
        raise Refusal(f"未覆盖的不变式 {len(missing)} 条：{missing} ⇒ 8 条不齐，拒签")
    if len(got) != len(ids):
        raise Refusal(f"覆盖 {len(got)} 条 ≠ {len(ids)} 条（多余/未知条目）")
    return got


# ── 逐档现算：配置覆写检查 + 组切片合法性 ────────────────────────────────────
def _schema_defaults() -> dict[tuple[str, str], object]:
    sys.path.insert(0, str(ROOT / "src"))
    from graspo.core.schema import RewardConfig, TrainingConfig  # noqa: PLC0415

    defaults: dict[tuple[str, str], object] = {}
    for section, cls in (("training", TrainingConfig), ("reward", RewardConfig)):
        for key, field in cls.model_fields.items():
            defaults[(section, key)] = field.get_default(call_default_factory=True)
    return defaults


def _load_yaml(path: Path) -> dict:
    import yaml  # noqa: PLC0415

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refusal(f"档位配置不可读：{path}（{exc}）") from exc
    if not isinstance(doc, dict):
        raise Refusal(f"档位配置不是 YAML 映射：{path}")
    return doc


def tier_check(tier: str, tier_doc: dict, *, manifest_rel: str) -> dict:
    """该档的机核事实：配置是否覆写算法参数 + 组切片是否合法（现算，⛔ 不猜）。"""
    cfg_field = str(tier_doc.get("config") or "").strip()
    if not cfg_field:
        raise Refusal(f"{tier}: 清单未登记 config 路径 ⇒ 无法核该档配置")
    cfg_path = ROOT / cfg_field
    cfg = _load_yaml(cfg_path)
    defaults = _schema_defaults()

    overrides: dict[str, dict] = {}
    for section, key in ALGO_KEYS:
        configured = (cfg.get(section) or {}).get(key, "<未设置>")
        default = defaults.get((section, key), "<schema 无此字段>")
        same = configured in ("<未设置>", default)
        if not same:
            overrides[f"{section}.{key}"] = {"configured": configured, "default": default}
    if overrides:
        raise Refusal(f"{tier}: 该档覆写了算法关键参数 {overrides} ⇒ 「共享实现证据」不成立，"
                      "必须逐档专有证据后再签（本文件不预填）")

    backend = str(cfg.get("backend") or tier_doc.get("backend") or "").strip()
    g = int((cfg.get("training") or {}).get("rollout_group_size",
                                             defaults[("training", "rollout_group_size")]))
    slicing: dict
    if backend == "native":
        dp = int((cfg.get("native") or {}).get("dp_size", 1))
        from graspo.flow.backend.native.assignment import (  # noqa: PLC0415
            rollouts_per_rank,
            validate_rollout_slicing,
        )

        try:
            validate_rollout_slicing(g, dp)
            rpr = rollouts_per_rank(g, dp)
        except Exception as exc:  # noqa: BLE001 - 切片不合法即不能签
            raise Refusal(f"{tier}: native 组切片不合法（G={g}, dp={dp}）：{exc}") from exc
        slicing = {"kind": "native", "rollout_group_size": g, "dp_size": dp,
                   "rollouts_per_rank": rpr,
                   "why": (f"validate_rollout_slicing(G={g}, dp={dp}) 通过 ⇒ G%dp==0、"
                           f"每卡 {rpr} 条、各卡槽位并集 = 整组 G 条")}
    elif backend in ("msswift", "ms-swift"):
        from graspo.flow.backend.msswift.group_sharding import (  # noqa: PLC0415
            local_batch_rows,
            steps_per_generation_for,
        )

        per_device = int((cfg.get("msswift") or {}).get("per_device_train_batch_size", 1) or 1)
        world = int((cfg.get("msswift") or {}).get("nproc_per_node", 1) or 1) * int(
            (cfg.get("msswift") or {}).get("nnodes", 1) or 1)
        steps = steps_per_generation_for(per_device, g)
        rows = local_batch_rows(per_device, g, steps)
        if rows % g != 0:
            raise Refusal(f"{tier}: ms-swift 每进程行数 {rows} 不是 G={g} 的整数倍 ⇒ 会切组")
        slicing = {"kind": "msswift", "rollout_group_size": g, "per_device": per_device,
                   "world_size": world, "steps_per_generation": steps,
                   "local_rows": rows, "groups_per_rank": rows // g,
                   "why": (f"steps_per_generation_for({per_device}, G={g})={steps}、"
                           f"local_batch_rows={rows} ⇒ 每进程恰 {rows // g} 个整组")}
    else:
        raise Refusal(f"{tier}: 后端 {backend!r} 不在 native/msswift 之内 ⇒ 无组切片口径可核")

    return {
        "config": cfg_field,
        "manifest": manifest_rel,
        "backend": backend,
        "algo_param_overrides": overrides,
        "defaults_source": "src/graspo/core/schema.py（TrainingConfig/RewardConfig 字段默认值）",
        "group_slicing": slicing,
    }


# ── 组装签署件 ──────────────────────────────────────────────────────────────
def build_doc(tier: str, tier_doc: dict, *, doc_items: list[str],
              ids: tuple[tuple[str, str], ...], evidence: dict[str, dict],
              check: dict, manifest_rel: str, html_rel: str, caveats: list[str]) -> dict:
    invariants = []
    for (iid, name), ordinal, doc_text in zip(ids, range(1, len(ids) + 1), doc_items, strict=True):
        ent = evidence[iid]
        tier_line = (f"{check['config']}: 该档未覆写算法关键参数（核对见 tier_check）")
        invariants.append({
            "id": iid,
            "ordinal": ordinal,
            "name": name,
            "doc_text": doc_text,
            "doc_source": f"{html_rel} 第 9 节第 {ordinal} 条",
            "verdict": True,
            "pass": True,
            "evidence": [*ent["evidence"], tier_line],
            "checker": ent["checker"],
            "checked_at": ent["checked_at"],
            "note": ent["note"] or None,
        })
    all_evidence: list[str] = []
    for inv in invariants:
        all_evidence.extend(inv["evidence"])
    items = [{"id": inv["id"], "pass": True,
              "evidence": "；".join(inv["evidence"])} for inv in invariants]
    return {
        "$schema_version": SCHEMA_VERSION,
        "doc": ("判据③「算法正确性检查通过」的 GRASPO 逐档签署件：《GRASPO 算法说明》第 9 节 8 条"
                "不变式逐条核验结论。判据原文见《GRASPO 训练框架功能测试报告》第 2 章判据③，"
                "逐条清单的唯一来源是算法说明第 9 节（本文件条文为现解析，未手抄）。"),
        "how_to_change": ("⛔ 不得手填 true：verdict 只能来自真读过被引代码/跑过被引单测后的结论；"
                          "相关代码一改必须重核并更新 checked_at/checker；"
                          "缺件/不合契约时 record_checks.py 不写 algo_checks_passed 键（fail-closed）。"),
        "tier": tier,
        "algo": "GRASPO",
        "algorithm": "GRASPO",
        "verdict": "pass",
        "checker": ", ".join(sorted({inv["checker"] for inv in invariants})),
        "checked_at": invariants[0]["checked_at"],
        "manifest": manifest_rel,
        "manifest_tier": {k: tier_doc.get(k) for k in
                          ("tier_id", "algorithm", "backend", "tuner_type", "mode",
                           "parallel_dims", "cards")},
        "tier_config": check["config"],
        "tier_check": check,
        "invariants": invariants,
        "items": items,
        "evidence": all_evidence,
        "caveats": caveats,
    }


# ── 落盘：幂等 + 变更台账 ───────────────────────────────────────────────────
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def emit(path: Path, text: str, *, tier: str, overwrite_log: Path | None,
         ledger_note: str, applied: list[str], unchanged: list[str]) -> None:
    """内容不变 ⇒ 不重写；有变更 ⇒ 必须有 overwrite-log（否则拒签，零写入）。"""
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == text:
        unchanged.append(str(path))
        return
    if old is not None and overwrite_log is None:
        raise Refusal(f"{path} 内容将变化（sha {_sha(old)[:12]} → {_sha(text)[:12]}），"
                      "但未给 --overwrite-log ⇒ 拒绝静默改写签署件")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    applied.append(str(path))
    if old is not None and overwrite_log is not None:
        overwrite_log.parent.mkdir(parents=True, exist_ok=True)
        line = (f"{datetime.now(timezone.utc).replace(microsecond=0).isoformat()} "
                f"tier={tier} path={path} old_sha256={_sha(old)} new_sha256={_sha(text)} "
                f"{ledger_note}\n")
        with overwrite_log.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="判据③ GRASPO 逐档签署件生成器（第 9 节 8 条不变式；缺证据即拒签、rc=2、零写入）",
        epilog="⛔ 本文件不预填 true、不发明条文、不放宽 record_checks.py 的判定；"
               "证据必须由 --evidence 外部给出（含实现坐标与单测名）。")
    ap.add_argument("--algo-html", default=str(DEFAULT_HTML),
                    help=f"《GRASPO 算法说明》（解析第 9 节 8 条条文）；缺省 {_rel(DEFAULT_HTML)}")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help=f"跑次侧清单（现算 GRASPO 档集合）；缺省 {_rel(DEFAULT_MANIFEST)}")
    ap.add_argument("--evidence", default=None,
                    help="逐条不变式的外部证据件（JSON；每条 {invariant,verdict,evidence[],"
                         "checker,checked_at}）；--apply 时必填")
    ap.add_argument("--tiers", default=None, help="逗号分隔档号（如 M019,M020）；或与 --all 二选一")
    ap.add_argument("--all", action="store_true", help="清单现算的全部 GRASPO 档")
    ap.add_argument("--apply", action="store_true", help="真落盘（默认只打印计划，不写文件）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"签署件落点（文件名 {SIG_PREFIX}<tier>.json）；缺省 {_rel(DEFAULT_OUT_DIR)}")
    ap.add_argument("--algo-checks-dir", default=None,
                    help="额外写一份字节相同的 <tier>.json 到该目录（= record_checks.py "
                         "--algo-checks-dir 的找件命名）")
    ap.add_argument("--overwrite-log", default=None,
                    help="内容变更时的台账（**必须在 .local/ 下**；缺它则内容一有变化就 rc=2）")
    args = ap.parse_args(argv)

    try:
        # ── 校验 1/2/3/4：条文、契约、证据 ────────────────────────────────
        html_path = Path(args.algo_html)
        doc_items = section9_items(html_path)
        ids = consumer_invariants()
        if len(doc_items) != len(ids):
            raise Refusal(f"第 9 节 {len(doc_items)} 条 ≠ 消费端 {len(ids)} 条 ⇒ 先对齐")
        # ── 校验 5：档号属于清单现算的 GRASPO 集合 ─────────────────────────
        manifest_path = Path(args.manifest)
        tiers = graspo_tiers(manifest_path)
        if args.all and args.tiers:
            raise Refusal("--all 与 --tiers 只能给一个")
        if not args.all and not args.tiers:
            raise Refusal("必须给 --all 或 --tiers M0nn[,…]（本文件不猜要签哪些档）")
        if args.all:
            selected = sorted(tiers)
        else:
            selected = [x.strip() for x in args.tiers.split(",") if x.strip()]
            bad = [t for t in selected if t not in tiers]
            if bad:
                raise Refusal(f"档号 {bad} 不属于清单现算的 GRASPO 档集合"
                              f"（{len(tiers)} 档：{sorted(tiers)[:5]}…）")
            if len(set(selected)) != len(selected):
                raise Refusal(f"--tiers 里有重复档号：{selected}")

        evidence_path = Path(args.evidence) if args.evidence else None
        evidence = None
        if args.apply:
            if evidence_path is None:
                raise Refusal("--apply 必须给 --evidence（⛔ 不预填 true、不生成结论）")
        if evidence_path is not None:
            evidence = load_evidence(evidence_path, tuple(i for i, _ in ids))
        if args.overwrite_log is not None:
            log_path = Path(args.overwrite_log)
            if ".local" not in log_path.resolve().parts:
                raise Refusal(f"--overwrite-log 必须在 .local/ 下（产物不入 git）：{log_path}")

        # ── 逐档现算（校验 6/7）与组装 ────────────────────────────────────
        manifest_rel = _rel(manifest_path)
        html_rel = _rel(html_path)
        caveats = []
        if evidence_path is not None:
            try:
                cdoc = json.loads(evidence_path.read_text(encoding="utf-8"))
                caveats.extend(str(x) for x in (cdoc.get("caveats") or []) if str(x).strip())
            except (OSError, json.JSONDecodeError):
                caveats = []
        out_dir = Path(args.out_dir)
        docs: dict[str, dict] = {}
        for tier in selected:
            tier_doc = tiers[tier]
            check = tier_check(tier, tier_doc, manifest_rel=manifest_rel)
            if evidence is None:
                docs[tier] = {"_plan_only": check}
                continue
            doc = build_doc(tier, tier_doc, doc_items=doc_items, ids=ids, evidence=evidence,
                            check=check, manifest_rel=manifest_rel, html_rel=html_rel,
                            caveats=caveats)
            docs[tier] = doc

        # ── 计划打印 ─────────────────────────────────────────────────────
        print(f"[sign] 算法说明 = {html_rel}（第 9 节 {len(doc_items)} 条，"
              f"与 record_checks.GRASPO_INVARIANTS 逐条对齐）")
        print(f"[sign] 清单 = {manifest_rel} ⇒ GRASPO 档 {len(tiers)} 个；本次 {len(selected)} 个")
        print(f"[sign] 落点 = {_rel(out_dir)}/{SIG_PREFIX}<tier>.json"
              + (f"；另写 <tier>.json 到 {args.algo_checks_dir}" if args.algo_checks_dir else ""))
        if evidence is not None:
            print("[sign] 证据：" + "、".join(
                f"{iid}={len(ev['evidence'])} 项/{ev['checker']}@{ev['checked_at']}"
                for iid, ev in evidence.items()))
        else:
            print("[sign] 证据：<未给 --evidence ⇒ 只打印计划，不会落盘>")

        if not args.apply:
            for tier in selected:
                doc = docs[tier]
                check = doc.get("tier_check") or doc["_plan_only"]
                print(f"  - {tier}: 计划可签（配置 {check['config']}、"
                      f"{check['group_slicing']['why']}）")
            print("[sign] 干跑结束：未写任何文件（加 --apply 才落盘）。")
            return 0

        applied: list[str] = []
        unchanged: list[str] = []
        overwrite_log = Path(args.overwrite_log) if args.overwrite_log else None
        for tier in selected:
            text = json.dumps(docs[tier], ensure_ascii=False, indent=1) + "\n"
            emit(out_dir / f"{SIG_PREFIX}{tier}.json", text, tier=tier,
                 overwrite_log=overwrite_log,
                 ledger_note=f"checker={docs[tier]['checker']} checked_at={docs[tier]['checked_at']}",
                 applied=applied, unchanged=unchanged)
            if args.algo_checks_dir:
                emit(Path(args.algo_checks_dir) / f"{tier}.json", text, tier=tier,
                     overwrite_log=overwrite_log,
                     ledger_note=f"consumer-copy checker={docs[tier]['checker']}",
                     applied=applied, unchanged=unchanged)
        print(f"[sign] 写入 {len(applied)} 个文件，{len(unchanged)} 个内容未变（未重写）。")
        for p in applied:
            print(f"  + {p}")
        return 0
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        print("REFUSE: 本次零写入（签署件未被创建/改写）。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
