#!/usr/bin/env python3
"""在远端机（或任意实验机）上**逐档具名**生成 ``subsets-v2/T0NN.jsonl`` 并逐档校验。

为什么是独立脚本：子集条数在 v2 里是**逐档**的（``48 × 卡数``，只有 9 档），
而 runner（``run_matrix54.sh``）只会在真跑时把 ``head -n $SUBSET <单一源>`` 写进
``<RUN_ROOT>/<T>/subsets/<T>.jsonl``。本脚本产出的是**冻结的、可审计的预生成件**
（``subsets-v2/``），两者是"冻结件"与"逐 run 副本"的关系，不是同一个文件。

★ 纪律（对齐宪法 §2.4 / §6.1）：
  · **只新增**：只创建 ``<root>/subsets-v2/`` 与其中的 ``T0NN.jsonl`` / 清单文件；
    不删除、不改写任何既有文件；不写根盘临时目录。
  · **逐档具名**：51 个 yaml 档号在源码里**显式列出**（``TIER_IDS``），
    不用通配符循环"静默跳过"——少一个档号会被下面的集合断言当场抓住。
  · **逐档校验**：每个产物现场核对 (a) 行数 == 期望条数，(b) 与源文件 ``head -n N``
    逐字节相同（sha256 比对），(c) JSON 逐行可解析。任一不满足即整体失败（非零退出）。

数据源（逐档，与 v2 冻结配置同源）：
  · 9 档 ``GRASPO × native``（T028/T029/T030/T034/T035/T036/T040/T041/T042）
    ⇒ **ELAM V5 正式集** ``<data_root>/data/train.jsonl`` 前 ``P × cards`` 条：
      默认 ``P=48``（48/96/192）；★ T040/T041/T042 按 2026-10-02 批准的 E-49 例外 ``P=72``
      （⇒ 72/144/288，失效条件见下方 E-50）。
  · 其余 45 档（含 42 个可跑档 + 3 个 not_applicable 档）⇒ **mini 冒烟集**
    ``<data_root>/mini-dataset/mini-short-mm-train.jsonl`` 前 N 条：
      - CPT/SFT 27 档 = 100（历来如此）；
      - ★ **2026-10-16 主席裁定「18 个 20 行档放大到 100 行」** ⇒ 原 18 个 20 行档的**声明口径 = 100**；
        其中 **15 档已落地**（`T031/T032/T033`、`T037/T038/T039`、`T043/T044/T045`、`T046…T051`），
        另 **3 档豁免未落地**（`T052–T054`：`not_applicable` ⇒ 无生成件、⛔ 不伪造）——具名登记见
        :data:`DECLARED_NOT_LANDED`。
      - ⚠ 本表 :data:`MINI_TIERS` 是**在盘落地真值**（15 档 = 100、`T052–T054` = 20）；
        **声明口径真值**是 :data:`MINI_SUBSET_TIERS_DECLARED`（18 档 = 100）。两者**不得混用**。

用法::

    python3 tests/e2e/make_subsets.py \
        --data-root  <ELAM V5 宿主根，如 ${GRASPO_ELAM_DATA_ROOT}> \
        --out-dir    <产物根，如 <实验根>/subsets-v2> \
        [--dry-run]

★ M 档（v3「68 档」世代，2026-10-22）另走一条通路，**不需要 `--data-root`**（源在仓里）::

    python3 tests/e2e/make_subsets.py --tier-set m68 --out-dir subsets-v2 [--dry-run]

``--data-root`` / ``--out-dir`` 都是**运行时环境信息**，不写死在脚本里（宪法 §15.1/§16）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# ── v2 冻结规则（与 samples/configs/matrix54-v2/ 及 verify_matrix_html.py 同源）──
#: 9 档「GRASPO × native」：每卡 prompt 数 48 ⇒ 子集条数 = 48 × 卡数。
#: ★ 例外（2026-10-02 批准，已声明数据变更，登记册 E-49）：T040/T041/T042 每卡 prompt 数 = 72
#:   （原 48）⇒ 子集 72/144/288。理由：这三档 ① 因 rollout 有效样本被 perfect_skip/invalid 吃掉
#:   而达不到计划步数；扩池 + 允许重采以补齐有效样本。★ 失效条件：日后子集再放大必须重锚（E-50）。
FORMAL_TIERS: dict[str, int] = {
    "T028": 48,
    "T029": 96,
    "T030": 192,
    "T034": 48,
    "T035": 96,
    "T036": 192,
    "T040": 72,
    "T041": 144,
    "T042": 288,
}

#: ★★ **原 18 个「20 行档」的声明口径**（2026-10-16 主席裁定：「18 个 20 行档放大到 100 行」）。
#: 这是**声明真值**（矩阵/权威清单据此声明 `subset_size`），与 :data:`MINI_TIERS`（在盘落地真值）
#: **是两张不同的表**：声明 ⊇ 落地，差额 = :data:`DECLARED_NOT_LANDED`（具名豁免）。
#: ★ 纪律：新增/追加时**必须**同时更新本表与 :data:`MINI_TIERS`（或登记进豁免表），
#:   否则 `generate_matrix_manifest_source.py::check()` 的冻结子集交叉核对会当场报 MISMATCH（fail-closed）。
MINI_SUBSET_TIERS_DECLARED: dict[str, int] = {
    "T031": 100, "T032": 100, "T033": 100,
    "T037": 100, "T038": 100, "T039": 100,
    "T043": 100, "T044": 100, "T045": 100,
    "T046": 100, "T047": 100, "T048": 100,
    "T049": 100, "T050": 100, "T051": 100,
    "T052": 100, "T053": 100, "T054": 100,
}

#: ★ 声明口径为 100、但**数据未落地**的具名豁免（⛔ 不是"偷偷跳过"：理由逐档写下、逐次打印）。
#: T052–T054 = `not_applicable`（师生权重常驻约 100 GiB、依裁定不实施）⇒ **无生成件**，
#: ⛔ 不为它们伪造 100 行冻结件；"声明未落地"这一事实本身在账本里显式登记。
DECLARED_NOT_LANDED: dict[str, str] = {
    "T052": "not_applicable（主席裁定不实施；师生权重常驻约 100 GiB）⇒ 无生成件，⛔ 不伪造冻结子集",
    "T053": "not_applicable（同上）⇒ 无生成件，⛔ 不伪造冻结子集",
    "T054": "not_applicable（同上）⇒ 无生成件，⛔ 不伪造冻结子集",
}

#: 其余 45 档的 v2 子集条数 = **在盘落地真值**（逐档显式）。
#: ★ 27 档 CPT/SFT = 100（历来如此）；★ 15 档（原 20 行档中已落地者）= 100（2026-10-16 扩面）；
#: ★ T052–T054 = 20（`not_applicable` 豁免，见 :data:`DECLARED_NOT_LANDED`——仍产出 20 行件
#:   只是为了清单完整、逐档可核，⛔ 不代表声明口径）。
MINI_TIERS: dict[str, int] = {
    # CPT × 9（T004–T006 为全量档，条数口径不变）
    "T001": 100, "T002": 100, "T003": 100, "T004": 100, "T005": 100,
    "T006": 100, "T007": 100, "T008": 100, "T009": 100,
    # SFT × 18
    "T010": 100, "T011": 100, "T012": 100, "T013": 100, "T014": 100,
    "T015": 100, "T016": 100, "T017": 100, "T018": 100, "T019": 100,
    "T020": 100, "T021": 100, "T022": 100, "T023": 100, "T024": 100,
    "T025": 100, "T026": 100, "T027": 100,
    # GRASPO × ms-swift × 9（★ 2026-10-16：原 20 行档 ⇒ 100 行，已全部落地）
    "T031": 100, "T032": 100, "T033": 100,
    "T037": 100, "T038": 100, "T039": 100,
    "T043": 100, "T044": 100, "T045": 100,
    # OPD × 9（★ T046–T051 = 2026-10-16 扩面落地；T052–T054 = not_applicable 豁免，仍 20）
    "T046": 100, "T047": 100, "T048": 100, "T049": 100, "T050": 100, "T051": 100,
    "T052": 20, "T053": 20, "T054": 20,
}

#: 51 个可跑档（有 yaml 的档）。T052–T054 为 not_applicable，只产出说明文件、无 yaml。
RUNNABLE_TIER_IDS: tuple[str, ...] = tuple(
    f"T{i:03d}" for i in range(1, 52)
)
NOT_APPLICABLE_TIER_IDS: tuple[str, ...] = ("T052", "T053", "T054")

#: 必须与上面两个表**恰好**构成 1..54（少一个档号即 fail-closed）。
assert set(FORMAL_TIERS) | set(MINI_TIERS) == set(RUNNABLE_TIER_IDS) | set(
    NOT_APPLICABLE_TIER_IDS
), "逐档表覆盖不全"
assert set(FORMAL_TIERS) & set(MINI_TIERS) == set(), "逐档表有重叠"
assert sorted(set(FORMAL_TIERS) | set(MINI_TIERS)) == [f"T{i:03d}" for i in range(1, 55)]

#: ★★ 声明口径表 vs 落地真值表的**自洽断言**（load 即校验 ⇒ 任何一处漏改立即 fail-closed）。
assert len(MINI_SUBSET_TIERS_DECLARED) == 18, (
    f"声明口径表应恰 18 档（2026-10-16 裁定的 18 个原 20 行档），实为 {len(MINI_SUBSET_TIERS_DECLARED)}"
)
assert set(MINI_SUBSET_TIERS_DECLARED) - set(MINI_TIERS) == set(), "声明口径表里有非 mini 档"
assert {t: v for t, v in MINI_SUBSET_TIERS_DECLARED.items() if v != 100} == {}, (
    "声明口径表的值应全部为 100（18 档放大到 100 行）"
)
assert set(DECLARED_NOT_LANDED) == set(NOT_APPLICABLE_TIER_IDS), (
    "声明未落地的豁免集合必须与 NOT_APPLICABLE_TIER_IDS 逐档一致"
)
assert all(MINI_TIERS[t] < MINI_SUBSET_TIERS_DECLARED[t] for t in DECLARED_NOT_LANDED), (
    "豁免档的在盘件条数必须小于声明口径（否则它就不是'未落地'，应删出豁免表）"
)
assert all(MINI_TIERS[t] == MINI_SUBSET_TIERS_DECLARED[t]
           for t in set(MINI_SUBSET_TIERS_DECLARED) - set(DECLARED_NOT_LANDED)), (
    "非豁免的声明口径档必须已按声明落地（在盘真值 == 声明口径）"
)

FORMAL_TRAIN_REL = "data/train.jsonl"
MINI_TRAIN_REL = "mini-dataset/mini-short-mm-train.jsonl"

# ══════════════════════════════════════════════════════════════════════════════════════
# ★★ M 档（能力矩阵 v3「68 档」世代，2026-10-22 第一轮基线）
#
#   口径（数据件 `meta.criteria` 改动三 + 人读件第 2/3 章）：
#     · 68 档（M001–M068）**全部**训练样本数 = 100 条；
#     · 68 档**共用同一份**文件（内容层面逐字节相同），sha256 = :data:`M68_SOURCE_SHA256`；
#     · 该 sha 同时是数据件判据锚点（100 行 / `5ad2d7cc…`）与
#       `subsets-v2.manifest.json::source_files.m68_frozen.sha256`。
#   ⇒ 因此**不需要重新造数据**：M 档产物 = 仓内 100 条冻结实体的逐档副本
#     （`subsets-v2/balanced-100-seed20261022-sha5ad2d7cc588b.jsonl` 就是该实体 —— 2026-10-22 语料切换后取代
#      旧 `subsets-v2/T001.jsonl`（= mini 100 条）；旧件保留作历史，不再被 M 档引用）。
#     ⛔ 不依赖远端机 / 不依赖宿主数据根即可在本仓生成与复核。
#
#   ★ T 档行为**逐字保持**：`--tier-set v2`（默认）走原 54 档通路，代码路径未改。
# ══════════════════════════════════════════════════════════════════════════════════════
#: 68 个 M 档号（**逐档具名**：少一个档号即被下面的集合断言当场抓住，⛔ 不用通配符静默跳过）。
M68_TIER_IDS: tuple[str, ...] = tuple(f"M{i:03d}" for i in range(1, 69))

#: 每档训练样本数（68 档一律 100，逐档显式）。
M68_TIERS: dict[str, int] = {tid: 100 for tid in M68_TIER_IDS}

#: 100 条共用语料的**冻结 sha256**（= 均衡子集 sha = 数据件判据锚点）。
M68_SOURCE_SHA256 = "5ad2d7cc588bab7b1b72750b3070d3c3f2de1690e105d46e541ca333566d3a6d"

#: 仓内实体（唯一；2026-10-22 语料切换后）：`subsets-v2/balanced-100-seed20261022-sha5ad2d7cc588b.jsonl`。
#: 历史：切换前为 `subsets-v2/T001.jsonl`（在仓里）→ mini 源（宿主数据根下）。
M68_REPO_SOURCE_CANDIDATES: tuple[str, ...] = ("subsets-v2/balanced-100-seed20261022-sha5ad2d7cc588b.jsonl",)

assert len(M68_TIERS) == 68, f"M 档表应恰 68 档，实为 {len(M68_TIERS)}"
assert sorted(M68_TIERS) == [f"M{i:03d}" for i in range(1, 69)], "M 档号必须恰为 M001–M068"
assert set(M68_TIERS.values()) == {100}, "M 档口径：68 档训练样本数一律 100"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def head_lines(path: Path, count: int) -> bytes:
    """取 ``path`` 的前 ``count`` 行（保留行尾换行；与 ``head -n`` 语义一致）。"""
    with path.open("rb") as handle:
        out: list[bytes] = []
        for i, line in enumerate(handle):
            if i >= count:
                break
            out.append(line)
    return b"".join(out)


def build_manifest(out_dir: Path, data_root: Path, records: list[dict], source_hashes: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "tests/e2e/make_subsets.py",
        "config_version": "v2",
        "purpose": (
            "能力矩阵第一轮跑批的**逐档数据子集冻结件**（9 档取 ELAM V5 正式集前 N 条，"
            "其余 45 档取 mini 冒烟集前 N 条）。真跑时 runner 会另把 `head -n $SUBSET <源>` "
            "写进 <RUN_ROOT>/<T>/subsets/<T>.jsonl —— 本目录是**预生成/可审计**的那一份。"
        ),
        "read_contract": (
            "★ 逐档读来源（防呆）：档号矩阵的 tier 定义在追踪文件里由**档号字面量**选择来源，"
            "不靠全局 export 切换。"
        ),
        "source_files": {
            "formal": {"relative_path": FORMAL_TRAIN_REL, "sha256": source_hashes["formal"]},
            "mini": {"relative_path": MINI_TRAIN_REL, "sha256": source_hashes["mini"]},
        },
        "rules": {
            "formal_tiers": {k: FORMAL_TIERS[k] for k in sorted(FORMAL_TIERS)},
            "formal_rule": "subset_size = 48 × cards（每卡 prompt 数 48）",
            "mini_tiers": {k: MINI_TIERS[k] for k in sorted(MINI_TIERS)},
            "mini_rule": "沿用 v1 算法下限（CPT/SFT=100、GRASPO/OPD=20）",
        },
        "resolved_paths": {
            "data_root": str(data_root),
            "out_dir": str(out_dir),
        },
        "not_applicable_tiers": list(NOT_APPLICABLE_TIER_IDS),
        "tiers": records,
    }


def resolve_m68_source(args) -> tuple[Path | None, str | None]:
    """m68 通路：定位 100 条共用语料源。返回 ``(path, why_not)``（二者互斥）。"""
    if args.m68_source is not None:
        p = Path(args.m68_source)
        return (p, None) if p.is_file() else (None, f"--m68-source 不存在：{p}")
    repo = Path(args.repo_root)
    tried = []
    for rel in M68_REPO_SOURCE_CANDIDATES:
        p = repo / rel
        if p.is_file():
            return p, None
        tried.append(rel)
    return None, ("仓内 100 条共用语料实体未找到（试过："
                  + "、".join(tried) + "）；可用 --m68-source 显式指定")


def extend_manifest_with_m68(manifest: dict, records: list[dict],
                             source_path: Path, source_sha: str) -> dict:
    """把 68 个 M 档**追加**进既有清单：T 档历史条目与全部既有键**逐字保留不动**。"""
    out = dict(manifest)                       # ★ 浅拷贝：既有键原样带过去（不重算、不重排）
    others = [t for t in (manifest.get("tiers") or [])
              if str(t.get("tier_id") or "") not in M68_TIERS]
    dropped = len(manifest.get("tiers") or []) - len(others)
    if dropped:                                # 幂等重跑：旧 M 条目先剔除再加，避免重复登记
        print(f"[manifest] 已存在 {dropped} 条 M 档旧登记 ⇒ 以本次在盘事实替换（T 档不受影响）")
    out["tiers"] = others + records
    sf = dict(manifest.get("source_files") or {})
    sf["m68_frozen"] = {"relative_path": str(source_path), "sha256": source_sha}
    out["source_files"] = sf
    rules = dict(manifest.get("rules") or {})
    rules["m68_tiers"] = {k: M68_TIERS[k] for k in sorted(M68_TIERS)}
    rules["m68_rule"] = ("68 档训练样本数一律 100，且 68 档共用同一份 100 行冻结语料"
                         "（内容逐字节相同，sha256 = M68_SOURCE_SHA256）")
    out["rules"] = rules
    out["m68"] = {
        "generated_by": "tests/e2e/make_subsets.py --tier-set m68",
        "subset_size": 100,
        "tier_count": len(M68_TIERS),
        "source_sha256": source_sha,
        "path_convention": ("`tiers[].path` 为**仓根相对路径**（M 档条目），"
                            "可与 `docs/capability-matrix.data.json::rows[].train_path` 逐字比对；"
                            "T 档历史条目仍用 `${GRASPO_PREGEN_SUBSET_DIR}/…` 符号路径，未改动"),
        "read_contract": ("逐档登记 {subset_size, lines, sha256, path}；"
                          "校验入口 `tools/matrix/check_68_readiness.py`（③）逐档复核在盘 sha256 与登记值"),
    }
    return out


def run_m68(args) -> int:
    """m68 通路：产出 ``M0nn.jsonl`` ×68（只新增/等价覆盖），并扩登记清单。"""
    out_dir: Path = args.out_dir
    src_path, why_not = resolve_m68_source(args)
    if src_path is None:
        print(f"FATAL: {why_not}", file=sys.stderr)
        return 2
    src_bytes = src_path.read_bytes()
    src_sha = sha256_bytes(src_bytes)
    src_lines = src_bytes.count(b"\n")
    print(f"[source] m68 ={src_path} 行数={src_lines} sha256={src_sha}")
    # ★ fail-closed：源必须**就是**那份 100 条共用语料（sha 锚定），否则它不能代表 68 档的同一份数据
    if src_sha != M68_SOURCE_SHA256:
        print(f"FATAL: 源 sha256={src_sha} != 冻结口径 {M68_SOURCE_SHA256}"
              f"（该档子集必须与数据件判据锚点是同一份文件）", file=sys.stderr)
        return 2
    if src_lines < 100:
        print(f"FATAL: 源只有 {src_lines} 行（需 ≥100）", file=sys.stderr)
        return 2

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    failures: list[str] = []
    for tier_id in M68_TIER_IDS:                     # ★ 逐档具名（⛔ 不靠通配符）
        size = M68_TIERS[tier_id]
        expected = head_lines(src_path, size)
        expected_lines = expected.count(b"\n")
        if expected_lines != size:
            failures.append(f"{tier_id}: 源行数不足：{expected_lines} < {size}")
            continue
        # (c) 逐行 JSON 可解析 —— 不静默放过坏行
        bad = []
        for i, line in enumerate(expected.splitlines(), start=1):
            if not line.strip():
                bad.append(i)
                continue
            try:
                json.loads(line)
            except Exception as exc:  # noqa: BLE001 - 逐档如实报告
                bad.append((i, str(exc)[:80]))
        if bad:
            failures.append(f"{tier_id}: 源前 {size} 行里有不可解析 JSON：{bad[:3]}")
            continue

        digest = sha256_bytes(expected)
        out_path = out_dir / f"{tier_id}.jsonl"
        status = "dry-run"
        if not args.dry_run:
            if out_path.exists():
                existing = out_path.read_bytes()
                if existing != expected:
                    # ★ 只新增/等价覆盖（§2.4）：内容不同即拒绝，不静默改写既有产物
                    failures.append(f"{tier_id}: {out_path} 已存在且内容与预期不同 —— 拒绝覆盖（§2.4）")
                    continue
                status = "unchanged"
            else:
                status = "written"
            out_path.write_bytes(expected)
            # (a)(b) 落盘后**再核一次**（磁盘事实，不是内存事实）
            on_disk = out_path.read_bytes()
            if sha256_bytes(on_disk) != digest or on_disk.count(b"\n") != size:
                failures.append(f"{tier_id}: 落盘校验失败（{out_path}）")
                continue
            if status == "written":
                status = "written"

        records.append({
            "tier_id": tier_id,
            "tier_set": "m68",
            "source": "m68_frozen",
            "source_jsonl": str(src_path),
            "subset_size": size,
            "lines": expected_lines,
            "sha256": digest,
            "path": f"subsets-v2/{tier_id}.jsonl",     # ★ 仓根相对路径（与数据件 train_path 同字面）
            "status": status,
            "not_applicable": False,
        })
        print(f"[{status:9s}] {tier_id}  lines={expected_lines:4d}  sha256={digest[:16]}…  -> {out_path}")

    if failures:
        print("\nFATAL: 以下 M 档未通过校验（未产出清单）:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 3
    if len(records) != 68:
        print(f"FATAL: M 档产物记录数 {len(records)} != 68", file=sys.stderr)
        return 3

    manifest_path = out_dir / "subsets-v2.manifest.json"
    if not manifest_path.is_file():
        print(f"FATAL: 既有清单不存在：{manifest_path}"
              f"（T 档历史条目必须保留 ⇒ ⛔ 不从零重建清单）", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_manifest = extend_manifest_with_m68(manifest, records, src_path, src_sha)
    if not args.dry_run:
        manifest_path.write_text(
            json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nOK: {len(records)} 个 M 档子集已逐档校验并落盘")
    print(f"清单: {manifest_path}（tiers 条目 {len(new_manifest['tiers'])} 条 = "
          f"T 档 {len(new_manifest['tiers']) - len(records)} + M 档 {len(records)}）")
    print(f"分组摘要: {json.dumps({'m68:100': len(records)}, ensure_ascii=False)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 v2 逐档数据子集（只新增，不删改）")
    parser.add_argument("--data-root", default=None, type=Path,
                        help="ELAM V5 宿主数据根（v2 通路必需；m68 通路不需要——源在仓内）")
    parser.add_argument("--out-dir", required=True, type=Path, help="产物根（其下写 T0NN.jsonl）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将写什么，不落盘")
    parser.add_argument(
        "--overwrite-tiers", default="",
        help="★ 具名覆盖授权（逗号分隔档号）。仅这些档允许覆盖内容不同的既有子集；"
        "未点名档一律维持「不静默改写」防呆。前像 sha256 逐件记入 同名 .overwrite-log.jsonl",
    )
    parser.add_argument(
        "--tier-set", default="v2", choices=("v2", "m68"),
        help="v2（默认）= 原 54 档 T0nn 通路，行为逐字不变；"
             "m68 = 68 档 M0nn 通路（100 条共用语料，源取仓内冻结实体，⛔ 不依赖宿主数据根）",
    )
    parser.add_argument("--repo-root", default=".", type=Path,
                        help="仓根（m68 通路用它在仓内定位 100 条共用语料的冻结实体）")
    parser.add_argument("--m68-source", default=None, type=Path,
                        help="m68 通路：显式指定 100 条共用语料源（缺省按 M68_REPO_SOURCE_CANDIDATES 现找）")
    args = parser.parse_args(argv)

    if args.tier_set == "m68":
        return run_m68(args)

    if args.data_root is None:
        print("FATAL: v2 通路必须给 --data-root（m68 通路不需要）", file=sys.stderr)
        return 2
    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    formal_path = data_root / FORMAL_TRAIN_REL
    mini_path = data_root / MINI_TRAIN_REL

    problems: list[str] = []
    overwrite_tiers = {t.strip() for t in (getattr(args, "overwrite_tiers", "") or "").split(",") if t.strip()}
    overwrite_log: list[dict] = []
    for label, path in (("formal", formal_path), ("mini", mini_path)):
        if not path.is_file():
            problems.append(f"{label} 源不存在：{path}")
    if not (data_root / "images").is_dir():
        problems.append(f"图像根不存在：{data_root / 'images'}（相对媒体路径的锚点）")
    if problems:
        for p in problems:
            print(f"FATAL: {p}", file=sys.stderr)
        return 2

    formal_bytes = formal_path.read_bytes()
    mini_bytes = mini_path.read_bytes()
    source_hashes = {
        "formal": sha256_bytes(formal_bytes),
        "mini": sha256_bytes(mini_bytes),
    }
    formal_total = formal_bytes.count(b"\n")
    mini_total = mini_bytes.count(b"\n")
    print(f"[source] formal={formal_path} 行数={formal_total} sha256={source_hashes['formal']}")
    print(f"[source] mini  ={mini_path} 行数={mini_total} sha256={source_hashes['mini']}")

    if mini_total < 100:
        print(f"FATAL: mini 源只有 {mini_total} 行（需 ≥100）", file=sys.stderr)
        return 2

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    failures: list[str] = []
    # ★ 逐档具名：先 formal 的 9 档（显式档号），再 mini 的 45 档（显式档号）。
    plan: list[tuple[str, str, int]] = [
        (tid, "formal", size) for tid, size in sorted(FORMAL_TIERS.items())
    ] + [(tid, "mini", size) for tid, size in sorted(MINI_TIERS.items())]
    assert len(plan) == 54, f"计划档数 {len(plan)} != 54"

    for tier_id, source, size in plan:
        src_path = formal_path if source == "formal" else mini_path
        expected = head_lines(src_path, size)
        expected_lines = expected.count(b"\n")
        if expected_lines != size:
            failures.append(f"{tier_id}: 源 {src_path} 行数不足：{expected_lines} < {size}")
            continue
        # 逐行 JSON 可解析（不静默放过坏行）
        bad = []
        for i, line in enumerate(expected.splitlines(), start=1):
            if not line.strip():
                bad.append(i)
                continue
            try:
                json.loads(line)
            except Exception as exc:  # noqa: BLE001 - 逐档如实报告
                bad.append((i, str(exc)[:80]))
        if bad:
            failures.append(f"{tier_id}: 源前 {size} 行里有不可解析 JSON：{bad[:3]}")
            continue

        digest = sha256_bytes(expected)
        out_path = out_dir / f"{tier_id}.jsonl"
        status = "dry-run"
        if not args.dry_run:
            if out_path.exists():
                existing = out_path.read_bytes()
                if existing != expected:
                    # 只新增/等价覆盖：内容不同即拒绝（防呆：不静默改写既有产物）
                    if tier_id in overwrite_tiers:
                        # ★ 具名覆盖授权（2026-10-02 批准）：前像 sha 逐件留档
                        pre = sha256_bytes(existing)
                        overwrite_log.append({
                            "tier_id": tier_id,
                            "preimage_sha256": pre,
                            "postimage_sha256": digest,
                            "authorized_by": "--overwrite-tiers",
                        })
                        print(f"[overwrite-authorized] {tier_id} 前像={pre[:16]}… → 后像={digest[:16]}…")
                    else:
                        failures.append(
                            f"{tier_id}: {out_path} 已存在且内容与预期不同 —— 拒绝覆盖（§2.4）"
                        )
                        continue
            out_path.write_bytes(expected)
            # 落盘后**再核一次**（磁盘事实，不是内存事实）
            on_disk = out_path.read_bytes()
            if sha256_bytes(on_disk) != digest or on_disk.count(b"\n") != size:
                failures.append(f"{tier_id}: 落盘校验失败（{out_path}）")
                continue
            status = "written"

        records.append(
            {
                "tier_id": tier_id,
                "source": source,
                "source_jsonl": str(src_path),
                "source_relpath_under_data_root": (
                    FORMAL_TRAIN_REL if source == "formal" else MINI_TRAIN_REL
                ),
                "subset_size": size,
                "lines": expected_lines,
                "sha256": digest,
                "path": str(out_path),
                "status": status,
                "not_applicable": tier_id in NOT_APPLICABLE_TIER_IDS,
            }
        )
        print(
            f"[{status:7s}] {tier_id}  source={source:6s} lines={expected_lines:4d}  "
            f"sha256={digest[:16]}…  -> {out_path}"
        )

    if overwrite_log and not args.dry_run:
        (out_dir / "subsets-v2.overwrite-log.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in overwrite_log) + "\n",
            encoding="utf-8",
        )
        print(f"具名覆盖日志: {out_dir / 'subsets-v2.overwrite-log.jsonl'}（{len(overwrite_log)} 件，含前像 sha）")

    if failures:
        print("\nFATAL: 以下档位未通过校验（未产出清单）:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 3

    if len(records) != 54:
        print(f"FATAL: 产物记录数 {len(records)} != 54", file=sys.stderr)
        return 3

    manifest = build_manifest(out_dir, data_root, records, source_hashes)
    manifest_path = out_dir / "subsets-v2.manifest.json"
    if not args.dry_run:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"\nOK: {len(records)} 个档位子集已逐档校验并落盘")
    print(f"清单: {manifest_path}（sha256={sha256_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode()+b'\\n') if not args.dry_run else 'dry-run'}）")
    # 汇总：按来源/条数分组（给报告用的摘要）
    by_size: dict[str, int] = {}
    for r in records:
        by_size[f"{r['source']}:{r['subset_size']}"] = by_size.get(f"{r['source']}:{r['subset_size']}", 0) + 1
    print("分组摘要:", json.dumps(by_size, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
