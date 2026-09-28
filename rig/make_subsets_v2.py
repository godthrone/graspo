#!/usr/bin/env python3
"""在 228（或任意实验机）上**逐档具名**生成 ``subsets-v2/T0NN.jsonl`` 并逐档校验。

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
    ⇒ **ELAM V5 正式集** ``<data_root>/data/train.jsonl`` 前 ``48 × cards`` 条（48/96/192）。
  · 其余 45 档（含 42 个可跑档 + 3 个 not_applicable 档）⇒ **mini 冒烟集**
    ``<data_root>/mini-dataset/mini-short-mm-train.jsonl`` 前 100 条
    （CPT/SFT=100、GRASPO/OPD=20）。

用法::

    python3 tests/e2e/make_subsets_v2.py \
        --data-root  <ELAM V5 宿主根，如 /home/<user>/<dataset-volume>> \
        --out-dir    <产物根，如 <实验根>/subsets-v2> \
        [--dry-run]

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
FORMAL_TIERS: dict[str, int] = {
    "T028": 48,
    "T029": 96,
    "T030": 192,
    "T034": 48,
    "T035": 96,
    "T036": 192,
    "T040": 48,
    "T041": 96,
    "T042": 192,
}

#: 其余 45 档的 v2 子集条数（= v1 算法下限，逐档显式；CPT/SFT=100、GRASPO/OPD=20）。
MINI_TIERS: dict[str, int] = {
    # CPT × 9（T004–T006 为全量档，条数口径不变）
    "T001": 100, "T002": 100, "T003": 100, "T004": 100, "T005": 100,
    "T006": 100, "T007": 100, "T008": 100, "T009": 100,
    # SFT × 18
    "T010": 100, "T011": 100, "T012": 100, "T013": 100, "T014": 100,
    "T015": 100, "T016": 100, "T017": 100, "T018": 100, "T019": 100,
    "T020": 100, "T021": 100, "T022": 100, "T023": 100, "T024": 100,
    "T025": 100, "T026": 100, "T027": 100,
    # GRASPO × ms-swift × 9（v2 未改：仍取门槛下限 20）
    "T031": 20, "T032": 20, "T033": 20,
    "T037": 20, "T038": 20, "T039": 20,
    "T043": 20, "T044": 20, "T045": 20,
    # OPD × 9（T052–T054 为 not_applicable，仍生成以便清单完整、逐档可核）
    "T046": 20, "T047": 20, "T048": 20, "T049": 20, "T050": 20, "T051": 20,
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

FORMAL_TRAIN_REL = "data/train.jsonl"
MINI_TRAIN_REL = "mini-dataset/mini-short-mm-train.jsonl"


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
        "generator": "tests/e2e/make_subsets_v2.py",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 v2 逐档数据子集（只新增，不删改）")
    parser.add_argument("--data-root", required=True, type=Path, help="ELAM V5 宿主数据根")
    parser.add_argument("--out-dir", required=True, type=Path, help="产物根（其下写 T0NN.jsonl）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将写什么，不落盘")
    args = parser.parse_args(argv)

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    formal_path = data_root / FORMAL_TRAIN_REL
    mini_path = data_root / MINI_TRAIN_REL

    problems: list[str] = []
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
