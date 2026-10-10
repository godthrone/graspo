#!/usr/bin/env python3
"""合成长序列产物的**结构性守卫校验器** —— 证明"两个变体绝不可混填同一列"（纯标准库）。

**为什么需要它**（指挥官 2026-09-19 裁定 3）

"两个变体不可混填「最大可行上下文」列"如果只靠纪律，迟早会被一次复制粘贴破坏。
本脚本把这条口径变成**可机器核验的断言**：任何"声称填列但不是 v1-image-copy 来源"的
产物、任何"v1 产物却没留实测回填字段"的产物，都会被判为**失败**。

**核验项（逐条对应一条口径）**

| # | 断言 | 依据 |
|---|---|---|
| G1 | 每个 provenance 的 `fills_context_column` 必须等于"其目录/变体"推出的期望值
| 裁定 3；§2 防呆 |
| G2 | `fills_context_column=True` 的产物，其 `variant` **必须**是 `v1-image-copy`
| 裁定 3（唯一来源） |
| G3 | `fills_context_column=False` 的产物**不得**被写进任何"填列"清单
（本脚本据此报出可填列清单） | 口径 2 |
| G4 | 每档必须留 **实测回填字段**（`tokens_measured` 键存在；可为 `null` = 待上机）
| 裁定 1 硬要求 |
| G5 | `tokens_measured` 一旦非空，其必须带 `measured: true` 与非空 `source`
（挡"算式冒充实测"） | 本包判据纪律 |
| **G6** | **单跑档的 `a4_status` 不得为 `pass`**；显示名必须一致
（台账照抄 `未适用（单跑）`） | 指挥官 2026-09-19 裁定 1 |

**用法**

    python3 verify_synth_artifacts.py --root <产物根，可多次> [--require-measured]

``--require-measured``：在"上机实测已回填"时用来收紧门槛（缺实测即失败）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_synth_seq import (  # noqa: E402  - 单一真相源，不复制第二套口径
    A4_STATUS_DISPLAY,
    FILLS_COLUMN_VARIANT,
    VARIANTS,
    A4StatusInvariantError,
    FillsColumnInvariantError,
    assert_a4_status_invariant,
    assert_fills_column_invariant,
)


def _iter_provenance(root: Path):
    """找出产物根下的所有 provenance JSON（`*.provenance.json`）。"""
    return sorted(root.rglob("*.provenance.json"))


def _expected_variant(root: Path, provenance_path: Path) -> str | None:
    """从路径推出期望变体：产物根形如 `<...>/<tier>/<variant>/subsets/...`。"""
    for part in provenance_path.relative_to(root).parts:
        if part in VARIANTS:
            return part
    return None


def verify_root(root: Path, *, require_measured: bool) -> tuple[list[str], list[str]]:
    """核验一个产物根；返回 ``(failures, fillable)``。"""
    failures: list[str] = []
    fillable: list[str] = []
    files = _iter_provenance(root)
    if not files:
        failures.append(f"{root}: 没有任何 *.provenance.json（产物缺失或未生成）")
        return failures, fillable

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{path}: 不是合法 JSON：{exc}")
            continue
        if not isinstance(doc, dict):
            failures.append(f"{path}: provenance 不是 JSON 对象")
            continue

        variant = doc.get("variant")
        fills = doc.get("fills_context_column")
        tag = doc.get("tier_tag")

        # G1：路径推出的变体必须与 provenance 自称的一致（挡"复制到别的目录就当另一口径"）。
        expected_from_path = _expected_variant(root, path)
        if expected_from_path is not None and expected_from_path != variant:
            failures.append(
                f"G1 {path}: 目录变体 {expected_from_path!r} ≠ provenance 变体 {variant!r}"
            )

        # G2：结构性守卫本体（非 v1 声称填列 ⇒ 硬失败）。
        if not isinstance(variant, str) or not isinstance(fills, bool):
            failures.append(
                f"G2 {path}: variant/fills_context_column 类型非法（{variant!r}/{fills!r}）"
            )
            continue
        try:
            assert_fills_column_invariant(variant, fills)
        except FillsColumnInvariantError as exc:
            failures.append(f"G2 {path}: {exc}")

        # G3：只有填列口径进"可填列清单"。
        if fills:
            if variant != FILLS_COLUMN_VARIANT:
                failures.append(
                    f"G3 {path}: 填列清单里出现非 {FILLS_COLUMN_VARIANT} 的变体 {variant!r}"
                )
            else:
                fillable.append(f"{root}::{tag}={doc.get('target_tokens')}")

        # G4：必须留实测回填字段（裁定 1 的硬要求："不得让实测读数永久缺失"）。
        if "tokens_measured" not in doc:
            failures.append(
                f"G4 {path}: 缺实测回填字段 tokens_measured（上机量出的 token 数必须能回填进来）"
            )
        measured = doc.get("tokens_measured")
        if measured is not None:
            # G5：实测回填必须自证是实测（挡算式冒充）。
            if not isinstance(measured, dict) or measured.get("measured") is not True:
                failures.append(
                    f"G5 {path}: tokens_measured 未标 measured=true（疑似算式冒充实测）"
                )
            elif not measured.get("source"):
                failures.append(f"G5 {path}: tokens_measured 缺 source（无法追溯量法）")
        elif require_measured:
            failures.append(
                f"G4 {path}: --require-measured 已开，但 tokens_measured 为空（待上机）"
            )

        # G6（裁定 1 硬要求）：**单跑档的 A4 不得为 pass**。
        # 这条与"变体不可混填"同级——都是"把没跑的记成跑了"。守卫本体在
        # build_synth_seq.assert_a4_status_invariant（单一真相源），此处只做后验复核。
        if "a4_dual_run" not in doc:
            failures.append(f"G6 {path}: 缺 a4_dual_run 字段（无法判定 A4 是否适用）")
        else:
            try:
                assert_a4_status_invariant(bool(doc["a4_dual_run"]), doc.get("a4_status"))
            except A4StatusInvariantError as exc:
                failures.append(f"G6 {path}: {exc}")
            # G6b：显示名也要对得上（台账要照抄 `未适用（单跑）`，不能在抄写时变成"通过"）。
            expected_display = (
                A4_STATUS_DISPLAY.get(doc.get("a4_status")) if doc.get("a4_status") else "待跑"
            )
            if doc.get("a4_status_display") != expected_display:
                failures.append(
                    f"G6 {path}: a4_status_display={doc.get('a4_status_display')!r} "
                    f"≠ 期望 {expected_display!r}（台账显示名必须与状态一致）"
                )
    return failures, fillable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成长序列产物结构性守卫校验")
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        type=Path,
        help="产物根（可多次；如 .../T013/v1-image-copy 与 .../v2-text-only）",
    )
    parser.add_argument(
        "--require-measured",
        action="store_true",
        help="要求每档都已回填实测 token 数（上机后收紧门槛用）",
    )
    args = parser.parse_args(argv)

    all_failures: list[str] = []
    all_fillable: list[str] = []
    for root in args.root:
        if not root.is_dir():
            all_failures.append(f"{root}: 不是目录")
            continue
        failures, fillable = verify_root(root, require_measured=args.require_measured)
        all_failures.extend(failures)
        all_fillable.extend(fillable)
        print(f"[root] {root}: failures={len(failures)} fillable={len(fillable)}")

    print("\n=== 可填「最大可行上下文」列的产物（只允许 v1-image-copy 来源）===")
    if all_fillable:
        for item in all_fillable:
            print(f"  OK  {item}")
    else:
        print("  （无——尚未生成任何 v1 产物）")

    if all_failures:
        print("\n=== FAILURES ===")
        for item in all_failures:
            print(f"  FAIL {item}")
        return 1
    print("\nGUARD OK: 全部产物通过 G1–G6 结构性守卫")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
