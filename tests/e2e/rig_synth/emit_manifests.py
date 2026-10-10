#!/usr/bin/env python3
"""合成长序列的**runner 清单生成器** —— 单档清单 + 变体字段（纯标准库）。

**职责边界**

- **本模块**：从仓内 ``tests/e2e/matrix54_manifest.json`` 取 ``T013`` 档，逐（长度 × 变体）
  产出一份**单档清单**，供 ``t013_host.sh`` 的既有链路使用。**不改仓库 manifest**
  （与 ``task-r3-lenramp/rig/mk_manifests.py`` 同口径：只改 ``config``/``data.subset_size``/
  ``acceptance.formal_gate``/``rig_context_*`` 四个位置）。
- **不负责**：配置生成（``emit_configs.py``）、数据构造（``build_synth_seq.py``）、
  执行（``run_synth.sh``）。

**★ 变体区分（裁定 3：绝不可混填同一列）**

每份清单在 tier 上带两个**可机器核验**的字段：

- ``rig_context_variant``：``v1-image-copy`` / ``v2-text-only``（与配置目录同名）；
- ``rig_context_fills_column``：布尔，**只有 v1 为 true**；并由
  :func:`_assert_fills_column_invariant` 在写盘前强制（非 v1 置 true ⇒ 硬失败）。

台账回填脚本据此**结构上不可能**把 v2 的读数填进该列。

**★ 门槛调整（必须登记，不得悄悄改）**

合成档每档**只有 1 行 ⇒ 1 step**（合成行极长，只有第 1 步有意义）。因此
``acceptance.formal_gate.min_optimizer_steps`` 必须从基线 ``T013`` 的 **5** 降到 **1**——
否则 A2 恒判失败、台账会记成"跑过但未过"，掩盖真因。``min_train_samples`` 同步降到 1。
**这条偏离写进清单的 ``rig_gate_note`` 字段**，并须在报告与台账标注里显式声明。

**用法**

    python3 emit_manifests.py --src <仓内 matrix54_manifest.json> --out-dir <.../manifests> \\
        --config-root <分号或逗号分隔的 变体=配置根>  [--tiers 4k,8k,...]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

TIERS: dict[str, int] = {
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "96k": 98304,
    "128k": 131072,
}

VARIANTS: dict[str, bool] = {
    "v1-image-copy": True,  # 填列主口径
    "v2-text-only": False,  # ★ 不得填列
}

#: 唯一允许填「最大可行上下文」列的变体（与 build_synth_seq / verify_synth_artifacts 同源口径）。
FILLS_COLUMN_VARIANT = "v1-image-copy"

# --------------------------------------------------------------------------- #
# ★ A4 适用策略（指挥官 2026-09-19 裁定 1）——口径与 build_synth_seq 逐字同源
# --------------------------------------------------------------------------- #
#: 4K–32K 双跑；64K 起单跑。
A4_DUAL_RUN_TIERS: tuple[str, ...] = ("4k", "8k", "16k", "32k")

A4_STATUS_NOT_APPLICABLE_SINGLE_RUN = "not_applicable_single_run"
A4_STATUS_DISPLAY_SINGLE_RUN = "未适用（单跑）"

#: 双跑档适用判据 = A1–A6；单跑档 = A1/A2/A3/A5/A6（A4 记「未适用（单跑）」）。
A4_APPLICABLE_CRITERIA_DUAL_RUN = ["A1", "A2", "A3", "A4", "A5", "A6"]
A4_APPLICABLE_CRITERIA_SINGLE_RUN = ["A1", "A2", "A3", "A5", "A6"]

#: 合成档的步数：1 行 = 1 step。
SYNTH_STEPS = 1

GATE_NOTE = (
    "★ 合成档门槛调整（登记）：每档只有 1 行合成样本（1 行 = 1 step）⇒ "
    f"min_optimizer_steps 由基线 5 降为 {SYNTH_STEPS}、min_train_samples 由 10 降为 {SYNTH_STEPS}。"
    "合成行极长，只有第 1 步有意义（峰值在第 1 步即定型）。"
    "若沿用 5，A2 恒判失败并把真因掩盖成'跑过但未过'。"
)


class ManifestEmitError(Exception):
    """清单生成越界（如非 v1 声称填列）⇒ fail-closed。"""


def _assert_fills_column_invariant(variant: str, fills: bool) -> None:
    """★ 结构性守卫：只有 :data:`FILLS_COLUMN_VARIANT` 可置 ``rig_context_fills_column=True``。"""
    if variant not in VARIANTS:
        raise ManifestEmitError(f"未知变体 {variant!r}")
    if fills != (variant == FILLS_COLUMN_VARIANT):
        raise ManifestEmitError(
            f"变体 {variant!r} 的 rig_context_fills_column={fills}，"
            f"但唯一允许填列的变体是 {FILLS_COLUMN_VARIANT!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成长序列 runner 清单生成器")
    parser.add_argument("--src", required=True, type=Path, help="仓内 matrix54_manifest.json")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--config-root",
        required=True,
        help="变体=配置根 的逗号分隔表，如 "
        "v1-image-copy=.local/.../configs/v1-image-copy,"
        "v2-text-only=.local/.../configs/v2-text-only",
    )
    parser.add_argument(
        "--config-prefix",
        default=".local/hb-workspace/20260915-152747/task-synth-seq",
        help="写入清单的 config 路径前缀（相对仓库根）",
    )
    parser.add_argument("--tiers", default=",".join(TIERS))
    parser.add_argument("--tier-id", default="T013")
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.src.read_text(encoding="utf-8"))
        tiers = {tier["tier_id"]: tier for tier in manifest["tiers"]}
        if args.tier_id not in tiers:
            raise ManifestEmitError(f"源 manifest 里没有档 {args.tier_id!r}")
        base = tiers[args.tier_id]

        config_roots: dict[str, str] = {}
        for item in args.config_root.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            key, value = key.strip(), value.strip()
            if key not in VARIANTS:
                raise ManifestEmitError(f"未知变体 {key!r}")
            if not value:
                raise ManifestEmitError(f"变体 {key!r} 缺配置根")
            config_roots[key] = value
        missing = [v for v in VARIANTS if v not in config_roots]
        if missing:
            raise ManifestEmitError(f"缺变体的配置根：{missing}")

        wanted = [tag.strip() for tag in args.tiers.split(",") if tag.strip()]
        for tag in wanted:
            if tag not in TIERS:
                raise ManifestEmitError(f"未知档 {tag!r}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        for variant, fills in VARIANTS.items():
            _assert_fills_column_invariant(variant, fills)  # ★ 写盘前强制
            for tag in wanted:
                length = TIERS[tag]
                tier = copy.deepcopy(base)
                config_rel = (
                    f"{args.config_prefix}/rig_synth/configs/{config_roots[variant]}/"
                    f"T013-synth-{variant}-{tag}.yaml"
                )
                tier["config"] = config_rel
                tier["data"] = dict(tier["data"], subset_size=SYNTH_STEPS)
                gate = dict(
                    tier["acceptance"]["formal_gate"],
                    min_train_samples=SYNTH_STEPS,
                    min_optimizer_steps=SYNTH_STEPS,
                )
                tier["acceptance"] = dict(tier["acceptance"], formal_gate=gate)
                tier["rig_context_length"] = length
                tier["rig_context_variant"] = variant
                tier["rig_context_fills_column"] = fills
                tier["rig_context_target_tokens"] = length
                tier["rig_gate_note"] = GATE_NOTE
                # ★ A4 适用性（裁定 1）：单跑档的 a4_status 直接是「未适用（单跑）」，
                #   **绝不是 pass**——与 provenance 的字段逐字同口径。
                dual_run = tag in A4_DUAL_RUN_TIERS
                tier["rig_context_a4_dual_run"] = dual_run
                tier["rig_context_a4_status"] = (
                    None if dual_run else A4_STATUS_NOT_APPLICABLE_SINGLE_RUN
                )
                tier["rig_context_a4_status_display"] = (
                    "待跑" if dual_run else A4_STATUS_DISPLAY_SINGLE_RUN
                )
                tier["rig_context_applicable_criteria"] = (
                    list(A4_APPLICABLE_CRITERIA_DUAL_RUN)
                    if dual_run
                    else list(A4_APPLICABLE_CRITERIA_SINGLE_RUN)
                )
                # 清单里的 acceptance.criteria 同步反映"该档实际适用的判据"，
                # 使"单跑档不判 A4"在**清单层**也是显式的（不靠读注释）。
                tier["acceptance"] = dict(
                    tier["acceptance"],
                    criteria=(
                        list(A4_APPLICABLE_CRITERIA_DUAL_RUN)
                        if dual_run
                        else list(A4_APPLICABLE_CRITERIA_SINGLE_RUN)
                    ),
                    a4_note=(
                        "A4 双跑（同 config 同 seed）"
                        if dual_run
                        else f"A4 = {A4_STATUS_DISPLAY_SINGLE_RUN}（本档单跑，"
                        "**不得记为通过**；裁定 1）"
                    ),
                )

                doc = dict(manifest)
                doc["tiers"] = [tier]
                doc["rig_note"] = (
                    "rig 本地单档清单（task-synth-seq · 合成长序列）；"
                    "只用于 run_matrix54.sh 的档解析，"
                    "不回写仓库 manifest。★ 数据来源是**合成宿主数据根**（非原始 ELAM V5 分布），"
                    "经 GRASPO_ELAM_HOST_ROOT 注入。"
                )
                name = f"T013-synth-{variant}-{tag}.json"
                out_path = args.out_dir / name
                out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
                written.append(str(out_path))
                print(
                    f"[ok] {name}  length={length} variant={variant} fills_column={fills} "
                    f"steps={SYNTH_STEPS} a4_dual_run={dual_run} "
                    f"a4_status={'待跑' if dual_run else A4_STATUS_DISPLAY_SINGLE_RUN}"
                )

        print(f"[ok] wrote {len(written)} 清单")
        print(
            "★ 提醒：v1-image-copy 的清单 fills_column=true（可填列）；"
            "v2-text-only 全为 false（**不得填列**，仅作单独标注的数据点）。"
        )
        return 0
    except ManifestEmitError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
