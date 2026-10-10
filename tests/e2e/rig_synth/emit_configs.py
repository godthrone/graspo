#!/usr/bin/env python3
"""合成长序列的**配套配置生成器** —— 只改 `data.max_prompt_length`（纯标准库）。

**职责边界**

- **本模块**：把仓内 ``samples/configs/matrix54/T013.yaml`` 逐字复制成每个目标长度一份，
  **唯一改动**是 ``data.max_prompt_length``（以及必须唯一的 ``training.run_name``——
  否则同名 run 在 collector 里不可区分）。其余字段**逐字节照抄**，并写一份 diff 证据。
- **不负责**：数据构造（``build_synth_seq.py``）、上机执行（``run_synth.sh``）、
  清单（``emit_manifests.py``）。

**为什么只改这一个字段**（口径：长度是唯一自变量）

`_config_mapping.py:433` 实测把 ``data.max_prompt_length`` 透传成 ms-swift ``--max_length``。
本包的目标就是"让长度成为自变量"——其余任何字段变化都会污染对照，所以：

- ``data.train_path`` **保持** ``/work/elam-v5/subsets/<tier>.jsonl`` **不动**（合成数据经
  ``GRASPO_ELAM_HOST_ROOT`` 注入，见 ``run_synth.sh``；这样 ``run_matrix54.sh:210`` 的
  路径一致性检查通过，**零改动既有 runner**）；
- ``model`` / ``training``（除 ``run_name``）/ ``lora`` / ``msswift`` **逐字不动**；
- ``output_dir`` 保持 ``/out/<TIER>``（宿主侧由 ``RUN_ROOT`` 逐 case 隔离）。

**用法**

    python3 emit_configs.py --base <仓内 T013.yaml> --out-dir .local/.../rig_synth/configs \\
        [--tiers 4k,8k,16k,32k,64k,96k,128k] [--variants v1-image-copy,v2-text-only]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

#: 档 → 目标长度（与 build_synth_seq.TIERS 同源；此处独立声明只为避免 import 依赖）。
TIERS: dict[str, int] = {
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "96k": 98304,
    "128k": 131072,
}

#: 变体 → 产物目录名（与 build_synth_seq.VARIANTS 的 tag 一致；命名隔离即防呆）。
VARIANTS: dict[str, str] = {
    "v1-image-copy": "v1-image-copy",
    "v2-text-only": "v2-text-only",
}

#: 变体 → 是否用于填「最大可行上下文」列（★ 与 build_synth_seq 的守卫同口径）。
FILLS_COLUMN: dict[str, bool] = {
    "v1-image-copy": True,
    "v2-text-only": False,
}

#: ★ A4 适用策略（指挥官 2026-09-19 裁定 1；与 build_synth_seq / emit_manifests 同源口径）：
#: 4K–32K 双跑（A4 适用）；64K 起单跑（A4 = 未适用（单跑），**不得记为通过**）。
A4_DUAL_RUN_TIERS: tuple[str, ...] = ("4k", "8k", "16k", "32k")

#: ★ 合成档的 `save_steps` 覆盖（指挥官 2026-09-19 裁定 3 批准）。
#:
#: **为什么覆盖**：A3（checkpoint 可重载）是本档的**适用判据**之一，其判定对象是
#: `scripts/collect_results.py` 的 `SWIFT_CHECKPOINT_GLOB = "checkpoint-*"` ——
#: 必须有 `checkpoint-<step>/` 目录。合成档只有 **1 step**，而基线是 `save_steps: 100`。
#:
#: **★ 诚实登记**：上游 ms-swift 4.5.3 **已经**保证末步保存
#: （`.local/refs/ms-swift-4.5.3/src/swift/trainers/patcher.py:61-72` 的
#: `DefaultFlowCallbackNew.on_step_end`：`state.global_step == state.max_steps` 且
#: `save_strategy != NO` ⇒ 强制 `control.should_save = True`；
#: `patcher.py:107` 用 `trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]` **替换**默认回调
#: ⇒ 确实生效）。所以本覆盖是 **"冗余保险"（belt and suspenders），不是"修 bug"**：
#: 1 step 档下 `save_steps: 1` 与 `100` **恰好都产出 1 个 `checkpoint-1`**，产物完全相同。
#: 覆盖的价值在**显式可读**（配置里看得见，无需读 ms-swift 源码）与**双机制冗余**。
#: 证据见 `.local/…/task-synth-seq/evidence/save_steps_investigation.md`。
SYNTH_SAVE_STEPS = 1

#: 配置文件里**唯一**允许被改写的键（逐行文本级改写，避免 yaml 库依赖）。
#: ★ 恰三个，且每个都有裁定依据 —— 这不是"可以随便改配置"，是"只能改这三个"。
ALLOWED_PATCH_KEY = "  max_prompt_length:"
ALLOWED_RUN_NAME_KEY = "  run_name:"
ALLOWED_SAVE_STEPS_KEY = "  save_steps:"
ALLOWED_KEYS = (ALLOWED_PATCH_KEY, ALLOWED_RUN_NAME_KEY, ALLOWED_SAVE_STEPS_KEY)


class ConfigEmitError(Exception):
    """配置改写越界（改写到了不允许的键）⇒ fail-closed，不产出配置。"""


def _patch_config(
    text: str, *, max_prompt_length: int, run_name: str, save_steps: int = SYNTH_SAVE_STEPS
) -> tuple[str, list[str]]:
    """逐行改写；**每一条改动都必须落在白名单键上**，否则抛错（§2 防呆）。

    返回 ``(新文本, 改动说明列表)``。
    """
    out_lines: list[str] = []
    changes: list[str] = []
    seen_length = False
    seen_run_name = False
    seen_save_steps = False
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith(ALLOWED_PATCH_KEY):
            if seen_length:
                raise ConfigEmitError("配置里出现多个 max_prompt_length（单一真相源被破坏）")
            seen_length = True
            old = stripped
            out_lines.append(f"  max_prompt_length: {max_prompt_length}\n")
            changes.append(
                f"data.max_prompt_length: {old.split(':', 1)[1].strip()} -> {max_prompt_length}"
            )
            continue
        if stripped.startswith(ALLOWED_RUN_NAME_KEY):
            if seen_run_name:
                raise ConfigEmitError("配置里出现多个 run_name")
            seen_run_name = True
            old = stripped
            out_lines.append(f"  run_name: {run_name}\n")
            changes.append(f"training.run_name: {old.split(':', 1)[1].strip()} -> {run_name}")
            continue
        if stripped.startswith(ALLOWED_SAVE_STEPS_KEY):
            if seen_save_steps:
                raise ConfigEmitError("配置里出现多个 save_steps")
            seen_save_steps = True
            old = stripped
            out_lines.append(f"  save_steps: {save_steps}\n")
            changes.append(
                f"training.save_steps: {old.split(':', 1)[1].strip()} -> {save_steps}"
                "（★ 合成档 A3 保障，见 SYNTH_SAVE_STEPS 注释）"
            )
            continue
        out_lines.append(line)
    if not seen_length:
        raise ConfigEmitError(f"配置里找不到 {ALLOWED_PATCH_KEY!r} —— 基线格式变了？")
    if not seen_run_name:
        raise ConfigEmitError(f"配置里找不到 {ALLOWED_RUN_NAME_KEY!r} —— 基线格式变了？")
    if not seen_save_steps:
        raise ConfigEmitError(f"配置里找不到 {ALLOWED_SAVE_STEPS_KEY!r} —— 基线格式变了？")
    return "".join(out_lines), changes


def _assert_only_allowed_diff(base_text: str, new_text: str) -> None:
    """断言 base→new 的 diff **只**触及白名单键（第二道防线，独立于上面的逐行改写）。"""
    diff = [
        line
        for line in difflib.unified_diff(
            base_text.splitlines(), new_text.splitlines(), lineterm="", n=0
        )
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    offenders = [line for line in diff if not any(line[1:].startswith(key) for key in ALLOWED_KEYS)]
    if offenders:
        raise ConfigEmitError(
            "diff 触及了白名单之外的键（只允许改 "
            + " / ".join(key.strip().rstrip(":") for key in ALLOWED_KEYS)
            + "）：\n  "
            + "\n  ".join(offenders)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成长序列配套配置生成器（只改长度字段）")
    parser.add_argument("--base", required=True, type=Path, help="仓内基线配置（T013.yaml）")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tiers", default=",".join(TIERS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--diff-out", type=Path, default=None, help="diff 证据落点（文本）")
    args = parser.parse_args(argv)

    try:
        base_text = args.base.read_text(encoding="utf-8")
        # ★ 基线指纹：把"我抄的是哪一份"钉死，基线漂移时可检出（可复现，§6）。
        base_sha256 = hashlib.sha256(base_text.encode("utf-8")).hexdigest()
        wanted_tiers = [tag.strip() for tag in args.tiers.split(",") if tag.strip()]
        wanted_variants = [tag.strip() for tag in args.variants.split(",") if tag.strip()]
        for tag in wanted_tiers:
            if tag not in TIERS:
                raise ConfigEmitError(f"未知档 {tag!r}；可用 {sorted(TIERS)}")
        for tag in wanted_variants:
            if tag not in VARIANTS:
                raise ConfigEmitError(f"未知变体 {tag!r}；可用 {sorted(VARIANTS)}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        diffs: list[str] = []
        written: list[str] = []

        for variant in wanted_variants:
            fills = FILLS_COLUMN[variant]
            for tag in wanted_tiers:
                length = TIERS[tag]
                variant_dir = args.out_dir / variant
                variant_dir.mkdir(parents=True, exist_ok=True)
                # 文件名带变体与档，**结构上不可能混**：同一目录里不会同时出现两个变体的同档配置。
                run_name = f"T013-synth-{variant}-{tag}"
                new_text, changes = _patch_config(
                    base_text, max_prompt_length=length, run_name=run_name
                )
                _assert_only_allowed_diff(base_text, new_text)

                banner = (
                    f"# ==== 合成数据配置（task-synth-seq）====\n"
                    f"# 变体: {variant}  |  档: {tag}（目标 {length} token）\n"
                    f"# 是否用于填「最大可行上下文」列: {fills}"
                    + ("" if fills else "  ⚠ 本变体**不得**用于填该列（仅作单独标注的数据点）")
                    + "\n"
                    "# A4 适用性: "
                    + (
                        "双跑（4K–32K）⇒ A4 适用"
                        if tag in A4_DUAL_RUN_TIERS
                        else "单跑（64K+）⇒ A4 = 未适用（单跑），**不得记为通过**"
                    )
                    + "\n"
                    f"# 数据来源: **合成宿主数据根**（GRASPO_ELAM_HOST_ROOT 指向），"
                    f"非原始 ELAM V5 分布\n"
                    f"# A3 保障: training.save_steps 覆盖为 {SYNTH_SAVE_STEPS}"
                    f"（合成档仅 1 步、须保证 A3 有可判定对象 checkpoint-*；"
                    f"**基线仍为 save_steps 100**）\n"
                    f"#   └ 说明: 本覆盖是**冗余保险**——上游 ms-swift 4.5.3 "
                    f"patcher.py:61-72 已强制末步保存（DEFAULT_CALLBACKS 被替换，见 :107），\n"
                    f"#     故 1 步档下 save_steps 1 与 100 都恰好产出 1 个 checkpoint-1、"
                    f"产物相同。\n"
                    # ★ 基线指纹（§6 可复现 / §1.4 单一真相源）：基线若漂移，由此可检出。
                    f"# 基线文件: {args.base}  sha256={base_sha256}\n"
                    f"# 本配置相对基线的改动（恰 {len(ALLOWED_KEYS)} 个白名单键）: "
                    + "; ".join(changes)
                    + "\n"
                    "# ==== 以下为基线逐字内容 ====\n"
                )
                out_path = variant_dir / f"T013-synth-{variant}-{tag}.yaml"
                out_path.write_text(banner + new_text, encoding="utf-8")
                written.append(str(out_path))

                diffs.append(
                    f"### {out_path}\n"
                    + "".join(
                        difflib.unified_diff(
                            base_text.splitlines(keepends=True),
                            new_text.splitlines(keepends=True),
                            fromfile=str(args.base),
                            tofile=str(out_path),
                            n=1,
                        )
                    )
                    + f"\n（白名单校验：仅 {ALLOWED_PATCH_KEY!r} 与 "
                    + f"{ALLOWED_RUN_NAME_KEY!r} 被改写）\n"
                )
                print(f"[ok] {out_path}  changes={changes}")

        (args.out_dir / "baseline_provenance.json").write_text(
            json.dumps(
                {
                    "base_path": str(args.base),
                    "base_sha256": base_sha256,
                    "base_bytes": len(base_text.encode("utf-8")),
                    "allowed_patch_keys": [key.strip().rstrip(":") for key in ALLOWED_KEYS],
                    # ★ 完整记录"我从基线派生出了什么"——必须与 base_sha256 一起读才完整：
                    #   base_sha256 = 我抄的是哪一版基线；overrides_applied = 我改了哪些键。
                    "overrides_applied": [change.split(":")[0].strip() for change in changes],
                    "save_steps_override": {
                        "value": SYNTH_SAVE_STEPS,
                        "baseline_value_note": "基线 T013.yaml 当前为 save_steps: 100",
                        "rationale": (
                            "合成档仅 1 步 ⇒ 须保证 A3（checkpoint 可重载）有可判定对象"
                            "（collect_results.py 的 SWIFT_CHECKPOINT_GLOB='checkpoint-*'）。"
                            "裁定 3 批准。"
                        ),
                        "honest_caveat": (
                            "★ 上游 ms-swift 4.5.3 patcher.py:61-72 已强制末步保存"
                            "（:107 替换 DEFAULT_CALLBACKS 使其生效）⇒ 本覆盖是"
                            "**冗余保险**，不是修 bug：1 步档下 save_steps 1 与 100 "
                            "恰好都产出 1 个 checkpoint-1、产物相同。"
                        ),
                        "evidence": "evidence/save_steps_investigation.md",
                    },
                    "note": (
                        "★ 全部配置只改这两个键（逐字白名单 + diff 双重断言）。"
                        "基线 sha256 是「我抄的是哪一份」的指纹——基线漂移时可检出。"
                    ),
                    "a4_dual_run_tiers": list(A4_DUAL_RUN_TIERS),
                    "generated_configs": written,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"[ok] baseline sha256 = {base_sha256}")
        if args.diff_out:
            args.diff_out.parent.mkdir(parents=True, exist_ok=True)
            args.diff_out.write_text("\n".join(diffs), encoding="utf-8")
            print(f"[ok] diff -> {args.diff_out}")
        print(f"[ok] wrote {len(written)} 配置")
        return 0
    except ConfigEmitError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
