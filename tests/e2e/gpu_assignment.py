#!/usr/bin/env python3
"""矩阵跑批的「卡集合覆盖点」——生成器与运行器共用的**唯一**实现（宪法 §1.4 单一真相源）。

## 为什么需要它

`generate_matrix.GPU_SETS` 写死 ``{1: (0,), 2: (0, 1), 4: (0, 1, 2, 3)}`` ⇒ **全部 10 个
1 卡档都落在 GPU0**，只能串行；且 GPU0 一旦被他人作业占用，整批开不了工。228 是共享机，
卡随他人作业变动 ⇒ 需要一个**显式、集中、可审计**的覆盖点，把某档（或某几档）改到别的
空闲卡上跑。

## 覆盖点长什么样（**只有这一处**）

`SPEC` 语法（**逐档显式映射，无通配符**，§2.2 显式即防呆）：

    <档号>=<卡号 csv> ; <档号>=<卡号 csv> ; …
    T001=4 ; T010=4,5 ; T023=4,5

- 档号必须是 ``T###``（T001–T054），**不允许** ``*`` / 范围 / 正则——通配符展开的结果
  对操作者不可见（§2.4），卡集合又直接决定产物是否可比，不能靠猜。
- 未在 SPEC 里出现的档 ⇒ 沿用按卡数的默认集合（``GPU_SETS``），并**显式打印**为「默认」。
- SPEC 为空（或不给）⇒ **完全不覆盖**，产物与旧版逐字节相同。
- SPEC 里出现未知档号 ⇒ fail-closed（防"写错档号被静默忽略"）。

两个调用点（同一实现，不另造一套）：

1. **生成期**：`generate_matrix.py --gpu-override SPEC`（校验后写进 manifest 每档
   ``tiers[].gpus``；manifest 即「档 → 卡集合」的落盘记录）。
2. **运行期**：runner 从清单取 ``tiers[].gpus`` ⇒ 不改 runner、不改档位 YAML；
   换一套卡集合 = 用 ``GRASPO_RUNNER_MANIFEST`` 指向另一份清单（既有环境变量覆盖点）。
   本次运行实际用了哪几张卡由既有守卫写进运行产物：``[gpu-guard] OK:
   NVIDIA_VISIBLE_DEVICES=…``。

## 范围守卫（fail-closed，生成期与运行期同一条判据）

- 卡号必须 ⊆ ``{0..5}``；出现 **GPU6/7（生产 vLLM，各约 74 GiB，永不触碰）** 即拒绝；
- 卡数 ≤ ``4``；非空；不得重复；
- **卡数必须等于该档的 ``cards``**——覆盖只改「用哪几张」，**不改卡数**；
  不符即拒绝（不"将就"，§2.3 边界校验即防呆）。
- 跨 NUMA 只做**信息性提示**（GPU0–3 = NUMA0，GPU4–7 = NUMA1），不阻断。

## A4「同档两跑必须同卡位」

同一档的两次运行若卡位不同，会引入 GPU 内核/归约差异，破坏 ``A4 = 同 config 同 seed
双跑一致``。因此：

- 卡集合**只由清单给出**（档 → 卡集合映射，生成期落盘）；
- 两次运行必须引用**同一份清单**（``GRASPO_RUNNER_MANIFEST`` 指同一路径）；
- :func:`gpu_assignment_fingerprint` 给出映射的 sha256 指纹，生成时打印、可随时复核：
  ``python3 tests/e2e/gpu_assignment.py check --manifest <清单> --expect-fingerprint <指纹>``
  指纹不一致 ⇒ 非零退出 ⇒ 这两跑**不可比**，必须在同一份清单下重跑。

本模块**只跑 stdlib**（生成器要求 stdlib-only，运行期也要在裸解释器下可用）；
不读源码、不碰 GPU、不启动任何进程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# ── 边界常量（单一真相源；生成器与 runner 侧校验都引用这里）────────────────────
#: 允许使用的卡（228 上仅 GPU0–5 可用于矩阵跑批）。
GPU_ALLOWED: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
#: **永不触碰**的生产卡：GPU6/7 跑生产 vLLM（各约 74741 MiB）。
GPU_FORBIDDEN_PRODUCTION: tuple[int, ...] = (6, 7)
#: 单次作业卡数上限。
MAX_CARDS_PER_JOB = 4
#: NUMA 归属（GPU0–3 = NUMA0，GPU4–7 = NUMA1）；仅用于信息性提示，不作判据。
NUMA_NODE_BY_GPU: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

_TIER_ID_RE = re.compile(r"^T\d{3}$")


class GpuAssignmentError(ValueError):
    """卡集合覆盖/fail-closed 校验失败（不合法即拒绝，绝不"将就"）。"""


# ── SPEC 解析 ───────────────────────────────────────────────────────────────


def parse_override(spec: str) -> dict[str, tuple[int, ...]]:
    """把 ``SPEC`` 解析成 ``{档号: 卡号元组}``；空串 ⇒ 空映射（= 不覆盖）。

    显式拒绝：空段、缺 ``=``、档号不合 ``T###``、重复档号、卡号非整数、空卡号列表。
    **不**做通配符/范围展开（§2.4：展开结果对操作者不可见即不许用）。
    """
    if not isinstance(spec, str):
        raise GpuAssignmentError(f"覆盖值必须是字符串，收到 {type(spec).__name__}")
    override: dict[str, tuple[int, ...]] = {}
    for raw in spec.split(";"):
        segment = raw.strip()
        if not segment:
            # 只容忍首尾/多余分隔符造成的空段（如 "T001=4;"）；整串为空即"不覆盖"。
            continue
        if "=" not in segment:
            raise GpuAssignmentError(
                f"覆盖段 {segment!r} 缺少 '='；正确形态：T001=4;T010=4,5（无通配符）"
            )
        tier_id, _, raw_gpus = segment.partition("=")
        tier_id = tier_id.strip()
        if not _TIER_ID_RE.match(tier_id):
            raise GpuAssignmentError(
                f"档号 {tier_id!r} 不合规（必须形如 T001–T054；不允许通配符*、范围或正则）"
            )
        if tier_id in override:
            raise GpuAssignmentError(f"档号 {tier_id} 在覆盖值里出现多次（歧义即拒绝）")
        raw_items = [item.strip() for item in raw_gpus.split(",")]
        if not [item for item in raw_items if item]:
            raise GpuAssignmentError(f"{tier_id} 的卡号列表为空；覆盖必须显式给出卡号")
        gpus: list[int] = []
        for item in raw_items:
            if not item:
                raise GpuAssignmentError(f"{tier_id} 的卡号列表含空元素（多余逗号？）")
            try:
                gpus.append(int(item))
            except ValueError as exc:
                raise GpuAssignmentError(f"{tier_id} 的卡号 {item!r} 不是整数") from exc
        override[tier_id] = tuple(gpus)
    return override


# ── 范围守卫 ────────────────────────────────────────────────────────────────


def check_gpus(tier_id: str, gpus: Sequence[int], cards: int) -> tuple[int, ...]:
    """校验一组卡；返回规范化元组。任何违规 ⇒ :class:`GpuAssignmentError`（fail-closed）。

    判据：非空、≤4 卡、去重、⊆{0..5}、不含 GPU6/7、**卡数 == 该档 ``cards``**。
    """
    normalized = tuple(gpus)
    if not normalized:
        raise GpuAssignmentError(f"{tier_id}: 卡集合为空（必须至少 1 张卡）")
    if len(normalized) > MAX_CARDS_PER_JOB:
        raise GpuAssignmentError(
            f"{tier_id}: 卡数 {len(normalized)} 超过单次作业上限 {MAX_CARDS_PER_JOB} 卡"
        )
    duplicates = sorted({gpu for gpu in normalized if list(normalized).count(gpu) > 1})
    if duplicates:
        raise GpuAssignmentError(f"{tier_id}: 卡集合 {list(normalized)} 含重复卡 {duplicates}")
    forbidden = [gpu for gpu in normalized if gpu in GPU_FORBIDDEN_PRODUCTION]
    if forbidden:
        raise GpuAssignmentError(
            f"{tier_id}: 卡集合 {list(normalized)} 含生产卡 GPU{forbidden}"
            f"（GPU6/7 跑生产 vLLM，永不触碰）"
        )
    out_of_range = [gpu for gpu in normalized if gpu not in GPU_ALLOWED]
    if out_of_range:
        raise GpuAssignmentError(
            f"{tier_id}: 卡集合 {list(normalized)} 含越界卡 {out_of_range}"
            f"（只允许 GPU0–{GPU_ALLOWED[-1]}）"
        )
    if len(normalized) != cards:
        raise GpuAssignmentError(
            f"{tier_id}: 卡数不符 —— 覆盖给了 {len(normalized)} 张 "
            f"{list(normalized)}，该档 cards={cards}。"
            f"覆盖只改『用哪几张』，**不改卡数**（TP × PP × DP = 卡数 的不变量）"
        )
    return normalized


def numa_nodes(gpus: Sequence[int]) -> tuple[int, ...]:
    """这组卡落在哪些 NUMA 节点（信息性，不参与校验）。"""
    return tuple(sorted({NUMA_NODE_BY_GPU[gpu] for gpu in gpus if gpu in NUMA_NODE_BY_GPU}))


def crosses_numa(gpus: Sequence[int]) -> bool:
    """是否跨 NUMA（信息性提示；跨 NUMA 只提示、不阻断）。"""
    return len(numa_nodes(gpus)) > 1


def numa_hint(gpus: Sequence[int]) -> str:
    """可读的 NUMA 提示串。"""
    nodes = numa_nodes(gpus)
    if not nodes:
        return "NUMA=未知"
    if crosses_numa(gpus):
        return f"跨 NUMA{'/'.join(str(n) for n in nodes)}（仅提示，不阻断）"
    return f"NUMA{''.join(str(n) for n in nodes)}"


# ── 生效值解析（生成器与 CLI 共用）──────────────────────────────────────────


def resolve_assignment(
    default_gpus_by_tier: Mapping[str, Sequence[int]],
    cards_by_tier: Mapping[str, int],
    override: Mapping[str, Sequence[int]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """算出每档生效卡集合与来源（``"override"`` / ``"default"``）；全程 fail-closed。

    未知档号（SPEC 里出现、台账里没有）⇒ 拒绝：写错档号被静默忽略会让"以为改了、
    实际没改"，是最危险的隐性失败。
    """
    unknown = sorted(set(override) - set(default_gpus_by_tier))
    if unknown:
        raise GpuAssignmentError(
            f"覆盖值含未知档号 {unknown}；合法档号见台账（T001–T054）。"
            f"（防呆：不静默忽略写错的档号）"
        )
    resolved: dict[str, tuple[int, ...]] = {}
    sources: dict[str, str] = {}
    for tier_id, default in default_gpus_by_tier.items():
        if tier_id in override:
            resolved[tier_id] = check_gpus(tier_id, override[tier_id], cards_by_tier[tier_id])
            sources[tier_id] = "override"
        else:
            resolved[tier_id] = check_gpus(tier_id, default, cards_by_tier[tier_id])
            sources[tier_id] = "default"
    return resolved, sources


def gpu_assignment_fingerprint(pairs: Iterable[tuple[str, Sequence[int]]]) -> str:
    """档 → 卡集合映射的 sha256 指纹（**与顺序无关**，排序后计算）。

    A4 用途：两次运行引用同一份清单 ⇒ 指纹一致；指纹不一致 ⇒ 两次运行**不可比**。
    """
    lines = [f"{tier_id}:{','.join(str(gpu) for gpu in gpus)}" for tier_id, gpus in pairs]
    payload = "\n".join(sorted(lines)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_assignment_line(
    tier_id: str, gpus: Sequence[int], cards: int, source: str
) -> str:
    """单档可读行：本档用哪几张卡 + 卡数 + 来源 + NUMA 提示。"""
    source_label = "覆盖" if source == "override" else "默认（GPU_SETS）"
    return (
        f"{tier_id}: GPU{list(gpus)}（{cards}卡，来源={source_label}，{numa_hint(gpus)}）"
    )


def describe_default_sets(default_by_cards: Mapping[int, Sequence[int]]) -> str:
    """按卡数的默认集合，一行说清（默认行为必须可被一句话说清）。"""
    return "; ".join(
        f"{cards}卡 -> GPU{list(gpus)}" for cards, gpus in sorted(default_by_cards.items())
    )


# ── CLI：清单/覆盖值的独立复核入口（供报告与 A4 取证使用）────────────────────


def load_manifest(manifest_path: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    """读清单，返回 ``({档号: {cards, gpus}}, 清单内记录的覆盖值或 None)``。"""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise GpuAssignmentError(f"{manifest_path}: 清单里没有非空 tiers 列表")
    result: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        tier_id = str(tier["tier_id"])
        result[tier_id] = {"cards": int(tier["cards"]), "gpus": list(tier["gpus"])}
    recorded = (payload.get("gpu_assignment") or {}).get("override_spec")
    return result, (None if recorded is None else str(recorded))


def _cmd_check(args: argparse.Namespace) -> int:
    tiers, recorded_spec = load_manifest(Path(args.manifest))
    cards_by_tier = {tier_id: info["cards"] for tier_id, info in tiers.items()}
    defaults = {tier_id: info["gpus"] for tier_id, info in tiers.items()}
    effective_spec = args.spec if args.spec is not None else recorded_spec
    if effective_spec is None:
        override: dict[str, tuple[int, ...]] = {}
        origin = "清单无 gpu_assignment 段 ⇒ 默认 GPU_SETS"
    else:
        override = parse_override(effective_spec)
        origin = (
            f"覆盖 SPEC: {effective_spec}"
            + ("" if args.spec is not None else "（来自清单 gpu_assignment.override_spec）")
        )
        print(origin, file=sys.stderr)
    resolved, sources = resolve_assignment(defaults, cards_by_tier, override)
    # 清单自洽性：若清单自己记了覆盖值，按覆盖值重算的卡集合必须与清单里的逐档 gpus 一致
    # （不一致 = 有人手改过清单，或覆盖值/卡集合漂移 ⇒ fail-closed）。
    if effective_spec is not None and args.spec is None:
        drifted = sorted(
            tier_id
            for tier_id, gpus in resolved.items()
            if gpus != tuple(tiers[tier_id]["gpus"])
        )
        if drifted:
            raise GpuAssignmentError(
                f"清单自相矛盾：按 gpu_assignment.override_spec 重算的卡集合与 tiers[].gpus "
                f"不一致，差异档 {drifted}（清单被手改过？）"
            )
    for tier_id in sorted(resolved):
        if not args.quiet:
            line = format_assignment_line(
                tier_id, resolved[tier_id], cards_by_tier[tier_id], sources[tier_id]
            )
            print("  " + line, file=sys.stderr)
    fingerprint = gpu_assignment_fingerprint((tier_id, gpus) for tier_id, gpus in resolved.items())
    overridden = sum(1 for source in sources.values() if source == "override")
    print(
        f"生效档数={len(resolved)} 覆盖档数={overridden} "
        f"来源={'覆盖' if overridden else '默认（GPU_SETS）'} "
        f"指纹(sha256)={fingerprint}"
    )
    if args.expect_fingerprint:
        if fingerprint != args.expect_fingerprint:
            print(
                f"FATAL: 卡集合指纹不一致 —— 期望 {args.expect_fingerprint}，实际 {fingerprint}。"
                f"两次运行若指纹不同则**同卡位不成立**，A4 双跑不可比，必须用同一份清单重跑。",
                file=sys.stderr,
            )
            return 1
        print("指纹与期望一致：A4『同档两跑同卡位』成立。", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="矩阵卡集合覆盖值的独立复核入口（只读清单，不碰 GPU、不改文件）。",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="校验一份清单（可叠加 --spec）并打印卡集合指纹")
    check.add_argument("--manifest", required=True, help="matrix54_manifest.json")
    check.add_argument("--spec", default=None, help="可选的覆盖值 T001=4;T010=4,5")
    check.add_argument("--expect-fingerprint", default=None, help="期望的卡集合指纹（A4 复核）")
    check.add_argument("--quiet", action="store_true", help="只打印指纹行")
    check.set_defaults(func=_cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except GpuAssignmentError as exc:
        print(f"FATAL(gpu-assignment): {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as exc:
        print(f"FATAL(gpu-assignment): 读取清单失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
