#!/usr/bin/env python3
"""regen_field_guard.py — 生成链「防丢失」共享守卫（清单生成器通用）。

## 为什么单独成模块（§1.4 单一真相源）

同一个病在生成链上被撞到过**多次**：**重新生成会静默丢掉在盘的真值**。清单生成器
（``generate_matrix_manifest_source.py`` / ``generate_matrix_manifest_fsdp.py`` /
``generate_matrix.py``）各写一套"别丢字段"的逻辑，迟早漂移；所以判定逻辑只有这一份。
三个消费者都只依赖 stdlib，本模块也**只依赖 stdlib**（不破坏它们各自的独立加载约束）。

## 硬不变量

1. ``field_paths()`` 递归收集 JSON 的**全部字段路径**（列表下标不进路径：逐档字段在 54 档
   之间是同一个字段，不该因为某档没回填就判"少了"）。
2. ``guard_regeneration_keeps_on_disk_fields()``：**重生成不得移除在盘已有的字段**。
   要写出的新内容若比在盘旧内容**少任何字段路径** ⇒ **fail-closed**（逐条列出，调用方
   不写盘）。命中的路径若在 post-fill 注册表里，会标注 ``POST-FILL``（已知由生成器之外
   回填），否则标注 ``UNKNOWN``（陌生漂移，更要看）。
3. 例外只能来自**显式具名清单**（``allowed_drop`` 形参，精确路径或含 ``*`` 的 fnmatch
   模式），且每次用到都会打印。

## 病史（三处实证，详见 .local/hb-workspace/20261014-sft-kickoff/）

* 病 1 ``build_data.py --preserve-from`` 白名单过窄 ⇒ 静默丢逐档 16 类回填字段，四态由
  13/41/0/0 崩成 13/23/15/3（另见 ``tools/matrix/build_data.py`` 内同名不变量）。
* 病 2 清单重生成丢"后填字段"（status/availability/matrix_conclusion/实测显存 …）。
* 病 3 runner 的修复只落在产物、没回写生成器 ⇒ 下次重生成即复发（由逐字节测试兜住）。
"""

from __future__ import annotations

import copy
import fnmatch
import json
import sys
from pathlib import Path

__all__ = [
    "field_paths",
    "matches_any",
    "post_fill_kind",
    "guard_regeneration_keeps_on_disk_fields",
    "prune_like",
    "apply_like_at",
    "field_universe_delta",
]


def field_paths(obj, prefix: str = "") -> set[str]:
    """递归收集 JSON 里出现的**全部字段路径**（索引无关：列表元素统一记 ``[]``）。

    路径形如 ``/tiers[]/status``、``/feasibility_model/opd_teacher/measured``。
    """
    out: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}/{key}"
            out.add(path)
            out |= field_paths(value, path)
    elif isinstance(obj, list):
        path = prefix + "[]"
        out.add(path)
        for value in obj:
            out |= field_paths(value, path)
    return out


def matches_any(path: str, patterns) -> bool:
    """``path`` 是否命中 ``patterns`` 里某条：精确相等、或位于其下（``pat + "/"`` 前缀）、
    或（模式含 ``*``/``?`` 时）fnmatch 命中。

    ⛔ 前缀匹配只对**显式给出的**具名路径生效；不许用 ``/*`` 之类把整棵子树放行。
    """
    for pat in patterns:
        if path == pat or path.startswith(pat + "/"):
            return True
        if ("*" in pat or "?" in pat) and fnmatch.fnmatchcase(path, pat):
            return True
    return False


def post_fill_kind(path: str, post_fill_patterns) -> str:
    """``POST-FILL``（在注册表里）/ ``UNKNOWN``（不在）。"""
    return "POST-FILL" if matches_any(path, post_fill_patterns) else "UNKNOWN"


def field_universe_delta(on_disk_doc, new_doc) -> set[str]:
    """在盘有、新内容没有的字段路径（= 重生成**会丢掉**的字段）。"""
    return field_paths(on_disk_doc) - field_paths(new_doc)


def guard_regeneration_keeps_on_disk_fields(
    new_doc,
    out_path: Path,
    *,
    post_fill_patterns=(),
    allowed_drop=(),
    what: str = "清单",
) -> list[str]:
    """★ fail-closed：要写的新内容不得比 ``out_path`` 上已有的旧内容**少字段路径**。

    返回**仍未获豁免**的被丢路径（空列表 = 通过）。调用方拿到非空列表就**不得写盘**。
    首次生成（``out_path`` 不存在）没有可比对象 ⇒ 直接通过。
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        return []
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    on_disk_paths = field_paths(on_disk)
    new_paths = field_paths(new_doc)
    lost = sorted(on_disk_paths - new_paths)
    if not lost:
        print(
            f"字段全集不减少 ✓ {what}: {out_path}（在盘 {len(on_disk_paths)} 条路径，"
            f"新内容 {len(new_paths)} 条，丢弃 0 条）"
        )
        return []
    excused = [p for p in lost if matches_any(p, allowed_drop)]
    for path in excused:
        print(f"ALLOW-DROP（显式具名例外）: {path}")
    missing = [p for p in lost if not matches_any(p, allowed_drop)]
    if not missing:
        return []
    print(
        f"FATAL: 重生成会从在盘 {what} 移除 {len(missing)} 条字段路径 ⇒ 拒绝写盘 {out_path}：",
        file=sys.stderr,
    )
    for path in missing:
        print(f"  DROPPED [{post_fill_kind(path, post_fill_patterns)}] {path}", file=sys.stderr)
    print(
        "  处置：① 让生成器产出该字段；② post-fill 字段应先在生成器里登记；"
        "③ 确需丢弃 ⇒ 进显式具名例外清单（并写明理由）。",
        file=sys.stderr,
    )
    return missing


def prune_like(obj, like):
    """把 ``obj`` 剪成 ``like`` 的形状：只保留 ``like`` 里也有的 dict 键；列表按位对齐。

    ★ 用途：把"在盘内容"投影到"生成器字段全集"上，再与生成器输出逐值比对。
    形状不一致（如某键一边是标量一边是对象、或列表长度不同）时**原样返回**，让比较自然失败。
    """
    if isinstance(obj, dict) and isinstance(like, dict):
        return {key: prune_like(value, like[key]) for key, value in obj.items() if key in like}
    if isinstance(obj, list) and isinstance(like, list) and len(obj) == len(like):
        return [prune_like(a, b) for a, b in zip(obj, like)]
    return obj


def apply_like_at(node, like, path: str, patterns):
    """把 ``like`` 在命中 ``patterns`` 的路径处的值覆盖到 ``node`` 上（列表按位对齐）。

    ★ 用途：``/tiers[]/status`` 这类字段生成器**也产出**，但在盘的值是**后填**的
    （生成器算 ready/unverified，在盘已回填 judged/blocked/running）⇒ 比对时以生成值为准，
    但仍要求它们**是同一个字段**（存在性/形状由 ``prune_like`` 与字段全集断言兜住）。
    """
    if matches_any(path, patterns):
        return copy.deepcopy(like)
    if isinstance(node, dict) and isinstance(like, dict):
        return {
            key: apply_like_at(value, like.get(key), f"{path}/{key}", patterns)
            for key, value in node.items()
        }
    if isinstance(node, list) and isinstance(like, list) and len(node) == len(like):
        return [apply_like_at(a, b, path + "[]", patterns) for a, b in zip(node, like)]
    return node
