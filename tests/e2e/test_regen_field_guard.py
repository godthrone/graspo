#!/usr/bin/env python3
"""生成链防丢失守卫（``regen_field_guard``）的回归——**断言本身必须能失败**。

背景：同一个病（"重新生成静默丢掉在盘真值"）已在三个独立的地方被撞到：
`build_data --preserve-from` 白名单、清单重生成丢后填字段、runner 修复未回写生成器。
本文件只钉住**判定逻辑**本身：字段路径收集、fail-closed 守卫、以及"投影到生成器字段全集"
的两个纯函数。真实产物的防线在各自生成器 + 各自用例里（见 test_generate_matrix.py 等）。

★ 这里的每个"必须失败"用例都是**阳性对照**：构造一个会被丢字段的输入，实测守卫**真的**报错。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_SPEC = importlib.util.spec_from_file_location(
    "regen_field_guard_ut", _HERE.parent / "regen_field_guard.py"
)
assert _SPEC is not None and _SPEC.loader is not None
guard = importlib.util.module_from_spec(_SPEC)
sys.modules["regen_field_guard_ut"] = guard
_SPEC.loader.exec_module(guard)


def test_field_paths_is_recursive_and_index_agnostic():
    doc = {"a": {"b": 1}, "rows": [{"tier": "T1", "x": {"y": 2}}, {"tier": "T2"}]}
    assert guard.field_paths(doc) == {
        "/a", "/a/b", "/rows", "/rows[]", "/rows[]/tier", "/rows[]/x", "/rows[]/x/y",
    }


def test_guard_fails_closed_when_regeneration_would_drop_a_field(tmp_path, capsys):
    """★阳性对照：新内容少了 1 条字段路径 ⇒ 返回该路径（调用方据此不写盘）。"""
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps({"keep": 1, "lost_field": {"deep": 2}}), encoding="utf-8")
    lost = guard.guard_regeneration_keeps_on_disk_fields(
        {"keep": 1}, out, what="测试清单",
    )
    assert lost == ["/lost_field", "/lost_field/deep"]
    err = capsys.readouterr().err
    assert "FATAL" in err and "DROPPED [UNKNOWN] /lost_field" in err


def test_guard_passes_when_new_is_a_superset(tmp_path):
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps({"keep": 1}), encoding="utf-8")
    assert guard.guard_regeneration_keeps_on_disk_fields({"keep": 1, "added": 2}, out) == []


def test_guard_is_noop_on_first_generation(tmp_path):
    assert guard.guard_regeneration_keeps_on_disk_fields({"a": 1}, tmp_path / "absent.json") == []


def test_guard_explicit_named_exception_is_printed(tmp_path, capsys):
    """例外只来自**显式具名**清单，且用到就打印（⛔ 不是默认放行）。"""
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps({"tier": {"legacy_probe": 1}}), encoding="utf-8")
    lost = guard.guard_regeneration_keeps_on_disk_fields(
        {"tier": {}}, out, allowed_drop=("/tier/legacy_probe",),
    )
    assert lost == []
    assert "ALLOW-DROP（显式具名例外）: /tier/legacy_probe" in capsys.readouterr().out


def test_prune_like_and_apply_like_at_project_on_disk_to_generator_universe():
    regen = {"tiers": [{"status": "ready", "config": "a.yaml"}]}
    on_disk = {"tiers": [{"status": "judged", "config": "a.yaml", "matrix_conclusion": "x"}],
               "extra_top": 1}
    projected = guard.prune_like(on_disk, regen)
    assert projected == {"tiers": [{"status": "judged", "config": "a.yaml"}]}  # 多出的被剪掉
    projected = guard.apply_like_at(projected, regen, "", ("/tiers[]/status",))
    assert projected == regen  # 登记的后填覆盖字段以生成值为准
