#!/usr/bin/env python3
"""③2 允许集"单一真相源"机核 —— 不允许存在**第二处定义**允许卡集合。

## 背景

B7 收口时修好了一处真实漂移：``src/graspo/eval/guard.py`` 的
``ALLOWED_GPU_INDICES`` 曾是 ``frozenset(range(ALLOWED_MAX_INDEX + 1))``（= ``{0..7}``，
把 0–3 也算成允许卡），与真相源的正向白名单冲突（宪法 §1.4）。
⇒ 允许集合这**一个量**必须有唯一家：``src/graspo/core/gpu_guard.py::ALLOWED_INDICES``。

## 它查什么（全仓 Python）

1. **第二处定义**：任何名字形如允许集常量的赋值（``ALLOWED*INDICES*`` /
   ``ALLOWED_GPU*`` / ``GPU_ALLOWED*`` / ``GPU_FORBIDDEN*`` / ``RESERVED*INDICES*`` /
   ``ALLOWED_MAX_INDEX``），必须满足二者之一：
   · 就是真相源本身（``core/gpu_guard.py`` 的 ``ALLOWED_INDICES`` / ``RESERVED_INDICES``）；
   · **纯派生/别名**——取值里引用 ``ALLOWED_INDICES`` / ``RESERVED_INDICES``
     （如 ``frozenset(ALLOWED_INDICES)`` / ``max(ALLOWED_INDICES)`` / ``= RESERVED_INDICES``）。
   取值是字面量或不引用真相源符号的派生 ⇒ **FAIL**（这正是历史漂移的形态）。
2. **第二个字面量定义**：全仓任何赋值（不限名字）的字面量取值若等于真相源的允许集合
   （或 ``frozenset/tuple/set/list(...)`` 包一层）⇒ **FAIL**。真相源那一处除外。

判据的期望值是**运行时从真相源 import 的**（``ALLOWED_INDICES``），不在本文件另抄一份。

## 边界说明（不查什么）

* 散文/消息/注释里**提及** ``{4,5,6,7}`` 不算"定义"（否则真相源的用法提示会被自己判死）；
* 逐卡数的**卡池分配**（如 ``GPU_POOL_BY_CARDS = {4: [4,5,6,7]}``）是另一个量
  （"这一档用哪几张卡"≠"哪些卡允许"），不在本机核范围。

## 阳性对照

``test_positive_control_*`` 用**同一个** ``analyze_sources()`` 注入历史漂移形态与字面量
重定义，必须被抓；``test_negative_control_*`` 证明纯派生别名不被误报。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]

#: 允许集合的**唯一真相源**。
CANONICAL_PATH = "src/graspo/core/gpu_guard.py"
CANONICAL_NAME = "ALLOWED_INDICES"
#: 真相源允许集合内**同类**常量（同为真相源文件内的规范定义）。
CANONICAL_EXTRA_OK = {"RESERVED_INDICES"}

#: 全仓扫描根（"全仓"= 在役 Python 面：src / tests / scripts / tools）。
SCAN_ROOTS = ("src", "tests", "scripts", "tools")

#: 形如"允许集常量"的名字。`_ENV` 结尾的是环境变量**名字**（字符串），不是集合定义。
_ALLOWLIST_NAME_RX = re.compile(
    r"^(ALLOWED.*INDICES.*|ALLOWED_GPU.*|GPU_ALLOWED.*|GPU_FORBIDDEN.*|RESERVED.*INDICES.*|ALLOWED_MAX_INDEX)$"
)
#: 派生/别名取值里必须引用到的真相源符号。
_SOURCE_SYMBOLS = frozenset({"ALLOWED_INDICES", "RESERVED_INDICES"})
_SEQ_FACTORIES = frozenset({"frozenset", "tuple", "set", "list"})


class Problem(NamedTuple):
    path: str
    lineno: int
    kind: str
    detail: str


def canonical_allowed_indices() -> tuple[int, ...]:
    """从真相源读取允许集合（⛔ 不在本文件另抄一份）。"""
    from graspo.core.gpu_guard import ALLOWED_INDICES

    return tuple(ALLOWED_INDICES)


def _iter_py_sources(repo: Path = REPO, roots: tuple[str, ...] = SCAN_ROOTS):
    for root in roots:
        base = repo / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path.relative_to(repo).as_posix(), path.read_text(encoding="utf-8", errors="replace")


def _literal_seq(node: ast.AST) -> tuple | None:
    """取"字面量序列"的元组值；支持 ``frozenset/tuple/set/list(字面量)`` 包一层。"""
    if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        if isinstance(node, ast.Set) and not node.elts:
            return None
        try:
            return tuple(ast.literal_eval(node))
        except (ValueError, SyntaxError, TypeError, MemoryError):
            return None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _SEQ_FACTORIES and len(node.args) == 1):
        return _literal_seq(node.args[0])
    return None


def analyze_sources(sources: dict[str, str], allowed: tuple[int, ...],
                    canonical_path: str = CANONICAL_PATH) -> list[Problem]:
    """扫描 ``{relpath: source}``，返回"第二处定义"问题清单。纯函数，供阳性对照复用。"""
    problems: list[Problem] = []
    for rel, source in sorted(sources.items()):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            if value is None:
                continue
            names = sorted({n.id for n in ast.walk(value) if isinstance(n, ast.Name)})
            text = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                name = t.id
                # 规则 2：任何名字的赋值，字面量取值 == 真相源允许集合（真相源那一处除外）
                seq = _literal_seq(value)
                if seq is not None and seq == allowed and not (rel == canonical_path and name == CANONICAL_NAME):
                    problems.append(Problem(
                        rel, node.lineno, "literal_definition",
                        f"{name} = {seq!r}（允许集的第二个字面量定义）:: {text}"))
                # 规则 1：允许集常量名必须"是真相源"或"纯派生自真相源"
                if not _ALLOWLIST_NAME_RX.match(name) or name.endswith("_ENV"):
                    continue
                if rel == canonical_path and name in ({CANONICAL_NAME} | CANONICAL_EXTRA_OK):
                    continue
                if set(names) & _SOURCE_SYMBOLS:
                    continue
                problems.append(Problem(
                    rel, node.lineno, "second_definition",
                    f"{name} 未引用真相源 {sorted(_SOURCE_SYMBOLS)}（取值引用 {names}）:: {text}"))
    return problems


#: 已裁定登记的**例外**（用到即打印）。
#: ★ 2026-10-17（卡池口径对齐）：`tests/e2e/gpu_assignment.py` 是**stdlib-only** 模块
#:   （生成器与运行期都要在**裸解释器**下可用，⛔ 不得 import `graspo.core.gpu_guard`）
#:   ⇒ 它的 `GPU_ALLOWED` 只能是**本机推导量** `tuple(range(GPU_MAX_INDEX + 1))`，
#:   **结构上**不可能引用真相源符号。这不是"第二处真相源"，而是**唯一不得不重复的一份派生定义**；
#:   它由 `tests/compliance/test_gpu_caliber_consistency.py` 与真相源**逐值交叉断言**，
#:   并由 `test_gpu_contract_retired_sweep.py` 保证"卡数上限字面量/旧允许集字面量"不复活。
#:   ★ 计数由 2 降为 1：旧口径的 `GPU_FORBIDDEN_PRODUCTION`（第二处定义之一）**已删除**。
REGISTERED_LEGACY: dict[str, dict[str, int]] = {
    "tests/e2e/gpu_assignment.py": {"second_definition": 1},
}


def _tally(problems: list[Problem]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for p in problems:
        out.setdefault(p.path, {})
        out[p.path][p.kind] = out[p.path].get(p.kind, 0) + 1
    return out


def diff_against_baseline(tally: dict[str, dict[str, int]],
                          baseline: dict[str, dict[str, int]] = REGISTERED_LEGACY) -> list[str]:
    problems: list[str] = []
    for path, kinds in sorted(tally.items()):
        reg = baseline.get(path, {})
        for kind, n in sorted(kinds.items()):
            if reg.get(kind, 0) != n:
                problems.append(f"{path}  [{kind}] 命中 {n} 处，基线登记 {reg.get(kind, 0)} 处")
    for path in sorted(baseline):
        if path not in tally:
            problems.append(f"{path}  基线已登记但实测零命中 ⇒ 请清理基线（防白名单漂移）")
    return problems


def test_allowlist_has_a_single_source_of_truth_in_the_repo() -> None:
    allowed = canonical_allowed_indices()
    assert allowed, "真相源 ALLOWED_INDICES 为空——单一真相源本身不成立"
    problems = analyze_sources(dict(_iter_py_sources()), allowed)
    tally = _tally(problems)
    offenders = diff_against_baseline(tally)
    if offenders:
        detail = "\n".join(f"  {p.path}:{p.lineno}  [{p.kind}]  {p.detail}" for p in problems)
        raise AssertionError(
            f"★ 发现允许集合（真相源 {CANONICAL_PATH}::{CANONICAL_NAME} = {allowed}）的"
            "【第二处定义】（命中即 FAIL）：\n"
            + "\n".join("  - " + o for o in offenders)
            + "\n\n全部命中（含已登记）：\n" + detail
        )
    # 真相源必须确实存在且是"字面量定义"（不允许真相源自己也是派生来的）
    canonical_src = dict(_iter_py_sources())[CANONICAL_PATH]
    assert f"{CANONICAL_NAME}: tuple[int, ...] = (" in canonical_src, (
        f"真相源 {CANONICAL_PATH} 里的 {CANONICAL_NAME} 不是约定的字面量定义——请核对"
    )


def test_positive_control_historical_drift_is_caught() -> None:
    """阳性对照：B7 修掉的那处漂移形态（派生自"上界"而非真相源）必须被抓。"""
    drift = {"src/graspo/eval/new_guard.py":
             "from graspo.core.gpu_guard import ALLOWED_MAX_INDEX\n"
             "ALLOWED_GPU_INDICES = frozenset(range(ALLOWED_MAX_INDEX + 1))\n"}
    problems = analyze_sources(drift, (4, 5, 6, 7))
    assert [(p.kind, p.path) for p in problems] == [("second_definition", "src/graspo/eval/new_guard.py")], problems


def test_positive_control_literal_redefinition_is_caught() -> None:
    """阳性对照：允许集的第二个**字面量**定义必须被抓。"""
    literal = {"src/graspo/flow/guard/pool.py": "FOUR_CARD_SET = (4, 5, 6, 7)\n"}
    problems = analyze_sources(literal, (4, 5, 6, 7))
    assert [p.kind for p in problems] == ["literal_definition"], problems
    wrapped = {"src/graspo/flow/guard/pool.py": "FOUR_CARD_SET = frozenset({4, 5, 6, 7})\n"}
    assert [p.kind for p in analyze_sources(wrapped, (4, 5, 6, 7))] == ["literal_definition"]


def test_positive_control_new_allowlist_reader_is_caught() -> None:
    """阳性对照：另起一份"允许卡"字面量常量（任何名字）必须被抓。"""
    second = {"src/graspo/eval/another_guard.py": "GPU_ALLOWED_IDS = (4, 5, 6, 7)\n"}
    problems = analyze_sources(second, (4, 5, 6, 7))
    kinds = sorted({p.kind for p in problems})
    assert kinds == ["literal_definition", "second_definition"], problems


def test_negative_control_pure_derivation_is_not_flagged() -> None:
    """阴性对照：纯派生/别名（引用真相源符号）不得被误报。"""
    ok = {"src/graspo/eval/guard_like.py": (
        "from graspo.core.gpu_guard import ALLOWED_INDICES, RESERVED_INDICES\n"
        "ALLOWED_GPU_INDICES = frozenset(ALLOWED_INDICES)\n"
        "RESERVED_GPU_INDICES = RESERVED_INDICES\n"
        "ALLOWED_MAX_INDEX = max(ALLOWED_INDICES)\n"
        "ALLOWED_INDICES_ENV = 'GRASPO_ALLOWED_GPU_INDICES'\n"
    )}
    assert analyze_sources(ok, (4, 5, 6, 7)) == []
