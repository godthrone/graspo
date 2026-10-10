#!/usr/bin/env python3
"""③1 契约扫查常设机核 —— 已退役卡位契约痕迹的**自动枚举**扫描。

## 为什么要有它（不可省略的立项目的）

B7 把「保留卡 / 生产卡」语义正式退役，卡位判据只剩**一个**：允许集合
``{4,5,6,7}``，其余一律拒绝（``src/graspo/core/gpu_guard.py``，§1.4 单一真相源）。
但收口过程中"还剩多少处旧契约引用"被人工枚举了**三次仍不全**（15 → 24 → 27 条 RED：
指挥部点名 15 + B6 裁定并入 9 + 扫查才发现 3）。⇒ **枚举必须机器生成，不能手写。**

## 它查什么

对 ``tests/`` 与 ``src/graspo/``（"测试树与必要的 src"）下全部 ``.py`` 做**行级 + AST** 扫描：

1. **退役词**：``生产卡`` / ``保留卡`` / ``production card`` / ``reserved card`` / ``生产 vLLM``
   —— 但**排除"退役声明"本身**（同一处上下文含 ``退役``/``retired``/``不再``/``不属``/
   ``Exited``/``失效`` 等标记时，视为新契约的正确表述，不算命中；真相源
   ``core/gpu_guard.py`` 的退役注释正在此列）。
2. **退役符号**：``GPU_ALLOWED`` / ``GPU_FORBIDDEN_PRODUCTION`` / ``production_card``
   （旧契约的**具名定义**，不带退役声明就必须消灭）。
3. **旧允许集字面量**（AST）：任何等于 ``(0,1,2,3,4,5)`` 的 tuple/set/list，或
   卡位语境下 ``range(0, 6)`` —— 旧"仅上限"口径的产物。
4. **``RESERVED*`` 名单当判据**（AST）：成员测试（``x in RESERVED_INDICES``）或与非空值
   比较 ⇒ 退役语义复活。放行"退役断言"（``RESERVED_INDICES == ()``，那是新契约的证明）。

**命中即 FAIL 并逐条列出位置。** 唯一例外是 ``REGISTERED_LEGACY`` 里显式登记、
已由 B7 §3 裁定"登记不动"的 v1 e2e 冻结子系统（改动它会波及冻结的 v1 清单
``tests/e2e/matrix54_manifest.json``，超出本批范围）。基线**自身也机器校验**：
登记条目必须仍能命中，否则 FAIL —— 防"白名单漂移"变成新的盲区。

## 阳性对照

``test_positive_control_*`` 把违规样本喂给**同一个** ``scan_sources()``，必须被抓到。
即"该 FAIL 时确实 FAIL"——否则扫描器本身失效而无人知。

自排除：本文件是**模式的定义处**（模式字面量就在源码里），故扫描时排除自身与其姊妹
机核文件；这是唯一豁免，理由与位置在此写明。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]

#: 扫描根（相对 REPO）：测试树 + 必要的 src（卡位契约实际生效面）。
SCAN_ROOTS = ("tests", "src/graspo")

#: 扫描时排除的文件：模式的定义处（含模式字面量，扫自己必然自命中）。
SELF_EXCLUDED = {
    "tests/compliance/test_gpu_contract_retired_sweep.py",
    "tests/compliance/test_gpu_allowlist_single_source.py",
}

#: "退役声明"标记：命中行 ±3 行内出现任一标记 ⇒ 视为**新契约的正确表述**，不算命中。
RETIREMENT_MARKERS = (
    "退役", "retired", "不再", "不属", "已退出", "Exited", "失效",
    "不是", "历史", "旧用例",
)

#: 旧契约用词（需退役声明豁免）。
_RE_RETIRED_WORD = re.compile(r"生产卡|保留卡|production[_ ]card|reserved[_ ]card", re.IGNORECASE)
#: 旧契约具名符号（不豁免）。
_RE_RETIRED_SYMBOL = re.compile(r"\b(GPU_ALLOWED|GPU_FORBIDDEN_PRODUCTION|production_card)\b")
#: 旧"生产 vLLM"语义（需退役声明豁免）。
_RE_PRODUCTION_VLLM = re.compile(r"生产\s*vLLM", re.IGNORECASE)
#: 卡位上下文（`range(0,6)` 这类**通用**写法只有在卡位语境里才算旧允许集痕迹）。
_RE_GPU_CONTEXT = re.compile(r"GPU|gpu|卡|ALLOWED|INDICES|INDEX|设备")

#: 旧允许集字面量（AST 判定）。
_OLD_ALLOWED_SEQ = (0, 1, 2, 3, 4, 5)


class Hit(NamedTuple):
    path: str
    lineno: int
    pattern: str
    line: str


def _iter_py_sources(repo: Path = REPO, roots: tuple[str, ...] = SCAN_ROOTS):
    for root in roots:
        base = repo / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo).as_posix()
            if rel in SELF_EXCLUDED:
                continue
            yield rel, path.read_text(encoding="utf-8", errors="replace")


def _has_retirement_context(lines: list[str], idx0: int) -> bool:
    lo, hi = max(0, idx0 - 3), min(len(lines), idx0 + 4)
    window = "\n".join(lines[lo:hi])
    return any(mark in window for mark in RETIREMENT_MARKERS)


def _old_allowlist_literals(source: str, lines: list[str]) -> dict[int, str]:
    """AST：找出等于旧允许集 ``(0,1,2,3,4,5)`` 的序列字面量 / 卡位语境下的 ``range(0,6)``。"""
    found: dict[int, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.Set, ast.List)) and not isinstance(node, ast.Starred):
            if isinstance(node, ast.Set) and not node.elts:
                continue
            try:
                val = tuple(ast.literal_eval(node))
            except (ValueError, SyntaxError, TypeError, MemoryError):
                continue
            if val == _OLD_ALLOWED_SEQ:
                found[node.lineno] = f"旧允许集字面量 {val}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            try:
                if node.keywords:
                    raise ValueError("range 带关键字参数")
                args = tuple(ast.literal_eval(a) for a in node.args)
                rng = tuple(range(*args))
            except (ValueError, SyntaxError, TypeError, MemoryError):
                continue
            # `range(6)` = 0..5 是**通用**写法（如"6 个 rollout 组"）⇒ 只在卡位语境里才判为痕迹。
            ctx = "\n".join(lines[max(0, node.lineno - 3):node.lineno + 2])
            if rng == _OLD_ALLOWED_SEQ and _RE_GPU_CONTEXT.search(ctx):
                found[node.lineno] = f"旧允许集 range{args}"
    return found


def _reserved_judgements(source: str) -> dict[int, str]:
    """AST：``RESERVED*`` 名单被当**判据**用（成员测试 / 与非空值比较）⇒ 退役语义复活。

    放行"退役断言"——与本文件常量"恒为空元组"的事实对照（``RESERVED_INDICES == ()``），
    那是新契约的**证明**而非旧语义。
    """
    found: dict[int, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id.startswith("RESERVED")]
        if not names:
            continue

        def _is_empty_literal(expr: ast.AST) -> bool:
            if isinstance(expr, ast.Tuple) and not expr.elts:
                return True
            return (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                    and expr.func.id == "tuple" and not expr.args)

        ops = node.ops
        comps = node.comparators
        if all(isinstance(op, (ast.Eq, ast.Is)) and _is_empty_literal(c) for op, c in zip(ops, comps)):
            continue  # 退役断言：与空元组对照
        found[node.lineno] = f"RESERVED 名单被当判据（退役语义复活）：{ast.unparse(node)[:90]}"
    return found


def scan_sources(sources: dict[str, str]) -> list[Hit]:
    """扫描 ``{relpath: source}``，返回命中清单（行级 + AST）。纯函数，供阳性对照复用。"""
    hits: list[Hit] = []
    for rel, source in sorted(sources.items()):
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if _RE_RETIRED_SYMBOL.search(line):
                hits.append(Hit(rel, i + 1, "retired_symbol", line.strip()))
            if _RE_RETIRED_WORD.search(line) and not _has_retirement_context(lines, i):
                hits.append(Hit(rel, i + 1, "retired_word", line.strip()))
            if _RE_PRODUCTION_VLLM.search(line) and not _has_retirement_context(lines, i):
                hits.append(Hit(rel, i + 1, "production_vllm", line.strip()))
        for lineno, desc in _old_allowlist_literals(source, lines).items():
            text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
            hits.append(Hit(rel, lineno, "old_allowlist_literal", f"{desc} :: {text}"))
        for lineno, desc in _reserved_judgements(source).items():
            text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
            hits.append(Hit(rel, lineno, "reserved_judgement", f"{desc} :: {text}"))
    return hits


#: ★ 已裁定登记的**冻结 v1 e2e 子系统**（B7 §3「登记不动」）：
#: 该子系统整体停在旧契约、自洽且全绿；改动会波及冻结的 v1 清单
#: ``tests/e2e/matrix54_manifest.json``（⛔ 必须逐字节不变）⇒ 不在本批范围，
#: 建议另立"v1 链路卡池收口"批。
REGISTERED_LEGACY: dict[str, dict[str, int]] = {
    "tests/e2e/gpu_assignment.py": {"retired_symbol": 5, "retired_word": 2, "production_vllm": 3,
                                    "old_allowlist_literal": 1},
    "tests/e2e/generate_matrix.py": {"retired_symbol": 6, "retired_word": 1, "production_vllm": 3,
                                     "old_allowlist_literal": 1},
    "tests/e2e/test_generate_matrix.py": {"retired_word": 3, "production_vllm": 1,
                                          "old_allowlist_literal": 1},
}


def _tally(hits: list[Hit]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for h in hits:
        out.setdefault(h.path, {})
        out[h.path][h.pattern] = out[h.path].get(h.pattern, 0) + 1
    return out


def _describe(hit: Hit) -> str:
    return f"{hit.path}:{hit.lineno}  [{hit.pattern}]  {hit.line}"


def diff_against_baseline(
    tally: dict[str, dict[str, int]],
    baseline: dict[str, dict[str, int]] = REGISTERED_LEGACY,
) -> list[str]:
    """把实测命中计数与基线比对；返回"漂移"清单（空 = 全登记）。纯函数，供阳性对照复用。"""
    problems: list[str] = []
    for path, pats in sorted(tally.items()):
        reg = baseline.get(path, {})
        for pat, n in sorted(pats.items()):
            if reg.get(pat, 0) != n:
                problems.append(f"{path}  [{pat}] 命中 {n} 处，基线登记 {reg.get(pat, 0)} 处")
    for path in sorted(baseline):
        if path not in tally:
            problems.append(f"{path}  基线已登记但实测零命中 ⇒ 请清理基线（防白名单漂移）")
    return problems


def test_real_tree_has_no_unregistered_retired_contract_traces() -> None:
    hits = scan_sources(dict(_iter_py_sources()))
    tally = _tally(hits)
    offenders = diff_against_baseline(tally)
    if offenders:
        detail = "\n".join("  " + _describe(h) for h in hits)
        raise AssertionError(
            "★ 发现未登记的【已退役卡位契约】痕迹（命中即 FAIL）——\n"
            "  要么按新契约（允许集合 {4,5,6,7}，其余拒绝；「保留/生产卡」语义已退役）改写，\n"
            "  要么说明理由后登记进 REGISTERED_LEGACY：\n"
            + "\n".join("  - " + o for o in offenders)
            + "\n\n全部命中（含已登记）：\n" + detail
        )
    assert tally, "扫描未命中任何文件——扫描根或模式失效，机核自身可疑"


def test_positive_control_retired_word_is_caught() -> None:
    """阳性对照：注入一处旧契约用词（无退役声明）⇒ 必须被抓。"""
    injected = {"src/graspo/fake.py": 'x = "GPU6/7 生产卡就在这堆卡里，必须拒绝。\\n"\n'}
    hits = scan_sources(injected)
    assert [h.pattern for h in hits] == ["retired_word"], hits


def test_positive_control_retired_symbol_is_caught() -> None:
    """阳性对照：注入旧契约具名符号 ⇒ 必须被抓。"""
    injected = {"tests/fake/test_x.py": "GPU_ALLOWED = (0, 1, 2, 3, 4, 5)\n"}
    hits = scan_sources(injected)
    kinds = sorted({h.pattern for h in hits})
    assert "retired_symbol" in kinds and "old_allowlist_literal" in kinds, hits


def test_positive_control_baseline_diff_flags_injected_trace() -> None:
    """阳性对照：把注入的痕迹并入真树计数 ⇒ **同一条基线比对**必须报出漂移。

    这一条走的是 ``test_real_tree_*`` 完全相同的判据路径（``diff_against_baseline``），
    证明"真树+新痕迹"确实会被抓，而不只是扫描函数单独可用。
    """
    real = _tally(scan_sources(dict(_iter_py_sources())))
    assert diff_against_baseline(real) == [], "真树本应全登记"
    polluted = {**real, "src/graspo/flow/guard/new_mod.py": {"retired_word": 1}}
    problems = diff_against_baseline(polluted)
    assert problems and "new_mod.py" in problems[0], problems


def test_positive_control_old_allowlist_literal_is_caught() -> None:
    """阳性对照：注入旧允许集字面量（AST 判定，含卡位语境）⇒ 必须被抓。"""
    by_range = {"tests/fake/test_y.py": "ALLOWED_GPU_IDS = tuple(range(0, 6))  # 允许卡\n"}
    assert [h.pattern for h in scan_sources(by_range)] == ["old_allowlist_literal"], by_range
    by_literal = {"tests/fake/test_z.py": "GPU_IDS = (0, 1, 2, 3, 4, 5)  # 允许卡\n"}
    assert [h.pattern for h in scan_sources(by_literal)] == ["old_allowlist_literal"], by_literal


def test_positive_control_reserved_judgement_is_caught() -> None:
    """阳性对照：把 RESERVED 名单重新当判据用 ⇒ 必须被抓。"""
    by_membership = {"src/graspo/flow/guard/x.py": (
        "from graspo.core.gpu_guard import RESERVED_INDICES\n"
        "def f(card):\n    return card in RESERVED_INDICES\n"
    )}
    assert [h.pattern for h in scan_sources(by_membership)] == ["reserved_judgement"]
    by_compare = {"src/graspo/eval/y.py": (
        "RESERVED_GPU_INDICES = RESERVED_INDICES\n"
        "if RESERVED_GPU_INDICES != (6, 7):\n    pass\n"
    )}
    assert "reserved_judgement" in {h.pattern for h in scan_sources(by_compare)}


def test_negative_control_retired_assertion_is_not_caught() -> None:
    """阴性对照：`RESERVED_INDICES == ()` 这种**退役断言**不得被误报。"""
    ok = {"tests/eval/fake_guard.py": (
        "from graspo.core.gpu_guard import RESERVED_INDICES, RESERVED_GPU_INDICES\n"
        "assert RESERVED_INDICES == ()\n"
        "assert RESERVED_GPU_INDICES == ()\n"
    )}
    assert scan_sources(ok) == []


def test_negative_control_retirement_statement_is_not_caught() -> None:
    """阴性对照：**正确的退役声明**不得被误报（否则真相源注释会被自己判死）。"""
    ok = {"src/graspo/gpu_guard_like.py": (
        "# ★「生产卡 / reserved」语义**已退役**（生产 vLLM 已 Exited 并释放 6/7）。\n"
        "#   · 0–3：第三方工作负载先到先得，**不属**任何形式的「保留卡」；\n"
    )}
    assert scan_sources(ok) == []
