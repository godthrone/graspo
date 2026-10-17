#!/usr/bin/env python3
"""③1 卡池口径扫查常设机核 —— 已退役卡池口径痕迹的**自动枚举**扫描。

## 现行口径（唯一标尺）

★★ 主席 **2026-10-17** 裁定，逐字：

> 「**228 先到先得，随便用。**」
> 另：「**单个作业不限制卡数。**」

⇒ 卡号判据只剩**存在性**：本机全卡 **GPU0–7 一律可用**。
   ⛔ 不预留、⛔ 不禁用、⛔ 无「生产锁卡」、⛔ 无「邻居卡」、⛔ 无单作业卡数上限。
   ★ 唯一保留的硬要求 =「**不抢占他人**」：起跑前逐卡实测空闲
     （**剩余显存 ≥ 本档头寸 ∧ util == 0**），被占则换卡或等，
     ⛔ 绝不 kill 他人进程 / 绝不停他人容器。

## 为什么必须有它（立项目的保持）

收口过程中"还剩多少处旧契约引用"被人工枚举了**三次仍不全** ⇒ **枚举必须机器生成，不能手写。**
同一件事在 2026-10-17 又发生了一次：口径已由主席裁定换代，而旧口径仍以**常量 / 判据 / 词表**
的形式活在生成器、测试与文档里 —— 本机核把它固化成**能失败**的断言。

## 它查什么（对 `tests` / `src/graspo` / `scripts` / `tools` 下全部 `.py`）

1. **退役符号的"真实代码引用"**（**AST**）：`GPU_FORBIDDEN_PRODUCTION` / `NEIGHBOR_CARDS` /
   `LOCKED_CARDS` / `CHAIRMAN_BANNED_CARDS` / `legal_card_set` / `locked_card_violation` /
   `non_legal_card_violation` / `MAX_CARDS_PER_JOB` / `production_card`
   —— 这些**概念已废除**（"生产锁卡 / 邻居卡 / 主席禁用卡 / 单作业卡数上限"），
   其**定义或调用**一旦复活即 FAIL。
   ★ 用 AST 而非行级正则：**注释 / 字符串里提及**它们（写"XX 已废"）**不算**代码引用，
   故不必为"解释退役"的散文开豁免口子（旧版的符号 leg 会在自己的退役说明上自撞）。
2. **卡数上限的字面量复活**（AST）：`MAX_CARDS` / `MAX_CARDS_PER_JOB` 被赋成**整数常量**
   （`= 4`）即 FAIL —— 现行 `MAX_CARDS` 是**派生量**（``len(ALLOWED_INDICES)``，语义 =
   本机卡数），⛔ 不是"每个作业最多几张"的阈值。
3. **旧允许集字面量**（AST）：任何等于 ``(0,1,2,3,4,5)`` 的 tuple/set/list，
   或卡位语境下的 ``range(0, 6)`` —— 旧"只允许到 GPU5"口径的产物。
4. **退役词**（行级，需退役声明豁免）：``生产卡`` / ``保留卡`` / ``production card`` /
   ``reserved card`` / ``生产 vLLM``。
5. **旧空闲判据**（行级，需退役声明豁免）：``memory.used <= N MiB`` / ``used > 64 MiB`` /
   ``utilization.gpu <= 5`` / ``util > 5%`` 这类**绝对占用阈值** —— 现行判据是
   **「剩余 ≥ 本档头寸 ∧ util == 0」**（B10）。这是 2026-10-17 机算发现的一处**真实注释漂移**
   （`cli/gpu_monitor.py`、`run_matrix54.sh` 的生成端）⇒ 固化成判据。
6. **``RESERVED*`` 名单当判据**（AST）：成员测试（``x in RESERVED_INDICES``）或与非空值比较
   ⇒ 退役语义复活。放行"退役断言"（``RESERVED_INDICES == ()``，那是新契约的证明）。

**命中即 FAIL 并逐条列出位置。**

## 阳性对照

``test_positive_control_*`` 把违规样本喂给**同一个** ``scan_sources()``，必须被抓到 ——
即"该 FAIL 时确实 FAIL"，否则扫描器本身失效而无人知。

自排除：本文件是**模式的定义处**（模式字面量就在源码里），故扫描时排除自身与其姊妹机核文件；
这是唯一豁免，理由与位置在此写明。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]

#: 扫描根（相对 REPO）：测试树 + 必要的 src + 宿主侧脚本/工具（卡池判据在这些地方也会成形）。
SCAN_ROOTS = ("tests", "src/graspo", "scripts", "tools")

#: 扫描时排除的文件：模式的定义处（含模式字面量，扫自己必然自命中）。
SELF_EXCLUDED = {
    "tests/compliance/test_gpu_contract_retired_sweep.py",
    "tests/compliance/test_gpu_allowlist_single_source.py",
    "tests/compliance/test_gpu_caliber_consistency.py",
}

#: ★ 已退役的卡池符号（**真实代码引用**即命中；注释/字符串里提及不算）。
#: ⛔ `GPU_ALLOWED` 已从本表移除：现行口径下它是**合法**名字（= 本机全卡 0–7）；
#: 它若被改回旧集合，由第 3 条「旧允许集字面量」抓住。
RETIRED_SYMBOLS: frozenset[str] = frozenset({
    "GPU_FORBIDDEN_PRODUCTION",
    "NEIGHBOR_CARDS",
    "LOCKED_CARDS",
    "CHAIRMAN_BANNED_CARDS",
    "legal_card_set",
    "locked_card_violation",
    "non_legal_card_violation",
    "MAX_CARDS_PER_JOB",
    "production_card",
})

#: 卡数上限常量名：被赋成**整数字面量**即命中（现行必须是派生量）。
CARD_COUNT_CONSTANTS: frozenset[str] = frozenset({"MAX_CARDS", "MAX_CARDS_PER_JOB"})

#: "退役声明"标记：命中行 ±3 行内出现任一标记 ⇒ 视为**新契约的正确表述**，不算命中。
RETIREMENT_MARKERS = (
    "退役", "retired", "不再", "不属", "已退出", "Exited", "失效",
    "不是", "历史", "旧用例", "已废", "作废", "废除", "旧口径", "旧判据", "改为",
)

#: 旧契约用词（需退役声明豁免）。
_RE_RETIRED_WORD = re.compile(r"生产卡|保留卡|production[_ ]card|reserved[_ ]card", re.IGNORECASE)
#: 旧"生产 vLLM"语义（需退役声明豁免）。
_RE_PRODUCTION_VLLM = re.compile(r"生产\s*vLLM", re.IGNORECASE)
#: 旧**绝对占用阈值**形式的空闲判据（需退役声明豁免）。★ 现行 =「剩余 ≥ 头寸 ∧ util == 0」。
_RE_OLD_IDLE_CRITERION = re.compile(
    r"(memory\.used\s*(<=|<|≤)\s*\d+\s*MiB)"
    r"|(used\s*(>|>=|≥)\s*64\s*MiB)"
    r"|(utilization\.gpu\s*(<=|<|≤)\s*5)"
    r"|(util\s*(>|>=)\s*5\s*%)"
    r"|(util\s*(<=|<|≤)\s*5\s*%)",
    re.IGNORECASE,
)
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


def _retired_symbol_refs(source: str) -> dict[int, str]:
    """AST：退役符号的**真实代码引用**（Name / Attribute / 定义名 / 形参）。

    ⛔ 注释与字符串里的**提及**不算（那正是"写明它已退役"的地方）。
    """
    found: dict[int, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        name = None
        kind = ""
        if isinstance(node, ast.Name) and node.id in RETIRED_SYMBOLS:
            name, kind = node.id, "引用"
        elif isinstance(node, ast.Attribute) and node.attr in RETIRED_SYMBOLS:
            name, kind = node.attr, "属性引用"
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in RETIRED_SYMBOLS:
            name, kind = node.name, "定义"
        elif isinstance(node, ast.arg) and node.arg in RETIRED_SYMBOLS:
            name, kind = node.arg, "形参"
        if name is not None:
            found[node.lineno] = f"退役符号（{kind}）{name}"
    return found


def _card_count_cap_literals(source: str) -> dict[int, str]:
    """AST：`MAX_CARDS` / `MAX_CARDS_PER_JOB` 被赋成**整数字面量** ⇒ 卡数上限复活。"""
    found: dict[int, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        if value is None:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in CARD_COUNT_CONSTANTS:
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    found[node.lineno] = (
                        f"卡数上限字面量复活：{t.id} = {value.value}"
                        f"（现行必须是派生量，语义 = 本机卡数）"
                    )
    return found


#: "允许集"这**一个量**的名字特征 —— 旧字面量只有落在它身上才算"旧允许集"。
#: ★ 为什么加这一层上下文（2026-10-17）：同一条 `(0,1,2,3,4,5)` 在**测试期望值**里
#:   是"6 张卡的设备元组"（新口径下合法的输入/输出），在 `GPU_ALLOWED = ...` 里才是
#:   "只允许到 GPU5"的旧口径。不加区分会把前者误报（实测：本包改写后的
#:   `tests/core/test_gpu_guard.py` 两处 `assert_gpu_lock("0,1,2,3,4,5") == (...)`）。
_ALLOWLIST_NAME_RX = re.compile(r"ALLOWED|allowlist|白名单|允许集")


def _seq_literal(node: ast.AST) -> "tuple | None":
    """取"字面量序列"的元组值；非字面量（含含名字的表达式）⇒ None。"""
    if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        if isinstance(node, ast.Set) and not node.elts:
            return None
        try:
            return tuple(ast.literal_eval(node))
        except (ValueError, SyntaxError, TypeError, MemoryError):
            return None
    return None


def _is_allowlist_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and bool(_ALLOWLIST_NAME_RX.search(node.id))


def _old_allowlist_literals(source: str, lines: list[str]) -> dict[int, str]:
    """AST：找出**旧允许集** ``(0,1,2,3,4,5)`` 的两种形态。

    ① 允许集常量被**赋成**该字面量（``GPU_ALLOWED = (0,1,2,3,4,5)``）；
    ② 该字面量**与**允许集常量**比较**（``assert GPU_ALLOWED == (0,1,2,3,4,5)``）；
    ③ 卡位语境下的 ``range(0, 6)`` / ``range(6)``。

    ★ 上下文限定（①/②）是**判据变准**而非变松：它专抓"允许集合这一个量"，
      不再把"6 张卡的设备元组期望值"误报成旧口径。
    """
    found: dict[int, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        # ① 赋值：允许集常量 = 旧字面量
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            val = _seq_literal(node.value) if node.value is not None else None
            if val == _OLD_ALLOWED_SEQ:
                for t in targets:
                    if _is_allowlist_name(t):
                        found[node.lineno] = (
                            f"旧允许集字面量 {val}（只允许到 GPU5 的口径）"
                        )
        # ② 比较：允许集常量 与 旧字面量
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if _OLD_ALLOWED_SEQ in [_seq_literal(o) for o in operands] and any(
                _is_allowlist_name(o) for o in operands
            ):
                found[node.lineno] = f"与允许集常量比较的旧字面量 {_OLD_ALLOWED_SEQ}"
        # ③ range
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

    放行"退役断言"——与真相源常量"恒为空元组"的事实对照（``RESERVED_INDICES == ()``），
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
    """扫描 ``{relpath: source}``，返回命中清单（AST + 行级）。纯函数，供阳性对照复用。"""
    hits: list[Hit] = []
    for rel, source in sorted(sources.items()):
        lines = source.splitlines()

        def _line(lineno: int) -> str:
            return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""

        for i, line in enumerate(lines):
            if _RE_RETIRED_WORD.search(line) and not _has_retirement_context(lines, i):
                hits.append(Hit(rel, i + 1, "retired_word", line.strip()))
            if _RE_PRODUCTION_VLLM.search(line) and not _has_retirement_context(lines, i):
                hits.append(Hit(rel, i + 1, "production_vllm", line.strip()))
            if _RE_OLD_IDLE_CRITERION.search(line) and not _has_retirement_context(lines, i):
                hits.append(Hit(rel, i + 1, "old_idle_criterion", line.strip()))
        for lineno, desc in _retired_symbol_refs(source).items():
            hits.append(Hit(rel, lineno, "retired_symbol_ref", f"{desc} :: {_line(lineno)}"))
        for lineno, desc in _card_count_cap_literals(source).items():
            hits.append(Hit(rel, lineno, "card_count_cap_literal", f"{desc} :: {_line(lineno)}"))
        for lineno, desc in _old_allowlist_literals(source, lines).items():
            hits.append(Hit(rel, lineno, "old_allowlist_literal", f"{desc} :: {_line(lineno)}"))
        for lineno, desc in _reserved_judgements(source).items():
            hits.append(Hit(rel, lineno, "reserved_judgement", f"{desc} :: {_line(lineno)}"))
    return hits


#: ★ 已裁定登记的**冻结 v1 e2e 子系统**（B7 §3「登记不动」）。
#: ★ 2026-10-17：v1 e2e 链路（`gpu_assignment.py` / `generate_matrix.py` /
#: `test_generate_matrix.py`）**已随卡池口径一起对齐**（`GPU_ALLOWED=(0..7)`、无生产锁卡、
#: 无卡数上限）⇒ 本登记表**清空**。空表由 ``test_baseline_is_empty_and_must_stay_empty``
#: 显式钉住 —— 一旦有人为了"让红变绿"往表里塞条目，那条断言会先红，逼他走"改口径 /
#: 说明理由"的正路（而不是靠白名单把红压成绿）。
REGISTERED_LEGACY: dict[str, dict[str, int]] = {}


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


def test_real_tree_has_no_retired_card_pool_traces() -> None:
    """★ 真树必须**零命中**：旧口径（生产锁卡 / 邻居卡 / 卡数上限 / 旧空闲阈值）一处不留。"""
    sources = dict(_iter_py_sources())
    # ★ 扫描器"没死"的守卫：不靠"必须有命中"（真树干净时那条会误报），
    #   而靠"确实读到了足量文件" + 下面六条阳性对照。
    assert len(sources) >= 100, f"扫描到的 .py 文件数异常偏少（{len(sources)}）⇒ 扫描根失效"
    hits = scan_sources(sources)
    offenders = diff_against_baseline(_tally(hits))
    if offenders:
        detail = "\n".join("  " + _describe(h) for h in hits)
        raise AssertionError(
            "★ 发现【已退役卡池口径】痕迹（命中即 FAIL）——\n"
            "  现行口径 = 主席 2026-10-17：「228 先到先得，随便用。」＋「单个作业不限制卡数。」\n"
            "  ⇒ 本机全卡 0–7 一律可用；⛔ 无预留/禁用/生产锁卡/邻居卡、⛔ 无卡数上限；\n"
            "     唯一硬要求 =「不抢占他人」（起跑前逐卡实测：剩余 ≥ 头寸 ∧ util == 0）。\n"
            "  要么按现行口径改写，要么说明理由后登记进 REGISTERED_LEGACY：\n"
            + "\n".join("  - " + o for o in offenders)
            + "\n\n全部命中（含已登记）：\n" + detail
        )


def test_baseline_is_empty_and_must_stay_empty() -> None:
    """★ 登记表必须为空：为"让红变绿"往白名单里塞条目是**禁止**路径（防白名单漂移）。"""
    assert REGISTERED_LEGACY == {}, (
        f"REGISTERED_LEGACY 非空（{sorted(REGISTERED_LEGACY)}）⇒ 必须走『改口径』的正路，"
        f"⛔ 不得用登记表压红"
    )


def test_positive_control_retired_word_is_caught() -> None:
    """阳性对照：注入一处旧契约用词（无退役声明）⇒ 必须被抓。"""
    injected = {"src/graspo/fake.py": 'x = "GPU6/7 生产卡就在这堆卡里，必须拒绝。\\n"\n'}
    hits = scan_sources(injected)
    assert [h.pattern for h in hits] == ["retired_word"], hits


def test_positive_control_retired_symbol_is_caught() -> None:
    """阳性对照：注入旧契约具名符号的**真实代码引用** ⇒ 必须被抓。"""
    injected = {"tests/fake/test_x.py": (
        "from gpu_assignment import GPU_FORBIDDEN_PRODUCTION\n"
        "LOCKED_CARDS = GPU_FORBIDDEN_PRODUCTION + (2,)\n"
    )}
    kinds = sorted({h.pattern for h in scan_sources(injected)})
    assert "retired_symbol_ref" in kinds, kinds


def test_negative_control_retired_symbol_in_prose_is_not_caught() -> None:
    """阴性对照：**注释/字符串里提及**退役符号（写"XX 已废"）不算代码引用，不得误报。"""
    ok = {"src/graspo/gpu_guard_like.py": (
        "# ★ `GPU_FORBIDDEN_PRODUCTION` / `NEIGHBOR_CARDS` / `LOCKED_CARDS` 均已废除。\n"
        'MSG = "old NEIGHBOR_CARDS / legal_card_set() are gone"\n'
    )}
    assert scan_sources(ok) == []


def test_positive_control_card_count_cap_literal_is_caught() -> None:
    """阳性对照：把单作业卡数上限写成字面量 ⇒ 必须被抓（现行必须是派生量）。"""
    injected = {"src/graspo/core/gpu_guard_fake.py": "MAX_CARDS = 4\n"}
    hits = scan_sources(injected)
    assert [h.pattern for h in hits] == ["card_count_cap_literal"], hits


def test_positive_control_old_allowlist_literal_is_caught() -> None:
    """阳性对照：注入旧允许集字面量（AST 判定，含卡位语境）⇒ 必须被抓。"""
    by_range = {"tests/fake/test_y.py": "ALLOWED_GPU_IDS = tuple(range(0, 6))  # 允许卡\n"}
    assert [h.pattern for h in scan_sources(by_range)] == ["old_allowlist_literal"], by_range
    by_literal = {"tests/fake/test_z.py": "GPU_ALLOWED_IDS = (0, 1, 2, 3, 4, 5)  # 允许卡\n"}
    assert [h.pattern for h in scan_sources(by_literal)] == ["old_allowlist_literal"], by_literal
    by_compare = {"tests/fake/test_w.py": (
        "assert GPU_ALLOWED == (0, 1, 2, 3, 4, 5)\n"
    )}
    assert [h.pattern for h in scan_sources(by_compare)] == ["old_allowlist_literal"], by_compare


def test_negative_control_device_tuple_expected_value_is_not_caught() -> None:
    """阴性对照：`(0,1,2,3,4,5)` 作为**设备元组期望值**（不是"允许集"这一个量）不得误报。"""
    ok = {"tests/core/fake_guard_test.py": (
        'assert assert_gpu_lock("0,1,2,3,4,5") == (0, 1, 2, 3, 4, 5)\n'
        "EXPECTED_SIX_DEVICES = (0, 1, 2, 3, 4, 5)\n"
    )}
    assert scan_sources(ok) == []


def test_positive_control_old_idle_criterion_is_caught() -> None:
    """阳性对照：把旧空闲阈值（绝对占用）塞回 ⇒ 必须被抓（现行 = 剩余 ≥ 头寸 ∧ util==0）。"""
    injected = {
        "src/graspo/cli/gpu_monitor_fake.py": 'HELP = "assert idle (memory.used <= 64 MiB)"\n'
    }
    assert [h.pattern for h in scan_sources(injected)] == ["old_idle_criterion"], injected


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
        "# ★「生产卡 / reserved」语义与「单作业卡数上限」均已**作废**（2026-10-17 主席令）。\n"
        "#   现行空闲判据 = 剩余 ≥ 头寸 ∧ util == 0（旧的 memory.used <= 64 MiB 已废）。\n"
    )}
    assert scan_sources(ok) == []
