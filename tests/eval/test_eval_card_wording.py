"""评测侧卡池口径**自述与事实不得分叉**的常设机核（`src/graspo/eval/**`）。

## 现行口径（唯一标尺）

主席 **2026-10-17** 裁定（已脱敏转述）：「**远端机先到先得，随便用。**」
＋「**单个作业不限制卡数。**」
⇒ 本机全卡 **GPU0–7 一律可用**（⛔ 无预留/禁用/生产锁卡/邻居卡、⛔ **无卡数上限**）；
唯一保留的硬要求 =「**不抢占他人**」（起跑前逐卡实测：剩余 ≥ 本档头寸 ∧ util == 0）。

## 为什么需要这个文件（与姊妹机核的分工）

- `tests/compliance/test_gpu_contract_retired_sweep.py`：全树**退役符号/词表**（AST ＋行级）；
- `tests/compliance/test_gpu_caliber_consistency.py`：**跨件相等 + 判据本身**；
- **本文件**：管**评测侧（`src/graspo/eval/**`）的自述**——这是姊妹机核都不覆盖的一面：
  `eval/guard.py` 里那个 ``MAX_GPU_COUNT: int = MAX_CARDS`` 旧别名**不在** P2 的退役符号表里，
  它本身不构成拦截，却是"这里有个卡数上限"的**误解诱因**（后来者可能据此把"至多四张"接回来）。
  ⛔ 不重抄姊妹机核的规则（`MAX_CARDS = <int>` 这类由 sweep 负责），
  只做"评测侧别名不得复活 ＋ 卡数上限文案不得复活 ＋ 真拦截调用点"这三件事。

## 判据（命中即 FAIL 并给位置）

1. **退役别名复活**（AST，**注释/字符串里提及不算**）：`src/graspo/eval/**` 任何
   `.py` 里出现名为 ``MAX_GPU_COUNT`` 的**代码引用**（赋值/读取/导入/`__all__` 条目）。
2. **卡数上限文案复活**（行级，具名旧字面量）：如 ``卡数 ≤ 4`` / ``最多 4 卡`` /
   ``4-card cap``。★ 只列**已作废的确切表述**，不列"卡数上限"这种现行说明里也要用的词。
3. **真拦截调用点**（AST）：评测侧任何调用传了 ``max_cards=<整数>`` ⇒ 那是"会在运行期
   真的拒绝 >N 卡"的上限（`core.gpu_guard.assert_gpu_lock` 的显式自限入口），必须 FAIL。
4. **必须写出新口径**（缺一即 FAIL）：`eval/guard.py` 必须含「不限制卡数」与「不抢占他人」。

## 阳性对照

`test_checker_actually_bites()` 把违规样本喂给**同一个** `problems_for_source()`，必须被抓到 ——
即"该 FAIL 时确实 FAIL"，否则扫描器本身失效而无人知。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "src" / "graspo" / "eval"

#: 已退役的评测侧卡数别名（**代码引用**即命中；注释/字符串里提及不算）。
RETIRED_CARD_SYMBOLS: frozenset[str] = frozenset({"MAX_GPU_COUNT"})

#: 作废的**卡数上限文案**（确切旧表述；命中即 FAIL）。
RETIRED_CARD_WORDING: tuple[str, ...] = (
    "卡数 ≤ 4",
    "卡数<=4",
    "卡数 <= 4",
    "最多 4 卡",
    "最多 4 张",
    "4-card cap",
    "limited to 4 cards",
    "at most 4 cards",
)

#: 现行口径里**必须出现**的要素（缺一即 FAIL —— 防"旧口径删了却没写清新口径"）。
REQUIRED_WORDING: tuple[tuple[str, str], ...] = (
    ("src/graspo/eval/guard.py", "不限制卡数"),
    ("src/graspo/eval/guard.py", "不抢占他人"),
)

#: 评测侧调用**不得**传的显式卡数上限（`core.gpu_guard.assert_gpu_lock` 的收窄入口；
#: 传了 = 运行期真的会拒绝 >N 卡）。判据按**关键字名**，不绑死被调方。
_CAP_KEYWORD = "max_cards"


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _all_entries(tree: ast.Module) -> list[ast.Constant]:
    """`__all__ = [...]` 里的字符串常量（符号导出清单）。"""
    entries: list[ast.Constant] = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            entries.extend(
                item
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return entries


def problems_for_source(rel: str, source: str) -> list[str]:
    """单个源文件的判据（纯函数，供真树与阳性对照共用）。

    Returns:
        问题清单；空 = 该文件合规。
    """
    problems: list[str] = []

    # ② 卡数上限文案（行级，确切旧字面量）
    for literal in RETIRED_CARD_WORDING:
        start = 0
        while (hit := source.find(literal, start)) != -1:
            lineno = source[:hit].count("\n") + 1
            problems.append(f"{rel}:{lineno}: 作废的卡数上限文案 {literal!r}")
            start = hit + len(literal)

    tree = ast.parse(source, filename=rel)

    # ① 退役别名：任何**代码引用**（Name / Attribute / import / __all__ 条目）
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in RETIRED_CARD_SYMBOLS:
            problems.append(f"{rel}:{node.lineno}: 退役卡数别名 {node.id!r} 的代码引用（No.1）")
        elif isinstance(node, ast.Attribute) and node.attr in RETIRED_CARD_SYMBOLS:
            problems.append(f"{rel}:{node.lineno}: 退役卡数别名 {node.attr!r} 的属性引用（No.1）")
        elif isinstance(node, ast.alias):
            name = (node.name or "").split(".")[-1]
            if name in RETIRED_CARD_SYMBOLS or (node.asname in RETIRED_CARD_SYMBOLS):
                problems.append(f"{rel}:{node.lineno}: 退役卡数别名 {node.name!r} 的导入（No.1）")
    for entry in _all_entries(tree):
        if entry.value in RETIRED_CARD_SYMBOLS:
            problems.append(
                f"{rel}:{entry.lineno}: `__all__` 仍导出退役卡数别名 {entry.value!r}（No.1）"
            )

    # ③ 真拦截调用点：任何调用传 `max_cards=<int>`
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != _CAP_KEYWORD:
                continue
            value = keyword.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)
            ):
                callee = _call_name(node) or "<expr>"
                problems.append(
                    f"{rel}:{node.lineno}: 评测侧给 {callee} 传了显式卡数上限 "
                    f"max_cards={value.value}（运行期真拦截，违反「单个作业不限制卡数」）"
                )
    return problems


def _eval_sources() -> dict[str, str]:
    """`src/graspo/eval/**/*.py` 的源码文本（相对 REPO 路径 → 文本）。"""
    return {
        str(path.relative_to(REPO)): path.read_text(encoding="utf-8")
        for path in sorted(EVAL_DIR.rglob("*.py"))
    }


def eval_card_caliber_problems(sources: dict[str, str] | None = None) -> list[str]:
    """对整片评测侧求问题清单（供真树断言与阳性对照共用）。"""
    texts = _eval_sources() if sources is None else sources
    problems: list[str] = []
    for rel, source in sorted(texts.items()):
        problems.extend(problems_for_source(rel, source))
    for rel, needle in REQUIRED_WORDING:
        if needle not in texts.get(rel, ""):
            problems.append(f"{rel}: 缺现行口径要素 {needle!r}（No.4）")
    return problems


# ── 真树断言 ────────────────────────────────────────────────────────────────


def test_eval_surface_has_no_retired_card_alias_or_cap():
    """★ 真树：`src/graspo/eval/**` 不得有退役别名 / 卡数上限文案 / 真拦截调用点。"""
    problems = eval_card_caliber_problems()
    assert problems == [], "评测侧卡池自述与事实分叉：\n  - " + "\n  - ".join(problems)


def test_eval_guard_states_the_current_caliber():
    """★ 真树：`eval/guard.py` 必须写出「不限制卡数」「不抢占他人」。"""
    problems = [entry for entry in eval_card_caliber_problems() if "缺现行口径要素" in entry]
    assert problems == [], "评测侧缺现行口径自述：\n  - " + "\n  - ".join(problems)


def test_removed_alias_is_really_gone_from_the_module():
    """符号级复核：`graspo.eval.guard` 里既没有该属性，也不在 `__all__`。"""
    from graspo.eval import guard as eval_guard

    assert not hasattr(eval_guard, "MAX_GPU_COUNT")
    assert "MAX_GPU_COUNT" not in eval_guard.__all__
    # 真实派生量仍可从未删的真相源取（不是靠"删了就没人能用"）
    from graspo.core.gpu_guard import ALLOWED_INDICES, MAX_CARDS

    assert MAX_CARDS == len(ALLOWED_INDICES) == 8
    assert eval_guard.ALLOWED_GPU_INDICES == frozenset(ALLOWED_INDICES)


# ── 阳性对照（扫描器自证能红）───────────────────────────────────────────────

_MUTANTS: tuple[tuple[str, str], ...] = (
    ("别名复活（赋值）", "MAX_GPU_COUNT: int = 4\n"),
    ("别名复活（导入）", "from graspo.eval.guard import MAX_GPU_COUNT\n"),
    ("别名复活（__all__ 条目）", '__all__ = ["MAX_GPU_COUNT"]\n'),
    ("真拦截调用点", "plan = resolve_gpu_plan(gpus, max_cards=4)\n"),
    ("真拦截调用点（另一入口）", "assert_gpu_lock(gpus, max_cards=2)\n"),
    ("作废文案复活", "# 卡数 ≤ 4，超过就拒绝\n"),
    ("作废文案复活（英文）", "# 4-card cap applies here\n"),
)


def test_checker_actually_bites():
    """★ 阳性对照：逐项把违规样本喂给同一个 `problems_for_source()` ⇒ 必须报出来。"""
    for name, source in _MUTANTS:
        problems = problems_for_source("src/graspo/eval/_mutant.py", source)
        assert problems, f"变异体未被抓到（扫描器失效）：{name}"


def test_checker_does_not_bite_on_the_real_tree_baseline():
    """★ 反面对照：合规样本（含"提及退役"的注释）不得被误判 —— 不是一律变红。"""
    clean = (
        "# 旧透传别名 MAX_GPU_COUNT（= MAX_CARDS）已删；这里不设卡数常量。\n"
        "ALLOWED_GPU_INDICES = frozenset(ALLOWED_INDICES)\n"
        "plan = resolve_gpu_plan(gpus)\n"
    )
    assert problems_for_source("src/graspo/eval/_clean.py", clean) == []


def test_required_wording_checker_actually_bites():
    """★ 阳性对照：新口径自述被删 ⇒ No.4 必须报出来。"""
    sources = _eval_sources()
    mutant = dict(sources)
    mutant["src/graspo/eval/guard.py"] = mutant["src/graspo/eval/guard.py"].replace(
        "不限制卡数", "卡数受限", 1
    ).replace("不抢占他人", "抢占他人", 1)
    assert any("缺现行口径要素" in p for p in eval_card_caliber_problems(mutant))
