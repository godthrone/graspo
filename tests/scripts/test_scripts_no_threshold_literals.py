"""★ 机核（修正轮 1，审 B1；采纳全局审查建议①）：`scripts/**` 不得出现**阈值字面量比较**。

病史：`scripts/gen_tier_json.py` 曾对判据② 通过线写死
``status = PASS if delta_pp >= 5.0 else FAIL`` —— 这是**第二份阈值副本**（宪法 §1.4 破裂），
SFT 档 +7.3 pp 会被误判为「通过」，且与回填器的矩阵冻结口径交叉核对冲突。

判据：脚本层只需要*读*阈值（唯一读取通道 `src/graspo/eval/thresholds.py`），
**不得**自己拿 `5` / `5.0` / `10` / `10.0` 做大小比较来做通过判定。
本断言用 AST 扫描**代码**（不含注释 / 字符串），故不会把说明文字里的 `>= 5.0` 误报。

⛔ 只认 `5` / `10` 两个数值：`>= 0`（不劣化线的 0.0）在脚本里普遍用于非口径的空值/计数守卫，
纳入扫描会产生大量误报；CPT/OPD 的不劣化判定同样必须经唯一读取通道。
"""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"

#: 判据② 在役的两个「Δ 通过线」数值（SFT +10.0 pp / GRASPO +5.0 pp）。
#: ★ 这里是**检测器**的输入（要拦住哪些字面量），⛔ 不是任何判定用的阈值。
THRESHOLD_LITERALS = (5, 5.0, 10, 10.0)

_COMPARE_OPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


def scan_source_for_threshold_literals(source: str, filename: str) -> list[tuple[str, int, str]]:
    """在**一份源码**里找阈值字面量比较；返回 `[(file, lineno, 表达式)]`。"""
    hits: list[tuple[str, int, str]] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, _COMPARE_OPS) for op in node.ops):
            continue
        operands = [node.left] + list(node.comparators)
        for operand in operands:
            if (isinstance(operand, ast.Constant) and not isinstance(operand.value, bool)
                    and operand.value in THRESHOLD_LITERALS):
                hits.append((filename, node.lineno, ast.unparse(node)))
                break
    return hits


def scan_tree_for_threshold_literals(root: Path) -> list[tuple[str, int, str]]:
    """递归扫描 `root/**/*.py`（跳过 `__pycache__`）。"""
    hits: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hits.extend(scan_source_for_threshold_literals(
            path.read_text(encoding="utf-8"), str(path)))
    return hits


def test_scripts_have_no_threshold_literal_comparison():
    """★ 真件：`scripts/**` 内不得出现拿 5/5.0/10/10.0 做比较的通过判定。"""
    hits = scan_tree_for_threshold_literals(_SCRIPTS)
    assert hits == [], (
        "★ scripts/** 出现阈值字面量比较（第二份阈值副本）⇒ 必须改为读唯一真相源"
        f"（`src/graspo/eval/thresholds.py`）：{hits}"
    )


def test_scanner_catches_injected_literal(tmp_path: Path):
    """★ 能失败对照：塞一处 `>= 5.0` / `>= 10` 的通过判定 ⇒ 扫描器**必红**。

    ⛔ 对照物写临时目录，不碰仓内脚本。
    """
    probe = tmp_path / "leaky.py"
    probe.write_text(
        "def judgement(delta_pp):\n"
        "    status = '通过' if delta_pp >= 5.0 else '失败'\n"
        "    ok = delta_pp >= 10\n"
        "    return status, ok\n",
        encoding="utf-8")
    hits = scan_tree_for_threshold_literals(tmp_path)
    assert len(hits) == 2, f"扫描器漏检注入的阈值字面量：{hits}"
    assert all(h[1] in (2, 3) for h in hits), hits


def test_scanner_ignores_comments_and_strings(tmp_path: Path):
    """★ 反面：说明文字 / 注释里的 `>= 5.0` **不得**误报（AST 只认代码）。"""
    probe = tmp_path / "doc.py"
    probe.write_text(
        '"""口径：判据② 不得写死 `delta_pp >= 5.0`。"""\n'
        "NOTE = '旧口径 status = PASS if delta_pp >= 10 else FAIL'\n"
        "# 曾经是：status = PASS if delta_pp >= 5.0 else FAIL\n"
        "def fine(delta_pp, threshold):\n"
        "    return delta_pp >= threshold\n",
        encoding="utf-8")
    assert scan_tree_for_threshold_literals(tmp_path) == []
