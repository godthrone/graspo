"""标注模块测试：用 annotation_testset_v3.jsonl 数据集驱动验证。

数据集 55 条覆盖矩阵（tool call 34 + JSON 21），每条断言：
- 标注与期望逐字符一致（S/V/T/W/E/D）
- 长度与 completion 一致
- E 之后全部为 D（严格对齐截断）
- error_pos 等于首个 E 下标

v3 相对于 v2 的变更（2026-08-07，与 ELAM v14 双工具数据同步）：
- T 系列 33 条换 v14 双工具格式（rotate_arm/extend_arm，按 GT action_type 映射）；
  T30（get_weather 跨函数）保留原样
- 标注语义更新：多余/未知参数 → 参数名首字符 E（原 E 落在 p/< 处）；
  双调用（闭合后第二个 <tool_call>）→ 第二个调用首字符 E（v20 毒药形态）
- 新增 4 条：T33-T36（毒药双调用/工具名错/跨工具参数错/尾随回归）
"""

import json
from pathlib import Path

import pytest

from graspo.ripple.annotation.labeler import AnnotationInput, annotate

TESTSET = Path(__file__).resolve().parents[2] / "data" / "annotation_testset_v3.jsonl"

# J02 等是无围栏 JSON 场景（check_json_markdown=False）
NO_FENCE_CASES = {"J02", "J18", "J20", "J21", "J22"}
# think 场景（check_think=True）
THINK_CASES = {"T11", "T24", "T25"}


def _load_testset() -> list[dict[str, str]]:
    with open(TESTSET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_targets(format_type: str, gt: str) -> list[dict]:
    if format_type == "tool_call":
        # gt 形如 robot_atomic_control(action_type=逆时针旋转, angle_deg=40.0)
        name, rest = gt.split("(", 1)
        args_str = rest.rstrip(")").strip()
        arguments: dict = {}
        for pair in args_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split("=", 1)
            v = v.strip()
            try:
                arguments[k.strip()] = float(v)
            except ValueError:
                arguments[k.strip()] = v
        return [{"output": {"tool_calls": [{"name": name.strip(), "arguments": arguments}]}}]
    try:
        content = json.loads(gt)
    except (TypeError, ValueError, json.JSONDecodeError):
        content = {}
    return [{"output": {"content": content}}]


@pytest.mark.parametrize("row", _load_testset(), ids=lambda r: r["id"])
def test_annotation_matches_testset(row: dict[str, str]) -> None:
    completion = row["completion"]
    expected = row["annotation"]
    cid = row["id"]

    result = annotate(
        AnnotationInput(
            completion_text=completion,
            targets=_make_targets(row["type"], row["ground_truth"]),
            tokenizer=None,
            format_type=row["type"],
            check_json_markdown=cid not in NO_FENCE_CASES,
            check_think=cid in THINK_CASES,
        )
    )
    actual = "".join(t.value for t in result.tags)

    assert len(actual) == len(completion), (
        f"{cid} 标注长度不匹配: actual={len(actual)}, expected={len(completion)}"
    )
    assert actual == expected, (
        f"{cid} 标注不匹配\n"
        f"  completion: {completion!r}\n"
        f"  expected:   {expected}\n"
        f"  actual:     {actual}"
    )


def test_all_tags_are_valid_enum() -> None:
    for row in _load_testset():
        for ch in row["annotation"]:
            assert ch in "SVTWED", f"{row['id']} 含非法标注字符 {ch!r}"


def test_error_then_dropped_invariant() -> None:
    """E 之后必须全部为 D（严格对齐截断）。"""
    for row in _load_testset():
        ann = row["annotation"]
        e = ann.find("E")
        if e == -1:
            assert "D" not in ann, f"{row['id']} 无 E 但含 D"
        else:
            tail = ann[e + 1 :]
            assert all(c == "D" for c in tail), f"{row['id']} E 之后存在非 D 字符"


def test_error_pos_matches_first_e() -> None:
    for row in _load_testset():
        e = row["annotation"].find("E")
        if row["error_pos"]:
            assert int(row["error_pos"]) == e, (
                f"{row['id']} error_pos={row['error_pos']} != 首个 E@{e}"
            )
        else:
            assert e == -1, f"{row['id']} 无 error_pos 但存在 E@{e}"


def test_all_cases_marked_correct() -> None:
    """数据集应经人工/agent 核验（correct=yes 或 pending 标记）。"""
    for row in _load_testset():
        assert row["correct"] in ("yes", "pending"), (
            f"{row['id']} correct={row['correct']!r}，期望 yes 或 pending"
        )
