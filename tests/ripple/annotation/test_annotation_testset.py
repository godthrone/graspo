"""标注模块测试：用 annotation_testset_v3.jsonl 数据集驱动验证。

数据集 65 条覆盖矩阵（tool call 44 + JSON 21），每条断言：
- 标注与期望逐字符一致（S/V/T/W/E/D）
- 长度与 completion 一致
- E 之后全部为 D（严格对齐截断）
- error_pos 等于首个 E 下标（无 E 的行为空，见 v3.3）

v3 相对于 v2 的变更（2026-08-07，与 v14 双工具数据同步）：
- T 系列 33 条换 v14 双工具格式（rotate_arm/extend_arm，按 GT action_type 映射）；
  T30（get_weather 跨函数）保留原样
- 标注语义更新：多余/未知参数 → 参数名首字符 E（原 E 落在 p/< 处）；
  双调用（闭合后第二个 <tool_call>）→ 第二个调用首字符 E（v20 毒药形态）
- 新增 4 条：T33-T36（毒药双调用/工具名错/跨工具参数错/尾随回归）

v3.1 变更（2026-08-08，用户裁定闭合后一刀切规则）：
- **闭合后多余内容检测取代紧邻检测**：</tool_call> 后第一个字符（含换行）
  直接 E + 其后全 D——工具调用之后没有思考/结构，只可能是乱码，要压制
  而不是放弃（v21 实测多调用 0.5%→9.8% 增长根因：间隔杂散标签漏检）。
  千问模板无 </tool_call> 后换行规定（实测 1312 条合法样本闭合后均空串）。
  统一规则：换行标注跟随其后内容——结构→S、值→W（v3.2）、结束后的多余→E。
- T17/T33/T36/T40 期望标注按新规则更新（T36 从"自然语言尾随 W"改 E+D；
  T33/T40 从"换行 W + 第二个 <tool_call> 的 < 为 E"改"闭合后换行即 E"）
- 新增 5 条真实 rollout 形态：T42-T46（间隔杂散标签夹 JSON/自然语言、
  紧邻双调用、截断第二调用、首调用错误+尾部多调用）

v3.3 变更（2026-11-18，P1：工具名是可评分字段）：
- **工具名写错 ⇒ 名字段整段 VALUE（不是 E）**：``<function=NAME>`` 的 NAME 按身份字段评分，
  写错时从首个不匹配字符起整段标 V（该段每个 token 拿 −1），其后照旧 drop。
  改前只把首个不匹配字符标 E（名字前 22 个结构 token 全 +1 ⇒ 整条净得 +21，**正的**）。
- 受影响的行：T13（``robot_contol`` 拼错）、T34/T46（选错/写坏工具名）⇒
  annotation 由 ``…E D…`` 改为 ``…V… D…``，``error_pos`` 如实置空（这些行没有 E 标签了）。
- 不变式随之推广：``test_error_then_dropped_invariant`` → ``test_dropped_tail_invariant``
  （切点 = 首个 E 或首个 D，其后仍必须全 D）。

v3.3 修正（2026-11-18 复审阻断 B1：工具名后缺 ``>``）：
- 判据修正：``<function=NAME`` 后面本该是 ``>``。若 NAME 段里撞见 ``<``
  （形如 ``<function=think</function>`` / ``<function=rotate_arm<parameter=…>``），
  说明找到的 ``>`` 属于**后面那个标签**，不是名字的闭合 ⇒ 按「缺 ``>``」处理：
  E 于「应出现 ``>`` 的位置」+ 其后全 D，**不产生 V 段**。
- 修正对象：T45 原期望是 ``…15 个 V…``，其中 10 个是 ``</function`` 这串**结构字符**
  被当成"值错"（连它前面写对的工具名一起被罚）⇒ 现恢复为 ``22S + E(22) + D…``
  （与 HEAD 一致，``error_pos`` 回到 22）。T46 不受影响：它的 ``think>`` 有 ``>``，
  名字写错仍按 P1 走 V 段。
- 为什么之前没被抓住：T45/T46 的「名字段 V 到哪为止」完全由实现决定，没有独立断言。
  现在缺 ``>`` 的形态由 ``test_tool_name_advantage.py``「判据 4」用字面期望钉住。

v3.4 变更（2026-11-24，B1：工具名是**值**，不是结构）：
- **工具名写对 ⇒ 名字段整段 VALUE（不是 S）**：划分口径是「写错 = 结构 / 选错 = 值」，
  工具名属于**选错**。写对时它仍拿 +1.0（值域满分），但不再计入 P3 的结构字符数——
  否则 ``k = 组内各条 S 字符数的最小值`` 会伸进工具名内部，把**写对的工具名** mask 成 0.0
  （反例见 ``test_group_structure_mask.py`` 第 3 节）。
- 受影响的行：32 条 tool_call 行的名字区间由 ``S…`` 改为 ``V…``（T01/T03/T04/T06/T07/T11/
  T12/T15/T16/T17/T18/T19/T20/T22/T25–T29/T30–T33/T35–T37/T39–T44），改的是**同一段字符的
  标签种类**，字符数、``error_pos``、``correct`` 一律不变。
- 名字前后的格式符（``<function=`` / ``>``）仍是 S；名字**写错**的行（T13/T34/T46 等）不受影响。
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


def test_dropped_tail_invariant() -> None:
    """错误/截断点之后必须全部为 D（严格对齐截断）。

    P1（2026-11-18）把「错误点」的判据从"首个 E"推广为"首个 E **或** 首个 D"：
    工具名写错的 completion **不产生 E**（名字段是 VALUE，按字段评分拿 −1），
    其后照旧 drop ⇒ 形态是 ``S…V…D…``。判据本身没放松——切点之后仍必须全是 D，
    ⛔ 不允许出现 D 之后再冒出一个 S/V/W/T。
    """
    for row in _load_testset():
        ann = row["annotation"]
        cut = next((i for i, char in enumerate(ann) if char in "ED"), -1)
        if cut == -1:
            continue
        tail = ann[cut + 1 :]
        assert all(c == "D" for c in tail), (
            f"{row['id']} 错误/截断点（@{cut}={ann[cut]}）之后存在非 D 字符"
        )


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
