"""标注模块修复回归测试（2026-08-07）。

覆盖两个已修复 bug：
- J16 嵌套对象：json_labeler 内层 key 校验此前只查顶层 GT，`"name"` 被误判
  多余字段 → E@17。修复：gt_ctx 上下文栈随 pending_key 下降一层（_child_gt）。
- T32 参数标签缺 `>`：tool_call_labeler 的 `find(">")` 此前跨行扫描，把后续
  `</function>` 的 `>` 误当标签闭合，参数名含换行走多余参数分支。修复：
  换行即未闭合 → name_end<0 → E 于 `<` 处截断。
"""

from graspo.ripple.annotation.labeler import AnnotationInput, annotate
from graspo.ripple.annotation.roles import CharTag


def _annotate(completion: str, targets: list[dict], format_type: str) -> str:
    result = annotate(
        AnnotationInput(
            completion_text=completion,
            targets=targets,
            tokenizer=None,
            format_type=format_type,
            check_json_markdown=False,
        )
    )
    return "".join(t.value for t in result.tags)


# ── J16 嵌套对象（gt_ctx 上下文栈）──


def test_nested_object_inner_key_valid() -> None:
    """内层 key 存在：GT 沿 pending_key 下降后校验通过，全 S/V 无 E。"""
    targets = [{"output": {"content": {"user": {"name": "Alice", "age": 30}}}}]
    ann = _annotate('{"user":{"name":"Alice","age":30}}', targets, "json")
    assert "E" not in ann
    assert "D" not in ann
    assert ann.count("V") >= 2  # Alice + 30


def test_nested_object_inner_key_typo() -> None:
    """内层 key 拼错：GT 下降后前缀匹配 → typo 分支 E。"""
    targets = [{"output": {"content": {"user": {"name": "Alice"}}}}]
    ann = _annotate('{"user":{"namx":"Alice"}}', targets, "json")
    assert "E" in ann
    # 前缀 nam 匹配后首个不匹配字符（x）为 E
    e_pos = ann.find("E")
    completion = '{"user":{"namx":"Alice"}}'
    assert completion[e_pos] == "x"


def test_nested_object_inner_key_extra() -> None:
    """内层多余 key：GT 下降后仍不在子对象 → extra 分支 E。"""
    targets = [{"output": {"content": {"user": {"name": "Alice"}}}}]
    ann = _annotate('{"user":{"extra_key":1}}', targets, "json")
    assert ann.find("E") >= 0


def test_nested_object_inner_value_type_error() -> None:
    """内层值类型错：GT 数字 vs 模型字符串 → E（类型校验也走 gt_ctx）。"""
    targets = [{"output": {"content": {"user": {"age": 30}}}}]
    ann = _annotate('{"user":{"age":"30"}}', targets, "json")
    e_pos = ann.find("E")
    completion = '{"user":{"age":"30"}}'
    # E 应落在值首字符（"age" 后的引号位置）
    assert e_pos > completion.find("age")


def test_nested_object_array_element_gt() -> None:
    """数组元素对象：GT 值为 list 时取首个 dict 元素校验内层 key。"""
    targets = [{"output": {"content": {"items": [{"x": 1, "y": 2}]}}}]
    ann = _annotate('{"items":[{"x":1,"y":2}]}', targets, "json")
    assert "E" not in ann


def test_nested_object_gt_none_no_crash() -> None:
    """GT 缺失（content 非 dict）：gt_ctx 顶层为 None，不崩、不误标。"""
    targets = [{"output": {"content": 5}}]
    ann = _annotate('{"a":{"b":1}}', targets, "json")
    assert "E" not in ann


# ── T32 参数标签缺 >（find(">") 限制当前行）──


def _tc_targets() -> list[dict]:
    return [
        {
            "output": {
                "tool_calls": [
                    {
                        "name": "robot_atomic_control",
                        "arguments": {"action_type": "逆时针旋转", "angle_deg": 40.0},
                    }
                ]
            }
        }
    ]


def test_param_tag_missing_gt_e_at_gt_position() -> None:
    """参数标签缺 >：换行即未闭合 → E 于"应出现 > 的位置"（参数名后首字符），
    `<parameter=` 与参数名标 S（v3.1 用户裁定）。"""
    completion = (
        "<tool_call>\n<function=robot_atomic_control>\n"
        "<parameter=action_type\n</function>\n</tool_call>"
    )
    ann = _annotate(completion, _tc_targets(), "tool_call")
    e_pos = ann.find("E")
    assert e_pos >= 0
    assert completion[e_pos] == "\n"  # E 在参数名后的换行（应出现 > 处）
    assert all(c == "S" for c in ann[:e_pos])  # 标签与参数名是正确结构
    assert all(c == "D" for c in ann[e_pos + 1:])


def test_param_tag_normal_line_still_works() -> None:
    """正常参数标签（同行闭合）不受影响：全 S/V 无 E。"""
    completion = (
        "<tool_call>\n<function=robot_atomic_control>\n"
        "<parameter=action_type>\n逆时针旋转\n</parameter>\n"
        "<parameter=angle_deg>\n40.0\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    ann = _annotate(completion, _tc_targets(), "tool_call")
    assert "E" not in ann
    assert CharTag.VALUE in ann


def test_param_tag_newline_inside_name_is_error() -> None:
    """参数名内嵌换行（`<parameter=action\n_type>`）：行内 find 失败 →
    E 于换行处（应出现 > 的位置，v3.1 语义）。"""
    completion = (
        "<tool_call>\n<function=robot_atomic_control>\n"
        "<parameter=action\n_type>\nx\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    ann = _annotate(completion, _tc_targets(), "tool_call")
    e_pos = ann.find("E")
    assert e_pos >= 0
    assert completion[e_pos] == "\n"  # action 名后换行（应为 >）
