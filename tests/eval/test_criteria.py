"""``graspo.eval.criteria`` 的单测：口径与 ELAM v3 脚本逐项对齐。

测试重点不是"函数返回什么"，而是"口径有没有按预期变化"——口径一改，历史数字
就不可比，所以这里锁死的是**语义**，不是实现细节。
"""

from __future__ import annotations

import pytest

from graspo.eval.criteria import (
    ACCURACY_CRITERIA_VERSION,
    GroundTruth,
    Prediction,
    accuracy,
    extract_ground_truth,
    extract_prediction,
    is_all_right,
)


def test_extract_ground_truth_reads_name_and_action():
    targets = [
        {
            "output": {
                "tool_calls": [
                    {"name": "rotate_arm", "arguments": {"action_type": "逆时针"}},
                ]
            }
        }
    ]
    gt = extract_ground_truth(targets)
    assert gt.tool_name == "rotate_arm"
    assert gt.action_type == "逆时针"


def test_extract_ground_truth_last_tool_call_wins_like_v3():
    """v3 用普通赋值而非 break，因此取到的是**最后一个** tool_call。"""
    targets = [
        {"output": {"tool_calls": [{"name": "a", "arguments": {"action_type": "x"}}]}},
        {"output": {"tool_calls": [{"name": "b", "arguments": {"action_type": "y"}}]}},
    ]
    gt = extract_ground_truth(targets)
    assert (gt.tool_name, gt.action_type) == ("b", "y")


def test_extract_ground_truth_empty_targets_is_blank_not_none():
    gt = extract_ground_truth(None)
    assert gt.tool_name == ""
    assert gt.action_type == ""


def test_extract_ground_truth_tolerates_arguments_as_json_string():
    targets = [{"output": {"tool_calls": [{"name": "n", "arguments": '{"action_type": "up"}'}]}}]
    assert extract_ground_truth(targets).action_type == "up"


def test_extract_prediction_parses_json_arguments():
    message = {
        "tool_calls": [{"function": {"name": "lift_arm", "arguments": '{"action_type": "抬高"}'}}]
    }
    prediction = extract_prediction(message)
    assert prediction.tool_name == "lift_arm"
    assert prediction.action_type == "抬高"
    assert prediction.parse_error is None


def test_extract_prediction_without_tool_calls_is_blank():
    prediction = extract_prediction({"content": "sorry"})
    assert prediction.tool_name == ""
    assert prediction.action_type == ""
    assert prediction.parse_error is None


def test_extract_prediction_records_parse_error_on_bad_json():
    message = {"tool_calls": [{"function": {"name": "n", "arguments": "{not json"}}]}
    prediction = extract_prediction(message)
    assert prediction.tool_name == "n"
    assert prediction.parse_error is not None
    assert prediction.parse_error.startswith("json_decode_error")


def test_extract_prediction_rejects_non_object_arguments():
    message = {"tool_calls": [{"function": {"name": "n", "arguments": "[1,2]"}}]}
    prediction = extract_prediction(message)
    assert prediction.parse_error is not None
    assert prediction.parse_error.startswith("arguments_not_object")


@pytest.mark.parametrize(
    ("pred", "gt", "expected"),
    [
        (("rotate_arm", "left"), ("rotate_arm", "left"), True),
        (("rotate_arm", "right"), ("rotate_arm", "left"), False),
        (("lift_arm", "left"), ("rotate_arm", "left"), False),
        (("rotate_arm", ""), ("rotate_arm", "left"), False),
        (("", ""), ("rotate_arm", "left"), False),
        (("", ""), ("", ""), True),  # 两边都空 → 朴素相等为真（与 v3 行为一致）
    ],
)
def test_is_all_right_is_strict_equality_of_both_fields(pred, gt, expected):
    """all_right = 工具名 AND 动作方向同时精确匹配（v3 run_vllm_eval.py:111）。"""
    prediction = Prediction(tool_name=pred[0], action_type=pred[1])
    ground_truth = GroundTruth(tool_name=gt[0], action_type=gt[1])
    assert is_all_right(prediction, ground_truth) is expected


def test_is_all_right_does_not_normalize_case_or_whitespace():
    """口径是严格相等——擅自归一（大小写/空格）会让准确率虚高、与历史不可比。"""
    prediction = Prediction(tool_name="Rotate_Arm", action_type=" left")
    ground_truth = GroundTruth(tool_name="rotate_arm", action_type="left")
    assert is_all_right(prediction, ground_truth) is False


def test_accuracy_uses_valid_samples_as_denominator():
    """分母只算有效样本（v3 stat_matrix.py 的 n_ok），错误样本两边都不计。"""
    assert accuracy(correct=53, valid=100) == pytest.approx(0.53)
    assert accuracy(correct=0, valid=0) == 0.0
    # 错误样本被剔除后，分母是 90——不是 100
    assert accuracy(correct=53, valid=90) == pytest.approx(53 / 90)


def test_criteria_version_is_pinned():
    """口径版本是产物里的复现锚点，改动必须是有意识的行为。"""
    assert ACCURACY_CRITERIA_VERSION == "elam-v3-allright-1"
