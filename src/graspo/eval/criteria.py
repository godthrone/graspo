"""评测口径的单一真相源（accuracy criteria）。

**职责**：定义"一条预测算不算对"以及"综合准确率怎么算"。全项目只有这一份口径，
其它模块（聚合、Δ 配对、CLI、报告）只能调用这里，不得各自重算。

**本文件不负责**：数据集读取（`dataset.py`）、模型推理（`vllm_client.py`）、
产物结构（`schema.py`）、GPU 防呆（`guard.py`）。这里只有纯函数——不读文件、
不连网络、不碰 torch，可在 CPU 上独立单测（宪法 §1.3 计算与设施分离）。

**口径溯源（ELAM v3 脚本，逐项对齐，详见工位 report.md §口径对照）**

- `all_right = (pred_name == gt_name) and (pred_action == gt_action)`
  —— 工具名 AND 动作方向同时精确匹配。来源：ELAM
  `experiments/v3-single-step/eval/run_vllm_eval.py:111`。
- 综合准确率 = `all_right` 样本数 ÷ **有效样本数**（无 error 的样本）。
  来源：同目录 `stat_matrix.py:19-27,47`（`n_ok` 分母，`n_err` 剔除）。
- gt 字段路径 `targets[*].output.tool_calls[*].name` /
  `.arguments.action_type`。来源：`run_vllm_eval.py:66-69`。
- pred 字段路径 `choices[0].message.tool_calls[0].function.name` /
  `.function.arguments`（JSON 串）`.action_type`。来源：`run_vllm_eval.py:101-109`。

**口径版本号**：任何影响"对/错"判定或分母定义的改动都必须递增
`ACCURACY_CRITERIA_VERSION`，并同步更新本 docstring 的溯源段落——产物里记录的
版本号是事后复现的唯一锚点。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

#: 综合准确率口径版本。改动 `is_all_right` / `accuracy` / gt·pred 取值路径即必须递增。
ACCURACY_CRITERIA_VERSION = "elam-v3-allright-1"

#: 该口径的来源脚本（产物中留痕，便于独立复现时回溯）。
ACCURACY_CRITERIA_SOURCE = (
    "ELAM experiments/v3-single-step/eval/run_vllm_eval.py:111 + stat_matrix.py:19-27,47"
)


class GroundTruth(BaseModel):
    """单样本的标注答案：工具名 + 动作方向。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = ""
    action_type: str = ""


class Prediction(BaseModel):
    """单样本的模型预测：工具名 + 动作方向（解析失败时为空串）。

    带 ``parse_error`` 时表示模型有输出但无法解析出 action_type，
    此时按口径判为错误（与 v3 一致：v3 解析异常时把原始 arguments 串
    塞进 pred_action，必然与 gt 不等）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = ""
    action_type: str = ""
    parse_error: str | None = None


def extract_ground_truth(targets: list[dict[str, Any]] | None) -> GroundTruth:
    """从 ELAM 样本的 ``targets`` 提取标注答案。

    与 v3 语义一致：遍历全部 targets、全部 tool_calls，**后者覆盖前者**
    （v3 用的是普通赋值而非 break），因此最终取到的是最后一个 tool_call。
    单样本只判一个工具调用（system prompt 明示"每轮只输出一个工具调用"）。

    Args:
        targets: 样本的 ``targets`` 字段，形如
            ``[{"output": {"tool_calls": [{"name": ..., "arguments": {...}}]}}]``。

    Returns:
        ``GroundTruth``；无标注时两个字段均为空串。
    """
    tool_name = ""
    action_type = ""
    for target in targets or []:
        output = target.get("output") or {}
        for tool_call in output.get("tool_calls") or []:
            tool_name = str(tool_call.get("name") or "")
            arguments = tool_call.get("arguments") or {}
            if isinstance(arguments, str):
                # 防御：标注侧偶尔是 JSON 串而非 dict。
                arguments = _parse_arguments_str(arguments)
            action_type = (
                str(arguments.get("action_type") or "") if isinstance(arguments, dict) else ""
            )
    return GroundTruth(tool_name=tool_name, action_type=action_type)


def extract_prediction(message: dict[str, Any] | None) -> Prediction:
    """从 vLLM Chat Completions 响应里的 ``message`` 提取预测。

    取第一个 tool_call（v3 取 ``tcs[0]``）。无 tool_calls 时返回空预测
    ——按口径判错，不抛异常（v3 同样写一行 all_right=False）。

    Args:
        message: ``choices[0].message``。

    Returns:
        ``Prediction``；``parse_error`` 仅在 arguments 串无法解析为 JSON 时给出。
    """
    tool_calls = (message or {}).get("tool_calls") or []
    if not tool_calls:
        return Prediction()
    function = tool_calls[0].get("function") or {}
    tool_name = str(function.get("name") or "")
    raw_arguments = function.get("arguments", "")
    if isinstance(raw_arguments, dict):
        arguments: dict[str, Any] = raw_arguments
        parse_error: str | None = None
    else:
        text = str(raw_arguments)
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            return Prediction(
                tool_name=tool_name,
                action_type="",
                parse_error=f"json_decode_error: {exc}",
            )
        if not isinstance(parsed, dict):
            return Prediction(
                tool_name=tool_name,
                action_type="",
                parse_error=f"arguments_not_object: {type(parsed).__name__}",
            )
        arguments = parsed
        parse_error = None
    action_type = str(arguments.get("action_type") or "")
    return Prediction(tool_name=tool_name, action_type=action_type, parse_error=parse_error)


def is_all_right(prediction: Prediction, ground_truth: GroundTruth) -> bool:
    """口径核心：工具名 AND 动作方向同时精确匹配（v3 ``run_vllm_eval.py:111``）。

    注意这是**严格相等**，不做大小写归一、不做同义归一。历史口径如此，
    不得擅自放宽（放宽会让准确率虚高，与历史数字不可比）。
    """
    return (
        prediction.tool_name == ground_truth.tool_name
        and prediction.action_type == ground_truth.action_type
    )


def accuracy(correct: int, valid: int) -> float:
    """综合准确率 = 正确数 ÷ 有效样本数；有效样本数为 0 时返回 0.0。

    分母只算**有效样本**（无 error 的样本），与 v3 ``stat_matrix.py`` 的
    ``n_ok`` 一致。错误样本既不加分子也不加分母。
    """
    if valid <= 0:
        return 0.0
    return correct / valid


def _parse_arguments_str(text: str) -> Any:
    """把 JSON 串形式的 arguments 解析为 dict；失败时原样返回字符串。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
