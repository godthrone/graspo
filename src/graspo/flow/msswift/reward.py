"""graspo 奖励在 ms-swift 奖励通道上的适配器（设施层，算法仍在 ``ripple``）。

**职责边界**

- **本模块**：把 ms-swift 的 reward 调用约定（``__call__(completions, **dataset_columns)``，
  返回 ``List[float]``）接到 ``ripple`` 的奖励实现上；并向 ms-swift 的 ``orms``
  注册表登记一个名字（``graspo``），使 ``msswift`` 段/argv 可以按名字引用。
- **不负责**：奖励算法本身（在 ``graspo.ripple.reward``，native 与 msswift 同源，
  宪法 §1.3 计算与设施分离）；打分之外的任何训练逻辑。

**为什么需要适配器而不是直接用 ``GraspoReward``**：ms-swift 的 reward 通道是
"批量 completion + 数据集列"，graspo 的是"单条 completion + targets"；两者形状不同，
适配层只做形状转换，不做任何打分逻辑。

**数据列契约**：GRPO 数据集必须携带 ``targets`` 列（JSON 字符串，ARD/graspo 的
``targets`` 原样序列化）。缺列时不静默给 0 分——直接抛错（§2.3 边界校验）。

**工具调用档必须走 ``score_parsed(is_tool_call=True)``**：graspo 的奖励有两条
互斥的判分路径——纯文本档看 ``targets[*].output.content``（fenced JSON），
工具调用档看 ``targets[*].output.tool_calls``。ELAM 一类纯 tool-call 数据集
的 targets **只有 ``tool_calls``**，此时文本路径的 ``check_targets`` 为空 ⇒
``_max_reward([]) == 0.0`` ⇒ 归一化分母为 0 ⇒ 奖励**恒为 0**（不是"小"，
是数学上的 0）。恒零奖励 ⇒ 组内无方差 ⇒ ``classify_group`` 判 RETRY/INVALID
⇒ 零 advantage ⇒ ``grad_norm`` 恒 0（T031 实测）。native 侧同口径走
``flow/trainer/rollout.py`` 的 ``score_parsed(..., is_tool_call=...)``，
本适配器必须与它同源，不得只在文本路径上判分。
"""

from __future__ import annotations

import json
from typing import Any

from graspo.core.schema import RewardConfig
from graspo.ripple.parsing.qwen_tool_parser import parse_qwen_tool_completion
from graspo.ripple.reward.reward import GraspoReward

#: ms-swift 奖励注册表里的名字（argv ``--reward_funcs <此名>``）。
GRASPO_REWARD_NAME = "graspo"

#: GRPO 数据集里承载 graspo targets 的列名（单一真相源，与 dataset.py 一致）。
GRASPO_TARGETS_COLUMN = "targets"

#: GRPO 数据集里承载工具 schema 的列名（与 ``dataset.py`` 的 ``tools`` 列一致）。
#: 列缺失时**透明降级**为 ``None``：XML 路径的 required 参数校验不再生效，
#: 但工具调用的名称/参数比对（dict_compare_score）仍然完整——缺失只影响
#: 「少给参数」这一种失败形态的检出率，不影响奖励非零这一必要条件。
GRASPO_TOOLS_COLUMN = "tools"


def expects_tool_calls(targets: Any) -> bool:
    """判据与 ``core/schema.py::Sample.expects_tool_calls`` 同源：目标里有 ``tool_calls``。"""
    if not isinstance(targets, list):
        return False
    for target in targets:
        output = target.get("output") if isinstance(target, dict) else None
        if isinstance(output, dict) and output.get("tool_calls") is not None:
            return True
    return False


def _parse_targets(raw: Any) -> Any:
    """把数据集列还原成 graspo 的 ``targets`` 结构（JSON 字符串 → list）。"""
    if raw is None:
        raise ValueError(
            f"GRPO dataset is missing the '{GRASPO_TARGETS_COLUMN}' column required by the "
            "graspo reward; refusing to score with a default of 0.0 (that would silently "
            "train on a constant reward)."
        )
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _parse_tools(raw: Any) -> list[dict[str, Any]] | None:
    """把 ``tools`` 列还原成工具 schema 列表；列缺失/为空时返回 ``None``（透明降级）。"""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        return None
    return raw


class GraspoMsSwiftReward:
    """ms-swift ORM：``(completions, targets, **columns) -> List[float]``。

    实例由 ms-swift 的 ``GRPOTrainer._prepare_rewards`` 用 ``GraspoMsSwiftReward(args=args)``
    构造（因此签名必须接受 ``args``），打分委托给 ``ripple`` 的 ``GraspoReward``。
    """

    def __init__(self, args: Any = None, reward_config: RewardConfig | None = None) -> None:
        self.args = args
        self._reward = GraspoReward(reward_config)

    def __call__(self, completions: list[str], **columns: Any) -> list[float]:
        targets_column = columns.get(GRASPO_TARGETS_COLUMN)
        if targets_column is None:
            raise ValueError(
                f"GRPO reward received no '{GRASPO_TARGETS_COLUMN}' column; "
                f"available columns: {sorted(columns)}"
            )
        tools_column = columns.get(GRASPO_TOOLS_COLUMN)
        scores: list[float] = []
        for index, completion in enumerate(completions):
            raw = targets_column[index] if isinstance(targets_column, list) else targets_column
            targets = _parse_targets(raw)
            # 每条的格式族由它自己的 targets 决定，不由批次猜测（同组样本同源，
            # 但逐条判定不会因批次混装而误判）。
            is_tool_call = expects_tool_calls(targets)
            if is_tool_call:
                tools_raw = tools_column[index] if isinstance(tools_column, list) else tools_column
                tools = _parse_tools(tools_raw)
                parsed = parse_qwen_tool_completion(
                    str(completion),
                    expect_tool_calls=True,
                    tools=tools,
                )
                reward = self._reward.score_parsed(parsed, targets, is_tool_call=True).reward
            else:
                reward = self._reward.score(str(completion), targets).reward
            scores.append(float(reward))
        return scores


def register_graspo_reward() -> None:
    """把 ``graspo`` 注册进 ms-swift 的 ``orms`` 表（幂等）。

    注册到 ms-swift 的公开扩展点（``swift.rewards.orms``）而不是改 ms-swift 源码，
    符合宪法 §1.2「新增实现只注册、不改现有代码」。
    """
    from swift.rewards import orms  # 延迟导入：无 ms-swift 环境不触发

    orms[GRASPO_REWARD_NAME] = GraspoMsSwiftReward
