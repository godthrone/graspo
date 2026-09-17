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
"""

from __future__ import annotations

import json
from typing import Any

from graspo.ripple.reward.reward import GraspoReward
from graspo.core.schema import RewardConfig

#: ms-swift 奖励注册表里的名字（argv ``--reward_funcs <此名>``）。
GRASPO_REWARD_NAME = "graspo"

#: GRPO 数据集里承载 graspo targets 的列名（单一真相源，与 dataset.py 一致）。
GRASPO_TARGETS_COLUMN = "targets"


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
        scores: list[float] = []
        for index, completion in enumerate(completions):
            raw = targets_column[index] if isinstance(targets_column, list) else targets_column
            targets = _parse_targets(raw)
            scores.append(float(self._reward.score(str(completion), targets).reward))
        return scores


def register_graspo_reward() -> None:
    """把 ``graspo`` 注册进 ms-swift 的 ``orms`` 表（幂等）。

    注册到 ms-swift 的公开扩展点（``swift.rewards.orms``）而不是改 ms-swift 源码，
    符合宪法 §1.2「新增实现只注册、不改现有代码」。
    """
    from swift.rewards import orms  # 延迟导入：无 ms-swift 环境不触发

    orms[GRASPO_REWARD_NAME] = GraspoMsSwiftReward
