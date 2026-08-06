"""ripple.reward — 奖励计算域：评分、结构比较、归一化。

- reward.py: GraspoReward 三层评分（marker/content/target）
- compare.py: 结构化 dict 比较（dcs/base_dcs/all_right）
- normalize.py: target 归一化

token 级 advantage 已迁至 ripple/annotation/advantages.py（v0.20.0 标注驱动）。
"""

from graspo.ripple.reward.reward import GraspoReward, RewardConfig, RewardResult

__all__ = ["GraspoReward", "RewardConfig", "RewardResult"]
