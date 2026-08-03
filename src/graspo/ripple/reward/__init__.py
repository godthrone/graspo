"""ripple.reward — 奖励计算域：评分、结构比较、归一化、token 级奖励。

- reward.py: GraspoReward 三层评分（marker/content/target）
- compare.py: 结构化 dict 比较（dcs/base_dcs/all_right）
- normalize.py: target 归一化
- token_reward.py: GRASPO-Ripple 算法本体（token 级奖励涟漪传播）
"""

from graspo.ripple.reward.reward import GraspoReward, RewardConfig, RewardResult

__all__ = ["GraspoReward", "RewardConfig", "RewardResult"]
