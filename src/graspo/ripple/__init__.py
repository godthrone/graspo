"""graspo-ripple 算法层：GRASPO 训练方法论的算法实现。

Ripple（涟漪）命名哲学：在强化学习中，一个 Token 的即时奖励并非孤立，
它会对前后 Token 的梯度传播产生"涟漪效应"（PPO 的 GAE 或时序差分）。
Ripple 层承载逐 Token 传播的微观动态与信用分配的波浪式回传——
GRASPO-Ripple token 级奖励算法（annotation/advantages.py，v0.20.0 标注驱动）
即本层命名出处。

层边界：
- 允许：标准库、pydantic、torch 纯张量计算（CPU 可单测）
- 禁止：GPU 设备调用（torch.cuda）、分布式、网络、文件 IO

本层所有模块可在 CPU 上独立测试，不启动 GPU。
"""

from graspo.ripple.parity import group_advantages, has_reward_variance

__all__ = ["group_advantages", "has_reward_variance"]
