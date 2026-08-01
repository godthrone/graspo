"""graspo-ripple 算法层：纯计算，零设施依赖。

Ripple（涟漪）命名哲学：在强化学习中，一个 Token 的即时奖励并非孤立，
它会对前后 Token 的梯度传播产生"涟漪效应"（PPO 的 GAE 或时序差分）。
Ripple 层承载逐 Token 传播的微观动态与信用分配的波浪式回传。

本层不 import 任何设施（GPU/分布式/IO），所有模块可在单线程本地测试。
"""
