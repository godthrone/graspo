"""core — 通用契约层：配置模型与不依赖训练方法论的通用件。

ripple/ 承载训练方法论算法（reward/group_decision/parsing/monitoring），
core/ 只保留跨层共享的通用契约（配置模型、chat template）。
"""

from graspo.core.schema import GraspoConfig, Sample

__all__ = ["GraspoConfig", "Sample"]
