"""Qwen3.5/3.6 模型适配器 — 设施层，Hybrid-text 模型的适配器与模型加载。"""

from graspo.flow.backend.native.models.qwen35_36.model_adapter import Qwen35ModelAdapter
from graspo.flow.backend.native.models.qwen35_36.model import Qwen35HybridTextModel

__all__ = [
    "Qwen35ModelAdapter",
    "Qwen35HybridTextModel",
]
