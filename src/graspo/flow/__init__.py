"""GraspoFlow — 设施层：GPU / 分布式 / 网络 / 文件 IO 的执行载体。

分层职责：
- ripple/  算法层：什么值给多少分、怎么判格式、怎么判健康
- core/    通用契约：配置模型、chat template
- flow/    设施层：怎么跑起来、怎么上 GPU（本层）

子模块按功能域组织：parallel/（TP/PP 状态）、
lora/、logger/、adapters/（模型族）、trainer/（训练循环）。
"""

from graspo.flow.runtime import GraspoFlowRuntime
from graspo.flow.trainer import GraspoFlowTrainer

__all__ = ["GraspoFlowTrainer", "GraspoFlowRuntime"]
