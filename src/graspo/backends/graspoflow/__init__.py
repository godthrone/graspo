"""GraspoFlow — unified tensor/pipeline parallel training backend.

Flink-style architecture:
  Layer 0: Scheduling framework (operator, schedule, graph, memory)
  Layer 1: Generic Transformer (transformer_op, transformer_adapter)
  Layer 2: Training orchestration (base_adapter, runtime, trainer)
  Layer 3: Model families (models/qwen3/, models/qwen35_36/)
"""

from graspo.backends.graspoflow.runtime import GraspoFlowRuntime
from graspo.backends.graspoflow.trainer import GraspoFlowTrainer

__all__ = ["GraspoFlowTrainer", "GraspoFlowRuntime"]
