"""flow.adapters — 模型适配层：适配器抽象、通用逻辑与模型族实现。

- base.py: BaseGraspoFlowAdapter 抽象协议
- transformer.py: TransformerAdapter 通用适配逻辑
- multimodal_tensors.py: 多模态张量切片/offset 工具
- models/: 模型族实现（common 共享层 + qwen3 + qwen35_36）
"""
