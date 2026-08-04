"""多模态纯逻辑：rows 构建与防呆契约。

本模块是 ripple 算法层的一部分——零设施依赖（不 import torch/GPU/IO）。
职责边界：
- rows 构建、metadata 读写：纯数据变换（rows.py）
- 防呆契约校验：纯逻辑校验，缺图即抛异常（contract.py）
- processor 编码、张量移动：属于 flow 设施层（flow/adapters/transformer_adapter.py）
"""
