"""flow.adapter — ★ 算法 ↔ 后端对接契约（形状 + 不变量）。

**一名一物**：本包的 "adapter" 专指「算法与后端之间的对接契约」，
不是「模型族 / LoRA 适配」（后者住 `flow/backend/native/models/<model>/model_adapter.py`），
也不是「并行度适配」（后者住 `flow/backend/parallel_degree.py`）。

- base_graspo_flow_adapter.py: BaseGraspoFlowAdapter —— 模型无关的对接契约基类
- multimodal_tensors.py: 多模态张量切片 / offset 形状口径
- contract.py: RolloutGroup / 三条不变量 I1–I3（P2 落位）
"""
