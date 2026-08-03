"""flow.parallel — 并行计算设施：分布式状态、层放置、张量工具。

- parallel_state.py: TP/PP 分布式状态（RANK 等）
- placement.py: 层放置规划
- tensor_utils.py: TP all-reduce / cuda snapshot / safetensors / collate
- tensor_math.py: 纯张量数学（RoPE / mask / 采样）
"""
