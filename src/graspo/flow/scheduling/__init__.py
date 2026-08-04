"""flow.scheduling — 流水线调度设施：Flink 风格调度原语与执行。

- operator.py: ComputeOperator/Microbatch/OpBuffer 原语
- schedule.py: GPipe/1F1B/Async 纯调度器
- graph.py: 流水线组装与执行
- transformer_stage_op.py: 通用流水线 stage（P2P 收发）
- optimize_pipeline.py: 1F1B 训练流水线
- rollout_pipeline.py: 前向生成流水线
"""
