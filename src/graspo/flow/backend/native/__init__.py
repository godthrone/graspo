"""flow.backend.native — native 后端实现。

- ``models/<model>/model_adapter.py`` = 模型族 / LoRA 适配（旧名为 ``adapter.py``）
- ``models/<model>/`` = native 模型实现（前向 / 生成 / 训练）
- ``transformer_model_adapter.py`` = 模型族适配公共基类
- ``config.py`` = native 后端**默认值唯一真相源**（经注册表注入 core）
- ``runtime.py`` = native 运行时（``GraspoFlowRuntime``）
- ``trainer/`` = native 训练循环；``parallel/`` = TP/PP/DP 状态与调度
- ``logger/`` = native rollout 日志；``lora/`` = LoRA 线性层与 IO
★ 并行度唯一真相源在 ``flow/backend/parallel_degree.py``（跨 backend）。
"""
