"""flow.backend.native — native 后端实现。

- ``models/<model>/model_adapter.py`` = 模型族 / LoRA 适配（旧名为 ``adapter.py``）
- ``models/<model>/`` = native 模型实现（前向 / 生成 / 训练）
- ``transformer_model_adapter.py`` = 模型族适配公共基类
- ``parallel_degree.py`` = 并行度（dp/tp/pp 的规模与秩，单一真相源）
"""
