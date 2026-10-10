"""flow.backend — 可切换后端实现（native / msswift）与并行度。

分层：
- ``flow/adapter/`` = 算法 ↔ 后端对接契约（形状 + 不变量）
- ``flow/backend/`` = 可切换实现（谁把结果接进真实世界）
- ``flow/backend/native/`` = native 后端（自带模型实现与执行载体：
  ``runtime.py`` / ``trainer/`` / ``parallel/`` / ``logger/`` / ``lora/`` / ``models/``）
- ``flow/backend/msswift/`` = ms-swift 后端
- ``flow/backend/selection.py`` = 后端选择器（横跨两后端）
- ``flow/backend/parallel_degree.py`` = 并行度（dp/tp/pp 规模与秩，单一真相源）
"""
